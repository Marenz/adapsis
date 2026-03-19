//! Coroutine runtime for Forge async IO.
//!
//! Each Forge coroutine runs on its own tokio task with a dedicated evaluator.
//! `+await` operations send IO requests to the runtime and block (async) until
//! the result comes back. `+spawn` creates a new coroutine.
//!
//! The runtime bridges Forge's synchronous evaluator with tokio's async world
//! using oneshot channels: the evaluator blocks on a channel receive, the
//! runtime completes the IO and sends the result back.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use anyhow::{Result, bail};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot, Mutex};

use crate::eval::Value;

/// A handle that Forge code uses to represent sockets/connections.
pub type Handle = i64;

/// IO operations that Forge code can request via +await.
#[derive(Debug)]
pub enum IoRequest {
    TcpListen { port: u16, reply: oneshot::Sender<Result<Handle>> },
    TcpAccept { listener: Handle, reply: oneshot::Sender<Result<Handle>> },
    TcpRead { conn: Handle, reply: oneshot::Sender<Result<String>> },
    TcpWrite { conn: Handle, data: String, reply: oneshot::Sender<Result<()>> },
    TcpClose { conn: Handle, reply: oneshot::Sender<Result<()>> },
    FileRead { path: String, reply: oneshot::Sender<Result<String>> },
    FileWrite { path: String, data: String, reply: oneshot::Sender<Result<()>> },
    FileExists { path: String, reply: oneshot::Sender<Result<bool>> },
    ListDir { path: String, reply: oneshot::Sender<Result<Vec<String>>> },
    TcpConnect { host: String, port: u16, reply: oneshot::Sender<Result<Handle>> },
    StdinReadLine { prompt: String, reply: oneshot::Sender<Result<String>> },
    Print { text: String, newline: bool, reply: oneshot::Sender<Result<()>> },
    Sleep { ms: u64, reply: oneshot::Sender<Result<()>> },
    Spawn { function_name: String, args: Vec<Value> },
}

/// The coroutine runtime — manages IO resources and dispatches operations.
pub struct Runtime {
    /// Channel for receiving IO requests from coroutines
    io_tx: mpsc::Sender<IoRequest>,
    /// Handle counter
    next_handle: AtomicI64,
    /// Active TCP listeners
    listeners: Mutex<HashMap<Handle, Arc<TcpListener>>>,
    /// Active TCP connections
    connections: Mutex<HashMap<Handle, Arc<Mutex<TcpStream>>>>,
}

impl Runtime {
    pub fn new() -> (Self, mpsc::Receiver<IoRequest>) {
        let (io_tx, io_rx) = mpsc::channel(256);
        (
            Self {
                io_tx,
                next_handle: AtomicI64::new(1),
                listeners: Mutex::new(HashMap::new()),
                connections: Mutex::new(HashMap::new()),
            },
            io_rx,
        )
    }

    fn next_handle(&self) -> Handle {
        self.next_handle.fetch_add(1, Ordering::Relaxed)
    }

    pub fn io_sender(&self) -> mpsc::Sender<IoRequest> {
        self.io_tx.clone()
    }

    /// Process a single IO request. Called from the event loop.
    pub async fn handle_io(&self, request: IoRequest) {
        match request {
            IoRequest::TcpListen { port, reply } => {
                match TcpListener::bind(format!("0.0.0.0:{port}")).await {
                    Ok(listener) => {
                        let handle = self.next_handle();
                        self.listeners.lock().await.insert(handle, Arc::new(listener));
                        let _ = reply.send(Ok(handle));
                    }
                    Err(e) => { let _ = reply.send(Err(e.into())); }
                }
            }
            IoRequest::TcpAccept { listener, reply } => {
                let listeners = self.listeners.lock().await;
                if let Some(l) = listeners.get(&listener) {
                    let l = l.clone();
                    drop(listeners);
                    match l.accept().await {
                        Ok((stream, _addr)) => {
                            let handle = self.next_handle();
                            self.connections.lock().await.insert(
                                handle,
                                Arc::new(Mutex::new(stream)),
                            );
                            let _ = reply.send(Ok(handle));
                        }
                        Err(e) => { let _ = reply.send(Err(e.into())); }
                    }
                } else {
                    let _ = reply.send(Err(anyhow::anyhow!("invalid listener handle {listener}")));
                }
            }
            IoRequest::TcpRead { conn, reply } => {
                let connections = self.connections.lock().await;
                if let Some(stream) = connections.get(&conn) {
                    let stream = stream.clone();
                    drop(connections);
                    let mut buf = vec![0u8; 8192];
                    let mut s = stream.lock().await;
                    match s.read(&mut buf).await {
                        Ok(n) => {
                            let data = String::from_utf8_lossy(&buf[..n]).to_string();
                            let _ = reply.send(Ok(data));
                        }
                        Err(e) => { let _ = reply.send(Err(e.into())); }
                    }
                } else {
                    let _ = reply.send(Err(anyhow::anyhow!("invalid connection handle {conn}")));
                }
            }
            IoRequest::TcpWrite { conn, data, reply } => {
                let connections = self.connections.lock().await;
                if let Some(stream) = connections.get(&conn) {
                    let stream = stream.clone();
                    drop(connections);
                    let mut s = stream.lock().await;
                    match s.write_all(data.as_bytes()).await {
                        Ok(()) => { let _ = reply.send(Ok(())); }
                        Err(e) => { let _ = reply.send(Err(e.into())); }
                    }
                } else {
                    let _ = reply.send(Err(anyhow::anyhow!("invalid connection handle {conn}")));
                }
            }
            IoRequest::TcpClose { conn, reply } => {
                self.connections.lock().await.remove(&conn);
                let _ = reply.send(Ok(()));
            }
            IoRequest::FileRead { path, reply } => {
                match tokio::fs::read_to_string(&path).await {
                    Ok(contents) => { let _ = reply.send(Ok(contents)); }
                    Err(e) => { let _ = reply.send(Err(e.into())); }
                }
            }
            IoRequest::FileWrite { path, data, reply } => {
                match tokio::fs::write(&path, data.as_bytes()).await {
                    Ok(()) => { let _ = reply.send(Ok(())); }
                    Err(e) => { let _ = reply.send(Err(e.into())); }
                }
            }
            IoRequest::FileExists { path, reply } => {
                let exists = tokio::fs::try_exists(&path).await.unwrap_or(false);
                let _ = reply.send(Ok(exists));
            }
            IoRequest::ListDir { path, reply } => {
                match tokio::fs::read_dir(&path).await {
                    Ok(mut entries) => {
                        let mut names = Vec::new();
                        while let Ok(Some(entry)) = entries.next_entry().await {
                            if let Ok(name) = entry.file_name().into_string() {
                                names.push(name);
                            }
                        }
                        let _ = reply.send(Ok(names));
                    }
                    Err(e) => { let _ = reply.send(Err(e.into())); }
                }
            }
            IoRequest::TcpConnect { host, port, reply } => {
                match TcpStream::connect(format!("{host}:{port}")).await {
                    Ok(stream) => {
                        let handle = self.next_handle();
                        self.connections.lock().await.insert(
                            handle,
                            Arc::new(Mutex::new(stream)),
                        );
                        let _ = reply.send(Ok(handle));
                    }
                    Err(e) => { let _ = reply.send(Err(e.into())); }
                }
            }
            IoRequest::StdinReadLine { prompt, reply } => {
                // Run stdin read on a blocking task since it blocks
                tokio::task::spawn_blocking(move || {
                    use std::io::Write;
                    print!("{prompt}");
                    std::io::stdout().flush().ok();
                    let mut line = String::new();
                    match std::io::stdin().read_line(&mut line) {
                        Ok(0) => reply.send(Err(anyhow::anyhow!("EOF"))),
                        Ok(_) => reply.send(Ok(line.trim_end_matches('\n').trim_end_matches('\r').to_string())),
                        Err(e) => reply.send(Err(e.into())),
                    }
                }).await.ok();
            }
            IoRequest::Print { text, newline, reply } => {
                if newline {
                    println!("{text}");
                } else {
                    print!("{text}");
                    std::io::Write::flush(&mut std::io::stdout()).ok();
                }
                let _ = reply.send(Ok(()));
            }
            IoRequest::Sleep { ms, reply } => {
                tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
                let _ = reply.send(Ok(()));
            }
            IoRequest::Spawn { .. } => {
                // Spawn is handled at a higher level
            }
        }
    }
}

/// A coroutine handle — gives Forge code access to the IO runtime.
/// This is passed into the evaluator as a context.
#[derive(Clone, Debug)]
pub struct CoroutineHandle {
    io_tx: mpsc::Sender<IoRequest>,
}

impl CoroutineHandle {
    pub fn new(io_tx: mpsc::Sender<IoRequest>) -> Self {
        Self { io_tx }
    }

    pub fn io_sender(&self) -> mpsc::Sender<IoRequest> {
        self.io_tx.clone()
    }

    /// Execute an await operation — sends IO request and blocks until result.
    /// This is called from the synchronous evaluator, so we use block_on
    /// within a spawn_blocking context.
    pub fn execute_await(&self, op: &str, args: &[Value]) -> Result<Value> {
        let io_tx = self.io_tx.clone();
        let op = op.to_string();
        let args: Vec<Value> = args.to_vec();

        // We're in a sync context (the evaluator). Use a oneshot channel
        // and block on it. The evaluator runs inside spawn_blocking, so
        // blocking is fine.
        let (result_tx, result_rx) = oneshot::channel();

        let request = match op.as_str() {
            "tcp_listen" => {
                let port = match &args[0] {
                    Value::Int(p) => *p as u16,
                    _ => bail!("tcp_listen expects Int port"),
                };
                IoRequest::TcpListen { port, reply: result_tx }
            }
            "tcp_accept" => {
                let handle = match &args[0] {
                    Value::Int(h) => *h,
                    _ => bail!("tcp_accept expects handle"),
                };
                IoRequest::TcpAccept { listener: handle, reply: result_tx }
            }
            "tcp_read" => {
                let handle = match &args[0] {
                    Value::Int(h) => *h,
                    _ => bail!("tcp_read expects handle"),
                };
                let (tx, rx) = oneshot::channel();
                let req = IoRequest::TcpRead { conn: handle, reply: tx };
                // Send request
                io_tx.blocking_send(req)
                    .map_err(|e| anyhow::anyhow!("IO runtime closed: {e}"))?;
                // Wait for result
                let data = rx.blocking_recv()
                    .map_err(|e| anyhow::anyhow!("IO result channel closed: {e}"))??;
                return Ok(Value::String(data));
            }
            "tcp_write" => {
                let handle = match &args[0] {
                    Value::Int(h) => *h,
                    _ => bail!("tcp_write expects handle"),
                };
                let data = match &args[1] {
                    Value::String(s) => s.clone(),
                    other => format!("{other}"),
                };
                let (tx, rx) = oneshot::channel();
                let req = IoRequest::TcpWrite { conn: handle, data, reply: tx };
                io_tx.blocking_send(req)
                    .map_err(|e| anyhow::anyhow!("IO runtime closed: {e}"))?;
                rx.blocking_recv()
                    .map_err(|e| anyhow::anyhow!("IO result channel closed: {e}"))??;
                return Ok(Value::Int(0));
            }
            "tcp_close" => {
                let handle = match &args[0] {
                    Value::Int(h) => *h,
                    _ => bail!("tcp_close expects handle"),
                };
                let (tx, rx) = oneshot::channel();
                let req = IoRequest::TcpClose { conn: handle, reply: tx };
                io_tx.blocking_send(req)
                    .map_err(|e| anyhow::anyhow!("IO runtime closed: {e}"))?;
                rx.blocking_recv()
                    .map_err(|e| anyhow::anyhow!("IO result channel closed: {e}"))??;
                return Ok(Value::Int(0));
            }
            "tcp_connect" => {
                let host = match &args[0] {
                    Value::String(s) => s.clone(),
                    _ => bail!("tcp_connect expects String host"),
                };
                let port = match &args[1] {
                    Value::Int(p) => *p as u16,
                    _ => bail!("tcp_connect expects Int port"),
                };
                let (tx, rx) = oneshot::channel();
                io_tx.blocking_send(IoRequest::TcpConnect { host, port, reply: tx })
                    .map_err(|e| anyhow::anyhow!("IO runtime closed: {e}"))?;
                let handle = rx.blocking_recv()
                    .map_err(|e| anyhow::anyhow!("IO result channel closed: {e}"))??;
                return Ok(Value::Int(handle));
            }
            "read_line" | "stdin_read_line" => {
                let prompt = if args.is_empty() {
                    String::new()
                } else {
                    match &args[0] {
                        Value::String(s) => s.clone(),
                        other => format!("{other}"),
                    }
                };
                let (tx, rx) = oneshot::channel();
                io_tx.blocking_send(IoRequest::StdinReadLine { prompt, reply: tx })
                    .map_err(|e| anyhow::anyhow!("IO runtime closed: {e}"))?;
                let line = rx.blocking_recv()
                    .map_err(|e| anyhow::anyhow!("IO result channel closed: {e}"))??;
                return Ok(Value::String(line));
            }
            "print" => {
                let text = match &args[0] {
                    Value::String(s) => s.clone(),
                    other => format!("{other}"),
                };
                let (tx, rx) = oneshot::channel();
                io_tx.blocking_send(IoRequest::Print { text, newline: false, reply: tx })
                    .map_err(|e| anyhow::anyhow!("IO runtime closed: {e}"))?;
                rx.blocking_recv()
                    .map_err(|e| anyhow::anyhow!("IO result channel closed: {e}"))??;
                return Ok(Value::Int(0));
            }
            "println" => {
                let text = match &args[0] {
                    Value::String(s) => s.clone(),
                    other => format!("{other}"),
                };
                let (tx, rx) = oneshot::channel();
                io_tx.blocking_send(IoRequest::Print { text, newline: true, reply: tx })
                    .map_err(|e| anyhow::anyhow!("IO runtime closed: {e}"))?;
                rx.blocking_recv()
                    .map_err(|e| anyhow::anyhow!("IO result channel closed: {e}"))??;
                return Ok(Value::Int(0));
            }
            "sleep" => {
                let ms = match &args[0] {
                    Value::Int(ms) => *ms as u64,
                    _ => bail!("sleep expects Int milliseconds"),
                };
                let (tx, rx) = oneshot::channel();
                let req = IoRequest::Sleep { ms, reply: tx };
                io_tx.blocking_send(req)
                    .map_err(|e| anyhow::anyhow!("IO runtime closed: {e}"))?;
                rx.blocking_recv()
                    .map_err(|e| anyhow::anyhow!("IO result channel closed: {e}"))??;
                return Ok(Value::Int(0));
            }
            "file_read" | "read_file" => {
                let path = match &args[0] {
                    Value::String(s) => s.clone(),
                    _ => bail!("file_read expects String path"),
                };
                let (tx, rx) = oneshot::channel();
                io_tx.blocking_send(IoRequest::FileRead { path, reply: tx })
                    .map_err(|e| anyhow::anyhow!("IO runtime closed: {e}"))?;
                let contents = rx.blocking_recv()
                    .map_err(|e| anyhow::anyhow!("IO result channel closed: {e}"))??;
                return Ok(Value::String(contents));
            }
            "file_write" | "write_file" => {
                let path = match &args[0] {
                    Value::String(s) => s.clone(),
                    _ => bail!("file_write expects String path"),
                };
                let data = match &args[1] {
                    Value::String(s) => s.clone(),
                    other => format!("{other}"),
                };
                let (tx, rx) = oneshot::channel();
                io_tx.blocking_send(IoRequest::FileWrite { path, data, reply: tx })
                    .map_err(|e| anyhow::anyhow!("IO runtime closed: {e}"))?;
                rx.blocking_recv()
                    .map_err(|e| anyhow::anyhow!("IO result channel closed: {e}"))??;
                return Ok(Value::Int(0));
            }
            "file_exists" => {
                let path = match &args[0] {
                    Value::String(s) => s.clone(),
                    _ => bail!("file_exists expects String path"),
                };
                let (tx, rx) = oneshot::channel();
                io_tx.blocking_send(IoRequest::FileExists { path, reply: tx })
                    .map_err(|e| anyhow::anyhow!("IO runtime closed: {e}"))?;
                let exists = rx.blocking_recv()
                    .map_err(|e| anyhow::anyhow!("IO result channel closed: {e}"))??;
                return Ok(Value::Bool(exists));
            }
            "list_dir" => {
                let path = match &args[0] {
                    Value::String(s) => s.clone(),
                    _ => bail!("list_dir expects String path"),
                };
                let (tx, rx) = oneshot::channel();
                io_tx.blocking_send(IoRequest::ListDir { path, reply: tx })
                    .map_err(|e| anyhow::anyhow!("IO runtime closed: {e}"))?;
                let names = rx.blocking_recv()
                    .map_err(|e| anyhow::anyhow!("IO result channel closed: {e}"))??;
                let values = names.into_iter().map(Value::String).collect();
                return Ok(Value::List(values));
            }
            _ => bail!("unknown await operation: {op}"),
        };

        // Send the request and wait for the result
        io_tx.blocking_send(request)
            .map_err(|e| anyhow::anyhow!("IO runtime closed: {e}"))?;
        let result = result_rx.blocking_recv()
            .map_err(|e| anyhow::anyhow!("IO result channel closed: {e}"))??;
        Ok(Value::Int(result))
    }
}
