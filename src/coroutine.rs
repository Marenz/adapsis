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
    ShellExec { command: String, reply: oneshot::Sender<Result<(String, String, i32)>> },
    SelfRestart { reply: oneshot::Sender<Result<()>> },
    StdinReadLine { prompt: String, reply: oneshot::Sender<Result<String>> },
    Print { text: String, newline: bool, reply: oneshot::Sender<Result<()>> },
    Sleep { ms: u64, reply: oneshot::Sender<Result<()>> },
    /// Single LLM text generation — no agentic loop, just prompt → response
    LlmCall {
        model: Option<String>,
        system: String,
        prompt: String,
        reply: oneshot::Sender<Result<String>>,
    },
    /// Full agentic LLM loop — takes control until DONE
    LlmAgent {
        model: Option<String>,
        system: String,
        task: String,
        reply: oneshot::Sender<Result<String>>,
    },
    Spawn { function_name: String, args: Vec<Value> },
}

/// The coroutine runtime — manages IO resources and dispatches operations.
pub struct Runtime {
    io_tx: mpsc::Sender<IoRequest>,
    next_handle: AtomicI64,
    listeners: Mutex<HashMap<Handle, Arc<TcpListener>>>,
    connections: Mutex<HashMap<Handle, Arc<Mutex<TcpStream>>>>,
    /// LLM config for llm_call/llm_agent
    pub llm_url: String,
    pub llm_default_model: String,
    pub llm_api_key: Option<String>,
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
                llm_url: String::new(),
                llm_default_model: String::new(),
                llm_api_key: None,
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
            IoRequest::ShellExec { command, reply } => {
                tokio::task::spawn_blocking(move || {
                    let output = std::process::Command::new("sh")
                        .arg("-c")
                        .arg(&command)
                        .output();
                    match output {
                        Ok(out) => {
                            let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                            let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                            let code = out.status.code().unwrap_or(-1);
                            reply.send(Ok((stdout, stderr, code)))
                        }
                        Err(e) => reply.send(Err(e.into())),
                    }
                }).await.ok();
            }
            IoRequest::SelfRestart { reply } => {
                // Save session, then exec() the same binary with same args
                let _ = reply.send(Ok(()));
                // Give a moment for the reply to be sent
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                let exe = std::env::current_exe().unwrap_or_default();
                let args: Vec<String> = std::env::args().collect();
                eprintln!("ForgeOS restarting: {} {}", exe.display(), args[1..].join(" "));
                let err = exec::execvp(&exe, &args);
                eprintln!("restart failed: {err}");
            }
            IoRequest::StdinReadLine { prompt, reply } => {
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
            IoRequest::LlmCall { model, system, prompt, reply } => {
                let url = self.llm_url.clone();
                let default_model = self.llm_default_model.clone();
                let api_key = self.llm_api_key.clone();
                let model = model.unwrap_or(default_model);

                tokio::spawn(async move {
                    let llm = crate::llm::LlmClient::new_with_model_and_key(&url, &model, api_key);
                    let messages = vec![
                        crate::llm::ChatMessage::system(system),
                        crate::llm::ChatMessage::user(prompt),
                    ];
                    match llm.generate(messages).await {
                        Ok(output) => {
                            let text = if !output.code.is_empty() { output.code } else { output.text };
                            let _ = reply.send(Ok(text));
                        }
                        Err(e) => { let _ = reply.send(Err(e)); }
                    }
                });
            }
            IoRequest::LlmAgent { model, system, task, reply } => {
                let url = self.llm_url.clone();
                let default_model = self.llm_default_model.clone();
                let api_key = self.llm_api_key.clone();
                let model = model.unwrap_or(default_model);

                tokio::spawn(async move {
                    let llm = crate::llm::LlmClient::new_with_model_and_key(&url, &model, api_key);
                    let builtins = crate::builtins::format_for_prompt();
                    let full_system = format!("{}\n\n{builtins}\n\nWork step by step. When done, respond with DONE.", system);
                    let mut messages = vec![
                        crate::llm::ChatMessage::system(full_system),
                        crate::llm::ChatMessage::user(task),
                    ];

                    let mut final_result = String::new();
                    for _iter in 0..10 {
                        match llm.generate(messages.clone()).await {
                            Ok(output) => {
                                messages.push(crate::llm::ChatMessage::assistant(&output.text));
                                let code = &output.code;
                                if code.trim() == "DONE" || code.is_empty() {
                                    final_result = output.text;
                                    break;
                                }
                                // Feed results back
                                messages.push(crate::llm::ChatMessage::user(
                                    "Continue with the next step, or DONE if complete.".to_string()
                                ));
                                final_result = output.text;
                            }
                            Err(e) => {
                                let _ = reply.send(Err(e));
                                return;
                            }
                        }
                    }
                    let _ = reply.send(Ok(final_result));
                });
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
            "shell_exec" | "exec" => {
                let command = match &args[0] {
                    Value::String(s) => s.clone(),
                    _ => bail!("shell_exec expects String command"),
                };
                let (tx, rx) = oneshot::channel();
                io_tx.blocking_send(IoRequest::ShellExec { command, reply: tx })
                    .map_err(|e| anyhow::anyhow!("IO runtime closed: {e}"))?;
                let (stdout, stderr, code) = rx.blocking_recv()
                    .map_err(|e| anyhow::anyhow!("IO result channel closed: {e}"))??;
                // Return a struct-like result: {stdout, stderr, code}
                // For simplicity, return stdout if code==0, otherwise stderr
                if code == 0 {
                    return Ok(Value::String(stdout));
                } else {
                    return Ok(Value::String(format!("EXIT {code}: {stderr}")));
                }
            }
            "self_restart" | "restart" => {
                let (tx, rx) = oneshot::channel();
                io_tx.blocking_send(IoRequest::SelfRestart { reply: tx })
                    .map_err(|e| anyhow::anyhow!("IO runtime closed: {e}"))?;
                rx.blocking_recv()
                    .map_err(|e| anyhow::anyhow!("IO result channel closed: {e}"))??;
                return Ok(Value::String("restarting...".to_string()));
            }
            "llm_call" => {
                // llm_call(system_prompt, user_prompt) or llm_call(system, user, model)
                let system = match args.get(0) {
                    Some(Value::String(s)) => s.clone(),
                    _ => bail!("llm_call expects (system:String, prompt:String[, model:String])"),
                };
                let prompt = match args.get(1) {
                    Some(Value::String(s)) => s.clone(),
                    _ => bail!("llm_call expects (system:String, prompt:String[, model:String])"),
                };
                let model = args.get(2).and_then(|v| match v {
                    Value::String(s) => Some(s.clone()),
                    _ => None,
                });
                let (tx, rx) = oneshot::channel();
                io_tx.blocking_send(IoRequest::LlmCall { model, system, prompt, reply: tx })
                    .map_err(|e| anyhow::anyhow!("IO runtime closed: {e}"))?;
                let result = rx.blocking_recv()
                    .map_err(|e| anyhow::anyhow!("IO result channel closed: {e}"))??;
                return Ok(Value::String(result));
            }
            "llm_agent" => {
                // llm_agent(system_prompt, task) or llm_agent(system, task, model)
                let system = match args.get(0) {
                    Some(Value::String(s)) => s.clone(),
                    _ => bail!("llm_agent expects (system:String, task:String[, model:String])"),
                };
                let task = match args.get(1) {
                    Some(Value::String(s)) => s.clone(),
                    _ => bail!("llm_agent expects (system:String, task:String[, model:String])"),
                };
                let model = args.get(2).and_then(|v| match v {
                    Value::String(s) => Some(s.clone()),
                    _ => None,
                });
                let (tx, rx) = oneshot::channel();
                io_tx.blocking_send(IoRequest::LlmAgent { model, system, task, reply: tx })
                    .map_err(|e| anyhow::anyhow!("IO runtime closed: {e}"))?;
                let result = rx.blocking_recv()
                    .map_err(|e| anyhow::anyhow!("IO result channel closed: {e}"))??;
                return Ok(Value::String(result));
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
