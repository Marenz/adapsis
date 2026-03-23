//! Coroutine runtime for Adapsis async IO.
//!
//! Each Adapsis coroutine runs on its own tokio task with a dedicated evaluator.
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
use serde::Serialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot, Mutex};

use crate::eval::Value;

/// A handle that Adapsis code uses to represent sockets/connections.
pub type Handle = i64;

/// Unique ID for a spawned task.
pub type TaskId = i64;

/// What a task is currently waiting on.
#[derive(Debug, Clone, Serialize)]
pub enum WaitReason {
    Running,
    TcpListen(u16),
    TcpAccept(Handle),
    TcpRead(Handle),
    TcpWrite(Handle),
    TcpConnect(String, u16),
    FileRead(String),
    FileWrite(String),
    ShellExec(String),
    Sleep(u64),
    LlmCall,
    LlmAgent,
    StdinRead,
    HttpGet(String),
    HttpPost(String),
    Completed(String),
    Failed(String),
}

impl std::fmt::Display for WaitReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WaitReason::Running => write!(f, "running"),
            WaitReason::TcpListen(p) => write!(f, "tcp_listen(:{p})"),
            WaitReason::TcpAccept(h) => write!(f, "tcp_accept(h{h})"),
            WaitReason::TcpRead(h) => write!(f, "tcp_read(h{h})"),
            WaitReason::TcpWrite(h) => write!(f, "tcp_write(h{h})"),
            WaitReason::TcpConnect(host, port) => write!(f, "tcp_connect({host}:{port})"),
            WaitReason::FileRead(p) => write!(f, "file_read({p})"),
            WaitReason::FileWrite(p) => write!(f, "file_write({p})"),
            WaitReason::ShellExec(cmd) => write!(f, "shell({cmd})"),
            WaitReason::Sleep(ms) => write!(f, "sleep({ms}ms)"),
            WaitReason::LlmCall => write!(f, "llm_call"),
            WaitReason::LlmAgent => write!(f, "llm_agent"),
            WaitReason::StdinRead => write!(f, "stdin_read"),
            WaitReason::HttpGet(url) => write!(f, "http_get({})", url.chars().take(50).collect::<String>()),
            WaitReason::HttpPost(url) => write!(f, "http_post({})", url.chars().take(50).collect::<String>()),
            WaitReason::Completed(v) => write!(f, "done: {}", v.chars().take(50).collect::<String>()),
            WaitReason::Failed(e) => write!(f, "failed: {}", e.chars().take(50).collect::<String>()),
        }
    }
}

/// Info about a running or completed task.
#[derive(Debug, Clone, Serialize)]
pub struct TaskInfo {
    pub id: TaskId,
    pub function_name: String,
    pub status: WaitReason,
    pub started_at: String,
}

/// Shared task registry — tracks all spawned tasks.
pub type TaskRegistry = Arc<std::sync::Mutex<HashMap<TaskId, TaskInfo>>>;

/// IO operations that Adapsis code can request via +await.
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
    Spawn { function_name: String, args: Vec<Value>, reply: oneshot::Sender<Result<TaskId>> },
    HttpGet { url: String, reply: oneshot::Sender<Result<String>> },
    HttpPost { url: String, body: String, content_type: String, reply: oneshot::Sender<Result<String>> },
}

/// The coroutine runtime — manages IO resources and dispatches operations.
pub struct Runtime {
    io_tx: mpsc::Sender<IoRequest>,
    next_handle: AtomicI64,
    next_task_id: AtomicI64,
    listeners: Mutex<HashMap<Handle, Arc<TcpListener>>>,
    connections: Mutex<HashMap<Handle, Arc<Mutex<TcpStream>>>>,
    /// Shared task registry — tracks all spawned async tasks.
    pub task_registry: TaskRegistry,
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
                next_task_id: AtomicI64::new(1),
                listeners: Mutex::new(HashMap::new()),
                connections: Mutex::new(HashMap::new()),
                task_registry: Arc::new(std::sync::Mutex::new(HashMap::new())),
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

    pub fn next_task_id(&self) -> TaskId {
        self.next_task_id.fetch_add(1, Ordering::Relaxed)
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
                            let data = String::from_utf8_lossy(&buf[..n]).into_owned();
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
                eprintln!("AdapsisOS restarting: {} {}", exe.display(), args[1..].join(" "));
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
            IoRequest::HttpGet { url, reply } => {
                tokio::spawn(async move {
                    let client = reqwest::Client::new();
                    match client.get(&url).send().await {
                        Ok(resp) => {
                            // text() decodes using the charset from Content-Type,
                            // defaulting to UTF-8 when none is specified (correct
                            // for application/json per RFC 8259).
                            match resp.text().await {
                                Ok(body) => { let _ = reply.send(Ok(body)); }
                                Err(e) => { let _ = reply.send(Err(e.into())); }
                            }
                        }
                        Err(e) => { let _ = reply.send(Err(e.into())); }
                    }
                });
            }
            IoRequest::HttpPost { url, body, content_type, reply } => {
                tokio::spawn(async move {
                    let client = reqwest::Client::new();
                    // Ensure charset=utf-8 is present in Content-Type for
                    // text-based types so servers know the body encoding.
                    let ct = if !content_type.contains("charset=") &&
                        (content_type.starts_with("application/json") ||
                         content_type.starts_with("text/"))
                    {
                        format!("{content_type}; charset=utf-8")
                    } else {
                        content_type
                    };
                    match client.post(&url)
                        .header("Content-Type", &ct)
                        .body(body)
                        .send().await
                    {
                        Ok(resp) => {
                            match resp.text().await {
                                Ok(body) => { let _ = reply.send(Ok(body)); }
                                Err(e) => { let _ = reply.send(Err(e.into())); }
                            }
                        }
                        Err(e) => { let _ = reply.send(Err(e.into())); }
                    }
                });
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

/// A coroutine handle — gives Adapsis code access to the IO runtime.
/// This is passed into the evaluator as a context.
#[derive(Clone, Debug)]
pub struct CoroutineHandle {
    io_tx: mpsc::Sender<IoRequest>,
    /// If this coroutine is a tracked task, its ID in the registry.
    pub task_id: Option<TaskId>,
    /// Shared task registry for updating wait reasons.
    task_registry: Option<TaskRegistry>,
    /// Mock responses for testing — if set, IO calls check here first.
    mocks: Option<Vec<crate::session::IoMock>>,
}

impl CoroutineHandle {
    pub fn new(io_tx: mpsc::Sender<IoRequest>) -> Self {
        Self { io_tx, task_id: None, task_registry: None, mocks: None }
    }

    pub fn new_with_task(io_tx: mpsc::Sender<IoRequest>, task_id: TaskId, registry: TaskRegistry) -> Self {
        Self { io_tx, task_id: Some(task_id), task_registry: Some(registry), mocks: None }
    }

    /// Create a mock handle for testing — no real IO, returns mock responses.
    pub fn new_mock(mocks: Vec<crate::session::IoMock>) -> Self {
        let (tx, _) = mpsc::channel(1); // dummy channel, never used
        Self { io_tx: tx, task_id: None, task_registry: None, mocks: Some(mocks) }
    }

    pub fn io_sender(&self) -> mpsc::Sender<IoRequest> {
        self.io_tx.clone()
    }

    pub fn registry(&self) -> Option<&TaskRegistry> {
        self.task_registry.as_ref()
    }

    /// Mark this task as waiting on a specific operation.
    fn set_wait(&self, reason: WaitReason) {
        if let (Some(id), Some(reg)) = (self.task_id, &self.task_registry) {
            if let Ok(mut tasks) = reg.lock() {
                if let Some(info) = tasks.get_mut(&id) {
                    info.status = reason;
                }
            }
        }
    }

    /// Mark this task as running (no longer waiting).
    fn clear_wait(&self) {
        self.set_wait(WaitReason::Running);
    }

    /// Helper: send an IO request and wait for the result, tracking wait reason.
    fn send_and_wait<T>(&self, reason: WaitReason, req: IoRequest, rx: oneshot::Receiver<Result<T>>) -> Result<T> {
        self.set_wait(reason);
        self.io_tx.blocking_send(req)
            .map_err(|e| anyhow::anyhow!("IO runtime closed: {e}"))?;
        let result = rx.blocking_recv()
            .map_err(|e| anyhow::anyhow!("IO result channel closed: {e}"))??;
        self.clear_wait();
        Ok(result)
    }

    /// Execute an await operation — sends IO request and blocks until result.
    /// This is called from the synchronous evaluator, so we use block_on
    /// within a spawn_blocking context.
    pub fn execute_await(&self, op: &str, args: &[Value]) -> Result<Value> {
        // Check mock table first — if a mock matches, return it without real IO
        if let Some(mocks) = &self.mocks {
            let arg_str = args.iter().map(|a| format!("{a}")).collect::<Vec<_>>().join(" ");
            for mock in mocks {
                if mock.operation == op && arg_str.contains(&mock.pattern) {
                    return Ok(Value::String(mock.response.clone()));
                }
            }
            // No mock matched — for mock-only handles, return an error
            if self.io_tx.is_closed() {
                bail!("no mock for {op}({arg_str}) — add !mock {op} \"<pattern>\" -> \"<response>\"");
            }
        }

        let op = op.to_string();
        let args: Vec<Value> = args.to_vec();

        match op.as_str() {
            "tcp_listen" => {
                let port = match &args[0] {
                    Value::Int(p) => *p as u16,
                    _ => bail!("tcp_listen expects Int port"),
                };
                let (tx, rx) = oneshot::channel();
                let handle = self.send_and_wait(
                    WaitReason::TcpListen(port),
                    IoRequest::TcpListen { port, reply: tx },
                    rx,
                )?;
                return Ok(Value::Int(handle));
            }
            "tcp_accept" => {
                let listener = match &args[0] {
                    Value::Int(h) => *h,
                    _ => bail!("tcp_accept expects handle"),
                };
                let (tx, rx) = oneshot::channel();
                let handle = self.send_and_wait(
                    WaitReason::TcpAccept(listener),
                    IoRequest::TcpAccept { listener, reply: tx },
                    rx,
                )?;
                return Ok(Value::Int(handle));
            }
            "tcp_read" => {
                let conn = match &args[0] {
                    Value::Int(h) => *h,
                    _ => bail!("tcp_read expects handle"),
                };
                let (tx, rx) = oneshot::channel();
                let data = self.send_and_wait(
                    WaitReason::TcpRead(conn),
                    IoRequest::TcpRead { conn, reply: tx },
                    rx,
                )?;
                return Ok(Value::String(data));
            }
            "tcp_write" => {
                let conn = match &args[0] {
                    Value::Int(h) => *h,
                    _ => bail!("tcp_write expects handle"),
                };
                let data = match &args[1] {
                    Value::String(s) => s.clone(),
                    other => format!("{other}"),
                };
                let (tx, rx) = oneshot::channel();
                self.send_and_wait(
                    WaitReason::TcpWrite(conn),
                    IoRequest::TcpWrite { conn, data, reply: tx },
                    rx,
                )?;
                return Ok(Value::Int(0));
            }
            "tcp_close" => {
                let conn = match &args[0] {
                    Value::Int(h) => *h,
                    _ => bail!("tcp_close expects handle"),
                };
                let (tx, rx) = oneshot::channel();
                self.send_and_wait(
                    WaitReason::Running, // tcp_close is instantaneous
                    IoRequest::TcpClose { conn, reply: tx },
                    rx,
                )?;
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
                let handle = self.send_and_wait(
                    WaitReason::TcpConnect(host.clone(), port),
                    IoRequest::TcpConnect { host, port, reply: tx },
                    rx,
                )?;
                return Ok(Value::Int(handle));
            }
            "read_line" | "stdin_read_line" => {
                let prompt = if args.is_empty() { String::new() } else {
                    match &args[0] { Value::String(s) => s.clone(), other => format!("{other}") }
                };
                let (tx, rx) = oneshot::channel();
                let line = self.send_and_wait(WaitReason::StdinRead, IoRequest::StdinReadLine { prompt, reply: tx }, rx)?;
                return Ok(Value::String(line));
            }
            "print" => {
                let text = match &args[0] { Value::String(s) => s.clone(), other => format!("{other}") };
                let (tx, rx) = oneshot::channel();
                self.send_and_wait(WaitReason::Running, IoRequest::Print { text, newline: false, reply: tx }, rx)?;
                return Ok(Value::Int(0));
            }
            "println" => {
                let text = match &args[0] { Value::String(s) => s.clone(), other => format!("{other}") };
                let (tx, rx) = oneshot::channel();
                self.send_and_wait(WaitReason::Running, IoRequest::Print { text, newline: true, reply: tx }, rx)?;
                return Ok(Value::Int(0));
            }
            "sleep" => {
                let ms = match &args[0] { Value::Int(ms) => *ms as u64, _ => bail!("sleep expects Int milliseconds") };
                let (tx, rx) = oneshot::channel();
                self.send_and_wait(WaitReason::Sleep(ms), IoRequest::Sleep { ms, reply: tx }, rx)?;
                return Ok(Value::Int(0));
            }
            "file_read" | "read_file" => {
                let path = match &args[0] { Value::String(s) => s.clone(), _ => bail!("file_read expects String path") };
                let (tx, rx) = oneshot::channel();
                let contents = self.send_and_wait(WaitReason::FileRead(path.clone()), IoRequest::FileRead { path, reply: tx }, rx)?;
                return Ok(Value::String(contents));
            }
            "file_write" | "write_file" => {
                let path = match &args[0] { Value::String(s) => s.clone(), _ => bail!("file_write expects String path") };
                let data = match &args[1] { Value::String(s) => s.clone(), other => format!("{other}") };
                let (tx, rx) = oneshot::channel();
                self.send_and_wait(WaitReason::FileWrite(path.clone()), IoRequest::FileWrite { path, data, reply: tx }, rx)?;
                return Ok(Value::Int(0));
            }
            "file_exists" => {
                let path = match &args[0] { Value::String(s) => s.clone(), _ => bail!("file_exists expects String path") };
                let (tx, rx) = oneshot::channel();
                let exists = self.send_and_wait(WaitReason::Running, IoRequest::FileExists { path, reply: tx }, rx)?;
                return Ok(Value::Bool(exists));
            }
            "list_dir" => {
                let path = match &args[0] { Value::String(s) => s.clone(), _ => bail!("list_dir expects String path") };
                let (tx, rx) = oneshot::channel();
                let names = self.send_and_wait(WaitReason::Running, IoRequest::ListDir { path, reply: tx }, rx)?;
                return Ok(Value::List(names.into_iter().map(Value::String).collect()));
            }
            "shell_exec" | "exec" => {
                let command = match &args[0] { Value::String(s) => s.clone(), _ => bail!("shell_exec expects String command") };
                let (tx, rx) = oneshot::channel();
                let (stdout, stderr, code) = self.send_and_wait(
                    WaitReason::ShellExec(command.chars().take(40).collect()),
                    IoRequest::ShellExec { command, reply: tx }, rx,
                )?;
                if code == 0 { return Ok(Value::String(stdout)); }
                else { return Ok(Value::String(format!("EXIT {code}: {stderr}"))); }
            }
            "self_restart" | "restart" => {
                let (tx, rx) = oneshot::channel();
                self.send_and_wait(WaitReason::Running, IoRequest::SelfRestart { reply: tx }, rx)?;
                return Ok(Value::String("restarting...".to_string()));
            }
            "http_get" => {
                let url = match args.get(0) { Some(Value::String(s)) => s.clone(), _ => bail!("http_get expects (url:String)") };
                let (tx, rx) = oneshot::channel();
                let result = self.send_and_wait(WaitReason::HttpGet(url.clone()), IoRequest::HttpGet { url, reply: tx }, rx)?;
                return Ok(Value::String(result));
            }
            "http_post" => {
                let url = match args.get(0) { Some(Value::String(s)) => s.clone(), _ => bail!("http_post expects (url:String, body:String, content_type:String)") };
                let body = match args.get(1) { Some(Value::String(s)) => s.clone(), _ => bail!("http_post expects (url:String, body:String, content_type:String)") };
                let content_type = match args.get(2) { Some(Value::String(s)) => s.clone(), _ => "application/json".to_string() };
                let (tx, rx) = oneshot::channel();
                let result = self.send_and_wait(WaitReason::HttpPost(url.clone()), IoRequest::HttpPost { url, body, content_type, reply: tx }, rx)?;
                return Ok(Value::String(result));
            }
            "llm_call" => {
                let system = match args.get(0) { Some(Value::String(s)) => s.clone(), _ => bail!("llm_call expects (system:String, prompt:String[, model:String])") };
                let prompt = match args.get(1) { Some(Value::String(s)) => s.clone(), _ => bail!("llm_call expects (system:String, prompt:String[, model:String])") };
                let model = args.get(2).and_then(|v| match v { Value::String(s) => Some(s.clone()), _ => None });
                let (tx, rx) = oneshot::channel();
                let result = self.send_and_wait(WaitReason::LlmCall, IoRequest::LlmCall { model, system, prompt, reply: tx }, rx)?;
                return Ok(Value::String(result));
            }
            "llm_agent" => {
                let system = match args.get(0) { Some(Value::String(s)) => s.clone(), _ => bail!("llm_agent expects (system:String, task:String[, model:String])") };
                let task = match args.get(1) { Some(Value::String(s)) => s.clone(), _ => bail!("llm_agent expects (system:String, task:String[, model:String])") };
                let model = args.get(2).and_then(|v| match v { Value::String(s) => Some(s.clone()), _ => None });
                let (tx, rx) = oneshot::channel();
                let result = self.send_and_wait(WaitReason::LlmAgent, IoRequest::LlmAgent { model, system, task, reply: tx }, rx)?;
                return Ok(Value::String(result));
            }
            _ => bail!("unknown await operation: {op}"),
        }
    }
}
