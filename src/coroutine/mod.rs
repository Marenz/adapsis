//! Coroutine runtime for Adapsis async IO.
//!
//! Each Adapsis coroutine runs on its own tokio task with a dedicated evaluator.
//! `+await` operations send IO requests to the runtime and block (async) until
//! the result comes back. `+spawn` creates a new coroutine.
//!
//! The runtime bridges Adapsis's synchronous evaluator with tokio's async world
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
    LlmTakeover(String),
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
            WaitReason::LlmTakeover(ctx) => write!(f, "llm_takeover({ctx})"),
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

/// Inspectable snapshot of a running task's interpreter state.
/// Values are rendered as display strings — no recursive Value serialization.
#[derive(Debug, Clone, Serialize)]
pub struct TaskSnapshot {
    pub task_id: TaskId,
    pub function_name: String,
    pub current_stmt_id: Option<String>,
    pub frame_depth: usize,
    pub locals: Vec<(String, String)>,
    pub wait_reason: String,
}

/// Shared snapshot registry — updated by the evaluator as tasks execute.
pub type TaskSnapshotRegistry = Arc<std::sync::Mutex<HashMap<TaskId, TaskSnapshot>>>;

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
    /// Multipart file upload: POST a file with optional extra form fields.
    HttpUpload { url: String, file_path: String, file_field: String, extra_fields: Vec<(String, String)>, reply: oneshot::Sender<Result<String>> },
    /// Register a source (timer, channel, event) on a module.
    SourceAdd {
        module_name: String,
        source_type: String,  // "timer", "channel", or "event:Module.event_name"
        interval_ms: Option<u64>,
        alias: String,
        handler: String,  // fully-qualified handler like "MyModule.on_tick"
        reply: oneshot::Sender<Result<String>>,
    },
    /// Conversational LLM call with per-context history.
    /// Returns the text reply immediately; code execution happens in background.
    LlmTakeover {
        context: String,
        message: String,
        reply_fn: Option<String>,
        reply_arg: Option<String>,
        reply: oneshot::Sender<Result<String>>,
    },
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
    /// Shared snapshot registry — live interpreter state for each task.
    pub snapshot_registry: TaskSnapshotRegistry,
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
                snapshot_registry: Arc::new(std::sync::Mutex::new(HashMap::new())),
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
                // Prefer worktree binary if it exists (after !opencode rebuild)
                let worktree_binary = std::env::args()
                    .position(|a| a == "--opencode-git-dir")
                    .and_then(|pos| std::env::args().nth(pos + 1))
                    .map(|d| std::path::PathBuf::from(d).join("target/release/adapsis"))
                    .filter(|p| p.exists());
                // Install to ~/.local/bin/adapsis if we have a worktree build
                if let Some(ref wb) = worktree_binary {
                    if let Some(home) = std::env::var_os("HOME") {
                        let install = std::path::PathBuf::from(home).join(".local/bin/adapsis");
                        if install.parent().map(|d| d.exists()).unwrap_or(false) {
                            match std::fs::copy(wb, &install) {
                                Ok(_) => eprintln!("[restart] installed binary to {}", install.display()),
                                Err(e) => eprintln!("[restart] failed to install binary: {e}"),
                            }
                        }
                    }
                }
                let exe = std::env::current_exe().unwrap_or_else(|_| {
                    std::env::args().next()
                        .map(std::path::PathBuf::from)
                        .unwrap_or_default()
                });
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
                    let client = reqwest::Client::builder()
                        .timeout(std::time::Duration::from_secs(30))
                        .build()
                        .unwrap_or_default();
                    match client.get(&url).send().await {
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
            IoRequest::HttpPost { url, body, content_type, reply } => {
                tokio::spawn(async move {
                    let client = reqwest::Client::builder()
                        .timeout(std::time::Duration::from_secs(30))
                        .build()
                        .unwrap_or_default();
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
            IoRequest::HttpUpload { url, file_path, file_field, extra_fields, reply } => {
                tokio::spawn(async move {
                    let client = reqwest::Client::builder()
                        .timeout(std::time::Duration::from_secs(120))
                        .build()
                        .unwrap_or_default();
                    let file_bytes = match tokio::fs::read(&file_path).await {
                        Ok(b) => b,
                        Err(e) => {
                            let _ = reply.send(Err(anyhow::anyhow!("http_upload: cannot read file '{}': {}", file_path, e)));
                            return;
                        }
                    };
                    let file_name = std::path::Path::new(&file_path)
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| "file".to_string());
                    // Guess MIME type from extension
                    let mime = match std::path::Path::new(&file_path).extension().and_then(|e| e.to_str()) {
                        Some("wav") => "audio/wav",
                        Some("mp3") => "audio/mpeg",
                        Some("ogg") => "audio/ogg",
                        Some("flac") => "audio/flac",
                        Some("jpg") | Some("jpeg") => "image/jpeg",
                        Some("png") => "image/png",
                        Some("pdf") => "application/pdf",
                        _ => "application/octet-stream",
                    };
                    let file_part = reqwest::multipart::Part::bytes(file_bytes)
                        .file_name(file_name)
                        .mime_str(mime)
                        .unwrap_or_else(|_| reqwest::multipart::Part::bytes(vec![]));
                    let mut form = reqwest::multipart::Form::new()
                        .part(file_field, file_part);
                    for (key, value) in extra_fields {
                        form = form.text(key, value);
                    }
                    match client.post(&url).multipart(form).send().await {
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
                // Spawn is handled at a higher level (main.rs IO loop)
            }
            IoRequest::SourceAdd { .. } => {
                // SourceAdd is handled at a higher level (main.rs IO loop)
            }
            IoRequest::LlmTakeover { .. } => {
                // LlmTakeover is handled at a higher level (main.rs IO loop)
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
    /// Shared snapshot registry for live interpreter state inspection.
    snapshot_registry: Option<TaskSnapshotRegistry>,
    /// Mock responses for testing — if set, IO calls check here first.
    mocks: Option<Vec<crate::session::IoMock>>,
    /// Function stubs for testing — intercepts user function calls.
    stubs: Option<Vec<crate::session::FunctionStub>>,
}

impl CoroutineHandle {
    pub fn new(io_tx: mpsc::Sender<IoRequest>) -> Self {
        Self { io_tx, task_id: None, task_registry: None, snapshot_registry: None, mocks: None, stubs: None }
    }

    pub fn new_with_task(
        io_tx: mpsc::Sender<IoRequest>,
        task_id: TaskId,
        registry: TaskRegistry,
        snapshot_registry: TaskSnapshotRegistry,
    ) -> Self {
        Self { io_tx, task_id: Some(task_id), task_registry: Some(registry), snapshot_registry: Some(snapshot_registry), mocks: None, stubs: None }
    }

    /// Create a mock handle for testing — no real IO, returns mock responses.
    #[allow(dead_code)]
    pub fn new_mock(mocks: Vec<crate::session::IoMock>) -> Self {
        let (tx, _) = mpsc::channel(1);
        Self { io_tx: tx, task_id: None, task_registry: None, snapshot_registry: None, mocks: Some(mocks), stubs: None }
    }

    /// Create a mock handle with function stubs for testing.
    pub fn new_mock_with_stubs(mocks: Vec<crate::session::IoMock>, stubs: Vec<crate::session::FunctionStub>) -> Self {
        let (tx, _) = mpsc::channel(1);
        Self { io_tx: tx, task_id: None, task_registry: None, snapshot_registry: None, mocks: Some(mocks), stubs: Some(stubs) }
    }

    /// Create a handle with mocks AND a real IO sender — mocks are checked first,
    /// unmatched operations fall through to real IO via the sender.
    pub fn new_mock_with_sender(mocks: Vec<crate::session::IoMock>, io_tx: mpsc::Sender<IoRequest>) -> Self {
        Self { io_tx, task_id: None, task_registry: None, snapshot_registry: None, mocks: Some(mocks), stubs: None }
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

    /// Update the live interpreter snapshot for this task.
    /// Called by the evaluator before each statement.
    /// `locals` should be pre-computed via `env.snapshot_bindings()` by the caller.
    pub fn update_snapshot(
        &self,
        function_name: &str,
        current_stmt_id: Option<String>,
        frame_depth: usize,
        locals: Vec<(String, String)>,
    ) {
        if let (Some(id), Some(snap_reg)) = (self.task_id, &self.snapshot_registry) {
            let wait_str = if let Some(task_reg) = &self.task_registry {
                task_reg.lock().ok()
                    .and_then(|tasks| tasks.get(&id).map(|t| format!("{}", t.status)))
                    .unwrap_or_else(|| "unknown".to_string())
            } else {
                "unknown".to_string()
            };
            let snap = TaskSnapshot {
                task_id: id,
                function_name: function_name.to_string(),
                current_stmt_id,
                frame_depth,
                locals,
                wait_reason: wait_str,
            };
            if let Ok(mut snaps) = snap_reg.lock() {
                snaps.insert(id, snap);
            }
        }
    }

    /// Mark the snapshot as completed (keep last known state queryable).
    pub fn complete_snapshot(&self, function_name: &str) {
        if let (Some(id), Some(snap_reg)) = (self.task_id, &self.snapshot_registry) {
            if let Ok(mut snaps) = snap_reg.lock() {
                if let Some(snap) = snaps.get_mut(&id) {
                    snap.function_name = function_name.to_string();
                    snap.current_stmt_id = None;
                    // wait_reason will be updated from TaskInfo on query
                }
            }
        }
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

    pub fn try_mock_io(&self, op: &str, args: &[Value], fail_on_missing: bool) -> Result<Option<Value>> {
        let Some(mocks) = &self.mocks else {
            return Ok(None);
        };

        let arg_strs: Vec<String> = args.iter().map(|a| format!("{a}")).collect();
        let arg_str = arg_strs.join(" ");
        'mock_loop: for mock in mocks {
            if mock.operation != op {
                continue;
            }
            if mock.patterns.len() == 1 {
                if arg_str.contains(&mock.patterns[0]) {
                    return Ok(Some(Value::string(mock.response.clone())));
                }
            } else {
                if mock.patterns.len() > arg_strs.len() {
                    continue;
                }
                for (pat, arg) in mock.patterns.iter().zip(arg_strs.iter()) {
                    if !arg.contains(pat) {
                        continue 'mock_loop;
                    }
                }
                return Ok(Some(Value::string(mock.response.clone())));
            }
        }

        if fail_on_missing && self.io_tx.is_closed() {
            bail!("no mock for {op}({arg_str}) — add !mock {op} \"<pattern>\" -> \"<response>\"");
        }

        Ok(None)
    }

    /// Check if a function call matches a registered stub.
    /// Returns the raw expression string if matched.
    pub fn try_stub(&self, func_name: &str, args: &[Value]) -> Result<Option<String>> {
        let Some(stubs) = &self.stubs else {
            return Ok(None);
        };
        let arg_strs: Vec<String> = args.iter().map(|a| format!("{a}")).collect();
        let arg_str = arg_strs.join(" ");
        'stub_loop: for stub in stubs {
            if stub.function_name != func_name {
                continue;
            }
            if stub.patterns.len() == 1 {
                if arg_str.contains(&stub.patterns[0]) {
                    return Ok(Some(stub.response_expr.clone()));
                }
            } else {
                if stub.patterns.len() > arg_strs.len() {
                    continue;
                }
                for (pat, arg) in stub.patterns.iter().zip(arg_strs.iter()) {
                    if !arg.contains(pat) {
                        continue 'stub_loop;
                    }
                }
                return Ok(Some(stub.response_expr.clone()));
            }
        }
        Ok(None)
    }

    /// Execute an await operation — sends IO request and blocks until result.
    /// This is called from the synchronous evaluator, so we use block_on
    /// within a spawn_blocking context.
    pub fn execute_await(&self, op: &str, args: &[Value]) -> Result<Value> {
        if op != "mock_set" && op != "mock_clear" {
            if let Some(result) = self.try_mock_io(op, args, false)? {
                return Ok(result);
            }
        }

        // Try in-process operations (shared state, queries, mutations, misc)
        if let Some(result) = self.execute_local_op(op, args)? {
            return Ok(result);
        }

        if let Some(result) = self.try_mock_io(op, args, true)? {
            return Ok(result);
        }

        self.execute_transport_op(op, args)
    }

    /// In-process operations handled via thread-local shared state.
    /// Covers roadmap/plan/inbox/mock ops, queries, AST mutations, and misc commands.
    /// Returns `Ok(Some(val))` if handled, `Ok(None)` to fall through to transport ops.
    fn execute_local_op(&self, op: &str, args: &[Value]) -> Result<Option<Value>> {
        match op {
            "roadmap_list" => {
                let result = crate::shared_state::get_shared_meta()
                    .and_then(|meta| meta.lock().ok().map(|m| crate::session::roadmap_list(&m.roadmap)))
                    .unwrap_or_else(|| "Roadmap is empty.".to_string());
                return Ok(Some(Value::string(result)));
            }
            "roadmap_add" => {
                let desc = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    Some(other) => format!("{other}"),
                    None => bail!("roadmap_add expects (description:String)"),
                };
                if desc.trim().is_empty() {
                    bail!("roadmap_add: description must not be empty");
                }
                let result = crate::shared_state::get_shared_meta()
                    .ok_or_else(|| anyhow::anyhow!("roadmap_add: no meta available"))
                    .and_then(|meta| {
                        meta.lock()
                            .map_err(|_| anyhow::anyhow!("roadmap_add: could not lock meta"))
                            .map(|mut m| {
                                crate::session::roadmap_add(&mut m.roadmap, &desc);
                                desc.clone()
                            })
                    })?;
                return Ok(Some(Value::string(result)));
            }
            "roadmap_done" => {
                let n = match args.first() {
                    Some(Value::Int(n)) => *n,
                    _ => bail!("roadmap_done expects (n:Int)"),
                };
                if n < 1 {
                    bail!("roadmap_done: index must be >= 1");
                }
                let result = crate::shared_state::get_shared_meta()
                    .ok_or_else(|| anyhow::anyhow!("roadmap_done: no meta available"))
                    .and_then(|meta| {
                        meta.lock()
                            .map_err(|_| anyhow::anyhow!("roadmap_done: could not lock meta"))
                            .and_then(|mut m| {
                                let msg = crate::session::roadmap_done_checked(&m, n as usize)?;
                                // Apply the actual state change
                                m.roadmap[(n as usize).saturating_sub(1)].done = true;
                                Ok(msg)
                            })
                    })?;
                return Ok(Some(Value::string(result)));
            }

            // ── Plan operations — via SHARED_META ──
            "plan_show" => {
                let result = crate::shared_state::get_shared_meta()
                    .and_then(|meta| meta.lock().ok().map(|m| crate::session::plan_show(&m.plan)))
                    .unwrap_or_else(|| "No plan set.".to_string());
                return Ok(Some(Value::string(result)));
            }
            "plan_set" => {
                let steps_str = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    Some(other) => format!("{other}"),
                    None => bail!("plan_set expects (steps:String)"),
                };
                let descriptions: Vec<String> = steps_str
                    .lines()
                    .map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty())
                    .collect();
                if descriptions.is_empty() {
                    bail!("plan_set: steps must not be empty");
                }
                let result = crate::shared_state::get_shared_meta()
                    .ok_or_else(|| anyhow::anyhow!("plan_set: no meta available"))
                    .and_then(|meta| {
                        meta.lock()
                            .map_err(|_| anyhow::anyhow!("plan_set: could not lock meta"))
                            .map(|mut m| crate::session::plan_set(&mut m.plan, &descriptions))
                    })?;
                return Ok(Some(Value::string(result)));
            }
            "plan_done" => {
                let n = match args.first() {
                    Some(Value::Int(n)) => *n,
                    _ => bail!("plan_done expects (n:Int)"),
                };
                if n < 1 {
                    bail!("plan_done: index must be >= 1");
                }
                let result = crate::shared_state::get_shared_meta()
                    .ok_or_else(|| anyhow::anyhow!("plan_done: no meta available"))
                    .and_then(|meta| {
                        meta.lock()
                            .map_err(|_| anyhow::anyhow!("plan_done: could not lock meta"))
                            .and_then(|mut m| crate::session::plan_done(&mut m.plan, n as usize))
                    })?;
                return Ok(Some(Value::string(result)));
            }
            "plan_fail" => {
                let n = match args.first() {
                    Some(Value::Int(n)) => *n,
                    _ => bail!("plan_fail expects (n:Int)"),
                };
                if n < 1 {
                    bail!("plan_fail: index must be >= 1");
                }
                let result = crate::shared_state::get_shared_meta()
                    .ok_or_else(|| anyhow::anyhow!("plan_fail: no meta available"))
                    .and_then(|meta| {
                        meta.lock()
                            .map_err(|_| anyhow::anyhow!("plan_fail: could not lock meta"))
                            .and_then(|mut m| crate::session::plan_fail(&mut m.plan, n as usize))
                    })?;
                return Ok(Some(Value::string(result)));
            }
            // ── Query operations — access program AST via thread-local ──
            // These reuse the same logic as the ? query commands in typeck.rs.
            "query_symbols" | "symbols_list" => {
                let program = crate::shared_state::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_symbols: program not available (no async context)"))?;
                let table = crate::typeck::build_symbol_table(&program);
                let result = crate::typeck::handle_query(&program, &table, "?symbols", &[]);
                return Ok(Some(Value::string(result)));
            }
            "query_symbols_detail" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("query_symbols_detail expects (name:String)"),
                };
                let program = crate::shared_state::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_symbols_detail: program not available"))?;
                let table = crate::typeck::build_symbol_table(&program);
                let query = format!("?symbols {name}");
                let result = crate::typeck::handle_query(&program, &table, &query, &[]);
                return Ok(Some(Value::string(result)));
            }
            "query_source" | "source_get" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("query_source expects (name:String)"),
                };
                let program = crate::shared_state::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_source: program not available"))?;
                let table = crate::typeck::build_symbol_table(&program);
                let query = format!("?source {name}");
                let result = crate::typeck::handle_query(&program, &table, &query, &[]);
                return Ok(Some(Value::string(result)));
            }
            "query_callers" | "callers_get" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("query_callers expects (name:String)"),
                };
                let program = crate::shared_state::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_callers: program not available"))?;
                let table = crate::typeck::build_symbol_table(&program);
                let query = format!("?callers {name}");
                let result = crate::typeck::handle_query(&program, &table, &query, &[]);
                return Ok(Some(Value::string(result)));
            }
            "query_callees" | "callees_get" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("query_callees expects (name:String)"),
                };
                let program = crate::shared_state::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_callees: program not available"))?;
                let table = crate::typeck::build_symbol_table(&program);
                let query = format!("?callees {name}");
                let result = crate::typeck::handle_query(&program, &table, &query, &[]);
                return Ok(Some(Value::string(result)));
            }
            "query_deps" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("query_deps expects (name:String)"),
                };
                let program = crate::shared_state::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_deps: program not available"))?;
                let table = crate::typeck::build_symbol_table(&program);
                let query = format!("?deps {name}");
                let result = crate::typeck::handle_query(&program, &table, &query, &[]);
                return Ok(Some(Value::string(result)));
            }
            "query_deps_all" | "deps_get" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("query_deps_all expects (name:String)"),
                };
                let program = crate::shared_state::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_deps_all: program not available"))?;
                let table = crate::typeck::build_symbol_table(&program);
                let query = format!("?deps-all {name}");
                let result = crate::typeck::handle_query(&program, &table, &query, &[]);
                return Ok(Some(Value::string(result)));
            }
            "query_routes" | "routes_list" => {
                let program = crate::shared_state::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_routes: program not available"))?;
                // Get HTTP routes from SharedRuntime
                let routes = crate::shared_state::get_shared_runtime()
                    .and_then(|rt| rt.read().ok().map(|state| state.http_routes.clone()))
                    .unwrap_or_default();
                let table = crate::typeck::build_symbol_table(&program);
                let result = crate::typeck::handle_query(&program, &table, "?routes", &routes);
                return Ok(Some(Value::string(result)));
            }
            "query_tasks" => {
                let result = if let Some(reg) = &self.task_registry {
                    crate::api::format_tasks(&Some(reg.clone()))
                } else {
                    "No task registry (not in async context).".to_string()
                };
                return Ok(Some(Value::string(result)));
            }
            "query_inbox" => {
                if !args.is_empty() {
                    bail!("query_inbox expects no arguments");
                }
                let result = crate::shared_state::get_shared_meta()
                    .map(|meta| {
                        meta.lock()
                            .map_err(|_| anyhow::anyhow!("query_inbox: could not lock meta"))
                            .map(|meta| {
                                let msgs = crate::session::peek_messages(&meta, "main");
                                if msgs.is_empty() {
                                    "No messages.".to_string()
                                } else {
                                    msgs.iter()
                                        .map(|m| format!("[{}] from {}: {}", m.timestamp, m.from, m.content))
                                        .collect::<Vec<_>>()
                                        .join("\n")
                                }
                            })
                    })
                    .transpose()?
                    .unwrap_or_else(|| "No messages.".to_string());
                return Ok(Some(Value::string(result)));
            }
            "inbox_clear" => {
                if !args.is_empty() {
                    bail!("inbox_clear expects no arguments");
                }

                let cleared = if let Some(meta) = crate::shared_state::get_shared_meta() {
                    meta.lock()
                        .map_err(|_| anyhow::anyhow!("inbox_clear: could not lock meta"))?
                        .agent_mailbox
                        .remove("main")
                        .map(|msgs| msgs.len())
                        .unwrap_or(0)
                } else {
                    let rt = crate::shared_state::get_shared_runtime()
                        .ok_or_else(|| anyhow::anyhow!("inbox_clear: no runtime available"))?;
                    rt.write()
                        .map_err(|_| anyhow::anyhow!("inbox_clear: could not access runtime"))?
                        .agent_mailbox
                        .remove("main")
                        .map(|msgs| msgs.len())
                        .unwrap_or(0)
                };

                if let Some(rt) = crate::shared_state::get_shared_runtime() {
                    if let Ok(mut state) = rt.write() {
                        state.agent_mailbox.remove("main");
                    }
                }

                return Ok(Some(Value::string(format!("cleared {cleared} messages"))));
            }
            "query_library" => {
                let program = crate::shared_state::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_library: program not available"))?;
                let result = crate::library::query_library(&program, None);
                return Ok(Some(Value::string(result)));
            }

            // ── library_errors — formatted string of all library errors ──
            "library_errors" => {
                let result = crate::shared_state::get_shared_runtime()
                    .and_then(|rt| rt.read().ok().map(|state| {
                        let mut out = String::new();
                        // Structured load errors (module_name, error_message)
                        if !state.library_load_errors.is_empty() {
                            out.push_str(&format!("Load errors ({}):\n", state.library_load_errors.len()));
                            for (module_name, error) in &state.library_load_errors {
                                out.push_str(&format!("  {}: {}\n", module_name, error));
                            }
                        }
                        // General errors
                        if !state.library_errors.is_empty() {
                            if !out.is_empty() {
                                out.push('\n');
                            }
                            out.push_str(&format!("Errors this session ({}):\n", state.library_errors.len()));
                            for e in &state.library_errors {
                                out.push_str(&format!("  {}\n", e));
                            }
                        }
                        if out.is_empty() {
                            "No library errors.".to_string()
                        } else {
                            out.trim_end().to_string()
                        }
                    }))
                    .unwrap_or_else(|| "No library errors.".to_string());
                return Ok(Some(Value::string(result)));
            }
            "failure_history" => {
                if !args.is_empty() {
                    bail!("failure_history expects no arguments");
                }
                let result = crate::shared_state::get_shared_runtime()
                    .and_then(|rt| rt.read().ok().map(|state| crate::session::format_failure_history(&state)))
                    .unwrap_or_else(|| "No recent mutation failures.".to_string());
                return Ok(Some(Value::string(result)));
            }
            "clear_failure_history" => {
                if !args.is_empty() {
                    bail!("clear_failure_history expects no arguments");
                }
                if let Some(rt) = crate::shared_state::get_shared_runtime()
                    && let Ok(mut state) = rt.write()
                {
                    crate::session::clear_failure_history(&mut state);
                }
                return Ok(Some(Value::string("cleared")));
            }
            "failure_patterns" => {
                if !args.is_empty() {
                    bail!("failure_patterns expects no arguments");
                }
                let result = crate::shared_state::get_shared_runtime()
                    .and_then(|rt| rt.read().ok().map(|state| crate::session::summarize_failure_patterns(&state)))
                    .unwrap_or_else(|| "No recent mutation failures.".to_string());
                return Ok(Some(Value::string(result)));
            }
            // ── library_reload — reload module(s) from disk ──
            "library_reload" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    Some(other) => format!("{other}"),
                    None => String::new(),
                };
                let program_lock = crate::shared_state::get_shared_program_mut()
                    .ok_or_else(|| anyhow::anyhow!("library_reload: program not available (no async context)"))?;
                let mut program = program_lock.write()
                    .map_err(|_| anyhow::anyhow!("library_reload: could not acquire program write lock"))?;

                let result = crate::library::reload_module(&mut program, &name)?;
                // Update the read-only snapshot so query builtins see the changes
                crate::shared_state::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                return Ok(Some(Value::string(result)));
            }

            "run_module_startups" => {
                let program_lock = crate::shared_state::get_shared_program_mut()
                    .ok_or_else(|| anyhow::anyhow!("run_module_startups: program not available (no async context)"))?;
                let program = program_lock.read()
                    .map_err(|_| anyhow::anyhow!("run_module_startups: could not acquire program read lock"))?;

                // Collect modules with startup blocks, sorted alphabetically
                let mut modules_with_startup: Vec<(String, crate::ast::LifecycleBlock)> = program.modules.iter()
                    .filter_map(|m| m.startup.as_ref().map(|s| (m.name.clone(), s.clone())))
                    .collect();
                modules_with_startup.sort_by(|a, b| a.0.cmp(&b.0));

                if modules_with_startup.is_empty() {
                    return Ok(Some(Value::string("no modules have startup blocks".to_string())));
                }

                let mut results = Vec::new();
                for (module_name, startup_fn) in &modules_with_startup {
                    eprintln!("[run_module_startups] executing {}.startup", module_name);
                    // Create a child env with coroutine handle and module name
                    let mut startup_env = crate::eval::Env::new_with_shared_interner(&program.shared_interner);
                    startup_env.populate_shared_from_program(&program);
                    // Pass along the coroutine handle from the caller
                    startup_env.set("__coroutine_handle", Value::CoroutineHandle(self.clone()));
                    startup_env.set("__module_name", Value::String(std::sync::Arc::new(module_name.clone())));
                    match crate::eval::eval_function_body_named(
                        &program,
                        &format!("{}.startup", module_name),
                        &startup_fn.body,
                        &mut startup_env,
                    ) {
                        Ok(val) => {
                            let msg = format!("{}.startup -> {}", module_name, val);
                            eprintln!("[run_module_startups] {}", msg);
                            results.push(msg);
                        }
                        Err(e) => {
                            let msg = format!("{}.startup error: {}", module_name, e);
                            eprintln!("[run_module_startups] {}", msg);
                            results.push(msg);
                        }
                    }
                }

                // Also auto-register module-level source declarations
                let modules_with_sources: Vec<(String, Vec<crate::ast::SourceDecl>)> = program.modules.iter()
                    .filter(|m| !m.sources.is_empty())
                    .map(|m| (m.name.clone(), m.sources.clone()))
                    .collect();
                drop(program); // release read lock before IO calls

                for (module_name, sources) in modules_with_sources {
                    for src in &sources {
                        let interval_ms = src.config.iter()
                            .find(|(k, _)| k == "interval")
                            .and_then(|(_, v)| v.parse::<u64>().ok());
                        let handler = if src.handler.contains('.') {
                            src.handler.clone()
                        } else {
                            format!("{}.{}", module_name, src.handler)
                        };
                        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
                        let _ = self.io_tx.blocking_send(IoRequest::SourceAdd {
                            module_name: module_name.clone(),
                            source_type: src.source_type.clone(),
                            interval_ms,
                            alias: src.name.clone(),
                            handler,
                            reply: reply_tx,
                        });
                        match reply_rx.blocking_recv() {
                            Ok(Ok(msg)) => results.push(format!("source {}.{}: {}", module_name, src.name, msg)),
                            Ok(Err(e)) => results.push(format!("source {}.{} error: {}", module_name, src.name, e)),
                            Err(_) => results.push(format!("source {}.{}: channel closed", module_name, src.name)),
                        }
                    }
                }

                let count = modules_with_startup.len();
                return Ok(Some(Value::string(format!("executed {} startup(s): {}", count, results.join("; ")))));
            }

            "query_startups" => {
                let program = crate::shared_state::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_startups: program not available"))?;

                let mut lines = Vec::new();
                for module in &program.modules {
                    if let Some(ref startup) = module.startup {
                        let effects = startup.effects.iter()
                            .map(|e| format!("{e:?}").to_lowercase())
                            .collect::<Vec<_>>()
                            .join(",");
                        let stmt_count = startup.body.len();
                        lines.push(format!("{}: +startup [{}] ({} statement{})",
                            module.name, effects, stmt_count,
                            if stmt_count == 1 { "" } else { "s" }));
                    }
                    if let Some(ref shutdown) = module.shutdown {
                        let effects = shutdown.effects.iter()
                            .map(|e| format!("{e:?}").to_lowercase())
                            .collect::<Vec<_>>()
                            .join(",");
                        let stmt_count = shutdown.body.len();
                        lines.push(format!("{}: +shutdown [{}] ({} statement{})",
                            module.name, effects, stmt_count,
                            if stmt_count == 1 { "" } else { "s" }));
                    }
                }

                if lines.is_empty() {
                    return Ok(Some(Value::string("No modules have startup or shutdown blocks.".to_string())));
                }
                return Ok(Some(Value::string(lines.join("\n"))));
            }

            // ── Mutation operations — write to program AST via thread-local ──
            "mutate" => {
                let code = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    Some(other) => format!("{other}"),
                    None => bail!("mutate expects (code:String)"),
                };
                if code.trim().is_empty() {
                    bail!("mutate: code must not be empty");
                }
                let program_lock = crate::shared_state::get_shared_program_mut()
                    .ok_or_else(|| anyhow::anyhow!("mutate: program not available (no async context)"))?;
                let mut program = program_lock.write()
                    .map_err(|_| anyhow::anyhow!("mutate: could not acquire program write lock"))?;

                let operations = crate::parser::parse(&code)
                    .map_err(|e| anyhow::anyhow!("mutate: parse error: {e}"))?;

                let mut applied = 0usize;
                let mut messages = Vec::new();
                for op in &operations {
                    // Skip non-mutation operations (tests, evals, queries, etc.)
                    match op {
                        crate::parser::Operation::Test(_)
                        | crate::parser::Operation::Trace(_)
                        | crate::parser::Operation::Eval(_)
                        | crate::parser::Operation::Query(_) => continue,
                        _ => {}
                    }
                    match crate::validator::apply_and_validate(&mut program, op) {
                        Ok(msg) => {
                            applied += 1;
                            messages.push(msg);
                        }
                        Err(e) => {
                            bail!("mutate: error applying operation {}: {e}", applied + 1);
                        }
                    }
                }
                // Update the read-only snapshot so query builtins see the changes
                crate::shared_state::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                if let Some(rt) = crate::shared_state::get_shared_runtime() {
                    crate::eval::init_missing_shared_runtime_vars(&program, &rt);
                }
                let summary = if applied == 1 {
                    format!("Applied 1 mutation: {}", messages[0])
                } else {
                    format!("Applied {applied} mutations")
                };
                return Ok(Some(Value::string(summary)));
            }
            "fn_remove" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("fn_remove expects (name:String)"),
                };
                let program_lock = crate::shared_state::get_shared_program_mut()
                    .ok_or_else(|| anyhow::anyhow!("fn_remove: program not available (no async context)"))?;
                let mut program = program_lock.write()
                    .map_err(|_| anyhow::anyhow!("fn_remove: could not acquire program write lock"))?;

                if let Some((mod_name, fn_name)) = name.split_once('.') {
                    // Remove function from module
                    if let Some(m) = program.modules.iter_mut().find(|m| m.name == mod_name) {
                        let before = m.functions.len();
                        m.functions.retain(|f| f.name != fn_name);
                        if m.functions.len() < before {
                            program.rebuild_function_index();
                            crate::shared_state::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                            return Ok(Some(Value::string(format!("Removed {name}"))));
                        }
                        bail!("fn_remove: function `{fn_name}` not found in module `{mod_name}`");
                    }
                    bail!("fn_remove: module `{mod_name}` not found");
                }
                // Top-level function removal
                if let Some(pos) = program.functions.iter().position(|f| f.name == name) {
                    program.functions.remove(pos);
                    program.rebuild_function_index();
                    crate::shared_state::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                    return Ok(Some(Value::string(format!("Removed {name}"))));
                }
                bail!("fn_remove: function `{name}` not found");
            }
            "type_remove" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("type_remove expects (name:String)"),
                };
                let program_lock = crate::shared_state::get_shared_program_mut()
                    .ok_or_else(|| anyhow::anyhow!("type_remove: program not available (no async context)"))?;
                let mut program = program_lock.write()
                    .map_err(|_| anyhow::anyhow!("type_remove: could not acquire program write lock"))?;

                if let Some((mod_name, type_name)) = name.split_once('.') {
                    // Remove type from module
                    if let Some(m) = program.modules.iter_mut().find(|m| m.name == mod_name) {
                        let before = m.types.len();
                        m.types.retain(|t| t.name() != type_name);
                        if m.types.len() < before {
                            crate::shared_state::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                            return Ok(Some(Value::string(format!("Removed {name}"))));
                        }
                        bail!("type_remove: type `{type_name}` not found in module `{mod_name}`");
                    }
                    bail!("type_remove: module `{mod_name}` not found");
                }
                // Top-level type removal
                let name_str: &str = &name;
                if let Some(pos) = program.types.iter().position(|t| t.name() == name_str) {
                    program.types.remove(pos);
                    crate::shared_state::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                    return Ok(Some(Value::string(format!("Removed {name}"))));
                }
                bail!("type_remove: type `{name}` not found");
            }
            "module_remove" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("module_remove expects (name:String)"),
                };
                let program_lock = crate::shared_state::get_shared_program_mut()
                    .ok_or_else(|| anyhow::anyhow!("module_remove: program not available (no async context)"))?;
                let mut program = program_lock.write()
                    .map_err(|_| anyhow::anyhow!("module_remove: could not acquire program write lock"))?;

                if let Some(pos) = program.modules.iter().position(|m| m.name == name) {
                    program.modules.remove(pos);
                    program.rebuild_function_index();
                    crate::shared_state::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                    return Ok(Some(Value::string(format!("Removed module {name}"))));
                }
                bail!("module_remove: module `{name}` not found");
            }

            // ── move_symbols — programmatic !move ──
            "move_symbols" => {
                let symbols_str = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    Some(other) => format!("{other}"),
                    None => bail!("move_symbols expects (symbols:String, target_module:String)"),
                };
                let target_module = match args.get(1) {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    Some(other) => format!("{other}"),
                    None => bail!("move_symbols expects (symbols:String, target_module:String)"),
                };
                if symbols_str.trim().is_empty() {
                    bail!("move_symbols: symbols must not be empty");
                }
                if target_module.trim().is_empty() {
                    bail!("move_symbols: target_module must not be empty");
                }
                let names: Vec<String> = symbols_str
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                if names.is_empty() {
                    bail!("move_symbols: no valid symbol names found in '{symbols_str}'");
                }

                let program_lock = crate::shared_state::get_shared_program_mut()
                    .ok_or_else(|| anyhow::anyhow!("move_symbols: program not available (no async context)"))?;
                let mut program = program_lock.write()
                    .map_err(|_| anyhow::anyhow!("move_symbols: could not acquire program write lock"))?;

                let result = crate::validator::apply_move(&mut program, &names, &target_module)?;
                // Update read-only snapshot
                crate::shared_state::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                return Ok(Some(Value::string(result)));
            }

            // ── trace_run — programmatic !trace ──
            "trace_run" => {
                let fn_name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("trace_run expects (fn_name:String, args:String)"),
                };
                let args_str = match args.get(1) {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    Some(other) => format!("{other}"),
                    None => String::new(),
                };
                let program = crate::shared_state::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("trace_run: program not available (no async context)"))?;

                let input_expr = if args_str.trim().is_empty() {
                    crate::parser::Expr::StructLiteral(vec![])
                } else {
                    crate::parser::parse_test_input(0, args_str.trim())
                        .map_err(|e| anyhow::anyhow!("trace_run: failed to parse args: {e}"))?
                };

                let steps = crate::eval::trace_function(&program, &fn_name, &input_expr)?;
                let output = steps.iter()
                    .map(|s| format!("{s}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                return Ok(Some(Value::string(
                    if output.is_empty() {
                        format!("Trace of {fn_name}: (no steps)")
                    } else {
                        format!("Trace of {fn_name}:\n{output}")
                    },
                )));
            }

            // ── msg_send — programmatic !msg ──
            "msg_send" => {
                let target = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("msg_send expects (target:String, message:String)"),
                };
                let message = match args.get(1) {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    Some(other) => format!("{other}"),
                    None => bail!("msg_send expects (target:String, message:String)"),
                };
                if target.trim().is_empty() {
                    bail!("msg_send: target must not be empty");
                }
                let rt = crate::shared_state::get_shared_runtime()
                    .ok_or_else(|| anyhow::anyhow!("msg_send: no runtime available"))?;
                let mut state = rt.write()
                    .map_err(|_| anyhow::anyhow!("msg_send: could not access runtime"))?;

                let timestamp = format!("{}s", std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs());
                let msg = crate::session::AgentMessage {
                    from: "main".to_string(),
                    to: target.clone(),
                    content: message.clone(),
                    timestamp,
                };
                if let Some(meta) = crate::shared_state::get_shared_meta() {
                    if let Ok(mut meta) = meta.lock() {
                        meta.agent_mailbox
                            .entry(target.clone())
                            .or_default()
                            .push(msg.clone());
                    }
                }
                state.agent_mailbox
                    .entry(target.clone())
                    .or_default()
                    .push(msg);

                return Ok(Some(Value::string(format!("Message sent to '{target}'"))));
            }

            // ── inbox_read — programmatic inbox drain ──
            "inbox_read" => {
                if !args.is_empty() {
                    bail!("inbox_read expects no arguments");
                }

                let mut messages = if let Some(meta) = crate::shared_state::get_shared_meta() {
                    meta.lock()
                        .map_err(|_| anyhow::anyhow!("inbox_read: could not lock meta"))?
                        .agent_mailbox
                        .remove("main")
                        .unwrap_or_default()
                } else {
                    let rt = crate::shared_state::get_shared_runtime()
                        .ok_or_else(|| anyhow::anyhow!("inbox_read: no runtime available"))?;
                    rt.write()
                        .map_err(|_| anyhow::anyhow!("inbox_read: could not access runtime"))?
                        .agent_mailbox
                        .remove("main")
                        .unwrap_or_default()
                };

                if let Some(rt) = crate::shared_state::get_shared_runtime() {
                    if let Ok(mut state) = rt.write() {
                        state.agent_mailbox.remove("main");
                    }
                }

                let contents: Vec<String> = messages.drain(..).map(|msg| msg.content).collect();
                let payload = serde_json::to_string(&contents)
                    .map_err(|e| anyhow::anyhow!("inbox_read: failed to serialize inbox: {e}"))?;
                return Ok(Some(Value::string(payload)));
            }

            // ── watch_start — programmatic !watch ──
            "watch_start" => {
                let fn_name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("watch_start expects (fn_name:String, interval_ms:Int)"),
                };
                let interval_ms = match args.get(1) {
                    Some(Value::Int(n)) => *n,
                    _ => bail!("watch_start expects (fn_name:String, interval_ms:Int)"),
                };
                if fn_name.trim().is_empty() {
                    bail!("watch_start: fn_name must not be empty");
                }
                if interval_ms <= 0 {
                    bail!("watch_start: interval_ms must be > 0");
                }
                // Verify the function exists
                let program = crate::shared_state::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("watch_start: program not available (no async context)"))?;
                if program.get_function(&fn_name).is_none() {
                    bail!("watch_start: function `{fn_name}` not found");
                }

                // Queue the watch command for API-layer processing
                let cmd = format!("!watch {fn_name} {interval_ms}");
                let rt = crate::shared_state::get_shared_runtime()
                    .ok_or_else(|| anyhow::anyhow!("watch_start: no runtime available"))?;
                rt.write()
                    .map_err(|_| anyhow::anyhow!("watch_start: could not access runtime"))?
                    .pending_commands.push(cmd);

                return Ok(Some(Value::string(format!("Watching {fn_name} every {interval_ms}ms (queued)"))));
            }

            // ── agent_spawn — programmatic !agent ──
            "agent_spawn" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("agent_spawn expects (name:String, scope:String, task:String)"),
                };
                let scope = match args.get(1) {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("agent_spawn expects (name:String, scope:String, task:String)"),
                };
                let task = match args.get(2) {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("agent_spawn expects (name:String, scope:String, task:String)"),
                };
                if name.trim().is_empty() {
                    bail!("agent_spawn: name must not be empty");
                }
                if task.trim().is_empty() {
                    bail!("agent_spawn: task must not be empty");
                }

                // Queue the agent command for API-layer processing
                let cmd = format!("!agent {name} --scope {scope}\n{task}");
                let rt = crate::shared_state::get_shared_runtime()
                    .ok_or_else(|| anyhow::anyhow!("agent_spawn: no runtime available"))?;
                rt.write()
                    .map_err(|_| anyhow::anyhow!("agent_spawn: could not access runtime"))?
                    .pending_commands.push(cmd);

                return Ok(Some(Value::string(format!("Agent '{name}' spawned (scope: {scope})"))));
            }

            // ── route_list — list registered HTTP routes ──
            "route_list" => {
                let routes = crate::shared_state::get_shared_runtime()
                    .and_then(|rt| rt.read().ok().map(|state| state.http_routes.clone()))
                    .unwrap_or_default();
                if routes.is_empty() {
                    return Ok(Some(Value::string("No routes registered.")));
                }
                let mut out = String::new();
                for r in &routes {
                    out.push_str(&format!("{} {} -> `{}`\n", r.method, r.path, r.handler_fn));
                }
                return Ok(Some(Value::string(out.trim_end().to_string())));
            }

            // ── route_add — register an HTTP route ──
            "route_add" => {
                let method = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("route_add expects (method:String, path:String, handler:String)"),
                };
                let path = match args.get(1) {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("route_add expects (method:String, path:String, handler:String)"),
                };
                let handler = match args.get(2) {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("route_add expects (method:String, path:String, handler:String)"),
                };
                let method_upper = method.to_uppercase();
                if !matches!(method_upper.as_str(), "GET" | "POST" | "PUT" | "DELETE" | "PATCH") {
                    bail!("route_add: method must be GET, POST, PUT, DELETE, or PATCH (got '{method}')");
                }
                if !path.starts_with('/') {
                    bail!("route_add: path must start with '/' (got '{path}')");
                }
                let rt = crate::shared_state::get_shared_runtime()
                    .ok_or_else(|| anyhow::anyhow!("route_add: no runtime available"))?;
                let mut state = rt.write()
                    .map_err(|_| anyhow::anyhow!("route_add: could not access runtime"))?;
                // Upsert: update if method+path already exists
                if let Some(existing) = state.http_routes.iter_mut()
                    .find(|r| r.method == method_upper && r.path == path)
                {
                    let old_fn = existing.handler_fn.clone();
                    existing.handler_fn = handler.clone();
                    return Ok(Some(Value::string(format!("updated route {method_upper} {path} -> `{handler}` (was `{old_fn}`)"))));
                }
                state.http_routes.push(crate::ast::HttpRoute {
                    method: method_upper.clone(),
                    path: path.clone(),
                    handler_fn: handler.clone(),
                });
                return Ok(Some(Value::string(format!("added route {method_upper} {path} -> `{handler}`"))));
            }

            // ── route_remove — remove an HTTP route by method+path ──
            "route_remove" => {
                let method = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("route_remove expects (method:String, path:String)"),
                };
                let path = match args.get(1) {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("route_remove expects (method:String, path:String)"),
                };
                let method_upper = method.to_uppercase();
                let rt = crate::shared_state::get_shared_runtime()
                    .ok_or_else(|| anyhow::anyhow!("route_remove: no runtime available"))?;
                let mut state = rt.write()
                    .map_err(|_| anyhow::anyhow!("route_remove: could not access runtime"))?;
                let before = state.http_routes.len();
                let mut removed_handler = None;
                state.http_routes.retain(|r| {
                    if r.method == method_upper && r.path == path {
                        removed_handler = Some(r.handler_fn.clone());
                        false
                    } else {
                        true
                    }
                });
                if state.http_routes.len() < before {
                    return Ok(Some(Value::string(format!(
                        "removed route {} {} (was -> `{}`)",
                        method_upper, path, removed_handler.unwrap_or_default()
                    ))));
                }
                bail!("route_remove: no route found for {method_upper} {path}");
            }

            // ── undo — revert the last mutation ──
            "undo" => {
                let rt = crate::shared_state::get_shared_runtime()
                    .ok_or_else(|| anyhow::anyhow!("undo: no runtime available"))?;
                rt.write()
                    .map_err(|_| anyhow::anyhow!("undo: could not access runtime"))?
                    .pending_commands.push("!undo".to_string());
                return Ok(Some(Value::string("Undo queued — will revert last mutation")));
            }

            // ── sandbox_enter — enter sandbox mode ──
            "sandbox_enter" => {
                let rt = crate::shared_state::get_shared_runtime()
                    .ok_or_else(|| anyhow::anyhow!("sandbox_enter: no runtime available"))?;
                rt.write()
                    .map_err(|_| anyhow::anyhow!("sandbox_enter: could not access runtime"))?
                    .pending_commands.push("!sandbox enter".to_string());
                return Ok(Some(Value::string("Sandbox enter queued — mutations will be isolated")));
            }

            // ── sandbox_merge — merge sandbox changes ──
            "sandbox_merge" => {
                let rt = crate::shared_state::get_shared_runtime()
                    .ok_or_else(|| anyhow::anyhow!("sandbox_merge: no runtime available"))?;
                rt.write()
                    .map_err(|_| anyhow::anyhow!("sandbox_merge: could not access runtime"))?
                    .pending_commands.push("!sandbox merge".to_string());
                return Ok(Some(Value::string("Sandbox merge queued — changes will be kept")));
            }

            // ── sandbox_discard — discard sandbox changes ──
            "sandbox_discard" => {
                let rt = crate::shared_state::get_shared_runtime()
                    .ok_or_else(|| anyhow::anyhow!("sandbox_discard: no runtime available"))?;
                rt.write()
                    .map_err(|_| anyhow::anyhow!("sandbox_discard: could not access runtime"))?
                    .pending_commands.push("!sandbox discard".to_string());
                return Ok(Some(Value::string("Sandbox discard queued — changes will be reverted")));
            }

            // ── mock_set — register an IO mock response ──
            "mock_set" => {
                let operation = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("mock_set expects (operation:String, pattern:String, response:String)"),
                };
                let pattern = match args.get(1) {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("mock_set expects (operation:String, pattern:String, response:String)"),
                };
                let response = match args.get(2) {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("mock_set expects (operation:String, pattern:String, response:String)"),
                };
                if operation.trim().is_empty() {
                    bail!("mock_set: operation must not be empty");
                }
                let patterns: Vec<String> = pattern.split_whitespace().map(|s| s.to_string()).collect();
                let meta = crate::shared_state::get_shared_meta()
                    .ok_or_else(|| anyhow::anyhow!("mock_set: no meta available"))?;
                meta.lock()
                    .map_err(|_| anyhow::anyhow!("mock_set: could not lock meta"))?
                    .io_mocks.push(crate::session::IoMock {
                        operation: operation.clone(),
                        patterns: patterns.clone(),
                        response: response.clone(),
                    });
                let pattern_display = if patterns.is_empty() { "*".to_string() } else { patterns.join(" ") };
                return Ok(Some(Value::string(format!(
                    "mock: {operation} {pattern_display} -> \"{}\"",
                    response.chars().take(50).collect::<String>()
                ))));
            }

            // ── mock_clear — clear all IO mocks ──
            "mock_clear" => {
                let meta = crate::shared_state::get_shared_meta()
                    .ok_or_else(|| anyhow::anyhow!("mock_clear: no meta available"))?;
                let count = {
                    let mut m = meta.lock()
                        .map_err(|_| anyhow::anyhow!("mock_clear: could not lock meta"))?;
                    let count = m.io_mocks.len();
                    m.io_mocks.clear();
                    count
                };
                return Ok(Some(Value::string(format!("cleared {count} mocks"))));
            }

            // ── sse_send — send event to SSE listeners ──
            "sse_send" | "sse_broadcast" => {
                let event_type = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("sse_send expects (event_type:String, data:String)"),
                };
                let data = match args.get(1) {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    Some(other) => format!("{other}"),
                    None => bail!("sse_send expects (event_type:String, data:String)"),
                };
                if event_type.trim().is_empty() {
                    bail!("sse_send: event_type must not be empty");
                }
                let sender = crate::shared_state::get_shared_event_broadcast()
                    .ok_or_else(|| anyhow::anyhow!("sse_send: no event broadcast available"))?;
                let payload = serde_json::json!({"type": event_type, "data": data}).to_string();
                sender.send(payload)
                    .map_err(|e| anyhow::anyhow!("sse_send: failed to send event: {e}"))?;
                return Ok(Some(Value::string("sent")));
            }

            // ── module_create — create/switch to a module ──
            "module_create" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("module_create expects (name:String)"),
                };
                if name.trim().is_empty() {
                    bail!("module_create: name must not be empty");
                }
                // Check if the first character is uppercase (module naming convention)
                if !name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    bail!("module_create: module name must start with an uppercase letter (got '{name}')");
                }
                let program_lock = crate::shared_state::get_shared_program_mut()
                    .ok_or_else(|| anyhow::anyhow!("module_create: program not available (no async context)"))?;
                let mut program = program_lock.write()
                    .map_err(|_| anyhow::anyhow!("module_create: could not acquire program write lock"))?;
                // Check if module already exists
                if program.modules.iter().any(|m| m.name == name) {
                    return Ok(Some(Value::string(format!("module '{name}' already exists"))));
                }
                // Create empty module
                let code = format!("!module {name}");
                let operations = crate::parser::parse(&code)
                    .map_err(|e| anyhow::anyhow!("module_create: parse error: {e}"))?;
                for op in &operations {
                    crate::validator::apply_and_validate(&mut program, op)
                        .map_err(|e| anyhow::anyhow!("module_create: {e}"))?;
                }
                crate::shared_state::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                return Ok(Some(Value::string(format!("created module '{name}'"))));
            }

            // ── test_run — run stored tests for a function ──
            "test_run" => {
                let fn_name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("test_run expects (fn_name:String)"),
                };
                let program = crate::shared_state::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("test_run: program not available"))?;
                let func = program.get_function(&fn_name)
                    .ok_or_else(|| anyhow::anyhow!("test_run: function `{fn_name}` not found"))?;
                let ast_cases = func.tests.clone();
                if ast_cases.is_empty() {
                    return Ok(Some(Value::string(format!("no stored tests for `{fn_name}`"))));
                }
                // Reconstruct test source from stored AST test cases
                let bare = fn_name.rsplit('.').next().unwrap_or(&fn_name);
                let mut test_src = format!("!test {bare}\n");
                for case in &ast_cases {
                    let expect_str = crate::session::reconstruct_expect_pub(&case.expected, case.matcher.as_deref());
                    test_src.push_str(&format!("  +with {} -> expect {}\n", case.input, expect_str));
                    for ac in &case.after_checks {
                        test_src.push_str(&format!("  +after {} {} \"{}\"\n", ac.target, ac.matcher, ac.value));
                    }
                }
                // Get IO mocks from meta and routes from runtime for test execution
                let io_mocks = crate::shared_state::get_shared_meta()
                    .and_then(|meta| meta.lock().ok().map(|m| m.io_mocks.clone()))
                    .unwrap_or_default();
                let http_routes = crate::shared_state::get_shared_runtime()
                    .and_then(|rt| rt.read().ok().map(|state| state.http_routes.clone()))
                    .unwrap_or_default();
                // Parse and run the reconstructed test
                let ops = crate::parser::parse(&test_src)
                    .map_err(|e| anyhow::anyhow!("test_run: failed to reconstruct tests: {e}"))?;
                let mut results = Vec::new();
                for test_op in &ops {
                    if let crate::parser::Operation::Test(test) = test_op {
                        for case in &test.cases {
                            match crate::eval::eval_test_case_with_mocks(
                                &program, &test.function_name, case, &io_mocks, &http_routes,
                            ) {
                                Ok(msg) => results.push(format!("PASS: {msg}")),
                                Err(e) => results.push(format!("FAIL: {e}")),
                            }
                        }
                    }
                }
                return Ok(Some(Value::string(results.join("\n"))));
            }

            // ── fn_replace — replace a statement in a function ──
            "fn_replace" => {
                let target = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("fn_replace expects (target:String, new_code:String)"),
                };
                let new_code = match args.get(1) {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("fn_replace expects (target:String, new_code:String)"),
                };
                if target.trim().is_empty() {
                    bail!("fn_replace: target must not be empty");
                }
                if new_code.trim().is_empty() {
                    bail!("fn_replace: new_code must not be empty");
                }
                let program_lock = crate::shared_state::get_shared_program_mut()
                    .ok_or_else(|| anyhow::anyhow!("fn_replace: program not available (no async context)"))?;
                let mut program = program_lock.write()
                    .map_err(|_| anyhow::anyhow!("fn_replace: could not acquire program write lock"))?;
                // Build !replace source and parse it
                let replace_src = format!("!replace {target}\n{new_code}\n+end");
                let operations = crate::parser::parse(&replace_src)
                    .map_err(|e| anyhow::anyhow!("fn_replace: parse error: {e}"))?;
                let mut result_msg = String::new();
                for op in &operations {
                    match crate::validator::apply_and_validate(&mut program, op) {
                        Ok(msg) => result_msg = msg,
                        Err(e) => bail!("fn_replace: {e}"),
                    }
                }
                crate::shared_state::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                return Ok(Some(Value::string(result_msg)));
            }

            _ => return Ok(None), // not a local op
        }
    }

    /// Channel-based IO operations dispatched to the async runtime via `send_and_wait`.
    /// Handles TCP, file, shell, HTTP, LLM, and other operations that require the
    /// tokio IO loop.
    fn execute_transport_op(&self, op: &str, args: &[Value]) -> Result<Value> {
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
                return Ok(Value::string(data));
            }
            "tcp_write" => {
                let conn = match &args[0] {
                    Value::Int(h) => *h,
                    _ => bail!("tcp_write expects handle"),
                };
                let data = match &args[1] {
                    Value::String(s) => s.as_ref().clone(),
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
                    Value::String(s) => s.as_ref().clone(),
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
                    match &args[0] { Value::String(s) => s.as_ref().clone(), other => format!("{other}") }
                };
                let (tx, rx) = oneshot::channel();
                let line = self.send_and_wait(WaitReason::StdinRead, IoRequest::StdinReadLine { prompt, reply: tx }, rx)?;
                return Ok(Value::string(line));
            }
            "print" => {
                let text = match &args[0] { Value::String(s) => s.as_ref().clone(), other => format!("{other}") };
                let (tx, rx) = oneshot::channel();
                self.send_and_wait(WaitReason::Running, IoRequest::Print { text, newline: false, reply: tx }, rx)?;
                return Ok(Value::Int(0));
            }
            "println" => {
                let text = match &args[0] { Value::String(s) => s.as_ref().clone(), other => format!("{other}") };
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
                let path = match &args[0] { Value::String(s) => s.as_ref().clone(), _ => bail!("file_read expects String path") };
                let (tx, rx) = oneshot::channel();
                let contents = self.send_and_wait(WaitReason::FileRead(path.clone()), IoRequest::FileRead { path, reply: tx }, rx)?;
                return Ok(Value::string(contents));
            }
            "file_write" | "write_file" => {
                let path = match &args[0] { Value::String(s) => s.as_ref().clone(), _ => bail!("file_write expects String path") };
                let data = match &args[1] { Value::String(s) => s.as_ref().clone(), other => format!("{other}") };
                let (tx, rx) = oneshot::channel();
                self.send_and_wait(WaitReason::FileWrite(path.clone()), IoRequest::FileWrite { path, data, reply: tx }, rx)?;
                return Ok(Value::string("OK"));
            }
            "file_exists" => {
                let path = match &args[0] { Value::String(s) => s.as_ref().clone(), _ => bail!("file_exists expects String path") };
                let (tx, rx) = oneshot::channel();
                let exists = self.send_and_wait(WaitReason::Running, IoRequest::FileExists { path, reply: tx }, rx)?;
                return Ok(Value::Bool(exists));
            }
            "list_dir" => {
                let path = match &args[0] { Value::String(s) => s.as_ref().clone(), _ => bail!("list_dir expects String path") };
                let (tx, rx) = oneshot::channel();
                let names = self.send_and_wait(WaitReason::Running, IoRequest::ListDir { path, reply: tx }, rx)?;
                return Ok(Value::list(names.into_iter().map(Value::string).collect()));
            }
            "shell_exec" | "exec" => {
                let command = match &args[0] { Value::String(s) => s.as_ref().clone(), _ => bail!("shell_exec expects String command") };
                let (tx, rx) = oneshot::channel();
                let (stdout, stderr, code) = self.send_and_wait(
                    WaitReason::ShellExec(command.chars().take(40).collect()),
                    IoRequest::ShellExec { command, reply: tx }, rx,
                )?;
                if code == 0 { return Ok(Value::string(stdout)); }
                else { return Ok(Value::string(format!("EXIT {code}: {stderr}"))); }
            }
            "self_restart" | "restart" => {
                let (tx, rx) = oneshot::channel();
                self.send_and_wait(WaitReason::Running, IoRequest::SelfRestart { reply: tx }, rx)?;
                return Ok(Value::string("restarting..."));
            }
            "http_get" => {
                let url = match args.get(0) { Some(Value::String(s)) => s.as_ref().clone(), _ => bail!("http_get expects (url:String)") };
                let (tx, rx) = oneshot::channel();
                let result = self.send_and_wait(WaitReason::HttpGet(url.clone()), IoRequest::HttpGet { url, reply: tx }, rx)?;
                return Ok(Value::string(result));
            }
            "http_post" => {
                let url = match args.get(0) { Some(Value::String(s)) => s.as_ref().clone(), _ => bail!("http_post expects (url:String, body:String, content_type:String)") };
                let body = match args.get(1) { Some(Value::String(s)) => s.as_ref().clone(), _ => bail!("http_post expects (url:String, body:String, content_type:String)") };
                let content_type = match args.get(2) { Some(Value::String(s)) => s.as_ref().clone(), _ => "application/json".to_string() };
                let (tx, rx) = oneshot::channel();
                let result = self.send_and_wait(WaitReason::HttpPost(url.clone()), IoRequest::HttpPost { url, body, content_type, reply: tx }, rx)?;
                return Ok(Value::string(result));
            }
            "http_upload" => {
                let url = match args.get(0) { Some(Value::String(s)) => s.as_ref().clone(), _ => bail!("http_upload expects (url:String, file_path:String, file_field:String[, extra_fields:String])") };
                let file_path = match args.get(1) { Some(Value::String(s)) => s.as_ref().clone(), _ => bail!("http_upload expects (url:String, file_path:String, file_field:String[, extra_fields:String])") };
                let file_field = match args.get(2) { Some(Value::String(s)) => s.as_ref().clone(), _ => "file".to_string() };
                // Parse extra_fields from "key1=val1&key2=val2" format
                let extra_fields: Vec<(String, String)> = match args.get(3) {
                    Some(Value::String(s)) if !s.is_empty() => {
                        s.split('&').filter_map(|pair| {
                            let mut parts = pair.splitn(2, '=');
                            match (parts.next(), parts.next()) {
                                (Some(k), Some(v)) => Some((k.to_string(), v.to_string())),
                                _ => None,
                            }
                        }).collect()
                    }
                    _ => vec![],
                };
                let (tx, rx) = oneshot::channel();
                let result = self.send_and_wait(
                    WaitReason::HttpPost(url.clone()),
                    IoRequest::HttpUpload { url, file_path, file_field, extra_fields, reply: tx },
                    rx,
                )?;
                return Ok(Value::string(result));
            }
            "llm_call" => {
                let system = match args.get(0) { Some(Value::String(s)) => s.as_ref().clone(), _ => bail!("llm_call expects (system:String, prompt:String[, model:String])") };
                let prompt = match args.get(1) { Some(Value::String(s)) => s.as_ref().clone(), _ => bail!("llm_call expects (system:String, prompt:String[, model:String])") };
                let model = args.get(2).and_then(|v| match v { Value::String(s) => Some(s.as_ref().clone()), _ => None });
                let (tx, rx) = oneshot::channel();
                let result = self.send_and_wait(WaitReason::LlmCall, IoRequest::LlmCall { model, system, prompt, reply: tx }, rx)?;
                return Ok(Value::string(result));
            }
            "llm_agent" => {
                let system = match args.get(0) { Some(Value::String(s)) => s.as_ref().clone(), _ => bail!("llm_agent expects (system:String, task:String[, model:String])") };
                let task = match args.get(1) { Some(Value::String(s)) => s.as_ref().clone(), _ => bail!("llm_agent expects (system:String, task:String[, model:String])") };
                let model = args.get(2).and_then(|v| match v { Value::String(s) => Some(s.as_ref().clone()), _ => None });
                let (tx, rx) = oneshot::channel();
                let result = self.send_and_wait(WaitReason::LlmAgent, IoRequest::LlmAgent { model, system, task, reply: tx }, rx)?;
                return Ok(Value::string(result));
            }
            "llm_takeover" => {
                let context = match args.get(0) { Some(Value::String(s)) => s.as_ref().clone(), _ => bail!("llm_takeover expects (context:String, message:String[, reply_fn:String, reply_arg:String])") };
                let message = match args.get(1) { Some(Value::String(s)) => s.as_ref().clone(), _ => bail!("llm_takeover expects (context:String, message:String[, reply_fn:String, reply_arg:String])") };
                let reply_fn = args.get(2).and_then(|v| match v { Value::String(s) => Some(s.as_ref().clone()), _ => None });
                let reply_arg = args.get(3).and_then(|v| match v { Value::String(s) => Some(s.as_ref().clone()), _ => None });
                let (tx, rx) = oneshot::channel();
                let result = self.send_and_wait(
                    WaitReason::LlmTakeover(context.clone()),
                    IoRequest::LlmTakeover { context, message, reply_fn, reply_arg, reply: tx },
                    rx,
                )?;
                return Ok(Value::string(result));
            }
            _ => bail!("unknown await operation: {op}"),
        }
    }
}

#[cfg(test)]
mod tests;
