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
                // Use args[0] instead of current_exe() because after cargo rebuild
                // the old inode is deleted and /proc/self/exe shows "(deleted)".
                let exe = std::env::args().next()
                    .map(std::path::PathBuf::from)
                    .and_then(|p| std::fs::canonicalize(&p).ok().or(Some(p)))
                    .unwrap_or_else(|| std::env::current_exe().unwrap_or_default());
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
    /// Shared snapshot registry for live interpreter state inspection.
    snapshot_registry: Option<TaskSnapshotRegistry>,
    /// Mock responses for testing — if set, IO calls check here first.
    mocks: Option<Vec<crate::session::IoMock>>,
}

impl CoroutineHandle {
    pub fn new(io_tx: mpsc::Sender<IoRequest>) -> Self {
        Self { io_tx, task_id: None, task_registry: None, snapshot_registry: None, mocks: None }
    }

    pub fn new_with_task(
        io_tx: mpsc::Sender<IoRequest>,
        task_id: TaskId,
        registry: TaskRegistry,
        snapshot_registry: TaskSnapshotRegistry,
    ) -> Self {
        Self { io_tx, task_id: Some(task_id), task_registry: Some(registry), snapshot_registry: Some(snapshot_registry), mocks: None }
    }

    /// Create a mock handle for testing — no real IO, returns mock responses.
    /// Unmatched operations error with "no mock for..." since there's no real
    /// IO sender to fall through to.
    #[allow(dead_code)]
    pub fn new_mock(mocks: Vec<crate::session::IoMock>) -> Self {
        let (tx, _) = mpsc::channel(1); // dummy channel, never used
        Self { io_tx: tx, task_id: None, task_registry: None, snapshot_registry: None, mocks: Some(mocks) }
    }

    /// Create a handle with mocks AND a real IO sender — mocks are checked first,
    /// unmatched operations fall through to real IO via the sender.
    pub fn new_mock_with_sender(mocks: Vec<crate::session::IoMock>, io_tx: mpsc::Sender<IoRequest>) -> Self {
        Self { io_tx, task_id: None, task_registry: None, snapshot_registry: None, mocks: Some(mocks) }
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
    pub fn update_snapshot(
        &self,
        function_name: &str,
        current_stmt_id: Option<String>,
        frame_depth: usize,
        env: &crate::eval::Env,
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
                locals: env.snapshot_bindings(),
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

    /// Execute an await operation — sends IO request and blocks until result.
    /// This is called from the synchronous evaluator, so we use block_on
    /// within a spawn_blocking context.
    pub fn execute_await(&self, op: &str, args: &[Value]) -> Result<Value> {
        // ── Roadmap operations — handled locally, no IO channel needed ──
        // These run before mock/IO dispatch since they only access the
        // thread-local SharedRuntime and never touch the IO channel.
        match op {
            "roadmap_list" => {
                let result = crate::eval::get_shared_runtime()
                    .and_then(|rt| rt.read().ok().map(|state| {
                        let items: Vec<String> = state.roadmap.iter().enumerate().map(|(i, item)| {
                            format!("{} {}: {}", if item.done { "[x]" } else { "[ ]" }, i + 1, item.description)
                        }).collect();
                        if items.is_empty() {
                            "Roadmap is empty.".to_string()
                        } else {
                            format!("Roadmap:\n{}", items.join("\n"))
                        }
                    }))
                    .unwrap_or_else(|| "Roadmap is empty.".to_string());
                return Ok(Value::String(result));
            }
            "roadmap_add" => {
                let desc = match args.first() {
                    Some(Value::String(s)) => s.clone(),
                    Some(other) => format!("{other}"),
                    None => bail!("roadmap_add expects (description:String)"),
                };
                if desc.trim().is_empty() {
                    bail!("roadmap_add: description must not be empty");
                }
                if let Some(rt) = crate::eval::get_shared_runtime() {
                    if let Ok(mut state) = rt.write() {
                        state.roadmap.push(crate::session::RoadmapItem {
                            description: desc.clone(),
                            done: false,
                        });
                    }
                }
                return Ok(Value::String(desc));
            }
            "roadmap_done" => {
                let n = match args.first() {
                    Some(Value::Int(n)) => *n,
                    _ => bail!("roadmap_done expects (n:Int)"),
                };
                if n < 1 {
                    bail!("roadmap_done: index must be >= 1");
                }
                let result = crate::eval::get_shared_runtime()
                    .ok_or_else(|| anyhow::anyhow!("roadmap_done: no runtime available"))
                    .and_then(|rt| {
                        rt.write()
                            .map_err(|_| anyhow::anyhow!("roadmap_done: could not access runtime"))
                            .and_then(|mut state| {
                                let idx = (n as usize).saturating_sub(1);
                                if idx < state.roadmap.len() {
                                    state.roadmap[idx].done = true;
                                    Ok(format!("Roadmap: #{n} done."))
                                } else {
                                    Err(anyhow::anyhow!("roadmap_done: item #{n} not found (roadmap has {} items)", state.roadmap.len()))
                                }
                            })
                    })?;
                return Ok(Value::String(result));
            }

            // ── Plan operations — handled locally, no IO channel needed ──
            "plan_show" => {
                let result = crate::eval::get_shared_runtime()
                    .and_then(|rt| rt.read().ok().map(|state| {
                        let steps: Vec<String> = state.plan.iter().enumerate().map(|(i, step)| {
                            let icon = match step.status {
                                crate::session::PlanStatus::Pending => "[ ]",
                                crate::session::PlanStatus::InProgress => "[~]",
                                crate::session::PlanStatus::Done => "[x]",
                                crate::session::PlanStatus::Failed => "[!]",
                            };
                            format!("{} {}: {}", icon, i + 1, step.description)
                        }).collect();
                        if steps.is_empty() {
                            "No plan set.".to_string()
                        } else {
                            steps.join("\n")
                        }
                    }))
                    .unwrap_or_else(|| "No plan set.".to_string());
                return Ok(Value::String(result));
            }
            "plan_set" => {
                let steps_str = match args.first() {
                    Some(Value::String(s)) => s.clone(),
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
                let count = descriptions.len();
                if let Some(rt) = crate::eval::get_shared_runtime() {
                    if let Ok(mut state) = rt.write() {
                        state.plan = descriptions
                            .into_iter()
                            .map(|d| crate::session::PlanStep {
                                description: d,
                                status: crate::session::PlanStatus::Pending,
                            })
                            .collect();
                    }
                }
                return Ok(Value::String(format!("Plan set with {count} steps.")));
            }
            "plan_done" => {
                let n = match args.first() {
                    Some(Value::Int(n)) => *n,
                    _ => bail!("plan_done expects (n:Int)"),
                };
                if n < 1 {
                    bail!("plan_done: index must be >= 1");
                }
                let result = crate::eval::get_shared_runtime()
                    .ok_or_else(|| anyhow::anyhow!("plan_done: no runtime available"))
                    .and_then(|rt| {
                        rt.write()
                            .map_err(|_| anyhow::anyhow!("plan_done: could not access runtime"))
                            .and_then(|mut state| {
                                let idx = (n as usize).saturating_sub(1);
                                if idx < state.plan.len() {
                                    state.plan[idx].status = crate::session::PlanStatus::Done;
                                    Ok(format!("Plan: step {n} done."))
                                } else {
                                    Err(anyhow::anyhow!("plan_done: step {n} not found (plan has {} steps)", state.plan.len()))
                                }
                            })
                    })?;
                return Ok(Value::String(result));
            }
            "plan_fail" => {
                let n = match args.first() {
                    Some(Value::Int(n)) => *n,
                    _ => bail!("plan_fail expects (n:Int)"),
                };
                if n < 1 {
                    bail!("plan_fail: index must be >= 1");
                }
                let result = crate::eval::get_shared_runtime()
                    .ok_or_else(|| anyhow::anyhow!("plan_fail: no runtime available"))
                    .and_then(|rt| {
                        rt.write()
                            .map_err(|_| anyhow::anyhow!("plan_fail: could not access runtime"))
                            .and_then(|mut state| {
                                let idx = (n as usize).saturating_sub(1);
                                if idx < state.plan.len() {
                                    state.plan[idx].status = crate::session::PlanStatus::Failed;
                                    Ok(format!("Plan: step {n} failed."))
                                } else {
                                    Err(anyhow::anyhow!("plan_fail: step {n} not found (plan has {} steps)", state.plan.len()))
                                }
                            })
                    })?;
                return Ok(Value::String(result));
            }
            // ── Query operations — access program AST via thread-local ──
            // These reuse the same logic as the ? query commands in typeck.rs.
            "query_symbols" | "symbols_list" => {
                let program = crate::eval::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_symbols: program not available (no async context)"))?;
                let table = crate::typeck::build_symbol_table(&program);
                let result = crate::typeck::handle_query(&program, &table, "?symbols", &[]);
                return Ok(Value::String(result));
            }
            "query_symbols_detail" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.clone(),
                    _ => bail!("query_symbols_detail expects (name:String)"),
                };
                let program = crate::eval::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_symbols_detail: program not available"))?;
                let table = crate::typeck::build_symbol_table(&program);
                let query = format!("?symbols {name}");
                let result = crate::typeck::handle_query(&program, &table, &query, &[]);
                return Ok(Value::String(result));
            }
            "query_source" | "source_get" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.clone(),
                    _ => bail!("query_source expects (name:String)"),
                };
                let program = crate::eval::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_source: program not available"))?;
                let table = crate::typeck::build_symbol_table(&program);
                let query = format!("?source {name}");
                let result = crate::typeck::handle_query(&program, &table, &query, &[]);
                return Ok(Value::String(result));
            }
            "query_callers" | "callers_get" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.clone(),
                    _ => bail!("query_callers expects (name:String)"),
                };
                let program = crate::eval::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_callers: program not available"))?;
                let table = crate::typeck::build_symbol_table(&program);
                let query = format!("?callers {name}");
                let result = crate::typeck::handle_query(&program, &table, &query, &[]);
                return Ok(Value::String(result));
            }
            "query_callees" | "callees_get" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.clone(),
                    _ => bail!("query_callees expects (name:String)"),
                };
                let program = crate::eval::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_callees: program not available"))?;
                let table = crate::typeck::build_symbol_table(&program);
                let query = format!("?callees {name}");
                let result = crate::typeck::handle_query(&program, &table, &query, &[]);
                return Ok(Value::String(result));
            }
            "query_deps" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.clone(),
                    _ => bail!("query_deps expects (name:String)"),
                };
                let program = crate::eval::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_deps: program not available"))?;
                let table = crate::typeck::build_symbol_table(&program);
                let query = format!("?deps {name}");
                let result = crate::typeck::handle_query(&program, &table, &query, &[]);
                return Ok(Value::String(result));
            }
            "query_deps_all" | "deps_get" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.clone(),
                    _ => bail!("query_deps_all expects (name:String)"),
                };
                let program = crate::eval::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_deps_all: program not available"))?;
                let table = crate::typeck::build_symbol_table(&program);
                let query = format!("?deps-all {name}");
                let result = crate::typeck::handle_query(&program, &table, &query, &[]);
                return Ok(Value::String(result));
            }
            "query_routes" | "routes_list" => {
                let program = crate::eval::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_routes: program not available"))?;
                // Get HTTP routes from SharedRuntime
                let routes = crate::eval::get_shared_runtime()
                    .and_then(|rt| rt.read().ok().map(|state| state.http_routes.clone()))
                    .unwrap_or_default();
                let table = crate::typeck::build_symbol_table(&program);
                let result = crate::typeck::handle_query(&program, &table, "?routes", &routes);
                return Ok(Value::String(result));
            }
            "query_tasks" => {
                let result = if let Some(reg) = &self.task_registry {
                    crate::api::format_tasks(&Some(reg.clone()))
                } else {
                    "No task registry (not in async context).".to_string()
                };
                return Ok(Value::String(result));
            }
            "query_library" => {
                let program = crate::eval::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_library: program not available"))?;
                let result = crate::library::query_library(&program, None);
                return Ok(Value::String(result));
            }

            // ── Mutation operations — write to program AST via thread-local ──
            "mutate" => {
                let code = match args.first() {
                    Some(Value::String(s)) => s.clone(),
                    Some(other) => format!("{other}"),
                    None => bail!("mutate expects (code:String)"),
                };
                if code.trim().is_empty() {
                    bail!("mutate: code must not be empty");
                }
                let program_lock = crate::eval::get_shared_program_mut()
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
                crate::eval::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                let summary = if applied == 1 {
                    format!("Applied 1 mutation: {}", messages[0])
                } else {
                    format!("Applied {applied} mutations")
                };
                return Ok(Value::String(summary));
            }
            "fn_remove" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.clone(),
                    _ => bail!("fn_remove expects (name:String)"),
                };
                let program_lock = crate::eval::get_shared_program_mut()
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
                            crate::eval::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                            return Ok(Value::String(format!("Removed {name}")));
                        }
                        bail!("fn_remove: function `{fn_name}` not found in module `{mod_name}`");
                    }
                    bail!("fn_remove: module `{mod_name}` not found");
                }
                // Top-level function removal
                if let Some(pos) = program.functions.iter().position(|f| f.name == name) {
                    program.functions.remove(pos);
                    program.rebuild_function_index();
                    crate::eval::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                    return Ok(Value::String(format!("Removed {name}")));
                }
                bail!("fn_remove: function `{name}` not found");
            }
            "type_remove" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.clone(),
                    _ => bail!("type_remove expects (name:String)"),
                };
                let program_lock = crate::eval::get_shared_program_mut()
                    .ok_or_else(|| anyhow::anyhow!("type_remove: program not available (no async context)"))?;
                let mut program = program_lock.write()
                    .map_err(|_| anyhow::anyhow!("type_remove: could not acquire program write lock"))?;

                if let Some((mod_name, type_name)) = name.split_once('.') {
                    // Remove type from module
                    if let Some(m) = program.modules.iter_mut().find(|m| m.name == mod_name) {
                        let before = m.types.len();
                        m.types.retain(|t| t.name() != type_name);
                        if m.types.len() < before {
                            crate::eval::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                            return Ok(Value::String(format!("Removed {name}")));
                        }
                        bail!("type_remove: type `{type_name}` not found in module `{mod_name}`");
                    }
                    bail!("type_remove: module `{mod_name}` not found");
                }
                // Top-level type removal
                let name_str: &str = &name;
                if let Some(pos) = program.types.iter().position(|t| t.name() == name_str) {
                    program.types.remove(pos);
                    crate::eval::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                    return Ok(Value::String(format!("Removed {name}")));
                }
                bail!("type_remove: type `{name}` not found");
            }
            "module_remove" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.clone(),
                    _ => bail!("module_remove expects (name:String)"),
                };
                let program_lock = crate::eval::get_shared_program_mut()
                    .ok_or_else(|| anyhow::anyhow!("module_remove: program not available (no async context)"))?;
                let mut program = program_lock.write()
                    .map_err(|_| anyhow::anyhow!("module_remove: could not acquire program write lock"))?;

                if let Some(pos) = program.modules.iter().position(|m| m.name == name) {
                    program.modules.remove(pos);
                    program.rebuild_function_index();
                    crate::eval::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                    return Ok(Value::String(format!("Removed module {name}")));
                }
                bail!("module_remove: module `{name}` not found");
            }

            _ => {} // fall through to mock/IO dispatch
        }

        // Check mock table first — if a mock matches, return it without real IO
        if let Some(mocks) = &self.mocks {
            let arg_strs: Vec<String> = args.iter().map(|a| format!("{a}")).collect();
            let arg_str = arg_strs.join(" ");
            'mock_loop: for mock in mocks {
                if mock.operation != op {
                    continue;
                }
                // Match each pattern against the corresponding arg position.
                // Single-pattern mocks match against the joined arg string (backward compat).
                if mock.patterns.len() == 1 {
                    if arg_str.contains(&mock.patterns[0]) {
                        return Ok(Value::String(mock.response.clone()));
                    }
                } else {
                    // Multi-pattern: each pattern must match the corresponding arg
                    if mock.patterns.len() > arg_strs.len() {
                        continue;
                    }
                    for (pat, arg) in mock.patterns.iter().zip(arg_strs.iter()) {
                        if !arg.contains(pat) {
                            continue 'mock_loop;
                        }
                    }
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
                return Ok(Value::String("OK".to_string()));
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: set up a SharedRuntime with an empty roadmap and install it as
    /// the thread-local, returning the handle and runtime for assertions.
    fn setup_roadmap_runtime() -> (CoroutineHandle, crate::session::SharedRuntime) {
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState {
            http_routes: vec![],
            shared_vars: std::collections::HashMap::new(),
            roadmap: vec![],
            plan: vec![],
        }));
        crate::eval::set_shared_runtime(Some(rt.clone()));
        let handle = CoroutineHandle::new_mock(vec![]);
        (handle, rt)
    }

    fn unwrap_string(v: Value) -> String {
        match v {
            Value::String(s) => s,
            other => panic!("expected String, got {other}"),
        }
    }

    #[test]
    fn roadmap_list_empty() {
        let (handle, _rt) = setup_roadmap_runtime();
        let result = unwrap_string(handle.execute_await("roadmap_list", &[]).unwrap());
        assert_eq!(result, "Roadmap is empty.");
    }

    #[test]
    fn roadmap_add_and_list() {
        let (handle, rt) = setup_roadmap_runtime();

        // Add an item
        let result = unwrap_string(
            handle.execute_await("roadmap_add", &[Value::String("Build feature X".into())]).unwrap()
        );
        assert_eq!(result, "Build feature X");

        // Verify it's in the runtime state
        assert_eq!(rt.read().unwrap().roadmap.len(), 1);
        assert_eq!(rt.read().unwrap().roadmap[0].description, "Build feature X");
        assert!(!rt.read().unwrap().roadmap[0].done);

        // List should show it
        let list = unwrap_string(handle.execute_await("roadmap_list", &[]).unwrap());
        assert!(list.contains("Build feature X"), "list should contain the item: {list}");
        assert!(list.contains("[ ] 1:"), "item should be unchecked: {list}");
    }

    #[test]
    fn roadmap_add_empty_fails() {
        let (handle, _rt) = setup_roadmap_runtime();
        let result = handle.execute_await("roadmap_add", &[Value::String("  ".into())]);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("must not be empty"),
            "should reject empty description"
        );
    }

    #[test]
    fn roadmap_add_no_args_fails() {
        let (handle, _rt) = setup_roadmap_runtime();
        let result = handle.execute_await("roadmap_add", &[]);
        assert!(result.is_err());
    }

    #[test]
    fn roadmap_done_marks_item() {
        let (handle, rt) = setup_roadmap_runtime();

        // Add two items
        handle
            .execute_await("roadmap_add", &[Value::String("Item A".into())])
            .unwrap();
        handle
            .execute_await("roadmap_add", &[Value::String("Item B".into())])
            .unwrap();

        // Mark item 2 as done
        let result = unwrap_string(
            handle.execute_await("roadmap_done", &[Value::Int(2)]).unwrap()
        );
        assert!(result.contains("#2 done"), "confirmation: {result}");

        // Verify state
        assert!(!rt.read().unwrap().roadmap[0].done);
        assert!(rt.read().unwrap().roadmap[1].done);

        // List should show [x] for item 2
        let list = unwrap_string(handle.execute_await("roadmap_list", &[]).unwrap());
        assert!(list.contains("[ ] 1: Item A"), "A unchecked: {list}");
        assert!(list.contains("[x] 2: Item B"), "B checked: {list}");
    }

    #[test]
    fn roadmap_done_out_of_bounds() {
        let (handle, _rt) = setup_roadmap_runtime();
        handle
            .execute_await("roadmap_add", &[Value::String("Only item".into())])
            .unwrap();

        let result = handle.execute_await("roadmap_done", &[Value::Int(5)]);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("not found"),
            "should error on out-of-bounds index"
        );
    }

    #[test]
    fn roadmap_done_zero_index_fails() {
        let (handle, _rt) = setup_roadmap_runtime();
        let result = handle.execute_await("roadmap_done", &[Value::Int(0)]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains(">= 1"));
    }

    #[test]
    fn roadmap_done_wrong_type_fails() {
        let (handle, _rt) = setup_roadmap_runtime();
        let result = handle.execute_await("roadmap_done", &[Value::String("1".into())]);
        assert!(result.is_err());
    }

    // ═════════════════════════════════════════════════════════════════════
    // Plan IO builtins
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn plan_show_empty() {
        let (handle, _rt) = setup_roadmap_runtime();
        let result = unwrap_string(handle.execute_await("plan_show", &[]).unwrap());
        assert_eq!(result, "No plan set.");
    }

    #[test]
    fn plan_set_and_show() {
        let (handle, rt) = setup_roadmap_runtime();

        let result = unwrap_string(
            handle.execute_await("plan_set", &[Value::String("Parse input\nValidate data\nStore results".into())]).unwrap()
        );
        assert_eq!(result, "Plan set with 3 steps.");

        // Verify state
        assert_eq!(rt.read().unwrap().plan.len(), 3);
        assert_eq!(rt.read().unwrap().plan[0].description, "Parse input");
        assert_eq!(rt.read().unwrap().plan[1].description, "Validate data");
        assert_eq!(rt.read().unwrap().plan[2].description, "Store results");

        // Show should list all steps
        let show = unwrap_string(handle.execute_await("plan_show", &[]).unwrap());
        assert!(show.contains("[ ] 1: Parse input"), "step 1: {show}");
        assert!(show.contains("[ ] 2: Validate data"), "step 2: {show}");
        assert!(show.contains("[ ] 3: Store results"), "step 3: {show}");
    }

    #[test]
    fn plan_set_empty_fails() {
        let (handle, _rt) = setup_roadmap_runtime();
        let result = handle.execute_await("plan_set", &[Value::String("  \n  \n  ".into())]);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("must not be empty"),
            "should reject empty steps"
        );
    }

    #[test]
    fn plan_set_no_args_fails() {
        let (handle, _rt) = setup_roadmap_runtime();
        let result = handle.execute_await("plan_set", &[]);
        assert!(result.is_err());
    }

    #[test]
    fn plan_set_skips_blank_lines() {
        let (handle, rt) = setup_roadmap_runtime();

        let result = unwrap_string(
            handle.execute_await("plan_set", &[Value::String("Step A\n\n  \nStep B".into())]).unwrap()
        );
        assert_eq!(result, "Plan set with 2 steps.");
        assert_eq!(rt.read().unwrap().plan.len(), 2);
        assert_eq!(rt.read().unwrap().plan[0].description, "Step A");
        assert_eq!(rt.read().unwrap().plan[1].description, "Step B");
    }

    #[test]
    fn plan_done_marks_step() {
        let (handle, rt) = setup_roadmap_runtime();

        handle.execute_await("plan_set", &[Value::String("Alpha\nBravo".into())]).unwrap();

        let result = unwrap_string(
            handle.execute_await("plan_done", &[Value::Int(2)]).unwrap()
        );
        assert_eq!(result, "Plan: step 2 done.");

        // Verify state
        assert_eq!(rt.read().unwrap().plan[0].status, crate::session::PlanStatus::Pending);
        assert_eq!(rt.read().unwrap().plan[1].status, crate::session::PlanStatus::Done);

        // Show should reflect [x]
        let show = unwrap_string(handle.execute_await("plan_show", &[]).unwrap());
        assert!(show.contains("[ ] 1: Alpha"), "Alpha pending: {show}");
        assert!(show.contains("[x] 2: Bravo"), "Bravo done: {show}");
    }

    #[test]
    fn plan_done_out_of_bounds() {
        let (handle, _rt) = setup_roadmap_runtime();
        handle.execute_await("plan_set", &[Value::String("Only step".into())]).unwrap();

        let result = handle.execute_await("plan_done", &[Value::Int(5)]);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("not found"),
            "should error on out-of-bounds index"
        );
    }

    #[test]
    fn plan_done_zero_index_fails() {
        let (handle, _rt) = setup_roadmap_runtime();
        let result = handle.execute_await("plan_done", &[Value::Int(0)]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains(">= 1"));
    }

    #[test]
    fn plan_done_wrong_type_fails() {
        let (handle, _rt) = setup_roadmap_runtime();
        let result = handle.execute_await("plan_done", &[Value::String("1".into())]);
        assert!(result.is_err());
    }

    #[test]
    fn plan_fail_marks_step() {
        let (handle, rt) = setup_roadmap_runtime();

        handle.execute_await("plan_set", &[Value::String("First\nSecond\nThird".into())]).unwrap();

        let result = unwrap_string(
            handle.execute_await("plan_fail", &[Value::Int(1)]).unwrap()
        );
        assert_eq!(result, "Plan: step 1 failed.");

        // Verify state
        assert_eq!(rt.read().unwrap().plan[0].status, crate::session::PlanStatus::Failed);
        assert_eq!(rt.read().unwrap().plan[1].status, crate::session::PlanStatus::Pending);

        // Show should reflect [!]
        let show = unwrap_string(handle.execute_await("plan_show", &[]).unwrap());
        assert!(show.contains("[!] 1: First"), "First failed: {show}");
        assert!(show.contains("[ ] 2: Second"), "Second pending: {show}");
    }

    #[test]
    fn plan_fail_out_of_bounds() {
        let (handle, _rt) = setup_roadmap_runtime();
        handle.execute_await("plan_set", &[Value::String("Only step".into())]).unwrap();

        let result = handle.execute_await("plan_fail", &[Value::Int(3)]);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("not found"),
            "should error on out-of-bounds index"
        );
    }

    #[test]
    fn plan_fail_zero_index_fails() {
        let (handle, _rt) = setup_roadmap_runtime();
        let result = handle.execute_await("plan_fail", &[Value::Int(0)]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains(">= 1"));
    }

    #[test]
    fn plan_fail_wrong_type_fails() {
        let (handle, _rt) = setup_roadmap_runtime();
        let result = handle.execute_await("plan_fail", &[Value::String("1".into())]);
        assert!(result.is_err());
    }

    #[test]
    fn plan_set_replaces_existing() {
        let (handle, rt) = setup_roadmap_runtime();

        handle.execute_await("plan_set", &[Value::String("Old step 1\nOld step 2".into())]).unwrap();
        assert_eq!(rt.read().unwrap().plan.len(), 2);

        handle.execute_await("plan_set", &[Value::String("New step".into())]).unwrap();
        assert_eq!(rt.read().unwrap().plan.len(), 1);
        assert_eq!(rt.read().unwrap().plan[0].description, "New step");
    }

    // ═════════════════════════════════════════════════════════════════════
    // Query IO builtins
    // ═════════════════════════════════════════════════════════════════════

    /// Helper: build a program from Adapsis source and install it as the thread-local.
    fn setup_query_runtime(source: &str) -> CoroutineHandle {
        let ops = crate::parser::parse(source).expect("parse failed");
        let mut program = crate::ast::Program::default();
        for op in &ops {
            match op {
                crate::parser::Operation::Test(_) | crate::parser::Operation::Eval(_) => {}
                _ => {
                    crate::validator::apply_and_validate(&mut program, op)
                        .expect("validation failed");
                }
            }
        }
        program.rebuild_function_index();
        crate::eval::set_shared_program(Some(std::sync::Arc::new(program)));

        // Also set up a runtime for query_routes/query_tasks
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState {
            http_routes: vec![],
            shared_vars: std::collections::HashMap::new(),
            roadmap: vec![],
            plan: vec![],
        }));
        crate::eval::set_shared_runtime(Some(rt));

        CoroutineHandle::new_mock(vec![])
    }

    #[test]
    fn query_symbols_empty_program() {
        let handle = setup_query_runtime("");
        let result = unwrap_string(handle.execute_await("query_symbols", &[]).unwrap());
        assert!(result.contains("Types:"), "should contain Types header: {result}");
        assert!(result.contains("Functions:"), "should contain Functions header: {result}");
    }

    #[test]
    fn query_symbols_with_function() {
        let handle = setup_query_runtime(
            "+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end"
        );
        let result = unwrap_string(handle.execute_await("query_symbols", &[]).unwrap());
        assert!(result.contains("greet"), "should list greet function: {result}");
    }

    #[test]
    fn query_symbols_detail_existing() {
        let handle = setup_query_runtime(
            "+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end"
        );
        let result = unwrap_string(
            handle.execute_await("query_symbols_detail", &[Value::String("greet".into())]).unwrap()
        );
        assert!(result.contains("greet"), "should show greet details: {result}");
        assert!(result.contains("params") || result.contains("String"),
            "should show parameter info: {result}");
    }

    #[test]
    fn query_symbols_detail_not_found() {
        let handle = setup_query_runtime("");
        let result = unwrap_string(
            handle.execute_await("query_symbols_detail", &[Value::String("nonexistent".into())]).unwrap()
        );
        assert!(result.contains("not found"), "should say not found: {result}");
    }

    #[test]
    fn query_symbols_detail_wrong_type_fails() {
        let handle = setup_query_runtime("");
        let result = handle.execute_await("query_symbols_detail", &[Value::Int(42)]);
        assert!(result.is_err(), "should fail with non-String arg");
    }

    #[test]
    fn query_source_existing_function() {
        let handle = setup_query_runtime(
            "+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end"
        );
        let result = unwrap_string(
            handle.execute_await("query_source", &[Value::String("greet".into())]).unwrap()
        );
        assert!(result.contains("+fn greet"), "should contain function definition: {result}");
        assert!(result.contains("concat"), "should contain function body: {result}");
    }

    #[test]
    fn query_source_not_found() {
        let handle = setup_query_runtime("");
        let result = unwrap_string(
            handle.execute_await("query_source", &[Value::String("missing".into())]).unwrap()
        );
        assert!(result.contains("not found"), "should say not found: {result}");
    }

    #[test]
    fn query_source_wrong_type_fails() {
        let handle = setup_query_runtime("");
        let result = handle.execute_await("query_source", &[Value::Int(1)]);
        assert!(result.is_err(), "should fail with non-String arg");
    }

    #[test]
    fn query_callers_no_callers() {
        let handle = setup_query_runtime(
            "+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end"
        );
        let result = unwrap_string(
            handle.execute_await("query_callers", &[Value::String("greet".into())]).unwrap()
        );
        assert!(result.contains("no callers"), "should say no callers: {result}");
    }

    #[test]
    fn query_callers_with_caller() {
        let handle = setup_query_runtime(
            "+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end\n\
             +fn main ()->String\n  +let x:String = greet(\"world\")\n  +return x\n+end"
        );
        let result = unwrap_string(
            handle.execute_await("query_callers", &[Value::String("greet".into())]).unwrap()
        );
        assert!(result.contains("main"), "should list main as caller: {result}");
    }

    #[test]
    fn query_callees_lists_calls() {
        let handle = setup_query_runtime(
            "+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end\n\
             +fn main ()->String\n  +let x:String = greet(\"world\")\n  +return x\n+end"
        );
        let result = unwrap_string(
            handle.execute_await("query_callees", &[Value::String("main".into())]).unwrap()
        );
        assert!(result.contains("greet"), "should list greet as callee: {result}");
    }

    #[test]
    fn query_callees_wrong_type_fails() {
        let handle = setup_query_runtime("");
        let result = handle.execute_await("query_callees", &[Value::Int(1)]);
        assert!(result.is_err(), "should fail with non-String arg");
    }

    #[test]
    fn query_deps_same_as_callees() {
        let handle = setup_query_runtime(
            "+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end\n\
             +fn main ()->String\n  +let x:String = greet(\"world\")\n  +return x\n+end"
        );
        let result = unwrap_string(
            handle.execute_await("query_deps", &[Value::String("main".into())]).unwrap()
        );
        assert!(result.contains("greet"), "should list greet as dependency: {result}");
    }

    #[test]
    fn query_deps_all_transitive() {
        let handle = setup_query_runtime(
            "+fn a ()->String\n  +return \"hello\"\n+end\n\
             +fn b ()->String\n  +let x:String = a()\n  +return x\n+end\n\
             +fn c ()->String\n  +let x:String = b()\n  +return x\n+end"
        );
        let result = unwrap_string(
            handle.execute_await("query_deps_all", &[Value::String("c".into())]).unwrap()
        );
        // c -> b -> a, so both a and b should appear
        assert!(result.contains("a"), "should include transitive dep 'a': {result}");
        assert!(result.contains("b"), "should include direct dep 'b': {result}");
    }

    #[test]
    fn query_deps_all_no_deps() {
        let handle = setup_query_runtime(
            "+fn a ()->String\n  +return \"hello\"\n+end"
        );
        let result = unwrap_string(
            handle.execute_await("query_deps_all", &[Value::String("a".into())]).unwrap()
        );
        assert!(result.contains("no dependencies"), "should say no dependencies: {result}");
    }

    #[test]
    fn query_routes_empty() {
        let handle = setup_query_runtime("");
        let result = unwrap_string(handle.execute_await("query_routes", &[]).unwrap());
        assert!(result.contains("No HTTP routes"), "should say no routes: {result}");
    }

    #[test]
    fn query_routes_with_routes() {
        // Set up runtime with routes
        let program = crate::ast::Program::default();
        crate::eval::set_shared_program(Some(std::sync::Arc::new(program)));
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState {
            http_routes: vec![crate::ast::HttpRoute {
                method: "GET".to_string(),
                path: "/health".to_string(),
                handler_fn: "health_check".to_string(),
            }],
            shared_vars: std::collections::HashMap::new(),
            roadmap: vec![],
            plan: vec![],
        }));
        crate::eval::set_shared_runtime(Some(rt));
        let handle = CoroutineHandle::new_mock(vec![]);

        let result = unwrap_string(handle.execute_await("query_routes", &[]).unwrap());
        assert!(result.contains("GET"), "should contain GET method: {result}");
        assert!(result.contains("/health"), "should contain /health path: {result}");
        assert!(result.contains("health_check"), "should contain handler: {result}");
    }

    #[test]
    fn query_tasks_no_registry() {
        let handle = setup_query_runtime("");
        let result = unwrap_string(handle.execute_await("query_tasks", &[]).unwrap());
        // Mock handles don't have a task registry
        assert!(result.contains("No task registry") || result.contains("No tasks"),
            "should indicate no tasks available: {result}");
    }

    #[test]
    fn query_tasks_with_registry() {
        let registry: TaskRegistry = std::sync::Arc::new(std::sync::Mutex::new(HashMap::new()));
        // Add a task
        registry.lock().unwrap().insert(1, TaskInfo {
            id: 1,
            function_name: "my_task".to_string(),
            status: WaitReason::Running,
            started_at: "2025-01-01T00:00:00Z".to_string(),
        });

        let (tx, _) = mpsc::channel(1);
        let handle = CoroutineHandle {
            io_tx: tx,
            task_id: None,
            task_registry: Some(registry),
            snapshot_registry: None,
            mocks: Some(vec![]),
        };

        let result = unwrap_string(handle.execute_await("query_tasks", &[]).unwrap());
        assert!(result.contains("my_task"), "should show task: {result}");
        assert!(result.contains("running"), "should show status: {result}");
    }

    #[test]
    fn query_library_returns_string() {
        let handle = setup_query_runtime("");
        let result = unwrap_string(handle.execute_await("query_library", &[]).unwrap());
        assert!(result.contains("Module library"), "should contain library info: {result}");
    }

    #[test]
    fn query_no_program_errors() {
        // Clear the thread-local program
        crate::eval::set_shared_program(None);
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState {
            http_routes: vec![],
            shared_vars: std::collections::HashMap::new(),
            roadmap: vec![],
            plan: vec![],
        }));
        crate::eval::set_shared_runtime(Some(rt));
        let handle = CoroutineHandle::new_mock(vec![]);

        // All query builtins that need the program should error
        let result = handle.execute_await("query_symbols", &[]);
        assert!(result.is_err(), "query_symbols should fail without program");
        assert!(result.unwrap_err().to_string().contains("program not available"),
            "should mention program not available");

        let result = handle.execute_await("query_source", &[Value::String("x".into())]);
        assert!(result.is_err(), "query_source should fail without program");

        let result = handle.execute_await("query_library", &[]);
        assert!(result.is_err(), "query_library should fail without program");
    }

    // ═════════════════════════════════════════════════════════════════════
    // Mutation IO builtins
    // ═════════════════════════════════════════════════════════════════════

    /// Helper: build a program from Adapsis source and install both read-only
    /// and mutable program thread-locals. Returns the handle and the mutable
    /// program Arc for post-mutation assertions.
    fn setup_mutation_runtime(source: &str) -> (CoroutineHandle, std::sync::Arc<std::sync::RwLock<crate::ast::Program>>) {
        let ops = crate::parser::parse(source).expect("parse failed");
        let mut program = crate::ast::Program::default();
        for op in &ops {
            match op {
                crate::parser::Operation::Test(_) | crate::parser::Operation::Eval(_) => {}
                _ => {
                    crate::validator::apply_and_validate(&mut program, op)
                        .expect("validation failed");
                }
            }
        }
        program.rebuild_function_index();
        crate::eval::set_shared_program(Some(std::sync::Arc::new(program.clone())));

        let program_mut = std::sync::Arc::new(std::sync::RwLock::new(program));
        crate::eval::set_shared_program_mut(Some(program_mut.clone()));

        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState {
            http_routes: vec![],
            shared_vars: std::collections::HashMap::new(),
            roadmap: vec![],
            plan: vec![],
        }));
        crate::eval::set_shared_runtime(Some(rt));

        (CoroutineHandle::new_mock(vec![]), program_mut)
    }

    // ── mutate ──

    #[test]
    fn mutate_add_function() {
        let (handle, prog) = setup_mutation_runtime("");
        let code = "+fn hello ()->String\n  +return \"hi\"\n+end";
        let result = unwrap_string(
            handle.execute_await("mutate", &[Value::String(code.into())]).unwrap()
        );
        assert!(result.contains("Applied 1 mutation"), "should report 1 mutation: {result}");
        assert!(result.contains("hello"), "should mention function name: {result}");

        // Verify function was actually added
        let p = prog.read().unwrap();
        assert!(p.get_function("hello").is_some(), "hello function should exist in program");
    }

    #[test]
    fn mutate_add_module_with_functions() {
        let (handle, prog) = setup_mutation_runtime("");
        let code = "!module Greeter\n+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end";
        let result = unwrap_string(
            handle.execute_await("mutate", &[Value::String(code.into())]).unwrap()
        );
        assert!(result.contains("Applied"), "should report mutations: {result}");

        let p = prog.read().unwrap();
        assert!(p.get_function("Greeter.greet").is_some(), "Greeter.greet should exist");
    }

    #[test]
    fn mutate_empty_code_fails() {
        let (handle, _prog) = setup_mutation_runtime("");
        let result = handle.execute_await("mutate", &[Value::String("".into())]);
        assert!(result.is_err(), "mutate with empty code should fail");
        assert!(result.unwrap_err().to_string().contains("empty"), "error should mention empty");
    }

    #[test]
    fn mutate_invalid_syntax_fails() {
        let (handle, _prog) = setup_mutation_runtime("");
        let result = handle.execute_await("mutate", &[Value::String("+fn".into())]);
        assert!(result.is_err(), "mutate with invalid syntax should fail");
    }

    #[test]
    fn mutate_no_args_fails() {
        let (handle, _prog) = setup_mutation_runtime("");
        let result = handle.execute_await("mutate", &[]);
        assert!(result.is_err(), "mutate with no args should fail");
    }

    #[test]
    fn mutate_no_program_fails() {
        crate::eval::set_shared_program_mut(None);
        let handle = CoroutineHandle::new_mock(vec![]);
        let result = handle.execute_await("mutate", &[Value::String("+fn x ()->Int\n  +return 1\n+end".into())]);
        assert!(result.is_err(), "mutate without program should fail");
        assert!(result.unwrap_err().to_string().contains("program not available"));
    }

    // ── fn_remove ──

    #[test]
    fn fn_remove_top_level() {
        let (handle, prog) = setup_mutation_runtime(
            "+fn hello ()->String\n  +return \"hi\"\n+end"
        );
        // Verify function exists
        assert!(prog.read().unwrap().get_function("hello").is_some());

        let result = unwrap_string(
            handle.execute_await("fn_remove", &[Value::String("hello".into())]).unwrap()
        );
        assert_eq!(result, "Removed hello");

        // Verify function was removed
        assert!(prog.read().unwrap().get_function("hello").is_none());
    }

    #[test]
    fn fn_remove_from_module() {
        let (handle, prog) = setup_mutation_runtime(
            "!module MyMod\n+fn greet (name:String)->String\n  +return name\n+end"
        );
        assert!(prog.read().unwrap().get_function("MyMod.greet").is_some());

        let result = unwrap_string(
            handle.execute_await("fn_remove", &[Value::String("MyMod.greet".into())]).unwrap()
        );
        assert_eq!(result, "Removed MyMod.greet");
        assert!(prog.read().unwrap().get_function("MyMod.greet").is_none());
    }

    #[test]
    fn fn_remove_not_found() {
        let (handle, _prog) = setup_mutation_runtime("");
        let result = handle.execute_await("fn_remove", &[Value::String("nonexistent".into())]);
        assert!(result.is_err(), "fn_remove should fail for missing function");
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn fn_remove_wrong_type_fails() {
        let (handle, _prog) = setup_mutation_runtime("");
        let result = handle.execute_await("fn_remove", &[Value::Int(42)]);
        assert!(result.is_err(), "fn_remove should fail with non-String arg");
    }

    #[test]
    fn fn_remove_module_not_found() {
        let (handle, _prog) = setup_mutation_runtime("");
        let result = handle.execute_await("fn_remove", &[Value::String("NoModule.func".into())]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("module `NoModule` not found"));
    }

    // ── type_remove ──

    #[test]
    fn type_remove_top_level() {
        let (handle, prog) = setup_mutation_runtime(
            "+type Color = Red | Green | Blue"
        );
        assert!(!prog.read().unwrap().types.is_empty());

        let result = unwrap_string(
            handle.execute_await("type_remove", &[Value::String("Color".into())]).unwrap()
        );
        assert_eq!(result, "Removed Color");
        assert!(prog.read().unwrap().types.is_empty());
    }

    #[test]
    fn type_remove_not_found() {
        let (handle, _prog) = setup_mutation_runtime("");
        let result = handle.execute_await("type_remove", &[Value::String("Missing".into())]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn type_remove_wrong_type_fails() {
        let (handle, _prog) = setup_mutation_runtime("");
        let result = handle.execute_await("type_remove", &[Value::Int(1)]);
        assert!(result.is_err());
    }

    // ── module_remove ──

    #[test]
    fn module_remove_existing() {
        let (handle, prog) = setup_mutation_runtime(
            "!module MyMod\n+fn hello ()->String\n  +return \"hi\"\n+end"
        );
        assert!(!prog.read().unwrap().modules.is_empty());

        let result = unwrap_string(
            handle.execute_await("module_remove", &[Value::String("MyMod".into())]).unwrap()
        );
        assert_eq!(result, "Removed module MyMod");
        assert!(prog.read().unwrap().modules.is_empty());
        assert!(prog.read().unwrap().get_function("MyMod.hello").is_none());
    }

    #[test]
    fn module_remove_not_found() {
        let (handle, _prog) = setup_mutation_runtime("");
        let result = handle.execute_await("module_remove", &[Value::String("NoModule".into())]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn module_remove_wrong_type_fails() {
        let (handle, _prog) = setup_mutation_runtime("");
        let result = handle.execute_await("module_remove", &[Value::Int(1)]);
        assert!(result.is_err());
    }

    // ── Mutation builtins update read-only snapshot ──

    #[test]
    fn mutate_updates_query_snapshot() {
        let (handle, _prog) = setup_mutation_runtime("");

        // Add a function via mutate
        handle.execute_await("mutate", &[Value::String(
            "+fn test_func ()->Int\n  +return 42\n+end".into()
        )]).unwrap();

        // Query builtins should see the new function via updated snapshot
        let result = unwrap_string(
            handle.execute_await("query_symbols", &[]).unwrap()
        );
        assert!(result.contains("test_func"), "query_symbols should see mutated function: {result}");
    }

    #[test]
    fn fn_remove_updates_query_snapshot() {
        let (handle, _prog) = setup_mutation_runtime(
            "+fn to_remove ()->String\n  +return \"bye\"\n+end"
        );

        // Remove the function
        handle.execute_await("fn_remove", &[Value::String("to_remove".into())]).unwrap();

        // Query builtins should no longer see it
        let result = unwrap_string(
            handle.execute_await("query_symbols", &[]).unwrap()
        );
        assert!(!result.contains("to_remove"), "query_symbols should not see removed function: {result}");
    }

    // ═════════════════════════════════════════════════════════════════════
    // Query alias builtins — verify aliases work with Program access
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn symbols_list_alias_works() {
        let handle = setup_query_runtime(
            "+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end"
        );
        let result = unwrap_string(handle.execute_await("symbols_list", &[]).unwrap());
        assert!(result.contains("greet"), "symbols_list alias should list greet function: {result}");
    }

    #[test]
    fn source_get_alias_works() {
        let handle = setup_query_runtime(
            "+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end"
        );
        let result = unwrap_string(
            handle.execute_await("source_get", &[Value::String("greet".into())]).unwrap()
        );
        assert!(result.contains("+fn greet"), "source_get alias should return source: {result}");
    }

    #[test]
    fn callers_get_alias_works() {
        let handle = setup_query_runtime(
            "+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end\n\
             +fn main ()->String\n  +let x:String = greet(\"world\")\n  +return x\n+end"
        );
        let result = unwrap_string(
            handle.execute_await("callers_get", &[Value::String("greet".into())]).unwrap()
        );
        assert!(result.contains("main"), "callers_get alias should list main as caller: {result}");
    }

    #[test]
    fn callees_get_alias_works() {
        let handle = setup_query_runtime(
            "+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end\n\
             +fn main ()->String\n  +let x:String = greet(\"world\")\n  +return x\n+end"
        );
        let result = unwrap_string(
            handle.execute_await("callees_get", &[Value::String("main".into())]).unwrap()
        );
        assert!(result.contains("greet"), "callees_get alias should list greet as callee: {result}");
    }

    #[test]
    fn deps_get_alias_works() {
        let handle = setup_query_runtime(
            "+fn a ()->String\n  +return \"hello\"\n+end\n\
             +fn b ()->String\n  +let x:String = a()\n  +return x\n+end\n\
             +fn c ()->String\n  +let x:String = b()\n  +return x\n+end"
        );
        let result = unwrap_string(
            handle.execute_await("deps_get", &[Value::String("c".into())]).unwrap()
        );
        assert!(result.contains("a"), "deps_get alias should include transitive dep 'a': {result}");
        assert!(result.contains("b"), "deps_get alias should include direct dep 'b': {result}");
    }

    #[test]
    fn routes_list_alias_works() {
        let program = crate::ast::Program::default();
        crate::eval::set_shared_program(Some(std::sync::Arc::new(program)));
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState {
            http_routes: vec![crate::ast::HttpRoute {
                method: "POST".to_string(),
                path: "/api/data".to_string(),
                handler_fn: "handle_data".to_string(),
            }],
            shared_vars: std::collections::HashMap::new(),
            roadmap: vec![],
            plan: vec![],
        }));
        crate::eval::set_shared_runtime(Some(rt));
        let handle = CoroutineHandle::new_mock(vec![]);

        let result = unwrap_string(handle.execute_await("routes_list", &[]).unwrap());
        assert!(result.contains("POST"), "routes_list alias should contain POST method: {result}");
        assert!(result.contains("/api/data"), "routes_list alias should contain path: {result}");
    }

    #[test]
    fn alias_no_program_errors() {
        // Clear the thread-local program
        crate::eval::set_shared_program(None);
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState {
            http_routes: vec![],
            shared_vars: std::collections::HashMap::new(),
            roadmap: vec![],
            plan: vec![],
        }));
        crate::eval::set_shared_runtime(Some(rt));
        let handle = CoroutineHandle::new_mock(vec![]);

        // All alias builtins that need the program should error
        let result = handle.execute_await("symbols_list", &[]);
        assert!(result.is_err(), "symbols_list should fail without program");
        assert!(result.unwrap_err().to_string().contains("program not available"),
            "should mention program not available");

        let result = handle.execute_await("source_get", &[Value::String("x".into())]);
        assert!(result.is_err(), "source_get should fail without program");

        let result = handle.execute_await("callers_get", &[Value::String("x".into())]);
        assert!(result.is_err(), "callers_get should fail without program");

        let result = handle.execute_await("callees_get", &[Value::String("x".into())]);
        assert!(result.is_err(), "callees_get should fail without program");

        let result = handle.execute_await("deps_get", &[Value::String("x".into())]);
        assert!(result.is_err(), "deps_get should fail without program");

        let result = handle.execute_await("routes_list", &[]);
        assert!(result.is_err(), "routes_list should fail without program");
    }
}
