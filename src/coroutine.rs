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

    fn try_mock_io(&self, op: &str, args: &[Value], fail_on_missing: bool) -> Result<Option<Value>> {
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

    /// Execute an await operation — sends IO request and blocks until result.
    /// This is called from the synchronous evaluator, so we use block_on
    /// within a spawn_blocking context.
    pub fn execute_await(&self, op: &str, args: &[Value]) -> Result<Value> {
        if op != "mock_set" && op != "mock_clear" {
            if let Some(result) = self.try_mock_io(op, args, false)? {
                return Ok(result);
            }
        }

        // ── Roadmap and plan operations — handled locally via SHARED_META ──
        // These access SessionMeta directly (the single source of truth).
        // No IO channel needed; no more syncing between runtime and meta.
        match op {
            "roadmap_list" => {
                let result = crate::eval::get_shared_meta()
                    .and_then(|meta| meta.lock().ok().map(|m| crate::session::roadmap_list(&m.roadmap)))
                    .unwrap_or_else(|| "Roadmap is empty.".to_string());
                return Ok(Value::string(result));
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
                let result = crate::eval::get_shared_meta()
                    .ok_or_else(|| anyhow::anyhow!("roadmap_add: no meta available"))
                    .and_then(|meta| {
                        meta.lock()
                            .map_err(|_| anyhow::anyhow!("roadmap_add: could not lock meta"))
                            .map(|mut m| {
                                crate::session::roadmap_add(&mut m.roadmap, &desc);
                                desc.clone()
                            })
                    })?;
                return Ok(Value::string(result));
            }
            "roadmap_done" => {
                let n = match args.first() {
                    Some(Value::Int(n)) => *n,
                    _ => bail!("roadmap_done expects (n:Int)"),
                };
                if n < 1 {
                    bail!("roadmap_done: index must be >= 1");
                }
                let result = crate::eval::get_shared_meta()
                    .ok_or_else(|| anyhow::anyhow!("roadmap_done: no meta available"))
                    .and_then(|meta| {
                        meta.lock()
                            .map_err(|_| anyhow::anyhow!("roadmap_done: could not lock meta"))
                            .and_then(|mut m| crate::session::roadmap_done(&mut m.roadmap, n as usize))
                    })?;
                return Ok(Value::string(result));
            }

            // ── Plan operations — via SHARED_META ──
            "plan_show" => {
                let result = crate::eval::get_shared_meta()
                    .and_then(|meta| meta.lock().ok().map(|m| crate::session::plan_show(&m.plan)))
                    .unwrap_or_else(|| "No plan set.".to_string());
                return Ok(Value::string(result));
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
                let result = crate::eval::get_shared_meta()
                    .ok_or_else(|| anyhow::anyhow!("plan_set: no meta available"))
                    .and_then(|meta| {
                        meta.lock()
                            .map_err(|_| anyhow::anyhow!("plan_set: could not lock meta"))
                            .map(|mut m| crate::session::plan_set(&mut m.plan, &descriptions))
                    })?;
                return Ok(Value::string(result));
            }
            "plan_done" => {
                let n = match args.first() {
                    Some(Value::Int(n)) => *n,
                    _ => bail!("plan_done expects (n:Int)"),
                };
                if n < 1 {
                    bail!("plan_done: index must be >= 1");
                }
                let result = crate::eval::get_shared_meta()
                    .ok_or_else(|| anyhow::anyhow!("plan_done: no meta available"))
                    .and_then(|meta| {
                        meta.lock()
                            .map_err(|_| anyhow::anyhow!("plan_done: could not lock meta"))
                            .and_then(|mut m| crate::session::plan_done(&mut m.plan, n as usize))
                    })?;
                return Ok(Value::string(result));
            }
            "plan_fail" => {
                let n = match args.first() {
                    Some(Value::Int(n)) => *n,
                    _ => bail!("plan_fail expects (n:Int)"),
                };
                if n < 1 {
                    bail!("plan_fail: index must be >= 1");
                }
                let result = crate::eval::get_shared_meta()
                    .ok_or_else(|| anyhow::anyhow!("plan_fail: no meta available"))
                    .and_then(|meta| {
                        meta.lock()
                            .map_err(|_| anyhow::anyhow!("plan_fail: could not lock meta"))
                            .and_then(|mut m| crate::session::plan_fail(&mut m.plan, n as usize))
                    })?;
                return Ok(Value::string(result));
            }
            // ── Query operations — access program AST via thread-local ──
            // These reuse the same logic as the ? query commands in typeck.rs.
            "query_symbols" | "symbols_list" => {
                let program = crate::eval::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_symbols: program not available (no async context)"))?;
                let table = crate::typeck::build_symbol_table(&program);
                let result = crate::typeck::handle_query(&program, &table, "?symbols", &[]);
                return Ok(Value::string(result));
            }
            "query_symbols_detail" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("query_symbols_detail expects (name:String)"),
                };
                let program = crate::eval::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_symbols_detail: program not available"))?;
                let table = crate::typeck::build_symbol_table(&program);
                let query = format!("?symbols {name}");
                let result = crate::typeck::handle_query(&program, &table, &query, &[]);
                return Ok(Value::string(result));
            }
            "query_source" | "source_get" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("query_source expects (name:String)"),
                };
                let program = crate::eval::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_source: program not available"))?;
                let table = crate::typeck::build_symbol_table(&program);
                let query = format!("?source {name}");
                let result = crate::typeck::handle_query(&program, &table, &query, &[]);
                return Ok(Value::string(result));
            }
            "query_callers" | "callers_get" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("query_callers expects (name:String)"),
                };
                let program = crate::eval::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_callers: program not available"))?;
                let table = crate::typeck::build_symbol_table(&program);
                let query = format!("?callers {name}");
                let result = crate::typeck::handle_query(&program, &table, &query, &[]);
                return Ok(Value::string(result));
            }
            "query_callees" | "callees_get" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("query_callees expects (name:String)"),
                };
                let program = crate::eval::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_callees: program not available"))?;
                let table = crate::typeck::build_symbol_table(&program);
                let query = format!("?callees {name}");
                let result = crate::typeck::handle_query(&program, &table, &query, &[]);
                return Ok(Value::string(result));
            }
            "query_deps" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("query_deps expects (name:String)"),
                };
                let program = crate::eval::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_deps: program not available"))?;
                let table = crate::typeck::build_symbol_table(&program);
                let query = format!("?deps {name}");
                let result = crate::typeck::handle_query(&program, &table, &query, &[]);
                return Ok(Value::string(result));
            }
            "query_deps_all" | "deps_get" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("query_deps_all expects (name:String)"),
                };
                let program = crate::eval::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_deps_all: program not available"))?;
                let table = crate::typeck::build_symbol_table(&program);
                let query = format!("?deps-all {name}");
                let result = crate::typeck::handle_query(&program, &table, &query, &[]);
                return Ok(Value::string(result));
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
                return Ok(Value::string(result));
            }
            "query_tasks" => {
                let result = if let Some(reg) = &self.task_registry {
                    crate::api::format_tasks(&Some(reg.clone()))
                } else {
                    "No task registry (not in async context).".to_string()
                };
                return Ok(Value::string(result));
            }
            "query_inbox" => {
                if !args.is_empty() {
                    bail!("query_inbox expects no arguments");
                }
                let result = crate::eval::get_shared_meta()
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
                return Ok(Value::string(result));
            }
            "inbox_clear" => {
                if !args.is_empty() {
                    bail!("inbox_clear expects no arguments");
                }

                let cleared = if let Some(meta) = crate::eval::get_shared_meta() {
                    meta.lock()
                        .map_err(|_| anyhow::anyhow!("inbox_clear: could not lock meta"))?
                        .agent_mailbox
                        .remove("main")
                        .map(|msgs| msgs.len())
                        .unwrap_or(0)
                } else {
                    let rt = crate::eval::get_shared_runtime()
                        .ok_or_else(|| anyhow::anyhow!("inbox_clear: no runtime available"))?;
                    rt.write()
                        .map_err(|_| anyhow::anyhow!("inbox_clear: could not access runtime"))?
                        .agent_mailbox
                        .remove("main")
                        .map(|msgs| msgs.len())
                        .unwrap_or(0)
                };

                if let Some(rt) = crate::eval::get_shared_runtime() {
                    if let Ok(mut state) = rt.write() {
                        state.agent_mailbox.remove("main");
                    }
                }

                return Ok(Value::string(format!("cleared {cleared} messages")));
            }
            "query_library" => {
                let program = crate::eval::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("query_library: program not available"))?;
                let result = crate::library::query_library(&program, None);
                return Ok(Value::string(result));
            }

            // ── library_errors — formatted string of all library errors ──
            "library_errors" => {
                let result = crate::eval::get_shared_runtime()
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
                return Ok(Value::string(result));
            }
            "failure_history" => {
                if !args.is_empty() {
                    bail!("failure_history expects no arguments");
                }
                let result = crate::eval::get_shared_runtime()
                    .and_then(|rt| rt.read().ok().map(|state| crate::session::format_failure_history(&state)))
                    .unwrap_or_else(|| "No recent mutation failures.".to_string());
                return Ok(Value::string(result));
            }
            "failure_patterns" => {
                if !args.is_empty() {
                    bail!("failure_patterns expects no arguments");
                }
                let result = crate::eval::get_shared_runtime()
                    .and_then(|rt| rt.read().ok().map(|state| crate::session::summarize_failure_patterns(&state)))
                    .unwrap_or_else(|| "No recent mutation failures.".to_string());
                return Ok(Value::string(result));
            }
            // ── library_reload — reload module(s) from disk ──
            "library_reload" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    Some(other) => format!("{other}"),
                    None => String::new(),
                };
                let program_lock = crate::eval::get_shared_program_mut()
                    .ok_or_else(|| anyhow::anyhow!("library_reload: program not available (no async context)"))?;
                let mut program = program_lock.write()
                    .map_err(|_| anyhow::anyhow!("library_reload: could not acquire program write lock"))?;

                let result = crate::library::reload_module(&mut program, &name)?;
                // Update the read-only snapshot so query builtins see the changes
                crate::eval::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                return Ok(Value::string(result));
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
                if let Some(rt) = crate::eval::get_shared_runtime() {
                    crate::eval::init_missing_shared_runtime_vars(&program, &rt);
                }
                let summary = if applied == 1 {
                    format!("Applied 1 mutation: {}", messages[0])
                } else {
                    format!("Applied {applied} mutations")
                };
                return Ok(Value::string(summary));
            }
            "fn_remove" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
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
                            return Ok(Value::string(format!("Removed {name}")));
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
                    return Ok(Value::string(format!("Removed {name}")));
                }
                bail!("fn_remove: function `{name}` not found");
            }
            "type_remove" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
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
                            return Ok(Value::string(format!("Removed {name}")));
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
                    return Ok(Value::string(format!("Removed {name}")));
                }
                bail!("type_remove: type `{name}` not found");
            }
            "module_remove" => {
                let name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
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
                    return Ok(Value::string(format!("Removed module {name}")));
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

                let program_lock = crate::eval::get_shared_program_mut()
                    .ok_or_else(|| anyhow::anyhow!("move_symbols: program not available (no async context)"))?;
                let mut program = program_lock.write()
                    .map_err(|_| anyhow::anyhow!("move_symbols: could not acquire program write lock"))?;

                let result = crate::validator::apply_move(&mut program, &names, &target_module)?;
                // Update read-only snapshot
                crate::eval::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                return Ok(Value::string(result));
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
                let program = crate::eval::get_shared_program()
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
                return Ok(Value::string(
                    if output.is_empty() {
                        format!("Trace of {fn_name}: (no steps)")
                    } else {
                        format!("Trace of {fn_name}:\n{output}")
                    },
                ));
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
                let rt = crate::eval::get_shared_runtime()
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
                if let Some(meta) = crate::eval::get_shared_meta() {
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

                return Ok(Value::string(format!("Message sent to '{target}'")));
            }

            // ── inbox_read — programmatic inbox drain ──
            "inbox_read" => {
                if !args.is_empty() {
                    bail!("inbox_read expects no arguments");
                }

                let mut messages = if let Some(meta) = crate::eval::get_shared_meta() {
                    meta.lock()
                        .map_err(|_| anyhow::anyhow!("inbox_read: could not lock meta"))?
                        .agent_mailbox
                        .remove("main")
                        .unwrap_or_default()
                } else {
                    let rt = crate::eval::get_shared_runtime()
                        .ok_or_else(|| anyhow::anyhow!("inbox_read: no runtime available"))?;
                    rt.write()
                        .map_err(|_| anyhow::anyhow!("inbox_read: could not access runtime"))?
                        .agent_mailbox
                        .remove("main")
                        .unwrap_or_default()
                };

                if let Some(rt) = crate::eval::get_shared_runtime() {
                    if let Ok(mut state) = rt.write() {
                        state.agent_mailbox.remove("main");
                    }
                }

                let contents: Vec<String> = messages.drain(..).map(|msg| msg.content).collect();
                let payload = serde_json::to_string(&contents)
                    .map_err(|e| anyhow::anyhow!("inbox_read: failed to serialize inbox: {e}"))?;
                return Ok(Value::string(payload));
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
                let program = crate::eval::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("watch_start: program not available (no async context)"))?;
                if program.get_function(&fn_name).is_none() {
                    bail!("watch_start: function `{fn_name}` not found");
                }

                // Queue the watch command for API-layer processing
                let cmd = format!("!watch {fn_name} {interval_ms}");
                let rt = crate::eval::get_shared_runtime()
                    .ok_or_else(|| anyhow::anyhow!("watch_start: no runtime available"))?;
                rt.write()
                    .map_err(|_| anyhow::anyhow!("watch_start: could not access runtime"))?
                    .pending_commands.push(cmd);

                return Ok(Value::string(format!("Watching {fn_name} every {interval_ms}ms (queued)")));
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
                let rt = crate::eval::get_shared_runtime()
                    .ok_or_else(|| anyhow::anyhow!("agent_spawn: no runtime available"))?;
                rt.write()
                    .map_err(|_| anyhow::anyhow!("agent_spawn: could not access runtime"))?
                    .pending_commands.push(cmd);

                return Ok(Value::string(format!("Agent '{name}' spawned (scope: {scope})")));
            }

            // ── route_list — list registered HTTP routes ──
            "route_list" => {
                let routes = crate::eval::get_shared_runtime()
                    .and_then(|rt| rt.read().ok().map(|state| state.http_routes.clone()))
                    .unwrap_or_default();
                if routes.is_empty() {
                    return Ok(Value::string("No routes registered."));
                }
                let mut out = String::new();
                for r in &routes {
                    out.push_str(&format!("{} {} -> `{}`\n", r.method, r.path, r.handler_fn));
                }
                return Ok(Value::string(out.trim_end().to_string()));
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
                let rt = crate::eval::get_shared_runtime()
                    .ok_or_else(|| anyhow::anyhow!("route_add: no runtime available"))?;
                let mut state = rt.write()
                    .map_err(|_| anyhow::anyhow!("route_add: could not access runtime"))?;
                // Upsert: update if method+path already exists
                if let Some(existing) = state.http_routes.iter_mut()
                    .find(|r| r.method == method_upper && r.path == path)
                {
                    let old_fn = existing.handler_fn.clone();
                    existing.handler_fn = handler.clone();
                    return Ok(Value::string(format!("updated route {method_upper} {path} -> `{handler}` (was `{old_fn}`)")));
                }
                state.http_routes.push(crate::ast::HttpRoute {
                    method: method_upper.clone(),
                    path: path.clone(),
                    handler_fn: handler.clone(),
                });
                return Ok(Value::string(format!("added route {method_upper} {path} -> `{handler}`")));
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
                let rt = crate::eval::get_shared_runtime()
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
                    return Ok(Value::string(format!(
                        "removed route {} {} (was -> `{}`)",
                        method_upper, path, removed_handler.unwrap_or_default()
                    )));
                }
                bail!("route_remove: no route found for {method_upper} {path}");
            }

            // ── undo — revert the last mutation ──
            "undo" => {
                let rt = crate::eval::get_shared_runtime()
                    .ok_or_else(|| anyhow::anyhow!("undo: no runtime available"))?;
                rt.write()
                    .map_err(|_| anyhow::anyhow!("undo: could not access runtime"))?
                    .pending_commands.push("!undo".to_string());
                return Ok(Value::string("Undo queued — will revert last mutation"));
            }

            // ── sandbox_enter — enter sandbox mode ──
            "sandbox_enter" => {
                let rt = crate::eval::get_shared_runtime()
                    .ok_or_else(|| anyhow::anyhow!("sandbox_enter: no runtime available"))?;
                rt.write()
                    .map_err(|_| anyhow::anyhow!("sandbox_enter: could not access runtime"))?
                    .pending_commands.push("!sandbox enter".to_string());
                return Ok(Value::string("Sandbox enter queued — mutations will be isolated"));
            }

            // ── sandbox_merge — merge sandbox changes ──
            "sandbox_merge" => {
                let rt = crate::eval::get_shared_runtime()
                    .ok_or_else(|| anyhow::anyhow!("sandbox_merge: no runtime available"))?;
                rt.write()
                    .map_err(|_| anyhow::anyhow!("sandbox_merge: could not access runtime"))?
                    .pending_commands.push("!sandbox merge".to_string());
                return Ok(Value::string("Sandbox merge queued — changes will be kept"));
            }

            // ── sandbox_discard — discard sandbox changes ──
            "sandbox_discard" => {
                let rt = crate::eval::get_shared_runtime()
                    .ok_or_else(|| anyhow::anyhow!("sandbox_discard: no runtime available"))?;
                rt.write()
                    .map_err(|_| anyhow::anyhow!("sandbox_discard: could not access runtime"))?
                    .pending_commands.push("!sandbox discard".to_string());
                return Ok(Value::string("Sandbox discard queued — changes will be reverted"));
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
                let meta = crate::eval::get_shared_meta()
                    .ok_or_else(|| anyhow::anyhow!("mock_set: no meta available"))?;
                meta.lock()
                    .map_err(|_| anyhow::anyhow!("mock_set: could not lock meta"))?
                    .io_mocks.push(crate::session::IoMock {
                        operation: operation.clone(),
                        patterns: patterns.clone(),
                        response: response.clone(),
                    });
                let pattern_display = if patterns.is_empty() { "*".to_string() } else { patterns.join(" ") };
                return Ok(Value::string(format!(
                    "mock: {operation} {pattern_display} -> \"{}\"",
                    response.chars().take(50).collect::<String>()
                )));
            }

            // ── mock_clear — clear all IO mocks ──
            "mock_clear" => {
                let meta = crate::eval::get_shared_meta()
                    .ok_or_else(|| anyhow::anyhow!("mock_clear: no meta available"))?;
                let count = {
                    let mut m = meta.lock()
                        .map_err(|_| anyhow::anyhow!("mock_clear: could not lock meta"))?;
                    let count = m.io_mocks.len();
                    m.io_mocks.clear();
                    count
                };
                return Ok(Value::string(format!("cleared {count} mocks")));
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
                let sender = crate::eval::get_shared_event_broadcast()
                    .ok_or_else(|| anyhow::anyhow!("sse_send: no event broadcast available"))?;
                let payload = serde_json::json!({"type": event_type, "data": data}).to_string();
                sender.send(payload)
                    .map_err(|e| anyhow::anyhow!("sse_send: failed to send event: {e}"))?;
                return Ok(Value::string("sent"));
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
                let program_lock = crate::eval::get_shared_program_mut()
                    .ok_or_else(|| anyhow::anyhow!("module_create: program not available (no async context)"))?;
                let mut program = program_lock.write()
                    .map_err(|_| anyhow::anyhow!("module_create: could not acquire program write lock"))?;
                // Check if module already exists
                if program.modules.iter().any(|m| m.name == name) {
                    return Ok(Value::string(format!("module '{name}' already exists")));
                }
                // Create empty module
                let code = format!("!module {name}");
                let operations = crate::parser::parse(&code)
                    .map_err(|e| anyhow::anyhow!("module_create: parse error: {e}"))?;
                for op in &operations {
                    crate::validator::apply_and_validate(&mut program, op)
                        .map_err(|e| anyhow::anyhow!("module_create: {e}"))?;
                }
                crate::eval::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                return Ok(Value::string(format!("created module '{name}'")));
            }

            // ── test_run — run stored tests for a function ──
            "test_run" => {
                let fn_name = match args.first() {
                    Some(Value::String(s)) => s.as_ref().clone(),
                    _ => bail!("test_run expects (fn_name:String)"),
                };
                let program = crate::eval::get_shared_program()
                    .ok_or_else(|| anyhow::anyhow!("test_run: program not available"))?;
                let func = program.get_function(&fn_name)
                    .ok_or_else(|| anyhow::anyhow!("test_run: function `{fn_name}` not found"))?;
                let ast_cases = func.tests.clone();
                if ast_cases.is_empty() {
                    return Ok(Value::string(format!("no stored tests for `{fn_name}`")));
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
                let io_mocks = crate::eval::get_shared_meta()
                    .and_then(|meta| meta.lock().ok().map(|m| m.io_mocks.clone()))
                    .unwrap_or_default();
                let http_routes = crate::eval::get_shared_runtime()
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
                return Ok(Value::string(results.join("\n")));
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
                let program_lock = crate::eval::get_shared_program_mut()
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
                crate::eval::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                return Ok(Value::string(result_msg));
            }

            _ => {} // fall through to mock/IO dispatch
        }

        if let Some(result) = self.try_mock_io(op, args, true)? {
            return Ok(result);
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
            _ => bail!("unknown await operation: {op}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: set up SharedMeta and SharedRuntime with empty roadmap/plan
    /// and install them as thread-locals, returning the handle and meta for assertions.
    fn setup_roadmap_runtime() -> (CoroutineHandle, crate::session::SharedMeta) {
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState::default()));
        let meta = std::sync::Arc::new(std::sync::Mutex::new(crate::session::SessionMeta::new()));
        crate::eval::set_shared_runtime(Some(rt.clone()));
        crate::eval::set_shared_meta(Some(meta.clone()));
        let handle = CoroutineHandle::new_mock(vec![]);
        (handle, meta)
    }

    fn unwrap_string(v: Value) -> String {
        match v {
            Value::String(s) => s.as_ref().clone(),
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
        let (handle, meta) = setup_roadmap_runtime();

        // Add an item
        let result = unwrap_string(
            handle.execute_await("roadmap_add", &[Value::string("Build feature X")]).unwrap()
        );
        assert_eq!(result, "Build feature X");

        // Verify it's in the meta state
        assert_eq!(meta.lock().unwrap().roadmap.len(), 1);
        assert_eq!(meta.lock().unwrap().roadmap[0].description, "Build feature X");
        assert!(!meta.lock().unwrap().roadmap[0].done);

        // List should show it
        let list = unwrap_string(handle.execute_await("roadmap_list", &[]).unwrap());
        assert!(list.contains("Build feature X"), "list should contain the item: {list}");
        assert!(list.contains("[ ] 1:"), "item should be unchecked: {list}");
    }

    #[test]
    fn roadmap_add_empty_fails() {
        let (handle, _rt) = setup_roadmap_runtime();
        let result = handle.execute_await("roadmap_add", &[Value::string("  ")]);
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
        let (handle, meta) = setup_roadmap_runtime();

        // Add two items
        handle
            .execute_await("roadmap_add", &[Value::string("Item A")])
            .unwrap();
        handle
            .execute_await("roadmap_add", &[Value::string("Item B")])
            .unwrap();

        // Mark item 2 as done
        let result = unwrap_string(
            handle.execute_await("roadmap_done", &[Value::Int(2)]).unwrap()
        );
        assert!(result.contains("#2 done"), "confirmation: {result}");

        // Verify state
        assert!(!meta.lock().unwrap().roadmap[0].done);
        assert!(meta.lock().unwrap().roadmap[1].done);

        // List should show [x] for item 2
        let list = unwrap_string(handle.execute_await("roadmap_list", &[]).unwrap());
        assert!(list.contains("[ ] 1: Item A"), "A unchecked: {list}");
        assert!(list.contains("[x] 2: Item B"), "B checked: {list}");
    }

    #[test]
    fn roadmap_done_out_of_bounds() {
        let (handle, _rt) = setup_roadmap_runtime();
        handle
            .execute_await("roadmap_add", &[Value::string("Only item")])
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
        let result = handle.execute_await("roadmap_done", &[Value::string("1")]);
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
        let (handle, meta) = setup_roadmap_runtime();

        let result = unwrap_string(
            handle.execute_await("plan_set", &[Value::string("Parse input\nValidate data\nStore results")]).unwrap()
        );
        assert_eq!(result, "Plan set with 3 steps.");

        // Verify state
        assert_eq!(meta.lock().unwrap().plan.len(), 3);
        assert_eq!(meta.lock().unwrap().plan[0].description, "Parse input");
        assert_eq!(meta.lock().unwrap().plan[1].description, "Validate data");
        assert_eq!(meta.lock().unwrap().plan[2].description, "Store results");

        // Show should list all steps
        let show = unwrap_string(handle.execute_await("plan_show", &[]).unwrap());
        assert!(show.contains("[ ] 1: Parse input"), "step 1: {show}");
        assert!(show.contains("[ ] 2: Validate data"), "step 2: {show}");
        assert!(show.contains("[ ] 3: Store results"), "step 3: {show}");
    }

    #[test]
    fn plan_set_empty_fails() {
        let (handle, _rt) = setup_roadmap_runtime();
        let result = handle.execute_await("plan_set", &[Value::string("  \n  \n  ")]);
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
        let (handle, meta) = setup_roadmap_runtime();

        let result = unwrap_string(
            handle.execute_await("plan_set", &[Value::string("Step A\n\n  \nStep B")]).unwrap()
        );
        assert_eq!(result, "Plan set with 2 steps.");
        assert_eq!(meta.lock().unwrap().plan.len(), 2);
        assert_eq!(meta.lock().unwrap().plan[0].description, "Step A");
        assert_eq!(meta.lock().unwrap().plan[1].description, "Step B");
    }

    #[test]
    fn plan_done_marks_step() {
        let (handle, meta) = setup_roadmap_runtime();

        handle.execute_await("plan_set", &[Value::string("Alpha\nBravo")]).unwrap();

        let result = unwrap_string(
            handle.execute_await("plan_done", &[Value::Int(2)]).unwrap()
        );
        assert_eq!(result, "Plan: step 2 done.");

        // Verify state
        assert_eq!(meta.lock().unwrap().plan[0].status, crate::session::PlanStatus::Pending);
        assert_eq!(meta.lock().unwrap().plan[1].status, crate::session::PlanStatus::Done);

        // Show should reflect [x]
        let show = unwrap_string(handle.execute_await("plan_show", &[]).unwrap());
        assert!(show.contains("[ ] 1: Alpha"), "Alpha pending: {show}");
        assert!(show.contains("[x] 2: Bravo"), "Bravo done: {show}");
    }

    #[test]
    fn plan_done_out_of_bounds() {
        let (handle, _rt) = setup_roadmap_runtime();
        handle.execute_await("plan_set", &[Value::string("Only step")]).unwrap();

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
        let result = handle.execute_await("plan_done", &[Value::string("1")]);
        assert!(result.is_err());
    }

    #[test]
    fn plan_fail_marks_step() {
        let (handle, meta) = setup_roadmap_runtime();

        handle.execute_await("plan_set", &[Value::string("First\nSecond\nThird")]).unwrap();

        let result = unwrap_string(
            handle.execute_await("plan_fail", &[Value::Int(1)]).unwrap()
        );
        assert_eq!(result, "Plan: step 1 failed.");

        // Verify state
        assert_eq!(meta.lock().unwrap().plan[0].status, crate::session::PlanStatus::Failed);
        assert_eq!(meta.lock().unwrap().plan[1].status, crate::session::PlanStatus::Pending);

        // Show should reflect [!]
        let show = unwrap_string(handle.execute_await("plan_show", &[]).unwrap());
        assert!(show.contains("[!] 1: First"), "First failed: {show}");
        assert!(show.contains("[ ] 2: Second"), "Second pending: {show}");
    }

    #[test]
    fn plan_fail_out_of_bounds() {
        let (handle, _rt) = setup_roadmap_runtime();
        handle.execute_await("plan_set", &[Value::string("Only step")]).unwrap();

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
        let result = handle.execute_await("plan_fail", &[Value::string("1")]);
        assert!(result.is_err());
    }

    #[test]
    fn plan_set_replaces_existing() {
        let (handle, meta) = setup_roadmap_runtime();

        handle.execute_await("plan_set", &[Value::string("Old step 1\nOld step 2")]).unwrap();
        assert_eq!(meta.lock().unwrap().plan.len(), 2);

        handle.execute_await("plan_set", &[Value::string("New step")]).unwrap();
        assert_eq!(meta.lock().unwrap().plan.len(), 1);
        assert_eq!(meta.lock().unwrap().plan[0].description, "New step");
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
            ..Default::default()
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
            handle.execute_await("query_symbols_detail", &[Value::string("greet")]).unwrap()
        );
        assert!(result.contains("greet"), "should show greet details: {result}");
        assert!(result.contains("params") || result.contains("String"),
            "should show parameter info: {result}");
    }

    #[test]
    fn query_symbols_detail_not_found() {
        let handle = setup_query_runtime("");
        let result = unwrap_string(
            handle.execute_await("query_symbols_detail", &[Value::string("nonexistent")]).unwrap()
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
            handle.execute_await("query_source", &[Value::string("greet")]).unwrap()
        );
        assert!(result.contains("+fn greet"), "should contain function definition: {result}");
        assert!(result.contains("concat"), "should contain function body: {result}");
    }

    #[test]
    fn query_source_not_found() {
        let handle = setup_query_runtime("");
        let result = unwrap_string(
            handle.execute_await("query_source", &[Value::string("missing")]).unwrap()
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
            handle.execute_await("query_callers", &[Value::string("greet")]).unwrap()
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
            handle.execute_await("query_callers", &[Value::string("greet")]).unwrap()
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
            handle.execute_await("query_callees", &[Value::string("main")]).unwrap()
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
            handle.execute_await("query_deps", &[Value::string("main")]).unwrap()
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
            handle.execute_await("query_deps_all", &[Value::string("c")]).unwrap()
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
            handle.execute_await("query_deps_all", &[Value::string("a")]).unwrap()
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
            ..Default::default()
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
    fn query_inbox_matches_query_output() {
        let (handle, meta) = setup_roadmap_runtime();
        {
            let mut meta = meta.lock().unwrap();
            crate::session::send_agent_message(&mut meta, "agent1", "main", "hello");
            crate::session::send_agent_message(&mut meta, "agent2", "main", "status update");
        }

        let result = unwrap_string(handle.execute_await("query_inbox", &[]).unwrap());
        assert!(result.contains("from agent1: hello"), "got: {result}");
        assert!(result.contains("from agent2: status update"), "got: {result}");
    }

    #[test]
    fn query_inbox_with_args_fails() {
        let (handle, _meta) = setup_roadmap_runtime();
        let result = handle.execute_await("query_inbox", &[Value::string("unexpected")]);
        assert!(result.is_err(), "should fail with extra args");
        assert!(result.unwrap_err().to_string().contains("expects no arguments"));
    }

    #[test]
    fn inbox_clear_removes_messages() {
        let (handle, meta) = setup_roadmap_runtime();
        {
            let mut meta = meta.lock().unwrap();
            crate::session::send_agent_message(&mut meta, "agent1", "main", "first");
            crate::session::send_agent_message(&mut meta, "agent2", "main", "second");
        }
        let rt = crate::eval::get_shared_runtime().unwrap();
        rt.write().unwrap().agent_mailbox = meta.lock().unwrap().agent_mailbox.clone();

        let result = unwrap_string(handle.execute_await("inbox_clear", &[]).unwrap());
        assert_eq!(result, "cleared 2 messages");
        assert!(meta.lock().unwrap().agent_mailbox.get("main").is_none());
        assert!(rt.read().unwrap().agent_mailbox.get("main").is_none());
    }

    #[test]
    fn inbox_clear_with_args_fails() {
        let (handle, _meta) = setup_roadmap_runtime();
        let result = handle.execute_await("inbox_clear", &[Value::string("unexpected")]);
        assert!(result.is_err(), "should fail with extra args");
        assert!(result.unwrap_err().to_string().contains("expects no arguments"));
    }

    #[test]
    fn query_library_returns_string() {
        let handle = setup_query_runtime("");
        let result = unwrap_string(handle.execute_await("query_library", &[]).unwrap());
        assert!(result.contains("Module library"), "should contain library info: {result}");
    }

    #[test]
    fn failure_history_returns_recent_failures() {
        let (handle, _meta) = setup_roadmap_runtime();
        let rt = crate::eval::get_shared_runtime().unwrap();
        if let Ok(mut state) = rt.write() {
            crate::session::record_failure(&mut state, "undefined variable `user_id`");
            crate::session::record_failure(&mut state, "type mismatch in let binding");
        }
        let result = unwrap_string(handle.execute_await("failure_history", &[]).unwrap());
        assert!(result.contains(": type mismatch in let binding"), "got: {result}");
        assert!(result.contains(": undefined variable `user_id`"), "got: {result}");
    }

    #[test]
    fn failure_patterns_groups_similar_errors() {
        let (handle, _meta) = setup_roadmap_runtime();
        let rt = crate::eval::get_shared_runtime().unwrap();
        if let Ok(mut state) = rt.write() {
            crate::session::record_failure(&mut state, "undefined variable `user_id`");
            crate::session::record_failure(&mut state, "undefined variable `account_id`");
            crate::session::record_failure(&mut state, "parse error: unexpected +end");
        }
        let result = unwrap_string(handle.execute_await("failure_patterns", &[]).unwrap());
        assert!(result.contains("2x undefined variable errors"), "got: {result}");
        assert!(result.contains("latest: `account_id`"), "got: {result}");
        assert!(result.contains("1x parse errors"), "got: {result}");
    }

    #[test]
    fn failure_patterns_with_args_fails() {
        let (handle, _meta) = setup_roadmap_runtime();
        let result = handle.execute_await("failure_patterns", &[Value::string("unexpected")]);
        assert!(result.is_err(), "should fail with extra args");
        assert!(result.unwrap_err().to_string().contains("expects no arguments"));
    }

    #[test]
    fn query_no_program_errors() {
        // Clear the thread-local program
        crate::eval::set_shared_program(None);
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState {
            http_routes: vec![],
            shared_vars: std::collections::HashMap::new(),
            ..Default::default()
        }));
        crate::eval::set_shared_runtime(Some(rt));
        let handle = CoroutineHandle::new_mock(vec![]);

        // All query builtins that need the program should error
        let result = handle.execute_await("query_symbols", &[]);
        assert!(result.is_err(), "query_symbols should fail without program");
        assert!(result.unwrap_err().to_string().contains("program not available"),
            "should mention program not available");

        let result = handle.execute_await("query_source", &[Value::string("x")]);
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
            ..Default::default()
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
            handle.execute_await("mutate", &[Value::string(code)]).unwrap()
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
            handle.execute_await("mutate", &[Value::string(code)]).unwrap()
        );
        assert!(result.contains("Applied"), "should report mutations: {result}");

        let p = prog.read().unwrap();
        assert!(p.get_function("Greeter.greet").is_some(), "Greeter.greet should exist");
    }

    #[test]
    fn mutate_empty_code_fails() {
        let (handle, _prog) = setup_mutation_runtime("");
        let result = handle.execute_await("mutate", &[Value::string("")]);
        assert!(result.is_err(), "mutate with empty code should fail");
        assert!(result.unwrap_err().to_string().contains("empty"), "error should mention empty");
    }

    #[test]
    fn mutate_invalid_syntax_fails() {
        let (handle, _prog) = setup_mutation_runtime("");
        let result = handle.execute_await("mutate", &[Value::string("+fn")]);
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
        let result = handle.execute_await("mutate", &[Value::string("+fn x ()->Int\n  +return 1\n+end")]);
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
            handle.execute_await("fn_remove", &[Value::string("hello")]).unwrap()
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
            handle.execute_await("fn_remove", &[Value::string("MyMod.greet")]).unwrap()
        );
        assert_eq!(result, "Removed MyMod.greet");
        assert!(prog.read().unwrap().get_function("MyMod.greet").is_none());
    }

    #[test]
    fn fn_remove_not_found() {
        let (handle, _prog) = setup_mutation_runtime("");
        let result = handle.execute_await("fn_remove", &[Value::string("nonexistent")]);
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
        let result = handle.execute_await("fn_remove", &[Value::string("NoModule.func")]);
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
            handle.execute_await("type_remove", &[Value::string("Color")]).unwrap()
        );
        assert_eq!(result, "Removed Color");
        assert!(prog.read().unwrap().types.is_empty());
    }

    #[test]
    fn type_remove_not_found() {
        let (handle, _prog) = setup_mutation_runtime("");
        let result = handle.execute_await("type_remove", &[Value::string("Missing")]);
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
            handle.execute_await("module_remove", &[Value::string("MyMod")]).unwrap()
        );
        assert_eq!(result, "Removed module MyMod");
        assert!(prog.read().unwrap().modules.is_empty());
        assert!(prog.read().unwrap().get_function("MyMod.hello").is_none());
    }

    #[test]
    fn module_remove_not_found() {
        let (handle, _prog) = setup_mutation_runtime("");
        let result = handle.execute_await("module_remove", &[Value::string("NoModule")]);
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
        handle.execute_await("mutate", &[Value::string(
            "+fn test_func ()->Int\n  +return 42\n+end"
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
        handle.execute_await("fn_remove", &[Value::string("to_remove")]).unwrap();

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
            handle.execute_await("source_get", &[Value::string("greet")]).unwrap()
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
            handle.execute_await("callers_get", &[Value::string("greet")]).unwrap()
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
            handle.execute_await("callees_get", &[Value::string("main")]).unwrap()
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
            handle.execute_await("deps_get", &[Value::string("c")]).unwrap()
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
            ..Default::default()
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
            ..Default::default()
        }));
        crate::eval::set_shared_runtime(Some(rt));
        let handle = CoroutineHandle::new_mock(vec![]);

        // All alias builtins that need the program should error
        let result = handle.execute_await("symbols_list", &[]);
        assert!(result.is_err(), "symbols_list should fail without program");
        assert!(result.unwrap_err().to_string().contains("program not available"),
            "should mention program not available");

        let result = handle.execute_await("source_get", &[Value::string("x")]);
        assert!(result.is_err(), "source_get should fail without program");

        let result = handle.execute_await("callers_get", &[Value::string("x")]);
        assert!(result.is_err(), "callers_get should fail without program");

        let result = handle.execute_await("callees_get", &[Value::string("x")]);
        assert!(result.is_err(), "callees_get should fail without program");

        let result = handle.execute_await("deps_get", &[Value::string("x")]);
        assert!(result.is_err(), "deps_get should fail without program");

        let result = handle.execute_await("routes_list", &[]);
        assert!(result.is_err(), "routes_list should fail without program");
    }

    // ═════════════════════════════════════════════════════════════════════
    // move_symbols builtin
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn move_symbols_moves_function() {
        let (handle, prog) = setup_mutation_runtime(
            "+fn helper ()->String\n  +return \"hi\"\n+end"
        );
        assert!(prog.read().unwrap().get_function("helper").is_some());

        let result = unwrap_string(
            handle.execute_await("move_symbols", &[
                Value::string("helper"),
                Value::string("Utils"),
            ]).unwrap()
        );
        assert!(result.contains("moved"), "should confirm move: {result}");
        assert!(result.contains("Utils"), "should mention target module: {result}");

        // Function should now be in Utils module, not top-level
        let p = prog.read().unwrap();
        assert!(p.get_function("Utils.helper").is_some(), "helper should be in Utils");
        assert!(p.functions.iter().all(|f| f.name != "helper"), "helper should not be in top-level functions");
    }

    #[test]
    fn move_symbols_multiple_comma_separated() {
        let (handle, prog) = setup_mutation_runtime(
            "+fn foo ()->Int\n  +return 1\n+end\n\
             +fn bar ()->Int\n  +return 2\n+end"
        );
        let result = unwrap_string(
            handle.execute_await("move_symbols", &[
                Value::string("foo, bar"),
                Value::string("Helpers"),
            ]).unwrap()
        );
        assert!(result.contains("moved"), "should confirm move: {result}");
        let p = prog.read().unwrap();
        assert!(p.get_function("Helpers.foo").is_some());
        assert!(p.get_function("Helpers.bar").is_some());
    }

    #[test]
    fn move_symbols_empty_symbols_fails() {
        let (handle, _prog) = setup_mutation_runtime("");
        let result = handle.execute_await("move_symbols", &[
            Value::string(""),
            Value::string("Target"),
        ]);
        assert!(result.is_err(), "should fail with empty symbols");
    }

    #[test]
    fn move_symbols_empty_target_fails() {
        let (handle, _prog) = setup_mutation_runtime(
            "+fn x ()->Int\n  +return 1\n+end"
        );
        let result = handle.execute_await("move_symbols", &[
            Value::string("x"),
            Value::string(""),
        ]);
        assert!(result.is_err(), "should fail with empty target");
    }

    #[test]
    fn move_symbols_not_found_fails() {
        let (handle, _prog) = setup_mutation_runtime("");
        let result = handle.execute_await("move_symbols", &[
            Value::string("nonexistent"),
            Value::string("Target"),
        ]);
        assert!(result.is_err(), "should fail when symbol not found");
    }

    #[test]
    fn move_symbols_no_args_fails() {
        let (handle, _prog) = setup_mutation_runtime("");
        let result = handle.execute_await("move_symbols", &[]);
        assert!(result.is_err(), "should fail with no args");
    }

    #[test]
    fn move_symbols_no_program_fails() {
        crate::eval::set_shared_program_mut(None);
        let handle = CoroutineHandle::new_mock(vec![]);
        let result = handle.execute_await("move_symbols", &[
            Value::string("x"),
            Value::string("Target"),
        ]);
        assert!(result.is_err(), "should fail without program");
        assert!(result.unwrap_err().to_string().contains("program not available"));
    }

    // ═════════════════════════════════════════════════════════════════════
    // trace_run builtin
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn trace_run_simple_function() {
        let handle = setup_query_runtime(
            "+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end"
        );
        let result = unwrap_string(
            handle.execute_await("trace_run", &[
                Value::string("greet"),
                Value::string("\"world\""),
            ]).unwrap()
        );
        assert!(result.contains("Trace of greet"), "should have trace header: {result}");
        assert!(result.contains("return"), "should have a return step: {result}");
    }

    #[test]
    fn trace_run_no_args() {
        let handle = setup_query_runtime(
            "+fn get_one ()->Int\n  +return 1\n+end"
        );
        let result = unwrap_string(
            handle.execute_await("trace_run", &[
                Value::string("get_one"),
                Value::string(""),
            ]).unwrap()
        );
        assert!(result.contains("Trace of get_one"), "should trace: {result}");
    }

    #[test]
    fn trace_run_function_not_found() {
        let handle = setup_query_runtime("");
        let result = handle.execute_await("trace_run", &[
            Value::string("nonexistent"),
            Value::string(""),
        ]);
        assert!(result.is_err(), "should fail for missing function");
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn trace_run_wrong_type_fails() {
        let handle = setup_query_runtime("");
        let result = handle.execute_await("trace_run", &[Value::Int(42)]);
        assert!(result.is_err(), "should fail with non-String fn_name");
    }

    #[test]
    fn trace_run_no_program_fails() {
        crate::eval::set_shared_program(None);
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState {
            http_routes: vec![],
            shared_vars: std::collections::HashMap::new(),
            ..Default::default()
        }));
        crate::eval::set_shared_runtime(Some(rt));
        let handle = CoroutineHandle::new_mock(vec![]);
        let result = handle.execute_await("trace_run", &[
            Value::string("x"),
            Value::string(""),
        ]);
        assert!(result.is_err(), "should fail without program");
        assert!(result.unwrap_err().to_string().contains("program not available"));
    }

    // ═════════════════════════════════════════════════════════════════════
    // msg_send builtin
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn msg_send_delivers_message() {
        let (handle, _meta) = setup_roadmap_runtime();
        let rt = crate::eval::get_shared_runtime().unwrap();
        let result = unwrap_string(
            handle.execute_await("msg_send", &[
                Value::string("agent1"),
                Value::string("hello from main"),
            ]).unwrap()
        );
        assert!(result.contains("Message sent to 'agent1'"), "confirmation: {result}");

        // Verify message is in the mailbox (agent_mailbox is still in RuntimeState)
        let state = rt.read().unwrap();
        let inbox = state.agent_mailbox.get("agent1").unwrap();
        assert_eq!(inbox.len(), 1);
        assert_eq!(inbox[0].content, "hello from main");
        assert_eq!(inbox[0].from, "main");
        assert_eq!(inbox[0].to, "agent1");
    }

    #[test]
    fn msg_send_multiple_messages() {
        let (handle, _meta) = setup_roadmap_runtime();
        let rt = crate::eval::get_shared_runtime().unwrap();
        handle.execute_await("msg_send", &[
            Value::string("agent1"),
            Value::string("first"),
        ]).unwrap();
        handle.execute_await("msg_send", &[
            Value::string("agent1"),
            Value::string("second"),
        ]).unwrap();

        let state = rt.read().unwrap();
        let inbox = state.agent_mailbox.get("agent1").unwrap();
        assert_eq!(inbox.len(), 2);
        assert_eq!(inbox[0].content, "first");
        assert_eq!(inbox[1].content, "second");
    }

    #[test]
    fn msg_send_syncs_meta_mailbox() {
        let (handle, meta) = setup_roadmap_runtime();
        handle.execute_await("msg_send", &[
            Value::string("agent1"),
            Value::string("hello from main"),
        ]).unwrap();

        let meta = meta.lock().unwrap();
        let inbox = meta.agent_mailbox.get("agent1").unwrap();
        assert_eq!(inbox.len(), 1);
        assert_eq!(inbox[0].content, "hello from main");
    }

    #[test]
    fn msg_send_empty_target_fails() {
        let (handle, _rt) = setup_roadmap_runtime();
        let result = handle.execute_await("msg_send", &[
            Value::string(""),
            Value::string("hello"),
        ]);
        assert!(result.is_err(), "should fail with empty target");
        assert!(result.unwrap_err().to_string().contains("must not be empty"));
    }

    #[test]
    fn msg_send_no_args_fails() {
        let (handle, _rt) = setup_roadmap_runtime();
        let result = handle.execute_await("msg_send", &[]);
        assert!(result.is_err(), "should fail with no args");
    }

    #[test]
    fn msg_send_wrong_type_fails() {
        let (handle, _rt) = setup_roadmap_runtime();
        let result = handle.execute_await("msg_send", &[Value::Int(42)]);
        assert!(result.is_err(), "should fail with non-String target");
    }

    #[test]
    fn msg_send_missing_message_fails() {
        let (handle, _rt) = setup_roadmap_runtime();
        let result = handle.execute_await("msg_send", &[Value::string("agent1")]);
        assert!(result.is_err(), "should fail without message arg");
    }

    // ═════════════════════════════════════════════════════════════════════
    // inbox_read builtin
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn inbox_read_returns_json_and_clears_inbox() {
        let (handle, meta) = setup_roadmap_runtime();
        {
            let mut meta = meta.lock().unwrap();
            crate::session::send_agent_message(&mut meta, "agent1", "main", "first");
            crate::session::send_agent_message(&mut meta, "agent2", "main", "second");
        }
        let rt = crate::eval::get_shared_runtime().unwrap();
        rt.write().unwrap().agent_mailbox = meta.lock().unwrap().agent_mailbox.clone();

        let result = unwrap_string(handle.execute_await("inbox_read", &[]).unwrap());
        assert_eq!(result, "[\"first\",\"second\"]");
        assert!(meta.lock().unwrap().agent_mailbox.get("main").is_none());
        assert!(rt.read().unwrap().agent_mailbox.get("main").is_none());

        let second = unwrap_string(handle.execute_await("inbox_read", &[]).unwrap());
        assert_eq!(second, "[]");
    }

    #[test]
    fn inbox_read_mock_intercepts_and_preserves_inbox() {
        let (_handle, meta) = setup_roadmap_runtime();
        {
            let mut meta = meta.lock().unwrap();
            crate::session::send_agent_message(&mut meta, "agent1", "main", "first");
        }
        let handle = CoroutineHandle::new_mock(vec![crate::session::IoMock {
            operation: "inbox_read".to_string(),
            patterns: vec!["".to_string()],
            response: "[\"mocked\"]".to_string(),
        }]);

        let result = unwrap_string(handle.execute_await("inbox_read", &[]).unwrap());
        assert_eq!(result, "[\"mocked\"]");
        let meta = meta.lock().unwrap();
        let inbox = meta.agent_mailbox.get("main").unwrap();
        assert_eq!(inbox.len(), 1);
        assert_eq!(inbox[0].content, "first");
    }

    #[test]
    fn inbox_read_with_args_fails() {
        let (handle, _meta) = setup_roadmap_runtime();
        let result = handle.execute_await("inbox_read", &[Value::string("unexpected")]);
        assert!(result.is_err(), "should fail with extra args");
        assert!(result.unwrap_err().to_string().contains("expects no arguments"));
    }

    // ═════════════════════════════════════════════════════════════════════
    // watch_start builtin
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn watch_start_queues_command() {
        // Set up with a function to watch
        let (handle, _prog) = setup_mutation_runtime(
            "+fn checker ()->Int\n  +return 42\n+end"
        );
        let result = unwrap_string(
            handle.execute_await("watch_start", &[
                Value::string("checker"),
                Value::Int(1000),
            ]).unwrap()
        );
        assert!(result.contains("Watching checker"), "confirmation: {result}");
        assert!(result.contains("1000ms"), "should mention interval: {result}");

        // Verify command was queued
        let rt = crate::eval::get_shared_runtime().unwrap();
        let state = rt.read().unwrap();
        assert_eq!(state.pending_commands.len(), 1);
        assert!(state.pending_commands[0].contains("!watch checker 1000"));
    }

    #[test]
    fn watch_start_function_not_found_fails() {
        let handle = setup_query_runtime("");
        let result = handle.execute_await("watch_start", &[
            Value::string("nonexistent"),
            Value::Int(1000),
        ]);
        assert!(result.is_err(), "should fail for missing function");
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn watch_start_zero_interval_fails() {
        let handle = setup_query_runtime(
            "+fn x ()->Int\n  +return 1\n+end"
        );
        let result = handle.execute_await("watch_start", &[
            Value::string("x"),
            Value::Int(0),
        ]);
        assert!(result.is_err(), "should fail with zero interval");
    }

    #[test]
    fn watch_start_negative_interval_fails() {
        let handle = setup_query_runtime(
            "+fn x ()->Int\n  +return 1\n+end"
        );
        let result = handle.execute_await("watch_start", &[
            Value::string("x"),
            Value::Int(-100),
        ]);
        assert!(result.is_err(), "should fail with negative interval");
    }

    #[test]
    fn watch_start_wrong_type_fn_name_fails() {
        let (handle, _) = setup_roadmap_runtime();
        let result = handle.execute_await("watch_start", &[Value::Int(42)]);
        assert!(result.is_err(), "should fail with non-String fn_name");
    }

    #[test]
    fn watch_start_empty_fn_name_fails() {
        let handle = setup_query_runtime(
            "+fn x ()->Int\n  +return 1\n+end"
        );
        let result = handle.execute_await("watch_start", &[
            Value::string(""),
            Value::Int(1000),
        ]);
        assert!(result.is_err(), "should fail with empty fn_name");
    }

    // ═════════════════════════════════════════════════════════════════════
    // agent_spawn builtin
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn agent_spawn_queues_command() {
        let (handle, _rt) = setup_roadmap_runtime();
        let result = unwrap_string(
            handle.execute_await("agent_spawn", &[
                Value::string("worker1"),
                Value::string("new-only"),
                Value::string("Build a calculator module"),
            ]).unwrap()
        );
        assert!(result.contains("Agent 'worker1' spawned"), "confirmation: {result}");
        assert!(result.contains("new-only"), "should mention scope: {result}");

        // Verify command was queued
        let rt = crate::eval::get_shared_runtime().unwrap();
        let state = rt.read().unwrap();
        assert_eq!(state.pending_commands.len(), 1);
        assert!(state.pending_commands[0].contains("!agent worker1"));
        assert!(state.pending_commands[0].contains("--scope new-only"));
        assert!(state.pending_commands[0].contains("Build a calculator module"));
    }

    #[test]
    fn agent_spawn_empty_name_fails() {
        let (handle, _rt) = setup_roadmap_runtime();
        let result = handle.execute_await("agent_spawn", &[
            Value::string(""),
            Value::string("full"),
            Value::string("do something"),
        ]);
        assert!(result.is_err(), "should fail with empty name");
        assert!(result.unwrap_err().to_string().contains("must not be empty"));
    }

    #[test]
    fn agent_spawn_empty_task_fails() {
        let (handle, _rt) = setup_roadmap_runtime();
        let result = handle.execute_await("agent_spawn", &[
            Value::string("worker"),
            Value::string("full"),
            Value::string(""),
        ]);
        assert!(result.is_err(), "should fail with empty task");
    }

    #[test]
    fn agent_spawn_no_args_fails() {
        let (handle, _rt) = setup_roadmap_runtime();
        let result = handle.execute_await("agent_spawn", &[]);
        assert!(result.is_err(), "should fail with no args");
    }

    #[test]
    fn agent_spawn_missing_task_fails() {
        let (handle, _rt) = setup_roadmap_runtime();
        let result = handle.execute_await("agent_spawn", &[
            Value::string("worker"),
            Value::string("full"),
        ]);
        assert!(result.is_err(), "should fail without task arg");
    }

    // ═════════════════════════════════════════════════════════════════════
    // New builtins are registered
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn new_io_builtins_registered() {
        for name in &[
            "move_symbols", "watch_start", "agent_spawn", "msg_send", "query_inbox", "inbox_read", "inbox_clear", "trace_run",
            "route_list", "route_add", "route_remove",
            "undo", "sandbox_enter", "sandbox_merge", "sandbox_discard",
            "mock_set", "mock_clear", "sse_send", "failure_history", "failure_patterns",
            "module_create", "test_run", "fn_replace",
        ] {
            assert!(
                crate::builtins::is_io_builtin(name),
                "IO builtin '{name}' should be registered"
            );
            assert!(
                crate::builtins::is_builtin(name),
                "'{name}' should also return true for is_builtin"
            );
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // route_list builtin
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn route_list_empty() {
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState {
            http_routes: vec![],
            ..Default::default()
        }));
        crate::eval::set_shared_runtime(Some(rt));
        let handle = CoroutineHandle::new_mock(vec![]);
        let result = unwrap_string(handle.execute_await("route_list", &[]).unwrap());
        assert_eq!(result, "No routes registered.");
    }

    #[test]
    fn route_list_with_routes() {
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState {
            http_routes: vec![
                crate::ast::HttpRoute {
                    method: "GET".into(), path: "/api/foo".into(), handler_fn: "Mod.foo".into(),
                },
                crate::ast::HttpRoute {
                    method: "POST".into(), path: "/api/bar".into(), handler_fn: "Mod.bar".into(),
                },
            ],
            ..Default::default()
        }));
        crate::eval::set_shared_runtime(Some(rt));
        let handle = CoroutineHandle::new_mock(vec![]);
        let result = unwrap_string(handle.execute_await("route_list", &[]).unwrap());
        assert!(result.contains("GET /api/foo -> `Mod.foo`"), "should list first route: {result}");
        assert!(result.contains("POST /api/bar -> `Mod.bar`"), "should list second route: {result}");
    }

    // ═════════════════════════════════════════════════════════════════════
    // route_add builtin
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn route_add_new_route() {
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState::default()));
        crate::eval::set_shared_runtime(Some(rt.clone()));
        let handle = CoroutineHandle::new_mock(vec![]);
        let result = unwrap_string(handle.execute_await("route_add", &[
            Value::string("POST"),
            Value::string("/api/test"),
            Value::string("Handler.test"),
        ]).unwrap());
        assert!(result.contains("added route POST /api/test"), "should confirm add: {result}");
        assert_eq!(rt.read().unwrap().http_routes.len(), 1);
        assert_eq!(rt.read().unwrap().http_routes[0].handler_fn, "Handler.test");
    }

    #[test]
    fn route_add_upserts_existing() {
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState {
            http_routes: vec![crate::ast::HttpRoute {
                method: "GET".into(), path: "/api/data".into(), handler_fn: "Old.handler".into(),
            }],
            ..Default::default()
        }));
        crate::eval::set_shared_runtime(Some(rt.clone()));
        let handle = CoroutineHandle::new_mock(vec![]);
        let result = unwrap_string(handle.execute_await("route_add", &[
            Value::string("GET"),
            Value::string("/api/data"),
            Value::string("New.handler"),
        ]).unwrap());
        assert!(result.contains("updated route"), "should say updated: {result}");
        assert!(result.contains("Old.handler"), "should mention old handler: {result}");
        assert_eq!(rt.read().unwrap().http_routes.len(), 1);
        assert_eq!(rt.read().unwrap().http_routes[0].handler_fn, "New.handler");
    }

    #[test]
    fn route_add_invalid_method() {
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState::default()));
        crate::eval::set_shared_runtime(Some(rt));
        let handle = CoroutineHandle::new_mock(vec![]);
        let result = handle.execute_await("route_add", &[
            Value::string("FOOBAR"),
            Value::string("/api/x"),
            Value::string("H.x"),
        ]);
        assert!(result.is_err(), "invalid method should fail");
        assert!(result.unwrap_err().to_string().contains("method must be"), "should mention valid methods");
    }

    #[test]
    fn route_add_invalid_path() {
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState::default()));
        crate::eval::set_shared_runtime(Some(rt));
        let handle = CoroutineHandle::new_mock(vec![]);
        let result = handle.execute_await("route_add", &[
            Value::string("GET"),
            Value::string("no-leading-slash"),
            Value::string("H.x"),
        ]);
        assert!(result.is_err(), "path without / should fail");
    }

    // ═════════════════════════════════════════════════════════════════════
    // route_remove builtin
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn route_remove_existing() {
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState {
            http_routes: vec![crate::ast::HttpRoute {
                method: "POST".into(), path: "/api/rm".into(), handler_fn: "Rm.handler".into(),
            }],
            ..Default::default()
        }));
        crate::eval::set_shared_runtime(Some(rt.clone()));
        let handle = CoroutineHandle::new_mock(vec![]);
        let result = unwrap_string(handle.execute_await("route_remove", &[
            Value::string("POST"),
            Value::string("/api/rm"),
        ]).unwrap());
        assert!(result.contains("removed route"), "should confirm removal: {result}");
        assert_eq!(rt.read().unwrap().http_routes.len(), 0);
    }

    #[test]
    fn route_remove_not_found() {
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState::default()));
        crate::eval::set_shared_runtime(Some(rt));
        let handle = CoroutineHandle::new_mock(vec![]);
        let result = handle.execute_await("route_remove", &[
            Value::string("GET"),
            Value::string("/nonexistent"),
        ]);
        assert!(result.is_err(), "removing nonexistent route should fail");
        assert!(result.unwrap_err().to_string().contains("no route found"), "should say no route found");
    }

    // ═════════════════════════════════════════════════════════════════════
    // undo builtin
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn undo_queues_command() {
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState::default()));
        crate::eval::set_shared_runtime(Some(rt.clone()));
        let handle = CoroutineHandle::new_mock(vec![]);
        let result = unwrap_string(handle.execute_await("undo", &[]).unwrap());
        assert!(result.contains("Undo queued"), "should confirm queued: {result}");
        let state = rt.read().unwrap();
        assert_eq!(state.pending_commands.len(), 1);
        assert_eq!(state.pending_commands[0], "!undo");
    }

    // ═════════════════════════════════════════════════════════════════════
    // sandbox builtins
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn sandbox_enter_queues_command() {
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState::default()));
        crate::eval::set_shared_runtime(Some(rt.clone()));
        let handle = CoroutineHandle::new_mock(vec![]);
        let result = unwrap_string(handle.execute_await("sandbox_enter", &[]).unwrap());
        assert!(result.contains("Sandbox enter queued"), "should confirm queued: {result}");
        assert_eq!(rt.read().unwrap().pending_commands[0], "!sandbox enter");
    }

    #[test]
    fn sandbox_merge_queues_command() {
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState::default()));
        crate::eval::set_shared_runtime(Some(rt.clone()));
        let handle = CoroutineHandle::new_mock(vec![]);
        let result = unwrap_string(handle.execute_await("sandbox_merge", &[]).unwrap());
        assert!(result.contains("Sandbox merge queued"), "should confirm queued: {result}");
        assert_eq!(rt.read().unwrap().pending_commands[0], "!sandbox merge");
    }

    #[test]
    fn sandbox_discard_queues_command() {
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState::default()));
        crate::eval::set_shared_runtime(Some(rt.clone()));
        let handle = CoroutineHandle::new_mock(vec![]);
        let result = unwrap_string(handle.execute_await("sandbox_discard", &[]).unwrap());
        assert!(result.contains("Sandbox discard queued"), "should confirm queued: {result}");
        assert_eq!(rt.read().unwrap().pending_commands[0], "!sandbox discard");
    }

    // ═════════════════════════════════════════════════════════════════════
    // mock_set and mock_clear builtins
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn mock_set_adds_mock() {
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState::default()));
        let meta = std::sync::Arc::new(std::sync::Mutex::new(crate::session::SessionMeta::new()));
        crate::eval::set_shared_runtime(Some(rt));
        crate::eval::set_shared_meta(Some(meta.clone()));
        let handle = CoroutineHandle::new_mock(vec![]);
        let result = unwrap_string(handle.execute_await("mock_set", &[
            Value::string("http_get"),
            Value::string("example.com"),
            Value::string("mock response body"),
        ]).unwrap());
        assert!(result.contains("mock: http_get"), "should confirm mock: {result}");
        let m = meta.lock().unwrap();
        assert_eq!(m.io_mocks.len(), 1);
        assert_eq!(m.io_mocks[0].operation, "http_get");
        assert_eq!(m.io_mocks[0].response, "mock response body");
    }

    #[test]
    fn mock_set_empty_operation_fails() {
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState::default()));
        let meta = std::sync::Arc::new(std::sync::Mutex::new(crate::session::SessionMeta::new()));
        crate::eval::set_shared_runtime(Some(rt));
        crate::eval::set_shared_meta(Some(meta));
        let handle = CoroutineHandle::new_mock(vec![]);
        let result = handle.execute_await("mock_set", &[
            Value::string(""),
            Value::string("pattern"),
            Value::string("response"),
        ]);
        assert!(result.is_err(), "empty operation should fail");
    }

    #[test]
    fn mock_clear_clears_all_mocks() {
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState::default()));
        let mut initial_meta = crate::session::SessionMeta::new();
        initial_meta.io_mocks = vec![
            crate::session::IoMock {
                operation: "http_get".into(),
                patterns: vec!["x".into()],
                response: "y".into(),
            },
            crate::session::IoMock {
                operation: "http_post".into(),
                patterns: vec![],
                response: "z".into(),
            },
        ];
        let meta = std::sync::Arc::new(std::sync::Mutex::new(initial_meta));
        crate::eval::set_shared_runtime(Some(rt));
        crate::eval::set_shared_meta(Some(meta.clone()));
        let handle = CoroutineHandle::new_mock(vec![]);
        let result = unwrap_string(handle.execute_await("mock_clear", &[]).unwrap());
        assert!(result.contains("cleared 2 mocks"), "should report count: {result}");
        assert_eq!(meta.lock().unwrap().io_mocks.len(), 0);
    }

    #[test]
    fn mock_clear_empty_returns_zero() {
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState::default()));
        let meta = std::sync::Arc::new(std::sync::Mutex::new(crate::session::SessionMeta::new()));
        crate::eval::set_shared_runtime(Some(rt));
        crate::eval::set_shared_meta(Some(meta));
        let handle = CoroutineHandle::new_mock(vec![]);
        let result = unwrap_string(handle.execute_await("mock_clear", &[]).unwrap());
        assert!(result.contains("cleared 0 mocks"), "should report 0: {result}");
    }

    #[test]
    fn sse_send_sends_json_event() {
        let (tx, mut rx) = tokio::sync::broadcast::channel(16);
        crate::eval::set_shared_event_broadcast(Some(tx));
        let handle = CoroutineHandle::new_mock(vec![]);

        let result = unwrap_string(handle.execute_await("sse_send", &[
            Value::string("mutation"),
            Value::string("updated module"),
        ]).unwrap());
        assert_eq!(result, "sent");
        let payload = rx.try_recv().unwrap();
        assert_eq!(payload, "{\"data\":\"updated module\",\"type\":\"mutation\"}");
    }

    #[test]
    fn sse_send_without_sender_fails() {
        crate::eval::set_shared_event_broadcast(None);
        let handle = CoroutineHandle::new_mock(vec![]);
        let result = handle.execute_await("sse_send", &[
            Value::string("mutation"),
            Value::string("updated module"),
        ]);
        assert!(result.is_err(), "missing sender should fail");
        assert!(result.unwrap_err().to_string().contains("no event broadcast available"));
    }

    // ═════════════════════════════════════════════════════════════════════
    // module_create builtin
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn module_create_new_module() {
        let (handle, prog) = setup_mutation_runtime("");
        let result = unwrap_string(
            handle.execute_await("module_create", &[Value::string("MyMod")]).unwrap()
        );
        assert!(result.contains("created module"), "should confirm creation: {result}");
        let p = prog.read().unwrap();
        assert!(p.modules.iter().any(|m| m.name == "MyMod"), "module should exist");
    }

    #[test]
    fn module_create_already_exists() {
        let (handle, _prog) = setup_mutation_runtime("!module Existing");
        let result = unwrap_string(
            handle.execute_await("module_create", &[Value::string("Existing")]).unwrap()
        );
        assert!(result.contains("already exists"), "should say already exists: {result}");
    }

    #[test]
    fn module_create_lowercase_fails() {
        let (handle, _prog) = setup_mutation_runtime("");
        let result = handle.execute_await("module_create", &[Value::string("lowercase")]);
        assert!(result.is_err(), "lowercase module name should fail");
        assert!(result.unwrap_err().to_string().contains("uppercase"), "should mention uppercase");
    }

    #[test]
    fn module_create_empty_name_fails() {
        let (handle, _prog) = setup_mutation_runtime("");
        let result = handle.execute_await("module_create", &[Value::string("")]);
        assert!(result.is_err(), "empty name should fail");
    }

    // ═════════════════════════════════════════════════════════════════════
    // test_run builtin
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn test_run_with_stored_tests() {
        // Build a program with a function that has stored tests
        let source = "+fn double (x:Int)->Int\n  +return x * 2\n+end\n\
                      !test double\n  +with 3 -> expect 6\n  +with 5 -> expect 10\n";
        let ops = crate::parser::parse(source).unwrap();
        let mut program = crate::ast::Program::default();
        for op in &ops {
            match op {
                crate::parser::Operation::Test(test) => {
                    // Store the tests on the function
                    if let Some(func) = program.get_function_mut(&test.function_name) {
                        func.tests = test.cases.iter().map(|c| crate::ast::TestCase {
                            input: crate::session::format_expr_pub(&c.input),
                            expected: crate::session::format_expr_pub(&c.expected),
                            passed: true,
                            matcher: None,
                            after_checks: vec![],
                        }).collect();
                    }
                }
                _ => {
                    crate::validator::apply_and_validate(&mut program, op).unwrap();
                }
            }
        }
        program.rebuild_function_index();
        crate::eval::set_shared_program(Some(std::sync::Arc::new(program)));
        let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState::default()));
        crate::eval::set_shared_runtime(Some(rt));
        let handle = CoroutineHandle::new_mock(vec![]);

        let result = unwrap_string(
            handle.execute_await("test_run", &[Value::string("double")]).unwrap()
        );
        assert!(result.contains("PASS"), "should have passing tests: {result}");
        // Each test case should appear
        let pass_count = result.matches("PASS").count();
        assert_eq!(pass_count, 2, "should have 2 passing tests: {result}");
    }

    #[test]
    fn test_run_no_stored_tests() {
        let (handle, _prog) = setup_mutation_runtime(
            "+fn foo ()->Int\n  +return 1\n+end"
        );
        let result = unwrap_string(
            handle.execute_await("test_run", &[Value::string("foo")]).unwrap()
        );
        assert!(result.contains("no stored tests"), "should say no tests: {result}");
    }

    #[test]
    fn test_run_function_not_found() {
        let (handle, _prog) = setup_mutation_runtime("");
        let result = handle.execute_await("test_run", &[Value::string("nonexistent")]);
        assert!(result.is_err(), "nonexistent function should fail");
        assert!(result.unwrap_err().to_string().contains("not found"), "should say not found");
    }

    // ═════════════════════════════════════════════════════════════════════
    // fn_replace builtin
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn fn_replace_single_statement() {
        let (handle, prog) = setup_mutation_runtime(
            "+fn greet (name:String)->String\n  +return concat(\"Hello \", name)\n+end"
        );
        let result = unwrap_string(handle.execute_await("fn_replace", &[
            Value::string("greet.s1"),
            Value::string("  +return concat(\"Hi \", name)"),
        ]).unwrap());
        // Should succeed
        assert!(!result.is_empty(), "should return a message: {result}");
        // Verify the function was modified
        let p = prog.read().unwrap();
        let func = p.get_function("greet").expect("greet should still exist");
        // The body should have the replaced statement
        assert_eq!(func.body.len(), 1, "should still have 1 statement");
    }

    #[test]
    fn fn_replace_empty_target_fails() {
        let (handle, _prog) = setup_mutation_runtime(
            "+fn dummy ()->Int\n  +return 1\n+end"
        );
        let result = handle.execute_await("fn_replace", &[
            Value::string(""),
            Value::string("  +return 2"),
        ]);
        assert!(result.is_err(), "empty target should fail");
    }

    #[test]
    fn fn_replace_empty_code_fails() {
        let (handle, _prog) = setup_mutation_runtime(
            "+fn dummy ()->Int\n  +return 1\n+end"
        );
        let result = handle.execute_await("fn_replace", &[
            Value::string("dummy.s1"),
            Value::string(""),
        ]);
        assert!(result.is_err(), "empty code should fail");
    }

    #[test]
    fn fn_replace_nonexistent_function_fails() {
        let (handle, _prog) = setup_mutation_runtime("");
        let result = handle.execute_await("fn_replace", &[
            Value::string("nonexistent.s1"),
            Value::string("  +return 42"),
        ]);
        assert!(result.is_err(), "replacing in nonexistent function should fail");
    }

    // ── library_reload ──

    #[test]
    fn library_reload_nonexistent_module_fails() {
        let (handle, _prog) = setup_mutation_runtime("");
        let result = handle.execute_await("library_reload", &[
            Value::string("NonExistentModule99999"),
        ]);
        assert!(result.is_err(), "should fail for nonexistent module");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not found") || err.contains("could not"),
            "error should mention not found: {err}"
        );
    }

    #[test]
    fn library_reload_empty_name_reloads_all() {
        let (handle, _prog) = setup_mutation_runtime("");
        // With empty name, it reloads all modules from the library dir.
        // This should succeed even if the library dir is empty.
        let result = handle.execute_await("library_reload", &[
            Value::string(""),
        ]);
        assert!(result.is_ok(), "should succeed with empty name: {result:?}");
        let msg = unwrap_string(result.unwrap());
        assert!(msg.contains("Reloaded"), "should report reloaded: {msg}");
    }

    // ── library_errors ──

    #[test]
    fn library_errors_no_errors() {
        let (handle, _rt) = setup_roadmap_runtime();
        let result = unwrap_string(handle.execute_await("library_errors", &[]).unwrap());
        assert_eq!(result, "No library errors.");
    }

    #[test]
    fn library_errors_with_load_errors() {
        let (handle, _meta) = setup_roadmap_runtime();
        let rt = crate::eval::get_shared_runtime().unwrap();
        // Add some library load errors to the runtime state
        if let Ok(mut state) = rt.write() {
            state.library_load_errors = vec![
                ("BadModule".to_string(), "parse error on line 5".to_string()),
                ("BrokenMod".to_string(), "no !module declaration found".to_string()),
            ];
        }
        let result = unwrap_string(handle.execute_await("library_errors", &[]).unwrap());
        assert!(result.contains("Load errors (2):"), "should show load error count: {result}");
        assert!(result.contains("BadModule: parse error on line 5"), "should show first error: {result}");
        assert!(result.contains("BrokenMod: no !module declaration found"), "should show second error: {result}");
    }

    #[test]
    fn library_errors_with_general_errors() {
        let (handle, _meta) = setup_roadmap_runtime();
        let rt = crate::eval::get_shared_runtime().unwrap();
        if let Ok(mut state) = rt.write() {
            state.library_errors = vec![
                "failed to persist module `Foo`: disk full".to_string(),
            ];
        }
        let result = unwrap_string(handle.execute_await("library_errors", &[]).unwrap());
        assert!(result.contains("Errors this session (1):"), "should show session error count: {result}");
        assert!(result.contains("failed to persist module `Foo`: disk full"), "should show the error: {result}");
    }

    #[test]
    fn library_errors_with_both_error_types() {
        let (handle, _meta) = setup_roadmap_runtime();
        let rt = crate::eval::get_shared_runtime().unwrap();
        if let Ok(mut state) = rt.write() {
            state.library_load_errors = vec![
                ("FailedMod".to_string(), "syntax error".to_string()),
            ];
            state.library_errors = vec![
                "could not read library dir".to_string(),
            ];
        }
        let result = unwrap_string(handle.execute_await("library_errors", &[]).unwrap());
        assert!(result.contains("Load errors (1):"), "should show load errors: {result}");
        assert!(result.contains("FailedMod: syntax error"), "should show load error detail: {result}");
        assert!(result.contains("Errors this session (1):"), "should show session errors: {result}");
        assert!(result.contains("could not read library dir"), "should show session error detail: {result}");
    }
}
