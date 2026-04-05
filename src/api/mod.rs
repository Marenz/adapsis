// test comment
//! HTTP API for AdapsisOS — programmatic access to the session.
//!
//! Endpoints:
//!   POST /api/mutate     — apply Adapsis source code
//!   POST /api/eval       — evaluate a function call
//!   POST /api/test       — run tests for a function
//!   GET  /api/query      — semantic query (?symbols, ?callers, etc.)
//!   GET  /api/status     — program state + revision info
//!   GET  /api/history    — mutation log + working history
//!   POST /api/rewind     — rewind to a revision
//!   POST /api/ask        — send task to LLM, apply generated code

use std::sync::Arc;

use axum::extract::State;
use axum::response::Html;
use axum::Json;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::eval;
use crate::parser;
use crate::session::{RuntimeState, SandboxState, SessionMeta};
use crate::typeck;

/// Default timeout for `!eval` execution (seconds).
pub(crate) const EVAL_TIMEOUT_SECS: u64 = 30;
use crate::validator;

pub mod execute;
pub use execute::{
    execute_code,
    CodeExecutionResult,
    AgentCompletionCallback,
    format_tasks,
    format_inspect_task,
    parse_inspect_task_query,
};
pub mod llm_handlers;
pub use llm_handlers::{ask, ask_stream, handle_llm_takeover};
use llm_handlers::EventSender;

impl AppConfig {
    /// Build a temporary working snapshot from the live tiers.
    async fn snapshot_working_set(&self) -> WorkingSet {
        WorkingSet {
            program: self.program.read().await.clone(),
            runtime: self.runtime.read().unwrap().clone(),
            meta: self.meta.lock().unwrap().clone(),
            sandbox: None,
        }
    }

    /// Write a temporary working snapshot back into the live tiers.
    async fn write_back_working_set(&self, ws: &WorkingSet) {
        *self.program.write().await = ws.program.clone();
        *self.meta.lock().unwrap() = ws.meta.clone();
        if let Ok(mut rt) = self.runtime.write() {
            rt.shared_vars = ws.runtime.shared_vars.clone();
            rt.http_routes = ws.runtime.http_routes.clone();
            rt.agent_mailbox = ws.meta.agent_mailbox.clone();
            // Sync library errors from LibraryState into RuntimeState
            if let Some(ref lib_state) = ws.meta.library_state {
                if let Ok(errs) = lib_state.errors.lock() {
                    rt.library_errors = errs.clone();
                }
                if let Ok(load_errs) = lib_state.load_errors.lock() {
                    rt.library_load_errors = load_errs.iter()
                        .map(|le| (le.module_name.clone(), le.error.clone()))
                        .collect();
                }
            }
        }
        self.notify_save();
    }

    /// Pull state that async eval/top-level IO can mutate via SHARED_META/
    /// SHARED_RUNTIME back into the local working snapshot before it gets
    /// written over by stale data.
    fn sync_async_side_effects_into(&self, ws: &mut WorkingSet) {
        if let Ok(meta) = self.meta.lock() {
            ws.meta.roadmap = meta.roadmap.clone();
            ws.meta.plan = meta.plan.clone();
            ws.meta.agent_mailbox = meta.agent_mailbox.clone();
            ws.meta.io_mocks = meta.io_mocks.clone();
        }
        if let Ok(runtime) = self.runtime.read() {
            ws.runtime = runtime.clone();
        }
    }
}

fn emit_event(state: &AppConfig, event_json: &str) {
    let _ = state.event_broadcast.send(event_json.to_string());
}

fn make_sse_event(event: &str, payload: serde_json::Value) -> String {
    let mut obj = match payload {
        serde_json::Value::Object(map) => map,
        other => {
            let mut map = serde_json::Map::new();
            map.insert("data".to_string(), other);
            map
        }
    };
    obj.insert("type".to_string(), serde_json::Value::String(event.to_string()));
    serde_json::Value::Object(obj).to_string()
}

fn encode_broadcast_event(event: &serde_json::Value) -> String {
    if let (Some(kind), Some(data)) = (event.get("event").and_then(|v| v.as_str()), event.get("data")) {
        return make_sse_event(kind, serde_json::json!({
            "data": data.as_str().map(str::to_owned).unwrap_or_else(|| data.to_string())
        }));
    }

    let kind = event.get("type").and_then(|v| v.as_str()).unwrap_or("message");
    let data = event.get("message")
        .or_else(|| event.get("detail"))
        .or_else(|| event.get("text"))
        .or_else(|| event.get("code"))
        .or_else(|| event.get("result"))
        .map(|v| v.as_str().map(str::to_owned).unwrap_or_else(|| v.to_string()))
        .unwrap_or_else(|| event.to_string());
    make_sse_event(kind, serde_json::json!({"data": data}))
}

/// Temporary mutable working state used by complex handlers.
/// Live truth is still the three tiers in AppConfig; this is just a local bag.
pub(crate) struct WorkingSet {
    program: crate::ast::Program,
    runtime: RuntimeState,
    meta: SessionMeta,
    sandbox: Option<SandboxState>,
}

/// Thread-safe session manager: maps session IDs to independent Program instances.
/// The "main" session uses the existing `session` field in AppConfig; additional
/// sessions are stored here with isolated Program state.
pub type SessionManager = Arc<Mutex<std::collections::HashMap<String, Arc<tokio::sync::Mutex<crate::ast::Program>>>>>;

/// Extended state for the API, including LLM and OpenCode configuration.
#[derive(Clone)]
pub struct AppConfig {
    /// Tier 1: Program AST — read-heavy, use read() for queries, write() briefly for mutations
    pub program: std::sync::Arc<tokio::sync::RwLock<crate::ast::Program>>,
    /// Tier 3: Session metadata — chat history, plan, roadmap, mocks, mutation log.
    /// Brief locks only; never hold during LLM calls or IO.
    /// Uses std::sync::Mutex (not tokio) since we never hold across .await points.
    /// This is also the SharedMeta passed to spawn_blocking tasks via set_shared_meta().
    pub meta: crate::session::SharedMeta,
    pub llm_url: String,
    pub llm_model: String,
    pub llm_api_key: Option<String>,
    pub project_dir: String,
    pub io_sender: Option<tokio::sync::mpsc::Sender<crate::coroutine::IoRequest>>,
    /// Channel for self-triggering: system events that should invoke the AI
    pub self_trigger: tokio::sync::mpsc::Sender<String>,
    /// Task registry for tracking spawned async tasks
    pub task_registry: Option<crate::coroutine::TaskRegistry>,
    /// Task snapshot registry for live interpreter state inspection
    pub snapshot_registry: Option<crate::coroutine::TaskSnapshotRegistry>,
    /// Structured log file for AI activity logging
    pub log_file: Option<std::sync::Arc<tokio::sync::Mutex<tokio::fs::File>>>,
    /// JIT compilation cache — reuses compiled modules across evals when revision unchanged
    pub jit_cache: crate::eval::JitCache,
    /// Broadcast channel for SSE events — all activity visible to all subscribers (web UI)
    pub event_broadcast: tokio::sync::broadcast::Sender<String>,
    /// Directory where !opencode runs and builds (fixed checkout, AdapsisOS must run from here)
    pub opencode_git_dir: String,
    /// Sequential lock for !opencode — only one at a time
    pub opencode_lock: std::sync::Arc<tokio::sync::Mutex<()>>,
    /// Message queue for injecting messages into the autonomous loop
    pub message_queue: std::sync::Arc<tokio::sync::Mutex<Vec<String>>>,
    /// Maximum iterations per AI request
    pub max_iterations: usize,
    /// JSONL training data log — one entry per iteration with input/output/outcome
    pub training_log: Option<std::sync::Arc<tokio::sync::Mutex<tokio::fs::File>>>,
    /// Shared runtime state (Tier 2) — HTTP routes, shared variables
    pub runtime: crate::session::SharedRuntime,
    /// Multi-session manager: maps session IDs to independent Program instances.
    /// The "main" session uses the existing `session` field above.
    pub sessions: SessionManager,
    /// Save notification channel — send `()` to trigger a debounced save.
    /// `None` in tests and in temporary AppConfig instances (e.g. llm_takeover).
    pub save_notify: Option<tokio::sync::mpsc::Sender<()>>,
}

impl AppConfig {
    /// Install the shared runtime, meta, and event broadcast into the current
    /// thread's thread-locals. Call at the top of async handler functions that
    /// run eval code on the tokio thread (before any spawn_blocking).
    pub fn install_handler_locals(&self) {
        crate::eval::set_shared_runtime(Some(self.runtime.clone()));
        crate::eval::set_shared_meta(Some(self.meta.clone()));
        crate::eval::set_shared_event_broadcast(Some(self.event_broadcast.clone()));
    }

    /// Signal that state has changed and a save should be performed.
    /// Uses `try_send` so it never blocks — if a save is already pending the
    /// notification is silently dropped (the pending save will cover it).
    pub fn notify_save(&self) {
        if let Some(ref tx) = self.save_notify {
            let _ = tx.try_send(());
        }
    }
}

#[derive(Deserialize)]
pub struct MutateRequest {
    pub source: String,
}

#[derive(Serialize)]
pub struct MutateResponse {
    pub revision: usize,
    pub results: Vec<MutationResult>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MutationResult {
    pub message: String,
    pub success: bool,
}

pub async fn mutate(
    State(config): State<AppConfig>,
    Json(req): Json<MutateRequest>,
) -> Json<MutateResponse> {
    eprintln!("[web:mutate] {}", req.source.chars().take(100).collect::<String>());
    let mut program = config.program.read().await.clone();
    let mut runtime = config.runtime.read().unwrap().clone();
    let mut meta = config.meta.lock().unwrap().clone();
    let mut sandbox = None;
    match crate::session::apply_to_tiers_async(&mut program, &mut runtime, &mut meta, &mut sandbox, &req.source, config.io_sender.as_ref()).await {
        Ok(results) => {
            let response_results: Vec<MutationResult> = results
                .into_iter()
                .map(|(message, success)| {
                    let message = if success {
                        message
                    } else if let Some(hint) = crate::session::recent_failure_hint(&runtime, &message) {
                        format!("{message}\n{hint}")
                    } else {
                        message
                    };
                    MutationResult { message, success }
                })
                .collect();
            let summary = if response_results.is_empty() {
                "Applied 0 mutations".to_string()
            } else {
                response_results.iter()
                    .map(|r| r.message.clone())
                    .collect::<Vec<_>>()
                    .join("; ")
            };
            *config.program.write().await = program;
            *config.runtime.write().unwrap() = runtime;
            *config.meta.lock().unwrap() = meta.clone();
            config.notify_save();
            emit_event(&config, &make_sse_event("mutation", serde_json::json!({
                "revision": meta.revision,
                "summary": summary,
            })));
            Json(MutateResponse { revision: meta.revision, results: response_results })
        }
        Err(e) => {
            let message = format!("error: {e}");
            let message = if let Some(hint) = crate::session::recent_failure_hint(&runtime, &message) {
                format!("{message}\n{hint}")
            } else {
                message
            };
            *config.runtime.write().unwrap() = runtime;
            Json(MutateResponse {
                revision: meta.revision,
                results: vec![MutationResult {
                    message,
                    success: false,
                }],
            })
        },
    }
}

#[derive(Deserialize)]
pub struct EvalRequest {
    pub function: String,
    #[serde(default)]
    pub input: String,
    /// Inline expression to evaluate directly (e.g. "1 + 2", "concat(\"a\", \"b\")").
    /// When set, `function` and `input` are ignored.
    #[serde(default)]
    pub expression: Option<String>,
}

#[derive(Serialize)]
pub struct EvalResponse {
    pub result: String,
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compiled: Option<bool>,
}

pub async fn eval_fn(
    State(config): State<AppConfig>,
    Json(req): Json<EvalRequest>,
) -> Json<EvalResponse> {
    config.install_handler_locals();

    // Handle inline expression evaluation (e.g. "1 + 2", "concat(\"a\", \"b\")")
    if let Some(ref expr_str) = req.expression {
        eprintln!("[web:eval] inline: {expr_str}");
        let expr_str = expr_str.trim();
        if expr_str.is_empty() {
            let response = EvalResponse {
                result: "empty expression".to_string(),
                success: false,
                compiled: None,
            };
            emit_event(&config, &make_sse_event("eval", serde_json::json!({
                "expression": expr_str,
                "result": response.result.clone(),
            })));
            return Json(response);
        }
        match parser::parse_expr_pub(0, expr_str) {
            Ok(expr) => {
                // Check if the expression contains IO builtins — run async if so
                if eval::expr_contains_io_builtin(&expr) {
                    if let Some(sender) = &config.io_sender {
                        let program = config.program.read().await.clone();
                        let program_mut = crate::eval::make_shared_program_mut(&program);
                        let sender = sender.clone();
                        let ctx = eval::EvalContext::new(config.runtime.clone(), config.meta.clone(), config.event_broadcast.clone(), &program, program_mut.clone());
                        let eval_result = tokio::task::spawn_blocking(move || {
                            ctx.install();
                            eval::eval_inline_expr_with_io(&program, &expr, sender)
                        }).await;
                        // Sync mutations back to session if any occurred
                        if let Some(mutated) = crate::eval::read_back_program_mutations(&program_mut) {
                            *config.program.write().await = mutated;
                        }
                        let response = match eval_result {
                            Ok(Ok(val)) => EvalResponse {
                                result: format!("{val}"),
                                success: true,
                                compiled: Some(false),
                            },
                            Ok(Err(e)) => EvalResponse {
                                result: format!("{e}"),
                                success: false,
                                compiled: None,
                            },
                            Err(e) => EvalResponse {
                                result: format!("task error: {e}"),
                                success: false,
                                compiled: None,
                            },
                        };
                        emit_event(&config, &make_sse_event("eval", serde_json::json!({
                            "expression": expr_str,
                            "result": response.result.clone(),
                        })));
                        return Json(response);
                    }
                    // No IO sender available — fall through to sync eval which will error
                }
                // Tier 1: read program briefly for eval
                let program = config.program.read().await;
                match eval::eval_inline_expr(&program, &expr) {
                    Ok(val) => {
                        let response = EvalResponse {
                            result: format!("{val}"),
                            success: true,
                            compiled: Some(false),
                        };
                        emit_event(&config, &make_sse_event("eval", serde_json::json!({
                            "expression": expr_str,
                            "result": response.result.clone(),
                        })));
                        return Json(response);
                    }
                    Err(e) => {
                        let response = EvalResponse {
                            result: format!("{e}"),
                            success: false,
                            compiled: None,
                        };
                        emit_event(&config, &make_sse_event("eval", serde_json::json!({
                            "expression": expr_str,
                            "result": response.result.clone(),
                        })));
                        return Json(response);
                    }
                }
            }
            Err(e) => {
                let response = EvalResponse {
                    result: format!("parse error: {e}"),
                    success: false,
                    compiled: None,
                };
                emit_event(&config, &make_sse_event("eval", serde_json::json!({
                    "expression": expr_str,
                    "result": response.result.clone(),
                })));
                return Json(response);
            }
        }
    }

    eprintln!("[web:eval] {} {}", req.function, req.input);

    // Parse input directly instead of reconstructing "!eval fn input" source
    // which breaks when the input contains unescaped quotes or special chars.
    let input_expr = if req.input.trim().is_empty() {
        parser::Expr::StructLiteral(vec![])
    } else {
        match parser::parse_test_input(0, &req.input) {
            Ok(expr) => expr,
            Err(e) => {
                let response = EvalResponse {
                    result: format!("parse error: {e}"),
                    success: false,
                    compiled: None,
                };
                emit_event(&config, &make_sse_event("eval", serde_json::json!({
                    "expression": req.input,
                    "result": response.result.clone(),
                })));
                return Json(response);
            }
        }
    };
    let ev = parser::EvalMutation {
        function_name: req.function.clone(),
        input: input_expr,
        inline_expr: None,
    };

    // Tier 1: read program to check testedness and get function info
    // Clone what we need so the lock is released before eval runs
    let (_require_modules, needs_async, revision) = {
        let program = config.program.read().await;
        let meta = config.meta.lock().unwrap();

        // Block eval of untested functions (>2 statements) in AdapsisOS mode
        if program.require_modules {
            if let Some(func) = program.get_function(&ev.function_name) {
                let is_tested = !func.tests.is_empty() && func.tests.iter().all(|t| t.passed);
                if func.body.len() > 2 && !is_tested {
                    let response = EvalResponse {
                        result: format!("function `{}` has {} statements but no passing tests. Write +test blocks first.", ev.function_name, func.body.len()),
                        success: false,
                        compiled: None,
                    };
                    emit_event(&config, &make_sse_event("eval", serde_json::json!({
                        "expression": ev.function_name,
                        "result": response.result.clone(),
                    })));
                    return Json(response);
                }
            }
        }

        let needs_async = program.get_function(&ev.function_name)
            .is_some_and(|f| f.effects.iter().any(|e|
                matches!(e, crate::ast::Effect::Io | crate::ast::Effect::Async)));

        (program.require_modules, needs_async, meta.revision)
    };

    if needs_async {
        if let Some(sender) = &config.io_sender {
            // Tier 1: clone program for the blocking task (lock released before blocking)
            let program = config.program.read().await.clone();
            let program_mut = crate::eval::make_shared_program_mut(&program);
            let fn_name = ev.function_name.clone();
            let input = ev.input.clone();
            let sender = sender.clone();
            let ctx = eval::EvalContext::new(config.runtime.clone(), config.meta.clone(), config.event_broadcast.clone(), &program, program_mut.clone());

            let eval_result = tokio::task::spawn_blocking(move || {
                ctx.install();
                let func = program.get_function(&fn_name)
                    .ok_or_else(|| anyhow::anyhow!("function not found"))?;
                let handle = crate::coroutine::CoroutineHandle::new(sender);
                let mut env = eval::Env::new_with_shared_interner(&program.shared_interner);
                env.populate_shared_from_program(&program);
                env.set("__coroutine_handle", eval::Value::CoroutineHandle(handle));
                let input_val = eval::eval_parser_expr_with_program(&input, &program)?;
                eval::bind_input_to_params(&program, func, &input_val, &mut env);
                eval::eval_function_body_pub(&program, &func.body, &mut env)
            }).await;

            // Sync mutations back to session if any occurred
            if let Some(mutated) = crate::eval::read_back_program_mutations(&program_mut) {
                *config.program.write().await = mutated;
            }

            let response = match eval_result {
                Ok(Ok(val)) => EvalResponse {
                    result: format!("{val}"),
                    success: true,
                    compiled: Some(false),
                },
                Ok(Err(e)) => EvalResponse {
                    result: format!("{e}"),
                    success: false,
                    compiled: None,
                },
                Err(e) => EvalResponse {
                    result: format!("task error: {e}"),
                    success: false,
                    compiled: None,
                },
            };
            emit_event(&config, &make_sse_event("eval", serde_json::json!({
                "expression": ev.function_name,
                "result": response.result.clone(),
            })));
            return Json(response);
        }
    }

    // Tier 1: read program for eval (no lock held during eval itself for sync path)
    let program = config.program.read().await;
    match eval::eval_compiled_or_interpreted_cached(&program, &ev.function_name, &ev.input, Some(&config.jit_cache), revision) {
        Ok((result, compiled)) => {
            drop(program); // release Tier 1 before writing Tier 3
            // Tier 3: record eval in history (brief lock)
            {
                let mut meta = config.meta.lock().unwrap();
                let rev = meta.revision;
                meta.history.push(crate::session::HistoryEntry::Eval {
                    revision: rev,
                    function: ev.function_name.clone(),
                    input: req.input.clone(),
                    result: result.clone(),
                });
            }
            let response = EvalResponse {
                result,
                success: true,
                compiled: Some(compiled),
            };
            emit_event(&config, &make_sse_event("eval", serde_json::json!({
                "expression": ev.function_name,
                "result": response.result.clone(),
            })));
            Json(response)
        }
        Err(e) => {
            let response = EvalResponse {
                result: format!("{e}"),
                success: false,
                compiled: None,
            };
            emit_event(&config, &make_sse_event("eval", serde_json::json!({
                "expression": ev.function_name,
                "result": response.result.clone(),
            })));
            Json(response)
        }
    }
}

#[derive(Deserialize)]
pub struct TestRequest {
    pub source: String,
}

#[derive(Serialize)]
pub struct TestResponse {
    pub passed: usize,
    pub failed: usize,
    pub results: Vec<TestCaseResult>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TestCaseResult {
    pub message: String,
    pub pass: bool,
}

pub async fn test_fn(
    State(config): State<AppConfig>,
    Json(req): Json<TestRequest>,
) -> Json<TestResponse> {
    config.install_handler_locals();
    let operations = match parser::parse(&req.source) {
        Ok(ops) => ops,
        Err(e) => {
            let response = TestResponse {
                passed: 0,
                failed: 1,
                results: vec![TestCaseResult {
                    message: format!("parse error: {e}"),
                    pass: false,
                }],
            };
            emit_event(&config, &make_sse_event("test", serde_json::json!({
                "function": "(parse)",
                "passed": response.passed,
                "failed": response.failed,
            })));
            return Json(response);
        }
    };

    let mut passed = 0;
    let mut failed = 0;
    let mut results = Vec::new();

    // Collect all test operations, including those embedded inside module bodies.
    let mut all_test_ops: Vec<&parser::TestMutation> = Vec::new();
    for op in &operations {
        if let parser::Operation::Test(test) = op {
            all_test_ops.push(test);
        }
        if let parser::Operation::Module(m) = op {
            for body_op in &m.body {
                if let parser::Operation::Test(test) = body_op {
                    all_test_ops.push(test);
                }
            }
        }
    }

    for test in &all_test_ops {
        let start_passed = passed;
        let start_failed = failed;
        // Tier 1: read program to check async needs; Tier 3: get mocks
        // Clone what we need so locks are released before test execution
        let (program_snapshot, needs_async, mocks, routes) = {
            let program = config.program.read().await;
            let meta = config.meta.lock().unwrap();
            let needs_async = program.get_function(&test.function_name)
                .is_some_and(|f| f.effects.iter().any(|e|
                    matches!(e, crate::ast::Effect::Io | crate::ast::Effect::Async)));
            let routes = config.runtime.read().unwrap().http_routes.clone();
            (program.clone(), needs_async, meta.io_mocks.clone(), routes)
        }; // All locks released

        for case in &test.cases {
            // NO LOCKS HELD during test execution
            let case_result = if needs_async {
                if let Some(sender) = &config.io_sender {
                    eval::eval_test_case_async(
                        &program_snapshot, &test.function_name, case, &mocks, sender.clone(), &routes,
                    ).await
                } else {
                    eval::eval_test_case_with_mocks(
                        &program_snapshot, &test.function_name, case, &mocks, &routes,
                    )
                }
            } else {
                eval::eval_test_case_with_mocks(
                    &program_snapshot, &test.function_name, case, &mocks, &routes,
                )
            };

            match case_result {
                Ok(msg) => {
                    passed += 1;
                    results.push(TestCaseResult {
                        message: msg,
                        pass: true,
                    });
                }
                Err(e) => {
                    failed += 1;
                    results.push(TestCaseResult {
                        message: format!("{e}"),
                        pass: false,
                    });
                }
            }
        }
        // Tier 3: record test results (brief lock)
        {
            let mut meta = config.meta.lock().unwrap();
            let rev = meta.revision;
            let details: Vec<String> = results.iter().map(|r| {
                format!("{}: {}", if r.pass { "PASS" } else { "FAIL" }, r.message)
            }).collect();
            meta.history.push(crate::session::HistoryEntry::Test {
                revision: rev,
                function: test.function_name.clone(),
                passed,
                failed,
                details,
            });
        }
        // Tier 1: store tests on the program if all passed (brief write lock)
        if failed == 0 && !test.cases.is_empty() {
            let mut program = config.program.write().await;
            crate::session::store_test(&mut program, &test.function_name, &test.cases);
            config.notify_save();
        }
        emit_event(&config, &make_sse_event("test", serde_json::json!({
            "function": test.function_name,
            "passed": passed - start_passed,
            "failed": failed - start_failed,
        })));
    }

    let response = TestResponse {
        passed,
        failed,
        results,
    };
    Json(response)
}

#[derive(Deserialize)]
pub struct QueryRequest {
    pub query: String,
}

#[derive(Serialize)]
pub struct QueryResponse {
    pub response: String,
}

pub async fn query(
    State(config): State<AppConfig>,
    Json(req): Json<QueryRequest>,
) -> Json<QueryResponse> {
    let response = if req.query.trim() == "?inbox" || req.query.trim().starts_with("?inbox") {
        // Tier 3: read meta (brief lock)
        let meta = config.meta.lock().unwrap();
        let msgs = meta.agent_mailbox.get("main").map(|v| v.as_slice()).unwrap_or(&[]);
        if msgs.is_empty() { "No messages.".to_string() }
        else { msgs.iter().map(|m| format!("[{}] from {}: {}", m.timestamp, m.from, m.content)).collect::<Vec<_>>().join("\n") }
    } else if req.query.trim() == "?tasks" {
        format_tasks(&config.task_registry)
    } else if let Some(task_id) = parse_inspect_task_query(req.query.trim()) {
        format_inspect_task(&config.task_registry, &config.snapshot_registry, task_id)
    } else if req.query.trim() == "?library" {
        // Tier 1 + Tier 3: read program + meta
        let program = config.program.read().await;
        let meta = config.meta.lock().unwrap();
        crate::library::query_library(&program, meta.library_state.as_ref())
    } else {
        // Tier 1 + Tier 2: read program + runtime
        let program = config.program.read().await;
        let routes = config.runtime.read().unwrap().http_routes.clone();
        let table = typeck::build_symbol_table(&program);
        typeck::handle_query(&program, &table, &req.query, &routes)
    };
    // Tier 3: brief write to record query in history
    {
        let mut meta = config.meta.lock().unwrap();
        let rev = meta.revision;
        meta.history.push(crate::session::HistoryEntry::Query {
            revision: rev,
            query: req.query.clone(),
            response: response.clone(),
        });
    }
    Json(QueryResponse { response })
}

#[derive(Serialize)]
pub struct StatusResponse {
    pub revision: usize,
    pub mutations: usize,
    pub history_entries: usize,
    pub functions: Vec<String>,
    pub types: Vec<String>,
    pub routes: Vec<RouteInfo>,
    pub program_summary: String,
    pub plan: Vec<PlanStepResponse>,
    pub roadmap: Vec<RoadmapItemResponse>,
}

#[derive(Serialize, Clone)]
pub struct RoadmapItemResponse {
    pub description: String,
    pub done: bool,
}

#[derive(Serialize, Clone)]
pub struct RouteInfo {
    pub method: String,
    pub path: String,
    pub handler_fn: String,
}

#[derive(Serialize, Clone)]
pub struct PlanStepResponse {
    pub description: String,
    pub status: String,
}

pub async fn status(State(config): State<AppConfig>) -> Json<StatusResponse> {
    // Tier 1: read program (RwLock read — non-exclusive, fast)
    let (functions, types, program_summary) = {
        let program = config.program.read().await;
        (
            program.functions.iter().map(|f| f.name.clone()).collect(),
            program.types.iter().map(|t| t.name().to_string()).collect(),
            validator::program_summary(&program),
        )
    };
    // Tier 2: read runtime (RwLock read — non-exclusive, fast)
    let routes = {
        let rt = config.runtime.read().unwrap();
        rt.http_routes.iter().map(|r| RouteInfo {
            method: r.method.clone(),
            path: r.path.clone(),
            handler_fn: r.handler_fn.clone(),
        }).collect()
    };
    // Tier 3: read meta (brief lock)
    let (revision, mutations, history_entries, plan, roadmap) = {
        let meta = config.meta.lock().unwrap();
        (
            meta.revision,
            meta.mutations.len(),
            meta.history.len(),
            meta.plan.iter().map(|s| PlanStepResponse {
                description: s.description.clone(),
                status: match s.status {
                    crate::session::PlanStatus::Pending => "pending",
                    crate::session::PlanStatus::InProgress => "in_progress",
                    crate::session::PlanStatus::Done => "done",
                    crate::session::PlanStatus::Failed => "failed",
                }.to_string(),
            }).collect(),
            meta.roadmap.iter().map(|r| RoadmapItemResponse {
                description: r.description.clone(),
                done: r.done,
            }).collect(),
        )
    };
    Json(StatusResponse {
        revision,
        mutations,
        history_entries,
        plan,
        roadmap,
        functions,
        types,
        routes,
        program_summary,
    })
}

#[derive(Deserialize)]
pub struct HistoryRequest {
    pub limit: Option<usize>,
}

#[derive(Serialize)]
pub struct HistoryResponse {
    pub formatted: String,
    pub mutations: Vec<crate::session::MutationEntry>,
    pub history: Vec<crate::session::HistoryEntry>,
}

pub async fn history(
    State(config): State<AppConfig>,
    Json(req): Json<HistoryRequest>,
) -> Json<HistoryResponse> {
    // Tier 3: read meta (brief lock)
    let meta = config.meta.lock().unwrap();
    let limit = req.limit.unwrap_or(20);
    // format_recent_history is on Session, but we can reconstruct from meta directly.
    // For now, build a temporary Session just for the formatting helper.
    // TODO: move format_recent_history to SessionMeta.
    let formatted = {
        let temp = crate::session::Session {
            program: crate::ast::Program::default(),
            runtime: crate::session::RuntimeState::default(),
            meta: meta.clone(),
            sandbox: None,
        };
        temp.format_recent_history(limit)
    };
    Json(HistoryResponse {
        formatted,
        mutations: meta.mutations.clone(),
        history: meta.history.clone(),
    })
}

#[derive(Deserialize)]
pub struct RewindRequest {
    pub revision: usize,
}

#[derive(Serialize)]
pub struct RewindResponse {
    pub revision: usize,
    pub success: bool,
    pub message: String,
}

pub async fn rewind(
    State(config): State<AppConfig>,
    Json(req): Json<RewindRequest>,
) -> Json<RewindResponse> {
    let mut program = config.program.write().await;
    let mut runtime = config.runtime.write().unwrap();
    let mut meta = config.meta.lock().unwrap();
    let mut sandbox = None;
    match crate::session::rewind_to(&mut program, &mut runtime, &mut meta, &mut sandbox, req.revision) {
        Ok(()) => {
            Json(RewindResponse {
                revision: meta.revision,
                success: true,
                message: format!("rewound to revision {}", req.revision),
            })
        }
        Err(e) => Json(RewindResponse {
            revision: meta.revision,
            success: false,
            message: format!("{e}"),
        }),
    }
}

// === Detailed program endpoint ===

#[derive(Serialize)]
pub struct ProgramResponse {
    pub revision: usize,
    pub types: Vec<TypeDetail>,
    pub functions: Vec<FunctionDetail>,
    pub modules: Vec<ModuleDetail>,
}

#[derive(Serialize)]
pub struct ModuleDetail {
    pub name: String,
    pub types: Vec<TypeDetail>,
    pub functions: Vec<FunctionDetail>,
    pub modules: Vec<ModuleDetail>,
}

#[derive(Serialize)]
pub struct TypeDetail {
    pub name: String,
    pub kind: String,
    pub fields: Vec<FieldDetail>,
}

#[derive(Serialize)]
pub struct FieldDetail {
    pub name: String,
    pub ty: String,
}

#[derive(Serialize)]
pub struct FunctionDetail {
    pub name: String,
    pub params: Vec<FieldDetail>,
    pub return_type: String,
    pub effects: Vec<String>,
    pub statements: Vec<StatementDetail>,
    pub compilable: bool,
}

#[derive(Serialize)]
pub struct StatementDetail {
    pub id: String,
    pub kind: String,
    pub summary: String,
}

fn format_type(ty: &crate::ast::Type) -> String {
    match ty {
        crate::ast::Type::Int => "Int".into(),
        crate::ast::Type::Float => "Float".into(),
        crate::ast::Type::Bool => "Bool".into(),
        crate::ast::Type::String => "String".into(),
        crate::ast::Type::Byte => "Byte".into(),
        crate::ast::Type::List(t) => format!("List<{}>", format_type(t)),
        crate::ast::Type::Set(t) => format!("Set<{}>", format_type(t)),
        crate::ast::Type::Map(k, v) => format!("Map<{},{}>", format_type(k), format_type(v)),
        crate::ast::Type::Option(t) => format!("Option<{}>", format_type(t)),
        crate::ast::Type::Result(t) => format!("Result<{}>", format_type(t)),
        crate::ast::Type::Struct(name) => name.clone(),
        crate::ast::Type::TaggedUnion(name) => name.clone(),
    }
}

fn stmt_summary(kind: &crate::ast::StatementKind) -> (String, String) {
    match kind {
        crate::ast::StatementKind::Let { name, ty, .. } => {
            ("let".into(), format!("let {}:{}", name, format_type(ty)))
        }
        crate::ast::StatementKind::Call { binding, call, .. } => {
            let bind = binding.as_ref().map(|b| format!("{}:{}", b.name, format_type(&b.ty))).unwrap_or_default();
            ("call".into(), format!("call {} = {}()", bind, call.callee))
        }
        crate::ast::StatementKind::Check { label, on_fail, .. } => {
            ("check".into(), format!("check {} ~{}", label, on_fail))
        }
        crate::ast::StatementKind::Return { .. } => ("return".into(), "return ...".into()),
        crate::ast::StatementKind::Branch { .. } => ("branch".into(), "if/else".into()),
        crate::ast::StatementKind::Each { binding, .. } => {
            ("each".into(), format!("each {}:{}", binding.name, format_type(&binding.ty)))
        }
        crate::ast::StatementKind::Set { name, .. } => {
            ("set".into(), format!("set {name} = ..."))
        }
        crate::ast::StatementKind::While { .. } => ("while".into(), "while ...".into()),
        crate::ast::StatementKind::Await { name, call, .. } => {
            ("await".into(), format!("await {name} = {}()", call.callee))
        }
        crate::ast::StatementKind::Spawn { call, .. } => {
            ("spawn".into(), format!("spawn {}()", call.callee))
        }
        crate::ast::StatementKind::Match { .. } => ("match".into(), "match ...".into()),
        crate::ast::StatementKind::Yield { .. } => ("yield".into(), "yield ...".into()),
        crate::ast::StatementKind::Source(op) => match op {
            crate::ast::SourceOp::Add { alias, .. } => ("source".into(), format!("source add as {alias}")),
            crate::ast::SourceOp::Remove { alias } => ("source".into(), format!("source remove {alias}")),
            crate::ast::SourceOp::Replace { alias, .. } => ("source".into(), format!("source replace {alias}")),
            crate::ast::SourceOp::List => ("source".into(), "source list".into()),
        },
        crate::ast::StatementKind::Event(op) => match op {
            crate::ast::EventOp::Register { name, .. } => ("event".into(), format!("event register {name}")),
            crate::ast::EventOp::Emit { name, .. } => ("event".into(), format!("event emit {name}")),
        },
    }
}

pub async fn program(State(config): State<AppConfig>) -> Json<ProgramResponse> {
    // Tier 1: read program (RwLock read — non-exclusive)
    let prog = config.program.read().await;
    // Tier 3: read meta for revision (brief)
    let revision = config.meta.lock().unwrap().revision;

    let types = prog.types.iter().map(|td| {
        match td {
            crate::ast::TypeDecl::Struct(s) => TypeDetail {
                name: s.name.clone(),
                kind: "struct".into(),
                fields: s.fields.iter().map(|f| FieldDetail {
                    name: f.name.clone(),
                    ty: format_type(&f.ty),
                }).collect(),
            },
            crate::ast::TypeDecl::TaggedUnion(u) => TypeDetail {
                name: u.name.clone(),
                kind: "union".into(),
                fields: u.variants.iter().map(|v| FieldDetail {
                    name: v.name.clone(),
                    ty: v.payload.iter().map(format_type).collect::<Vec<_>>().join(", "),
                }).collect(),
            },
        }
    }).collect();

    let functions = prog.functions.iter().map(|f| {
        let (stmts, _) = f.body.iter().map(|s| {
            let (kind, summary) = stmt_summary(&s.kind);
            StatementDetail { id: s.id.clone(), kind, summary }
        }).fold((Vec::new(), ()), |(mut v, _), s| { v.push(s); (v, ()) });

        FunctionDetail {
            name: f.name.clone(),
            params: f.params.iter().map(|p| FieldDetail {
                name: p.name.clone(),
                ty: format_type(&p.ty),
            }).collect(),
            return_type: format_type(&f.return_type),
            effects: f.effects.iter().map(|e| format!("{e:?}")).collect(),
            statements: stmts,
            compilable: crate::compiler::is_compilable_function(f),
        }
    }).collect();

    let modules = prog.modules.iter().map(|m| {
        fn build_module_detail(m: &crate::ast::Module) -> ModuleDetail {
            let mod_types = m.types.iter().map(|td| match td {
                crate::ast::TypeDecl::Struct(s) => TypeDetail {
                    name: s.name.clone(), kind: "struct".into(),
                    fields: s.fields.iter().map(|f| FieldDetail { name: f.name.clone(), ty: format_type(&f.ty) }).collect(),
                },
                crate::ast::TypeDecl::TaggedUnion(u) => TypeDetail {
                    name: u.name.clone(), kind: "union".into(),
                    fields: u.variants.iter().map(|v| FieldDetail { name: v.name.clone(), ty: v.payload.iter().map(format_type).collect::<Vec<_>>().join(", ") }).collect(),
                },
            }).collect();
            let mod_funcs = m.functions.iter().map(|f| {
                let stmts = f.body.iter().map(|s| { let (kind, summary) = stmt_summary(&s.kind); StatementDetail { id: s.id.clone(), kind, summary } }).collect();
                FunctionDetail {
                    name: f.name.clone(),
                    params: f.params.iter().map(|p| FieldDetail { name: p.name.clone(), ty: format_type(&p.ty) }).collect(),
                    return_type: format_type(&f.return_type),
                    effects: f.effects.iter().map(|e| format!("{e:?}")).collect(),
                    statements: stmts,
                    compilable: crate::compiler::is_compilable_function(f),
                }
            }).collect();
            let sub_modules = m.modules.iter().map(build_module_detail).collect();
            ModuleDetail { name: m.name.clone(), types: mod_types, functions: mod_funcs, modules: sub_modules }
        }
        build_module_detail(m)
    }).collect();

    Json(ProgramResponse {
        revision,
        types,
        functions,
        modules,
    })
}

// === Ask endpoint (LLM chat) ===

#[derive(Deserialize)]
pub struct AskRequest {
    pub message: String,
}

#[derive(Deserialize)]
pub struct InjectRequest {
    pub message: String,
    /// Conversation context to inject into (default: "main").
    /// Use "main" for the primary conversation, or a specific context
    /// like "telegram:12345" for a targeted conversation.
    #[serde(default = "default_inject_context")]
    pub context: String,
}

fn default_inject_context() -> String { "main".to_string() }

#[derive(Serialize, Deserialize)]
pub struct AskResponse {
    pub reply: String,
    pub code: String,
    pub results: Vec<MutationResult>,
    pub test_results: Vec<TestCaseResult>,
    pub has_errors: bool,
}


pub async fn opencode_task(
    State(config): State<AppConfig>,
    Json(req): Json<OpenCodeRequest>,
) -> Json<OpenCodeResponse> {
    use tokio::io::{AsyncBufReadExt, BufReader};

    let project_dir = &config.project_dir;
    let tx = EventSender::broadcast_only(config.event_broadcast.clone());

    tx.send(serde_json::json!({"type": "opencode_start", "task": req.task})).await;

    // Spawn with piped stdout so we can stream lines and emit SSE events
    // as they happen (instead of buffering until exit).
    let child = tokio::process::Command::new("opencode")
        .arg("run")
        .arg("--format")
        .arg("json")
        .arg(&req.task)
        .current_dir(project_dir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn();

    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[web:opencode:err] {e}");
            tx.send(serde_json::json!({"type": "opencode_error", "exit_code": -1, "message": format!("Failed to run opencode: {e}")})).await;
            return Json(OpenCodeResponse {
                stdout: String::new(),
                stderr: format!("Failed to run opencode: {e}"),
                exit_code: -1,
                success: false,
            });
        }
    };

    let stdout = child.stdout.take().unwrap();
    let stderr_pipe = child.stderr.take().unwrap();
    let mut lines = BufReader::new(stdout).lines();

    // Drain stderr in background so it doesn't block the process.
    let stderr_handle = tokio::spawn(async move {
        use tokio::io::AsyncReadExt;
        let mut buf = String::new();
        let mut reader = BufReader::new(stderr_pipe);
        let _ = reader.read_to_string(&mut buf).await;
        buf
    });

    // Read stdout line-by-line, emitting SSE events in real time.
    let mut text_parts = Vec::new();
    let mut tool_results = Vec::new();
    let mut raw_lines = Vec::new();
    while let Ok(Some(line)) = lines.next_line().await {
        raw_lines.push(line.clone());
        if let Ok(event) = serde_json::from_str::<serde_json::Value>(&line) {
            match event.get("type").and_then(|t| t.as_str()) {
                Some("text") => {
                    if let Some(text) = event.pointer("/part/text").and_then(|t| t.as_str()) {
                        text_parts.push(text.to_string());
                        tx.send(serde_json::json!({"type": "opencode_progress", "kind": "text", "text": text})).await;
                    }
                }
                Some("tool_result") => {
                    if let Some(content) = event.pointer("/part/content") {
                        tool_results.push(content.to_string());
                        tx.send(serde_json::json!({"type": "opencode_progress", "kind": "tool_result", "content": content})).await;
                    }
                }
                _ => {}
            }
        }
    }

    // Wait for process exit and stderr collection.
    let status = child.wait().await;
    let stderr = stderr_handle.await.unwrap_or_default();
    let code = status.map(|s| s.code().unwrap_or(-1)).unwrap_or(-1);

    let raw = raw_lines.join("\n");
    let summary = if !text_parts.is_empty() {
        text_parts.join("\n")
    } else if !tool_results.is_empty() {
        tool_results.join("\n")
    } else {
        raw.chars().take(500).collect()
    };

    eprintln!("[web:opencode] exit={code} text={}chars tools={}", summary.len(), tool_results.len());
    if !summary.is_empty() {
        eprintln!("[web:opencode:text] {}", summary.chars().take(300).collect::<String>());
    }

    if code == 0 {
        tx.send(serde_json::json!({"type": "opencode_done", "exit_code": code})).await;
    } else {
        tx.send(serde_json::json!({"type": "opencode_error", "exit_code": code, "message": stderr})).await;
    }

    Json(OpenCodeResponse {
        stdout: summary,
        stderr,
        exit_code: code,
        success: code == 0,
    })
}

#[derive(Deserialize)]
pub struct OpenCodeRequest {
    pub task: String,
}

#[derive(Serialize)]
pub struct OpenCodeResponse {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub success: bool,
}

/// Build the full router with LLM support.
pub async fn agents(State(config): State<AppConfig>) -> Json<Vec<crate::session::AgentStatus>> {
    // Tier 3: read meta (brief lock)
    let meta = config.meta.lock().unwrap();
    Json(meta.agent_log.clone())
}

// ═══════════════════════════════════════════════════════════════════════
// Multi-session endpoints
// ═══════════════════════════════════════════════════════════════════════

/// GET /api/sessions — list all session IDs (always includes "main").
async fn list_sessions(
    State(config): State<AppConfig>,
) -> Json<serde_json::Value> {
    let sessions = config.sessions.lock().await;
    let mut ids: Vec<String> = vec!["main".to_string()];
    ids.extend(sessions.keys().cloned());
    ids.sort();
    ids.dedup();
    Json(serde_json::json!(ids))
}

#[derive(Deserialize)]
pub struct CreateSessionRequest {
    pub session_id: String,
}

/// POST /api/sessions — create a new named session with an empty Program.
async fn create_session(
    State(config): State<AppConfig>,
    Json(req): Json<CreateSessionRequest>,
) -> (axum::http::StatusCode, Json<serde_json::Value>) {
    let session_id = req.session_id.trim().to_string();
    if session_id.is_empty() {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "session_id must not be empty"})),
        );
    }
    if session_id == "main" {
        return (
            axum::http::StatusCode::CONFLICT,
            Json(serde_json::json!({"error": "cannot create session with reserved id 'main'"})),
        );
    }

    let mut sessions = config.sessions.lock().await;
    if sessions.contains_key(&session_id) {
        return (
            axum::http::StatusCode::CONFLICT,
            Json(serde_json::json!({"error": format!("session '{}' already exists", session_id)})),
        );
    }

    sessions.insert(
        session_id.clone(),
        Arc::new(tokio::sync::Mutex::new(crate::ast::Program::default())),
    );

    (
        axum::http::StatusCode::CREATED,
        Json(serde_json::json!({"session_id": session_id, "status": "created"})),
    )
}

/// DELETE /api/sessions/:id — delete a session.
async fn delete_session(
    State(config): State<AppConfig>,
    axum::extract::Path(session_id): axum::extract::Path<String>,
) -> (axum::http::StatusCode, Json<serde_json::Value>) {
    if session_id == "main" {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "cannot delete the 'main' session"})),
        );
    }

    let mut sessions = config.sessions.lock().await;
    if sessions.remove(&session_id).is_some() {
        (
            axum::http::StatusCode::OK,
            Json(serde_json::json!({"status": "deleted"})),
        )
    } else {
        (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("session '{}' not found", session_id)})),
        )
    }
}

/// POST /api/sessions/:id/eval — evaluate a function in a specific session's Program.
/// Accepts the same JSON body as /api/eval (EvalRequest).
async fn session_eval(
    State(config): State<AppConfig>,
    axum::extract::Path(session_id): axum::extract::Path<String>,
    Json(req): Json<EvalRequest>,
) -> Json<EvalResponse> {
    // For "main", delegate to the main session
    if session_id == "main" {
        return eval_fn(State(config), Json(req)).await;
    }

    // Look up the session's Program
    let sessions = config.sessions.lock().await;
    let program_lock = match sessions.get(&session_id) {
        Some(p) => p.clone(),
        None => {
            return Json(EvalResponse {
                result: format!("session '{}' not found", session_id),
                success: false,
                compiled: None,
            });
        }
    };
    drop(sessions); // release SessionManager lock

    let program = program_lock.lock().await;

    // Handle inline expression evaluation
    if let Some(ref expr_str) = req.expression {
        eprintln!("[session:{session_id}:eval] inline: {expr_str}");
        let expr_str = expr_str.trim();
        if expr_str.is_empty() {
            return Json(EvalResponse {
                result: "empty expression".to_string(),
                success: false,
                compiled: None,
            });
        }
        match parser::parse_expr_pub(0, expr_str) {
            Ok(expr) => {
                // Check if the expression contains IO builtins — run async if so
                if eval::expr_contains_io_builtin(&expr) {
                    if let Some(sender) = &config.io_sender {
                        let program = program.clone();
                        let program_mut = crate::eval::make_shared_program_mut(&program);
                        let sender = sender.clone();
                        let ctx = eval::EvalContext::new(config.runtime.clone(), config.meta.clone(), config.event_broadcast.clone(), &program, program_mut.clone());
                        let eval_result = tokio::task::spawn_blocking(move || {
                            ctx.install();
                            eval::eval_inline_expr_with_io(&program, &expr, sender)
                        }).await;
                        // Note: session eval doesn't sync back to main session — mutations
                        // are scoped to the session's program_lock
                        return match eval_result {
                            Ok(Ok(val)) => Json(EvalResponse {
                                result: format!("{val}"),
                                success: true,
                                compiled: Some(false),
                            }),
                            Ok(Err(e)) => Json(EvalResponse {
                                result: format!("{e}"),
                                success: false,
                                compiled: None,
                            }),
                            Err(e) => Json(EvalResponse {
                                result: format!("task error: {e}"),
                                success: false,
                                compiled: None,
                            }),
                        };
                    }
                    // No IO sender — fall through to sync eval which will error
                }
                match eval::eval_inline_expr(&program, &expr) {
                    Ok(val) => {
                        return Json(EvalResponse {
                            result: format!("{val}"),
                            success: true,
                            compiled: Some(false),
                        });
                    }
                    Err(e) => {
                        return Json(EvalResponse {
                            result: format!("{e}"),
                            success: false,
                            compiled: None,
                        });
                    }
                }
            }
            Err(e) => {
                return Json(EvalResponse {
                    result: format!("parse error: {e}"),
                    success: false,
                    compiled: None,
                });
            }
        }
    }

    eprintln!("[session:{session_id}:eval] {} {}", req.function, req.input);

    // Parse input
    let input_expr = if req.input.trim().is_empty() {
        parser::Expr::StructLiteral(vec![])
    } else {
        match parser::parse_test_input(0, &req.input) {
            Ok(expr) => expr,
            Err(e) => {
                return Json(EvalResponse {
                    result: format!("parse error: {e}"),
                    success: false,
                    compiled: None,
                });
            }
        }
    };

    let ev = parser::EvalMutation {
        function_name: req.function.clone(),
        input: input_expr,
        inline_expr: None,
    };

    // Evaluate (interpreted only for session-scoped programs — no JIT cache)
    match eval::eval_compiled_or_interpreted(&program, &ev.function_name, &ev.input) {
        Ok((result, compiled)) => Json(EvalResponse {
            result,
            success: true,
            compiled: Some(compiled),
        }),
        Err(e) => Json(EvalResponse {
            result: format!("{e}"),
            success: false,
            compiled: None,
        }),
    }
}

/// POST /api/sessions/:id/mutate — apply mutations to a specific session's Program.
async fn session_mutate(
    State(config): State<AppConfig>,
    axum::extract::Path(session_id): axum::extract::Path<String>,
    Json(req): Json<MutateRequest>,
) -> Json<MutateResponse> {
    // For "main", delegate to the main session
    if session_id == "main" {
        return mutate(State(config), Json(req)).await;
    }

    // Look up the session's Program
    let sessions = config.sessions.lock().await;
    let program_lock = match sessions.get(&session_id) {
        Some(p) => p.clone(),
        None => {
            return Json(MutateResponse {
                revision: 0,
                results: vec![MutationResult {
                    message: format!("session '{}' not found", session_id),
                    success: false,
                }],
            });
        }
    };
    drop(sessions);

    let mut program = program_lock.lock().await;

    eprintln!("[session:{session_id}:mutate] {}", req.source.chars().take(100).collect::<String>());

    match parser::parse(&req.source) {
        Ok(ops) => {
            let mut results = Vec::new();
            for op in &ops {
                match op {
                    // Skip non-mutation operations
                    parser::Operation::Test(_)
                    | parser::Operation::Trace(_)
                    | parser::Operation::Eval(_)
                    | parser::Operation::Query(_) => {}
                    _ => match validator::apply_and_validate(&mut program, op) {
                        Ok(msg) => results.push(MutationResult { message: msg, success: true }),
                        Err(e) => results.push(MutationResult { message: format!("{e}"), success: false }),
                    },
                }
            }
            Json(MutateResponse { revision: 0, results })
        }
        Err(e) => Json(MutateResponse {
            revision: 0,
            results: vec![MutationResult {
                message: format!("error: {e}"),
                success: false,
            }],
        }),
    }
}

pub fn router_with_llm(config: AppConfig) -> axum::Router {
    use axum::routing::{get, post};

    let config_routes = axum::Router::new()
        .route("/ui", get(ui_page))
        // Read-only handlers (migrated to tier locks)
        .route("/api/status", get(status))
        .route("/api/program", get(program))
        .route("/api/history", post(history))
        .route("/api/rewind", post(rewind))
        .route("/api/agents", get(agents))
        .route("/api/routes", get(list_routes))
        // Write handlers (still using session shim)
        .route("/api/mutate", post(mutate))
        .route("/api/eval", post(eval_fn))
        .route("/api/test", post(test_fn))
        .route("/api/query", post(query))
        .route("/api/ask", post(ask))
        .route("/api/ask-stream", post(ask_stream))
        .route("/api/inject", post(inject_message))
        .route("/api/drain-queue", post(drain_queue))
        .route("/api/opencode", post(opencode_task))
        .route("/api/tasks", get(tasks))
        .route("/api/log", get(get_log))
        .route("/api/events", get(events_stream))
        .with_state(config.clone());

    let multi_session_routes = axum::Router::new()
        .route("/api/sessions", get(list_sessions))
        .route("/api/sessions", post(create_session))
        .route("/api/sessions/{id}", axum::routing::delete(delete_session))
        .route("/api/sessions/{id}/eval", post(session_eval))
        .route("/api/sessions/{id}/mutate", post(session_mutate))
        .with_state(config.clone());

    // Adapsis-registered HTTP route dispatch (e.g. webhook endpoints)
    let webhook_fallback = axum::Router::new()
        .fallback(adapsis_route_dispatch)
        .with_state(config);

    config_routes.merge(multi_session_routes).merge(webhook_fallback)
}

/// GET /api/routes — list all Adapsis-registered HTTP routes.
async fn list_routes(State(config): State<AppConfig>) -> Json<serde_json::Value> {
    // Tier 2: read runtime (RwLock read — non-exclusive, fast)
    let rt = config.runtime.read().unwrap();
    let routes: Vec<serde_json::Value> = rt
        .http_routes
        .iter()
        .map(|r| {
            serde_json::json!({
                "method": r.method,
                "path": r.path,
                "handler_fn": r.handler_fn,
            })
        })
        .collect();
    Json(serde_json::json!({ "routes": routes }))
}

/// Fallback handler: dispatch incoming requests to Adapsis-registered HTTP routes.
/// Matches method + path against `program.http_routes`, calls the named Adapsis
/// function with the request body as a String parameter, returns the result as
/// 200 text/plain.
async fn adapsis_route_dispatch(
    State(config): State<AppConfig>,
    method: axum::http::Method,
    uri: axum::http::Uri,
    body: axum::body::Bytes,
) -> axum::response::Response {
    config.install_handler_locals();
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    let path = uri.path();
    let method_str = method.as_str();

    // Skip /api/ paths and the root — those are handled by explicit routes
    if path.starts_with("/api/") || path == "/" {
        return (StatusCode::NOT_FOUND, "not found").into_response();
    }

    // Tier 2: look up a matching registered route (brief read lock)
    let handler_fn = {
        let rt = config.runtime.read().unwrap();
        rt.http_routes
            .iter()
            .find(|r| r.method == method_str && r.path == path)
            .map(|r| r.handler_fn.clone())
    };

    let Some(handler_fn) = handler_fn else {
        return (StatusCode::NOT_FOUND, format!("no Adapsis route for {method_str} {path}")).into_response();
    };

    // Tier 1: read program to verify handler exists and clone for eval (brief read lock)
    let program = {
        let prog = config.program.read().await;
        if prog.get_function(&handler_fn).is_none() {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("route handler function `{handler_fn}` not found in program"),
            )
                .into_response();
        }
        prog.clone()
    };

    let body_str = String::from_utf8_lossy(&body).to_string();

    let io_sender_for_blocking = config.io_sender.clone();
    let program_mut = crate::eval::make_shared_program_mut(&program);
    let ctx = eval::EvalContext::new(config.runtime.clone(), config.meta.clone(), config.event_broadcast.clone(), &program, program_mut.clone());

    eprintln!("[webhook] {method_str} {path} -> {handler_fn}({} bytes)", body_str.len());

    let handler_fn_for_log = handler_fn.clone();
    // Evaluate the handler function with the body as a String argument
    let eval_result = tokio::task::spawn_blocking(move || {
        ctx.install();
        let func = program
            .get_function(&handler_fn)
            .ok_or_else(|| anyhow::anyhow!("function `{handler_fn}` not found"))?;
        // Initialize shared runtime vars so +shared defaults are available
        if let Some(rt) = crate::eval::get_shared_runtime() {
            crate::eval::init_missing_shared_runtime_vars(&program, &rt);
        }
        let mut env = eval::Env::new_with_shared_interner(&program.shared_interner);
        env.populate_shared_from_program(&program);
        // Set up coroutine handle so async functions (+await) work in route handlers
        if let Some(sender) = io_sender_for_blocking {
            let handle = crate::coroutine::CoroutineHandle::new(sender);
            env.set("__coroutine_handle", eval::Value::CoroutineHandle(handle));
        }
        let input = eval::Value::string(body_str);
        eval::bind_input_to_params(&program, func, &input, &mut env);
        eval::eval_function_body_named(&program, &handler_fn, &func.body, &mut env)
    })
    .await;

    // Sync mutations back to program tier if any occurred
    if let Some(mutated) = crate::eval::read_back_program_mutations(&program_mut) {
        *config.program.write().await = mutated;
    }

    match eval_result {
        Ok(Ok(val)) => {
            // Extract the raw string for HTTP response (no JSON quoting).
            // Infer content-type from the response body.
            let response_body = match &val {
                eval::Value::String(s) => s.as_ref().clone(),
                other => format!("{other}"),
            };
            let content_type = {
                let lower = response_body.trim_start().to_ascii_lowercase();
                if lower.starts_with("<!doctype") || lower.starts_with("<html") {
                    "text/html; charset=utf-8"
                } else if lower.starts_with('{') || lower.starts_with('[') {
                    "application/json; charset=utf-8"
                } else if lower.starts_with("<?xml") || lower.starts_with("<svg") {
                    "application/xml; charset=utf-8"
                } else {
                    "text/plain; charset=utf-8"
                }
            };
            (
                StatusCode::OK,
                [("content-type", content_type)],
                response_body,
            )
                .into_response()
        }
        Ok(Err(e)) => {
            eprintln!("[webhook] handler error in {handler_fn_for_log}: {e:#}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("handler error: {e:#}"),
            )
                .into_response()
        }
        Err(e) => {
            eprintln!("[webhook] task panic in {handler_fn_for_log}: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("task error: {e}"),
            )
                .into_response()
        }
    }
}

/// GET /api/events — SSE stream of all AI activity (subscribe from web UI).
/// POST /api/inject — inject a message into a conversation context.
///
/// Body: `{"message": "...", "context": "main"}` (context defaults to "main")
///
/// For the "main" context, the message is also pushed to the global message_queue
/// so that an active `ask_stream` session picks it up mid-flight.
async fn inject_message(
    State(config): State<AppConfig>,
    Json(req): Json<InjectRequest>,
) -> Json<serde_json::Value> {
    let context = req.context.clone();
    eprintln!("[inject:{context}] {}...", req.message.chars().take(80).collect::<String>());

    // Push into the conversation's message history
    {
        let mut meta = config.meta.lock().unwrap();
        let conv = meta.conversations.get_or_create(&context);
        conv.push_user(&req.message);
    }

    // For "main" context, also push to the global queue so an active
    // ask_stream session picks it up mid-flight.
    if context == "main" {
        config.message_queue.lock().await.push(req.message.clone());
    }

    Json(serde_json::json!({"status": "injected", "context": context, "message": req.message}))
}

/// POST /api/drain-queue — drain queued messages (called by autonomous loop)
async fn drain_queue(
    State(config): State<AppConfig>,
) -> Json<serde_json::Value> {
    let mut queue = config.message_queue.lock().await;
    let messages: Vec<String> = queue.drain(..).collect();
    Json(serde_json::json!({"messages": messages}))
}

async fn events_stream(
    State(config): State<AppConfig>,
) -> impl axum::response::IntoResponse {
    use axum::response::sse::{Event, KeepAlive};
    let mut rx = config.event_broadcast.subscribe();
    let stream = async_stream::stream! {
        loop {
            match rx.recv().await {
                Ok(event) => {
                    let sse_event = match serde_json::from_str::<serde_json::Value>(&event)
                        .ok()
                        .and_then(|value| value.get("type").and_then(|v| v.as_str()).map(str::to_owned))
                    {
                        Some(kind) => Event::default().event(kind).data(event),
                        None => Event::default().data(event),
                    };
                    yield Ok::<Event, std::convert::Infallible>(sse_event)
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    };
    (
        [
            (axum::http::header::CACHE_CONTROL, "no-cache"),
            (axum::http::header::CONNECTION, "keep-alive"),
        ],
        axum::response::sse::Sse::new(stream).keep_alive(KeepAlive::default()),
    )
}

async fn ui_page() -> Html<&'static str> {
    Html(r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AdapsisOS Dashboard</title>
  <style>
    :root { color-scheme: dark; }
    body { margin: 0; font: 14px/1.5 monospace; background: #020617; color: #e2e8f0; }
    header { padding: 16px 20px; background: linear-gradient(135deg, #111827, #1e293b); border-bottom: 1px solid #334155; }
    h1 { margin: 0; font-size: 20px; }
    .layout { display: grid; grid-template-columns: minmax(320px, 420px) 1fr; min-height: calc(100vh - 73px); }
    .panel { padding: 18px 20px; box-sizing: border-box; }
    .controls { border-right: 1px solid #334155; background: #0f172a; }
    textarea { width: 100%; min-height: 280px; margin: 12px 0; padding: 12px; box-sizing: border-box; border: 1px solid #475569; border-radius: 8px; background: #111827; color: #e2e8f0; }
    .actions { display: flex; gap: 10px; }
    button { padding: 10px 14px; border: 0; border-radius: 8px; background: #2563eb; color: #fff; cursor: pointer; }
    button.secondary { background: #0f766e; }
    #status { margin-top: 12px; color: #93c5fd; white-space: pre-wrap; }
    #log { height: calc(100vh - 109px); overflow: auto; }
    .event { margin: 0 0 10px; padding: 10px 12px; background: #111827; border-left: 4px solid #38bdf8; border-radius: 6px; white-space: pre-wrap; }
    .meta { color: #94a3b8; margin-bottom: 4px; }
    @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } .controls { border-right: 0; border-bottom: 1px solid #334155; } #log { height: auto; min-height: 40vh; } }
  </style>
</head>
<body>
  <header><h1>AdapsisOS Dashboard</h1></header>
  <main class="layout">
    <section class="panel controls">
      <div>Use the editor for inline eval expressions or mutation source.</div>
      <textarea id="source" spellcheck="false" placeholder='1 + 2

or

+fn hello ()->String
  +return "hi"
+end'></textarea>
      <div class="actions">
        <button id="eval">Eval</button>
        <button id="apply" class="secondary">Apply</button>
      </div>
      <div id="status"></div>
    </section>
    <section class="panel"><div id="log"></div></section>
  </main>
  <script>
    const log = document.getElementById('log');
    const status = document.getElementById('status');
    const editor = document.getElementById('source');
    const formatPayload = (payload) => {
      if (payload == null) return '(empty event)';
      if (typeof payload === 'string') return payload || '(empty event)';
      const preferred = ['data', 'detail', 'message', 'result', 'summary'];
      for (const key of preferred) {
        const value = payload[key];
        if (value === undefined || value === null) continue;
        if (typeof value === 'string') return value || '(empty string)';
        return JSON.stringify(value);
      }
      const filtered = Object.fromEntries(Object.entries(payload).filter(([key, value]) => key !== 'type' && key !== 'event' && value !== undefined));
      const text = JSON.stringify(filtered);
      return text && text !== '{}' ? text : JSON.stringify(payload);
    };
    const append = (payload) => {
      const el = document.createElement('div');
      el.className = 'event';
      const meta = document.createElement('div');
      meta.className = 'meta';
      const type = payload && typeof payload === 'object' ? (payload.event || payload.type || 'message') : 'message';
      meta.textContent = `${new Date().toLocaleTimeString()} - ${String(type)}`;
      const body = document.createElement('div');
      body.textContent = formatPayload(payload);
      el.append(meta, body);
      log.appendChild(el);
      log.scrollTop = log.scrollHeight;
    };
    const postJson = async (url, body) => {
      const response = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      status.textContent = await response.text();
    };
    const source = new EventSource('/api/events');
    source.onmessage = (event) => {
      try { append(JSON.parse(event.data)); }
      catch (_) { append({ event: 'raw', data: event.data }); }
    };
    source.onerror = () => append({ event: 'status', data: 'connection issue, retrying...' });
    document.getElementById('eval').onclick = () => postJson('/api/eval', { function: '', input: '', expression: editor.value });
    document.getElementById('apply').onclick = () => postJson('/api/mutate', { source: editor.value });
  </script>
</body>
</html>"#)
}

/// GET /api/log?tail=N — get recent log entries.
async fn get_log(
    State(config): State<AppConfig>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> String {
    let tail = params.get("tail").and_then(|t| t.parse::<usize>().ok()).unwrap_or(50);
    if let Some(log) = &config.log_file {
        use tokio::io::{AsyncReadExt, AsyncSeekExt};
        let mut f = log.lock().await;
        let _ = f.seek(std::io::SeekFrom::Start(0)).await;
        let mut content = String::new();
        let _ = f.read_to_string(&mut content).await;
        // Return last N lines
        let lines: Vec<&str> = content.lines().collect();
        let start = lines.len().saturating_sub(tail);
        lines[start..].join("\n")
    } else {
        "No log file configured.".to_string()
    }
}



/// GET /api/tasks — list all spawned async tasks and their status.
async fn tasks(State(config): State<AppConfig>) -> Json<serde_json::Value> {
    let Some(reg) = &config.task_registry else {
        return Json(serde_json::json!({"tasks": [], "message": "no task registry"}));
    };
    let tasks = reg.lock().unwrap();
    let list: Vec<serde_json::Value> = tasks.values().map(|t| {
        serde_json::json!({
            "id": t.id,
            "function": t.function_name,
            "status": format!("{}", t.status),
            "started_at": t.started_at,
        })
    }).collect();
    Json(serde_json::json!({"tasks": list}))
}


#[cfg(test)]
mod tests;
