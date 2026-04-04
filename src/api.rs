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
const EVAL_TIMEOUT_SECS: u64 = 30;
use crate::validator;

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

fn collect_opencode_tasks(ops: &[crate::parser::Operation]) -> Vec<String> {
    ops.iter()
        .filter_map(|op| match op {
            crate::parser::Operation::OpenCode(task) => Some(task.clone()),
            _ => None,
        })
        .collect()
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
    /// Structured log file for AI activity
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
    crate::eval::set_shared_runtime(Some(config.runtime.clone()));
    crate::eval::set_shared_meta(Some(config.meta.clone()));
    crate::eval::set_shared_event_broadcast(Some(config.event_broadcast.clone()));

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
                        let program_mut_clone = program_mut.clone();
                        let sender = sender.clone();
                        let runtime_for_blocking = config.runtime.clone();
                        let meta_for_blocking = config.meta.clone();
                        let event_broadcast = config.event_broadcast.clone();
                        let eval_result = tokio::task::spawn_blocking(move || {
                            crate::eval::set_shared_runtime(Some(runtime_for_blocking));
                            eval::set_shared_meta(Some(meta_for_blocking));
                            eval::set_shared_event_broadcast(Some(event_broadcast));
                            crate::eval::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                            crate::eval::set_shared_program_mut(Some(program_mut_clone));
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
                        result: format!("function `{}` has {} statements but no passing tests. Write !test blocks first.", ev.function_name, func.body.len()),
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
            let program_mut_clone = program_mut.clone();
            let fn_name = ev.function_name.clone();
            let input = ev.input.clone();
            let sender = sender.clone();
            let runtime_for_blocking = config.runtime.clone();
            let meta_for_blocking = config.meta.clone();
            let event_broadcast = config.event_broadcast.clone();

            let eval_result = tokio::task::spawn_blocking(move || {
                crate::eval::set_shared_runtime(Some(runtime_for_blocking));
                eval::set_shared_meta(Some(meta_for_blocking));
                eval::set_shared_event_broadcast(Some(event_broadcast));
                crate::eval::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                crate::eval::set_shared_program_mut(Some(program_mut_clone));
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

#[derive(Serialize, Deserialize, Clone)]
pub struct TestCaseResult {
    pub message: String,
    pub pass: bool,
}

pub async fn test_fn(
    State(config): State<AppConfig>,
    Json(req): Json<TestRequest>,
) -> Json<TestResponse> {
    crate::eval::set_shared_runtime(Some(config.runtime.clone()));
    crate::eval::set_shared_meta(Some(config.meta.clone()));
    crate::eval::set_shared_event_broadcast(Some(config.event_broadcast.clone()));
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

/// Parse `?inspect task N` queries, returning the task ID if matched.
pub fn parse_inspect_task_query(query: &str) -> Option<i64> {
    let parts: Vec<&str> = query.split_whitespace().collect();
    if parts.len() == 3 && parts[0] == "?inspect" && parts[1] == "task" {
        parts[2].parse::<i64>().ok()
    } else {
        None
    }
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

#[derive(Serialize, Deserialize)]
pub struct AskResponse {
    pub reply: String,
    pub code: String,
    pub results: Vec<MutationResult>,
    pub test_results: Vec<TestCaseResult>,
    pub has_errors: bool,
}

/// Channel wrapper that sends events to the broadcast channel (and optionally an mpsc
/// response channel used by the SSE streaming endpoint).  Both `/api/ask` and
/// `/api/ask-stream` use this so events always appear on `/api/events`.
///
/// The `log` method is the preferred single entry-point for emitting events: it
/// sends to the broadcast channel, the per-request mpsc (if present), writes to
/// the structured log file, and prints a short preview to stderr — all in one
/// call, replacing the previous pattern of separate `tx.send()` + `log_activity()`
/// + `eprintln!()` calls.
struct EventSender {
    tx: Option<tokio::sync::mpsc::Sender<serde_json::Value>>,
    broadcast: tokio::sync::broadcast::Sender<String>,
    log_file: Option<std::sync::Arc<tokio::sync::Mutex<tokio::fs::File>>>,
}

impl EventSender {
    /// Broadcast-only sender (used by plain `/api/ask`).
    fn broadcast_only(broadcast: tokio::sync::broadcast::Sender<String>) -> Self {
        Self { tx: None, broadcast, log_file: None }
    }

    /// Broadcast + per-request mpsc sender (used by `/api/ask-stream`).
    fn with_mpsc(
        tx: tokio::sync::mpsc::Sender<serde_json::Value>,
        broadcast: tokio::sync::broadcast::Sender<String>,
        log_file: Option<std::sync::Arc<tokio::sync::Mutex<tokio::fs::File>>>,
    ) -> Self {
        Self { tx: Some(tx), broadcast, log_file }
    }

    /// Send a raw event value to broadcast + mpsc.
    async fn send(&self, event: serde_json::Value) {
        let encoded = encode_broadcast_event(&event);
        let _ = self.broadcast.send(encoded);
        if let Some(tx) = &self.tx {
            let _ = tx.send(event).await;
        } else {
            // Yield so broadcast receivers get a chance to process the event.
            tokio::task::yield_now().await;
        }
    }

    /// Unified logging: broadcast event + write to log file + stderr preview.
    ///
    /// This replaces the separate `tx.send()` / `log_activity()` / `eprintln!()`
    /// pattern.  `event` is a short tag (e.g. "iter", "code", "feedback") and
    /// `detail` is the full text.  The method also constructs an appropriate
    /// broadcast JSON payload.
    async fn log(&self, event: &str, detail: &str) {
        // 1. Write to structured log file
        write_log_file(&self.log_file, event, detail).await;

        // 2. Stderr: short preview
        let preview: String = detail.chars().take(200).collect();
        eprintln!("[{event}] {preview}");

        // 3. Broadcast (+ mpsc if present) — build a JSON event
        let json = match event {
            "iter" => serde_json::json!({"type": "iteration", "detail": detail}),
            "code" => serde_json::json!({"type": "code", "code": detail}),
            "think" => serde_json::json!({"type": "thinking", "text": detail}),
            "feedback" => serde_json::json!({"type": "feedback", "message": detail}),
            "ai-text" => serde_json::json!({"type": "text", "text": detail}),
            "user" => serde_json::json!({"type": "user", "text": detail}),
            "done" => serde_json::json!({"type": "done", "detail": detail}),
            "done-rejected" => serde_json::json!({"type": "result", "message": detail, "success": false}),
            "llm-error" => serde_json::json!({"type": "error", "message": detail}),
            _ => serde_json::json!({"type": event, "detail": detail}),
        };
        self.send(json).await;
    }
}

/// Write a structured entry to the log file (if configured).
/// Shared by both `log_activity` and `EventSender::log`.
async fn write_log_file(
    log_file: &Option<std::sync::Arc<tokio::sync::Mutex<tokio::fs::File>>>,
    event: &str,
    detail: &str,
) {
    if let Some(f) = log_file {
        use tokio::io::AsyncWriteExt;
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let h = (secs / 3600) % 24;
        let m = (secs / 60) % 60;
        let s = secs % 60;

        let line = match event {
            "iter" => format!("\n============================================================\n[{h:02}:{m:02}:{s:02}] {detail}\n============================================================\n"),
            "code" => format!("[{h:02}:{m:02}:{s:02}] CODE:\n{detail}\n"),
            "think" => format!("[{h:02}:{m:02}:{s:02}] THINK:\n{detail}\n"),
            "feedback" => {
                let has_err = detail.contains("ERROR") || detail.contains("FAIL") || detail.contains("Fix the errors");
                let prefix = if has_err { "FEEDBACK (ERRORS)" } else { "FEEDBACK (ok)" };
                format!("[{h:02}:{m:02}:{s:02}] {prefix}:\n{detail}\n")
            }
            "ai-text" => format!("[{h:02}:{m:02}:{s:02}] AI: {detail}\n"),
            "done" | "done-rejected" => format!("[{h:02}:{m:02}:{s:02}] >>> {event}: {detail}\n"),
            "llm-error" => format!("[{h:02}:{m:02}:{s:02}] !!! LLM ERROR: {detail}\n"),
            "user" => format!("[{h:02}:{m:02}:{s:02}] USER:\n{detail}\n"),
            _ => format!("[{h:02}:{m:02}:{s:02}] [{event}] {detail}\n"),
        };
        let mut f = f.lock().await;
        let _ = f.write_all(line.as_bytes()).await;
        let _ = f.flush().await;
    }
}

/// Legacy standalone logging function — kept during migration so existing code
/// that doesn't yet have an `EventSender` can still log.
/// New code should prefer `EventSender::log()`.
async fn log_activity(
    log_file: &Option<std::sync::Arc<tokio::sync::Mutex<tokio::fs::File>>>,
    event: &str,
    detail: &str,
) {
    write_log_file(log_file, event, detail).await;
    // Stderr: short preview
    let preview: String = detail.chars().take(200).collect();
    eprintln!("[{event}] {preview}");
}

/// Write a training data entry (JSONL) for one iteration.
async fn log_training_data(
    training_log: &Option<std::sync::Arc<tokio::sync::Mutex<tokio::fs::File>>>,
    model: &str,
    context: &str,
    thinking: &str,
    code: &str,
    feedback: &[String],
    has_errors: bool,
    tests_passed: usize,
    tests_failed: usize,
) {
    let Some(f) = training_log else { return };
    use tokio::io::AsyncWriteExt;
    let entry = serde_json::json!({
        "model": model,
        "context": context,
        "thinking": thinking,
        "code": code,
        "outcome": if has_errors { "error" } else { "success" },
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
        "feedback": feedback,
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_secs(),
    });
    let mut line = serde_json::to_string(&entry).unwrap_or_default();
    line.push('\n');
    let mut f = f.lock().await;
    let _ = f.write_all(line.as_bytes()).await;
    let _ = f.flush().await;
}

/// Format the task registry for display.
pub fn format_tasks(registry: &Option<crate::coroutine::TaskRegistry>) -> String {
    let Some(reg) = registry else { return "No task registry (async not available).".to_string() };
    let tasks = reg.lock().unwrap();
    if tasks.is_empty() {
        return "No tasks.".to_string();
    }
    let mut out = String::new();
    let mut sorted: Vec<_> = tasks.values().collect();
    sorted.sort_by_key(|t| t.id);
    for t in sorted {
        out.push_str(&format!("  task {} [{}] — {}\n", t.id, t.function_name, t.status));
    }
    out
}

/// Format a detailed inspection of a single task, combining TaskInfo and TaskSnapshot.
pub fn format_inspect_task(
    task_registry: &Option<crate::coroutine::TaskRegistry>,
    snapshot_registry: &Option<crate::coroutine::TaskSnapshotRegistry>,
    task_id: i64,
) -> String {
    let Some(task_reg) = task_registry else {
        return "No task registry (async not available).".to_string();
    };
    let tasks = task_reg.lock().unwrap();
    let Some(info) = tasks.get(&task_id) else {
        return format!("No task with id {task_id}.");
    };

    let mut out = String::new();
    out.push_str(&format!("Task {}\n", info.id));
    out.push_str(&format!("  function: {}\n", info.function_name));
    out.push_str(&format!("  started:  {}\n", info.started_at));
    out.push_str(&format!("  status:   {}\n", info.status));

    if let Some(snap_reg) = snapshot_registry {
        if let Ok(snaps) = snap_reg.lock() {
            if let Some(snap) = snaps.get(&task_id) {
                if let Some(ref stmt_id) = snap.current_stmt_id {
                    out.push_str(&format!("  stmt:     {}\n", stmt_id));
                }
                out.push_str(&format!("  depth:    {}\n", snap.frame_depth));
                if snap.locals.is_empty() {
                    out.push_str("  locals:   (none)\n");
                } else {
                    out.push_str("  locals:\n");
                    for (name, val) in &snap.locals {
                        out.push_str(&format!("    {} = {}\n", name, val));
                    }
                }
            } else {
                out.push_str("  snapshot: (not yet captured)\n");
            }
        }
    } else {
        out.push_str("  snapshot: (registry not available)\n");
    }

    out
}

/// Build plan context string and whether the AI needs to create a new plan.
fn build_plan_context(plan: &[crate::session::PlanStep]) -> (String, bool) {
    if plan.is_empty() {
        return (String::new(), true);
    }
    let all_done = plan.iter().all(|s| matches!(s.status, crate::session::PlanStatus::Done | crate::session::PlanStatus::Failed));
    let steps = plan.iter().enumerate().map(|(i, s)| {
        let icon = match s.status {
            crate::session::PlanStatus::Pending => "[ ]",
            crate::session::PlanStatus::InProgress => "[~]",
            crate::session::PlanStatus::Done => "[x]",
            crate::session::PlanStatus::Failed => "[!]",
        };
        format!("{} {}: {}", icon, i + 1, s.description)
    }).collect::<Vec<_>>().join("\n");
    (format!("\nCurrent plan:\n{steps}\n"), all_done)
}

/// Format library load errors for inclusion in the AI context.
/// Returns an empty string if there are no load errors.
fn format_library_load_errors(meta: &crate::session::SessionMeta) -> String {
    if let Some(ref lib_state) = meta.library_state {
        if let Some(text) = lib_state.format_load_errors() {
            return format!("\nWARNING — Library module load failures:\n{text}Use `+await result:String = library_reload(\"\")` or `+await result:String = library_reload(\"ModuleName\")` to retry.\n");
        }
    }
    String::new()
}

// ═══════════════════════════════════════════════════════════════════════
// Operation dispatch helpers
// ═══════════════════════════════════════════════════════════════════════

/// Result of processing a single operation or a batch of operations.
struct OperationResult {
    feedback: Vec<String>,
    has_errors: bool,
    tests_passed: usize,
    tests_failed: usize,
    /// Signals the main loop should stop after this iteration.
    accepted_done: bool,
}

impl OperationResult {
    fn new() -> Self {
        Self {
            feedback: Vec::new(),
            has_errors: false,
            tests_passed: 0,
            tests_failed: 0,
            accepted_done: false,
        }
    }

    fn ok(&mut self, msg: impl Into<String>) {
        let entry = format!("OK: {}", msg.into());
        eprintln!("[op] {}", entry.chars().take(200).collect::<String>());
        self.feedback.push(entry);
    }

    /// Log to stderr but do NOT include in LLM feedback.
    /// Use for bookkeeping ops (mock registration, plan progress) where
    /// the information is already conveyed via plan_summary or is noise.
    fn ok_silent(&self, msg: impl Into<String>) {
        let entry = format!("OK: {}", msg.into());
        eprintln!("[op] {}", entry.chars().take(200).collect::<String>());
    }

    fn pass(&mut self, msg: impl Into<String>) {
        self.tests_passed += 1;
        let entry = format!("PASS: {}", msg.into());
        eprintln!("[op] {}", entry.chars().take(200).collect::<String>());
        self.feedback.push(entry);
    }

    fn fail(&mut self, msg: impl Into<String>) {
        self.tests_failed += 1;
        self.has_errors = true;
        let entry = format!("FAIL: {}", msg.into());
        eprintln!("[op] {}", entry.chars().take(200).collect::<String>());
        self.feedback.push(entry);
    }

    fn error(&mut self, msg: impl Into<String>) {
        self.has_errors = true;
        let entry = format!("ERROR: {}", msg.into());
        eprintln!("[op] {}", entry.chars().take(200).collect::<String>());
        self.feedback.push(entry);
    }

    fn info(&mut self, msg: impl Into<String>) {
        let entry = msg.into();
        eprintln!("[op] {}", entry.chars().take(200).collect::<String>());
        self.feedback.push(entry);
    }
}

/// Process a `!plan` operation.
async fn process_plan(
    action: &crate::parser::PlanAction,
    meta: &mut crate::session::SessionMeta,
    tx: &EventSender,
    result: &mut OperationResult,
) {
    match action {
        crate::parser::PlanAction::Set(steps) => {
            meta.plan = steps
                .iter()
                .map(|s| crate::session::PlanStep {
                    description: s.clone(),
                    status: crate::session::PlanStatus::Pending,
                })
                .collect();
            let plan_json: Vec<serde_json::Value> = meta
                .plan
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "description": s.description,
                        "status": format!("{:?}", s.status).to_lowercase()
                    })
                })
                .collect();
            let _ = tx.send(serde_json::json!({"type": "plan", "plan": plan_json})).await;
            result.ok_silent(format!("Plan: {} steps", steps.len()));
            let _ = tx.send(serde_json::json!({"type": "result", "message": format!("Plan: {} steps", steps.len()), "success": true})).await;
        }
        crate::parser::PlanAction::Progress(n) => {
            if let Some(step) = meta.plan.get_mut(n.saturating_sub(1)) {
                step.status = crate::session::PlanStatus::Done;
                let plan_json: Vec<serde_json::Value> = meta
                    .plan
                    .iter()
                    .map(|s| {
                        serde_json::json!({
                            "description": s.description,
                            "status": format!("{:?}", s.status).to_lowercase()
                        })
                    })
                    .collect();
                let _ = tx.send(serde_json::json!({"type": "plan", "plan": plan_json})).await;
                result.ok_silent(format!("Step {n} done"));
                let _ = tx.send(serde_json::json!({"type": "result", "message": format!("Step {n} done"), "success": true})).await;
            }
        }
        _ => {}
    }
}

/// Process a `?query` operation.
async fn process_query(
    query: &str,
    session: &WorkingSet,
    config: &AppConfig,
    tx: &EventSender,
    result: &mut OperationResult,
) {
    let response = if query.trim() == "?inbox" || query.trim().starts_with("?inbox") {
        let msgs = crate::session::peek_messages(&session.meta, "main");
        if msgs.is_empty() {
            "No messages.".to_string()
        } else {
            msgs.iter()
                .map(|m| format!("[{}] from {}: {}", m.timestamp, m.from, m.content))
                .collect::<Vec<_>>()
                .join("\n")
        }
    } else if query.trim() == "?tasks" {
        format_tasks(&config.task_registry)
    } else if let Some(tid) = parse_inspect_task_query(query.trim()) {
        format_inspect_task(&config.task_registry, &config.snapshot_registry, tid)
    } else if query.trim() == "?library" {
        crate::library::query_library(&session.program, session.meta.library_state.as_ref())
    } else {
        let table = crate::typeck::build_symbol_table(&session.program);
        crate::typeck::handle_query(
            &session.program,
            &table,
            query,
            &session.runtime.http_routes,
        )
    };
    result.info(format!("{query}:\n{response}"));
    let _ = tx.send(serde_json::json!({"type": "query", "query": query, "response": response})).await;
}

/// Process `!done` — checks for untested functions and signals completion.
async fn process_done(
    session: &WorkingSet,
    iteration: usize,
    tx: &EventSender,
    result: &mut OperationResult,
) -> bool {
    if session.program.require_modules {
        let untested: Vec<String> = session
            .program
            .modules
            .iter()
            .flat_map(|m| {
                m.functions.iter().filter_map(|f| {
                    let qname = format!("{}.{}", m.name, f.name);
                    if f.body.len() > 2 && !crate::session::is_function_tested(&session.program, &qname) {
                        Some(qname)
                    } else {
                        None
                    }
                })
            })
            .collect();
        if !untested.is_empty() {
            let challenge = format!(
                "Cannot accept !done: {} untested functions: {}. Write !test blocks for them.",
                untested.len(),
                untested.join(", ")
            );
            tx.log("done-rejected", &challenge).await;
            result.error(challenge);
            return false; // not accepted
        }
    }
    tx.log("done", &format!("AI said !done at iteration {}", iteration + 1)).await;
    result.accepted_done = true;
    true // accepted
}

/// Process `!mock` — register an IO mock.
fn process_mock(
    operation: &str,
    patterns: &[String],
    response: &str,
    meta: &mut crate::session::SessionMeta,
    result: &mut OperationResult,
) {
    let pattern_display = patterns
        .iter()
        .map(|p| format!("\"{p}\""))
        .collect::<Vec<_>>()
        .join(" ");
    meta.io_mocks.push(crate::session::IoMock {
        operation: operation.to_string(),
        patterns: patterns.to_vec(),
        response: response.to_string(),
    });
    result.ok_silent(format!("mock: {operation} {pattern_display}"));
}

/// Process `!unmock` — clear all IO mocks.
fn process_unmock(meta: &mut crate::session::SessionMeta, result: &mut OperationResult) {
    let count = meta.io_mocks.len();
    meta.io_mocks.clear();
    result.ok_silent(format!("cleared {count} mocks"));
}

/// Result returned by `execute_code()` — no SSE events, just data.
pub struct CodeExecutionResult {
    pub mutation_results: Vec<MutationResult>,
    pub test_results: Vec<TestCaseResult>,
    pub has_errors: bool,
    /// True when `!opencode` succeeded and we need to `exec` into the new binary.
    pub needs_opencode_restart: bool,
    /// True when `!agent` was encountered — caller should know a background agent was spawned.
    pub agent_spawned: bool,
    /// Names of agents that were spawned.
    pub spawned_agent_names: Vec<String>,
}

impl CodeExecutionResult {
    fn new() -> Self {
        Self {
            mutation_results: Vec::new(),
            test_results: Vec::new(),
            has_errors: false,
            needs_opencode_restart: false,
            agent_spawned: false,
            spawned_agent_names: Vec::new(),
        }
    }

    fn push_ok(&mut self, msg: impl Into<String>) {
        self.mutation_results.push(MutationResult { message: msg.into(), success: true });
    }

    fn push_err(&mut self, msg: impl Into<String>) {
        let msg = msg.into();
        eprintln!("[execute_code:err] {}", msg.chars().take(200).collect::<String>());
        self.has_errors = true;
        self.mutation_results.push(MutationResult { message: msg, success: false });
    }

    fn push_test_pass(&mut self, msg: impl Into<String>) {
        self.test_results.push(TestCaseResult { message: msg.into(), pass: true });
    }

    fn push_test_fail(&mut self, msg: impl Into<String>) {
        self.has_errors = true;
        self.test_results.push(TestCaseResult { message: msg.into(), pass: false });
    }
}

/// Execute a block of Adapsis code against a working snapshot.
///
/// Optional callback info for agent completion notifications.
/// When set, spawned agents will call the LLM with the conversation context
/// after completion and deliver the result via the reply function.
#[derive(Clone)]
pub struct AgentCompletionCallback {
    /// Conversation context name (e.g. "telegram:1815217")
    pub context: String,
    /// Function to call with the result (e.g. "TelegramBot.send_reply")
    pub reply_fn: String,
    /// Argument to pass to reply_fn (e.g. "1815217")  
    pub reply_arg: String,
    /// LLM config for generating the summary
    pub llm_url: String,
    pub llm_model: String,
    pub llm_key: Option<String>,
}

/// All mutations, tests, evals, queries, watches, agents, and `!opencode` are
/// handled here.  SSE events are **not** sent — the caller is responsible for
/// inspecting the returned `CodeExecutionResult` and emitting any events it needs.
pub async fn execute_code(
    code: &str,
    config: &AppConfig,
    session: &mut WorkingSet,
    agent_callback: Option<AgentCompletionCallback>,
) -> CodeExecutionResult {
    let mut result = CodeExecutionResult::new();

    match crate::parser::parse(code) {
        Ok(ops) => {
            let opencode_tasks = collect_opencode_tasks(&ops);
            let mut needs_opencode_restart = false;

            // Remove duplicate function/type definitions before applying mutations
            let mut fns_removed = false;
            for op in &ops {
                match op {
                    crate::parser::Operation::Function(f) => {
                        session.program.functions.retain(|existing| existing.name != f.name);
                        fns_removed = true;
                    }
                    crate::parser::Operation::Type(t) => {
                        let name = t.name.clone();
                        session.program.types.retain(|existing: &crate::ast::TypeDecl| existing.name() != name);
                    }
                    _ => {}
                }
            }
            if fns_removed {
                session.program.rebuild_function_index();
            }

            // Handle !undo
            let has_undo = ops.iter().any(|op| matches!(op, crate::parser::Operation::Undo));
            if has_undo {
                if session.meta.revision > 0 {
                    let prev = session.meta.revision - 1;
                    match crate::session::rewind_to(&mut session.program, &mut session.runtime, &mut session.meta, &mut session.sandbox, prev) {
                        Ok(()) => result.push_ok(format!("Undone to rev {prev}")),
                        Err(e) => result.push_err(format!("Undo: {e}")),
                    }
                }
            }

            // Handle !plan
            let has_plan_ops = ops.iter().any(|op| matches!(op, crate::parser::Operation::Plan(_)));
            for op in &ops {
                if let crate::parser::Operation::Plan(action) = op {
                    match action {
                        crate::parser::PlanAction::Set(steps) => {
                            session.meta.plan = steps.iter().map(|s| crate::session::PlanStep {
                                description: s.clone(),
                                status: crate::session::PlanStatus::Pending,
                            }).collect();
                            result.push_ok(format!("Plan set: {} steps", steps.len()));
                        }
                        crate::parser::PlanAction::Progress(n) => {
                            let idx = n.saturating_sub(1);
                            if let Some(step) = session.meta.plan.get_mut(idx) {
                                step.status = crate::session::PlanStatus::Done;
                                result.push_ok(format!("Step {n} done: {}", step.description));
                            }
                        }
                        crate::parser::PlanAction::Fail(n) => {
                            let idx = n.saturating_sub(1);
                            if let Some(step) = session.meta.plan.get_mut(idx) {
                                step.status = crate::session::PlanStatus::Failed;
                                result.push_ok(format!("Step {n} failed: {}", step.description));
                            }
                        }
                        crate::parser::PlanAction::Show => {
                            let plan_str = session.meta.plan.iter().enumerate().map(|(i, s)| {
                                let icon = match s.status {
                                    crate::session::PlanStatus::Pending => "[ ]",
                                    crate::session::PlanStatus::InProgress => "[~]",
                                    crate::session::PlanStatus::Done => "[x]",
                                    crate::session::PlanStatus::Failed => "[!]",
                                };
                                format!("  {} {}: {}", icon, i + 1, s.description)
                            }).collect::<Vec<_>>().join("\n");
                            result.push_ok(if plan_str.is_empty() { "No plan set".to_string() } else { format!("Plan:\n{plan_str}") });
                        }
                    }
                }
            }
            if has_plan_ops {
                config.write_back_working_set(session).await;
            }

            // Apply mutations
            let has_mutations = ops.iter().any(|op| !matches!(op,
                crate::parser::Operation::Test(_) | crate::parser::Operation::Trace(_)
                | crate::parser::Operation::Eval(_) | crate::parser::Operation::Query(_)
                | crate::parser::Operation::Undo | crate::parser::Operation::Plan(_)
                | crate::parser::Operation::Watch { .. }
                | crate::parser::Operation::Agent { .. }
                | crate::parser::Operation::Message { .. }
                | crate::parser::Operation::Done
                | crate::parser::Operation::OpenCode(_)));

            if has_mutations {
                match crate::session::apply_to_tiers(&mut session.program, &mut session.runtime, &mut session.meta, &mut session.sandbox, code) {
                    Ok(res) => {
                        config.write_back_working_set(session).await;
                        for (msg, ok) in res {
                            eprintln!("[execute_code:{}] {msg}", if ok { "ok" } else { "err" });
                            if !ok { result.has_errors = true; }
                            result.mutation_results.push(MutationResult { message: msg, success: ok });
                        }
                    }
                    Err(e) => {
                        result.push_err(format!("{e}"));
                    }
                }
            }

            // Handle tests, evals, queries, watches, agents, messages
            for op in &ops {
                match op {
                    crate::parser::Operation::Test(test) => {
                        let mut all_passed = true;
                        let needs_async = session.program.get_function(&test.function_name)
                            .is_some_and(|f| f.effects.iter().any(|e|
                                matches!(e, crate::ast::Effect::Io | crate::ast::Effect::Async)));

                        for case in &test.cases {
                            let case_result = if needs_async {
                                if let Some(sender) = &config.io_sender {
                                    let program = session.program.clone();
                                    let fn_name = test.function_name.clone();
                                    let case = case.clone();
                                    let mocks = session.meta.io_mocks.clone();
                                    let routes = session.runtime.http_routes.clone();
                                    let sender = sender.clone();
                                    crate::eval::eval_test_case_async(
                                        &program, &fn_name, &case, &mocks, sender, &routes,
                                    ).await
                                } else {
                                    crate::eval::eval_test_case_with_mocks(
                                        &session.program, &test.function_name, case, &session.meta.io_mocks, &session.runtime.http_routes,
                                    )
                                }
                            } else {
                                crate::eval::eval_test_case_with_mocks(
                                    &session.program, &test.function_name, case, &session.meta.io_mocks, &session.runtime.http_routes,
                                )
                            };
                            match case_result {
                                Ok(msg) => {
                                    eprintln!("[execute_code:pass] {msg}");
                                    result.push_test_pass(msg);
                                }
                                Err(e) => {
                                    all_passed = false;
                                    eprintln!("[execute_code:fail] {e}");
                                    result.push_test_fail(format!("{e}"));
                                }
                            }
                        }
                        if all_passed && !test.cases.is_empty() {
                            crate::session::store_test(&mut session.program, &test.function_name, &test.cases);
                        }
                    }
                    crate::parser::Operation::Eval(ev) => {
                        // Inline expression: evaluate directly
                        if let Some(ref expr) = ev.inline_expr {
                            if crate::eval::expr_contains_io_builtin(expr) {
                                if let Some(sender) = &config.io_sender {
                                    let program = session.program.clone();
                                    let program_mut = crate::eval::make_shared_program_mut(&program);
                                    let program_mut_clone = program_mut.clone();
                                    let expr = expr.clone();
                                    let sender = sender.clone();
                                    let runtime_for_blocking = config.runtime.clone();
                                    let meta_for_blocking = config.meta.clone();
                                    let event_broadcast = config.event_broadcast.clone();
                                    let eval_result = tokio::time::timeout(
                                        std::time::Duration::from_secs(EVAL_TIMEOUT_SECS),
                                        tokio::task::spawn_blocking(move || {
                                            crate::eval::set_shared_runtime(Some(runtime_for_blocking));
                                            eval::set_shared_meta(Some(meta_for_blocking));
                                            eval::set_shared_event_broadcast(Some(event_broadcast));
                                            crate::eval::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                                            crate::eval::set_shared_program_mut(Some(program_mut_clone));
                                            crate::eval::eval_inline_expr_with_io(&program, &expr, sender)
                                        })
                                    ).await;
                                    let (msg, success) = match &eval_result {
                                        Ok(Ok(Ok(val))) => (format!("= {val}"), true),
                                        Ok(Ok(Err(e))) => { (format!("eval error: {e}"), false) }
                                        Ok(Err(e)) => { (format!("eval task error: {e}"), false) }
                                        Err(_) => { (format!("eval timed out after {EVAL_TIMEOUT_SECS}s"), false) }
                                    };
                                    eprintln!("[execute_code:eval] {msg}");
                                    if success {
                                        result.push_ok(msg);
                                    } else {
                                        result.push_err(msg);
                                    }
                                    if let Some(mutated) = crate::eval::read_back_program_mutations(&program_mut) {
                                        session.program = mutated.clone();
                                        *config.program.write().await = mutated;
                                    }
                                    config.sync_async_side_effects_into(session);
                                    continue;
                                }
                            }
                            match crate::eval::eval_inline_expr(&session.program, expr) {
                                Ok(val) => {
                                    let msg = format!("= {val}");
                                    eprintln!("[execute_code:eval] {msg}");
                                    result.push_ok(msg);
                                }
                                Err(e) => {
                                    let msg = format!("eval error: {e}");
                                    eprintln!("[execute_code:eval:err] {msg}");
                                    result.push_err(msg);
                                }
                            }
                            continue;
                        }

                        // Block eval of untested functions in AdapsisOS mode
                        if session.program.require_modules {
                            if let Some(func) = session.program.get_function(&ev.function_name) {
                                if func.body.len() > 2 && !crate::session::is_function_tested(&session.program, &ev.function_name) {
                                    result.push_err(format!(
                                        "function `{}` has {} statements but no passing tests. Write !test blocks first.",
                                        ev.function_name, func.body.len()
                                    ));
                                    continue;
                                }
                            }
                        }

                        let needs_async = session.program.get_function(&ev.function_name)
                            .is_some_and(|f| f.effects.iter().any(|e|
                                matches!(e, crate::ast::Effect::Io | crate::ast::Effect::Async)));

                        if needs_async {
                            if let Some(sender) = &config.io_sender {
                                let program = session.program.clone();
                                let program_mut = crate::eval::make_shared_program_mut(&program);
                                let program_mut_clone = program_mut.clone();
                                let fn_name = ev.function_name.clone();
                                let input = ev.input.clone();
                                let sender = sender.clone();
                                let runtime_for_blocking = config.runtime.clone();
                                let meta_for_blocking = config.meta.clone();
                                let event_broadcast = config.event_broadcast.clone();
                                let eval_fn_name = ev.function_name.clone();
                                let eval_result = tokio::time::timeout(
                                    std::time::Duration::from_secs(EVAL_TIMEOUT_SECS),
                                    tokio::task::spawn_blocking(move || {
                                        crate::eval::set_shared_runtime(Some(runtime_for_blocking));
                                        eval::set_shared_meta(Some(meta_for_blocking));
                                        eval::set_shared_event_broadcast(Some(event_broadcast));
                                        crate::eval::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                                        crate::eval::set_shared_program_mut(Some(program_mut_clone));
                                        let func = program.get_function(&fn_name)
                                            .ok_or_else(|| anyhow::anyhow!("function not found"))?;
                                        let handle = crate::coroutine::CoroutineHandle::new(sender);
                                        let mut env = crate::eval::Env::new_with_shared_interner(&program.shared_interner);
                                        env.populate_shared_from_program(&program);
                                        env.set("__coroutine_handle", crate::eval::Value::CoroutineHandle(handle));
                                        let input_val = crate::eval::eval_parser_expr_with_program(&input, &program)?;
                                        crate::eval::bind_input_to_params(&program, func, &input_val, &mut env);
                                        crate::eval::eval_function_body_pub(&program, &func.body, &mut env)
                                    })
                                ).await;
                                let (msg, success) = match &eval_result {
                                    Ok(Ok(Ok(val))) => (format!("eval {}() = {val}", eval_fn_name), true),
                                    Ok(Ok(Err(e))) => { (format!("eval error: {e}"), false) }
                                    Ok(Err(e)) => { (format!("eval task error: {e}"), false) }
                                    Err(_) => { (format!("eval {}() timed out after {EVAL_TIMEOUT_SECS}s", eval_fn_name), false) }
                                };
                                eprintln!("[execute_code:eval] {msg}");
                                if success {
                                    result.push_ok(msg);
                                } else {
                                    result.push_err(msg);
                                }
                                if let Some(mutated) = crate::eval::read_back_program_mutations(&program_mut) {
                                    session.program = mutated.clone();
                                    *config.program.write().await = mutated;
                                }
                                config.sync_async_side_effects_into(session);
                            }
                        } else {
                            match crate::eval::eval_compiled_or_interpreted_cached(&session.program, &ev.function_name, &ev.input, Some(&config.jit_cache), session.meta.revision) {
                                Ok((val, compiled)) => {
                                    let tag = if compiled { " [compiled]" } else { "" };
                                    let msg = format!("eval {}() = {val}{tag}", ev.function_name);
                                    eprintln!("[execute_code:eval] {msg}");
                                    result.push_ok(msg);
                                }
                                Err(e) => {
                                    let msg = format!("eval error: {e}");
                                    eprintln!("[execute_code:eval:err] {msg}");
                                    result.push_err(msg);
                                }
                            }
                        }
                    }
                    crate::parser::Operation::Query(query) => {
                        let response = if query.trim() == "?inbox" || query.trim().starts_with("?inbox") {
                            let msgs = crate::session::peek_messages(&session.meta, "main");
                            if msgs.is_empty() {
                                "No messages.".to_string()
                            } else {
                                msgs.iter().map(|m| format!("[{}] from {}: {}", m.timestamp, m.from, m.content)).collect::<Vec<_>>().join("\n")
                            }
                        } else if query.trim() == "?tasks" {
                            format_tasks(&config.task_registry)
                        } else if let Some(tid) = parse_inspect_task_query(query.trim()) {
                            format_inspect_task(&config.task_registry, &config.snapshot_registry, tid)
                        } else if query.trim() == "?library" {
                            crate::library::query_library(&session.program, session.meta.library_state.as_ref())
                        } else {
                            let table = crate::typeck::build_symbol_table(&session.program);
                            crate::typeck::handle_query(&session.program, &table, query, &session.runtime.http_routes)
                        };
                        result.push_ok(response);
                    }
                    crate::parser::Operation::Watch { function_name, args, interval_ms } => {
                        eprintln!("[execute_code:watch] {function_name}({args}) every {interval_ms}ms");
                        let fn_name = function_name.clone();
                        let fn_args = args.clone();
                        let interval = *interval_ms;
                        let watch_program = config.program.clone();
                        let watch_meta = config.meta.clone();
                        let trigger = config.self_trigger.clone();
                        let watch_jit_cache = config.jit_cache.clone();

                        tokio::spawn(async move {
                            let mut last_result = String::new();
                            loop {
                                tokio::time::sleep(std::time::Duration::from_millis(interval)).await;
                                let result = {
                                    let program = watch_program.read().await;
                                    let meta = watch_meta.lock().unwrap();
                                    let input_expr = if fn_args.trim().is_empty() {
                                        crate::parser::Expr::StructLiteral(vec![])
                                    } else {
                                        match crate::parser::parse_test_input(0, &fn_args) {
                                            Ok(expr) => expr,
                                            Err(_) => break,
                                        }
                                    };
                                    match crate::eval::eval_compiled_or_interpreted_cached(
                                        &program, &fn_name, &input_expr,
                                        Some(&watch_jit_cache), meta.revision,
                                    ) {
                                        Ok((r, _)) => r,
                                        Err(e) => format!("error: {e}"),
                                    }
                                };
                                if result != last_result && !last_result.is_empty() {
                                    eprintln!("[execute_code:watch:trigger] {fn_name} changed: {last_result} → {result}");
                                    let msg = format!("Watcher '{fn_name}' triggered: result changed from '{last_result}' to '{result}'");
                                    let _ = trigger.send(msg).await;
                                }
                                last_result = result;
                            }
                        });
                        result.push_ok(format!("Watching {function_name}({args}) every {interval_ms}ms"));
                    }
                    crate::parser::Operation::Agent { name, scope, task } => {
                        eprintln!("[execute_code:agent] spawning '{name}' scope={scope} task={}", task.chars().take(80).collect::<String>());

                        let agent_scope = crate::session::AgentScope::parse(scope);
                        let branch = crate::session::AgentBranch::fork_from_parts(name, agent_scope, task, &session.program, &session.runtime, &session.meta);
                        let program_summary = crate::validator::program_summary_compact(&session.program);
                        let agent_task = task.clone();
                        let agent_name = name.clone();
                        let llm_url = config.llm_url.clone();
                        let llm_model = config.llm_model.clone();
                        let llm_key = config.llm_api_key.clone();
                        let agent_program = config.program.clone();
                        let agent_meta = config.meta.clone();
                        let agent_runtime = config.runtime.clone();
                        let agent_callback = agent_callback.clone();
                        let agent_io_sender = config.io_sender.clone();

                        tokio::spawn(async move {
                            eprintln!("[agent:{agent_name}] starting");
                            let agent_llm = crate::llm::LlmClient::new_with_model_and_key(&llm_url, &llm_model, llm_key);

                            let scope_desc = match &branch.scope {
                                crate::session::AgentScope::ReadOnly =>
                                    "SCOPE: read-only. You CAN: write !test blocks, use !eval, use ?queries. You CANNOT: define new functions or types, modify existing code.".to_string(),
                                crate::session::AgentScope::NewOnly =>
                                    "SCOPE: new-only. You CAN: define NEW functions and types, write !test blocks, use !eval. You CANNOT: modify or replace existing functions.".to_string(),
                                crate::session::AgentScope::Module(m) =>
                                    format!("SCOPE: module {m}. You CAN: modify anything in module {m}, add new functions to it. You CANNOT: modify code outside module {m}."),
                                crate::session::AgentScope::Full =>
                                    "SCOPE: full. You can modify anything.".to_string(),
                            };
                            let agent_system = format!(
                                "{}\n\n{}\n\nYou are agent '{agent_name}'.\n{scope_desc}\n\nYour task:\n{agent_task}\n\nWork step by step. Always include a <code> block with Adapsis code. When done, respond with !done in a <code> block.",
                                crate::prompt::system_prompt(),
                                crate::builtins::format_for_prompt()
                            );

                            let mut agent_messages = vec![
                                crate::llm::ChatMessage::system(agent_system),
                                crate::llm::ChatMessage::user(format!("Program state:\n{program_summary}\n\nTask: {agent_task}")),
                            ];

                            let mut branch = branch;
                            for agent_iter in 0..10 {
                                {
                                    let mut meta = agent_meta.lock().unwrap();
                                    let inbox = meta.agent_mailbox.remove(&agent_name).unwrap_or_default();
                                    if !inbox.is_empty() {
                                        let inbox_text = inbox.iter()
                                            .map(|m| format!("[from {}] {}", m.from, m.content))
                                            .collect::<Vec<_>>().join("\n");
                                        eprintln!("[agent:{agent_name}] received {} messages", inbox.len());
                                        agent_messages.push(crate::llm::ChatMessage::user(
                                            format!("Messages received:\n{inbox_text}\n\nIncorporate this information and continue.")
                                        ));
                                    }
                                }

                                let output = match agent_llm.generate(agent_messages.clone()).await {
                                    Ok(o) => o,
                                    Err(e) => { eprintln!("[agent:{agent_name}] LLM error: {e}"); break; }
                                };

                                agent_messages.push(crate::llm::ChatMessage::assistant(&output.text));

                                let code = output.code.clone();
                                if code.trim() == "!done" || code.is_empty() {
                                    eprintln!("[agent:{agent_name}] done at iter {agent_iter}");
                                    break;
                                }

                                if let Ok(ops) = crate::parser::parse(&code) {
                                    for op in &ops {
                                        if let crate::parser::Operation::Message { to, content } = op {
                                            eprintln!("[agent:{agent_name}] !msg → {to}: {content}");
                                            let mut meta = agent_meta.lock().unwrap();
                                            let msg = crate::session::AgentMessage {
                                                from: agent_name.clone(),
                                                to: to.clone(),
                                                content: content.clone(),
                                                timestamp: crate::session::now(),
                                            };
                                            meta.agent_mailbox.entry(to.clone()).or_default().push(msg);
                                        }
                                    }
                                }

                                match branch.apply(&code) {
                                    Ok(results) => {
                                        let mut has_err = false;
                                        for (msg, ok) in &results {
                                            eprintln!("[agent:{agent_name}] {}: {msg}", if *ok {"ok"} else {"err"});
                                            if !*ok { has_err = true; }
                                        }
                                        let feedback = results.iter()
                                            .map(|(msg, ok)| format!("{}: {msg}", if *ok {"OK"} else {"ERROR"}))
                                            .collect::<Vec<_>>().join("\n");
                                        if has_err {
                                            agent_messages.push(crate::llm::ChatMessage::user(format!("Errors:\n{feedback}\nFix and continue.")));
                                        } else {
                                            agent_messages.push(crate::llm::ChatMessage::user(format!("Results:\n{feedback}\nContinue or !done.")));
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("[agent:{agent_name}] apply error: {e}");
                                        agent_messages.push(crate::llm::ChatMessage::user(format!("Error: {e}\nFix and continue.")));
                                    }
                                }
                            }

                            // Merge branch back
                            let mut program = agent_program.read().await.clone();
                            let mut runtime = agent_runtime.read().unwrap().clone();
                            let mut meta = agent_meta.lock().unwrap().clone();
                            let mut sandbox = None;
                            let conflicts = branch.merge_into_parts(&mut program, &mut runtime, &mut meta, &mut sandbox);
                            if conflicts.is_empty() {
                                eprintln!("[agent:{agent_name}] merged successfully");
                                meta.chat_messages.push(crate::session::ChatMessage {
                                    role: "system".to_string(),
                                    content: format!("Agent '{agent_name}' completed and merged successfully."),
                                });
                                if let Some(s) = meta.agent_log.iter_mut().rev().find(|s| s.name == agent_name && s.status == "running") {
                                    s.status = "merged".to_string();
                                    s.message = "completed and merged".to_string();
                                }
                            } else {
                                eprintln!("[agent:{agent_name}] merge conflicts: {:?}", conflicts);
                                meta.chat_messages.push(crate::session::ChatMessage {
                                    role: "system".to_string(),
                                    content: format!("Agent '{agent_name}' finished but had merge conflicts:\n{}", conflicts.join("\n")),
                                });
                                if let Some(s) = meta.agent_log.iter_mut().rev().find(|s| s.name == agent_name && s.status == "running") {
                                    s.status = "conflict".to_string();
                                    s.message = conflicts.join("; ");
                                }
                            }
                            *agent_program.write().await = program;
                            *agent_runtime.write().unwrap() = runtime;
                            *agent_meta.lock().unwrap() = meta;

                            // Notify conversation context if callback configured
                            if let Some(cb) = agent_callback {
                                let agent_result = if conflicts.is_empty() {
                                    format!("Agent '{agent_name}' completed and merged successfully.")
                                } else {
                                    format!("Agent '{agent_name}' had merge conflicts: {}", conflicts.join(", "))
                                };
                                eprintln!("[agent:{agent_name}] notifying context '{}'", cb.context);

                                // Append result to conversation as user message to prompt summary
                                {
                                    let mut meta_guard = agent_meta.lock().unwrap();
                                    if let Some(conv) = meta_guard.conversations.get_mut(&cb.context) {
                                        conv.push_user(format!("[System: {}] Summarize the result briefly for the user.", agent_result));
                                    }
                                }

                                // Call LLM for a user-facing summary
                                let summary_messages = {
                                    let meta_guard = agent_meta.lock().unwrap();
                                    if let Some(conv) = meta_guard.conversations.get(&cb.context) {
                                        conv.messages.iter().map(|m| match m.role.as_str() {
                                            "system" => crate::llm::ChatMessage::system(m.content.clone()),
                                            "assistant" => crate::llm::ChatMessage::assistant(&m.content),
                                            _ => crate::llm::ChatMessage::user(m.content.clone()),
                                        }).collect::<Vec<_>>()
                                    } else {
                                        vec![]
                                    }
                                };

                                if !summary_messages.is_empty() {
                                    let summary_llm = crate::llm::LlmClient::new_with_model_and_key(
                                        &cb.llm_url, &cb.llm_model, cb.llm_key,
                                    );
                                    eprintln!("[agent:{agent_name}] calling LLM for completion summary ({} messages)", summary_messages.len());
                                    match summary_llm.generate(summary_messages).await {
                                        Ok(output) => {
                                            let mut reply = output.text.clone();
                                            while let Some(s) = reply.find("<think>") {
                                                if let Some(e) = reply[s..].find("</think>") {
                                                    reply.replace_range(s..s + e + 8, "");
                                                } else { break; }
                                            }
                                            while let Some(s) = reply.find("<code>") {
                                                if let Some(e) = reply[s..].find("</code>") {
                                                    reply.replace_range(s..s + e + 7, "");
                                                } else { break; }
                                            }
                                            let reply = reply.trim().to_string();

                                            if !reply.is_empty() {
                                                // Store in conversation
                                                {
                                                    let mut meta_guard = agent_meta.lock().unwrap();
                                                    if let Some(conv) = meta_guard.conversations.get_mut(&cb.context) {
                                                        conv.push_assistant(&reply);
                                                    }
                                                }
                                                // Deliver via callback
                                                eprintln!("[agent:{agent_name}] delivering reply via {}({})", cb.reply_fn, cb.reply_arg);
                                                if let Some(sender) = agent_io_sender {
                                                    let (tx, _rx) = tokio::sync::oneshot::channel();
                                                    let _ = sender.send(crate::coroutine::IoRequest::Spawn {
                                                        function_name: cb.reply_fn,
                                                        args: vec![
                                                            crate::eval::Value::string(cb.reply_arg),
                                                            crate::eval::Value::string(reply),
                                                        ],
                                                        reply: tx,
                                                    }).await;
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            eprintln!("[agent:{agent_name}] summary LLM call failed: {e}");
                                        }
                                    }
                                }
                            }
                        });

                        result.push_ok(format!("Agent '{name}' spawned (background)"));
                        result.agent_spawned = true;
                        result.spawned_agent_names.push(name.clone());

                        session.meta.agent_log.push(crate::session::AgentStatus {
                            name: name.clone(),
                            task: task.chars().take(100).collect(),
                            scope: scope.clone(),
                            status: "running".to_string(),
                            message: String::new(),
                        });
                    }
                    crate::parser::Operation::Message { to, content } => {
                        eprintln!("[execute_code:msg] → {to}: {content}");
                        crate::session::send_agent_message(&mut session.meta, "main", to, content);
                        config.write_back_working_set(session).await;
                        result.push_ok(format!("Message sent to '{to}'"));
                    }
                    crate::parser::Operation::OpenCode(_) => {}
                    // Top-level statements: execute immediately
                    op @ (crate::parser::Operation::Call(_)
                    | crate::parser::Operation::Let(_)
                    | crate::parser::Operation::Set(_)
                    | crate::parser::Operation::Await(_)
                    | crate::parser::Operation::Spawn(_)
                    | crate::parser::Operation::If(_)
                    | crate::parser::Operation::While(_)
                    | crate::parser::Operation::Each(_)
                    | crate::parser::Operation::Match(_)
                    | crate::parser::Operation::Check(_)
                    | crate::parser::Operation::Branch(_)
                    | crate::parser::Operation::Return(_)) => {
                        match crate::validator::convert_statement_op(op) {
                            Ok(stmt) => {
                                let mut env = crate::eval::Env::new_with_shared_interner(&session.program.shared_interner);
                                env.populate_shared_from_program(&session.program);
                                if let Some(sender) = &config.io_sender {
                                    env.set("__coroutine_handle", crate::eval::Value::CoroutineHandle(
                                        crate::coroutine::CoroutineHandle::new(sender.clone())
                                    ));
                                }
                                match crate::eval::eval_function_body_pub(&session.program, &[stmt], &mut env) {
                                    Ok(val) => {
                                        let msg = format!("executed: {val}");
                                        eprintln!("[execute_code:exec] {msg}");
                                        result.push_ok(msg);
                                    }
                                    Err(e) => {
                                        let msg = format!("exec error: {e}");
                                        eprintln!("[execute_code:exec:err] {msg}");
                                        result.push_err(msg);
                                    }
                                }
                                config.sync_async_side_effects_into(session);
                            }
                            Err(e) => {
                                result.push_err(format!("statement error: {e}"));
                            }
                        }
                    }
                    _ => {}
                }
            }

            // Handle !opencode tasks
            for task in opencode_tasks {
                eprintln!("[execute_code:opencode] {task}");
                let oc_result = tokio::time::timeout(
                    std::time::Duration::from_secs(3600),
                    tokio::process::Command::new("opencode")
                        .arg("run").arg("--format").arg("json")
                        .arg("--attach").arg("http://localhost:4096")
                        .arg("--dir").arg(&config.project_dir)
                        .arg(task)
                        .current_dir(&config.project_dir)
                        .output()
                ).await;
                match oc_result {
                    Ok(Ok(output)) if output.status.success() => {
                        eprintln!("[execute_code:opencode:done] rebuilding...");
                        let build = tokio::process::Command::new("cargo")
                            .arg("build").arg("--release").current_dir(&config.project_dir).output().await;
                        match build {
                            Ok(b) if b.status.success() => {
                                result.push_ok("OpenCode + rebuild successful. Restart to apply.".to_string());
                                needs_opencode_restart = true;
                            }
                            _ => {
                                result.push_err("OpenCode done but build failed".to_string());
                            }
                        }
                    }
                    _ => {
                        result.push_err("OpenCode failed or timed out".to_string());
                    }
                }
            }

            if needs_opencode_restart {
                result.needs_opencode_restart = true;
            }
        }
        Err(e) => {
            result.push_err(format!("Parse error: {e}"));
        }
    }

    result
}

pub async fn ask(
    State(config): State<AppConfig>,
    Json(req): Json<AskRequest>,
) -> Json<AskResponse> {
    crate::eval::set_shared_runtime(Some(config.runtime.clone()));
    crate::eval::set_shared_meta(Some(config.meta.clone()));
    crate::eval::set_shared_event_broadcast(Some(config.event_broadcast.clone()));
    eprintln!("\n[web:user] {}", req.message);
    let tx = EventSender::broadcast_only(config.event_broadcast.clone());
    tx.send(serde_json::json!({"type": "start", "message": req.message})).await;
    let llm = crate::llm::LlmClient::new_with_model_and_key(
        &config.llm_url, &config.llm_model, config.llm_api_key.clone(),
    );

    let max_iterations = config.max_iterations;
    let mut all_results: Vec<MutationResult> = vec![];
    let mut all_test_results: Vec<TestCaseResult> = vec![];
    let mut all_code = String::new();
    let mut reply_text = String::new();

    let system_prompt = {
        let base = crate::prompt::system_prompt();
        let builtins = crate::builtins::format_for_prompt();
        let identity = crate::prompt::adapsis_identity();
        format!("{base}\n\n{builtins}\n\n{identity}")
    };

    // Build messages from conversation history
    let mut session = config.snapshot_working_set().await;
    let mut messages = {
        if session.meta.chat_messages.is_empty() {
            session.meta.chat_messages.push(crate::session::ChatMessage {
                role: "system".to_string(),
                content: system_prompt,
            });
        }
        let (plan_ctx, needs_plan) = build_plan_context(&session.meta.plan);
        let plan_hint = if needs_plan {
            "\n\nYour previous plan is completed (or none exists). Create a new plan with !plan set for this task before writing code. You can update it anytime with !plan set / !plan done N."
        } else { "" };
        let load_errors_ctx = format_library_load_errors(&session.meta);
        let context = format!(
            "Working directory: {}\n{}{}{}\nUser: {}{}",
            config.project_dir,
            crate::validator::program_summary_compact(&session.program),
            load_errors_ctx,
            plan_ctx,
            req.message,
            plan_hint
        );
        session.meta.chat_messages.push(crate::session::ChatMessage {
            role: "user".to_string(),
            content: context,
        });
        session.meta.chat_messages.iter().map(|m| match m.role.as_str() {
            "system" => crate::llm::ChatMessage::system(m.content.clone()),
            "assistant" => crate::llm::ChatMessage::assistant(&m.content),
            _ => crate::llm::ChatMessage::user(m.content.clone()),
        }).collect::<Vec<_>>()
    };

    for iteration in 0..max_iterations {
        eprintln!("[web:iter {}/{}]", iteration + 1, max_iterations);

        // Call LLM
        let output = match llm.generate(messages.clone()).await {
            Ok(o) => o,
            Err(e) => {
                eprintln!("[web:error] LLM: {e}");
                tx.send(serde_json::json!({"type": "error", "message": format!("LLM error: {e}")})).await;
                reply_text.push_str(&format!("\n\nLLM error: {e}"));
                break;
            }
        };

        messages.push(crate::llm::ChatMessage::assistant(&output.text));

        let code = output.code.clone();

        // Build reply from thinking + prose
        let mut clean = output.text.clone();
        while let Some(s) = clean.find("<think>") {
            if let Some(e) = clean[s..].find("</think>") { clean.replace_range(s..s+e+8, ""); } else { break; }
        }
        while let Some(s) = clean.find("<code>") {
            if let Some(e) = clean[s..].find("</code>") { clean.replace_range(s..s+e+7, ""); } else { break; }
        }
        let clean = clean.trim();
        if !clean.is_empty() {
            tx.send(serde_json::json!({"type": "text", "text": clean})).await;
            if !reply_text.is_empty() { reply_text.push_str("\n\n"); }
            reply_text.push_str(clean);
        }
        if !output.thinking.is_empty() {
            eprintln!("[web:think] {}...", output.thinking.chars().take(100).collect::<String>());
            tx.send(serde_json::json!({"type": "thinking", "text": output.thinking})).await;
        }

        // Check for !done or no code (AI is asking a question / responding with text)
        if code.trim() == "!done" {
            eprintln!("[web:done] model said !done at iteration {}", iteration + 1);
            tx.send(serde_json::json!({"type": "done"})).await;
            break;
        }
        if code.is_empty() {
            // No code block = AI is responding with text only (question or explanation)
            eprintln!("[web:text-only] no code block, stopping");
            tx.send(serde_json::json!({"type": "done"})).await;
            break;
        }

        eprintln!("[web:code]\n{}", code.chars().take(200).collect::<String>());
        tx.send(serde_json::json!({"type": "code", "code": code})).await;
        if !all_code.is_empty() { all_code.push_str("\n\n// --- iteration ---\n"); }
        all_code.push_str(&code);

        // Execute code via shared execute_code() function
        let mut session = config.snapshot_working_set().await;
        let exec_result = execute_code(&code, &config, &mut session, None).await;

        // Send SSE events for results
        for r in &exec_result.mutation_results {
            tx.send(serde_json::json!({"type": "result", "message": r.message, "success": r.success})).await;
        }
        for t in &exec_result.test_results {
            tx.send(serde_json::json!({"type": "test", "pass": t.pass, "message": t.message})).await;
        }

        let iter_has_errors = exec_result.has_errors;

        // Handle opencode restart
        if exec_result.needs_opencode_restart {
            tokio::spawn(async {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                let exe = std::env::args().next()
                    .map(std::path::PathBuf::from)
                    .and_then(|p| std::fs::canonicalize(&p).ok().or(Some(p)))
                    .unwrap_or_else(|| std::env::current_exe().unwrap_or_default());
                let args: Vec<String> = std::env::args().collect();
                let _ = exec::execvp(&exe, &args);
            });
        }

        all_results.extend(exec_result.mutation_results.clone());
        all_test_results.extend(exec_result.test_results.clone());
        // Write mutations back to tiers after each iteration
        config.write_back_working_set(&session).await;

        // Build feedback for next iteration
        if iter_has_errors {
            let errors: Vec<String> = exec_result.mutation_results.iter().filter(|r| !r.success).map(|r| r.message.clone())
                .chain(exec_result.test_results.iter().filter(|r| !r.pass).map(|r| r.message.clone()))
                .collect();
            let feedback = format!("Errors:\n{}\n\nFix and continue.", errors.join("\n"));
            eprintln!("[web:feedback] → retrying");
            messages.push(crate::llm::ChatMessage::user(feedback));
        } else {
            // Success — tell the AI to continue or finish
            let results_summary: Vec<String> = exec_result.mutation_results.iter().map(|r| r.message.clone()).collect();
            let feedback = format!(
                "Results:\n{}\n\nIf the task is complete, respond with !done. Otherwise continue with the next step.",
                results_summary.join("\n")
            );
            messages.push(crate::llm::ChatMessage::user(feedback));
        }
    }

    // Save conversation — write assistant reply directly into meta tier
    {
        let summary = format!("{}\n{}", reply_text.chars().take(200).collect::<String>(),
            all_results.iter().map(|r| format!("{}: {}", if r.success {"OK"} else {"ERR"}, r.message)).collect::<Vec<_>>().join("\n"));
        let mut meta = config.meta.lock().unwrap();
        meta.chat_messages.push(crate::session::ChatMessage {
            role: "assistant".to_string(), content: summary,
        });
        if meta.chat_messages.len() > 50 {
            let system = meta.chat_messages[0].clone();
            let start = meta.chat_messages.len() - 49;
            let keep: Vec<_> = meta.chat_messages[start..].to_vec();
            meta.chat_messages = vec![system];
            meta.chat_messages.extend(keep);
        }
    }

    let has_errors = all_results.iter().any(|r| !r.success) || all_test_results.iter().any(|r| !r.pass);
    if has_errors {
        tx.send(serde_json::json!({"type": "error", "message": "request completed with errors"})).await;
    }
    tx.send(serde_json::json!({"type": "done"})).await;
    Json(AskResponse {
        reply: reply_text,
        code: all_code,
        results: all_results,
        test_results: all_test_results,
        has_errors,
    })
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

/// SSE streaming version of /api/ask — streams events as they happen.
pub async fn ask_stream(
    State(config): State<AppConfig>,
    Json(req): Json<AskRequest>,
) -> axum::response::sse::Sse<impl futures::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>> {
    use axum::response::sse::{Event, KeepAlive};
    use tokio::sync::mpsc;

    let (raw_tx, mut rx) = mpsc::channel::<serde_json::Value>(100);

    // Spawn the processing loop
    let config_clone = config.clone();
    tokio::spawn(async move {
        crate::eval::set_shared_runtime(Some(config_clone.runtime.clone()));
        crate::eval::set_shared_meta(Some(config_clone.meta.clone()));
        crate::eval::set_shared_event_broadcast(Some(config_clone.event_broadcast.clone()));
        let tx = EventSender::with_mpsc(raw_tx, config_clone.event_broadcast.clone(), config_clone.log_file.clone());
        let llm = crate::llm::LlmClient::new_with_model_and_key(
            &config_clone.llm_url, &config_clone.llm_model, config_clone.llm_api_key.clone(),
        );

        let _ = tx.send(serde_json::json!({"type": "start", "message": req.message})).await;

        let system_prompt = {
            let base = crate::prompt::system_prompt();
            let builtins = crate::builtins::format_for_prompt();
            let identity = crate::prompt::adapsis_identity();
            format!("{base}\n\n{builtins}\n\n{identity}")
        };

            let mut messages = {
            // Tier 1: read program briefly for summary
            let program_summary = {
                let program = config_clone.program.read().await;
                crate::validator::program_summary_compact(&program)
            };
            // Tier 3: read/write meta briefly for chat history + plan context
            // Note: guard must be dropped before any .await — std::sync::MutexGuard is not Send.
            let (context, msgs, meta_snapshot) = {
                let mut meta = config_clone.meta.lock().unwrap();
                if meta.chat_messages.is_empty() {
                    meta.chat_messages.push(crate::session::ChatMessage {
                        role: "system".to_string(), content: system_prompt,
                    });
                }
                let (plan_ctx, needs_plan) = build_plan_context(&meta.plan);
                let plan_hint = if needs_plan {
                    "\n\nYour previous plan is completed (or none exists). Create a new plan with !plan set for this task before writing code. You can update it anytime with !plan set / !plan done N."
                } else { "" };
                let load_errors_ctx = format_library_load_errors(&meta);
                let context = format!("Working directory: {}\n{}{}{}\nUser: {}{}",
                    config_clone.project_dir,
                    program_summary,
                    load_errors_ctx,
                    plan_ctx, req.message, plan_hint);
                meta.chat_messages.push(crate::session::ChatMessage {
                    role: "user".to_string(), content: context.clone(),
                });
                let msgs = meta.chat_messages.iter().map(|m| match m.role.as_str() {
                    "system" => crate::llm::ChatMessage::system(m.content.clone()),
                    "assistant" => crate::llm::ChatMessage::assistant(&m.content),
                    _ => crate::llm::ChatMessage::user(m.content.clone()),
                }).collect::<Vec<_>>();
                let meta_snapshot = meta.clone();
                // guard dropped here — before any .await
                (context, msgs, meta_snapshot)
            };
            tx.log("user", &context).await;
            msgs
        }; // All locks released before LLM call

        let max_iterations = config_clone.max_iterations;
        let mut last_context = req.message.clone();
        for iteration in 0..max_iterations {
            // Check for injected messages and append to conversation
            {
                let mut queue = config_clone.message_queue.lock().await;
                for injected in queue.drain(..) {
                    tx.log("inject", &injected).await;
                    messages.push(crate::llm::ChatMessage::user(injected));
                }
            }

            tx.log("iter", &format!("iteration {}/{}", iteration + 1, max_iterations)).await;

            // === Streaming LLM call ===
            // generate_streaming() retries the HTTP connection internally.
            // Once connected, chunks flow through the channel — no "waiting"
            // timer needed because the chunks themselves are progress.
            let output = {
                let mut rx = match llm.generate_streaming(messages.clone()).await {
                    Ok(rx) => rx,
                    Err(e) => {
                        tx.log("llm-error", &format!("{e}")).await;
                        break;
                    }
                };

                // Forward incremental chunks to SSE
                let mut final_output = None;
                while let Some(chunk) = rx.recv().await {
                    match chunk {
                        crate::llm::StreamChunk::Thinking(text) => {
                            let _ = tx.send(serde_json::json!({"type": "thinking", "text": text})).await;
                        }
                        crate::llm::StreamChunk::Content(text) => {
                            let _ = tx.send(serde_json::json!({"type": "content", "text": text})).await;
                        }
                        crate::llm::StreamChunk::Done(output) => {
                            final_output = Some(output);
                        }
                    }
                }

                match final_output {
                    Some(o) => o,
                    None => {
                        // Channel closed without Done — stream error
                        tx.log("llm-error", "LLM stream ended without completing").await;
                        break;
                    }
                }
            };

            messages.push(crate::llm::ChatMessage::assistant(&output.text));

            // Log thinking (full text for the log file, already streamed incrementally above)
            if !output.thinking.is_empty() {
                write_log_file(&config_clone.log_file, "think", &output.thinking).await;
            }

            // Extract prose and send as a single text event
            let mut clean = output.text.clone();
            while let Some(s) = clean.find("<think>") { if let Some(e) = clean[s..].find("</think>") { clean.replace_range(s..s+e+8, ""); } else { break; } }
            while let Some(s) = clean.find("<code>") { if let Some(e) = clean[s..].find("</code>") { clean.replace_range(s..s+e+7, ""); } else { break; } }
            let clean = clean.trim();
            if !clean.is_empty() {
                write_log_file(&config_clone.log_file, "ai-text", clean).await;
                let _ = tx.send(serde_json::json!({"type": "text", "text": clean})).await;
            }

            let code = output.code.trim().to_string();

            // Empty code = AI responded with prose only, no operations
            if code.is_empty() {
                // Push a user message so the conversation doesn't end on assistant
                // (some models like MiMo require the last message to be from the user)
                messages.push(crate::llm::ChatMessage::user(
                    "Your response contained no Adapsis operations. Write code with +, !, or ? prefixes.".to_string()
                ));
                continue;
            }

            tx.log("code", &code).await;

            // Apply code
            let mut session = config_clone.snapshot_working_set().await;
            let mut op_result = OperationResult::new();

            match crate::parser::parse(&code) {
                Ok(ops) => {
                    // Remove duplicates
                    let mut fns_removed = false;
                    for op in &ops {
                        match op {
                            crate::parser::Operation::Function(f) => { session.program.functions.retain(|e| e.name != f.name); fns_removed = true; }
                            crate::parser::Operation::Type(t) => { let n = t.name.clone(); session.program.types.retain(|e| e.name() != n); }
                            _ => {}
                        }
                    }
                    if fns_removed {
                        session.program.rebuild_function_index();
                    }

                    // Handle plan
                    let has_plan_ops = ops.iter().any(|op| matches!(op, crate::parser::Operation::Plan(_)));
                    for op in &ops {
                        if let crate::parser::Operation::Plan(action) = op {
                            process_plan(action, &mut session.meta, &tx, &mut op_result).await;
                        }
                    }
                    if has_plan_ops {
                        config_clone.write_back_working_set(&session).await;
                    }

                    let has_mutations = ops.iter().any(|op| !matches!(op,
                        crate::parser::Operation::Test(_) | crate::parser::Operation::Eval(_)
                        | crate::parser::Operation::Query(_) | crate::parser::Operation::Trace(_)
                        | crate::parser::Operation::Undo | crate::parser::Operation::Plan(_)
                        | crate::parser::Operation::Watch { .. } | crate::parser::Operation::Agent { .. }
                        | crate::parser::Operation::Message { .. }
                        | crate::parser::Operation::Done
                        | crate::parser::Operation::OpenCode(_)));

                    if has_mutations {
                        match crate::session::apply_to_tiers(&mut session.program, &mut session.runtime, &mut session.meta, &mut session.sandbox, &code) {
                            Ok(res) => {
                                config_clone.write_back_working_set(&session).await;
                                for (msg, ok) in &res {
                                    if *ok {
                                        op_result.ok(msg);
                                    } else {
                                        op_result.error(msg);
                                    }
                                    let _ = tx.send(serde_json::json!({"type": "result", "message": msg, "success": ok})).await;
                                }
                            }
                            Err(e) => {
                                op_result.error(format!("{e}"));
                                let _ = tx.send(serde_json::json!({"type": "result", "message": format!("{e}"), "success": false})).await;
                            }
                        }
                    }

                    // Tests and evals
                    let mut needs_opencode_restart = false;
                    for op in &ops {
                        match op {
                            crate::parser::Operation::Test(test) => {
                                let mut all_passed = true;
                                let needs_async = session.program.get_function(&test.function_name)
                                    .is_some_and(|f| f.effects.iter().any(|e|
                                        matches!(e, crate::ast::Effect::Io | crate::ast::Effect::Async)));

                                for case in &test.cases {
                                    let case_result = if needs_async {
                                        if let Some(sender) = &config_clone.io_sender {
                                            let program = session.program.clone();
                                            let fn_name = test.function_name.clone();
                                            let case = case.clone();
                                            let mocks = session.meta.io_mocks.clone();
                                            let routes = session.runtime.http_routes.clone();
                                            let sender = sender.clone();
                                            let result = crate::eval::eval_test_case_async(
                                                &program, &fn_name, &case, &mocks, sender, &routes,
                                            ).await;
                                            result
                                        } else {
                                            crate::eval::eval_test_case_with_mocks(
                                                &session.program, &test.function_name, case, &session.meta.io_mocks, &session.runtime.http_routes,
                                            )
                                        }
                                    } else {
                                        crate::eval::eval_test_case_with_mocks(
                                            &session.program, &test.function_name, case, &session.meta.io_mocks, &session.runtime.http_routes,
                                        )
                                    };
                                    match case_result {
                                        Ok(msg) => {
                                            op_result.pass(&msg);
                                            let _ = tx.send(serde_json::json!({"type": "test", "pass": true, "message": msg})).await;
                                        }
                                        Err(e) => {
                                            all_passed = false;
                                            op_result.fail(format!("{e}"));
                                            let _ = tx.send(serde_json::json!({"type": "test", "pass": false, "message": format!("{e}")})).await;
                                        }
                                    }
                                }
                                if all_passed && !test.cases.is_empty() {
                                    crate::session::store_test(&mut session.program, &test.function_name, &test.cases);
                                }
                            }
                            crate::parser::Operation::Eval(ev) => {
                                // Inline expression: evaluate directly
                                if let Some(ref expr) = ev.inline_expr {
                                    // Check if expression contains IO builtins — run async if so
                                    if crate::eval::expr_contains_io_builtin(expr) {
                                        if let Some(sender) = &config_clone.io_sender {
                                            let program = session.program.clone();
                                            let program_mut = crate::eval::make_shared_program_mut(&program);
                                            let program_mut_clone = program_mut.clone();
                                            let expr = expr.clone();
                                            let sender = sender.clone();
                                            let runtime_for_blocking = config_clone.runtime.clone();
                                            let meta_for_blocking = config_clone.meta.clone();
                                            let event_broadcast = config_clone.event_broadcast.clone();
                                            let eval_result = tokio::time::timeout(
                                                std::time::Duration::from_secs(EVAL_TIMEOUT_SECS),
                                                tokio::task::spawn_blocking(move || {
                                                    crate::eval::set_shared_runtime(Some(runtime_for_blocking));
                                                    eval::set_shared_meta(Some(meta_for_blocking));
                                                    eval::set_shared_event_broadcast(Some(event_broadcast));
                                                    crate::eval::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                                                    crate::eval::set_shared_program_mut(Some(program_mut_clone));
                                                    crate::eval::eval_inline_expr_with_io(&program, &expr, sender)
                                                })
                                            ).await;
                                            let (msg, success) = match &eval_result {
                                                Ok(Ok(Ok(val))) => (format!("{val}"), true),
                                                Ok(Ok(Err(e))) => (format!("eval error: {e}"), false),
                                                Ok(Err(e)) => (format!("eval task error: {e}"), false),
                                                Err(_) => (format!("eval timed out after {EVAL_TIMEOUT_SECS}s"), false),
                                            };
                                            if success {
                                                op_result.info(format!("= {msg}"));
                                            } else {
                                                op_result.error(&msg);
                                            }
                                            let _ = tx.send(serde_json::json!({"type": "eval", "result": msg, "function": "(inline)", "success": success})).await;
                                            // Sync mutations back
                                            if let Some(mutated) = crate::eval::read_back_program_mutations(&program_mut) {
                                                session.program = mutated.clone();
                                                *config_clone.program.write().await = mutated;
                                            }
                                            config_clone.sync_async_side_effects_into(&mut session);
                                            continue;
                                        }
                                        // No IO sender — fall through to sync eval which will error
                                    }
                                    match crate::eval::eval_inline_expr(&session.program, expr) {
                                        Ok(val) => {
                                            let msg = format!("{val}");
                                            op_result.info(format!("= {msg}"));
                                            let _ = tx.send(serde_json::json!({"type": "eval", "result": msg, "function": "(inline)", "success": true})).await;
                                        }
                                        Err(e) => {
                                            let msg = format!("eval error: {e}");
                                            op_result.error(&msg);
                                            let _ = tx.send(serde_json::json!({"type": "eval", "result": msg, "function": "(inline)", "success": false})).await;
                                        }
                                    }
                                    continue;
                                }

                                // Block eval of untested functions (>2 statements) in AdapsisOS mode
                                if session.program.require_modules {
                                    if let Some(func) = session.program.get_function(&ev.function_name) {
                                        if func.body.len() > 2 && !crate::session::is_function_tested(&session.program, &ev.function_name) {
                                            let msg = format!("function `{}` has {} statements but no passing tests. Write !test blocks first.", ev.function_name, func.body.len());
                                            op_result.error(&msg);
                                            let _ = tx.send(serde_json::json!({"type": "eval", "result": msg, "function": ev.function_name, "success": false})).await;
                                            continue;
                                        }
                                    }
                                }

                                let needs_async = session.program.get_function(&ev.function_name)
                                    .is_some_and(|f| f.effects.iter().any(|e|
                                        matches!(e, crate::ast::Effect::Io | crate::ast::Effect::Async)));

                                if needs_async {
                                    if let Some(sender) = &config_clone.io_sender {
                                        let program = session.program.clone();
                                        let program_mut = crate::eval::make_shared_program_mut(&program);
                                        let program_mut_clone = program_mut.clone();
                                        let fn_name = ev.function_name.clone();
                                        let input = ev.input.clone();
                                        let sender = sender.clone();
                                        let runtime_for_blocking = config_clone.runtime.clone();
                                        let meta_for_blocking = config_clone.meta.clone();
                                        let event_broadcast = config_clone.event_broadcast.clone();
                                        let eval_fn_name = ev.function_name.clone();
                                        let eval_result = tokio::time::timeout(
                                            std::time::Duration::from_secs(EVAL_TIMEOUT_SECS),
                                            tokio::task::spawn_blocking(move || {
                                                crate::eval::set_shared_runtime(Some(runtime_for_blocking));
                                                eval::set_shared_meta(Some(meta_for_blocking));
                                                eval::set_shared_event_broadcast(Some(event_broadcast));
                                                crate::eval::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                                                crate::eval::set_shared_program_mut(Some(program_mut_clone));
                                                let func = program.get_function(&fn_name)
                                                    .ok_or_else(|| anyhow::anyhow!("function not found"))?;
                                                let handle = crate::coroutine::CoroutineHandle::new(sender);
                                                let mut env = crate::eval::Env::new_with_shared_interner(&program.shared_interner);
                                                env.populate_shared_from_program(&program);
                                                env.set("__coroutine_handle", crate::eval::Value::CoroutineHandle(handle));
                                                let input_val = crate::eval::eval_parser_expr_with_program(&input, &program)?;
                                                crate::eval::bind_input_to_params(&program, func, &input_val, &mut env);
                                                crate::eval::eval_function_body_pub(&program, &func.body, &mut env)
                                            })
                                        ).await;
                                        let (msg, success) = match &eval_result {
                                            Ok(Ok(Ok(val))) => (format!("{val}"), true),
                                            Ok(Ok(Err(e))) => (format!("error: {e}"), false),
                                            Ok(Err(e)) => (format!("task error: {e}"), false),
                                            Err(_) => (format!("eval {}() timed out after {EVAL_TIMEOUT_SECS}s", eval_fn_name), false),
                                        };
                                        if success {
                                            op_result.info(format!("eval {}() = {msg}", ev.function_name));
                                        } else {
                                            op_result.error(format!("eval {}() = {msg} [FAILED]", ev.function_name));
                                        }
                                        let _ = tx.send(serde_json::json!({"type": "eval", "result": msg, "function": ev.function_name, "success": success})).await;
                                        // Sync mutations back
                                        if let Some(mutated) = crate::eval::read_back_program_mutations(&program_mut) {
                                            session.program = mutated.clone();
                                            *config_clone.program.write().await = mutated;
                                        }
                                        config_clone.sync_async_side_effects_into(&mut session);
                                    } else {
                                        op_result.error(format!("eval {}() = async not available [FAILED]", ev.function_name));
                                        let _ = tx.send(serde_json::json!({"type": "eval", "result": "async not available", "function": ev.function_name, "success": false})).await;
                                    }
                                } else {
                                    match crate::eval::eval_compiled_or_interpreted_cached(&session.program, &ev.function_name, &ev.input, Some(&config_clone.jit_cache), session.meta.revision) {
                                        Ok((result, compiled)) => {
                                            let tag = if compiled { " [compiled]" } else { "" };
                                            op_result.info(format!("eval {}() = {result}{tag}", ev.function_name));
                                            let _ = tx.send(serde_json::json!({"type": "eval", "result": format!("{result}{tag}"), "function": ev.function_name})).await;
                                        }
                                        Err(e) => {
                                            op_result.error(format!("eval {}() error: {e} [FAILED]", ev.function_name));
                                            let _ = tx.send(serde_json::json!({"type": "eval", "result": format!("error: {e}"), "function": ev.function_name})).await;
                                        }
                                    }
                                }
                            }
                            crate::parser::Operation::Query(query) => {
                                process_query(query, &session, &config_clone, &tx, &mut op_result).await;
                            }
                            // Top-level statements: execute immediately
                            crate::parser::Operation::Call(_)
                            | crate::parser::Operation::Let(_)
                            | crate::parser::Operation::Set(_)
                            | crate::parser::Operation::Await(_)
                            | crate::parser::Operation::Spawn(_)
                            | crate::parser::Operation::If(_)
                            | crate::parser::Operation::While(_)
                            | crate::parser::Operation::Each(_)
                            | crate::parser::Operation::Match(_)
                            | crate::parser::Operation::Check(_)
                            | crate::parser::Operation::Branch(_)
                            | crate::parser::Operation::Return(_) => {
                                match crate::validator::convert_statement_op(op) {
                                    Ok(stmt) => {
                                        let mut env = crate::eval::Env::new_with_shared_interner(&session.program.shared_interner);
                                        env.populate_shared_from_program(&session.program);
                                        if let Some(sender) = &config_clone.io_sender {
                                            env.set("__coroutine_handle", crate::eval::Value::CoroutineHandle(
                                                crate::coroutine::CoroutineHandle::new(sender.clone())
                                            ));
                                        }
                                        let program = session.program.clone();
                                        let program_mut = crate::eval::make_shared_program_mut(&program);
                                        let program_mut_clone = program_mut.clone();
                                        let runtime_for_blocking = config_clone.runtime.clone();
                                        let meta_for_blocking = config_clone.meta.clone();
                                        let event_broadcast = config_clone.event_broadcast.clone();
                                        let result = tokio::task::spawn_blocking(move || {
                                            crate::eval::set_shared_runtime(Some(runtime_for_blocking));
                                            eval::set_shared_meta(Some(meta_for_blocking));
                                            eval::set_shared_event_broadcast(Some(event_broadcast));
                                            crate::eval::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                                            crate::eval::set_shared_program_mut(Some(program_mut_clone));
                                            crate::eval::eval_function_body_pub(&program, &[stmt], &mut env)
                                        }).await;
                                        match result {
                                            Ok(Ok(val)) => {
                                                let msg = format!("executed: {val}");
                                                op_result.ok(&msg);
                                                let _ = tx.send(serde_json::json!({"type": "result", "message": msg, "success": true})).await;
                                            }
                                            Ok(Err(e)) => {
                                                op_result.error(format!("{e}"));
                                                let _ = tx.send(serde_json::json!({"type": "result", "message": format!("{e}"), "success": false})).await;
                                            }
                                            Err(e) => {
                                                op_result.error(format!("task error: {e}"));
                                                let _ = tx.send(serde_json::json!({"type": "result", "message": format!("task error: {e}"), "success": false})).await;
                                            }
                                        }
                                        // Sync mutations back
                                        if let Some(mutated) = crate::eval::read_back_program_mutations(&program_mut) {
                                            session.program = mutated.clone();
                                            *config_clone.program.write().await = mutated;
                                        }
                                        config_clone.sync_async_side_effects_into(&mut session);
                                    }
                                    Err(e) => {
                                        op_result.error(format!("statement error: {e}"));
                                        let _ = tx.send(serde_json::json!({"type": "result", "message": format!("statement error: {e}"), "success": false})).await;
                                    }
                                }
                            }
                            crate::parser::Operation::Done => {
                                let accepted = process_done(&session, iteration, &tx, &mut op_result).await;
                                if !accepted {
                                    continue; // untested functions — keep going
                                }
                                break; // accepted done
                            }
                            crate::parser::Operation::Mock { operation, patterns, response } => {
                                process_mock(operation, patterns, response, &mut session.meta, &mut op_result);
                                config_clone.write_back_working_set(&session).await;
                                let pattern_display = patterns.iter().map(|p| format!("\"{p}\"")).collect::<Vec<_>>().join(" ");
                                let _ = tx.send(serde_json::json!({"type": "result", "message": format!("mock: {operation} {pattern_display}"), "success": true})).await;
                            }
                            crate::parser::Operation::Unmock => {
                                let count = session.meta.io_mocks.len();
                                process_unmock(&mut session.meta, &mut op_result);
                                config_clone.write_back_working_set(&session).await;
                                let _ = tx.send(serde_json::json!({"type": "result", "message": format!("cleared {count} mocks"), "success": true})).await;
                            }
                            crate::parser::Operation::Stub { function_name, patterns, response_expr } => {
                                let pattern_display = patterns.iter().map(|p| format!("\"{p}\"")).collect::<Vec<_>>().join(" ");
                                session.meta.function_stubs.push(crate::session::FunctionStub {
                                    function_name: function_name.clone(),
                                    patterns: patterns.clone(),
                                    response_expr: response_expr.clone(),
                                });
                                config_clone.write_back_working_set(&session).await;
                                op_result.ok_silent(format!("stub: {function_name} {pattern_display} -> {}", response_expr.chars().take(60).collect::<String>()));
                                let _ = tx.send(serde_json::json!({"type": "result", "message": format!("stub: {function_name} {pattern_display}"), "success": true})).await;
                            }
                            crate::parser::Operation::Unstub => {
                                let count = session.meta.function_stubs.len();
                                session.meta.function_stubs.clear();
                                config_clone.write_back_working_set(&session).await;
                                op_result.ok_silent(format!("cleared {count} stubs"));
                                let _ = tx.send(serde_json::json!({"type": "result", "message": format!("cleared {count} stubs"), "success": true})).await;
                            }
                            crate::parser::Operation::Message { to, content } => {
                                crate::session::send_agent_message(&mut session.meta, "main", to, content);
                                config_clone.write_back_working_set(&session).await;
                                op_result.ok_silent(format!("Message sent to '{to}'"));
                                let _ = tx.send(serde_json::json!({"type": "result", "message": format!("Message sent to '{to}'"), "success": true})).await;
                            }
                            crate::parser::Operation::OpenCode(task) => {
                                eprintln!("[web:opencode:stream] {task}");
                                log_activity(&config_clone.log_file, "opencode", &task).await;
                                // Sequential lock — only one !opencode at a time
                                let _opencode_guard = config_clone.opencode_lock.lock().await;
                                let _ = tx.send(serde_json::json!({"type": "result", "message": format!("Running !opencode: {}", task.chars().take(80).collect::<String>()), "success": true})).await;
                                // Use the configured opencode directory (fixed checkout, not dynamic worktrees)
                                let work_dir = config_clone.opencode_git_dir.clone();
                                log_activity(&config_clone.log_file, "opencode-dir", &work_dir).await;
                                use tokio::io::{AsyncBufReadExt, BufReader};

                                // Get existing OpenCode session ID to continue building on top
                                let oc_session_id = {
                                    let meta = config_clone.meta.lock().unwrap();
                                    meta.opencode_session_id.clone()
                                };

                                let recent_lines = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
                                let recent_for_stream = recent_lines.clone();
                                let had_tool_calls = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                                let had_tool_calls_stream = had_tool_calls.clone();
                                let killed_by_idle = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                                let killed_by_idle_inner = killed_by_idle.clone();
                                let last_text = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
                                let last_text_stream = last_text.clone();
                                let full_task = format!("{task}. IMPORTANT: Do NOT ask for clarification — proceed with your best judgment. When done: 1) Write Rust tests for the changes you made — cover the happy path and at least one error case. 2) Update src/prompt.rs to document any new builtins, IO operations, commands, or language features you added — the AI inside AdapsisOS needs to know about them. 3) Register new builtins in src/builtins.rs. 4) Run `cargo test` and fix any failures. 5) Create clean atomic git commits with descriptive messages for each logical change.");
                                let oc_result = tokio::time::timeout(
                                    std::time::Duration::from_secs(3600),
                                    async {
                                        let mut cmd = tokio::process::Command::new("opencode");
                                        cmd.arg("run").arg("--format").arg("json")
                                            .arg("--attach").arg("http://localhost:4096")
                                            .arg("--dir").arg(&work_dir);
                                        if let Some(sid) = &oc_session_id {
                                            cmd.arg("--session").arg(sid).arg("--fork");
                                        }
                                        cmd.arg(&full_task);
                                        let mut child = cmd
                                            .current_dir(&work_dir)
                                            .stdout(std::process::Stdio::piped())
                                            .stderr(std::process::Stdio::piped())
                                            .process_group(0) // Create new process group so we can kill all children
                                            .spawn()?;

                                        let stdout = child.stdout.take().unwrap();
                                        let mut reader = BufReader::new(stdout).lines();
                                        // Track last activity from both stdout and stderr.
                                        // The idle timeout kills only if NEITHER stream has
                                        // produced output for the timeout duration.
                                        let last_activity = std::sync::Arc::new(std::sync::Mutex::new(std::time::Instant::now()));
                                        let last_activity_stderr = last_activity.clone();

                                        // Spawn stderr reader that updates last_activity
                                        let stderr = child.stderr.take();
                                        if let Some(stderr) = stderr {
                        tokio::spawn(async move {
                                                let mut reader = BufReader::new(stderr).lines();
                                                while let Ok(Some(_)) = reader.next_line().await {
                                                    *last_activity_stderr.lock().unwrap() = std::time::Instant::now();
                                                }
                                            });
                                        }

                                        let idle_timeout = std::time::Duration::from_secs(7200); // 2 hours
                                        loop {
                                            let line = match tokio::time::timeout(std::time::Duration::from_secs(30), reader.next_line()).await {
                                                Ok(Ok(Some(line))) => {
                                                    *last_activity.lock().unwrap() = std::time::Instant::now();
                                                    line
                                                }
                                                Ok(Ok(None)) => break, // EOF
                                                Ok(Err(e)) => { eprintln!("[opencode] read error: {e}"); break; }
                                                Err(_) => {
                                                    // No stdout line for 30s — check if stderr had activity
                                                    let elapsed = last_activity.lock().unwrap().elapsed();
                                                    if elapsed >= idle_timeout {
                                                        let context = recent_for_stream.lock().unwrap().join("\n");
                                                        let msg = format!(
                                                            "ERROR: !opencode idle timeout — no output for {}s. The task is NOT done.\n\
                                                             Last output before silence:\n{context}\n\n\
                                                             The opencode subprocess went silent. This usually means it got stuck on a \
                                                             long operation or API rate limit. Try breaking the task into smaller pieces.",
                                                            elapsed.as_secs()
                                                        );
                                                        eprintln!("[opencode] IDLE TIMEOUT: {}s", elapsed.as_secs());
                                                        log_activity(&config_clone.log_file, "opencode-timeout", &msg).await;
                                                        op_result.error(&msg);
                                                        killed_by_idle_inner.store(true, std::sync::atomic::Ordering::Relaxed);
                                                        // Kill entire process group to clean up opencode subprocesses
                                                        if let Some(pid) = child.id() {
                                                            unsafe { libc::kill(-(pid as i32), libc::SIGKILL); }
                                                        }
                                                        let _ = child.kill().await;
                                                        break;
                                                    }
                                                    continue; // stderr had recent activity, keep waiting
                                                }
                                            };
                                            // Keep last 20 lines for error context
                                            {
                                                let mut rl = recent_for_stream.lock().unwrap();
                                                rl.push(line.clone());
                                                if rl.len() > 20 { rl.remove(0); }
                                            }
                                            // Parse JSON events from opencode
                                            if let Ok(event) = serde_json::from_str::<serde_json::Value>(&line) {
                                                let event_type = event.get("type").and_then(|t| t.as_str()).unwrap_or("");
                                                match event_type {
                                                    "text" => {
                                                        if let Some(part) = event.get("part") {
                                                            if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                                                let preview: String = text.chars().take(200).collect();
                                                                eprintln!("[opencode:text] {preview}");
                                                                log_activity(&config_clone.log_file, "opencode-text", text).await;
                                                                let _ = tx.send(serde_json::json!({"type": "result", "message": format!("OpenCode: {preview}"), "success": true})).await;
                                                                let mut lt = last_text_stream.lock().unwrap();
                                                                lt.push_str(text);
                                                            }
                                                        }
                                                    }
                                                    "tool_call" | "tool_result" => {
                                                        had_tool_calls_stream.store(true, std::sync::atomic::Ordering::Relaxed);
                                                        let summary = event.get("part")
                                                            .and_then(|p| p.get("name").or(p.get("tool")))
                                                            .and_then(|n| n.as_str())
                                                            .unwrap_or("tool");
                                                        eprintln!("[opencode:{event_type}] {summary}");
                                                        log_activity(&config_clone.log_file, &format!("opencode-{event_type}"), summary).await;
                                                        let _ = tx.send(serde_json::json!({"type": "result", "message": format!("OpenCode {event_type}: {summary}"), "success": true})).await;
                                                    }
                                                    "step_start" | "step_finish" => {
                                                        // Capture session ID for reuse
                                                        if let Some(sid) = event.get("sessionID").and_then(|s| s.as_str()) {
                                                            let mut meta = config_clone.meta.lock().unwrap();
                                                            if meta.opencode_session_id.as_deref() != Some(sid) {
                                                                meta.opencode_session_id = Some(sid.to_string());
                                                                eprintln!("[opencode] session ID: {sid}");
                                                            }
                                                        }
                                                    }
                                                    _ => {
                                                        // Extract a short summary for non-text/tool events
                                                        let detail = event.get("part")
                                                            .and_then(|p| {
                                                                p.get("name").or(p.get("tool")).or(p.get("status"))
                                                                    .and_then(|v| v.as_str())
                                                                    .map(|s| s.to_string())
                                                            })
                                                            .or_else(|| event.get("error").and_then(|e| e.as_str()).map(|s| s.chars().take(100).collect()))
                                                            .unwrap_or_default();
                                                        if detail.is_empty() {
                                                            eprintln!("[opencode:event] {event_type}");
                                                        } else {
                                                            eprintln!("[opencode:event] {event_type}: {detail}");
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        child.wait().await
                                    }
                                ).await;

                                // Detect stale session: exit 0 but zero output means --session/--fork silently failed.
                                // Retry without session reuse.
                                let got_output = !recent_lines.lock().unwrap().is_empty();
                                if !got_output && oc_session_id.is_some() {
                                    if let Ok(Ok(status)) = &oc_result {
                                        if status.success() {
                                            eprintln!("[opencode] stale session (exit 0, no output), retrying fresh");
                                            log_activity(&config_clone.log_file, "opencode-retry", "stale session, retrying without --fork").await;
                                            let _ = tx.send(serde_json::json!({"type": "result", "message": "OpenCode session stale, retrying fresh...", "success": true})).await;
                                            // Clear session ID
                                            {
                                                let mut meta = config_clone.meta.lock().unwrap();
                                                meta.opencode_session_id = None;
                                            }
                                            // Retry without --session/--fork
                                            let recent_for_retry = recent_lines.clone();
                                            let retry_result = tokio::time::timeout(
                                                std::time::Duration::from_secs(3600),
                                                async {
                                                    let mut cmd2 = tokio::process::Command::new("opencode");
                                                    cmd2.arg("run").arg("--format").arg("json")
                                                        .arg("--attach").arg("http://localhost:4096")
                                                        .arg("--dir").arg(&work_dir)
                                                        .arg(&full_task);
                                                    let mut child2 = cmd2
                                                        .current_dir(&work_dir)
                                                        .stdout(std::process::Stdio::piped())
                                                        .stderr(std::process::Stdio::piped())
                                                        .process_group(0)
                                                        .spawn()?;
                                                    let stdout2 = child2.stdout.take().unwrap();
                                                    let mut reader2 = BufReader::new(stdout2).lines();
                                                    let idle_timeout = std::time::Duration::from_secs(7200); // 2 hours
                                                    loop {
                                                        let line = match tokio::time::timeout(idle_timeout, reader2.next_line()).await {
                                                            Ok(Ok(Some(line))) => line,
                                                            Ok(Ok(None)) => break,
                                                            Ok(Err(_)) => break,
                                                            Err(_) => {
                                                                if let Some(pid) = child2.id() {
                                                                    unsafe { libc::kill(-(pid as i32), libc::SIGKILL); }
                                                                }
                                                                let _ = child2.kill().await;
                                                                break;
                                                            }
                                                        };
                                                        {
                                                            let mut rl = recent_for_retry.lock().unwrap();
                                                            rl.push(line.clone());
                                                            if rl.len() > 20 { rl.remove(0); }
                                                        }
                                                        if let Ok(event) = serde_json::from_str::<serde_json::Value>(&line) {
                                                            // Capture new session ID
                                                            if let Some(sid) = event.get("sessionID").and_then(|s| s.as_str()) {
                                                                let mut meta = config_clone.meta.lock().unwrap();
                                                                if meta.opencode_session_id.is_none() {
                                                                    meta.opencode_session_id = Some(sid.to_string());
                                                                }
                                                            }
                                                            let event_type = event.get("type").and_then(|t| t.as_str()).unwrap_or("");
                                                            if event_type == "text" {
                                                                if let Some(text) = event.get("part").and_then(|p| p.get("text")).and_then(|t| t.as_str()) {
                                                                    let preview: String = text.chars().take(200).collect();
                                                                    eprintln!("[opencode:text] {preview}");
                                                                    log_activity(&config_clone.log_file, "opencode-text", text).await;
                                                                }
                                                            }
                                                        }
                                                    }
                                                    child2.wait().await
                                                }
                                            ).await;
                                            // Use retry result
                                            match retry_result {
                                                Ok(Ok(s)) if s.success() => {} // fall through to success handling
                                                _ => {
                                                    let ctx = recent_lines.lock().unwrap().join("\n");
                                                    op_result.error(format!("OpenCode retry failed\n{ctx}"));
                                                    continue;
                                                }
                                            }
                                        }
                                    }
                                }

                                // Detect no-op exit: OpenCode exited successfully but made no
                                // file changes.  This usually means it asked for clarification
                                // or only read files.  Report the text as an error so the
                                // Adapsis LLM can retry with a better prompt.
                                if let Ok(Ok(ref status)) = oc_result {
                                    if status.success() {
                                        let has_changes = tokio::process::Command::new("git")
                                            .args(["diff", "--stat", "HEAD"])
                                            .current_dir(&work_dir)
                                            .output()
                                            .await
                                            .map(|o| !o.stdout.is_empty())
                                            .unwrap_or(true); // assume changes if git fails
                                        let has_new_commits = tokio::process::Command::new("git")
                                            .args(["log", "--oneline", "-1", "--since=2 minutes ago"])
                                            .current_dir(&work_dir)
                                            .output()
                                            .await
                                            .map(|o| !o.stdout.is_empty())
                                            .unwrap_or(true);
                                        if !has_changes && !has_new_commits {
                                            let text = last_text.lock().unwrap().clone();
                                            let preview: String = text.chars().take(500).collect();
                                             let msg = format!("ERROR: !opencode made no changes. The task is NOT done. OpenCode said: {preview}\n\nYou must either retry !opencode with a clearer description, or acknowledge this item cannot be completed right now.");
                                            eprintln!("[opencode:no-changes] {msg}");
                                            log_activity(&config_clone.log_file, "opencode-no-changes", &msg).await;
                                            op_result.error(&msg);
                                            let _ = tx.send(serde_json::json!({"type": "result", "message": msg, "success": false})).await;
                                            continue;
                                        }
                                    }
                                }

                                if killed_by_idle.load(std::sync::atomic::Ordering::Relaxed) {
                                    // Already reported via op_result.error — skip the generic SIGKILL match
                                } else { match oc_result {
                                    Ok(Ok(status)) if status.success() => {
                                        eprintln!("[web:opencode:stream:done] rebuilding...");
                                        log_activity(&config_clone.log_file, "opencode-done", "rebuilding...").await;
                                        let _ = tx.send(serde_json::json!({"type": "result", "message": "OpenCode done, rebuilding...", "success": true})).await;

                                        let build = tokio::process::Command::new("cargo")
                                            .arg("build").arg("--release").current_dir(&work_dir).output().await;
                                        match build {
                                            Ok(b) if b.status.success() => {
                                                log_activity(&config_clone.log_file, "opencode-restart", "rebuild successful, attempting restart").await;
                                                let _ = tx.send(serde_json::json!({"type": "result", "message": "OpenCode + rebuild successful. Restarting...", "success": true})).await;
                                                // Save session before restart — include opencode output
                                                // so the AI has context after the execvp restart.
                                                {
                                                    let opencode_text = last_text.lock().unwrap().clone();
                                                    if !opencode_text.is_empty() {
                                                        config_clone.meta.lock().unwrap().last_opencode_output = Some(opencode_text);
                                                    }
                                                    let snap = config_clone.snapshot_working_set().await;
                                                    if let Some(path) = std::env::args().nth(std::env::args().position(|a| a == "--session").unwrap_or(999) + 1) {
                                                        let snap = crate::session::Session { program: snap.program, runtime: snap.runtime, meta: snap.meta, sandbox: snap.sandbox };
                                                        let _ = snap.save(std::path::Path::new(&path));
                                                    }
                                                }
                                                // Defer restart until all operations are processed
                                                needs_opencode_restart = true;
                                                let _ = tx.send(serde_json::json!({"type": "result", "message": "OpenCode + rebuild successful. Will restart after remaining operations.", "success": true})).await;
                                                log_activity(&config_clone.log_file, "opencode-restart", "deferred restart after remaining ops").await;
                                                op_result.ok("OpenCode + rebuild successful. Restart deferred.");
                                            }
                                            Ok(b) => {
                                                let stderr = String::from_utf8_lossy(&b.stderr);
                                                let msg = format!("ERROR: !opencode made changes but cargo build failed. The task is NOT done.\n{stderr}\n\nRetry !opencode to fix the build errors.");
                                                op_result.error(&msg);
                                                log_activity(&config_clone.log_file, "opencode-build-fail", &msg).await;
                                                let _ = tx.send(serde_json::json!({"type": "result", "message": msg, "success": false})).await;
                                            }
                                            Err(e) => {
                                                op_result.error(format!("build error: {e}"));
                                                let _ = tx.send(serde_json::json!({"type": "result", "message": format!("build error: {e}"), "success": false})).await;
                                            }
                                        }
                                    }
                                    Ok(Ok(status)) => {
                                        let context = recent_lines.lock().unwrap().join("\n");
                                        let msg = format!("ERROR: !opencode failed (exit {status}). The task is NOT done.\nLast output:\n{context}\n\nRetry with a different approach or acknowledge this item cannot be completed.");
                                        op_result.error(&msg);
                                        log_activity(&config_clone.log_file, "opencode-fail", &msg).await;
                                        let _ = tx.send(serde_json::json!({"type": "result", "message": msg, "success": false})).await;
                                    }
                                    Ok(Err(e)) => {
                                        let context = recent_lines.lock().unwrap().join("\n");
                                        let msg = format!("ERROR: !opencode crashed: {e}. The task is NOT done.\nLast output:\n{context}\n\nRetry or acknowledge this item cannot be completed.");
                                        op_result.error(&msg);
                                        log_activity(&config_clone.log_file, "opencode-error", &msg).await;
                                        let _ = tx.send(serde_json::json!({"type": "result", "message": msg, "success": false})).await;
                                    }
                                    Err(_) => {
                                        let context = recent_lines.lock().unwrap().join("\n");
                                        let msg = format!("ERROR: !opencode timed out (30 min limit). The task is NOT done.\nLast output:\n{context}\n\nBreak the task into smaller pieces and retry, or acknowledge this item cannot be completed.");
                                        op_result.error(&msg);
                                        log_activity(&config_clone.log_file, "opencode-timeout", &msg).await;
                                        let _ = tx.send(serde_json::json!({"type": "result", "message": msg, "success": false})).await;
                                    }
                                } } // close match + if/else
                            }
                            _ => {}
                        }
                    }

                    // Deferred restart after all operations processed
                    if needs_opencode_restart {
                        // Save session one more time (may have changed since first opencode)
                        {
                            let snap = config_clone.snapshot_working_set().await;
                            if let Some(path) = std::env::args().nth(std::env::args().position(|a| a == "--session").unwrap_or(999) + 1) {
                                let snap = crate::session::Session { program: snap.program, runtime: snap.runtime, meta: snap.meta, sandbox: snap.sandbox };
                                let _ = snap.save(std::path::Path::new(&path));
                            }
                        }
                        // Prefer newly-built binary from the opencode worktree
                        let worktree_binary = {
                            let wb = std::path::PathBuf::from(&config_clone.opencode_git_dir).join("target/release/adapsis");
                            if wb.exists() { Some(wb) } else { None }
                        };
                        // Install to ~/.local/bin/adapsis if we have a worktree build
                        if let Some(ref wb) = worktree_binary {
                            let install_path = dirs::home_dir()
                                .map(|h| h.join(".local/bin/adapsis"))
                                .filter(|p| p.parent().map(|d| d.exists()).unwrap_or(false));
                            if let Some(install) = install_path {
                                match std::fs::copy(wb, &install) {
                                    Ok(_) => eprintln!("[opencode] installed binary to {}", install.display()),
                                    Err(e) => eprintln!("[opencode] failed to install binary: {e}"),
                                }
                            }
                        }
                        let exe = std::env::current_exe().unwrap_or_else(|_| {
                            std::env::args().next()
                                .map(std::path::PathBuf::from)
                                .unwrap_or_default()
                        });
                        let args: Vec<String> = std::env::args().collect();
                        eprintln!("[opencode] restarting with binary: {}", exe.display());
                        let err = exec::execvp(&exe, &args);
                        let msg = format!("RESTART FAILED: exec::execvp returned: {err}. The new binary is built but NOT running. Manual restart required.");
                        eprintln!("[opencode] {msg}");
                        op_result.error(&msg);
                    }
                }
                Err(e) => {
                    // Extract line number from error and show surrounding code context
                    let err_str = format!("{e}");
                    let context = if let Some(rest) = err_str.strip_prefix("line ") {
                        if let Some(n) = rest.split(':').next().and_then(|s| s.trim().parse::<usize>().ok()) {
                            let lines: Vec<&str> = code.lines().collect();
                            let start = n.saturating_sub(2);
                            let end = (n + 1).min(lines.len());
                            let ctx: Vec<String> = (start..end).map(|i| {
                                let marker = if i + 1 == n { ">>>" } else { "   " };
                                format!("{marker} {}: {}", i + 1, lines.get(i).unwrap_or(&""))
                            }).collect();
                            format!("\nContext:\n{}", ctx.join("\n"))
                        } else { String::new() }
                    } else { String::new() };
                    let msg = format!("Parse error: {e}{context}");
                    op_result.error(&msg);
                    let _ = tx.send(serde_json::json!({"type": "result", "message": msg, "success": false})).await;
                }
            }

            // Write session mutations back to tiers after each iteration
            config_clone.write_back_working_set(&session).await;

            // Build detailed feedback with ALL results so the AI can see them
            // Also re-run queries and check inbox
            {
                if let Ok(ops) = crate::parser::parse(&code) {
                    for op in &ops {
                        if let crate::parser::Operation::Query(query) = op {
                            let response = if query.trim() == "?tasks" {
                                format_tasks(&config_clone.task_registry)
                            } else if let Some(tid) = parse_inspect_task_query(query.trim()) {
                                format_inspect_task(&config_clone.task_registry, &config_clone.snapshot_registry, tid)
                            } else if query.trim() == "?library" {
                                crate::library::query_library(&session.program, session.meta.library_state.as_ref())
                            } else if query.trim() == "?inbox" || query.trim().starts_with("?inbox") {
                                let msgs = crate::session::peek_messages(&session.meta, "main");
                                if msgs.is_empty() { "No messages.".to_string() }
                                else { msgs.iter().map(|m| format!("[{}] from {}: {}", m.timestamp, m.from, m.content)).collect::<Vec<_>>().join("\n") }
                            } else {
                                let table = crate::typeck::build_symbol_table(&session.program);
                                crate::typeck::handle_query(&session.program, &table, query, &session.runtime.http_routes)
                            };
                            op_result.info(format!("{query}:\n{response}"));
                        }
                    }
                }
                // Check for messages from agents addressed to main
                let inbox = crate::session::drain_messages(&mut session.meta, "main");
                if !inbox.is_empty() {
                    let inbox_text = inbox.iter()
                        .map(|m| format!("[from {}] {}", m.from, m.content))
                        .collect::<Vec<_>>().join("\n");
                    op_result.info(format!("Agent messages:\n{inbox_text}"));
                    let _ = tx.send(serde_json::json!({"type": "result", "message": format!("Messages from agents: {}", inbox.len()), "success": true})).await;
                }

                // Note untested functions in feedback so the AI knows what needs tests
                if session.program.require_modules {
                    let all_fns: Vec<String> = {
                        let mut fns = Vec::new();
                        for m in &session.program.modules {
                            for f in &m.functions {
                                let qname = format!("{}.{}", m.name, f.name);
                                if f.body.len() > 2 && !crate::session::is_function_tested(&session.program, &qname) {
                                    fns.push(qname);
                                }
                            }
                        }
                        fns
                    };
                    if !all_fns.is_empty() {
                        op_result.info(format!(
                            "Untested functions (blocked from !eval): {}",
                            all_fns.join(", ")
                        ));
                    }
                }
            }

            // Build plan status summary for feedback
            let plan_summary = {
                let meta = config_clone.meta.lock().unwrap();
                let in_progress: Vec<_> = meta.plan.iter().filter(|s| matches!(s.status, crate::session::PlanStatus::InProgress)).collect();
                let pending: Vec<_> = meta.plan.iter().filter(|s| matches!(s.status, crate::session::PlanStatus::Pending)).collect();
                let failed: Vec<_> = meta.plan.iter().filter(|s| matches!(s.status, crate::session::PlanStatus::Failed)).collect();
                let total = meta.plan.len();
                let done = meta.plan.iter().filter(|s| matches!(s.status, crate::session::PlanStatus::Done)).count();
                if total == 0 {
                    "No plan set. Create one with !plan set.".to_string()
                } else if pending.is_empty() && in_progress.is_empty() && failed.is_empty() {
                    format!("All {total} plan steps completed. Verify everything works, then !done.")
                } else {
                    let mut msg = format!("Plan: {done}/{total} done.");
                    if !failed.is_empty() {
                        msg.push_str(&format!("\n  {} failed.", failed.len()));
                    }
                    if !in_progress.is_empty() {
                        for s in &in_progress {
                            msg.push_str(&format!("\n  Current: {}", s.description));
                        }
                        for s in pending.iter().take(1) {
                            msg.push_str(&format!("\n  Next: {}", s.description));
                        }
                    } else {
                        if let Some(first) = pending.first() {
                            msg.push_str(&format!("\n  Current: {}", first.description));
                        }
                        if let Some(next) = pending.get(1) {
                            msg.push_str(&format!("\n  Next: {}", next.description));
                        }
                    }
                    msg
                }
            };

            let feedback_details = &op_result.feedback;
            let has_errors = op_result.has_errors;
            let accepted_done = op_result.accepted_done;

            if has_errors {
                let errors: Vec<&str> = feedback_details.iter()
                    .filter(|d| d.starts_with("ERROR:") || d.starts_with("FAIL:") || d.contains("[FAILED]"))
                    .map(|s| s.as_str()).collect();
                let feedback = format!(
                    "Results:\n{}\n\n{}\n\nFix the errors and continue.",
                    feedback_details.join("\n"),
                    plan_summary
                );
                tx.log("feedback", &format!("Errors found ({} issues), retrying...\n{feedback}", errors.len())).await;
                log_training_data(&config_clone.training_log, &config_clone.llm_model, &last_context, &output.thinking, &code, feedback_details, true, op_result.tests_passed, op_result.tests_failed).await;
                last_context = feedback.clone();
                messages.push(crate::llm::ChatMessage::user(feedback));
            } else {
                let results_section = if feedback_details.is_empty() {
                    String::new()
                } else {
                    format!("Results:\n{}\n\n", feedback_details.join("\n"))
                };
                let feedback = format!("{}{}", results_section, plan_summary);
                tx.log("feedback", &feedback).await;
                log_training_data(&config_clone.training_log, &config_clone.llm_model, &last_context, &output.thinking, &code, feedback_details, true, op_result.tests_passed, op_result.tests_failed).await;
                last_context = feedback.clone();
                messages.push(crate::llm::ChatMessage::user(feedback));
                if accepted_done { break; }
            }
        }

        let _ = tx.send(serde_json::json!({"type": "end"})).await;
    });

    // Convert channel to SSE stream
    let stream = async_stream::stream! {
        while let Some(event) = rx.recv().await {
            let data = serde_json::to_string(&event).unwrap_or_default();
            yield Ok(Event::default().data(data));
        }
    };

    axum::response::sse::Sse::new(stream).keep_alive(KeepAlive::default())
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
                        let program_mut_clone = program_mut.clone();
                        let sender = sender.clone();
                        let runtime_for_blocking = config.runtime.clone();
                            let meta_for_blocking = config.meta.clone();
                        let eval_result = tokio::task::spawn_blocking(move || {
                            crate::eval::set_shared_runtime(Some(runtime_for_blocking));
                                eval::set_shared_meta(Some(meta_for_blocking));
                            crate::eval::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                            crate::eval::set_shared_program_mut(Some(program_mut_clone));
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
    crate::eval::set_shared_runtime(Some(config.runtime.clone()));
    crate::eval::set_shared_meta(Some(config.meta.clone()));
    crate::eval::set_shared_event_broadcast(Some(config.event_broadcast.clone()));
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

    let runtime_for_blocking = config.runtime.clone();
    let meta_for_blocking = config.meta.clone();
    let event_broadcast = config.event_broadcast.clone();
    let io_sender_for_blocking = config.io_sender.clone();
    let program_mut = crate::eval::make_shared_program_mut(&program);
    let program_mut_clone = program_mut.clone();

    eprintln!("[webhook] {method_str} {path} -> {handler_fn}({} bytes)", body_str.len());

    let handler_fn_for_log = handler_fn.clone();
    // Evaluate the handler function with the body as a String argument
    let eval_result = tokio::task::spawn_blocking(move || {
        crate::eval::set_shared_runtime(Some(runtime_for_blocking));
        eval::set_shared_meta(Some(meta_for_blocking));
        eval::set_shared_event_broadcast(Some(event_broadcast));
        crate::eval::set_shared_program(Some(std::sync::Arc::new(program.clone())));
        crate::eval::set_shared_program_mut(Some(program_mut_clone));
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
/// POST /api/inject — queue a message for the autonomous loop (no parallel stream)
async fn inject_message(
    State(config): State<AppConfig>,
    Json(req): Json<AskRequest>,
) -> Json<serde_json::Value> {
    config.message_queue.lock().await.push(req.message.clone());
    eprintln!("[inject] queued: {}...", req.message.chars().take(80).collect::<String>());
    Json(serde_json::json!({"status": "queued", "message": req.message}))
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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;

    /// Helper: build a minimal AppConfig for testing multi-session endpoints.
    fn test_config() -> AppConfig {
        let (trigger_tx, _trigger_rx) = tokio::sync::mpsc::channel::<String>(1);
        AppConfig {
            program: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::ast::Program::default(),
            )),
            meta: std::sync::Arc::new(std::sync::Mutex::new(
                crate::session::SessionMeta::new(),
            )),
            llm_url: String::new(),
            llm_model: String::new(),
            llm_api_key: None,
            project_dir: ".".to_string(),
            io_sender: None,
            self_trigger: trigger_tx,
            task_registry: None,
            snapshot_registry: None,
            log_file: None,
            training_log: None,
            jit_cache: crate::eval::new_jit_cache(),
            event_broadcast: tokio::sync::broadcast::channel(16).0,
            opencode_git_dir: ".".to_string(),
            opencode_lock: std::sync::Arc::new(tokio::sync::Mutex::new(())),
            message_queue: std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new())),
            max_iterations: 1,
            runtime: std::sync::Arc::new(std::sync::RwLock::new(
                crate::session::RuntimeState::default(),
            )),
            sessions: std::sync::Arc::new(tokio::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
        }
    }

    async fn recv_event(rx: &mut tokio::sync::broadcast::Receiver<String>) -> serde_json::Value {
        let raw = rx.recv().await.unwrap();
        let value: serde_json::Value = serde_json::from_str(&raw).unwrap_or_else(|_| serde_json::json!({"raw": raw.clone()}));
        if let Some(encoded) = value.as_str() {
            serde_json::from_str(encoded).unwrap_or(value)
        } else {
            value
        }
    }

    #[test]
    fn collect_opencode_tasks_keeps_all_in_order() {
        let ops = crate::parser::parse("!opencode First task\n!opencode Second task\n+fn hi ()->Int\n  +return 1\n").unwrap();
        let tasks = collect_opencode_tasks(&ops);
        assert_eq!(tasks, vec!["First task".to_string(), "Second task".to_string()]);
    }

    #[test]
    fn collect_opencode_tasks_ignores_non_opencode_ops() {
        let ops = crate::parser::parse("+fn hi ()->Int\n  +return 1\n!eval hi\n").unwrap();
        let tasks = collect_opencode_tasks(&ops);
        assert!(tasks.is_empty());
    }

    #[tokio::test]
    async fn events_stream_sets_sse_headers() {
        let config = test_config();
        let response = events_stream(State(config)).await.into_response();
        assert_eq!(response.headers()[axum::http::header::CONTENT_TYPE], "text/event-stream");
        assert_eq!(response.headers()[axum::http::header::CACHE_CONTROL], "no-cache");
        assert_eq!(response.headers()[axum::http::header::CONNECTION], "keep-alive");
    }

    // ═════════════════════════════════════════════════════════════════════
    // GET /api/sessions — list sessions
    // ═════════════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn list_sessions_includes_main() {
        let config = test_config();
        let Json(result) = list_sessions(State(config)).await;
        let ids = result.as_array().unwrap();
        assert!(
            ids.iter().any(|v| v.as_str() == Some("main")),
            "list should always include 'main'"
        );
    }

    #[tokio::test]
    async fn list_sessions_includes_created_sessions() {
        let config = test_config();
        // Create a session directly in the map
        config
            .sessions
            .lock()
            .await
            .insert(
                "test-session".to_string(),
                std::sync::Arc::new(tokio::sync::Mutex::new(
                    crate::ast::Program::default(),
                )),
            );

        let Json(result) = list_sessions(State(config)).await;
        let ids: Vec<&str> = result
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(ids.contains(&"main"));
        assert!(ids.contains(&"test-session"));
    }

    // ═════════════════════════════════════════════════════════════════════
    // POST /api/sessions — create session
    // ═════════════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn create_session_happy_path() {
        let config = test_config();
        let (status, Json(body)) = create_session(
            State(config.clone()),
            Json(CreateSessionRequest {
                session_id: "my-session".to_string(),
            }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::CREATED);
        assert_eq!(body["session_id"], "my-session");
        assert_eq!(body["status"], "created");

        // Verify it's in the map
        let sessions = config.sessions.lock().await;
        assert!(sessions.contains_key("my-session"));
    }

    #[tokio::test]
    async fn create_session_empty_id_rejected() {
        let config = test_config();
        let (status, Json(body)) = create_session(
            State(config),
            Json(CreateSessionRequest {
                session_id: "  ".to_string(),
            }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert!(body["error"].as_str().unwrap().contains("must not be empty"));
    }

    #[tokio::test]
    async fn create_session_main_reserved() {
        let config = test_config();
        let (status, Json(body)) = create_session(
            State(config),
            Json(CreateSessionRequest {
                session_id: "main".to_string(),
            }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::CONFLICT);
        assert!(body["error"].as_str().unwrap().contains("reserved"));
    }

    #[tokio::test]
    async fn create_session_duplicate_rejected() {
        let config = test_config();
        // Create first
        create_session(
            State(config.clone()),
            Json(CreateSessionRequest {
                session_id: "dup".to_string(),
            }),
        )
        .await;
        // Create duplicate
        let (status, Json(body)) = create_session(
            State(config),
            Json(CreateSessionRequest {
                session_id: "dup".to_string(),
            }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::CONFLICT);
        assert!(body["error"].as_str().unwrap().contains("already exists"));
    }

    // ═════════════════════════════════════════════════════════════════════
    // DELETE /api/sessions/:id — delete session
    // ═════════════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn delete_session_happy_path() {
        let config = test_config();
        // Create then delete
        create_session(
            State(config.clone()),
            Json(CreateSessionRequest {
                session_id: "to-delete".to_string(),
            }),
        )
        .await;
        let (status, Json(body)) = delete_session(
            State(config.clone()),
            axum::extract::Path("to-delete".to_string()),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(body["status"], "deleted");

        // Verify removed
        let sessions = config.sessions.lock().await;
        assert!(!sessions.contains_key("to-delete"));
    }

    #[tokio::test]
    async fn delete_session_main_rejected() {
        let config = test_config();
        let (status, Json(body)) = delete_session(
            State(config),
            axum::extract::Path("main".to_string()),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert!(body["error"].as_str().unwrap().contains("cannot delete"));
    }

    #[tokio::test]
    async fn delete_session_not_found() {
        let config = test_config();
        let (status, Json(body)) = delete_session(
            State(config),
            axum::extract::Path("nonexistent".to_string()),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::NOT_FOUND);
        assert!(body["error"].as_str().unwrap().contains("not found"));
    }

    // ═════════════════════════════════════════════════════════════════════
    // POST /api/sessions/:id/eval — eval in session
    // ═════════════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn session_eval_not_found() {
        let config = test_config();
        let Json(response) = session_eval(
            State(config),
            axum::extract::Path("nonexistent".to_string()),
            Json(EvalRequest {
                function: "foo".to_string(),
                input: String::new(),
                expression: None,
            }),
        )
        .await;
        assert!(!response.success);
        assert!(response.result.contains("not found"));
    }

    #[tokio::test]
    async fn session_eval_inline_expression() {
        let config = test_config();
        // Create a session
        create_session(
            State(config.clone()),
            Json(CreateSessionRequest {
                session_id: "eval-test".to_string(),
            }),
        )
        .await;

        // Eval an inline expression (no functions needed)
        let Json(response) = session_eval(
            State(config),
            axum::extract::Path("eval-test".to_string()),
            Json(EvalRequest {
                function: String::new(),
                input: String::new(),
                expression: Some("1 + 2".to_string()),
            }),
        )
        .await;
        assert!(response.success, "eval should succeed: {}", response.result);
        assert_eq!(response.result, "3");
    }

    #[tokio::test]
    async fn session_eval_function_in_session() {
        let config = test_config();
        // Create session
        create_session(
            State(config.clone()),
            Json(CreateSessionRequest {
                session_id: "fn-test".to_string(),
            }),
        )
        .await;

        // Add a function via session_mutate
        session_mutate(
            State(config.clone()),
            axum::extract::Path("fn-test".to_string()),
            Json(MutateRequest {
                source: "+fn double (x:Int)->Int\n  +return x * 2\n".to_string(),
            }),
        )
        .await;

        // Eval the function
        let Json(response) = session_eval(
            State(config),
            axum::extract::Path("fn-test".to_string()),
            Json(EvalRequest {
                function: "double".to_string(),
                input: "5".to_string(),
                expression: None,
            }),
        )
        .await;
        assert!(response.success, "eval should succeed: {}", response.result);
        assert_eq!(response.result, "10");
    }

    // ═════════════════════════════════════════════════════════════════════
    // POST /api/sessions/:id/mutate — mutate in session
    // ═════════════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn session_mutate_happy_path() {
        let config = test_config();
        create_session(
            State(config.clone()),
            Json(CreateSessionRequest {
                session_id: "mut-test".to_string(),
            }),
        )
        .await;

        let Json(response) = session_mutate(
            State(config.clone()),
            axum::extract::Path("mut-test".to_string()),
            Json(MutateRequest {
                source: "+fn greet ()->String\n  +return \"hello\"\n".to_string(),
            }),
        )
        .await;
        assert!(
            response.results.iter().all(|r| r.success),
            "mutate should succeed: {:?}",
            response.results
        );

        // Verify the function exists in the session
        let sessions = config.sessions.lock().await;
        let program = sessions.get("mut-test").unwrap().lock().await;
        assert!(program.get_function("greet").is_some());
    }

    #[tokio::test]
    async fn session_mutate_not_found() {
        let config = test_config();
        let Json(response) = session_mutate(
            State(config),
            axum::extract::Path("nonexistent".to_string()),
            Json(MutateRequest {
                source: "+fn greet ()->String\n  +return \"hello\"\n".to_string(),
            }),
        )
        .await;
        assert!(response.results.iter().any(|r| !r.success));
        assert!(response.results[0].message.contains("not found"));
    }

    // ═════════════════════════════════════════════════════════════════════
    // Isolation: sessions have independent Programs
    // ═════════════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn sessions_are_isolated() {
        let config = test_config();

        // Create two sessions
        create_session(
            State(config.clone()),
            Json(CreateSessionRequest {
                session_id: "session-a".to_string(),
            }),
        )
        .await;
        create_session(
            State(config.clone()),
            Json(CreateSessionRequest {
                session_id: "session-b".to_string(),
            }),
        )
        .await;

        // Add a function to session-a only
        session_mutate(
            State(config.clone()),
            axum::extract::Path("session-a".to_string()),
            Json(MutateRequest {
                source: "+fn only_in_a ()->Int\n  +return 42\n".to_string(),
            }),
        )
        .await;

        // session-a should have the function
        let Json(resp_a) = session_eval(
            State(config.clone()),
            axum::extract::Path("session-a".to_string()),
            Json(EvalRequest {
                function: "only_in_a".to_string(),
                input: String::new(),
                expression: None,
            }),
        )
        .await;
        assert!(resp_a.success, "session-a should have the function");

        // session-b should NOT have it
        let Json(resp_b) = session_eval(
            State(config),
            axum::extract::Path("session-b".to_string()),
            Json(EvalRequest {
                function: "only_in_a".to_string(),
                input: String::new(),
                expression: None,
            }),
        )
        .await;
        assert!(!resp_b.success, "session-b should not have session-a's function");
    }

    #[tokio::test]
    async fn sync_async_side_effects_preserves_live_roadmap_and_runtime_state() {
        let config = test_config();
        let mut session = config.snapshot_working_set().await;

        session.meta.plan.push(crate::session::PlanStep {
            description: "keep local plan".to_string(),
            status: crate::session::PlanStatus::Pending,
        });
        config.write_back_working_set(&session).await;

        {
            let mut meta = config.meta.lock().unwrap();
            crate::session::roadmap_add(&mut meta.roadmap, "#30 Sync GitHub issues");
            meta.io_mocks.push(crate::session::IoMock {
                operation: "http_get".to_string(),
                patterns: vec!["api.github.com".to_string()],
                response: "[]".to_string(),
            });
            crate::session::send_agent_message(&mut meta, "main", "worker", "hello");
        }
        {
            let mut runtime = config.runtime.write().unwrap();
            runtime.http_routes.push(crate::ast::HttpRoute {
                method: "GET".to_string(),
                path: "/health".to_string(),
                handler_fn: "health_check".to_string(),
            });
        }

        config.sync_async_side_effects_into(&mut session);
        config.write_back_working_set(&session).await;

        let meta = config.meta.lock().unwrap();
        assert_eq!(meta.plan.len(), 1, "local plan should survive sync");
        assert_eq!(meta.roadmap.len(), 1, "async roadmap additions should survive write-back");
        assert_eq!(meta.io_mocks.len(), 1, "async mock mutations should survive write-back");
        assert_eq!(meta.agent_mailbox.get("worker").map(Vec::len), Some(1));
        drop(meta);

        let runtime = config.runtime.read().unwrap();
        assert_eq!(runtime.http_routes.len(), 1, "async runtime mutations should survive write-back");
    }

    #[tokio::test]
    async fn mutate_broadcasts_sse_event() {
        let config = test_config();
        let mut rx = config.event_broadcast.subscribe();

        let Json(response) = mutate(
            State(config),
            Json(MutateRequest {
                source: "+fn ping ()->String\n  +return \"pong\"\n+end".to_string(),
            }),
        ).await;

        assert!(response.results.iter().all(|r| r.success));
        let event = recv_event(&mut rx).await;
        assert_eq!(event["type"], "mutation");
        assert_eq!(event["revision"], 1);
        assert_eq!(event["summary"], response.results[0].message);
    }

    #[tokio::test]
    async fn mutate_repeated_errors_add_hint() {
        let config = test_config();
        let bad_source = "+fn bad ()->String\n  +let user:String = {first: \"a\" second: \"b\"}\n  +return user\n+end";

        let Json(first) = mutate(
            State(config.clone()),
            Json(MutateRequest { source: bad_source.to_string() }),
        ).await;
        assert!(!first.results[0].success);
        assert!(!first.results[0].message.contains("Suggestion:"));

        let Json(response) = mutate(
            State(config),
            Json(MutateRequest { source: bad_source.to_string() }),
        ).await;

        assert!(!response.results[0].success);
        assert!(response.results[0].message.contains("Suggestion:"), "got: {}", response.results[0].message);
        assert!(response.results[0].message.contains("missing commas between fields"), "got: {}", response.results[0].message);
    }

    #[tokio::test]
    async fn eval_broadcasts_sse_event() {
        let config = test_config();
        let mut rx = config.event_broadcast.subscribe();
        let _ = mutate(
            State(config.clone()),
            Json(MutateRequest {
                source: "+fn forty_two ()->Int\n  +return 42\n+end".to_string(),
            }),
        ).await;
        let _ = recv_event(&mut rx).await;

        let Json(response) = eval_fn(
            State(config),
            Json(EvalRequest {
                function: "forty_two".to_string(),
                input: "".to_string(),
                expression: None,
            }),
        ).await;

        assert!(response.success);
        let event = recv_event(&mut rx).await;
        assert_eq!(event["type"], "eval");
        assert_eq!(event["expression"], "forty_two");
        assert_eq!(event["result"], "42");
    }

    #[tokio::test]
    async fn eval_error_broadcasts_sse_event() {
        let config = test_config();
        let mut rx = config.event_broadcast.subscribe();

        let Json(response) = eval_fn(
            State(config),
            Json(EvalRequest {
                function: String::new(),
                input: String::new(),
                expression: Some(String::new()),
            }),
        ).await;

        assert!(!response.success);
        let event = recv_event(&mut rx).await;
        assert_eq!(event["type"], "eval");
        assert_eq!(event["expression"], "");
        assert_eq!(event["result"], "empty expression");
    }

    #[tokio::test]
    async fn test_endpoint_broadcasts_sse_event() {
        let config = test_config();
        let mut rx = config.event_broadcast.subscribe();
        let _ = mutate(
            State(config.clone()),
            Json(MutateRequest {
                source: "+fn one ()->Int\n  +return 1\n+end".to_string(),
            }),
        ).await;
        let _ = recv_event(&mut rx).await;

        let Json(response) = test_fn(
            State(config),
            Json(TestRequest {
                source: "!test one\n  +with -> expect 1".to_string(),
            }),
        ).await;

        assert_eq!(response.passed, 1);
        let event = recv_event(&mut rx).await;
        assert_eq!(event["type"], "test");
        assert_eq!(event["function"], "one");
        assert_eq!(event["passed"], 1);
        assert_eq!(event["failed"], 0);
    }

    #[tokio::test]
    async fn ui_page_contains_event_source_client() {
        let Html(page) = ui_page().await;
        assert!(page.contains("AdapsisOS Dashboard"));
        assert!(page.contains("textarea"));
        assert!(page.contains("Eval"));
        assert!(page.contains("Apply"));
        assert!(page.contains("/api/events"));
        assert!(page.contains("EventSource"));
        assert!(page.contains("const formatPayload = (payload) =>"));
        assert!(page.contains("(empty event)"));
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

/// Handle a `llm_takeover` IO request: call LLM with per-context conversation
/// history, return text reply immediately, execute any code in background.
pub async fn handle_llm_takeover(
    context: String,
    message: String,
    reply_fn: Option<String>,
    reply_arg: Option<String>,
    meta: crate::session::SharedMeta,
    program: std::sync::Arc<tokio::sync::RwLock<crate::ast::Program>>,
    runtime: crate::session::SharedRuntime,
    llm_url: &str,
    llm_model: &str,
    llm_key: Option<String>,
    io_sender: tokio::sync::mpsc::Sender<crate::coroutine::IoRequest>,
    task_registry: crate::coroutine::TaskRegistry,
    snap_registry: crate::coroutine::TaskSnapshotRegistry,
) -> anyhow::Result<String> {
    let llm = crate::llm::LlmClient::new_with_model_and_key(llm_url, llm_model, llm_key.clone());

    // Get or create conversation, update callback info, build messages
    let program_summary = {
        let prog = program.read().await;
        crate::validator::program_summary_compact(&prog)
    };
    let messages = {
        let mut meta_guard = meta.lock().unwrap();
        let conv = meta_guard.conversations.get_or_create(&context);

        // Update callback info if provided
        if reply_fn.is_some() {
            conv.reply_fn = reply_fn.clone();
        }
        if reply_arg.is_some() {
            conv.reply_arg = reply_arg.clone();
        }

        // Add system prompt if this is a new conversation
        if conv.messages.is_empty() {
            let system = conv.system_prompt.clone().unwrap_or_else(|| {
                format!(
                    "{}\n\n{}\n\nCurrent program state:\n{}\n\n\
                     You are in conversation context '{context}'. Respond naturally. \
                     If you need to do work (modify code, create modules, run tasks), include the \
                     Adapsis commands in your response. For long-running tasks, use `!agent`. \
                     IMPORTANT: The user only sees text BEFORE the first <code> block. \
                     Text after <code> blocks is discarded. Put your response to the user first, \
                     then the code. Do not narrate what the code does after the <code> block — \
                     the user won't see it. For !agent tasks, just say what you'll do, then the \
                     <code> block. The agent completion result will be delivered separately.",
                    crate::prompt::system_prompt(),
                    crate::builtins::format_for_prompt(),
                    program_summary,
                )
            });
            conv.push_system(system);
        }

        // Add user message
        conv.push_user(&message);

        // Build LLM messages from conversation history
        conv.messages.iter().map(|m| match m.role.as_str() {
            "system" => crate::llm::ChatMessage::system(m.content.clone()),
            "assistant" => crate::llm::ChatMessage::assistant(&m.content),
            _ => crate::llm::ChatMessage::user(m.content.clone()),
        }).collect::<Vec<_>>()
    }; // meta_guard dropped here

    eprintln!("[llm_takeover:{context}] calling LLM with {} messages", messages.len());

    // Build a temporary AppConfig so we can reuse execute_code()
    let (self_trigger_tx, _self_trigger_rx) = tokio::sync::mpsc::channel::<String>(1);
    let tmp_config = AppConfig {
        program: program.clone(),
        meta: meta.clone(),
        llm_url: llm_url.to_string(),
        llm_model: llm_model.to_string(),
        llm_api_key: llm_key.clone(),
        project_dir: ".".to_string(),
        io_sender: Some(io_sender.clone()),
        self_trigger: self_trigger_tx,
        task_registry: Some(task_registry.clone()),
        snapshot_registry: Some(snap_registry.clone()),
        log_file: None,
        training_log: None,
        jit_cache: crate::eval::new_jit_cache(),
        event_broadcast: tokio::sync::broadcast::channel(16).0,
        opencode_git_dir: ".".to_string(),
        opencode_lock: std::sync::Arc::new(tokio::sync::Mutex::new(())),
        message_queue: std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new())),
        max_iterations: 10,
        runtime: runtime.clone(),
        sessions: std::sync::Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new())),
    };

    // Build agent completion callback from conversation's reply info
    let agent_cb = {
        let meta_guard = meta.lock().unwrap();
        meta_guard.conversations.get(&context).and_then(|conv| {
            match (&conv.reply_fn, &conv.reply_arg) {
                (Some(rf), Some(ra)) => Some(AgentCompletionCallback {
                    context: context.clone(),
                    reply_fn: rf.clone(),
                    reply_arg: ra.clone(),
                    llm_url: llm_url.to_string(),
                    llm_model: llm_model.to_string(),
                    llm_key: llm_key.clone(),
                }),
                _ => None,
            }
        })
    };

    // Iterative loop: call LLM → execute_code → feed results back (max 10 iterations)
    let mut llm_messages = messages;
    let mut reply_text = String::new();

    for iteration in 0..10 {
        eprintln!("[llm_takeover:{context}] iteration {}/{}", iteration + 1, 10);

        let output = match llm.generate(llm_messages.clone()).await {
            Ok(o) => o,
            Err(e) => {
                eprintln!("[llm_takeover:{context}] LLM error: {e}");
                break;
            }
        };

        // Store assistant response in conversation
        {
            let mut meta_guard = meta.lock().unwrap();
            if let Some(conv) = meta_guard.conversations.get_mut(&context) {
                conv.push_assistant(&output.text);
            }
        }

        llm_messages.push(crate::llm::ChatMessage::assistant(&output.text));

        // Extract text reply (prose without code)
        let prose = {
            let mut clean = output.text.clone();
            while let Some(s) = clean.find("<think>") {
                if let Some(e) = clean[s..].find("</think>") {
                    clean.replace_range(s..s + e + 8, "");
                } else { break; }
            }
            // Truncate at first <code> block - everything after is narration about code execution
            if let Some(s) = clean.find("<code>") {
                clean.truncate(s);
            }
            clean.trim().to_string()
        };

        if !prose.is_empty() {
            if !reply_text.is_empty() { reply_text.push_str("\n\n"); }
            reply_text.push_str(&prose);
        }

        eprintln!("[llm_takeover:{context}] reply: {}...", prose.chars().take(80).collect::<String>());

        let code = output.code.trim().to_string();

        // If !done or no code, stop iterating
        if code == "!done" || code.is_empty() {
            eprintln!("[llm_takeover:{context}] done at iteration {}", iteration + 1);
            break;
        }

        // Check for !agent — spawn it and return prose (agent delivers callback later)
        if let Ok(ops) = crate::parser::parse(&code) {
            let has_agent = ops.iter().any(|op| matches!(op, crate::parser::Operation::Agent { .. }));
            if has_agent {
                eprintln!("[llm_takeover:{context}] !agent detected, spawning via execute_code and breaking");
                let mut session = WorkingSet {
                    program: program.read().await.clone(),
                    runtime: runtime.read().unwrap().clone(),
                    meta: meta.lock().unwrap().clone(),
                    sandbox: None,
                };
                execute_code(&code, &tmp_config, &mut session, agent_cb.clone()).await;
                // Write back mutations
                *program.write().await = session.program;
                *runtime.write().unwrap() = session.runtime;
                *meta.lock().unwrap() = session.meta;
                break;
            }
        }

        // Execute code inline
        let mut session = WorkingSet {
            program: program.read().await.clone(),
            runtime: runtime.read().unwrap().clone(),
            meta: meta.lock().unwrap().clone(),
            sandbox: None,
        };
        let exec_result = execute_code(&code, &tmp_config, &mut session, agent_cb.clone()).await;

        // Write back mutations to shared state
        *program.write().await = session.program;
        *runtime.write().unwrap() = session.runtime;
        *meta.lock().unwrap() = session.meta;

        // Build feedback and append to conversation
        let feedback: Vec<String> = exec_result.mutation_results.iter()
            .map(|r| format!("{}: {}", if r.success { "OK" } else { "ERROR" }, r.message))
            .chain(exec_result.test_results.iter()
                .map(|t| format!("{}: {}", if t.pass { "PASS" } else { "FAIL" }, t.message)))
            .collect();

        eprintln!("[llm_takeover:{context}] code results: {}", feedback.join("; "));

        let feedback_msg = if exec_result.has_errors {
            format!("Errors:\n{}\n\nFix and continue.", feedback.join("\n"))
        } else {
            format!("Results:\n{}\n\nContinue or !done.", feedback.join("\n"))
        };

        {
            let mut meta_guard = meta.lock().unwrap();
            if let Some(conv) = meta_guard.conversations.get_mut(&context) {
                conv.push_user(feedback_msg.clone());
            }
        }
        llm_messages.push(crate::llm::ChatMessage::user(feedback_msg));

        // Handle opencode restart if needed
        if exec_result.needs_opencode_restart {
            eprintln!("[llm_takeover:{context}] opencode restart triggered");
            let exe = std::env::current_exe().unwrap_or_default();
            let args: Vec<String> = std::env::args().collect();
            let _ = exec::execvp(&exe, &args);
        }
    }

    Ok(reply_text)
}
