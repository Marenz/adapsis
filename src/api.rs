// test comment
//! HTTP API for AdapsisOS — programmatic access to the session.
//!
//! Endpoints:
//!   POST /api/mutate     — apply Forge source code
//!   POST /api/eval       — evaluate a function call
//!   POST /api/test       — run tests for a function
//!   GET  /api/query      — semantic query (?symbols, ?callers, etc.)
//!   GET  /api/status     — program state + revision info
//!   GET  /api/history    — mutation log + working history
//!   POST /api/rewind     — rewind to a revision
//!   POST /api/ask        — send task to LLM, apply generated code

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::eval;
use crate::parser;
use crate::session::Session;
use crate::typeck;
use crate::validator;

pub type SharedSession = Arc<Mutex<Session>>;

/// Thread-safe session manager: maps session IDs to independent Program instances.
/// The "main" session uses the existing `session` field in AppConfig; additional
/// sessions are stored here with isolated Program state.
pub type SessionManager = Arc<Mutex<std::collections::HashMap<String, Arc<tokio::sync::Mutex<crate::ast::Program>>>>>;

/// Extended state for the API, including LLM and OpenCode configuration.
#[derive(Clone)]
pub struct AppConfig {
    pub session: SharedSession,
    /// Tier 1: Program AST — read-heavy, use read() for queries, write() briefly for mutations
    pub program: std::sync::Arc<tokio::sync::RwLock<crate::ast::Program>>,
    /// Tier 3: Session metadata — chat history, plan, roadmap, mocks, mutation log.
    /// Brief locks only; never hold during LLM calls or IO.
    pub meta: std::sync::Arc<tokio::sync::Mutex<crate::session::SessionMeta>>,
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
    pub event_broadcast: tokio::sync::broadcast::Sender<serde_json::Value>,
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
    let mut session = config.session.lock().await;
    match session.apply_async(&req.source, config.io_sender.as_ref()).await {
        Ok(results) => {
            // Sync shared vars to the Arc<RwLock> runtime
            if let Ok(mut rt) = config.runtime.write() {
                rt.shared_vars = session.runtime.shared_vars.clone();
            }
            Json(MutateResponse {
                revision: session.meta.revision,
                results: results
                    .into_iter()
                    .map(|(message, success)| MutationResult { message, success })
                    .collect(),
            })
        }
        Err(e) => Json(MutateResponse {
            revision: session.meta.revision,
            results: vec![MutationResult {
                message: format!("error: {e}"),
                success: false,
            }],
        }),
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
    let mut session = config.session.lock().await;

    // Handle inline expression evaluation (e.g. "1 + 2", "concat(\"a\", \"b\")")
    if let Some(ref expr_str) = req.expression {
        eprintln!("[web:eval] inline: {expr_str}");
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
                match eval::eval_inline_expr(&session.program, &expr) {
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

    eprintln!("[web:eval] {} {}", req.function, req.input);

    // Parse input directly instead of reconstructing "!eval fn input" source
    // which breaks when the input contains unescaped quotes or special chars.
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

    // Block eval of untested functions (>2 statements) in AdapsisOS mode
    if session.program.require_modules {
        if let Some(func) = session.program.get_function(&ev.function_name) {
            if func.body.len() > 2 && !session.is_function_tested(&ev.function_name) {
                return Json(EvalResponse {
                    result: format!("function `{}` has {} statements but no passing tests. Write !test blocks first.", ev.function_name, func.body.len()),
                    success: false,
                    compiled: None,
                });
            }
        }
    }

    let needs_async = session.program.get_function(&ev.function_name)
        .is_some_and(|f| f.effects.iter().any(|e|
            matches!(e, crate::ast::Effect::Io | crate::ast::Effect::Async)));

    if needs_async {
        if let Some(sender) = &config.io_sender {
            let program = session.program.clone();
            let fn_name = ev.function_name.clone();
            let input = ev.input.clone();
            let sender = sender.clone();

            drop(session); // release lock before blocking

            let eval_result = tokio::task::spawn_blocking(move || {
                let func = program.get_function(&fn_name)
                    .ok_or_else(|| anyhow::anyhow!("function not found"))?;
                let handle = crate::coroutine::CoroutineHandle::new(sender);
                let mut env = eval::Env::new();
                env.populate_shared_from_program(&program);
                env.set("__coroutine_handle", eval::Value::CoroutineHandle(handle));
                let input_val = eval::eval_parser_expr_with_program(&input, &program)?;
                eval::bind_input_to_params(&program, func, &input_val, &mut env);
                eval::eval_function_body_pub(&program, &func.body, &mut env)
            }).await;

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
    }

    match eval::eval_compiled_or_interpreted_cached(&session.program, &ev.function_name, &ev.input, Some(&config.jit_cache), session.meta.revision) {
        Ok((result, compiled)) => {
            // re-acquire session for recording (we may have dropped it above for async)
            session.record_eval(&ev.function_name, &req.input, &result);
            Json(EvalResponse {
                result,
                success: true,
                compiled: Some(compiled),
            })
        }
        Err(e) => {
            Json(EvalResponse {
                result: format!("{e}"),
                success: false,
                compiled: None,
            })
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
    let mut session = config.session.lock().await;
    let operations = match parser::parse(&req.source) {
        Ok(ops) => ops,
        Err(e) => {
            return Json(TestResponse {
                passed: 0,
                failed: 1,
                results: vec![TestCaseResult {
                    message: format!("parse error: {e}"),
                    pass: false,
                }],
            });
        }
    };

    let mut passed = 0;
    let mut failed = 0;
    let mut results = Vec::new();

    for op in &operations {
        if let parser::Operation::Test(test) = op {
            // Check if the function under test has async/io effects
            let needs_async = session.program.get_function(&test.function_name)
                .is_some_and(|f| f.effects.iter().any(|e|
                    matches!(e, crate::ast::Effect::Io | crate::ast::Effect::Async)));

            for case in &test.cases {
                let case_result = if needs_async {
                    if let Some(sender) = &config.io_sender {
                        // Run through real coroutine runtime (with mock fallback)
                        let program = session.program.clone();
                        let fn_name = test.function_name.clone();
                        let case = case.clone();
                        let mocks = session.meta.io_mocks.clone();
                        let routes = session.runtime.http_routes.clone();
                        let sender = sender.clone();

                        drop(session); // release lock before blocking

                        let result = eval::eval_test_case_async(
                            &program, &fn_name, &case, &mocks, sender, &routes,
                        ).await;

                        session = config.session.lock().await;
                        result
                    } else {
                        // No IO sender — fall back to mock-only execution
                        eval::eval_test_case_with_mocks(
                            &session.program, &test.function_name, case, &session.meta.io_mocks, &session.runtime.http_routes,
                        )
                    }
                } else {
                    // Sync function — run directly
                    eval::eval_test_case_with_mocks(
                        &session.program, &test.function_name, case, &session.meta.io_mocks, &session.runtime.http_routes,
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
            let details: Vec<String> = results.iter().map(|r| {
                format!("{}: {}", if r.pass { "PASS" } else { "FAIL" }, r.message)
            }).collect();
            session.record_test(&test.function_name, passed, failed, details);
            if failed == 0 && !test.cases.is_empty() {
                session.store_test(&test.function_name, &test.cases);
            }
        }
    }

    Json(TestResponse {
        passed,
        failed,
        results,
    })
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
    let mut session = config.session.lock().await;
    let response = if req.query.trim() == "?inbox" || req.query.trim().starts_with("?inbox") {
        let msgs = session.peek_messages("main");
        if msgs.is_empty() { "No messages.".to_string() }
        else { msgs.iter().map(|m| format!("[{}] from {}: {}", m.timestamp, m.from, m.content)).collect::<Vec<_>>().join("\n") }
    } else if req.query.trim() == "?tasks" {
        format_tasks(&config.task_registry)
    } else if let Some(task_id) = parse_inspect_task_query(req.query.trim()) {
        format_inspect_task(&config.task_registry, &config.snapshot_registry, task_id)
    } else if req.query.trim() == "?library" {
        crate::library::query_library(&session.program, session.meta.library_state.as_ref())
    } else {
        let table = typeck::build_symbol_table(&session.program);
        typeck::handle_query(&session.program, &table, &req.query, &session.runtime.http_routes)
    };
    session.record_query(&req.query, &response);
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
        let meta = config.meta.lock().await;
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
    let meta = config.meta.lock().await;
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
    // Rewind needs all tiers — use the session shim for now (rare operation)
    let mut session = config.session.lock().await;
    match session.rewind_to(req.revision) {
        Ok(()) => {
            // Sync tiers from session after rewind
            *config.program.write().await = session.program.clone();
            {
                let mut meta = config.meta.lock().await;
                *meta = session.meta.clone();
            }
            Json(RewindResponse {
                revision: session.meta.revision,
                success: true,
                message: format!("rewound to revision {}", req.revision),
            })
        }
        Err(e) => Json(RewindResponse {
            revision: session.meta.revision,
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
    }
}

pub async fn program(State(config): State<AppConfig>) -> Json<ProgramResponse> {
    // Tier 1: read program (RwLock read — non-exclusive)
    let prog = config.program.read().await;
    // Tier 3: read meta for revision (brief)
    let revision = config.meta.lock().await.revision;

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
struct EventSender {
    tx: Option<tokio::sync::mpsc::Sender<serde_json::Value>>,
    broadcast: tokio::sync::broadcast::Sender<serde_json::Value>,
}

impl EventSender {
    /// Broadcast-only sender (used by plain `/api/ask`).
    fn broadcast_only(broadcast: tokio::sync::broadcast::Sender<serde_json::Value>) -> Self {
        Self { tx: None, broadcast }
    }

    /// Broadcast + per-request mpsc sender (used by `/api/ask-stream`).
    fn with_mpsc(tx: tokio::sync::mpsc::Sender<serde_json::Value>, broadcast: tokio::sync::broadcast::Sender<serde_json::Value>) -> Self {
        Self { tx: Some(tx), broadcast }
    }

    async fn send(&self, event: serde_json::Value) {
        let _ = self.broadcast.send(event.clone());
        if let Some(tx) = &self.tx {
            let _ = tx.send(event).await;
        } else {
            // Yield so broadcast receivers get a chance to process the event.
            tokio::task::yield_now().await;
        }
    }
}

async fn log_activity(log_file: &Option<std::sync::Arc<tokio::sync::Mutex<tokio::fs::File>>>, event: &str, detail: &str) {
    if let Some(f) = log_file {
        use tokio::io::AsyncWriteExt;
        // Human-readable timestamp
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
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

pub async fn ask(
    State(config): State<AppConfig>,
    Json(req): Json<AskRequest>,
) -> Json<AskResponse> {
    crate::eval::set_shared_runtime(Some(config.runtime.clone()));
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
    let mut messages = {
        let mut session = config.session.lock().await;
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
        let context = format!(
            "Working directory: {}\n{}{}\nUser: {}{}",
            config.project_dir,
            crate::validator::program_summary_compact(&session.program),
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
            if let Some(e) = clean.find("</think>") { clean.replace_range(s..e+8, ""); } else { break; }
        }
        while let Some(s) = clean.find("<code>") {
            if let Some(e) = clean.find("</code>") { clean.replace_range(s..e+7, ""); } else { break; }
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

        // Apply code
        let mut iter_results: Vec<MutationResult> = vec![];
        let mut iter_test_results: Vec<TestCaseResult> = vec![];
        let mut iter_has_errors = false;

        let mut session = config.session.lock().await;

        match crate::parser::parse(&code) {
            Ok(ops) => {
                // Remove duplicates
                let mut fns_removed = false;
                for op in &ops {
                    match op {
                        crate::parser::Operation::Function(f) => {
                            session.program.functions.retain(|existing| existing.name != f.name);
                            fns_removed = true;
                        }
                        crate::parser::Operation::Type(t) => {
                            let name = t.name.clone();
                            session.program.types.retain(|existing| existing.name() != name);
                        }
                        _ => {}
                    }
                }
                if fns_removed {
                    session.program.rebuild_function_index();
                }

                // Handle !undo and !plan before apply
                let has_undo = ops.iter().any(|op| matches!(op, crate::parser::Operation::Undo));
                if has_undo {
                    if session.meta.revision > 0 {
                        let prev = session.meta.revision - 1;
                        match session.rewind_to(prev) {
                            Ok(()) => iter_results.push(MutationResult { message: format!("Undone to rev {prev}"), success: true }),
                            Err(e) => { iter_has_errors = true; iter_results.push(MutationResult { message: format!("Undo: {e}"), success: false }); }
                        }
                    }
                }
                for op in &ops {
                    if let crate::parser::Operation::Plan(action) = op {
                        match action {
                            crate::parser::PlanAction::Set(steps) => {
                                session.meta.plan = steps.iter().map(|s| crate::session::PlanStep {
                                    description: s.clone(),
                                    status: crate::session::PlanStatus::Pending,
                                }).collect();
                                iter_results.push(MutationResult { message: format!("Plan set: {} steps", steps.len()), success: true });
                            }
                            crate::parser::PlanAction::Progress(n) => {
                                let idx = n.saturating_sub(1);
                                if let Some(step) = session.meta.plan.get_mut(idx) {
                                    step.status = crate::session::PlanStatus::Done;
                                    iter_results.push(MutationResult { message: format!("Step {n} done: {}", step.description), success: true });
                                }
                            }
                            crate::parser::PlanAction::Fail(n) => {
                                let idx = n.saturating_sub(1);
                                if let Some(step) = session.meta.plan.get_mut(idx) {
                                    step.status = crate::session::PlanStatus::Failed;
                                    iter_results.push(MutationResult { message: format!("Step {n} failed: {}", step.description), success: true });
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
                                iter_results.push(MutationResult { message: if plan_str.is_empty() { "No plan set".to_string() } else { format!("Plan:\n{plan_str}") }, success: true });
                            }
                        }
                    }
                }

                // Apply mutations
                let has_mutations = ops.iter().any(|op| !matches!(op,
                    crate::parser::Operation::Test(_) | crate::parser::Operation::Trace(_)
                    | crate::parser::Operation::Eval(_) | crate::parser::Operation::Query(_)
                    | crate::parser::Operation::Undo | crate::parser::Operation::Plan(_)
                    | crate::parser::Operation::Watch { .. }
                    | crate::parser::Operation::Agent { .. }
                    | crate::parser::Operation::Message { .. }
                    | crate::parser::Operation::OpenCode(_)));

                if has_mutations {
                    match session.apply(&code) {
                        Ok(res) => {
                            // Sync shared vars to the Arc<RwLock> runtime
                            if let Ok(mut rt) = config.runtime.write() {
                                rt.shared_vars = session.runtime.shared_vars.clone();
                            }
                            for (msg, ok) in res {
                                eprintln!("[web:{}] {msg}", if ok { "ok" } else { "err" });
                                if !ok { iter_has_errors = true; }
                                iter_results.push(MutationResult { message: msg, success: ok });
                            }
                        }
                        Err(e) => {
                            eprintln!("[web:err] {e}");
                            iter_has_errors = true;
                            iter_results.push(MutationResult { message: format!("{e}"), success: false });
                        }
                    }
                }

                // Handle tests, evals, queries, opencode
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
                                        drop(session);
                                        let result = crate::eval::eval_test_case_async(
                                            &program, &fn_name, &case, &mocks, sender, &routes,
                                        ).await;
                                        session = config.session.lock().await;
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
                                    Ok(msg) => { eprintln!("[web:pass] {msg}"); iter_test_results.push(TestCaseResult { message: msg, pass: true }); }
                                    Err(e) => { all_passed = false; eprintln!("[web:fail] {e}"); iter_has_errors = true; iter_test_results.push(TestCaseResult { message: format!("{e}"), pass: false }); }
                                }
                            }
                            if all_passed && !test.cases.is_empty() {
                                session.store_test(&test.function_name, &test.cases);
                            }
                        }
                        crate::parser::Operation::Eval(ev) => {
                            // Inline expression: evaluate directly
                            if let Some(ref expr) = ev.inline_expr {
                                match crate::eval::eval_inline_expr(&session.program, expr) {
                                    Ok(val) => {
                                        let msg = format!("= {val}");
                                        eprintln!("[web:eval] {msg}");
                                        iter_results.push(MutationResult { message: msg, success: true });
                                    }
                                    Err(e) => {
                                        iter_has_errors = true;
                                        let msg = format!("eval error: {e}");
                                        eprintln!("[web:eval:err] {msg}");
                                        iter_results.push(MutationResult { message: msg, success: false });
                                    }
                                }
                                continue;
                            }

                            // Block eval of untested functions in AdapsisOS mode
                            if session.program.require_modules {
                                if let Some(func) = session.program.get_function(&ev.function_name) {
                                    if func.body.len() > 2 && !session.is_function_tested(&ev.function_name) {
                                        iter_has_errors = true;
                                        iter_results.push(MutationResult {
                                            message: format!("function `{}` has {} statements but no passing tests. Write !test blocks first.", ev.function_name, func.body.len()),
                                            success: false,
                                        });
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
                                    let fn_name = ev.function_name.clone();
                                    let input = ev.input.clone();
                                    let sender = sender.clone();
                                    drop(session);
                                    let eval_result = tokio::task::spawn_blocking(move || {
                                        let func = program.get_function(&fn_name)
                                            .ok_or_else(|| anyhow::anyhow!("function not found"))?;
                                        let handle = crate::coroutine::CoroutineHandle::new(sender);
                                        let mut env = crate::eval::Env::new();
                                        env.populate_shared_from_program(&program);
                                        env.set("__coroutine_handle", crate::eval::Value::CoroutineHandle(handle));
                                        let input_val = crate::eval::eval_parser_expr_with_program(&input, &program)?;
                                        crate::eval::bind_input_to_params(&program, func, &input_val, &mut env);
                                        crate::eval::eval_function_body_pub(&program, &func.body, &mut env)
                                    }).await;
                                    let (msg, success) = match &eval_result {
                                        Ok(Ok(val)) => (format!("eval {}() = {val}", ev.function_name), true),
                                        Ok(Err(e)) => { iter_has_errors = true; (format!("eval error: {e}"), false) }
                                        Err(e) => { iter_has_errors = true; (format!("eval task error: {e}"), false) }
                                    };
                                    eprintln!("[web:eval] {msg}");
                                    iter_results.push(MutationResult { message: msg, success });
                                    session = config.session.lock().await;
                                }
                            } else {
                                match crate::eval::eval_compiled_or_interpreted_cached(&session.program, &ev.function_name, &ev.input, Some(&config.jit_cache), session.meta.revision) {
                                    Ok((result, compiled)) => {
                                        let tag = if compiled { " [compiled]" } else { "" };
                                        let msg = format!("eval {}() = {result}{tag}", ev.function_name);
                                        eprintln!("[web:eval] {msg}");
                                        iter_results.push(MutationResult { message: msg, success: true });
                                    }
                                    Err(e) => {
                                        iter_has_errors = true;
                                        let msg = format!("eval error: {e}");
                                        eprintln!("[web:eval:err] {msg}");
                                        iter_results.push(MutationResult { message: msg, success: false });
                                    }
                                }
                            }
                        }
                        crate::parser::Operation::Query(query) => {
                            let response = if query.trim() == "?inbox" || query.trim().starts_with("?inbox") {
                                let msgs = session.peek_messages("main");
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
                            iter_results.push(MutationResult { message: response, success: true });
                        }
                        crate::parser::Operation::Watch { function_name, args, interval_ms } => {
                            eprintln!("[web:watch] {function_name}({args}) every {interval_ms}ms");
                            let fn_name = function_name.clone();
                            let fn_args = args.clone();
                            let interval = *interval_ms;
                            let session_ref = config.session.clone();
                            let io_sender = config.io_sender.clone();
                            let trigger = config.self_trigger.clone();
                            let watch_jit_cache = config.jit_cache.clone();

                            tokio::spawn(async move {
                                let mut last_result = String::new();
                                loop {
                                    tokio::time::sleep(std::time::Duration::from_millis(interval)).await;

                                    // Evaluate the function
                                    let result = {
                                        let session = session_ref.lock().await;
                                        let input_expr = if fn_args.trim().is_empty() {
                                            crate::parser::Expr::StructLiteral(vec![])
                                        } else {
                                            match crate::parser::parse_test_input(0, &fn_args) {
                                                Ok(expr) => expr,
                                                Err(e) => {
                                                    format!("parse error: {e}")
                                                        .clone(); // type doesn't matter, we break below
                                                    break;
                                                }
                                            }
                                        };
                                        match crate::eval::eval_compiled_or_interpreted_cached(
                                            &session.program, &fn_name, &input_expr,
                                            Some(&watch_jit_cache), session.meta.revision,
                                        ) {
                                            Ok((r, _)) => r,
                                            Err(e) => format!("error: {e}"),
                                        }
                                    };

                                    if result != last_result && !last_result.is_empty() {
                                        eprintln!("[web:watch:trigger] {fn_name} changed: {last_result} → {result}");
                                        let msg = format!("Watcher '{fn_name}' triggered: result changed from '{last_result}' to '{result}'");
                                        // Trigger the AI to respond
                                        let _ = trigger.send(msg).await;
                                    }
                                    last_result = result;
                                }
                            });

                            iter_results.push(MutationResult {
                                message: format!("Watching {function_name}({args}) every {interval_ms}ms"),
                                success: true,
                            });
                        }
                        crate::parser::Operation::Agent { name, scope, task } => {
                            eprintln!("[web:agent] spawning '{name}' scope={scope} task={}", task.chars().take(80).collect::<String>());

                            let agent_scope = crate::session::AgentScope::parse(scope);
                            let branch = crate::session::AgentBranch::fork(name, agent_scope, task, &session);
                            let program_summary = crate::validator::program_summary_compact(&session.program);
                            let agent_task = task.clone();
                            let agent_name = name.clone();
                            let llm_url = config.llm_url.clone();
                            let llm_model = config.llm_model.clone();
                            let llm_key = config.llm_api_key.clone();
                            let session_ref = config.session.clone();

                            drop(session);

                            // Run agent in background
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
                                    // Check inbox for messages from other agents or main
                                    {
                                        let mut session = session_ref.lock().await;
                                        let inbox = session.drain_messages(&agent_name);
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

                                    // Handle !msg commands from agent code before applying mutations
                                    if let Ok(ops) = crate::parser::parse(&code) {
                                        for op in &ops {
                                            if let crate::parser::Operation::Message { to, content } = op {
                                                eprintln!("[agent:{agent_name}] !msg → {to}: {content}");
                                                let mut session = session_ref.lock().await;
                                                session.send_agent_message(&agent_name, to, content);
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

                                // Merge branch back into main session
                                let mut session = session_ref.lock().await;
                                let conflicts = branch.merge_into(&mut session);
                                if conflicts.is_empty() {
                                    eprintln!("[agent:{agent_name}] merged successfully");
                                    session.meta.chat_messages.push(crate::session::ChatMessage {
                                        role: "system".to_string(),
                                        content: format!("Agent '{agent_name}' completed and merged successfully."),
                                    });
                                    if let Some(s) = session.meta.agent_log.iter_mut().rev().find(|s| s.name == agent_name && s.status == "running") {
                                        s.status = "merged".to_string();
                                        s.message = "completed and merged".to_string();
                                    }
                                } else {
                                    eprintln!("[agent:{agent_name}] merge conflicts: {:?}", conflicts);
                                    session.meta.chat_messages.push(crate::session::ChatMessage {
                                        role: "system".to_string(),
                                        content: format!("Agent '{agent_name}' finished but had merge conflicts:\n{}", conflicts.join("\n")),
                                    });
                                    if let Some(s) = session.meta.agent_log.iter_mut().rev().find(|s| s.name == agent_name && s.status == "running") {
                                        s.status = "conflict".to_string();
                                        s.message = conflicts.join("; ");
                                    }
                                }
                            });

                            iter_results.push(MutationResult {
                                message: format!("Agent '{name}' spawned (background)"),
                                success: true,
                            });

                            session = config.session.lock().await;
                            session.meta.agent_log.push(crate::session::AgentStatus {
                                name: name.clone(),
                                task: task.chars().take(100).collect(),
                                scope: scope.clone(),
                                status: "running".to_string(),
                                message: String::new(),
                            });
                        }
                        crate::parser::Operation::Message { to, content } => {
                            eprintln!("[web:msg] → {to}: {content}");
                            session.send_agent_message("main", &to, &content);
                            iter_results.push(MutationResult {
                                message: format!("Message sent to '{to}'"),
                                success: true,
                            });
                        }
                        crate::parser::Operation::OpenCode(task) => {
                            eprintln!("[web:opencode] {task}");
                            drop(session);
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
                                    eprintln!("[web:opencode:done] rebuilding...");
                                    let build = tokio::process::Command::new("cargo")
                                        .arg("build").arg("--release").current_dir(&config.project_dir).output().await;
                                    match build {
                                        Ok(b) if b.status.success() => {
                                            iter_results.push(MutationResult {
                                                message: "OpenCode + rebuild successful. Restart to apply.".to_string(), success: true,
                                            });
                                            tokio::spawn(async {
                                                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                                                // Use args[0] instead of current_exe() — after rebuild the old
                                                // inode is deleted and /proc/self/exe shows "(deleted)".
                                                let exe = std::env::args().next()
                                                    .map(std::path::PathBuf::from)
                                                    .and_then(|p| std::fs::canonicalize(&p).ok().or(Some(p)))
                                                    .unwrap_or_else(|| std::env::current_exe().unwrap_or_default());
                                                let args: Vec<String> = std::env::args().collect();
                                                let _ = exec::execvp(&exe, &args);
                                            });
                                        }
                                        _ => {
                                            iter_has_errors = true;
                                            iter_results.push(MutationResult { message: "OpenCode done but build failed".to_string(), success: false });
                                        }
                                    }
                                }
                                _ => {
                                    iter_has_errors = true;
                                    iter_results.push(MutationResult { message: "OpenCode failed or timed out".to_string(), success: false });
                                }
                            }
                            session = config.session.lock().await;
                        }
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
                                    let mut env = crate::eval::Env::new();
                                    env.populate_shared_from_program(&session.program);
                                    if let Some(sender) = &config.io_sender {
                                        env.set("__coroutine_handle", crate::eval::Value::CoroutineHandle(
                                            crate::coroutine::CoroutineHandle::new(sender.clone())
                                        ));
                                    }
                                    match crate::eval::eval_function_body_pub(&session.program, &[stmt], &mut env) {
                                        Ok(val) => {
                                            let msg = format!("executed: {val}");
                                            eprintln!("[web:exec] {msg}");
                                            iter_results.push(MutationResult { message: msg, success: true });
                                        }
                                        Err(e) => {
                                            iter_has_errors = true;
                                            let msg = format!("exec error: {e}");
                                            eprintln!("[web:exec:err] {msg}");
                                            iter_results.push(MutationResult { message: msg, success: false });
                                        }
                                    }
                                }
                                Err(e) => {
                                    iter_has_errors = true;
                                    iter_results.push(MutationResult { message: format!("statement error: {e}"), success: false });
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            Err(e) => {
                iter_has_errors = true;
                iter_results.push(MutationResult { message: format!("Parse error: {e}"), success: false });
            }
        }

        all_results.extend(iter_results.clone());
        all_test_results.extend(iter_test_results.clone());
        drop(session);

        // Build feedback for next iteration
        if iter_has_errors {
            let errors: Vec<String> = iter_results.iter().filter(|r| !r.success).map(|r| r.message.clone())
                .chain(iter_test_results.iter().filter(|r| !r.pass).map(|r| r.message.clone()))
                .collect();
            let feedback = format!("Errors:\n{}\n\nFix and continue.", errors.join("\n"));
            eprintln!("[web:feedback] → retrying");
            messages.push(crate::llm::ChatMessage::user(feedback));
        } else {
            // Success — tell the AI to continue or finish
            let results_summary: Vec<String> = iter_results.iter().map(|r| r.message.clone()).collect();
            let feedback = format!(
                "Results:\n{}\n\nIf the task is complete, respond with !done. Otherwise continue with the next step.",
                results_summary.join("\n")
            );
            messages.push(crate::llm::ChatMessage::user(feedback));
        }
    }

    // Save conversation
    {
        let mut session = config.session.lock().await;
        let summary = format!("{}\n{}", reply_text.chars().take(200).collect::<String>(),
            all_results.iter().map(|r| format!("{}: {}", if r.success {"OK"} else {"ERR"}, r.message)).collect::<Vec<_>>().join("\n"));
        session.meta.chat_messages.push(crate::session::ChatMessage {
            role: "assistant".to_string(), content: summary,
        });
        if session.meta.chat_messages.len() > 50 {
            let system = session.meta.chat_messages[0].clone();
            let start = session.meta.chat_messages.len() - 49;
            let keep: Vec<_> = session.meta.chat_messages[start..].to_vec();
            session.meta.chat_messages = vec![system];
            session.meta.chat_messages.extend(keep);
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
    let meta = config.meta.lock().await;
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
        let tx = EventSender::with_mpsc(raw_tx, config_clone.event_broadcast.clone());
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
            let mut session = config_clone.session.lock().await;
            if session.meta.chat_messages.is_empty() {
                session.meta.chat_messages.push(crate::session::ChatMessage {
                    role: "system".to_string(), content: system_prompt,
                });
            }
            let (plan_ctx, needs_plan) = build_plan_context(&session.meta.plan);
            let plan_hint = if needs_plan {
                "\n\nYour previous plan is completed (or none exists). Create a new plan with !plan set for this task before writing code. You can update it anytime with !plan set / !plan done N."
            } else { "" };
            let context = format!("Working directory: {}\n{}{}\nUser: {}{}",
                config_clone.project_dir,
                crate::validator::program_summary_compact(&session.program),
                plan_ctx, req.message, plan_hint);
            log_activity(&config_clone.log_file, "user", &context).await;
            session.meta.chat_messages.push(crate::session::ChatMessage {
                role: "user".to_string(), content: context,
            });
            session.meta.chat_messages.iter().map(|m| match m.role.as_str() {
                "system" => crate::llm::ChatMessage::system(m.content.clone()),
                "assistant" => crate::llm::ChatMessage::assistant(&m.content),
                _ => crate::llm::ChatMessage::user(m.content.clone()),
            }).collect::<Vec<_>>()
        };

        let max_iterations = config_clone.max_iterations;
        let mut last_context = req.message.clone();
        for iteration in 0..max_iterations {
            // Check for injected messages and append to conversation
            {
                let mut queue = config_clone.message_queue.lock().await;
                for injected in queue.drain(..) {
                    eprintln!("[inject] processing: {}...", injected.chars().take(80).collect::<String>());
                    log_activity(&config_clone.log_file, "inject", &injected).await;
                    messages.push(crate::llm::ChatMessage::user(injected));
                    let _ = tx.send(serde_json::json!({"type": "result", "message": "Injected message received", "success": true})).await;
                }
            }

            let _ = tx.send(serde_json::json!({"type": "iteration", "n": iteration + 1})).await;
            log_activity(&config_clone.log_file, "iter", &format!("iteration {}/{}", iteration + 1, max_iterations)).await;

            // Retry LLM calls on transient errors (network, timeout)
            let output = {
                let mut last_err = String::new();
                let mut result = None;
                let waiting_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
                let wf = waiting_flag.clone();
                let log_ref = config_clone.log_file.clone();
                let broadcast_ref = config_clone.event_broadcast.clone();
                let iter_num = iteration + 1;
                tokio::spawn(async move {
                    let mut secs = 0u64;
                    loop {
                        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                        if !wf.load(std::sync::atomic::Ordering::Relaxed) { break; }
                        secs += 30;
                        let msg = format!("Waiting for LLM response... ({secs}s, iteration {iter_num})");
                        eprintln!("[waiting] {msg}");
                        log_activity(&log_ref, "waiting", &msg).await;
                        let _ = broadcast_ref.send(serde_json::json!({"type": "result", "message": msg, "success": true}));
                    }
                });
                for retry in 0..3 {
                    match llm.generate(messages.clone()).await {
                        Ok(o) => { result = Some(o); break; }
                        Err(e) => {
                            last_err = format!("{e}");
                            log_activity(&config_clone.log_file, "llm-error", &format!("attempt {}/{}: {e}", retry + 1, 3)).await;
                            if retry < 2 {
                                let _ = tx.send(serde_json::json!({"type": "error", "message": format!("LLM error (retrying): {e}")})).await;
                                tokio::time::sleep(std::time::Duration::from_secs(5 * (retry as u64 + 1))).await;
                            }
                        }
                    }
                }
                waiting_flag.store(false, std::sync::atomic::Ordering::Relaxed);
                match result {
                    Some(o) => o,
                    None => {
                        let _ = tx.send(serde_json::json!({"type": "error", "message": format!("LLM failed after 3 retries: {last_err}")})).await;
                        break;
                    }
                }
            };

            messages.push(crate::llm::ChatMessage::assistant(&output.text));

            if !output.thinking.is_empty() {
                log_activity(&config_clone.log_file, "think", &output.thinking).await;
                let _ = tx.send(serde_json::json!({"type": "thinking", "text": output.thinking})).await;
            }

            // Extract prose
            let mut clean = output.text.clone();
            while let Some(s) = clean.find("<think>") { if let Some(e) = clean.find("</think>") { clean.replace_range(s..e+8, ""); } else { break; } }
            while let Some(s) = clean.find("<code>") { if let Some(e) = clean.find("</code>") { clean.replace_range(s..e+7, ""); } else { break; } }
            let clean = clean.trim();
            if !clean.is_empty() {
                log_activity(&config_clone.log_file, "ai-text", clean).await;
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

            log_activity(&config_clone.log_file, "code", &code).await;
            let _ = tx.send(serde_json::json!({"type": "code", "code": code})).await;

            // Apply code
            let mut session = config_clone.session.lock().await;
            let mut has_errors = false;
            let mut feedback_details: Vec<String> = Vec::new();
            let mut train_tests_passed: usize = 0;
            let mut train_tests_failed: usize = 0;
            let mut accepted_done = false;

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

                    // Handle plan, undo
                    for op in &ops {
                        if let crate::parser::Operation::Plan(action) = op {
                            match action {
                                crate::parser::PlanAction::Set(steps) => {
                                    session.meta.plan = steps.iter().map(|s| crate::session::PlanStep {
                                        description: s.clone(), status: crate::session::PlanStatus::Pending,
                                    }).collect();
                                    let plan_json: Vec<serde_json::Value> = session.meta.plan.iter().map(|s| {
                                        serde_json::json!({"description": s.description, "status": format!("{:?}", s.status).to_lowercase()})
                                    }).collect();
                                    let _ = tx.send(serde_json::json!({"type": "plan", "plan": plan_json})).await;
                                    let _ = tx.send(serde_json::json!({"type": "result", "message": format!("Plan: {} steps", steps.len()), "success": true})).await;
                                }
                                crate::parser::PlanAction::Progress(n) => {
                                    if let Some(step) = session.meta.plan.get_mut(n.saturating_sub(1)) {
                                        step.status = crate::session::PlanStatus::Done;
                                        let plan_json: Vec<serde_json::Value> = session.meta.plan.iter().map(|s| {
                                            serde_json::json!({"description": s.description, "status": format!("{:?}", s.status).to_lowercase()})
                                        }).collect();
                                        let _ = tx.send(serde_json::json!({"type": "plan", "plan": plan_json})).await;
                                        let _ = tx.send(serde_json::json!({"type": "result", "message": format!("Step {n} done"), "success": true})).await;
                                    }
                                }
                                _ => {}
                            }
                        }
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
                        match session.apply(&code) {
                            Ok(res) => {
                                // Sync shared vars to the Arc<RwLock> runtime
                                if let Ok(mut rt) = config_clone.runtime.write() {
                                    rt.shared_vars = session.runtime.shared_vars.clone();
                                }
                                for (msg, ok) in &res {
                                    if !*ok { has_errors = true; }
                                    feedback_details.push(format!("{}: {msg}", if *ok {"OK"} else {"ERROR"}));
                                    let _ = tx.send(serde_json::json!({"type": "result", "message": msg, "success": ok})).await;
                                }
                            }
                            Err(e) => {
                                has_errors = true;
                                feedback_details.push(format!("ERROR: {e}"));
                                let _ = tx.send(serde_json::json!({"type": "result", "message": format!("{e}"), "success": false})).await;
                            }
                        }
                    }

                    // Tests and evals
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
                                            drop(session);
                                            let result = crate::eval::eval_test_case_async(
                                                &program, &fn_name, &case, &mocks, sender, &routes,
                                            ).await;
                                            session = config_clone.session.lock().await;
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
                                            train_tests_passed += 1;
                                            feedback_details.push(format!("PASS: {msg}"));
                                            let _ = tx.send(serde_json::json!({"type": "test", "pass": true, "message": msg})).await;
                                        }
                                        Err(e) => {
                                            all_passed = false;
                                            has_errors = true;
                                            train_tests_failed += 1;
                                            feedback_details.push(format!("FAIL: {e}"));
                                            let _ = tx.send(serde_json::json!({"type": "test", "pass": false, "message": format!("{e}")})).await;
                                        }
                                    }
                                }
                                if all_passed && !test.cases.is_empty() {
                                    session.store_test(&test.function_name, &test.cases);
                                }
                            }
                            crate::parser::Operation::Eval(ev) => {
                                // Inline expression: evaluate directly
                                if let Some(ref expr) = ev.inline_expr {
                                    match crate::eval::eval_inline_expr(&session.program, expr) {
                                        Ok(val) => {
                                            let msg = format!("{val}");
                                            feedback_details.push(format!("= {msg}"));
                                            let _ = tx.send(serde_json::json!({"type": "eval", "result": msg, "function": "(inline)", "success": true})).await;
                                        }
                                        Err(e) => {
                                            has_errors = true;
                                            let msg = format!("eval error: {e}");
                                            feedback_details.push(format!("ERROR: {msg}"));
                                            let _ = tx.send(serde_json::json!({"type": "eval", "result": msg, "function": "(inline)", "success": false})).await;
                                        }
                                    }
                                    continue;
                                }

                                // Block eval of untested functions (>2 statements) in AdapsisOS mode
                                if session.program.require_modules {
                                    if let Some(func) = session.program.get_function(&ev.function_name) {
                                        if func.body.len() > 2 && !session.is_function_tested(&ev.function_name) {
                                            has_errors = true;
                                            let msg = format!("function `{}` has {} statements but no passing tests. Write !test blocks first.", ev.function_name, func.body.len());
                                            feedback_details.push(format!("ERROR: {msg}"));
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
                                        let fn_name = ev.function_name.clone();
                                        let input = ev.input.clone();
                                        let sender = sender.clone();
                                        drop(session);
                                        let eval_result = tokio::task::spawn_blocking(move || {
                                            let func = program.get_function(&fn_name)
                                                .ok_or_else(|| anyhow::anyhow!("function not found"))?;
                                            let handle = crate::coroutine::CoroutineHandle::new(sender);
                                            let mut env = crate::eval::Env::new();
                                            env.populate_shared_from_program(&program);
                                            env.set("__coroutine_handle", crate::eval::Value::CoroutineHandle(handle));
                                            let input_val = crate::eval::eval_parser_expr_with_program(&input, &program)?;
                                            crate::eval::bind_input_to_params(&program, func, &input_val, &mut env);
                                            crate::eval::eval_function_body_pub(&program, &func.body, &mut env)
                                        }).await;
                                        let (msg, success) = match &eval_result {
                                            Ok(Ok(val)) => (format!("{val}"), true),
                                            Ok(Err(e)) => { has_errors = true; (format!("error: {e}"), false) }
                                            Err(e) => { has_errors = true; (format!("task error: {e}"), false) }
                                        };
                                        feedback_details.push(format!("eval {}() = {}{}", ev.function_name, msg, if success {""} else {" [FAILED]"}));
                                        let _ = tx.send(serde_json::json!({"type": "eval", "result": msg, "function": ev.function_name, "success": success})).await;
                                        session = config_clone.session.lock().await;
                                    } else {
                                        has_errors = true;
                                        feedback_details.push(format!("eval {}() = async not available [FAILED]", ev.function_name));
                                        let _ = tx.send(serde_json::json!({"type": "eval", "result": "async not available", "function": ev.function_name, "success": false})).await;
                                    }
                                } else {
                                    match crate::eval::eval_compiled_or_interpreted_cached(&session.program, &ev.function_name, &ev.input, Some(&config_clone.jit_cache), session.meta.revision) {
                                        Ok((result, compiled)) => {
                                            let tag = if compiled { " [compiled]" } else { "" };
                                            feedback_details.push(format!("eval {}() = {result}{tag}", ev.function_name));
                                            let _ = tx.send(serde_json::json!({"type": "eval", "result": format!("{result}{tag}"), "function": ev.function_name})).await;
                                        }
                                        Err(e) => {
                                            has_errors = true;
                                            feedback_details.push(format!("eval {}() error: {e} [FAILED]", ev.function_name));
                                            let _ = tx.send(serde_json::json!({"type": "eval", "result": format!("error: {e}"), "function": ev.function_name})).await;
                                        }
                                    }
                                }
                            }
                            crate::parser::Operation::Query(query) => {
                                let response = if query.trim() == "?inbox" || query.trim().starts_with("?inbox") {
                                    let msgs = session.peek_messages("main");
                                    if msgs.is_empty() {
                                        "No messages.".to_string()
                                    } else {
                                        msgs.iter().map(|m| format!("[{}] from {}: {}", m.timestamp, m.from, m.content)).collect::<Vec<_>>().join("\n")
                                    }
                                } else if query.trim() == "?tasks" {
                                    format_tasks(&config_clone.task_registry)
                                } else if let Some(tid) = parse_inspect_task_query(query.trim()) {
                                    format_inspect_task(&config_clone.task_registry, &config_clone.snapshot_registry, tid)
                                } else if query.trim() == "?library" {
                                    crate::library::query_library(&session.program, session.meta.library_state.as_ref())
                                } else {
                                    let table = crate::typeck::build_symbol_table(&session.program);
                                    crate::typeck::handle_query(&session.program, &table, query, &session.runtime.http_routes)
                                };
                                let _ = tx.send(serde_json::json!({"type": "query", "query": query, "response": response})).await;
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
                                // Convert to AST statement and execute
                                match crate::validator::convert_statement_op(op) {
                                    Ok(stmt) => {
                                        let mut env = crate::eval::Env::new();
                                        env.populate_shared_from_program(&session.program);
                                        // Propagate coroutine handle if available
                                        if let Some(sender) = &config_clone.io_sender {
                                            env.set("__coroutine_handle", crate::eval::Value::CoroutineHandle(
                                                crate::coroutine::CoroutineHandle::new(sender.clone())
                                            ));
                                        }
                                        let program = session.program.clone();
                                        drop(session);
                                        let result = tokio::task::spawn_blocking(move || {
                                            crate::eval::eval_function_body_pub(&program, &[stmt], &mut env)
                                        }).await;
                                        match result {
                                            Ok(Ok(val)) => {
                                                let msg = format!("executed: {val}");
                                                feedback_details.push(format!("OK: {msg}"));
                                                let _ = tx.send(serde_json::json!({"type": "result", "message": msg, "success": true})).await;
                                            }
                                            Ok(Err(e)) => {
                                                has_errors = true;
                                                feedback_details.push(format!("ERROR: {e}"));
                                                let _ = tx.send(serde_json::json!({"type": "result", "message": format!("{e}"), "success": false})).await;
                                            }
                                            Err(e) => {
                                                has_errors = true;
                                                feedback_details.push(format!("ERROR: task error: {e}"));
                                                let _ = tx.send(serde_json::json!({"type": "result", "message": format!("task error: {e}"), "success": false})).await;
                                            }
                                        }
                                        session = config_clone.session.lock().await;
                                    }
                                    Err(e) => {
                                        has_errors = true;
                                        feedback_details.push(format!("ERROR: statement error: {e}"));
                                        let _ = tx.send(serde_json::json!({"type": "result", "message": format!("statement error: {e}"), "success": false})).await;
                                    }
                                }
                            }
                            crate::parser::Operation::Done => {
                                // Check for untested functions before accepting
                                if session.program.require_modules {
                                    let untested: Vec<String> = session.program.modules.iter().flat_map(|m| {
                                        m.functions.iter().filter_map(|f| {
                                            let qname = format!("{}.{}", m.name, f.name);
                                            if f.body.len() > 2 && !session.is_function_tested(&qname) {
                                                Some(qname)
                                            } else { None }
                                        })
                                    }).collect();
                                    if !untested.is_empty() {
                                        let challenge = format!(
                                            "Cannot accept !done: {} untested functions: {}. Write !test blocks for them.",
                                            untested.len(), untested.join(", ")
                                        );
                                        log_activity(&config_clone.log_file, "done-rejected", &challenge).await;
                                        feedback_details.push(format!("ERROR: {challenge}"));
                                        has_errors = true;
                                        continue;
                                    }
                                }
                                log_activity(&config_clone.log_file, "done", &format!("AI said !done at iteration {}", iteration + 1)).await;
                                let _ = tx.send(serde_json::json!({"type": "done"})).await;
                                accepted_done = true;
                                break;
                            }
                            crate::parser::Operation::Mock { operation, patterns, response } => {
                                let pattern_display = patterns.iter().map(|p| format!("\"{p}\"")).collect::<Vec<_>>().join(" ");
                                session.meta.io_mocks.push(crate::session::IoMock {
                                    operation: operation.clone(), patterns: patterns.clone(), response: response.clone(),
                                });
                                feedback_details.push(format!("mock: {operation} {pattern_display}"));
                                let _ = tx.send(serde_json::json!({"type": "result", "message": format!("mock: {operation} {pattern_display}"), "success": true})).await;
                            }
                            crate::parser::Operation::Unmock => {
                                let count = session.meta.io_mocks.len();
                                session.meta.io_mocks.clear();
                                feedback_details.push(format!("cleared {count} mocks"));
                                let _ = tx.send(serde_json::json!({"type": "result", "message": format!("cleared {count} mocks"), "success": true})).await;
                            }
                            crate::parser::Operation::Message { to, content } => {
                                eprintln!("[web:msg] → {to}: {content}");
                                session.send_agent_message("main", &to, &content);
                                feedback_details.push(format!("Message sent to '{to}'"));
                                let _ = tx.send(serde_json::json!({"type": "result", "message": format!("Message sent to '{to}'"), "success": true})).await;
                            }
                            crate::parser::Operation::OpenCode(task) => {
                                eprintln!("[web:opencode:stream] {task}");
                                log_activity(&config_clone.log_file, "opencode", &task).await;
                                // Sequential lock — only one !opencode at a time
                                let _opencode_guard = config_clone.opencode_lock.lock().await;
                                let _ = tx.send(serde_json::json!({"type": "result", "message": format!("Running !opencode: {}", task.chars().take(80).collect::<String>()), "success": true})).await;
                                drop(session);
                                // Use the configured opencode directory (fixed checkout, not dynamic worktrees)
                                let work_dir = config_clone.opencode_git_dir.clone();
                                log_activity(&config_clone.log_file, "opencode-dir", &work_dir).await;
                                use tokio::io::{AsyncBufReadExt, BufReader};

                                // Get existing OpenCode session ID to continue building on top
                                let oc_session_id = {
                                    let s = config_clone.session.lock().await;
                                    s.meta.opencode_session_id.clone()
                                };

                                let recent_lines = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
                                let recent_for_stream = recent_lines.clone();
                                let had_tool_calls = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                                let had_tool_calls_stream = had_tool_calls.clone();
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

                                        let idle_timeout = std::time::Duration::from_secs(600); // 10 min
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
                                                        let msg = format!("[opencode] IDLE TIMEOUT: no output on stdout or stderr for {}s — killing OpenCode process", elapsed.as_secs());
                                                        eprintln!("{msg}");
                                                        log_activity(&config_clone.log_file, "opencode-timeout", &msg).await;
                                                        has_errors = true;
                                                        feedback_details.push(format!("ERROR: {msg}"));
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
                                                            let mut s = config_clone.session.lock().await;
                                                            if s.meta.opencode_session_id.as_deref() != Some(sid) {
                                                                s.meta.opencode_session_id = Some(sid.to_string());
                                                                eprintln!("[opencode] session ID: {sid}");
                                                            }
                                                        }
                                                    }
                                                    _ => {
                                                        eprintln!("[opencode:event] {event_type}");
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
                                                let mut s = config_clone.session.lock().await;
                                                s.meta.opencode_session_id = None;
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
                                                    let idle_timeout = std::time::Duration::from_secs(300);
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
                                                                let mut s = config_clone.session.lock().await;
                                                                if s.meta.opencode_session_id.is_none() {
                                                                    s.meta.opencode_session_id = Some(sid.to_string());
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
                                                    has_errors = true;
                                                    let ctx = recent_lines.lock().unwrap().join("\n");
                                                    feedback_details.push(format!("ERROR: OpenCode retry failed\n{ctx}"));
                                                    session = config_clone.session.lock().await;
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
                                            let msg = format!("OpenCode exited without making any file changes. It may have asked for clarification instead of proceeding. OpenCode said: {preview}");
                                            eprintln!("[opencode:no-changes] {msg}");
                                            log_activity(&config_clone.log_file, "opencode-no-changes", &msg).await;
                                            has_errors = true;
                                            feedback_details.push(format!("ERROR: {msg}"));
                                            let _ = tx.send(serde_json::json!({"type": "result", "message": msg, "success": false})).await;
                                            session = config_clone.session.lock().await;
                                            continue;
                                        }
                                    }
                                }

                                match oc_result {
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
                                                // Save session before restart
                                                {
                                                    let session = config_clone.session.lock().await;
                                                    if let Some(path) = std::env::args().nth(std::env::args().position(|a| a == "--session").unwrap_or(999) + 1) {
                                                        let _ = session.save(std::path::Path::new(&path));
                                                    }
                                                }
                                                // execvp replaces the process — if it returns, it failed
                                                // Use args[0] instead of current_exe() because after cargo rebuild
                                                // the old inode is deleted and /proc/self/exe shows "(deleted)".
                                                let exe = std::env::args().next()
                                                    .map(std::path::PathBuf::from)
                                                    .and_then(|p| std::fs::canonicalize(&p).ok().or(Some(p)))
                                                    .unwrap_or_else(|| std::env::current_exe().unwrap_or_default());
                                                let args: Vec<String> = std::env::args().collect();
                                                let err = exec::execvp(&exe, &args);
                                                // If we get here, execvp failed
                                                let msg = format!("RESTART FAILED: exec::execvp returned: {err}. The new binary is built but NOT running. Manual restart required.");
                                                eprintln!("[opencode] {msg}");
                                                log_activity(&config_clone.log_file, "opencode-restart-FAILED", &msg).await;
                                                feedback_details.push(format!("ERROR: {msg}"));
                                                has_errors = true;
                                            }
                                            Ok(b) => {
                                                has_errors = true;
                                                let stderr = String::from_utf8_lossy(&b.stderr);
                                                let msg = format!("OpenCode done but cargo build failed:\n{stderr}");
                                                feedback_details.push(format!("ERROR: {msg}"));
                                                log_activity(&config_clone.log_file, "opencode-build-fail", &msg).await;
                                                let _ = tx.send(serde_json::json!({"type": "result", "message": msg, "success": false})).await;
                                            }
                                            Err(e) => {
                                                has_errors = true;
                                                feedback_details.push(format!("ERROR: build error: {e}"));
                                                let _ = tx.send(serde_json::json!({"type": "result", "message": format!("build error: {e}"), "success": false})).await;
                                            }
                                        }
                                    }
                                    Ok(Ok(status)) => {
                                        has_errors = true;
                                        let context = recent_lines.lock().unwrap().join("\n");
                                        let msg = format!("OpenCode exited with status: {status}\nLast output:\n{context}");
                                        feedback_details.push(format!("ERROR: {msg}"));
                                        log_activity(&config_clone.log_file, "opencode-fail", &msg).await;
                                        let _ = tx.send(serde_json::json!({"type": "result", "message": msg, "success": false})).await;
                                    }
                                    Ok(Err(e)) => {
                                        has_errors = true;
                                        let context = recent_lines.lock().unwrap().join("\n");
                                        let msg = format!("OpenCode error: {e}\nLast output:\n{context}");
                                        feedback_details.push(format!("ERROR: {msg}"));
                                        log_activity(&config_clone.log_file, "opencode-error", &msg).await;
                                        let _ = tx.send(serde_json::json!({"type": "result", "message": msg, "success": false})).await;
                                    }
                                    Err(_) => {
                                        has_errors = true;
                                        let context = recent_lines.lock().unwrap().join("\n");
                                        let msg = format!("OpenCode timed out (30 min limit)\nLast output:\n{context}");
                                        feedback_details.push(format!("ERROR: {msg}"));
                                        log_activity(&config_clone.log_file, "opencode-timeout", &msg).await;
                                        let _ = tx.send(serde_json::json!({"type": "result", "message": msg, "success": false})).await;
                                    }
                                }
                                session = config_clone.session.lock().await;
                            }
                            _ => {}
                        }
                    }
                }
                Err(e) => {
                    has_errors = true;
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
                    feedback_details.push(format!("ERROR: {msg}"));
                    let _ = tx.send(serde_json::json!({"type": "result", "message": msg, "success": false})).await;
                }
            }

            drop(session);

            // Build detailed feedback with ALL results so the AI can see them
            // Also re-run queries and check inbox
            {
                let mut session = config_clone.session.lock().await;
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
                                let msgs = session.peek_messages("main");
                                if msgs.is_empty() { "No messages.".to_string() }
                                else { msgs.iter().map(|m| format!("[{}] from {}: {}", m.timestamp, m.from, m.content)).collect::<Vec<_>>().join("\n") }
                            } else {
                                let table = crate::typeck::build_symbol_table(&session.program);
                                crate::typeck::handle_query(&session.program, &table, query, &session.runtime.http_routes)
                            };
                            feedback_details.push(format!("{query}:\n{response}"));
                        }
                    }
                }
                // Check for messages from agents addressed to main
                let inbox = session.drain_messages("main");
                if !inbox.is_empty() {
                    let inbox_text = inbox.iter()
                        .map(|m| format!("[from {}] {}", m.from, m.content))
                        .collect::<Vec<_>>().join("\n");
                    feedback_details.push(format!("Agent messages:\n{inbox_text}"));
                    let _ = tx.send(serde_json::json!({"type": "result", "message": format!("Messages from agents: {}", inbox.len()), "success": true})).await;
                }

                // Note untested functions in feedback so the AI knows what needs tests
                if session.program.require_modules {
                    let all_fns: Vec<String> = {
                        let mut fns = Vec::new();
                        for m in &session.program.modules {
                            for f in &m.functions {
                                let qname = format!("{}.{}", m.name, f.name);
                                if f.body.len() > 2 && !session.is_function_tested(&qname) {
                                    fns.push(qname);
                                }
                            }
                        }
                        fns
                    };
                    if !all_fns.is_empty() {
                        feedback_details.push(format!(
                            "Untested functions (blocked from !eval): {}",
                            all_fns.join(", ")
                        ));
                    }
                }
            }

            // Build plan status summary for feedback
            let plan_summary = {
                let session = config_clone.session.lock().await;
                let in_progress: Vec<_> = session.meta.plan.iter().filter(|s| matches!(s.status, crate::session::PlanStatus::InProgress)).collect();
                let pending: Vec<_> = session.meta.plan.iter().filter(|s| matches!(s.status, crate::session::PlanStatus::Pending)).collect();
                let failed: Vec<_> = session.meta.plan.iter().filter(|s| matches!(s.status, crate::session::PlanStatus::Failed)).collect();
                let total = session.meta.plan.len();
                let done = session.meta.plan.iter().filter(|s| matches!(s.status, crate::session::PlanStatus::Done)).count();
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

            if has_errors {
                let errors: Vec<&str> = feedback_details.iter()
                    .filter(|d| d.starts_with("ERROR:") || d.starts_with("FAIL:") || d.contains("[FAILED]"))
                    .map(|s| s.as_str()).collect();
                let _ = tx.send(serde_json::json!({"type": "feedback", "message": format!("Errors found ({} issues), retrying...", errors.len())})).await;
                let feedback = format!(
                    "Results:\n{}\n\n{}\n\nFix the errors and continue.",
                    feedback_details.join("\n"),
                    plan_summary
                );
                log_activity(&config_clone.log_file, "feedback", &feedback).await;
                log_training_data(&config_clone.training_log, &config_clone.llm_model, &last_context, &output.thinking, &code, &feedback_details, true, train_tests_passed, train_tests_failed).await;
                last_context = feedback.clone();
                messages.push(crate::llm::ChatMessage::user(feedback));
            } else {
                let results_section = if feedback_details.is_empty() {
                    String::new()
                } else {
                    format!("Results:\n{}\n\n", feedback_details.join("\n"))
                };
                let feedback = format!("{}{}", results_section, plan_summary);
                log_activity(&config_clone.log_file, "feedback", &feedback).await;
                log_training_data(&config_clone.training_log, &config_clone.llm_model, &last_context, &output.thinking, &code, &feedback_details, false, train_tests_passed, train_tests_failed).await;
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
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    let path = uri.path();
    let method_str = method.as_str();

    // Skip /api/ paths and the root — those are handled by explicit routes
    if path.starts_with("/api/") || path == "/" {
        return (StatusCode::NOT_FOUND, "not found").into_response();
    }

    // Look up a matching registered route
    let session = config.session.lock().await;
    let route = session
        .runtime
        .http_routes
        .iter()
        .find(|r| r.method == method_str && r.path == path);

    let Some(route) = route else {
        return (StatusCode::NOT_FOUND, format!("no Adapsis route for {method_str} {path}")).into_response();
    };

    let handler_fn = route.handler_fn.clone();

    // Check the handler function exists
    let func = session.program.get_function(&handler_fn);
    if func.is_none() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("route handler function `{handler_fn}` not found in program"),
        )
            .into_response();
    }

    let body_str = String::from_utf8_lossy(&body).to_string();

    // Clone program and drop the session lock before blocking eval
    let program = session.program.clone();
    drop(session);

    eprintln!("[webhook] {method_str} {path} -> {handler_fn}({} bytes)", body_str.len());

    // Evaluate the handler function with the body as a String argument
    let eval_result = tokio::task::spawn_blocking(move || {
        let func = program
            .get_function(&handler_fn)
            .ok_or_else(|| anyhow::anyhow!("function `{handler_fn}` not found"))?;
        let mut env = eval::Env::new();
        env.populate_shared_from_program(&program);
        let input = eval::Value::String(body_str);
        eval::bind_input_to_params(&program, func, &input, &mut env);
        eval::eval_function_body_pub(&program, &func.body, &mut env)
    })
    .await;

    match eval_result {
        Ok(Ok(val)) => {
            // Extract the raw string for HTTP response (no JSON quoting).
            // Infer content-type from the response body.
            let response_body = match &val {
                eval::Value::String(s) => s.clone(),
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
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("handler error: {e}"),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("task error: {e}"),
        )
            .into_response(),
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
) -> axum::response::sse::Sse<impl futures::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>> {
    use axum::response::sse::{Event, KeepAlive};
    let mut rx = config.event_broadcast.subscribe();
    let stream = async_stream::stream! {
        while let Ok(event) = rx.recv().await {
            let data = serde_json::to_string(&event).unwrap_or_default();
            yield Ok(Event::default().data(data));
        }
    };
    axum::response::sse::Sse::new(stream).keep_alive(KeepAlive::default())
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

    /// Helper: build a minimal AppConfig for testing multi-session endpoints.
    fn test_config() -> AppConfig {
        let session = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::session::Session::new(),
        ));
        let (trigger_tx, _trigger_rx) = tokio::sync::mpsc::channel::<String>(1);
        AppConfig {
            session,
            program: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::ast::Program::default(),
            )),
            meta: std::sync::Arc::new(tokio::sync::Mutex::new(
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
