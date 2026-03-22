//! HTTP API for ForgeOS — programmatic access to the session.
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

/// Extended state for the API, including LLM and OpenCode configuration.
#[derive(Clone)]
pub struct AppConfig {
    pub session: SharedSession,
    pub llm_url: String,
    pub llm_model: String,
    pub llm_api_key: Option<String>,
    pub project_dir: String,
    pub io_sender: Option<tokio::sync::mpsc::Sender<crate::coroutine::IoRequest>>,
    /// Channel for self-triggering: system events that should invoke the AI
    pub self_trigger: tokio::sync::mpsc::Sender<String>,
    /// Task registry for tracking spawned async tasks
    pub task_registry: Option<crate::coroutine::TaskRegistry>,
    /// Structured log file for AI activity
    pub log_file: Option<std::sync::Arc<tokio::sync::Mutex<tokio::fs::File>>>,
    /// JIT compilation cache — reuses compiled modules across evals when revision unchanged
    pub jit_cache: crate::eval::JitCache,
    /// Broadcast channel for SSE events — all activity visible to all subscribers (web UI)
    pub event_broadcast: tokio::sync::broadcast::Sender<serde_json::Value>,
    /// Directory where !opencode runs and builds (fixed checkout, ForgeOS must run from here)
    pub opencode_git_dir: String,
    /// Sequential lock for !opencode — only one at a time
    pub opencode_lock: std::sync::Arc<tokio::sync::Mutex<()>>,
    /// Maximum iterations per AI request
    pub max_iterations: usize,
    /// JSONL training data log — one entry per iteration with input/output/outcome
    pub training_log: Option<std::sync::Arc<tokio::sync::Mutex<tokio::fs::File>>>,
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

#[derive(Serialize, Clone)]
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
        Ok(results) => Json(MutateResponse {
            revision: session.revision,
            results: results
                .into_iter()
                .map(|(message, success)| MutationResult { message, success })
                .collect(),
        }),
        Err(e) => Json(MutateResponse {
            revision: session.revision,
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
    pub input: String,
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
    eprintln!("[web:eval] {} {}", req.function, req.input);
    let mut session = config.session.lock().await;

    let input_source = format!("!eval {} {}", req.function, req.input);
    let operations = match parser::parse(&input_source) {
        Ok(ops) => ops,
        Err(e) => {
            return Json(EvalResponse {
                result: format!("parse error: {e}"),
                success: false,
                compiled: None,
            });
        }
    };

    for op in &operations {
        if let parser::Operation::Eval(ev) = op {
            // Block eval of untested functions (>2 statements) in ForgeOS mode
            if session.program.require_modules {
                if let Some(func) = session.program.get_function(&ev.function_name) {
                    if func.body.len() > 2 && !session.tested_functions.contains(&ev.function_name) {
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
                        env.set("__coroutine_handle", eval::Value::CoroutineHandle(handle));
                        let input_val = eval::eval_parser_expr_standalone(&input)?;
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

            match eval::eval_compiled_or_interpreted_cached(&session.program, &ev.function_name, &ev.input, Some(&config.jit_cache), session.revision) {
                Ok((result, compiled)) => {
                    session.record_eval(&ev.function_name, &req.input, &result);
                    return Json(EvalResponse {
                        result,
                        success: true,
                        compiled: Some(compiled),
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
    }

    Json(EvalResponse {
        result: "no eval operation found".to_string(),
        success: false,
        compiled: None,
    })
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

#[derive(Serialize, Clone)]
pub struct TestCaseResult {
    pub message: String,
    pub pass: bool,
}

pub async fn test_fn(
    State(config): State<AppConfig>,
    Json(req): Json<TestRequest>,
) -> Json<TestResponse> {
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
                        let mocks = session.io_mocks.clone();
                        let sender = sender.clone();

                        drop(session); // release lock before blocking

                        let result = eval::eval_test_case_async(
                            &program, &fn_name, &case, &mocks, sender,
                        ).await;

                        session = config.session.lock().await;
                        result
                    } else {
                        // No IO sender — fall back to mock-only execution
                        eval::eval_test_case_with_mocks(
                            &session.program, &test.function_name, case, &session.io_mocks,
                        )
                    }
                } else {
                    // Sync function — run directly
                    eval::eval_test_case_with_mocks(
                        &session.program, &test.function_name, case, &session.io_mocks,
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
            if passed > 0 && failed == 0 {
                session.tested_functions.insert(test.function_name.clone());
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
    } else {
        let table = typeck::build_symbol_table(&session.program);
        typeck::handle_query(&session.program, &table, &req.query)
    };
    session.record_query(&req.query, &response);
    Json(QueryResponse { response })
}

#[derive(Serialize)]
pub struct StatusResponse {
    pub revision: usize,
    pub mutations: usize,
    pub history_entries: usize,
    pub functions: Vec<String>,
    pub types: Vec<String>,
    pub program_summary: String,
    pub plan: Vec<PlanStepResponse>,
}

#[derive(Serialize, Clone)]
pub struct PlanStepResponse {
    pub description: String,
    pub status: String,
}

pub async fn status(State(session): State<SharedSession>) -> Json<StatusResponse> {
    let session = session.lock().await;
    let plan = session.plan.iter().map(|s| PlanStepResponse {
        description: s.description.clone(),
        status: match s.status {
            crate::session::PlanStatus::Pending => "pending",
            crate::session::PlanStatus::InProgress => "in_progress",
            crate::session::PlanStatus::Done => "done",
            crate::session::PlanStatus::Failed => "failed",
        }.to_string(),
    }).collect();
    Json(StatusResponse {
        revision: session.revision,
        mutations: session.mutations.len(),
        history_entries: session.history.len(),
        plan,
        functions: session
            .program
            .functions
            .iter()
            .map(|f| f.name.clone())
            .collect(),
        types: session.program.types.iter().map(|t| t.name().to_string()).collect(),
        program_summary: validator::program_summary(&session.program),
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
    State(session): State<SharedSession>,
    Json(req): Json<HistoryRequest>,
) -> Json<HistoryResponse> {
    let session = session.lock().await;
    let limit = req.limit.unwrap_or(20);
    Json(HistoryResponse {
        formatted: session.format_recent_history(limit),
        mutations: session.mutations.clone(),
        history: session.history.clone(),
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
    State(session): State<SharedSession>,
    Json(req): Json<RewindRequest>,
) -> Json<RewindResponse> {
    let mut session = session.lock().await;
    match session.rewind_to(req.revision) {
        Ok(()) => Json(RewindResponse {
            revision: session.revision,
            success: true,
            message: format!("rewound to revision {}", req.revision),
        }),
        Err(e) => Json(RewindResponse {
            revision: session.revision,
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

pub async fn program(State(session): State<SharedSession>) -> Json<ProgramResponse> {
    let session = session.lock().await;

    let types = session.program.types.iter().map(|td| {
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

    let functions = session.program.functions.iter().map(|f| {
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

    let modules = session.program.modules.iter().map(|m| {
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
        revision: session.revision,
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

#[derive(Serialize)]
pub struct AskResponse {
    pub reply: String,
    pub code: String,
    pub results: Vec<MutationResult>,
    pub test_results: Vec<TestCaseResult>,
    pub has_errors: bool,
}

/// Write a structured log entry to the AI activity log.
/// Channel wrapper that sends to both the response mpsc and the broadcast channel.
struct EventSender {
    tx: tokio::sync::mpsc::Sender<serde_json::Value>,
    broadcast: tokio::sync::broadcast::Sender<serde_json::Value>,
}

impl EventSender {
    async fn send(&self, event: serde_json::Value) {
        let _ = self.broadcast.send(event.clone());
        let _ = self.tx.send(event).await;
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
fn format_tasks(registry: &Option<crate::coroutine::TaskRegistry>) -> String {
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
    eprintln!("\n[web:user] {}", req.message);
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
        let identity = crate::prompt::forgeos_identity();
        format!("{base}\n\n{builtins}\n\n{identity}")
    };

    // Build messages from conversation history
    let mut messages = {
        let mut session = config.session.lock().await;
        if session.chat_messages.is_empty() {
            session.chat_messages.push(crate::session::ChatMessage {
                role: "system".to_string(),
                content: system_prompt,
            });
        }
        let (plan_ctx, needs_plan) = build_plan_context(&session.plan);
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
        session.chat_messages.push(crate::session::ChatMessage {
            role: "user".to_string(),
            content: context,
        });
        session.chat_messages.iter().map(|m| match m.role.as_str() {
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
            if !reply_text.is_empty() { reply_text.push_str("\n\n"); }
            reply_text.push_str(clean);
        }
        if !output.thinking.is_empty() {
            eprintln!("[web:think] {}...", output.thinking.chars().take(100).collect::<String>());
        }

        // Check for DONE or no code (AI is asking a question / responding with text)
        if code.trim() == "DONE" {
            eprintln!("[web:done] model said DONE at iteration {}", iteration + 1);
            break;
        }
        if code.is_empty() {
            // No code block = AI is responding with text only (question or explanation)
            eprintln!("[web:text-only] no code block, stopping");
            break;
        }

        eprintln!("[web:code]\n{}", code.chars().take(200).collect::<String>());
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
                    if session.revision > 0 {
                        let prev = session.revision - 1;
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
                                session.plan = steps.iter().map(|s| crate::session::PlanStep {
                                    description: s.clone(),
                                    status: crate::session::PlanStatus::Pending,
                                }).collect();
                                iter_results.push(MutationResult { message: format!("Plan set: {} steps", steps.len()), success: true });
                            }
                            crate::parser::PlanAction::Progress(n) => {
                                let idx = n.saturating_sub(1);
                                if let Some(step) = session.plan.get_mut(idx) {
                                    step.status = crate::session::PlanStatus::Done;
                                    iter_results.push(MutationResult { message: format!("Step {n} done: {}", step.description), success: true });
                                }
                            }
                            crate::parser::PlanAction::Fail(n) => {
                                let idx = n.saturating_sub(1);
                                if let Some(step) = session.plan.get_mut(idx) {
                                    step.status = crate::session::PlanStatus::Failed;
                                    iter_results.push(MutationResult { message: format!("Step {n} failed: {}", step.description), success: true });
                                }
                            }
                            crate::parser::PlanAction::Show => {
                                let plan_str = session.plan.iter().enumerate().map(|(i, s)| {
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
                                        let mocks = session.io_mocks.clone();
                                        let sender = sender.clone();
                                        drop(session);
                                        let result = crate::eval::eval_test_case_async(
                                            &program, &fn_name, &case, &mocks, sender,
                                        ).await;
                                        session = config.session.lock().await;
                                        result
                                    } else {
                                        crate::eval::eval_test_case_with_mocks(
                                            &session.program, &test.function_name, case, &session.io_mocks,
                                        )
                                    }
                                } else {
                                    crate::eval::eval_test_case_with_mocks(
                                        &session.program, &test.function_name, case, &session.io_mocks,
                                    )
                                };
                                match case_result {
                                    Ok(msg) => { eprintln!("[web:pass] {msg}"); iter_test_results.push(TestCaseResult { message: msg, pass: true }); }
                                    Err(e) => { all_passed = false; eprintln!("[web:fail] {e}"); iter_has_errors = true; iter_test_results.push(TestCaseResult { message: format!("{e}"), pass: false }); }
                                }
                            }
                            if all_passed && !test.cases.is_empty() {
                                session.tested_functions.insert(test.function_name.clone());
                            }
                        }
                        crate::parser::Operation::Eval(ev) => {
                            // Block eval of untested functions in ForgeOS mode
                            if session.program.require_modules {
                                if let Some(func) = session.program.get_function(&ev.function_name) {
                                    if func.body.len() > 2 && !session.tested_functions.contains(&ev.function_name) {
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
                                        env.set("__coroutine_handle", crate::eval::Value::CoroutineHandle(handle));
                                        let input_val = crate::eval::eval_parser_expr_standalone(&input)?;
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
                                match crate::eval::eval_compiled_or_interpreted_cached(&session.program, &ev.function_name, &ev.input, Some(&config.jit_cache), session.revision) {
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
                            } else {
                                let table = crate::typeck::build_symbol_table(&session.program);
                                crate::typeck::handle_query(&session.program, &table, query)
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
                                        let input_source = format!("!eval {fn_name} {fn_args}");
                                        match crate::parser::parse(&input_source) {
                                            Ok(ops) => {
                                                let mut result_str = String::new();
                                                for op in &ops {
                                                    if let crate::parser::Operation::Eval(ev) = op {
                                                        match crate::eval::eval_compiled_or_interpreted_cached(
                                                            &session.program, &ev.function_name, &ev.input,
                                                            Some(&watch_jit_cache), session.revision,
                                                        ) {
                                                            Ok((r, _)) => result_str = r,
                                                            Err(e) => result_str = format!("error: {e}"),
                                                        }
                                                    }
                                                }
                                                result_str
                                            }
                                            Err(e) => format!("parse error: {e}"),
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
                                    "{}\n\n{}\n\nYou are agent '{agent_name}'.\n{scope_desc}\n\nYour task:\n{agent_task}\n\nWork step by step. Always include a <code> block with Forge code. When done, respond with DONE in a <code> block.",
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
                                    if code.trim() == "DONE" || code.is_empty() {
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
                                                agent_messages.push(crate::llm::ChatMessage::user(format!("Results:\n{feedback}\nContinue or DONE.")));
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
                                    session.chat_messages.push(crate::session::ChatMessage {
                                        role: "system".to_string(),
                                        content: format!("Agent '{agent_name}' completed and merged successfully."),
                                    });
                                    if let Some(s) = session.agent_log.iter_mut().rev().find(|s| s.name == agent_name && s.status == "running") {
                                        s.status = "merged".to_string();
                                        s.message = "completed and merged".to_string();
                                    }
                                } else {
                                    eprintln!("[agent:{agent_name}] merge conflicts: {:?}", conflicts);
                                    session.chat_messages.push(crate::session::ChatMessage {
                                        role: "system".to_string(),
                                        content: format!("Agent '{agent_name}' finished but had merge conflicts:\n{}", conflicts.join("\n")),
                                    });
                                    if let Some(s) = session.agent_log.iter_mut().rev().find(|s| s.name == agent_name && s.status == "running") {
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
                            session.agent_log.push(crate::session::AgentStatus {
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
                                std::time::Duration::from_secs(1800),
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
                                        .arg("build").current_dir(&config.project_dir).output().await;
                                    match build {
                                        Ok(b) if b.status.success() => {
                                            iter_results.push(MutationResult {
                                                message: "OpenCode + rebuild successful. Restart to apply.".to_string(), success: true,
                                            });
                                            tokio::spawn(async {
                                                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                                                let exe = std::env::current_exe().unwrap_or_default();
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
                "Results:\n{}\n\nIf the task is complete, respond with DONE. Otherwise continue with the next step.",
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
        session.chat_messages.push(crate::session::ChatMessage {
            role: "assistant".to_string(), content: summary,
        });
        if session.chat_messages.len() > 50 {
            let system = session.chat_messages[0].clone();
            let start = session.chat_messages.len() - 49;
            let keep: Vec<_> = session.chat_messages[start..].to_vec();
            session.chat_messages = vec![system];
            session.chat_messages.extend(keep);
        }
    }

    let has_errors = all_results.iter().any(|r| !r.success) || all_test_results.iter().any(|r| !r.pass);
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
    let project_dir = &config.project_dir;

    // Run opencode with the task in the project directory
    let result = tokio::process::Command::new("opencode")
        .arg("run")
        .arg("--format")
        .arg("json")
        .arg(&req.task)
        .current_dir(project_dir)
        .output()
        .await;

    match result {
        Ok(output) => {
            let raw = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let code = output.status.code().unwrap_or(-1);

            // Parse JSON events to extract text and tool results
            let mut text_parts = Vec::new();
            let mut tool_results = Vec::new();
            for line in raw.lines() {
                if let Ok(event) = serde_json::from_str::<serde_json::Value>(line) {
                    match event.get("type").and_then(|t| t.as_str()) {
                        Some("text") => {
                            if let Some(text) = event.pointer("/part/text").and_then(|t| t.as_str()) {
                                text_parts.push(text.to_string());
                            }
                        }
                        Some("tool_result") => {
                            if let Some(content) = event.pointer("/part/content") {
                                tool_results.push(content.to_string());
                            }
                        }
                        _ => {}
                    }
                }
            }

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

            Json(OpenCodeResponse {
                stdout: summary,
                stderr,
                exit_code: code,
                success: code == 0,
            })
        }
        Err(e) => {
            eprintln!("[web:opencode:err] {e}");
            Json(OpenCodeResponse {
                stdout: String::new(),
                stderr: format!("Failed to run opencode: {e}"),
                exit_code: -1,
                success: false,
            })
        }
    }
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
pub async fn agents(State(session): State<SharedSession>) -> Json<Vec<crate::session::AgentStatus>> {
    let session = session.lock().await;
    Json(session.agent_log.clone())
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
        let tx = EventSender { tx: raw_tx, broadcast: config_clone.event_broadcast.clone() };
        let llm = crate::llm::LlmClient::new_with_model_and_key(
            &config_clone.llm_url, &config_clone.llm_model, config_clone.llm_api_key.clone(),
        );

        let _ = tx.send(serde_json::json!({"type": "start", "message": req.message})).await;

        let system_prompt = {
            let base = crate::prompt::system_prompt();
            let builtins = crate::builtins::format_for_prompt();
            let identity = crate::prompt::forgeos_identity();
            format!("{base}\n\n{builtins}\n\n{identity}")
        };

        let mut messages = {
            let mut session = config_clone.session.lock().await;
            if session.chat_messages.is_empty() {
                session.chat_messages.push(crate::session::ChatMessage {
                    role: "system".to_string(), content: system_prompt,
                });
            }
            let (plan_ctx, needs_plan) = build_plan_context(&session.plan);
            let plan_hint = if needs_plan {
                "\n\nYour previous plan is completed (or none exists). Create a new plan with !plan set for this task before writing code. You can update it anytime with !plan set / !plan done N."
            } else { "" };
            let context = format!("Working directory: {}\n{}{}\nUser: {}{}",
                config_clone.project_dir,
                crate::validator::program_summary_compact(&session.program),
                plan_ctx, req.message, plan_hint);
            log_activity(&config_clone.log_file, "user", &context).await;
            session.chat_messages.push(crate::session::ChatMessage {
                role: "user".to_string(), content: context,
            });
            session.chat_messages.iter().map(|m| match m.role.as_str() {
                "system" => crate::llm::ChatMessage::system(m.content.clone()),
                "assistant" => crate::llm::ChatMessage::assistant(&m.content),
                _ => crate::llm::ChatMessage::user(m.content.clone()),
            }).collect::<Vec<_>>()
        };

        let max_iterations = config_clone.max_iterations;
        let mut last_context = req.message.clone();
        for iteration in 0..max_iterations {
            let _ = tx.send(serde_json::json!({"type": "iteration", "n": iteration + 1})).await;
            log_activity(&config_clone.log_file, "iter", &format!("iteration {}/{}", iteration + 1, max_iterations)).await;

            // Retry LLM calls on transient errors (network, timeout)
            let output = {
                let mut last_err = String::new();
                let mut result = None;
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

            // Strip "DONE" from code — the AI sometimes appends it to a code block
            let raw_code = output.code.trim().to_string();
            let (code, is_done) = if raw_code.ends_with("\nDONE") || raw_code.ends_with("\n\nDONE") {
                (raw_code.rsplit_once('\n').map(|(before, _)| before.trim().to_string()).unwrap_or_default(), true)
            } else if raw_code == "DONE" || raw_code.is_empty() {
                (String::new(), true)
            } else {
                (raw_code, false)
            };

            if code.is_empty() {
                // Check for untested functions before accepting DONE
                if config_clone.session.lock().await.program.require_modules {
                    let session = config_clone.session.lock().await;
                    let untested: Vec<String> = session.program.modules.iter().flat_map(|m| {
                        m.functions.iter().filter_map(|f| {
                            let qname = format!("{}.{}", m.name, f.name);
                            if f.body.len() > 2 && !session.tested_functions.contains(&qname) {
                                Some(qname)
                            } else { None }
                        })
                    }).collect();
                    if !untested.is_empty() {
                        let challenge = format!(
                            "Cannot accept DONE: {} untested functions: {}. Write !test blocks for them.",
                            untested.len(), untested.join(", ")
                        );
                        log_activity(&config_clone.log_file, "done-rejected", &challenge).await;
                        let _ = tx.send(serde_json::json!({"type": "feedback", "message": "DONE rejected: untested functions"})).await;
                        messages.push(crate::llm::ChatMessage::user(challenge));
                        continue;
                    }
                }
                log_activity(&config_clone.log_file, "done", &format!("AI said DONE at iteration {}", iteration + 1)).await;
                let _ = tx.send(serde_json::json!({"type": "done"})).await;
                break;
            }
            // Code has content — process it, then check is_done after feedback

            log_activity(&config_clone.log_file, "code", &code).await;
            let _ = tx.send(serde_json::json!({"type": "code", "code": code})).await;

            // Apply code
            let mut session = config_clone.session.lock().await;
            let mut has_errors = false;
            let mut feedback_details: Vec<String> = Vec::new();
            let mut train_tests_passed: usize = 0;
            let mut train_tests_failed: usize = 0;

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
                                    session.plan = steps.iter().map(|s| crate::session::PlanStep {
                                        description: s.clone(), status: crate::session::PlanStatus::Pending,
                                    }).collect();
                                    let plan_json: Vec<serde_json::Value> = session.plan.iter().map(|s| {
                                        serde_json::json!({"description": s.description, "status": format!("{:?}", s.status).to_lowercase()})
                                    }).collect();
                                    let _ = tx.send(serde_json::json!({"type": "plan", "plan": plan_json})).await;
                                    let _ = tx.send(serde_json::json!({"type": "result", "message": format!("Plan: {} steps", steps.len()), "success": true})).await;
                                }
                                crate::parser::PlanAction::Progress(n) => {
                                    if let Some(step) = session.plan.get_mut(n.saturating_sub(1)) {
                                        step.status = crate::session::PlanStatus::Done;
                                        let plan_json: Vec<serde_json::Value> = session.plan.iter().map(|s| {
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
                        | crate::parser::Operation::OpenCode(_)));

                    if has_mutations {
                        match session.apply(&code) {
                            Ok(res) => {
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
                                            let mocks = session.io_mocks.clone();
                                            let sender = sender.clone();
                                            drop(session);
                                            let result = crate::eval::eval_test_case_async(
                                                &program, &fn_name, &case, &mocks, sender,
                                            ).await;
                                            session = config_clone.session.lock().await;
                                            result
                                        } else {
                                            crate::eval::eval_test_case_with_mocks(
                                                &session.program, &test.function_name, case, &session.io_mocks,
                                            )
                                        }
                                    } else {
                                        crate::eval::eval_test_case_with_mocks(
                                            &session.program, &test.function_name, case, &session.io_mocks,
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
                                    session.tested_functions.insert(test.function_name.clone());
                                }
                            }
                            crate::parser::Operation::Eval(ev) => {
                                // Block eval of untested functions (>2 statements) in ForgeOS mode
                                if session.program.require_modules {
                                    if let Some(func) = session.program.get_function(&ev.function_name) {
                                        if func.body.len() > 2 && !session.tested_functions.contains(&ev.function_name) {
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
                                            env.set("__coroutine_handle", crate::eval::Value::CoroutineHandle(handle));
                                            let input_val = crate::eval::eval_parser_expr_standalone(&input)?;
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
                                    match crate::eval::eval_compiled_or_interpreted_cached(&session.program, &ev.function_name, &ev.input, Some(&config_clone.jit_cache), session.revision) {
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
                                } else {
                                    let table = crate::typeck::build_symbol_table(&session.program);
                                    crate::typeck::handle_query(&session.program, &table, query)
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
                            crate::parser::Operation::Mock { operation, pattern, response } => {
                                session.io_mocks.push(crate::session::IoMock {
                                    operation: operation.clone(), pattern: pattern.clone(), response: response.clone(),
                                });
                                feedback_details.push(format!("mock: {operation} \"{pattern}\""));
                                let _ = tx.send(serde_json::json!({"type": "result", "message": format!("mock: {operation} \"{pattern}\""), "success": true})).await;
                            }
                            crate::parser::Operation::Unmock => {
                                let count = session.io_mocks.len();
                                session.io_mocks.clear();
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
                                    s.opencode_session_id.clone()
                                };

                                let recent_lines = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
                                let recent_for_stream = recent_lines.clone();
                                let oc_result = tokio::time::timeout(
                                    std::time::Duration::from_secs(1800),
                                    async {
                                        let mut cmd = tokio::process::Command::new("opencode");
                                        cmd.arg("run").arg("--format").arg("json")
                                            .arg("--attach").arg("http://localhost:4096")
                                            .arg("--dir").arg(&work_dir);
                                        // Continue existing session if we have one
                                        if let Some(sid) = &oc_session_id {
                                            cmd.arg("--session").arg(sid);
                                        }
                                        let full_task = format!("{task}. When done, create clean atomic git commits with descriptive messages for each logical change.");
                                        cmd.arg(&full_task);
                                        let mut child = cmd
                                            .current_dir(&work_dir)
                                            .stdout(std::process::Stdio::piped())
                                            .stderr(std::process::Stdio::piped())
                                            .spawn()?;

                                        let stdout = child.stdout.take().unwrap();
                                        let mut reader = BufReader::new(stdout).lines();
                                        while let Some(line) = reader.next_line().await? {
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
                                                            }
                                                        }
                                                    }
                                                    "tool_call" | "tool_result" => {
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
                                                            if s.opencode_session_id.as_deref() != Some(sid) {
                                                                s.opencode_session_id = Some(sid.to_string());
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

                                match oc_result {
                                    Ok(Ok(status)) if status.success() => {
                                        eprintln!("[web:opencode:stream:done] rebuilding...");
                                        log_activity(&config_clone.log_file, "opencode-done", "rebuilding...").await;
                                        let _ = tx.send(serde_json::json!({"type": "result", "message": "OpenCode done, rebuilding...", "success": true})).await;

                                        let build = tokio::process::Command::new("cargo")
                                            .arg("build").arg("--release").current_dir(&work_dir).output().await;
                                        match build {
                                            Ok(b) if b.status.success() => {
                                                feedback_details.push("OK: OpenCode + rebuild successful. Restarting...".to_string());
                                                log_activity(&config_clone.log_file, "opencode-restart", "rebuild successful, restarting").await;
                                                let _ = tx.send(serde_json::json!({"type": "result", "message": "OpenCode + rebuild successful. Restarting...", "success": true})).await;
                                                tokio::spawn(async {
                                                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                                                    let exe = std::env::current_exe().unwrap_or_default();
                                                    let args: Vec<String> = std::env::args().collect();
                                                    let _ = exec::execvp(&exe, &args);
                                                });
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
                            } else if query.trim() == "?inbox" || query.trim().starts_with("?inbox") {
                                let msgs = session.peek_messages("main");
                                if msgs.is_empty() { "No messages.".to_string() }
                                else { msgs.iter().map(|m| format!("[{}] from {}: {}", m.timestamp, m.from, m.content)).collect::<Vec<_>>().join("\n") }
                            } else {
                                let table = crate::typeck::build_symbol_table(&session.program);
                                crate::typeck::handle_query(&session.program, &table, query)
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
                                if f.body.len() > 2 && !session.tested_functions.contains(&qname) {
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
                let pending: Vec<_> = session.plan.iter().filter(|s| matches!(s.status, crate::session::PlanStatus::Pending | crate::session::PlanStatus::InProgress)).collect();
                let failed: Vec<_> = session.plan.iter().filter(|s| matches!(s.status, crate::session::PlanStatus::Failed)).collect();
                let total = session.plan.len();
                let done = session.plan.iter().filter(|s| matches!(s.status, crate::session::PlanStatus::Done)).count();
                if total == 0 {
                    "No plan set. Create one with !plan set.".to_string()
                } else if pending.is_empty() && failed.is_empty() {
                    format!("All {total} plan steps completed. Verify everything works, then DONE.")
                } else {
                    let mut msg = format!("Plan: {done}/{total} done.");
                    if !failed.is_empty() {
                        msg.push_str(&format!(" {} failed.", failed.len()));
                    }
                    // Show next 2 pending steps only
                    for s in pending.iter().take(2) {
                        msg.push_str(&format!(" Next: {}", s.description));
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
                // If the AI said DONE alongside this code and there were no errors, check untested
                if is_done {
                    if config_clone.session.lock().await.program.require_modules {
                        let session = config_clone.session.lock().await;
                        let untested: Vec<String> = session.program.modules.iter().flat_map(|m| {
                            m.functions.iter().filter_map(|f| {
                                let qname = format!("{}.{}", m.name, f.name);
                                if f.body.len() > 2 && !session.tested_functions.contains(&qname) {
                                    Some(qname)
                                } else { None }
                            })
                        }).collect();
                        if !untested.is_empty() {
                            let challenge = format!(
                                "Cannot accept DONE: {} untested functions: {}. Write !test blocks for them.",
                                untested.len(), untested.join(", ")
                            );
                            log_activity(&config_clone.log_file, "done-rejected", &challenge).await;
                            let _ = tx.send(serde_json::json!({"type": "feedback", "message": "DONE rejected: untested functions"})).await;
                            messages.push(crate::llm::ChatMessage::user(challenge));
                            continue;
                        }
                    }
                    log_activity(&config_clone.log_file, "done", &format!("AI said DONE (with code) at iteration {}", iteration + 1)).await;
                    let _ = tx.send(serde_json::json!({"type": "done"})).await;
                    break;
                }
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

pub fn router_with_llm(config: AppConfig) -> axum::Router {
    use axum::routing::{get, post};

    let session_routes = axum::Router::new()
        .route("/api/status", get(status))
        .route("/api/program", get(program))
        .route("/api/history", post(history))
        .route("/api/rewind", post(rewind))
        .route("/api/agents", get(agents))
        .with_state(config.session.clone());

    let config_routes = axum::Router::new()
        .route("/api/mutate", post(mutate))
        .route("/api/eval", post(eval_fn))
        .route("/api/test", post(test_fn))
        .route("/api/query", post(query))
        .route("/api/ask", post(ask))
        .route("/api/ask-stream", post(ask_stream))
        .route("/api/opencode", post(opencode_task))
        .route("/api/tasks", get(tasks))
        .route("/api/log", get(get_log))
        .route("/api/events", get(events_stream))
        .with_state(config);

    session_routes.merge(config_routes)
}

/// GET /api/events — SSE stream of all AI activity (subscribe from web UI).
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
