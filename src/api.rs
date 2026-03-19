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

/// Extended state for the API, including LLM configuration.
#[derive(Clone)]
pub struct AppConfig {
    pub session: SharedSession,
    pub llm_url: String,
    pub llm_model: String,
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

#[derive(Serialize)]
pub struct MutationResult {
    pub message: String,
    pub success: bool,
}

pub async fn mutate(
    State(session): State<SharedSession>,
    Json(req): Json<MutateRequest>,
) -> Json<MutateResponse> {
    let mut session = session.lock().await;
    match session.apply(&req.source) {
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
    State(session): State<SharedSession>,
    Json(req): Json<EvalRequest>,
) -> Json<EvalResponse> {
    let mut session = session.lock().await;

    // Parse the input as a test input expression
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
            match eval::eval_compiled_or_interpreted(&session.program, &ev.function_name, &ev.input) {
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

#[derive(Serialize)]
pub struct TestCaseResult {
    pub message: String,
    pub pass: bool,
}

pub async fn test_fn(
    State(session): State<SharedSession>,
    Json(req): Json<TestRequest>,
) -> Json<TestResponse> {
    let mut session = session.lock().await;
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
            for case in &test.cases {
                match eval::eval_test_case(&session.program, &test.function_name, case) {
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
    State(session): State<SharedSession>,
    Json(req): Json<QueryRequest>,
) -> Json<QueryResponse> {
    let mut session = session.lock().await;
    let table = typeck::build_symbol_table(&session.program);
    let response = typeck::handle_query(&session.program, &table, &req.query);
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
}

pub async fn status(State(session): State<SharedSession>) -> Json<StatusResponse> {
    let session = session.lock().await;
    Json(StatusResponse {
        revision: session.revision,
        mutations: session.mutations.len(),
        history_entries: session.history.len(),
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
        crate::ast::StatementKind::Spawn { call } => {
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

    Json(ProgramResponse {
        revision: session.revision,
        types,
        functions,
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
}

pub async fn ask(
    State(config): State<AppConfig>,
    Json(req): Json<AskRequest>,
) -> Json<AskResponse> {
    let llm = crate::llm::LlmClient::new_with_model(&config.llm_url, &config.llm_model);

    let mut session = config.session.lock().await;
    let context = format!(
        "{}\n\n{}\n\nUser request: {}",
        crate::validator::program_summary(&session.program),
        session.format_recent_history(10),
        req.message
    );

    let messages = vec![
        crate::llm::ChatMessage::system(crate::prompt::system_prompt()),
        crate::llm::ChatMessage::user(context),
    ];

    let output = match llm.generate(messages).await {
        Ok(o) => o,
        Err(e) => {
            return Json(AskResponse {
                reply: format!("LLM error: {e}"),
                code: String::new(),
                results: vec![],
                test_results: vec![],
            });
        }
    };

    // Only use code from <code> blocks — don't try to extract from prose
    let code = output.code.clone();

    // Build the reply text — combine thinking + prose, strip tags
    let mut reply_text = String::new();
    if !output.thinking.is_empty() {
        reply_text.push_str(&output.thinking);
        reply_text.push_str("\n\n");
    }
    // Get the raw text without <think> and <code> blocks
    let mut clean_text = output.text.clone();
    // Strip <think>...</think>
    while let Some(start) = clean_text.find("<think>") {
        if let Some(end) = clean_text.find("</think>") {
            clean_text.replace_range(start..end + 8, "");
        } else {
            break;
        }
    }
    // Strip <code>...</code>
    while let Some(start) = clean_text.find("<code>") {
        if let Some(end) = clean_text.find("</code>") {
            clean_text.replace_range(start..end + 7, "");
        } else {
            break;
        }
    }
    let clean_text = clean_text.trim();
    if !clean_text.is_empty() {
        reply_text.push_str(clean_text);
    }
    let reply_text = reply_text.trim().to_string();

    let mut results = vec![];
    let mut test_results = vec![];

    if !code.is_empty() && code.trim() != "DONE" {
        if let Ok(ops) = crate::parser::parse(&code) {
            let has_definitions = ops.iter().any(|op| {
                matches!(
                    op,
                    crate::parser::Operation::Module(_)
                        | crate::parser::Operation::Function(_)
                        | crate::parser::Operation::Type(_)
                )
            });

            // If the code has definitions, reset program state first
            // (the model sends the complete program each time)
            if has_definitions {
                session.program = crate::ast::Program::default();
            }

            let has_mutations = ops.iter().any(|op| {
                !matches!(
                    op,
                    crate::parser::Operation::Test(_)
                        | crate::parser::Operation::Trace(_)
                        | crate::parser::Operation::Eval(_)
                        | crate::parser::Operation::Query(_)
                )
            });

            if has_mutations {
                match session.apply(&code) {
                    Ok(res) => {
                        for (msg, ok) in res {
                            results.push(MutationResult { message: msg, success: ok });
                        }
                    }
                    Err(e) => {
                        results.push(MutationResult {
                            message: format!("{e}"),
                            success: false,
                        });
                    }
                }
            }

            // Run tests and evals
            for op in &ops {
                match op {
                    crate::parser::Operation::Test(test) => {
                        for case in &test.cases {
                            match crate::eval::eval_test_case(&session.program, &test.function_name, case) {
                                Ok(msg) => test_results.push(TestCaseResult { message: msg, pass: true }),
                                Err(e) => test_results.push(TestCaseResult { message: format!("{e}"), pass: false }),
                            }
                        }
                    }
                    crate::parser::Operation::Eval(ev) => {
                        match crate::eval::eval_compiled_or_interpreted(
                            &session.program,
                            &ev.function_name,
                            &ev.input,
                        ) {
                            Ok((result, compiled)) => {
                                let tag = if compiled { " [compiled]" } else { "" };
                                results.push(MutationResult {
                                    message: format!("eval {}() = {result}{tag}", ev.function_name),
                                    success: true,
                                });
                            }
                            Err(e) => {
                                results.push(MutationResult {
                                    message: format!("eval error: {e}"),
                                    success: false,
                                });
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    Json(AskResponse {
        reply: reply_text,
        code,
        results,
        test_results,
    })
}

/// Build the API router (without LLM support).
#[allow(dead_code)]
pub fn router(session: SharedSession) -> axum::Router {
    use axum::routing::{get, post};

    axum::Router::new()
        .route("/api/mutate", post(mutate))
        .route("/api/eval", post(eval_fn))
        .route("/api/test", post(test_fn))
        .route("/api/query", post(query))
        .route("/api/status", get(status))
        .route("/api/program", get(program))
        .route("/api/history", post(history))
        .route("/api/rewind", post(rewind))
        .with_state(session)
}

/// Build the full router with LLM support.
pub fn router_with_llm(config: AppConfig) -> axum::Router {
    use axum::routing::{get, post};

    // The session-only routes need SharedSession state
    let session_routes = axum::Router::new()
        .route("/api/mutate", post(mutate))
        .route("/api/eval", post(eval_fn))
        .route("/api/test", post(test_fn))
        .route("/api/query", post(query))
        .route("/api/status", get(status))
        .route("/api/program", get(program))
        .route("/api/history", post(history))
        .route("/api/rewind", post(rewind))
        .with_state(config.session.clone());

    // The ask route needs the full config
    let ask_route = axum::Router::new()
        .route("/api/ask", post(ask))
        .with_state(config);

    session_routes.merge(ask_route)
}
