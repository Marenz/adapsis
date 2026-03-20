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
    pub project_dir: String,
    pub io_sender: Option<tokio::sync::mpsc::Sender<crate::coroutine::IoRequest>>,
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
    eprintln!("[web:mutate] {}", req.source.chars().take(100).collect::<String>());
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

pub async fn ask(
    State(config): State<AppConfig>,
    Json(req): Json<AskRequest>,
) -> Json<AskResponse> {
    eprintln!("\n[web:user] {}", req.message);
    let llm = crate::llm::LlmClient::new_with_model(&config.llm_url, &config.llm_model);

    let mut results: Vec<MutationResult> = vec![];
    let mut test_results: Vec<TestCaseResult> = vec![];
    let mut code = String::new();
    let mut reply_text = String::new();
    let mut has_errors = false;

    let system_prompt = {
        let base = crate::prompt::system_prompt();
        format!(
            "{base}\n\n## ForgeOS Interactive Mode\n\
             Program state PERSISTS. Do NOT resend existing types/functions.\n\
             Only send NEW code or modifications.\n\
             IO builtins work as tools (minimal function + !eval) or building blocks.\n\
             If an eval fails, the error will be shown and you'll be asked to fix it."
        )
    };

    // Add to conversation history and build messages
    let messages = {
        let mut session = config.session.lock().await;
        if session.chat_messages.is_empty() {
            session.chat_messages.push(crate::session::ChatMessage {
                role: "system".to_string(),
                content: system_prompt,
            });
        }
        let context = format!(
            "Working directory: {}\n{}\nUser: {}",
            config.project_dir,
            crate::validator::program_summary_compact(&session.program),
            req.message
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

    // Single LLM call
    let output = match llm.generate(messages).await {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[web:error] LLM: {e}");
            return Json(AskResponse {
                reply: format!("LLM error: {e}"),
                code: String::new(),
                results: vec![],
                test_results: vec![],
                has_errors: false,
            });
        }
    };

    // Extract code and reply
    code = output.code.clone();
    if !output.thinking.is_empty() {
        reply_text.push_str(&output.thinking);
        eprintln!("[web:think] {}...", output.thinking.chars().take(150).collect::<String>());
    }
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

    if !code.is_empty() {
        eprintln!("[web:code]\n{code}");
    }

    if !code.is_empty() && code.trim() != "DONE" {
        let mut session = config.session.lock().await;

        if let Ok(ops) = crate::parser::parse(&code) {
            // Remove duplicates
            for op in &ops {
                match op {
                    crate::parser::Operation::Function(f) => {
                        session.program.functions.retain(|existing| existing.name != f.name);
                    }
                    crate::parser::Operation::Type(t) => {
                        let name = t.name.clone();
                        session.program.types.retain(|existing| existing.name() != name);
                    }
                    _ => {}
                }
            }

            let has_mutations = ops.iter().any(|op| !matches!(op,
                crate::parser::Operation::Test(_) | crate::parser::Operation::Trace(_)
                | crate::parser::Operation::Eval(_) | crate::parser::Operation::Query(_)));

            if has_mutations {
                match session.apply(&code) {
                    Ok(res) => {
                        for (msg, ok) in res {
                            eprintln!("[web:{}] {msg}", if ok { "ok" } else { "err" });
                            if !ok { has_errors = true; }
                            results.push(MutationResult { message: msg, success: ok });
                        }
                    }
                    Err(e) => {
                        eprintln!("[web:err] {e}");
                        has_errors = true;
                        results.push(MutationResult { message: format!("{e}"), success: false });
                    }
                }
            }

            for op in &ops {
                match op {
                    crate::parser::Operation::Test(test) => {
                        for case in &test.cases {
                            match crate::eval::eval_test_case(&session.program, &test.function_name, case) {
                                Ok(msg) => { eprintln!("[web:pass] {msg}"); test_results.push(TestCaseResult { message: msg, pass: true }); }
                                Err(e) => { eprintln!("[web:fail] {e}"); has_errors = true; test_results.push(TestCaseResult { message: format!("{e}"), pass: false }); }
                            }
                        }
                    }
                    crate::parser::Operation::Eval(ev) => {
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
                                    Ok(Err(e)) => { has_errors = true; (format!("eval error: {e}"), false) }
                                    Err(e) => { has_errors = true; (format!("eval task error: {e}"), false) }
                                };
                                eprintln!("[web:eval] {msg}");
                                results.push(MutationResult { message: msg, success });

                                session = config.session.lock().await;
                            }
                        } else {
                            match crate::eval::eval_compiled_or_interpreted(&session.program, &ev.function_name, &ev.input) {
                                Ok((result, compiled)) => {
                                    let tag = if compiled { " [compiled]" } else { "" };
                                    let msg = format!("eval {}() = {result}{tag}", ev.function_name);
                                    eprintln!("[web:eval] {msg}");
                                    results.push(MutationResult { message: msg, success: true });
                                }
                                Err(e) => {
                                    has_errors = true;
                                    let msg = format!("eval error: {e}");
                                    eprintln!("[web:eval:err] {msg}");
                                    results.push(MutationResult { message: msg, success: false });
                                }
                            }
                        }
                    }
                    crate::parser::Operation::Query(query) => {
                        let table = crate::typeck::build_symbol_table(&session.program);
                        let response = crate::typeck::handle_query(&session.program, &table, query);
                        eprintln!("[web:query] {}", response.chars().take(100).collect::<String>());
                        results.push(MutationResult { message: response, success: true });
                    }
                    _ => {}
                }
            }
        } else if let Err(e) = crate::parser::parse(&code) {
            has_errors = true;
            let msg = format!("Parse error: {e}");
            eprintln!("[web:parse:err] {msg}");
            results.push(MutationResult { message: msg, success: false });
        }
    }

    // Save assistant response to conversation history
    {
        let mut session = config.session.lock().await;
        let summary = if code.is_empty() {
            reply_text.chars().take(500).collect::<String>()
        } else {
            let result_summary = results.iter()
                .map(|r| format!("{}: {}", if r.success { "OK" } else { "ERROR" }, r.message))
                .collect::<Vec<_>>().join("\n");
            format!("{}\n<code>\n{}\n</code>\nResults:\n{}",
                reply_text.chars().take(200).collect::<String>(),
                code.chars().take(500).collect::<String>(),
                result_summary)
        };
        session.chat_messages.push(crate::session::ChatMessage {
            role: "assistant".to_string(),
            content: summary,
        });
        if session.chat_messages.len() > 50 {
            let system = session.chat_messages[0].clone();
            let start = session.chat_messages.len() - 49;
            let keep: Vec<_> = session.chat_messages[start..].to_vec();
            session.chat_messages = vec![system];
            session.chat_messages.extend(keep);
        }
    }

    eprintln!("[web:done] errors={has_errors}");

    Json(AskResponse {
        reply: reply_text,
        code,
        results,
        test_results,
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
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let code = output.status.code().unwrap_or(-1);
            Json(OpenCodeResponse {
                stdout,
                stderr,
                exit_code: code,
                success: code == 0,
            })
        }
        Err(e) => Json(OpenCodeResponse {
            stdout: String::new(),
            stderr: format!("Failed to run opencode: {e}"),
            exit_code: -1,
            success: false,
        }),
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
pub fn router_with_llm(config: AppConfig) -> axum::Router {
    use axum::routing::{get, post};

    let session_routes = axum::Router::new()
        .route("/api/mutate", post(mutate))
        .route("/api/test", post(test_fn))
        .route("/api/query", post(query))
        .route("/api/status", get(status))
        .route("/api/program", get(program))
        .route("/api/history", post(history))
        .route("/api/rewind", post(rewind))
        .with_state(config.session.clone());

    let config_routes = axum::Router::new()
        .route("/api/eval", post(eval_fn))
        .route("/api/ask", post(ask))
        .route("/api/opencode", post(opencode_task))
        .with_state(config);

    session_routes.merge(config_routes)
}
