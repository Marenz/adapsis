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
}

pub async fn ask(
    State(config): State<AppConfig>,
    Json(req): Json<AskRequest>,
) -> Json<AskResponse> {
    eprintln!("\n[web:user] {}", req.message);
    let llm = crate::llm::LlmClient::new_with_model(&config.llm_url, &config.llm_model);

    let max_iterations = 5;
    let mut all_results: Vec<MutationResult> = vec![];
    let mut all_test_results: Vec<TestCaseResult> = vec![];
    let mut all_code = String::new();
    let mut reply_text = String::new();

    // Build initial messages
    let system = {
        let base = crate::prompt::system_prompt();
        format!(
            "{base}\n\n## IMPORTANT: ForgeOS Interactive Mode\n\
             \n\
             The program state PERSISTS between responses. Do NOT resend existing types or functions.\n\
             Only send NEW code or modifications. Use !replace to modify existing functions.\n\
             \n\
             ## IO Builtins as Tools\n\
             \n\
             IO builtins serve two purposes:\n\
             1. As TOOLS: for quick answers, write a minimal function and !eval it immediately.\n\
             2. As BUILDING BLOCKS: for larger programs, compose them into proper functions.\n\
             \n\
             If an eval fails, FIX the issue and try again — don't stop.\n\
             Always keep going until the user's request is fulfilled."
        )
    };

    let initial_context = {
        let session = config.session.lock().await;
        format!(
            "Working directory: {}\n\n{}\n\n{}\n\nUser request: {}",
            config.project_dir,
            crate::validator::program_summary(&session.program),
            session.format_recent_history(10),
            req.message
        )
    };

    let mut messages = vec![
        crate::llm::ChatMessage::system(system),
        crate::llm::ChatMessage::user(initial_context),
    ];

    for iteration in 0..max_iterations {
        if iteration > 0 {
            eprintln!("[web:retry {iteration}/{max_iterations}]");
        }

        // Call LLM
        let output = match llm.generate(messages.clone()).await {
            Ok(o) => o,
            Err(e) => {
                eprintln!("[web:error] LLM: {e}");
                reply_text = format!("LLM error: {e}");
                break;
            }
        };

        messages.push(crate::llm::ChatMessage::assistant(&output.text));

        // Extract code and reply
        let code = output.code.clone();

        if !output.thinking.is_empty() {
            eprintln!("[web:think] {}...", output.thinking.chars().take(150).collect::<String>());
        }

        // Build reply from thinking + prose
        let mut iter_reply = String::new();
        if !output.thinking.is_empty() {
            iter_reply.push_str(&output.thinking);
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
            if !iter_reply.is_empty() { iter_reply.push_str("\n\n"); }
            iter_reply.push_str(clean);
        }
        if !iter_reply.is_empty() {
            if !reply_text.is_empty() { reply_text.push_str("\n\n---\n\n"); }
            reply_text.push_str(&iter_reply);
        }

        if code.is_empty() || code.trim() == "DONE" {
            eprintln!("[web:done] no code / DONE");
            break;
        }

        eprintln!("[web:code]\n{code}");
        if !all_code.is_empty() { all_code.push_str("\n\n// --- iteration ---\n"); }
        all_code.push_str(&code);

        // Apply code
        let mut iter_results: Vec<MutationResult> = vec![];
        let mut iter_test_results: Vec<TestCaseResult> = vec![];

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

            // Apply mutations
            let has_mutations = ops.iter().any(|op| {
                !matches!(op,
                    crate::parser::Operation::Test(_) | crate::parser::Operation::Trace(_)
                    | crate::parser::Operation::Eval(_) | crate::parser::Operation::Query(_))
            });

            if has_mutations {
                match session.apply(&code) {
                    Ok(res) => {
                        for (msg, ok) in res {
                            eprintln!("[web:{}] {msg}", if ok { "ok" } else { "err" });
                            iter_results.push(MutationResult { message: msg, success: ok });
                        }
                    }
                    Err(e) => {
                        eprintln!("[web:err] {e}");
                        iter_results.push(MutationResult { message: format!("{e}"), success: false });
                    }
                }
            }

            // Run tests, evals, queries
            for op in &ops {
                match op {
                    crate::parser::Operation::Test(test) => {
                        for case in &test.cases {
                            match crate::eval::eval_test_case(&session.program, &test.function_name, case) {
                                Ok(msg) => {
                                    eprintln!("[web:pass] {msg}");
                                    iter_test_results.push(TestCaseResult { message: msg, pass: true });
                                }
                                Err(e) => {
                                    eprintln!("[web:fail] {e}");
                                    iter_test_results.push(TestCaseResult { message: format!("{e}"), pass: false });
                                }
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

                                let msg = match &eval_result {
                                    Ok(Ok(val)) => { let m = format!("eval {}() = {val}", ev.function_name); eprintln!("[web:eval] {m}"); m }
                                    Ok(Err(e)) => { let m = format!("eval error: {e}"); eprintln!("[web:eval:err] {m}"); m }
                                    Err(e) => { let m = format!("eval task error: {e}"); eprintln!("[web:eval:err] {m}"); m }
                                };
                                let success = matches!(eval_result, Ok(Ok(_)));
                                iter_results.push(MutationResult { message: msg, success });

                                session = config.session.lock().await;
                            }
                        } else {
                            match crate::eval::eval_compiled_or_interpreted(&session.program, &ev.function_name, &ev.input) {
                                Ok((result, compiled)) => {
                                    let tag = if compiled { " [compiled]" } else { "" };
                                    let msg = format!("eval {}() = {result}{tag}", ev.function_name);
                                    eprintln!("[web:eval] {msg}");
                                    iter_results.push(MutationResult { message: msg, success: true });
                                }
                                Err(e) => {
                                    let msg = format!("eval error: {e}");
                                    eprintln!("[web:eval:err] {msg}");
                                    iter_results.push(MutationResult { message: msg, success: false });
                                }
                            }
                        }
                    }
                    crate::parser::Operation::Query(query) => {
                        let table = crate::typeck::build_symbol_table(&session.program);
                        let response = crate::typeck::handle_query(&session.program, &table, query);
                        eprintln!("[web:query] {}", response.chars().take(100).collect::<String>());
                        iter_results.push(MutationResult { message: response, success: true });
                    }
                    _ => {}
                }
            }
        } else if let Err(e) = crate::parser::parse(&code) {
            let msg = format!("Parse error: {e}");
            eprintln!("[web:parse:err] {msg}");
            iter_results.push(MutationResult { message: msg, success: false });
        }

        // Check for errors
        let has_errors = iter_results.iter().any(|r| !r.success) || iter_test_results.iter().any(|r| !r.pass);

        all_results.extend(iter_results);
        all_test_results.extend(iter_test_results);

        if !has_errors {
            eprintln!("[web:done] all passed, iteration {iteration}");
            break;
        }

        // Feed errors back to LLM
        let error_summary: String = all_results.iter()
            .filter(|r| !r.success)
            .map(|r| r.message.clone())
            .chain(all_test_results.iter().filter(|r| !r.pass).map(|r| r.message.clone()))
            .collect::<Vec<_>>()
            .join("\n");

        let program_state = crate::validator::program_summary(&session.program);
        drop(session);

        let feedback = format!(
            "There were errors:\n{error_summary}\n\nCurrent program state:\n{program_state}\n\nFix the issues and try again."
        );
        eprintln!("[web:feedback] errors → retrying");
        messages.push(crate::llm::ChatMessage::user(feedback));
    }

    eprintln!("[web:reply] {}...", reply_text.chars().take(200).collect::<String>());

    Json(AskResponse {
        reply: reply_text,
        code: all_code,
        results: all_results,
        test_results: all_test_results,
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
