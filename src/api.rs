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

#[derive(Serialize, Clone)]
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

/// Scan eval.rs for builtin function names not already in the prompt.
pub async fn ask(
    State(config): State<AppConfig>,
    Json(req): Json<AskRequest>,
) -> Json<AskResponse> {
    eprintln!("\n[web:user] {}", req.message);
    let llm = crate::llm::LlmClient::new_with_model_and_key(
        &config.llm_url, &config.llm_model, config.llm_api_key.clone(),
    );

    let max_iterations = 10;
    let mut all_results: Vec<MutationResult> = vec![];
    let mut all_test_results: Vec<TestCaseResult> = vec![];
    let mut all_code = String::new();
    let mut reply_text = String::new();

    let system_prompt = {
        let base = crate::prompt::system_prompt();
        let builtins = crate::builtins::format_for_prompt();
        format!(
            "{base}\n\n{builtins}\n\n## ForgeOS Interactive Mode\n\
             Program state PERSISTS. Do NOT resend existing types/functions.\n\
             Only send NEW code or modifications.\n\
             You can !eval builtins directly: !eval concat a=\"hello \" b=\"world\"\n\
             For IO builtins, write a minimal [io,async] function and !eval it.\n\
             When your task is COMPLETE, respond with just DONE in a <code> block.\n\
             If you need to ask the user a question, just respond with text (no <code> block).\n\
             Keep working step by step until the task is fully done.\n\
             \n\
             If you hit a limitation that CANNOT be solved in Forge, emit !opencode <description>."
        )
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

                // Handle !undo before apply
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

                // Apply mutations
                let has_mutations = ops.iter().any(|op| !matches!(op,
                    crate::parser::Operation::Test(_) | crate::parser::Operation::Trace(_)
                    | crate::parser::Operation::Eval(_) | crate::parser::Operation::Query(_)
                    | crate::parser::Operation::Undo | crate::parser::Operation::Agent { .. }
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
                            for case in &test.cases {
                                match crate::eval::eval_test_case(&session.program, &test.function_name, case) {
                                    Ok(msg) => { eprintln!("[web:pass] {msg}"); iter_test_results.push(TestCaseResult { message: msg, pass: true }); }
                                    Err(e) => { eprintln!("[web:fail] {e}"); iter_has_errors = true; iter_test_results.push(TestCaseResult { message: format!("{e}"), pass: false }); }
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
                                        Ok(Err(e)) => { iter_has_errors = true; (format!("eval error: {e}"), false) }
                                        Err(e) => { iter_has_errors = true; (format!("eval task error: {e}"), false) }
                                    };
                                    eprintln!("[web:eval] {msg}");
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
                                        iter_has_errors = true;
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
                            iter_results.push(MutationResult { message: response, success: true });
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
                        crate::parser::Operation::OpenCode(task) => {
                            eprintln!("[web:opencode] {task}");
                            drop(session);
                            let oc_result = tokio::time::timeout(
                                std::time::Duration::from_secs(300),
                                tokio::process::Command::new("opencode")
                                    .arg("run").arg("--format").arg("json").arg(task)
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
        .route("/api/agents", get(agents))
        .with_state(config.session.clone());

    let config_routes = axum::Router::new()
        .route("/api/eval", post(eval_fn))
        .route("/api/ask", post(ask))
        .route("/api/opencode", post(opencode_task))
        .with_state(config);

    session_routes.merge(config_routes)
}
