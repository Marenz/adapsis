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

    let mut session = config.session.lock().await;
    let context = format!(
        "Working directory: {}\n\n{}\n\n{}\n\nUser request: {}",
        config.project_dir,
        crate::validator::program_summary(&session.program),
        session.format_recent_history(10),
        req.message
    );

    let system = format!(
        "{}\n\n## IMPORTANT: ForgeOS Interactive Mode\n\
         \n\
         The program state PERSISTS between responses. Do NOT resend existing types or functions.\n\
         Only send NEW code or modifications. Use !replace to modify existing functions.\n\
         Use !move to reorganize code into modules.\n\
         \n\
         ## IO Builtins as Tools\n\
         \n\
         IO builtins (file_read, list_dir, shell_exec, etc.) serve two purposes:\n\
         1. As TOOLS: for quick answers, write a minimal function and !eval it immediately.\n\
            Example: user asks 'what files are in /tmp?' →\n\
            +fn q ()->List<String> [io,async]\\n  +await r:List<String> = list_dir(\"/tmp\")\\n  +return r\\n!eval q\n\
         2. As BUILDING BLOCKS: for larger programs, compose them into proper functions.\n\
         \n\
         For questions, prefer the tool pattern — minimal function, immediate eval.\n\
         For building software, create well-named reusable functions.\n\
         \n\
         If an eval fails with an error, FIX the issue and try again — don't give up.\n\
         Always follow up on errors until the user's request is fulfilled.",
        crate::prompt::system_prompt()
    );

    let system_clone = system.clone();
    let context_clone = context.clone();
    let messages = vec![
        crate::llm::ChatMessage::system(system),
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
    let mut code = output.code.clone();

    // Echo to CLI
    if !output.thinking.is_empty() {
        eprintln!("[web:ai:think] {}", output.thinking.chars().take(200).collect::<String>());
    }
    if !code.is_empty() {
        eprintln!("[web:ai:code]\n{code}");
    }

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
    let mut reply_text = reply_text.trim().to_string();

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

            // Remove duplicates before applying — allow the model to redefine things
            if has_definitions {
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
                        // Check if the function needs async (has io/async effects)
                        let needs_async = session.program.get_function(&ev.function_name)
                            .is_some_and(|f| f.effects.iter().any(|e| 
                                matches!(e, crate::ast::Effect::Io | crate::ast::Effect::Async)));
                        
                        if needs_async {
                            if let Some(sender) = &config.io_sender {
                                let program = session.program.clone();
                                let fn_name = ev.function_name.clone();
                                let input = ev.input.clone();
                                let sender = sender.clone();
                                
                                let eval_result = tokio::task::spawn_blocking(move || {
                                    let func = program.get_function(&fn_name)
                                        .ok_or_else(|| anyhow::anyhow!("function not found"))?;
                                    let handle = crate::coroutine::CoroutineHandle::new(sender);
                                    let mut env = crate::eval::Env::new();
                                    env.set("__coroutine_handle", crate::eval::Value::CoroutineHandle(handle));
                                    crate::eval::bind_input_to_params(&program, func, &crate::eval::eval_parser_expr_standalone(&input)?, &mut env);
                                    crate::eval::eval_function_body_pub(&program, &func.body, &mut env)
                                }).await;

                                match eval_result {
                                    Ok(Ok(val)) => {
                                        results.push(MutationResult {
                                            message: format!("eval {}() = {val}", ev.function_name),
                                            success: true,
                                        });
                                    }
                                    Ok(Err(e)) => {
                                        results.push(MutationResult {
                                            message: format!("eval error: {e}"),
                                            success: false,
                                        });
                                    }
                                    Err(e) => {
                                        results.push(MutationResult {
                                            message: format!("eval task error: {e}"),
                                            success: false,
                                        });
                                    }
                                }
                            } else {
                                results.push(MutationResult {
                                    message: "eval error: async not available (no coroutine runtime)".to_string(),
                                    success: false,
                                });
                            }
                        } else {
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
                    }
                    crate::parser::Operation::Query(query) => {
                        let table = crate::typeck::build_symbol_table(&session.program);
                        let response = crate::typeck::handle_query(&session.program, &table, query);
                        results.push(MutationResult {
                            message: response,
                            success: true,
                        });
                    }
                    _ => {}
                }
            }
        }
    }

    // If there were errors, retry with the error context
    let has_errors = results.iter().any(|r| !r.success) || test_results.iter().any(|r| !r.pass);

    if has_errors {
        let error_summary: String = results.iter()
            .filter(|r| !r.success)
            .map(|r| r.message.clone())
            .chain(test_results.iter().filter(|r| !r.pass).map(|r| r.message.clone()))
            .collect::<Vec<_>>()
            .join("\n");

        let retry_context = format!(
            "Your code had errors:\n{error_summary}\n\nFix the issues and try again. The program state is:\n{}",
            crate::validator::program_summary(&session.program)
        );

        drop(session); // release lock for retry

        let retry_messages = vec![
            crate::llm::ChatMessage::system(system_clone),
            crate::llm::ChatMessage::user(context_clone),
            crate::llm::ChatMessage::assistant(&output.text),
            crate::llm::ChatMessage::user(retry_context),
        ];

        if let Ok(retry_output) = llm.generate(retry_messages).await {
            let retry_code = retry_output.code.clone();
            if !retry_code.is_empty() && retry_code.trim() != "DONE" {
                let mut session = config.session.lock().await;
                // Remove duplicates before applying
                if let Ok(ops) = crate::parser::parse(&retry_code) {
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
                }
                if let Ok(retry_results) = session.apply(&retry_code) {
                    let mut retry_mut_results: Vec<MutationResult> = retry_results.into_iter()
                        .map(|(msg, ok)| MutationResult { message: msg, success: ok })
                        .collect();
                    results.append(&mut retry_mut_results);
                }
                // Re-run evals from retry code
                if let Ok(ops) = crate::parser::parse(&retry_code) {
                    for op in &ops {
                        if let crate::parser::Operation::Eval(ev) = op {
                            match crate::eval::eval_compiled_or_interpreted(
                                &session.program, &ev.function_name, &ev.input,
                            ) {
                                Ok((result, compiled)) => {
                                    let tag = if compiled { " [compiled]" } else { "" };
                                    results.push(MutationResult {
                                        message: format!("[retry] eval {}() = {result}{tag}", ev.function_name),
                                        success: true,
                                    });
                                }
                                Err(e) => {
                                    results.push(MutationResult {
                                        message: format!("[retry] eval error: {e}"),
                                        success: false,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            // Append retry thinking
            if !retry_output.thinking.is_empty() {
                reply_text.push_str(&format!("\n\n[retry] {}", retry_output.thinking));
            }
            code = format!("{code}\n\n// --- retry ---\n{retry_code}");
        }
    }

    // Echo results to CLI
    for r in &results {
        if r.success {
            eprintln!("[web:result] OK: {}", r.message);
        } else {
            eprintln!("[web:result] ERROR: {}", r.message);
        }
    }
    for r in &test_results {
        eprintln!("[web:test] {}: {}", if r.pass { "PASS" } else { "FAIL" }, r.message);
    }
    if !reply_text.is_empty() {
        eprintln!("[web:reply] {}", reply_text.chars().take(300).collect::<String>());
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
        .route("/api/test", post(test_fn))
        .route("/api/query", post(query))
        .route("/api/status", get(status))
        .route("/api/program", get(program))
        .route("/api/history", post(history))
        .route("/api/rewind", post(rewind))
        .with_state(config.session.clone());

    // Routes that need the full config (async IO, LLM, OpenCode)
    let config_routes = axum::Router::new()
        .route("/api/eval", post(eval_fn))
        .route("/api/ask", post(ask))
        .route("/api/opencode", post(opencode_task))
        .with_state(config);

    session_routes.merge(config_routes)
}

// === OpenCode integration ===

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
