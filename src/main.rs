mod api;
mod ast;
pub mod bytecode;
pub mod builtins;
mod compiler;
mod coroutine;
mod eval;
mod events;
pub mod intern;
pub mod library;
mod llm;
mod orchestrator;
mod parser;
mod prompt;
mod repl;
mod server;
mod session;
mod telegram;
mod typeck;
mod validator;
mod vm;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

async fn snapshot_from_tiers(
    program: &std::sync::Arc<tokio::sync::RwLock<crate::ast::Program>>,
    meta: &crate::session::SharedMeta,
    runtime: &crate::session::SharedRuntime,
) -> crate::session::Session {
    crate::session::Session {
        program: program.read().await.clone(),
        runtime: runtime.read().unwrap().clone(),
        meta: meta.lock().unwrap().clone(),
        sandbox: None,
    }
}

#[derive(Parser)]
#[command(name = "adapsis", about = "Adapsis — the adaptive, self-modifying AI programming environment")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Run the interactive feedback loop with the LLM (CLI mode)
    Run {
        /// Natural language task description for the model
        #[arg(short, long)]
        task: String,

        /// LLM server URL (OpenAI-compatible)
        #[arg(short, long, default_value = "http://127.0.0.1:8081")]
        url: String,

        /// Model name to use
        #[arg(long, env = "FORGE_MODEL", default_value = "default")]
        model: String,

        /// API key for the LLM provider (sent as Bearer token)
        #[arg(long, env = "LLM_API_KEY")]
        api_key: Option<String>,

        /// Maximum feedback loop iterations
        #[arg(short, long, default_value_t = 20)]
        max_iterations: usize,
    },

    /// Architect mode: design first, then implement per-function
    Architect {
        /// Natural language task description for the model
        #[arg(short, long)]
        task: String,

        /// LLM server URL (OpenAI-compatible)
        #[arg(short, long, default_value = "http://127.0.0.1:8081")]
        url: String,

        /// Model name to use
        #[arg(long, env = "FORGE_MODEL", default_value = "default")]
        model: String,

        /// API key for the LLM provider (sent as Bearer token)
        #[arg(long, env = "LLM_API_KEY")]
        api_key: Option<String>,

        /// Maximum feedback loop iterations per function
        #[arg(short, long, default_value_t = 5)]
        max_iterations: usize,

        /// Web server port (0 to disable browser UI)
        #[arg(short, long, default_value_t = 3000)]
        port: u16,
    },

    /// Run with browser interface
    Serve {
        /// Natural language task description for the model
        #[arg(short, long)]
        task: String,

        /// LLM server URL (OpenAI-compatible)
        #[arg(short, long, default_value = "http://127.0.0.1:8081")]
        url: String,

        /// API key for the LLM provider (sent as Bearer token)
        #[arg(long, env = "LLM_API_KEY")]
        api_key: Option<String>,

        /// Maximum feedback loop iterations
        #[arg(short, long, default_value_t = 20)]
        max_iterations: usize,

        /// Web server port
        #[arg(short, long, default_value_t = 3000)]
        port: u16,
    },

    /// Parse a .ax file and validate it
    Check {
        /// Path to .ax file
        path: String,
    },

    /// Parse a .ax file and run its !test blocks
    Test {
        /// Path to .ax file
        path: String,
    },

    /// Compile a .ax file to native code and run it
    Compile {
        /// Path to .ax file
        path: String,

        /// Function to call
        #[arg(short, long)]
        func: String,

        /// Arguments (comma-separated integers)
        #[arg(short, long, default_value = "")]
        args: String,
    },

    /// Run an Adapsis program with async IO (coroutine runtime)
    RunAsync {
        /// Path to .ax file
        path: String,

        /// Function to call (default: main)
        #[arg(short, long, default_value = "main")]
        func: String,

        /// LLM server URL
        #[arg(short, long, default_value = "http://127.0.0.1:8081")]
        url: String,

        /// Model name
        #[arg(long, env = "FORGE_MODEL", default_value = "default")]
        model: String,
    },

    /// Interactive REPL (auto-starts AdapsisOS if not running)
    Repl {
        /// AdapsisOS API URL (auto-detected if not specified)
        #[arg(short, long, default_value = "http://127.0.0.1:3001")]
        api: String,

        /// Session name or path (used when auto-starting AdapsisOS)
        #[arg(short, long, default_value = "repl")]
        session: String,

        /// LLM server URL (used when auto-starting)
        #[arg(short, long, env = "FORGE_LLM_URL", default_value = "http://127.0.0.1:8081")]
        url: String,

        /// Model name (used when auto-starting)
        #[arg(long, env = "FORGE_MODEL")]
        model: Option<String>,
    },

    /// Start AdapsisOS — HTTP API + browser UI + session persistence
    Os {
        /// HTTP port
        #[arg(short, long, default_value_t = 3001)]
        port: u16,

        /// Session name or path. Plain names (e.g. "opus-run") are stored in
        /// ~/.config/adapsis/sessions/<name>.json. Absolute paths are used as-is.
        #[arg(short, long, default_value = "default")]
        session: String,

        /// LLM server URL (OpenAI-compatible)
        #[arg(short, long, default_value = "http://127.0.0.1:8081")]
        url: String,

        /// Model name (required, e.g. anthropic/claude-haiku-4-5-20251001)
        #[arg(long, env = "FORGE_MODEL")]
        model: String,

        /// API key for the LLM provider (sent as Bearer token)
        #[arg(long, env = "LLM_API_KEY")]
        api_key: Option<String>,

        /// Daemonize: fork to background after server is ready
        #[arg(short, long)]
        daemonize: bool,

        /// Autonomous mode: inject a goal and let the AI work without user input.
        /// Pass a goal string, or "roadmap" to use the current priority from ROADMAP.md.
        #[arg(long)]
        autonomous: Option<String>,

        /// Log file for structured AI activity logging (what it sees, thinks, does)
        #[arg(long, default_value = "adapsisos.log")]
        log_file: String,

        /// JSONL training data log (one entry per iteration: input/output/outcome)
        #[arg(long, default_value = "training.jsonl")]
        training_log: String,

        /// Directory where !opencode runs and builds. AdapsisOS should be started from
        /// {dir}/target/release/adapsis so exec restart picks up rebuilt binaries.
        #[arg(long, env = "FORGE_OPENCODE_GIT_DIR")]
        opencode_git_dir: Option<String>,

        /// Maximum iterations per AI request (default 20)
        #[arg(long, default_value_t = 20)]
        max_iterations: usize,

        /// Telegram bot token (enables Telegram bot when set)
        #[arg(long, env = "TELEGRAM_BOT_TOKEN")]
        telegram_token: Option<String>,

        /// Telegram admin chat ID (messages from this chat route through /api/ask)
        #[arg(long, env = "TELEGRAM_ADMIN_CHAT_ID", default_value_t = 1815217)]
        telegram_admin_chat_id: i64,
    },

    /// Send a message to a running AdapsisOS instance
    Ask {
        /// The message to send
        message: Vec<String>,

        /// AdapsisOS API URL
        #[arg(short, long, default_value = "http://127.0.0.1:3001")]
        api: String,
    },

    /// Show status of a running AdapsisOS instance
    Status {
        /// AdapsisOS API URL
        #[arg(short, long, default_value = "http://127.0.0.1:3001")]
        api: String,
    },

    /// Apply Adapsis code to a running AdapsisOS instance
    Mutate {
        /// Adapsis source code
        source: Vec<String>,

        /// AdapsisOS API URL
        #[arg(short, long, default_value = "http://127.0.0.1:3001")]
        api: String,
    },

    /// Eval a function on a running AdapsisOS instance
    Eval {
        /// Function name and arguments
        expr: Vec<String>,

        /// AdapsisOS API URL
        #[arg(short, long, default_value = "http://127.0.0.1:3001")]
        api: String,
    },

    /// Query a running AdapsisOS instance
    Query {
        /// Query string (?symbols, ?source fn, ?deps fn, etc.)
        query: Vec<String>,

        /// AdapsisOS API URL
        #[arg(short, long, default_value = "http://127.0.0.1:3001")]
        api: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Run {
            task,
            url,
            model,
            api_key,
            max_iterations,
        } => {
            let llm_client = llm::LlmClient::new_with_model_and_key(&url, &model, api_key);
            let mut orch = orchestrator::Orchestrator::new(llm_client, max_iterations);
            orch.run(&task).await?;
        }
        Command::Architect {
            task,
            url,
            model,
            api_key,
            max_iterations,
            port,
        } => {
            let llm_client = llm::LlmClient::new_with_model_and_key(&url, &model, api_key);
            if port > 0 {
                // Run with browser UI
                let event_bus = events::EventBus::new();
                let state = std::sync::Arc::new(server::AppState {
                    event_bus: event_bus.clone(),
                    program: tokio::sync::Mutex::new(ast::Program::default()),
                    llm: llm_client.clone(),
                    max_iterations,
                });

                let app = axum::Router::new()
                    .route("/", axum::routing::get(|| async {
                        axum::response::Html(include_str!("../web/index.html"))
                    }))
                    .route("/ws", axum::routing::get(server::ws_handler))
                    .layer(tower_http::cors::CorsLayer::permissive())
                    .with_state(state);

                let listener =
                    tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await?;
                println!("Adapsis architect UI at http://127.0.0.1:{port}");

                let server_task = axum::serve(listener, app);
                let orch_task = async {
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    let mut orch = orchestrator::Orchestrator::with_event_bus(
                        llm_client,
                        max_iterations,
                        event_bus,
                    );
                    orch.run_architect(&task).await
                };

                tokio::select! {
                    r = server_task => { r?; }
                    r = orch_task => {
                        r?;
                        println!("Architect complete. Server still running — Ctrl+C to stop.");
                        std::future::pending::<()>().await;
                    }
                }
            } else {
                let mut orch = orchestrator::Orchestrator::new(llm_client, max_iterations);
                orch.run_architect(&task).await?;
            }
        }
        Command::Serve {
            task,
            url,
            api_key,
            max_iterations,
            port,
        } => {
            let llm_client = llm::LlmClient::new_with_model_and_key(&url, "default", api_key);
            server::serve_and_run(llm_client, max_iterations, port, task).await?;
        }
        Command::Check { path } => {
            let source = std::fs::read_to_string(&path)?;
            let operations = parser::parse(&source)?;
            let mut program = ast::Program::default();
            for op in &operations {
                match op {
                    parser::Operation::Test(_)
                    | parser::Operation::Trace(_)
                    | parser::Operation::Eval(_)
                    | parser::Operation::Query(_) => {}
                    _ => match validator::apply_and_validate(&mut program, op) {
                        Ok(msg) => println!("OK: {msg}"),
                        Err(e) => eprintln!("ERROR: {e}"),
                    },
                }
            }
            let table = typeck::build_symbol_table(&program);
            for func in &program.functions {
                for error in typeck::check_function(&table, func) {
                    eprintln!("TYPE WARNING: {error}");
                }
            }
            println!("\n--- Program state ---");
            println!("{program}");
        }
        Command::Test { path } => {
            let source = std::fs::read_to_string(&path)?;
            let operations = parser::parse(&source)?;
            let mut program = ast::Program::default();
            let mut test_ops = vec![];
            let mut io_mocks: Vec<session::IoMock> = vec![];
            // Standalone registries for ?tasks / ?inspect queries (empty but real).
            let task_registry: coroutine::TaskRegistry = std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
            let snapshot_registry: coroutine::TaskSnapshotRegistry = std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
            for op in &operations {
                match op {
                    parser::Operation::Test(_) => test_ops.push(op.clone()),
                    parser::Operation::Module(m) => {
                        // Collect tests embedded inside module bodies
                        for body_op in &m.body {
                            if let parser::Operation::Test(_) = body_op {
                                test_ops.push(body_op.clone());
                            }
                        }
                        // Still apply the module itself
                        match validator::apply_and_validate(&mut program, op) {
                            Ok(msg) => println!("OK: {msg}"),
                            Err(e) => eprintln!("ERROR: {e}"),
                        }
                    }
                    parser::Operation::Mock { operation, patterns, response } => {
                        let pattern_display = patterns.iter().map(|p| format!("\"{p}\"")).collect::<Vec<_>>().join(" ");
                        io_mocks.push(session::IoMock {
                            operation: operation.clone(),
                            patterns: patterns.clone(),
                            response: response.clone(),
                        });
                        println!("OK: mock {operation} {pattern_display}");
                    }
                    parser::Operation::Unmock => {
                        let count = io_mocks.len();
                        io_mocks.clear();
                        println!("OK: cleared {count} mocks");
                    }
                    parser::Operation::Trace(trace) => {
                        println!("\n--- Tracing {} ---", trace.function_name);
                        match eval::trace_function(&program, &trace.function_name, &trace.input) {
                            Ok(steps) => {
                                for step in &steps {
                                    println!("  > {step}");
                                }
                            }
                            Err(e) => eprintln!("  TRACE ERROR: {e}"),
                        }
                    }
                    parser::Operation::Eval(ev) => {
                        if let Some(ref expr) = ev.inline_expr {
                            // Inline expression: evaluate directly
                            match eval::eval_inline_expr(&program, expr) {
                                Ok(val) => println!("  = {val}"),
                                Err(e) => eprintln!("  EVAL ERROR: {e}"),
                            }
                        } else {
                            match eval::eval_compiled_or_interpreted(
                                &program,
                                &ev.function_name,
                                &ev.input,
                            ) {
                                Ok((result, compiled)) => {
                                    let tag = if compiled { " [compiled]" } else { "" };
                                    println!("  eval {}(...) = {result}{tag}", ev.function_name);
                                }
                                Err(e) => eprintln!("  EVAL ERROR: {e}"),
                            }
                        }
                    }
                    parser::Operation::Query(query) => {
                        let response = if query.trim() == "?tasks" {
                            api::format_tasks(&Some(task_registry.clone()))
                        } else if let Some(tid) = api::parse_inspect_task_query(query.trim()) {
                            api::format_inspect_task(&Some(task_registry.clone()), &Some(snapshot_registry.clone()), tid)
                        } else {
                            let table = typeck::build_symbol_table(&program);
                            typeck::handle_query(&program, &table, query, &[])
                        };
                        println!("\n--- Query: {query} ---\n{response}");
                    }
                    _ => match validator::apply_and_validate(&mut program, op) {
                        Ok(msg) => println!("OK: {msg}"),
                        Err(e) => eprintln!("ERROR: {e}"),
                    },
                }
            }
            for test_op in &test_ops {
                if let parser::Operation::Test(test) = test_op {
                    println!("\n--- Testing {} ---", test.function_name);
                    for (i, case) in test.cases.iter().enumerate() {
                        match eval::eval_test_case_with_mocks(&program, &test.function_name, case, &io_mocks, &[]) {
                            Ok(msg) => println!("  PASS [{i}]: {msg}"),
                            Err(e) => eprintln!("  FAIL [{i}]: {e}"),
                        }
                    }
                }
            }
        }
        Command::Compile { path, func, args } => {
            let source = std::fs::read_to_string(&path)?;
            let operations = parser::parse(&source)?;
            let mut program = ast::Program::default();
            for op in &operations {
                match op {
                    parser::Operation::Test(_)
                    | parser::Operation::Trace(_)
                    | parser::Operation::Eval(_)
                    | parser::Operation::Query(_) => {}
                    _ => {
                        validator::apply_and_validate(&mut program, op)?;
                    }
                }
            }

            println!("Compiling...");
            let mut compiled = compiler::compile(&program)?;
            println!("Compiled {} function(s)", program.functions.len());

            let int_args: Vec<i64> = if args.is_empty() {
                vec![]
            } else {
                args.split(',')
                    .map(|s| s.trim().parse::<i64>())
                    .collect::<std::result::Result<Vec<_>, _>>()?
            };

            // Find the function to check its return type
            let returns_string = program
                .functions
                .iter()
                .find(|f| f.name == func)
                .is_some_and(|f| matches!(f.return_type, ast::Type::String));

            println!("Calling {}({})...", func, args);
            if returns_string {
                let result = compiled.call_string(&func, &int_args)?;
                println!("Result: \"{result}\"");
            } else {
                let result = compiled.call_i64(&func, &int_args)?;
                println!("Result: {result}");
            }
        }
        Command::RunAsync { path, func, url, model } => {
            let source = std::fs::read_to_string(&path)?;
            let operations = parser::parse(&source)?;
            let mut program = ast::Program::default();
            for op in &operations {
                match op {
                    parser::Operation::Test(_)
                    | parser::Operation::Trace(_)
                    | parser::Operation::Eval(_)
                    | parser::Operation::Query(_) => {}
                    _ => {
                        validator::apply_and_validate(&mut program, op)?;
                    }
                }
            }

            println!("Running {func}() with coroutine runtime...");

            let (mut runtime, mut io_rx) = coroutine::Runtime::new();
            runtime.llm_url = url;
            runtime.llm_default_model = model;
            let runtime = std::sync::Arc::new(runtime);
            let handle = coroutine::CoroutineHandle::new(runtime.io_sender());

            // Spawn the main evaluator on a blocking thread
            let program_clone = program.clone();
            let program_mut = eval::make_shared_program_mut(&program);
            let program_mut_clone = program_mut.clone();
            let func_clone = func.clone();
            let eval_task = tokio::task::spawn_blocking(move || {
                eval::set_shared_program(Some(std::sync::Arc::new(program_clone.clone())));
                eval::set_shared_program_mut(Some(program_mut_clone));
                let func_decl = program_clone.get_function(&func_clone)
                    .ok_or_else(|| anyhow::anyhow!("function `{func_clone}` not found"))?;

                let mut env = eval::Env::new_with_shared_interner(&program_clone.shared_interner);
                env.set("__coroutine_handle", eval::Value::CoroutineHandle(handle));

                eval::eval_function_body_pub(&program_clone, &func_decl.body, &mut env)
            });

            // Event loop — process IO requests from coroutines
            let rt = runtime.clone();
            let program_for_spawn = program.clone();
            let io_sender_for_spawn = runtime.io_sender();
            let task_registry_for_spawn = runtime.task_registry.clone();
            let snap_registry_for_spawn = runtime.snapshot_registry.clone();
            let rt_for_id = runtime.clone();
            let io_loop = async move {
                while let Some(request) = io_rx.recv().await {
                    match request {
                        coroutine::IoRequest::Spawn { function_name, args, reply } => {
                            // Register the task
                            let task_id = rt_for_id.next_task_id();
                            let task_info = coroutine::TaskInfo {
                                id: task_id,
                                function_name: function_name.clone(),
                                status: coroutine::WaitReason::Running,
                                started_at: format!("{}s", std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()),
                            };
                            task_registry_for_spawn.lock().unwrap().insert(task_id, task_info);
                            let _ = reply.send(Ok(task_id));

                            // Spawn a new coroutine for this function
                            let prog = program_for_spawn.clone();
                            let sender = io_sender_for_spawn.clone();
                            let registry = task_registry_for_spawn.clone();
                            let snap_reg = snap_registry_for_spawn.clone();
                            tokio::task::spawn_blocking(move || {
                                eval::set_shared_program(Some(std::sync::Arc::new(prog.clone())));
                                eval::set_shared_program_mut(Some(eval::make_shared_program_mut(&prog)));
                                let func_decl = match prog.get_function(&function_name) {
                                    Some(f) => f,
                                    None => {
                                        eprintln!("spawn: function `{function_name}` not found");
                                        if let Ok(mut tasks) = registry.lock() {
                                            if let Some(info) = tasks.get_mut(&task_id) {
                                                info.status = coroutine::WaitReason::Failed(format!("function `{function_name}` not found"));
                                            }
                                        }
                                        return;
                                    }
                                };
                                let handle = coroutine::CoroutineHandle::new_with_task(sender, task_id, registry.clone(), snap_reg);
                                let mut env = eval::Env::new_with_shared_interner(&prog.shared_interner);
                                env.set("__coroutine_handle", eval::Value::CoroutineHandle(handle));
                                // Bind args to params
                                for (i, param) in func_decl.params.iter().enumerate() {
                                    if let Some(val) = args.get(i) {
                                        env.set(&param.name, val.clone());
                                    }
                                }
                                match eval::eval_function_body_named(&prog, &function_name, &func_decl.body, &mut env) {
                                    Ok(val) => {
                                        if let Ok(mut tasks) = registry.lock() {
                                            if let Some(info) = tasks.get_mut(&task_id) {
                                                info.status = coroutine::WaitReason::Completed(format!("{val}"));
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("spawn {function_name}: {e}");
                                        if let Ok(mut tasks) = registry.lock() {
                                            if let Some(info) = tasks.get_mut(&task_id) {
                                                info.status = coroutine::WaitReason::Failed(format!("{e}"));
                                            }
                                        }
                                    }
                                }
                            });
                        }
                        coroutine::IoRequest::SourceAdd {
                            module_name, source_type, interval_ms, alias, handler, reply,
                        } => {
                            if source_type == "timer" {
                                if let Some(ms) = interval_ms {
                                    let _ = reply.send(Ok(format!("timer source '{}' registered ({}ms)", alias, ms)));
                                    let prog = program_for_spawn.clone();
                                    let sender = io_sender_for_spawn.clone();
                                    let registry = task_registry_for_spawn.clone();
                                    let snap_reg = snap_registry_for_spawn.clone();
                                    tokio::spawn(async move {
                                        let mut interval = tokio::time::interval(std::time::Duration::from_millis(ms));
                                        interval.tick().await; // skip first immediate tick
                                        loop {
                                            interval.tick().await;
                                            let handler_name = handler.clone();
                                            let prog = prog.clone();
                                            let sender = sender.clone();
                                            let registry = registry.clone();
                                            let snap_reg = snap_reg.clone();
                                            let alias = alias.clone();
                                            let module_name = module_name.clone();
                                            tokio::task::spawn_blocking(move || {
                                                eval::set_shared_program(Some(std::sync::Arc::new(prog.clone())));
                                                eval::set_shared_program_mut(Some(eval::make_shared_program_mut(&prog)));
                                                let func = match prog.get_function(&handler_name) {
                                                    Some(f) => f.clone(),
                                                    None => { eprintln!("[timer:{}] handler `{}` not found", alias, handler_name); return; }
                                                };
                                                let handle = coroutine::CoroutineHandle::new_with_task(sender, 0, registry, snap_reg);
                                                let mut env = eval::Env::new_with_shared_interner(&prog.shared_interner);
                                                env.set("__coroutine_handle", eval::Value::CoroutineHandle(handle));
                                                env.set("__module_name", eval::Value::String(std::sync::Arc::new(module_name)));
                                                match eval::eval_function_body_named(&prog, &handler_name, &func.body, &mut env) {
                                                    Ok(val) => eprintln!("[timer:{}] {} -> {}", alias, handler_name, val),
                                                    Err(e) => eprintln!("[timer:{}] {} error: {}", alias, handler_name, e),
                                                }
                                            });
                                        }
                                    });
                                } else {
                                    let _ = reply.send(Err(anyhow::anyhow!("timer source requires interval_ms")));
                                }
                            } else if source_type == "channel" {
                                let _ = reply.send(Ok(format!("channel source '{}' registered", alias)));
                            } else {
                                let _ = reply.send(Ok(format!("event source '{}' registered ({})", alias, source_type)));
                            }
                        }
                        _ => {
                            let rt = rt.clone();
                            tokio::spawn(async move {
                                rt.handle_io(request).await;
                            });
                        }
                    }
                }
            };

            tokio::select! {
                result = eval_task => {
                    match result? {
                        Ok(val) => println!("Result: {val}"),
                        Err(e) => eprintln!("Error: {e}"),
                    }
                }
                _ = io_loop => {}
            }
        }
        Command::Repl { api, session, url, model } => {
            // Check if AdapsisOS is already running
            let client = reqwest::Client::new();
            let running = client.get(format!("{api}/api/status"))
                .send().await
                .map(|r| r.status().is_success())
                .unwrap_or(false);

            let api_url = if running {
                api
            } else {
                // Auto-start AdapsisOS in the background
                let model = model.unwrap_or_else(|| {
                    eprintln!("No model specified. Set FORGE_MODEL env var or use --model.");
                    eprintln!("Example: FORGE_MODEL=anthropic/claude-haiku-4-5-20251001 adapsis repl");
                    std::process::exit(1);
                });

                // Extract port from api URL
                let port = api.rsplit(':').next()
                    .and_then(|p| p.parse::<u16>().ok())
                    .unwrap_or(3001);

                eprintln!("No AdapsisOS instance detected. Starting one...");

                let exe = std::env::current_exe()?;
                let mut cmd = std::process::Command::new(&exe);
                cmd.arg("os")
                    .arg("--session").arg(&session)
                    .arg("--port").arg(port.to_string())
                    .arg("--url").arg(&url)
                    .arg("--model").arg(&model)
                    .arg("--daemonize");

                let output = cmd.output()?;
                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    eprintln!("Failed to start AdapsisOS: {stderr}");
                    std::process::exit(1);
                }

                // Print the startup output (includes "Daemonized: PID ...")
                let stdout = String::from_utf8_lossy(&output.stdout);
                eprint!("{stdout}");

                // Wait for it to be ready
                for _ in 0..20 {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    if client.get(format!("{api}/api/status"))
                        .send().await
                        .map(|r| r.status().is_success())
                        .unwrap_or(false)
                    {
                        break;
                    }
                }

                api
            };

            repl::run_repl(&api_url).await?;
        }
        Command::Os { port, session, url, model, api_key, daemonize, autonomous, log_file, training_log, opencode_git_dir, max_iterations, telegram_token, telegram_admin_chat_id } => {
            // Resolve session path: plain names go to ~/.config/adapsis/sessions/,
            // absolute paths or paths with directory separators are used as-is.
            let session = if std::path::Path::new(&session).is_absolute() || session.contains('/') || session.contains('\\') {
                session
            } else {
                let dir = dirs::config_dir()
                    .unwrap_or_else(|| std::path::PathBuf::from("."))
                    .join("adapsis")
                    .join("sessions");
                std::fs::create_dir_all(&dir).ok();
                let name = if session.ends_with(".json") { session } else { format!("{session}.json") };
                dir.join(name).to_string_lossy().to_string()
            };

            // Prevent session file from living inside the opencode git dir —
            // !opencode modifies that directory and could corrupt or delete the session.
            if let Some(ref git_dir) = opencode_git_dir {
                let session_canonical = std::fs::canonicalize(&session).unwrap_or_else(|_| std::path::PathBuf::from(&session));
                let git_dir_canonical = std::fs::canonicalize(git_dir).unwrap_or_else(|_| std::path::PathBuf::from(git_dir));
                if session_canonical.starts_with(&git_dir_canonical) {
                    eprintln!("ERROR: Session file '{}' is inside the opencode git directory '{}'.", session, git_dir);
                    eprintln!("       !opencode modifies that directory and could corrupt the session.");
                    eprintln!("       Use a plain name (e.g. --session opus-run) to store in ~/.config/adapsis/sessions/");
                    std::process::exit(1);
                }
            }

            let session_path = std::path::Path::new(&session);
            let mut sess = if session_path.exists() {
                println!("Loading session from {session}...");
                let s = session::Session::load(session_path)?;
                println!(
                    "Loaded: revision {}, {} mutations",
                    s.meta.revision,
                    s.meta.mutations.len()
                );
                s
            } else {
                println!("New session (saving to {session})");
                session::Session::new()
            };

            // In AdapsisOS mode, enforce modules and tests
            sess.program.require_modules = true;

            // Auto-load persistent module library (~/.config/adapsis/modules/)
            let lib_state = library::load_module_library(&mut sess.program);
            if !lib_state.loaded_modules.is_empty() {
                sess.program.rebuild_function_index();
            }
            sess.meta.library_state = Some(lib_state);
            sess.init_shared_vars();

            // Build restart context before the session is split into tier locks.
            // This consumes last_opencode_output so it's only shown once.
            let restart_context = if sess.meta.chat_messages.len() > 1 {
                Some(sess.restart_context())
            } else {
                None
            };

            let initial_runtime = sess.runtime.clone();
            let shared_runtime: crate::session::SharedRuntime =
                std::sync::Arc::new(std::sync::RwLock::new(initial_runtime));
            let shared_meta: crate::session::SharedMeta =
                std::sync::Arc::new(std::sync::Mutex::new(sess.meta.clone()));

            // Set up coroutine runtime for async IO
            let (mut runtime, mut io_rx) = coroutine::Runtime::new();
            runtime.llm_url = url.clone();
            runtime.llm_default_model = model.clone();
            runtime.llm_api_key = api_key.clone();
            let runtime = std::sync::Arc::new(runtime);
            let io_sender = runtime.io_sender();

            // Spawn IO event loop (including +spawn support)
            let rt = runtime.clone();
            let rt_for_id = runtime.clone();
            let task_registry_for_spawn = runtime.task_registry.clone();
            let snap_registry_for_spawn2 = runtime.snapshot_registry.clone();
            let io_sender_for_spawn = runtime.io_sender();
            let shared_runtime_for_spawn = shared_runtime.clone();
            let shared_meta_for_spawn = shared_meta.clone();
            let shared_program_for_spawn = std::sync::Arc::new(tokio::sync::RwLock::new(sess.program.clone()));
            // Clone resources for startup execution (before IO loop moves them)
            let io_sender_for_startup = runtime.io_sender();
            let startup_registry = runtime.task_registry.clone();
            let startup_snap_reg = runtime.snapshot_registry.clone();
            let startup_runtime = shared_runtime.clone();
            let startup_meta = shared_meta.clone();
            let startup_program = shared_program_for_spawn.clone();
            tokio::spawn(async move {
                while let Some(request) = io_rx.recv().await {
                    match request {
                        coroutine::IoRequest::Spawn { function_name, args, reply } => {
                            let task_id = rt_for_id.next_task_id();
                            let task_info = coroutine::TaskInfo {
                                id: task_id,
                                function_name: function_name.clone(),
                                status: coroutine::WaitReason::Running,
                                started_at: format!("{}s", std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()),
                            };
                            task_registry_for_spawn.lock().unwrap().insert(task_id, task_info);
                            let _ = reply.send(Ok(task_id));

                            let sender = io_sender_for_spawn.clone();
                            let registry = task_registry_for_spawn.clone();
                            let snap_reg = snap_registry_for_spawn2.clone();
                            let runtime_for_blocking = shared_runtime_for_spawn.clone();
                            let meta_for_blocking = shared_meta_for_spawn.clone();
                            let program_for_blocking = shared_program_for_spawn.clone();
                            tokio::task::spawn_blocking(move || {
                                eval::set_shared_runtime(Some(runtime_for_blocking));
                                eval::set_shared_meta(Some(meta_for_blocking));
                                let program = program_for_blocking.blocking_read().clone();
                                let func_decl = match program.get_function(&function_name) {
                                    Some(f) => f.clone(),
                                    None => {
                                        eprintln!("spawn: function `{function_name}` not found");
                                        if let Ok(mut tasks) = registry.lock() {
                                            if let Some(info) = tasks.get_mut(&task_id) {
                                                info.status = coroutine::WaitReason::Failed(format!("function not found"));
                                            }
                                        }
                                        return;
                                    }
                                };
                                eval::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                                eval::set_shared_program_mut(Some(eval::make_shared_program_mut(&program)));

                                let handle = coroutine::CoroutineHandle::new_with_task(sender, task_id, registry.clone(), snap_reg);
                                let mut env = eval::Env::new_with_shared_interner(&program.shared_interner);
                                env.set("__coroutine_handle", eval::Value::CoroutineHandle(handle));
                                for (i, param) in func_decl.params.iter().enumerate() {
                                    if let Some(val) = args.get(i) {
                                        env.set(&param.name, val.clone());
                                    }
                                }
                                match eval::eval_function_body_named(&program, &function_name, &func_decl.body, &mut env) {
                                    Ok(val) => {
                                        if let Ok(mut tasks) = registry.lock() {
                                            if let Some(info) = tasks.get_mut(&task_id) {
                                                info.status = coroutine::WaitReason::Completed(format!("{val}"));
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("spawn {function_name}: {e}");
                                        if let Ok(mut tasks) = registry.lock() {
                                            if let Some(info) = tasks.get_mut(&task_id) {
                                                info.status = coroutine::WaitReason::Failed(format!("{e}"));
                                            }
                                        }
                                    }
                                }
                            });
                        }
                        coroutine::IoRequest::SourceAdd {
                            module_name, source_type, interval_ms, alias, handler, reply,
                        } => {
                            if source_type == "timer" {
                                if let Some(ms) = interval_ms {
                                    let _ = reply.send(Ok(format!("timer source '{}' registered ({}ms)", alias, ms)));
                                    let program_for_timer = shared_program_for_spawn.clone();
                                    let sender_for_timer = io_sender_for_spawn.clone();
                                    let registry_for_timer = task_registry_for_spawn.clone();
                                    let snap_reg_for_timer = snap_registry_for_spawn2.clone();
                                    let runtime_for_timer = shared_runtime_for_spawn.clone();
                                    let meta_for_timer = shared_meta_for_spawn.clone();
                                    tokio::spawn(async move {
                                        let mut interval = tokio::time::interval(std::time::Duration::from_millis(ms));
                                        interval.tick().await; // skip first immediate tick
                                        loop {
                                            interval.tick().await;
                                            let handler_name = handler.clone();
                                            let prog = program_for_timer.read().await.clone();
                                            let func = match prog.get_function(&handler_name) {
                                                Some(f) => f.clone(),
                                                None => {
                                                    eprintln!("[timer:{}] handler `{}` not found", alias, handler_name);
                                                    continue;
                                                }
                                            };
                                            let sender = sender_for_timer.clone();
                                            let registry = registry_for_timer.clone();
                                            let snap_reg = snap_reg_for_timer.clone();
                                            let rt_for_tick = runtime_for_timer.clone();
                                            let meta_for_tick = meta_for_timer.clone();
                                            let alias_for_tick = alias.clone();
                                            let module_for_tick = module_name.clone();
                                            tokio::task::spawn_blocking(move || {
                                                eval::set_shared_runtime(Some(rt_for_tick));
                                                eval::set_shared_meta(Some(meta_for_tick));
                                                eval::set_shared_program(Some(std::sync::Arc::new(prog.clone())));
                                                eval::set_shared_program_mut(Some(eval::make_shared_program_mut(&prog)));
                                                let task_id = 0; // timer tasks don't need unique IDs for now
                                                let handle = coroutine::CoroutineHandle::new_with_task(sender, task_id, registry, snap_reg);
                                                let mut env = eval::Env::new_with_shared_interner(&prog.shared_interner);
                                                env.set("__coroutine_handle", eval::Value::CoroutineHandle(handle));
                                                env.set("__module_name", eval::Value::String(std::sync::Arc::new(module_for_tick)));
                                                match eval::eval_function_body_named(&prog, &handler_name, &func.body, &mut env) {
                                                    Ok(val) => {
                                                        eprintln!("[timer:{}] {} -> {}", alias_for_tick, handler_name, val);
                                                    }
                                                    Err(e) => {
                                                        eprintln!("[timer:{}] {} error: {}", alias_for_tick, handler_name, e);
                                                    }
                                                }
                                            });
                                        }
                                    });
                                } else {
                                    let _ = reply.send(Err(anyhow::anyhow!("timer source requires interval_ms")));
                                }
                            } else if source_type == "channel" {
                                let _ = reply.send(Ok(format!("channel source '{}' registered", alias)));
                                // Channel dispatch will be implemented in a later phase
                            } else {
                                // Event source (source_type starts with "event:")
                                let _ = reply.send(Ok(format!("event source '{}' registered ({})", alias, source_type)));
                                // Event dispatch will be implemented in a later phase
                            }
                        }
                        _ => {
                            let rt = rt.clone();
                            tokio::spawn(async move {
                                rt.handle_io(request).await;
                            });
                        }
                    }
                }
            });

            // Execute module startup blocks and auto-register module-level sources
            {
                let modules_with_startup: Vec<(String, std::sync::Arc<ast::FunctionDecl>)> = {
                    let prog = startup_program.blocking_read();
                    prog.modules.iter()
                        .filter_map(|m| m.startup.as_ref().map(|s| (m.name.clone(), s.clone())))
                        .collect()
                };
                for (module_name, startup_fn) in modules_with_startup {
                    eprintln!("[startup] executing {}.startup", module_name);
                    let prog_clone = startup_program.clone();
                    let sender = io_sender_for_startup.clone();
                    let registry = startup_registry.clone();
                    let snap_reg = startup_snap_reg.clone();
                    let rt = startup_runtime.clone();
                    let meta = startup_meta.clone();
                    let mod_name = module_name.clone();
                    tokio::task::spawn_blocking(move || {
                        let prog = prog_clone.blocking_read().clone();
                        eval::set_shared_runtime(Some(rt));
                        eval::set_shared_meta(Some(meta));
                        eval::set_shared_program(Some(std::sync::Arc::new(prog.clone())));
                        eval::set_shared_program_mut(Some(eval::make_shared_program_mut(&prog)));
                        let task_id = 0;
                        let handle = coroutine::CoroutineHandle::new_with_task(sender, task_id, registry, snap_reg);
                        let mut env = eval::Env::new_with_shared_interner(&prog.shared_interner);
                        env.set("__coroutine_handle", eval::Value::CoroutineHandle(handle));
                        env.set("__module_name", eval::Value::String(std::sync::Arc::new(mod_name.clone())));
                        match eval::eval_function_body_named(&prog, &format!("{}.startup", mod_name), &startup_fn.body, &mut env) {
                            Ok(val) => eprintln!("[startup] {}.startup -> {}", mod_name, val),
                            Err(e) => eprintln!("[startup] {}.startup error: {}", mod_name, e),
                        }
                    });
                }

                // Auto-register module-level source declarations
                let module_sources: Vec<(String, Vec<ast::SourceDecl>)> = {
                    let prog = startup_program.blocking_read();
                    prog.modules.iter()
                        .filter(|m| !m.sources.is_empty())
                        .map(|m| (m.name.clone(), m.sources.clone()))
                        .collect()
                };
                for (module_name, sources) in module_sources {
                    for src in sources {
                        let interval_ms = src.config.iter()
                            .find(|(k, _)| k == "interval")
                            .and_then(|(_, v)| v.parse::<u64>().ok());
                        let handler = if src.handler.contains('.') {
                            src.handler.clone()
                        } else {
                            format!("{}.{}", module_name, src.handler)
                        };
                        eprintln!("[startup] registering source {}.{} ({} {})",
                            module_name, src.name, src.source_type,
                            src.config.iter().map(|(k,v)| format!("{}={}", k, v)).collect::<Vec<_>>().join(" "));
                        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
                        let _ = io_sender_for_startup.blocking_send(
                            coroutine::IoRequest::SourceAdd {
                                module_name: module_name.clone(),
                                source_type: src.source_type.clone(),
                                interval_ms,
                                alias: src.name.clone(),
                                handler,
                                reply: reply_tx,
                            }
                        );
                        match reply_rx.blocking_recv() {
                            Ok(Ok(msg)) => eprintln!("[startup] source {}.{}: {}", module_name, src.name, msg),
                            Ok(Err(e)) => eprintln!("[startup] source {}.{} error: {}", module_name, src.name, e),
                            Err(_) => eprintln!("[startup] source {}.{}: reply channel closed", module_name, src.name),
                        }
                    }
                }
            }

            let project_dir = std::env::current_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| ".".to_string());

            // Validate: binary must be inside the git dir for !opencode restart to work
            let resolved_git_dir = opencode_git_dir.as_deref().unwrap_or(&project_dir);
            let exe_path = std::env::current_exe().unwrap_or_default();
            let exe_str = exe_path.to_string_lossy();
            if !exe_str.contains(resolved_git_dir) {
                eprintln!("WARNING: AdapsisOS binary ({}) is not inside the opencode git dir ({}).", exe_str, resolved_git_dir);
                eprintln!("  !opencode self-restart will not pick up rebuilt binaries.");
                eprintln!("  Run from: {}/target/release/adapsis", resolved_git_dir);
            }

            // Self-trigger channel: events feed back into the AI
            let (trigger_tx, mut trigger_rx) = tokio::sync::mpsc::channel::<String>(32);

            // Set up structured log file
            let ai_log = {
                let f = tokio::fs::OpenOptions::new()
                    .create(true).append(true)
                    .open(&log_file).await?;
                Some(std::sync::Arc::new(tokio::sync::Mutex::new(f)))
            };
            let train_log = {
                let f = tokio::fs::OpenOptions::new()
                    .create(true).append(true)
                    .open(&training_log).await?;
                Some(std::sync::Arc::new(tokio::sync::Mutex::new(f)))
            };

            // Build the three independent tiers from the loaded session.
            let tier1_program = std::sync::Arc::new(tokio::sync::RwLock::new(sess.program.clone()));

            let config = api::AppConfig {
                program: tier1_program,
                meta: shared_meta.clone(),
                llm_url: url.clone(),
                llm_model: model.clone(),
                llm_api_key: api_key.clone(),
                project_dir: project_dir.clone(),
                io_sender: Some(io_sender),
                self_trigger: trigger_tx,
                task_registry: Some(runtime.task_registry.clone()),
                snapshot_registry: Some(runtime.snapshot_registry.clone()),
                log_file: ai_log,
                training_log: train_log,
                jit_cache: eval::new_jit_cache(),
                event_broadcast: tokio::sync::broadcast::channel(256).0,
                max_iterations,
                opencode_lock: std::sync::Arc::new(tokio::sync::Mutex::new(())),
                message_queue: std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new())),
                opencode_git_dir: opencode_git_dir.unwrap_or_else(|| project_dir.clone()),
                runtime: shared_runtime.clone(),
                sessions: std::sync::Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new())),
            };

            // Clone tier handles before config is moved into the router
            let save_program = config.program.clone();
            let save_meta = config.meta.clone();
            let save_runtime = config.runtime.clone();

            let app = axum::Router::new()
                .route(
                    "/",
                    axum::routing::get(|| async {
                        axum::response::Html(include_str!("../web/adapsis.html"))
                    }),
                )
                .merge(api::router_with_llm(config))
                .layer(tower_http::cors::CorsLayer::permissive());

            let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
                .await
                .map_err(|e| anyhow::anyhow!("Cannot bind port {port}: {e}. Try -p {}", port + 1))?;
            println!("AdapsisOS running at http://127.0.0.1:{port}");
            println!("  API:     http://127.0.0.1:{port}/api/");
            println!("  Browser: http://127.0.0.1:{port}/");
            println!();

            if daemonize {
                // We verified the port works. Now respawn without -d.
                // Use SO_REUSEADDR equivalent by dropping listener first and
                // giving the OS a moment.
                drop(listener);
                std::thread::sleep(std::time::Duration::from_millis(100));
                
                let exe = std::env::current_exe()?;
                let mut args: Vec<String> = std::env::args().collect();
                args.retain(|a| a != "-d" && a != "--daemonize");
                
                let log_file = std::fs::File::create("/tmp/adapsisos.log")
                    .unwrap_or_else(|_| std::fs::File::open("/dev/null").unwrap());
                
                let child = std::process::Command::new(&exe)
                    .args(&args[1..])
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::from(log_file.try_clone().unwrap()))
                    .stderr(std::process::Stdio::from(log_file))
                    .spawn()?;
                println!("Daemonized: PID {}", child.id());
                return Ok(());
            }

            // Auto-save session periodically.
            // Sync tier state into the session shim before saving so that
            // changes made through the tier locks are persisted.
            let save_path = session.clone();
            let autosave_program = save_program.clone();
            let autosave_meta = save_meta.clone();
            let autosave_runtime = save_runtime.clone();
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                    let session = snapshot_from_tiers(&autosave_program, &autosave_meta, &autosave_runtime).await;
                    if let Err(e) = session.save(std::path::Path::new(&save_path)) {
                        eprintln!("auto-save failed: {e}");
                    }
                }
            });

            // Self-trigger loop: process system events through the AI
            // (use save_* clones since config was moved into the router above)
            let trigger_program = save_program.clone();
            let trigger_meta = save_meta.clone();
            let trigger_runtime = save_runtime.clone();
            let trigger_url = url.clone();
            let trigger_model = model.clone();
            let trigger_key = api_key.clone();
            tokio::spawn(async move {
                while let Some(event_message) = trigger_rx.recv().await {
                    eprintln!("[self-trigger] {}", event_message.chars().take(80).collect::<String>());
                    let llm = llm::LlmClient::new_with_model_and_key(&trigger_url, &trigger_model, trigger_key.clone());

                    // Add event as tool message — AI decides whether to act
                    let messages = {
                        let mut meta = trigger_meta.lock().unwrap();
                        meta.chat_messages.push(crate::session::ChatMessage {
                            role: "tool".to_string(),
                            content: event_message.clone(),
                        });
                        meta.chat_messages.iter().map(|m| match m.role.as_str() {
                            "system" => llm::ChatMessage::system(m.content.clone()),
                            "assistant" => llm::ChatMessage::assistant(&m.content),
                            _ => llm::ChatMessage::user(m.content.clone()),
                        }).collect::<Vec<_>>()
                    };

                    match llm.generate(messages).await {
                        Ok(output) => {
                            let code = output.code.clone();
                            eprintln!("[self-trigger:response] {}...", output.text.chars().take(100).collect::<String>());

                            // Apply code if any
                            if !code.is_empty() && code.trim() != "DONE" {
                                let mut program = trigger_program.read().await.clone();
                                let mut runtime = trigger_runtime.read().unwrap().clone();
                                let mut meta = trigger_meta.lock().unwrap().clone();
                                let mut sandbox = None;
                                if let Ok(ops) = crate::parser::parse(&code) {
                                    let mut fns_removed = false;
                                    for op in &ops {
                                        match op {
                                            crate::parser::Operation::Function(f) => { program.functions.retain(|e| e.name != f.name); fns_removed = true; }
                                            crate::parser::Operation::Type(t) => { let n = t.name.clone(); program.types.retain(|e| e.name() != n); }
                                            _ => {}
                                        }
                                    }
                                    if fns_removed {
                                        program.rebuild_function_index();
                                    }
                                    if let Ok(results) = crate::session::apply_to_tiers(&mut program, &mut runtime, &mut meta, &mut sandbox, &code) {
                                        for (msg, ok) in &results {
                                            eprintln!("[self-trigger:{}] {msg}", if *ok { "ok" } else { "err" });
                                        }
                                    }
                                }
                                meta.chat_messages.push(crate::session::ChatMessage {
                                    role: "assistant".to_string(),
                                    content: format!("[auto-response] {}", output.text.chars().take(200).collect::<String>()),
                                });
                                *trigger_program.write().await = program;
                                *trigger_runtime.write().unwrap() = runtime;
                                *trigger_meta.lock().unwrap() = meta;
                            }
                        }
                        Err(e) => {
                            eprintln!("[self-trigger:error] {e}");
                        }
                    }
                }
            });

            // Autonomous mode: build the initial message from restart context + goal.
            {
                let goal = autonomous;
                let goal_message = if let Some(context) = restart_context {
                    // Restarted with prior session — include full context
                    eprintln!("[autonomous] restarting with context");
                    match goal {
                        Some(ref g) => format!("{context}\n\n## Goal\n{g}"),
                        None => format!("{context}\n\nContinue where you left off."),
                    }
                } else if goal.as_deref() == Some("roadmap") {
                    let roadmap_path = format!("{}/ROADMAP.md", project_dir);
                    match std::fs::read_to_string(&roadmap_path) {
                        Ok(content) => format!(
                            "You are running in autonomous mode. Here is the project roadmap:\n\n{content}\n\n\
                             First, use !roadmap add to populate your roadmap with the undone items above. \
                             Then check !roadmap, pick the first undone item, create a !plan, and start working. \
                             Use !roadmap done N when you finish an item. Use !opencode for Rust-level changes. \
                             Keep going — when one item is done, move to the next."
                        ),
                        Err(_) => "You are running in autonomous mode. Check !roadmap for tasks. If empty, identify improvements and !roadmap add them. Then start working.".to_string(),
                    }
                } else if let Some(ref g) = goal {
                    format!(
                        "You are running in autonomous mode. Your goal:\n\n{g}\n\n\
                         Create a plan, then start building. Use !opencode when you need Rust-level changes. \
                         Keep going until the goal is complete or you get stuck and need user input."
                    )
                } else {
                    "Check !roadmap for undone items. If there are any, pick the first one, create a !plan, and start working. If empty, idle.".to_string()
                };

                eprintln!("[autonomous] injecting goal: {}...", goal_message.chars().take(100).collect::<String>());
                let auto_port = port;
                tokio::spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    let client = reqwest::Client::new();
                    let mut is_first = true;
                    loop {
                        let msg = if is_first {
                            is_first = false;
                            goal_message.clone()
                        } else {
                            // Check for injected messages first
                            let injected = match client.post(format!("http://127.0.0.1:{auto_port}/api/drain-queue"))
                                .send().await {
                                    Ok(r) => r.json::<serde_json::Value>().await.ok()
                                        .and_then(|q| q.get("messages").cloned())
                                        .and_then(|m| m.as_array().cloned())
                                        .and_then(|arr| {
                                            let non_empty: Vec<String> = arr.iter()
                                                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                                .filter(|s| !s.is_empty())
                                                .collect();
                                            if non_empty.is_empty() { None } else { Some(non_empty.join("\n\n")) }
                                        }),
                                    Err(_) => None,
                                };

                            if let Some(injected_msg) = injected {
                                eprintln!("[autonomous] processing injected: {}...", injected_msg.chars().take(80).collect::<String>());
                                injected_msg
                            } else {
                                // Check session state to give the right nudge
                                let status = match client.get(format!("http://127.0.0.1:{auto_port}/api/status"))
                                    .send().await {
                                        Ok(r) => r.json::<serde_json::Value>().await.ok(),
                                        Err(_) => None,
                                    };

                                if let Some(ref status) = status {
                                    let plan = status.get("plan").and_then(|p| p.as_array());
                                    let has_pending_plan = plan.map(|p| p.iter().any(|s| {
                                        let st = s.get("status").and_then(|s| s.as_str()).unwrap_or("");
                                        st == "pending" || st == "in_progress"
                                    })).unwrap_or(false);

                                    let roadmap = status.get("roadmap").and_then(|r| r.as_array());
                                    let has_undone_roadmap = roadmap.map(|r| r.iter().any(|item| {
                                        item.get("done").and_then(|d| d.as_bool()) == Some(false)
                                    })).unwrap_or(false);

                                    if has_pending_plan {
                                        "You hit the iteration limit but your plan has unfinished steps. Continue working on the current plan.".to_string()
                                    } else if has_undone_roadmap {
                                        "Plan completed. Use !roadmap done N to mark the current roadmap item done, then check !roadmap for the next undone item. Create a new !plan and start working on it.".to_string()
                                    } else {
                                        // Nothing left — idle, but check queue periodically
                                        eprintln!("[autonomous] all roadmap items done, idling...");
                                        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                                        continue;
                                    }
                                } else {
                                    "Continue working. Check !roadmap and !plan for current state.".to_string()
                                }
                            }
                        };
                        eprintln!("[autonomous] sending: {}...", msg.chars().take(80).collect::<String>());
                        match client.post(format!("http://127.0.0.1:{auto_port}/api/ask-stream"))
                            .json(&serde_json::json!({"message": msg}))
                            .send().await
                        {
                            Ok(resp) => {
                                use futures::StreamExt;
                                let mut stream = resp.bytes_stream();
                                let mut raw_buf: Vec<u8> = Vec::new();
                                while let Some(chunk) = stream.next().await {
                                    if let Ok(bytes) = chunk {
                                        raw_buf.extend_from_slice(&bytes);
                                        let valid_up_to = match std::str::from_utf8(&raw_buf) {
                                            Ok(_) => raw_buf.len(),
                                            Err(e) => e.valid_up_to(),
                                        };
                                        if valid_up_to == 0 { continue; }
                                        let text = std::str::from_utf8(&raw_buf[..valid_up_to]).unwrap();
                                        let mut got_end = false;
                                        for line in text.lines() {
                                            if let Some(data) = line.strip_prefix("data: ") {
                                                if let Ok(event) = serde_json::from_str::<serde_json::Value>(data) {
                                                    let etype = event.get("type").and_then(|t| t.as_str()).unwrap_or("");
                                                    match etype {
                                                        // High-frequency streaming chunks — don't spam logs
                                                        "content" | "thinking" => {}
                                                        "end" | "done" => {
                                                            eprintln!("[autonomous] {etype}");
                                                            got_end = true;
                                                        }
                                                        "text" => {
                                                            let preview = event.get("text")
                                                                .and_then(|t| t.as_str())
                                                                .map(|s| s.chars().take(120).collect::<String>())
                                                                .unwrap_or_default();
                                                            eprintln!("[ai-text] {preview}");
                                                        }
                                                        "code" => {
                                                            let preview = event.get("code")
                                                                .and_then(|t| t.as_str())
                                                                .map(|s| s.chars().take(120).collect::<String>())
                                                                .unwrap_or_default();
                                                            eprintln!("[code] {preview}");
                                                        }
                                                        "error" => {
                                                            let msg = event.get("message")
                                                                .and_then(|t| t.as_str())
                                                                .unwrap_or("unknown");
                                                            eprintln!("[error] {msg}");
                                                        }
                                                        "feedback" => {
                                                            let msg = event.get("message")
                                                                .and_then(|t| t.as_str())
                                                                .map(|s| s.chars().take(200).collect::<String>())
                                                                .unwrap_or_default();
                                                            eprintln!("[feedback] {msg}");
                                                        }
                                                        _ => {
                                                            let detail = event.get("message")
                                                                .and_then(|t| t.as_str())
                                                                .map(|s| format!(": {}", s.chars().take(100).collect::<String>()))
                                                                .unwrap_or_default();
                                                            eprintln!("[autonomous] {etype}{detail}");
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        raw_buf.drain(..valid_up_to);
                                        if got_end { break; }
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("[autonomous] request failed: {e}, retrying in 10s");
                                tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                            }
                        }
                        // Brief pause between autonomous rounds
                        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    }
                });
            }

            // Telegram bot (long-polling, runs alongside the HTTP server)
            if let Some(token) = telegram_token {
                let bot = telegram::TelegramBot::new(
                    token,
                    telegram_admin_chat_id,
                    port,
                    url.clone(),
                    model.clone(),
                    api_key.clone(),
                );
                tokio::spawn(async move {
                    if let Err(e) = bot.run().await {
                        eprintln!("[telegram] bot exited with error: {e}");
                    }
                });
                eprintln!("[telegram] bot polling started (admin_chat_id={telegram_admin_chat_id})");
            }

            match axum::serve(listener, app).await {
                Ok(()) => eprintln!("[adapsis] server exited cleanly"),
                Err(e) => eprintln!("[adapsis] server error: {e}"),
            }
        }
        Command::Ask { message, api } => {
            let msg = message.join(" ");
            let client = reqwest::Client::new();
            let resp: serde_json::Value = client
                .post(format!("{api}/api/ask"))
                .json(&serde_json::json!({ "message": msg }))
                .send().await?
                .json().await?;
            if let Some(reply) = resp.get("reply").and_then(|r| r.as_str()) {
                if !reply.is_empty() { println!("{reply}"); }
            }
            if let Some(code) = resp.get("code").and_then(|c| c.as_str()) {
                if !code.is_empty() { println!("\x1b[36m{code}\x1b[0m"); }
            }
            if let Some(results) = resp.get("results").and_then(|r| r.as_array()) {
                for r in results {
                    let ok = r.get("success").and_then(|s| s.as_bool()).unwrap_or(false);
                    let msg = r.get("message").and_then(|m| m.as_str()).unwrap_or("");
                    if ok {
                        println!("\x1b[32m  OK: {msg}\x1b[0m");
                    } else {
                        println!("\x1b[31m  ERR: {msg}\x1b[0m");
                    }
                }
            }
            if let Some(tests) = resp.get("test_results").and_then(|r| r.as_array()) {
                for r in tests {
                    let pass = r.get("pass").and_then(|s| s.as_bool()).unwrap_or(false);
                    let msg = r.get("message").and_then(|m| m.as_str()).unwrap_or("");
                    if pass {
                        println!("\x1b[32m  PASS: {msg}\x1b[0m");
                    } else {
                        println!("\x1b[31m  FAIL: {msg}\x1b[0m");
                    }
                }
            }
        }
        Command::Status { api } => {
            let resp: serde_json::Value = reqwest::get(format!("{api}/api/status"))
                .await?.json().await?;
            println!("Revision: {}", resp.get("revision").unwrap_or(&serde_json::json!(0)));
            if let Some(fns) = resp.get("functions").and_then(|f| f.as_array()) {
                println!("Functions ({}): {}", fns.len(), fns.iter().filter_map(|f| f.as_str()).collect::<Vec<_>>().join(", "));
            }
            if let Some(types) = resp.get("types").and_then(|t| t.as_array()) {
                if !types.is_empty() {
                    println!("Types ({}): {}", types.len(), types.iter().filter_map(|t| t.as_str()).collect::<Vec<_>>().join(", "));
                }
            }
        }
        Command::Mutate { source, api } => {
            let src = source.join(" ");
            let client = reqwest::Client::new();
            let resp: serde_json::Value = client
                .post(format!("{api}/api/mutate"))
                .json(&serde_json::json!({ "source": src }))
                .send().await?.json().await?;
            println!("Revision: {}", resp.get("revision").unwrap_or(&serde_json::json!(0)));
            if let Some(results) = resp.get("results").and_then(|r| r.as_array()) {
                for r in results {
                    let ok = r.get("success").and_then(|s| s.as_bool()).unwrap_or(false);
                    let msg = r.get("message").and_then(|m| m.as_str()).unwrap_or("");
                    println!("  {}: {msg}", if ok { "OK" } else { "ERR" });
                }
            }
        }
        Command::Eval { expr, api } => {
            let parts = expr.join(" ");
            // Try parsing as inline expression first; if it succeeds and isn't
            // a bare identifier (which is the existing func-name syntax), send
            // it as an inline expression.
            let is_inline = if let Ok(parsed) = parser::parse_expr_pub(0, &parts) {
                !matches!(parsed, parser::Expr::Ident(_))
            } else {
                false
            };
            let client = reqwest::Client::new();
            let resp: serde_json::Value = if is_inline {
                client
                    .post(format!("{api}/api/eval"))
                    .json(&serde_json::json!({ "function": "", "expression": parts }))
                    .send().await?.json().await?
            } else {
                let (func, input) = parts.split_once(' ').unwrap_or((&parts, ""));
                client
                    .post(format!("{api}/api/eval"))
                    .json(&serde_json::json!({ "function": func, "input": input }))
                    .send().await?.json().await?
            };
            let result = resp.get("result").and_then(|r| r.as_str()).unwrap_or("(none)");
            let success = resp.get("success").and_then(|s| s.as_bool()).unwrap_or(false);
            let compiled = resp.get("compiled").and_then(|c| c.as_bool()).unwrap_or(false);
            let tag = if compiled { " [compiled]" } else { "" };
            if success {
                println!("= {result}{tag}");
            } else {
                println!("Error: {result}");
            }
        }
        Command::Query { query, api } => {
            let q = query.join(" ");
            let client = reqwest::Client::new();
            let resp: serde_json::Value = client
                .post(format!("{api}/api/query"))
                .json(&serde_json::json!({ "query": q }))
                .send().await?.json().await?;
            let response = resp.get("response").and_then(|r| r.as_str()).unwrap_or("(none)");
            println!("{response}");
        }
    }

    Ok(())
}
