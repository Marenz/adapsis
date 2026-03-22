mod api;
mod ast;
pub mod builtins;
mod compiler;
mod coroutine;
mod eval;
mod events;
mod llm;
mod orchestrator;
mod parser;
mod prompt;
mod repl;
mod server;
mod session;
mod typeck;
mod validator;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "forge", about = "Forge — AI-first programming language")]
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

    /// Parse a .forge file and validate it
    Check {
        /// Path to .forge file
        path: String,
    },

    /// Parse a .forge file and run its !test blocks
    Test {
        /// Path to .forge file
        path: String,
    },

    /// Compile a .forge file to native code and run it
    Compile {
        /// Path to .forge file
        path: String,

        /// Function to call
        #[arg(short, long)]
        func: String,

        /// Arguments (comma-separated integers)
        #[arg(short, long, default_value = "")]
        args: String,
    },

    /// Run a Forge program with async IO (coroutine runtime)
    RunAsync {
        /// Path to .forge file
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

    /// Interactive REPL (auto-starts ForgeOS if not running)
    Repl {
        /// ForgeOS API URL (auto-detected if not specified)
        #[arg(short, long, default_value = "http://127.0.0.1:3001")]
        api: String,

        /// Session file (used when auto-starting ForgeOS)
        #[arg(short, long, default_value = "forgeos-session.json")]
        session: String,

        /// LLM server URL (used when auto-starting)
        #[arg(short, long, env = "FORGE_LLM_URL", default_value = "http://127.0.0.1:8081")]
        url: String,

        /// Model name (used when auto-starting)
        #[arg(long, env = "FORGE_MODEL")]
        model: Option<String>,
    },

    /// Start ForgeOS — HTTP API + browser UI + session persistence
    Os {
        /// HTTP port
        #[arg(short, long, default_value_t = 3001)]
        port: u16,

        /// Session file path
        #[arg(short, long, default_value = "forgeos-session.json")]
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
        #[arg(long, default_value = "forgeos.log")]
        log_file: String,

        /// Git repository for !opencode worktrees (defaults to project dir)
        #[arg(long, env = "FORGE_OPENCODE_GIT_DIR")]
        opencode_git_dir: Option<String>,

        /// Directory for !opencode worktrees (defaults to {project_dir}/../forge-opencode-work)
        #[arg(long, env = "FORGE_OPENCODE_WORKTREE_DIR")]
        opencode_worktree_dir: Option<String>,

        /// Maximum iterations per AI request (default 20)
        #[arg(long, default_value_t = 20)]
        max_iterations: usize,
    },

    /// Send a message to a running ForgeOS instance
    Ask {
        /// The message to send
        message: Vec<String>,

        /// ForgeOS API URL
        #[arg(short, long, default_value = "http://127.0.0.1:3001")]
        api: String,
    },

    /// Show status of a running ForgeOS instance
    Status {
        /// ForgeOS API URL
        #[arg(short, long, default_value = "http://127.0.0.1:3001")]
        api: String,
    },

    /// Apply Forge code to a running ForgeOS instance
    Mutate {
        /// Forge source code
        source: Vec<String>,

        /// ForgeOS API URL
        #[arg(short, long, default_value = "http://127.0.0.1:3001")]
        api: String,
    },

    /// Eval a function on a running ForgeOS instance
    Eval {
        /// Function name and arguments
        expr: Vec<String>,

        /// ForgeOS API URL
        #[arg(short, long, default_value = "http://127.0.0.1:3001")]
        api: String,
    },

    /// Query a running ForgeOS instance
    Query {
        /// Query string (?symbols, ?source fn, ?deps fn, etc.)
        query: Vec<String>,

        /// ForgeOS API URL
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
                println!("Forge architect UI at http://127.0.0.1:{port}");

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
            for op in &operations {
                match op {
                    parser::Operation::Test(_) => test_ops.push(op.clone()),
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
                    parser::Operation::Query(query) => {
                        let table = typeck::build_symbol_table(&program);
                        let response = typeck::handle_query(&program, &table, query);
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
                        match eval::eval_test_case(&program, &test.function_name, case) {
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
            let func_clone = func.clone();
            let eval_task = tokio::task::spawn_blocking(move || {
                let func_decl = program_clone.get_function(&func_clone)
                    .ok_or_else(|| anyhow::anyhow!("function `{func_clone}` not found"))?;

                let mut env = eval::Env::new();
                env.set("__coroutine_handle", eval::Value::CoroutineHandle(handle));

                eval::eval_function_body_pub(&program_clone, &func_decl.body, &mut env)
            });

            // Event loop — process IO requests from coroutines
            let rt = runtime.clone();
            let program_for_spawn = program.clone();
            let io_sender_for_spawn = runtime.io_sender();
            let task_registry_for_spawn = runtime.task_registry.clone();
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
                            tokio::task::spawn_blocking(move || {
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
                                let handle = coroutine::CoroutineHandle::new_with_task(sender, task_id, registry.clone());
                                let mut env = eval::Env::new();
                                env.set("__coroutine_handle", eval::Value::CoroutineHandle(handle));
                                // Bind args to params
                                for (i, param) in func_decl.params.iter().enumerate() {
                                    if let Some(val) = args.get(i) {
                                        env.set(&param.name, val.clone());
                                    }
                                }
                                match eval::eval_function_body_pub(&prog, &func_decl.body, &mut env) {
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
            // Check if ForgeOS is already running
            let client = reqwest::Client::new();
            let running = client.get(format!("{api}/api/status"))
                .send().await
                .map(|r| r.status().is_success())
                .unwrap_or(false);

            let api_url = if running {
                api
            } else {
                // Auto-start ForgeOS in the background
                let model = model.unwrap_or_else(|| {
                    eprintln!("No model specified. Set FORGE_MODEL env var or use --model.");
                    eprintln!("Example: FORGE_MODEL=anthropic/claude-haiku-4-5-20251001 forge repl");
                    std::process::exit(1);
                });

                // Extract port from api URL
                let port = api.rsplit(':').next()
                    .and_then(|p| p.parse::<u16>().ok())
                    .unwrap_or(3001);

                eprintln!("No ForgeOS instance detected. Starting one...");

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
                    eprintln!("Failed to start ForgeOS: {stderr}");
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
        Command::Os { port, session, url, model, api_key, daemonize, autonomous, log_file, opencode_git_dir, opencode_worktree_dir, max_iterations } => {
            let session_path = std::path::Path::new(&session);
            let mut sess = if session_path.exists() {
                println!("Loading session from {session}...");
                let s = session::Session::load(session_path)?;
                println!(
                    "Loaded: revision {}, {} mutations",
                    s.revision,
                    s.mutations.len()
                );
                s
            } else {
                println!("New session (saving to {session})");
                session::Session::new()
            };

            // In ForgeOS mode, enforce modules and tests
            sess.program.require_modules = true;

            let shared_session = std::sync::Arc::new(tokio::sync::Mutex::new(sess));

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
            let io_sender_for_spawn = runtime.io_sender();
            let shared_session_for_spawn = shared_session.clone();
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
                            let session_ref = shared_session_for_spawn.clone();
                            tokio::task::spawn_blocking(move || {
                                let session = session_ref.blocking_lock();
                                let func_decl = match session.program.get_function(&function_name) {
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
                                let program = session.program.clone();
                                drop(session);

                                let handle = coroutine::CoroutineHandle::new_with_task(sender, task_id, registry.clone());
                                let mut env = eval::Env::new();
                                env.set("__coroutine_handle", eval::Value::CoroutineHandle(handle));
                                for (i, param) in func_decl.params.iter().enumerate() {
                                    if let Some(val) = args.get(i) {
                                        env.set(&param.name, val.clone());
                                    }
                                }
                                match eval::eval_function_body_pub(&program, &func_decl.body, &mut env) {
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
                        _ => {
                            let rt = rt.clone();
                            tokio::spawn(async move {
                                rt.handle_io(request).await;
                            });
                        }
                    }
                }
            });

            let project_dir = std::env::current_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| ".".to_string());

            // Self-trigger channel: events feed back into the AI
            let (trigger_tx, mut trigger_rx) = tokio::sync::mpsc::channel::<String>(32);

            // Set up structured log file
            let ai_log = {
                let f = tokio::fs::OpenOptions::new()
                    .create(true).append(true)
                    .open(&log_file).await?;
                Some(std::sync::Arc::new(tokio::sync::Mutex::new(f)))
            };

            let config = api::AppConfig {
                session: shared_session.clone(),
                llm_url: url.clone(),
                llm_model: model.clone(),
                llm_api_key: api_key.clone(),
                project_dir: project_dir.clone(),
                io_sender: Some(io_sender),
                self_trigger: trigger_tx,
                task_registry: Some(runtime.task_registry.clone()),
                log_file: ai_log,
                jit_cache: eval::new_jit_cache(),
                event_broadcast: tokio::sync::broadcast::channel(256).0,
                max_iterations,
                opencode_lock: std::sync::Arc::new(tokio::sync::Mutex::new(())),
                opencode_git_dir: opencode_git_dir.unwrap_or_else(|| project_dir.clone()),
                opencode_worktree_dir: opencode_worktree_dir.unwrap_or_else(|| {
                    let p = std::path::Path::new(&project_dir).parent().unwrap_or(std::path::Path::new("."));
                    p.join("forge-opencode-work").to_string_lossy().to_string()
                }),
            };

            let app = axum::Router::new()
                .route(
                    "/",
                    axum::routing::get(|| async {
                        axum::response::Html(include_str!("../web/forgeos.html"))
                    }),
                )
                .merge(api::router_with_llm(config))
                .layer(tower_http::cors::CorsLayer::permissive());

            let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
                .await
                .map_err(|e| anyhow::anyhow!("Cannot bind port {port}: {e}. Try -p {}", port + 1))?;
            println!("ForgeOS running at http://127.0.0.1:{port}");
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
                
                let log_file = std::fs::File::create("/tmp/forgeos.log")
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

            // Auto-save session periodically
            let save_session = shared_session.clone();
            let save_path = session.clone();
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                    let session = save_session.lock().await;
                    if let Err(e) = session.save(std::path::Path::new(&save_path)) {
                        eprintln!("auto-save failed: {e}");
                    }
                }
            });

            // Self-trigger loop: process system events through the AI
            let trigger_session = shared_session.clone();
            let trigger_url = url.clone();
            let trigger_model = model.clone();
            let trigger_key = api_key.clone();
            tokio::spawn(async move {
                while let Some(event_message) = trigger_rx.recv().await {
                    eprintln!("[self-trigger] {}", event_message.chars().take(80).collect::<String>());
                    let llm = llm::LlmClient::new_with_model_and_key(&trigger_url, &trigger_model, trigger_key.clone());

                    // Add event as tool message — AI decides whether to act
                    let messages = {
                        let mut session = trigger_session.lock().await;
                        session.chat_messages.push(crate::session::ChatMessage {
                            role: "tool".to_string(),
                            content: event_message.clone(),
                        });
                        session.chat_messages.iter().map(|m| match m.role.as_str() {
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
                                let mut session = trigger_session.lock().await;
                                if let Ok(ops) = crate::parser::parse(&code) {
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
                                    if let Ok(results) = session.apply(&code) {
                                        for (msg, ok) in &results {
                                            eprintln!("[self-trigger:{}] {msg}", if *ok { "ok" } else { "err" });
                                        }
                                    }
                                }
                                session.chat_messages.push(crate::session::ChatMessage {
                                    role: "assistant".to_string(),
                                    content: format!("[auto-response] {}", output.text.chars().take(200).collect::<String>()),
                                });
                            }
                        }
                        Err(e) => {
                            eprintln!("[self-trigger:error] {e}");
                        }
                    }
                }
            });

            // Autonomous mode: inject goal as the first message after startup
            // Skip if session already has chat history (e.g. after !opencode restart)
            let session_has_history = shared_session.lock().await.chat_messages.len() > 1;
            if let Some(goal) = autonomous {
                if session_has_history {
                    eprintln!("[autonomous] session has history, injecting continue message instead of full goal");
                    let auto_port = port;
                    tokio::spawn(async move {
                        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                        let client = reqwest::Client::new();
                        let _ = client.post(format!("http://127.0.0.1:{auto_port}/api/ask-stream"))
                            .json(&serde_json::json!({"message": "ForgeOS was restarted after an !opencode change. The runtime has been updated. Continue where you left off — check ?symbols and ?tasks, then keep working on your plan."}))
                            .send().await;
                    });
                } else {
                let goal_message = if goal == "roadmap" {
                    // Read the current priority from ROADMAP.md
                    let roadmap_path = format!("{}/ROADMAP.md", project_dir);
                    match std::fs::read_to_string(&roadmap_path) {
                        Ok(content) => format!(
                            "You are running in autonomous mode. Here is the project roadmap:\n\n{}\n\n\
                             Work on the current priority (the first item under 'In Progress' or the top of 'Next Targets'). \
                             Create a plan, then start building. Use !opencode when you need Rust-level changes. \
                             Keep going until the goal is complete or you get stuck and need user input.",
                            content
                        ),
                        Err(_) => "You are running in autonomous mode. Identify the most impactful improvement you can make to ForgeOS and start working on it.".to_string(),
                    }
                } else {
                    format!(
                        "You are running in autonomous mode. Your goal:\n\n{}\n\n\
                         Create a plan, then start building. Use !opencode when you need Rust-level changes. \
                         Keep going until the goal is complete or you get stuck and need user input.",
                        goal
                    )
                };

                eprintln!("[autonomous] injecting goal: {}...", goal_message.chars().take(100).collect::<String>());
                let auto_port = port;
                tokio::spawn(async move {
                    // Wait for the server to be ready
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    let client = reqwest::Client::new();
                    match client.post(format!("http://127.0.0.1:{auto_port}/api/ask-stream"))
                        .json(&serde_json::json!({"message": goal_message}))
                        .send().await
                    {
                        Ok(resp) => {
                            // Stream the response to log it
                            use futures::StreamExt;
                            let mut stream = resp.bytes_stream();
                            while let Some(chunk) = stream.next().await {
                                if let Ok(bytes) = chunk {
                                    let text = String::from_utf8_lossy(&bytes);
                                    // SSE events are logged via the handler, just consume them
                                    for line in text.lines() {
                                        if let Some(data) = line.strip_prefix("data: ") {
                                            if let Ok(event) = serde_json::from_str::<serde_json::Value>(data) {
                                                if event.get("type").and_then(|t| t.as_str()) == Some("end") {
                                                    eprintln!("[autonomous] first goal iteration complete");
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => eprintln!("[autonomous] failed to inject goal: {e}"),
                    }
                });
                } // else (fresh session)
            }

            match axum::serve(listener, app).await {
                Ok(()) => eprintln!("[forge] server exited cleanly"),
                Err(e) => eprintln!("[forge] server error: {e}"),
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
            let (func, input) = parts.split_once(' ').unwrap_or((&parts, ""));
            let client = reqwest::Client::new();
            let resp: serde_json::Value = client
                .post(format!("{api}/api/eval"))
                .json(&serde_json::json!({ "function": func, "input": input }))
                .send().await?.json().await?;
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
