mod api;
mod ast;
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
        #[arg(long, default_value = "default")]
        model: String,

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
        #[arg(long, default_value = "default")]
        model: String,

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
    },

    /// Interactive REPL
    Repl {
        /// LLM server URL (OpenAI-compatible)
        #[arg(short, long, default_value = "http://127.0.0.1:8081")]
        url: String,

        /// Model name
        #[arg(long, default_value = "default")]
        model: String,

        /// Session file path (auto-saves)
        #[arg(short, long)]
        session: Option<String>,
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
            max_iterations,
        } => {
            let llm_client = llm::LlmClient::new_with_model(&url, &model);
            let mut orch = orchestrator::Orchestrator::new(llm_client, max_iterations);
            orch.run(&task).await?;
        }
        Command::Architect {
            task,
            url,
            model,
            max_iterations,
            port,
        } => {
            let llm_client = llm::LlmClient::new_with_model(&url, &model);
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
            max_iterations,
            port,
        } => {
            let llm_client = llm::LlmClient::new(&url);
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
        Command::RunAsync { path, func } => {
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

            let (runtime, mut io_rx) = coroutine::Runtime::new();
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
            let io_loop = async move {
                while let Some(request) = io_rx.recv().await {
                    match request {
                        coroutine::IoRequest::Spawn { function_name, args } => {
                            // Spawn a new coroutine for this function
                            let prog = program_for_spawn.clone();
                            let sender = io_sender_for_spawn.clone();
                            tokio::task::spawn_blocking(move || {
                                let func_decl = match prog.get_function(&function_name) {
                                    Some(f) => f,
                                    None => {
                                        eprintln!("spawn: function `{function_name}` not found");
                                        return;
                                    }
                                };
                                let handle = coroutine::CoroutineHandle::new(sender);
                                let mut env = eval::Env::new();
                                env.set("__coroutine_handle", eval::Value::CoroutineHandle(handle));
                                // Bind args to params
                                for (i, param) in func_decl.params.iter().enumerate() {
                                    if let Some(val) = args.get(i) {
                                        env.set(&param.name, val.clone());
                                    }
                                }
                                if let Err(e) = eval::eval_function_body_pub(&prog, &func_decl.body, &mut env) {
                                    // Errors in spawned coroutines are logged, not fatal
                                    eprintln!("spawn {function_name}: {e}");
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
        Command::Repl { url, model, session } => {
            let llm_client = llm::LlmClient::new_with_model(&url, &model);
            let session_path = session.map(std::path::PathBuf::from);
            repl::run_repl(llm_client, session_path).await?;
        }
        Command::Os { port, session, url } => {
            let session_path = std::path::Path::new(&session);
            let sess = if session_path.exists() {
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

            let shared_session = std::sync::Arc::new(tokio::sync::Mutex::new(sess));

            // Set up coroutine runtime for async IO
            let (runtime, mut io_rx) = coroutine::Runtime::new();
            let runtime = std::sync::Arc::new(runtime);
            let io_sender = runtime.io_sender();

            // Spawn IO event loop
            let rt = runtime.clone();
            tokio::spawn(async move {
                while let Some(request) = io_rx.recv().await {
                    let rt = rt.clone();
                    tokio::spawn(async move {
                        rt.handle_io(request).await;
                    });
                }
            });

            let project_dir = std::env::current_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| ".".to_string());

            let config = api::AppConfig {
                session: shared_session.clone(),
                llm_url: url.clone(),
                llm_model: "default".to_string(),
                project_dir,
                io_sender: Some(io_sender),
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

            axum::serve(listener, app).await?;
        }
    }

    Ok(())
}
