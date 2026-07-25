mod api;
mod ast;
mod attachment;
mod builtins;
mod compiler;
mod coroutine;
mod eval;
mod events;
pub mod intern;
mod permissions;
pub mod library;
mod llm;
mod memory_graph;
mod orchestrator;
mod parser;
mod prompt;
mod repl;
mod server;
mod session;
mod shared_state;
mod typeck;
mod validator;
mod vm;

use anyhow::{Context as _, Result};
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

fn migrate_conversations_to_memory(
    session_path: &std::path::Path,
    database_path: &std::path::Path,
) -> Result<()> {
    let session = session::Session::load(session_path)
        .with_context(|| format!("load session {}", session_path.display()))?;
    let graph = memory_graph::MemoryGraph::open(database_path)
        .with_context(|| format!("open memory graph {}", database_path.display()))?;
    let modified_at_ms = std::fs::metadata(session_path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |duration| duration.as_millis().min(i64::MAX as u128) as i64);
    let mut contexts: Vec<_> = session.meta.conversations.contexts.iter().collect();
    contexts.sort_by_key(|(context, _)| *context);
    let total_messages: usize = contexts
        .iter()
        .map(|(_, conversation)| conversation.messages.len())
        .sum();
    let mut imported = 0usize;

    for (context, conversation) in contexts {
        for (index, message) in conversation.messages.iter().enumerate() {
            let graph_message_id = format!("legacy:{context}:{index}");
            if graph.has_message(&graph_message_id)? {
                imported += 1;
                continue;
            }
            let (speaker_id, speaker_name, content) = match message.role.as_str() {
                "assistant" => (
                    "agent:kronk".to_string(),
                    "Kronk".to_string(),
                    message.content.clone(),
                ),
                "system" => (
                    "system:adapsis".to_string(),
                    "Adapsis".to_string(),
                    message.content.clone(),
                ),
                _ => legacy_user_source(context, &message.content),
            };
            let age_from_end = conversation.messages.len().saturating_sub(index) as i64;
            graph.ingest_message(
                &memory_graph::SourceMessage {
                    id: graph_message_id,
                    platform_message_id: None,
                    context_id: context.clone(),
                    context_kind: if context.starts_with("telegram:group:") {
                        "telegram_group".to_string()
                    } else if context.starts_with("telegram:") {
                        "telegram_direct".to_string()
                    } else {
                        "internal".to_string()
                    },
                    speaker_id,
                    speaker_name,
                    role: message.role.clone(),
                    content,
                    created_at_ms: modified_at_ms.saturating_sub(age_from_end * 1_000),
                },
                "telegram:user:1815217",
            )?;
            imported += 1;
        }
    }

    println!(
        "Imported {imported}/{total_messages} messages from {} into {}",
        session_path.display(),
        database_path.display()
    );
    Ok(())
}

fn legacy_user_source(context: &str, content: &str) -> (String, String, String) {
    if let Some(rest) = content.strip_prefix("[user:") {
        if let Some((id, text)) = rest.split_once("] ") {
            let name = match id {
                "1815217" => "Marenz",
                "520125610" => "Sven",
                "47128798" => "Kata",
                _ => id,
            };
            return (
                format!("telegram:user:{id}"),
                name.to_string(),
                text.to_string(),
            );
        }
    }
    if let Some(id) = context
        .strip_prefix("telegram:user:")
        .or_else(|| context.strip_prefix("telegram:"))
    {
        return (
            format!("telegram:user:{id}"),
            id.to_string(),
            content.to_string(),
        );
    }
    (
        "system:runtime".to_string(),
        "Adapsis runtime".to_string(),
        content.to_string(),
    )
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

        /// OpenCode server URL to attach to (e.g. http://localhost:4096).
        /// If set, !opencode uses `--attach` to connect to a running server.
        /// If unset (default), opencode runs in standalone mode.
        #[arg(long)]
        opencode_attach: Option<String>,

        /// Maximum iterations per AI request (default 20)
        #[arg(long, default_value_t = 20)]
        max_iterations: usize,

        /// Process-level access cap: full, adapsis-only, user-only, execute-only.
        /// Defaults to adapsis-only so !opencode (which rebuilds and re-execs the
        /// runtime) is OFF by default — pass --access-level full to opt in.
        #[arg(long, default_value = "adapsis-only")]
        access_level: String,

        /// Path to permissions TOML file for per-model permissions
        #[arg(long)]
        permissions_file: Option<String>,

        /// Address(es) to bind the HTTP API to. Defaults to 127.0.0.1 (loopback
        /// only) so the code-executing API is not exposed on the network. Pass
        /// 0.0.0.0 to expose on all interfaces, or repeat the flag / use a
        /// comma-separated list to bind several specific interfaces, e.g.
        /// `--host 127.0.0.1 --host 10.0.0.4` or `--host 127.0.0.1,10.0.0.4`.
        #[arg(long, default_value = "127.0.0.1", value_delimiter = ',')]
        host: Vec<String>,
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

    /// Import persisted JSON conversations into the Ladybug memory graph
    MemoryMigrate {
        /// Session JSON file to import
        session: std::path::PathBuf,

        /// Ladybug database path
        #[arg(long, env = "ADAPSIS_MEMORY_DB")]
        database: Option<std::path::PathBuf>,
    },

    /// Show Ladybug memory graph statistics and episode checkpoints
    MemoryStats {
        /// Ladybug database path
        #[arg(long, env = "ADAPSIS_MEMORY_DB")]
        database: Option<std::path::PathBuf>,

        /// Also inspect the uncheckpointed prefix for this context
        #[arg(long)]
        context: Option<String>,
    },

    /// Recover a Ladybug WAL to its last checksum-valid transaction and checkpoint it
    MemoryRecover {
        /// Ladybug database path
        #[arg(long, env = "ADAPSIS_MEMORY_DB")]
        database: Option<std::path::PathBuf>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Ensure panics in any thread are logged to stderr
    std::panic::set_hook(Box::new(|info| {
        eprintln!("[PANIC] {info}");
        let bt = std::backtrace::Backtrace::force_capture();
        eprintln!("{bt}");
    }));
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::MemoryRecover { database } => {
            let database = database.unwrap_or_else(|| {
                dirs::config_dir()
                    .unwrap_or_else(|| std::path::PathBuf::from("."))
                    .join("adapsis")
                    .join("memory.lbug")
            });
            memory_graph::MemoryGraph::recover(&database)?;
            println!("Recovered and checkpointed {}", database.display());
        }
        Command::MemoryStats { database, context } => {
            let database = database.unwrap_or_else(|| {
                dirs::config_dir()
                    .unwrap_or_else(|| std::path::PathBuf::from("."))
                    .join("adapsis")
                    .join("memory.lbug")
            });
            let graph = memory_graph::MemoryGraph::open(&database)?;
            print!("{}", graph.describe()?);
            if let Some(context) = context {
                let pending = graph.pending_context_messages(&context, usize::MAX)?;
                let characters: usize = pending.iter().map(|message| message.content.chars().count()).sum();
                println!("Pending {context}: {} messages, {characters} characters", pending.len());
            }
        }
        Command::MemoryMigrate { session, database } => {
            let database = database.unwrap_or_else(|| {
                dirs::config_dir()
                    .unwrap_or_else(|| std::path::PathBuf::from("."))
                    .join("adapsis")
                    .join("memory.lbug")
            });
            migrate_conversations_to_memory(&session, &database)?;
        }
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
            let mut fn_stubs: Vec<session::FunctionStub> = vec![];
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
                    parser::Operation::Stub { function_name, patterns, response_expr } => {
                        let pattern_display = patterns.iter().map(|p| format!("\"{p}\"")).collect::<Vec<_>>().join(" ");
                        fn_stubs.push(session::FunctionStub {
                            function_name: function_name.clone(),
                            patterns: patterns.clone(),
                            response_expr: response_expr.clone(),
                        });
                        println!("OK: stub {function_name} {pattern_display} -> {response_expr}");
                    }
                    parser::Operation::Unstub => {
                        let count = fn_stubs.len();
                        fn_stubs.clear();
                        println!("OK: cleared {count} stubs");
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
            // Set up shared meta so function stubs are accessible during tests
            {
                let mut meta = session::SessionMeta::new();
                meta.io_mocks = io_mocks.clone();
                meta.function_stubs = fn_stubs.clone();
                eval::set_shared_meta(Some(std::sync::Arc::new(std::sync::Mutex::new(meta))));
            }
            let mut passed = 0usize;
            let mut failed = 0usize;
            for test_op in &test_ops {
                if let parser::Operation::Test(test) = test_op {
                    println!("\n--- Testing {} ---", test.function_name);
                    for (i, case) in test.cases.iter().enumerate() {
                        match eval::eval_test_case_with_mocks(&program, &test.function_name, case, &io_mocks, &[]) {
                            Ok(msg) => {
                                passed += 1;
                                println!("  PASS [{i}]: {msg}");
                            }
                            Err(e) => {
                                failed += 1;
                                eprintln!("  FAIL [{i}]: {e}");
                            }
                        }
                    }
                }
            }
            println!("\nTEST SUMMARY: {passed} passed, {failed} failed");
            if failed > 0 {
                std::process::exit(1);
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
                let ctx = eval::EvalContext {
                    runtime: None, meta: None, event_broadcast: None,
                    program_snapshot: Some(std::sync::Arc::new(program_clone.clone())),
                    program_mut: Some(program_mut_clone),
                };
                ctx.install();
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
            let source_registry: coroutine::SourceRegistry =
                std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
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
                                let ctx = eval::EvalContext {
                                    runtime: None, meta: None, event_broadcast: None,
                                    program_snapshot: Some(std::sync::Arc::new(prog.clone())),
                                    program_mut: Some(eval::make_shared_program_mut(&prog)),
                                };
                                ctx.install();
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
                                    // Replace semantics: drop any existing source with
                                    // the same module.alias key (abort its task) first.
                                    coroutine::remove_source(&source_registry, &module_name, &alias);
                                    let prog = program_for_spawn.clone();
                                    let sender = io_sender_for_spawn.clone();
                                    let registry = task_registry_for_spawn.clone();
                                    let snap_reg = snap_registry_for_spawn.clone();
                                    let timer_alias = alias.clone();
                                    let timer_module = module_name.clone();
                                    let timer_handler = handler.clone();
                                    let join = tokio::spawn(async move {
                                        let mut interval = tokio::time::interval(std::time::Duration::from_millis(ms));
                                        interval.tick().await; // skip first immediate tick
                                        loop {
                                            interval.tick().await;
                                            let handler_name = timer_handler.clone();
                                            let prog = prog.clone();
                                            let sender = sender.clone();
                                            let registry = registry.clone();
                                            let snap_reg = snap_reg.clone();
                                            let alias = timer_alias.clone();
                                            let module_name = timer_module.clone();
                                            tokio::task::spawn_blocking(move || {
                                                let ctx = eval::EvalContext {
                                                    runtime: None, meta: None, event_broadcast: None,
                                                    program_snapshot: Some(std::sync::Arc::new(prog.clone())),
                                                    program_mut: Some(eval::make_shared_program_mut(&prog)),
                                                };
                                                ctx.install();
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
                                    if let Ok(mut reg) = source_registry.lock() {
                                        reg.insert(
                                            coroutine::source_key(&module_name, &alias),
                                            coroutine::ActiveSource {
                                                module: module_name.clone(),
                                                alias: alias.clone(),
                                                source_type: "timer".to_string(),
                                                handler: handler.clone(),
                                                interval_ms: Some(ms),
                                                abort: Some(join.abort_handle()),
                                            },
                                        );
                                    }
                                    let _ = reply.send(Ok(format!("timer source '{}' registered ({}ms)", alias, ms)));
                                } else {
                                    let _ = reply.send(Err(anyhow::anyhow!("timer source requires interval_ms")));
                                }
                            } else if source_type == "channel" {
                                let _ = reply.send(Ok(format!("channel source '{}' registered", alias)));
                            } else {
                                let _ = reply.send(Ok(format!("event source '{}' registered ({})", alias, source_type)));
                            }
                        }
                        coroutine::IoRequest::SourceRemove { module_name, alias, reply } => {
                            let removed = coroutine::remove_source(&source_registry, &module_name, &alias);
                            if removed {
                                let _ = reply.send(Ok(format!("source '{}.{}' removed", module_name, alias)));
                            } else {
                                let _ = reply.send(Ok(format!("source '{}.{}' not found (nothing removed)", module_name, alias)));
                            }
                        }
                        coroutine::IoRequest::SourceList { reply } => {
                            let _ = reply.send(Ok(coroutine::format_source_list(&source_registry)));
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
        Command::Os { port, session, url, model, api_key, daemonize, autonomous, log_file, training_log, opencode_git_dir, opencode_attach, max_iterations, access_level, permissions_file, host } => {
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
            // Drain any routes declared inside library modules via +route
            let pending_routes: Vec<_> = sess.program.pending_routes.drain(..).collect();
            for route in pending_routes {
                let msg = sess.add_route(route);
                eprintln!("[library] {msg}");
            }
            eprintln!("[library] http_routes after drain: {:?}", sess.runtime.http_routes.iter().map(|r| format!("{} {} -> {}", r.method, r.path, r.handler_fn)).collect::<Vec<_>>());
            sess.meta.library_state = Some(lib_state);
            sess.init_shared_vars();

            // Build restart context before the session is split into tier locks.
            // This consumes last_opencode_output so it's only shown once.
            let restart_context = if sess.meta.conversations.get("main").is_some_and(|c| c.messages.len() > 1) {
                Some(sess.restart_context())
            } else {
                None
            };

            let initial_runtime = sess.runtime.clone();
            let shared_runtime: crate::session::SharedRuntime =
                std::sync::Arc::new(std::sync::RwLock::new(initial_runtime));
            let shared_meta: crate::session::SharedMeta =
                std::sync::Arc::new(std::sync::Mutex::new(sess.meta.clone()));

            let memory_path = std::env::var_os("ADAPSIS_MEMORY_DB")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| {
                    dirs::config_dir()
                        .unwrap_or_else(|| std::path::PathBuf::from("."))
                        .join("adapsis")
                        .join("memory.lbug")
                });
            let memory_graph = std::sync::Arc::new(
                memory_graph::MemoryGraph::open(&memory_path)
                    .with_context(|| format!("open memory graph at {}", memory_path.display()))?,
            );
            let memory_embedder = std::sync::Arc::new(
                memory_graph::MemoryEmbedder::new().context("initialize memory embeddings")?,
            );
            eprintln!("Ladybug memory graph: {}", memory_path.display());

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
             let source_registry: coroutine::SourceRegistry =
                 std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
             let llm_url_for_spawn = url.clone();
            let llm_model_shared: std::sync::Arc<std::sync::RwLock<String>> = std::sync::Arc::new(std::sync::RwLock::new(model.clone()));
            let llm_model_for_spawn = llm_model_shared.clone();
            let llm_key_for_spawn = api_key.clone();
            let opencode_lock = std::sync::Arc::new(tokio::sync::Mutex::new(()));
            let opencode_lock_for_spawn = opencode_lock.clone();
            let opencode_git_dir_shared: std::sync::Arc<std::sync::RwLock<String>> = std::sync::Arc::new(std::sync::RwLock::new(".".to_string()));
            let opencode_git_dir_for_spawn = opencode_git_dir_shared.clone();
            let training_log_shared: std::sync::Arc<std::sync::RwLock<Option<std::sync::Arc<tokio::sync::Mutex<tokio::fs::File>>>>> = std::sync::Arc::new(std::sync::RwLock::new(None));
            let training_log_for_spawn = training_log_shared.clone();
            // Late-filled after the save channel / log file are created below
            // (the IO loop spawns before they exist). Same pattern as
            // training_log_shared. Lets llm_takeover persist conversations
            // and write to --log-file.
            let save_notify_shared: std::sync::Arc<std::sync::RwLock<Option<tokio::sync::mpsc::Sender<()>>>> = std::sync::Arc::new(std::sync::RwLock::new(None));
            let save_notify_for_spawn = save_notify_shared.clone();
            let ai_log_shared: std::sync::Arc<std::sync::RwLock<Option<std::sync::Arc<tokio::sync::Mutex<tokio::fs::File>>>>> = std::sync::Arc::new(std::sync::RwLock::new(None));
            let ai_log_for_spawn = ai_log_shared.clone();
            let memory_graph_for_spawn = memory_graph.clone();
            let memory_embedder_for_spawn = memory_embedder.clone();
            let access_level_parsed: permissions::AccessLevel = access_level.parse().expect("invalid --access-level");
            let perm_config = if let Some(ref path) = permissions_file {
                permissions::PermissionConfig::load(std::path::Path::new(path)).expect("failed to load permissions file")
            } else {
                permissions::PermissionConfig::default()
            };
            let access_level_for_spawn = access_level_parsed;
            let perm_config_for_spawn = std::sync::Arc::new(perm_config.clone());
            let perm_config_for_config = perm_config_for_spawn.clone();
            // Clone resources for startup execution (before IO loop moves them)
            let io_sender_for_startup = runtime.io_sender();
            let io_sender_for_autonomous = runtime.io_sender();
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
                                let runtime_for_env = runtime_for_blocking.clone();
                                let ctx = eval::EvalContext::new_minimal(
                                    runtime_for_blocking, meta_for_blocking,
                                    &program, eval::make_shared_program_mut(&program),
                                );
                                ctx.install();

                                let handle = coroutine::CoroutineHandle::new_with_task(sender, task_id, registry.clone(), snap_reg);
                                let mut env = eval::Env::new_with_shared_interner(&program.shared_interner);
                                env.populate_shared_from_program(&program);
                                // Set shared runtime so +shared vars resolve from runtime state
                                env.set_shared_runtime(runtime_for_env);
                                // Initialize shared runtime vars so +shared defaults are available
                                if let Some(rt) = eval::get_shared_runtime() {
                                    eval::init_missing_shared_runtime_vars(&program, &rt);
                                }
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
                                    // Replace semantics: abort any existing source on
                                    // the same module.alias key before spawning anew.
                                    coroutine::remove_source(&source_registry, &module_name, &alias);
                                    let program_for_timer = shared_program_for_spawn.clone();
                                    let sender_for_timer = io_sender_for_spawn.clone();
                                    let registry_for_timer = task_registry_for_spawn.clone();
                                    let snap_reg_for_timer = snap_registry_for_spawn2.clone();
                                    let runtime_for_timer = shared_runtime_for_spawn.clone();
                                    let meta_for_timer = shared_meta_for_spawn.clone();
                                    let timer_alias = alias.clone();
                                    let timer_module = module_name.clone();
                                    let timer_handler = handler.clone();
                                    let join = tokio::spawn(async move {
                                        let mut interval = tokio::time::interval(std::time::Duration::from_millis(ms));
                                        interval.tick().await; // skip first immediate tick
                                        loop {
                                            interval.tick().await;
                                            let handler_name = timer_handler.clone();
                                            let prog = program_for_timer.read().await.clone();
                                            let func = match prog.get_function(&handler_name) {
                                                Some(f) => f.clone(),
                                                None => {
                                                    eprintln!("[timer:{}] handler `{}` not found", timer_alias, handler_name);
                                                    continue;
                                                }
                                            };
                                            let sender = sender_for_timer.clone();
                                            let registry = registry_for_timer.clone();
                                            let snap_reg = snap_reg_for_timer.clone();
                                            let rt_for_tick = runtime_for_timer.clone();
                                            let meta_for_tick = meta_for_timer.clone();
                                            let alias_for_tick = timer_alias.clone();
                                            let module_for_tick = timer_module.clone();
                                            tokio::task::spawn_blocking(move || {
                                                let ctx = eval::EvalContext::new_minimal(
                                                    rt_for_tick, meta_for_tick,
                                                    &prog, eval::make_shared_program_mut(&prog),
                                                );
                                                ctx.install();
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
                                    if let Ok(mut reg) = source_registry.lock() {
                                        reg.insert(
                                            coroutine::source_key(&module_name, &alias),
                                            coroutine::ActiveSource {
                                                module: module_name.clone(),
                                                alias: alias.clone(),
                                                source_type: "timer".to_string(),
                                                handler: handler.clone(),
                                                interval_ms: Some(ms),
                                                abort: Some(join.abort_handle()),
                                            },
                                        );
                                    }
                                    let _ = reply.send(Ok(format!("timer source '{}' registered ({}ms)", alias, ms)));
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
                        coroutine::IoRequest::SourceRemove { module_name, alias, reply } => {
                            let removed = coroutine::remove_source(&source_registry, &module_name, &alias);
                            if removed {
                                let _ = reply.send(Ok(format!("source '{}.{}' removed", module_name, alias)));
                            } else {
                                let _ = reply.send(Ok(format!("source '{}.{}' not found (nothing removed)", module_name, alias)));
                            }
                        }
                        coroutine::IoRequest::SourceList { reply } => {
                            let _ = reply.send(Ok(coroutine::format_source_list(&source_registry)));
                        }
                        coroutine::IoRequest::LlmTakeover { context, message, attachment, source_metadata, reply_fn, reply_arg, permission_model, reply } => {
                            // Set or clear permission_model on the conversation.
                            // This is per-message: admin messages clear any restriction
                            // that a non-admin may have set on a shared group context.
                            {
                                let mut meta_guard = shared_meta_for_spawn.lock().unwrap();
                                let conv = meta_guard.conversations.get_or_create(&context);
                                conv.permission_model = permission_model.clone();
                            }

                            let meta = shared_meta_for_spawn.clone();
                            let program = shared_program_for_spawn.clone();
                            let runtime = shared_runtime_for_spawn.clone();
                            let llm_url = llm_url_for_spawn.clone();
                            let llm_model = llm_model_for_spawn.read().unwrap().clone();
                            let llm_key = llm_key_for_spawn.clone();
                            let io_sender = io_sender_for_spawn.clone();
                            let task_registry = task_registry_for_spawn.clone();
                            let snap_registry = snap_registry_for_spawn2.clone();
                            let oc_lock = opencode_lock_for_spawn.clone();
                            let oc_git_dir = opencode_git_dir_for_spawn.read().unwrap().clone();
                            let t_log = training_log_for_spawn.read().unwrap().clone();
                            let al = access_level_for_spawn;
                            let pc = perm_config_for_spawn.clone();
                            let save_notify = save_notify_for_spawn.read().unwrap().clone();
                            let ai_log = ai_log_for_spawn.read().unwrap().clone();
                            let memory_graph = memory_graph_for_spawn.clone();
                            let memory_embedder = memory_embedder_for_spawn.clone();

                            tokio::spawn(async move {
                                let result = crate::api::handle_llm_takeover(
                                    context, message, attachment, source_metadata, memory_graph, memory_embedder, reply_fn, reply_arg,
                                    meta, program, runtime,
                                    &llm_url, &llm_model, llm_key,
                                    io_sender, task_registry, snap_registry,
                                    oc_lock, oc_git_dir, t_log,
                                    al, pc,
                                    save_notify, ai_log,
                                ).await;
                                let _ = reply.send(result);
                            });
                        }
                        coroutine::IoRequest::MemoryCypher { principal_id, query, reply } => {
                            let graph = memory_graph_for_spawn.clone();
                            tokio::task::spawn_blocking(move || {
                                let result = graph.authorized_cypher(&principal_id, &query);
                                let _ = reply.send(result);
                            });
                        }
                        coroutine::IoRequest::SetLlmModel { name, reply } => {
                            // Verify the model loads by making a test request
                            let old_model = llm_model_for_spawn.read().unwrap().clone();
                            let test_url = llm_url_for_spawn.clone();
                            let test_name = name.clone();
                            let model_ref = llm_model_for_spawn.clone();
                            tokio::spawn(async move {
                                let client = reqwest::Client::builder()
                                    .timeout(std::time::Duration::from_secs(60))
                                    .build()
                                    .unwrap_or_default();
                                let test_req = serde_json::json!({
                                    "model": &test_name,
                                    "messages": [{"role": "user", "content": "."}],
                                    "max_tokens": 1
                                });
                                eprintln!("[llm_set_model] testing model '{test_name}'...");
                                match client.post(format!("{test_url}/v1/chat/completions"))
                                    .json(&test_req)
                                    .send()
                                    .await
                                {
                                    Ok(resp) if resp.status().is_success() => {
                                        *model_ref.write().unwrap() = test_name.clone();
                                        eprintln!("[llm_set_model] switched to '{test_name}'");
                                        let _ = reply.send(Ok(format!("model set to: {test_name}")));
                                    }
                                    Ok(resp) => {
                                        let status = resp.status();
                                        let body = resp.text().await.unwrap_or_default();
                                        eprintln!("[llm_set_model] model '{test_name}' failed: {status} {body}");
                                        let _ = reply.send(Err(anyhow::anyhow!(
                                            "model '{test_name}' failed to load: {status}. Staying on '{old_model}'"
                                        )));
                                    }
                                    Err(e) => {
                                        eprintln!("[llm_set_model] model '{test_name}' unreachable: {e}");
                                        let _ = reply.send(Err(anyhow::anyhow!(
                                            "model '{test_name}' unreachable: {e}. Staying on '{old_model}'"
                                        )));
                                    }
                                }
                            });
                        }
                        coroutine::IoRequest::GetLlmModel { reply } => {
                            let name = llm_model_for_spawn.read().unwrap().clone();
                            let _ = reply.send(Ok(name));
                        }
                        coroutine::IoRequest::ConversationNotify { context, message, attachment, reply } => {
                            let meta = shared_meta_for_spawn.clone();
                            let io_sender = io_sender_for_spawn.clone();
                            tokio::spawn(async move {
                                // Push message with attachment to conversation
                                let (cb_fn, cb_arg) = {
                                    let mut meta_guard = meta.lock().unwrap();
                                    let conv = meta_guard.conversations.get_or_create(&context);
                                    let mut msg = crate::session::ChatMessage {
                                        role: "system".to_string(),
                                        content: message.clone(),
                                        attachments: vec![],
                                    };
                                    if let Some(att) = attachment.clone() {
                                        msg.attachments.push(att);
                                    }
                                    conv.messages.push(msg);
                                    (conv.reply_fn.clone(), conv.reply_arg.clone())
                                };

                                // Trigger reply callback with text, and attachment if present
                                if let (Some(func_name), Some(arg)) = (cb_fn, cb_arg) {
                                    if let Some(att) = attachment {
                                        // Call <reply_fn>_with_attachment(arg, text, attachment)
                                        let att_fn = format!("{func_name}_with_attachment");
                                        eprintln!("[conversation_notify:{context}] delivering with attachment via {att_fn}({arg})");
                                        let args = vec![
                                            crate::eval::Value::string(arg),
                                            crate::eval::Value::string(message),
                                            crate::eval::Value::Attachment(att),
                                        ];
                                        let (tx, _rx) = tokio::sync::oneshot::channel();
                                        match io_sender.try_send(crate::coroutine::IoRequest::Spawn {
                                            function_name: att_fn.clone(),
                                            args,
                                            reply: tx,
                                        }) {
                                            Ok(()) => eprintln!("[conversation_notify:{context}] spawn dispatched {att_fn}"),
                                            Err(e) => eprintln!("[conversation_notify:{context}] spawn send failed: {e}"),
                                        }
                                    } else {
                                        // Text only
                                        eprintln!("[conversation_notify:{context}] delivering text via {func_name}({arg})");
                                        let args = vec![
                                            crate::eval::Value::string(arg),
                                            crate::eval::Value::string(message),
                                        ];
                                        let (tx, _rx) = tokio::sync::oneshot::channel();
                                        match io_sender.try_send(crate::coroutine::IoRequest::Spawn {
                                            function_name: func_name,
                                            args,
                                            reply: tx,
                                        }) {
                                            Ok(()) => eprintln!("[conversation_notify:{context}] spawn dispatched"),
                                            Err(e) => eprintln!("[conversation_notify:{context}] spawn send failed: {e}"),
                                        }
                                    }
                                }

                                let _ = reply.send(Ok("notified".to_string()));
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

            // Execute module startup blocks and auto-register module-level sources
            {
                let modules_with_startup: Vec<(String, ast::LifecycleBlock)> = {
                    let prog = startup_program.read().await;
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
                        let ctx = eval::EvalContext::new_minimal(
                            rt, meta,
                            &prog, eval::make_shared_program_mut(&prog),
                        );
                        ctx.install();
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
                    let prog = startup_program.read().await;
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
                        let _ = io_sender_for_startup.send(
                            coroutine::IoRequest::SourceAdd {
                                module_name: module_name.clone(),
                                source_type: src.source_type.clone(),
                                interval_ms,
                                alias: src.name.clone(),
                                handler,
                                reply: reply_tx,
                            }
                        ).await;
                        match reply_rx.await {
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

            // Resolve opencode git directory:
            // 0. If !opencode is disabled by the access level (anything below
            //    Full), skip all git/worktree setup entirely. Such deployments
            //    can never rebuild their own runtime, so cloning the source repo
            //    is both unnecessary and an avoidable dependency/attack surface
            //    (and would hard-fail on boxes without git installed).
            // 1. If explicitly set, validate it's a git repo with Cargo.toml
            // 2. Otherwise, auto-setup a bare repo + per-session worktree
            //    Each session gets its own isolated worktree so multiple
            //    sessions don't interfere with each other.
            let resolved_git_dir = if access_level_parsed != permissions::AccessLevel::Full {
                eprintln!("[opencode] access level {:?} disables !opencode; skipping git/worktree setup", access_level_parsed);
                ".".to_string()
            } else if let Some(ref dir) = opencode_git_dir {
                // Validate the explicit directory
                let p = std::path::Path::new(dir);
                if !p.join(".git").exists() {
                    eprintln!("ERROR: --opencode-git-dir '{}' is not a git repository (no .git directory)", dir);
                    std::process::exit(1);
                }
                if !p.join("Cargo.toml").exists() {
                    eprintln!("ERROR: --opencode-git-dir '{}' does not look like an adapsis repo (no Cargo.toml)", dir);
                    std::process::exit(1);
                }
                dir.clone()
            } else {
                // Derive session name for the worktree
                let session_stem = std::path::Path::new(&session)
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| session.clone());

                // Auto-setup: bare repo at ~/.local/share/adapsis/repo.git
                // Worktrees at ~/.local/share/adapsis/worktrees/<session>/
                let data_base = dirs::data_dir()
                    .unwrap_or_else(|| std::path::PathBuf::from(
                        format!("{}/.local/share", std::env::var("HOME").unwrap_or_else(|_| ".".to_string()))
                    ))
                    .join("adapsis");
                let bare_repo = data_base.join("repo.git");
                let worktrees_dir = data_base.join("worktrees");
                let worktree_dir = worktrees_dir.join(&session_stem);
                let worktree_dir_str = worktree_dir.to_string_lossy().to_string();

                // Step 1: Ensure bare repo exists
                if !bare_repo.exists() {
                    // Default source: the public GitHub repo.
                    // Can be overridden via ADAPSIS_OPENCODE_SOURCE env var
                    // (useful for forks or offline setups).
                    let source_url = std::env::var("ADAPSIS_OPENCODE_SOURCE")
                        .unwrap_or_else(|_| "https://github.com/Marenz/adapsis.git".to_string());

                    eprintln!("[opencode] Creating bare repo from {}", source_url);
                    let bare_str = bare_repo.to_string_lossy();
                    let output = std::process::Command::new("git")
                        .args(["clone", "--bare", &source_url, &bare_str])
                        .output();
                    match output {
                        Ok(o) if o.status.success() => {
                            eprintln!("[opencode] Bare repo at {}", bare_str);
                        }
                        Ok(o) => {
                            let stderr = String::from_utf8_lossy(&o.stderr);
                            eprintln!("ERROR: Failed to create bare repo at {}: {}", bare_str, stderr);
                            std::process::exit(1);
                        }
                        Err(e) => {
                            eprintln!("ERROR: Failed to run git clone: {}", e);
                            std::process::exit(1);
                        }
                    }
                }

                // Step 2: Ensure worktree exists for this session
                std::fs::create_dir_all(&worktrees_dir).unwrap_or_else(|e| {
                    eprintln!("ERROR: Failed to create worktrees dir: {}", e);
                    std::process::exit(1);
                });

                if !worktree_dir.exists() {
                    eprintln!("[opencode] Creating worktree for session '{}': {}", session_stem, worktree_dir_str);
                    let bare_str = bare_repo.to_string_lossy();
                    // Fetch latest before creating worktree
                    let _ = std::process::Command::new("git")
                        .args(["-C", &bare_str, "fetch", "origin"])
                        .output();
                    // Use session name as branch to avoid conflicts with other sessions
                    let branch_name = session_stem.to_lowercase().replace(' ', "-");
                    let output = std::process::Command::new("git")
                        .args(["-C", &bare_str, "worktree", "add", "-b", &branch_name, &worktree_dir_str, "master"])
                        .output();
                    match output {
                        Ok(o) if o.status.success() => {
                            eprintln!("[opencode] Worktree ready at {}", worktree_dir_str);
                        }
                        Ok(o) => {
                            let stderr = String::from_utf8_lossy(&o.stderr);
                            // If worktree already exists from a previous run, that's fine
                            if !stderr.contains("already a") && !stderr.contains("already exists") {
                                eprintln!("ERROR: Failed to create worktree: {}", stderr);
                                std::process::exit(1);
                            }
                        }
                        Err(e) => {
                            eprintln!("ERROR: Failed to run git worktree add: {}", e);
                            std::process::exit(1);
                        }
                    }
                }

                worktree_dir_str
            };

            // Note: binary may be installed at ~/.local/bin/adapsis while worktree is separate.
            // !opencode will copy the rebuilt binary to the installed location before restart.

            // Update the shared git dir so the IO loop can use it
            *opencode_git_dir_shared.write().unwrap() = resolved_git_dir.clone();

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
            *training_log_shared.write().unwrap() = train_log.clone();
            *ai_log_shared.write().unwrap() = ai_log.clone();

            // Build the three independent tiers from the loaded session.
            let tier1_program = std::sync::Arc::new(tokio::sync::RwLock::new(sess.program.clone()));

            // Save-on-change: bounded channel with capacity 1.
            // Any code path calls try_send(()) when state changes; the background
            // task below debounces and saves after 2 seconds of quiet.
            let (save_tx, save_rx) = tokio::sync::mpsc::channel::<()>(1);
            *save_notify_shared.write().unwrap() = Some(save_tx.clone());

            let permission_config = perm_config_for_config;
            let access_level_for_config = access_level_parsed;

            let config = api::AppConfig {
                program: tier1_program,
                meta: shared_meta.clone(),
                llm_url: url.clone(),
                llm_model: llm_model_shared.clone(),
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
                opencode_lock: opencode_lock,
                opencode_attach: opencode_attach.clone(),
                message_queue: std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new())),
                opencode_git_dir: resolved_git_dir,
                runtime: shared_runtime.clone(),
                sessions: std::sync::Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new())),
                save_notify: Some(save_tx),
                access_level: access_level_for_config,
                permission_config,
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

            // Normalize the requested bind addresses: trim, drop empties, and
            // de-duplicate while preserving order so `--host 0.0.0.0` and
            // `--host 127.0.0.1 --host 10.0.0.4` both behave sensibly.
            let mut hosts: Vec<String> = Vec::new();
            for h in &host {
                let h = h.trim();
                if !h.is_empty() && !hosts.iter().any(|existing| existing == h) {
                    hosts.push(h.to_string());
                }
            }
            if hosts.is_empty() {
                hosts.push("127.0.0.1".to_string());
            }

            // Bind one listener per requested host. Binding fails fast so a typo
            // or an already-used interface is reported before the server starts.
            let mut listeners = Vec::with_capacity(hosts.len());
            for h in &hosts {
                let listener = tokio::net::TcpListener::bind(format!("{h}:{port}"))
                    .await
                    .map_err(|e| anyhow::anyhow!("Cannot bind {h}:{port}: {e}. Try -p {}", port + 1))?;
                listeners.push(listener);
            }

            println!("AdapsisOS running, bound to {} interface(s):", hosts.len());
            for h in &hosts {
                println!("  API:     http://{h}:{port}/api/");
                println!("  Browser: http://{h}:{port}/");
            }
            println!();

            if daemonize {
                // We verified the port works. Now respawn without -d.
                // Use SO_REUSEADDR equivalent by dropping listeners first and
                // giving the OS a moment.
                drop(listeners);
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

            // Save-on-change background task.
            // Waits for a save notification, then debounces for 2 seconds
            // (coalescing rapid changes), snapshots the tiers, persists library
            // modules, and saves the session file.
            let save_path = session.clone();
            let autosave_program = save_program.clone();
            let autosave_meta = save_meta.clone();
            let autosave_runtime = save_runtime.clone();
            tokio::spawn(async move {
                let mut rx = save_rx;
                loop {
                    // Wait for first notification
                    if rx.recv().await.is_none() {
                        break; // channel closed, shut down
                    }
                    // Debounce: drain additional notifications for up to 2 seconds
                    loop {
                        match tokio::time::timeout(
                            std::time::Duration::from_secs(2),
                            rx.recv(),
                        ).await {
                            Ok(Some(())) => {} // more notifications — keep waiting
                            Ok(None) => break, // channel closed
                            Err(_) => break,   // 2-second quiet window elapsed
                        }
                    }
                    // Snapshot tiers and save
                    let sess = snapshot_from_tiers(&autosave_program, &autosave_meta, &autosave_runtime).await;
                    let conv_count = sess.meta.conversations.contexts.len();
                    eprintln!("[save] triggered: {} conversation(s), revision {}", conv_count, sess.meta.revision);
                    // Persist all library modules (picks up test changes, shared var updates, etc.)
                    let mut lib_errors = 0usize;
                    for module in &sess.program.modules {
                        if let Err(e) = crate::library::persist_module(module) {
                            eprintln!("[save] failed to persist module `{}`: {e}", module.name);
                            lib_errors += 1;
                        }
                    }
                    if lib_errors > 0 {
                        eprintln!("[save] {lib_errors} module(s) failed to persist");
                    }
                    if let Err(e) = sess.save(std::path::Path::new(&save_path)) {
                        eprintln!("[save] failed: {e}");
                    } else {
                        eprintln!("[save] session saved to {save_path}");
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
                        let conv = meta.conversations.get_or_create("main");
                        conv.messages.push(crate::session::ChatMessage {
                            role: "tool".to_string(),
                            content: event_message.clone(),
                            attachments: vec![],
                        });
                        conv.to_llm_messages()
                    };

                    match llm.generate(messages).await {
                        Ok(output) => {
                            let code = output.code.clone();
                            eprintln!("[self-trigger:response] {}...", output.text.chars().take(100).collect::<String>());

                            // Apply code if any
                            if !code.is_empty() && code.trim() != "DONE" {
                                let base_program = trigger_program.read().await.clone();
                                let mut program = base_program.clone();
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
                                let conv = meta.conversations.get_or_create("main");
                                conv.push_assistant(format!("[auto-response] {}", output.text.chars().take(200).collect::<String>()));
                                // Per-function CoW merge (issue #9).
                                trigger_program.write().await.merge_changed_from(&base_program, &program);
                                if let Ok(mut rt) = trigger_runtime.write() {
                                    rt.shared_vars.replace_from(runtime.shared_vars.snapshot());
                                    rt.http_routes = runtime.http_routes.clone();
                                    rt.failure_history = runtime.failure_history.clone();
                                    rt.agent_mailbox = runtime.agent_mailbox.clone();
                                    rt.pending_commands = runtime.pending_commands.clone();
                                    rt.library_errors = runtime.library_errors.clone();
                                    rt.library_load_errors = runtime.library_load_errors.clone();
                                }
                                *trigger_meta.lock().unwrap() = meta;
                            }
                        }
                        Err(e) => {
                            eprintln!("[self-trigger:error] {e}");
                        }
                    }
                }
            });

            // If a startup goal was provided (--autonomous), inject it into the "main" context.
            // This is a one-shot trigger, not a polling loop. The LLM processes it once
            // and stops. Further work requires explicit triggers (inject, webhook, timer).
            if let Some(goal) = autonomous {
                let inject_sender = io_sender_for_autonomous.clone();
                tokio::spawn(async move {
                    // Wait briefly for the server to be ready
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    let (tx, _rx) = tokio::sync::oneshot::channel();
                    let msg = if let Some(context) = restart_context {
                        format!("{context}\n\n## Goal\n{goal}")
                    } else {
                        goal
                    };
                    eprintln!("[startup] injecting goal into 'main' context: {}...", msg.chars().take(80).collect::<String>());
                    let _ = inject_sender.send(crate::coroutine::IoRequest::LlmTakeover {
                        context: "main".to_string(),
                        message: msg,
                        attachment: None,
                        source_metadata: None,
                        reply_fn: None,
                        reply_arg: None,
                        permission_model: None,
                        reply: tx,
                    }).await;
                });
            }

            eprintln!("[adapsis] starting HTTP server on port {port} across {} interface(s)", listeners.len());
            // Serve the same router on every bound interface concurrently. The
            // router is cloned per listener (axum Routers are cheap to clone);
            // if any listener errors, the whole server tears down.
            let serve_futures = listeners
                .into_iter()
                .zip(hosts.iter().cloned())
                .map(|(listener, h)| {
                    let app = app.clone();
                    async move {
                        let result = axum::serve(listener, app).await;
                        if let Err(e) = &result {
                            eprintln!("[adapsis] server error on {h}:{port}: {e}");
                        }
                        result
                    }
                });
            match futures::future::try_join_all(serve_futures).await {
                Ok(_) => eprintln!("[adapsis] server exited cleanly — THIS SHOULD NOT HAPPEN"),
                Err(e) => eprintln!("[adapsis] server error: {e}"),
            }
            eprintln!("[adapsis] process exiting");
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
