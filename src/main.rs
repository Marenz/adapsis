mod ast;
mod eval;
mod events;
mod llm;
mod orchestrator;
mod parser;
mod prompt;
mod server;
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
            max_iterations,
        } => {
            let llm_client = llm::LlmClient::new(&url);
            let mut orch = orchestrator::Orchestrator::new(llm_client, max_iterations);
            orch.run(&task).await?;
        }
        Command::Architect {
            task,
            url,
            max_iterations,
            port,
        } => {
            let llm_client = llm::LlmClient::new(&url);
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
    }

    Ok(())
}
