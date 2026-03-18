mod ast;
mod eval;
mod llm;
mod orchestrator;
mod parser;
mod prompt;
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
    /// Run the interactive feedback loop with the LLM
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
        Command::Check { path } => {
            let source = std::fs::read_to_string(&path)?;
            let operations = parser::parse(&source)?;
            let mut program = ast::Program::default();
            for op in &operations {
                match validator::apply_and_validate(&mut program, op) {
                    Ok(msg) => println!("OK: {msg}"),
                    Err(e) => eprintln!("ERROR: {e}"),
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
