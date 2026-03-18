use anyhow::Result;
use tracing::info;

use crate::ast;
use crate::eval;
use crate::llm::{ChatMessage, LlmBackend, LlmClient};
use crate::parser;
use crate::prompt;
use crate::validator;

pub struct Orchestrator<B: LlmBackend = crate::llm::OpenAiBackend> {
    llm: LlmClient<B>,
    max_iterations: usize,
}

impl<B: LlmBackend> Orchestrator<B> {
    pub fn new(llm: LlmClient<B>, max_iterations: usize) -> Self {
        Self {
            llm,
            max_iterations,
        }
    }

    pub async fn run(&mut self, task: &str) -> Result<()> {
        let mut program = ast::Program::default();
        let mut messages = vec![
            ChatMessage::system(prompt::system_prompt()),
            ChatMessage::user(prompt::task_message(task)),
        ];

        println!("=== Forge Feedback Loop ===");
        println!("Task: {task}");
        println!("Max iterations: {}", self.max_iterations);
        println!();

        for iteration in 1..=self.max_iterations {
            println!("--- Iteration {iteration}/{} ---", self.max_iterations);

            // Get LLM response
            let output = self.llm.generate(messages.clone()).await?;
            println!(); // newline after streaming output

            // Check if the model signals completion
            let code = if output.code.is_empty() {
                // If no <code> block, try to extract code from the raw text
                // (the model might not always use tags perfectly)
                extract_forge_code(&output.text)
            } else {
                output.code.clone()
            };

            if code.trim() == "DONE" || code.trim().is_empty() {
                println!("\n=== Model signals completion ===");
                println!("{program}");
                return Ok(());
            }

            // Add assistant message to history
            messages.push(ChatMessage::assistant(&output.text));

            // Parse the code
            let operations = match parser::parse(&code) {
                Ok(ops) => ops,
                Err(e) => {
                    let error_msg = format!("Parse error: {e}");
                    println!("  {error_msg}");
                    let feedback = prompt::feedback_message(
                        &[(error_msg, false)],
                        &[],
                        &validator::program_summary(&program),
                    );
                    messages.push(ChatMessage::user(feedback));
                    continue;
                }
            };

            // Each response is a complete program — start fresh
            // (unless it contains only !replace or !test operations)
            let has_definitions = operations.iter().any(|op| {
                matches!(
                    op,
                    parser::Operation::Module(_)
                        | parser::Operation::Function(_)
                        | parser::Operation::Type(_)
                )
            });
            if has_definitions {
                program = ast::Program::default();
            }

            // Apply and validate each operation
            let mut results: Vec<(String, bool)> = vec![];
            let mut test_ops: Vec<parser::TestMutation> = vec![];

            for op in &operations {
                match op {
                    parser::Operation::Test(test) => {
                        test_ops.push(test.clone());
                    }
                    _ => match validator::apply_and_validate(&mut program, op) {
                        Ok(msg) => {
                            println!("  OK: {msg}");
                            results.push((msg, true));
                        }
                        Err(e) => {
                            let msg = format!("{e}");
                            println!("  ERROR: {msg}");
                            results.push((msg, false));
                        }
                    },
                }
            }

            // Run tests
            let mut test_results: Vec<(String, bool)> = vec![];
            for test in &test_ops {
                println!("  Testing {}:", test.function_name);
                for (i, case) in test.cases.iter().enumerate() {
                    match eval::eval_test_case(&program, &test.function_name, case) {
                        Ok(msg) => {
                            println!("    PASS [{i}]: {msg}");
                            test_results.push((msg, true));
                        }
                        Err(e) => {
                            let msg = format!("{e}");
                            println!("    FAIL [{i}]: {msg}");
                            test_results.push((msg, false));
                        }
                    }
                }
            }

            // Build feedback
            let all_ok = results.iter().all(|(_, s)| *s) && test_results.iter().all(|(_, s)| *s);

            let feedback = prompt::feedback_message(
                &results,
                &test_results,
                &validator::program_summary(&program),
            );
            messages.push(ChatMessage::user(feedback));

            if all_ok && !results.is_empty() {
                info!("iteration {iteration}: all passed");
                // Don't break yet — let the model decide if it's done
            }
        }

        println!("\n=== Max iterations reached ===");
        println!("{program}");
        Ok(())
    }
}

/// Try to extract Forge code from raw text when <code> tags are missing.
fn extract_forge_code(text: &str) -> String {
    // Look for lines starting with + or ! which are Forge operations
    let mut code_lines = vec![];
    let mut in_code = false;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('+') || trimmed.starts_with('!') || trimmed == "end" {
            in_code = true;
            code_lines.push(line);
        } else if in_code && (trimmed.is_empty() || trimmed.starts_with("//")) {
            code_lines.push(line);
        } else if in_code {
            // Non-forge line after forge content — might be end of code block
            in_code = false;
        }
    }

    code_lines.join("\n")
}
