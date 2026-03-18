use anyhow::Result;
use tracing::info;

use crate::ast;
use crate::eval;
use crate::events::{self, EventBus, ForgeEvent};
use crate::llm::{ChatMessage, LlmBackend, LlmClient};
use crate::parser;
use crate::prompt;
use crate::typeck;
use crate::validator;

pub struct Orchestrator<B: LlmBackend = crate::llm::OpenAiBackend> {
    llm: LlmClient<B>,
    max_iterations: usize,
    event_bus: Option<EventBus>,
}

impl<B: LlmBackend> Orchestrator<B> {
    pub fn new(llm: LlmClient<B>, max_iterations: usize) -> Self {
        Self {
            llm,
            max_iterations,
            event_bus: None,
        }
    }

    pub fn with_event_bus(llm: LlmClient<B>, max_iterations: usize, bus: EventBus) -> Self {
        Self {
            llm,
            max_iterations,
            event_bus: Some(bus),
        }
    }

    fn emit(&self, event: ForgeEvent) {
        if let Some(bus) = &self.event_bus {
            bus.send(event);
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
            self.emit(ForgeEvent::IterationStart {
                iteration,
                max_iterations: self.max_iterations,
            });

            // Get LLM response
            let output = self.llm.generate(messages.clone()).await?;
            println!(); // newline after streaming output

            if !output.thinking.is_empty() {
                self.emit(ForgeEvent::Thinking {
                    text: output.thinking.clone(),
                });
            }

            // Check if the model signals completion
            let code = if output.code.is_empty() {
                extract_forge_code(&output.text)
            } else {
                output.code.clone()
            };

            if !code.is_empty() {
                self.emit(ForgeEvent::Code { text: code.clone() });
            }

            if code.trim() == "DONE" || code.trim().is_empty() {
                println!("\n=== Model signals completion ===");
                println!("{program}");
                self.emit(events::snapshot_program(&program));
                self.emit(ForgeEvent::Done);
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
                    self.emit(ForgeEvent::MutationError {
                        message: error_msg.clone(),
                    });
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
                    parser::Operation::Trace(trace) => {
                        println!("  Tracing {}:", trace.function_name);
                        match eval::trace_function(&program, &trace.function_name, &trace.input) {
                            Ok(steps) => {
                                for step in &steps {
                                    println!("    > {step}");
                                    self.emit(ForgeEvent::TraceStep {
                                        stmt_id: step.stmt_id.clone(),
                                        description: step.description.clone(),
                                        result: step.result.clone(),
                                        status: format!("{:?}", step.status),
                                    });
                                }
                                results.push((
                                    format!(
                                        "traced {} ({} steps)",
                                        trace.function_name,
                                        steps.len()
                                    ),
                                    true,
                                ));
                            }
                            Err(e) => {
                                let msg = format!("trace error: {e}");
                                println!("    {msg}");
                                self.emit(ForgeEvent::MutationError {
                                    message: msg.clone(),
                                });
                                results.push((msg, false));
                            }
                        }
                    }
                    parser::Operation::Query(query) => {
                        let table = typeck::build_symbol_table(&program);
                        let response = typeck::handle_query(&program, &table, query);
                        println!("  Query `{query}`:\n{response}");
                        self.emit(ForgeEvent::QueryResult {
                            query: query.clone(),
                            response: response.clone(),
                        });
                        results.push((format!("query: {query}"), true));
                    }
                    _ => match validator::apply_and_validate(&mut program, op) {
                        Ok(msg) => {
                            println!("  OK: {msg}");
                            self.emit(ForgeEvent::MutationOk {
                                message: msg.clone(),
                            });
                            results.push((msg, true));
                        }
                        Err(e) => {
                            let msg = format!("{e}");
                            println!("  ERROR: {msg}");
                            self.emit(ForgeEvent::MutationError {
                                message: msg.clone(),
                            });
                            results.push((msg, false));
                        }
                    },
                }
            }

            // Run type checking
            {
                let table = typeck::build_symbol_table(&program);
                for func in &program.functions {
                    for error in typeck::check_function(&table, func) {
                        println!("  TYPE WARNING: {error}");
                        self.emit(ForgeEvent::TypeWarning {
                            message: error.clone(),
                        });
                        results.push((format!("type warning: {error}"), true));
                    }
                }
                for module in &program.modules {
                    for func in &module.functions {
                        for error in typeck::check_function(&table, func) {
                            println!("  TYPE WARNING: {error}");
                            self.emit(ForgeEvent::TypeWarning {
                                message: error.clone(),
                            });
                            results.push((
                                format!(
                                    "type warning in {}.{}: {}",
                                    module.name, func.name, error
                                ),
                                true,
                            ));
                        }
                    }
                }
            }

            // Send program snapshot
            self.emit(events::snapshot_program(&program));

            // Run tests
            let mut test_results: Vec<(String, bool)> = vec![];
            for test in &test_ops {
                println!("  Testing {}:", test.function_name);
                for (i, case) in test.cases.iter().enumerate() {
                    match eval::eval_test_case(&program, &test.function_name, case) {
                        Ok(msg) => {
                            println!("    PASS [{i}]: {msg}");
                            self.emit(ForgeEvent::TestPass {
                                function: test.function_name.clone(),
                                index: i,
                                message: msg.clone(),
                            });
                            test_results.push((msg, true));
                        }
                        Err(e) => {
                            let msg = format!("{e}");
                            println!("    FAIL [{i}]: {msg}");
                            self.emit(ForgeEvent::TestFail {
                                function: test.function_name.clone(),
                                index: i,
                                message: msg.clone(),
                            });
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

            self.emit(ForgeEvent::ProgramState {
                summary: validator::program_summary(&program),
            });

            if all_ok && !results.is_empty() {
                info!("iteration {iteration}: all passed");
            }
        }

        println!("\n=== Max iterations reached ===");
        println!("{program}");
        self.emit(ForgeEvent::Done);
        Ok(())
    }
}

/// Try to extract Forge code from raw text when <code> tags are missing.
fn extract_forge_code(text: &str) -> String {
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
            in_code = false;
        }
    }

    code_lines.join("\n")
}
