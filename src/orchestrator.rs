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

    /// Architect mode: design first, then implement per-function.
    pub async fn run_architect(&mut self, task: &str) -> Result<()> {
        let mut program = ast::Program::default();

        println!("=== Forge Architect Mode ===");
        println!("Task: {task}");
        println!();

        // Phase 1: Design
        println!("--- Phase 1: Design ---");
        self.emit(ForgeEvent::IterationStart {
            iteration: 1,
            max_iterations: self.max_iterations,
        });

        let mut messages = vec![
            ChatMessage::system(prompt::architect_system_prompt()),
            ChatMessage::user(prompt::architect_design_message(task)),
        ];

        let mut design_ok = false;
        for attempt in 1..=3 {
            println!("  Design attempt {attempt}/3...");
            let output = self.llm.generate(messages.clone()).await?;
            println!();

            if !output.thinking.is_empty() {
                self.emit(ForgeEvent::Thinking {
                    text: output.thinking.clone(),
                });
            }

            let code = if output.code.is_empty() {
                extract_forge_code(&output.text)
            } else {
                output.code.clone()
            };

            if !code.is_empty() {
                self.emit(ForgeEvent::Code { text: code.clone() });
            }

            messages.push(ChatMessage::assistant(&output.text));

            let operations = match parser::parse(&code) {
                Ok(ops) => ops,
                Err(e) => {
                    let error_msg = format!("Parse error: {e}");
                    println!("  {error_msg}");
                    self.emit(ForgeEvent::MutationError {
                        message: error_msg.clone(),
                    });
                    let feedback = prompt::architect_design_feedback(
                        &[(error_msg, false)],
                        &validator::program_summary(&program),
                        &[],
                    );
                    messages.push(ChatMessage::user(feedback));
                    continue;
                }
            };

            program = ast::Program::default();
            let mut results: Vec<(String, bool)> = vec![];

            for op in &operations {
                match op {
                    parser::Operation::Test(_)
                    | parser::Operation::Trace(_)
                    | parser::Operation::Query(_) => {}
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

            self.emit(events::snapshot_program(&program));

            let all_ok = results.iter().all(|(_, s)| *s);
            let stub_names: Vec<String> = program
                .functions
                .iter()
                .map(|f| f.name.clone())
                .chain(
                    program
                        .modules
                        .iter()
                        .flat_map(|m| m.functions.iter().map(|f| format!("{}.{}", m.name, f.name))),
                )
                .collect();

            let feedback = prompt::architect_design_feedback(
                &results,
                &validator::program_summary(&program),
                &stub_names,
            );
            messages.push(ChatMessage::user(feedback));

            if all_ok && !stub_names.is_empty() {
                println!(
                    "\n  Design validated! {} functions to implement: {}",
                    stub_names.len(),
                    stub_names.join(", ")
                );
                design_ok = true;
                break;
            }
        }

        if !design_ok {
            println!("\n=== Design phase failed after 3 attempts ===");
            self.emit(ForgeEvent::Done);
            return Ok(());
        }

        // Phase 2: Implement each function
        let functions_to_implement: Vec<String> = program
            .functions
            .iter()
            .map(|f| f.name.clone())
            .collect();

        // Also collect module functions
        let module_functions: Vec<String> = program
            .modules
            .iter()
            .flat_map(|m| m.functions.iter().map(|f| f.name.clone()))
            .collect();

        let all_functions: Vec<String> = functions_to_implement
            .iter()
            .chain(module_functions.iter())
            .cloned()
            .collect();

        for (fn_idx, fn_name) in all_functions.iter().enumerate() {
            println!(
                "\n--- Phase 2: Implement `{fn_name}` ({}/{}) ---",
                fn_idx + 1,
                all_functions.len()
            );
            self.emit(ForgeEvent::IterationStart {
                iteration: fn_idx + 2,
                max_iterations: all_functions.len() + 1,
            });

            let implement_msg = prompt::architect_implement_message(
                fn_name,
                &validator::program_summary(&program),
            );
            messages.push(ChatMessage::user(implement_msg));

            let mut fn_ok = false;
            for attempt in 1..=self.max_iterations {
                println!("  Attempt {attempt}...");
                let output = self.llm.generate(messages.clone()).await?;
                println!();

                if !output.thinking.is_empty() {
                    self.emit(ForgeEvent::Thinking {
                        text: output.thinking.clone(),
                    });
                }

                let code = if output.code.is_empty() {
                    extract_forge_code(&output.text)
                } else {
                    output.code.clone()
                };

                if !code.is_empty() {
                    self.emit(ForgeEvent::Code { text: code.clone() });
                }

                if code.trim() == "DONE" {
                    break;
                }

                messages.push(ChatMessage::assistant(&output.text));

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

                // Replace the function in the program (don't reset everything)
                // Remove existing function with same name, then add new one
                let mut results: Vec<(String, bool)> = vec![];
                let mut test_ops: Vec<parser::TestMutation> = vec![];

                for op in &operations {
                    match op {
                        parser::Operation::Function(fd) if fd.name == *fn_name => {
                            // Remove old stub
                            program.functions.retain(|f| f.name != *fn_name);
                            // Also remove from modules
                            for m in &mut program.modules {
                                m.functions.retain(|f| f.name != *fn_name);
                            }
                            match validator::apply_and_validate(&mut program, op) {
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
                            }
                        }
                        parser::Operation::Test(test) => {
                            test_ops.push(test.clone());
                        }
                        parser::Operation::Type(_) => {
                            // Allow adding new types during implementation
                            match validator::apply_and_validate(&mut program, op) {
                                Ok(msg) => {
                                    results.push((msg, true));
                                }
                                Err(_) => {} // Duplicate type is fine — already defined in design
                            }
                        }
                        _ => {
                            // Skip other operations (other functions, etc.)
                        }
                    }
                }

                // Type check
                {
                    let table = typeck::build_symbol_table(&program);
                    for func in &program.functions {
                        if func.name == *fn_name {
                            for error in typeck::check_function(&table, func) {
                                println!("  TYPE WARNING: {error}");
                                self.emit(ForgeEvent::TypeWarning {
                                    message: error.clone(),
                                });
                            }
                        }
                    }
                }

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

                let all_ok =
                    results.iter().all(|(_, s)| *s) && test_results.iter().all(|(_, s)| *s);

                let feedback = prompt::feedback_message(
                    &results,
                    &test_results,
                    &validator::program_summary(&program),
                );
                messages.push(ChatMessage::user(feedback));

                if all_ok && !results.is_empty() {
                    println!("  `{fn_name}` implemented successfully!");
                    fn_ok = true;
                    break;
                }
            }

            if !fn_ok {
                println!("  WARNING: `{fn_name}` implementation incomplete after max iterations");
            }
        }

        println!("\n=== Architect mode complete ===");
        println!("{program}");
        self.emit(events::snapshot_program(&program));
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
