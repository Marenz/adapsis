//! Shared implementation of `adapsis test`; Python enters with typed operations.
use crate::{api, ast, coroutine, eval, parser, session, typeck, validator};

#[derive(Debug)]
pub(crate) struct Summary {
    pub passed: usize,
    pub failed: usize,
}

pub(crate) fn run(operations: &[parser::Operation]) -> Summary {
    let mut run = Run {
        program: ast::Program::default(),
        tests: vec![],
        mocks: vec![],
        stubs: vec![],
        tasks: Default::default(),
        snapshots: Default::default(),
    };
    for operation in operations {
        run.apply(operation);
    }
    let mut meta = session::SessionMeta::new();
    meta.io_mocks = run.mocks.clone();
    meta.function_stubs = run.stubs;
    eval::set_shared_meta(Some(std::sync::Arc::new(std::sync::Mutex::new(meta))));
    let mut summary = Summary {
        passed: 0,
        failed: 0,
    };
    for test in &run.tests {
        println!("\n--- Testing {} ---", test.function_name);
        for (index, case) in test.cases.iter().enumerate() {
            match eval::eval_test_case_with_mocks(
                &run.program,
                &test.function_name,
                case,
                &run.mocks,
                &[],
            ) {
                Ok(message) => {
                    summary.passed += 1;
                    println!("  PASS [{index}]: {message}");
                }
                Err(error) => {
                    summary.failed += 1;
                    eprintln!("  FAIL [{index}]: {error}");
                }
            }
        }
    }
    println!(
        "\nTEST SUMMARY: {} passed, {} failed",
        summary.passed, summary.failed
    );
    summary
}

struct Run {
    program: ast::Program,
    tests: Vec<parser::TestMutation>,
    mocks: Vec<session::IoMock>,
    stubs: Vec<session::FunctionStub>,
    tasks: coroutine::TaskRegistry,
    snapshots: coroutine::TaskSnapshotRegistry,
}

impl Run {
    fn apply(&mut self, operation: &parser::Operation) {
        match operation {
            parser::Operation::Test(test) => self.tests.push(test.clone()),
            parser::Operation::Module(module) => {
                for op in &module.body {
                    if let parser::Operation::Test(test) = op {
                        self.tests.push(test.clone());
                    }
                }
                self.declaration(operation);
            }
            parser::Operation::Mock {
                operation,
                patterns,
                response,
            } => {
                let display = patterns
                    .iter()
                    .map(|p| format!("\"{p}\""))
                    .collect::<Vec<_>>()
                    .join(" ");
                self.mocks.push(session::IoMock {
                    operation: operation.clone(),
                    patterns: patterns.clone(),
                    response: response.clone(),
                });
                println!("OK: mock {operation} {display}");
            }
            parser::Operation::Unmock => {
                let count = self.mocks.len();
                self.mocks.clear();
                println!("OK: cleared {count} mocks");
            }
            parser::Operation::Stub {
                function_name,
                patterns,
                response_expr,
            } => {
                let display = patterns
                    .iter()
                    .map(|p| format!("\"{p}\""))
                    .collect::<Vec<_>>()
                    .join(" ");
                self.stubs.push(session::FunctionStub {
                    function_name: function_name.clone(),
                    patterns: patterns.clone(),
                    response_expr: response_expr.clone(),
                });
                println!("OK: stub {function_name} {display} -> {response_expr}");
            }
            parser::Operation::Unstub => {
                let count = self.stubs.len();
                self.stubs.clear();
                println!("OK: cleared {count} stubs");
            }
            parser::Operation::Trace(trace) => {
                println!("\n--- Tracing {} ---", trace.function_name);
                match eval::trace_function(&self.program, &trace.function_name, &trace.input) {
                    Ok(steps) => {
                        for step in steps {
                            println!("  > {step}");
                        }
                    }
                    Err(error) => eprintln!("  TRACE ERROR: {error}"),
                }
            }
            parser::Operation::Eval(value) => self.evaluate(value),
            parser::Operation::Query(query) => {
                let response = if query.trim() == "?tasks" {
                    api::format_tasks(&Some(self.tasks.clone()))
                } else if let Some(id) = api::parse_inspect_task_query(query.trim()) {
                    api::format_inspect_task(
                        &Some(self.tasks.clone()),
                        &Some(self.snapshots.clone()),
                        id,
                    )
                } else {
                    typeck::handle_query(
                        &self.program,
                        &typeck::build_symbol_table(&self.program),
                        query,
                        &[],
                    )
                };
                println!("\n--- Query: {query} ---\n{response}");
            }
            _ => self.declaration(operation),
        }
    }

    fn declaration(&mut self, operation: &parser::Operation) {
        match validator::apply_and_validate(&mut self.program, operation) {
            Ok(message) => println!("OK: {message}"),
            Err(error) => eprintln!("ERROR: {error}"),
        }
    }

    fn evaluate(&self, value: &parser::EvalMutation) {
        if let Some(expression) = &value.inline_expr {
            match eval::eval_inline_expr(&self.program, expression) {
                Ok(value) => println!("  = {value}"),
                Err(error) => eprintln!("  EVAL ERROR: {error}"),
            }
        } else {
            match eval::eval_compiled_or_interpreted(
                &self.program,
                &value.function_name,
                &value.input,
            ) {
                Ok((result, compiled)) => {
                    let tag = if compiled { " [compiled]" } else { "" };
                    println!("  eval {}(...) = {result}{tag}", value.function_name);
                }
                Err(error) => eprintln!("  EVAL ERROR: {error}"),
            }
        }
    }
}
