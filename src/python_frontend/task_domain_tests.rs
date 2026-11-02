//! #120: real Python frontend and runtime gates; no Rust task-domain implementation.
use crate::{ast, eval, parser, validator};

const DOMAIN: &str = include_str!("../../examples/tasks/domain.py");
const SCENARIOS: &str = include_str!("../../examples/tasks/scenarios.py");
const ASSERTIONS: &str = include_str!("../../examples/tasks/test_scenarios.py");
const CASES: usize = 47;

fn load(source: &str) -> ast::Program {
    let lowered = super::lower_module(source, "TaskDomain").unwrap();
    let mut program = ast::Program::default();
    for operation in &lowered.operations {
        validator::apply_and_validate(&mut program, operation).unwrap();
    }
    program.rebuild_function_index();
    super::validate_runtime_arguments(&lowered.operations, &program).unwrap();
    program
}

fn roundtrip(program: &ast::Program) -> ast::Program {
    let printed = super::print_module(program, "TaskDomain").unwrap();
    let rebuilt = load(&printed);
    assert!(
        super::ast_equality::program(program, &rebuilt),
        "{}",
        super::ast_equality::module_difference(&program.modules[0], &rebuilt.modules[0])
    );
    assert_eq!(
        super::print_module(&rebuilt, "TaskDomain").unwrap(),
        printed
    );
    rebuilt
}

#[test]
fn task_domain_exact_module_roundtrip() {
    let program = load(DOMAIN);
    assert_eq!(program.modules.len(), 1);
    assert_eq!(program.modules[0].types.len(), 15);
    assert!(program.modules[0].shared_vars.is_empty());
    assert!(program.pending_routes.is_empty());
    for function in &program.modules[0].functions {
        assert!(function.effects.is_empty(), "pure module only");
    }
    roundtrip(&program);
}

#[test]
fn task_domain_python_scenarios_before_and_after_exact_roundtrip() {
    // Keep test constructors separate from domain source; no records/calls added
    // to the frontend's literal-only test-data policy.
    let original = load(&format!("{DOMAIN}\n{SCENARIOS}"));
    let rebuilt = roundtrip(&original);
    let tests = super::lower_tests(ASSERTIONS, "TaskDomain").unwrap();
    for program in [&original, &rebuilt] {
        super::validate_runtime_arguments(&tests.operations, program).unwrap();
        let mut count = 0;
        for operation in &tests.operations {
            let parser::Operation::Test(test) = operation else {
                panic!("only paired assertion declarations expected")
            };
            assert!(
                program
                    .get_function(&test.function_name)
                    .unwrap()
                    .effects
                    .is_empty()
            );
            for case in &test.cases {
                eval::eval_test_case(program, &test.function_name, case)
                    .unwrap_or_else(|error| panic!("{} case {count}: {error}", test.function_name));
                count += 1;
            }
        }
        assert_eq!(count, CASES);
    }
}
