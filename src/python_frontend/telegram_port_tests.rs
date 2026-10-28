//! Separate partial corrected-origin port gate. Frozen corpus status is untouched.
use crate::{ast, eval, parser, validator, vm};

const ORIGIN: &str =
    include_str!("../../tests/fixtures/telegram-python-port/origin/TelegramBot.ax");
const PYTHON: &str =
    include_str!("../../tests/fixtures/telegram-python-port/python/TelegramBot.py");
const TESTS: &str =
    include_str!("../../tests/fixtures/telegram-python-port/python/test_telegram_bot.py");
const FROZEN: &str = include_str!("../../tests/fixtures/python-parity/library/TelegramBot.ax");
mod io_contracts;
mod wrapper_contract;

fn program(operations: &[parser::Operation]) -> ast::Program {
    let mut program = ast::Program::default();
    for operation in operations {
        validator::apply_and_validate(&mut program, operation).unwrap();
    }
    program.rebuild_function_index();
    program
}

#[test]
fn telegram_port_pure_origin_and_python_exact_roundtrip() {
    let origin = program(&parser::parse(ORIGIN).unwrap());
    let lowered = super::lower_module(PYTHON, "TelegramBot").unwrap();
    let python = program(&lowered.operations);
    assert!(super::ast_equality::program(&origin, &python));
    let printed = super::print_module(&python, "TelegramBot").unwrap();
    let second = program(
        &super::lower_module(&printed, "TelegramBot")
            .unwrap()
            .operations,
    );
    assert!(super::ast_equality::program(&python, &second));
    assert_eq!(
        super::print_module(&second, "TelegramBot").unwrap(),
        printed
    );

    // Validate only the explicitly selected pure closure against the frozen
    // function ASTs. The original whole module is still invalid (#91).
    let mut operations = parser::parse(FROZEN).unwrap();
    let parser::Operation::Module(module) = &mut operations[0] else {
        panic!()
    };
    module.body.retain(|op| matches!(op, parser::Operation::Function(f) if origin.get_function(&format!("TelegramBot.{}", f.name)).is_some()));
    module.doc = None;
    let selected = program(&operations);
    assert_eq!(selected.modules[0].functions.len(), 10);
    for function in &selected.modules[0].functions {
        let target = format!("TelegramBot.{}", function.name);
        assert!(
            super::ast_equality::function(function, origin.get_function(&target).unwrap()),
            "{target}"
        );
    }
}

fn run_paired(
    program: &ast::Program,
    tests: &[parser::Operation],
    check_vm: bool,
) -> (usize, Vec<String>) {
    let mut count = 0;
    let mut failures = Vec::new();
    for operation in tests {
        let parser::Operation::Test(test) = operation else {
            panic!("only paired test declarations")
        };
        let function = program.get_function(&test.function_name).unwrap();
        assert!(
            function.effects.is_empty(),
            "this runner admits pure functions only"
        );
        for case in &test.cases {
            eval::eval_test_case(program, &test.function_name, case).unwrap();
            count += 1;
            if !check_vm {
                continue;
            }
            let args = if let parser::Expr::StructLiteral(fields) = &case.input {
                function
                    .params
                    .iter()
                    .enumerate()
                    .map(|(i, param)| {
                        let field = fields
                            .iter()
                            .find(|f| f.name == param.name || f.name == format!("_{i}"))
                            .unwrap();
                        eval::eval_parser_expr_with_program(&field.value, program).unwrap()
                    })
                    .collect()
            } else {
                assert_eq!(function.params.len(), 1);
                vec![eval::eval_parser_expr_with_program(&case.input, program).unwrap()]
            };
            let bytecode = vm::compile_function(function, program).unwrap();
            let actual = vm::execute_with_io(&bytecode, args, program, &|_, _| {
                anyhow::bail!("pure port attempted IO")
            });
            let expected = eval::eval_parser_expr_with_program(&case.expected, program).unwrap();
            println!(
                "PORT_CASE {count} {} tree=pass vm={actual:?} expected={expected:?}",
                test.function_name
            );
            match actual {
                Ok(actual) if actual.to_string() == expected.to_string() => {}
                other => failures.push(format!(
                    "case {count} {}: {other:?}; expected {expected:?}",
                    test.function_name
                )),
            }
        }
    }
    (count, failures)
}

#[test]
fn telegram_port_paired_pure_cases_run_both_origins_and_tiers() {
    let origin = program(&parser::parse(ORIGIN).unwrap());
    let lowered = super::lower_module(PYTHON, "TelegramBot").unwrap();
    let python = program(&lowered.operations);
    let tests = super::lower_tests(TESTS, "TelegramBot").unwrap();
    super::validate_runtime_arguments(&tests.operations, &python).unwrap();
    let (origin_count, origin_failures) = run_paired(&origin, &tests.operations, true);
    let (python_count, python_failures) = run_paired(&python, &tests.operations, true);
    assert_eq!(origin_count, 28);
    assert_eq!(python_count, 28);
    // This gate stays RED while direct VM behavior differs. No allowlist of
    // known failures: later fixes must make every paired assertion pass.
    assert!(
        origin_failures.is_empty() && python_failures.is_empty(),
        "origin: {origin_failures:#?}\npython: {python_failures:#?}"
    );
}

#[test]
fn telegram_port_paired_pure_tree_gate() {
    let origin = program(&parser::parse(ORIGIN).unwrap());
    let python = program(
        &super::lower_module(PYTHON, "TelegramBot")
            .unwrap()
            .operations,
    );
    let tests = super::lower_tests(TESTS, "TelegramBot").unwrap();
    assert_eq!(run_paired(&origin, &tests.operations, false).0, 28);
    assert_eq!(run_paired(&python, &tests.operations, false).0, 28);
}

#[test]
fn telegram_port_tgformat_pure_subset() {
    let origin = program(
        &parser::parse(include_str!(
            "../../tests/fixtures/telegram-python-port/origin/TgFormat.ax"
        ))
        .unwrap(),
    );
    let python = program(
        &super::lower_module(
            include_str!("../../tests/fixtures/telegram-python-port/python/TgFormat.py"),
            "TgFormat",
        )
        .unwrap()
        .operations,
    );
    assert!(super::ast_equality::program(&origin, &python));
    let printed = super::print_module(&python, "TgFormat").unwrap();
    let second = program(
        &super::lower_module(&printed, "TgFormat")
            .unwrap()
            .operations,
    );
    assert!(super::ast_equality::program(&python, &second));
    assert_eq!(super::print_module(&second, "TgFormat").unwrap(), printed);
    let frozen = program(&parser::parse(include_str!("../../tests/fixtures/python-parity/library/TgFormat.ax")).unwrap().into_iter().filter_map(|op| {
        if let parser::Operation::Module(mut module) = op {
            module.body.retain(|op| matches!(op, parser::Operation::Function(f) if ["escape_html", "apply_inline", "tag"].contains(&f.name.as_str())));
            Some(parser::Operation::Module(module))
        } else { None }
    }).collect::<Vec<_>>());
    for function in &origin.modules[0].functions {
        assert!(super::ast_equality::function(
            function,
            frozen
                .get_function(&format!("TgFormat.{}", function.name))
                .unwrap()
        ));
    }
    let tests = super::lower_tests(
        include_str!("../../tests/fixtures/telegram-python-port/python/test_tg_format.py"),
        "TgFormat",
    )
    .unwrap();
    assert_eq!(run_paired(&origin, &tests.operations, true), (7, vec![]));
    assert_eq!(run_paired(&python, &tests.operations, true), (7, vec![]));
}

#[test]
fn telegram_port_provenance_inventory_is_explicit_and_partial() {
    use std::collections::BTreeSet;
    let manifest: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/telegram-python-port/manifest.json"
    ))
    .unwrap();
    assert_eq!(manifest["full_port_complete"], false);
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    assert_eq!(
        git_blob(&root.join("tests/fixtures/python-parity/manifest.json")),
        manifest["frozen_manifest_git_blob"].as_str().unwrap()
    );
    for module in manifest["modules"].as_array().unwrap() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(module["source"].as_str().unwrap());
        assert_eq!(git_blob(&path), module["source_git_blob"].as_str().unwrap());
        let operations = parser::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
        let parser::Operation::Module(source) = &operations[0] else {
            panic!()
        };
        let actual = source
            .body
            .iter()
            .filter_map(|op| match op {
                parser::Operation::Function(f) => Some(f.name.as_str()),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let done = module["implemented"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap())
            .collect::<BTreeSet<_>>();
        let pending = module["pending"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap())
            .collect::<BTreeSet<_>>();
        assert!(done.is_disjoint(&pending));
        assert!(!pending.is_empty());
        assert_eq!(actual, done.union(&pending).copied().collect());
        let origin_path = root
            .join("tests/fixtures/telegram-python-port")
            .join(module["origin"].as_str().unwrap());
        let origin =
            program(&parser::parse(&std::fs::read_to_string(origin_path).unwrap()).unwrap());
        assert_eq!(
            origin.modules[0]
                .functions
                .iter()
                .map(|f| f.name.as_str())
                .collect::<BTreeSet<_>>(),
            done
        );
        assert_eq!(
            module["ledger"]
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            done
        );
    }
}

fn git_blob(path: &std::path::Path) -> String {
    let output = std::process::Command::new("git")
        .arg("hash-object")
        .arg(path)
        .output()
        .unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout).unwrap().trim().into()
}
