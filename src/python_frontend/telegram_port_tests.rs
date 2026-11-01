//! Separate partial corrected-origin port gate. Frozen corpus status is untouched.
use crate::{ast, eval, parser, validator, vm};

const ORIGIN: &str =
    include_str!("../../tests/fixtures/telegram-python-port/origin/TelegramBot.ax");
const PYTHON: &str =
    include_str!("../../tests/fixtures/telegram-python-port/python/TelegramBot.py");
const TESTS: &str =
    include_str!("../../tests/fixtures/telegram-python-port/python/test_telegram_bot.py");
const FROZEN: &str = include_str!("../../tests/fixtures/python-parity/library/TelegramBot.ax");
const TELEGRAM_CASES: usize = 84;
const FORMAT_CASES: usize = 38;
mod io_contracts;
mod intake;
mod outbound;
mod polling;
mod media;
mod qualification;
mod routing;
mod send_chain;
mod wrapper_contract;
mod voice;

fn program(operations: &[parser::Operation]) -> ast::Program {
    let mut program = ast::Program::default();
    for operation in operations {
        validator::apply_and_validate(&mut program, operation).unwrap();
    }
    program.rebuild_function_index();
    program
}

fn format_operations() -> Vec<parser::Operation> {
    parser::parse(include_str!(
        "../../tests/fixtures/telegram-python-port/origin/TgFormat.ax"
    ))
    .unwrap()
}

fn lower_telegram(source: &str) -> super::LoweredSource {
    super::lower_module_with_program(source, "TelegramBot", &program(&format_operations())).unwrap()
}

fn attach_format(program: &mut ast::Program, python: bool) {
    let operations = if python {
        super::lower_module(
            include_str!("../../tests/fixtures/telegram-python-port/python/TgFormat.py"),
            "TgFormat",
        )
        .unwrap()
        .operations
    } else {
        format_operations()
    };
    for operation in operations {
        validator::apply_and_validate(program, &operation).unwrap();
    }
    program.rebuild_function_index();
}

#[test]
fn telegram_port_pure_origin_and_python_exact_roundtrip() {
    let mut origin = program(&parser::parse(ORIGIN).unwrap());
    let lowered = lower_telegram(PYTHON);
    let mut python = program(&lowered.operations);
    attach_format(&mut origin, false);
    attach_format(&mut python, true);
    super::validate_runtime_arguments(&lowered.operations, &python).unwrap();
    assert!(super::ast_equality::program(&origin, &python));
    let printed = super::print_module(&python, "TelegramBot").unwrap();
    let mut second = program(&lower_telegram(&printed).operations);
    attach_format(&mut second, true);
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
    module.body.retain(|op| match op {
        // #91 malformed original cannot be validated as a function. Its explicit
        // authored repair has a separate source/signature/ledger gate in voice.rs.
        parser::Operation::Function(f) => f.name != "transcribe_voice" && origin
            .get_function(&format!("TelegramBot.{}", f.name))
            .is_some(),
        parser::Operation::SharedVar(shared) => [
            "telegram_token",
            "admin_user_ids",
            "allowed_user_ids",
            "sandbox_perm_model",
            "intake_mode",
            "poll_offset",
            "poll_running",
        ]
        .contains(&shared.name.as_str()),
        _ => false,
    });
    module.doc = None;
    let selected = program(&operations);
    assert_eq!(selected.modules[0].functions.len(), 41);
    assert_eq!(
        selected.modules[0].shared_vars,
        origin.modules[0].shared_vars
    );
    verify_qualification(&selected, &origin, "TelegramBot");
}

fn run_paired(
    program: &ast::Program,
    tests: &[parser::Operation],
    check_vm: bool,
) -> (usize, Vec<String>) {
    assert_paired_coverage(program, tests);
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
                Ok(actual) if actual.matches(&expected) => {}
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
    let lowered = lower_telegram(PYTHON);
    let python = program(&lowered.operations);
    let tests = super::lower_tests(TESTS, "TelegramBot").unwrap();
    super::validate_runtime_arguments(&tests.operations, &python).unwrap();
    let (origin_count, origin_failures) = run_paired(&origin, &tests.operations, true);
    let (python_count, python_failures) = run_paired(&python, &tests.operations, true);
    assert_eq!(origin_count, TELEGRAM_CASES);
    assert_eq!(python_count, TELEGRAM_CASES);
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
    let python = program(&lower_telegram(PYTHON).operations);
    let tests = super::lower_tests(TESTS, "TelegramBot").unwrap();
    assert_eq!(
        run_paired(&origin, &tests.operations, false).0,
        TELEGRAM_CASES
    );
    assert_eq!(
        run_paired(&python, &tests.operations, false).0,
        TELEGRAM_CASES
    );
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
    let frozen = program(
        &parser::parse(include_str!(
            "../../tests/fixtures/python-parity/library/TgFormat.ax"
        ))
        .unwrap()
        .into_iter()
        .filter_map(|op| {
            if let parser::Operation::Module(mut module) = op {
                module
                    .body
                    .retain(|op| matches!(op, parser::Operation::Function(_)));
                Some(parser::Operation::Module(module))
            } else {
                None
            }
        })
        .collect::<Vec<_>>(),
    );
    assert_eq!(origin.modules[0].functions.len(), 18);
    assert_eq!(origin.modules[0].doc, frozen.modules[0].doc);
    verify_qualification(&frozen, &origin, "TgFormat");
    let tests = super::lower_tests(
        include_str!("../../tests/fixtures/telegram-python-port/python/test_tg_format.py"),
        "TgFormat",
    )
    .unwrap();
    assert_eq!(
        run_paired(&origin, &tests.operations, true),
        (FORMAT_CASES, vec![])
    );
    assert_eq!(
        run_paired(&python, &tests.operations, true),
        (FORMAT_CASES, vec![])
    );
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
        assert!(
            !pending.is_empty()
                || !module["pending_non_function_surface"]
                    .as_array()
                    .unwrap()
                    .is_empty()
        );
        assert_eq!(actual, done.union(&pending).copied().collect());
        let origin_path = root
            .join("tests/fixtures/telegram-python-port")
            .join(module["origin"].as_str().unwrap());
        let origin =
            program(&parser::parse(&std::fs::read_to_string(origin_path).unwrap()).unwrap());
        let mut implemented_and_new = done.clone();
        if let Some(introduced) = module["introduced_functions"].as_object() {
            implemented_and_new.extend(introduced.keys().map(String::as_str));
        }
        assert_eq!(
            origin.modules[0]
                .functions
                .iter()
                .map(|f| f.name.as_str())
                .collect::<BTreeSet<_>>(),
            implemented_and_new
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

fn verify_qualification(original: &ast::Program, corrected: &ast::Program, module: &str) {
    let locals = original.modules[0]
        .functions
        .iter()
        .map(|f| f.name.clone())
        .collect();
    let manifest: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/telegram-python-port/manifest.json"
    ))
    .unwrap();
    let ledger = manifest["modules"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["name"] == module)
        .unwrap();
    for function in &original.modules[0].functions {
        let target = format!("{module}.{}", function.name);
        if let Some(repair) = ledger["semantic_repairs"].get(&function.name) {
            if repair["kind"] == "typed_poll_loop_v1" {
                polling::looping::verify_repair(function, corrected.get_function(&target).unwrap());
            } else if repair["kind"] == "typed_poll_once_v1" {
                polling::cycle::verify_repair(function, corrected.get_function(&target).unwrap());
            } else if repair["kind"] == "typed_dispatch_v1" {
                polling::verify_repair(function, corrected.get_function(&target).unwrap());
            } else if repair["kind"] == "typed_attachment_reply_v1" || repair["kind"] == "typed_audio_upload_v1" {
                outbound::verify_repair(function, corrected.get_function(&target).unwrap());
            } else if repair["kind"] == "typed_intake_v1" {
                intake::verify_repair(function, corrected.get_function(&target).unwrap());
            } else if repair["kind"] == "transcript_kind_guard_v1" {
                voice::verify_helper(function, corrected.get_function(&target).unwrap());
            } else if repair["kind"] == "file_path_await_v1" {
                media::verify_file_path(function, corrected.get_function(&target).unwrap());
            } else if repair["kind"] == "text_handler_v1"
                || repair["kind"] == "attachment_handler_v1"
            {
                routing::verify_handler(function, corrected.get_function(&target).unwrap(), repair);
            } else {
                send_chain::verify_repair(
                    function,
                    corrected.get_function(&target).unwrap(),
                    repair,
                );
            }
            continue;
        }
        let count = qualification::verify(
            function,
            corrected.get_function(&target).unwrap(),
            module,
            &locals,
        );
        assert_eq!(
            count as u64,
            ledger["qualified_call_sites"][&function.name]
                .as_u64()
                .unwrap_or(0),
            "{target}: qualification site ledger"
        );
    }
}

fn assert_paired_coverage(program: &ast::Program, tests: &[parser::Operation]) {
    use std::collections::BTreeSet;
    let expected: BTreeSet<_> = program
        .modules
        .iter()
        .flat_map(|module| {
            module
                .functions
                .iter()
                .filter(|function| {
                    function.effects.is_empty()
                        && !routing::SHARED_FUNCTIONS.contains(&function.name.as_str())
                        && function.name != "transcription_upload_response"
                        && function.name != "telegram_upload_attempt"
                })
                .map(|function| format!("{}.{}", module.name, function.name))
        })
        .collect();
    let actual: BTreeSet<_> = tests
        .iter()
        .map(|operation| {
            let parser::Operation::Test(test) = operation else {
                panic!("only paired test declarations")
            };
            assert!(!test.cases.is_empty());
            test.function_name.clone()
        })
        .collect();
    assert_eq!(
        expected, actual,
        "every pure implemented function needs a paired assertion; effectful coverage is separately required"
    );
    let effectful: BTreeSet<_> = program
        .modules
        .iter()
        .flat_map(|module| {
            module
                .functions
                .iter()
                .filter(|f| !f.effects.is_empty())
                .map(|f| format!("{}.{}", module.name, f.name))
        })
        .collect();
    assert!(
        effectful.is_empty()
            || effectful
                == [
                    "TelegramBot.send_message_part".to_owned(),
                    "TelegramBot.send_message_parts".to_owned(),
                    "TelegramBot.send_message".to_owned(),
                    "TelegramBot.send_reply".to_owned(),
                    "TelegramBot.handle_group".to_owned(),
                    "TelegramBot.handle_admin_dm".to_owned(),
                    "TelegramBot.handle_user_dm".to_owned(),
                    "TelegramBot.process_text_update".to_owned(),
                    "TelegramBot.telegram_file_path".to_owned(),
                    "TelegramBot.handle_group_attachment".to_owned(),
                    "TelegramBot.handle_admin_dm_attachment".to_owned(),
                    "TelegramBot.handle_user_dm_attachment".to_owned(),
                    "TelegramBot.transcribe_voice".to_owned(),
                    "TelegramBot.process_update".to_owned(),
                    "TelegramBot.send_reply_with_attachment".to_owned(),
                    "TelegramBot.send_audio_file".to_owned(),
                    "TelegramBot.init".to_owned(),
                    "TelegramBot.handle_update".to_owned(),
                    "TelegramBot.dispatch_updates".to_owned(),
                    "TelegramBot.poll_once".to_owned(),
                    "TelegramBot.poll_loop".to_owned()
                ]
                .into_iter()
                .collect(),
        "effectful additions require normal-dispatch contract coverage"
    );
}

#[test]
fn telegram_port_metadata_missing_timestamp_preserves_source_error() {
    let origin = program(&parser::parse(ORIGIN).unwrap());
    let python = program(&lower_telegram(PYTHON).operations);
    for program in [&origin, &python] {
        let args = vec![eval::Value::string("{}")];
        for result in [
            super::logical_tier_tests::tree(program, "TelegramBot.source_metadata", args.clone()),
            super::logical_tier_tests::vm_run(program, "TelegramBot.source_metadata", args),
        ] {
            let error = result.unwrap_err().to_string();
            assert_eq!(error, "to_int: cannot parse '\"\"' as integer");
        }
    }
}

#[test]
fn telegram_port_inherited_oversized_parts_exceed_documented_delivery_limit() {
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
    let content = "x".repeat(5000);
    for text in [content.clone(), format!("<pre>{content}</pre>")] {
        for program in [&origin, &python] {
            let args = vec![eval::Value::string(&text), eval::Value::Int(3800)];
            for result in [
                super::logical_tier_tests::tree(
                    program,
                    "TgFormat.split_for_telegram",
                    args.clone(),
                ),
                super::logical_tier_tests::vm_run(program, "TgFormat.split_for_telegram", args),
            ] {
                let eval::Value::List(parts) = result.unwrap() else {
                    panic!("expected parts")
                };
                assert_eq!(parts.len(), 1);
                assert!(matches!(&parts[0], eval::Value::String(value) if value.as_str() == text));
                // Even excluding HTML tags, 5000 ASCII content characters exceed
                // Telegram's documented 4096 cap. No live Telegram call is made.
                assert!(content.len() > 4096);
            }
        }
    }
}
