//! Slice D2: the 23 stored `+test` blocks / 61 cases of the FROZEN TelegramBot module.
//! A manifest-driven gate assigns exactly one disposition to every frozen case and
//! proves each disposition against the parsed fixtures. Redacted values are never
//! recovered from any non-fixture source: they are recorded as unrecoverable.
use super::{FROZEN, ORIGIN, TESTS};
use crate::{
    ast,
    eval::{self, Value},
    parser, session, shared_state,
};
use serde_json::Value as Json;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, RwLock};

const MANIFEST: &str = include_str!("../../../tests/fixtures/telegram-python-port/manifest.json");
const THIS_FILE: &str = include_str!("stored_tests.rs");
const REPLY_CLOSURE: &str = include_str!("send_chain/runtime/reply_closure.rs");
const SEND_RUNTIME: &str = include_str!("send_chain/runtime.rs");
const DISPOSITIONS: [&str; 5] = [
    "carried",
    "carried_ax_only",
    "covered_elsewhere",
    "redacted",
    "obsolete",
];

fn dbg(case: &parser::TestCase) -> String {
    format!("{case:?}")
}

/// Module-body `+test` blocks in source order: (function, cases).
fn blocks(source: &str) -> Vec<(String, Vec<parser::TestCase>)> {
    let operations = parser::parse(source).unwrap();
    let parser::Operation::Module(module) = &operations[0] else {
        panic!("module expected")
    };
    module
        .body
        .iter()
        .filter_map(|operation| match operation {
            parser::Operation::Test(test) => Some((test.function_name.clone(), test.cases.clone())),
            _ => None,
        })
        .collect()
}

/// Python tests: (def name, qualified function, case) in source order.
fn python_cases() -> Vec<(String, String, parser::TestCase)> {
    let names: Vec<String> = TESTS
        .lines()
        .filter_map(|line| line.strip_prefix("def "))
        .filter_map(|rest| rest.split('(').next())
        .filter(|name| name.starts_with("test_"))
        .map(str::to_owned)
        .collect();
    let lowered = super::super::lower_tests(TESTS, "TelegramBot").unwrap();
    assert_eq!(
        names.len(),
        lowered.operations.len(),
        "one operation per def"
    );
    names
        .into_iter()
        .zip(lowered.operations)
        .map(|(name, operation)| {
            let parser::Operation::Test(test) = operation else {
                panic!("only test declarations")
            };
            assert_eq!(test.cases.len(), 1);
            (name, test.function_name, test.cases[0].clone())
        })
        .collect()
}

fn manifest() -> Json {
    serde_json::from_str(MANIFEST).unwrap()
}

fn strings(value: &Json) -> BTreeSet<String> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item.as_str().unwrap().to_owned())
        .collect()
}

fn has_test_fn(name: &str, sources: &[&str]) -> bool {
    sources
        .iter()
        .any(|source| source.contains(&format!("fn {name}(")))
}

#[test]
fn telegram_port_stored_tests_every_frozen_case_has_exactly_one_disposition() {
    let manifest = manifest();
    let module = &manifest["modules"][0];
    let stored = &module["stored_tests"];
    let frozen = blocks(FROZEN);
    let table = stored["dispositions"].as_object().unwrap();
    // Set equality (not counts): same functions, same case arity, 23 blocks / 61 cases.
    let frozen_functions: BTreeSet<_> = frozen.iter().map(|(name, _)| name.clone()).collect();
    assert_eq!(frozen.len(), 23);
    assert_eq!(frozen_functions.len(), 23, "one block per function");
    assert_eq!(
        frozen_functions,
        table.keys().cloned().collect::<BTreeSet<_>>()
    );
    assert_eq!(
        frozen.iter().map(|(_, cases)| cases.len()).sum::<usize>(),
        61
    );
    assert_eq!(stored["frozen_blocks"], 23);
    assert_eq!(stored["frozen_cases"], 61);

    let origin = blocks(ORIGIN);
    let python = python_cases();
    let python_by_name: BTreeMap<_, _> = python
        .iter()
        .map(|(name, function, case)| (name.clone(), (function.clone(), dbg(case))))
        .collect();
    let mut expected_python = BTreeSet::new();
    let mut summary: BTreeMap<String, usize> = BTreeMap::new();
    let mut by_kind: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    for (function, cases) in &frozen {
        let marks: Vec<&str> = table[function]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_str().unwrap())
            .collect();
        assert_eq!(
            marks.len(),
            cases.len(),
            "{function}: one disposition per case"
        );
        let mut expected_origin = Vec::new();
        for (index, (case, mark)) in cases.iter().zip(&marks).enumerate() {
            assert!(
                DISPOSITIONS.contains(mark),
                "{function}[{}]: {mark}",
                index + 1
            );
            *summary.entry((*mark).to_owned()).or_default() += 1;
            by_kind
                .entry((*mark).to_owned())
                .or_default()
                .insert(function.clone());
            // Redaction is judged from the frozen case itself, in BOTH directions.
            assert_eq!(
                *mark == "redacted",
                dbg(case).contains("<redacted>"),
                "{function}[{}] redaction marker vs disposition",
                index + 1
            );
            let python_name = format!("test_stored_{function}_{}", index + 1);
            match *mark {
                "carried" => {
                    expected_origin.push(dbg(case));
                    expected_python.insert(python_name.clone());
                    let (qualified, python_case) = python_by_name
                        .get(&python_name)
                        .unwrap_or_else(|| panic!("missing Python test {python_name}"));
                    assert_eq!(*qualified, format!("TelegramBot.{function}"));
                    assert_eq!(*python_case, dbg(case), "{python_name}");
                }
                "carried_ax_only" => expected_origin.push(dbg(case)),
                _ => {}
            }
        }
        // The corrected-origin block is EXACTLY the ordered carried subsequence: no
        // redacted, obsolete or covered-elsewhere case is invented or smuggled in.
        let actual_origin: Vec<String> = origin
            .iter()
            .filter(|(name, _)| name == function)
            .flat_map(|(_, cases)| cases.iter().map(dbg))
            .collect();
        assert_eq!(
            actual_origin, expected_origin,
            "{function}: corrected-origin cases"
        );
        assert!(
            origin.iter().filter(|(name, _)| name == function).count()
                <= usize::from(!expected_origin.is_empty()),
            "{function}: at most one corrected-origin block"
        );
    }
    // No stored Python def exists beyond the carried set, and no corrected-origin block
    // exists for a function that has nothing to carry.
    let actual_python: BTreeSet<String> = python_by_name
        .keys()
        .filter(|name| name.starts_with("test_stored_"))
        .cloned()
        .collect();
    assert_eq!(actual_python, expected_python);
    let origin_functions: BTreeSet<_> = origin.iter().map(|(name, _)| name.clone()).collect();
    let carrying: BTreeSet<_> = table
        .iter()
        .filter(|(_, marks)| {
            marks
                .as_array()
                .unwrap()
                .iter()
                .any(|m| matches!(m.as_str().unwrap(), "carried" | "carried_ax_only"))
        })
        .map(|(name, _)| name.clone())
        .collect();
    assert_eq!(origin_functions, carrying);

    // Summary is derived, not trusted.
    let summary_manifest = stored["summary"].as_object().unwrap();
    for kind in DISPOSITIONS {
        assert_eq!(
            summary.get(kind).copied().unwrap_or(0) as u64,
            summary_manifest[kind].as_u64().unwrap(),
            "{kind}"
        );
    }
    assert_eq!(summary.values().sum::<usize>(), 61);
    assert_eq!(
        summary["carried"] + summary["carried_ax_only"] + summary["covered_elsewhere"],
        44
    );

    // Per-disposition detail maps must match the table by set equality.
    let detail = |name: &str| -> BTreeSet<String> {
        stored[name].as_object().unwrap().keys().cloned().collect()
    };
    let empty = BTreeSet::new();
    assert_eq!(
        detail("carried_ax_only"),
        *by_kind.get("carried_ax_only").unwrap_or(&empty)
    );
    assert_eq!(
        detail("covered_elsewhere"),
        *by_kind.get("covered_elsewhere").unwrap_or(&empty)
    );
    assert_eq!(
        detail("obsolete"),
        *by_kind.get("obsolete").unwrap_or(&empty)
    );
    // covered_elsewhere / carried_ax_only name their tests, which exist and are
    // registered in contract_tests; case indexes match the table.
    let contracts = module["contract_tests"].as_object().unwrap();
    let mut named = Vec::new();
    for (function, entry) in stored["covered_elsewhere"].as_object().unwrap() {
        named.push((
            function.clone(),
            vec![entry["test"].as_str().unwrap().to_owned()],
            &entry["cases"],
        ));
    }
    for (function, entry) in stored["carried_ax_only"].as_object().unwrap() {
        let tests = entry["tests"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_str().unwrap().to_owned())
            .collect();
        named.push((function.clone(), tests, &entry["cases"]));
    }
    for (function, tests, cases) in named {
        for test in tests {
            assert!(has_test_fn(&test, &[THIS_FILE]), "{test} must exist");
            assert!(
                strings(&contracts[&test]).contains(&function),
                "{test} contract"
            );
        }
        let indexes: Vec<u64> = cases
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_u64().unwrap())
            .collect();
        let marked: Vec<u64> = table[&function]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .filter(|(_, mark)| {
                matches!(
                    mark.as_str().unwrap(),
                    "covered_elsewhere" | "carried_ax_only"
                )
            })
            .map(|(index, _)| index as u64 + 1)
            .collect();
        assert_eq!(indexes, marked, "{function}");
    }
    // Obsolete means a LEDGERED repair that changed the contract; its replacement
    // tests exist.
    for (function, entry) in stored["obsolete"].as_object().unwrap() {
        assert_eq!(
            entry["repair"], module["semantic_repairs"][function]["kind"],
            "{function}"
        );
        for test in entry["analogue_tests"].as_array().unwrap() {
            let test = test.as_str().unwrap();
            assert!(has_test_fn(test, &[REPLY_CLOSURE, SEND_RUNTIME]), "{test}");
            assert!(contracts.contains_key(test), "{test} registered");
        }
    }
    // The execution and disposition tests are registered against the right functions.
    for (test, functions) in [
        (
            "telegram_port_stored_tests_every_frozen_case_has_exactly_one_disposition",
            frozen_functions.clone(),
        ),
        (
            "telegram_port_stored_carried_cases_execute_on_both_origins_and_tiers",
            by_kind["carried"].clone(),
        ),
    ] {
        assert!(has_test_fn(test, &[THIS_FILE]));
        assert_eq!(strings(&contracts[test]), functions, "{test}");
    }
    assert_eq!(
        module["pending_non_function_surface"],
        serde_json::json!([])
    );
}

fn qualified_carried(program_tests: &[(String, Vec<parser::TestCase>)]) -> Vec<parser::Operation> {
    program_tests
        .iter()
        .filter(|(name, _)| name != "transcription_upload_response")
        .map(|(name, cases)| {
            parser::Operation::Test(parser::TestMutation {
                function_name: format!("TelegramBot.{name}"),
                cases: cases.clone(),
            })
        })
        .collect()
}

#[test]
fn telegram_port_stored_carried_cases_execute_on_both_origins_and_tiers() {
    // The carried cases come from the corrected-origin .ax module body (already proved
    // identical to the frozen cases and the Python-lowered cases) and run on BOTH
    // programs, tree and direct VM, with the same runner as the paired pure gate.
    let carried = qualified_carried(&blocks(ORIGIN));
    let [origin, python] = super::send_chain::programs();
    for program in [&origin, &python] {
        let (count, failures) = super::run_cases(program, &carried, true);
        assert_eq!(count, 41);
        assert!(failures.is_empty(), "{failures:#?}");
    }
    // The runner is not vacuous: a corrupted expectation is reported as a failure.
    let mut corrupted = carried.clone();
    let parser::Operation::Test(test) = &mut corrupted[0] else {
        unreachable!()
    };
    test.cases[0].expected = parser::Expr::String("corrupted".into());
    let result = std::panic::catch_unwind(|| {
        let [origin, _] = super::send_chain::programs();
        super::run_cases(&origin, &corrupted, true)
    });
    assert!(result.is_err() || !result.unwrap().1.is_empty());
}

struct StateRestore {
    runtime: Option<session::SharedRuntime>,
    program: Option<Arc<ast::Program>>,
}
impl Drop for StateRestore {
    fn drop(&mut self) {
        shared_state::set_shared_runtime(self.runtime.take());
        shared_state::set_shared_program(self.program.take());
    }
}

#[test]
fn telegram_port_stored_is_allowed_default_state_covers_frozen_case() {
    let frozen = blocks(FROZEN);
    let (_, cases) = frozen
        .iter()
        .find(|(name, _)| name == "is_allowed")
        .unwrap();
    assert_eq!(cases.len(), 1);
    for program in super::send_chain::programs() {
        // Declared default state only: nothing seeded, so allowed_user_ids is the
        // declared "" and user 999 is not on the list.
        let _restore = StateRestore {
            runtime: shared_state::get_shared_runtime(),
            program: shared_state::get_shared_program(),
        };
        shared_state::set_shared_runtime(Some(Arc::new(RwLock::new(
            session::RuntimeState::default(),
        ))));
        eval::eval_test_case(&program, "TelegramBot.is_allowed", &cases[0]).unwrap();
        // Not vacuous: the same input with the opposite expectation fails.
        let mut wrong = cases[0].clone();
        wrong.expected = parser::Expr::Bool(true);
        assert!(eval::eval_test_case(&program, "TelegramBot.is_allowed", &wrong).is_err());
    }
}

/// `Ok("..")` / `Err("..")` input from a frozen case -> a trusted runtime Value. The
/// text comes from the frozen case, not from retyped literals.
fn trusted_result(case: &parser::TestCase) -> Value {
    let parser::Expr::StructLiteral(fields) = &case.input else {
        panic!("struct input expected")
    };
    let [field] = fields.as_slice() else {
        panic!("one field expected")
    };
    assert_eq!(field.name, "response");
    let parser::Expr::Call { callee, args } = &field.value else {
        panic!("Ok/Err call expected")
    };
    let parser::Expr::Ident(kind) = callee.as_ref() else {
        panic!("identifier callee expected")
    };
    let [parser::Expr::String(text)] = args.as_slice() else {
        panic!("one String argument expected")
    };
    match kind.as_str() {
        "Ok" => Value::Ok(Box::new(Value::string(text.clone()))),
        "Err" => Value::Err(text.clone()),
        other => panic!("unexpected constructor {other}"),
    }
}

#[test]
fn telegram_port_stored_transcription_cases_run_ax_origin_and_trusted_results() {
    let frozen = blocks(FROZEN);
    let (_, cases) = frozen
        .iter()
        .find(|(name, _)| name == "transcription_upload_response")
        .unwrap();
    let carried = blocks(ORIGIN);
    let (_, ax_cases) = carried
        .iter()
        .find(|(name, _)| name == "transcription_upload_response")
        .unwrap();
    assert_eq!(
        cases.iter().map(dbg).collect::<Vec<_>>(),
        ax_cases.iter().map(dbg).collect::<Vec<_>>()
    );
    let [origin, python] = super::send_chain::programs();
    // (1) The carried `+test` cases execute through the stored-test runner on the
    //     corrected-origin .ax program (the Python test-data literal policy is unchanged
    //     and cannot express Ok/Err inputs).
    for case in ax_cases {
        eval::eval_test_case(&origin, "TelegramBot.transcription_upload_response", case).unwrap();
    }
    // (2) The same frozen cases as Rust-constructed Result Values on BOTH origins,
    //     tree and direct VM.
    for program in [&origin, &python] {
        for case in cases {
            let parser::Expr::String(expected) = &case.expected else {
                panic!("String expectation expected")
            };
            for result in [
                super::super::logical_tier_tests::tree(
                    program,
                    "TelegramBot.transcription_upload_response",
                    vec![trusted_result(case)],
                ),
                super::super::logical_tier_tests::vm_run(
                    program,
                    "TelegramBot.transcription_upload_response",
                    vec![trusted_result(case)],
                ),
            ] {
                assert!(
                    matches!(result.unwrap(), Value::String(ref text) if text.as_str() == expected)
                );
            }
        }
    }
}
