//! Slices D2/D3: the stored `+test` blocks of the FROZEN TelegramBot (23 blocks / 61
//! cases) and TgFormat (10 blocks / 57 cases) modules. A manifest-driven gate assigns
//! exactly one disposition to every frozen case and proves each disposition against
//! the parsed fixtures. Redacted values are never recovered from any non-fixture
//! source: they are recorded as unrecoverable.
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
const TG_FROZEN: &str = include_str!("../../../tests/fixtures/python-parity/library/TgFormat.ax");
const TG_ORIGIN: &str =
    include_str!("../../../tests/fixtures/telegram-python-port/origin/TgFormat.ax");
const TG_TESTS: &str =
    include_str!("../../../tests/fixtures/telegram-python-port/python/test_tg_format.py");
const DISPOSITIONS: [&str; 5] = [
    "carried",
    "carried_ax_only",
    "covered_elsewhere",
    "redacted",
    "obsolete",
];

/// One frozen module whose stored tests are dispositioned.
struct Subject {
    module: &'static str,
    index: usize,
    frozen: &'static str,
    origin: &'static str,
    tests: &'static str,
    blocks: usize,
    cases: usize,
    /// carried + carried_ax_only + covered_elsewhere
    covered: usize,
    gate_test: &'static str,
    exec_test: &'static str,
    /// Dispositions whose functions the execution test must be registered for.
    exec_kinds: &'static [&'static str],
}

const TELEGRAM: Subject = Subject {
    module: "TelegramBot",
    index: 0,
    frozen: FROZEN,
    origin: ORIGIN,
    tests: TESTS,
    blocks: 23,
    cases: 61,
    covered: 44,
    gate_test: "telegram_port_stored_tests_every_frozen_case_has_exactly_one_disposition",
    exec_test: "telegram_port_stored_carried_cases_execute_on_both_origins_and_tiers",
    exec_kinds: &["carried"],
};

const TGFORMAT: Subject = Subject {
    module: "TgFormat",
    index: 1,
    frozen: TG_FROZEN,
    origin: TG_ORIGIN,
    tests: TG_TESTS,
    blocks: 10,
    cases: 57,
    covered: 56,
    gate_test: "telegram_port_tgformat_stored_every_frozen_case_has_exactly_one_disposition",
    exec_test: "telegram_port_tgformat_stored_cases_execute_verbatim_on_both_origins_and_tiers",
    exec_kinds: &["carried", "carried_ax_only"],
};

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
fn python_cases(s: &Subject) -> Vec<(String, String, parser::TestCase)> {
    let names: Vec<String> = s
        .tests
        .lines()
        .filter_map(|line| line.strip_prefix("def "))
        .filter_map(|rest| rest.split('(').next())
        .filter(|name| name.starts_with("test_"))
        .map(str::to_owned)
        .collect();
    let lowered = super::super::lower_tests(s.tests, s.module).unwrap();
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
    check_dispositions(&TELEGRAM);
}

#[test]
fn telegram_port_tgformat_stored_every_frozen_case_has_exactly_one_disposition() {
    check_dispositions(&TGFORMAT);
}

fn check_dispositions(subject: &Subject) {
    let manifest = manifest();
    let module = &manifest["modules"][subject.index];
    assert_eq!(module["name"], subject.module);
    let stored = &module["stored_tests"];
    let frozen = blocks(subject.frozen);
    let table = stored["dispositions"].as_object().unwrap();
    // Set equality (not counts): same functions, same case arity, blocks / cases.
    let frozen_functions: BTreeSet<_> = frozen.iter().map(|(name, _)| name.clone()).collect();
    assert_eq!(frozen.len(), subject.blocks);
    assert_eq!(
        frozen_functions.len(),
        subject.blocks,
        "one block per function"
    );
    assert_eq!(
        frozen_functions,
        table.keys().cloned().collect::<BTreeSet<_>>()
    );
    assert_eq!(
        frozen.iter().map(|(_, cases)| cases.len()).sum::<usize>(),
        subject.cases
    );
    assert_eq!(stored["frozen_blocks"], subject.blocks);
    assert_eq!(stored["frozen_cases"], subject.cases);

    let origin = blocks(subject.origin);
    let python = python_cases(subject);
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
                    assert_eq!(*qualified, format!("{}.{function}", subject.module));
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
    assert_eq!(summary.values().sum::<usize>(), subject.cases);
    let count = |kind: &str| summary.get(kind).copied().unwrap_or(0);
    assert_eq!(
        count("carried") + count("carried_ax_only") + count("covered_elsewhere"),
        subject.covered
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
    let executed: BTreeSet<String> = subject
        .exec_kinds
        .iter()
        .flat_map(|kind| by_kind.get(*kind).cloned().unwrap_or_default())
        .collect();
    for (test, functions) in [
        (subject.gate_test, frozen_functions.clone()),
        (subject.exec_test, executed),
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
                super::super::logical_tier_tests::vm_result_opcode_fixture_run(
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

#[test]
fn telegram_port_tgformat_stored_cases_execute_verbatim_on_both_origins_and_tiers() {
    // Every visible frozen TgFormat case (39 carried + 17 ax-only, the latter with
    // calls such as `code_wrap(..)` / `sample_lines()` resolved by their bare module
    // names exactly as in the frozen module) runs from the corrected-origin .ax blocks
    // on BOTH programs, tree and direct VM, with the shared pure-case runner.
    let operations: Vec<parser::Operation> = blocks(TG_ORIGIN)
        .into_iter()
        .map(|(name, cases)| {
            parser::Operation::Test(parser::TestMutation {
                function_name: format!("TgFormat.{name}"),
                cases,
            })
        })
        .collect();
    let [origin, python] = super::send_chain::programs();
    for program in [&origin, &python] {
        let (count, failures) = super::run_cases(program, &operations, true);
        assert_eq!(count, 56);
        assert!(failures.is_empty(), "{failures:#?}");
    }
    // Not vacuous: the redacted case is absent, and a corrupted expectation (a call
    // expression, so the evaluator path is exercised) is reported as a failure.
    assert!(!TG_ORIGIN.contains("<redacted>"));
    let mut corrupted = operations.clone();
    let target = corrupted
        .iter_mut()
        .find_map(|op| match op {
            parser::Operation::Test(test) if test.function_name == "TgFormat.inline_code" => {
                Some(test)
            }
            _ => None,
        })
        .unwrap();
    target.cases[0].expected = parser::Expr::String("corrupted".into());
    let outcome = std::panic::catch_unwind(|| {
        let [origin, _] = super::send_chain::programs();
        super::run_cases(&origin, &corrupted, true)
    });
    assert!(outcome.is_err() || !outcome.unwrap().1.is_empty());
}

/// Independent oracle for the strict-harness test printer (B1): the 80 carried frozen
/// cases were written by a script, not by `tests_print`, so the printer must reproduce
/// each stored Python assertion byte-for-byte, and must refuse exactly the cases the
/// port carries in the `.ax` only (calls and Result inputs stay outside the policy).
#[test]
fn telegram_port_stored_printer_reproduces_carried_python_assertions_and_refuses_ax_only() {
    for subject in [&TELEGRAM, &TGFORMAT] {
        let manifest = manifest();
        let table = manifest["modules"][subject.index]["stored_tests"]["dispositions"]
            .as_object()
            .unwrap()
            .clone();
        let (mut printed, mut refused) = (0, 0);
        for (function, cases) in blocks(subject.frozen) {
            for (index, case) in cases.iter().enumerate() {
                let mark = table[&function][index].as_str().unwrap();
                let result = super::super::tests_print::print_case(&function, case);
                match mark {
                    "carried" => {
                        let name = format!("def test_stored_{function}_{}() -> None:\n", index + 1);
                        let start = subject.tests.find(&name).expect("stored def") + name.len();
                        let end = subject.tests[start..]
                            .find("\n\n")
                            .map_or(subject.tests.len(), |offset| start + offset);
                        let stored = format!("{}\n", subject.tests[start..end].trim_end());
                        assert_eq!(
                            result.unwrap(),
                            stored,
                            "{} {function}[{}]",
                            subject.module,
                            index + 1
                        );
                        printed += 1;
                    }
                    "carried_ax_only" => {
                        assert!(
                            result.is_err(),
                            "{} {function}[{}] must stay outside the Python test-data policy",
                            subject.module,
                            index + 1
                        );
                        refused += 1;
                    }
                    _ => {}
                }
            }
        }
        let (want_printed, want_refused) = if subject.module == "TelegramBot" {
            (41, 2)
        } else {
            (39, 17)
        };
        assert_eq!(
            (printed, refused),
            (want_printed, want_refused),
            "{}",
            subject.module
        );
    }
}
