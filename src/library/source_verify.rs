//! Startup re-run of stored test cases for a freshly loaded managed set.
//!
//! Cases parsed from source are untested: parsing is not evidence. After the
//! whole set has validated, every stored case is run once through the existing
//! test evaluator, whose coroutine handle is strictly mock-only (no mocks are
//! supplied, so any IO request fails closed and never reaches a real IO loop).
//! A passing case is marked passed; a failing or erroring case stays untested.
//! Diagnostics carry module/function names and counts only, never case content.
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use crate::{ast, parser};

#[derive(Debug, Default)]
pub(super) struct Rerun {
    pub(super) cases: usize,
    pub(super) passed: usize,
    pub(super) elapsed: Duration,
    /// module -> function -> number of cases that did not pass.
    pub(super) not_passing: BTreeMap<String, BTreeMap<String, usize>>,
}

/// `tests` holds, per managed module, the parsed cases in stored order (the
/// same order `store_test_payload` wrote them onto each function).
pub(super) fn rerun_stored_cases(
    candidate: &mut ast::Program,
    tests: &BTreeMap<String, Vec<parser::TestMutation>>,
) -> Rerun {
    let started = Instant::now();
    // Module route declarations are static source facts; `+after routes` checks
    // see them without any registration having happened.
    let routes: Vec<ast::HttpRoute> = candidate
        .modules
        .iter()
        .flat_map(|module| module.routes.clone())
        .collect();
    // The evaluator installs thread-local snapshots; restore the caller's.
    let previous_program = crate::shared_state::get_shared_program();
    let mut outcomes: Vec<(String, String, usize, bool)> = Vec::new();
    for (module, groups) in tests {
        for group in groups {
            for (index, case) in group.cases.iter().enumerate() {
                let passed = crate::eval::eval_test_case_with_mocks(
                    candidate,
                    &group.function_name,
                    case,
                    &[],
                    &routes,
                )
                .is_ok();
                outcomes.push((module.clone(), group.function_name.clone(), index, passed));
            }
        }
    }
    crate::shared_state::set_shared_program(previous_program);
    let mut summary = Rerun {
        cases: outcomes.len(),
        ..Rerun::default()
    };
    for (module, function, index, passed) in outcomes {
        if passed {
            summary.passed += 1;
            if let Some(case) = candidate
                .get_function_mut(&function)
                .and_then(|decl| decl.tests.get_mut(index))
            {
                case.passed = true;
            }
        } else {
            let bare = function.rsplit('.').next().unwrap_or(&function).to_string();
            *summary
                .not_passing
                .entry(module)
                .or_default()
                .entry(bare)
                .or_default() += 1;
        }
    }
    candidate.rebuild_function_index();
    summary.elapsed = started.elapsed();
    summary
}

/// One startup log line per affected module and one journal entry per module.
pub(super) fn report(summary: &Rerun) {
    eprintln!(
        "[library] stored tests re-run mock-only: {} case(s), {} passed, {} untested, {} ms",
        summary.cases,
        summary.passed,
        summary.cases - summary.passed,
        summary.elapsed.as_millis()
    );
    for (module, functions) in &summary.not_passing {
        let cases: usize = functions.values().sum();
        let names = functions
            .iter()
            .map(|(name, count)| format!("{name} ({count})"))
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!(
            "[library] stored tests: module `{module}`: {cases} case(s) did not pass and stay untested: {names}"
        );
        crate::error_journal::record(
            crate::error_journal::Category::Library,
            module,
            format!(
                "{cases} stored test case(s) in {} function(s) did not pass the mock-only startup run and stay untested",
                functions.len()
            ),
        );
    }
}
