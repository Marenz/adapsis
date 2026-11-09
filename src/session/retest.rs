//! Reject-on-fail rollback shared by textual and typed submission. A rejected
//! replacement restores the complete prior FunctionDecl (signature, effects,
//! body, docs and test flags), reusing the original Arc so CoW identity holds.
use super::*;

impl Session {
    pub(super) fn invalidate_and_retest(
        &mut self,
        operation: &parser::Operation,
        backups: HashMap<String, std::sync::Arc<ast::FunctionDecl>>,
    ) -> Vec<(String, bool, String)> {
        let affected = Self::affected_function_names(operation);
        for name in &affected {
            self.test_flags(name, false);
        }
        let mut results = Vec::new();
        for name in affected {
            let cases = self
                .program
                .get_function(&name)
                .map(|f| f.tests.clone())
                .unwrap_or_default();
            if cases.is_empty() {
                continue;
            }
            let operations = match stored_tests::operations(&name, &cases) {
                Ok(operations) => operations,
                Err(error) => {
                    self.restore_tested_body(&name, backups.get(&name));
                    results.push((
                        name,
                        false,
                        format!("retest parse error (reverted): {error}"),
                    ));
                    continue;
                }
            };
            for operation in operations {
                let parser::Operation::Test(test) = operation else {
                    continue;
                };
                let passed = self.retest_cases(&name, &test, &mut results);
                if passed {
                    self.test_flags(&name, true);
                } else if self.restore_tested_body(&name, backups.get(&name)) {
                    results.push((
                        name.clone(),
                        false,
                        "REJECTED: replacement reverted because existing tests failed".into(),
                    ));
                }
            }
        }
        results
    }

    fn retest_cases(
        &self,
        name: &str,
        test: &parser::TestMutation,
        results: &mut Vec<(String, bool, String)>,
    ) -> bool {
        let mut passed = true;
        for case in &test.cases {
            match crate::eval::eval_test_case_with_mocks(
                &self.program,
                &test.function_name,
                case,
                &self.meta.io_mocks,
                &self.runtime.http_routes,
            ) {
                Ok(message) => results.push((name.into(), true, format!("retest PASS: {message}"))),
                Err(error) => {
                    passed = false;
                    results.push((name.into(), false, format!("retest FAIL: {error}")));
                }
            }
        }
        passed
    }

    fn test_flags(&mut self, name: &str, passed: bool) {
        if let Some(function) = self.program.get_function_mut(name) {
            for test in &mut function.tests {
                test.passed = passed;
            }
        }
    }

    fn restore_tested_body(
        &mut self,
        name: &str,
        previous: Option<&std::sync::Arc<ast::FunctionDecl>>,
    ) -> bool {
        let Some(previous) = previous else {
            return false;
        };
        if let Some(slot) = function_slot(&mut self.program, name) {
            *slot = previous.clone();
            self.program.rebuild_function_index();
        }
        true
    }
}

/// The Arc slot `Program::get_function` resolves for `name`.
pub(super) fn function_slot<'a>(
    program: &'a mut ast::Program,
    name: &str,
) -> Option<&'a mut std::sync::Arc<ast::FunctionDecl>> {
    if let Some((module, function)) = name.split_once('.') {
        return program
            .modules
            .iter_mut()
            .find(|candidate| candidate.name == module)?
            .functions
            .iter_mut()
            .find(|candidate| candidate.name == function);
    }
    if let Some(index) = program.functions.iter().position(|f| f.name == name) {
        return program.functions.get_mut(index);
    }
    program
        .modules
        .iter_mut()
        .flat_map(|module| module.functions.iter_mut())
        .find(|candidate| candidate.name == name)
}
