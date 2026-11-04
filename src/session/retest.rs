//! Existing reject-on-fail body rollback shared by textual and typed submission.
use super::*;

impl Session {
    pub(super) fn invalidate_and_retest(
        &mut self,
        operation: &parser::Operation,
        backups: HashMap<String, Vec<ast::Statement>>,
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

    fn restore_tested_body(&mut self, name: &str, body: Option<&Vec<ast::Statement>>) -> bool {
        let Some(body) = body else { return false };
        if let Some(function) = self.program.get_function_mut(name) {
            function.body = body.clone();
            for test in &mut function.tests {
                test.passed = true;
            }
        }
        true
    }
}
