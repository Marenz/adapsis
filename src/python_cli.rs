//! Deterministic CLI demonstration of the production Python frontend.
use anyhow::{Context, Result, bail};

pub fn test_operations(
    path: &str,
    language: &str,
    module: Option<&str>,
    tests: Option<&str>,
) -> Result<Vec<crate::parser::Operation>> {
    match language {
        "adapsis" => {
            if module.is_some() || tests.is_some() {
                bail!(
                    "--module and --tests require --language python; Adapsis files already contain their declarations and tests"
                );
            }
            let source = std::fs::read_to_string(path)
                .with_context(|| format!("read test source {path}"))?;
            crate::parser::parse(&source)
        }
        "python" => {
            let module = module
                .filter(|value| !value.is_empty())
                .context("Python tests require --module <Name>")?;
            let tests = tests.context("Python tests require --tests <paired-test-file.py>; zero-test runs are not success")?;
            let source = std::fs::read_to_string(path)
                .with_context(|| format!("read Python source {path}"))?;
            let assertions = std::fs::read_to_string(tests)
                .with_context(|| format!("read Python assertions {tests}"))?;
            let module = crate::python_frontend::lower_module(&source, module)?;
            let tests = crate::python_frontend::lower_tests(&assertions, module_name(&module)?)?;
            if !tests.operations.iter().any(|op| {
                matches!(op,
                crate::parser::Operation::Test(test) if !test.cases.is_empty())
            }) {
                bail!("Python assertion file must contain at least one nonempty test_* assertion");
            }
            let mut program = crate::ast::Program::default();
            for operation in &module.operations {
                crate::validator::apply_and_validate(&mut program, operation)?;
            }
            crate::python_frontend::validate_runtime_arguments(&module.operations, &program)
                .map_err(anyhow::Error::msg)?;
            crate::python_frontend::validate_runtime_arguments(&tests.operations, &program)
                .map_err(anyhow::Error::msg)?;
            let mut operations = module.operations;
            operations.extend(tests.operations);
            Ok(operations)
        }
        _ => bail!("unsupported test language; expected adapsis or python"),
    }
}

fn module_name(lowered: &crate::python_frontend::LoweredSource) -> Result<&str> {
    match lowered.operations.as_slice() {
        [crate::parser::Operation::Module(module)] => Ok(&module.name),
        _ => bail!("Python module lowering did not produce exactly one explicit module"),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn typed_python_cli_normal_path_without_legacy_exporter() {
        if !crate::logging::tests::child() {
            crate::logging::tests::capture_with_env(
                "python_cli::tests::typed_python_cli_normal_path_without_legacy_exporter",
                false,
                &[("ADAPSIS_TEST_DENY_LEGACY_EXPORT", "1")],
            );
            return;
        }
        let root = env!("CARGO_MANIFEST_DIR");
        for (file, module, count) in [
            ("basic", "CliNoExportBasic", 2),
            ("flow", "CliNoExportFlow", 7),
        ] {
            let source = format!("{root}/examples/python/{file}.py");
            let tests = format!("{root}/examples/python/test_{file}.py");
            let operations =
                super::test_operations(&source, "python", Some(module), Some(&tests)).unwrap();
            let summary = crate::cli_test::run(&operations);
            assert_eq!((summary.passed, summary.failed), (count, 0));
        }
        let lowered =
            crate::python_frontend::lower_module("def value() -> int:\n    return 1\n", "Control")
                .unwrap();
        assert_eq!(
            lowered.legacy_export().unwrap_err().message,
            "legacy exporter unavailable in fixture"
        );
        assert_eq!(
            crate::python_frontend::lower_input("7")
                .unwrap_err()
                .message,
            "legacy exporter unavailable in fixture"
        );
    }
}
