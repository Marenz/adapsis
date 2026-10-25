//! Deterministic CLI demonstration of the production Python frontend.
use anyhow::{Context, Result, bail};

pub fn test_source(
    path: &str,
    language: &str,
    module: Option<&str>,
    tests: Option<&str>,
) -> Result<String> {
    match language {
        "adapsis" => {
            if module.is_some() || tests.is_some() {
                bail!(
                    "--module and --tests require --language python; Adapsis files already contain their declarations and tests"
                );
            }
            std::fs::read_to_string(path).with_context(|| format!("read test source {path}"))
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
            Ok(format!(
                "{}\n{}",
                module.canonical_adapsis, tests.canonical_adapsis
            ))
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
