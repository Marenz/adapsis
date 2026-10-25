use std::process::{Command, Output};

fn run(source: &str, assertions: &str) -> Output {
    let directory = tempfile::tempdir().unwrap();
    let module = directory.path().join("module.py");
    let tests = directory.path().join("test_module.py");
    std::fs::write(&module, source).unwrap();
    std::fs::write(&tests, assertions).unwrap();
    Command::new(env!("CARGO_BIN_EXE_adapsis"))
        .arg("test")
        .arg(&module)
        .args(["--language", "python", "--module", "PyCli"])
        .arg("--tests")
        .arg(&tests)
        .env("ADAPSIS_MODULES_DIR", directory.path().join("modules"))
        .env("ADAPSIS_MEMORY_DB", directory.path().join("unused.lbug"))
        .output()
        .unwrap()
}

#[test]
fn python_cli_runs_paired_examples_through_the_production_frontend() {
    for (source, tests, passed) in [
        (
            include_str!("../examples/python/basic.py"),
            include_str!("../examples/python/test_basic.py"),
            2,
        ),
        (
            include_str!("../examples/python/flow.py"),
            include_str!("../examples/python/test_flow.py"),
            7,
        ),
    ] {
        let output = run(source, tests);
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout)
                .contains(&format!("TEST SUMMARY: {passed} passed, 0 failed"))
        );
    }
}

#[test]
fn python_cli_failed_expectation_has_nonzero_status() {
    let output = run(
        "def value() -> int:\n    return 1\n",
        "def test_value() -> None:\n    assert value() == 2\n",
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("1 failed"));
}

#[test]
fn python_cli_unsupported_syntax_has_location_and_nonzero_status() {
    let output = run(
        "class Unsupported:\n    pass\n",
        "def test_value() -> None:\n    assert value() == 1\n",
    );
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("PyCli") && error.contains(":1:"), "{error}");
}

#[test]
fn python_cli_requires_explicit_namespace_and_nonempty_test_file() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("module.py");
    std::fs::write(&source, "def value() -> int:\n    return 1\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_adapsis"))
        .arg("test")
        .arg(&source)
        .args(["--language", "python", "--module", "PyCli"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("--tests"));
    assert!(
        !run("def value() -> int:\n    return 1\n", "")
            .status
            .success()
    );
}

#[test]
fn python_cli_rejects_ignored_or_misnamed_assertion_arguments_before_testing() {
    let source = "def constant() -> int:\n    return 1\n\ndef ordered(x: int, y: int) -> int:\n    return x * 10 + y\n";
    for expression in [
        "constant(999)",
        "constant(unexpected=999)",
        "ordered(1)",
        "ordered(1, 2, 3)",
        "ordered(x=1, y=2, extra=3)",
        "ordered(wrong=1, y=2)",
        "ordered(_0=1, _1=2)",
    ] {
        let tests = format!("def test_shape() -> None:\n    assert {expression} == 1\n");
        let output = run(source, &tests);
        assert!(!output.status.success(), "{expression} unexpectedly passed");
        assert!(!String::from_utf8_lossy(&output.stdout).contains("PASS"));
    }
    let output = run(
        "def bad(_0: int) -> int:\n    return _0\n",
        "def test_bad():\n    assert bad(1) == 1\n",
    );
    assert!(!output.status.success());
}
