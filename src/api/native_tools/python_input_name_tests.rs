//! G2 through the production native path: `input` as a parameter/local binding name
//! applies, tests, evaluates and inspects; as a callee/function/type name it is refused
//! without changing the program.
use super::python_catalog_tests::{inspect, mutate, permissions};
use super::*;
use serde_json::json;

#[tokio::test]
async fn native_python_accepts_input_as_a_parameter_and_local_name() {
    let baseline =
        build_program("+module NtInputOther\n+fn one ()->Int\n  +return 1\n+end\n").await;
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        permissions(None),
        baseline.clone(),
    );
    let mut ws = cfg.snapshot_working_set().await;
    let source = "def measure(input: str) -> int:\n    n: int = len(input)\n    return n\n\ndef shout(text: str) -> str:\n    input: str = concat(text, '!')\n    return input\n";
    let applied = mutate(&cfg, &mut ws, &baseline, "NtInputName", source).await;
    assert!(!applied.execution.has_errors, "{}", applied.content);
    cfg.write_back_working_set(&ws).await;
    for (index, (call, expected)) in [("measure('abc')", "3"), ("shout('hi')", "'hi!'")]
        .into_iter()
        .enumerate()
    {
        let tested = execute_native_tool(
            "test",
            &json!({"language":"python","module":"NtInputName","source":format!("def test_case_{index}():\n    assert {call} == {expected}\n")}).to_string(),
            &cfg,
            &mut ws,
            &baseline,
        )
        .await;
        assert!(!tested.execution.has_errors, "{}", tested.content);
        assert!(!tested.execution.test_results.is_empty());
    }
    // The evaluate protocol names the parameter `input` as a keyword.
    let before = serde_json::to_value(&ws.program).unwrap();
    let evaluated = execute_native_tool(
        "evaluate",
        &json!({"language":"python","function":"NtInputName.measure","input":"input='abcd'"})
            .to_string(),
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert_eq!(before, serde_json::to_value(&ws.program).unwrap());
    assert!(!evaluated.execution.has_errors, "{}", evaluated.content);
    let content: serde_json::Value = serde_json::from_str(&evaluated.content).unwrap();
    assert!(
        content["eval_results"].as_str().unwrap().contains("= 4"),
        "{}",
        evaluated.content
    );
    // Inspection prints the stored function back, parameter and local included.
    let measure = inspect(&cfg, &mut ws, &baseline, "NtInputName.measure").await;
    assert!(
        measure.contains("def measure(input: str) -> int:"),
        "{measure}"
    );
    let shout = inspect(&cfg, &mut ws, &baseline, "NtInputName.shout").await;
    assert!(
        shout.contains("input: str = concat(text, \"!\")"),
        "{shout}"
    );
}

#[tokio::test]
async fn native_python_still_refuses_input_as_callee_function_and_type() {
    let baseline =
        build_program("+module NtInputOther\n+fn one ()->Int\n  +return 1\n+end\n").await;
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        permissions(None),
        baseline.clone(),
    );
    let mut ws = cfg.snapshot_working_set().await;
    for (index, source) in [
        "def f(x: str) -> str:\n    return input(x)\n",
        "def f(input: str) -> str:\n    return input(input)\n",
        "def input(x: int) -> int:\n    return x\n",
        "@dataclass\nclass input:\n    x: int\n",
        "input: int = 0\n",
    ]
    .into_iter()
    .enumerate()
    {
        let before = serde_json::to_value(&ws.program).unwrap();
        let rejected = mutate(
            &cfg,
            &mut ws,
            &baseline,
            &format!("NtInputRefused{index}"),
            source,
        )
        .await;
        assert!(
            rejected.execution.has_errors,
            "{source}\n{}",
            rejected.content
        );
        assert_eq!(
            before,
            serde_json::to_value(&ws.program).unwrap(),
            "{source}"
        );
    }
}
