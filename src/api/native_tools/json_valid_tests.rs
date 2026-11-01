use super::python_catalog_tests::{mutate, permissions};
use super::*;
use serde_json::json;

fn evaluated_value(result: &super::super::NativeToolResult) -> String {
    let content: serde_json::Value = serde_json::from_str(&result.content).unwrap();
    content["eval_results"].as_str().unwrap().to_owned()
}

#[tokio::test]
async fn native_python_json_valid_mutate_test_inspect_evaluate_with_hidden_state() {
    let baseline=build_program("+module NtJsonValidHidden\n+fn secret ()->String\n  +return \"JSON_VALID_PRIVATE_CANARY\"\n+end\n").await;
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        permissions(Some("NtJsonValidHidden")),
        baseline.clone(),
    );
    let mut ws = cfg.snapshot_working_set().await;
    let created = mutate(
        &cfg,
        &mut ws,
        &baseline,
        "NtJsonValidProbe",
        "@effects('fail')\ndef probe(payload: str) -> bool:\n    return json_valid(payload)\n",
    )
    .await;
    assert!(!created.execution.has_errors, "{}", created.content);
    cfg.write_back_working_set(&ws).await;
    let source =
        super::python_catalog_tests::inspect(&cfg, &mut ws, &baseline, "NtJsonValidProbe.probe")
            .await;
    assert!(source.contains("json_valid"));
    assert!(!source.contains("PRIVATE_CANARY"));
    // Malformed input is an ordinary `false` answer: unlike json_kind there is no
    // native "evaluated Err with has_errors=false" artefact for any String.
    for (index, (payload, expected)) in [
        ("{}", true),
        ("[]", true),
        ("null", true),
        ("18446744073709551615", true),
        ("not JSON", false),
        ("", false),
        ("{", false),
        ("1e400", false),
        ("<html>bad gateway</html>", false),
    ]
    .into_iter()
    .enumerate()
    {
        let literal = serde_json::to_string(payload).unwrap();
        let tested=execute_native_tool("test",&json!({"language":"python","module":"NtJsonValidProbe","source":format!("def test_case_{index}():\n    assert probe({literal}) == {}\n", if expected {"True"} else {"False"})}).to_string(),&cfg,&mut ws,&baseline).await;
        assert!(!tested.execution.has_errors, "{}", tested.content);
        assert!(!tested.execution.test_results.is_empty());
        let before = serde_json::to_value(&ws.program).unwrap();
        let evaluated=execute_native_tool("evaluate",&json!({"language":"python","function":"NtJsonValidProbe.probe","input":format!("payload={literal}")}).to_string(),&cfg,&mut ws,&baseline).await;
        assert_eq!(before, serde_json::to_value(&ws.program).unwrap());
        assert!(!evaluated.execution.has_errors, "{}", evaluated.content);
        let value = evaluated_value(&evaluated);
        assert!(
            value.contains(&expected.to_string()) && !value.contains("Err("),
            "{payload:?}: {value}"
        );
        assert!(!evaluated.content.contains("PRIVATE_CANARY"));
    }
    for (i, expression) in ["json_valid()", "json_valid('{}','x')", "json_valid(0)"]
        .iter()
        .enumerate()
    {
        let rejected = mutate(
            &cfg,
            &mut ws,
            &baseline,
            &format!("NtJsonValidInvalid{i}"),
            &format!("def probe() -> bool:\n    return {expression}\n"),
        )
        .await;
        assert!(rejected.execution.has_errors);
        assert!(!rejected.content.contains("PRIVATE_CANARY"));
    }
    let denied = mutate(
        &cfg,
        &mut ws,
        &baseline,
        "NtJsonValidDenied",
        "def probe() -> bool:\n    return json_valid(NtJsonValidHidden.secret())\n",
    )
    .await;
    assert!(denied.execution.has_errors);
    assert!(!denied.content.contains("PRIVATE_CANARY"));
}
