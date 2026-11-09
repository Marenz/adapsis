//! C runtime contract through native tooling. Canonical declarations deliberately retain
//! the existing Python G7/user-Result-await guard; no frontend policy relaxation here.
use super::{python_catalog_tests::permissions, *};
use serde_json::json;

#[tokio::test]
async fn native_result_declaration_and_ordinary_consumer_share_the_runtime_contract() {
    let baseline = Program::default();
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        permissions(None),
        baseline.clone(),
    );
    let mut ws = cfg.snapshot_working_set().await;
    let source = "+module CCoreNative\n+fn plain ()->Result<Int>\n  +return 7\n+end\n+fn rejected ()->Result<Int> [fail]\n  +check valid false ~invalid\n  +return 7\n+end\n+fn observe ()->Bool\n  +match CCoreNative.plain()\n  +case Ok(value)\n    +return value == 7\n  +case Err(error)\n    +return false\n  +end\n+end\n";
    let created = execute_native_tool(
        "mutate",
        &json!({"source":source}).to_string(),
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(!created.execution.has_errors, "{}", created.content);
    let tested = execute_native_tool(
        "test",
        &json!({"source":"+test CCoreNative.observe\n  +with -> expect true\n"}).to_string(),
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(!tested.execution.has_errors, "{}", tested.content);
    assert!(!tested.execution.test_results.is_empty());
    for (function, expected) in [
        ("plain", "Ok(7)"),
        ("rejected", "Err(invalid)"),
        ("observe", "true"),
    ] {
        let evaluated = execute_native_tool(
            "evaluate",
            &json!({"function":format!("CCoreNative.{function}")}).to_string(),
            &cfg,
            &mut ws,
            &baseline,
        )
        .await;
        assert!(!evaluated.execution.has_errors, "{}", evaluated.content);
        let value: serde_json::Value = serde_json::from_str(&evaluated.content).unwrap();
        assert!(
            value["eval_results"]
                .as_str()
                .unwrap()
                .contains(&format!("= {expected}")),
            "{}",
            evaluated.content
        );
    }
}

#[tokio::test]
async fn native_operational_error_flags_are_structural_not_message_classification() {
    let baseline = Program::default();
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        permissions(None),
        baseline.clone(),
    );
    let mut ws = cfg.snapshot_working_set().await;
    let source = "+module CCoreError\n+fn application ()->Result<String>\n  +return Err(\"division by zero\")\n+end\n+fn text ()->String\n  +return \"division by zero\"\n+end\n+fn broken_result ()->Result<Int>\n  +return 1 / 0\n+end\n+fn broken_scalar ()->Int [fail]\n  +return 1 / 0\n+end\n";
    let created = execute_native_tool(
        "mutate",
        &json!({"source":source}).to_string(),
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(!created.execution.has_errors, "{}", created.content);
    for (name, failed) in [
        ("application", false),
        ("text", false),
        ("broken_result", true),
        ("broken_scalar", true),
    ] {
        let before = serde_json::to_value(&ws.program).unwrap();
        let result = execute_native_tool(
            "evaluate",
            &json!({"function":format!("CCoreError.{name}")}).to_string(),
            &cfg,
            &mut ws,
            &baseline,
        )
        .await;
        assert_eq!(result.execution.has_errors, failed, "{}", result.content);
        assert!(
            result.content.contains("division by zero"),
            "{}",
            result.content
        );
        assert_eq!(before, serde_json::to_value(&ws.program).unwrap());
    }
}
