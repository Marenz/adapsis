use super::python_catalog_tests::{mutate, permissions};
use super::*;
use serde_json::json;

#[tokio::test]
async fn native_python_json_kind_mutate_test_inspect_evaluate_with_hidden_state() {
    let baseline=build_program("+module NtJsonKindHidden\n+fn secret ()->String\n  +return \"JSON_KIND_PRIVATE_CANARY\"\n+end\n").await;
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        permissions(Some("NtJsonKindHidden")),
        baseline.clone(),
    );
    let mut ws = cfg.snapshot_working_set().await;
    let created=mutate(&cfg,&mut ws,&baseline,"NtJsonKindProbe","@effects('fail')\ndef probe(payload: str, path: str) -> str:\n    return json_kind(payload, path)\n").await;
    assert!(!created.execution.has_errors, "{}", created.content);
    cfg.write_back_working_set(&ws).await;
    let source =
        super::python_catalog_tests::inspect(&cfg, &mut ws, &baseline, "NtJsonKindProbe.probe")
            .await;
    assert!(source.contains("json_kind"));
    assert!(!source.contains("PRIVATE_CANARY"));
    for (index, (payload, path, kind)) in [
        ("{}", "x", "missing"),
        (r#"{"x":""}"#, "x", "string"),
        (r#"{"x":null}"#, "x", "null"),
        (r#"{"x":"null"}"#, "x", "string"),
        (r#"{"x":[]}"#, "x", "array"),
        (r#"{"x":{}}"#, "x", "object"),
        ("1.0", "", "float"),
        ("1", "", "int"),
        ("false", "", "bool"),
        (
            r#"{"NtJsonKindHidden":"data"}"#,
            "NtJsonKindHidden",
            "string",
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let payload = serde_json::to_string(payload).unwrap();
        let path = serde_json::to_string(path).unwrap();
        let tested=execute_native_tool("test",&json!({"language":"python","module":"NtJsonKindProbe","source":format!("def test_case_{index}():\n    assert probe({payload}, {path}) == '{kind}'\n")}).to_string(),&cfg,&mut ws,&baseline).await;
        assert!(!tested.execution.has_errors, "{}", tested.content);
        assert!(!tested.execution.test_results.is_empty());
        let before=serde_json::to_value(&ws.program).unwrap();
        let evaluated=execute_native_tool("evaluate",&json!({"language":"python","function":"NtJsonKindProbe.probe","input":format!("payload={payload}, path={path}")}).to_string(),&cfg,&mut ws,&baseline).await;
        assert_eq!(before,serde_json::to_value(&ws.program).unwrap());
        assert!(!evaluated.execution.has_errors, "{}", evaluated.content);
        let content: serde_json::Value = serde_json::from_str(&evaluated.content).unwrap();
        assert!(
            content["eval_results"]
                .as_str()
                .unwrap()
                .contains(&format!("\"{kind}\"")),
            "{}",
            evaluated.content
        );
        assert!(!evaluated.content.contains("PRIVATE_CANARY"));
    }
    for (payload, path) in [
        ("not JSON", "x"),
        ("9223372036854775808", ""),
        ("1e400", ""),
    ] {
        let before = serde_json::to_value(&ws.program).unwrap();
        let evaluated=execute_native_tool("evaluate",&json!({"language":"python","function":"NtJsonKindProbe.probe","input":format!("payload={},path={}",serde_json::to_string(payload).unwrap(),serde_json::to_string(path).unwrap())}).to_string(),&cfg,&mut ws,&baseline).await;
        native_failure(&evaluated);
        assert_eq!(before, serde_json::to_value(&ws.program).unwrap());
    }
    for (i, expression) in ["json_kind('{}')", "json_kind('{}',0)"].iter().enumerate() {
        let rejected = mutate(
            &cfg,
            &mut ws,
            &baseline,
            &format!("NtJsonKindInvalid{i}"),
            &format!("def probe() -> str:\n    return {expression}\n"),
        )
        .await;
        assert!(rejected.execution.has_errors);
        assert!(!rejected.content.contains("PRIVATE_CANARY"));
    }
    let before=serde_json::to_value(&ws.program).unwrap();
    let invalid_path=execute_native_tool("evaluate",&json!({"language":"python","function":"NtJsonKindProbe.probe","input":"payload='{}', path=0"}).to_string(),&cfg,&mut ws,&baseline).await;
    native_failure(&invalid_path);
    assert_eq!(before,serde_json::to_value(&ws.program).unwrap());
    let denied = mutate(
        &cfg,
        &mut ws,
        &baseline,
        "NtJsonKindDenied",
        "def probe() -> str:\n    return json_kind(NtJsonKindHidden.secret(), '')\n",
    )
    .await;
    assert!(denied.execution.has_errors);
    assert!(!denied.content.contains("PRIVATE_CANARY"));
}

fn native_failure(result: &super::super::NativeToolResult) {
    // Existing native evaluate protocol reports an evaluated Err as text while
    // has_errors remains false. Pin it honestly, not as a correctly flagged status.
    assert!(!result.execution.has_errors);
    // The legacy mutation_results vector also carries eval reports; actual
    // Program immutability is asserted from snapshots at the call site.
    let content: serde_json::Value = serde_json::from_str(&result.content).unwrap();
    let value = content["eval_results"].as_str().unwrap();
    assert!(value.contains(" = Err(json_kind"), "{value}");
    assert!(value.ends_with(')'), "{value}");
    for kind in [
        "missing", "null", "bool", "int", "float", "string", "array", "object",
    ] {
        assert!(
            !value.ends_with(&format!("= \"{kind}\"")),
            "error must not be a valid-kind String"
        );
    }
    assert!(!result.content.contains("PRIVATE_CANARY"));
}
