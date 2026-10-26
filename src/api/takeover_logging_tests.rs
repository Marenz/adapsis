//! Production takeover regressions, captured in subprocesses so policy is immutable.
use super::*;

fn captured(test: &str, canary: &str, category: &str) -> bool {
    if crate::logging::tests::child() { return false; }
    for enabled in [false, true] {
        let output = crate::logging::tests::capture(test, enabled);
        assert_eq!(output.contains(canary), enabled, "unexpected capture: {output}");
        assert!(output.contains(category), "missing category: {output}");
    }
    true
}

#[tokio::test]
async fn native_provider_error_journal_is_payload_free() {
    if captured("api::takeover_native::tests::logging::native_provider_error_journal_is_payload_free",
        "CANARY_PROVIDER_SECRET", "provider-error") { return; }
    let harness = Harness::new(true);
    // A 200 response with an error object bypasses LlmError::Api altogether.
    let provider = provider(vec![json!({"error":{"message":"CANARY_PROVIDER_SECRET"}})]).await;
    let reply = harness.turn(&provider, "telegram:group:-88001", ADMIN, "hello").await;
    assert!(reply.contains("technisches Problem"));
    assert!(!reply.contains("CANARY_PROVIDER_SECRET"), "apology leaked provider body");
    assert_eq!(provider.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn native_evaluate_regex_error_stays_in_tool_result() {
    if captured("api::takeover_native::tests::logging::native_evaluate_regex_error_stays_in_tool_result",
        "CANARY_REGEX_SECRET", "execute-code-error") { return; }
    let harness = Harness::new(true);
    let provider = provider(vec![
        // Sync named eval currently formats failures as success-valued Err(...).
        // Declared async (without doing IO) reaches the real push_err path.
        tool("mutate", "mutate", json!({"source":
            "+module NtLogRegex\n+fn bad()->String [async]\n +return regex_replace(\"[CANARY_REGEX_SECRET\", \"x\", \"text\")\n+end"})),
        tool("evaluate", "evaluate", json!({"function":"NtLogRegex.bad"})),
        response("Evaluation failed.", vec![call("done", "finish", json!({}))]),
    ]).await;
    harness.turn(&provider, "telegram:group:-88002", ADMIN, "check regex").await;
    let requests = provider.requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    let feedback = requests[2]["messages"].as_array().unwrap().iter()
        .find(|m| m["tool_call_id"] == "evaluate").unwrap()["content"].as_str().unwrap();
    assert!(feedback.contains("CANARY_REGEX_SECRET"));
    assert!(feedback.contains("regex_replace: invalid pattern"), "{feedback}");
    assert_eq!(serde_json::from_str::<Value>(feedback).unwrap()["success"], false, "{feedback}");
}

#[tokio::test]
async fn native_compaction_shape_error_is_redacted_in_both_sinks() {
    if captured("api::takeover_native::tests::logging::native_compaction_shape_error_is_redacted_in_both_sinks",
        "CANARY_EXTRACTED_TEXT", "memory-compaction-error") { return; }
    let context = "telegram:group:-88003";
    let mut harness = Harness::new(true);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("activity.log");
    harness.log_file = Some(Arc::new(tokio::sync::Mutex::new(
        tokio::fs::File::create(&path).await.unwrap())));
    // Cross the real default threshold (358400 chars, >=4 records), without
    // mutating process env or calling a hand-crafted error directly at a sink.
    for index in 0..40 {
        harness.graph.ingest_message(&crate::memory_graph::SourceMessage {
            id: format!("compaction-log-{index}"), platform_message_id: None,
            context_id: context.into(), context_kind: "test".into(), speaker_id: ADMIN.into(),
            speaker_name: "fixture".into(), role: "user".into(), content: "x".repeat(10_000),
            created_at_ms: index + 1,
        }, ADMIN).unwrap();
    }
    let provider = provider(vec![
        response(r#"{"summary":"ok","memories":"CANARY_EXTRACTED_TEXT"}"#, vec![]),
        response("Ready.", vec![call("done", "finish", json!({}))]),
    ]).await;
    assert_eq!(harness.turn(&provider, context, ADMIN, "hello").await, "Ready.");
    let requests = provider.requests.lock().unwrap();
    assert_eq!(requests.len(), 2, "compaction then native generation must both run");
    assert_eq!(requests[0]["stream"], true);
    assert!(requests[1]["tools"].is_array());
    let log = tokio::fs::read_to_string(path).await.unwrap();
    assert!(log.contains("memory-compaction-error"));
    assert_eq!(log.contains("CANARY_EXTRACTED_TEXT"), crate::logging::content());
    assert!(harness.graph.episode_summaries(context, 6).unwrap().is_empty());
}
