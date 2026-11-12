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
    // Cross the real default threshold for an unknown model (192000 serialized
    // bytes at the default 2.5 bytes/token; one 40000-byte chunk of >=4 records),
    // without mutating process env or calling a hand-crafted error directly at a sink.
    for index in 0..40 {
        harness.graph.ingest_message(&crate::memory_graph::SourceMessage {
            id: format!("compaction-log-{index}"), platform_message_id: None,
            context_id: context.into(), context_kind: "test".into(), speaker_id: ADMIN.into(),
            speaker_name: "fixture".into(), role: "user".into(), content: "x".repeat(6_000),
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

/// #129/#88: a `run` block's source and result are content. Through a complete native
/// takeover turn with a `--log-file`, neither reaches stdout, stderr or the log file by
/// default; the model still receives both results in its tool messages.
#[tokio::test]
async fn native_run_source_and_result_stay_out_of_the_journal_and_the_log_file() {
    if captured("api::takeover_native::tests::logging::native_run_source_and_result_stay_out_of_the_journal_and_the_log_file",
        "CANARY_RUN_TURN", "run-error") { return; }
    let mut harness = Harness::new(true);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("activity.log");
    harness.log_file = Some(Arc::new(tokio::sync::Mutex::new(
        tokio::fs::File::create(&path).await.unwrap())));
    let provider = provider(vec![
        tool("value", "run", json!({"source": "text: str = \"CANARY_RUN_TURN_SOURCE\"\nreturn concat(text, \"_RESULT\")\n"})),
        tool("failure", "run", json!({"source": "check(\"probe\", 1 > 2, \"CANARY_RUN_TURN_ERROR\")\nreturn 1\n"})),
        response("Ran both.", vec![call("done", "finish", json!({}))]),
    ]).await;
    assert_eq!(harness.turn(&provider, "telegram:group:-88010", ADMIN, "run it").await, "Ran both.");
    let requests = provider.requests.lock().unwrap();
    let delivered = requests[2]["messages"].to_string();
    assert!(delivered.contains("CANARY_RUN_TURN_SOURCE_RESULT"), "{delivered}");
    assert!(delivered.contains("CANARY_RUN_TURN_ERROR"), "{delivered}");
    let log = tokio::fs::read_to_string(path).await.unwrap();
    // The takeover log file carries user/ai-text entries only; tool content never.
    assert!(!log.contains("CANARY_RUN_TURN"), "{log}");
    assert_eq!(log.contains("Ran both."), crate::logging::content(), "{log}");
}

#[tokio::test]
async fn native_provider_error_feeds_the_runtime_error_journal() {
    let last = |text: &str| serde_json::from_str::<Value>(text).unwrap();
    let before = last(&crate::error_journal::query(u64::MAX - 1))["last_seq"].as_u64().unwrap();
    let harness = Harness::new(true);
    let provider = provider(vec![json!({"error":{"message":"JOURNAL_PROVIDER_BODY"}})]).await;
    harness.turn(&provider, "telegram:group:-88009", ADMIN, "private journal chat text").await;
    let journal = last(&crate::error_journal::query(before));
    let entries = journal["entries"].as_array().unwrap();
    let entry = entries.iter()
        .find(|e| e["category"] == "provider" && e["source"] == "llm_takeover_native")
        .unwrap_or_else(|| panic!("no provider entry: {entries:#?}"));
    assert!(entry["fingerprint"].as_str().unwrap().starts_with("provider:"));
    assert!(entries.iter().all(|e| !e.to_string().contains("private journal chat text")));
    // The provider's own error text is kept (redacted, 1 KiB): it is the diagnosis.
    assert!(entry["message"].as_str().unwrap().starts_with("provider error"));
}

/// Workspace tools: paths, patterns and file content are content. The default log has
/// one metadata line per call (tool, success, workspace name, sizes) and none of them.
#[tokio::test]
async fn workspace_tool_calls_log_metadata_only() {
    if captured("api::takeover_native::tests::logging::workspace_tool_calls_log_metadata_only",
        "CANARY_WS_SECRET", "tool=read ok=true workspace=log-repo/logged result_bytes=") { return; }
    let repo = crate::workspace::test_support::RepoFixture::new();
    crate::workspace::test_support::write(&repo.repository.checkout, "CANARY_WS_SECRET.txt", "CANARY_WS_SECRET content\n");
    crate::workspace::test_support::commit_all(&repo.repository.checkout, "canary");
    let mut harness = Harness::new(true);
    harness.workspaces = Some(repo.site("log-repo"));
    let provider = provider(vec![
        tool("open", "workspace_open", json!({"repo": "log-repo", "name": "logged"})),
        response("", vec![
            call("rd", "read", json!({"path": "CANARY_WS_SECRET.txt"})),
            call("se", "search", json!({"pattern": "CANARY_WS_SECRET"})),
            call("gl", "glob", json!({"pattern": "CANARY_WS_SECRET*"})),
            call("miss", "read", json!({"path": "CANARY_WS_SECRET/absent"})),
        ]),
        response("Read.", vec![call("done", "finish", json!({}))]),
    ]).await;
    assert_eq!(harness.turn(&provider, "telegram:user:100000001", ADMIN, "read it").await, "Read.");
    // The model did get the content.
    let requests = provider.requests.lock().unwrap();
    assert!(all_text(&requests[2]).contains("CANARY_WS_SECRET content"));
}
