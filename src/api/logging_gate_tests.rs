//! Capture the actual event/log-file sinks with process-isolated policies.
use super::*;

#[tokio::test]
async fn event_sender_logs_content_only_when_opted_in() {
    if !crate::logging::tests::child() {
        for enabled in [false, true] {
            let captured = crate::logging::tests::capture(
                "api::llm_handlers::logging_gate_tests::event_sender_logs_content_only_when_opted_in", enabled);
            assert_eq!(captured.contains("CANARY_EVENT"), enabled);
        }
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("events.log");
    let log_file = Some(std::sync::Arc::new(tokio::sync::Mutex::new(
        tokio::fs::File::create(&path).await.unwrap())));
    let (tx, mut rx) = tokio::sync::mpsc::channel(16);
    let broadcast = tokio::sync::broadcast::channel(16).0;
    let sender = EventSender::with_mpsc(tx, broadcast, log_file.clone());
    for tag in ["iter", "done", "done-rejected", "llm-error", "inject", "code",
        "think", "feedback", "user", "ai-text", "memory-compaction-error", "CANARY_EVENT_TAG"] {
        let detail = format!("CANARY_EVENT_{tag}");
        sender.log(tag, &detail).await;
        // UI/caller receives the complete diagnostic even when the journal does not.
        assert!(rx.recv().await.unwrap().to_string().contains(&detail));
        // Direct file callers must not bypass the same boundary.
        write_log_file(&log_file, tag, &detail).await;
    }
    let contents = tokio::fs::read_to_string(&path).await.unwrap();
    assert_eq!(contents.contains("CANARY_EVENT"), crate::logging::content());
    for category in ["iteration", "done", "done-rejected", "llm-error", "activity"] {
        assert!(contents.contains(category));
    }
}
