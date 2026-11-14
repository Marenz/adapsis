//! Native takeover outcome guard; no live provider, callbacks or attachments.
use super::*;
use crate::api::native_outcome::INCOMPLETE;

fn silent(id: &str) -> Value {
    response("", vec![call(id, "finish", json!({}))])
}

#[tokio::test]
async fn native_outcome_takeover_corrects_finish_only_and_keeps_one_final_answer() {
    let harness = Harness::new(true);
    let provider = provider(vec![
        response(
            "<iteration_budget>1</iteration_budget>",
            vec![call("empty", "finish", json!({}))],
        ),
        response(
            "The original constraints still apply; this answers the follow-up.",
            vec![call("answered", "finish", json!({}))],
        ),
    ])
    .await;
    let reply = harness
        .turn(
            &provider,
            "telegram:outcome-followup",
            ADMIN,
            "What about replacing it midway?",
        )
        .await;
    assert_eq!(
        reply,
        "The original constraints still apply; this answers the follow-up."
    );
    assert_eq!(provider.requests.lock().unwrap().len(), 2);
    let meta = harness.meta.lock().unwrap();
    let first = meta
        .conversations
        .get("telegram:outcome-followup")
        .unwrap()
        .messages
        .iter()
        .find(|message| message.tool_call_id.as_deref() == Some("empty"))
        .unwrap();
    let result: Value = serde_json::from_str(&first.content).unwrap();
    assert_eq!(result["done_accepted"], false);
    assert_eq!(result["success"], false);
}

#[tokio::test]
async fn native_outcome_takeover_repeated_finish_only_returns_incomplete_not_empty_success() {
    let harness = Harness::new(true);
    let provider = provider(vec![
        silent("first"),
        silent("second"),
        response("must not run", vec![]),
    ])
    .await;
    let reply = harness
        .turn(
            &provider,
            "telegram:outcome-repeat",
            ADMIN,
            "A question awaiting an answer",
        )
        .await;
    assert_eq!(reply, INCOMPLETE);
    assert_eq!(provider.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn native_outcome_takeover_provider_failure_after_refusal_keeps_apology() {
    let harness = Harness::new(true);
    let provider = provider(vec![silent("empty"), json!({"http_status":400})]).await;
    let reply = harness
        .turn(
            &provider,
            "telegram:outcome-provider",
            ADMIN,
            "A question awaiting an answer",
        )
        .await;
    assert!(reply.contains("technisches Problem"));
    assert!(!reply.is_empty());
    assert_eq!(provider.requests.lock().unwrap().len(), 2);
}
