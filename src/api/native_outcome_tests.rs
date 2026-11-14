//! #151 through actual HTTP/SSE handlers and captured provider requests.
use super::*;
use crate::api::native_outcome::{INCOMPLETE, REFUSAL};

fn silent(id: &str, content: &str) -> Value {
    response(content, vec![call(id, "finish", json!({}))])
}

fn result(fixture: &Fixture, id: &str) -> Value {
    let meta = fixture.config.meta.lock().unwrap();
    let message = meta
        .conversations
        .get("main")
        .unwrap()
        .messages
        .iter()
        .find(|message| message.tool_call_id.as_deref() == Some(id))
        .unwrap();
    serde_json::from_str(&message.content).unwrap()
}

fn assert_refusal(fixture: &Fixture, id: &str) {
    let result = result(fixture, id);
    assert_eq!(result["done_accepted"], false);
    assert_eq!(result["success"], false);
    assert_eq!(result["has_errors"], true);
    assert_eq!(result["output"], REFUSAL);
    assert!(
        result["mutation_results"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["success"] == false)
    );
    assert!(!result.to_string().contains("done accepted"));
}

#[tokio::test]
async fn native_outcome_continuation_question_gets_one_correction_not_silent_success() {
    for streaming in [false, true] {
        let mut fixture = Fixture::new(
            vec![
                silent("empty", ""),
                response(
                    "You can replace it only at a supported pause; the task is still waiting.",
                    vec![call("answered", "finish", json!({}))],
                ),
            ],
            "",
        )
        .await;
        let training = tempfile::NamedTempFile::new_in("/tmp/opencode").unwrap();
        fixture.config.training_log = Some(Arc::new(tokio::sync::Mutex::new(
            tokio::fs::File::from_std(training.reopen().unwrap()),
        )));
        let response = fixture
            .run_message(
                streaming,
                "Can it be replaced midway? Keep the original constraints.",
            )
            .await;
        assert_eq!(response.completed, Some(true));
        assert!(response.reply.contains("supported pause"));
        assert!(!response.reply.contains(INCOMPLETE));
        assert_refusal(&fixture, "empty");
        assert_eq!(result(&fixture, "answered")["done_accepted"], true);
        let requests = fixture.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(
            requests[1]
                .to_string()
                .contains("without repeating completed actions")
        );
        drop(requests);
        let records = tokio::fs::read_to_string(training.path()).await.unwrap();
        let first: Value = serde_json::from_str(records.lines().next().unwrap()).unwrap();
        let normalized: Value =
            serde_json::from_str(first["feedback"][0].as_str().unwrap()).unwrap();
        assert_eq!(normalized["done_accepted"], false);
        assert_eq!(normalized["success"], false);
    }
}

#[tokio::test]
async fn native_outcome_repeated_finish_is_incomplete_and_does_not_use_extra_rounds() {
    for streaming in [false, true] {
        let fixture = Fixture::new(
            vec![
                silent("first", ""),
                silent("second", ""),
                response(
                    "must not be requested",
                    vec![call("extra", "finish", json!({}))],
                ),
            ],
            "",
        )
        .await;
        let response = fixture.run(streaming).await;
        assert_eq!(response.completed, Some(false));
        assert!(response.has_errors);
        assert!(response.reply.contains(INCOMPLETE));
        assert_eq!(fixture.requests.lock().unwrap().len(), 2);
        assert_refusal(&fixture, "first");
        assert_refusal(&fixture, "second");
    }
}

#[tokio::test]
async fn native_outcome_budget_one_has_visible_fallback_without_extra_generation() {
    for streaming in [false, true] {
        let fixture = Fixture::new(vec![silent("only", "")], "").await;
        let response = fixture.run(streaming).await;
        assert_eq!(response.completed, Some(false));
        assert!(response.reply.contains(INCOMPLETE));
        assert_eq!(fixture.requests.lock().unwrap().len(), 1);
        assert_refusal(&fixture, "only");
    }
}

#[tokio::test]
async fn native_outcome_inspect_and_action_are_not_answers_and_are_not_replayed() {
    for streaming in [false, true] {
        let source = "+module OutcomeAction\n+fn advance ()->String [io,async]\n  +await sent:Result<String> = http_post(\"https://outcome.invalid/action\", \"synthetic action\", \"text/plain\")\n  +return \"notified\"\n+end\n";
        let mut fixture = Fixture::new(
            vec![
                tool("inspect", "inspect", json!({"kind":"symbols"})),
                tool(
                    "action",
                    "evaluate",
                    json!({"function":"OutcomeAction.advance"}),
                ),
                silent("no-answer", ""),
                response(
                    "The action ran once; the larger task is not complete.",
                    vec![call("reported", "finish", json!({}))],
                ),
            ],
            source,
        )
        .await;
        let (sender, mut receiver) = tokio::sync::mpsc::channel(8);
        fixture.config.io_sender = Some(sender);
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let observed = count.clone();
        let driver = tokio::spawn(async move {
            while let Some(request) = receiver.recv().await {
                let crate::coroutine::IoRequest::HttpPost { url, reply, .. } = request else {
                    panic!("unexpected operation")
                };
                assert_eq!(url, "https://outcome.invalid/action");
                observed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let _ = reply.send(Ok("{}".into()));
            }
        });
        let response = fixture.run(streaming).await;
        assert_eq!(response.completed, Some(true), "{}", response.reply);
        assert_refusal(&fixture, "no-answer");
        assert_eq!(result(&fixture, "action")["success"], true);
        assert!(
            result(&fixture, "action")["eval_results"]
                .as_str()
                .unwrap()
                .contains("notified")
        );
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 1);
        let requests = fixture.requests.lock().unwrap();
        assert_eq!(requests.len(), 4);
        driver.abort();
    }
}

#[tokio::test]
async fn native_outcome_whitespace_hidden_reasoning_and_code_only_do_not_satisfy_finish() {
    for content in [
        "   \n\t",
        "<think>hidden</think><iteration_budget>1</iteration_budget>",
        "<code>+module OutcomeCodeOnly\n+fn ignored ()->Int\n  +return 1\n+end</code>",
    ] {
        let mut first = silent("hidden", content);
        first["choices"][0]["message"]["reasoning_content"] = json!("PRIVATE_THINKING");
        let fixture = Fixture::new(vec![first, silent("still-empty", "")], "").await;
        let response = fixture.run(false).await;
        assert_eq!(response.completed, Some(false));
        assert!(response.reply.contains(INCOMPLETE));
        assert_refusal(&fixture, "hidden");
        assert!(
            !fixture
                .config
                .program
                .read()
                .await
                .modules
                .iter()
                .any(|module| module.name == "OutcomeCodeOnly")
        );
    }
}

#[tokio::test]
async fn native_outcome_provider_error_after_refusal_stays_failed_and_visible() {
    for streaming in [false, true] {
        let fixture = Fixture::new(vec![silent("empty", ""), json!({"http_status":400})], "").await;
        let response = fixture.run(streaming).await;
        assert_eq!(response.completed, Some(false));
        assert!(response.has_errors);
        assert!(response.reply.contains("provider error"));
        assert_eq!(fixture.requests.lock().unwrap().len(), 2);
        assert_refusal(&fixture, "empty");
    }
}

#[tokio::test]
async fn native_outcome_normal_answer_keeps_one_reply_and_one_generation() {
    for streaming in [false, true] {
        let fixture = Fixture::new(
            vec![response(
                "An ordinary answer.",
                vec![call("done", "finish", json!({}))],
            )],
            "",
        )
        .await;
        let response = fixture.run(streaming).await;
        assert_eq!(response.completed, Some(true));
        assert!(!response.has_errors);
        if streaming {
            let finals = response
                .reply
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .filter_map(|line| serde_json::from_str::<Value>(line.trim()).ok())
                .filter(|event| event["type"] == "text" && event["text"] == "An ordinary answer.")
                .count();
            assert_eq!(
                finals, 1,
                "one final text event, separate from streaming deltas"
            );
        } else {
            assert_eq!(response.reply.matches("An ordinary answer.").count(), 1);
        }
        assert_eq!(fixture.requests.lock().unwrap().len(), 1);
    }
}
