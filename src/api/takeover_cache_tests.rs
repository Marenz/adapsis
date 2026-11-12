//! Cold start and cache accounting through the real takeover loop (#68), with
//! the mock provider. The clock is real, so pauses are produced by back-dating
//! the conversation's last request time.
use super::*;

const OPUS: &str = "anthropic/claude-opus-5";

fn now_ms() -> u64 {
    crate::cache_lifetime::now_ms()
}

/// `turns` answered turns of about `bytes` each, as stored history.
fn prefill(harness: &Harness, context: &str, turns: usize, bytes: usize) {
    let mut meta = harness.meta.lock().unwrap();
    let conv = meta.conversations.get_or_create(context);
    for i in 0..turns {
        conv.push_user(format!("old question {i} {}", "u".repeat(bytes / 2)));
        conv.push_assistant(format!("old answer {i} {}", "r".repeat(bytes / 2)));
    }
}

fn conv<T>(harness: &Harness, context: &str, read: impl FnOnce(&mut crate::session::Conversation) -> T) -> T {
    read(harness.meta.lock().unwrap().conversations.get_or_create(context))
}

fn wire_bytes(request: &Value) -> usize {
    request["messages"].as_array().unwrap().iter().map(|m| m.to_string().len()).sum()
}

fn finish(text: &str, id: &str) -> Value {
    response(text, vec![call(id, "finish", json!({}))])
}

/// A turn that starts after a pause over the route's cache lifetime sends a
/// compact history; a turn that starts within it, and every round inside a
/// turn, sends the full append-only history.
#[tokio::test]
async fn takeover_cold_starts_only_at_a_turn_start_after_a_long_pause() {
    let mut harness = Harness::new(true);
    harness.model = OPUS;
    let context = "telegram:group:-88080";
    // ≈600 KB ≈ 240k tokens of history: far under the 1M window, over the 100k
    // cold-start size.
    prefill(&harness, context, 100, 6_000);
    let provider = provider(vec![
        finish("warm answer", "f0"),
        tool("look", "inspect", json!({"kind":"symbols"})),
        finish("second answer", "f1"),
        finish("cold answer", "f2"),
        finish("after the cold start", "f3"),
    ]).await;
    // Turn 1: no earlier request is known (an old session): no cold start.
    assert_eq!(harness.turn(&provider, context, ADMIN, "first").await, "warm answer");
    assert_eq!(conv(&harness, context, |c| c.view_start), 0);
    let first_request = conv(&harness, context, |c| c.last_request_ms).expect("the request time is recorded");
    assert!(now_ms() - first_request < 30_000);
    // Turn 2 right away, with a tool round: two requests, nothing cut.
    assert_eq!(harness.turn(&provider, context, ADMIN, "second").await, "second answer");
    assert_eq!(conv(&harness, context, |c| c.view_start), 0);
    // The conversation's last request was two hours ago (lifetime: one hour).
    conv(&harness, context, |c| c.last_request_ms = Some(now_ms() - 2 * 3_600_000));
    let stored = conv(&harness, context, |c| c.messages.len());
    assert_eq!(harness.turn(&provider, context, ADMIN, "third, after a long pause").await, "cold answer");
    let start = conv(&harness, context, |c| c.view_start);
    assert!(start > 0, "the view was cut at the turn start");
    assert_eq!(conv(&harness, context, |c| c.messages[start].role.clone()), "user");
    assert!(conv(&harness, context, |c| c.messages.len()) > stored, "persisted history only grew");
    // Turn 4 right away: append-only again.
    assert_eq!(harness.turn(&provider, context, ADMIN, "fourth").await, "after the cold start");
    assert_eq!(conv(&harness, context, |c| c.view_start), start);

    let requests = provider.requests.lock().unwrap().clone();
    assert_eq!(requests.len(), 5);
    let sizes: Vec<usize> = requests.iter().map(wire_bytes).collect();
    // Warm requests carry the whole history (≈600 KB); the two rounds of turn 2
    // only append.
    assert!(sizes[0] > 600_000 && sizes[1] > sizes[0] && sizes[2] > sizes[1], "{sizes:?}");
    // The cold request carries at most the cold-start size (100k tokens ≈ 250 KB
    // at the default ratio) plus prompt and tail; the next one appends to it.
    assert!(sizes[3] < 320_000 && sizes[3] > 150_000, "{sizes:?}");
    assert!(sizes[4] > sizes[3] && sizes[4] < 330_000, "{sizes:?}");
    let text = requests[3]["messages"].to_string();
    assert!(text.contains("third, after a long pause") && text.contains("older message(s) are omitted"));
    assert!(text.contains("second answer"), "the newest history is what is kept");
    assert!(!text.contains("old question 0 "), "the oldest history is omitted from the view");
    // History is a byte-stable prefix from the cold request to the next.
    let strip = |m: &Value| { let mut m = m.clone(); m.as_object_mut().unwrap().remove("cache_control"); m.to_string() };
    let (cold, next) = (requests[3]["messages"].as_array().unwrap(), requests[4]["messages"].as_array().unwrap());
    for (a, b) in cold[..cold.len() - 1].iter().zip(next.iter()) {
        assert_eq!(strip(a), strip(b), "history changed after the cold start");
    }
    // Every request is acceptable to the gateway and asks for the 1h lifetime.
    for (index, request) in requests.iter().enumerate() {
        let messages = request["messages"].as_array().unwrap();
        assert_eq!(crate::context_log::gateway_boundary_errors(messages), Vec::<String>::new(), "request {index}");
        assert!(messages.iter().filter_map(|m| m.get("cache_control")).all(|marker| marker["ttl"] == "1h"), "request {index}");
    }
    // The cold request starts a new cache epoch: only the stable prefix and the
    // new history end are boundaries, not a position of the old view.
    let cold_marks = cold.iter().filter(|m| m.get("cache_control").is_some()).count();
    assert_eq!(cold_marks, 2);
}

/// A route without a known cache lifetime never cold-starts, however long the
/// pause; and the cold start can be switched off per model.
#[tokio::test]
async fn takeover_does_not_cold_start_on_a_route_with_an_unknown_lifetime() {
    for model in ["takeover-fixture", "chatgpt/gpt-6-astra"] {
        let mut harness = Harness::new(true);
        harness.model = model;
        let context = "telegram:group:-88081";
        // ≈200 KB ≈ 80k tokens: under the 128k window's trigger with the reserve.
        prefill(&harness, context, 20, 6_000);
        conv(&harness, context, |c| c.last_request_ms = Some(now_ms() - 30 * 24 * 3_600_000));
        let provider = provider(vec![finish("answer", "f0")]).await;
        assert_eq!(harness.turn(&provider, context, ADMIN, "after a month").await, "answer");
        assert_eq!(conv(&harness, context, |c| c.view_start), 0, "{model}");
        let requests = provider.requests.lock().unwrap().clone();
        assert!(requests[0]["messages"].to_string().contains("old question 0 "), "{model}: full history sent");
    }
}

/// The accounting records what the real loop sent: the generating model, the
/// request time, the real runtime tail, and cache evidence per route only when
/// a hit could be expected. A refused legacy turn is no evidence and is not the
/// conversation's last request.
#[tokio::test]
async fn takeover_accounts_requests_and_skips_refused_turns() {
    let mut harness = Harness::new(false);
    harness.model = OPUS;
    let context = "telegram:group:-88082";
    prefill(&harness, context, 20, 6_000); // ≈120 KB ≈ 48k tokens
    let usage = |read: u64, created: u64| json!({"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,
        "cache_read_input_tokens":read,"cache_creation_input_tokens":created});
    let mut first = response("one<code>!done</code>", vec![]);
    first["usage"] = usage(0, 50_000);
    let mut second = response("two<code>!done</code>", vec![]);
    second["usage"] = usage(49_000, 1_000);
    let mut refused = response("", vec![]);
    refused["choices"][0]["finish_reason"] = json!("content_filter");
    refused["usage"] = usage(0, 50_000);
    let mut fourth = response("four<code>!done</code>", vec![]);
    fourth["usage"] = usage(0, 50_000);
    let provider = provider(vec![first, second, refused.clone(), refused, fourth]).await;
    let evidence = |harness: &Harness| harness.meta.lock().unwrap().conversations.cache_lifetimes.evidence(OPUS).cloned();

    harness.turn(&provider, context, ADMIN, "first").await;
    // First request of the process for this context: no hit was expected.
    assert_eq!(evidence(&harness), None);
    let calibration = conv(&harness, context, |c| c.token_calibration.clone()).expect("calibrated");
    assert_eq!((calibration.model.as_str(), calibration.input_tokens), (OPUS, 50_010));
    // The real tail of this turn (authority text and what recall added) was measured.
    let tail = conv(&harness, context, |c| c.tail_bytes).expect("tail measured");
    assert!(tail > 100 && tail < 64 * 1024, "{tail}");
    // Second turn 90 s after the first request: the history only appended and
    // the provider read it from the cache: a hit at that pause.
    conv(&harness, context, |c| c.last_request_ms = Some(now_ms() - 90_000));
    harness.turn(&provider, context, ADMIN, "second").await;
    let recorded = evidence(&harness).expect("a hit was recorded");
    assert_eq!((recorded.hit_count, recorded.miss_count), (1, 0));
    assert!(recorded.longest_hit_secs.is_some_and(|secs| (89..=95).contains(&secs)), "{recorded:?}");
    // Third turn: the provider refuses (and claims a cache write).
    conv(&harness, context, |c| c.last_request_ms = Some(now_ms() - 200_000));
    let before = conv(&harness, context, |c| c.last_request_ms);
    harness.turn(&provider, context, ADMIN, "third").await;
    assert_eq!(evidence(&harness).unwrap(), recorded, "a refused request is neither hit nor miss");
    assert_eq!(conv(&harness, context, |c| c.last_request_ms), before, "and not the last request");
    // Fourth turn: its first request reads nothing (the refused turn wrote
    // nothing), which must not be learned as a miss.
    harness.turn(&provider, context, ADMIN, "fourth").await;
    assert_eq!(evidence(&harness).unwrap(), recorded, "no miss after a refused turn");
    assert!(conv(&harness, context, |c| c.last_request_ms) > before);
}

/// Providers cache the tool schemas ahead of the system prompt. Two speakers who cannot
/// change code share the restricted prompt text, but not the tool list (a guest's
/// profile tools, another guest's, an execute-only speaker's). A turn whose tool list
/// differs from the previous turn's starts a new cache epoch: its unavoidable miss is not
/// lifetime evidence, and no boundary is set at a history position the provider cannot
/// have cached. The same speaker's next turn continues the epoch.
#[tokio::test]
async fn takeover_a_changed_tool_list_is_a_new_cache_epoch_and_no_miss_evidence() {
    const MODEL: &str = "takeover-fixture";
    const VIEWER: &str = "telegram:user:100000778";
    const OTHER_GUEST: &str = "telegram:user:100000781";
    let usage = |read: u64, created: u64| json!({"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,
        "cache_read_input_tokens":read,"cache_creation_input_tokens":created});
    let answer = |id: &str, read: u64| {
        let mut answer = finish("ok", id);
        answer["usage"] = usage(read, 50_000 - read);
        answer
    };
    let tools = |request: &Value| -> Vec<String> {
        request["tools"].as_array().unwrap().iter().map(|tool| tool["function"]["name"].as_str().unwrap().to_string()).collect()
    };
    let marks = |request: &Value| request["messages"].as_array().unwrap().iter().filter(|m| m.get("cache_control").is_some()).count();
    for (label, context, second) in [
        ("guest then execute-only", "telegram:group:-88090", VIEWER),
        ("two guests with different profiles", "telegram:group:-88091", OTHER_GUEST),
    ] {
        let mut harness = Harness::new(true);
        harness.permissions = restricted::permissions();
        // The intake module's generic non-admin profile; `[principals]` narrows a guest further.
        harness.set_profile(context, Some("viewer"));
        prefill(&harness, context, 20, 6_000); // ≈120 KB ≈ 48k tokens
        let evidence = |harness: &Harness| harness.meta.lock().unwrap().conversations.cache_lifetimes.evidence(MODEL).cloned();
        let provider = provider(vec![answer("f0", 0), answer("f1", 0), answer("f2", 49_000)]).await;

        harness.turn(&provider, context, restricted::GUEST, "first").await;
        // 90 s later another speaker of the same authority class, with another tool list:
        // the provider reads nothing from its cache.
        conv(&harness, context, |c| c.last_request_ms = Some(now_ms() - 90_000));
        harness.turn(&provider, context, second, "second").await;
        assert_eq!(evidence(&harness), None, "{label}: the miss after a tool list change is not evidence");
        // The same speaker again: the history only appended and the tool list is the same.
        conv(&harness, context, |c| c.last_request_ms = Some(now_ms() - 90_000));
        harness.turn(&provider, context, second, "third").await;
        let recorded = evidence(&harness).unwrap_or_else(|| panic!("{label}: the hit of an unchanged tool list is evidence"));
        assert_eq!((recorded.hit_count, recorded.miss_count), (1, 0), "{label}");

        let requests = provider.requests.lock().unwrap().clone();
        assert_eq!(requests.len(), 3, "{label}");
        assert_ne!(tools(&requests[0]), tools(&requests[1]), "{label}: the fixture must change the tool list");
        assert_eq!(tools(&requests[1]), tools(&requests[2]), "{label}");
        assert_eq!(requests[0]["messages"][0], requests[1]["messages"][0], "{label}: the stable prefix text is shared");
        // New epoch: stable prefix and history end only. Continued epoch: also the
        // previous turn's history end.
        assert_eq!((marks(&requests[1]), marks(&requests[2])), (2, 3), "{label}");
        for request in &requests {
            assert_eq!(crate::context_log::gateway_boundary_errors(request["messages"].as_array().unwrap()), Vec::<String>::new(), "{label}");
        }
    }
}
