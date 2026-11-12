//! Offline tests for request accounting, tail calibration and the cold start.
//! No provider is called: usage reports are constructed.
use super::*;
use crate::api::http_context::checkpointed_view_at;
use crate::cache_lifetime::{Lifetime, Source};
use crate::llm::{ChatMessage, LlmUsage, ToolCall};
use crate::model_limits::DEFAULT_BYTES_PER_TOKEN;
use crate::session::{Conversation, SharedMeta};

/// 1M window, 1-hour cache lifetime requested (default lifetime 3600 s), cold
/// view 100k tokens.
const OPUS: &str = "anthropic/claude-opus-5";
/// The default lifetime of `OPUS` in seconds.
const HOUR: u64 = 3_600;
const T0: u64 = 1_800_000_000_000;

fn secs(seconds: u64) -> u64 {
    seconds * 1_000
}

fn new_meta() -> SharedMeta {
    std::sync::Arc::new(std::sync::Mutex::new(crate::session::SessionMeta::new()))
}

fn usage(prompt: u64, read: Option<u64>, created: Option<u64>) -> LlmUsage {
    LlmUsage { prompt_tokens: prompt, cache_read_input_tokens: read, cache_creation_input_tokens: created, ..Default::default() }
}

/// `turns` old turns of about `bytes` each (a user message and an answer).
fn fill(meta: &SharedMeta, context: &str, turns: usize, bytes: usize) {
    let mut guard = meta.lock().unwrap();
    let conv = guard.conversations.get_or_create(context);
    for i in 0..turns {
        conv.push_user(format!("q{i} {}", "u".repeat(bytes / 2)));
        conv.push_assistant(format!("a{i} {}", "r".repeat(bytes / 2)));
    }
}

/// Start a turn: the user message, the turn's view, and the cache expectation
/// the epoch tracker would report (the history only appended unless this is the
/// conversation's first request or the view was just cut).
fn start_turn(meta: &SharedMeta, context: &str, model: &str, now_ms: u64) -> View {
    start_turn_with(meta, context, model, now_ms, "next question".to_string())
}

fn start_turn_with(meta: &SharedMeta, context: &str, model: &str, now_ms: u64, message: String) -> View {
    // History length of each conversation's previous turn start, as the epoch
    // tracker would report it (`Epoch::Continued { previous_len }`).
    static PREVIOUS: std::sync::Mutex<Vec<(usize, String, usize)>> = std::sync::Mutex::new(Vec::new());
    let key = (std::sync::Arc::as_ptr(meta) as usize, context.to_string());
    let mut guard = meta.lock().unwrap();
    guard.conversations.get_or_create(context).push_user(message);
    let view = checkpointed_view_at(&mut guard.conversations, context, model, "s", 0, now_ms);
    let history = &view.messages[1..view.messages.len() - 1];
    let mut previous = PREVIOUS.lock().unwrap();
    let slot = previous.iter().position(|(meta, context, _)| (*meta, context) == (key.0, &key.1));
    let previous_len = slot.map(|slot| previous[slot].2).filter(|_| !view.checkpointed);
    expect_cached_prefix(&mut guard.conversations, context, "s", history, previous_len);
    match slot {
        Some(slot) => previous[slot].2 = history.len(),
        None => previous.push((key.0, key.1, history.len())),
    }
    view
}

fn sent(model: &str, text_bytes: usize, now_ms: u64) -> SentRequest {
    SentRequest { model: model.into(), text_bytes, images: 0, started_ms: now_ms, same_tools: true }
}

/// Answer one request of the turn, sent to `model` at `now_ms`.
fn answer(meta: &SharedMeta, context: &str, model: &str, text_bytes: usize, now_ms: u64, usage: &LlmUsage) {
    sent(model, text_bytes, now_ms).calibrate(meta, context, false, false, usage);
}

fn with_conv<T>(meta: &SharedMeta, context: &str, read: impl FnOnce(&mut Conversation) -> T) -> T {
    read(meta.lock().unwrap().conversations.get_or_create(context))
}

fn view_start(meta: &SharedMeta, context: &str) -> usize {
    with_conv(meta, context, |conv| conv.view_start)
}

/// Projected tokens of the view's history (everything after the system prefix).
fn history_tokens(view: &View, ratio: f32) -> usize {
    crate::model_limits::tokens_for_bytes(text_bytes(&view.messages[1..]), ratio)
}

fn lifetime(meta: &SharedMeta, model: &str) -> Option<Lifetime> {
    meta.lock().unwrap().conversations.cache_lifetimes.lifetime(model, &crate::model_limits::for_model(model), T0)
}

#[test]
fn calibration_measures_sent_text_bytes_without_image_data() {
    let text = vec![ChatMessage::system("s".repeat(1_000)), ChatMessage::user("u".repeat(2_000))];
    let plain = SentRequest::measure("route/m", &text, 0);
    assert_eq!(plain.model, "route/m");
    let wire: usize = text.iter().map(|m| serde_json::to_string(m).unwrap().len()).sum();
    assert_eq!((plain.text_bytes, plain.images), (wire, 0));
    assert!(plain.started_ms > 1_700_000_000_000, "sent time is recorded");
    assert_eq!(SentRequest::measure("route/m", &text, 500).text_bytes, wire + 500, "tool schemas are part of the request");
    // The same request with a 300 KB image: the base64 data is not text.
    let photo = crate::attachment::Attachment::from_bytes(vec![7u8; 300_000], "image/png", "p.png").unwrap();
    let with_image = vec![text[0].clone(), ChatMessage::user_with_attachments("u".repeat(2_000), &[photo])];
    let measured = SentRequest::measure("route/m", &with_image, 0);
    assert_eq!(measured.images, 1);
    assert!(measured.text_bytes < wire + 200, "image bytes excluded: {} vs {wire}", measured.text_bytes);
}

#[test]
fn calibration_from_provider_usage_drives_the_next_view() {
    let meta = new_meta();
    let ratio = |meta: &SharedMeta, model: &str| with_conv(meta, "ctx", |conv| conv.bytes_per_token(model));
    let request = || sent(UNKNOWN_BIG, 1_460_000, T0);
    assert_eq!(ratio(&meta, UNKNOWN_BIG), DEFAULT_BYTES_PER_TOKEN);
    // Anthropic shape: the total processed input is prompt + read + created.
    request().calibrate(&meta, "ctx", false, false, &usage(3_447, Some(0), Some(590_000)));
    assert!((ratio(&meta, UNKNOWN_BIG) - 2.46).abs() < 0.01, "{}", ratio(&meta, UNKNOWN_BIG));
    // ChatGPT shape: cache reads are inside prompt_tokens.
    request().calibrate(&meta, "ctx", false, false, &usage(365_000, Some(300_000), None));
    assert_eq!(ratio(&meta, UNKNOWN_BIG), 4.0);
    // Missing/zero usage and a request the client shrank leave it unchanged.
    request().calibrate(&meta, "ctx", false, false, &LlmUsage::default());
    request().calibrate(&meta, "ctx", true, false, &usage(1_000_000, None, None));
    assert_eq!(ratio(&meta, UNKNOWN_BIG), 4.0);
    // The ratio belongs to the model that counted it.
    assert_eq!(ratio(&meta, OPUS), DEFAULT_BYTES_PER_TOKEN);
    // Implausible reports are clamped.
    request().calibrate(&meta, "ctx", false, false, &usage(5_000_000, None, None));
    assert_eq!(ratio(&meta, UNKNOWN_BIG), crate::model_limits::MIN_BYTES_PER_TOKEN);
    // The stored ratio is the one the next view projects with: ≈1.2 MB is under
    // the trigger at the default ratio and far over it at 1.5 bytes/token.
    fill(&meta, "ctx", 200, 6_000);
    fill(&meta, "fresh", 200, 6_000);
    assert!(!start_turn(&meta, "fresh", UNKNOWN_BIG, T0).checkpointed);
    assert!(start_turn(&meta, "ctx", UNKNOWN_BIG, T0).checkpointed);
    // Each context keeps its own measurement.
    assert_eq!(with_conv(&meta, "other", |conv| conv.bytes_per_token(UNKNOWN_BIG)), DEFAULT_BYTES_PER_TOKEN);
}

/// A model with a large window and NO sourced cache lifetime (the ChatGPT route
/// reports no cache use at all).
const UNKNOWN_BIG: &str = "chatgpt/gpt-6-astra";

/// The cold start: only at the start of a turn, only after a pause of at least
/// the route's lifetime, and never between the rounds of a running turn.
#[test]
fn cold_start_fires_only_at_a_turn_start_after_a_pause_over_the_lifetime() {
    let meta = new_meta();
    fill(&meta, "ctx", 200, 6_000); // ≈1.2 MB ≈ 490k tokens of history
    // No earlier request is recorded (an old session): behave as before.
    let view = start_turn(&meta, "ctx", OPUS, T0);
    assert!(!view.checkpointed);
    assert_eq!(view_start(&meta, "ctx"), 0);
    let bytes = text_bytes(&view.messages) + 30_000;
    answer(&meta, "ctx", OPUS, bytes, T0, &usage(10, Some(0), Some(480_000)));
    // The same turn runs for hours: a tool call returns after 2 h, another after
    // 4 h, both far over the lifetime. Rounds never rebuild the view, and
    // accounting never moves it.
    for (round, at) in [(1, T0 + secs(2 * HOUR)), (2, T0 + secs(4 * HOUR))] {
        with_conv(&meta, "ctx", |conv| {
            conv.push_native_batch(String::new(), vec![ToolCall { id: format!("c{round}"), name: "evaluate".into(), arguments: "{}".into() }], vec!["ok".into()])
        });
        answer(&meta, "ctx", OPUS, bytes + round * 200, at, &usage(10, Some(480_000), Some(100)));
        assert_eq!(view_start(&meta, "ctx"), 0, "round {round}: a pause inside a turn costs no context");
    }
    with_conv(&meta, "ctx", |conv| conv.push_assistant("done"));
    // The mid-turn hits after two hours have taught the route that its cache
    // lives that long.
    assert_eq!(lifetime(&meta, OPUS), Some(Lifetime { secs: 2 * HOUR + 1, source: Source::Learned }));
    // The next turn starts 100 s after the turn's LAST request: under the
    // lifetime, however long the turn itself ran.
    let second = T0 + secs(4 * HOUR + 100);
    assert!(!start_turn(&meta, "ctx", OPUS, second).checkpointed);
    assert_eq!(view_start(&meta, "ctx"), 0);
    answer(&meta, "ctx", OPUS, bytes, second, &usage(10, Some(480_000), Some(100)));
    with_conv(&meta, "ctx", |conv| conv.push_assistant("ok"));
    // A turn that starts after a pause over the lifetime cold-starts once.
    let before = with_conv(&meta, "ctx", |conv| conv.messages.len());
    let third = second + secs(2 * HOUR + 100);
    let cold = start_turn(&meta, "ctx", OPUS, third);
    assert!(cold.checkpointed, "reported as a checkpoint, so the cache epoch resets");
    let start = view_start(&meta, "ctx");
    assert!(start > 0);
    let kept = history_tokens(&cold, with_conv(&meta, "ctx", |conv| conv.bytes_per_token(OPUS)));
    assert!(kept <= 100_000 && kept > 90_000, "cut to the cold-start size in one step: {kept}");
    assert_eq!(with_conv(&meta, "ctx", |conv| conv.messages.len()), before + 1, "persisted history untouched");
    assert_eq!(with_conv(&meta, "ctx", |conv| conv.messages[start].role.clone()), "user", "view starts at a user turn");
    let text = serde_json::to_string(&cold.messages).unwrap();
    assert!(text.contains("next question") && text.contains("older message(s) are omitted"));
    // Then it grows append-only again: the next turn, seconds later, only appends.
    answer(&meta, "ctx", OPUS, text_bytes(&cold.messages), third, &usage(1_000, Some(0), Some(100_000)));
    with_conv(&meta, "ctx", |conv| conv.push_assistant("after the cold start"));
    let next = start_turn(&meta, "ctx", OPUS, third + secs(30));
    assert!(!next.checkpointed);
    assert_eq!(view_start(&meta, "ctx"), start);
    let wire = |view: &View| view.messages.iter().map(|m| serde_json::to_string(m).unwrap()).collect::<Vec<_>>();
    let (old, new) = (wire(&cold), wire(&next));
    assert_eq!(&new[..old.len()], &old[..], "byte-stable prefix after the cold start");
}

#[test]
fn cold_start_never_happens_with_an_unknown_lifetime_or_a_small_history() {
    let meta = new_meta();
    let week = secs(7 * 24 * HOUR);
    fill(&meta, "ctx", 200, 6_000);
    assert_eq!(lifetime(&meta, UNKNOWN_BIG), None);
    start_turn(&meta, "ctx", UNKNOWN_BIG, T0);
    answer(&meta, "ctx", UNKNOWN_BIG, 1_000_000, T0, &usage(400_000, None, None));
    // A week later, on a route with no default and nothing learned: as today.
    assert!(!start_turn(&meta, "ctx", UNKNOWN_BIG, T0 + week).checkpointed);
    assert_eq!(view_start(&meta, "ctx"), 0);
    // Hits alone are only a lower bound: still unknown, still no cold start.
    meta.lock().unwrap().conversations.cache_lifetimes.record(UNKNOWN_BIG, Outcome::Hit, 900, T0);
    assert!(!start_turn(&meta, "ctx", UNKNOWN_BIG, T0 + 2 * week).checkpointed);
    // A history already under the cold-start size is left alone on any route.
    fill(&meta, "small", 10, 6_000);
    start_turn(&meta, "small", OPUS, T0);
    answer(&meta, "small", OPUS, 70_000, T0, &usage(28_000, Some(0), Some(1)));
    assert!(!start_turn(&meta, "small", OPUS, T0 + week).checkpointed);
    assert_eq!(view_start(&meta, "small"), 0);
}

/// A cold start never keeps more than a budget checkpoint would: on a small
/// window the cold-start size is bounded by the checkpoint target.
#[test]
fn cold_start_size_is_bounded_by_the_checkpoint_target_on_small_windows() {
    // 128k window (unknown model limits), sourced lifetime 300 s.
    const SMALL: &str = "moonshot/kimi";
    let limits = crate::model_limits::for_model(SMALL);
    assert_eq!((limits.context_tokens, limits.cache_ttl_secs, limits.cold_start_tokens), (128_000, Some(300), 100_000));
    let meta = new_meta();
    fill(&meta, "ctx", 16, 6_000); // ≈98 KB ≈ 39k tokens: under the 85% trigger
    assert!(!start_turn(&meta, "ctx", SMALL, T0).checkpointed);
    answer(&meta, "ctx", SMALL, 100_000, T0, &usage(40_000, None, None));
    with_conv(&meta, "ctx", |conv| conv.push_assistant("ok"));
    let cold = start_turn(&meta, "ctx", SMALL, T0 + secs(400));
    assert!(cold.checkpointed);
    // Target 64k minus the 32k output reserve, system and measured tail: at most
    // 32k tokens of history, not the configured 100k (most of this window).
    let kept = history_tokens(&cold, with_conv(&meta, "ctx", |conv| conv.bytes_per_token(SMALL)));
    assert!(kept <= 32_000 && kept > 20_000, "{kept}");
}

#[test]
fn learned_lifetime_overrides_the_default_in_both_directions() {
    // Longer: a hit after 2.5 hours on a route whose default is one hour.
    let longer = new_meta();
    fill(&longer, "ctx", 200, 6_000);
    start_turn(&longer, "ctx", OPUS, T0);
    answer(&longer, "ctx", OPUS, 1_000_000, T0, &usage(400_000, Some(0), Some(1)));
    longer.lock().unwrap().conversations.cache_lifetimes.record(OPUS, Outcome::Hit, 9_000, T0);
    let second = T0 + secs(8_000);
    assert!(!start_turn(&longer, "ctx", OPUS, second).checkpointed, "8,000 s is within the learned lifetime (the default would cold-start)");
    answer(&longer, "ctx", OPUS, 1_000_000, second, &usage(10_000, Some(390_000), Some(1)));
    assert!(start_turn(&longer, "ctx", OPUS, second + secs(9_500)).checkpointed, "9,500 s is past it");
    // Shorter: two misses after about 20 minutes on the same default.
    let shorter = new_meta();
    fill(&shorter, "ctx", 200, 6_000);
    start_turn(&shorter, "ctx", OPUS, T0);
    answer(&shorter, "ctx", OPUS, 1_000_000, T0, &usage(400_000, Some(0), Some(1)));
    {
        let routes = &mut shorter.lock().unwrap().conversations.cache_lifetimes;
        routes.record(OPUS, Outcome::Miss, 1_200, T0);
        assert_eq!(routes.lifetime(OPUS, &crate::model_limits::for_model(OPUS), T0).unwrap().source, Source::Default, "one miss is not believed");
        routes.record(OPUS, Outcome::Miss, 1_500, T0);
    }
    assert_eq!(lifetime(&shorter, OPUS), Some(Lifetime { secs: 1_500, source: Source::Learned }));
    assert!(start_turn(&shorter, "ctx", OPUS, T0 + secs(2_000)).checkpointed, "2,000 s is under the default but past the learned lifetime");
    // Learned on a route without a default: two misses make it known.
    let unknown = new_meta();
    fill(&unknown, "ctx", 200, 6_000);
    start_turn(&unknown, "ctx", UNKNOWN_BIG, T0);
    answer(&unknown, "ctx", UNKNOWN_BIG, 1_000_000, T0, &usage(400_000, Some(0), None));
    for pause in [1_800, 2_000] {
        unknown.lock().unwrap().conversations.cache_lifetimes.record(UNKNOWN_BIG, Outcome::Miss, pause, T0);
    }
    assert!(!start_turn(&unknown, "ctx", UNKNOWN_BIG, T0 + secs(1_900)).checkpointed);
    answer(&unknown, "ctx", UNKNOWN_BIG, 1_000_000, T0 + secs(1_900), &usage(400_000, None, None));
    assert!(start_turn(&unknown, "ctx", UNKNOWN_BIG, T0 + secs(1_900 + 2_100)).checkpointed);
    // Cold starts switched off for a model (`cache_ttl_secs = 0`): never.
    let mut off = crate::model_limits::for_model(OPUS);
    off.cold_start = false;
    assert_eq!(shorter.lock().unwrap().conversations.cache_lifetimes.lifetime(OPUS, &off, T0), None);
}

/// Evidence comes from answered requests: only where a hit could be expected,
/// only on routes that report cache reads, never from a refused, shrunk or
/// failed request.
#[test]
fn cache_evidence_is_recorded_from_usage_only_when_a_hit_could_be_expected() {
    let meta = new_meta();
    let evidence = |meta: &SharedMeta, model: &str| meta.lock().unwrap().conversations.cache_lifetimes.evidence(model).cloned();
    // 20 answered turns of 6 KB: the cacheable prefix is ≈120 KB ≈ 48k tokens.
    let history: Vec<ChatMessage> = (0..20)
        .flat_map(|_| [ChatMessage::user("x".repeat(3_000)), ChatMessage::assistant("y".repeat(3_000))])
        .collect();
    // A new turn whose first request may expect the first `previous_len` history
    // messages in the cache, if the previous turn wrote them.
    let turn = |meta: &SharedMeta, context: &str, previous_wrote: bool, previous_len: Option<usize>| {
        let mut guard = meta.lock().unwrap();
        guard.conversations.get_or_create(context).turn_probe = Some(TurnProbe { previous_wrote, ..Default::default() });
        expect_cached_prefix(&mut guard.conversations, context, "s", &history, previous_len);
    };
    // Turn 1, first request after a restart (epoch reset): nothing expected.
    turn(&meta, "ctx", false, None);
    answer(&meta, "ctx", OPUS, 150_000, T0, &usage(10, Some(0), Some(60_000)));
    assert_eq!(evidence(&meta, OPUS), None, "a first request never counts as a miss");
    // Round 2 of the same turn, 4 minutes later: the first round's prefix hits.
    answer(&meta, "ctx", OPUS, 151_000, T0 + secs(240), &usage(10, Some(48_000), Some(12_000)));
    assert_eq!(evidence(&meta, OPUS).unwrap().longest_hit_secs, Some(240));
    // Turn 2 after 80 minutes with history that only appended: a real miss.
    let second = T0 + secs(240 + 4_800);
    turn(&meta, "ctx", true, Some(40));
    answer(&meta, "ctx", OPUS, 152_000, second, &usage(10, Some(0), Some(60_000)));
    let recorded = evidence(&meta, OPUS).unwrap();
    assert_eq!(recorded.misses.iter().map(|m| m.pause_secs).collect::<Vec<_>>(), vec![4_800]);
    assert_eq!((recorded.hit_count, recorded.miss_count), (1, 1));
    // A partial read is ambiguous: no record.
    answer(&meta, "ctx", OPUS, 152_000, second + secs(3_000), &usage(44_000, Some(15_000), Some(1_000)));
    assert_eq!(evidence(&meta, OPUS).unwrap(), recorded);
    // A request the client had to shrink is not comparable, and nothing later in
    // that turn can expect to read what it wrote.
    sent(OPUS, 152_000, second + secs(5_000)).calibrate(&meta, "ctx", true, false, &usage(10, Some(0), Some(60_000)));
    answer(&meta, "ctx", OPUS, 152_000, second + secs(9_000), &usage(10, Some(0), Some(60_000)));
    assert_eq!(evidence(&meta, OPUS).unwrap(), recorded, "nothing after the reduced request either");

    // A refused turn (MEASURED: it reports a cache write the next request cannot
    // read): no evidence, not the conversation's last request, and the NEXT turn
    // does not expect a hit from it.
    let third = second + secs(20_000);
    turn(&meta, "ctx", true, Some(40));
    let before = with_conv(&meta, "ctx", |conv| conv.last_request_ms);
    sent(OPUS, 152_000, third).calibrate(&meta, "ctx", false, true, &usage(10, Some(0), Some(60_000)));
    assert_eq!(evidence(&meta, OPUS).unwrap(), recorded, "a refused request is not a miss");
    assert_eq!(with_conv(&meta, "ctx", |conv| conv.last_request_ms), before, "a refused request does not refresh the pause");
    assert_eq!(with_conv(&meta, "ctx", |conv| conv.turn_probe.clone().unwrap().wrote_prefix), false);
    with_conv(&meta, "ctx", |conv| conv.push_user("after the refusal"));
    {
        let mut guard = meta.lock().unwrap();
        checkpointed_view_at(&mut guard.conversations, "ctx", OPUS, "s", 0, third + secs(60));
        expect_cached_prefix(&mut guard.conversations, "ctx", "s", &history, Some(40));
    }
    let probe = with_conv(&meta, "ctx", |conv| conv.turn_probe.clone().unwrap());
    assert_eq!((probe.previous_wrote, probe.first_prefix_bytes), (false, None), "the refused turn did not warm the cache");
    assert!(probe.turn_prefix_bytes.is_some_and(|bytes| bytes > 100_000), "later rounds of this turn can read its own write");
    answer(&meta, "ctx", OPUS, 152_000, third + secs(60), &usage(10, Some(0), Some(60_000)));
    assert_eq!(evidence(&meta, OPUS).unwrap(), recorded, "so the write after a refusal is not a miss");
    // The same holds after a FAILED turn (no answer is ever accounted): the probe
    // of a turn without an answered first request says it wrote nothing.
    turn(&meta, "ctx", true, Some(40));
    with_conv(&meta, "ctx", |conv| conv.push_user("after the failure"));
    meta.lock().map(|mut guard| checkpointed_view_at(&mut guard.conversations, "ctx", OPUS, "s", 0, third + secs(120))).unwrap();
    assert!(!with_conv(&meta, "ctx", |conv| conv.turn_probe.clone().unwrap().previous_wrote));

    // The expected prefix ends at a cache boundary (an assistant message): user
    // messages after it are not part of what the previous request wrote.
    let mut trailing = history.clone();
    trailing.extend((0..5).map(|_| ChatMessage::user("z".repeat(50_000))));
    {
        let mut guard = meta.lock().unwrap();
        guard.conversations.get_or_create("ctx").turn_probe = Some(TurnProbe { previous_wrote: true, ..Default::default() });
        expect_cached_prefix(&mut guard.conversations, "ctx", "s", &trailing, Some(45));
    }
    let probe = with_conv(&meta, "ctx", |conv| conv.turn_probe.clone().unwrap());
    assert_eq!(probe.first_prefix_bytes, Some(3 + text_bytes(&history)));
    assert_eq!(probe.turn_prefix_bytes, probe.first_prefix_bytes);

    // A route that does not report cache reads is never learned.
    turn(&meta, "plain", true, Some(40));
    for step in 0..4 {
        answer(&meta, "plain", UNKNOWN_BIG, 150_000, T0 + step * secs(2_000), &usage(60_000, None, None));
    }
    assert_eq!(evidence(&meta, UNKNOWN_BIG), None);
    assert_eq!(lifetime(&meta, UNKNOWN_BIG), None, "stays unknown");
    // The pause is measured per context, from request start to request start.
    assert_eq!(with_conv(&meta, "plain", |conv| conv.last_request_ms), Some(T0 + secs(6_000)));
}

/// The real runtime tail (first request of a turn minus the turn's view)
/// replaces the fixed allowance in the next projection.
#[test]
fn measured_runtime_tail_replaces_the_fixed_allowance() {
    use crate::session::conversation_view::Projection;
    let policy = |tail: Option<usize>, known: usize| {
        ViewPolicy::for_model(OPUS, "s", known, Projection { bytes_per_token: 2.5, tail_bytes: tail })
    };
    let base = policy(None, 0);
    // No measurement: the fixed allowance, plus whatever is known.
    assert_eq!(policy(None, 10_000).overhead_tokens, base.overhead_tokens + 4_000);
    assert_eq!(policy(None, 10_000).overhead_bytes, base.overhead_bytes + 10_000);
    // Measured 200 KB (with a quarter of margin = 250 KB = 100k tokens) instead
    // of the 16k-token / 64 KiB allowance.
    let measured = policy(Some(200_000), 10_000);
    assert_eq!(measured.overhead_tokens, base.overhead_tokens - ViewPolicy::TAIL_ALLOWANCE_TOKENS + 100_000);
    assert_eq!(measured.overhead_bytes, base.overhead_bytes - ViewPolicy::TAIL_ALLOWANCE_BYTES + 250_000);
    // A small real tail frees room the allowance used to reserve.
    assert!(policy(Some(8_000), 0).overhead_tokens < base.overhead_tokens);
    // The known part of this turn wins when it is larger than last turn's tail.
    assert_eq!(policy(Some(8_000), 50_000).overhead_bytes, base.overhead_bytes - ViewPolicy::TAIL_ALLOWANCE_BYTES + 50_000);
    // An absurd measurement cannot consume the route (capped at 1/4 of it).
    let limit = crate::model_limits::for_model(OPUS).max_request_bytes;
    assert!(policy(Some(usize::MAX / 8), 0).overhead_bytes <= limit / 4 + limit / 16 + 16);

    // Recorded from the first answered request of a turn only.
    let meta = new_meta();
    fill(&meta, "ctx", 10, 6_000);
    let view = start_turn(&meta, "ctx", OPUS, T0);
    let view_bytes = text_bytes(&view.messages);
    assert_eq!(with_conv(&meta, "ctx", |conv| conv.tail_bytes), None);
    answer(&meta, "ctx", OPUS, view_bytes + 45_000, T0, &usage(40_000, Some(0), Some(1)));
    assert_eq!(with_conv(&meta, "ctx", |conv| conv.tail_bytes), Some(45_000));
    // Later rounds carry the turn's own messages, not tail: not measured.
    answer(&meta, "ctx", OPUS, view_bytes + 300_000, T0 + 1_000, &usage(140_000, Some(1), Some(1)));
    assert_eq!(with_conv(&meta, "ctx", |conv| conv.tail_bytes), Some(45_000));
    // The next turn projects with it, and measures again.
    assert_eq!(with_conv(&meta, "ctx", |conv| conv.projection(OPUS).tail_bytes), Some(45_000));
    // The tail is a size in bytes, so it also holds for another model.
    assert_eq!(with_conv(&meta, "ctx", |conv| conv.projection(UNKNOWN_BIG)), Projection { bytes_per_token: DEFAULT_BYTES_PER_TOKEN, tail_bytes: Some(45_000) });
    let next = start_turn(&meta, "ctx", OPUS, T0 + 2_000);
    answer(&meta, "ctx", OPUS, text_bytes(&next.messages) + 12_000, T0 + 2_000, &usage(40_000, Some(1), Some(1)));
    assert_eq!(with_conv(&meta, "ctx", |conv| conv.tail_bytes), Some(12_000));
    // A reduced request says nothing about the tail.
    let third = start_turn(&meta, "ctx", OPUS, T0 + 3_000);
    sent(OPUS, text_bytes(&third.messages) + 90_000, T0 + 3_000).calibrate(&meta, "ctx", true, false, &usage(40_000, None, None));
    assert_eq!(with_conv(&meta, "ctx", |conv| conv.tail_bytes), Some(12_000));
    // Nor does a refused one.
    let fourth = start_turn(&meta, "ctx", OPUS, T0 + 4_000);
    sent(OPUS, text_bytes(&fourth.messages) + 70_000, T0 + 4_000).calibrate(&meta, "ctx", false, true, &usage(40_000, None, None));
    assert_eq!(with_conv(&meta, "ctx", |conv| conv.tail_bytes), Some(12_000));
}

#[test]
fn cold_start_keeps_batches_whole_and_the_newest_turn() {
    let meta = new_meta();
    {
        let mut guard = meta.lock().unwrap();
        let conv = guard.conversations.get_or_create("ctx");
        // Turns of one user message and ten 40 KB native batches (≈160k tokens).
        for turn in 0..6 {
            conv.push_user(format!("task {turn}"));
            for call in 0..10 {
                conv.push_native_batch(String::new(),
                    vec![ToolCall { id: format!("t{turn}c{call}"), name: "inspect".into(), arguments: "{}".into() }],
                    vec!["r".repeat(40_000)]);
            }
            conv.push_assistant(format!("done {turn}"));
        }
        conv.last_request_ms = Some(T0);
    }
    let view = start_turn(&meta, "ctx", OPUS, T0 + secs(HOUR + 60));
    assert!(view.checkpointed);
    let start = view_start(&meta, "ctx");
    // No whole older turn fits 100k tokens, so (as at a budget checkpoint) the
    // view starts at a batch boundary inside the previous turn: its newest work.
    with_conv(&meta, "ctx", |conv| {
        let first = &conv.messages[start];
        assert!(first.role == "assistant" && first.tool_calls.is_some(), "starts with a whole batch");
        assert_eq!(conv.messages[start + 1].role, "tool");
        assert!(start > 5 * 22, "inside the newest of the old turns: {start}");
    });
    let kept = history_tokens(&view, DEFAULT_BYTES_PER_TOKEN);
    assert!(kept <= 100_000 && kept > 80_000, "{kept}");
    // Every tool result in the view follows its call; no orphan at the cut.
    for (i, message) in view.messages.iter().enumerate() {
        if message.role == crate::llm::ChatRole::Tool {
            assert!(!view.messages[i - 1].tool_calls.is_empty(), "orphan tool result at {i}");
        }
    }
    assert_eq!(view.messages.last().unwrap().role, crate::llm::ChatRole::User);
    // A newest turn larger than the cold-start size is still sent whole.
    let big = new_meta();
    with_conv(&big, "ctx", |conv| {
        conv.push_user("old");
        conv.push_assistant("r".repeat(300_000));
        conv.push_user("u".repeat(300_000));
        conv.last_request_ms = Some(T0);
    });
    let mut guard = big.lock().unwrap();
    let view = checkpointed_view_at(&mut guard.conversations, "ctx", OPUS, "s", 0, T0 + secs(HOUR + 60));
    assert!(view.checkpointed);
    assert_eq!(guard.conversations.get("ctx").unwrap().view_start, 3);
    assert!(text_bytes(&view.messages) > 300_000, "the newest turn is never cut away");
}

#[test]
fn cold_start_state_persists_and_old_sessions_load() {
    let meta = new_meta();
    fill(&meta, "ctx", 4, 6_000);
    let view = start_turn(&meta, "ctx", OPUS, T0);
    answer(&meta, "ctx", OPUS, text_bytes(&view.messages) + 9_000, T0, &usage(13_300, Some(0), Some(1)));
    {
        let routes = &mut meta.lock().unwrap().conversations.cache_lifetimes;
        routes.record(OPUS, Outcome::Hit, 5_000, T0);
        routes.record(OPUS, Outcome::Miss, 9_000, T0);
    }
    let json = serde_json::to_string(&meta.lock().unwrap().conversations).unwrap();
    assert!(json.contains(r#""cache_lifetimes":{"anthropic/claude-opus-5":{"longest_hit_secs":5000"#), "{json}");
    assert!(json.contains(&format!(r#""last_request_ms":{T0}"#)) && json.contains(r#""tail_bytes":9000"#), "{json}");
    assert!(!json.contains("turn_probe"), "turn accounting is in-memory only");
    let restored: crate::session::ConversationManager = serde_json::from_str(&json).unwrap();
    let conv = restored.get("ctx").unwrap();
    assert_eq!((conv.last_request_ms, conv.tail_bytes, conv.turn_probe.clone()), (Some(T0), Some(9_000), None));
    assert_eq!(restored.cache_lifetimes, meta.lock().unwrap().conversations.cache_lifetimes);
    assert_eq!(
        restored.cache_lifetimes.lifetime(OPUS, &crate::model_limits::for_model(OPUS), T0),
        Some(Lifetime { secs: 5_001, source: Source::Learned })
    );
    // After a restart the pause is still known, so the cold start still works.
    let restarted: SharedMeta = std::sync::Arc::new(std::sync::Mutex::new(crate::session::SessionMeta::new()));
    restarted.lock().unwrap().conversations = restored;
    fill(&restarted, "ctx", 200, 6_000);
    assert!(!start_turn(&restarted, "ctx", OPUS, T0 + secs(4_900)).checkpointed, "within the learned lifetime");
    assert!(start_turn(&restarted, "ctx", OPUS, T0 + secs(5_100)).checkpointed, "the pause survives the restart");
    // A session written before any of this existed loads, and never cold-starts
    // on its first turn (no request time is known).
    let old: crate::session::ConversationManager =
        serde_json::from_str(r#"{"contexts":{"ctx":{"messages":[{"role":"user","content":"hi"}]}}}"#).unwrap();
    assert!(old.cache_lifetimes.is_empty());
    assert_eq!(old.get("ctx").unwrap().last_request_ms, None);
    let fresh = serde_json::to_string(&crate::session::ConversationManager::default()).unwrap();
    assert_eq!(fresh, r#"{"contexts":{}}"#, "nothing new is written for an empty manager");
}

/// One simulated turn against a provider whose cache really lives an hour: start,
/// three requests with tool rounds, final answer. Content tokenizes at 2.5
/// bytes/token. Returns (the view at turn start, largest request in real tokens).
fn simulate_turn(meta: &SharedMeta, model: &str, now_ms: u64, turn: usize, work_bytes: usize) -> (View, usize) {
    const REAL: f64 = 2.5;
    let previous = with_conv(meta, "sim", |conv| conv.last_request_ms);
    let view = start_turn_with(meta, "sim", model, now_ms, format!("turn {turn}: {}", "u".repeat(800)));
    let mut sent = text_bytes(&view.messages) + 40_000;
    let mut worst = 0;
    for round in 0..3u64 {
        let tokens = (sent as f64 / REAL).ceil() as u64;
        worst = worst.max(tokens as usize);
        // The provider reads the prefix from its cache unless it expired or the
        // prefix changed (first request ever, or a cut view).
        let cached = round > 0 || (!view.checkpointed && previous.is_some_and(|last| now_ms - last < secs(HOUR)));
        let report = if cached { usage(1_000, Some(tokens - 41_000 / 2), Some(20_000)) } else { usage(1_000, Some(0), Some(tokens)) };
        answer(meta, "sim", model, sent, now_ms + round * 5_000, &report);
        with_conv(meta, "sim", |conv| {
            conv.push_native_batch(String::new(),
                vec![ToolCall { id: format!("t{turn}r{round}"), name: "inspect".into(), arguments: "{}".into() }],
                vec!["r".repeat(work_bytes / 3)])
        });
        sent += work_bytes / 3 + 200;
    }
    with_conv(meta, "sim", |conv| conv.push_assistant(format!("done {turn}")));
    (view, worst)
}

/// A chat with long pauses sends at most the cold-start size plus the new turn;
/// an active session grows to the window and stays cache-stable.
#[test]
fn cold_start_simulation_paused_chat_versus_active_session() {
    let cold_tokens = crate::model_limits::for_model(OPUS).cold_start_tokens;
    let limits = crate::model_limits::for_model(OPUS);
    // Paused chat: 40 turns of ≈60 KB (24k tokens), two hours apart.
    let paused = new_meta();
    let (mut cold_starts, mut worst_paused) = (0, 0);
    for turn in 0..40 {
        let now = T0 + turn as u64 * secs(2 * HOUR);
        let (view, worst) = simulate_turn(&paused, OPUS, now, turn, 60_000);
        cold_starts += usize::from(view.checkpointed);
        let ratio = with_conv(&paused, "sim", |conv| conv.bytes_per_token(OPUS));
        let at_start = history_tokens(&view, ratio);
        assert!(at_start <= cold_tokens + 1_000, "turn {turn}: {at_start} tokens of history at turn start");
        // The whole turn: the cold view, the tail and the turn's own work.
        assert!(worst <= cold_tokens + 16_000 + 24_000 + 2_000, "turn {turn}: request of {worst} tokens");
        worst_paused = worst_paused.max(worst);
    }
    assert!(cold_starts >= 8, "the history is cut back again and again: {cold_starts}");
    assert_eq!(with_conv(&paused, "sim", |conv| conv.messages.len()), 1 + 40 * 8, "nothing is deleted");
    // The provider's misses after two hours only confirm the one-hour default.
    assert_eq!(lifetime(&paused, OPUS), Some(Lifetime { secs: HOUR, source: Source::Default }));
    // Active session: the same turns 30 s apart, 150 turns ≈ 3.6M tokens of work.
    let active = new_meta();
    let (mut checkpoints, mut worst_active, mut stable_turns) = (0, 0, 0);
    let mut previous: Vec<String> = Vec::new();
    for turn in 0..150 {
        let now = T0 + turn as u64 * 30_000;
        let (view, worst) = simulate_turn(&active, OPUS, now, turn, 60_000);
        worst_active = worst_active.max(worst);
        let bodies: Vec<String> = view.messages.iter().map(|m| serde_json::to_string(m).unwrap())
            .filter(|body| !body.contains("older message(s) are omitted")).collect();
        if view.checkpointed {
            checkpoints += 1;
            // A budget checkpoint cuts to half the window, not to the cold size.
            let ratio = with_conv(&active, "sim", |conv| conv.bytes_per_token(OPUS));
            assert!(history_tokens(&view, ratio) > 2 * cold_tokens, "turn {turn}: not a cold start");
        } else {
            assert_eq!(&bodies[..previous.len().min(bodies.len())], &previous[..], "turn {turn}: history only appends");
            stable_turns += 1;
        }
        // Drop the newest user message: next turn it is followed by this turn's work.
        previous = bodies[..bodies.len() - 1].to_vec();
    }
    eprintln!("SIM paused: {cold_starts} cold starts, worst request {worst_paused} tokens | active: {checkpoints} checkpoints, worst request {worst_active} tokens, {stable_turns}/150 cache-stable turns");
    // 3.6M tokens of work through a 1M window: each checkpoint frees ≈350k.
    assert!((6..=12).contains(&checkpoints), "only budget checkpoints, in large steps: {checkpoints}");
    assert!(worst_active > 600_000, "the window is used: {worst_active}");
    assert!(worst_active <= limits.trigger_tokens(), "and never overrun: {worst_active}");
    assert!(worst_paused * 4 < worst_active, "a paused chat stays small: {worst_paused} vs {worst_active}");
    // Every request of the active session read its prefix from the cache: the
    // route's evidence is hits only, and nothing shortened the lifetime.
    let routes = &active.lock().unwrap().conversations.cache_lifetimes;
    let evidence = routes.evidence(OPUS).unwrap();
    assert!(evidence.hit_count > 400 && evidence.miss_count == 0, "{evidence:?}");
}

/// A request sent with another tool list than the turn's first request (the final
/// status round sends none) cannot read the turn's cached prefix: its zero read is not
/// a miss, its sizes still calibrate, and the turn's other requests stay evidence.
#[test]
fn a_request_with_another_tool_list_is_no_cache_evidence() {
    let meta = new_meta();
    let evidence = |meta: &SharedMeta| meta.lock().unwrap().conversations.cache_lifetimes.evidence(OPUS).cloned();
    let history: Vec<ChatMessage> = (0..20)
        .flat_map(|_| [ChatMessage::user("x".repeat(3_000)), ChatMessage::assistant("y".repeat(3_000))])
        .collect();
    {
        let mut guard = meta.lock().unwrap();
        guard.conversations.get_or_create("ctx").turn_probe = Some(TurnProbe::default());
        expect_cached_prefix(&mut guard.conversations, "ctx", "s", &history, None);
    }
    // First request of the turn writes the prefix.
    answer(&meta, "ctx", OPUS, 150_000, T0, &usage(10, Some(0), Some(60_000)));
    // Ten minutes of tool work, then the status round without tools reads nothing.
    let last = T0 + secs(600);
    sent(OPUS, 140_000, last).with_other_tools().calibrate(&meta, "ctx", false, false, &usage(56_000, Some(0), None));
    assert_eq!(evidence(&meta), None, "not a miss");
    let calibration = with_conv(&meta, "ctx", |conv| conv.token_calibration.clone()).expect("sizes are still calibrated");
    assert_eq!((calibration.request_bytes, calibration.input_tokens), (140_000, 56_000));
    assert_eq!(with_conv(&meta, "ctx", |conv| conv.last_request_ms), Some(last), "it is the conversation's last request");
    // The same request with the turn's tools would have been a miss.
    answer(&meta, "ctx", OPUS, 140_000, last + secs(600), &usage(56_000, Some(0), None));
    assert_eq!(evidence(&meta).map(|e| e.miss_count), Some(1));
}
