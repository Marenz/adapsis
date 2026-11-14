use super::*;
use crate::llm::ToolCall;

fn image(n: u8) -> crate::attachment::Attachment {
    crate::attachment::Attachment::from_bytes(vec![n; 3], "image/png", "reference.png").unwrap()
}

/// Projection at the default (uncalibrated) ratio.
fn est(bytes: usize) -> usize {
    tokens_for_bytes(bytes, crate::model_limits::DEFAULT_BYTES_PER_TOKEN)
}

fn roomy() -> ViewPolicy {
    ViewPolicy::tokens_only(1_000_000, 500_000, 0, 100_000)
}

fn json(view: &View) -> String {
    serde_json::to_string(&view.messages).unwrap()
}

#[test]
fn photo_only_turn_survives() {
    let mut conv = Conversation::new();
    conv.push_system("system");
    conv.push_user_with_attachment("", image(1));
    assert!(json(&build(&mut conv, &roomy())).contains("data:image/png;base64,AQEB"));
    assert_eq!(conv.messages[1].attachments.len(), 1);
}

#[test]
fn captioned_image_is_sent_with_its_caption() {
    let mut conv = Conversation::new();
    conv.push_user_with_attachment("Use this reference", image(2));
    let text = json(&build(&mut conv, &roomy()));
    assert!(text.contains("Use this reference") && text.contains("data:image/png;base64,AgIC"));
}

#[test]
fn telegram_octet_stream_photos_survive_and_use_image_mime() {
    let mut conv = Conversation::new();
    let attachment = crate::attachment::Attachment::from_bytes(vec![0xff, 0xd8, 0xff], "application/octet-stream", "opaque-hash").unwrap();
    conv.push_user_with_attachment("", attachment);
    assert!(json(&build(&mut conv, &roomy())).contains("data:image/jpeg;base64,/9j/"));
}

#[test]
fn image_budget_overflow_moves_the_anchor_once_per_three_images() {
    let mut conv = Conversation::new();
    conv.push_system("system");
    let mut moves = 0;
    let mut previous = String::new();
    for n in 1..=7u8 {
        conv.push_user_with_attachment(format!("photo {n}"), image(n));
        let view = build(&mut conv, &roomy());
        let text = json(&view);
        assert!(text.contains(&format!("photo {n}")), "newest turn present");
        let b64 = |n: u8| { use base64::Engine; base64::engine::general_purpose::STANDARD.encode([n; 3]) };
        assert!(text.contains(&b64(n)), "newest image always sent");
        if view.checkpointed {
            moves += 1;
        } else if !previous.is_empty() {
            // Without a move the previous request's messages are a byte prefix.
            let prefix = previous.trim_end_matches(']');
            assert!(text.starts_with(prefix), "image retention changed without a checkpoint");
        }
        previous = text;
    }
    // Images 4 and 7 overflow the budget of 3; every other image appends stably.
    assert_eq!(moves, 2);
    assert_eq!(conv.messages.iter().map(|m| m.attachments.len()).sum::<usize>(), 7, "persisted attachments untouched");
}

#[test]
fn history_is_byte_stable_between_checkpoints_and_cut_respects_budget() {
    let mut conv = Conversation::new();
    conv.push_system("system");
    // Each turn: user 400 chars + assistant 400 chars; one message projects to
    // (402 + framing) bytes / 2.5 = 200 tokens, so the view grows to 18 messages
    // and is cut back to 8.
    let policy = ViewPolicy::tokens_only(4_000, 2_000, 400, 10_000);
    let mut previous: Option<Vec<String>> = None;
    let mut checkpoints = 0;
    for turn in 0..40 {
        conv.push_user(format!("u{turn:02}{}", "x".repeat(397)));
        let view = build(&mut conv, &policy);
        let bodies: Vec<String> = view.messages.iter().map(|m| serde_json::to_string(m).unwrap()).collect();
        // Newest turn always present.
        assert!(bodies.last().unwrap().contains(&format!("u{turn:02}")));
        let history_tokens: usize = view.messages.iter().skip(1)
            .map(|m| est(json_len(serde_json::to_value(m).unwrap()["content"].as_str().unwrap_or("")) + MESSAGE_FRAMING_BYTES))
            .sum();
        if view.checkpointed {
            checkpoints += 1;
            assert!(policy.overhead_tokens + history_tokens <= policy.target_tokens,
                "after a cut the projection is within the target: {history_tokens}");
        } else if let Some(previous) = &previous {
            // Byte-identical prefix (ignoring the volatile omission notice slot).
            let strip = |v: &[String]| v.iter().filter(|b| !b.contains("older message(s) are omitted")).cloned().collect::<Vec<_>>();
            let (old, new) = (strip(previous), strip(&bodies));
            assert_eq!(&new[..old.len()], &old[..], "turn {turn}: history must only append between checkpoints");
        }
        assert!(policy.overhead_tokens + history_tokens <= policy.trigger_tokens, "never above the trigger");
        previous = Some(bodies);
        conv.push_assistant(format!("a{turn:02}{}", "y".repeat(397)));
    }
    assert!((3..=8).contains(&checkpoints), "large steps, not every turn: {checkpoints}");
    assert_eq!(conv.messages.len(), 81, "persisted history is never trimmed");
}

#[test]
fn checkpoint_never_splits_a_native_batch_and_prefers_a_user_boundary() {
    let mut conv = Conversation::new();
    conv.push_system("system");
    for turn in 0..6 {
        conv.push_user(format!("task {turn} {}", "x".repeat(200)));
        conv.push_native_batch(
            String::new(),
            vec![ToolCall { id: format!("c{turn}"), name: "inspect".into(), arguments: "{}".into() }],
            vec!["r".repeat(200)],
        );
    }
    conv.push_user("now");
    let view = build(&mut conv, &ViewPolicy::tokens_only(300, 150, 0, 10_000));
    assert!(view.checkpointed);
    assert_eq!(conv.messages[conv.view_start].role, "user", "cut lands on a user turn");
    let roles: Vec<_> = view.messages.iter().map(|m| (m.role.clone(), !m.tool_calls.is_empty(), m.tool_call_id.is_some())).collect();
    for (i, (role, calls, _)) in roles.iter().enumerate() {
        if *calls {
            assert!(roles.get(i + 1).is_some_and(|r| r.2), "a call is followed by its result");
        }
        if *role == crate::llm::ChatRole::Tool {
            assert!(i > 0 && roles[i - 1].1, "no orphan result at the cut");
        }
    }
}

#[test]
fn per_message_cap_is_deterministic_not_budget_dependent() {
    let mut conv = Conversation::new();
    conv.push_user(format!("big{}", "z".repeat(5_000)));
    let first = json(&build(&mut conv, &ViewPolicy::tokens_only(100_000, 50_000, 0, 1_000)));
    conv.push_assistant("ok");
    conv.push_user("next");
    let second = json(&build(&mut conv, &ViewPolicy::tokens_only(100_000, 50_000, 40_000, 1_000)));
    assert!(second.starts_with(first.trim_end_matches(']')), "same cap, same bytes");
    assert!(first.contains("characters omitted"));
}

/// Owner requirement: a capped message tells the model it was cut and by how
/// much, and the persisted message stays complete.
#[test]
fn capped_message_states_the_omitted_character_count_and_storage_is_untouched() {
    let original = format!("HEAD{}TAIL", "ä".repeat(9_992));
    let mut conv = Conversation::new();
    conv.push_system("s");
    conv.push_user(original.clone());
    let view = build(&mut conv, &ViewPolicy::tokens_only(1_000_000, 500_000, 0, 1_000));
    let crate::llm::ChatContent::Text(shown) = &view.messages[1].content else { panic!("text") };
    // The cap is in UTF-8 bytes; the original is 19,992 bytes / 10,000 characters.
    assert!(shown.len() <= 1_000 && shown.len() > 990, "{}", shown.len());
    assert!(shown.starts_with("HEAD") && shown.ends_with("TAIL"));
    let marker = shown.lines().find(|line| line.starts_with("[... ")).expect("marker line");
    assert!(marker.contains("omitted here by the runtime") && marker.contains("the stored message is complete"), "{marker}");
    let omitted: usize = marker.split_whitespace().nth(1).unwrap().parse().unwrap();
    let kept = shown.chars().count() - marker.chars().count() - 2;
    assert_eq!(omitted + kept, 10_000, "the count is exact");
    assert_eq!(conv.messages[1].content, original, "persisted history keeps the full text");
    // Deterministic: the same message and cap give the same bytes.
    assert_eq!(truncate_middle(&original, 1_000), *shown);
    // A message within the cap is passed through without a marker.
    assert_eq!(truncate_middle("short", 1_000), "short");
    // Small caps use the short marker (still with the count); tiny caps keep the head.
    let small = truncate_middle(&original, 100);
    assert!(small.contains(" characters omitted ...]") && small.len() <= 100, "{small}");
    let omitted: usize = small.lines().find(|l| l.starts_with("[... ")).unwrap().split_whitespace().nth(1).unwrap().parse().unwrap();
    assert_eq!(omitted, 10_000 - (small.chars().count() - small.lines().find(|l| l.starts_with("[... ")).unwrap().chars().count() - 2));
    assert_eq!(truncate_middle(&original, 20), "HEADääääääää");
    assert_eq!(truncate_middle(&original, 21), "HEADääääääää", "never cuts inside a character");
    // Multi-byte text right at the cut positions stays valid at every cap.
    for cap in 0..400 {
        let cut = truncate_middle(&original, cap);
        assert!(cut.len() <= cap, "cap {cap}: {}", cut.len());
    }
    // The projection counts the same text the view sends.
    let policy = ViewPolicy::tokens_only(1_000_000, 500_000, 0, 1_000);
    assert_eq!(message_bytes(&conv.messages[1], 1_000), json_len(shown) + MESSAGE_FRAMING_BYTES);
    let _ = policy;
}

/// 30-turn coding simulation (20–50k tokens per turn, native batches of 8k-char
/// tool results) against a 1M-token window, compared with the pre-#68 rule
/// (newest 60k chars / 64 messages / 40k per message). Run with --nocapture
/// to see the numbers; assertions pin the new behaviour.
#[test]
fn conversation_view_thirty_turn_coding_simulation() {
    use crate::context_log::{Epoch, observe_epoch};
    let key = crate::context_prompt::StablePrefixKey { context: "sim".into(), may_write: true, identity_hash: 1, mesh_hash: 1, tools_hash: 0 };
    // Window-bound simulation: the route body limit is raised (gateway fix), so the
    // token budget decides. The 2 MB case is covered by the byte-limit test.
    let limits = crate::model_limits::ModelLimits {
        context_tokens: 1_000_000, output_tokens: 128_000,
        request_timeout: std::time::Duration::from_secs(1), chunk_timeout: std::time::Duration::from_secs(1),
        max_request_bytes: 64_000_000,
        ..crate::model_limits::for_model("unknown/test-model")
    };
    let system = "s".repeat(36_000); // ~9k tokens, like the live admin DM prefix
    // A conversation measured at 4 bytes/token (the sizes below are in those terms).
    let policy = ViewPolicy::from_limits(&limits, &system, 4.0);
    let mut seed = 0x2545_f491_u64;
    let mut next = move |lo: usize, hi: usize| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; lo + (seed as usize) % (hi - lo + 1) };
    let mut conv = Conversation::new();
    conv.push_system(system.clone());
    let tokens = |m: &[crate::llm::ChatMessage]| m.iter().map(|m| tokens_for_bytes(serde_json::to_string(m).unwrap().len(), 4.0)).sum::<usize>();
    let (mut new_resets, mut checkpoints, mut new_stable, mut new_total) = (0, 0, 0usize, 0usize);
    let (mut old_resets, mut old_stable, mut old_total) = (0, 0usize, 0usize);
    let mut previous_new: Vec<crate::llm::ChatMessage> = Vec::new();
    let mut previous_old: Vec<crate::llm::ChatMessage> = Vec::new();
    for turn in 0..30 {
        conv.push_user(format!("turn {turn}: {}", "u".repeat(next(500, 2_000))));
        let view = build(&mut conv, &policy);
        let history: Vec<_> = view.messages[1..view.messages.len() - 1].iter().filter(|m| !matches!(&m.content, crate::llm::ChatContent::Text(t) if t.contains("older message(s) are omitted"))).cloned().collect();
        let common = history.iter().zip(&previous_new).take_while(|(a, b)| serde_json::to_string(a).unwrap() == serde_json::to_string(b).unwrap()).count();
        let continued = previous_new.is_empty() || common == previous_new.len();
        match observe_epoch("sim-new", &key, "m", &system, &history, view.checkpointed) {
            Epoch::Continued { .. } => assert!(continued && !view.checkpointed),
            Epoch::Reset { .. } => new_resets += 1,
        }
        checkpoints += usize::from(view.checkpointed);
        new_stable += tokens(&history[..common]);
        new_total += tokens(&history);
        let request_tokens = policy.overhead_tokens + tokens(&view.messages[1..]);
        assert!(request_tokens <= limits.context_tokens, "request fits the window");
        previous_new = history;
        // Old rule over the same persisted messages.
        let old = old_view(&conv);
        let old_common = old.iter().zip(&previous_old).take_while(|(a, b)| serde_json::to_string(a).unwrap() == serde_json::to_string(b).unwrap()).count();
        if previous_old.is_empty() || old_common != previous_old.len() { old_resets += 1; }
        old_stable += tokens(&old[..old_common]);
        old_total += tokens(&old);
        previous_old = old;
        // The turn's work: native batches of 8k-char results until 20–50k tokens.
        let mut budget = next(20_000, 50_000) * 4;
        let mut call = 0;
        while budget > 9_000 {
            conv.push_native_batch(String::new(),
                vec![ToolCall { id: format!("t{turn}c{call}"), name: "inspect".into(), arguments: format!("{{\"q\":\"{}\"}}", "a".repeat(next(200, 1_000))) }],
                vec!["r".repeat(8_000)]);
            budget -= 9_000;
            call += 1;
        }
        conv.push_assistant(format!("done {turn}"));
    }
    eprintln!("SIM new: epoch resets {new_resets}/30, checkpoints {checkpoints}, cache-stable history share {:.1}%", 100.0 * new_stable as f64 / new_total as f64);
    eprintln!("SIM old: epoch resets {old_resets}/30, cache-stable history share {:.1}%, old view ~{} tokens", 100.0 * old_stable as f64 / old_total.max(1) as f64, tokens(&previous_old));
    assert_eq!(new_resets, 1 + checkpoints, "only the first turn and checkpoints reset");
    assert!(checkpoints >= 1 && checkpoints <= 2, "30 turns of ~35k tokens cross 85% of 1M about once: {checkpoints}");
}

fn old_view(conv: &Conversation) -> Vec<crate::llm::ChatMessage> {
    // Pre-#68 rule: newest groups within 60k chars and 64 messages, 40k per message.
    let mut remaining = 60_000usize;
    let mut out = Vec::new();
    for group in super::super::native_history::complete_groups(&conv.messages).into_iter().rev() {
        if group[0].role == "system" || out.len() + group.len() > 63 { continue; }
        let cost: usize = group.iter().map(raw_bytes).sum();
        if group.len() > 1 {
            if cost <= remaining && group.iter().all(|m| raw_bytes(m) <= 40_000) {
                out.extend(group.iter().rev().cloned());
                remaining -= cost;
            }
            continue;
        }
        let content = truncate_middle(&group[0].content, 40_000.min(remaining));
        if content.is_empty() { continue; }
        remaining -= content.chars().count();
        out.push(ChatMessage { role: group[0].role.clone(), content, ..Default::default() });
    }
    out.reverse();
    if out.last().is_some_and(|m| m.role == "user") { out.pop(); }
    out.iter().map(super::super::native_history::to_llm).collect()
}

#[test]
fn no_message_count_cap_only_the_token_budget_decides() {
    let mut conv = Conversation::new();
    conv.push_system("system");
    for i in 0..300 {
        conv.push_user(format!("m{i}"));
    }
    let view = build(&mut conv, &roomy());
    assert!(!view.checkpointed);
    assert_eq!(view.messages.len(), 301, "the old 64-message cap is gone");
}

/// Route body limit (gateway: 2 MiB): a history far above the byte limit is cut at
/// one checkpoint so the serialized request fits, images counted as base64.
#[test]
fn byte_limit_checkpoint_keeps_the_serialized_request_under_the_route_limit() {
    let limits = crate::model_limits::ModelLimits {
        context_tokens: 1_000_000, output_tokens: 128_000,
        request_timeout: std::time::Duration::from_secs(1), chunk_timeout: std::time::Duration::from_secs(1),
        max_request_bytes: 2_000_000,
        ..crate::model_limits::for_model("unknown/test-model")
    };
    let system = "s".repeat(36_000);
    // At the sparsest ratio the token target (≈2 MB of history) is looser than the
    // route's byte target (≈1 MB), so the byte budget decides the cut.
    let policy = ViewPolicy::from_limits(&limits, &system, crate::model_limits::MAX_BYTES_PER_TOKEN);
    assert!(policy.image_bytes <= 450_000, "images limited to ~30% of the route in base64");
    let mut conv = Conversation::new();
    conv.push_system(system.clone());
    // ~7.5 MB of multi-byte history against a 2 MB route.
    for i in 0..300 {
        conv.push_user(format!("q{i} {}", "ä".repeat(5_000)));
        conv.push_assistant(format!("a{i} {}", "€".repeat(5_000)));
    }
    let photo = crate::attachment::Attachment::from_bytes(vec![7u8; 140_000], "image/png", "p.png").unwrap();
    conv.push_user_with_attachment("latest with photo", photo);
    let view = build(&mut conv, &policy);
    assert!(view.checkpointed, "over the byte trigger");
    let wire = serde_json::to_vec(&view.messages).unwrap().len();
    // Request = view + runtime tail/tool schemas (allowance) must fit the route.
    assert!(wire + ViewPolicy::TAIL_ALLOWANCE_BYTES <= limits.max_request_bytes, "request {wire} bytes");
    assert!(wire <= limits.max_request_bytes / 2 + 64 * 1024, "cut down to about the 50% target: {wire}");
    let text = String::from_utf8(serde_json::to_vec(&view.messages).unwrap()).unwrap();
    assert!(text.contains("latest with photo") && text.contains("data:image/png;base64,"), "newest turn and its image kept");
    assert_eq!(conv.messages.len(), 602, "persisted history untouched");
    // Next turn appends without another checkpoint.
    conv.push_assistant("ok");
    conv.push_user("follow-up");
    assert!(!build(&mut conv, &policy).checkpointed);
}

#[test]
fn secondary_system_notifications_become_user_runtime_notices_and_keep_the_prefix_stable() {
    let mut conv = Conversation::new();
    conv.push_system("stable prefix");
    conv.push_user("first");
    conv.push_assistant("reply");
    let before = build(&mut conv, &roomy());
    // A runtime notification (append_conversation_notification) and the next turn.
    conv.push_system("job finished: result ready");
    conv.push_user("next");
    let after = build(&mut conv, &roomy());
    let wire = |v: &View| v.messages.iter().map(|m| serde_json::to_string(m).unwrap()).collect::<Vec<_>>();
    let (old, new) = (wire(&before), wire(&after));
    assert_eq!(&new[..old.len()], &old[..], "provider-visible prefix is byte-stable across a notification");
    let systems = after.messages.iter().filter(|m| m.role == crate::llm::ChatRole::System).count();
    assert_eq!(systems, 1, "only the primary prefix is a system message");
    let notice = &after.messages[3];
    assert_eq!(notice.role, crate::llm::ChatRole::User);
    let text = serde_json::to_string(notice).unwrap();
    assert!(text.contains("Runtime notice") && text.contains("job finished"), "{text}");
}

#[test]
fn image_projection_counts_three_images_total_not_three_per_message() {
    let photo = |n: u8| crate::attachment::Attachment::from_bytes(vec![n; 300_000], "image/png", "p.png").unwrap();
    let mut conv = Conversation::new();
    conv.push_system("s");
    let mut message = super::ChatMessage { role: "user".into(), content: "many".into(), ..Default::default() };
    message.attachments = (1..=5).map(photo).collect();
    conv.messages.push(message);
    let policy = ViewPolicy::tokens_only(10_000_000, 5_000_000, 0, 1_000_000);
    let all = groups(&conv, &policy);
    let sent: usize = kept_image_sizes(&conv, policy.image_bytes).values().map(Vec::len).sum();
    assert_eq!(sent, 3);
    let view = build(&mut conv, &policy);
    let sent_parts = serde_json::to_string(&view.messages).unwrap().matches("data:image/png;base64,").count();
    assert_eq!(sent_parts, 3, "projection and the sent request agree");
    // Text is projected from its serialized bytes; images at a fixed cost outside the ratio.
    assert_eq!(all[0].tokens, est(message_bytes(&conv.messages[1], policy.max_message_bytes)) + 3 * IMAGE_TOKENS);
}

#[test]
fn forced_checkpoint_keeps_only_the_newest_turn_and_batch_integrity() {
    let mut conv = Conversation::new();
    conv.push_system("s");
    conv.push_user("old");
    conv.push_native_batch(String::new(), vec![ToolCall { id: "c".into(), name: "inspect".into(), arguments: "{}".into() }], vec!["r".into()]);
    conv.push_user("newest");
    conv.push_native_batch(String::new(), vec![ToolCall { id: "d".into(), name: "inspect".into(), arguments: "{}".into() }], vec!["r2".into()]);
    assert!(force_checkpoint(&mut conv));
    assert_eq!(conv.messages[conv.view_start].content, "newest");
    let view = build(&mut conv, &roomy());
    let text = serde_json::to_string(&view.messages).unwrap();
    assert!(!text.contains("\"old\"") && text.contains("newest") && text.contains("r2"));
    assert!(!force_checkpoint(&mut conv), "idempotent once at the newest turn");
}

fn opus_like() -> crate::model_limits::ModelLimits {
    crate::model_limits::ModelLimits {
        context_tokens: 1_000_000, output_tokens: 128_000,
        request_timeout: std::time::Duration::from_secs(1), chunk_timeout: std::time::Duration::from_secs(1),
        max_request_bytes: 64_000_000,
        ..crate::model_limits::for_model("unknown/test-model")
    }
}

/// The measured ratio of the previous request decides where the next request's
/// checkpoint fires: the same history is kept whole or cut depending on it.
#[test]
fn calibration_from_usage_is_applied_to_the_next_projection() {
    use crate::model_limits::TokenCalibration;
    let limits = opus_like();
    let history = |conv: &mut Conversation| {
        conv.push_system("s");
        // 300 turns × ~6 KB ≈ 1.83 MB of serialized history.
        for i in 0..300 {
            conv.push_user(format!("q{i} {}", "u".repeat(3_000)));
            conv.push_assistant(format!("a{i} {}", "r".repeat(3_000)));
        }
        conv.push_user("newest");
    };
    let view = |conv: &mut Conversation| {
        let policy = ViewPolicy::from_limits(&limits, "s", conv.bytes_per_token("m"));
        build(conv, &policy)
    };
    // Uncalibrated: 2.5 bytes/token -> ≈743k history + 144k overhead > 850k: cut.
    let mut fresh = Conversation::new();
    history(&mut fresh);
    assert_eq!(fresh.bytes_per_token("m"), crate::model_limits::DEFAULT_BYTES_PER_TOKEN);
    assert!(view(&mut fresh).checkpointed, "the conservative default cuts");
    // Measured sparse (4 bytes/token): ≈464k + overhead stays under the trigger.
    let mut sparse = Conversation::new();
    history(&mut sparse);
    sparse.token_calibration = TokenCalibration::measure("m", 400_000, 0, 100_000);
    assert_eq!(sparse.bytes_per_token("m"), 4.0);
    assert!(!view(&mut sparse).checkpointed, "a measured 4 bytes/token keeps the whole history");
    assert_eq!(sparse.view_start, 0);
    // Measured dense (2 bytes/token): cut deeper than at the default.
    let mut dense = Conversation::new();
    history(&mut dense);
    dense.token_calibration = TokenCalibration::measure("m", 200_000, 0, 100_000);
    assert!(view(&mut dense).checkpointed);
    assert!(dense.view_start > fresh.view_start, "denser tokens leave room for less history: {} vs {}", dense.view_start, fresh.view_start);
    // The projection after the cut is within the target in the measured terms.
    let policy = ViewPolicy::from_limits(&limits, "s", dense.bytes_per_token("m"));
    let kept: usize = groups(&dense, &policy).iter().filter(|g| g.start >= dense.view_start).map(|g| g.tokens).sum();
    assert!(policy.overhead_tokens + kept <= limits.target_tokens(), "{kept}");
    // A ratio change alone never rewrites the visible history of an unchanged
    // conversation below the trigger: only the trigger position depends on it.
    let before = json(&view(&mut sparse));
    sparse.token_calibration = TokenCalibration::measure("m", 350_000, 0, 100_000);
    assert_eq!(json(&view(&mut sparse)), before);
}

/// Reviewer follow-up: the ratio is the tokenizer's, so a conversation whose
/// generating model changes projects with the default until the new model has
/// answered, and returns to nothing stale when it switches back.
#[test]
fn calibration_is_reset_when_the_generating_model_changes() {
    use crate::model_limits::{DEFAULT_BYTES_PER_TOKEN, TokenCalibration};
    let limits = opus_like();
    let mut conv = Conversation::new();
    conv.push_system("s");
    for i in 0..300 {
        conv.push_user(format!("q{i} {}", "u".repeat(3_000)));
        conv.push_assistant(format!("a{i} {}", "r".repeat(3_000)));
    }
    conv.push_user("newest");
    // Model A measured sparse tokens: the 1.83 MB history stays whole for A.
    conv.token_calibration = TokenCalibration::measure("route/a", 400_000, 0, 100_000);
    assert_eq!(conv.bytes_per_token("route/a"), 4.0);
    let view_for = |conv: &mut Conversation, model: &str| {
        let policy = ViewPolicy::from_limits(&limits, "s", conv.bytes_per_token(model));
        build(conv, &policy)
    };
    assert!(!view_for(&mut conv.clone(), "route/a").checkpointed);
    // The same conversation sent to model B: A's ratio does not apply, the
    // conservative default does, and the view is cut.
    assert_eq!(conv.bytes_per_token("route/b"), DEFAULT_BYTES_PER_TOKEN);
    assert!(view_for(&mut conv.clone(), "route/b").checkpointed);
    // B answers: its own measurement replaces A's.
    conv.token_calibration = TokenCalibration::measure("route/b", 300_000, 0, 100_000);
    assert_eq!(conv.bytes_per_token("route/b"), 3.0);
    // Back on A there is no stale ratio either: default until A answers again.
    assert_eq!(conv.bytes_per_token("route/a"), DEFAULT_BYTES_PER_TOKEN);
}

/// A coding session whose content really tokenizes at `real` bytes per token, run
/// against a 1M window. `calibrate` feeds each request's provider usage back into
/// the conversation (the #68 follow-up); without it the ratio stays at `fixed`.
/// Returns (largest real input tokens of any request, checkpoints).
fn run_dense_session(real: f64, fixed: Option<f32>) -> (usize, usize) {
    use crate::model_limits::TokenCalibration;
    let limits = opus_like();
    let system = "s".repeat(36_000);
    let mut seed = 0x9e37_79b9_u64;
    let mut next = move |lo: usize, hi: usize| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; lo + (seed as usize) % (hi - lo + 1) };
    let mut conv = Conversation::new();
    conv.push_system(system.clone());
    let (mut worst, mut checkpoints) = (0usize, 0usize);
    let request = |conv: &mut Conversation| {
        let ratio = fixed.unwrap_or_else(|| conv.bytes_per_token("m"));
        let view = build(conv, &ViewPolicy::from_limits(&limits, &system, ratio));
        // The request as sent: the view plus a 40 KB runtime tail and tool schemas.
        let bytes: usize = view.messages.iter().map(|m| serde_json::to_string(m).unwrap().len()).sum::<usize>() + 40_000;
        let provider_tokens = (bytes as f64 / real).ceil() as usize;
        if fixed.is_none() {
            if let Some(measured) = TokenCalibration::measure("m", bytes, 0, provider_tokens as u64) {
                conv.token_calibration = Some(measured);
            }
        }
        (provider_tokens, view.checkpointed)
    };
    for turn in 0..60 {
        conv.push_user(format!("turn {turn}: {}", "u".repeat(next(500, 2_000))));
        // Every round of the turn is one request: 8 KB tool results until 50–125 KB.
        let mut budget = next(50_000, 125_000);
        let mut call = 0;
        loop {
            let (tokens, checkpointed) = request(&mut conv);
            worst = worst.max(tokens);
            checkpoints += usize::from(checkpointed);
            if budget < 9_000 { break; }
            conv.push_native_batch(String::new(),
                vec![ToolCall { id: format!("t{turn}c{call}"), name: "inspect".into(), arguments: format!("{{\"q\":\"{}\"}}", "a".repeat(next(200, 1_000))) }],
                vec!["r".repeat(8_000)]);
            budget -= 9_000;
            call += 1;
        }
        conv.push_assistant(format!("done {turn}"));
    }
    (worst, checkpoints)
}

/// Live evidence (2026-11-10): the admin DM tokenized at ≈2.46 bytes/token while
/// the view was projected at 4 chars/token. With a real ratio of 2.5 the old fixed
/// rule sends requests over the window; the calibrated rule never does, and it
/// still does not when the content is denser than the conservative default.
#[test]
fn calibrated_projection_never_sends_a_request_over_the_window() {
    let limits = opus_like();
    let input_limit = limits.context_tokens - limits.output_tokens as usize;
    let (old_worst, _) = run_dense_session(2.5, Some(4.0));
    assert!(old_worst > limits.context_tokens, "the fixed 4/token rule overruns the window: {old_worst}");
    for real in [2.5, 2.0, 3.3, 4.0] {
        let (worst, checkpoints) = run_dense_session(real, None);
        eprintln!("SIM calibrated real={real}: worst request {worst} tokens, {checkpoints} checkpoints (old fixed rule at 2.5: {old_worst})");
        assert!(worst <= input_limit, "real {real}: request of {worst} tokens leaves no room for the output");
        // 85% of the window including the output reserve and the tail allowance.
        assert!(worst <= limits.trigger_tokens(), "real {real}: {worst} stays at or under the trigger");
        assert!(checkpoints >= 1, "real {real}: the session is long enough to checkpoint");
        // Still large steps: the window is used, not cut every turn.
        assert!(checkpoints <= 12, "real {real}: {checkpoints} checkpoints");
    }
}

#[test]
fn token_calibration_persists_with_the_conversation_and_old_sessions_load() {
    use crate::model_limits::TokenCalibration;
    let mut conv = Conversation::new();
    conv.push_user("hello");
    conv.token_calibration = TokenCalibration::measure("m", 1_460_000, 0, 593_447);
    let json = serde_json::to_string(&conv).unwrap();
    assert!(json.contains(r#""token_calibration":{"request_bytes":1460000,"input_tokens":593447,"model":"m"}"#), "{json}");
    let restored: Conversation = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.token_calibration, conv.token_calibration);
    assert!((restored.bytes_per_token("m") - 2.46).abs() < 0.01);
    // A conversation without a measurement does not serialize the field, and a
    // session written before the field existed loads with the default ratio.
    let fresh = serde_json::to_string(&Conversation::new()).unwrap();
    assert!(!fresh.contains("token_calibration"), "{fresh}");
    let old: Conversation = serde_json::from_str(r#"{"messages":[{"role":"user","content":"hi"}],"view_start":0}"#).unwrap();
    assert_eq!(old.token_calibration, None);
    assert_eq!(old.bytes_per_token("m"), crate::model_limits::DEFAULT_BYTES_PER_TOKEN);
    // A calibration stored before the model was recorded (4733b2f) loads and is
    // simply re-learned: it matches no model.
    let unlabelled: Conversation = serde_json::from_str(
        r#"{"messages":[],"token_calibration":{"request_bytes":400000,"input_tokens":100000}}"#).unwrap();
    assert_eq!(unlabelled.bytes_per_token("m"), crate::model_limits::DEFAULT_BYTES_PER_TOKEN);
    // Through the manager (the shape stored in the session file).
    let mut manager = crate::session::ConversationManager::default();
    manager.get_or_create("ctx").token_calibration = TokenCalibration::measure("m", 30_000, 0, 10_000);
    let restored: crate::session::ConversationManager = serde_json::from_str(&serde_json::to_string(&manager).unwrap()).unwrap();
    assert_eq!(restored.get("ctx").unwrap().bytes_per_token("m"), 3.0);
}

fn batch(conv: &mut Conversation, id: &str, name: &str, arguments: String, result: String) {
    conv.push_native_batch(String::new(), vec![ToolCall { id: id.into(), name: name.into(), arguments }], vec![result]);
}

/// A tool RESULT over the per-message cap is cut in the view with the omission marker
/// and its batch stays (after a switch to a model with a smaller window the earlier,
/// larger results are still there). Only a batch whose CALL exceeds the cap is left out
/// whole: tool-call arguments cannot be cut.
#[test]
fn an_oversized_tool_result_is_cut_in_the_view_and_its_batch_is_kept() {
    let mut conv = Conversation::new();
    conv.push_system("system");
    conv.push_user("read it");
    batch(&mut conv, "big-result", "read", r#"{"path":"a"}"#.into(), format!("HEAD {} TAIL", "r".repeat(50_000)));
    batch(&mut conv, "big-call", "write", format!(r#"{{"content":"{}"}}"#, "c".repeat(50_000)), "written".into());
    batch(&mut conv, "small", "read", r#"{"path":"b"}"#.into(), "small result".into());
    conv.push_user("next");
    let policy = ViewPolicy::tokens_only(1_000_000, 500_000, 0, 10_000);
    let (first, second) = (build(&mut conv, &policy), build(&mut conv, &policy));
    assert_eq!(json(&first), json(&second), "the cut is deterministic");
    let view = &first.messages;
    let result = view.iter().find(|message| message.tool_call_id.as_deref() == Some("big-result")).expect("the batch is kept");
    let text = match &result.content {
        crate::llm::ChatContent::Text(text) => text.clone(),
        other => panic!("{other:?}"),
    };
    assert!(text.len() <= 10_000 && text.starts_with("HEAD ") && text.ends_with(" TAIL"), "{}", text.len());
    assert!(text.contains("characters omitted here by the runtime to fit the model context; the stored message is complete"));
    let sent = json(&first);
    assert!(sent.contains("\"id\":\"big-result\"") && sent.contains("\"id\":\"small\"") && sent.contains("small result"));
    assert!(!sent.contains("big-call") && !sent.contains("written"), "a call that cannot be shown whole takes its batch with it");
    assert!(conv.messages.iter().any(|message| message.content.len() > 50_000), "stored complete");
    // Under a roomy cap nothing is cut.
    assert!(json(&build(&mut conv, &roomy())).contains(&"r".repeat(50_000)));
}

/// `hidden_tools`: a complete batch with a call to a hidden tool is left out of the view,
/// whatever else it calls; the view says how many, in the notice the request builders
/// move to the volatile tail; the stored history and the checkpoint are untouched.
#[test]
fn batches_of_hidden_tools_are_left_out_whole_with_a_notice() {
    fn hide(name: &str) -> bool {
        matches!(name, "read" | "workspace_open")
    }
    let mut conv = Conversation::new();
    conv.push_system("system");
    conv.push_user("first question");
    batch(&mut conv, "h1", "workspace_open", "{}".into(), "opened SECRET_PATH".into());
    conv.push_native_batch(
        "mixed".into(),
        vec![ToolCall { id: "h2".into(), name: "inspect".into(), arguments: "{}".into() }, ToolCall { id: "h3".into(), name: "read".into(), arguments: "{}".into() }],
        vec!["symbols".into(), "SECRET_CONTENT".into()],
    );
    batch(&mut conv, "v1", "inspect", "{}".into(), "visible symbols".into());
    conv.push_assistant("first answer");
    conv.push_user("second question");
    let stored = conv.messages.clone();
    let mut policy = roomy();
    let open = json(&build(&mut conv, &policy));
    assert!(open.contains("SECRET_CONTENT") && !open.contains("are not shown in this turn"));
    policy.hidden_tools = Some(hide);
    let view = build(&mut conv, &policy);
    let hidden = json(&view);
    for absent in ["SECRET_PATH", "SECRET_CONTENT", "workspace_open", "\"h1\"", "\"h2\"", "\"h3\"", "mixed"] {
        assert!(!hidden.contains(absent), "`{absent}` is in the view");
    }
    assert!(hidden.contains("visible symbols") && hidden.contains("first answer") && hidden.contains("first question"));
    assert!(!view.checkpointed && conv.view_start == 0, "hiding is not a checkpoint");
    let notice = match &view.messages[1].content {
        crate::llm::ChatContent::Text(text) => text.clone(),
        other => panic!("{other:?}"),
    };
    assert_eq!(notice, "[System: 2 earlier tool exchange(s) of this conversation are not shown in this turn.]");
    // The request builder recognises it and moves it out of the history.
    let log = crate::context_log::ContextLog::from_conversation(view.messages.clone());
    assert!(log.history().iter().all(|message| !serde_json::to_string(message).unwrap().contains("are not shown")));
    assert!(serde_json::to_string(&log.into_messages()).unwrap().contains("2 earlier tool exchange(s)"));
    assert_eq!(conv.messages.len(), stored.len());
    // Together with a checkpoint: one notice with both numbers; batches before the
    // checkpoint are counted as omitted, not as hidden.
    conv.view_start = 4;
    let both = build(&mut conv, &policy);
    assert_eq!(
        match &both.messages[1].content { crate::llm::ChatContent::Text(text) => text.clone(), other => panic!("{other:?}") },
        "[System: 3 older message(s) are omitted from this model context; the raw history is retained. 1 earlier tool exchange(s) of this conversation are not shown in this turn.]"
    );
    let log = crate::context_log::ContextLog::from_conversation(both.messages);
    assert!(log.history().iter().all(|message| !serde_json::to_string(message).unwrap().contains("[System:")));
}
