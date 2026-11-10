use super::*;
use crate::llm::ToolCall;

fn image(n: u8) -> crate::attachment::Attachment {
    crate::attachment::Attachment::from_bytes(vec![n; 3], "image/png", "reference.png").unwrap()
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
    // Each turn: user 400 chars (100 tokens) + assistant 400 chars.
    let policy = ViewPolicy::tokens_only(2_000, 1_000, 200, 10_000);
    let mut previous: Option<Vec<String>> = None;
    let mut checkpoints = 0;
    for turn in 0..40 {
        conv.push_user(format!("u{turn:02}{}", "x".repeat(397)));
        let view = build(&mut conv, &policy);
        let bodies: Vec<String> = view.messages.iter().map(|m| serde_json::to_string(m).unwrap()).collect();
        // Newest turn always present.
        assert!(bodies.last().unwrap().contains(&format!("u{turn:02}")));
        let history_tokens: usize = view.messages.iter().skip(1)
            .map(|m| crate::model_limits::estimate_tokens(serde_json::to_value(m).unwrap()["content"].as_str().map_or(0, |s| s.chars().count())))
            .sum();
        if view.checkpointed {
            checkpoints += 1;
            assert!(policy.overhead_tokens + history_tokens <= policy.target_tokens + 30,
                "after a cut the projection is within the target: {history_tokens}");
        } else if let Some(previous) = &previous {
            // Byte-identical prefix (ignoring the volatile omission notice slot).
            let strip = |v: &[String]| v.iter().filter(|b| !b.contains("older message(s) are omitted")).cloned().collect::<Vec<_>>();
            let (old, new) = (strip(previous), strip(&bodies));
            assert_eq!(&new[..old.len()], &old[..], "turn {turn}: history must only append between checkpoints");
        }
        assert!(policy.overhead_tokens + history_tokens <= policy.trigger_tokens + 100, "never far above the trigger");
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
    assert!(first.contains("content omitted"));
}

/// 30-turn coding simulation (20–50k tokens per turn, native batches of 8k-char
/// tool results) against a 1M-token window, compared with the pre-#68 rule
/// (newest 60k chars / 64 messages / 40k per message). Run with --nocapture
/// to see the numbers; assertions pin the new behaviour.
#[test]
fn conversation_view_thirty_turn_coding_simulation() {
    use crate::context_log::{Epoch, observe_epoch};
    let key = crate::context_prompt::StablePrefixKey { context: "sim".into(), may_write: true, identity_hash: 1, mesh_hash: 1 };
    // Window-bound simulation: the route body limit is raised (gateway fix), so the
    // token budget decides. The 2 MB case is covered by the byte-limit test.
    let limits = crate::model_limits::ModelLimits {
        context_tokens: 1_000_000, output_tokens: 128_000,
        request_timeout: std::time::Duration::from_secs(1), chunk_timeout: std::time::Duration::from_secs(1),
        max_request_bytes: 64_000_000,
    };
    let system = "s".repeat(36_000); // ~9k tokens, like the live admin DM prefix
    let policy = ViewPolicy::from_limits(&limits, &system);
    let mut seed = 0x2545_f491_u64;
    let mut next = move |lo: usize, hi: usize| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; lo + (seed as usize) % (hi - lo + 1) };
    let mut conv = Conversation::new();
    conv.push_system(system.clone());
    let tokens = |m: &[crate::llm::ChatMessage]| m.iter().map(|m| crate::model_limits::estimate_tokens(serde_json::to_string(m).unwrap().len())).sum::<usize>();
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
        let cost: usize = group.iter().map(chars).sum();
        if group.len() > 1 {
            if cost <= remaining && group.iter().all(|m| chars(m) <= 40_000) {
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
    };
    let system = "s".repeat(36_000);
    let policy = ViewPolicy::from_limits(&limits, &system);
    assert!(policy.image_bytes <= 450_000, "images limited to ~30% of the route in base64");
    let mut conv = Conversation::new();
    conv.push_system(system.clone());
    // ~15 MB of multi-byte history (3M chars ≈ 750k tokens, under the token
    // trigger): the byte budget alone must force the checkpoint.
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
    assert_eq!(all[0].tokens, crate::model_limits::estimate_tokens(4) + 3 * IMAGE_TOKENS);
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
