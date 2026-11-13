//! Real takeover + configured callback regressions; all transport is intercepted.
use super::*;
use crate::api::takeover_progress::{QUEUED, STARTED};
#[path = "takeover_progress_fixtures.rs"]
mod fixtures;
use fixtures::*;

const REPLY: &str = "ProgressFixture.reply";
const OLD_IMMEDIATE: &str = "Message received. I am starting to process it now.";

#[tokio::test]
async fn takeover_progress_immediate_lock_stays_silent_even_while_provider_is_held() {
    for (native, error) in [(true, false), (false, false), (true, true), (false, true)] {
        let harness = Arc::new(Harness::new(native));
        let io = Arc::new(CallbackIo::new(&harness, false).await);
        let (provider, release) = held_provider(native, error).await;
        let provider = Arc::new(provider);
        let context = format!("telegram:progress-immediate-{native}-{error}");
        let turn = {
            let (h, io, p, context) = (
                harness.clone(),
                io.clone(),
                provider.clone(),
                context.clone(),
            );
            tokio::spawn(async move {
                io.turn(
                    &h,
                    &p,
                    &context,
                    ADMIN,
                    "original input",
                    (Some(REPLY), Some("bound-destination")),
                )
                .await
            })
        };
        until(|| provider.requests.lock().unwrap().len() == 1).await;
        assert!(!turn.is_finished(), "provider is held");
        assert!(io.launches.lock().unwrap().is_empty());
        assert!(io.delivered.lock().unwrap().is_empty());
        // A different conversation's stored callback must not be consulted.
        {
            let mut meta = harness.meta.lock().unwrap();
            let other = meta
                .conversations
                .get_or_create("telegram:progress-foreign");
            other.reply_fn = Some("PrivateForeign.callback".into());
            other.reply_arg = Some("foreign-destination".into());
        }
        release.add_permits(1);
        let reply = turn.await.unwrap();
        if error {
            assert!(reply.contains("technisches Problem"));
        } else {
            assert_eq!(reply, "final answer");
        }
        assert!(io.launches.lock().unwrap().is_empty());
        let requests = provider.requests.lock().unwrap().clone();
        assert_eq!(requests.len(), 1);
        assert_eq!(all_text(&requests[0]).matches("original input").count(), 1);
        assert_clean(&harness, &io, &provider, &context).await;
    }
}

#[tokio::test]
async fn takeover_progress_queued_receipt_precedes_lock_start_and_keeps_pinned_callback() {
    for native in [true, false] {
        let harness = Arc::new(Harness::new(native));
        let io = Arc::new(CallbackIo::new(&harness, false).await);
        let (provider, release) = held_provider(native, false).await;
        let provider = Arc::new(provider);
        let context = format!("telegram:progress-queue-{native}");
        let spawn = |input: &'static str, destination: &'static str| {
            let (h, io, p, context) = (
                harness.clone(),
                io.clone(),
                provider.clone(),
                context.clone(),
            );
            tokio::spawn(async move {
                io.turn(
                    &h,
                    &p,
                    &context,
                    ADMIN,
                    input,
                    (Some(REPLY), Some(destination)),
                )
                .await
            })
        };
        let first = spawn("first turn", "first-destination");
        until(|| provider.requests.lock().unwrap().len() == 1).await;
        let second = spawn("queued turn", "second-destination");
        until(|| io.delivered.lock().unwrap().len() == 1).await;
        assert_eq!(
            provider.requests.lock().unwrap().len(),
            1,
            "queued turn must not call provider"
        );
        assert_eq!(io.launches.lock().unwrap()[0].message, QUEUED);
        assert!(
            !harness
                .meta
                .lock()
                .unwrap()
                .conversations
                .get(&context)
                .unwrap()
                .messages
                .iter()
                .any(|m| m.content == "queued turn")
        );
        {
            let mut meta = harness.meta.lock().unwrap();
            let conv = meta.conversations.get_mut(&context).unwrap();
            conv.reply_fn = Some("Foreign.reply".into());
            conv.reply_arg = Some("wrong-destination".into());
        }
        release.add_permits(1);
        assert_eq!(first.await.unwrap(), "final answer");
        until(|| {
            io.delivered.lock().unwrap().len() == 2 && provider.requests.lock().unwrap().len() == 2
        })
        .await;
        assert_eq!(
            io.launches.lock().unwrap()[1],
            Launch {
                function: REPLY.into(),
                destination: "second-destination".into(),
                message: STARTED.into(),
            }
        );
        release.add_permits(1);
        assert_eq!(second.await.unwrap(), "final answer");
        assert_eq!(io.launches.lock().unwrap().len(), 2);
        assert_clean(&harness, &io, &provider, &context).await;
    }
}

#[tokio::test]
async fn takeover_progress_short_queue_is_silent_in_native_and_legacy_turns() {
    for native in [true, false] {
        let harness = Harness::new(native);
        let mut io = CallbackIo::new(&harness, false).await;
        io.queue_delay = std::time::Duration::from_secs(86_400);
        let (provider, release) = held_provider(native, false).await;
        let context = format!("telegram:progress-short-queue-{native}");
        let first = io.turn(
            &harness,
            &provider,
            &context,
            ADMIN,
            "first",
            (Some(REPLY), Some("first")),
        );
        tokio::pin!(first);
        tokio::select! {
            _ = &mut first => panic!("first provider is held"),
            _ = until(|| provider.requests.lock().unwrap().len() == 1) => {},
        }
        let second = io.turn(
            &harness,
            &provider,
            &context,
            ADMIN,
            "second",
            (Some(REPLY), Some("second")),
        );
        tokio::pin!(second);
        assert!(futures::poll!(&mut second).is_pending());
        assert_eq!(provider.requests.lock().unwrap().len(), 1);
        release.add_permits(2);
        let (first, second) = tokio::join!(first, second);
        assert_eq!(
            (first.as_str(), second.as_str()),
            ("final answer", "final answer")
        );
        assert!(io.launches.lock().unwrap().is_empty());
        assert!(io.delivered.lock().unwrap().is_empty());
        assert_clean(&harness, &io, &provider, &context).await;
    }
}

#[tokio::test]
async fn takeover_progress_callback_failure_does_not_hide_provider_failure_or_retry() {
    if !crate::logging::tests::child() {
        let captured = crate::logging::tests::capture(
            "api::takeover_native::tests::progress::takeover_progress_callback_failure_does_not_hide_provider_failure_or_retry",
            false,
        );
        assert_eq!(captured.matches("[takeover-progress]").count(), 4);
        for private in [
            "CALLBACK_PRIVATE_ERROR",
            "ProgressFixture.fail",
            "private destination",
            "private failure input",
        ] {
            assert!(!captured.contains(private));
        }
        return;
    }
    let before: Value = serde_json::from_str(&crate::error_journal::query(u64::MAX - 1)).unwrap();
    for (native, error) in [(true, true), (false, true), (true, false), (false, false)] {
        let harness = Harness::new(native);
        let io = CallbackIo::new(&harness, false).await;
        let provider = provider(vec![if error {
            json!({"http_status":400})
        } else {
            output(native, 0)
        }])
        .await;
        let reply = io
            .queued_turn(
                &harness,
                &provider,
                &format!("telegram:progress-error-{native}-{error}"),
                ADMIN,
                "private failure input",
                (Some("ProgressFixture.fail"), Some("private destination")),
            )
            .await;
        if error {
            assert!(reply.contains("technisches Problem"));
        } else {
            assert_eq!(reply, "final answer");
        }
        assert_eq!(provider.requests.lock().unwrap().len(), 1);
        assert_eq!(io.launches.lock().unwrap().len(), 1);
        assert!(io.delivered.lock().unwrap().is_empty());
        assert!(
            io.registry
                .lock()
                .unwrap()
                .values()
                .all(|task| matches!(task.status, crate::coroutine::WaitReason::Failed(_)))
        );
    }
    let journal: Value = serde_json::from_str(&crate::error_journal::query(
        before["last_seq"].as_u64().unwrap(),
    ))
    .unwrap();
    let progress: Vec<_> = journal["entries"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|entry| entry["source"] == "takeover-progress")
        .collect();
    assert_eq!(
        progress.len(),
        4,
        "one categorical failure for each failed callback, with no retry"
    );
    for entry in progress {
        assert!(matches!(
            entry["message"].as_str(),
            Some("progress callback failed" | "progress callback timed out")
        ));
        let text = entry.to_string();
        for private in [
            "CALLBACK_PRIVATE_ERROR",
            "ProgressFixture.fail",
            "private destination",
            "private failure input",
        ] {
            assert!(!text.contains(private));
        }
    }
}

#[tokio::test]
async fn takeover_progress_quick_turns_are_silent_and_never_use_stored_callback() {
    for pair in [
        (None, None),
        (Some(REPLY), None),
        (None, Some("destination")),
        (Some(REPLY), Some("destination")),
    ] {
        let harness = Harness::new(true);
        let io = CallbackIo::new(&harness, false).await;
        let context = format!(
            "telegram:progress-missing-{}-{}",
            pair.0.is_some(),
            pair.1.is_some()
        );
        {
            let mut meta = harness.meta.lock().unwrap();
            let conv = meta.conversations.get_or_create(&context);
            conv.reply_fn = Some(REPLY.into());
            conv.reply_arg = Some("stored-destination".into());
        }
        let provider = provider(vec![output(true, 0)]).await;
        assert_eq!(
            io.turn(&harness, &provider, &context, ADMIN, "hello", pair)
                .await,
            "final answer"
        );
        assert!(io.launches.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn takeover_progress_callback_timeout_is_bounded_and_does_not_retry_provider() {
    let harness = Harness::new(true);
    let io = CallbackIo::new(&harness, true).await;
    let (provider, release) = held_provider(true, true).await;
    let turn = io.queued_turn(
        &harness,
        &provider,
        "telegram:progress-timeout",
        ADMIN,
        "timeout input",
        (Some(REPLY), Some("destination")),
    );
    tokio::pin!(turn);
    let started = std::time::Instant::now();
    tokio::select! {
        result = &mut turn => panic!("provider is held: {result}"),
        _ = until(|| io.launches.lock().unwrap().len() == 1 && provider.requests.lock().unwrap().len() == 1) => {},
    }
    release.add_permits(1);
    let reply = tokio::time::timeout(std::time::Duration::from_secs(7), &mut turn)
        .await
        .unwrap();
    assert!(reply.contains("technisches Problem"));
    assert!(started.elapsed() < std::time::Duration::from_secs(7));
    assert_eq!(io.launches.lock().unwrap().len(), 1);
    assert_eq!(provider.requests.lock().unwrap().len(), 1);
    assert!(io.delivered.lock().unwrap().is_empty());
    // Already-launched callback may deliver late; no second attempt is started.
    io.gate.add_permits(1);
    until(|| io.delivered.lock().unwrap().len() == 1).await;
    assert_eq!(io.launches.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn takeover_progress_fast_provider_error_suppresses_unlaunched_queued_start() {
    let harness = Arc::new(Harness::new(true));
    let io = Arc::new(CallbackIo::new(&harness, true).await);
    let (provider, release) = held_provider(true, true).await;
    let provider = Arc::new(provider);
    let context = "telegram:progress-fast-error";
    let first = {
        let (h, io, p) = (harness.clone(), io.clone(), provider.clone());
        tokio::spawn(async move { io.turn(&h, &p, context, ADMIN, "first", (None, None)).await })
    };
    until(|| provider.requests.lock().unwrap().len() == 1).await;
    let second = {
        let (h, io, p) = (harness.clone(), io.clone(), provider.clone());
        tokio::spawn(async move {
            io.turn(
                &h,
                &p,
                context,
                ADMIN,
                "queued",
                (Some(REPLY), Some("queued-destination")),
            )
            .await
        })
    };
    until(|| io.launches.lock().unwrap().len() == 1).await;
    assert_eq!(io.launches.lock().unwrap()[0].message, QUEUED);
    assert_eq!(provider.requests.lock().unwrap().len(), 1);
    release.add_permits(2);
    assert!(first.await.unwrap().contains("technisches Problem"));
    until(|| provider.requests.lock().unwrap().len() == 2).await;
    // Keep receipt delivery blocked until the bounded observer expires; no
    // wall-clock sleep is used to guess when the provider error has completed.
    assert!(second.await.unwrap().contains("technisches Problem"));
    assert_eq!(
        io.launches.lock().unwrap().len(),
        1,
        "pending STARTED was suppressed"
    );
    assert_eq!(provider.requests.lock().unwrap().len(), 2);
    io.gate.add_permits(1);
    until(|| io.delivered.lock().unwrap().len() == 1).await;
}

#[tokio::test]
async fn takeover_progress_guest_callback_does_not_widen_conversational_authority() {
    let mut harness = Harness::new(true);
    let principal = "telegram:user:progress-guest";
    let mut permissions = crate::permissions::PermissionConfig::default();
    permissions
        .principals
        .insert(principal.into(), "progress-guest".into());
    permissions
        .guest_profiles
        .insert("progress-guest".into(), vec!["web_search".into()]);
    harness.permissions = Arc::new(permissions);
    let io = CallbackIo::new(&harness, false).await;
    let provider = provider(vec![output(false, 0)]).await;
    assert_eq!(
        io.turn(
            &harness,
            &provider,
            "telegram:progress-guest",
            principal,
            "hello",
            (Some(REPLY), Some("guest-destination"))
        )
        .await,
        "final answer"
    );
    assert!(io.launches.lock().unwrap().is_empty());
    let requests = provider.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert!(
        requests[0].get("tools").is_none(),
        "existing guest legacy routing"
    );
    let request = all_text(&requests[0]);
    assert!(!request.contains("Add or modify code: yes"));
    assert!(!request.contains("CALLBACK_PRIVATE_ERROR"));
    assert!(!request.contains(OLD_IMMEDIATE));
}

async fn assert_clean(harness: &Harness, io: &CallbackIo, provider: &Provider, context: &str) {
    let history = serde_json::to_string(
        &harness
            .meta
            .lock()
            .unwrap()
            .conversations
            .get(context)
            .unwrap(),
    )
    .unwrap();
    let graph = harness
        .graph
        .pending_context_messages(context, 100_000)
        .unwrap();
    let graph_text: String = graph
        .iter()
        .map(|message| message.content.as_str())
        .collect();
    let training = tokio::fs::read_to_string(io.training_file.path())
        .await
        .unwrap();
    for notice in [OLD_IMMEDIATE, QUEUED, STARTED] {
        assert!(!history.contains(notice));
        assert!(!graph_text.contains(notice));
        assert!(!training.contains(notice));
        assert!(
            provider
                .requests
                .lock()
                .unwrap()
                .iter()
                .all(|request| !all_text(request).contains(notice))
        );
    }
}
