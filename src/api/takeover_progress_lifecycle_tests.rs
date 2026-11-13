//! Deterministic lifecycle races independent of provider or callback latency.
use super::*;
use crate::coroutine::{TaskInfo, WaitReason};

#[tokio::test]
async fn takeover_progress_ended_before_queued_start_never_launches_the_second_attempt() {
    let (sender, mut receiver) = mpsc::channel(2);
    let registry = TaskRegistry::default();
    let lock = Arc::new(tokio::sync::Mutex::new(()));
    let held = lock.clone().lock_owned().await;
    let mut progress = Progress::new(
        Some("Bound.reply".into()),
        Some("destination".into()),
        sender,
        registry.clone(),
    );
    let guard = {
        let acquire = progress.acquire_after(lock, std::future::ready(()));
        tokio::pin!(acquire);
        let request = tokio::select! {
            _ = &mut acquire => panic!("must wait for context lock"),
            request = receiver.recv() => request.unwrap(),
        };
        let IoRequest::Spawn { reply, args, .. } = request else {
            panic!("expected callback")
        };
        assert!(matches!(&args[1], crate::eval::Value::String(text) if text.as_str() == QUEUED));
        registry.lock().unwrap().insert(
            1,
            TaskInfo {
                id: 1,
                function_name: "Bound.reply".into(),
                status: WaitReason::Running,
                started_at: "0".into(),
            },
        );
        reply.send(Ok(1)).unwrap();
        drop(held);
        acquire.await
    };
    // STARTED has been emitted after acquisition, but receipt is still in flight.
    progress.end();
    registry.lock().unwrap().get_mut(&1).unwrap().status =
        WaitReason::Completed("attempted".into());
    progress.finish().await;
    assert!(receiver.try_recv().is_err());
    drop(guard);
}

#[tokio::test]
async fn takeover_progress_backpressure_cannot_launch_after_end() {
    let (sender, mut receiver) = mpsc::channel(1);
    let (reply, _answer) = tokio::sync::oneshot::channel();
    sender.send(IoRequest::GetLlmModel { reply }).await.unwrap();
    let mut progress = Progress::new(
        Some("Bound.reply".into()),
        Some("destination".into()),
        sender,
        Default::default(),
    );
    progress.state.lock().unwrap().waiting = true;
    progress.notify_waiting();
    tokio::task::yield_now().await;
    progress.end();
    assert!(matches!(
        receiver.recv().await,
        Some(IoRequest::GetLlmModel { .. })
    ));
    tokio::time::timeout(Duration::from_millis(200), progress.finish())
        .await
        .unwrap();
    assert!(receiver.try_recv().is_err());
}

#[tokio::test]
async fn takeover_progress_cancelled_lock_wait_drops_pending_start_and_never_retries() {
    let (sender, mut receiver) = mpsc::channel(2);
    let lock = Arc::new(tokio::sync::Mutex::new(()));
    let held = lock.clone().lock_owned().await;
    let turn = tokio::spawn(async move {
        let mut progress = Progress::new(
            Some("Bound.reply".into()),
            Some("destination".into()),
            sender,
            Default::default(),
        );
        let _guard = progress.acquire_after(lock, std::future::ready(())).await;
    });
    let request = receiver.recv().await.unwrap();
    turn.abort();
    assert!(turn.await.unwrap_err().is_cancelled());
    drop(held);
    let IoRequest::Spawn { reply, .. } = request else {
        panic!("expected queued receipt")
    };
    let _ = reply.send(Err(anyhow::anyhow!("callback failure with PRIVATE_ARGS")));
    tokio::task::yield_now().await;
    assert!(receiver.try_recv().is_err());
}

#[tokio::test]
async fn takeover_progress_immediate_path_starts_no_worker() {
    assert_eq!(QUEUE_NOTICE_AFTER, Duration::from_secs(10));
    let (sender, mut receiver) = mpsc::channel(2);
    let mut progress = Progress::new(
        Some("Bound.reply".into()),
        Some("destination".into()),
        sender,
        Default::default(),
    );
    let _guard = progress
        .acquire(Arc::new(tokio::sync::Mutex::new(())))
        .await;
    assert!(progress.worker.is_none());
    assert!(progress.events.is_none());
    progress.finish().await;
    assert!(receiver.try_recv().is_err());
}

#[tokio::test]
async fn takeover_progress_available_lock_wins_when_threshold_is_also_ready() {
    let (sender, mut receiver) = mpsc::channel(2);
    let mut progress = Progress::new(
        Some("Bound.reply".into()),
        Some("destination".into()),
        sender,
        Default::default(),
    );
    let lock = Arc::new(tokio::sync::Mutex::new(()));
    let held = lock.clone().lock_owned().await;
    let (fire, timer) = tokio::sync::oneshot::channel();
    let guard = {
        let acquire = progress.acquire_after(lock, async {
            timer.await.unwrap();
        });
        tokio::pin!(acquire);
        assert!(futures::poll!(&mut acquire).is_pending());
        drop(held);
        fire.send(()).unwrap();
        acquire.await
    };
    assert!(progress.worker.is_none());
    assert!(receiver.try_recv().is_err());
    drop(guard);
}

#[tokio::test]
async fn takeover_progress_acquired_lock_suppresses_queued_send_under_backpressure() {
    let (sender, mut receiver) = mpsc::channel(1);
    let (reply, _answer) = tokio::sync::oneshot::channel();
    sender.send(IoRequest::GetLlmModel { reply }).await.unwrap();
    let mut progress = Progress::new(
        Some("Bound.reply".into()),
        Some("destination".into()),
        sender,
        Default::default(),
    );
    let lock = Arc::new(tokio::sync::Mutex::new(()));
    let held = lock.clone().lock_owned().await;
    let guard = {
        let acquire = progress.acquire_after(lock, std::future::ready(()));
        tokio::pin!(acquire);
        assert!(futures::poll!(&mut acquire).is_pending());
        tokio::task::yield_now().await;
        drop(held);
        acquire.await
    };
    assert!(!progress.state.lock().unwrap().queue_launched);
    assert!(!progress.state.lock().unwrap().waiting);
    assert!(matches!(
        receiver.recv().await,
        Some(IoRequest::GetLlmModel { .. })
    ));
    // Keep the turn active: WAITING, not just end, must prevent the false notice.
    tokio::task::yield_now().await;
    assert!(receiver.try_recv().is_err());
    progress.finish().await;
    assert!(receiver.try_recv().is_err());
    drop(guard);
}

#[tokio::test]
async fn takeover_progress_cancelled_pending_timer_leaves_no_worker_or_notice() {
    let (sender, mut receiver) = mpsc::channel(2);
    let mut progress = Progress::new(
        Some("Bound.reply".into()),
        Some("destination".into()),
        sender,
        Default::default(),
    );
    let lock = Arc::new(tokio::sync::Mutex::new(()));
    let _held = lock.clone().lock_owned().await;
    let (fire, timer) = tokio::sync::oneshot::channel();
    {
        let acquire = progress.acquire_after(lock, async {
            let _ = timer.await;
        });
        tokio::pin!(acquire);
        assert!(futures::poll!(&mut acquire).is_pending());
    }
    assert!(fire.send(()).is_err(), "the local timer future was dropped");
    assert!(progress.worker.is_none());
    progress.finish().await;
    assert!(receiver.try_recv().is_err());
}
