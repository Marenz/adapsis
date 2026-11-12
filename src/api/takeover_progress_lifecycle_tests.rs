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
        let acquire = progress.acquire(lock);
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
    let _guard = progress
        .acquire(Arc::new(tokio::sync::Mutex::new(())))
        .await;
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
        let progress = Progress::new(
            Some("Bound.reply".into()),
            Some("destination".into()),
            sender,
            Default::default(),
        );
        let _guard = progress.acquire(lock).await;
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
