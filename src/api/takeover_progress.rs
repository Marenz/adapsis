//! Best-effort interface feedback, never conversation/model/memory content (#147).
use crate::coroutine::{IoRequest, TaskRegistry};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{OwnedMutexGuard, mpsc};

pub(super) const IMMEDIATE: &str = "Message received. I am starting to process it now.";
pub(super) const QUEUED: &str = "Message received. It is queued behind earlier work in this chat.";
pub(super) const STARTED: &str = "I am now processing your queued message.";
const ATTEMPT_TIMEOUT: Duration = Duration::from_secs(5);

pub(super) struct Progress {
    ended: Arc<Mutex<bool>>,
    events: Option<mpsc::Sender<&'static str>>,
    worker: Option<tokio::task::JoinHandle<()>>,
}

impl Progress {
    pub(super) fn new(
        function: Option<String>,
        argument: Option<String>,
        sender: mpsc::Sender<IoRequest>,
        registry: TaskRegistry,
    ) -> Self {
        let ended = Arc::new(Mutex::new(false));
        let Some((function, argument)) = function.zip(argument) else {
            return Self {
                ended,
                events: None,
                worker: None,
            };
        };
        let (events, receiver) = mpsc::channel(2);
        let worker = tokio::spawn(run(
            receiver,
            ended.clone(),
            function,
            argument,
            sender,
            registry,
        ));
        Self {
            ended,
            events: Some(events),
            worker: Some(worker),
        }
    }

    pub(super) async fn acquire(&self, lock: Arc<tokio::sync::Mutex<()>>) -> OwnedMutexGuard<()> {
        match lock.clone().try_lock_owned() {
            Ok(guard) => {
                self.emit(IMMEDIATE);
                guard
            }
            Err(_) => {
                self.emit(QUEUED);
                let guard = lock.lock_owned().await;
                self.emit(STARTED);
                guard
            }
        }
    }

    fn emit(&self, message: &'static str) {
        if let Some(events) = &self.events {
            // At most two events per turn, with no await or model involvement.
            let _ = events.try_send(message);
        }
    }

    fn end(&mut self) {
        if let Ok(mut ended) = self.ended.lock() {
            *ended = true;
        }
        self.events.take();
    }

    pub(super) async fn finish(&mut self) {
        // Suppress pending attempts BEFORE draining. No Program/runtime/meta lock
        // is held here. An in-flight Spawn cannot be cancelled: drain only its
        // observer for at most five seconds, never promise on-wire ordering.
        self.end();
        if let Some(mut worker) = self.worker.take() {
            if tokio::time::timeout(ATTEMPT_TIMEOUT, &mut worker)
                .await
                .is_err()
            {
                worker.abort();
            }
        }
    }
}

impl Drop for Progress {
    fn drop(&mut self) {
        // Also covers early errors and a cancelled takeover/lock wait.
        self.end();
    }
}

async fn run(
    mut events: mpsc::Receiver<&'static str>,
    ended: Arc<Mutex<bool>>,
    function: String,
    argument: String,
    sender: mpsc::Sender<IoRequest>,
    registry: TaskRegistry,
) {
    while let Some(message) = events.recv().await {
        let attempt = attempt(
            &ended,
            &function,
            &argument,
            message,
            &sender,
            registry.clone(),
        );
        let failure = match tokio::time::timeout(ATTEMPT_TIMEOUT, attempt).await {
            Ok(Ok(true)) => continue,
            Ok(Ok(false)) => break,
            Ok(Err(())) => "progress callback failed",
            Err(_) => "progress callback timed out",
        };
        // One fixed categorical failure per turn; never callback args/name,
        // evaluated payloads or user text. Failure never retries or fails work.
        crate::logging::diagnostic("takeover-progress", failure);
        crate::error_journal::record(
            crate::error_journal::Category::Task,
            "takeover-progress",
            failure,
        );
        break;
    }
}

async fn attempt(
    ended: &Mutex<bool>,
    function: &str,
    argument: &str,
    message: &str,
    sender: &mpsc::Sender<IoRequest>,
    registry: TaskRegistry,
) -> Result<bool, ()> {
    let permit = sender.reserve().await.map_err(|_| ())?;
    let (request, receiver) = crate::conversation_callback::request(
        function.to_string(),
        vec![
            crate::eval::Value::string(argument),
            crate::eval::Value::string(message),
        ],
    );
    {
        // End/check/launch is one synchronous critical section. In particular,
        // channel backpressure cannot launch a queued notice after turn end.
        let ended = ended.lock().map_err(|_| ())?;
        if *ended {
            return Ok(false);
        }
        permit.send(request);
    }
    crate::await_spawned_callback_with_timeout(receiver, registry, ATTEMPT_TIMEOUT)
        .await
        .map(|_| true)
        .map_err(|_| ())
}

#[cfg(test)]
#[path = "takeover_progress_lifecycle_tests.rs"]
mod tests;
