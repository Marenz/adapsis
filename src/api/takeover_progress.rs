//! Best-effort interface feedback, never conversation/model/memory content (#147).
use crate::coroutine::{IoRequest, TaskRegistry};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{OwnedMutexGuard, mpsc};

pub(super) const QUEUED: &str = "Your message is waiting behind earlier work in this chat.";
pub(super) const STARTED: &str = "I am now processing your queued message.";
const QUEUE_NOTICE_AFTER: Duration = Duration::from_secs(10);
const ATTEMPT_TIMEOUT: Duration = Duration::from_secs(5);

#[cfg(test)]
tokio::task_local! { pub(super) static TEST_QUEUE_DELAY: Duration; }

#[derive(Default)]
struct State {
    ended: bool,
    waiting: bool,
    queue_launched: bool,
}

struct Callback {
    function: String,
    argument: String,
    sender: mpsc::Sender<IoRequest>,
    registry: TaskRegistry,
}

pub(super) struct Progress {
    state: Arc<Mutex<State>>,
    callback: Option<Callback>,
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
        Self {
            state: Arc::new(Mutex::new(State::default())),
            callback: function.zip(argument).map(|(function, argument)| Callback {
                function,
                argument,
                sender,
                registry,
            }),
            events: None,
            worker: None,
        }
    }

    fn notify_waiting(&mut self) {
        let Some(callback) = self.callback.take() else {
            return;
        };
        let (events, receiver) = mpsc::channel(2);
        let worker = tokio::spawn(run(
            receiver,
            self.state.clone(),
            callback.function,
            callback.argument,
            callback.sender,
            callback.registry,
        ));
        self.events = Some(events);
        self.worker = Some(worker);
        self.emit(QUEUED);
    }

    pub(super) async fn acquire(
        &mut self,
        lock: Arc<tokio::sync::Mutex<()>>,
    ) -> OwnedMutexGuard<()> {
        let delay = QUEUE_NOTICE_AFTER;
        #[cfg(test)]
        let delay = TEST_QUEUE_DELAY.try_with(|delay| *delay).unwrap_or(delay);
        self.acquire_after(lock, tokio::time::sleep(delay)).await
    }

    async fn acquire_after(
        &mut self,
        lock: Arc<tokio::sync::Mutex<()>>,
        notice_after: impl std::future::Future<Output = ()>,
    ) -> OwnedMutexGuard<()> {
        if let Ok(guard) = lock.clone().try_lock_owned() {
            return guard;
        }
        let acquire = lock.lock_owned();
        tokio::pin!(acquire);
        tokio::select! {
            biased;
            guard = &mut acquire => return guard,
            _ = notice_after => {},
        }
        if let Ok(mut state) = self.state.lock() {
            state.waiting = true;
        }
        self.notify_waiting();
        let guard = acquire.await;
        let queue_launched = self
            .state
            .lock()
            .map(|mut state| {
                state.waiting = false;
                state.queue_launched
            })
            .unwrap_or(false);
        if queue_launched {
            self.emit(STARTED);
        }
        guard
    }

    fn emit(&self, message: &'static str) {
        if let Some(events) = &self.events {
            // At most two events per turn, with no await or model involvement.
            let _ = events.try_send(message);
        }
    }

    fn end(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            state.ended = true;
            state.waiting = false;
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
    state: Arc<Mutex<State>>,
    function: String,
    argument: String,
    sender: mpsc::Sender<IoRequest>,
    registry: TaskRegistry,
) {
    while let Some(message) = events.recv().await {
        let attempt = attempt(
            &state,
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
    state: &Mutex<State>,
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
        let mut state = state.lock().map_err(|_| ())?;
        if state.ended
            || (message == QUEUED && !state.waiting)
            || (message == STARTED && !state.queue_launched)
        {
            return Ok(false);
        }
        if message == QUEUED {
            state.queue_launched = true;
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
