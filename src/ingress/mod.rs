//! Finite authenticated receive substrate. Construction/activation is fixture-only.
//! No caller attestation issuance, task policy, live poll loop or API builtin.
#![allow(dead_code)]
mod state;
mod strict_json;
mod telegram_poll;
#[cfg(test)]
mod tests;

use state::IntakeStore;
use std::sync::Arc;

const PAGE_LIMIT: usize = 8;
const EVENT_LIMIT: usize = 16;
const EVENT_BYTES: usize = 32 * 1024;
const SOURCE_BYTES: usize = 256 * 1024;
const INCARNATION_LIMIT: i64 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IngressError {
    Transport,
    Protocol,
    WrongBot,
    Oversized,
    Capacity,
    StaleSource,
    AlteredReplay,
    StateUnavailable,
    StateUnknown,
    AlreadyStarted,
    Mismatch,
    Conflict,
}

#[derive(Clone, serde::Serialize)]
struct PrivateRoute {
    sender_id: i64,
    context: String,
    reply_handler: String,
}

struct SourceConfig {
    id: String,
    handler: String,
    bot_id: i64,
    credential: String,
    routes: Vec<PrivateRoute>,
    seed: i64,
    #[cfg(test)]
    endpoint: reqwest::Url,
}

/// Not serializable or constructible by an Adapsis source/event/payload.
pub(crate) struct VerifiedIngress {
    source: String,
    update_id: i64,
    epoch: i64,
    digest: String,
    principal: String,
    context: String,
    text: String,
    reply_handler: String,
    reply_arg: String,
}

impl std::fmt::Debug for VerifiedIngress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("VerifiedIngress(<private>)")
    }
}

impl VerifiedIngress {
    pub(crate) fn agrees(
        &self,
        principal: &str,
        context: &str,
        text: &str,
        reply_handler: &str,
        reply_arg: &str,
    ) -> bool {
        self.principal == principal
            && self.context == context
            && self.text == text
            && self.reply_handler == reply_handler
            && self.reply_arg == reply_arg
    }
}

/// Delivery data plus opaque origin evidence. Evidence is NOT a record grant.
pub(crate) struct Delivery {
    pub(crate) payload: String,
    proof: Option<VerifiedIngress>,
    update_id: i64,
    digest: String,
    epoch: i64,
    source: Arc<()>,
}

pub(crate) struct Admitted {
    pub(crate) handler: String,
    pub(crate) payload: String,
    pub(crate) proof: Option<VerifiedIngress>,
}

pub(crate) struct Received {
    pub(crate) events: Vec<Delivery>,
    pub(crate) unknown: usize,
}

pub(crate) struct PollSource {
    config: Arc<SourceConfig>,
    state: IntakeStore,
    epoch: i64,
    receiver: tokio::sync::Mutex<()>,
    stamp: Arc<()>,
}

impl PollSource {
    pub(crate) async fn receive_once(&self) -> Result<Received, IngressError> {
        let _single_receive = self.receiver.lock().await;
        let pending = self
            .state_work(false, |state, epoch| state.pending(epoch))
            .await?;
        if !pending.events.is_empty() {
            return self.deliveries(pending);
        }
        let cursor = self
            .state_work(false, |state, epoch| state.cursor(epoch))
            .await?;
        let events = telegram_poll::fetch(&self.config, cursor).await?;
        let pending = self
            .state_work(true, move |state, epoch| {
                for event in events {
                    state.accept(epoch, &event)?;
                }
                state.pending(epoch)
            })
            .await?;
        self.deliveries(pending)
    }

    async fn state_work<T: Send + 'static>(
        &self,
        write: bool,
        work: impl FnOnce(IntakeStore, i64) -> Result<T, IngressError> + Send + 'static,
    ) -> Result<T, IngressError> {
        let state = self.state.clone();
        let epoch = self.epoch;
        tokio::task::spawn_blocking(move || work(state, epoch))
            .await
            .map_err(|_| {
                if write {
                    IngressError::StateUnknown
                } else {
                    IngressError::StateUnavailable
                }
            })?
    }

    fn deliveries(&self, pending: state::Pending) -> Result<Received, IngressError> {
        let mut events = Vec::new();
        for event in pending.events {
            let proof = telegram_poll::private_text(&self.config, &event, self.epoch)?;
            events.push(Delivery {
                payload: event.raw,
                proof,
                update_id: event.id,
                digest: event.digest,
                epoch: self.epoch,
                source: self.stamp.clone(),
            });
        }
        Ok(Received {
            events,
            unknown: pending.unknown,
        })
    }

    /// Exactly one admission to handler dispatch. No delivery/LLM-success claim.
    pub(crate) async fn admit(&self, event: Delivery) -> Result<Admitted, IngressError> {
        if !Arc::ptr_eq(&event.source, &self.stamp)
            || event.epoch != self.epoch
            || telegram_poll::digest(&event.payload) != event.digest
        {
            return Err(IngressError::Mismatch);
        }
        let id = event.update_id;
        let digest = event.digest.clone();
        self.state_work(true, move |state, epoch| state.start(epoch, id, &digest))
            .await?;
        Ok(Admitted {
            handler: self.config.handler.clone(),
            payload: event.payload,
            proof: event.proof,
        })
    }

    #[cfg(test)]
    fn fixture(
        config: SourceConfig,
        store: crate::memory_graph::records::RecordStore,
        namespace: &str,
    ) -> Result<Self, IngressError> {
        let state = IntakeStore::fixture(store, namespace, &config.id);
        let identity = telegram_poll::source_identity(&config)?;
        let epoch = state.open(&identity, config.seed)?;
        Ok(Self {
            config: Arc::new(config),
            state,
            epoch,
            receiver: tokio::sync::Mutex::new(()),
            stamp: Arc::new(()),
        })
    }
}
