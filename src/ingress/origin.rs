//! Purpose-limited event admission reference, never a general record capability.
use super::{IngressError, VerifiedIngress, state::IntakeStore};
use std::sync::{Arc, Mutex};

struct Lease {
    proof: VerifiedIngress,
    /// Taken by the first takeover attempt (successful or not). Afterwards the
    /// origin is spent: handles stop propagating it, and long-lived descendants
    /// keep only this small spent shell, not the intake store.
    state: Mutex<Option<IntakeStore>>,
    epoch: i64,
    event: i64,
    digest: String,
}

/// One pointer wide: handles (and so `eval::Value`) carry it by value.
#[derive(Clone)]
pub(crate) struct InvocationOrigin {
    lease: Arc<Lease>,
}

pub(crate) struct ClaimedOrigin {
    proof: Arc<Lease>,
    state: IntakeStore,
    claim: String,
}

impl std::fmt::Debug for InvocationOrigin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("InvocationOrigin(<sealed>)")
    }
}

impl InvocationOrigin {
    pub(super) fn admitted(proof: VerifiedIngress, state: IntakeStore) -> Self {
        let lease = Lease {
            state: Mutex::new(Some(state)),
            epoch: proof.epoch,
            event: proof.update_id,
            digest: proof.digest.clone(),
            proof,
        };
        Self {
            lease: Arc::new(lease),
        }
    }

    /// A spent origin no longer authorizes anything and must not be forwarded.
    pub(crate) fn is_spent(&self) -> bool {
        self.lease
            .state
            .lock()
            .map_or(true, |state| state.is_none())
    }

    pub(crate) async fn consume(
        &self,
        context: &str,
        message: &str,
        metadata: Option<&str>,
        reply_fn: Option<&str>,
        reply_arg: Option<&str>,
    ) -> Result<ClaimedOrigin, IngressError> {
        // Spend first: any attempt, including a mismatched one, ends propagation.
        let state = self
            .lease
            .state
            .lock()
            .map_err(|_| IngressError::StateUnavailable)?
            .take()
            .ok_or(IngressError::AlreadyStarted)?;
        let speaker = metadata
            .and_then(|text| serde_json::from_str::<serde_json::Value>(text).ok())
            .and_then(|value| {
                value
                    .get("speaker_id")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            });
        if !self.lease.proof.agrees(
            speaker.as_deref().unwrap_or(""),
            context,
            message,
            reply_fn.unwrap_or(""),
            reply_arg.unwrap_or(""),
        ) {
            return Err(IngressError::Mismatch);
        }
        let lease = self.lease.clone();
        let claim = uuid::Uuid::new_v4().to_string();
        let attempt = claim.clone();
        let state = tokio::task::spawn_blocking(move || {
            state
                .takeover(lease.epoch, lease.event, &lease.digest, &attempt)
                .map(|()| state)
        })
        .await
        .map_err(|_| IngressError::StateUnknown)??;
        Ok(ClaimedOrigin {
            proof: self.lease.clone(),
            state,
            claim,
        })
    }
}

impl ClaimedOrigin {
    pub(crate) fn principal(&self) -> &str {
        &self.proof.proof.principal
    }
    pub(crate) fn context(&self) -> &str {
        &self.proof.proof.context
    }
    pub(crate) async fn finish(self, confirmed: bool) -> Result<(), IngressError> {
        let Self {
            proof,
            state,
            claim,
        } = self;
        tokio::task::spawn_blocking(move || {
            state.finish_takeover(proof.epoch, proof.event, &claim, confirmed)
        })
        .await
        .map_err(|_| IngressError::StateUnknown)?
    }
}
