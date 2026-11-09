//! Record authority is a per-turn adjunct; it never changes legacy turn identity.
use super::AppConfig;
use crate::{
    coroutine::{CoroutineHandle, IoRequest, TurnIdentity},
    ingress::ClaimedOrigin,
    memory_graph::records::{
        RecordStore,
        authority::{RecordCeiling, TrustedCaller},
    },
    permissions::{AccessLevel, PermissionConfig},
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

pub(crate) struct RecordTurn {
    principal: String,
    context: String,
    caller: Option<TrustedCaller>,
    store: Option<RecordStore>,
    /// The permission profile this verified turn was dispatched with. Shared
    /// conversation meta can be rewritten by a concurrent dispatch to the same
    /// context before this turn takes the context lock, so the live field alone is
    /// not the turn's own ceiling; both are intersected (narrowing only).
    turn_profile: Option<String>,
    complete: AtomicBool,
}

impl RecordTurn {
    pub(super) fn claimed(
        claim: &ClaimedOrigin,
        store: Option<RecordStore>,
        permissions: &PermissionConfig,
        process: AccessLevel,
        actual_model: &str,
        turn_profile: Option<&str>,
    ) -> Arc<Self> {
        let turn = TurnIdentity {
            principal: claim.principal().into(),
            context: claim.context().into(),
            may_write: true,
            guest_tools: None,
        };
        let caller = RecordCeiling::capture(
            &turn,
            permissions,
            process,
            Some(actual_model),
            turn_profile,
        )
        .and_then(|ceiling| TrustedCaller::from_ingress(claim, ceiling));
        Arc::new(Self {
            principal: turn.principal,
            context: turn.context,
            caller,
            store,
            turn_profile: turn_profile.map(str::to_owned),
            complete: AtomicBool::new(false),
        })
    }
    /// End of the verified turn (finished, unknown or error): every binding and
    /// handle derived from this turn is denied from now on.
    pub(super) fn revoke(&self) {
        if let Some(caller) = &self.caller {
            caller.revoke();
        }
    }
    pub(super) fn completed(&self) -> bool {
        self.complete.load(Ordering::SeqCst)
    }
    pub(super) fn confirm(&self) {
        self.complete.store(true, Ordering::SeqCst);
    }
}

impl AppConfig {
    pub(crate) fn coroutine_handle(
        &self,
        sender: tokio::sync::mpsc::Sender<IoRequest>,
    ) -> CoroutineHandle {
        let handle = CoroutineHandle::new(sender).with_turn(self.turn.clone());
        let (Some(scope), Some(turn)) = (&self.record_turn, &self.turn) else {
            return handle;
        };
        if scope.principal != turn.principal || scope.context != turn.context {
            return handle;
        }
        let (Some(caller), Some(store)) = (&scope.caller, &scope.store) else {
            return handle;
        };
        let model = self.llm_model.read().unwrap().clone();
        let profile = self
            .meta
            .lock()
            .unwrap()
            .conversations
            .get(&turn.context)
            .and_then(|c| c.permission_model.clone());
        let Some(current) = RecordCeiling::capture(
            turn,
            &self.permission_config,
            self.access_level,
            Some(&model),
            profile.as_deref(),
        )
        .and_then(|current| current.narrowed(scope.turn_profile.as_deref())) else {
            return handle;
        };
        match store.bind(turn, Some(caller), Some(&current)) {
            Ok(binding) => {
                let handle = handle.with_record_access(store.clone(), binding);
                #[cfg(test)]
                escaped::capture(store, &handle);
                handle
            }
            Err(_) => handle,
        }
    }

    pub(super) fn confirm_ingress_completion(&self) {
        if let Some(scope) = &self.record_turn {
            scope.confirm();
        }
    }
}

/// Test-only stand-in for a handle that escaped its turn (the reserved-name rule
/// removed every in-language way to copy one). Lets revocation be tested against a
/// handle minted by the real verified pipeline rather than a hand-built binding.
#[cfg(test)]
pub(crate) mod escaped {
    use crate::{coroutine::CoroutineHandle, memory_graph::records::RecordStore};
    static HANDLES: std::sync::Mutex<Vec<(String, CoroutineHandle)>> =
        std::sync::Mutex::new(Vec::new());

    pub(super) fn capture(store: &RecordStore, handle: &CoroutineHandle) {
        HANDLES
            .lock()
            .unwrap()
            .push((store.test_identity(), handle.clone()));
    }

    pub(crate) fn take(store: &RecordStore) -> Vec<CoroutineHandle> {
        let mut handles = HANDLES.lock().unwrap();
        let id = store.test_identity();
        let (mine, rest) = handles.drain(..).partition(|(owner, _)| *owner == id);
        *handles = rest;
        mine.into_iter().map(|(_, handle)| handle).collect()
    }
}
