//! Cancellation of EXISTING foreground workspace operations. Only direct
//! loopback HTTP callers may request it; no caller-supplied identity is used.
//! Entries live until the blocking worker finishes cleanup, not until its
//! requesting future is dropped. A later Telegram control can use this same
//! registry without creating a second cancellation channel.
use crate::workspace::Cancel;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

type Key = (PathBuf, String);
type Entries = HashMap<Key, Arc<Cancel>>;
fn registry() -> &'static Mutex<Entries> {
    static ENTRIES: OnceLock<Mutex<Entries>> = OnceLock::new();
    ENTRIES.get_or_init(Default::default)
}

pub(super) struct Active {
    key: Key,
    pub cancel: Arc<Cancel>,
}

impl Active {
    /// Caller already owns the conversation's work lock; never replace an
    /// entry that still belongs to an earlier running worker.
    pub(super) fn register(root: &Path, context: &str) -> Result<Self, &'static str> {
        let key = (root.to_path_buf(), context.to_string());
        let mut entries = registry().lock().unwrap_or_else(|p| p.into_inner());
        if entries.contains_key(&key) {
            return Err("an earlier foreground operation is still finishing; nothing was started");
        }
        let cancel = Arc::new(Cancel::new());
        entries.insert(key.clone(), cancel.clone());
        Ok(Self { key, cancel })
    }
}

impl Drop for Active {
    fn drop(&mut self) {
        let mut entries = registry().lock().unwrap_or_else(|p| p.into_inner());
        if entries
            .get(&self.key)
            .is_some_and(|entry| Arc::ptr_eq(entry, &self.cancel))
        {
            entries.remove(&self.key);
        }
    }
}

/// Kept in the async CALL future, separate from Active in the worker. Even
/// cancelling only this future (without dropping the whole turn) stops bash.
pub(super) struct OnDrop(pub Arc<Cancel>);
impl Drop for OnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

fn request(root: &Path, context: &str) -> bool {
    let entries = registry().lock().unwrap_or_else(|p| p.into_inner());
    match entries.get(&(root.to_path_buf(), context.to_string())) {
        Some(cancel) => {
            cancel.cancel();
            true
        }
        None => false,
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    context: String,
}

pub(super) async fn cancel(
    origin: super::origin::Origin,
    axum::extract::State(config): axum::extract::State<super::AppConfig>,
    axum::Json(input): axum::Json<Request>,
) -> (axum::http::StatusCode, axum::Json<serde_json::Value>) {
    use axum::http::StatusCode;
    let reply = |status, text| (status, axum::Json(serde_json::json!({"status": text})));
    // Same Origin extractor as ask: forwarding headers can only narrow. No
    // context lookup, entry creation, work start or replay on a refusal.
    if origin != super::origin::Origin::Local {
        return reply(StatusCode::FORBIDDEN, "local requests only");
    }
    if input.context.is_empty()
        || input.context.len() > crate::workspace::limits::BINDING_CONTEXT_MAX_BYTES
    {
        return reply(StatusCode::BAD_REQUEST, "invalid context length");
    }
    let store = config
        .workspaces
        .as_ref()
        .map(|site| site.bindings.clone())
        .unwrap_or_else(crate::workspace::BindingStore::standard);
    if request(store.root(), &input.context) {
        // This is a request, not proof that supervision has stopped/reaped yet.
        reply(
            StatusCode::ACCEPTED,
            "cancellation requested; stopped state is not yet confirmed",
        )
    } else {
        reply(
            StatusCode::NOT_FOUND,
            "no active foreground workspace operation",
        )
    }
}

#[cfg(test)]
#[path = "workspace_cancel_tests.rs"]
mod tests;
