//! Shared start/close reservation. Lock order: lifecycle-global (where used),
//! then workspace reservation, then private-state descriptor operations. Internal
//! list/prune/forget helpers do not reacquire a reservation already held.
use super::lifecycle::Workspace;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock, Weak};

pub fn for_workspace(ws: &Workspace) -> Arc<Mutex<()>> {
    static LOCKS: OnceLock<Mutex<HashMap<PathBuf, Weak<Mutex<()>>>>> = OnceLock::new();
    let mut locks = LOCKS.get_or_init(Default::default).lock().unwrap_or_else(|p| p.into_inner());
    locks.retain(|_, value| value.strong_count() > 0);
    let key = ws.path.clone();
    if let Some(lock) = locks.get(&key).and_then(Weak::upgrade) {
        return lock;
    }
    let lock = Arc::new(Mutex::new(()));
    locks.insert(key, Arc::downgrade(&lock));
    lock
}
