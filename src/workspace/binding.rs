//! Which workspace a conversation works in, and what it has read there.
//!
//! One private record per conversation, `<state root>/_bindings/<hash>.json`,
//! holding the repository and workspace NAMES and the read ledger. Nothing is
//! kept in memory between turns: a restart of this process, or a rewind of the
//! session, changes nothing here. A `Workspace` is not stored; the caller
//! binds again with `lifecycle::find`, which checks the worktree still exists.
//!
//! One workspace has one writer (DESIGN T4): a workspace is bound to at most
//! one conversation. The rules that keep that true:
//!
//! * A workspace is taken with [`BindingStore::claim`], one transaction under
//!   the workspace's reservation (the lock job start and close share): find
//!   every holder, get each one idle, remove their records durably, publish
//!   the claimant's record. Two claims on one workspace run one after the
//!   other, and the second finds the first as holder.
//! * A conversation's record is written or removed only while that
//!   conversation is held: by its own running tool call, or by a claimant that
//!   got it idle through the `idle` callback and keeps the returned guards
//!   until its own record is published.
//! * Everything in `_bindings` must be a record this code can attribute to a
//!   conversation. Anything else is an error for every claim, because it may
//!   be the record of the holder.
//!
//! The reservation and the guards are process-local. A second process on the
//! same state root is not excluded: one active daemon per state root.
//!
//! The file name is a hash of the conversation key, so neither a file name
//! nor an error text shows who a conversation is. `_bindings` cannot collide
//! with a repository directory: repository names have no underscore.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::error::{Error, Result};
use super::job_store::state_root;
use super::ledger::ReadLedger;
use super::lifecycle::Workspace;
use super::limits::{BINDING_CONTEXT_MAX_BYTES, BINDING_MAX_BYTES};
use super::state_io::Dir;

const DIRECTORY: &str = "_bindings";
const VERSION: u32 = 1;
/// In-flight atomic writes of `state_io` (`.tmp-<pid>-<n>`): not records.
const TEMPORARY_PREFIX: &str = ".tmp-";

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Binding {
    pub repo: String,
    pub workspace: String,
    pub ledger: ReadLedger,
}

impl Binding {
    pub fn new(repo: &str, workspace: &str) -> Self {
        Self { repo: repo.to_string(), workspace: workspace.to_string(), ledger: ReadLedger::default() }
    }

    fn is_for(&self, workspace: &Workspace) -> bool {
        self.repo == workspace.repo && self.workspace == workspace.name
    }
}

#[derive(Serialize, Deserialize)]
struct Record {
    version: u32,
    /// The conversation key, checked on load: two keys never share a record.
    context: String,
    repo: String,
    workspace: String,
    ledger: ReadLedger,
}

/// Proof that another conversation is idle (no turn, no running tool call),
/// kept alive for as long as it must stay idle.
pub type IdleGuard = Box<dyn std::any::Any + Send>;

/// What a successful claim did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Claimed {
    /// The workspace was bound to another conversation, which lost it.
    pub taken: bool,
}

/// Where a claim is in its transaction; tests stop it there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Step {
    /// Holders are idle; nothing has been changed.
    BeforeRemoval,
    /// The old holders' records are removed durably; nothing is published.
    AfterRemoval,
    /// The claimant's record is published; the old holders are still held.
    AfterPublish,
}

#[cfg(test)]
pub(crate) type Hook = std::sync::Arc<dyn Fn(Step) -> std::io::Result<()> + Send + Sync>;

#[derive(Clone)]
pub struct BindingStore {
    root: PathBuf,
    #[cfg(test)]
    writes: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    #[cfg(test)]
    pub(crate) hook: Option<Hook>,
}

impl std::fmt::Debug for BindingStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BindingStore").finish_non_exhaustive()
    }
}

impl BindingStore {
    /// The store of this process, under the standard state root.
    pub fn standard() -> Self {
        Self::at(&state_root())
    }

    pub fn at(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            #[cfg(test)]
            writes: Default::default(),
            #[cfg(test)]
            hook: None,
        }
    }

    /// A point of the claim transaction (a no-op outside tests).
    fn step(&self, step: Step) -> Result<()> {
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            return hook(step).map_err(state_error);
        }
        let _ = step;
        Ok(())
    }

    /// The error for an entry of `_bindings` that cannot be used. It names
    /// the file, because the remedy is the owner's: repair or remove it. The
    /// name is a hash (or whatever a stranger put there), never a conversation key.
    fn unusable(&self, name: &str, reason: &str) -> Error {
        let path = self.root.join(DIRECTORY).join(super::error::shown(name));
        journal_once(path.to_string_lossy().into_owned(), "binding", "workspace binding record unreadable");
        Error::Binding(format!(
            "the workspace binding record {} cannot be used ({reason}); nothing may be assumed unbound and nothing was changed. If it cannot be repaired the owner removes that file; the conversation it belonged to then has no workspace",
            path.display()
        ))
    }

    fn directory(&self, create: bool) -> Result<Option<Dir>> {
        let Some(root) = Dir::root(&self.root, create).map_err(state_error)? else { return Ok(None) };
        root.child(OsStr::new(DIRECTORY), create).map_err(state_error)
    }

    /// Read and check one entry of the directory. `None`: it is gone.
    fn record(&self, directory: &Dir, name: &str) -> Result<Option<Record>> {
        let Some(bytes) = directory.read(name, BINDING_MAX_BYTES).map_err(|error| self.unusable(name, &error.kind().to_string()))? else {
            return Ok(None);
        };
        let record: Record = serde_json::from_slice(&bytes).map_err(|_| self.unusable(name, "it is not a binding record"))?;
        if record.version != VERSION {
            return Err(self.unusable(name, &format!("it has version {}, this build reads version {VERSION}", record.version)));
        }
        if file_name(&record.context).ok().as_deref() != Some(name) {
            return Err(self.unusable(name, "its file name does not belong to the conversation it names"));
        }
        Ok(Some(record))
    }

    /// The binding of `context`, or `None` when it has none.
    pub fn load(&self, context: &str) -> Result<Option<Binding>> {
        let name = file_name(context)?;
        let Some(directory) = self.directory(false)? else { return Ok(None) };
        Ok(self.record(&directory, &name)?.map(|record| Binding { repo: record.repo, workspace: record.workspace, ledger: record.ledger }))
    }

    fn write(&self, directory: &Dir, context: &str, binding: &Binding) -> Result<()> {
        let record = Record {
            version: VERSION,
            context: context.to_string(),
            repo: binding.repo.clone(),
            workspace: binding.workspace.clone(),
            ledger: binding.ledger.clone(),
        };
        let bytes = serde_json::to_vec(&record).map_err(|_| Error::Binding("the workspace binding could not be encoded".into()))?;
        if bytes.len() as u64 > BINDING_MAX_BYTES {
            return Err(Error::Binding(format!("the workspace binding would exceed {BINDING_MAX_BYTES} bytes and was not stored")));
        }
        // Temporary file, fsync, rename, directory fsync.
        directory.write(&file_name(context)?, &bytes).map_err(state_error)?;
        #[cfg(test)]
        self.writes.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Every record of the directory with its file name. Fails closed: an
    /// entry that is not an attributable record is an error (it may be the
    /// record of a holder). Only our own in-flight temporary files are skipped.
    fn records(&self, directory: &Dir) -> Result<Vec<Record>> {
        let mut records = Vec::new();
        for name in directory.list().map_err(state_error)? {
            let Some(name) = name.to_str() else { return Err(self.unusable(&name.to_string_lossy(), "its name is not valid UTF-8")) };
            if name.starts_with(TEMPORARY_PREFIX) {
                continue;
            }
            // A record removed between the listing and the read held nothing.
            if let Some(record) = self.record(directory, name)? {
                records.push(record);
            }
        }
        Ok(records)
    }

    /// The conversations bound to `repo`/`workspace` (at most one in normal
    /// operation). Fails closed like every scan of the directory.
    pub fn holders(&self, repo: &str, workspace: &str) -> Result<Vec<String>> {
        let Some(directory) = self.directory(false)? else { return Ok(Vec::new()) };
        let mut holders: Vec<String> =
            self.records(&directory)?.into_iter().filter(|record| record.repo == repo && record.workspace == workspace).map(|record| record.context).collect();
        holders.sort();
        Ok(holders)
    }

    /// Make `workspace` the workspace of `context`, with an empty ledger, in
    /// one transaction under the workspace's reservation.
    ///
    /// `idle(other)` must return a guard only if conversation `other` is in no
    /// turn and runs no tool call, and must keep it so while the guard lives;
    /// it must not wait. The caller must hold `context` itself in the same
    /// sense. Order, with the guards held to the end:
    ///
    /// 1. read every record (fail closed);
    /// 2. get every other holder idle, else refuse with nothing changed;
    /// 3. remove their records and fsync the directory;
    /// 4. publish the claimant's record (which replaces its earlier one).
    ///
    /// A crash or an error between 3 and 4 leaves the workspace bound to
    /// nobody: no conversation can then change it on the strength of an old
    /// ledger, and `claim` gets it back. Two records never name one workspace.
    pub fn claim(&self, context: &str, workspace: &Workspace, idle: &dyn Fn(&str) -> Option<IdleGuard>) -> Result<Claimed> {
        file_name(context)?;
        let reservation = super::reservation::for_workspace(workspace);
        let _reserved = reservation.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let directory = self.directory(true)?.ok_or_else(|| Error::Binding("the workspace state directory could not be created".into()))?;
        let mut held: Vec<(String, IdleGuard)> = Vec::new();
        for record in self.records(&directory)? {
            if record.context == context || record.repo != workspace.repo || record.workspace != workspace.name {
                continue;
            }
            let Some(guard) = idle(&record.context) else {
                return Err(Error::Lifecycle(format!(
                    "workspace {}/{} is in use by another conversation that is working right now; nothing was changed. Try again when it has finished",
                    workspace.repo, workspace.name
                )));
            };
            held.push((record.context, guard));
        }
        self.step(Step::BeforeRemoval)?;
        for (holder, _) in &held {
            directory.remove_file_synced(OsStr::new(&file_name(holder)?)).map_err(state_error)?;
        }
        self.step(Step::AfterRemoval)?;
        self.write(&directory, context, &Binding::new(&workspace.repo, &workspace.name))?;
        self.step(Step::AfterPublish)?;
        Ok(Claimed { taken: !held.is_empty() })
    }

    /// Store what `context` has read in `workspace`, if `context` still holds
    /// it. Returns false, and writes nothing, when its record no longer binds
    /// that workspace (it was taken, or closed): a late writer must not bring
    /// a binding back.
    pub fn save_ledger(&self, context: &str, workspace: &Workspace, ledger: &ReadLedger) -> Result<bool> {
        let reservation = super::reservation::for_workspace(workspace);
        let _reserved = reservation.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(directory) = self.directory(false)? else { return Ok(false) };
        match self.record(&directory, &file_name(context)?)? {
            Some(record) if record.repo == workspace.repo && record.workspace == workspace.name => {}
            _ => return Ok(false),
        }
        let binding = Binding { repo: workspace.repo.clone(), workspace: workspace.name.clone(), ledger: ledger.clone() };
        self.write(&directory, context, &binding)?;
        Ok(true)
    }

    /// Remove the binding of `context` durably; removing none is not an
    /// error. With `only`, the record is removed only if it binds that
    /// workspace (after a close: never a newer binding).
    pub fn release(&self, context: &str, only: Option<&Workspace>) -> Result<()> {
        let name = file_name(context)?;
        let reservation = only.map(super::reservation::for_workspace);
        let _reserved = reservation.as_ref().map(|lock| lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner()));
        let Some(directory) = self.directory(false)? else { return Ok(()) };
        if let Some(workspace) = only {
            match self.load(context)? {
                Some(binding) if binding.is_for(workspace) => {}
                _ => return Ok(()),
            }
        }
        directory.remove_file_synced(OsStr::new(&name)).map_err(state_error)
    }

    /// Test fixture: a record without any of the claim rules.
    #[cfg(test)]
    pub(crate) fn put(&self, context: &str, binding: &Binding) -> Result<()> {
        let directory = self.directory(true)?.ok_or_else(|| Error::Binding("no state directory".into()))?;
        self.write(&directory, context, binding)
    }

    /// The state root: what scopes this store's conversations.
    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    #[cfg(test)]
    pub(crate) fn writes_for_tests(&self) -> usize {
        self.writes.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Journal `text` the first time `key` is seen in this process. The key (a
/// path) only deduplicates; the entry carries no path and no conversation key.
fn journal_once(key: String, source: &'static str, text: &'static str) {
    static SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> = std::sync::OnceLock::new();
    let mut seen = SEEN.get_or_init(Default::default).lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    // Bounded: past this many distinct records the journal already says enough.
    if seen.len() < 1024 && seen.insert(key) {
        crate::error_journal::record(crate::error_journal::Category::Workspace, source, text);
    }
}

/// `<hash of the conversation key>.json`. The key itself is only inside the
/// record, so neither the directory listing nor an error names a conversation.
pub(crate) fn file_name(context: &str) -> Result<String> {
    if context.is_empty() || context.len() > BINDING_CONTEXT_MAX_BYTES {
        return Err(Error::Binding(format!("a conversation key must have 1 to {BINDING_CONTEXT_MAX_BYTES} bytes to bind a workspace")));
    }
    let digest = Sha256::digest(context.as_bytes());
    let digest: String = digest[..16].iter().map(|byte| format!("{byte:02x}")).collect();
    Ok(format!("{digest}.json"))
}

fn state_error(error: std::io::Error) -> Error {
    journal_once(format!("state directory: {}", error.kind()), "binding-state", "workspace state directory unavailable");
    Error::Binding(format!("the workspace state directory is unavailable ({}); nothing may be assumed unbound", error.kind()))
}

#[cfg(test)]
#[path = "binding_tests.rs"]
mod tests;
