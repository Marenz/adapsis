//! One pass of the managed writer: enrol and add dependencies, save every
//! module, remove dependencies the saved sources no longer need.
//!
//! The source files of one save are separate renames, and a start decodes each
//! module against the files of its dependencies. So the renames are ordered:
//! a changed file is written only when the set on disk, with that one file
//! replaced, still loads (`source_io::would_load`, exactly the decode a start
//! does). Dependencies are tried first, and the rest again after every write,
//! so "a module gains a function, then its new caller" and "a caller stops
//! using a function, then the module drops it" both find their order. What
//! cannot be written that way:
//! - the complete new set loads but no single file can go first (functions
//!   moved between two modules in both directions): the files are written
//!   anyway; a crash between those renames leaves a set the next start refuses
//!   (exit 78). This is the remaining gap, and it is logged.
//! - the complete new set does not load (a module it needs could not be
//!   printed or written): those modules stay pending and the message names
//!   what blocks them.
//! A save that changes no file decodes nothing.
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::Result;

use super::source::SourceManifest;
use super::source_io::Prepared;
use super::LibraryState;
use crate::ast;

/// What one round of the writer left behind. Only the last round is reported.
#[derive(Default)]
pub(super) struct Round {
    pub(super) saved: Vec<String>,
    /// Enrolment and dependency updates that did not happen.
    pub(super) unresolved: Vec<String>,
    /// Managed modules whose source was not saved.
    pub(super) unsaved: Vec<String>,
    pub(super) qualified_call_sites: Vec<(String, usize)>,
    pub(super) order_dependent: Vec<(String, String)>,
}

pub(super) struct Saved {
    pub(super) round: Round,
    pub(super) enrolled: Vec<String>,
    pub(super) notes: Vec<String>,
}

impl LibraryState {
    fn managed_manifest(&self) -> Option<SourceManifest> {
        self.managed_sources.lock().ok()?.as_ref().map(|managed| managed.manifest.clone())
    }

    /// Listed modules, dependencies first.
    fn managed_order(&self) -> Vec<String> {
        self.managed_sources.lock().ok().and_then(|managed| managed.as_ref().map(|managed| managed.order.clone())).unwrap_or_default()
    }

    /// Everything listed module `name` depends on, directly or not.
    fn managed_dependency_names(&self, name: &str) -> BTreeSet<String> {
        self.managed_sources
            .lock()
            .ok()
            .and_then(|managed| managed.as_ref().filter(|managed| managed.manifest.modules.contains_key(name)).map(|managed| managed.dependency_names(name)))
            .unwrap_or_default()
    }
}

/// Under a manifest the manifest follows the live Program. One round is: add
/// dependencies and enrol new modules, save every module, then remove
/// dependencies the saved sources no longer need. A second round runs only when
/// the first left something behind that its own saves may have unblocked (a new
/// module using a function saved a moment ago).
pub(super) fn run(live: &ast::Program, state: Option<&LibraryState>, dir: &Path, governed: bool) -> Result<Saved> {
    let mut enrolled = Vec::new();
    let mut notes = Vec::new();
    let mut attempt = 0;
    let round = loop {
        let mut round = Round::default();
        if let Some(state) = state.filter(|_| governed) {
            match super::source_enrol::grow(live, state, dir) {
                Ok(result) => {
                    for (name, reason) in result.refused {
                        state.queue_source(&name)?;
                        round.unresolved.push(format!("module `{name}` is live but has NO source and was not added to the manifest; it will be dropped at the next restart unless this is fixed: {reason}"));
                    }
                    round.unresolved.extend(result.failures);
                    enrolled.extend(result.enrolled);
                    notes.extend(result.notes);
                }
                Err(error) => round.unresolved.push(format!("new modules could not be added to the source manifest; nothing was changed: {error:#}")),
            }
        }
        if let Some(state) = state {
            Saver { state, dir, requests: BTreeMap::new(), blocked: BTreeSet::new(), round: &mut round }.save(live, &mut notes)?;
        }
        if let Some(state) = state.filter(|_| governed) {
            let saved = round.saved.iter().cloned().collect();
            match super::source_enrol::shrink(live, state, dir, &saved) {
                Ok(result) => notes.extend(result),
                Err(error) => notes.push(format!("recorded dependencies were not reduced: {error:#}")),
            }
        }
        attempt += 1;
        if attempt == 2 || (round.unresolved.is_empty() && round.unsaved.is_empty()) {
            break round;
        }
    };
    Ok(Saved { round, enrolled, notes })
}

fn unsaved(name: &str, error: &dyn std::fmt::Display) -> String {
    format!("module `{name}` remains live but source was NOT saved: {error}; preserve the session recovery checkpoint, reconcile, and restart if source changed externally")
}

struct Saver<'a> {
    state: &'a LibraryState,
    dir: &'a Path,
    /// The pending request each module had when this pass looked at it.
    requests: BTreeMap<String, u64>,
    /// Modules whose source is not going to be written in this pass.
    blocked: BTreeSet<String>,
    round: &'a mut Round,
}

impl Saver<'_> {
    /// Saves every managed module from the live Program.
    fn save(&mut self, live: &ast::Program, notes: &mut Vec<String>) -> Result<()> {
        // 1. Print and check everything first; nothing is written yet.
        let mut prepared = Vec::new();
        for name in self.state.managed_module_names() {
            self.requests.insert(name.clone(), self.state.queue_source(&name)?);
            match super::source_io::prepare_live(live, &name, self.state, self.dir) {
                Ok(Some(module)) => prepared.push(module),
                Ok(None) => {}
                Err(error) => {
                    self.round.unsaved.push(unsaved(&name, &error));
                    self.blocked.insert(name);
                }
            }
        }
        let Some(manifest) = self.state.managed_manifest() else {
            return Ok(());
        };
        let order = self.state.managed_order();
        prepared.sort_by_key(|module| order.iter().position(|listed| listed == &module.name).unwrap_or(usize::MAX));
        // 2. Unchanged files are only verified; changed files go in an order
        // that keeps the set on disk loadable.
        let (overlay, waiting) = self.write_in_order(&prepared, &manifest);
        // 3. Whatever could not go first.
        self.write_or_explain(&prepared, waiting, overlay, &manifest, notes);
        Ok(())
    }

    /// Writes one prepared module. A failed write is reported once and not
    /// tried again in this pass.
    fn write(&mut self, module: &Prepared) -> bool {
        if let Err(error) = super::source_io::commit_live(module, self.state, self.dir) {
            self.round.unsaved.push(unsaved(&module.name, &error));
            self.blocked.insert(module.name.clone());
            return false;
        }
        if module.ledger.call_sites > 0 {
            self.round.qualified_call_sites.push((module.name.clone(), module.ledger.call_sites));
        }
        self.round.order_dependent.extend(module.ledger.order_dependent.iter().map(|callee| (module.name.clone(), callee.clone())));
        if let Ok(mut pending) = self.state.pending_sources.lock() {
            // A mutation may enqueue before its CoW commit waits for our live
            // read guard. Do not consume that later request.
            if pending.get(&module.name) == self.requests.get(&module.name) {
                pending.remove(&module.name);
            }
        }
        self.round.saved.push(module.name.clone());
        true
    }

    /// Returns what the disk holds after the writes done here (`overlay`: the
    /// new texts) and the changed modules that could not be written yet.
    fn write_in_order(&mut self, prepared: &[Prepared], manifest: &SourceManifest) -> (BTreeMap<String, String>, Vec<usize>) {
        let mut overlay = BTreeMap::new();
        let mut waiting = Vec::new();
        for (index, module) in prepared.iter().enumerate() {
            if module.changed {
                waiting.push(index);
            } else {
                self.write(module);
            }
        }
        loop {
            let before = waiting.len();
            let mut still = Vec::new();
            for index in waiting {
                let module = &prepared[index];
                overlay.insert(module.name.clone(), module.text.clone());
                if super::source_io::would_load(self.dir, manifest, &overlay).is_err() {
                    overlay.remove(&module.name);
                    still.push(index);
                } else if !self.write(module) {
                    overlay.remove(&module.name);
                }
            }
            waiting = still;
            if waiting.is_empty() || waiting.len() == before {
                return (overlay, waiting);
            }
        }
    }

    /// No waiting module can go first. If the complete new set loads, the
    /// files are written one after another (the remaining gap); otherwise they
    /// stay pending with the module that blocks them.
    fn write_or_explain(
        &mut self,
        prepared: &[Prepared],
        waiting: Vec<usize>,
        overlay: BTreeMap<String, String>,
        manifest: &SourceManifest,
        notes: &mut Vec<String>,
    ) {
        if waiting.is_empty() {
            return;
        }
        let names: Vec<String> = waiting.iter().map(|index| prepared[*index].name.clone()).collect();
        let mut together = overlay;
        together.extend(waiting.iter().map(|index| (prepared[*index].name.clone(), prepared[*index].text.clone())));
        if super::source_io::would_load(self.dir, manifest, &together).is_ok() {
            notes.push(format!(
                "modules {} only load together (functions moved between them in both directions): no rename order keeps every intermediate state loadable, so their files were written one after another",
                names.iter().map(|name| format!("`{name}`")).collect::<Vec<_>>().join(", ")
            ));
            let mut failed: Option<&str> = None;
            for index in waiting {
                let module = &prepared[index];
                match failed {
                    // Do not widen a half-written set: the rest waits.
                    Some(blocker) => self.round.unsaved.push(unsaved(&module.name, &format!("it is written together with `{blocker}`, whose source write failed"))),
                    None if self.write(module) => {}
                    None => failed = Some(&module.name),
                }
            }
            return;
        }
        for index in waiting {
            let module = &prepared[index];
            let blockers: Vec<String> = self
                .state
                .managed_dependency_names(&module.name)
                .iter()
                .filter(|name| self.blocked.contains(*name) || names.contains(name))
                .map(|name| format!("`{name}`"))
                .collect();
            let mut reason = "with this file the module sources on disk would not load at the next start".to_string();
            if !blockers.is_empty() {
                reason.push_str(&format!("; it waits for {}, whose source is not saved", blockers.join(", ")));
            }
            self.round.unsaved.push(unsaved(&module.name, &reason));
        }
    }
}
