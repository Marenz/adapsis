//! The manifest follows the live Program (#128).
//!
//! Under an active manifest the managed writer enrols every live module the
//! manifest does not list yet, and keeps every module's dependency set equal
//! to what its code references. Authorization happens where a module enters
//! the live Program (mutation entry points); nothing here grants authority.
//!
//! Two rules keep every manifest this module writes loadable with the source
//! files that are on disk at that moment:
//!
//! 1. Dependencies are only ADDED before a module's source is saved (`grow`)
//!    and only REMOVED after it was saved (`shrink`). A recorded dependency the
//!    source no longer needs is harmless; a missing one refuses the next start.
//! 2. Every commit is preceded by `source_io::would_load`: the listed sources
//!    as they are on disk (plus the first text of modules being enrolled) are
//!    decoded against the new manifest exactly as a start would decode them.
//!    A change that does not pass is not committed and stays pending.
//!
//! Write order per enrolment: (1) the module's `.py`, (2) the manifest (the
//! commit point), (3) the session, later, by the autosave. A crash after (1)
//! leaves an unlisted file that a start ignores; after (2) the source is
//! authoritative. A module that is only in the session cache is handled at
//! startup by `adopt_cached`: enrolled when it prints, otherwise dropped to a
//! recovery file.
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{Context, Result};

use super::source::{never_enrolled, ManagedSources, ModuleSource, SourceFormat, SourceManifest};
use super::source_unsaved::classify;
use super::{LibraryState, Unsaved, UnsavedKind};
use crate::{ast, python_frontend};

const MAX_MODULES: usize = 128;

#[derive(Debug, Default)]
pub(super) struct Reconciled {
    /// Modules added to the manifest by this pass.
    pub(super) enrolled: Vec<String>,
    /// Unlisted live modules that were NOT enrolled, with the reason.
    pub(super) refused: BTreeMap<String, Unsaved>,
    /// Listed modules whose new dependencies could not be recorded.
    pub(super) failures: Vec<(String, Unsaved)>,
    /// Categorical notes for the log (dependency updates, files moved aside).
    pub(super) notes: Vec<String>,
}

fn lock<T>(mutex: &std::sync::Mutex<T>) -> Result<std::sync::MutexGuard<'_, T>> {
    mutex.lock().map_err(|_| anyhow::anyhow!("managed source state lock unavailable"))
}

fn cycle_text(name: &str, partners: &BTreeSet<String>) -> String {
    let others = partners.iter().map(|p| format!("`{p}`")).collect::<Vec<_>>().join(", ");
    format!(
        "module `{name}` and {others} call each other; cyclic module dependencies cannot be loaded. \
         Break the cycle: move the functions both sides need into a third module that neither of them calls back from, then save again"
    )
}

const WAITS: &str = "it is retried with the next save";

fn set(source: &ModuleSource) -> BTreeSet<String> {
    source.dependencies.iter().cloned().collect()
}

fn declared(current: &SourceManifest, name: &str) -> BTreeSet<String> {
    current.modules.get(name).map(set).unwrap_or_default()
}

/// Everything reachable from `start` through `edges`.
fn reachable(start: &str, edges: &BTreeMap<String, BTreeSet<String>>) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    let mut pending: Vec<String> = edges.get(start).into_iter().flatten().cloned().collect();
    while let Some(next) = pending.pop() {
        if seen.insert(next.clone()) {
            pending.extend(edges.get(&next).into_iter().flatten().cloned());
        }
    }
    seen
}

/// Every node that reaches itself, with the nodes on its cycles.
fn on_cycles(edges: &BTreeMap<String, BTreeSet<String>>) -> BTreeMap<String, BTreeSet<String>> {
    let mut cyclic = BTreeMap::new();
    for name in edges.keys() {
        let reach = reachable(name, edges);
        if reach.contains(name) {
            let partners: BTreeSet<String> = reach
                .iter()
                .filter(|other| *other != name && reachable(other, edges).contains(name))
                .cloned()
                .collect();
            cyclic.insert(name.clone(), partners);
        }
    }
    cyclic
}

/// Unlisted live modules the writer would enrol.
fn candidates(program: &ast::Program, current: &SourceManifest) -> BTreeSet<String> {
    program
        .modules
        .iter()
        .map(|m| m.name.clone())
        .filter(|name| !current.modules.contains_key(name) && !never_enrolled(name))
        .collect()
}

/// Dependencies as the code has them, for every live module in scope, plus the
/// candidates whose dependencies cannot be determined. A listed module whose
/// dependencies cannot be determined keeps its declared set.
fn derive(
    program: &ast::Program,
    current: &SourceManifest,
    candidates: &BTreeSet<String>,
) -> (BTreeMap<String, BTreeSet<String>>, BTreeSet<String>) {
    let mut derived = BTreeMap::new();
    let mut undetermined = BTreeSet::new();
    for name in current.modules.keys().chain(candidates.iter()) {
        if !program.modules.iter().any(|m| &m.name == name) {
            continue;
        }
        match python_frontend::module_dependencies(program, name) {
            Ok(dependencies) => {
                derived.insert(name.clone(), dependencies);
            }
            Err(_) if current.modules.contains_key(name) => {}
            Err(_) => {
                undetermined.insert(name.clone());
            }
        }
    }
    (derived, undetermined)
}

/// Cycles in the code itself: every module in scope that reaches itself
/// through what the code references, with the modules on its cycles.
fn code_cycles(
    current: &SourceManifest,
    candidates: &BTreeSet<String>,
    derived: &BTreeMap<String, BTreeSet<String>>,
) -> BTreeMap<String, BTreeSet<String>> {
    let scope: BTreeSet<String> = current.modules.keys().cloned().chain(candidates.iter().cloned()).collect();
    let edges = scope
        .iter()
        .map(|name| {
            let wanted = derived.get(name).cloned().unwrap_or_else(|| declared(current, name));
            (name.clone(), wanted.intersection(&scope).cloned().collect())
        })
        .collect();
    on_cycles(&edges)
}

/// The manifest the writer wants before the saves: candidates listed with the
/// dependencies of their code, listed modules with their recorded dependencies
/// PLUS what their code needs now. `frozen` listed modules keep their entry.
fn grown(
    current: &SourceManifest,
    candidates: &BTreeSet<String>,
    derived: &BTreeMap<String, BTreeSet<String>>,
    frozen: &BTreeSet<String>,
) -> SourceManifest {
    let scope: BTreeSet<String> = current.modules.keys().cloned().chain(candidates.iter().cloned()).collect();
    let mut desired = current.clone();
    for name in &scope {
        let (Some(wanted), false) = (derived.get(name), frozen.contains(name)) else {
            continue;
        };
        let mut dependencies: BTreeSet<String> = wanted.intersection(&scope).cloned().collect();
        let format = match current.modules.get(name) {
            Some(listed) => {
                dependencies.extend(listed.dependencies.iter().cloned());
                if dependencies == set(listed) {
                    continue;
                }
                listed.format
            }
            None => SourceFormat::Python,
        };
        desired.modules.insert(name.clone(), ModuleSource { format, dependencies: dependencies.into_iter().collect() });
    }
    desired
}

fn edges(manifest: &SourceManifest) -> BTreeMap<String, BTreeSet<String>> {
    manifest.modules.iter().map(|(name, source)| (name.clone(), set(source))).collect()
}

/// The entries of `desired` that pass `would_load`: all of them when the whole
/// manifest loads, otherwise each changed module on its own, dependencies
/// first, on top of the ones accepted before it. Returns the accepted manifest
/// and the rejected module names.
fn loadable(
    dir: &Path,
    current: &SourceManifest,
    desired: &SourceManifest,
    overlay: &BTreeMap<String, String>,
) -> Result<(SourceManifest, BTreeSet<String>)> {
    if super::source_io::would_load(dir, desired, overlay).is_ok() {
        return Ok((desired.clone(), BTreeSet::new()));
    }
    let order = ManagedSources::trial(desired.clone())?.order;
    let mut accepted = current.clone();
    let mut rejected = BTreeSet::new();
    for name in order {
        if current.modules.get(&name) == desired.modules.get(&name) {
            continue;
        }
        let mut trial = accepted.clone();
        trial.modules.insert(name.clone(), desired.modules[&name].clone());
        if super::source_io::would_load(dir, &trial, overlay).is_ok() {
            accepted = trial;
        } else {
            rejected.insert(name);
        }
    }
    Ok((accepted, rejected))
}

fn dependency_notes(current: &SourceManifest, updated: &SourceManifest, notes: &mut Vec<String>) {
    for (name, source) in &updated.modules {
        if current.modules.get(name).is_some_and(|old| set(old) != set(source)) {
            notes.push(format!("manifest dependencies of `{name}` are now [{}]", source.dependencies.join(", ")));
        }
    }
}

/// Before the per-module saves: enrol new modules and ADD the dependencies the
/// code needs now. Must run inside the source transaction, with the Program the
/// caller has a read guard on (or, at startup, the not yet shared Program).
pub(super) fn grow(program: &ast::Program, state: &LibraryState, dir: &Path) -> Result<Reconciled> {
    grow_as(program, state, dir, Step::Write)
}

#[derive(Clone, Copy, PartialEq)]
enum Step {
    /// Decide everything, write nothing: `enrolled` is what `Write` would enrol.
    Plan,
    Write,
}

fn grow_as(program: &ast::Program, state: &LibraryState, dir: &Path, step: Step) -> Result<Reconciled> {
    let mut out = Reconciled::default();
    let Some(managed) = lock(&state.managed_sources)?.clone() else {
        return Ok(out);
    };
    let current = &managed.manifest;
    let live: BTreeMap<&str, &ast::Module> = program.modules.iter().map(|m| (m.name.as_str(), m)).collect();
    let mut candidates = candidates(program, current);
    if current.modules.len() + candidates.len() > MAX_MODULES {
        for name in std::mem::take(&mut candidates) {
            out.refused.insert(name, Unsaved::new(UnsavedKind::Other, format!("the source manifest is limited to {MAX_MODULES} modules")));
        }
    }
    let (derived, undetermined) = derive(program, current, &candidates);
    for name in undetermined {
        candidates.remove(&name);
        out.refused.insert(name, Unsaved::new(UnsavedKind::Other, "its module dependencies could not be determined (ambiguous type owner)"));
    }
    // Cycles in the code: a new module is not enrolled, a listed one keeps its
    // recorded dependencies. Both sides get the explanation.
    let mut frozen = BTreeSet::new();
    for (name, partners) in &code_cycles(current, &candidates, &derived) {
        let kind = UnsavedKind::Cycle(partners.iter().cloned().collect());
        if candidates.remove(name) {
            out.refused.insert(name.clone(), Unsaved::new(kind, cycle_text(name, partners)));
        } else {
            frozen.insert(name.clone());
            out.failures.push((name.clone(), Unsaved::new(kind, format!("{}; its recorded dependencies were left unchanged", cycle_text(name, partners)))));
        }
    }
    // Plan, then the checked print of every candidate against the manifest it
    // would join. Whatever leaves the plan can invalidate the rest of it.
    let (desired, texts) = loop {
        let desired = grown(current, &candidates, &derived, &frozen);
        // A cycle that exists only through a recorded dependency some module no
        // longer needs: wait until that module's source is saved without it.
        let stale: Vec<String> = on_cycles(&edges(&desired))
            .into_keys()
            .filter(|name| current.modules.get(name) != desired.modules.get(name))
            .collect();
        if !stale.is_empty() {
            for name in stale {
                let reason = format!("its new dependencies wait for another module's source save (a dependency that module no longer needs is still recorded); {WAITS}");
                if candidates.remove(&name) {
                    out.refused.insert(name, Unsaved::new(UnsavedKind::WaitsFor(Vec::new()), reason));
                } else {
                    frozen.insert(name.clone());
                    out.failures.push((name.clone(), Unsaved::new(UnsavedKind::WaitsFor(Vec::new()), format!("module `{name}`: {reason}"))));
                }
            }
            continue;
        }
        let trial = ManagedSources::trial(desired.clone())?;
        let mut texts = BTreeMap::new();
        let mut dropped = false;
        for name in &candidates {
            let printed = super::source_io::dependency_catalog(program, &trial, name).and_then(|mut catalog| {
                let module = live[name.as_str()];
                catalog.modules.push(module.clone());
                catalog.rebuild_function_index();
                super::source_io::checked_python(program, &catalog, module, name, SourceFormat::Python)
            });
            match printed {
                Ok((text, _)) => {
                    texts.insert(name.clone(), text);
                }
                Err(error) => {
                    out.refused.insert(name.clone(), classify(&error, UnsavedKind::Unprintable));
                    dropped = true;
                }
            }
        }
        if !dropped {
            break (desired, texts);
        }
        candidates.retain(|name| !out.refused.contains_key(name));
    };
    if desired.same_policy(current) {
        return Ok(out);
    }
    commit_growth(state, dir, &managed, &desired, candidates, &texts, &mut out, step)?;
    Ok(out)
}

/// Commits only what a start could load together with the sources on disk:
/// (1) the first source of every module being enrolled, (2) the manifest.
fn commit_growth(
    state: &LibraryState,
    dir: &Path,
    managed: &ManagedSources,
    desired: &SourceManifest,
    mut candidates: BTreeSet<String>,
    texts: &BTreeMap<String, String>,
    out: &mut Reconciled,
    step: Step,
) -> Result<()> {
    let current = &managed.manifest;
    let (accepted, rejected) = loadable(dir, current, desired, texts)?;
    for name in rejected {
        // What it waits for: the modules its new text depends on.
        let needs = UnsavedKind::WaitsFor(desired.modules.get(&name).map(|source| source.dependencies.clone()).unwrap_or_default());
        if candidates.remove(&name) {
            out.refused.insert(name, Unsaved::new(needs, format!("its source does not load yet together with the module sources on disk (a module it uses has unsaved changes); {WAITS}")));
        } else {
            out.failures.push((name.clone(), Unsaved::new(needs, format!("module `{name}`: its new dependencies could not be recorded because the module sources on disk would not load with them; {WAITS}"))));
        }
    }
    if accepted.same_policy(current) {
        return Ok(());
    }
    if step == Step::Plan {
        out.enrolled = candidates.into_iter().collect();
        return Ok(());
    }
    for name in &candidates {
        for aside in super::source::write_new_source(dir, name, &texts[name])
            .with_context(|| format!("could not write the first source of module `{name}`"))?
        {
            out.notes.push(format!("unlisted leftover file moved aside: {aside}"));
        }
    }
    dependency_notes(current, &accepted, &mut out.notes);
    let updated = managed.write_updated(dir, accepted).context("could not update the source manifest; it is unchanged")?;
    {
        let mut hashes = lock(&state.file_hashes)?;
        for name in &candidates {
            hashes.insert(name.clone(), super::source_hash(&texts[name]));
        }
    }
    *lock(&state.managed_sources)? = Some(updated);
    out.enrolled = candidates.into_iter().collect();
    Ok(())
}

/// After the per-module saves: REMOVE recorded dependencies that the saved
/// source of a module no longer needs. `saved` are the modules whose source on
/// disk is the live module. Returns notes for the log. A removal that would not
/// load is skipped: a recorded dependency too many is harmless.
pub(super) fn shrink(
    program: &ast::Program,
    state: &LibraryState,
    dir: &Path,
    saved: &BTreeSet<String>,
) -> Result<Vec<String>> {
    let mut notes = Vec::new();
    let Some(managed) = lock(&state.managed_sources)?.clone() else {
        return Ok(notes);
    };
    let current = &managed.manifest;
    let listed: BTreeSet<String> = current.modules.keys().cloned().collect();
    let mut desired = current.clone();
    for name in saved {
        let Some(source) = current.modules.get(name) else {
            continue;
        };
        if !program.modules.iter().any(|m| &m.name == name) {
            continue;
        }
        let Ok(derived) = python_frontend::module_dependencies(program, name) else {
            continue;
        };
        let wanted: BTreeSet<String> = derived.intersection(&listed).cloned().collect();
        let recorded = set(source);
        if wanted != recorded && wanted.is_subset(&recorded) {
            desired.modules.insert(
                name.clone(),
                ModuleSource { format: source.format, dependencies: wanted.into_iter().collect() },
            );
        }
    }
    if desired.same_policy(current) {
        return Ok(notes);
    }
    let (accepted, _kept) = loadable(dir, current, &desired, &BTreeMap::new())?;
    if accepted.same_policy(current) {
        return Ok(notes);
    }
    dependency_notes(current, &accepted, &mut notes);
    let updated = managed.write_updated(dir, accepted).context("could not update the source manifest; it is unchanged")?;
    *lock(&state.managed_sources)? = Some(updated);
    Ok(notes)
}

/// Startup: modules that exist only in the session cache. A printable one is
/// enrolled (the cache is the only copy of that work); anything else is
/// removed from the Program before it can run, with its AST kept in a
/// recovery file. Every adoption and every drop is logged and journaled.
///
/// Order: (1) plan without writing, (2) write the recovery file of every module
/// the plan will not enrol, (3) only then commit the enrolment. A recovery file
/// that cannot be written refuses the start in step (2), before any source or
/// manifest write. An error after step (3) is possible only when the enrolment
/// write itself failed AND the recovery file of a module it should have
/// enrolled cannot be written either; an unlisted `.py` may then remain (a
/// start ignores it).
pub(super) fn adopt_cached(program: &mut ast::Program, dir: &Path, state: &LibraryState) -> Result<BTreeSet<String>> {
    let listed = state.managed_module_names();
    let unlisted: Vec<String> =
        program.modules.iter().map(|m| m.name.clone()).filter(|name| !listed.contains(name)).collect();
    if unlisted.is_empty() {
        return Ok(BTreeSet::new());
    }
    let planned: BTreeSet<String> =
        grow_as(program, state, dir, Step::Plan).map(|plan| plan.enrolled.into_iter().collect()).unwrap_or_default();
    let mut kept = BTreeMap::new();
    let keep = |program: &ast::Program, name: &String, kept: &mut BTreeMap<String, String>| -> Result<()> {
        if kept.contains_key(name) || never_enrolled(name) {
            return Ok(());
        }
        let module = program.modules.iter().find(|m| &m.name == name).context("unlisted module vanished")?;
        let recovery = write_recovery(dir, module).with_context(|| {
            format!("cached module `{}` has no source and its recovery file could not be written", super::source::diagnostic_name(name))
        })?;
        kept.insert(name.clone(), recovery);
        Ok(())
    };
    for name in unlisted.iter().filter(|name| !planned.contains(*name)) {
        keep(program, name, &mut kept)?;
    }
    let (enrolled, refused): (BTreeSet<String>, BTreeMap<String, String>) = match grow_as(program, state, dir, Step::Write) {
        Ok(result) => {
            for note in &result.notes {
                eprintln!("[library] {note}");
            }
            (result.enrolled.into_iter().collect(), result.refused.into_iter().map(|(name, unsaved)| (name, unsaved.detail)).collect())
        }
        Err(error) => (BTreeSet::new(), unlisted.iter().map(|name| (name.clone(), format!("enrolment failed: {error:#}"))).collect()),
    };
    // Only after a failed enrolment write: planned modules that were not enrolled.
    for name in unlisted.iter().filter(|name| !enrolled.contains(*name)) {
        keep(program, name, &mut kept)?;
    }
    for name in &unlisted {
        let shown = super::source::diagnostic_name(name);
        if enrolled.contains(name) {
            let module = program.modules.iter().find(|m| &m.name == name).context("adopted module vanished")?;
            let routes = module.routes.clone();
            let functions = module.functions.len();
            program.pending_routes.extend(routes);
            eprintln!("[library] adopted cached module `{shown}` into the source manifest ({functions} function(s)); its source file was written from the session cache");
            crate::error_journal::record(
                crate::error_journal::Category::Library,
                &shown,
                format!("module was only in the session cache; adopted into the source manifest at startup ({functions} function(s))"),
            );
            continue;
        }
        program.modules.retain(|module| &module.name != name);
        let Some(recovery) = kept.get(name) else {
            eprintln!("[library] dropped cached scratch module `{shown}` (never part of the source library)");
            continue;
        };
        let reason = refused.get(name).cloned().unwrap_or_else(|| "it could not be enrolled".into());
        let reason: String = reason.chars().take(300).collect();
        eprintln!("[library] dropped cached module `{shown}`: it has no source and cannot be saved ({reason}); its definition was kept in {recovery}");
        let message = format!("cached module without source was dropped at startup and kept in {recovery}: {reason}");
        crate::error_journal::record(crate::error_journal::Category::Library, &shown, &message);
        state.push_error(format!("module `{shown}`: {message}"));
        if let Ok(mut dropped) = state.dropped_at_startup.lock() {
            dropped.insert(name.clone());
        }
    }
    program.rebuild_function_index();
    Ok(enrolled)
}

/// `<library>/recovery/<Module>-<unix ms>.json`, owner-only. Subdirectories are
/// never scanned for sources.
fn write_recovery(dir: &Path, module: &ast::Module) -> Result<String> {
    use std::io::Write;
    use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
    let folder = dir.join("recovery");
    if !folder.is_dir() {
        std::fs::DirBuilder::new().mode(0o700).create(&folder)?;
    }
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|e| e.as_millis()).unwrap_or_default();
    let name = format!("{}-{stamp}.json", super::source::diagnostic_name(&module.name));
    let mut file = std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(folder.join(&name))?;
    file.write_all(serde_json::to_string_pretty(module)?.as_bytes())?;
    file.sync_all()?;
    Ok(format!("recovery/{name}"))
}
