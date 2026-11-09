//! The sole runtime writer of managed source. Requests contain no AST snapshot.
//! Lock order: live Program read -> source transaction -> metadata. All work
//! under these guards runs on a blocking worker, without model/eval/IO awaits.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::RwLock;

use super::LibraryState;
use crate::ast;

pub struct SaveReceipt {
    /// Captured while the same read guard protected the source writes. It is a
    /// recovery checkpoint only; a later session-file write is not atomic with source.
    pub program_snapshot: ast::Program,
    pub saved_names: Vec<String>,
    pub failures: Vec<String>,
    /// Managed modules whose save qualified proven bare local calls, with the
    /// number of rewritten call sites (the only permitted AST normalization).
    pub qualified_call_sites: Vec<(String, usize)>,
    /// (module, callee) qualifications whose name several live modules declare:
    /// the save froze today's order-dependent binding. Shown to the owner.
    pub order_dependent: Vec<(String, String)>,
}

pub async fn flush_live_sources(
    program: Arc<RwLock<ast::Program>>,
    state: Option<LibraryState>,
) -> Result<SaveReceipt> {
    flush_live_sources_in(program, state, super::library_dir()).await
}

pub(super) async fn flush_live_sources_in(
    program: Arc<RwLock<ast::Program>>,
    state: Option<LibraryState>,
    dir: PathBuf,
) -> Result<SaveReceipt> {
    if let Some(state) = &state {
        for name in state.managed_module_names() {
            state.queue_source(&name)?;
        }
    }
    tokio::task::spawn_blocking(move || flush_blocking(&program, state.as_ref(), &dir))
        .await
        .context("source flush worker failed; pending edits were not acknowledged as saved")?
}

fn flush_blocking(
    program: &RwLock<ast::Program>,
    state: Option<&LibraryState>,
    dir: &std::path::Path,
) -> Result<SaveReceipt> {
    // Never acquire this lock inside a source transaction. A queued request
    // acquires the *current* program, even if mutations ran since it was queued.
    let live = program.blocking_read();
    let _transaction = state
        .map(|state| {
            state
                .source_transaction
                .lock()
                .map_err(|_| anyhow::anyhow!("source transaction lock unavailable"))
        })
        .transpose()?;
    std::fs::create_dir_all(dir)
        .context("source directory unavailable; pending edits remain unsaved")?;
    if let Some(state) = state {
        if let Err(error) = super::source_io::verify_live_policy(state, dir) {
            let message = format!("source policy unavailable; no source files saved and pending edits retained: {error}");
            state.record_error(message.clone());
            return Ok(SaveReceipt {
                program_snapshot: live.clone(),
                saved_names: Vec::new(),
                failures: vec![message],
                qualified_call_sites: Vec::new(),
                order_dependent: Vec::new(),
            });
        }
    }
    let managed = state
        .map(LibraryState::managed_module_names)
        .unwrap_or_default();
    let mut saved_names = Vec::new();
    let mut failures = Vec::new();
    let mut qualified_call_sites = Vec::new();
    let mut order_dependent = Vec::new();
    for name in managed.iter().cloned() {
        let source_state = state.context("managed source state unavailable")?;
        let pending_request = source_state.queue_source(&name)?;
        let result = super::source_io::persist_live(&live, &name, source_state, dir);
        match result {
            Ok(qualified) => {
                if let Some(ledger) = qualified {
                    if ledger.call_sites > 0 {
                        qualified_call_sites.push((name.clone(), ledger.call_sites));
                    }
                    order_dependent.extend(
                        ledger
                            .order_dependent
                            .into_iter()
                            .map(|callee| (name.clone(), callee)),
                    );
                }
                let mut pending = source_state
                    .pending_sources
                    .lock()
                    .map_err(|_| anyhow::anyhow!("source pending-state lock unavailable"))?;
                // A mutation may enqueue before its CoW commit waits for
                // our live read guard. Do not consume that later request.
                if pending.get(&name) == Some(&pending_request) {
                    pending.remove(&name);
                }
                saved_names.push(name);
            }
            Err(error) => {
                let message = format!("module `{name}` remains live but source was NOT saved: {error}; preserve the session recovery checkpoint, reconcile, and restart if source changed externally");
                if let Some(state) = state {
                    state.record_error(message.clone());
                }
                failures.push(message);
            }
        }
    }
    let program_snapshot = live.clone();
    drop(_transaction);
    drop(live);
    // Preserve the legacy snapshot writer, but do not hold the live Program
    // guard over unbounded legacy modules. Managed snapshot writes still refuse.
    for module in program_snapshot
        .modules
        .iter()
        .filter(|module| !managed.contains(&module.name))
    {
        match super::persist_program_module_to_dir(&program_snapshot, &module.name, state, dir) {
            Ok(()) => saved_names.push(module.name.clone()),
            Err(error) => {
                let message = format!(
                    "legacy module `{}` source save failed: {error}",
                    module.name
                );
                if let Some(state) = state {
                    state.record_error(message.clone());
                }
                failures.push(message);
            }
        }
    }
    Ok(SaveReceipt {
        program_snapshot,
        saved_names,
        failures,
        qualified_call_sites,
        order_dependent,
    })
}
