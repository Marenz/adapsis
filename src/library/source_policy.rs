//! What a manifest-governed library refuses up front (#128, #126): creating a
//! module from Adapsis source and removing a whole managed module. Called at
//! every mutation entry point before any state change.
use std::collections::BTreeSet;

use anyhow::{bail, Result};

use super::{library_dir, source, LibraryState};
use crate::{ast, parser};

/// Exit code of a start refused by the source policy (EX_CONFIG). The unit must
/// not restart on it: the same files would refuse again.
pub const EXIT_SOURCE_REFUSED: i32 = 78;

pub(super) const CREATE_REFUSAL: &str =
    "this library stores Python source; create the module with a Python mutation";

/// The managed module names when a source manifest governs this process, else
/// `None`. Prefers the retained state; without one (handler threads, tests) the
/// library directory decides. Inspection failures count as "governed".
fn governed_modules(state: Option<&LibraryState>) -> Result<Option<BTreeSet<String>>> {
    if let Some(state) = state {
        state.check_source_available()?;
        if state.has_source_manifest()? {
            return Ok(Some(state.managed_module_names()));
        }
    }
    let dir = library_dir();
    if matches!(std::fs::symlink_metadata(&dir), Err(error) if error.kind() == std::io::ErrorKind::NotFound) {
        return Ok(None);
    }
    if !source::manifest_present(&dir)? {
        return Ok(None);
    }
    Ok(Some(source::ManagedSources::read(&dir)?.map(|managed| managed.manifest.modules.keys().cloned().collect()).unwrap_or_default()))
}

/// Under a manifest a module is created by a Python mutation only (DESIGN L1);
/// Adapsis-source creation would be accepted in memory and could never be saved.
pub fn refuse_module_creation(state: Option<&LibraryState>, name: &str) -> Result<()> {
    if governed_modules(state)?.is_some() {
        bail!("module `{}` was not created: {CREATE_REFUSAL}", source::diagnostic_name(name));
    }
    Ok(())
}

/// Removing a whole managed module is not supported yet (#126): the source
/// would stay and the module would return at the next start.
pub fn refuse_module_removal(state: Option<&LibraryState>, name: &str) -> Result<()> {
    if governed_modules(state)?.is_some_and(|names| names.contains(name)) {
        bail!(
            "removing module `{}` is not supported yet for manifest-managed libraries (#126); nothing was changed. Remove its functions individually, or ask the owner to retire the module",
            source::diagnostic_name(name)
        );
    }
    Ok(())
}

/// Single check for a submission: Adapsis-source module creation (declaration
/// or `!move` into a new module) and whole-module removal are refused before
/// any state change when a manifest governs the library.
pub fn refuse_unsupported_operations(
    state: Option<&LibraryState>,
    program: &ast::Program,
    operations: &[parser::Operation],
    python_origin: bool,
) -> Result<()> {
    let exists = |name: &str| program.modules.iter().any(|module| module.name == name);
    for operation in operations {
        match operation {
            parser::Operation::Module(module) if !python_origin && !exists(&module.name) => {
                refuse_module_creation(state, &module.name)?
            }
            parser::Operation::Move { target_module, .. } if !exists(target_module) => {
                refuse_module_creation(state, target_module)?
            }
            parser::Operation::Remove(target) if !target.contains('.') && exists(target) => {
                refuse_module_removal(state, target)?
            }
            _ => {}
        }
    }
    Ok(())
}

/// A new module under a manifest: the managed writer enrols it with its first
/// save. Scratch and reserved modules are never source and get no notice. The
/// acknowledgement is neutral: whether the source was saved is reported by the
/// entry point after its write-back (`api/source_save.rs`), which knows.
pub(super) fn queue_new_module(program: &ast::Program, state: &LibraryState, name: &str) -> Option<(String, bool)> {
    if !program.modules.iter().any(|m| m.name == name) || source::never_enrolled(name) {
        return None;
    }
    Some(match state.queue_source(name) {
        Err(error) => (error.to_string(), false),
        Ok(_) => (format!("module `{name}` changed in memory (new module)"), true),
    })
}
