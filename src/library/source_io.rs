//! Small adapter between manifest-managed files and the trusted Python frontend.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use anyhow::{bail, Context, Result};

use super::source::{ManagedSources, SourceFormat, MANIFEST_FILE};
use super::LibraryState;
use crate::{ast, parser, python_frontend, validator};

fn lock<T>(mutex: &Mutex<T>) -> Result<MutexGuard<'_, T>> {
    mutex
        .lock()
        .map_err(|_| anyhow::anyhow!("managed source state lock unavailable; restart for recovery"))
}

pub(super) fn load(
    program: &mut ast::Program,
    dir: &Path,
    state: &LibraryState,
) -> Result<BTreeSet<String>> {
    let _transaction = lock(&state.source_transaction)?;
    let result = load_candidate(program, dir, state);
    if let Err(error) = &result {
        let message = error.to_string();
        *lock(&state.managed_error)? = Some(message.clone());
        state.record_load_error(MANIFEST_FILE.to_string(), message);
    }
    result
}

fn load_candidate(
    program: &mut ast::Program,
    dir: &Path,
    state: &LibraryState,
) -> Result<BTreeSet<String>> {
    let previous = lock(&state.managed_sources)?.clone();
    let Some(managed) = ManagedSources::read(dir)? else {
        if previous.is_some() {
            bail!("source manifest disappeared; restore it before reloading");
        }
        return Ok(BTreeSet::new());
    };
    let names: BTreeSet<_> = managed.manifest.modules.keys().cloned().collect();
    refuse_unlisted_cache(program, &names)?;
    if previous.as_ref().is_some_and(|old| {
        old.manifest
            .modules
            .keys()
            .any(|name| !names.contains(name))
    }) {
        bail!("source manifest removed a managed module; restart with a reconciled recovery cache instead of hot reloading");
    }
    let mut loaded = ast::Program::default();
    let mut hashes = BTreeMap::new();
    for name in &managed.order {
        let format = managed.manifest.modules[name].format;
        let filename = format!("{name}.{}", format.extension());
        let text = super::source::read_source(&dir.join(&filename))
            .map_err(|_| anyhow::anyhow!("authoritative source `{filename}` could not be read"))?;
        let mut catalog = dependency_catalog(&loaded, &managed, name)?;
        decode(&mut catalog, name, &text, format).map_err(|error| {
            let location = error.downcast_ref::<python_frontend::Diagnostic>()
                .map(|diagnostic| format!(" at line {}, column {}", diagnostic.line, diagnostic.column))
                .unwrap_or_default();
            anyhow::anyhow!("authoritative source `{filename}` failed module/test validation{location}; repair it before restarting")
        })?;
        let module = catalog
            .modules
            .into_iter()
            .find(|module| module.name == *name)
            .context("source decoder did not produce its declared module")?;
        loaded.modules.push(module);
        loaded.pending_routes.extend(catalog.pending_routes);
        loaded.rebuild_function_index();
        hashes.insert(name.clone(), super::source_hash(&text));
    }
    managed.verify_manifest(dir)?;
    for (name, hash) in &hashes {
        let path = dir.join(format!(
            "{name}.{}",
            managed.manifest.modules[name].format.extension()
        ));
        let bytes = super::source::read_source(&path).context("source disappeared during load")?;
        if super::source_hash(&bytes) != *hash {
            bail!("source changed during managed-set load; retry the whole reload");
        }
    }
    let mut candidate = program.clone();
    candidate
        .modules
        .retain(|module| !names.contains(&module.name));
    candidate.modules.extend(loaded.modules);
    candidate
        .pending_routes
        .retain(|route| !managed_handler(&route.handler_fn, &names));
    candidate.pending_routes.extend(loaded.pending_routes);
    candidate.rebuild_function_index();
    let mut state_hashes = lock(&state.file_hashes)?;
    let mut state_managed = lock(&state.managed_sources)?;
    let mut state_error = lock(&state.managed_error)?;
    *program = candidate;
    state_hashes.extend(hashes);
    *state_managed = Some(managed);
    *state_error = None;
    state.clear_load_error(MANIFEST_FILE);
    Ok(names)
}

pub(super) fn managed_handler(handler: &str, names: &BTreeSet<String>) -> bool {
    handler
        .split_once('.')
        .is_some_and(|(owner, _)| names.contains(owner))
}

fn dependency_catalog(
    program: &ast::Program,
    managed: &ManagedSources,
    name: &str,
) -> Result<ast::Program> {
    let mut catalog = ast::Program::default();
    for dependency in managed.dependency_names(name) {
        let module = program
            .modules
            .iter()
            .find(|module| module.name == dependency)
            .with_context(|| {
                format!("module `{name}` dependency is unavailable in this source snapshot")
            })?;
        catalog.modules.push(module.clone());
    }
    catalog.rebuild_function_index();
    Ok(catalog)
}

fn refuse_unlisted_cache(program: &ast::Program, names: &BTreeSet<String>) -> Result<()> {
    let unlisted: Vec<_> = program
        .modules
        .iter()
        .filter(|module| !names.contains(&module.name))
        .map(|module| module.name.as_str())
        .collect();
    if !unlisted.is_empty() {
        let shown = unlisted
            .iter()
            .take(20)
            .map(|name| super::source::diagnostic_name(name))
            .collect::<Vec<_>>()
            .join(", ");
        bail!("source manifest excludes {} cached module(s): {shown}; startup refused. Preserve the session recovery cache and add explicit source entries before restarting", unlisted.len());
    }
    if !program.functions.is_empty() || !program.types.is_empty() {
        bail!("manifest-managed startup refuses cached top-level functions/types; preserve the recovery cache and move them into explicit module sources");
    }
    Ok(())
}

fn decode(catalog: &mut ast::Program, name: &str, text: &str, format: SourceFormat) -> Result<()> {
    check_catalog_budget(catalog)?;
    let (declarations, tests) = match format {
        SourceFormat::Python => {
            let lowered = python_frontend::lower_python_module(text, name, catalog)?;
            (lowered.declarations.operations, lowered.tests)
        }
        SourceFormat::LegacyAx => legacy_declarations(text, name)?,
    };
    for declaration in declarations {
        if let parser::Operation::Module(module) = &declaration {
            unique_members(&module.body)?;
        }
        validator::apply_and_validate(catalog, &declaration)?;
    }
    catalog.rebuild_function_index();
    let mut seen = BTreeSet::new();
    for test in tests {
        if !test.function_name.starts_with(&format!("{name}."))
            || !seen.insert(test.function_name.clone())
            || catalog.get_function(&test.function_name).is_none()
        {
            bail!("stored test must name one local function with one grouped case set");
        }
        // Verified Python source cases carry typed payloads; transitional
        // legacy_ax imports keep the legacy string adapter. Parsing alone is
        // not a test run, so the pass flag is cleared below.
        let typed = format == SourceFormat::Python;
        crate::session::store_test_payload(catalog, &test.function_name, &test.cases, typed);
        if let Some(function) = catalog.get_function_mut(&test.function_name) {
            for case in &mut function.tests {
                case.passed = false;
            }
        }
    }
    // The checked printer also rejects unresolved owners/callees in a legacy
    // importer. Unsupported legacy modules must be repaired explicitly, not
    // silently accepted as a second independently editable authority.
    check_catalog_budget(catalog)?;
    // Registration is a startup-layer transaction, not source content. The
    // checked printer rejects pending queues, so retain this batch separately
    // while it checks the complete module declarations (including routes).
    let registrations = std::mem::take(&mut catalog.pending_routes);
    python_frontend::print_python_module(catalog, name)?;
    catalog.pending_routes = registrations;
    Ok(())
}

/// The ordinary validator merges declarations; a source snapshot must not use
/// that merge behavior to silently discard an earlier member of the same file.
fn unique_members(body: &[parser::Operation]) -> Result<()> {
    let mut seen = BTreeSet::new();
    for operation in body {
        let key = match operation {
            parser::Operation::Function(value) => format!("function:{}", value.name),
            parser::Operation::Type(value) => format!("type:{}", value.name),
            parser::Operation::SharedVar(value) => format!("shared:{}", value.name),
            parser::Operation::ModuleSource(value) => format!("source:{}", value.name),
            parser::Operation::Route { method, path, .. } => format!("route:{method}:{path}"),
            parser::Operation::Startup(_) => "startup".to_string(),
            parser::Operation::Shutdown(_) => "shutdown".to_string(),
            _ => continue,
        };
        if !seen.insert(key) {
            bail!("authoritative source contains a duplicate declaration; each member must appear exactly once");
        }
    }
    Ok(())
}

fn legacy_declarations(
    text: &str,
    name: &str,
) -> Result<(Vec<parser::Operation>, Vec<parser::TestMutation>)> {
    let mut declarations = Vec::new();
    let mut tests = Vec::new();
    for operation in parser::parse(text)? {
        match operation {
            parser::Operation::Module(mut module)
                if module.name == name && declarations.is_empty() =>
            {
                module.body.retain(|op| {
                    if let parser::Operation::Test(test) = op {
                        tests.push(test.clone());
                        false
                    } else {
                        true
                    }
                });
                declarations.push(parser::Operation::Module(module));
            }
            parser::Operation::Test(test) => tests.push(test),
            _ => bail!("managed legacy import requires exactly one matching module and its tests"),
        }
    }
    if declarations.len() != 1 {
        bail!("managed legacy import has no module");
    }
    for test in &mut tests {
        if !test.function_name.contains('.') {
            test.function_name = format!("{name}.{}", test.function_name);
        }
    }
    Ok((declarations, tests))
}

pub(super) fn persist_live(
    program: &ast::Program,
    name: &str,
    state: &LibraryState,
    dir: &Path,
) -> Result<Option<SaveLedger>> {
    state.check_source_available()?;
    let Some(managed) = lock(&state.managed_sources)?.clone() else {
        if super::source::manifest_present(dir)? {
            bail!("new source manifest has not been loaded; restart the server before saving");
        }
        return Ok(None);
    };
    managed.verify_manifest(dir)?;
    let Some(spec) = managed.manifest.modules.get(name) else {
        return Ok(None);
    };
    let mut catalog = dependency_catalog(program, &managed, name)?;
    let module = program
        .modules
        .iter()
        .find(|module| module.name == name)
        .context("managed module absent from current program; source deletion is not implicit")?;
    catalog.modules.push(module.clone());
    catalog.rebuild_function_index();
    check_catalog_budget(&catalog)?;
    let (python, ledger) = python_frontend::print_python_module_with_ledger(&catalog, name)
        .map_err(|diagnostic| anyhow::anyhow!("module `{name}` cannot be completely serialized as Python (line {}, column {}); source unchanged. Keep the session recovery checkpoint and repair unsupported members/tests before retrying", diagnostic.line, diagnostic.column))?;
    let (text, save_ledger) = match spec.format {
        SourceFormat::Python => {
            // The printer proved each qualification against the closed catalog.
            // The stored bare call ran against the WHOLE live program, so the
            // proof must also hold there, or qualification could change which
            // function runs.
            for callee in &ledger.names {
                let own = module.functions.iter().find(|f| &f.name == callee);
                let resolved = program.get_function(callee);
                if !matches!((own, resolved), (Some(own), Some(resolved)) if std::ptr::eq(own.as_ref(), resolved))
                {
                    bail!("module `{name}`: a bare local call resolves to another declaration in the live program; qualification would change runtime behaviour, source unchanged");
                }
            }
            // Classify against the WHOLE live program (the catalog may hide
            // modules that also declare a name): these freeze today's binding.
            let order_dependent = ledger
                .names
                .iter()
                .filter(|callee| python_frontend::declaration_count(program, callee) > 1)
                .cloned()
                .collect();
            (
                python,
                SaveLedger {
                    call_sites: ledger.call_sites,
                    order_dependent,
                },
            )
        }
        // legacy_ax persists the reconstructed .ax, which keeps bare calls
        // verbatim: no qualification reaches disk, so nothing needs re-proof.
        SourceFormat::LegacyAx => {
            let text = super::reconstruct_module_source(module);
            super::validate_reconstructed_source(name, &text)?;
            (text, SaveLedger::default())
        }
    };
    let hash = state.file_hash(name).context(
        "managed source was never successfully loaded; reconcile and restart before saving",
    )?;
    let filename = format!("{name}.{}", spec.format.extension());
    super::source::write_checked(dir, &filename, &text, &hash, &managed)?;
    state.record_file_hash(name, &text);
    Ok(Some(save_ledger))
}

/// Proven-qualification facts of one managed save.
#[derive(Debug, Default)]
pub(super) struct SaveLedger {
    pub(super) call_sites: usize,
    /// Qualified callees declared by several live modules (binding frozen).
    pub(super) order_dependent: Vec<String>,
}

pub(super) fn verify_live_policy(state: &LibraryState, dir: &Path) -> Result<()> {
    state.check_source_available()?;
    match lock(&state.managed_sources)?.as_ref() {
        Some(managed) => managed.verify_manifest(dir),
        None if super::source::manifest_present(dir)? => {
            bail!("source manifest has not been loaded; refusing legacy fallback until restart")
        }
        None => Ok(()),
    }
}

fn check_catalog_budget(program: &ast::Program) -> Result<()> {
    struct Budget(usize);
    impl std::io::Write for Budget {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 += bytes.len();
            if self.0 > 16 * 1024 * 1024 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "catalog budget exceeded",
                ));
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    serde_json::to_writer(Budget(0), program).map_err(|_| {
        anyhow::anyhow!("managed source catalog exceeds the 16 MiB AST budget; source unchanged")
    })
}
