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
    let result = load_candidate(program, dir, state).and_then(|mut names| {
        // Modules that exist only in the session cache: enrol or drop (#128).
        // Runs after the set is published and before anything can execute.
        if lock(&state.managed_sources)?.is_some() {
            names.extend(super::source_enrol::adopt_cached(program, dir, state)?);
        }
        Ok(names)
    });
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
    refuse_unlisted_cache(program)?;
    if previous.as_ref().is_some_and(|old| {
        old.manifest
            .modules
            .keys()
            .any(|name| !names.contains(name))
    }) {
        bail!("source manifest removed a managed module; restart with a reconciled recovery cache instead of hot reloading");
    }
    let DecodedSet { loaded, hashes, stored_cases } = decode_set(dir, &managed, &BTreeMap::new())?;
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
    // The set is valid. Earn the pass flags: mock-only re-run, no real IO.
    let rerun = super::source_verify::rerun_stored_cases(&mut candidate, &stored_cases);
    super::source_verify::report(&rerun);
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

pub(super) struct DecodedSet {
    pub(super) loaded: ast::Program,
    hashes: BTreeMap<String, String>,
    stored_cases: BTreeMap<String, Vec<parser::TestMutation>>,
}

/// Decodes every listed source in dependency order, each against its declared
/// dependency closure only: exactly what a start does. `overlay` supplies the
/// text of sources that are not on disk yet. Executes nothing.
pub(super) fn decode_set(
    dir: &Path,
    managed: &ManagedSources,
    overlay: &BTreeMap<String, String>,
) -> Result<DecodedSet> {
    let mut loaded = ast::Program::default();
    let mut hashes = BTreeMap::new();
    let mut stored_cases = BTreeMap::new();
    for name in &managed.order {
        let format = managed.manifest.modules[name].format;
        let filename = format!("{name}.{}", format.extension());
        let text = match overlay.get(name) {
            Some(text) => text.clone(),
            None => super::source::read_source(&dir.join(&filename))
                .map_err(|_| anyhow::anyhow!("authoritative source `{filename}` could not be read"))?,
        };
        let mut catalog = dependency_catalog(&loaded, managed, name)?;
        let cases = decode(&mut catalog, name, &text, format).map_err(|error| {
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
        stored_cases.insert(name.clone(), cases);
    }
    Ok(DecodedSet { loaded, hashes, stored_cases })
}

/// Would a start succeed with `manifest` and the sources on disk (plus
/// `overlay`)? Asked before every manifest commit of the writer.
pub(super) fn would_load(
    dir: &Path,
    manifest: &super::source::SourceManifest,
    overlay: &BTreeMap<String, String>,
) -> Result<()> {
    let trial = ManagedSources::trial(manifest.clone())?;
    decode_set(dir, &trial, overlay).map(|_| ())
}

pub(super) fn managed_handler(handler: &str, names: &BTreeSet<String>) -> bool {
    handler
        .split_once('.')
        .is_some_and(|(owner, _)| names.contains(owner))
}

pub(super) fn dependency_catalog(
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

/// Cached top-level functions/types have no module to live in and OS mode
/// cannot create them; a session that has them is refused (exit 78, no loop).
/// Cached MODULES the manifest does not list are handled after the load by
/// `source_enrol::adopt_cached`, never by a refusal.
fn refuse_unlisted_cache(program: &ast::Program) -> Result<()> {
    if !program.functions.is_empty() || !program.types.is_empty() {
        bail!("manifest-managed startup refuses cached top-level functions/types; preserve the recovery cache and move them into explicit module sources");
    }
    Ok(())
}

/// Applies one source file to `catalog` and returns its stored cases (grouped
/// per function, stored order) for the mock-only startup re-run.
fn decode(
    catalog: &mut ast::Program,
    name: &str,
    text: &str,
    format: SourceFormat,
) -> Result<Vec<parser::TestMutation>> {
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
    for test in &tests {
        if !test.function_name.starts_with(&format!("{name}."))
            || !seen.insert(test.function_name.clone())
            || catalog.get_function(&test.function_name).is_none()
        {
            bail!("stored test must name one local function with one grouped case set");
        }
        // Verified Python source cases carry typed payloads; transitional
        // legacy_ax imports keep the legacy string adapter. Parsing alone is
        // not a test run, so the pass flag is cleared here; the mock-only
        // startup re-run (source_verify) sets it for cases that really pass.
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
    Ok(tests)
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

/// The checked text one managed module would be saved as, from the live Program.
pub(super) struct Prepared {
    pub(super) name: String,
    pub(super) text: String,
    pub(super) ledger: SaveLedger,
    /// False when the file on disk already holds exactly this text.
    pub(super) changed: bool,
    filename: String,
    loaded_hash: String,
    managed: ManagedSources,
}

/// Prints and checks one managed module without writing anything. `None`: the
/// name is not manifest-managed.
pub(super) fn prepare_live(
    program: &ast::Program,
    name: &str,
    state: &LibraryState,
    dir: &Path,
) -> Result<Option<Prepared>> {
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
    let (python, python_ledger) = checked_python(program, &catalog, module, name, spec.format)?;
    let (text, ledger) = match spec.format {
        SourceFormat::Python => (python, python_ledger),
        // legacy_ax persists the reconstructed .ax, which keeps bare calls
        // verbatim: no qualification reaches disk, so nothing needs re-proof.
        SourceFormat::LegacyAx => {
            let text = super::reconstruct_module_source(module);
            super::validate_reconstructed_source(name, &text)?;
            (text, SaveLedger::default())
        }
    };
    let loaded_hash = state.file_hash(name).context(
        "managed source was never successfully loaded; reconcile and restart before saving",
    )?;
    let filename = format!("{name}.{}", spec.format.extension());
    let changed = super::source_hash(&text) != loaded_hash;
    Ok(Some(Prepared { name: name.to_string(), text, ledger, changed, filename, loaded_hash, managed }))
}

/// Writes a prepared module (temporary file, checks, rename, directory sync).
/// Also run for unchanged text: it verifies that the file on disk is still the
/// one this process loaded or wrote.
pub(super) fn commit_live(prepared: &Prepared, state: &LibraryState, dir: &Path) -> Result<()> {
    super::source::write_checked(dir, &prepared.filename, &prepared.text, &prepared.loaded_hash, &prepared.managed)?;
    state.record_file_hash(&prepared.name, &prepared.text);
    Ok(())
}

/// Proven-qualification facts of one managed save.
#[derive(Debug, Default)]
pub(super) struct SaveLedger {
    pub(super) call_sites: usize,
    /// Qualified callees declared by several live modules (binding frozen).
    pub(super) order_dependent: Vec<String>,
}

/// Checked Python for one module of the live Program: complete print, round
/// trip, and the proof that every qualified bare call still binds to the same
/// function in the WHOLE live Program (the catalog may omit modules).
pub(super) fn checked_python(
    program: &ast::Program,
    catalog: &ast::Program,
    module: &ast::Module,
    name: &str,
    format: SourceFormat,
) -> Result<(String, SaveLedger)> {
    use super::source_unsaved::refusal;
    use super::UnsavedKind::{Nesting, Unprintable};
    let (python, ledger) = python_frontend::print_python_module_with_ledger(catalog, name).map_err(|diagnostic| {
        if python_frontend::is_nesting_refusal(&diagnostic) {
            // The printer parenthesizes every binary operation, which roughly
            // doubles the depth of an operator chain (#140). Everything the
            // submission guard admits fits; this is reached by deeper `.ax` edits.
            let limit = python_frontend::MAX_STORED_NESTING;
            refusal(Nesting, format!("module `{name}` cannot be saved: the printed form of an expression near line {} exceeds the nesting limit of {limit} levels for stored source (printing adds parentheses, so about {} chained operators is the most that can be saved). Split the expression into typed local bindings or helper functions; source unchanged", diagnostic.line, (limit - 1) / 2))
        } else {
            refusal(Unprintable, format!("module `{name}` cannot be completely serialized as Python (line {}, column {}); source unchanged. Keep the session recovery checkpoint and repair unsupported members/tests before retrying", diagnostic.line, diagnostic.column))
        }
    })?;
    if format == SourceFormat::LegacyAx {
        // The reconstructed .ax keeps bare calls verbatim: nothing to re-prove.
        return Ok((python, SaveLedger::default()));
    }
    for callee in &ledger.names {
        let own = module.functions.iter().find(|f| &f.name == callee);
        let resolved = program.get_function(callee);
        if !matches!((own, resolved), (Some(own), Some(resolved)) if std::ptr::eq(own.as_ref(), resolved)) {
            return Err(refusal(Unprintable, format!("module `{name}`: a bare local call resolves to another declaration in the live program; qualification would change runtime behaviour, source unchanged")));
        }
    }
    // Names several live modules declare: the save freezes today's binding.
    let order_dependent = ledger
        .names
        .iter()
        .filter(|callee| python_frontend::declaration_count(program, callee) > 1)
        .cloned()
        .collect();
    Ok((python, SaveLedger { call_sites: ledger.call_sites, order_dependent }))
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
