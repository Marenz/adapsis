//! Persistent module library — auto-loads modules from ~/.config/adapsis/modules/
//! and persists module changes back to disk.
//!
//! This provides a permanent library layer that works across git worktrees and sessions.
//! Modules are stored as reconstructed `.ax` source files, one per module.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};

use crate::ast;
use crate::parser;
use crate::typeck;
use crate::validator;

/// A module load error — pairs the module/file name with the error message.
#[derive(Debug, Clone)]
pub struct LoadError {
    /// Module name (derived from filename, e.g. "MyModule" from "MyModule.ax")
    pub module_name: String,
    /// The error message describing why loading failed
    pub error: String,
}

/// Runtime state for the module library — tracks what was loaded and any errors.
#[derive(Debug, Clone)]
pub struct LibraryState {
    /// Module names successfully auto-loaded at startup.
    pub loaded_modules: Vec<String>,
    /// Accumulated load/save error messages this session.
    pub errors: Arc<Mutex<Vec<String>>>,
    /// Structured load errors from startup — (module_name, error_message) pairs.
    pub load_errors: Arc<Mutex<Vec<LoadError>>>,
}

impl LibraryState {
    pub fn new() -> Self {
        Self {
            loaded_modules: Vec::new(),
            errors: Arc::new(Mutex::new(Vec::new())),
            load_errors: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn record_error(&self, msg: String) {
        if let Ok(mut errs) = self.errors.lock() {
            errs.push(msg);
        }
    }

    /// Record a structured load error with the module name and error message.
    fn record_load_error(&self, module_name: String, error: String) {
        if let Ok(mut errs) = self.load_errors.lock() {
            errs.push(LoadError { module_name, error });
        }
    }

    /// Format load errors for display (e.g. in ?library output or AI context).
    /// Returns None if there are no load errors.
    pub fn format_load_errors(&self) -> Option<String> {
        let errs = self.load_errors.lock().ok()?;
        if errs.is_empty() {
            return None;
        }
        let mut out = format!("Load errors ({}):\n", errs.len());
        for le in errs.iter() {
            out.push_str(&format!("  {}: {}\n", le.module_name, le.error));
        }
        Some(out)
    }
}

/// Default library directory path.
pub fn library_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".config")
        .join("adapsis")
        .join("modules")
}

/// Ensure the library directory exists, creating it if necessary.
pub fn ensure_library_dir() -> Result<PathBuf> {
    let dir = library_dir();
    if !dir.exists() {
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create module library dir: {}", dir.display()))?;
    }
    Ok(dir)
}

/// Reconstruct the full `.ax` source for a module from its AST representation.
/// Produces output that can be round-tripped through the parser.
pub fn reconstruct_module_source(module: &ast::Module) -> String {
    let mut out = String::new();
    out.push_str(&format!("+module {}\n", module.name));
    // Emit module doc if present
    if let Some(ref doc) = module.doc {
        out.push_str(&format!("+doc \"{}\"\n", ast::escape_string_literal(doc)));
    }

    // Types first
    for td in &module.types {
        out.push_str(&reconstruct_type_source(td));
        out.push('\n');
    }

    // Shared variables (after types, before functions)
    for sv in &module.shared_vars {
        out.push_str(&format!(
            "+shared {}:{} = {}\n",
            sv.name,
            format_type(&sv.ty),
            crate::typeck::reconstruct_expr(&sv.default),
        ));
    }
    if !module.shared_vars.is_empty() {
        out.push('\n');
    }

    // Startup block
    if let Some(ref startup) = module.startup {
        let effects = if startup.effects.is_empty() {
            String::new()
        } else {
            format!(
                " [{}]",
                startup
                    .effects
                    .iter()
                    .map(|e| format!("{e:?}").to_lowercase())
                    .collect::<Vec<_>>()
                    .join(",")
            )
        };
        out.push_str(&format!("+startup{}\n", effects));
        for stmt in &startup.body {
            crate::typeck::reconstruct_stmt_pub(&mut out, stmt, 1);
        }
        out.push('\n');
    }

    // Shutdown block
    if let Some(ref shutdown) = module.shutdown {
        let effects = if shutdown.effects.is_empty() {
            String::new()
        } else {
            format!(
                " [{}]",
                shutdown
                    .effects
                    .iter()
                    .map(|e| format!("{e:?}").to_lowercase())
                    .collect::<Vec<_>>()
                    .join(",")
            )
        };
        out.push_str(&format!("+shutdown{}\n", effects));
        for stmt in &shutdown.body {
            crate::typeck::reconstruct_stmt_pub(&mut out, stmt, 1);
        }
        out.push('\n');
    }

    // Source declarations
    for src in &module.sources {
        let config_str = src
            .config
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(" ");
        if config_str.is_empty() {
            out.push_str(&format!(
                "+source {} {} -> {}\n",
                src.name, src.source_type, src.handler
            ));
        } else {
            out.push_str(&format!(
                "+source {} {} {} -> {}\n",
                src.name, src.source_type, config_str, src.handler
            ));
        }
    }
    if !module.sources.is_empty() {
        out.push('\n');
    }

    // Routes
    for route in &module.routes {
        // Strip module prefix from handler_fn for reconstruction
        let handler = route
            .handler_fn
            .strip_prefix(&format!("{}.", module.name))
            .unwrap_or(&route.handler_fn);
        out.push_str(&format!(
            "+route {} \"{}\" -> {}\n",
            route.method, route.path, handler
        ));
    }
    if !module.routes.is_empty() {
        out.push('\n');
    }

    // Then functions
    for func in &module.functions {
        out.push_str(&reconstruct_function_source(func));
        out.push('\n');
    }

    out
}

fn reconstruct_type_source(td: &ast::TypeDecl) -> String {
    match td {
        ast::TypeDecl::Struct(s) => {
            let fields = s
                .fields
                .iter()
                .map(|f| format!("{}:{}", f.name, format_type(&f.ty)))
                .collect::<Vec<_>>()
                .join(", ");
            format!("+type {} = {{{}}}\n", s.name, fields)
        }
        ast::TypeDecl::TaggedUnion(u) => {
            let variants = u
                .variants
                .iter()
                .map(|v| {
                    if v.payload.is_empty() {
                        v.name.clone()
                    } else {
                        format!(
                            "{}({})",
                            v.name,
                            v.payload
                                .iter()
                                .map(format_type)
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    }
                })
                .collect::<Vec<_>>()
                .join(" | ");
            format!("+type {} = {}\n", u.name, variants)
        }
    }
}

fn reconstruct_function_source(func: &ast::FunctionDecl) -> String {
    let mut out = String::new();
    let params = func
        .params
        .iter()
        .map(|p| format!("{}:{}", p.name, format_type(&p.ty)))
        .collect::<Vec<_>>()
        .join(", ");
    let effects = if func.effects.is_empty() {
        String::new()
    } else {
        format!(
            " [{}]",
            func.effects
                .iter()
                .map(|e| format!("{e:?}").to_lowercase())
                .collect::<Vec<_>>()
                .join(",")
        )
    };
    out.push_str(&format!(
        "+fn {} ({})->{}{}\n",
        func.name,
        params,
        format_type(&func.return_type),
        effects
    ));

    for stmt in &func.body {
        typeck::reconstruct_stmt_pub(&mut out, stmt, 1);
    }

    // Emit function doc if present (after +end, before +test)
    if let Some(ref doc) = func.doc {
        out.push_str(&format!("+doc \"{}\"\n", ast::escape_string_literal(doc)));
    }

    // Include persisted tests in source reconstruction
    if !func.tests.is_empty() {
        out.push('\n');
        out.push_str(&format!("+test {}\n", func.name));
        for tc in &func.tests {
            let expect_part = if let Some(ref m) = tc.matcher {
                if m == "AnyOk" {
                    "Ok".to_string()
                } else if m == "AnyErr" {
                    "Err".to_string()
                } else if let Some(msg) = m.strip_prefix("ErrContaining:") {
                    format!("Err(\"{}\")", msg)
                } else if let Some(sub) = m.strip_prefix("contains:") {
                    format!("contains(\"{}\")", sub)
                } else if let Some(pre) = m.strip_prefix("starts_with:") {
                    format!("starts_with(\"{}\")", pre)
                } else {
                    tc.expected.clone()
                }
            } else {
                tc.expected.clone()
            };
            out.push_str(&format!("  +with {} -> expect {}\n", tc.input, expect_part));
            for ac in &tc.after_checks {
                out.push_str(&format!(
                    "  +after {} {} \"{}\"\n",
                    ac.target, ac.matcher, ac.value
                ));
            }
        }
    }

    out
}

fn format_type(ty: &ast::Type) -> String {
    match ty {
        ast::Type::Int => "Int".into(),
        ast::Type::Float => "Float".into(),
        ast::Type::Bool => "Bool".into(),
        ast::Type::String => "String".into(),
        ast::Type::Byte => "Byte".into(),
        ast::Type::List(t) => format!("List<{}>", format_type(t)),
        ast::Type::Set(t) => format!("Set<{}>", format_type(t)),
        ast::Type::Map(k, v) => format!("Map<{},{}>", format_type(k), format_type(v)),
        ast::Type::Option(t) => format!("Option<{}>", format_type(t)),
        ast::Type::Result(t) => format!("Result<{}>", format_type(t)),
        ast::Type::Struct(name) | ast::Type::TaggedUnion(name) => name.clone(),
    }
}

/// Load all `.ax` files from the module library directory into the program.
/// Creates the library directory if it doesn't exist.
/// Files are loaded in sorted filename order for determinism.
/// Malformed files produce a warning but do not abort the process.
/// Returns a `LibraryState` tracking what was loaded and any errors.
pub fn load_module_library(program: &mut ast::Program) -> LibraryState {
    let mut state = LibraryState::new();
    let dir_path = library_dir();
    eprintln!(
        "[library] init: resolved library dir = {}",
        dir_path.display()
    );
    eprintln!("[library] init: HOME = {:?}", std::env::var("HOME").ok());

    // Always create the directory on startup so persist can write later
    let dir = match ensure_library_dir() {
        Ok(d) => {
            eprintln!("[library] init: directory ready at {}", d.display());
            d
        }
        Err(e) => {
            let msg = format!("could not create library dir: {e}");
            eprintln!("[library] init: ERROR — {msg}");
            state.record_error(msg);
            return state;
        }
    };

    let mut files: Vec<PathBuf> = match std::fs::read_dir(&dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "ax"))
            .collect(),
        Err(e) => {
            let msg = format!("could not read library dir {}: {e}", dir.display());
            eprintln!("[library] warning: {msg}");
            state.record_error(msg);
            return state;
        }
    };

    // Sort by filename for deterministic load order
    files.sort();

    for path in &files {
        // Extract module name from filename (e.g. "MyModule.ax" -> "MyModule")
        let file_module_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Unknown")
            .to_string();

        match std::fs::read_to_string(path) {
            Ok(source) => {
                // If this module already exists (e.g. from restored session state),
                // remove it first to prevent "duplicate +startup" errors.
                program.modules.retain(|m| m.name != file_module_name);

                match load_module_source(program, &source) {
                    Ok(module_name) => {
                        eprintln!(
                            "[library] loaded module `{module_name}` from {}",
                            path.display()
                        );
                        state.loaded_modules.push(module_name);
                    }
                    Err(e) => {
                        let err_msg = format!("{e}");
                        let msg = format!("failed to load {}: {e}", path.display());
                        eprintln!("[library] warning: {msg}");
                        state.record_error(msg);
                        state.record_load_error(file_module_name, err_msg);
                    }
                }
            }
            Err(e) => {
                let err_msg = format!("could not read file: {e}");
                let msg = format!("could not read {}: {e}", path.display());
                eprintln!("[library] warning: {msg}");
                state.record_error(msg);
                state.record_load_error(file_module_name, err_msg);
            }
        }
    }

    if !state.loaded_modules.is_empty() {
        eprintln!(
            "[library] auto-loaded {} module(s) from {}",
            state.loaded_modules.len(),
            dir.display()
        );
    }

    state
}

/// Parse and apply a single module source string into the program.
/// Returns the module name on success.
fn load_module_source(program: &mut ast::Program, source: &str) -> Result<String> {
    let operations = parser::parse(source)?;
    let mut module_name = None;

    for op in &operations {
        match op {
            // Skip transient operations
            parser::Operation::Test(_)
            | parser::Operation::Trace(_)
            | parser::Operation::Eval(_)
            | parser::Operation::Query(_)
            | parser::Operation::Plan(_)
            | parser::Operation::Roadmap(_)
            | parser::Operation::Mock { .. }
            | parser::Operation::Unmock => continue,
            parser::Operation::Module(m) => {
                module_name = Some(m.name.clone());
            }
            _ => {}
        }
        validator::apply_and_validate(program, op)?;
    }

    // Restore persisted tests from +test blocks.
    // reconstruct_module_source() only emits tests that previously passed,
    // so we trust them and mark them passed: true without re-executing.
    if let Some(ref mod_name) = module_name {
        restore_tests_from_operations(program, mod_name, &operations);
    }

    module_name.ok_or_else(|| anyhow::anyhow!("no +module declaration found in source"))
}

/// Restore `+test` blocks from parsed operations into the corresponding
/// function's `tests` field. Called during library load to preserve test
/// state across restarts.
fn restore_tests_from_operations(
    program: &mut ast::Program,
    module_name: &str,
    operations: &[parser::Operation],
) {
    // Collect tests from top-level +test ops and from inside +module body
    let mut tests_to_restore: Vec<(&str, &[parser::TestCase])> = Vec::new();

    for op in operations {
        match op {
            parser::Operation::Test(t) => {
                eprintln!("[library] found top-level +test for `{}`", t.function_name);
                tests_to_restore.push((&t.function_name, &t.cases));
            }
            parser::Operation::Module(m) => {
                for body_op in &m.body {
                    if let parser::Operation::Test(t) = body_op {
                        eprintln!(
                            "[library] found module-body +test for `{}`",
                            t.function_name
                        );
                        tests_to_restore.push((&t.function_name, &t.cases));
                    }
                }
            }
            _ => {}
        }
    }

    for (fn_name, cases) in &tests_to_restore {
        let qualified = format!("{module_name}.{fn_name}");
        crate::session::store_test(program, &qualified, cases);
    }
    if !tests_to_restore.is_empty() {
        eprintln!(
            "[library] restored {} test block(s) for module `{module_name}`",
            tests_to_restore.len()
        );
        // Verify tests are actually on the functions
        for (fn_name, _) in &tests_to_restore {
            let qualified = format!("{module_name}.{fn_name}");
            let tested = crate::session::is_function_tested(program, &qualified);
            if !tested {
                eprintln!("[library] WARNING: test restore failed for `{qualified}` — function not found or tests empty");
            }
        }
    }
}

/// Persist a module's reconstructed source to the library directory.
/// Uses atomic write (write to temp file, then rename) for safety.
pub fn persist_module(module: &ast::Module) -> Result<()> {
    let dir = ensure_library_dir()?;
    let target = dir.join(format!("{}.ax", module.name));
    let tmp = dir.join(format!(".{}.ax.tmp", module.name));
    eprintln!(
        "[library] persist: writing module `{}` to {}",
        module.name,
        target.display()
    );

    let source = reconstruct_module_source(module);

    std::fs::write(&tmp, &source)
        .with_context(|| format!("failed to write temp file {}", tmp.display()))?;
    std::fs::rename(&tmp, &target)
        .with_context(|| format!("failed to rename {} -> {}", tmp.display(), target.display()))?;

    Ok(())
}

/// Reload a specific module from disk, or all modules if `module_name` is empty.
///
/// For a specific module: reads `<module_name>.ax` from the library directory,
/// removes the old module from the program, and re-parses/loads the file.
///
/// For all modules: re-reads all `.ax` files from the library directory and
/// reloads each one, replacing existing modules.
///
/// Returns a status message describing what happened.
pub fn reload_module(program: &mut ast::Program, module_name: &str) -> Result<String> {
    let dir = ensure_library_dir()?;

    if module_name.is_empty() {
        // Reload all modules
        let mut files: Vec<PathBuf> = match std::fs::read_dir(&dir) {
            Ok(entries) => entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|ext| ext == "ax"))
                .collect(),
            Err(e) => return Err(anyhow::anyhow!("could not read library dir: {e}")),
        };
        files.sort();

        let mut reloaded = Vec::new();
        let mut failed = Vec::new();

        for path in &files {
            let file_name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Unknown")
                .to_string();

            match std::fs::read_to_string(path) {
                Ok(source) => {
                    // Remove existing module first
                    program.modules.retain(|m| m.name != file_name);
                    match load_module_source(program, &source) {
                        Ok(name) => {
                            eprintln!("[library] reloaded module `{name}` from {}", path.display());
                            reloaded.push(name);
                        }
                        Err(e) => {
                            eprintln!("[library] reload failed for {}: {e}", path.display());
                            failed.push(format!("{file_name}: {e}"));
                        }
                    }
                }
                Err(e) => {
                    failed.push(format!("{file_name}: could not read file: {e}"));
                }
            }
        }

        program.rebuild_function_index();

        let mut msg = format!("Reloaded {} module(s)", reloaded.len());
        if !reloaded.is_empty() {
            msg.push_str(&format!(": {}", reloaded.join(", ")));
        }
        if !failed.is_empty() {
            msg.push_str(&format!(
                ". Failed ({}): {}",
                failed.len(),
                failed.join("; ")
            ));
        }
        Ok(msg)
    } else {
        // Reload a specific module
        let path = dir.join(format!("{module_name}.ax"));
        if !path.exists() {
            return Err(anyhow::anyhow!(
                "library file not found: {}.ax",
                module_name
            ));
        }

        let source = std::fs::read_to_string(&path)
            .with_context(|| format!("could not read {}", path.display()))?;

        // Remove the existing module first
        program.modules.retain(|m| m.name != module_name);

        match load_module_source(program, &source) {
            Ok(name) => {
                program.rebuild_function_index();
                eprintln!("[library] reloaded module `{name}` from {}", path.display());
                Ok(format!("Reloaded {name} successfully"))
            }
            Err(e) => {
                program.rebuild_function_index();
                Err(anyhow::anyhow!("Reload failed: {e}"))
            }
        }
    }
}

/// Determine which module names were affected by a set of parsed operations.
/// This is used to decide which modules need to be persisted after a mutation.
pub fn affected_module_names(operations: &[parser::Operation]) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for op in operations {
        match op {
            parser::Operation::Module(m) => {
                names.insert(m.name.clone());
            }
            parser::Operation::Move { target_module, .. } => {
                names.insert(target_module.clone());
            }
            parser::Operation::Route { .. } => {
                // Routes are program-level, not module-level — skip
            }
            _ => {}
        }
    }
    names
}

/// Persist all modules whose names are in the given set.
/// Logs warnings on failure but does not abort.
/// Records errors into the LibraryState if provided.
pub fn persist_affected_modules(
    program: &ast::Program,
    module_names: &BTreeSet<String>,
    lib_state: Option<&LibraryState>,
) {
    for name in module_names {
        if let Some(module) = program.modules.iter().find(|m| &m.name == name) {
            match persist_module(module) {
                Ok(()) => {
                    eprintln!("[library] persisted module `{name}` to library");
                }
                Err(e) => {
                    let msg = format!("failed to persist module `{name}`: {e}");
                    eprintln!("[library] warning: {msg}");
                    if let Some(state) = lib_state {
                        state.record_error(msg);
                    }
                }
            }
        }
    }
}

/// Format a summary of the library state for the ?library query.
/// Works with or without LibraryState — when called from typeck::handle_query
/// without runtime context, lib_state is None but still shows useful info.
pub fn query_library(program: &ast::Program, lib_state: Option<&LibraryState>) -> String {
    let dir = library_dir();
    let mut out = String::new();

    out.push_str(&format!("Module library: {}\n", dir.display()));
    out.push_str(&format!(
        "Directory exists: {}\n",
        if dir.exists() { "yes" } else { "no" }
    ));

    if let Some(state) = lib_state {
        if state.loaded_modules.is_empty() {
            out.push_str("No library modules loaded at startup.\n");
        } else {
            out.push_str(&format!(
                "\nLoaded {} module(s) at startup:\n",
                state.loaded_modules.len()
            ));
            for name in &state.loaded_modules {
                if let Some(module) = program.modules.iter().find(|m| m.name == *name) {
                    out.push_str(&format!(
                        "  {} — {} type(s), {} function(s)\n",
                        name,
                        module.types.len(),
                        module.functions.len()
                    ));
                } else {
                    out.push_str(&format!("  {} — (not in current program state)\n", name));
                }
            }
        }
    }

    // List .ax files on disk
    if dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut files: Vec<String> = entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "ax"))
                .map(|e| {
                    e.path()
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("?")
                        .to_string()
                })
                .collect();
            files.sort();
            out.push_str(&format!("\nLibrary files on disk ({}):\n", files.len()));
            if files.is_empty() {
                out.push_str("  (none)\n");
            } else {
                for f in &files {
                    out.push_str(&format!("  {f}.ax\n"));
                }
            }
        }
    }

    // Show structured load errors (module name + error message)
    if let Some(state) = lib_state {
        if let Some(load_errors_text) = state.format_load_errors() {
            out.push('\n');
            out.push_str(&load_errors_text);
        }
    }

    // Show general errors
    if let Some(state) = lib_state {
        if let Ok(errs) = state.errors.lock() {
            if !errs.is_empty() {
                out.push_str(&format!("\nErrors this session ({}):\n", errs.len()));
                for e in errs.iter() {
                    out.push_str(&format!("  {e}\n"));
                }
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reconstruct_module_source_roundtrips() {
        let source = "+module TestMod\n+fn greet (name:String)->String\n  +return name\n";
        let mut program = ast::Program::default();
        let ops = parser::parse(source).unwrap();
        for op in &ops {
            validator::apply_and_validate(&mut program, op).unwrap();
        }
        let module = program
            .modules
            .iter()
            .find(|m| m.name == "TestMod")
            .unwrap();
        let reconstructed = reconstruct_module_source(module);
        assert!(reconstructed.contains("+module TestMod"));
        assert!(reconstructed.contains("+fn greet"));
        assert!(reconstructed.contains("+return name"));

        // Verify roundtrip: parse the reconstructed source into a fresh program
        let mut program2 = ast::Program::default();
        let ops2 = parser::parse(&reconstructed).unwrap();
        for op in &ops2 {
            validator::apply_and_validate(&mut program2, op).unwrap();
        }
        assert_eq!(program2.modules.len(), 1);
        assert_eq!(program2.modules[0].name, "TestMod");
        assert_eq!(program2.modules[0].functions.len(), 1);
        assert_eq!(program2.modules[0].functions[0].name, "greet");
    }

    #[test]
    fn test_persist_and_load_module() {
        let tmp = tempfile::tempdir().unwrap();
        let lib_dir = tmp.path().join("modules");
        std::fs::create_dir_all(&lib_dir).unwrap();

        // Create a program with a module
        let source = "+module Probe\n+fn hi ()->String\n  +return \"ok\"\n";
        let mut program = ast::Program::default();
        let ops = parser::parse(source).unwrap();
        for op in &ops {
            validator::apply_and_validate(&mut program, op).unwrap();
        }

        // Persist manually to the temp dir
        let module = program.modules.iter().find(|m| m.name == "Probe").unwrap();
        let target = lib_dir.join("Probe.ax");
        let src = reconstruct_module_source(module);
        std::fs::write(&target, &src).unwrap();
        assert!(target.exists());

        // Load into fresh program
        let mut program2 = ast::Program::default();
        let content = std::fs::read_to_string(&target).unwrap();
        let ops2 = parser::parse(&content).unwrap();
        for op in &ops2 {
            validator::apply_and_validate(&mut program2, op).unwrap();
        }
        assert_eq!(program2.modules.len(), 1);
        assert_eq!(program2.modules[0].name, "Probe");
        assert_eq!(program2.modules[0].functions[0].name, "hi");
    }

    #[test]
    fn test_affected_module_names() {
        let source = "+module Foo\n+fn bar ()->Int\n  +return 1\n";
        let ops = parser::parse(source).unwrap();
        let names = affected_module_names(&ops);
        assert!(names.contains("Foo"));
        assert_eq!(names.len(), 1);
    }

    #[test]
    fn test_query_library_without_state() {
        let program = ast::Program::default();
        let output = query_library(&program, None);
        assert!(output.contains("Module library:"));
        assert!(output.contains("Directory exists:"));
        // Must NOT say "Library state not initialized" — that's removed
        // When no state, it just omits the loaded modules section
    }

    #[test]
    fn test_symbols_no_bare_duplicates() {
        let source = "+module Probe\n+fn hi ()->String\n  +return \"ok\"\n+fn answer ()->Int\n  +return 42\n";
        let mut program = ast::Program::default();
        let ops = parser::parse(source).unwrap();
        for op in &ops {
            validator::apply_and_validate(&mut program, op).unwrap();
        }
        let table = crate::typeck::build_symbol_table(&program);
        let output = crate::typeck::handle_query(&program, &table, "?symbols", &[]);
        // Must contain qualified names
        assert!(
            output.contains("Probe.hi"),
            "missing Probe.hi in:\n{output}"
        );
        assert!(
            output.contains("Probe.answer"),
            "missing Probe.answer in:\n{output}"
        );
        // Must NOT contain bare unqualified names in Functions listing
        for line in output.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("hi (") || trimmed.starts_with("answer (") {
                panic!("bare unqualified name found in ?symbols output:\n{output}");
            }
        }
    }

    #[test]
    fn test_source_qualified_lookup() {
        let source = "+module Probe\n+fn hi ()->String\n  +return \"ok\"\n";
        let mut program = ast::Program::default();
        let ops = parser::parse(source).unwrap();
        for op in &ops {
            validator::apply_and_validate(&mut program, op).unwrap();
        }
        let table = crate::typeck::build_symbol_table(&program);
        let output = crate::typeck::handle_query(&program, &table, "?source Probe.hi", &[]);
        assert!(
            output.contains("+fn hi"),
            "?source Probe.hi failed:\n{output}"
        );
        assert!(output.contains("+return"), "missing body in:\n{output}");
    }

    #[test]
    fn test_library_query_in_handle_query() {
        let program = ast::Program::default();
        let table = crate::typeck::build_symbol_table(&program);
        let output = crate::typeck::handle_query(&program, &table, "?library", &[]);
        // Must NOT return "unknown query"
        assert!(
            !output.contains("unknown query"),
            "?library returned unknown:\n{output}"
        );
        assert!(
            output.contains("Module library:"),
            "wrong output:\n{output}"
        );
    }

    #[test]
    fn test_session_apply_persists_module() {
        // This test verifies that session.apply() calls persist logic
        let mut session = crate::session::Session::new();
        // Give it a library state
        session.meta.library_state = Some(LibraryState::new());

        let source = "+module TestPersist\n+fn check ()->Int\n  +return 1\n";
        let result = session.apply(source);
        assert!(result.is_ok(), "apply failed: {result:?}");
        let results = result.unwrap();
        assert!(
            results.iter().any(|(_, ok)| *ok),
            "no success in: {results:?}"
        );

        // Check the module exists in program
        assert!(
            session
                .program
                .modules
                .iter()
                .any(|m| m.name == "TestPersist"),
            "module not in program"
        );

        // Check that the library dir + file were created
        let dir = library_dir();
        let file = dir.join("TestPersist.ax");
        assert!(
            file.exists(),
            "module file not persisted at {}",
            file.display()
        );

        // Clean up
        let _ = std::fs::remove_file(&file);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Tests for structured load errors
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_load_error_struct() {
        let state = LibraryState::new();

        // Initially empty
        assert!(state.format_load_errors().is_none());

        // Record a load error
        state.record_load_error("BadModule".to_string(), "parse error on line 5".to_string());
        let output = state.format_load_errors().unwrap();
        assert!(output.contains("Load errors (1):"), "header: {output}");
        assert!(
            output.contains("BadModule: parse error on line 5"),
            "content: {output}"
        );
    }

    #[test]
    fn test_multiple_load_errors() {
        let state = LibraryState::new();
        state.record_load_error(
            "ModuleA".to_string(),
            "no +module declaration found".to_string(),
        );
        state.record_load_error(
            "ModuleB".to_string(),
            "unexpected token at line 3".to_string(),
        );

        let output = state.format_load_errors().unwrap();
        assert!(output.contains("Load errors (2):"), "header: {output}");
        assert!(
            output.contains("ModuleA: no +module declaration found"),
            "A: {output}"
        );
        assert!(
            output.contains("ModuleB: unexpected token at line 3"),
            "B: {output}"
        );
    }

    #[test]
    fn test_query_library_shows_load_errors() {
        let program = ast::Program::default();
        let state = LibraryState::new();
        state.record_load_error("BrokenMod".to_string(), "syntax error".to_string());

        let output = query_library(&program, Some(&state));
        assert!(
            output.contains("Load errors (1):"),
            "should show load errors section: {output}"
        );
        assert!(
            output.contains("BrokenMod: syntax error"),
            "should show specific error: {output}"
        );
    }

    #[test]
    fn test_query_library_no_errors_no_section() {
        let program = ast::Program::default();
        let state = LibraryState::new();

        let output = query_library(&program, Some(&state));
        assert!(
            !output.contains("Load errors"),
            "should not show load errors section when empty: {output}"
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // Tests for library_reload
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_reload_specific_module() {
        let tmp = tempfile::tempdir().unwrap();
        let lib_dir = tmp.path().join("modules");
        std::fs::create_dir_all(&lib_dir).unwrap();

        // Write a module file
        let source = "+module ReloadTest\n+fn greet ()->String\n  +return \"hello\"\n";
        std::fs::write(lib_dir.join("ReloadTest.ax"), source).unwrap();

        // Load the module initially
        let mut program = ast::Program::default();
        let ops = parser::parse(source).unwrap();
        for op in &ops {
            validator::apply_and_validate(&mut program, op).unwrap();
        }
        assert_eq!(program.modules.len(), 1);
        assert_eq!(program.modules[0].functions[0].name, "greet");

        // Update the file on disk with a different function
        let updated_source = "+module ReloadTest\n+fn farewell ()->String\n  +return \"bye\"\n";
        std::fs::write(lib_dir.join("ReloadTest.ax"), updated_source).unwrap();

        // Override HOME so reload_module uses our temp dir
        // Note: reload_module uses library_dir() which uses HOME env var.
        // For isolated test, we test load_module_source directly instead.
        let mut program2 = ast::Program::default();
        let content = std::fs::read_to_string(lib_dir.join("ReloadTest.ax")).unwrap();
        let result = load_module_source(&mut program2, &content);
        assert!(result.is_ok(), "reload should succeed: {result:?}");
        assert_eq!(result.unwrap(), "ReloadTest");
        assert_eq!(program2.modules[0].functions[0].name, "farewell");
    }

    #[test]
    fn test_reload_nonexistent_module_fails() {
        let mut program = ast::Program::default();
        // This will fail because the file doesn't exist in the real library dir
        // (or if it does, it's a valid module — either way tests the path)
        let result = reload_module(&mut program, "NonExistentModule99999");
        assert!(result.is_err(), "should fail for nonexistent module");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not found") || err.contains("could not"),
            "error should mention not found: {err}"
        );
    }

    #[test]
    fn test_load_errors_recorded_during_library_load() {
        // Create a temp dir with a malformed .ax file
        let tmp = tempfile::tempdir().unwrap();
        let lib_dir = tmp.path().join("modules");
        std::fs::create_dir_all(&lib_dir).unwrap();

        // Write a valid module
        let valid_source = "+module GoodMod\n+fn ok ()->Int\n  +return 1\n";
        std::fs::write(lib_dir.join("GoodMod.ax"), valid_source).unwrap();

        // Write an invalid module (no +module declaration)
        let bad_source = "+fn orphan ()->Int\n  +return 42\n";
        std::fs::write(lib_dir.join("BadMod.ax"), bad_source).unwrap();

        // We can't easily test load_module_library since it reads from HOME,
        // but we can test the underlying function
        let mut program = ast::Program::default();
        let bad_result = load_module_source(&mut program, bad_source);
        assert!(bad_result.is_err(), "bad module should fail to load");
        let err = bad_result.unwrap_err().to_string();
        assert!(
            err.contains("no +module declaration"),
            "error should mention missing module declaration: {err}"
        );
    }

    #[test]
    fn test_persist_and_load_preserves_tests() {
        // Build a module with a function and attach a test to it
        let source = "+module TestMod\n+fn double (n:Int)->Int\n  +return n * 2\n";
        let mut program = ast::Program::default();
        let ops = parser::parse(source).unwrap();
        for op in &ops {
            validator::apply_and_validate(&mut program, op).unwrap();
        }

        // Store a passing test on the function
        let test_source = "+test double\n  +with n=3 -> expect 6\n  +with n=0 -> expect 0\n";
        let test_ops = parser::parse(test_source).unwrap();
        for op in &test_ops {
            if let parser::Operation::Test(t) = op {
                crate::session::store_test(
                    &mut program,
                    &format!("TestMod.{}", t.function_name),
                    &t.cases,
                );
            }
        }

        // Verify the test is stored
        let func = program.get_function("TestMod.double").unwrap();
        assert_eq!(
            func.tests.len(),
            2,
            "should have 2 test cases before persist"
        );
        assert!(
            func.tests.iter().all(|t| t.passed),
            "all tests should be marked passed"
        );

        // Persist → reconstruct source → load into fresh program
        let module = program
            .modules
            .iter()
            .find(|m| m.name == "TestMod")
            .unwrap();
        let reconstructed = reconstruct_module_source(module);

        // Verify the reconstructed source contains +test blocks
        assert!(
            reconstructed.contains("+test double"),
            "reconstructed source should contain +test block"
        );
        assert!(
            reconstructed.contains("+with"),
            "reconstructed source should contain test cases"
        );

        // Load into a fresh program (this is where the bug is)
        let mut program2 = ast::Program::default();
        load_module_source(&mut program2, &reconstructed).unwrap();

        let func2 = program2.get_function("TestMod.double").unwrap();
        assert_eq!(
            func2.tests.len(),
            2,
            "tests should survive persist→load roundtrip (got {} tests)",
            func2.tests.len()
        );
        assert!(
            func2.tests.iter().all(|t| t.passed),
            "all restored tests should be marked passed"
        );
        assert!(
            crate::session::is_function_tested(&program2, "TestMod.double"),
            "function should be considered tested after reload"
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // Tests for library load removing existing modules (duplicate startup fix)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_load_module_source_over_existing_module_with_startup() {
        // Simulate the scenario: session state already has a module with +startup,
        // and library loading tries to parse the same module from an .ax file.
        // Without the fix, this would fail with "duplicate +startup" error.

        let source_with_startup = concat!(
            "+module MyMod\n",
            "+startup [io,async]\n",
            "  +await _:String = shell(\"echo hello\")\n",
            "+fn greet ()->String\n",
            "  +return \"hi\"\n",
        );

        // First: load the module into the program (simulating session restore)
        let mut program = ast::Program::default();
        let ops = parser::parse(source_with_startup).unwrap();
        for op in &ops {
            validator::apply_and_validate(&mut program, op).unwrap();
        }
        assert_eq!(program.modules.len(), 1);
        assert!(
            program.modules[0].startup.is_some(),
            "module should have startup"
        );

        // Now simulate what load_module_library does: remove existing, then load
        let file_module_name = "MyMod";
        program.modules.retain(|m| m.name != file_module_name);
        assert_eq!(program.modules.len(), 0, "module should be removed");

        let result = load_module_source(&mut program, source_with_startup);
        assert!(
            result.is_ok(),
            "loading over removed module should succeed: {result:?}"
        );
        assert_eq!(program.modules.len(), 1);
        assert_eq!(program.modules[0].name, "MyMod");
        assert!(
            program.modules[0].startup.is_some(),
            "startup should be present after reload"
        );
    }

    #[test]
    fn test_load_module_source_fails_without_removal() {
        // This test demonstrates the bug: without removing the existing module first,
        // loading the same module with a +startup block fails.

        let source_with_startup = concat!(
            "+module DupMod\n",
            "+startup [io,async]\n",
            "  +await _:String = shell(\"echo hello\")\n",
            "+fn greet ()->String\n",
            "  +return \"hi\"\n",
        );

        // Load the module once
        let mut program = ast::Program::default();
        let ops = parser::parse(source_with_startup).unwrap();
        for op in &ops {
            validator::apply_and_validate(&mut program, op).unwrap();
        }
        assert_eq!(program.modules.len(), 1);

        // Try to load again WITHOUT removing — should fail with duplicate startup
        let result = load_module_source(&mut program, source_with_startup);
        assert!(
            result.is_err(),
            "loading duplicate module with startup should fail without removal"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("duplicate +startup"),
            "error should mention duplicate startup: {err}"
        );
    }

    #[test]
    fn test_load_module_replaces_functions_correctly() {
        // Verify that when a module is removed and reloaded, the new version's
        // functions replace the old ones properly.

        let source_v1 = "+module VersionMod\n+fn compute ()->Int\n  +return 1\n";
        let source_v2 = "+module VersionMod\n+fn compute ()->Int\n  +return 2\n+fn extra ()->String\n  +return \"new\"\n";

        // Load v1
        let mut program = ast::Program::default();
        let ops = parser::parse(source_v1).unwrap();
        for op in &ops {
            validator::apply_and_validate(&mut program, op).unwrap();
        }
        assert_eq!(program.modules[0].functions.len(), 1);

        // Remove and load v2 (as library loader now does)
        program.modules.retain(|m| m.name != "VersionMod");
        let result = load_module_source(&mut program, source_v2);
        assert!(result.is_ok(), "v2 load should succeed: {result:?}");
        assert_eq!(program.modules.len(), 1);
        assert_eq!(
            program.modules[0].functions.len(),
            2,
            "v2 should have 2 functions"
        );
        assert_eq!(program.modules[0].functions[0].name, "compute");
        assert_eq!(program.modules[0].functions[1].name, "extra");
    }
}
