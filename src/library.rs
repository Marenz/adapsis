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

/// Runtime state for the module library — tracks what was loaded and any errors.
#[derive(Debug, Clone)]
pub struct LibraryState {
    /// Module names successfully auto-loaded at startup.
    pub loaded_modules: Vec<String>,
    /// Accumulated load/save error messages this session.
    pub errors: Arc<Mutex<Vec<String>>>,
}

impl LibraryState {
    pub fn new() -> Self {
        Self {
            loaded_modules: Vec::new(),
            errors: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn record_error(&self, msg: String) {
        if let Ok(mut errs) = self.errors.lock() {
            errs.push(msg);
        }
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
    out.push_str(&format!("!module {}\n", module.name));

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

    // Include persisted tests in source reconstruction
    if !func.tests.is_empty() {
        out.push('\n');
        out.push_str(&format!("!test {}\n", func.name));
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
        match std::fs::read_to_string(path) {
            Ok(source) => match load_module_source(program, &source) {
                Ok(module_name) => {
                    eprintln!(
                        "[library] loaded module `{module_name}` from {}",
                        path.display()
                    );
                    state.loaded_modules.push(module_name);
                }
                Err(e) => {
                    let msg = format!("failed to load {}: {e}", path.display());
                    eprintln!("[library] warning: {msg}");
                    state.record_error(msg);
                }
            },
            Err(e) => {
                let msg = format!("could not read {}: {e}", path.display());
                eprintln!("[library] warning: {msg}");
                state.record_error(msg);
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

    module_name.ok_or_else(|| anyhow::anyhow!("no !module declaration found in source"))
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

    // Show errors
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
        let source = "!module TestMod\n+fn greet (name:String)->String\n  +return name\n";
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
        assert!(reconstructed.contains("!module TestMod"));
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
        let source = "!module Probe\n+fn hi ()->String\n  +return \"ok\"\n";
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
        let source = "!module Foo\n+fn bar ()->Int\n  +return 1\n";
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
        let source = "!module Probe\n+fn hi ()->String\n  +return \"ok\"\n+fn answer ()->Int\n  +return 42\n";
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
        let source = "!module Probe\n+fn hi ()->String\n  +return \"ok\"\n";
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

        let source = "!module TestPersist\n+fn check ()->Int\n  +return 1\n";
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
}
