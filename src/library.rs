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

    // Always create the directory on startup so persist can write later
    let dir = match ensure_library_dir() {
        Ok(d) => d,
        Err(e) => {
            let msg = format!("could not create library dir: {e}");
            eprintln!("[library] {msg}");
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
    } else {
        out.push_str("Library state not initialized.\n");
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
