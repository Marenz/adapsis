//! Persistent module library — auto-loads modules from ~/.config/adapsis/modules/
//! and persists module changes back to disk.
//!
//! This provides a permanent library layer that works across git worktrees and sessions.
//! Modules are stored as reconstructed `.ax` source files, one per module.

use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::ast;
use crate::parser;
use crate::typeck;
use crate::validator;

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

    // Nested modules (rare but supported in AST)
    for sub in &module.modules {
        // Nested modules can't be expressed in current syntax, but emit a comment
        out.push_str(&format!(
            "# nested module {} ({} types, {} functions) — not persisted\n",
            sub.name,
            sub.types.len(),
            sub.functions.len()
        ));
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
        reconstruct_stmt(&mut out, stmt, 1);
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

fn reconstruct_stmt(out: &mut String, stmt: &ast::Statement, indent: usize) {
    typeck::reconstruct_stmt_pub(out, stmt, indent);
}

/// Load all `.ax` files from the module library directory into the program.
/// Files are loaded in sorted filename order for determinism.
/// Malformed files produce a warning but do not abort the process.
/// Returns the list of successfully loaded module names.
pub fn load_module_library(program: &mut ast::Program) -> Vec<String> {
    let dir = library_dir();
    if !dir.exists() {
        return vec![];
    }

    let mut files: Vec<PathBuf> = match std::fs::read_dir(&dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "ax"))
            .collect(),
        Err(e) => {
            eprintln!(
                "[library] warning: could not read module library dir {}: {e}",
                dir.display()
            );
            return vec![];
        }
    };

    // Sort by filename for deterministic load order
    files.sort();

    let mut loaded = Vec::new();
    for path in &files {
        match std::fs::read_to_string(path) {
            Ok(source) => match load_module_source(program, &source) {
                Ok(module_name) => {
                    eprintln!(
                        "[library] loaded module `{module_name}` from {}",
                        path.display()
                    );
                    loaded.push(module_name);
                }
                Err(e) => {
                    eprintln!("[library] warning: failed to load {}: {e}", path.display());
                }
            },
            Err(e) => {
                eprintln!("[library] warning: could not read {}: {e}", path.display());
            }
        }
    }

    if !loaded.is_empty() {
        eprintln!(
            "[library] auto-loaded {} module(s) from {}",
            loaded.len(),
            dir.display()
        );
    }

    loaded
}

/// Parse and apply a single module source string into the program.
/// Returns the module name on success.
fn load_module_source(program: &mut ast::Program, source: &str) -> Result<String> {
    let operations = parser::parse(source)?;
    let mut module_name = None;

    for op in &operations {
        match op {
            parser::Operation::Test(_)
            | parser::Operation::Trace(_)
            | parser::Operation::Eval(_)
            | parser::Operation::Query(_)
            | parser::Operation::Plan(_)
            | parser::Operation::Roadmap(_)
            | parser::Operation::Mock { .. }
            | parser::Operation::Unmock => {
                // Skip transient operations
                continue;
            }
            parser::Operation::Module(m) => {
                module_name = Some(m.name.clone());
            }
            _ => {}
        }
        match validator::apply_and_validate(program, op) {
            Ok(_msg) => {}
            Err(e) => {
                return Err(e);
            }
        }
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
pub fn persist_affected_modules(program: &ast::Program, module_names: &BTreeSet<String>) {
    for name in module_names {
        if let Some(module) = program.modules.iter().find(|m| &m.name == name) {
            if let Err(e) = persist_module(module) {
                eprintln!("[library] warning: failed to persist module `{name}`: {e}");
            } else {
                eprintln!("[library] persisted module `{name}` to library");
            }
        }
    }
}

/// Format a summary of which library modules are loaded, for the ?library query.
pub fn query_library(program: &ast::Program, loaded_modules: &[String]) -> String {
    let dir = library_dir();
    let mut out = String::new();

    out.push_str(&format!("Module library: {}\n", dir.display()));

    if loaded_modules.is_empty() {
        out.push_str("No library modules loaded.\n");
    } else {
        out.push_str(&format!(
            "Loaded {} module(s) at startup:\n",
            loaded_modules.len()
        ));
        for name in loaded_modules {
            // Find the module in the program to show stats
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

    // Also list .ax files on disk
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
            if !files.is_empty() {
                out.push_str(&format!("\nLibrary files on disk ({}):\n", files.len()));
                for f in &files {
                    out.push_str(&format!("  {f}.ax\n"));
                }
            }
        }
    }

    out
}
