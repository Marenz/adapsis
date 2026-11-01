//! Module-less (top-level) Python surface. TEST-ONLY (`#[cfg(test)]`): there is no
//! authority model for module-less declarations (native permissions are keyed by
//! module), so this must not be reachable from native tools or any runtime path.
//!
//! Scope rule: the declaration owner is the EMPTY string, exactly as in the runtime
//! catalog (`types::owned_name("", n) == n`, `catalog_from_program` inserts
//! `program.types` with owner ""). This slice (A1) admits type declarations only;
//! top-level functions (A2) and shared/lifecycle/route/source declarations (modules
//! only) are refused with an explicit reason.
use super::{types, Context, Diagnostic, LoweredSource};
use crate::{ast, parser as p};
use rustpython_parser::ast as py;
use std::collections::HashSet;

/// Lower a module-less submission to TOP-LEVEL `Operation::Type`s (never wrapped in a
/// `Module`). `program` supplies only the names the collision rules compare against;
/// an existing top-level type of the same name is NOT consulted here because the
/// validator itself rejects duplicates at apply time ("duplicate type declaration").
pub(super) fn lower(source: &str, program: &ast::Program) -> Result<LoweredSource, Diagnostic> {
    let cx = Context { source, module: "" };
    let suite = cx.parse()?;
    if suite.is_empty() {
        return Err(cx.error(
            0,
            "top-level submission requires at least one type declaration",
        ));
    }
    let mut declared: Vec<(&py::Stmt, p::TypeDecl)> = Vec::new();
    let mut names = HashSet::new();
    for stmt in &suite {
        match stmt {
            py::Stmt::ClassDef(class) => {
                let decl = types::lower_type_decl(&cx, class)?;
                if !names.insert(decl.name.clone()) {
                    return Err(cx.at(
                        stmt,
                        format!("duplicate type declaration {}; submit each type once", decl.name),
                    ));
                }
                declared.push((stmt, decl));
            }
            py::Stmt::AnnAssign(_) | py::Stmt::Assign(_) => {
                return Err(cx.at(
                    stmt,
                    "shared variables belong to a module; a top-level submission cannot declare them",
                ))
            }
            py::Stmt::FunctionDef(_) | py::Stmt::AsyncFunctionDef(_) => {
                return Err(cx.at(
                    stmt,
                    "top-level functions are not supported by this surface yet; declare them in a module",
                ))
            }
            _ => {
                return Err(cx.at(
                    stmt,
                    "a top-level submission may only contain @dataclass/@union type declarations \
                     (no module docstring, imports or statements)",
                ))
            }
        }
    }
    check_collisions(&cx, program, &declared)?;
    let operations = declared
        .iter()
        .map(|(_, decl)| p::Operation::Type(decl.clone()))
        .collect();
    let source_map = declared
        .iter()
        .enumerate()
        .map(|(index, (stmt, _))| cx.span(*stmt, index))
        .collect();
    cx.finish(operations, source_map)
}

/// Names that are already declared anywhere in the program, by kind.
#[derive(Default)]
struct Existing {
    types: HashSet<String>,
    variants: HashSet<String>,
    modules: HashSet<String>,
}

fn collect(modules: &[ast::Module], out: &mut Existing) {
    for module in modules {
        out.modules.insert(module.name.clone());
        for ty in &module.types {
            record(ty, out);
        }
        collect(&module.modules, out);
    }
}

fn record(ty: &ast::TypeDecl, out: &mut Existing) {
    out.types.insert(ty.name().to_owned());
    if let ast::TypeDecl::TaggedUnion(union) = ty {
        out.variants
            .extend(union.variants.iter().map(|v| v.name.clone()));
    }
}

/// Declaration-time collision rules for TYPES (function/variant rules belong to A2):
/// - a type must not share a spelling with a module (`Geo` would be both a type and
///   a namespace);
/// - a type must not share a spelling with ANY union variant (existing or submitted):
///   `Name(..)` would be both a struct constructor and a variant constructor (#95
///   class), and the tree walker prefers variants;
/// - symmetrically, a variant must not share a spelling with ANY type name.
/// Two unions that both declare the same variant (`Some`) are NOT rejected here: that
/// is a valid Adapsis program whose ambiguity is reported at the use site.
fn check_collisions(
    cx: &Context<'_>,
    program: &ast::Program,
    declared: &[(&py::Stmt, p::TypeDecl)],
) -> Result<(), Diagnostic> {
    let mut existing = Existing::default();
    for ty in &program.types {
        record(ty, &mut existing);
    }
    collect(&program.modules, &mut existing);
    let mut types = existing.types.clone();
    let mut variants = existing.variants.clone();
    for (_, decl) in declared {
        types.insert(decl.name.clone());
        if let p::TypeBody::Union(list) = &decl.body {
            variants.extend(list.iter().map(|v| v.name.clone()));
        }
    }
    for (stmt, decl) in declared {
        if existing.modules.contains(&decl.name) {
            return Err(cx.at(
                *stmt,
                format!(
                    "type name {0} collides with module {0}; rename the type or the module",
                    decl.name
                ),
            ));
        }
        if variants.contains(&decl.name) {
            return Err(cx.at(
                *stmt,
                format!(
                    "type name {0} collides with a union variant named {0}; rename the type or the variant",
                    decl.name
                ),
            ));
        }
        if let p::TypeBody::Union(list) = &decl.body {
            for variant in list {
                if types.contains(&variant.name) {
                    return Err(cx.at(
                        *stmt,
                        format!(
                            "union variant {0} collides with a type named {0}; rename the variant or the type",
                            variant.name
                        ),
                    ));
                }
            }
        }
    }
    Ok(())
}
