//! Module-less (top-level) Python surface. TEST-ONLY (`#[cfg(test)]`): there is no
//! authority model for module-less declarations (native permissions are keyed by
//! module), so this must not be reachable from native tools or any runtime path.
//!
//! Scope rule: the declaration owner is the EMPTY string, exactly as in the runtime
//! catalog (`types::owned_name("", n) == n`, `catalog_from_program` inserts
//! `program.types` with owner ""). A1 admits type declarations, A2 adds `def`/`async def`
//! functions (bare signature keys, so a bare call resolves exactly as the runtime's
//! top-level-first `Program::get_function`). Shared/lifecycle/route/source declarations
//! (modules only) are refused with an explicit reason.
use super::{types, Context, Diagnostic, LoweredSource};
use crate::{ast, parser as p};
use rustpython_parser::ast as py;
use std::collections::{HashMap, HashSet};

/// Lower a module-less submission to TOP-LEVEL `Operation::Type`/`Operation::Function`s
/// (never wrapped in a `Module`), in source order. `program` supplies the names the
/// collision rules compare against and the signatures/types the bodies resolve against.
/// An existing top-level TYPE of the same name is left to the validator ("duplicate type
/// declaration"); an existing top-level FUNCTION of the same name is refused here
/// because the signature map is keyed by bare name and would otherwise be overwritten
/// silently.
pub(super) fn lower(source: &str, program: &ast::Program) -> Result<LoweredSource, Diagnostic> {
    let cx = Context { source, module: "" };
    let suite = cx.parse()?;
    if suite.is_empty() {
        return Err(cx.error(
            0,
            "top-level submission requires at least one type or function declaration",
        ));
    }
    let mut declared: Vec<(&py::Stmt, p::TypeDecl)> = Vec::new();
    let mut defs: Vec<&py::Stmt> = Vec::new();
    let mut order = Vec::new();
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
                order.push(Item::Type(declared.len()));
                declared.push((stmt, decl));
            }
            py::Stmt::AnnAssign(_) | py::Stmt::Assign(_) => {
                return Err(cx.at(
                    stmt,
                    "shared variables belong to a module; a top-level submission cannot declare them",
                ))
            }
            py::Stmt::FunctionDef(_) | py::Stmt::AsyncFunctionDef(_) => {
                order.push(Item::Function(defs.len()));
                defs.push(stmt);
            }
            _ => {
                return Err(cx.at(
                    stmt,
                    "a top-level submission may only contain @dataclass/@union type declarations \
                     and annotated def/async def functions (no module docstring, imports or statements)",
                ))
            }
        }
    }
    let existing = check_collisions(&cx, program, &declared)?;
    let functions = lower_functions(&cx, program, &declared, &defs, &existing, &names)?;
    let operations: Vec<_> = order
        .iter()
        .map(|item| match item {
            Item::Type(i) => p::Operation::Type(declared[*i].1.clone()),
            Item::Function(i) => p::Operation::Function(functions[*i].clone()),
        })
        .collect();
    let source_map = order
        .iter()
        .enumerate()
        .map(|(index, item)| match item {
            Item::Type(i) => cx.span(declared[*i].0, index),
            Item::Function(i) => cx.span(defs[*i], index),
        })
        .collect();
    cx.finish(operations, source_map)
}

enum Item {
    Type(usize),
    Function(usize),
}

/// Lower the submitted functions against the program's qualified module signatures,
/// its BARE top-level signatures and the program + submitted type catalog. A bare
/// callee that is not a top-level function (module functions are only reachable
/// qualified) is refused by the shared call lowering as an unknown local callable.
fn lower_functions(
    cx: &Context<'_>,
    program: &ast::Program,
    declared: &[(&py::Stmt, p::TypeDecl)],
    defs: &[&py::Stmt],
    existing: &Existing,
    submitted_types: &HashSet<String>,
) -> Result<Vec<p::FunctionDecl>, Diagnostic> {
    if defs.is_empty() {
        return Ok(Vec::new());
    }
    let mut signatures = HashMap::new();
    super::printer::collect_signatures(&program.modules, "", &mut signatures);
    super::printer::collect_toplevel_signatures(&program.functions, &mut signatures);
    let mut catalog = types::catalog_from_program(program);
    for (_, decl) in declared {
        super::lower::insert_parser_type(decl, "", &mut catalog);
    }
    let mut variants = existing.variants.clone();
    for (_, decl) in declared {
        if let p::TypeBody::Union(list) = &decl.body {
            variants.extend(list.iter().map(|v| v.name.clone()));
        }
    }
    let toplevel: HashSet<&str> = program.functions.iter().map(|f| f.name.as_str()).collect();
    let check = |stmt: &py::Stmt, name: &str| -> Result<(), Diagnostic> {
        let refuse = |why: &str| Err(cx.at(stmt, format!("function name {name} {why}")));
        if toplevel.contains(name) {
            return refuse(
                "already exists as a top-level function; the validator rejects duplicates",
            );
        }
        if existing.modules.contains(name) {
            return refuse(
                "collides with a module of the same name; rename the function or the module",
            );
        }
        if variants.contains(name) {
            return refuse("collides with a union variant of the same name; `Name(..)` would be both a call and a constructor");
        }
        if existing.types.contains(name) || submitted_types.contains(name) {
            return refuse(
                "collides with a type of the same name; rename the function or the type",
            );
        }
        Ok(())
    };
    super::lower::toplevel_functions(cx, defs, signatures, &catalog, &check)
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

/// Declaration-time collision rules for TYPES (function rules live in `lower_functions`):
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
) -> Result<Existing, Diagnostic> {
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
    Ok(existing)
}
