//! Server-derived reference roles. Input JSON cannot provide this evidence.
//! Source shape is collected without runtime lookup; this module is invoked
//! only after target-module Write and explicit-reference Execute authorization.
use super::{
    expressions::Ty,
    lower::Preflight,
    types::{self, TypeCatalog, TypeInfo},
};
use crate::{ast, parser as p};
use std::collections::HashSet;
pub(super) mod usage;

pub struct NativeReferences {
    pub functions: Vec<String>,
    pub types: Vec<String>,
    module: String,
    constructors: HashSet<String>,
    shared: HashSet<String>,
    submitted: HashSet<String>,
    declared_types: Vec<String>,
    builtin_collisions: HashSet<String>,
    catalog: TypeCatalog,
}

impl NativeReferences {
    pub fn constructor(&self, module: &str, name: &str) -> bool {
        self.module == module && self.constructors.contains(name)
    }
    pub fn shared(&self, module: &str) -> HashSet<String> {
        if self.module == module {
            self.shared.clone()
        } else {
            HashSet::new()
        }
    }
    pub fn shadows_builtin(&self, name: &str) -> bool {
        self.builtin_collisions.contains(name)
    }
}

pub(crate) const COLLISION: &str = "Python constructor role cannot be preserved safely: a declaration is shadowed or ambiguous; use a distinct constructor name";

pub fn resolve_native_references(
    preflight: &Preflight,
    module: &str,
    authorized: &ast::Program,
    full: &ast::Program,
) -> Result<NativeReferences, String> {
    let full_catalog = types::catalog_from_program(full);
    let mut catalog = types::catalog_from_program(authorized);
    for decl in &preflight.declared_types {
        super::lower::insert_parser_type(decl, module, &mut catalog);
    }
    let own = authorized.modules.iter().find(|m| m.name == module);
    let submitted: HashSet<_> = preflight
        .declared_functions
        .iter()
        .map(|f| types::owned_name(module, f))
        .collect();
    // #95: a function and a union variant must never share a spelling, because
    // the tree walker prefers variants while the VM prefers functions. That state
    // is unreachable for Python execution and cannot be repaired by any native
    // tool, so refuse it at declaration time. The check is BIDIRECTIONAL and uses
    // the UNION of the AUTHORIZED post-submission catalog and the FULL program
    // catalog. The full half is load-bearing: a variant in a module the caller
    // cannot read still enters the same unreachable state once a function takes
    // its spelling, and that state is then permanent for every bare reachable
    // entry. Accepting this costs a constant-message name-existence oracle; the
    // alternative is a natively enterable brick. Cases covered:
    //   (a) a submitted function name matching any variant (submitted, authorized
    //       or hidden);
    //   (b) a submitted variant name matching any function (existing or submitted).
    for function in &preflight.declared_functions {
        if variant_declared_anywhere(&catalog, function)
            || variant_declared_anywhere(&full_catalog, function)
        {
            return Err("Python function name collides with a union variant name; rename the function or the variant".into());
        }
    }
    for decl in &preflight.declared_types {
        if let p::TypeBody::Union(variants) = &decl.body {
            for variant in variants {
                let collides = function_named(full, &variant.name)
                    || preflight.declared_functions.contains(&variant.name);
                if collides {
                    return Err("Python union variant name collides with a function name; rename the function or the variant".into());
                }
            }
        }
    }
    let mut functions = preflight.references.clone();
    let mut type_names = preflight.type_references.clone();
    let mut constructors = HashSet::new();
    let shared: HashSet<String> = own
        .map(|m| m.shared_vars.iter().map(|v| v.name.clone()).collect())
        .unwrap_or_default();
    let candidates: HashSet<_> = preflight
        .local_calls
        .iter()
        .chain(&preflight.value_references)
        .chain(&preflight.bare_calls)
        .collect();
    let mut builtin_collisions = HashSet::new();
    for info in catalog.values() {
        if let TypeInfo::Union {
            module: owner,
            variants,
        } = info
        {
            if owner == module {
                builtin_collisions.extend(
                    variants
                        .iter()
                        .filter(|(name, _)| crate::builtins::is_builtin(name))
                        .map(|(name, _)| name.clone()),
                );
            }
        }
    }
    for name in candidates {
        if shared.contains(name) && !preflight.bare_calls.contains(name) {
            continue;
        }
        let function = submitted.contains(&types::owned_name(module, name))
            || own.is_some_and(|m| m.functions.iter().any(|f| f.name == *name));
        let variants: Vec<_> = catalog
            .iter()
            .filter_map(|(key, info)| match info {
                TypeInfo::Union {
                    module: owner,
                    variants,
                } if owner == module && variants.iter().any(|(variant, _)| variant == name) => {
                    Some(key)
                }
                _ => None,
            })
            .collect();
        if variants.len() > 1 || (function && !variants.is_empty()) {
            return Err(COLLISION.into());
        }
        if let Some(key) = variants.first() {
            if function_named(full, name)
                || preflight.declared_functions.contains(name)
                || crate::builtins::is_builtin(name)
                || name == "None"
            {
                return Err(COLLISION.into());
            }
            constructors.insert(name.clone());
            type_names.push(types::bare_name(key).into());
        } else if function && preflight.local_calls.contains(name) {
            functions.push(types::owned_name(module, name));
        } else if preflight.local_calls.contains(name) || preflight.value_references.contains(name)
        {
            return Err("Python bare call has no supported target-module function or constructor; external bare variants are unsupported".into());
        }
    }
    functions.sort();
    functions.dedup();
    type_names.sort();
    type_names.dedup();
    Ok(NativeReferences {
        functions,
        types: type_names,
        module: module.into(),
        constructors,
        shared,
        builtin_collisions,
        submitted,
        declared_types: preflight
            .declared_types
            .iter()
            .map(|t| t.name.clone())
            .collect(),
        catalog,
    })
}

fn function_named(program: &ast::Program, name: &str) -> bool {
    fn modules(modules_: &[ast::Module], name: &str) -> bool {
        modules_
            .iter()
            .any(|m| m.functions.iter().any(|f| f.name == name) || modules(&m.modules, name))
    }
    program.functions.iter().any(|f| f.name == name) || modules(&program.modules, name)
}

/// Metadata-only recheck of the *reachable* call graph from one entry point.
/// VM call dispatch prefers user functions over union variants while the tree
/// walker prefers variants; if a reachable bare call spelling is ambiguous the
/// two tiers disagree, so refuse before execution. Legacy collisions elsewhere
/// in the program must still allow unrelated functions to evaluate (#95).
pub fn check_constructor_environment(program: &ast::Program, entry: &str) -> Result<(), String> {
    let variants = all_variant_names(program);
    let mut pending = vec![program.qualify_function_name(entry)];
    let mut seen = HashSet::new();
    while let Some(target) = pending.pop() {
        if !seen.insert(target.clone()) {
            continue;
        }
        let Some(function) = program.get_function(&target) else {
            continue;
        };
        reachable_calls(program, &function.body, &mut pending);
        // A function that shares a spelling with any declared variant is only a
        // hazard if it is itself reachable or dispatched bare.
        if variants.contains(types::bare_name(&target)) {
            return Err(COLLISION.into());
        }
    }
    Ok(())
}

fn variant_declared_anywhere(catalog: &TypeCatalog, name: &str) -> bool {
    catalog.values().any(|info| match info {
        TypeInfo::Union { variants, .. } => variants.iter().any(|(variant, _)| variant == name),
        TypeInfo::Struct { .. } => false,
    })
}

fn all_variant_names(program: &ast::Program) -> HashSet<String> {
    fn types(list: &[ast::TypeDecl], out: &mut HashSet<String>) {
        for ty in list {
            if let ast::TypeDecl::TaggedUnion(u) = ty {
                out.extend(u.variants.iter().map(|v| v.name.clone()));
            }
        }
    }
    fn modules(list: &[ast::Module], out: &mut HashSet<String>) {
        for module in list {
            types(&module.types, out);
            modules(&module.modules, out);
        }
    }
    let mut out = HashSet::new();
    types(&program.types, &mut out);
    modules(&program.modules, &mut out);
    out
}

fn reachable_calls(program: &ast::Program, body: &[ast::Statement], out: &mut Vec<String>) {
    fn expr(program: &ast::Program, value: &ast::Expr, out: &mut Vec<String>) {
        match value {
            ast::Expr::Call(call) => {
                out.push(program.qualify_function_name(&call.callee));
                for arg in &call.args {
                    expr(program, arg, out);
                }
            }
            ast::Expr::Binary { left, right, .. } => {
                expr(program, left, out);
                expr(program, right, out);
            }
            ast::Expr::Unary { expr: inner, .. } | ast::Expr::FieldAccess { base: inner, .. } => {
                expr(program, inner, out);
            }
            ast::Expr::StructInit { fields, .. } => {
                for field in fields {
                    expr(program, &field.value, out);
                }
            }
            _ => {}
        }
    }
    for statement in body {
        use ast::StatementKind::*;
        match &statement.kind {
            Let { value, .. } | Set { value, .. } | Return { value } | Yield { value } => {
                expr(program, value, out)
            }
            Call { call, .. } | Await { call, .. } | Spawn { call, .. } => {
                out.push(program.qualify_function_name(&call.callee));
                for arg in &call.args {
                    expr(program, arg, out);
                }
            }
            Check { condition, .. } => expr(program, condition, out),
            While {
                condition, body, ..
            } => {
                expr(program, condition, out);
                reachable_calls(program, body, out);
            }
            Branch {
                condition,
                then_body,
                else_body,
            } => {
                expr(program, condition, out);
                reachable_calls(program, then_body, out);
                reachable_calls(program, else_body, out);
            }
            Each { iterator, body, .. } => {
                expr(program, iterator, out);
                reachable_calls(program, body, out);
            }
            Match {
                expr: subject,
                arms,
            } => {
                expr(program, subject, out);
                for arm in arms {
                    reachable_calls(program, &arm.body, out);
                }
            }
            Event(ast::EventOp::Emit { value, .. }) => expr(program, value, out),
            Source(_) | Event(_) => {}
        }
    }
}

/// Retain an owner-qualified dependency closure, including callee signatures and
/// recursive payloads. Never seed this closure from unfiltered private schemas.
pub fn retain_native_dependencies(
    program: &mut ast::Program,
    roles: &NativeReferences,
) -> Result<(), String> {
    let mut needed = HashSet::new();
    for name in &roles.types {
        add_name(&roles.catalog, &roles.module, name, &mut needed)?;
    }
    for target in &roles.functions {
        if roles.submitted.contains(target) {
            continue;
        }
        if let (Some((owner, _)), Some(f)) = (target.rsplit_once('.'), program.get_function(target))
        {
            for param in &f.params {
                add_type(
                    &roles.catalog,
                    &types::ast_type_to_ty_in(&param.ty, owner),
                    &mut needed,
                )?;
            }
            add_type(
                &roles.catalog,
                &types::ast_type_to_ty_in(&f.return_type, owner),
                &mut needed,
            )?;
        }
    }
    // Submitted declarations can depend on existing fields/payload types even
    // when no function annotation names those dependencies directly.
    for name in &roles.declared_types {
        add_name(&roles.catalog, &roles.module, name, &mut needed)?;
    }
    program.types.retain(|t| needed.contains(t.name()));
    for module in &mut program.modules {
        module
            .types
            .retain(|t| needed.contains(&types::owned_name(&module.name, t.name())));
        module.functions.retain(|f| {
            roles
                .functions
                .contains(&types::owned_name(&module.name, &f.name))
        });
    }
    program.rebuild_function_index();
    Ok(())
}

fn add_name(
    catalog: &TypeCatalog,
    owner: &str,
    name: &str,
    needed: &mut HashSet<String>,
) -> Result<(), String> {
    let Some(key) = types::resolve_name(catalog, owner, name)? else {
        return Ok(());
    };
    if !needed.insert(key.clone()) {
        return Ok(());
    }
    match &catalog[&key] {
        TypeInfo::Struct { fields } => {
            for (_, ty) in fields {
                add_type(catalog, ty, needed)?;
            }
        }
        TypeInfo::Union { variants, .. } => {
            for (_, payload) in variants {
                for ty in payload {
                    add_type(catalog, ty, needed)?;
                }
            }
        }
    }
    Ok(())
}

fn add_type(catalog: &TypeCatalog, ty: &Ty, needed: &mut HashSet<String>) -> Result<(), String> {
    match ty {
        Ty::Named(name) => {
            let (owner, name) = name.rsplit_once('.').unwrap_or(("", name));
            add_name(catalog, owner, name, needed)
        }
        Ty::List(t) | Ty::Set(t) | Ty::Option(t) | Ty::Result(t) => add_type(catalog, t, needed),
        Ty::Map(k, v) => {
            add_type(catalog, k, needed)?;
            add_type(catalog, v, needed)
        }
        _ => Ok(()),
    }
}
