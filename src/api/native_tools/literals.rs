//! Data-only literal constructors with runtime shadowing checks.
use crate::{ast, parser::Expr};

pub(super) fn literal(expr: &Expr) -> Result<(), String> {
    match expr {
        Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) | Expr::String(_) => Ok(()),
        Expr::Unary { expr, .. } => literal(expr),
        Expr::StructLiteral(fields) => fields.iter().try_for_each(|field| literal(&field.value)),
        Expr::Call { callee, args } if matches!(callee.as_ref(), Expr::Ident(name) if name == "list") => {
            args.iter().try_for_each(literal)
        }
        _ => Err("native inputs, exact expectations and shared defaults must be literals; only checked list construction is supported, not calls or references".into()),
    }
}

pub(super) fn check_runtime(expr: &Expr, program: &ast::Program) -> Result<(), String> {
    check_runtime_variant(expr, program, &[])
}

/// Variant-aware runtime check. `variants` is empty everywhere except the shared
/// default position for a union annotation, where the caller supplies that
/// union's own constructor spellings. Admitted variant calls are data-only, so
/// their arguments recurse under the same rule; `list` keeps its shadow check.
pub(super) fn check_runtime_variant(
    expr: &Expr,
    program: &ast::Program,
    variants: &[String],
) -> Result<(), String> {
    match expr {
        Expr::Call { callee, args } if matches!(callee.as_ref(), Expr::Ident(name) if variants.iter().any(|v| v == name)) => {
            args.iter()
                .try_for_each(|arg| check_runtime_variant(arg, program, variants))
        }
        Expr::Call { args, .. } => {
            require_unshadowed(program, "list")?;
            args.iter()
                .try_for_each(|arg| check_runtime_variant(arg, program, variants))
        }
        Expr::Unary { expr, .. } => check_runtime_variant(expr, program, variants),
        Expr::StructLiteral(fields) => fields
            .iter()
            .try_for_each(|field| check_runtime_variant(&field.value, program, variants)),
        // Zero-payload variant default: the bare constructor spelling.
        Expr::Ident(name) if variants.iter().any(|v| v == name) => Ok(()),
        other => literal(other),
    }
}

pub(super) fn require_unshadowed(program: &ast::Program, name: &str) -> Result<(), String> {
    let collides = crate::python_frontend::runtime_shadowed(program, name);
    if collides {
        Err(format!(
            "cannot guarantee literal/frontend helper `{name}` semantics: a user-defined symbol shadows this builtin"
        ))
    } else {
        Ok(())
    }
}

/// Without frontend-origin metadata, conservatively refuse Python execution
/// while any runtime builtin can be intercepted by a user-defined symbol.
/// This also protects previously installed Python after later Adapsis edits.
pub(super) fn require_python_environment(program: &ast::Program) -> Result<(), String> {
    for name in crate::builtins::BUILTINS
        .iter()
        .chain(crate::builtins::IO_BUILTINS.iter())
        .flat_map(|builtin| std::iter::once(builtin.name).chain(builtin.aliases.iter().copied()))
    {
        require_unshadowed(program, name)?;
    }
    Ok(())
}

/// #95: an ambiguous reachable construct is refused, but only for the entry
/// point actually being executed. Legacy collisions elsewhere in the program
/// must not disable unrelated Python functions (they cannot be repaired by any
/// native tool). Declaration-time checks prevent new Python-authored collisions.
pub(super) fn require_python_entry(program: &ast::Program, entry: &str) -> Result<(), String> {
    require_python_environment(program)?;
    crate::python_frontend::check_constructor_environment(program, entry)
}

/// A declaration in this same mutation is not present in the runtime snapshot
/// yet. It must not intercept a shared default after declarations are applied.
pub(super) fn check_new_shared_defaults(ops: &[crate::parser::Operation]) -> Result<(), String> {
    if declares_list(ops) && has_list_default(ops) {
        return Err("literal list shared default conflicts with a list symbol declared in the same mutation".into());
    }
    Ok(())
}

fn declares_list(ops: &[crate::parser::Operation]) -> bool {
    use crate::parser::{Operation, TypeBody};
    ops.iter().any(|op| match op {
        Operation::Module(module) => declares_list(&module.body),
        Operation::Function(function) => function.name == "list",
        Operation::Type(ty) => matches!(&ty.body, TypeBody::Union(variants)
            if variants.iter().any(|variant| variant.name == "list")),
        _ => false,
    })
}

fn has_list_default(ops: &[crate::parser::Operation]) -> bool {
    use crate::parser::Operation;
    ops.iter().any(|op| match op {
        Operation::Module(module) => has_list_default(&module.body),
        Operation::SharedVar(value) => contains_list(&value.default),
        _ => false,
    })
}

fn contains_list(expr: &Expr) -> bool {
    match expr {
        Expr::Call { callee, .. } => matches!(callee.as_ref(), Expr::Ident(name) if name == "list"),
        Expr::StructLiteral(fields) => fields.iter().any(|field| contains_list(&field.value)),
        Expr::Unary { expr, .. } => contains_list(expr),
        _ => false,
    }
}
