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

/// Strict form used by every runtime_targets position (test/evaluate inputs and
/// expected values, non-shared defaults). It passes an empty `variants` slice,
/// which selects the strict arm in `check_runtime_variant`: those positions rely
/// on the parse-time `authored::literal` gate AND this check agreeing, so a
/// non-list call must be refused here too, not merely at parse time.
pub(super) fn check_runtime(expr: &Expr, program: &ast::Program) -> Result<(), String> {
    check_runtime_variant(expr, program, &[])
}

/// Variant-aware runtime check. `variants` is empty in every non-shared position
/// (which makes this behave exactly like the strict `literal()` gate, including
/// refusing any non-`list` call), and non-empty only for a shared default whose
/// annotation is a union — the caller supplies THAT union's own constructor
/// spellings. Admitted variant calls are data-only, so their arguments recurse
/// under the same rule; `list` keeps its shadow check.
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
        Expr::Call { callee, args } if matches!(callee.as_ref(), Expr::Ident(name) if name == "list") =>
        {
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
        // Non-call leaves fall through to `literal()`, which is the original
        // strict behaviour for non-shared positions. The widening is the two
        // variant arms above, and only when the caller supplies a non-empty
        // `variants` set from the shared-default position.
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

#[cfg(test)]
mod strict_contract_tests {
    use super::*;
    use crate::parser::parse_single_expr;

    /// #95 follow-up (a): `check_runtime` is the STRICT form. A non-list call must
    /// be refused even when the argument happens to be a literal. This was red
    /// before the guard because arm 2 accepted every call shape.
    #[test]
    fn strict_check_runtime_refuses_non_list_call() {
        let expr = parse_single_expr("helper(2)").unwrap();
        let program = ast::Program::default();
        assert!(
            check_runtime(&expr, &program).is_err(),
            "non-list call must be refused"
        );
    }

    #[test]
    fn strict_check_runtime_still_admits_list_construction() {
        let expr = parse_single_expr("list(1, 2)").unwrap();
        let program = ast::Program::default();
        assert!(check_runtime(&expr, &program).is_ok());
    }

    /// The variant-aware form keeps admitting an annotation-union constructor
    /// while still refusing an unrelated call.
    #[test]
    fn variant_form_admits_only_the_supplied_constructors() {
        let program = ast::Program::default();
        let variants = vec!["Leaf".to_string(), "Empty".to_string()];
        assert!(
            check_runtime_variant(&parse_single_expr("Leaf(7)").unwrap(), &program, &variants)
                .is_ok()
        );
        assert!(
            check_runtime_variant(&parse_single_expr("Empty").unwrap(), &program, &variants)
                .is_ok()
        );
        assert!(
            check_runtime_variant(
                &parse_single_expr("helper(2)").unwrap(),
                &program,
                &variants
            )
            .is_err()
        );
        assert!(
            check_runtime_variant(
                &parse_single_expr("Leaf(compute())").unwrap(),
                &program,
                &variants
            )
            .is_err()
        );
    }
}
