//! Slice 3: shared-variable defaults are data-only and type-checked against the
//! declared annotation using the complete local type catalog. This module never
//! executes or resolves runtime signatures; it delegates to the same data-only
//! lowering used by `Expressions::lower_shared_default`.
use super::{
    Context, Diagnostic,
    expressions::{Ty, compatible},
    types::{self, TypeCatalog},
};
use crate::parser as p;
use rustpython_parser::ast as py;
use std::collections::HashMap;

pub(super) fn check(
    cx: &Context<'_>,
    default: &py::Expr,
    expected: &Ty,
    types: &TypeCatalog,
) -> Result<(), Diagnostic> {
    let env = HashMap::new();
    let signatures = HashMap::new();
    let shared_vars = HashMap::new();
    // The caller supplies the complete module catalog; no clone is needed.
    let expressions = super::expressions::Expressions {
        cx,
        env: &env,
        signatures: &signatures,
        types,
        shared_vars: &shared_vars,
        effects: &[],
    };
    let (_expr, actual) = expressions.lower_shared_default(default)?;
    if !compatible(expected, &actual) {
        // An empty list literal is the one narrow exception: `[]` cannot infer an
        // element type, so allow it only when the annotation supplies one.
        let empty_list = matches!(default, py::Expr::List(list) if list.elts.is_empty())
            && matches!(expected, Ty::List(_));
        if !empty_list {
            return Err(cx.at(
                default,
                format!("shared default type mismatch: expected {expected:?}, got {actual:?}"),
            ));
        }
    }
    Ok(())
}
