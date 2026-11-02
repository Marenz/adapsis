//! G1: the runtime's non-struct `FieldAccess` forms, spelled as Python PROPERTIES.
//!
//! The tree walker gives `FieldAccess` meaning beyond struct fields: `.len`/`.is_empty` on
//! a String or List and `.is_ok`/`.is_err`/`.unwrap` on a Result. Lowering `s.len` to
//! `FieldAccess{s, "len"}` is the only spelling with the SAME AST (`len(s)` would be a
//! `Call`, `s.len()` a different node). The text is therefore NOT runnable CPython: `str`
//! has no `.len` and `Result` is an Adapsis type. That is the same position as `Ok(..)`,
//! `@union` and the reserved helpers; this frontend never executes Python.
//!
//! Scope is deliberately small and type-gated. A struct field of the same name always wins
//! (the caller consults this only for a non-`Named` base); `error`, `unwrap_err`,
//! `is_some`, `is_none` and Map/Set `len` are not implemented here or refused by the tree.
//! Execution tiers: the VM's `GetField` reads structs only, so `vm/field_admission.rs`
//! (#114) sends any function containing these accesses to the tree walker before execution.
use super::expressions::Ty;

/// Static result type of `<base>.<field>` when it is one of the supported pseudo-fields.
pub(super) fn pseudo_field_type(base: &Ty, field: &str) -> Option<Ty> {
    match (base, field) {
        (Ty::Str | Ty::List(_), "len") => Some(Ty::Int),
        (Ty::Str | Ty::List(_), "is_empty") => Some(Ty::Bool),
        (Ty::Result(_), "is_ok" | "is_err") => Some(Ty::Bool),
        (Ty::Result(inner), "unwrap") => Some(inner.as_ref().clone()),
        _ => None,
    }
}

/// The reason shown for a non-struct base that is not a supported pseudo-field access.
pub(super) fn refusal(base: &Ty, field: &str) -> String {
    match (base, field) {
        (Ty::Map(..) | Ty::Set(_), "len" | "is_empty") => {
            format!("`.{field}` is not supported on Map/Set (the runtime rejects it); use len(x)")
        }
        (Ty::Result(_), "len" | "is_empty") => {
            format!(
                "`.{field}` is not supported on a Result (the runtime rejects it); unwrap it first"
            )
        }
        _ => "field access requires a struct-typed base name; supported non-struct properties are \
              .len/.is_empty on str or list and .is_ok/.is_err/.unwrap on a Result"
            .to_string(),
    }
}
