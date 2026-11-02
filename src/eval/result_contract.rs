//! Approved Adapsis Result contract (#82/#97/#86), slice A.
//!
//! Only `wrap_success` is wired into existing runtime paths in this slice: it replaces
//! three identical success-wrapping blocks without changing their behavior. Invocation,
//! Check, await, VM unwind and frontend migration are later slices. In particular, the
//! legacy catch-all `+call Result` and test-error formatting are NOT changed here.
//!
//! At a genuine invocation boundary, declared Result<T> wraps successful plain T once;
//! explicit Ok/Err is preserved. A typed language failure becomes Err at that boundary,
//! but an operational/internal error remains an execution error. Ordinary calls and
//! user awaits preserve Result. `+call T` extracts one Ok or propagates an explicit Err;
//! `+call Result` explicitly captures language failure. These are Adapsis semantics,
//! not Python inference. Never apply invocation normalization to a branch/block return.
//!
//! Nested Result requires an explicit outer constructor: otherwise Ok(inner) could mean
//! either a complete outer Result or an inner value waiting for auto-wrapping. This module
//! checks wrapper depth, not arbitrary payload types; general type validation is separate.
use super::Value;
use crate::ast::Type;
use anyhow::Result;
use std::fmt;

/// Language control flow, deliberately distinct from untyped operational anyhow errors.
/// Match this type, never its Display text, when deciding whether a failure is catchable.
#[allow(dead_code)] // Constructed by Check/VM propagation in the later B/C migration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LanguageFailure {
    Check { label: String },
    Propagation { message: String },
}

impl fmt::Display for LanguageFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Check { label } => f.write_str(label),
            Self::Propagation { message } => f.write_str(message),
        }
    }
}

impl std::error::Error for LanguageFailure {}

/// Existing behavior: wrap a success value unless it is already an explicit Result.
/// This is NOT typed invocation normalization; it intentionally retains legacy behavior
/// even for nested Result values until the invocation sites are migrated together.
pub(super) fn wrap_success(value: Value) -> Value {
    match value {
        value @ (Value::Ok(_) | Value::Err(_)) => value,
        value => Value::Ok(Box::new(value)),
    }
}

/// Finish ONE invocation. For non-Result declarations neither successes nor errors are
/// changed. Operational errors (including a wrapped operational cause) never turn into
/// data merely because the function declares Result. Context-wrapped language failures
/// are recognized by type and preserve their original label/message, not context prose.
#[allow(dead_code)] // Contract API; do not wire independently into existing entry paths.
pub(crate) fn complete_invocation(
    function: &str,
    return_type: &Type,
    outcome: Result<Value>,
) -> Result<Value> {
    let Type::Result(inner) = return_type else {
        return outcome;
    };
    let value = capture_language_failure(outcome)?;
    if matches!(inner.as_ref(), Type::Result(_)) {
        require_explicit_nested_result(function, return_type, &value)?;
        Ok(value)
    } else {
        Ok(wrap_success(value))
    }
}

/// `+call x:T = f()` (also no-binding +call): unwrap one Ok, turn explicit Err into
/// nonlocal language failure, preserve plain scalar values and all existing failures.
#[allow(dead_code)] // B/C must change tree and VM call sites together.
pub(crate) fn propagate_call(outcome: Result<Value>) -> Result<Value> {
    match outcome? {
        Value::Ok(inner) => Ok(*inner),
        Value::Err(message) => Err(LanguageFailure::Propagation { message }.into()),
        value => Ok(value),
    }
}

/// Explicit `+call x:Result<T>` capture, preserving the established authored idiom.
/// Uses the same declared-Result contract (including nested-wrapper requirements).
#[allow(dead_code)] // Do not replace legacy catch-all before operational-error migration.
pub(crate) fn capture_call(
    function: &str,
    payload_type: &Type,
    outcome: Result<Value>,
) -> Result<Value> {
    complete_invocation(
        function,
        &Type::Result(Box::new(payload_type.clone())),
        outcome,
    )
}

fn capture_language_failure(outcome: Result<Value>) -> Result<Value> {
    match outcome {
        Err(error) => match error.downcast_ref::<LanguageFailure>() {
            Some(failure) => Ok(Value::Err(failure.to_string())),
            None => Err(error),
        },
        value => value,
    }
}

/// No recursive flattening, implicit inner wrapping or inspection of user-union names.
/// At each declared Result layer an explicit Err terminates the shape; Ok descends once.
/// A violation is an execution/contract error, not a catchable language failure.
fn require_explicit_nested_result(function: &str, ty: &Type, value: &Value) -> Result<()> {
    let (mut ty, mut value) = (ty, value);
    while let Type::Result(inner) = ty {
        match value {
            Value::Err(_) => return Ok(()),
            Value::Ok(payload) => {
                ty = inner;
                value = payload;
            }
            _ => anyhow::bail!(
                "function `{function}` returning nested Result requires explicit outer Ok/Err \
                 at every Result layer; implicit nested wrapping is not supported"
            ),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
