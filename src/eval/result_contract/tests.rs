//! Slice A tests are contract tests, not claims that old invocation paths are repaired.
//! B owns Check unwind, caller effect suppression and JIT refusal. C owns all invocation
//! boundaries, user await, stubs and test/CLI/native adapters (matrix in ROADMAP.md).
use super::*;
use std::sync::Arc;

fn result_type(payload: Type) -> Type {
    Type::Result(Box::new(payload))
}

fn check_failure(label: &str) -> anyhow::Error {
    LanguageFailure::Check {
        label: label.into(),
    }
    .into()
}

fn assert_ok_int(value: Value, expected: i64) {
    let Value::Ok(inner) = value else {
        panic!("expected explicit Ok")
    };
    assert!(matches!(*inner, Value::Int(n) if n == expected));
}

#[test]
fn plain_result_success_normalizes_once_across_repeated_boundaries() {
    let ty = result_type(Type::Int);
    let first = complete_invocation("M.validate", &ty, Ok(Value::Int(7))).unwrap();
    let forwarded = complete_invocation("M.forward", &ty, Ok(first)).unwrap();
    assert_ok_int(forwarded, 7);
}

#[test]
fn explicit_success_and_error_are_not_wrapped_or_parsed() {
    let ty = result_type(Type::Int);
    assert_ok_int(
        complete_invocation("M.ok", &ty, Ok(Value::Ok(Box::new(Value::Int(3))))).unwrap(),
        3,
    );
    let label = "Err(opaque)\nwith context";
    let value = complete_invocation("M.err", &ty, Ok(Value::Err(label.into()))).unwrap();
    assert!(matches!(value, Value::Err(text) if text == label));
}

#[test]
fn successful_payload_identity_survives_the_wrapper() {
    let text = Arc::new("payload".to_owned());
    let value = complete_invocation(
        "M.text",
        &result_type(Type::String),
        Ok(Value::String(text.clone())),
    )
    .unwrap();
    let Value::Ok(inner) = value else {
        panic!("expected Ok")
    };
    let Value::String(actual) = *inner else {
        panic!("expected String payload")
    };
    assert!(
        Arc::ptr_eq(&text, &actual),
        "normalization must not re-render payloads"
    );
}

#[test]
fn non_result_boundary_does_not_catch_or_coerce() {
    assert!(matches!(
        complete_invocation("M.scalar", &Type::Int, Ok(Value::Int(4))).unwrap(),
        Value::Int(4)
    ));
    let error =
        complete_invocation("M.scalar", &Type::Int, Err(check_failure("invalid"))).unwrap_err();
    assert_eq!(
        error.downcast_ref::<LanguageFailure>(),
        Some(&LanguageFailure::Check {
            label: "invalid".into(),
        })
    );
    // Non-Result normalization is not a new global runtime type checker. Existing
    // malformed scalar-returning functions need a separate compatibility audit.
    assert!(matches!(
        complete_invocation("M.legacy", &Type::Int, Ok(Value::Err("legacy".into()))).unwrap(),
        Value::Err(text) if text == "legacy"
    ));
}

#[test]
fn typed_check_failure_unwinds_scalars_then_is_caught_at_declared_result() {
    let failure = complete_invocation("M.inner", &Type::Int, Err(check_failure("bad_age")));
    let failure = complete_invocation("M.middle", &Type::Int, failure);
    assert!(
        failure.is_err(),
        "scalar callers must not receive an Err value as data"
    );
    let result = complete_invocation("M.outer", &result_type(Type::Int), failure).unwrap();
    assert!(matches!(result, Value::Err(label) if label == "bad_age"));
}

#[test]
fn context_wrapped_language_failure_keeps_its_original_label() {
    let error = check_failure("bad_name").context("caller diagnostic, not the language label");
    let result = complete_invocation("M.outer", &result_type(Type::String), Err(error)).unwrap();
    assert!(matches!(result, Value::Err(label) if label == "bad_name"));
}

#[test]
fn identical_error_text_cannot_make_an_operational_failure_catchable() {
    let label = "connection refused";
    let language = complete_invocation("M.f", &result_type(Type::Int), Err(check_failure(label)));
    assert!(matches!(language.unwrap(), Value::Err(text) if text == label));
    let operational = anyhow::anyhow!(label).context("transport context");
    let result = complete_invocation("M.f", &result_type(Type::Int), Err(operational));
    let error = result.unwrap_err();
    assert!(error.downcast_ref::<LanguageFailure>().is_none());
    assert_eq!(error.to_string(), "transport context");
    assert_eq!(error.root_cause().to_string(), label);
}

#[test]
fn operational_error_type_and_context_survive_all_contract_helpers() {
    let original = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "operator failure");
    let error = anyhow::Error::new(original).context("preserve source chain");
    let outcome = propagate_call(Err(error));
    let outcome = capture_call("M.catch", &Type::Int, outcome);
    let error = complete_invocation("M.result", &result_type(Type::Int), outcome).unwrap_err();
    assert!(error.downcast_ref::<LanguageFailure>().is_none());
    assert_eq!(
        error.downcast_ref::<std::io::Error>().unwrap().kind(),
        std::io::ErrorKind::PermissionDenied
    );
    assert_eq!(error.to_string(), "preserve source chain");
}

#[test]
fn explicit_err_propagates_and_an_explicit_result_binding_can_catch_it() {
    let outcome = propagate_call(Ok(Value::Err("no_write".into())));
    let error = outcome.unwrap_err();
    assert_eq!(
        error.downcast_ref::<LanguageFailure>(),
        Some(&LanguageFailure::Propagation {
            message: "no_write".into(),
        })
    );
    let value = capture_call("M.catch", &Type::Int, Err(error)).unwrap();
    assert!(matches!(value, Value::Err(message) if message == "no_write"));
}

#[test]
fn propagation_removes_exactly_one_ok_layer() {
    let nested = Value::Ok(Box::new(Value::Ok(Box::new(Value::Int(9)))));
    assert_ok_int(propagate_call(Ok(nested)).unwrap(), 9);
    let inner_error = Value::Ok(Box::new(Value::Err("inner data".into())));
    assert!(
        matches!(propagate_call(Ok(inner_error)).unwrap(), Value::Err(text) if text == "inner data")
    );
    assert!(matches!(
        propagate_call(Ok(Value::Int(5))).unwrap(),
        Value::Int(5)
    ));
}

#[test]
fn result_binding_preserves_catchable_values_and_wraps_plain_success() {
    assert_ok_int(
        capture_call("M.call", &Type::Int, Ok(Value::Int(2))).unwrap(),
        2,
    );
    assert_ok_int(
        capture_call("M.call", &Type::Int, Ok(Value::Ok(Box::new(Value::Int(2))))).unwrap(),
        2,
    );
    let captured = capture_call("M.call", &Type::Int, Err(check_failure("bad"))).unwrap();
    assert!(matches!(captured, Value::Err(label) if label == "bad"));
    let preserved =
        capture_call("M.call", &Type::Int, Ok(Value::Err("already data".into()))).unwrap();
    assert!(matches!(preserved, Value::Err(label) if label == "already data"));
}

#[test]
fn nested_result_requires_explicit_layers_and_never_flattens_them() {
    let ty = result_type(result_type(Type::Int));
    let value = Value::Ok(Box::new(Value::Ok(Box::new(Value::Int(11)))));
    let value = complete_invocation("M.nested", &ty, Ok(value)).unwrap();
    let Value::Ok(inner) = value else {
        panic!("missing outer Ok")
    };
    assert_ok_int(*inner, 11);
    let value = Value::Ok(Box::new(Value::Err("inner".into())));
    let value = complete_invocation("M.nested", &ty, Ok(value)).unwrap();
    let Value::Ok(inner) = value else {
        panic!("inner Err was flattened")
    };
    assert!(matches!(*inner, Value::Err(label) if label == "inner"));
    let value = complete_invocation("M.nested", &ty, Ok(Value::Err("outer".into()))).unwrap();
    assert!(matches!(value, Value::Err(label) if label == "outer"));
    let value = complete_invocation("M.nested", &ty, Err(check_failure("check"))).unwrap();
    assert!(matches!(value, Value::Err(label) if label == "check"));
}

#[test]
fn missing_nested_wrapper_is_an_uncatchable_contract_error() {
    let ty = result_type(result_type(Type::Int));
    for value in [Value::Int(3), Value::Ok(Box::new(Value::Int(3)))] {
        let error = complete_invocation("M.nested", &ty, Ok(value)).unwrap_err();
        assert!(error.to_string().contains("M.nested"));
        assert!(error.to_string().contains("explicit outer Ok/Err"));
        assert!(error.downcast_ref::<LanguageFailure>().is_none());
        assert!(capture_call("M.catcher", &Type::Int, Err(error)).is_err());
    }
    assert!(capture_call(
        "M.capture_nested",
        &result_type(Type::Int),
        Ok(Value::Int(3))
    )
    .is_err());
}

#[test]
fn constructor_spelling_in_a_user_union_is_not_a_result_wrapper() {
    // Use an arbitrary interned id: the helpers must not resolve/display it or compare
    // the variant's spelling with "Ok", "Err" or "Some".
    let variant = 12345;
    let union = Value::Union {
        variant,
        payload: vec![Value::Int(8)],
    };
    let ty = result_type(Type::TaggedUnion("UserUnion".into()));
    let result = complete_invocation("M.union", &ty, Ok(union)).unwrap();
    let Value::Ok(inner) = result else {
        panic!("union was mistaken for Result")
    };
    let Value::Union {
        variant: actual,
        payload,
    } = *inner
    else {
        panic!("lost union payload")
    };
    assert_eq!(actual, variant);
    assert!(matches!(payload.as_slice(), [Value::Int(8)]));
}

#[test]
fn legacy_success_helper_keeps_existing_nested_and_non_result_behavior() {
    // This is the only helper used by production callers in A. The stricter nested
    // invocation contract must not leak into the old tree/test/+call paths yet.
    let value = wrap_success(Value::Ok(Box::new(Value::Int(3))));
    assert_ok_int(value, 3);
    let value = wrap_success(Value::Ok(Box::new(Value::Ok(Box::new(Value::Int(3))))));
    let Value::Ok(inner) = value else {
        panic!("missing outer Ok")
    };
    assert_ok_int(*inner, 3);
    assert!(matches!(wrap_success(Value::Err("e".into())), Value::Err(label) if label == "e"));
    assert!(matches!(wrap_success(Value::None), Value::Ok(inner) if matches!(*inner, Value::None)));
}

#[test]
fn triple_result_requires_the_deepest_wrapper_but_middle_err_terminates() {
    let ty = result_type(result_type(result_type(Type::Int)));
    let missing = Value::Ok(Box::new(Value::Ok(Box::new(Value::Int(7)))));
    let error = complete_invocation("M.triple", &ty, Ok(missing)).unwrap_err();
    assert!(error.downcast_ref::<LanguageFailure>().is_none());
    let middle_err = Value::Ok(Box::new(Value::Err("middle".into())));
    let value = complete_invocation("M.triple", &ty, Ok(middle_err)).unwrap();
    let Value::Ok(inner) = value else {
        panic!("outer wrapper lost")
    };
    assert!(matches!(*inner, Value::Err(message) if message == "middle"));
}

#[test]
fn extra_terminal_wrapper_is_preserved_not_payload_type_checked() {
    // This helper validates required Result layers, not terminal payload types. Pin that
    // boundary explicitly: an extra wrapper at the terminal Int is left intact for the
    // separate type-validation policy; it must not be silently flattened here.
    let ty = result_type(result_type(Type::Int));
    let extra = Value::Ok(Box::new(Value::Ok(Box::new(Value::Ok(Box::new(
        Value::Int(7),
    ))))));
    let value = complete_invocation("M.extra", &ty, Ok(extra)).unwrap();
    let Value::Ok(inner) = value else {
        panic!("outer wrapper lost")
    };
    let Value::Ok(inner) = *inner else {
        panic!("middle wrapper lost")
    };
    assert_ok_int(*inner, 7);
}
