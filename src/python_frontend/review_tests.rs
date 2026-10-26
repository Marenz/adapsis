//! Discriminating tests for the T1+T2 independent-review findings.
use super::*;
use crate::{ast, parser, validator};

fn program(source: &str) -> ast::Program {
    let lowered = lower_module(source, "Example").unwrap_or_else(|e| panic!("{e}"));
    let mut program = ast::Program::default();
    for operation in &lowered.operations {
        validator::apply_and_validate(&mut program, operation).unwrap();
    }
    program
}

fn reject(source: &str, fragment: &str) -> Diagnostic {
    let error = lower_module(source, "Example").unwrap_err();
    assert!(
        error.message.contains(fragment),
        "expected {fragment:?}, got {error}"
    );
    error
}

#[test]
fn struct_constructor_requires_every_field_exactly_once() {
    reject(
        "@dataclass\nclass Point:\n    x: int\n    y: int\n\ndef f() -> Point:\n    return Point(x=1)\n",
        "missing field",
    );
    reject(
        "@dataclass\nclass Point:\n    x: int\n\ndef f() -> Point:\n    return Point(x=1, z=2)\n",
        "unknown field",
    );
}

#[test]
fn non_exhaustive_match_is_rejected_in_lowering() {
    reject(
        "@union\nclass E:\n    A: tuple[int]\n    B: tuple[int]\n\ndef f(e: E) -> int:\n    match e:\n        case A(x):\n            return x\n",
        "non-exhaustive",
    );
    // With a wildcard it is accepted.
    let source = "@union\nclass E:\n    A: tuple[int]\n    B: tuple[int]\n\ndef f(e: E) -> int:\n    match e:\n        case A(x):\n            return x\n        case _:\n            return 0\n";
    assert!(lower_module(source, "Example").is_ok());
}

#[test]
fn printer_refuses_non_exhaustive_adapsis_match() {
    let source = "+module Example\n+type E = A(Int) | B(Int)\n+fn f (e:E)->Int\n  +match e\n  +case A(x)\n    +return x\n  +end\n+end\n";
    let mut program = ast::Program::default();
    for op in parser::parse(source).unwrap() {
        validator::apply_and_validate(&mut program, &op).unwrap();
    }
    let error = printer::print_function(&program, "Example.f").unwrap_err();
    assert!(error.message.contains("non-exhaustive"));
}

#[test]
fn case_allowlist_rejects_arbitrary_builtins() {
    reject(
        "@union\nclass E:\n    A: tuple[int]\n    B: tuple[int]\n\ndef f(e: E) -> int:\n    match e:\n        case len(x):\n            return x\n        case _:\n            return 0\n",
        "unknown case variant",
    );
}

#[test]
fn io_match_subject_requires_async_and_io() {
    reject(
        "def f(url: str) -> str:\n    match http_get(url):\n        case Ok(b):\n            return b\n        case Err(e):\n            return e\n",
        "@effects('io')",
    );
}

#[test]
fn nested_capture_has_resolved_int_type() {
    let source = "@union\nclass Node:\n    Leaf: tuple[int]\n    Wrap: tuple[\"Node\"]\n\ndef unwrap(n: Node) -> int:\n    match n:\n        case Wrap(Leaf(v)):\n            return v + 1\n        case _:\n            return 0\n";
    let original = program(source);
    assert_eq!(
        crate::eval::eval_call_with_input(
            &original,
            "Example.unwrap",
            &parser::parse_test_input(1, "Wrap(Leaf(41))").unwrap()
        )
        .unwrap(),
        "42"
    );
}

#[test]
fn nested_zero_payload_variant_is_rejected() {
    reject(
        "@union\nclass Node:\n    Leaf: tuple[int]\n    Wrap: tuple[\"Node\"]\n\ndef f(n: Node) -> int:\n    match n:\n        case Wrap(Leaf()):\n            return 0\n        case _:\n            return 0\n",
        "nested zero-payload",
    );
}

#[test]
fn struct_literal_in_constructor_payload_roundtrips() {
    let source = "@dataclass\nclass Point:\n    x: int\n    y: int\n\ndef origin() -> Result[Point]:\n    return Ok(Point(x=0, y=0))\n";
    let original = program(source);
    let printed = printer::print_function(&original, "Example.origin").unwrap();
    assert!(printed.contains("Ok(Point(x=0, y=0))"));
    let type_text = printer::print_type(&original, "Example.Point").unwrap();
    let rebuilt = program(&format!("{type_text}\n{printed}"));
    assert_eq!(printer::print_function(&rebuilt, "Example.origin").unwrap(), printed);
}

#[test]
fn struct_function_name_collision_is_a_located_error() {
    reject(
        "@dataclass\nclass Point:\n    x: int\n    y: int\n\ndef Point() -> int:\n    return 1\n",
        "duplicate function Point",
    );
}

#[test]
fn variant_payload_lookup_is_current_module_first_and_deterministic() {
    let source = "+module Z\n+type Zeta = Add(Int) | B(Int)\n+end\n+module A\n+type Alpha = Add(Int) | C(Int)\n+end\n";
    let mut program = ast::Program::default();
    for op in parser::parse(source).unwrap() {
        validator::apply_and_validate(&mut program, &op).unwrap();
    }
    let catalog = super::types::catalog_from_program(&program);
    let fallback = super::types::variant_payloads(&catalog, "Add", "Other").unwrap_err();
    assert_eq!(fallback, super::types::variant_payloads(&catalog, "Add", "Other").unwrap_err());
    assert!(fallback.contains("ambiguous"));
    let local = super::types::variant_payloads(&catalog, "Add", "Z").unwrap();
    assert_eq!(local.map(|(name, _)| name).as_deref(), Some("Z.Zeta"));

    let conflict = "+module P\n+type One = Add(Int) | B(Int)\n+end\n+module Q\n+type Two = Add(Int, Int) | C(Int)\n+end\n";
    let mut p2 = ast::Program::default();
    for op in parser::parse(conflict).unwrap() {
        validator::apply_and_validate(&mut p2, &op).unwrap();
    }
    let cat2 = super::types::catalog_from_program(&p2);
    assert!(super::types::variant_payloads(&cat2, "Add", "Other").is_err());
}
