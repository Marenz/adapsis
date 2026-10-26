//! Roundtrip, idempotence and negative tests for Task 2: union variant
//! construction, `match`/`case` lowering, and printer reversal.
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

fn evaluate(program: &ast::Program, target: &str, input: &str) -> String {
    let input = parser::parse_test_input(1, input).unwrap();
    crate::eval::eval_call_with_input(program, target, &input).unwrap()
}

#[test]
fn recursive_union_match_and_variant_construction_roundtrip_and_eval() {
    let source = "@union\nclass Expr:\n    Literal: tuple[int]\n    Add: tuple[\"Expr\", \"Expr\"]\n\ndef eval_expr(e: Expr) -> int:\n    match e:\n        case Literal(v):\n            return v\n        case Add(left, right):\n            return eval_expr(left) + eval_expr(right)\n        case _:\n            return 0\n";
    let original = program(source);
    assert_eq!(
        evaluate(&original, "Example.eval_expr", "Add(Literal(1), Add(Literal(2), Literal(3)))"),
        "6"
    );
    let printed = printer::print_function(&original, "Example.eval_expr").unwrap();
    assert!(printed.contains("match e:"));
    assert!(printed.contains("case Add(left, right):"));
    let type_text = printer::print_type(&original, "Example.Expr").unwrap();
    let rebuilt = program(&format!("{type_text}\n{printed}"));
    assert_eq!(printer::print_function(&rebuilt, "Example.eval_expr").unwrap(), printed);
}

#[test]
fn result_match_on_io_builtin_without_await() {
    let source = "@effects('io')\nasync def fetch(url: str) -> str:\n    match http_get(url):\n        case Ok(body):\n            return body\n        case Err(e):\n            return concat('ERR ', e)\n";
    let original = program(source);
    let lowered = lower_module(source, "Example").unwrap();
    assert!(lowered.canonical_adapsis.contains("+match http_get(url)"));
    assert!(lowered.canonical_adapsis.contains("+case Ok(body)"));
    assert!(lowered.canonical_adapsis.contains("+case Err(e)"));
    let printed = printer::print_function(&original, "Example.fetch").unwrap();
    assert!(printed.contains("case Ok(body):"));
    assert!(printed.contains("case Err(e):"));
    let rebuilt = program(&printed);
    assert_eq!(printer::print_function(&rebuilt, "Example.fetch").unwrap(), printed);
}

#[test]
fn nested_patterns_and_wildcard_roundtrip() {
    let source = "@union\nclass Node:\n    Value: tuple[int]\n    Wrapped: tuple[\"Node\"]\n\ndef depth(n: Node) -> int:\n    match n:\n        case Value(_):\n            return 0\n        case Wrapped(Value(_)):\n            return 1\n        case _:\n            return -1\n";
    let original = program(source);
    let lowered = lower_module(source, "Example").unwrap();
    assert!(lowered.canonical_adapsis.contains("+case Wrapped(Value(_))"));
    let printed = printer::print_function(&original, "Example.depth").unwrap();
    assert!(printed.contains("case Wrapped(Value(_)):"));
    let type_text = printer::print_type(&original, "Example.Node").unwrap();
    let rebuilt = program(&format!("{type_text}\n{printed}"));
    assert_eq!(printer::print_function(&rebuilt, "Example.depth").unwrap(), printed);
}

#[test]
fn nested_capture_is_bound_for_the_arm_body() {
    let source = "@union\nclass Node:\n    Leaf: tuple[int]\n    Wrap: tuple[\"Node\"]\n\ndef unwrap(n: Node) -> int:\n    match n:\n        case Wrap(Wrap(Leaf(v))):\n            return v\n        case _:\n            return 0\n";
    let original = program(source);
    assert_eq!(evaluate(&original, "Example.unwrap", "Wrap(Wrap(Leaf(42)))"), "42");
    let printed = printer::print_function(&original, "Example.unwrap").unwrap();
    assert!(printed.contains("case Wrap(Wrap(Leaf(v))):"));
    let type_text = printer::print_type(&original, "Example.Node").unwrap();
    let rebuilt = program(&format!("{type_text}\n{printed}"));
    assert_eq!(printer::print_function(&rebuilt, "Example.unwrap").unwrap(), printed);
}

#[test]
fn literal_and_zero_payload_patterns_roundtrip() {
    let source = "@union\nclass Kind:\n    Tag: tuple[str]\n    Level: tuple[int]\n    Nil: tuple[()]\n\ndef kind_of(k: Kind) -> int:\n    match k:\n        case Tag(\"x\"):\n            return 1\n        case Level(0):\n            return 2\n        case Nil():\n            return 3\n        case _:\n            return 0\n";
    let original = program(source);
    let lowered = lower_module(source, "Example").unwrap();
    assert!(lowered.canonical_adapsis.contains("+case Tag(\"x\")"));
    assert!(lowered.canonical_adapsis.contains("+case Level(0)"));
    assert!(lowered.canonical_adapsis.contains("+case Nil"));
    let printed = printer::print_function(&original, "Example.kind_of").unwrap();
    assert!(printed.contains("case Nil():"));
    let type_text = printer::print_type(&original, "Example.Kind").unwrap();
    let rebuilt = program(&format!("{type_text}\n{printed}"));
    assert_eq!(printer::print_function(&rebuilt, "Example.kind_of").unwrap(), printed);
}

#[test]
fn none_variant_value_and_case_roundtrip() {
    let source = "@union\nclass Maybe:\n    Some: tuple[int]\n    Nope: tuple[()]\n\ndef is_some(o: Maybe) -> bool:\n    match o:\n        case Some(_):\n            return True\n        case Nope():\n            return False\n";
    let original = program(source);
    let lowered = lower_module(source, "Example").unwrap();
    assert!(lowered.canonical_adapsis.contains("+type Maybe = Some(Int) | Nope"));
    assert!(lowered.canonical_adapsis.contains("+case Nope"));
    assert_eq!(evaluate(&original, "Example.is_some", "Nope"), "false");
    assert_eq!(evaluate(&original, "Example.is_some", "Some(1)"), "true");
    let printed = printer::print_function(&original, "Example.is_some").unwrap();
    assert!(printed.contains("case Nope():"));
    let type_text = printer::print_type(&original, "Example.Maybe").unwrap();
    let rebuilt = program(&format!("{type_text}\n{printed}"));
    assert_eq!(printer::print_function(&rebuilt, "Example.is_some").unwrap(), printed);
}

#[test]
fn none_keyword_value_and_case_roundtrip_structurally() {
    let source = "def classify(o: Option[int]) -> str:\n    match o:\n        case Some(v):\n            return 'some'\n        case None:\n            return 'none'\n";
    let lowered = lower_module(source, "Example").unwrap();
    assert!(lowered.canonical_adapsis.contains("+case None"));
    let value = lower_module("def nothing() -> str:\n    return None\n", "Example").unwrap();
    assert!(value.canonical_adapsis.contains("+return None"));
    let original = program(source);
    let printed = printer::print_function(&original, "Example.classify").unwrap();
    assert!(printed.contains("case None:"));
    let rebuilt = program(&printed);
    assert_eq!(printer::print_function(&rebuilt, "Example.classify").unwrap(), printed);
}

#[test]
fn struct_literal_as_call_argument_is_typed_from_callee() {
    let source = "@dataclass\nclass Point:\n    x: int\n    y: int\n\ndef negate(p: Point) -> Point:\n    return Point(x=-p.x, y=-p.y)\n\ndef wrap() -> Point:\n    return Example.negate(Point(x=1, y=2))\n";
    let original = program(source);
    let printed = printer::print_function(&original, "Example.wrap").unwrap();
    assert!(printed.contains("negate(Point(x=1, y=2))"));
    let negate = printer::print_function(&original, "Example.negate").unwrap();
    let type_text = printer::print_type(&original, "Example.Point").unwrap();
    let rebuilt = program(&format!("{type_text}\n{negate}\n{printed}"));
    assert_eq!(printer::print_function(&rebuilt, "Example.wrap").unwrap(), printed);
}

#[test]
fn match_rejects_guards_alternatives_sequences_and_captures() {
    let union = "@union\nclass E:\n    A: tuple[int]\n    B: tuple[int]\n";
    reject(
        &format!("{union}def f(e: E) -> int:\n    match e:\n        case A(x) if x > 0:\n            return x\n        case _:\n            return 0\n"),
        "guards",
    );
    reject(
        &format!("{union}def f(e: E) -> int:\n    match e:\n        case A(x) | B(x):\n            return x\n        case _:\n            return 0\n"),
        "'|'",
    );
    reject(
        &format!("{union}def f(e: E) -> int:\n    match e:\n        case A(x) as whole:\n            return x\n        case _:\n            return 0\n"),
        "as-patterns",
    );
    reject(
        &format!("{union}def f(e: E) -> int:\n    match e:\n        case [x]:\n            return x\n        case _:\n            return 0\n"),
        "sequence",
    );
    reject(
        &format!("{union}def f(e: E) -> int:\n    match e:\n        case x:\n            return 0\n"),
        "capture",
    );
    reject(
        &format!("{union}def f(e: E) -> int:\n    match e:\n        case 0:\n            return 0\n"),
        "top-level literal",
    );
}

#[test]
fn match_rejects_unknown_variants_and_arity_mismatches() {
    let union = "@union\nclass E:\n    A: tuple[int]\n    B: tuple[int]\n";
    reject(
        &format!("{union}def f(e: E) -> int:\n    match e:\n        case Unknown(x):\n            return x\n        case _:\n            return 0\n"),
        "unknown case variant",
    );
    reject(
        &format!("{union}def f(e: E) -> int:\n    return A(1, 2)\n"),
        "expects 1 payload",
    );
    reject(
        &format!("{union}def f(e: E) -> E:\n    return A\n"),
        "requires 1 payload",
    );
}
