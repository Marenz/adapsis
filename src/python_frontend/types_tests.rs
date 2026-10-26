//! Roundtrip and negative tests for Task 1: struct/union type declarations,
//! struct literals, field access, and Map/Set/Option/Byte annotations.
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
fn struct_declaration_construction_and_field_access_roundtrip() {
    let source = "@dataclass\nclass Point:\n    x: int\n    y: int\n\ndef add(a: Point, b: Point) -> Point:\n    return Point(x=a.x + b.x, y=a.y + b.y)\n\ndef getx(p: Point) -> int:\n    return p.x\n";
    let original = program(source);
    let printed = printer::print_function(&original, "Example.add").unwrap();
    assert!(printed.contains("Point(x="));
    assert!(printed.contains("a.x"));
    let type_text = printer::print_type(&original, "Example.Point").unwrap();
    let rebuilt = program(&format!("{type_text}\n{printed}"));
    assert_eq!(printer::print_function(&rebuilt, "Example.add").unwrap(), printed);
    assert_eq!(
        crate::eval::eval_call_with_input(
            &original,
            "Example.getx",
            &parser::parse_test_input(1, "p={x: 42, y: 0}").unwrap()
        )
        .unwrap(),
        "42"
    );
}

#[test]
fn struct_type_prints_and_relowers_identically() {
    let source = "@dataclass\nclass User:\n    name: str\n    email: str\n    age: int\n";
    let original = program(source);
    let printed = printer::print_type(&original, "Example.User").unwrap();
    assert!(printed.contains("@dataclass"));
    assert!(printed.contains("name: str"));
    let rebuilt = program(&printed);
    assert_eq!(printer::print_type(&rebuilt, "Example.User").unwrap(), printed);
}

#[test]
fn recursive_union_declaration_prints_and_relowers() {
    let source = "@union\nclass Expr:\n    Literal: tuple[int]\n    Var: tuple[str]\n    Add: tuple[\"Expr\", \"Expr\"]\n    Nil: tuple[()]\n";
    let original = program(source);
    let printed = printer::print_type(&original, "Example.Expr").unwrap();
    assert!(printed.contains("@union"));
    assert!(printed.contains("Nil: tuple[()]"));
    let rebuilt = program(&printed);
    assert_eq!(printer::print_type(&rebuilt, "Example.Expr").unwrap(), printed);
    let canonical = lower_module(source, "Example").unwrap();
    assert!(canonical.canonical_adapsis.contains("+type Expr = Literal(Int) | Var(String) | Add(Expr, Expr) | Nil"));
}

#[test]
fn union_typed_parameter_and_return_roundtrip() {
    let source = "@union\nclass Expr:\n    Literal: tuple[int]\n    Add: tuple[\"Expr\", \"Expr\"]\n\ndef depth(e: Expr) -> Expr:\n    return e\n";
    let original = program(source);
    let printed = printer::print_function(&original, "Example.depth").unwrap();
    assert!(printed.contains("e: Expr"));
    assert!(printed.contains("-> Expr"));
    let type_text = printer::print_type(&original, "Example.Expr").unwrap();
    let rebuilt = program(&format!("{type_text}\n{printed}"));
    assert_eq!(printer::print_function(&rebuilt, "Example.depth").unwrap(), printed);
}

#[test]
fn map_set_option_byte_annotations_lower_to_adapsis() {
    let source = "@effects('mut')\ndef f(m: Map[str, int], s: Set[int], o: Option[int], b: Byte) -> Byte:\n    return b\n";
    let lowered = lower_module(source, "Example").unwrap();
    let canonical = lowered.canonical_adapsis;
    assert!(canonical.contains("m:Map<String, Int>"));
    assert!(canonical.contains("s:Set<Int>"));
    assert!(canonical.contains("o:Option<Int>"));
    assert!(canonical.contains("b:Byte"));
    assert!(canonical.contains("-> Byte"));
    let original = program(source);
    let printed = printer::print_function(&original, "Example.f").unwrap();
    assert!(printed.contains("Map[str, int]"));
    assert!(printed.contains("Set[int]"));
    assert!(printed.contains("Option[int]"));
    assert!(printed.contains("Byte"));
}

#[test]
fn struct_with_collection_fields_roundtrips() {
    let source = "@dataclass\nclass Env:\n    names: list[str]\n    counts: Map[str, int]\n    flags: Set[str]\n";
    let original = program(source);
    let printed = printer::print_type(&original, "Example.Env").unwrap();
    assert!(printed.contains("list[str]"));
    assert!(printed.contains("Map[str, int]"));
    let rebuilt = program(&printed);
    assert_eq!(printer::print_type(&rebuilt, "Example.Env").unwrap(), printed);
}

#[test]
fn nested_field_access_is_supported() {
    let source = "@dataclass\nclass Inner:\n    v: int\n@dataclass\nclass Outer:\n    inner: Inner\n\ndef read_inner(o: Outer) -> int:\n    return o.inner.v\n";
    let original = program(source);
    let printed = printer::print_function(&original, "Example.read_inner").unwrap();
    assert!(printed.contains("o.inner.v"));
    let type_text = format!(
        "{}\n{}",
        printer::print_type(&original, "Example.Inner").unwrap(),
        printer::print_type(&original, "Example.Outer").unwrap(),
    );
    let rebuilt = program(&format!("{type_text}\n{printed}"));
    assert_eq!(printer::print_function(&rebuilt, "Example.read_inner").unwrap(), printed);
}

#[test]
fn struct_construction_requires_keyword_fields() {
    reject(
        "@dataclass\nclass Point:\n    x: int\n\ndef f() -> Point:\n    return Point(1)\n",
        "keyword fields",
    );
    reject(
        "@dataclass\nclass Point:\n    x: int\n\ndef f() -> Point:\n    return Point(y=1)\n",
        "unknown field",
    );
    reject(
        "@dataclass\nclass Point:\n    x: int\n\ndef f() -> Point:\n    return Point(x=\"no\")\n",
        "expects Int",
    );
}

#[test]
fn struct_declarations_reject_defaults_methods_and_bad_decorators() {
    reject(
        "@dataclass\nclass Point:\n    x: int = 1\n",
        "default values",
    );
    reject(
        "@dataclass\nclass Point:\n    x: int\n    def method(self):\n        pass\n",
        "struct bodies may only contain",
    );
    reject(
        "@frozen\nclass Point:\n    x: int\n",
        "type decorator",
    );
    reject(
        "class Point:\n    x: int\n",
        "one decorator",
    );
    reject(
        "@dataclass\nclass list:\n    x: int\n",
        "reserved/unsafe type name",
    );
}

#[test]
fn union_declarations_reject_non_tuple_payloads_and_empty_union() {
    reject(
        "@union\nclass E:\n    A: int\n",
        "tuple",
    );
    reject("@union\nclass E:\n    pass\n", "union bodies may only contain");
    reject(
        "@union\nclass E:\n    A: tuple[int] = 1\n",
        "cannot have a value",
    );
    reject(
        "@union\nclass E:\n    A: tuple[int]\n",
        "at least two variants",
    );
    reject("@dataclass\nclass E:\n    pass\n", "struct bodies may only contain");
}

#[test]
fn field_access_requires_a_struct_typed_base() {
    reject(
        "def f(x: int) -> int:\n    return x.y\n",
        "field access requires",
    );
    reject(
        "def f() -> int:\n    return a.b.c\n",
        "field access requires",
    );
}

#[test]
fn type_diagnostics_carry_source_locations() {
    let error = reject(
        "@union\nclass E:\n    A: tuple[int]\n    B: int\n",
        "tuple",
    );
    assert_eq!((error.line, error.column), (4, 8));
    let error = reject(
        "@dataclass\nclass Point:\n    x: int\n\ndef f() -> Point:\n    return Point(z=1)\n",
        "unknown field",
    );
    assert_eq!((error.line, error.column), (6, 18));
}
