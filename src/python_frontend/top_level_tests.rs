//! A1: module-less type declarations with REAL owner "" semantics.
use super::{lower_toplevel, print_toplevel_type, types};
use crate::{ast, parser, validator};

fn program(ax: &str) -> ast::Program {
    let mut program = ast::Program::default();
    for op in parser::parse(ax).unwrap() {
        validator::apply_and_validate(&mut program, &op).unwrap();
    }
    program.rebuild_function_index();
    program
}

fn apply(program: &mut ast::Program, source: &str) -> Vec<parser::Operation> {
    let lowered = lower_toplevel(source, program).unwrap_or_else(|e| panic!("{source}\n{e}"));
    for op in &lowered.operations {
        validator::apply_and_validate(program, op).unwrap();
    }
    program.rebuild_function_index();
    lowered.operations
}

fn refused(source: &str, ax: &str) -> String {
    lower_toplevel(source, &program(ax))
        .unwrap_err()
        .to_string()
}

const POINT: &str = "@dataclass\nclass Point:\n    x: int\n    y: int\n";

#[test]
fn top_level_lowering_emits_unwrapped_type_operations() {
    let source =
        format!("{POINT}\n@union\nclass Shape:\n    Circle: tuple[int]\n    Dot: tuple[()]\n");
    let lowered = lower_toplevel(&source, &ast::Program::default()).unwrap();
    assert_eq!(lowered.operations.len(), 2);
    assert!(lowered
        .operations
        .iter()
        .all(|op| matches!(op, parser::Operation::Type(_))));
    assert!(lowered.legacy_export().unwrap().contains("+type Point"));
    assert!(!lowered.legacy_export().unwrap().contains("+module"));
    let mut fresh = ast::Program::default();
    apply(&mut fresh, &source);
    assert_eq!(fresh.types.len(), 2);
    assert!(fresh.modules.is_empty() && fresh.functions.is_empty());
}

#[test]
fn owner_is_the_empty_string_and_never_a_module() {
    // A top-level and a module type may share a bare name: different owners.
    let program = program(
        "+type Point = x:Int, y:Int\n+module Geo\n+type Point = a:Int\n+fn f ()->Int\n  +return 1\n+end\n",
    );
    let catalog = types::catalog_from_program(&program);
    assert!(catalog.contains_key("Point") && catalog.contains_key("Geo.Point"));
    assert_eq!(
        types::resolve_name(&catalog, "", "Point")
            .unwrap()
            .as_deref(),
        Some("Point")
    );
    assert_eq!(
        types::resolve_name(&catalog, "Geo", "Point")
            .unwrap()
            .as_deref(),
        Some("Geo.Point")
    );
    assert_eq!(types::owned_name("", "Point"), "Point");
    // The printer takes the TOP-LEVEL one only, and it roundtrips exactly.
    let text = print_toplevel_type(&program, "Point").unwrap();
    assert_eq!(text, "@dataclass\nclass Point:\n    x: int\n    y: int\n");
    assert!(print_toplevel_type(&program, "Geo.Point").is_err());
    // A top-level struct referencing `Point` resolves it to the top-level owner.
    let mut with_ref = program.clone();
    let ops = apply(
        &mut with_ref,
        "@dataclass\nclass Line:\n    start: Point\n    end: Point\n",
    );
    let parser::Operation::Type(line) = &ops[0] else {
        panic!()
    };
    let parser::TypeBody::Struct(fields) = &line.body else {
        panic!()
    };
    assert!(fields
        .iter()
        .all(|f| matches!(&f.ty, parser::TypeExpr::Named(n) if n == "Point")));
    let text = print_toplevel_type(&with_ref, "Line").unwrap();
    assert!(text.contains("start: Point"), "{text}");
}

#[test]
fn every_type_shape_roundtrips_and_reprints_identically() {
    for source in [
        POINT,
        "@dataclass\nclass Mixed:\n    items: list[int]\n    names: Map[str, int]\n    maybe: Option[int]\n    outcome: Result[str]\n    flag: bool\n    ratio: float\n",
        "@union\nclass IntList:\n    Cons: tuple[int, IntList]\n    Nil: tuple[()]\n",
        "@union\nclass Option:\n    Some: tuple[int]\n    None_: tuple[()]\n",
        "@union\nclass Expr:\n    Literal: tuple[int]\n    Add: tuple[Expr, Expr]\n    Neg: tuple[Expr]\n",
        // Declared variants named like constructor builtins are legal (existing member rule).
        "@union\nclass Echo:\n    Ok: tuple[int]\n    Err: tuple[str]\n",
    ] {
        let mut program = ast::Program::default();
        apply(&mut program, source);
        let name = program.types[0].name().to_owned();
        let text = print_toplevel_type(&program, &name).unwrap_or_else(|e| panic!("{source}\n{e}"));
        // Second print from the re-lowered declaration is byte-identical.
        let mut again = ast::Program::default();
        apply(&mut again, &text);
        assert_eq!(print_toplevel_type(&again, &name).unwrap(), text, "{source}");
    }
}

#[test]
fn collision_rules_refuse_ambiguous_type_declarations() {
    for (source, ax, needle) in [
        // Builtin, syntax, keyword and unsafe type names.
        ("@dataclass\nclass len:\n    x: int\n", "", "reserved/unsafe type name"),
        ("@dataclass\nclass concat:\n    x: int\n", "", "reserved/unsafe type name"),
        ("@dataclass\nclass Ok:\n    x: int\n", "", "reserved/unsafe type name"),
        ("@dataclass\nclass Some:\n    x: int\n", "", "reserved/unsafe type name"),
        ("@dataclass\nclass Result:\n    x: int\n", "", "reserved/unsafe type name"),
        ("@dataclass\nclass Map:\n    x: int\n", "", "reserved/unsafe type name"),
        ("@dataclass\nclass eval:\n    x: int\n", "", "reserved/unsafe type name"),
        ("@dataclass\nclass __dunder__:\n    x: int\n", "", "reserved/unsafe type name"),
        // Duplicate declarations in one submission, of either kind.
        ("@dataclass\nclass T:\n    x: int\n\n@dataclass\nclass T:\n    y: int\n", "", "duplicate type declaration T"),
        ("@dataclass\nclass T:\n    x: int\n\n@union\nclass T:\n    A: tuple[()]\n    B: tuple[()]\n", "", "duplicate type declaration T"),
        // A type may not share a spelling with a module.
        ("@dataclass\nclass Geo:\n    x: int\n", "+module Geo\n+fn f ()->Int\n  +return 1\n+end\n", "collides with module Geo"),
        // Type name vs union variant, existing (top-level and module) and submitted.
        ("@dataclass\nclass Circle:\n    r: int\n", "+type Shape = Circle(Int) | Dot\n", "collides with a union variant named Circle"),
        ("@dataclass\nclass Circle:\n    r: int\n", "+module G\n+type Shape = Circle(Int) | Dot\n+fn f ()->Int\n  +return 1\n+end\n", "collides with a union variant named Circle"),
        ("@union\nclass Shape:\n    Circle: tuple[int]\n    Dot: tuple[()]\n\n@dataclass\nclass Circle:\n    r: int\n", "", "Circle collides with a"),
        ("@union\nclass Shape:\n    Shape: tuple[int]\n    Dot: tuple[()]\n", "", "collides with a union variant named Shape"),
        // Symmetric: a variant may not reuse an existing or submitted type name.
        ("@union\nclass Shape:\n    Point: tuple[int]\n    Dot: tuple[()]\n", "+type Point = x:Int\n", "union variant Point collides with a type named Point"),
        ("@union\nclass Shape:\n    Point: tuple[int]\n    Dot: tuple[()]\n", "+module G\n+type Point = x:Int\n+fn f ()->Int\n  +return 1\n+end\n", "union variant Point collides with a type named Point"),
        ("@union\nclass Shape:\n    Point: tuple[int]\n    Dot: tuple[()]\n\n@dataclass\nclass Point:\n    x: int\n", "", "collides with a"),
        // Not type declarations at all.
        ("shared_counter: int = 0\n", "", "shared variables belong to a module"),
        // (Top-level functions are A2: see top_level_function_tests.rs.)
        ("\"module doc\"\n", "", "may only contain @dataclass/@union type declarations"),
        ("import os\n", "", "may only contain @dataclass/@union type declarations"),
        ("x = 1\n", "", "shared variables belong to a module"),
        ("", "", "at least one type or function declaration"),
    ] {
        let message = refused(source, ax);
        assert!(message.contains(needle), "{source:?}: {message}");
        assert!(message.starts_with("<top-level>:"), "{message}");
    }
}

#[test]
fn legitimate_overlaps_stay_accepted_and_exact() {
    // Two unions may both declare `Some`: a valid Adapsis program whose ambiguity is
    // reported where it is USED (function lowering), not at declaration.
    let mut both = ast::Program::default();
    apply(
        &mut both,
        "@union\nclass Maybe:\n    Some: tuple[int]\n    Nope: tuple[()]\n\n@union\nclass Option:\n    Some: tuple[int]\n    None_: tuple[()]\n",
    );
    for name in ["Maybe", "Option"] {
        print_toplevel_type(&both, name).unwrap();
    }
    // A module type with the same bare name as a top-level type is not a collision.
    let program = program("+module Geo\n+type Point = a:Int\n+fn f ()->Int\n  +return 1\n+end\n");
    lower_toplevel(POINT, &program).unwrap();
    // Replacement is the validator's job: re-applying an existing top-level type is
    // refused there ("duplicate type declaration"), never silently replaced.
    let mut twice = ast::Program::default();
    let ops = apply(&mut twice, POINT);
    let err = validator::apply_and_validate(&mut twice, &ops[0]).unwrap_err();
    assert!(
        err.to_string().contains("duplicate type declaration"),
        "{err}"
    );
}

#[test]
fn unprintable_members_are_reported_not_approximated() {
    let program = program("+type Env = vars:Int, other:Int\n");
    let message = print_toplevel_type(&program, "Env")
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("unsupported member name \"vars\""),
        "{message}"
    );
}
