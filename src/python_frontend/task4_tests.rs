//! Roundtrip and eval tests for Task 4: +call/+check/+spawn helpers, div/rem,
//! all effects, and unqualified-call resolution.
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
fn call_propagation_roundtrips_as_plus_call() {
    let source = "@effects('fail')\ndef f(name: str) -> str:\n    result: str = propagate(concat('hi ', name))\n    return result\n";
    let lowered = lower_module(source, "Example").unwrap();
    assert!(lowered.canonical_adapsis.contains("+call result:String = concat(\"hi \", name)"));
    let original = program(source);
    let printed = printer::print_function(&original, "Example.f").unwrap();
    assert!(printed.contains("propagate(concat("));
    let rebuilt = program(&printed);
    assert_eq!(printer::print_function(&rebuilt, "Example.f").unwrap(), printed);
}

#[test]
fn check_statement_roundtrips() {
    let source = "def f(x: int) -> int:\n    check('valid', x > 0, 'err_negative')\n    return x\n";
    let lowered = lower_module(source, "Example").unwrap();
    assert!(lowered.canonical_adapsis.contains("+check valid (x > 0) ~err_negative"));
    let original = program(source);
    let printed = printer::print_function(&original, "Example.f").unwrap();
    assert!(printed.contains("check(\"valid\", (x > 0), \"err_negative\")"));
    let rebuilt = program(&printed);
    assert_eq!(printer::print_function(&rebuilt, "Example.f").unwrap(), printed);
}

#[test]
fn spawn_bare_and_bound_roundtrip() {
    let source = "def task() -> str:\n    return 'done'\n\n@effects('io')\nasync def f() -> str:\n    spawn(task())\n    t: int = spawn(task())\n    return 'ok'\n";
    let lowered = lower_module(source, "Example").unwrap();
    assert!(lowered.canonical_adapsis.contains("+spawn Example.task()"));
    assert!(lowered.canonical_adapsis.contains("+spawn t:Int = Example.task()"));
    let original = program(source);
    let printed = printer::print_function(&original, "Example.f").unwrap();
    assert!(printed.contains("spawn(Example.task())"));
    assert!(printed.contains("t: int = spawn(Example.task())"));
    let task = printer::print_function(&original, "Example.task").unwrap();
    let rebuilt = program(&format!("{task}\n{printed}"));
    assert_eq!(printer::print_function(&rebuilt, "Example.f").unwrap(), printed);
}

#[test]
fn div_rem_helpers_lower_to_operators_and_eval() {
    let source = "def half(x: int) -> int:\n    return div(x, 2)\n\ndef parity(x: int) -> int:\n    return rem(x, 2)\n";
    let lowered = lower_module(source, "Example").unwrap();
    assert!(lowered.canonical_adapsis.contains("+return (x / 2)"));
    assert!(lowered.canonical_adapsis.contains("+return (x % 2)"));
    let original = program(source);
    assert_eq!(
        crate::eval::eval_call_with_input(&original, "Example.half", &parser::parse_test_input(1, "7").unwrap()).unwrap(),
        "3"
    );
    assert_eq!(
        crate::eval::eval_call_with_input(&original, "Example.parity", &parser::parse_test_input(1, "7").unwrap()).unwrap(),
        "1"
    );
    let printed = printer::print_function(&original, "Example.half").unwrap();
    assert!(printed.contains("div(x, 2)"));
    let rebuilt = program(&printed);
    assert_eq!(printer::print_function(&rebuilt, "Example.half").unwrap(), printed);
}

#[test]
fn all_effects_roundtrip_through_effects_decorator() {
    let source = "@effects('rand', 'parallel', 'unsafe', 'yield')\ndef f() -> int:\n    return 1\n";
    let lowered = lower_module(source, "Example").unwrap();
    assert!(lowered.canonical_adapsis.contains("[rand, parallel, unsafe, yield]"));
    let original = program(source);
    let printed = printer::print_function(&original, "Example.f").unwrap();
    assert!(printed.contains("@effects(\"rand\", \"parallel\", \"unsafe\", \"yield\")"));
    let rebuilt = program(&printed);
    assert_eq!(printer::print_function(&rebuilt, "Example.f").unwrap(), printed);
}

#[test]
fn unqualified_adapsis_calls_are_refused_without_changing_the_original() {
    let source = "+module Example\n+fn inc (x:Int)->Int\n  +return x + 1\n+end\n+fn twice (x:Int)->Int\n  +return inc(inc(x))\n+end\n";
    let mut prog = ast::Program::default();
    for op in parser::parse(source).unwrap() {
        validator::apply_and_validate(&mut prog, &op).unwrap();
    }
    let original = prog.clone();
    let error = printer::print_function(&prog, "Example.twice").unwrap_err();
    assert!(error.message.contains("exact Python roundtrip"), "{error}");
    assert_eq!(prog, original);
}

#[test]
fn reserved_helper_names_cannot_be_shadowed() {
    reject("def check() -> int:\n    return 1\n", "reserved/unsafe function name check");
    reject("def spawn() -> int:\n    return 1\n", "reserved/unsafe function name spawn");
    reject("def propagate() -> int:\n    return 1\n", "reserved/unsafe function name propagate");
    reject("def f(div: int) -> int:\n    return div\n", "shadows/reuses");
}
