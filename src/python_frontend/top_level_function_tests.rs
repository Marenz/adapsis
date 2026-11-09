//! A2: module-less (top-level) functions. TEST-ONLY surface with real owner "" semantics:
//! a bare call resolves to a top-level function exactly as `Program::get_function` does,
//! module functions stay reachable only qualified, and module-scoped lowering/printing is
//! unchanged.
use super::{lower_module, lower_toplevel, print_function, print_toplevel_function, types};
use crate::{ast, eval, parser, validator, vm};
use eval::{Env, Value};

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

fn tree(program: &ast::Program, target: &str, args: Vec<Value>) -> Value {
    let function = program.get_function(target).unwrap();
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    for (param, arg) in function.params.iter().zip(args) {
        env.set(&param.name, arg);
    }
    eval::eval_declared_function(program, target, function, &mut env).unwrap()
}

fn bytecode(program: &ast::Program, target: &str, args: Vec<Value>) -> Value {
    let compiled = vm::compile_function(program.get_function(target).unwrap(), program).unwrap();
    vm::execute_with_io(&compiled, args, program, &|_, _| {
        anyhow::bail!("unexpected IO")
    })
    .unwrap()
}

const FACT: &str =
    "def fact(n: int) -> int:\n    if n <= 1:\n        return 1\n    return n * fact(n - 1)\n";
const PARITY: &str = "def is_even(n: int) -> bool:\n    if n == 0:\n        return True\n    return is_odd(n - 1)\n\ndef is_odd(n: int) -> bool:\n    if n == 0:\n        return False\n    return is_even(n - 1)\n";

#[test]
fn functions_lower_to_unwrapped_top_level_operations() {
    let source = "def double(n: int) -> int:\n    return n * 2\n";
    let lowered = lower_toplevel(source, &ast::Program::default()).unwrap();
    assert!(
        matches!(lowered.operations.as_slice(), [parser::Operation::Function(f)] if f.name == "double")
    );
    assert!(lowered.legacy_export().unwrap().contains("+fn double"));
    assert!(!lowered.legacy_export().unwrap().contains("+module"));
    let mut fresh = ast::Program::default();
    apply(&mut fresh, source);
    assert_eq!(fresh.functions.len(), 1);
    assert!(fresh.modules.is_empty());
}

#[test]
fn recursion_is_a_bare_call_and_runs_on_both_tiers() {
    let mut p = ast::Program::default();
    let ops = apply(&mut p, FACT);
    let parser::Operation::Function(f) = &ops[0] else {
        panic!()
    };
    let text = format!("{:?}", f.body);
    assert!(
        text.contains("\"fact\"") && !text.contains("\".fact\""),
        "{text}"
    );
    assert!(matches!(
        tree(&p, "fact", vec![Value::Int(5)]),
        Value::Int(120)
    ));
    assert!(matches!(
        bytecode(&p, "fact", vec![Value::Int(5)]),
        Value::Int(120)
    ));
    let printed = print_toplevel_function(&p, "fact").unwrap();
    assert_eq!(
        printed,
        "def fact(n: int) -> int:\n    if (n <= 1):\n        return 1\n    return (n * fact((n - 1)))\n"
    );
}

#[test]
fn mutual_recursion_and_forward_reference_in_one_submission() {
    // `is_odd` is referenced before its definition: both signatures exist before bodies.
    let mut p = ast::Program::default();
    apply(&mut p, PARITY);
    assert!(matches!(
        tree(&p, "is_even", vec![Value::Int(10)]),
        Value::Bool(true)
    ));
    assert!(matches!(
        tree(&p, "is_odd", vec![Value::Int(7)]),
        Value::Bool(true)
    ));
    assert!(matches!(
        bytecode(&p, "is_even", vec![Value::Int(7)]),
        Value::Bool(false)
    ));
    let even = print_toplevel_function(&p, "is_even").unwrap();
    assert!(even.contains("return is_odd((n - 1))"), "{even}");
    let odd = print_toplevel_function(&p, "is_odd").unwrap();
    assert!(odd.contains("return is_even((n - 1))"), "{odd}");
}

#[test]
fn a_later_submission_calls_an_existing_top_level_function_bare() {
    let mut p = ast::Program::default();
    apply(&mut p, FACT);
    apply(
        &mut p,
        "def twice(n: int) -> int:\n    return fact(n) + fact(n)\n",
    );
    assert!(matches!(
        tree(&p, "twice", vec![Value::Int(4)]),
        Value::Int(48)
    ));
    assert!(print_toplevel_function(&p, "twice")
        .unwrap()
        .contains("fact(n)"));
}

#[test]
fn struct_literal_arguments_use_the_bare_callee_parameter_hint() {
    let mut p = ast::Program::default();
    apply(&mut p, "@dataclass\nclass Point:\n    x: int\n    y: int\n");
    apply(&mut p, "def norm(p: Point) -> int:\n    return p.x + p.y\n");
    apply(
        &mut p,
        "def run() -> int:\n    return norm(Point(x=1, y=2))\n",
    );
    assert!(matches!(tree(&p, "run", vec![]), Value::Int(3)));
    let printed = print_toplevel_function(&p, "run").unwrap();
    assert!(printed.contains("norm(Point(x=1, y=2))"), "{printed}");
}

#[test]
fn types_and_functions_in_one_submission_keep_source_order() {
    let mut p = ast::Program::default();
    let ops = apply(
        &mut p,
        "@dataclass\nclass Box:\n    v: int\n\ndef unbox(b: Box) -> int:\n    return b.v\n",
    );
    assert!(matches!(
        ops.as_slice(),
        [parser::Operation::Type(_), parser::Operation::Function(_)]
    ));
    assert_eq!(types::owned_name("", "unbox"), "unbox");
}

#[test]
fn qualified_calls_are_preserved_and_bare_module_calls_are_refused() {
    let base = "+module Geo\n+fn area (n:Int)->Int\n  +return n * n\n+end\n";
    let mut p = program(base);
    apply(&mut p, "def sq(n: int) -> int:\n    return Geo.area(n)\n");
    assert!(matches!(tree(&p, "sq", vec![Value::Int(3)]), Value::Int(9)));
    assert!(print_toplevel_function(&p, "sq")
        .unwrap()
        .contains("Geo.area(n)"));
    // A module function is NOT a bare top-level callee, even though the runtime's
    // first-module fallback would find it: the surface refuses what it cannot prove.
    let message = refused("def sq(n: int) -> int:\n    return area(n)\n", base);
    assert!(message.contains("unknown local callable area"), "{message}");
    // An unresolvable qualified callee is refused rather than typed `Unknown`.
    let message = refused("def sq(n: int) -> int:\n    return Nope.area(n)\n", base);
    assert!(message.contains("no supported signature"), "{message}");
    let message = refused("def sq(n: int) -> int:\n    return missing(n)\n", "");
    assert!(
        message.contains("unknown local callable missing"),
        "{message}"
    );
}

#[test]
fn collision_matrix_refuses_ambiguous_function_names() {
    let types_ax = "+type Shape = Circle(Int) | Dot\n+type Point = x:Int, y:Int\n+module Geo\n+fn area (n:Int)->Int\n  +return n\n+end\n+fn lone ()->Int\n  +return 1\n+end\n";
    let body = "(n: int) -> int:\n    return n\n";
    let cases: &[(&str, &str)] = &[
        ("len", "reserved/unsafe function name len"),
        ("length", "reserved/unsafe function name length"),
        ("http_get", "reserved/unsafe function name http_get"),
        ("spawn", "reserved/unsafe function name spawn"),
        ("propagate", "reserved/unsafe function name propagate"),
        ("eval", "reserved/unsafe function name eval"),
        ("int", "reserved/unsafe function name int"),
        ("Result", "reserved/unsafe function name Result"),
        ("startup", "blocks belong to a module"),
        ("shutdown", "blocks belong to a module"),
        ("Circle", "collides with a union variant"),
        ("Dot", "collides with a union variant"),
        ("Point", "collides with a type"),
        ("Shape", "collides with a type"),
        ("Geo", "collides with a module"),
    ];
    for (name, expected) in cases {
        let message = refused(&format!("def {name}{body}"), types_ax);
        assert!(message.contains(expected), "{name}: {message}");
    }
    // Duplicates: within one submission, and against an existing top-level function.
    let duplicate = format!("def twice{body}\ndef twice{body}");
    assert!(refused(&duplicate, "").contains("duplicate function twice"));
    let existing = "+fn once (n:Int)->Int\n  +return n\n+end\n";
    assert!(refused(&format!("def once{body}"), existing)
        .contains("already exists as a top-level function"));
    // A function named like a type declared in the SAME submission, and a variant declared there.
    let message = refused(
        "@dataclass\nclass Tag:\n    v: int\n\ndef Tag(n: int) -> int:\n    return n\n",
        "",
    );
    assert!(message.contains("collides with a type"), "{message}");
    let message = refused("@union\nclass Pick:\n    Lo: tuple[int]\n    Hi: tuple[()]\n\ndef Lo(n: int) -> int:\n    return n\n", "");
    assert!(
        message.contains("collides with a union variant"),
        "{message}"
    );
    // Decorators that only a module may carry.
    let message = refused(
        "@route(\"GET\", \"/x\")\ndef page(n: int) -> int:\n    return n\n",
        "",
    );
    assert!(message.contains("belong to a module"), "{message}");
    // A local binding named like a top-level function is a reserved name.
    let message = refused(
        &format!(
            "def helper{body}\ndef user(n: int) -> int:\n    helper: int = n\n    return helper\n"
        ),
        "",
    );
    assert!(message.contains("binding helper shadows"), "{message}");
}

#[test]
fn shared_variants_are_ambiguous_at_the_use_site_not_at_declaration() {
    // Two unions may declare the same variant; a use is reported, never guessed.
    let mut p = ast::Program::default();
    apply(&mut p, "@union\nclass A:\n    Pick: tuple[int]\n    NoA: tuple[()]\n\n@union\nclass B:\n    Pick: tuple[int]\n    NoB: tuple[()]\n");
    let message = refused(
        "def make() -> A:\n    return Pick(1)\n",
        "+type A = Pick(Int) | NoA\n+type B = Pick(Int) | NoB\n",
    );
    assert!(message.contains("ambiguous union variant"), "{message}");
    // A function with that spelling is still a declaration-time collision.
    let message = refused(
        "def Pick(n: int) -> int:\n    return n\n",
        "+type A = Pick(Int) | NoA\n+type B = Pick(Int) | NoB\n",
    );
    assert!(
        message.contains("collides with a union variant"),
        "{message}"
    );
}

#[test]
fn module_scoped_lowering_and_printing_are_unchanged() {
    // Module-local bare calls are still qualified by the lowerer.
    let lowered = lower_module(
        "def a(n: int) -> int:\n    return n\n\ndef b(n: int) -> int:\n    return a(n)\n",
        "M",
    )
    .unwrap();
    assert!(
        lowered.legacy_export().unwrap().contains("M.a("),
        "{}",
        lowered.legacy_export().unwrap()
    );
    // An unknown bare callee is refused exactly as before.
    let message = lower_module("def b(n: int) -> int:\n    return a(n)\n", "M")
        .unwrap_err()
        .to_string();
    assert!(message.contains("unknown local callable a"), "{message}");
    // A module function printing a bare call stays unprintable, even when a top-level
    // function of that name exists: only owner "" may print a bare user call.
    let p = program(
        "+fn a (n:Int)->Int\n  +return n\n+end\n+module M\n+fn b (n:Int)->Int\n  +return a(n)\n+end\n",
    );
    let message = print_function(&p, "M.b").unwrap_err().to_string();
    assert!(message.contains("unqualified user call a"), "{message}");
    // And the top-level printer never picks the module function of the same bare name.
    assert!(print_toplevel_function(&p, "b")
        .unwrap_err()
        .to_string()
        .contains("not found"));
}

#[test]
fn unprintable_top_level_calls_are_reported_not_approximated() {
    // A top-level function calling a function that exists only inside a module bare
    // (hand-built program): the printer must not invent a qualification.
    let p = program(
        "+fn caller (n:Int)->Int\n  +return helper(n)\n+end\n+module M\n+fn helper (n:Int)->Int\n  +return n\n+end\n",
    );
    let message = print_toplevel_function(&p, "caller")
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("unqualified user call helper"),
        "{message}"
    );
}

#[test]
fn bare_top_level_calls_obey_the_user_call_effect_and_await_rules() {
    // The bare callee must be classified as a USER call (not a builtin/constructor), so
    // the same site policy applies as for a qualified call.
    let pull = "@effects('io')\nasync def pull(n: int) -> int:\n    return n\n";
    let mut p = ast::Program::default();
    apply(&mut p, pull);
    let message = refused_in(&p, "def sync(n: int) -> int:\n    return pull(n)\n");
    assert!(message.contains("annotated await"), "{message}");
    let message = refused_in(
        &p,
        "async def a(n: int) -> int:\n    v: int = await pull(n)\n    return v\n",
    );
    assert!(
        message.contains("IO await requires @effects('io')"),
        "{message}"
    );
    apply(
        &mut p,
        "@effects('io')\nasync def ok(n: int) -> int:\n    v: int = await pull(n)\n    return v\n",
    );
    let pure = "def plain(n: int) -> int:\n    return n\n";
    apply(&mut p, pure);
    let message = refused_in(
        &p,
        "@effects('io')\nasync def b(n: int) -> int:\n    v: int = await plain(n)\n    return v\n",
    );
    assert!(
        message.contains("synchronous calls are not awaitable"),
        "{message}"
    );
}

fn refused_in(program: &ast::Program, source: &str) -> String {
    lower_toplevel(source, program).unwrap_err().to_string()
}
