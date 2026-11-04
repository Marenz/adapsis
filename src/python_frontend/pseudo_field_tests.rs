//! G1: `.len`/`.is_empty` on str/list and `.is_ok`/`.is_err`/`.unwrap` on a Result are
//! lowered as PROPERTIES to the same `FieldAccess` AST the `.ax` source produces, type-gated,
//! with a same-named struct field always winning. The text is not runnable CPython.
use super::{ast_equality, lower_module, lower_module_with_program, print_function};
use crate::{
    ast,
    coroutine::{CoroutineHandle, IoRequest},
    eval, parser, validator, vm,
};
use eval::{Env, Value};

fn build(operations: &[parser::Operation]) -> ast::Program {
    let mut program = ast::Program::default();
    for operation in operations {
        validator::apply_and_validate(&mut program, operation).unwrap();
    }
    program.rebuild_function_index();
    program
}

fn ax(source: &str) -> ast::Program {
    build(&parser::parse(source).unwrap())
}

fn py(source: &str, module: &str) -> ast::Program {
    build(
        &lower_module(source, module)
            .unwrap_or_else(|e| panic!("{source}\n{e}"))
            .operations,
    )
}

fn refused(source: &str) -> String {
    lower_module(source, "G1").unwrap_err().to_string()
}

fn value(program: &ast::Program, text: &str) -> Value {
    crate::intern::set_display_interner(&program.shared_interner);
    eval::eval_parser_expr_standalone(&parser::parse_single_expr(text).unwrap()).unwrap()
}

fn tree(program: &ast::Program, target: &str, args: Vec<Value>) -> String {
    crate::intern::set_display_interner(&program.shared_interner);
    let function = program.get_function(target).unwrap();
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    for (param, arg) in function.params.iter().zip(args) {
        env.set(&param.name, arg);
    }
    match eval::eval_declared_function(program, target, function, &mut env) {
        Ok(value) => format!("{value}"),
        Err(error) => format!("ERR {error}"),
    }
}

/// (python source, module-less .ax source, function, args, tree result)
fn forms() -> Vec<(
    &'static str,
    &'static str,
    &'static str,
    Vec<Value>,
    &'static str,
)> {
    let s = |t: &str| Value::string(t.to_string());
    let list = || Value::list(vec![Value::Int(1), Value::Int(2)]);
    vec![
        (
            "def f(s: str) -> int:\n    return s.len\n",
            "+fn f (s:String)->Int\n  +return s.len\n+end\n",
            "f",
            vec![s("héllo")],
            "6",
        ),
        (
            "def f(l: list[int]) -> int:\n    return l.len\n",
            "+fn f (l:List<Int>)->Int\n  +return l.len\n+end\n",
            "f",
            vec![list()],
            "2",
        ),
        (
            "def f(s: str) -> bool:\n    return s.is_empty\n",
            "+fn f (s:String)->Bool\n  +return s.is_empty\n+end\n",
            "f",
            vec![s("")],
            "true",
        ),
        (
            "def f(l: list[int]) -> bool:\n    return l.is_empty\n",
            "+fn f (l:List<Int>)->Bool\n  +return l.is_empty\n+end\n",
            "f",
            vec![list()],
            "false",
        ),
        (
            "def f(r: Result[int]) -> bool:\n    return r.is_ok\n",
            "+fn f (r:Result<Int>)->Bool\n  +return r.is_ok\n+end\n",
            "f",
            vec![Value::Ok(Box::new(Value::Int(3)))],
            "true",
        ),
        (
            "def f(r: Result[int]) -> bool:\n    return r.is_err\n",
            "+fn f (r:Result<Int>)->Bool\n  +return r.is_err\n+end\n",
            "f",
            vec![Value::Err("bad".into())],
            "true",
        ),
        (
            "def f(r: Result[int]) -> int:\n    return r.unwrap\n",
            "+fn f (r:Result<Int>)->Int\n  +return r.unwrap\n+end\n",
            "f",
            vec![Value::Ok(Box::new(Value::Int(3)))],
            "3",
        ),
    ]
}

#[test]
fn every_supported_form_lowers_to_the_ax_ast_and_runs_on_the_tree() {
    for (python, source, name, args, expected) in forms() {
        let from_python = py(python, "G1");
        let from_ax = ax(&format!("+module G1\n{source}"));
        assert!(
            ast_equality::function(
                from_python.get_function("G1.f").unwrap(),
                from_ax.get_function("G1.f").unwrap()
            ),
            "{python}"
        );
        assert_eq!(
            tree(&from_python, &format!("G1.{name}"), args),
            expected,
            "{python}"
        );
    }
}

#[test]
fn ax_functions_with_pseudo_fields_print_and_reprint_identically() {
    for (python, source, ..) in forms() {
        let program = ax(&format!("+module G1\n{source}"));
        let text = print_function(&program, "G1.f").unwrap_or_else(|e| panic!("{source}: {e}"));
        assert_eq!(text, python);
        let again = py(&text, "G1");
        assert_eq!(print_function(&again, "G1.f").unwrap(), text);
    }
}

#[test]
fn chains_conditions_and_loops_compose() {
    // `r.unwrap.name` (Result -> struct field), `rec.items.len` (struct field of list
    // type), property conditions and a `while` bound, as the corpus uses them.
    let source = "@dataclass\nclass Item:\n    name: str\n    tags: list[str]\n\ndef name_of(r: Result[Item]) -> str:\n    if r.is_ok:\n        return r.unwrap.name\n    return ''\n\ndef tag_count(item: Item) -> int:\n    return item.tags.len\n\ndef scan(text: str) -> int:\n    pos: int = 0\n    while pos < text.len:\n        pos = pos + 1\n    return pos\n";
    let program = py(source, "G1");
    let item = value(&program, "{name: \"a\", tags: list(\"x\", \"y\")}");
    assert_eq!(
        tree(
            &program,
            "G1.name_of",
            vec![Value::Ok(Box::new(item.clone()))]
        ),
        "\"a\""
    );
    assert_eq!(
        tree(&program, "G1.name_of", vec![Value::Err("e".into())]),
        "\"\""
    );
    assert_eq!(tree(&program, "G1.tag_count", vec![item]), "2");
    assert_eq!(
        tree(&program, "G1.scan", vec![Value::string("abcd".to_string())]),
        "4"
    );
    // Whole-module print roundtrip.
    let printed = print_function(&program, "G1.name_of").unwrap();
    assert!(
        printed.contains("r.unwrap.name") && printed.contains("if r.is_ok:"),
        "{printed}"
    );
}

#[test]
fn a_same_named_struct_field_always_wins() {
    let source = "@dataclass\nclass Count:\n    len: int\n    is_ok: bool\n    unwrap: str\n\ndef f(c: Count) -> int:\n    return c.len\n\ndef g(c: Count) -> bool:\n    return c.is_ok\n\ndef h(c: Count) -> str:\n    return c.unwrap\n";
    let program = py(source, "G1");
    let count = value(&program, "{len: 7, is_ok: false, unwrap: \"u\"}");
    assert_eq!(tree(&program, "G1.f", vec![count.clone()]), "7");
    assert_eq!(tree(&program, "G1.g", vec![count.clone()]), "false");
    assert_eq!(tree(&program, "G1.h", vec![count]), "\"u\"");
    // And on the VM too: these are genuine struct accesses (#114 keeps them there).
    let compiled = vm::compile_function(program.get_function("G1.f").unwrap(), &program);
    assert!(compiled.is_ok(), "{:?}", compiled.err());
    // A struct WITHOUT such a field does not get the pseudo-field: the tree would fail.
    let message = refused(
        "@dataclass\nclass Plain:\n    x: int\n\ndef f(p: Plain) -> int:\n    return p.len\n",
    );
    assert!(message.contains("unknown field len"), "{message}");
}

#[test]
fn everything_outside_the_type_gate_keeps_a_precise_refusal() {
    for (source, needle) in [
        // Wrong base types.
        (
            "def f(n: int) -> int:\n    return n.len\n",
            "supported non-struct properties",
        ),
        (
            "def f(b: bool) -> bool:\n    return b.is_ok\n",
            "supported non-struct properties",
        ),
        (
            "def f(s: str) -> bool:\n    return s.is_ok\n",
            "supported non-struct properties",
        ),
        (
            "def f(s: str) -> str:\n    return s.unwrap\n",
            "supported non-struct properties",
        ),
        (
            "def f(l: list[int]) -> int:\n    return l.unwrap\n",
            "supported non-struct properties",
        ),
        (
            "def f(o: Option[int]) -> int:\n    return o.unwrap\n",
            "supported non-struct properties",
        ),
        // The runtime rejects these, so they are refused with a reason, not approximated.
        (
            "def f(r: Result[int]) -> int:\n    return r.len\n",
            "not supported on a Result",
        ),
        (
            "def f(r: Result[str]) -> bool:\n    return r.is_empty\n",
            "not supported on a Result",
        ),
        (
            "def f(m: Map[str, int]) -> int:\n    return m.len\n",
            "not supported on Map/Set",
        ),
        (
            "def f(m: Map[str, int]) -> bool:\n    return m.is_empty\n",
            "not supported on Map/Set",
        ),
        (
            "def f(s: Set[int]) -> int:\n    return s.len\n",
            "not supported on Map/Set",
        ),
        // Names deliberately out of scope.
        (
            "def f(r: Result[int]) -> str:\n    return r.error\n",
            "supported non-struct properties",
        ),
        (
            "def f(r: Result[int]) -> str:\n    return r.unwrap_err\n",
            "supported non-struct properties",
        ),
        (
            "def f(o: Option[int]) -> bool:\n    return o.is_some\n",
            "supported non-struct properties",
        ),
        (
            "def f(o: Option[int]) -> bool:\n    return o.is_none\n",
            "supported non-struct properties",
        ),
        // Result of a pseudo-field is not itself a pseudo-field base.
        (
            "def f(s: str) -> int:\n    return s.len.len\n",
            "supported non-struct properties",
        ),
        // Method form and builtin form are different ASTs and stay unsupported here.
        (
            "def f(s: str) -> int:\n    return s.len()\n",
            "dynamic attributes",
        ),
    ] {
        let message = refused(source);
        assert!(message.contains(needle), "{source}\n{message}");
    }
    // Only a typed local/parameter name is a base: calls and literals are not.
    for source in [
        "def g() -> str:\n    return 'x'\n\ndef f() -> int:\n    return g().len\n",
        "def f() -> int:\n    return 'abc'.len\n",
        "def f(l: list[int]) -> int:\n    return [1, 2].len\n",
    ] {
        assert!(lower_module(source, "G1").is_err(), "{source}");
    }
    // A property is a value, never an assignment target.
    assert!(lower_module("def f(s: str) -> int:\n    s.len = 3\n    return 0\n", "G1").is_err());
}

#[test]
fn native_entry_and_preflight_agree_with_the_free_lowering() {
    for (python, ..) in forms() {
        let free = lower_module(python, "G1").unwrap().legacy_export().unwrap();
        let native = lower_module_with_program(python, "G1", &ast::Program::default())
            .unwrap_or_else(|e| panic!("{python}\n{e}"))
            .legacy_export().unwrap();
        assert_eq!(free, native);
    }
    for source in [
        "def f(m: Map[str, int]) -> int:\n    return m.len\n",
        "def f(r: Result[int]) -> int:\n    return r.len\n",
    ] {
        assert!(lower_module_with_program(source, "G1", &ast::Program::default()).is_err());
    }
}

#[test]
fn the_vm_never_runs_pseudo_field_functions_and_async_after_io_gives_the_tree_result() {
    let python = "@effects('io')\nasync def after(path: str) -> int:\n    text: str = await read_file(path)\n    return text.len\n\ndef plain(s: str) -> int:\n    return s.len\n";
    let program = py(python, "G1");
    for name in ["G1.after", "G1.plain"] {
        let error =
            vm::compile_function(program.get_function(name).unwrap(), &program).unwrap_err();
        assert!(
            error.to_string().contains("requires tree evaluation"),
            "{name}: {error}"
        );
    }
    let (sender, mut receiver) = tokio::sync::mpsc::channel(8);
    let responder = std::thread::spawn(move || {
        let mut seen = Vec::new();
        while let Some(request) = receiver.blocking_recv() {
            let IoRequest::FileRead { path, reply } = request else {
                panic!("only FileRead")
            };
            seen.push(path.clone());
            assert!(seen.len() <= 2, "replay");
            reply.send(Ok("héllo".to_string())).unwrap();
        }
        seen
    });
    let result = eval::eval_async_function(
        &program,
        "G1.after",
        &parser::parse_single_expr("{path: \"p\"}").unwrap(),
        CoroutineHandle::new(sender),
    );
    assert!(
        matches!(result.unwrap(), Value::Int(6)),
        "byte length of the read text"
    );
    assert_eq!(responder.join().unwrap(), ["p"]);
    // Sync normal dispatch agrees.
    let (text, _) = eval::eval_compiled_or_interpreted(
        &program,
        "G1.plain",
        &parser::parse_single_expr("{s: \"héllo\"}").unwrap(),
    )
    .unwrap();
    assert_eq!(text, "6");
}
