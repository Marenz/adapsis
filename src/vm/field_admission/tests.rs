use super::*;
use crate::{
    coroutine::{CoroutineHandle, IoRequest},
    eval::{self, Value},
    parser, validator, vm,
};

fn program(source: &str) -> ast::Program {
    let mut program = ast::Program::default();
    for operation in parser::parse(source).unwrap() {
        validator::apply_and_validate(&mut program, &operation).unwrap();
    }
    program.rebuild_function_index();
    program
}

fn refused(program: &ast::Program, target: &str) -> bool {
    match vm::compile_function(program.get_function(target).unwrap(), program) {
        Ok(_) => false,
        Err(e) => {
            assert!(e.to_string().contains(REQUIRES_TREE), "{target}: {e}");
            true
        }
    }
}

fn entry(program: &ast::Program, target: &str, input: &str) -> anyhow::Result<String> {
    eval::eval_compiled_or_interpreted(program, target, &parser::parse_single_expr(input).unwrap())
        .map(|(text, _)| text)
}

/// Async normal dispatch with a synthetic FileRead responder; returns result + requests.
fn run_async(
    program: &ast::Program,
    target: &str,
    input: &str,
) -> (anyhow::Result<Value>, Vec<String>) {
    let (sender, mut receiver) = tokio::sync::mpsc::channel(8);
    let responder = std::thread::spawn(move || {
        let mut seen = Vec::new();
        while let Some(request) = receiver.blocking_recv() {
            let IoRequest::FileRead { path, reply } = request else {
                panic!("only synthetic FileRead allowed")
            };
            seen.push(path.clone());
            assert!(seen.len() <= 4, "request replay/unbounded execution");
            reply.send(Ok(path)).unwrap();
        }
        seen
    });
    let result = eval::eval_async_function(
        program,
        target,
        &parser::parse_single_expr(input).unwrap(),
        CoroutineHandle::new(sender),
    );
    (result, responder.join().unwrap())
}

const POINT: &str = "+type Point = x:Int, y:Int\n";

fn tree(program: &ast::Program, target: &str, input: &str) -> String {
    eval::eval_call_with_input(program, target, &parser::parse_single_expr(input).unwrap())
        .unwrap_or_else(|e| panic!("{target}: {e}"))
}

fn vm_direct(program: &ast::Program, target: &str, input: &str) -> anyhow::Result<String> {
    crate::intern::set_display_interner(&program.shared_interner);
    let function = program.get_function(target).unwrap();
    let args = eval::input_to_vm_args(&parser::parse_single_expr(input).unwrap(), function)?;
    let compiled = vm::compile_function(function, program)?;
    vm::execute_with_io(&compiled, args, program, &|_, _| anyhow::bail!("no io"))
        .map(|v| format!("{v}"))
}

const PSEUDO: &str = "+type Point = x:Int, y:Int\n+fn slen (s:String)->Int\n  +return s.len\n+end\n+fn llen (l:List<Int>)->Int\n  +return l.len\n+end\n+fn sempty (s:String)->Bool\n  +return s.is_empty\n+end\n+fn lempty (l:List<Int>)->Bool\n  +return l.is_empty\n+end\n+fn rok (r:Result<Int>)->Bool\n  +return r.is_ok\n+end\n+fn rerr (r:Result<Int>)->Bool\n  +return r.is_err\n+end\n+fn runw (r:Result<Int>)->Int\n  +return r.unwrap\n+end\n+fn rerror (r:Result<Int>)->String\n  +return r.error\n+end\n+fn runwerr (r:Result<Int>)->String\n  +return r.unwrap_err\n+end\n+fn transp (r:Result<Point>)->Int\n  +return r.x\n+end\n";

#[test]
fn every_non_struct_field_access_the_tree_implements_is_refused_before_execution() {
    let p = program(PSEUDO);
    // (function, input, tree result): the tree walker is the reference.
    for (name, input, expected) in [
        ("slen", "{s: \"héllo\"}", "6"),
        ("llen", "{l: list(1, 2)}", "2"),
        ("sempty", "{s: \"\"}", "true"),
        ("lempty", "{l: list()}", "true"),
        ("rok", "{r: Ok(3)}", "true"),
        ("rerr", "{r: Err(\"bad\")}", "true"),
        ("runw", "{r: Ok(3)}", "3"),
        ("rerror", "{r: Err(\"bad\")}", "\"\"bad\"\""),
        ("runwerr", "{r: Err(\"bad\")}", "\"\"bad\"\""),
        ("transp", "{r: Ok({x: 3, y: 4})}", "3"),
    ] {
        assert!(refused(&p, name), "{name} must not compile for the VM");
        assert_eq!(tree(&p, name, input), expected, "{name}");
        // Normal sync dispatch reaches the same answer (VM refused up front).
        assert_eq!(
            entry(&p, name, input).unwrap(),
            expected,
            "{name} via dispatch"
        );
        // Direct VM: refused before execution, never a partial run.
        assert!(
            vm_direct(&p, name, input)
                .unwrap_err()
                .to_string()
                .contains(REQUIRES_TREE),
            "{name}"
        );
    }
}

fn statement(kind: ast::StatementKind) -> ast::Statement {
    ast::Statement {
        id: "field-admission-fixture".into(),
        kind,
    }
}

fn bad() -> ast::Expr {
    ast::Expr::FieldAccess {
        base: Box::new(ast::Expr::Identifier("s".into())),
        field: "len".into(),
    }
}

fn good() -> ast::Expr {
    ast::Expr::Identifier("s".into())
}

fn call(args: Vec<ast::Expr>) -> ast::CallExpr {
    ast::CallExpr {
        callee: "outer".into(),
        args,
    }
}

/// Every executable slot in the statement language, holding `x`.
fn slot_cases(x: impl Fn() -> ast::Expr) -> Vec<ast::StatementKind> {
    use ast::{Expr as E, StatementKind as S};
    let nested = || vec![statement(S::Return { value: x() })];
    let binding = || ast::Binding {
        name: "v".into(),
        ty: ast::Type::Int,
    };
    let literal = || E::Literal(ast::Literal::Bool(true));
    vec![
        S::Let {
            name: "v".into(),
            ty: ast::Type::Int,
            value: x(),
        },
        S::Set {
            name: "v".into(),
            value: x(),
        },
        S::Return { value: x() },
        S::Yield { value: x() },
        S::Call {
            binding: Some(binding()),
            call: call(vec![x()]),
        },
        S::Spawn {
            binding: None,
            call: call(vec![x()]),
        },
        S::Await {
            name: "v".into(),
            ty: ast::Type::Int,
            call: call(vec![x()]),
        },
        S::Check {
            label: "c".into(),
            condition: x(),
            on_fail: "failed".into(),
        },
        S::Branch {
            condition: x(),
            then_body: vec![],
            else_body: vec![],
        },
        S::Branch {
            condition: literal(),
            then_body: nested(),
            else_body: vec![],
        },
        S::Branch {
            condition: literal(),
            then_body: vec![],
            else_body: nested(),
        },
        S::While {
            condition: x(),
            body: vec![],
        },
        S::While {
            condition: literal(),
            body: nested(),
        },
        S::Each {
            iterator: x(),
            binding: binding(),
            body: vec![],
        },
        S::Each {
            iterator: literal(),
            binding: binding(),
            body: nested(),
        },
        S::Match {
            expr: x(),
            arms: vec![],
        },
        S::Match {
            expr: literal(),
            arms: vec![ast::MatchArm {
                variant: "_".into(),
                bindings: vec![],
                patterns: None,
                body: nested(),
            }],
        },
        S::Source(ast::SourceOp::Add {
            source_type: ast::SourceType::Timer(Box::new(x())),
            alias: "a".into(),
            handler: "h".into(),
        }),
        S::Source(ast::SourceOp::Replace {
            source_type: ast::SourceType::Timer(Box::new(x())),
            alias: "a".into(),
            handler: "h".into(),
        }),
        S::Event(ast::EventOp::Emit {
            name: "e".into(),
            value: Box::new(x()),
        }),
    ]
}

/// Every nesting position of an expression, holding `x`.
fn expression_cases(x: impl Fn() -> ast::Expr) -> Vec<ast::Expr> {
    use ast::Expr as E;
    vec![
        E::Binary {
            left: Box::new(x()),
            op: ast::BinaryOp::Add,
            right: Box::new(E::Literal(ast::Literal::Int(1))),
        },
        E::Binary {
            left: Box::new(E::Literal(ast::Literal::Int(1))),
            op: ast::BinaryOp::Add,
            right: Box::new(x()),
        },
        E::Unary {
            op: ast::UnaryOp::Not,
            expr: Box::new(x()),
        },
        E::StructInit {
            ty: "Record".into(),
            fields: vec![ast::StructFieldValue {
                name: "f".into(),
                value: x(),
            }],
        },
        E::Call(call(vec![E::Literal(ast::Literal::Int(0)), x()])),
        // A struct literal is a proven base, so the control stays admitted.
        E::FieldAccess {
            base: Box::new(E::StructInit {
                ty: "Record".into(),
                fields: vec![ast::StructFieldValue {
                    name: "f".into(),
                    value: x(),
                }],
            }),
            field: "f".into(),
        },
    ]
}

fn root_with(body: Vec<ast::Statement>) -> ast::Program {
    let mut base = program("+fn root (s:String)->Int\n  +return 0\n+end\n");
    base.get_function_mut("root").unwrap().body = body;
    base
}

#[test]
fn every_executable_slot_is_inspected() {
    for (index, kind) in slot_cases(bad).into_iter().enumerate() {
        let p = root_with(vec![statement(kind)]);
        assert!(
            analyze(p.get_function("root").unwrap(), &p),
            "statement slot {index}"
        );
    }
    for (index, expression) in expression_cases(bad).into_iter().enumerate() {
        let p = root_with(vec![statement(ast::StatementKind::Return {
            value: expression,
        })]);
        assert!(
            analyze(p.get_function("root").unwrap(), &p),
            "expression slot {index}"
        );
    }
    // Controls: the same shapes without the field access are admitted, so the markers
    // above are caused by the access and nothing else in the fixture.
    for (index, kind) in slot_cases(good).into_iter().enumerate() {
        let p = root_with(vec![statement(kind)]);
        assert!(
            !analyze(p.get_function("root").unwrap(), &p),
            "control statement {index}"
        );
    }
    for (index, expression) in expression_cases(good).into_iter().enumerate() {
        let p = root_with(vec![statement(ast::StatementKind::Return {
            value: expression,
        })]);
        assert!(
            !analyze(p.get_function("root").unwrap(), &p),
            "control expression {index}"
        );
    }
}

#[test]
fn a_deep_field_chain_is_decided_without_recursion() {
    let mut expression = ast::Expr::Identifier("s".into());
    for _ in 0..20_000 {
        expression = ast::Expr::FieldAccess {
            base: Box::new(expression),
            field: "f".into(),
        };
    }
    let p = root_with(vec![statement(ast::StatementKind::Return {
        value: expression,
    })]);
    assert!(analyze(p.get_function("root").unwrap(), &p));
    // Leak instead of dropping the 20k-deep Box chain recursively on a small test stack.
    std::mem::forget(p);
}

#[test]
fn reachable_graph_is_inspected_with_tree_lookup_and_cycles() {
    let helpers = "+fn leaf (s:String)->Int\n  +return s.len\n+end\n+fn clean (s:String)->Int\n  +return 1\n+end\n";
    for (body, expect) in [
        ("+return leaf(s)", true),
        ("+call v:Int = leaf(s)\n  +return v", true),
        ("+let v:Int = leaf(s)\n  +return v", true),
        ("+return clean(s) + leaf(s)", true),
        ("+return clean(s)", false),
    ] {
        let p = program(&format!(
            "{helpers}+fn root (s:String)->Int\n  {body}\n+end\n"
        ));
        assert_eq!(refused(&p, "root"), expect, "{body}");
    }
    // Transitive and cyclic graphs terminate and still see the access.
    let p = program("+fn a (s:String)->Int\n  +return b(s)\n+end\n+fn b (s:String)->Int\n  +call v:Int = a(s)\n  +return c(s)\n+end\n+fn c (s:String)->Int\n  +return s.len\n+end\n");
    assert!(refused(&p, "a") && refused(&p, "b") && refused(&p, "c"));
    let p = program(
        "+fn a (n:Int)->Int\n  +return b(n)\n+end\n+fn b (n:Int)->Int\n  +return a(n)\n+end\n",
    );
    assert!(!refused(&p, "a"));
    // Callee resolution is the tree's get_function, NOT caller-module-first: the bare
    // name resolves to the first module defining it.
    let p = program("+module A\n+fn child (s:String)->Int\n  +return s.len\n+end\n+module B\n+fn child (s:String)->Int\n  +return 1\n+end\n+fn bare (s:String)->Int\n  +return child(s)\n+end\n+fn qualified (s:String)->Int\n  +return B.child(s)\n+end\n");
    assert!(refused(&p, "B.bare"), "bare resolves to A.child");
    assert!(!refused(&p, "B.qualified"), "B.child is clean");
}

const STRUCTS: &str = "+type Point = x:Int, y:Int\n+type Line = a:Point, b:Point\n+type Count = len:Int, error:Int\n+fn dist (p:Point)->Int\n  +return p.x + p.y\n+end\n+fn span (l:Line)->Int\n  +return l.b.x - l.a.x\n+end\n+fn total (ps:List<Point>)->Int\n  +let t:Int = 0\n  +each ps p:Point\n    +set t = t + p.x\n  +end\n  +return t\n+end\n+fn lit ()->Int\n  +let p:Point = {x: 1, y: 2}\n  +return p.y\n+end\n+fn mkp (n:Int)->Point\n  +return {x: n, y: n}\n+end\n+fn viacall (n:Int)->Int\n  +let p:Point = mkp(n)\n  +return p.x\n+end\n+fn passes (n:Int)->Int\n  +let p:Point = mkp(n)\n  +return dist(p)\n+end\n+fn fieldnamed (c:Count)->Int\n  +return c.len + c.error\n+end\n";

#[test]
fn genuine_struct_field_access_still_compiles_and_runs_on_the_vm() {
    let p = program(STRUCTS);
    for (name, input, expected) in [
        ("dist", "{p: {x: 3, y: 4}}", "7"),
        ("span", "{l: {a: {x: 1, y: 0}, b: {x: 6, y: 0}}}", "5"),
        ("total", "{ps: list({x: 1, y: 0}, {x: 2, y: 0})}", "3"),
        ("lit", "{}", "2"),
        ("viacall", "{n: 5}", "5"),
        ("passes", "{n: 4}", "8"),
        ("fieldnamed", "{c: {len: 2, error: 3}}", "5"),
    ] {
        assert!(!refused(&p, name), "{name} must stay on the VM");
        assert_eq!(
            vm_direct(&p, name, input).unwrap(),
            expected,
            "{name} on the VM"
        );
        assert_eq!(tree(&p, name, input), expected, "{name} on the tree");
        assert_eq!(
            entry(&p, name, input).unwrap(),
            expected,
            "{name} via dispatch"
        );
    }
}

#[test]
fn unproven_bases_are_refused_even_when_the_field_names_are_ordinary() {
    let prefix = "+type Point = x:Int, y:Int\n+fn mk (n:Int)->Result<Point>\n  +return Ok({x: n, y: n})\n+end\n+fn getx (p:Point)->Int\n  +return p.x\n+end\n";
    for (body, why) in [
        ("+call v:Point = mk(n)\n  +return v.x", "+call binding"),
        ("+let v:Point = mk(n)\n  +return v.x", "let of a Result-returning call"),
        ("+let v:Point = 5\n  +return v.x", "let of a non-struct value"),
        ("+call v:Point = mk(n)\n  +return getx(v)", "unproven argument for a Struct parameter"),
        ("+let v:Point = {x: 1, y: 2}\n  +set v = n\n  +return v.x", "set to a non-struct"),
        ("+let v:Point = {x: 1, y: 2}\n  +let r:Result<Int> = Ok(n)\n  +match r\n  +case Ok(v)\n    +return v.x\n  +case Err(e)\n    +return 0\n  +end\n  +return 0", "match capture shadows a proven name"),
        ("+let v:Point = {x: 1, y: 2}\n  +let v:Int = n\n  +return v.x", "conflicting bindings"),
        ("+return missing.x", "unbound identifier"),
        ("+return mk(n).x", "builtin-or-Result call as a base"),
    ] {
        let p = program(&format!("{prefix}+fn root (n:Int)->Int\n  {body}\n+end\n"));
        assert!(refused(&p, "root"), "{why}");
    }
    // Controls for the same fixture: proven shapes stay admitted.
    for body in [
        "+let v:Point = {x: n, y: n}\n  +return getx(v)",
        "+let v:Point = {x: n, y: n}\n  +set v = {x: 1, y: 2}\n  +return v.x",
    ] {
        let p = program(&format!("{prefix}+fn root (n:Int)->Int\n  {body}\n+end\n"));
        assert!(!refused(&p, "root"), "{body}");
    }
    // An undeclared or ambiguous struct name cannot be proven (no unique declaration).
    // The declared `Struct` type proves a LEAF access without needing the declaration, but
    // a chained field needs the declaration to know the field's type: ambiguous => tree.
    let ambiguous = program("+module A\n+type Inner = v:Int\n+type Point = x:Int\n+module B\n+type Inner = v:Int, w:Int\n+type Point = x:Int, inner:Inner\n+fn leaf (p:Point)->Int\n  +return p.x\n+end\n+fn chain (p:Point)->Int\n  +return p.inner.v\n+end\n");
    assert!(!refused(&ambiguous, "B.leaf"));
    assert!(refused(&ambiguous, "B.chain"));
    let unique =
        program("+module A\n+type Point = x:Int\n+fn pick (p:Point)->Int\n  +return p.x\n+end\n");
    assert!(!refused(&unique, "A.pick"));
    // A shared variable is never a proven base.
    let shared = program("+module S\n+type Point = x:Int\n+shared origin:Point = {x: 0}\n+fn read ()->Int\n  +return origin.x\n+end\n");
    assert!(refused(&shared, "S.read"));
}

#[test]
fn rebinding_a_proven_name_poisons_it_for_every_binding_form() {
    // Each form rebinds `v` (declared `Point`, so initially proven) to something the
    // tree may hold differently from the declared type, then reads `v.x`.
    let prefix = "+type Point = x:Int, y:Int\n+fn mk (n:Int)->Result<Point>\n  +return Ok({x: n, y: n})\n+end\n+fn slow (n:Int)->Int [io,async]\n  +return n\n+end\n";
    for (body, why) in [
        ("+call v:Point = mk(n)\n  +return v.x", "+call rebinding"),
        ("+spawn v:Point = mk(n)\n  +return v.x", "+spawn rebinding"),
        ("+await v:Point = read_file(\"a\")\n  +return v.x", "+await rebinding"),
        ("+let r:Result<Int> = Ok(n)\n  +match r\n  +case Ok(v)\n    +return v.x\n  +case Err(e)\n    +return 0\n  +end\n  +return 0", "match capture"),
    ] {
        let p = program(&format!("{prefix}+fn root (v:Point, n:Int)->Int [io,async]\n  {body}\n+end\n"));
        assert!(refused(&p, "root"), "{why}");
    }
    // Control: without the rebinding the parameter stays proven.
    let p = program(&format!(
        "{prefix}+fn root (v:Point, n:Int)->Int\n  +return v.x\n+end\n"
    ));
    assert!(!refused(&p, "root"));
    // Nested variant patterns bind names too.
    let p = program("+type Point = x:Int, y:Int\n+type Box = Full(Int) | Empty\n+type Wrap = Wrapped(Box) | Bare\n+fn root (v:Point, w:Wrap)->Int\n  +match w\n  +case Wrapped(Full(v))\n    +return v.x\n  +case _\n    +return 0\n  +end\n  +return 0\n+end\n");
    assert!(refused(&p, "root"), "nested pattern binding");
}

#[test]
fn a_name_bound_to_different_structs_is_not_proven_for_chains() {
    // `p` is first `B` (inner: Int), then rebound by an `+each` to `A` (inner: Inner).
    // Merging to the last declaration would prove `p.inner.v` for the `B` value.
    let p = program("+type Inner = v:Int\n+type A = inner:Inner\n+type B = inner:Int\n+fn root (p:B, xs:List<A>)->Int\n  +each xs p:A\n    +return p.inner.v\n  +end\n  +return 0\n+end\n");
    assert!(refused(&p, "root"));
    let p = program("+type Inner = v:Int\n+type A = inner:Inner\n+fn root (p:A, xs:List<A>)->Int\n  +each xs p:A\n    +return p.inner.v\n  +end\n  +return 0\n+end\n");
    assert!(!refused(&p, "root"), "agreeing bindings stay proven");
}

#[test]
fn chained_access_needs_a_declared_struct_field_and_a_unique_declaration() {
    // `inner` is an Int field: `p.inner.v` is not a struct chain.
    let p = program("+type A = inner:Int\n+fn root (p:A)->Int\n  +return p.inner.v\n+end\n");
    assert!(refused(&p, "root"));
    // Undeclared field name.
    let p = program("+type A = inner:Int\n+fn root (p:A)->Int\n  +return p.nope.v\n+end\n");
    assert!(refused(&p, "root"));
    // Same type name in two modules: the declaration (and its field types) is ambiguous.
    let p = program("+module A\n+type Inner = v:Int\n+type Point = inner:Inner\n+module B\n+type Inner = v:Int, w:Int\n+type Point = inner:Inner\n+fn chain (p:Point)->Int\n  +return p.inner.v\n+end\n");
    assert!(refused(&p, "B.chain"));
}

#[test]
fn struct_parameters_are_sound_proof_for_internal_call_sites() {
    // #114 weak point 1: a `+call` binding holds Ok(struct) in the VM (the tree unwraps
    // it), so passing it to a Struct-typed parameter would reach GetField after IO.
    let p = program(&format!("{POINT}+fn mk (n:Int)->Result<Point>\n  +return Ok({{x: n, y: n}})\n+end\n+fn getx (p:Point)->Int\n  +return p.x\n+end\n+fn root (n:Int)->Int [io,async]\n  +await s:String = read_file(\"a\")\n  +call v:Point = mk(n)\n  +return getx(v)\n+end\n"));
    assert!(
        refused(&p, "root"),
        "unproven argument into a Struct parameter"
    );
    assert!(
        !refused(&p, "getx"),
        "the callee on its own stays on the VM"
    );
    let (result, requests) = run_async(&p, "root", "{n: 4}");
    assert!(matches!(result.unwrap(), Value::Int(4)));
    assert_eq!(
        requests,
        ["a"],
        "one request: tree ran the whole function once"
    );
}

#[test]
fn struct_parameters_are_sound_proof_for_entry_arguments() {
    // #114 weak point 2: entry input is external. `Ok({..})` reaches a Struct parameter,
    // the tree reads it transparently, the VM would fail after IO.
    let p = program(&format!("{POINT}+fn getx (p:Point)->Int\n  +return p.x\n+end\n+fn getx2 (p:Point)->Int [io,async]\n  +await s:String = read_file(\"a\")\n  +return p.x\n+end\n"));
    let ok = "{p: Ok({x: 1, y: 2})}";
    let plain = "{p: {x: 1, y: 2}}";
    // Direct VM: the guard fires before any instruction for a non-struct entry argument.
    assert!(vm_direct(&p, "getx", ok)
        .unwrap_err()
        .to_string()
        .contains(NON_STRUCT_ARGUMENT));
    assert!(vm_direct(&p, "getx", "{p: 5}")
        .unwrap_err()
        .to_string()
        .contains(NON_STRUCT_ARGUMENT));
    assert_eq!(vm_direct(&p, "getx", plain).unwrap(), "1");
    // Normal sync dispatch.
    assert_eq!(entry(&p, "getx", ok).unwrap(), "1");
    assert_eq!(entry(&p, "getx", plain).unwrap(), "1");
    // Normal async dispatch after an intercepted await.
    let (result, requests) = run_async(&p, "getx2", ok);
    assert!(matches!(result.unwrap(), Value::Int(1)));
    assert_eq!(requests, ["a"]);
    let (result, requests) = run_async(&p, "getx2", plain);
    assert!(matches!(result.unwrap(), Value::Int(1)));
    assert_eq!(requests, ["a"]);
    // A truly wrong value is the same failure in both tiers (and still one request).
    let (result, requests) = run_async(&p, "getx2", "{p: 5}");
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("cannot access field `x` on 5"));
    assert_eq!(requests, ["a"]);
}

#[test]
fn async_after_an_intercepted_await_runs_pseudo_field_functions_on_the_tree() {
    // The original #114 symptom: IO first, then `.len`/`.unwrap` on the VM failed with
    // "GetField: expected struct" and could not be retried after IO.
    let p = program("+fn slen (s:String)->Int\n  +return s.len\n+end\n+fn after (u:Int)->Int [io,async]\n  +await s:String = read_file(\"first\")\n  +return s.len\n+end\n+fn through (u:Int)->Int [io,async]\n  +await s:String = read_file(\"first\")\n  +let n:Int = slen(s)\n  +await t:String = read_file(\"second\")\n  +return n + t.len\n+end\n+fn unwrapped (r:Result<Int>)->Int [io,async]\n  +await s:String = read_file(\"first\")\n  +return r.unwrap\n+end\n");
    for target in ["after", "through", "unwrapped"] {
        assert!(refused(&p, target), "{target}");
    }
    let (result, requests) = run_async(&p, "after", "{u: 0}");
    assert!(matches!(result.unwrap(), Value::Int(5)));
    assert_eq!(requests, ["first"]);
    let (result, requests) = run_async(&p, "through", "{u: 0}");
    assert!(matches!(result.unwrap(), Value::Int(11)), "5 + 6");
    assert_eq!(requests, ["first", "second"]);
    let (result, requests) = run_async(&p, "unwrapped", "{r: Ok(7)}");
    assert!(matches!(result.unwrap(), Value::Int(7)));
    assert_eq!(requests, ["first"]);
}

#[test]
fn refusal_precedes_user_await_admission_and_both_stay_independent() {
    // A function needing the tree for either reason is refused once, up front; the await
    // admission message keeps its own text when the await is what forces the tree.
    let p = program("+fn child ()->String [io,async]\n  +await v:String = read_file(\"c\")\n  +return v\n+end\n+fn viaawait ()->String [io,async]\n  +await v:String = child()\n  +return v\n+end\n+fn viafield (s:String)->Int [io,async]\n  +await v:String = read_file(\"c\")\n  +return s.len\n+end\n");
    let error = vm::compile_function(p.get_function("viaawait").unwrap(), &p).unwrap_err();
    assert!(error.to_string().contains("user-function await"), "{error}");
    assert!(refused(&p, "viafield"));
}

#[test]
fn guard_ignores_non_struct_parameters_and_extra_arguments() {
    let p = program("+fn plain (n:Int, s:String)->Int\n  +return n\n+end\n");
    let compiled = vm::compile_function(p.get_function("plain").unwrap(), &p).unwrap();
    assert_eq!(compiled.struct_params, [false, false]);
    assert!(check_arguments(&compiled, &[Value::Int(1), Value::Int(2)]).is_ok());
    let q = program(&format!(
        "{POINT}+fn two (a:Point, n:Int)->Int\n  +return a.x\n+end\n"
    ));
    let compiled = vm::compile_function(q.get_function("two").unwrap(), &q).unwrap();
    assert_eq!(compiled.struct_params, [true, false]);
    assert!(check_arguments(&compiled, &[Value::Int(1), Value::Int(2)]).is_err());
    assert!(
        check_arguments(&compiled, &[]).is_ok(),
        "arity is not this guard's concern"
    );
}
