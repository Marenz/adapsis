//! #90: `and`/`or` must short-circuit identically in tree, VM and JIT, on one
//! truthiness definition. Tree short-circuit is normative; these tests are the
//! contract that keeps the three tiers from drifting again.
use crate::{ast, compiler, eval, parser, validator, vm};
use eval::Value;

fn build(source: &str) -> ast::Program {
    let mut prog = ast::Program::default();
    for op in parser::parse(source).unwrap() {
        validator::apply_and_validate(&mut prog, &op).unwrap();
    }
    prog.rebuild_function_index();
    crate::intern::set_display_interner(&prog.shared_interner);
    prog
}

fn tree(prog: &ast::Program, target: &str, args: Vec<Value>) -> anyhow::Result<Value> {
    let function = prog.get_function(target).unwrap();
    let mut env = eval::Env::new_with_shared_interner(&prog.shared_interner);
    for (param, arg) in function.params.iter().zip(args) {
        env.set(&param.name, arg);
    }
    eval::eval_function_body_named(prog, target, &function.body, &mut env)
}

fn vm_run(prog: &ast::Program, target: &str, args: Vec<Value>) -> anyhow::Result<Value> {
    let compiled = vm::compile_function(prog.get_function(target).unwrap(), prog)?;
    vm::execute_with_io(&compiled, args, prog, &|_, _| {
        anyhow::bail!("no IO expected")
    })
}

fn text(value: &Value) -> String {
    value.to_string()
}

/// The rhs is an erroring division. Short-circuit means the tier must NOT
/// evaluate it when the lhs decides the result.
#[test]
fn python_logical_rhs_error_is_skipped_on_short_circuit_in_tree_and_vm() {
    let source = "+module LogErr\n+fn rhs (n:Int)->Bool\n  +return 1/n>0\n+end\n+fn conj (flag:Bool,n:Int)->Bool\n  +return flag AND rhs(n)\n+end\n+fn disj (flag:Bool,n:Int)->Bool\n  +return flag OR rhs(n)\n+end\n";
    // AND with false lhs: rhs would divide by zero; tree and VM both skip it.
    let and_args = vec![Value::Bool(false), Value::Int(0)];
    assert_eq!(
        text(&tree(&build(source), "LogErr.conj", and_args.clone()).unwrap()),
        "false"
    );
    assert_eq!(
        text(&vm_run(&build(source), "LogErr.conj", and_args).unwrap()),
        "false"
    );
    // OR with true lhs: same.
    let or_args = vec![Value::Bool(true), Value::Int(0)];
    assert_eq!(
        text(&tree(&build(source), "LogErr.disj", or_args.clone()).unwrap()),
        "true"
    );
    assert_eq!(
        text(&vm_run(&build(source), "LogErr.disj", or_args).unwrap()),
        "true"
    );
    // Non-short-circuit paths still evaluate rhs and surface the same error.
    assert!(
        tree(
            &build(source),
            "LogErr.conj",
            vec![Value::Bool(true), Value::Int(0)]
        )
        .is_err()
    );
    assert!(
        vm_run(
            &build(source),
            "LogErr.conj",
            vec![Value::Bool(true), Value::Int(0)]
        )
        .is_err()
    );
    assert!(
        tree(
            &build(source),
            "LogErr.disj",
            vec![Value::Bool(false), Value::Int(0)]
        )
        .is_err()
    );
    assert!(
        vm_run(
            &build(source),
            "LogErr.disj",
            vec![Value::Bool(false), Value::Int(0)]
        )
        .is_err()
    );
}

/// A side-effecting rhs must not run when short-circuited. Shared variables need
/// a live runtime for isolated calls, so side effects are observed through a
/// stack-overflowing recursion: an evaluated rhs aborts, a skipped one does not.
#[test]
fn python_logical_rhs_side_effect_does_not_run_when_short_circuited() {
    let source = "+module LogSide\n+fn boom ()->Bool\n  +return LogSide.boom()\n+end\n+fn conj (flag:Bool)->Bool\n  +return flag AND boom()\n+end\n+fn disj (flag:Bool)->Bool\n  +return flag OR boom()\n+end\n";
    // AND false / OR true must skip the recursive rhs entirely in tree and VM.
    assert_eq!(
        text(&tree(&build(source), "LogSide.conj", vec![Value::Bool(false)]).unwrap()),
        "false"
    );
    assert_eq!(
        text(&vm_run(&build(source), "LogSide.conj", vec![Value::Bool(false)]).unwrap()),
        "false"
    );
    assert_eq!(
        text(&tree(&build(source), "LogSide.disj", vec![Value::Bool(true)]).unwrap()),
        "true"
    );
    assert_eq!(
        text(&vm_run(&build(source), "LogSide.disj", vec![Value::Bool(true)]).unwrap()),
        "true"
    );
}

/// Pure-value cases must agree in ALL THREE tiers, including the JIT.
#[test]
fn python_logical_pure_values_agree_across_tree_vm_and_jit() {
    let source = "+fn pure (a:Bool,b:Bool)->Bool\n  +return a AND b\n+end\n+fn pure_or (a:Bool,b:Bool)->Bool\n  +return a OR b\n+end\n";
    let prog = build(source);
    let mut jit = compiler::compile(&prog).unwrap();
    for (target, a, b, expected) in [
        ("pure", false, false, 0),
        ("pure", false, true, 0),
        ("pure", true, false, 0),
        ("pure", true, true, 1),
        ("pure_or", false, false, 0),
        ("pure_or", false, true, 1),
        ("pure_or", true, false, 1),
        ("pure_or", true, true, 1),
    ] {
        let args = vec![Value::Bool(a), Value::Bool(b)];
        assert!(
            matches!(tree(&prog, target, args.clone()).unwrap(), Value::Bool(v) if v == (expected == 1)),
            "{target} tree"
        );
        assert!(
            matches!(vm_run(&prog, target, args).unwrap(), Value::Bool(v) if v == (expected == 1)),
            "{target} vm"
        );
        assert_eq!(
            jit.call_i64(target, &[a as i64, b as i64]).unwrap(),
            expected,
            "{target} jit"
        );
    }
}

/// #90 contract matrix: truthiness is one definition, including the `.ax`-only
/// String/None/Err rows the Python frontend never produces (it only emits Bool).
#[test]
fn python_truthiness_matrix_is_shared_and_pinned() {
    for (value, expected) in [
        (Value::Bool(true), true),
        (Value::Bool(false), false),
        (Value::Int(0), false),
        (Value::Int(-3), true),
        (Value::string(""), false),
        (Value::string("x"), true),
        (Value::None, false),
        (Value::Err("boom".into()), false),
        (Value::list(vec![]), true),
        (Value::map(vec![]), true),
        (Value::set(vec![]), true),
    ] {
        assert_eq!(value.is_truthy(), expected, "{value}");
    }
    let prog = build("+fn neg (v:Bool)->Bool\n  +return NOT v\n+end\n");
    for (value, expected) in [(Value::Bool(true), false), (Value::Bool(false), true)] {
        assert!(
            matches!(tree(&prog, "neg", vec![value.clone()]).unwrap(), Value::Bool(b) if b == expected)
        );
        assert!(
            matches!(vm_run(&prog, "neg", vec![value]).unwrap(), Value::Bool(b) if b == expected)
        );
    }
}

/// Conditions (`if`) must agree with the same truthiness definition.
#[test]
fn python_if_condition_truthiness_agrees_tree_and_vm() {
    let prog =
        build("+fn choose (v:Bool)->Int\n  +if v\n    +return 1\n  +end\n  +return 0\n+end\n");
    for (value, expected) in [(Value::Bool(true), 1), (Value::Bool(false), 0)] {
        assert!(
            matches!(tree(&prog, "choose", vec![value.clone()]).unwrap(), Value::Int(n) if n == expected)
        );
        assert!(
            matches!(vm_run(&prog, "choose", vec![value]).unwrap(), Value::Int(n) if n == expected)
        );
    }
}
