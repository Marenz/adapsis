//! #90: `and`/`or` must short-circuit identically in tree, VM and JIT, on one
//! truthiness definition. Tree short-circuit is normative; these tests are the
//! contract that keeps the three tiers from drifting again.
use crate::{ast, compiler, coroutine, eval, parser, validator, vm};
use eval::Value;

pub(super) fn build(source: &str) -> ast::Program {
    let mut prog = ast::Program::default();
    for op in parser::parse(source).unwrap() {
        validator::apply_and_validate(&mut prog, &op).unwrap();
    }
    prog.rebuild_function_index();
    crate::intern::set_display_interner(&prog.shared_interner);
    prog
}

pub(super) fn tree(prog: &ast::Program, target: &str, args: Vec<Value>) -> anyhow::Result<Value> {
    let function = prog.get_function(target).unwrap();
    let mut env = eval::Env::new_with_shared_interner(&prog.shared_interner);
    for (param, arg) in function.params.iter().zip(args) {
        env.set(&param.name, arg);
    }
    eval::eval_function_body_named(prog, target, &function.body, &mut env)
}

pub(super) fn vm_run(prog: &ast::Program, target: &str, args: Vec<Value>) -> anyhow::Result<Value> {
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
    assert!(tree(
        &build(source),
        "LogErr.conj",
        vec![Value::Bool(true), Value::Int(0)]
    )
    .is_err());
    assert!(vm_run(
        &build(source),
        "LogErr.conj",
        vec![Value::Bool(true), Value::Int(0)]
    )
    .is_err());
    assert!(tree(
        &build(source),
        "LogErr.disj",
        vec![Value::Bool(false), Value::Int(0)]
    )
    .is_err());
    assert!(vm_run(
        &build(source),
        "LogErr.disj",
        vec![Value::Bool(false), Value::Int(0)]
    )
    .is_err());
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
    let source = "+fn pure (a:Bool,b:Bool)->Int\n  +if a AND b\n    +return 1\n  +end\n  +return 0\n+end\n+fn pure_or (a:Bool,b:Bool)->Int\n  +if a OR b\n    +return 1\n  +end\n  +return 0\n+end\n";
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
            matches!(tree(&prog, target, args.clone()).unwrap(), Value::Int(v) if v == expected),
            "{target} tree"
        );
        assert!(
            matches!(vm_run(&prog, target, args).unwrap(), Value::Int(v) if v == expected),
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

// ── #90 item (1): frontend call operands, verified by short-circuit probes ──

fn py_program(source: &str, module: &str) -> ast::Program {
    let lowered = super::lower_module(source, module).unwrap_or_else(|e| panic!("{e}"));
    let mut prog = ast::Program::default();
    for op in lowered.operations {
        validator::apply_and_validate(&mut prog, &op).unwrap();
    }
    prog.rebuild_function_index();
    crate::intern::set_display_interner(&prog.shared_interner);
    prog
}

/// Python-source `and`/`or` with a CALL operand: the skipped call must not run,
/// proven by recursion (an evaluated call aborts).
#[test]
fn python_frontend_logical_call_operand_short_circuits_tree_and_vm() {
    let source = "def boom() -> bool:\n    return boom()\ndef conj(flag: bool) -> bool:\n    return flag and boom()\ndef disj(flag: bool) -> bool:\n    return flag or boom()\n";
    let prog = py_program(source, "PyLogical");
    for (target, flag, expected) in [
        ("PyLogical.conj", false, "false"),
        ("PyLogical.disj", true, "true"),
    ] {
        let args = vec![Value::Bool(flag)];
        assert_eq!(
            text(&tree(&prog, target, args.clone()).unwrap()),
            expected,
            "{target} tree"
        );
        assert_eq!(
            text(&vm_run(&prog, target, args).unwrap()),
            expected,
            "{target} vm"
        );
    }
    // Non-short-circuit path does evaluate the recursive call.
    assert!(tree(&prog, "PyLogical.conj", vec![Value::Bool(true)]).is_err());
}

/// Python-source call operand whose rhs would divide by zero.
#[test]
fn python_frontend_logical_call_operand_skips_division_error() {
    let source = "def risky(n: int) -> bool:\n    return div(1, n) > 0\ndef conj(flag: bool, n: int) -> bool:\n    return flag and risky(n)\ndef disj(flag: bool, n: int) -> bool:\n    return flag or risky(n)\n";
    let prog = py_program(source, "PyRisky");
    for (target, flag) in [("PyRisky.conj", false), ("PyRisky.disj", true)] {
        let args = vec![Value::Bool(flag), Value::Int(0)];
        assert!(matches!(tree(&prog, target, args.clone()).unwrap(), Value::Bool(b) if b == flag));
        assert!(matches!(vm_run(&prog, target, args).unwrap(), Value::Bool(b) if b == flag));
    }
}

/// Python-source negation and if-condition carry call operands.
#[test]
fn python_frontend_not_and_if_condition_accept_calls() {
    let source = "def yes() -> bool:\n    return True\ndef neg() -> bool:\n    return not yes()\ndef choose(flag: bool) -> int:\n    if flag and yes():\n        return 1\n    return 0\n";
    let prog = py_program(source, "PyNotIf");
    assert_eq!(text(&tree(&prog, "PyNotIf.neg", vec![]).unwrap()), "false");
    assert_eq!(
        text(&vm_run(&prog, "PyNotIf.neg", vec![]).unwrap()),
        "false"
    );
    for (flag, expected) in [(true, 1), (false, 0)] {
        let args = vec![Value::Bool(flag)];
        assert!(
            matches!(tree(&prog, "PyNotIf.choose", args.clone()).unwrap(), Value::Int(n) if n == expected)
        );
        assert!(
            matches!(vm_run(&prog, "PyNotIf.choose", args).unwrap(), Value::Int(n) if n == expected)
        );
    }
}

/// Legacy .ax evidence only: short-circuited IO calls consume zero requests,
/// taken calls one. Python cannot express this implicit async call in a Boolean
/// operand under the current await policy; this is not Python-source IO coverage.
#[test]
fn ax_logical_call_operand_consumes_no_io_when_short_circuited() {
    use coroutine::{CoroutineHandle, IoRequest};
    let ax = "+module PyIoProbe\n+fn check (path:String)->Bool [io,async]\n  +await value:Bool = file_exists(path)\n  +return value\n+end\n+fn conj (flag:Bool,path:String)->Bool [io,async]\n  +return flag AND check(path)\n+end\n+fn disj (flag:Bool,path:String)->Bool [io,async]\n  +return flag OR check(path)\n+end\n";
    for (target, flag, expected, count) in [
        ("PyIoProbe.conj", false, false, 0),
        ("PyIoProbe.disj", true, true, 0),
        ("PyIoProbe.conj", true, true, 1),
        ("PyIoProbe.disj", false, true, 1),
    ] {
        let prog = build(ax);
        let (sender, mut receiver) = tokio::sync::mpsc::channel(2);
        let responder = std::thread::spawn(move || {
            let mut calls = 0;
            while let Some(request) = receiver.blocking_recv() {
                let IoRequest::FileExists { reply, .. } = request else {
                    panic!("only fixture")
                };
                calls += 1;
                reply.send(Ok(true)).unwrap();
            }
            calls
        });
        let mut env = eval::Env::new_with_shared_interner(&prog.shared_interner);
        env.set(
            "__coroutine_handle",
            Value::CoroutineHandle(CoroutineHandle::new(sender)),
        );
        let function = prog.get_function(target).unwrap();
        env.set("flag", Value::Bool(flag));
        env.set("path", Value::string("fixture"));
        let result =
            eval::eval_function_body_named(&prog, target, &function.body, &mut env).unwrap();
        drop(env);
        assert!(
            matches!(result, Value::Bool(b) if b == expected),
            "{target}/{flag}"
        );
        assert_eq!(
            responder.join().unwrap(),
            count,
            "{target}/{flag} tree mock count"
        );

        let compiled = vm::compile_function(function, &prog).unwrap();
        let seen = std::cell::RefCell::new(0);
        let result = vm::execute_with_io(
            &compiled,
            vec![Value::Bool(flag), Value::string("fixture")],
            &prog,
            &|_, _| {
                *seen.borrow_mut() += 1;
                Ok(Value::Bool(true))
            },
        )
        .unwrap();
        assert!(
            matches!(result, Value::Bool(b) if b == expected),
            "{target}/{flag} vm"
        );
        assert_eq!(*seen.borrow(), count, "{target}/{flag} vm mock count");
    }
}

/// #90 item (2): JIT `NOT` must follow truthiness, not boolean xor. `NOT 5` is
/// false (5 is truthy); `NOT 0` is true; Bool cases unchanged. The JIT's
/// `call_i64` cannot reify an I8 Bool return, so the NOT result is consumed by an
/// Int-returning wrapper.
#[test]
fn python_jit_not_uses_truthiness_not_xor() {
    let source = "+fn neg_int (n:Int)->Bool\n  +return NOT n\n+end\n+fn neg_bool (b:Bool)->Bool\n  +return NOT b\n+end\n+fn neg_int_as_int (n:Int)->Int\n  +if NOT n\n    +return 1\n  +end\n  +return 0\n+end\n+fn neg_bool_as_int (b:Bool)->Int\n  +if NOT b\n    +return 1\n  +end\n  +return 0\n+end\n";
    let prog = build(source);
    let mut jit = compiler::compile(&prog).unwrap();
    for (target, arg, expected) in [
        ("neg_int_as_int", 5i64, 0i64),
        ("neg_int_as_int", 0, 1),
        ("neg_int_as_int", -3, 0),
        ("neg_bool_as_int", 1, 0),
        ("neg_bool_as_int", 0, 1),
    ] {
        assert_eq!(
            jit.call_i64(target, &[arg]).unwrap(),
            expected,
            "{target}({arg}) jit"
        );
    }
    // Cross-tier agreement for the same wrappers.
    for (target, arg, expected) in [
        ("neg_int_as_int", Value::Int(5), 0),
        ("neg_int_as_int", Value::Int(0), 1),
        ("neg_int_as_int", Value::Int(-3), 0),
        ("neg_bool_as_int", Value::Bool(true), 0),
        ("neg_bool_as_int", Value::Bool(false), 1),
    ] {
        assert!(
            matches!(tree(&prog, target, vec![arg.clone()]).unwrap(), Value::Int(n) if n == expected),
            "{target} tree"
        );
        assert!(
            matches!(vm_run(&prog, target, vec![arg]).unwrap(), Value::Int(n) if n == expected),
            "{target} vm"
        );
    }
}

#[test]
fn python_jit_not_string_float_literal_representations() {
    for (operand, expected) in [
        ("\"\"", 1),
        ("\"x\"", 0),
        ("\"ä\"", 0),
        ("0.0", 0),
        ("-0.0", 0),
        ("1.5", 0),
        ("-2.5", 0),
    ] {
        let prog = build(&format!(
            "+fn probe ()->Int\n  +if NOT {operand}\n    +return 1\n  +end\n  +return 0\n+end\n"
        ));
        let mut jit = compiler::compile(&prog).unwrap();
        assert_eq!(
            jit.call_i64("probe", &[]).unwrap(),
            expected,
            "JIT NOT {operand}"
        );
        assert!(matches!(tree(&prog, "probe", vec![]).unwrap(), Value::Int(n) if n == expected));
        assert!(matches!(vm_run(&prog, "probe", vec![]).unwrap(), Value::Int(n) if n == expected));
    }
    // NaN has no source literal. Inject a real Float AST node, then run that
    // identical AST in all tiers (no fallback or Bool-return ABI coercion).
    let mut prog =
        build("+fn probe ()->Int\n  +if NOT 0.0\n    +return 1\n  +end\n  +return 0\n+end\n");
    let ast::StatementKind::Branch { condition, .. } =
        &mut prog.get_function_mut("probe").unwrap().body[0].kind
    else {
        panic!()
    };
    let ast::Expr::Unary { expr, .. } = condition else {
        panic!()
    };
    **expr = ast::Expr::Literal(ast::Literal::Float(f64::NAN));
    assert_eq!(
        compiler::compile(&prog)
            .unwrap()
            .call_i64("probe", &[])
            .unwrap(),
        0
    );
    assert!(matches!(
        tree(&prog, "probe", vec![]).unwrap(),
        Value::Int(0)
    ));
    assert!(matches!(
        vm_run(&prog, "probe", vec![]).unwrap(),
        Value::Int(0)
    ));
}

#[test]
fn python_jit_not_string_parameter_uses_length_not_pointer() {
    let prog =
        build("+fn probe (s:String)->Int\n  +if NOT s\n    +return 1\n  +end\n  +return 0\n+end\n");
    let mut jit = compiler::compile(&prog).unwrap();
    for (s, expected) in [("", 1), ("x", 0), ("ä", 0)] {
        // Existing ABI: String = pointer + byte length; result is genuinely I64.
        assert_eq!(
            jit.call_i64("probe", &[s.as_ptr() as i64, s.len() as i64])
                .unwrap(),
            expected
        );
        assert!(
            matches!(tree(&prog, "probe", vec![Value::string(s)]).unwrap(), Value::Int(n) if n == expected)
        );
        assert!(
            matches!(vm_run(&prog, "probe", vec![Value::string(s)]).unwrap(), Value::Int(n) if n == expected)
        );
    }
}

#[test]
fn python_jit_not_unproven_string_representations_refuse() {
    for source in [
        "+fn probe ()->Int\n  +let s:String = \"x\"\n  +if NOT s\n    +return 1\n  +end\n  +return 0\n+end\n",
        "+fn identity (s:String)->String\n  +return s\n+end\n+fn probe ()->Int\n  +if NOT identity(\"x\")\n    +return 1\n  +end\n  +return 0\n+end\n",
        "+fn probe (s:String,s_len:Int)->Int\n  +if NOT s\n    +return 1\n  +end\n  +return 0\n+end\n",
        "+fn probe (s:String)->Int\n  +set s = \"x\"\n  +if NOT s\n    +return 1\n  +end\n  +return 0\n+end\n",
        "+fn probe (s:String)->Int\n  +let alias:String = s\n  +if NOT alias\n    +return 1\n  +end\n  +return 0\n+end\n",
    ] {
        assert!(compiler::compile(&build(source)).is_err(), "must refuse unproven representation");
    }
}

#[test]
fn python_jit_not_string_parameter_survives_scratch_name_collision() {
    const CHILD: &str = "ADAPSIS_NOT_STRING_COLLISION_CHILD";
    if std::env::var_os(CHILD).is_none() {
        // Concurrent JIT allocations in the full suite can put PC-relative
        // call relocations out of range. Keep this real nested-call regression
        // in its own process; compilation/runtime failures must still fail it.
        let output = std::process::Command::new("timeout")
            .args(["--signal=KILL", "60s"])
            .arg(std::env::current_exe().unwrap())
            .args(["--exact", "python_frontend::logical_tier_tests::python_jit_not_string_parameter_survives_scratch_name_collision", "--nocapture"])
            .env(CHILD, "1").output().unwrap();
        assert!(output.status.success(), "{output:?}");
        assert!(
            String::from_utf8_lossy(&output.stdout)
                .contains("STRING_COLLISION_BOTH_DIRECTIONS_PASSED"),
            "{output:?}"
        );
        return;
    }
    for (dummy, argument, expected) in [("x", "", 1), ("", "x", 0)] {
        let source = format!(
            "+fn probe (_last_str:String)->Int\n  +let dummy:String = \"{dummy}\"\n  +if NOT _last_str\n    +return 1\n  +end\n  +return 0\n+end\n\
             +fn wrapper ()->Int\n  +return probe(\"{argument}\")\n+end\n"
        );
        let prog = build(&source);
        let mut jit = compiler::compile(&prog).unwrap();
        // Both entry and result are actual Int ABI calls; the nested compiled
        // call itself supplies the pointer/length pair to probe.
        assert_eq!(jit.call_i64("wrapper", &[]).unwrap(), expected);
        assert!(matches!(tree(&prog, "wrapper", vec![]).unwrap(), Value::Int(n) if n == expected));
        assert!(
            matches!(vm_run(&prog, "wrapper", vec![]).unwrap(), Value::Int(n) if n == expected)
        );
    }
    println!("STRING_COLLISION_BOTH_DIRECTIONS_PASSED");
}

#[test]
fn python_jit_not_float_call_preserves_operand_failure() {
    let prog = build("+fn fail_float ()->Float [fail]\n  +check required false ~failed\n  +return 0.0\n+end\n+fn probe ()->Int\n  +if NOT fail_float()\n    +return 1\n  +end\n  +return 0\n+end\n");
    let Err(error) = compiler::compile(&prog) else { panic!("fallible JIT graph admitted") };
    assert!(error.to_string().contains("jit: reachable Check"), "{error}");
    assert!(tree(&prog, "probe", vec![]).is_err());
    // #97 VM unwind is still pending: Err is consumed as falsy, then NOT makes it true.
    assert!(matches!(vm_run(&prog, "probe", vec![]).unwrap(), Value::Int(1)));
}

#[test]
fn python_jit_not_scalar_parameter_identity_survives_recursive_scratch_collision() {
    for (dummy, argument) in [("x", 0), ("", 1)] {
        for (operand, expected) in [
            ("_last_str_len", i64::from(argument == 0)),
            ("NOT _last_str_len", i64::from(argument != 0)),
            ("-_last_str_len", i64::from(argument == 0)),
            ("NOT NOT _last_str_len", i64::from(argument == 0)),
        ] {
            let prog = build(&format!(
                "+fn probe (_last_str_len:Int)->Int\n  +let dummy:String = \"{dummy}\"\n  +if NOT ({operand})\n    +return 1\n  +end\n  +return 0\n+end\n"
            ));
            let mut jit = compiler::compile(&prog).unwrap();
            assert_eq!(
                jit.call_i64("probe", &[argument]).unwrap(),
                expected,
                "{dummy:?}/{argument}/{operand}"
            );
            assert!(
                matches!(tree(&prog, "probe", vec![Value::Int(argument)]).unwrap(), Value::Int(n) if n == expected)
            );
            assert!(
                matches!(vm_run(&prog, "probe", vec![Value::Int(argument)]).unwrap(), Value::Int(n) if n == expected)
            );
        }
    }
}

#[test]
fn python_jit_not_unproven_compound_operands_refuse_before_execution() {
    for (operand, diagnostic, expected) in [
        ("_last_str_len + 0", "binary operands", 1),
        ("NOT (_last_str_len + 0)", "binary operands", 0),
        ("identity(_last_str_len)", "argument-bearing calls", 1),
        ("NOT identity(_last_str_len)", "argument-bearing calls", 0),
    ] {
        let prog = build(&format!(
            "+fn identity (n:Int)->Int\n  +return n\n+end\n+fn probe (_last_str_len:Int)->Int\n  +let dummy:String = \"x\"\n  +if NOT ({operand})\n    +return 1\n  +end\n  +return 0\n+end\n"
        ));
        let error = match compiler::compile(&prog) {
            Ok(_) => panic!("unproven operand must refuse: {operand}"),
            Err(error) => format!("{error:#}"),
        };
        assert!(error.contains(diagnostic), "{error}");
        assert!(
            matches!(tree(&prog, "probe", vec![Value::Int(0)]).unwrap(), Value::Int(n) if n == expected)
        );
        assert!(
            matches!(vm_run(&prog, "probe", vec![Value::Int(0)]).unwrap(), Value::Int(n) if n == expected)
        );
    }
}
