//! Direct compiled condition evidence; fallback evidence is explicitly separate.
use super::logical_tier_tests::{build, tree, vm_run};
use crate::{ast, compiler, eval, parser};
use eval::Value;

fn compare(source: &str, args: &[i64], expected: i64) {
    let program = build(source);
    let mut jit = compiler::compile(&program).unwrap();
    assert_eq!(jit.call_i64("probe", args).unwrap(), expected, "{source}");
    let values = args.iter().copied().map(Value::Int).collect::<Vec<_>>();
    assert!(
        matches!(tree(&program, "probe", values.clone()).unwrap(), Value::Int(n) if n == expected),
        "{source}"
    );
    assert!(
        matches!(vm_run(&program, "probe", values).unwrap(), Value::Int(n) if n == expected),
        "{source}"
    );
}

fn matrix() {
    for (operand, truth) in [
        ("0", false),
        ("2", true),
        ("-3", true),
        ("false", false),
        ("true", true),
        ("\"\"", false),
        ("\"ä\"", true),
        ("0.0", true),
        ("-0.0", true),
        ("-2.5", true),
    ] {
        for expression in [
            operand.to_owned(),
            format!("{operand} AND true"),
            format!("true AND {operand}"),
            format!("{operand} OR false"),
            format!("false OR {operand}"),
        ] {
            for control in ["if", "while"] {
                compare(&format!("+fn probe ()->Int\n  +{control} {expression}\n    +return 1\n  +end\n  +return 0\n+end\n"), &[], i64::from(truth));
            }
        }
        if truth {
            let program = build(&format!("+fn probe ()->Int [fail]\n  +check truth {operand} ~failed\n  +return 1\n+end\n"));
            // Check is now refused before JIT emission (#97). Preserve the operand
            // truthiness cases on both value-capable tiers, not a native trap path.
            assert!(compiler::compile(&program).is_err());
            assert!(matches!(tree(&program, "probe", vec![]).unwrap(), Value::Int(1)));
            assert!(matches!(vm_run(&program, "probe", vec![]).unwrap(), Value::Int(1)));
        }
    }
    for expression in [
        "2 > 1",
        "2 >= 2",
        "1 < 2",
        "2 <= 2",
        "2 == 2",
        "2 != 1",
        "1.5 < 2.5",
        "-0.0 <= 0.0",
        "true != false",
        "(2 > 1) AND (1 < 2)",
    ] {
        compare(
            &format!(
                "+fn probe ()->Int\n  +if {expression}\n    +return 1\n  +end\n  +return 0\n+end\n"
            ),
            &[],
            1,
        );
    }
    let mut program =
        build("+fn probe ()->Int\n  +if 0.0\n    +return 1\n  +end\n  +return 0\n+end\n");
    let ast::StatementKind::Branch { condition, .. } =
        &mut program.get_function_mut("probe").unwrap().body[0].kind
    else {
        panic!()
    };
    *condition = ast::Expr::Literal(ast::Literal::Float(f64::NAN));
    assert_eq!(
        compiler::compile(&program)
            .unwrap()
            .call_i64("probe", &[])
            .unwrap(),
        1
    );
    assert!(matches!(
        tree(&program, "probe", vec![]).unwrap(),
        Value::Int(1)
    ));
    assert!(matches!(
        vm_run(&program, "probe", vec![]).unwrap(),
        Value::Int(1)
    ));
}

fn collisions_and_calls() {
    for (dummy, argument, expected) in [("x", 0, 0), ("", 1, 1)] {
        for expression in [
            "_last_str_len AND true",
            "true AND _last_str_len",
            "_last_str_len OR false",
            "false OR _last_str_len",
            "_last_str_len > 0",
            "identity(identity(_last_str_len)) AND true",
        ] {
            compare(&format!("+fn identity (n:Int)->Int\n  +return n\n+end\n+fn probe (_last_str_len:Int)->Int\n  +let dummy:String = \"{dummy}\"\n  +if {expression}\n    +return 1\n  +end\n  +return 0\n+end\n"), &[argument], expected);
        }
        compare(&format!("+fn probe (_last_str_len:Int)->Int\n  +while _last_str_len\n    +let dummy:String = \"{dummy}\"\n    +if _last_str_len\n      +return 1\n    +end\n    +return 0\n  +end\n  +return 0\n+end\n"), &[argument], expected);
        for expression in [
            "_last_str AND true",
            "true AND _last_str",
            "_last_str OR false",
            "false OR _last_str",
        ] {
            let text = if argument == 0 { "" } else { "x" };
            compare(&format!("+fn child (_last_str:String)->Int\n  +let dummy:String = \"{dummy}\"\n  +if {expression}\n    +return 1\n  +end\n  +return 0\n+end\n+fn probe ()->Int\n  +return child(\"{text}\")\n+end\n"), &[], expected);
        }
    }
    // Preserve authored argument order through nested scalar call emission.
    compare("+fn identity (n:Int)->Int\n  +return n\n+end\n+fn pair (a:Int,b:Int)->Int\n  +return a*10+b\n+end\n+fn probe ()->Int\n  +if pair(identity(1), identity(2))==12\n    +return 1\n  +end\n  +return 0\n+end\n", &[], 1);
}

fn refusals_and_fallback() {
    for (declaration, expression, diagnostic) in [
        (
            "+fn callee (n:Int)->Int\n  +return n\n+end\n",
            "callee()",
            "arity",
        ),
        (
            "+fn callee (n:Float)->Int\n  +return 1\n+end\n",
            "callee(1)",
            "exact scalar types",
        ),
        (
            "+fn callee (n:String)->Int\n  +return 1\n+end\n",
            "callee(\"x\")",
            "exact scalar types",
        ),
        (
            "+fn callee ()->Int [async]\n  +return 1\n+end\n",
            "callee()",
            "IO/async",
        ),
        (
            "+fn callee ()->Int [io]\n  +return 1\n+end\n",
            "callee()",
            "IO/async",
        ),
        ("+fn callee (n:Int)->Int [fail]\n  +return n\n+end\n", "callee(1)", "jit: reachable Check"),
        ("+fn raises ()->Int [fail]\n  +return 1\n+end\n+fn callee (n:Int)->Int\n  +return n\n+end\n", "callee(raises())", "jit: reachable Check"),
        ("+fn raises ()->Bool [fail]\n  +return true\n+end\n+fn callee (n:Bool)->Int\n  +return 1\n+end\n", "callee(NOT raises())", "jit: reachable Check"),
        ("", "\"a\"==\"a\"", "homogeneous scalar comparison"),
    ] {
        // Even a short-circuited unsupported RHS must fail before execution.
        let program = build(&format!("{declaration}+fn probe ()->Int\n  +if false AND ({expression})\n    +return 1\n  +end\n  +return 0\n+end\n"));
        let error = match compiler::compile(&program) {
            Ok(_) => panic!("must refuse {expression}"),
            Err(e) => format!("{e:#}"),
        };
        assert!(error.contains(diagnostic), "{expression}: {error}");
    }
    let program = build(
        "+fn probe (n:Int)->Int\n  +while n>0\n    +set n = n-1\n  +end\n  +return n\n+end\n",
    );
    assert!(compiler::compile(&program).is_err());
    let input = parser::parse_single_expr("{n: 3}").unwrap();
    let (result, used_jit) = eval::eval_compiled_or_interpreted(&program, "probe", &input).unwrap();
    assert_eq!(result, "0");
    assert!(
        !used_jit,
        "mutable loop must not use its immutable entry snapshot"
    );
    assert!(matches!(
        tree(&program, "probe", vec![Value::Int(3)]).unwrap(),
        Value::Int(0)
    ));
    assert!(matches!(
        vm_run(&program, "probe", vec![Value::Int(3)]).unwrap(),
        Value::Int(0)
    ));
}

#[test]
fn python_jit_condition_normalization_matrix() {
    const CHILD: &str = "ADAPSIS_CONDITION_MATRIX_CHILD";
    if std::env::var_os(CHILD).is_none() {
        // #96 remains open: isolate direct nested calls from concurrent JIT
        // allocations, using this exact executable and a hard timeout.
        let output = std::process::Command::new("timeout")
            .args(["--signal=KILL", "60s"])
            .arg(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "python_frontend::condition_tier_tests::python_jit_condition_normalization_matrix",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("CONDITION_MATRIX_COMPLETE"),
            "{output:?}"
        );
        return;
    }
    matrix();
    collisions_and_calls();
    refusals_and_fallback();
    println!("CONDITION_MATRIX_COMPLETE");
}

#[test]
fn python_jit_float_equality_refuses_including_nested_conditions() {
    for op in ["==", "!="] {
        for operands in [
            ("0.0", "0.00000000000000001"),
            ("-0.0", "0.0"),
            ("infinite()", "infinite()"),
        ] {
            for condition in [
                format!("{} {op} {}", operands.0, operands.1),
                format!("false AND ({} {op} {})", operands.0, operands.1),
            ] {
                let program = build(&format!("+fn infinite ()->Float\n  +return 1.0/0.0\n+end\n+fn probe ()->Int\n  +if {condition}\n    +return 1\n  +end\n  +return 0\n+end\n"));
                let error = match compiler::compile(&program) {
                    Ok(_) => panic!("must refuse {condition}"),
                    Err(e) => format!("{e:#}"),
                };
                assert!(error.contains("Float equality is unsupported"), "{error}");
            }
        }
        let mut program = build(&format!(
            "+fn probe ()->Int\n  +if 0.0 {op} 0.0\n    +return 1\n  +end\n  +return 0\n+end\n"
        ));
        let ast::StatementKind::Branch { condition, .. } =
            &mut program.get_function_mut("probe").unwrap().body[0].kind
        else {
            panic!()
        };
        let ast::Expr::Binary { left, .. } = condition else {
            panic!()
        };
        **left = ast::Expr::Literal(ast::Literal::Float(f64::NAN));
        let error = match compiler::compile(&program) {
            Ok(_) => panic!("must refuse NaN equality"),
            Err(e) => format!("{e:#}"),
        };
        assert!(error.contains("Float equality is unsupported"), "{error}");
        // Deliberately do not claim VM fallback fixes unordered Neq: VM and
        // tree differ there already. Only the finite fallback below is certified.
    }
}

#[test]
fn python_jit_float_equality_finite_fallback_preserves_epsilon_rule() {
    for (op, expected) in [("==", 1), ("!=", 0)] {
        let program = build(&format!(
            "+fn probe ()->Int\n  +if 0.0 {op} 0.00000000000000001\n    +return 1\n  +end\n  +return 0\n+end\n"
        ));
        assert!(compiler::compile(&program).is_err());
        let input = parser::parse_single_expr("{}").unwrap();
        let (actual, used_jit) =
            eval::eval_compiled_or_interpreted(&program, "probe", &input).unwrap();
        assert!(!used_jit);
        assert_eq!(actual, expected.to_string());
        assert!(matches!(tree(&program, "probe", vec![]).unwrap(), Value::Int(n) if n == expected));
        assert!(
            matches!(vm_run(&program, "probe", vec![]).unwrap(), Value::Int(n) if n == expected)
        );
    }
}

#[test]
fn python_jit_condition_failure_and_short_circuit_paths() {
    let cases = [
        ("+check required 0 ~failed\n  +return 1", None),
        ("+check required \"\" ~failed\n  +return 1", None),
        ("+if fail_float()\n    +return 1\n  +end\n  +return 0", None),
        (
            "+while fail_float()\n    +return 1\n  +end\n  +return 0",
            None,
        ),
        ("+check required fail_float() ~failed\n  +return 1", None),
        (
            "+if true OR fail_float()\n    +return 1\n  +end\n  +return 0",
            Some(1),
        ),
        (
            "+if false AND fail_float()\n    +return 1\n  +end\n  +return 0",
            Some(0),
        ),
    ];
    for (index, (body, expected)) in cases.into_iter().enumerate() {
        let program = build(&format!("+fn fail_float ()->Float [fail]\n  +check required false ~failed\n  +return 0.0\n+end\n+fn probe ()->Int [fail]\n  {body}\n+end\n"));
        if let Some(expected) = expected {
            assert!(
                matches!(tree(&program, "probe", vec![]).unwrap(), Value::Int(n) if n == expected)
            );
            assert!(
                matches!(vm_run(&program, "probe", vec![]).unwrap(), Value::Int(n) if n == expected)
            );
        } else {
            assert!(tree(&program, "probe", vec![]).is_err());
            let vm_result = vm_run(&program, "probe", vec![]).unwrap();
            if index == 2 || index == 3 {
                // Existing VM failure contract: Check returns Err as a value
                // from the CALLEE; ordinary caller control flow consumes that
                // falsy value and reaches its following return. Unlike tree,
                // failure does not abort the caller. This gap remains OPEN.
                assert!(matches!(vm_result, Value::Int(0)), "{vm_result}");
            } else {
                // Direct/outer failed Check terminates before return1.
                assert!(
                    matches!(vm_result, Value::Err(ref e) if e == "failed"),
                    "{vm_result}"
                );
            }
        }
        // Historical native Check traps are no longer reachable through compile().
        // Refuse the complete graph, including short-circuited fallible branches.
        let Err(error) = compiler::compile(&program) else { panic!("unsafe JIT graph admitted") };
        assert!(error.to_string().contains("jit: reachable Check"), "{error}");
    }
}

/// Trusted .ax async fixture, not Python admission or JIT IO support. A VM
/// failure after the first actual IoRequest must not retry through tree-walking.
#[test]
fn trusted_ax_condition_failure_after_io_is_not_replayed() {
    use crate::coroutine::{CoroutineHandle, IoRequest};
    let program = build("+fn probe ()->Int [io,async]\n  +await exists:Bool = file_exists(\"fixture\")\n  +if exists AND (1/0>0)\n    +return 1\n  +end\n  +return 0\n+end\n");
    let (sender, mut receiver) = tokio::sync::mpsc::channel(4);
    let responder = std::thread::spawn(move || {
        let mut requests = 0;
        while let Some(request) = receiver.blocking_recv() {
            let IoRequest::FileExists { reply, .. } = request else {
                panic!("only trusted fixture IO")
            };
            requests += 1;
            reply.send(Ok(true)).unwrap();
        }
        requests
    });
    let input = parser::parse_single_expr("{}").unwrap();
    let error = eval::eval_async_function(&program, "probe", &input, CoroutineHandle::new(sender))
        .unwrap_err();
    assert!(error.to_string().contains("division by zero"), "{error:#}");
    assert_eq!(
        responder.join().unwrap(),
        1,
        "completed IO must not be replayed"
    );
}

/// Existing failure-propagation discrepancy, NOT equivalence evidence. Production
/// async dispatch selects the VM, whose ordinary caller continues after a failed
/// Check returned as an Err value. Tree aborts before the subsequent IO effect.
#[test]
fn trusted_ax_failed_check_caller_effect_characterization() {
    use crate::coroutine::{CoroutineHandle, IoRequest};
    let program = build("+fn fail_bool ()->Bool [fail]\n  +check required false ~failed\n  +return true\n+end\n+fn probe ()->Int [io,async]\n  +let answer:Bool = fail_bool()\n  +await after:Bool = file_exists(\"after-failed-callee\")\n  +return 0\n+end\n");
    for normal_dispatch in [false, true] {
        let (sender, mut receiver) = tokio::sync::mpsc::channel(4);
        let responder = std::thread::spawn(move || {
            let mut requests = 0;
            while let Some(request) = receiver.blocking_recv() {
                let IoRequest::FileExists { reply, .. } = request else {
                    panic!("only trusted fixture IO")
                };
                requests += 1;
                reply.send(Ok(true)).unwrap();
            }
            requests
        });
        let handle = CoroutineHandle::new(sender);
        if normal_dispatch {
            let input = parser::parse_single_expr("{}").unwrap();
            assert!(matches!(
                eval::eval_async_function(&program, "probe", &input, handle).unwrap(),
                Value::Int(0)
            ));
        } else {
            let mut env = eval::Env::new_with_shared_interner(&program.shared_interner);
            env.set("__coroutine_handle", Value::CoroutineHandle(handle));
            let body = &program.get_function("probe").unwrap().body;
            assert!(eval::eval_function_body_named(&program, "probe", body, &mut env).is_err());
        }
        assert_eq!(responder.join().unwrap(), usize::from(normal_dispatch));
    }
}
