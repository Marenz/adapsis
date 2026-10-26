//! #92: `i64::MIN / -1` and `i64::MIN % -1` must agree across tree, VM and JIT.
//! Rust's inherent operators overflow here, so the tiers share checked helpers:
//! division wraps to `i64::MIN`, remainder is `0`. Zero divisors are unchanged.
use crate::{ast, compiler, eval, parser, validator, vm};
use eval::Value;

fn program(source: &str) -> ast::Program {
    let mut program = ast::Program::default();
    for op in parser::parse(source).unwrap() {
        validator::apply_and_validate(&mut program, &op).unwrap();
    }
    program.rebuild_function_index();
    program
}

fn tiers(source: &str, target: &str, args: &[i64]) -> (i64, i64, i64) {
    let program = program(source);
    let function = program.get_function(target).unwrap();
    let values: Vec<Value> = args.iter().map(|n| Value::Int(*n)).collect();

    let mut env = eval::Env::new_with_shared_interner(&program.shared_interner);
    for (param, value) in function.params.iter().zip(values.iter()) {
        env.set(&param.name, value.clone());
    }
    let tree =
        match eval::eval_function_body_named(&program, target, &function.body, &mut env).unwrap() {
            Value::Int(n) => n,
            other => panic!("tree returned {other}"),
        };

    let compiled = vm::compile_function(function, &program).unwrap();
    let machine = vm::execute_with_io(&compiled, values, &program, &|_, _| {
        anyhow::bail!("no IO expected")
    })
    .unwrap();
    let vm_value = match machine {
        Value::Int(n) => n,
        other => panic!("vm returned {other}"),
    };

    // The public JIT path only reifies scalar returns and iterates top-level
    // functions, so exercise it through a top-level copy of the same body.
    let top = format!("{source}");
    let mut top_program = ast::Program::default();
    for op in parser::parse(&top).unwrap() {
        validator::apply_and_validate(&mut top_program, &op).unwrap();
    }
    top_program.rebuild_function_index();
    let jit = {
        let mut compiled = compiler::compile(&top_program).unwrap();
        compiled.call_i64(target, args).unwrap()
    };
    (tree, vm_value, jit)
}

const SOURCE: &str = "+fn divide (a:Int,b:Int)->Int\n  +return a / b\n+end\n+fn remainder (a:Int,b:Int)->Int\n  +return a % b\n+end\n";

#[test]
fn python_int_min_overflow_wraps_identically_in_all_tiers() {
    // i64::MIN / -1 overflows; all tiers wrap to i64::MIN.
    assert_eq!(
        tiers(SOURCE, "divide", &[i64::MIN, -1]),
        (i64::MIN, i64::MIN, i64::MIN)
    );
    // i64::MIN % -1 is 0 in every tier.
    assert_eq!(tiers(SOURCE, "remainder", &[i64::MIN, -1]), (0, 0, 0));
}

#[test]
fn python_int_min_overflow_is_reachable_from_python_frontend_operands() {
    // The Python frontend lowers `div`/`rem` to the same binary operators, so the
    // helper/operand path must agree too.
    let source = "def divide(a: int, b: int) -> int:\n    return div(a, b)\ndef remainder(a: int, b: int) -> int:\n    return rem(a, b)\n";
    let lowered = super::lower_module(source, "Overflow").unwrap();
    let mut program = ast::Program::default();
    for op in lowered.operations {
        validator::apply_and_validate(&mut program, &op).unwrap();
    }
    program.rebuild_function_index();
    for (target, expected) in [("Overflow.divide", i64::MIN), ("Overflow.remainder", 0)] {
        let function = program.get_function(target).unwrap();
        let mut env = eval::Env::new_with_shared_interner(&program.shared_interner);
        env.set("a", Value::Int(i64::MIN));
        env.set("b", Value::Int(-1));
        let value =
            eval::eval_function_body_named(&program, target, &function.body, &mut env).unwrap();
        assert!(
            matches!(value, Value::Int(n) if n == expected),
            "{target}: {value}"
        );
    }
}

#[test]
fn python_int_zero_divisor_still_errors_in_tree_and_vm() {
    for (target, message) in [
        ("divide", "division by zero"),
        ("remainder", "modulo by zero"),
    ] {
        // Tree
        let program = program(SOURCE);
        let function = program.get_function(target).unwrap();
        let mut env = eval::Env::new_with_shared_interner(&program.shared_interner);
        env.set("a", Value::Int(1));
        env.set("b", Value::Int(0));
        let error =
            eval::eval_function_body_named(&program, target, &function.body, &mut env).unwrap_err();
        assert!(error.to_string().contains(message), "{target}: {error}");
        // VM
        let compiled = vm::compile_function(function, &program).unwrap();
        let error = vm::execute_with_io(
            &compiled,
            vec![Value::Int(1), Value::Int(0)],
            &program,
            &|_, _| anyhow::bail!("no IO expected"),
        )
        .unwrap_err();
        assert!(error.to_string().contains(message), "{target} vm: {error}");
    }
}
