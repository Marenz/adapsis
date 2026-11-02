//! #98: heterogeneous equality shares tree semantics. Existing same-family and
//! Int/Float numeric behavior is deliberately unchanged pending #90/#86.
use super::{binary_cmp, pop_stack};
use crate::eval::Value;
use anyhow::Result;

pub(super) fn compare(stack: &mut Vec<Value>, negate: bool) -> Result<()> {
    let heterogeneous = match stack.as_slice() {
        [.., Value::Int(_), Value::Float(_)] | [.., Value::Float(_), Value::Int(_)] => false,
        [.., lhs, rhs] => std::mem::discriminant(lhs) != std::mem::discriminant(rhs),
        _ => false, // Preserve existing stack-underflow diagnostics.
    };
    if heterogeneous {
        let rhs = pop_stack(stack)?;
        let lhs = pop_stack(stack)?;
        let equal = lhs.matches(&rhs);
        stack.push(Value::Bool(if negate { !equal } else { equal }));
        return Ok(());
    }
    if negate {
        binary_cmp(stack, |a, b| a != b, |a, b| (a - b).abs() >= f64::EPSILON)
    } else {
        binary_cmp(stack, |a, b| a == b, |a, b| (a - b).abs() < f64::EPSILON)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ast, eval, parser, validator, vm};

    fn program(operator: &str) -> ast::Program {
        let mut program = ast::Program::default();
        for op in parser::parse(&format!(
            "+fn probe (a:Int,b:Int)->Bool\n  +return a {operator} b\n+end\n"
        ))
        .unwrap()
        {
            validator::apply_and_validate(&mut program, &op).unwrap();
        }
        program.rebuild_function_index();
        crate::intern::set_display_interner(&program.shared_interner);
        program
    }

    fn execute(
        program: &ast::Program,
        lhs: Value,
        rhs: Value,
    ) -> (anyhow::Result<Value>, anyhow::Result<Value>) {
        // Dynamic Value comparison contract: direct environments/VM args avoid
        // introducing a new frontend coercion or pretending annotations coerce.
        let function = program.get_function("probe").unwrap();
        let mut env = eval::Env::new_with_shared_interner(&program.shared_interner);
        env.set("a", lhs.clone());
        env.set("b", rhs.clone());
        let tree = eval::eval_declared_function(program, "probe", function, &mut env);
        let compiled = vm::compile_function(function, program).unwrap();
        let machine = vm::execute_with_io(&compiled, vec![lhs, rhs], program, &|_, _| {
            anyhow::bail!("comparison must not perform IO")
        });
        (tree, machine)
    }

    fn boolean(value: anyhow::Result<Value>) -> bool {
        match value.unwrap() {
            Value::Bool(value) => value,
            other => panic!("not Bool: {other}"),
        }
    }

    #[test]
    fn vm_heterogeneous_equality_matches_tree_in_both_orders() {
        for (operator, expected) in [("==", false), ("!=", true)] {
            let program = program(operator);
            for (left, right) in [
                (Value::Int(123), Value::string("")),
                (Value::Int(123), Value::string("123")),
                (Value::Float(1.25), Value::string("1.25")),
                (Value::Bool(true), Value::string("true")),
                (Value::None, Value::string("")),
                (Value::Ok(Box::new(Value::Int(7))), Value::string("7")),
                (Value::Err("error".into()), Value::string("error")),
                (
                    Value::Union {
                        variant: crate::intern::intern_display("Leaf"),
                        payload: vec![Value::Int(7)],
                    },
                    Value::string("Leaf"),
                ),
                (Value::Bool(true), Value::Int(1)),
            ] {
                for (lhs, rhs) in [(left.clone(), right.clone()), (right, left)] {
                    let (tree, machine) = execute(&program, lhs, rhs);
                    assert_eq!(boolean(tree), expected);
                    assert_eq!(boolean(machine), expected);
                }
            }
        }
    }

    #[test]
    fn vm_equality_same_scalar_controls_and_ordered_mixed_errors_are_preserved() {
        for (lhs, rhs, equal) in [
            (Value::Int(7), Value::Int(7), true),
            (Value::Int(7), Value::Int(8), false),
            (Value::Bool(true), Value::Bool(false), false),
            (Value::string("x"), Value::string("x"), true),
            (Value::string("x"), Value::string("y"), false),
            (Value::Float(0.0), Value::Float(1e-17), true),
        ] {
            for (op, expected) in [("==", equal), ("!=", !equal)] {
                let (tree, machine) = execute(&program(op), lhs.clone(), rhs.clone());
                assert_eq!(boolean(tree), expected);
                assert_eq!(boolean(machine), expected);
            }
        }
        for op in ["<", "<=", ">", ">="] {
            let (_, machine) = execute(&program(op), Value::Int(1), Value::string("2"));
            assert!(machine
                .unwrap_err()
                .to_string()
                .contains("comparison: unsupported types"));
        }
    }

    #[test]
    fn vm_equality_existing_numeric_and_aggregate_residuals_are_not_changed() {
        // Existing mixed-numeric coercion is NOT tree equality. Explicitly retain
        // and label it rather than widening #98 to all equality semantics.
        for (op, tree_expected, vm_expected) in [("==", false, true), ("!=", true, false)] {
            for (lhs, rhs) in [
                (Value::Int(1), Value::Float(1.0)),
                (Value::Float(1.0), Value::Int(1)),
            ] {
                let (tree, machine) = execute(&program(op), lhs, rhs);
                assert_eq!(boolean(tree), tree_expected);
                assert_eq!(boolean(machine), vm_expected);
            }
        }
        for value in [f64::NAN, f64::INFINITY] {
            let (tree, machine) = execute(&program("!="), Value::Float(value), Value::Float(value));
            assert!(boolean(tree));
            assert!(!boolean(machine));
            let (tree, machine) = execute(&program("=="), Value::Float(value), Value::Float(value));
            assert!(!boolean(tree));
            assert!(!boolean(machine));
        }
        for value in [
            Value::None,
            Value::Ok(Box::new(Value::Int(7))),
            Value::Err("x".into()),
            Value::Union {
                variant: crate::intern::intern_display("Leaf"),
                payload: vec![Value::Int(7)],
            },
        ] {
            for op in ["==", "!="] {
                let (_, machine) = execute(&program(op), value.clone(), value.clone());
                assert!(machine
                    .unwrap_err()
                    .to_string()
                    .contains("comparison: unsupported types"));
            }
        }
    }
}
