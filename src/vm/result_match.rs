//! #93: dedicated Result values participate in the existing match stack ABI.
//! BranchIfNot consumes the final Bool; the compiler then stores payloads in
//! reverse binding order. Valid Result arms bind zero or one payload, including
//! the discard binding `_`. Union payload order/count behavior is unchanged.
use crate::{
    eval::Value,
    intern::{InternedId, StringInterner},
};

pub(super) fn push_match(
    value: &Value,
    variant_id: InternedId,
    binding_count: usize,
    interner: &StringInterner,
    stack: &mut Vec<Value>,
) {
    let matched = match value {
        Value::Union { variant, payload } if *variant == variant_id => {
            for value in payload.iter().take(binding_count) {
                stack.push(value.clone());
            }
            true
        }
        Value::Ok(inner) if interner.resolve(variant_id) == Some("Ok") => {
            if binding_count > 0 {
                stack.push(inner.as_ref().clone());
            }
            true
        }
        Value::Err(message) if interner.resolve(variant_id) == Some("Err") => {
            if binding_count > 0 {
                stack.push(Value::string(message.clone()));
            }
            true
        }
        _ => false,
    };
    stack.push(Value::Bool(matched));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ast, eval, parser, validator, vm};

    fn program(source: &str) -> ast::Program {
        let mut program = ast::Program::default();
        for op in parser::parse(source).unwrap() {
            validator::apply_and_validate(&mut program, &op).unwrap();
        }
        program.rebuild_function_index();
        crate::intern::set_display_interner(&program.shared_interner);
        program
    }

    fn agree(program: &ast::Program, target: &str, argument: Value, expected: &str) {
        let function = program.get_function(target).unwrap();
        let mut env = eval::Env::new_with_shared_interner(&program.shared_interner);
        env.set(&function.params[0].name, argument.clone());
        let tree =
            eval::eval_function_body_named(program, target, &function.body, &mut env).unwrap();
        let compiled = vm::compile_function(function, program).unwrap();
        let vm = vm::execute_with_io(&compiled, vec![argument], program, &|_, _| {
            anyhow::bail!("no IO in matching fixture")
        })
        .unwrap();
        assert_eq!(tree.to_string(), expected);
        assert_eq!(vm.to_string(), expected);
    }

    #[test]
    fn dedicated_result_tag_only_and_payload_discard_match_tree() {
        let program = program(
            "+module ResultTags\n+fn tag (r:Result<Int>)->Int\n +match r\n +case Ok\n  +return 7\n +case Err\n  +return 8\n +end\n+end\n+fn discard (r:Result<Int>)->Int\n +match r\n +case Ok(_)\n  +return 7\n +case Err(_)\n  +return 8\n +end\n+end\n+fn fallback (r:Result<Int>)->Int\n +match r\n +case _\n  +return 9\n +end\n+end\n",
        );
        for name in ["tag", "discard"] {
            agree(
                &program,
                &format!("ResultTags.{name}"),
                Value::Ok(Box::new(Value::Int(5))),
                "7",
            );
            agree(
                &program,
                &format!("ResultTags.{name}"),
                Value::Err("fixture".into()),
                "8",
            );
        }
        for value in [
            Value::Ok(Box::new(Value::Int(5))),
            Value::Err("fixture".into()),
        ] {
            agree(&program, "ResultTags.fallback", value, "9");
        }
    }

    #[test]
    fn nested_match_statements_bind_inner_result_and_error_string() {
        // Nested matches are supported; this does not certify ignored nested
        // payload patterns such as Ok(Leaf(x)), which remain outside #93.
        let program = program(
            "+module NestedResult\n+fn read (r:Result<Result<Int>>)->String\n +match r\n +case Ok(inner)\n  +match inner\n  +case Ok(value)\n   +return to_string(value)\n  +case Err(message)\n   +return message\n  +end\n +case Err(outer)\n  +return outer\n +end\n+end\n",
        );
        for (value, expected) in [
            (
                Value::Ok(Box::new(Value::Ok(Box::new(Value::Int(5))))),
                "\"5\"",
            ),
            (
                Value::Ok(Box::new(Value::Err("inner fixture".into()))),
                "\"inner fixture\"",
            ),
            (Value::Err("outer fixture".into()), "\"outer fixture\""),
        ] {
            agree(&program, "NestedResult.read", value, expected);
        }
    }

    #[test]
    fn user_union_ok_err_labels_retain_payload_count_and_order() {
        let program = program(
            "+module UserResultLabels\n+type Outcome = Ok(Int,String) | Err(Int)\n+fn read (r:Outcome)->String\n +match r\n +case Ok(number,text)\n  +return concat(to_string(number), text)\n +case Err(number)\n  +return to_string(number)\n +end\n+end\n",
        );
        let ok = Value::Union {
            variant: program.interner.get("Ok").unwrap(),
            payload: vec![Value::Int(5), Value::string(":suffix")],
        };
        let err = Value::Union {
            variant: program.interner.get("Err").unwrap(),
            payload: vec![Value::Int(8)],
        };
        agree(&program, "UserResultLabels.read", ok, "\"5:suffix\"");
        agree(&program, "UserResultLabels.read", err, "\"8\"");
    }
}
