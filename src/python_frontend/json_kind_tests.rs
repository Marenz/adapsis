use crate::{ast, eval, parser, validator, vm};

fn build(ops: &[parser::Operation]) -> ast::Program {
    let mut program = ast::Program::default();
    for op in ops {
        validator::apply_and_validate(&mut program, op).unwrap();
    }
    program.rebuild_function_index();
    program
}

#[test]
fn python_json_kind_signature_roundtrip_and_no_io_tiers() {
    let source="@effects('fail')\ndef probe(payload: str, path: str) -> str:\n    return json_kind(payload, path)\n";
    let python = build(&super::lower_module(source, "Kind").unwrap().operations);
    let ax=build(&parser::parse("+module Kind\n+fn probe (payload:String,path:String)->String [fail]\n  +return json_kind(payload,path)\n+end\n").unwrap());
    assert!(super::ast_equality::program(&python, &ax));
    let printed = super::print_function(&python, "Kind.probe").unwrap();
    let rebuilt = build(&super::lower_module(&printed, "Kind").unwrap().operations);
    assert!(super::ast_equality::program(&python, &rebuilt));
    for program in [&python, &ax] {
        let function = program.get_function("Kind.probe").unwrap();
        for (payload, path, expected) in [
            ("{}", "x", "missing"),
            (r#"{"x":""}"#, "x", "string"),
            (r#"{"x":null}"#, "x", "null"),
            (r#"{"x":"null"}"#, "x", "string"),
            (r#"{"x":{}}"#, "x", "object"),
            (r#"{"x":"{}"}"#, "x", "string"),
            (r#"{"x":[]}"#, "x", "array"),
            (r#"{"x":"[]"}"#, "x", "string"),
            ("1", "", "int"),
            ("1.0", "", "float"),
            ("true", "", "bool"),
        ] {
            let args = vec![eval::Value::string(payload), eval::Value::string(path)];
            let mut env = eval::Env::new_with_shared_interner(&program.shared_interner);
            env.set("payload", args[0].clone());
            env.set("path", args[1].clone());
            let tree =
                eval::eval_function_body_named(program, "Kind.probe", &function.body, &mut env)
                    .unwrap();
            let code = vm::compile_function(function, program).unwrap();
            let machine = vm::execute_with_io(&code, args, program, &|_, _| {
                anyhow::bail!("json_kind must not request IO")
            })
            .unwrap();
            for result in [tree, machine] {
                assert!(
                    matches!(result,eval::Value::String(ref value) if value.as_str()==expected)
                );
            }
        }
    }
    assert!(crate::builtins::is_builtin("json_kind"));
    assert!(!crate::builtins::is_io_builtin("json_kind"));
    assert!(crate::builtins::BUILTINS
        .iter()
        .find(|entry| entry.name == "json_kind")
        .unwrap()
        .aliases
        .is_empty());
}

#[test]
fn python_json_kind_rejects_wrong_arity_types_and_await() {
    for expression in [
        "json_kind()",
        "json_kind('{}')",
        "json_kind('{}','x','extra')",
        "json_kind(1,'x')",
        "json_kind('{}',0)",
    ] {
        let error = super::lower_module(
            &format!("def probe() -> str:\n    return {expression}\n"),
            "KindBad",
        )
        .unwrap_err();
        assert_eq!(error.line, 2);
        assert!(error.to_string().contains("json_kind"), "{error}");
    }
    assert!(super::lower_module("@effects('io')\nasync def probe() -> str:\n    value: str = await json_kind('{}','x')\n    return value\n","KindBadAwait").is_err());
}
