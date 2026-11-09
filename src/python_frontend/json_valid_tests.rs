//! json_valid admission per tier. It is a total, pure `(str) -> bool` builtin.
use crate::{ast, compiler, eval, parser, validator, vm};

fn build(ops: &[parser::Operation]) -> ast::Program {
    let mut program = ast::Program::default();
    for op in ops {
        validator::apply_and_validate(&mut program, op).unwrap();
    }
    program.rebuild_function_index();
    program
}

const CASES: [(&str, bool); 12] = [
    ("{}", true),
    ("[]", true),
    ("null", true),
    (r#"{"x":"null"}"#, true),
    ("18446744073709551615", true),
    ("", false),
    ("{", false),
    ("{} x", false),
    ("<html>bad gateway</html>", false),
    ("1e400", false),
    ("\u{feff}{}", false),
    ("\0", false),
];

#[test]
fn python_json_valid_signature_roundtrip_and_tree_vm_dispatch_tiers() {
    let source =
        "@effects('fail')\ndef probe(payload: str) -> bool:\n    return json_valid(payload)\n";
    let python = build(&super::lower_module(source, "Valid").unwrap().operations);
    let ax = build(
        &parser::parse("+module Valid\n+fn probe (payload:String)->Bool [fail]\n  +return json_valid(payload)\n+end\n").unwrap(),
    );
    assert!(super::ast_equality::program(&python, &ax));
    let printed = super::print_function(&python, "Valid.probe").unwrap();
    let rebuilt = build(&super::lower_module(&printed, "Valid").unwrap().operations);
    assert!(super::ast_equality::program(&python, &rebuilt));
    for program in [&python, &ax] {
        let function = program.get_function("Valid.probe").unwrap();
        for (payload, expected) in CASES {
            let args = vec![eval::Value::string(payload)];
            let mut env = eval::Env::new_with_shared_interner(&program.shared_interner);
            env.set("payload", args[0].clone());
            let tree =
                eval::eval_declared_function(program, "Valid.probe", function, &mut env)
                    .unwrap();
            let code = vm::compile_function(function, program).unwrap();
            let machine = vm::execute_with_io(&code, args, program, &|_, _| {
                anyhow::bail!("json_valid must not request IO")
            })
            .unwrap();
            for result in [tree, machine] {
                assert!(
                    matches!(result, eval::Value::Bool(value) if value == expected),
                    "{payload:?}"
                );
            }
        }
    }
    assert!(crate::builtins::is_builtin("json_valid"));
    assert!(!crate::builtins::is_io_builtin("json_valid"));
    let entry = crate::builtins::BUILTINS
        .iter()
        .find(|entry| entry.name == "json_valid")
        .unwrap();
    assert!(entry.aliases.is_empty());
    for phrase in ["not strict RFC 8259", "127", "1e400", "Never errors"] {
        assert!(entry.long.contains(phrase), "{phrase}");
    }
}

#[test]
fn json_valid_jit_refuses_and_normal_dispatch_falls_back_to_correct_answers() {
    // The JIT only compiles top-level functions. [fail] now refuses before emission;
    // independently, without that annotation it still has no json_* lowering.
    let top = build(
        &parser::parse(
            "+fn probe (payload:String)->Bool [fail]\n  +return json_valid(payload)\n+end\n",
        )
        .unwrap(),
    );
    assert!(!compiler::is_compilable_function(
        top.get_function("probe").unwrap()
    ));
    let error = match compiler::compile(&top) {
        Ok(_) => panic!("the JIT must refuse json_valid"),
        Err(error) => format!("{error:#}"),
    };
    assert!(error.contains("jit: reachable Check"), "{error}");
    // json_kind is refused identically, so json_valid adds no new JIT surface.
    let kind = build(
        &parser::parse(
            "+fn probe (payload:String)->String [fail]\n  +return json_kind(payload, \"\")\n+end\n",
        )
        .unwrap(),
    );
    let kind_error = match compiler::compile(&kind) {
        Ok(_) => panic!("the JIT must refuse json_kind"),
        Err(error) => format!("{error:#}"),
    };
    assert!(
        kind_error.contains("jit: reachable Check"),
        "{kind_error}"
    );
    // Retain the old undefined-builtin coverage separately from fallible admission.
    for (name, function) in [("json_valid", &top), ("json_kind", &kind)] {
        let mut without_fail = function.clone();
        without_fail.get_function_mut("probe").unwrap().effects.clear();
        assert!(compiler::is_compilable_function(without_fail.get_function("probe").unwrap()));
        let Err(error) = compiler::compile(&without_fail) else { panic!("missing JSON JIT lowering") };
        assert!(error.to_string().contains(&format!("undefined function `{name}`")), "{error}");
    }
    // Module-qualified functions are never handed to the JIT at all, and normal
    // dispatch (also with a top-level function, whose String argument cannot use the
    // i64 JIT ABI) answers correctly through the interpreter.
    let module = build(
        &parser::parse("+module JitValid\n+fn probe (payload:String)->Bool [fail]\n  +return json_valid(payload)\n+end\n").unwrap(),
    );
    for (program, name) in [(&module, "JitValid.probe"), (&top, "probe")] {
        for (payload, expected) in CASES {
            let input = parser::parse_single_expr(&format!(
                "{{payload:{}}}",
                serde_json::to_string(payload).unwrap()
            ))
            .unwrap();
            let (result, used_jit) =
                eval::eval_compiled_or_interpreted(program, name, &input).unwrap();
            assert!(!used_jit, "{name} {payload:?}");
            assert_eq!(result, expected.to_string(), "{name} {payload:?}");
        }
    }
}

#[test]
fn python_json_valid_rejects_wrong_arity_types_await_and_user_shadowing() {
    for expression in [
        "json_valid()",
        "json_valid('{}','x')",
        "json_valid(1)",
        "json_valid(True)",
    ] {
        let error = super::lower_module(
            &format!("def probe() -> bool:\n    return {expression}\n"),
            "ValidBad",
        )
        .unwrap_err();
        assert_eq!(error.line, 2);
        assert!(error.to_string().contains("json_valid"), "{error}");
    }
    // The result is bool, not str: assigning it to a str binding is a type error.
    assert!(super::lower_module(
        "def probe() -> str:\n    value: str = json_valid('{}')\n    return value\n",
        "ValidType"
    )
    .is_err());
    assert!(super::lower_module(
        "@effects('io')\nasync def probe() -> bool:\n    value: bool = await json_valid('{}')\n    return value\n",
        "ValidAwait"
    )
    .is_err());
    assert!(super::lower_module(
        "def json_valid(payload: str) -> bool:\n    return True\n",
        "ValidShadow"
    )
    .is_err());
}
