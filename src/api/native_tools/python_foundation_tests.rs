//! P0 regressions exercise actual native admission, not just standalone lowering.
use super::*;
use serde_json::json;

async fn writer(program: Program) -> AppConfig {
    let permissions = toml::from_str("[model.writer]\nuser = 'write'\n").unwrap();
    config_for("writer", AccessLevel::Full, permissions, program)
}

#[tokio::test]
async fn python_p0_inspection_cannot_retarget_bare_calls_even_with_denied_competitor() {
    for denied in [false, true] {
        let suffix = if denied { "Hidden" } else { "Visible" };
        let a = format!("NtP0A{suffix}");
        let b = format!("NtP0B{suffix}");
        let program = build_program(&format!("+module {a}\n+fn helper ()->Int\n +return 1\n+end\n+module {b}\n+fn helper ()->Int\n +return 2\n+end\n+fn caller ()->Int\n +return helper()\n+end\n")).await;
        let input = crate::parser::parse_test_input(1, "").unwrap();
        let target = format!("{b}.caller");
        assert_eq!(
            crate::eval::eval_call_with_input(&program, &target, &input).unwrap(),
            "1"
        );
        let permissions = toml::from_str(&format!(
            "[groups]\ncompetitor = ['{a}']\n[model.writer]\ncompetitor = '{}'\nuser = 'write'\n",
            if denied { "none" } else { "write" }
        ))
        .unwrap();
        let cfg = config_for("writer", AccessLevel::Full, permissions, program.clone());
        let mut ws = cfg.snapshot_working_set().await;
        let before = ws.program.clone();
        let result = execute_native_tool(
            "inspect",
            &json!({"kind":"source","target":target,"format":"python"}).to_string(),
            &cfg,
            &mut ws,
            &program,
        )
        .await;
        assert!(
            result.execution.has_errors && result.content.contains("exact"),
            "{}",
            result.content
        );
        assert_eq!(ws.program, before);
        assert_eq!(
            crate::eval::eval_call_with_input(&ws.program, &target, &input).unwrap(),
            "1"
        );
    }
}

#[tokio::test]
async fn python_p0_native_mutation_rejects_unresolved_operands_and_missing_fields() {
    let baseline = Program::default();
    let cfg = writer(baseline.clone()).await;
    let mut ws = cfg.snapshot_working_set().await;
    for (index, (source, diagnostic)) in [
        (
            "def f() -> int:\n    return len(to_int('1'))\n",
            "statically known",
        ),
        (
            "def f() -> bool:\n    return True < abs(1)\n",
            "matching known",
        ),
        (
            "def f() -> int:\n    return not abs(1)\n",
            "not requires bool",
        ),
        ("def f() -> int:\n    return abs(1) + 1\n", "matching known"),
        (
            "def f() -> int:\n    if abs(1):\n        return 1\n    return 0\n",
            "statically bool",
        ),
        (
            "def f() -> str:\n    return concat('a', abs(1))\n",
            "concat requires",
        ),
        (
            "@dataclass\nclass Point:\n    x: int\ndef f(p: Point) -> int:\n    return p.missing\n",
            "unknown field",
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let before = ws.program.clone();
        let result = execute_native_tool(
            "mutate",
            &json!({"language":"python","module":format!("NtP0Operand{index}"),"source":source})
                .to_string(),
            &cfg,
            &mut ws,
            &baseline,
        )
        .await;
        assert!(
            result.execution.has_errors && result.content.contains(diagnostic),
            "{index}: {}",
            result.content
        );
        assert_eq!(ws.program, before, "invalid source must not mutate program");
    }
}

#[tokio::test]
async fn python_p0_known_bool_not_has_bool_result_and_valid_native_control() {
    let baseline = Program::default();
    let cfg = writer(baseline.clone()).await;
    let mut ws = cfg.snapshot_working_set().await;
    let rejected = execute_native_tool("mutate", &json!({"language":"python","module":"NtP0NotBad","source":"def f(flag: bool) -> int:\n    return not flag\n"}).to_string(), &cfg, &mut ws, &baseline).await;
    assert!(
        rejected.execution.has_errors && rejected.content.contains("type mismatch"),
        "{}",
        rejected.content
    );
    let accepted = execute_native_tool("mutate", &json!({"language":"python","module":"NtP0NotGood","source":"def f(flag: bool) -> bool:\n    return not flag\n"}).to_string(), &cfg, &mut ws, &baseline).await;
    assert!(!accepted.execution.has_errors, "{}", accepted.content);
}

#[tokio::test]
async fn python_p0_duplicate_type_catalog_must_not_borrow_other_owner() {
    let program = build_program(
        "+module NtP0TypeA\n+type Point = x:Int\n+module NtP0TypeB\n+type Point = x:String\n",
    )
    .await;
    let cfg = writer(program.clone()).await;
    let mut ws = cfg.snapshot_working_set().await;
    let before = ws.program.clone();
    let result = execute_native_tool("mutate", &json!({"language":"python","module":"NtP0TypeA","source":"def wrong(p: Point) -> str:\n    return p.x\n"}).to_string(), &cfg, &mut ws, &program).await;
    // A.Point.x is Int. Until an owner-qualified catalog exists, refusal is
    // required; B.Point.x must never make this invalid A function acceptable.
    assert!(
        result.execution.has_errors,
        "owner-colliding type was silently accepted: {}",
        result.content
    );
    assert_eq!(ws.program, before);
}

#[tokio::test]
async fn python_p0_inspect_replace_preserves_attached_tests_and_other_functions() {
    let mut program = build_program("+module NtP0Preserved\n+fn inc (x:Int)->Int\n +return x+1\n+end\n+fn untouched ()->Int\n +return 7\n+end\n").await;
    program
        .get_function_mut("NtP0Preserved.inc")
        .unwrap()
        .tests
        .push(crate::ast::TestCase {
            input: "1".into(),
            expected: "2".into(),
            passed: true,
            matcher: None,
            after_checks: vec![],
        });
    let original = program.get_function("NtP0Preserved.inc").unwrap().clone();
    let cfg = writer(program.clone()).await;
    let mut ws = cfg.snapshot_working_set().await;
    let result = execute_native_tool(
        "inspect",
        r#"{"kind":"source","target":"NtP0Preserved.inc","format":"python"}"#,
        &cfg,
        &mut ws,
        &program,
    )
    .await;
    assert!(!result.execution.has_errors, "{}", result.content);
    let envelope: serde_json::Value = serde_json::from_str(&result.content).unwrap();
    let printed = envelope["query_results"].as_str().unwrap();
    let edited = printed.replace("(x + 1)", "(1 + x)");
    assert_ne!(
        printed, edited,
        "test must actually edit the submitted body"
    );
    let result = execute_native_tool(
        "mutate",
        &json!({"language":"python","module":"NtP0Preserved","source":edited}).to_string(),
        &cfg,
        &mut ws,
        &program,
    )
    .await;
    assert!(!result.execution.has_errors, "{}", result.content);
    let replaced = ws.program.get_function("NtP0Preserved.inc").unwrap();
    assert_eq!(replaced.tests, original.tests);
    assert_eq!(
        replaced
            .params
            .iter()
            .map(|p| (&p.name, &p.ty))
            .collect::<Vec<_>>(),
        original
            .params
            .iter()
            .map(|p| (&p.name, &p.ty))
            .collect::<Vec<_>>()
    );
    assert_eq!(replaced.effects, original.effects);
    assert_eq!(replaced.return_type, original.return_type);
    assert_eq!(replaced.doc, original.doc);
    let crate::ast::StatementKind::Return {
        value: crate::ast::Expr::Binary { left, right, .. },
    } = &replaced.body[0].kind
    else {
        panic!("expected edited return")
    };
    assert!(matches!(
        left.as_ref(),
        crate::ast::Expr::Literal(crate::ast::Literal::Int(1))
    ));
    assert!(matches!(right.as_ref(), crate::ast::Expr::Identifier(name) if name == "x"));
    let old_other = &program.modules[0].functions[1];
    let new_other = &ws.program.modules[0].functions[1];
    assert!(
        Arc::ptr_eq(old_other, new_other),
        "unsubmitted function was replaced/deleted"
    );
}
