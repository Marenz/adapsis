use super::*;
use serde_json::json;

#[test]
fn python_native_metadata_is_explicit_closed_and_backward_compatible() {
    for (name, arguments) in [
        (
            "mutate",
            json!({"language":"python","source":"def f() -> int:\n    return 1\n"}),
        ),
        (
            "mutate",
            json!({"language":"adapsis","module":"M","source":"+module M"}),
        ),
        (
            "mutate",
            json!({"language":"python","module":"M\n+module Other","source":"def f() -> int:\n    return 1\n"}),
        ),
        (
            "test",
            json!({"language":"python","module":"M","source":"def test_f():\n    assert f() == 1\n","extra":true}),
        ),
        (
            "evaluate",
            json!({"language":"javascript","function":"M.f"}),
        ),
        (
            "evaluate",
            json!({"language":"python","function":"M.f","input":"Other.secret()"}),
        ),
        (
            "inspect",
            json!({"kind":"deps","target":"M.f","format":"python"}),
        ),
        ("finish", json!({"language":"python"})),
    ] {
        assert!(
            validate_native_tool(name, &arguments.to_string()).is_err(),
            "{name}: {arguments}"
        );
    }
    assert!(validate_native_tool("evaluate", r#"{"function":"M.f","input":"true"}"#).is_ok());
    assert!(
        validate_native_tool(
            "evaluate",
            r#"{"language":"python","function":"M.f","input":"True"}"#
        )
        .is_ok()
    );
    assert!(
        validate_native_tool(
            "evaluate",
            r#"{"language":"python","function":"M.f","input":"[1, 2]"}"#
        )
        .is_ok()
    );
    assert!(
        validate_native_tool(
            "evaluate",
            r#"{"function":"M.f","input":"list(Other.secret())"}"#
        )
        .is_err()
    );
}

async fn collision_fixture(
    suffix: &str,
    helper: &str,
    union: bool,
) -> (AppConfig, WorkingSet, Program, String) {
    let owner = format!("PyHelperOwner{suffix}");
    let target = format!("PyHelperTarget{suffix}");
    let declaration = if union {
        format!("+type Shadow = {helper}(Int) | Other")
    } else {
        format!("+fn {helper} ()->Int\n +return 999\n+end")
    };
    let program = build_program(&format!("+module {owner}\n{declaration}\n+module {target}\n+fn echo (values:List<Int>)->List<Int>\n +return values\n+end")).await;
    // Both modules are writable: the collision must be rejected for semantic
    // fidelity even when it would not violate a module permission by itself.
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        PermissionConfig::default(),
        program.clone(),
    );
    let ws = cfg.snapshot_working_set().await;
    (cfg, ws, program, target)
}

#[tokio::test]
async fn python_literal_lists_never_execute_authorized_function_or_union_collisions() {
    for union in [false, true] {
        let suffix = if union { "Union" } else { "Function" };
        let (cfg, mut ws, baseline, target) = collision_fixture(suffix, "list", union).await;
        for (tool, arguments) in [
            (
                "evaluate",
                json!({"language":"python","function":format!("{target}.echo"),"input":"[1, 2]"}),
            ),
            (
                "test",
                json!({"language":"python","module":target,"source":"def test_echo():\n    assert echo([1, 2]) == [1, 2]\n"}),
            ),
            (
                "mutate",
                json!({"source":format!("+module {target}\n+shared values:List<Int> = list(1, 2)\n")}),
            ),
        ] {
            let result =
                execute_native_tool(tool, &arguments.to_string(), &cfg, &mut ws, &baseline).await;
            assert!(
                result.execution.has_errors && result.content.contains("shadows"),
                "{}",
                result.content
            );
            assert!(result.execution.test_results.is_empty());
        }
    }
}

#[tokio::test]
async fn python_synthetic_helpers_reject_shadowing_before_mutation() {
    for (helper, source) in [
        (
            "concat",
            "def combine(left: str, right: str) -> str:\n    return left + right\n",
        ),
        (
            "len",
            "def measure(values: list[int]) -> int:\n    return len(values)\n",
        ),
        (
            "list",
            "def literal_values() -> list[int]:\n    return [1, 2]\n",
        ),
    ] {
        let (cfg, mut ws, baseline, target) =
            collision_fixture(&format!("Synthetic{helper}"), helper, false).await;
        let before = ws.program.clone();
        let result = execute_native_tool(
            "mutate",
            &json!({"language":"python","module":target,"source":source}).to_string(),
            &cfg,
            &mut ws,
            &baseline,
        )
        .await;
        assert!(
            result.execution.has_errors && result.content.contains("shadows"),
            "{}",
            result.content
        );
        assert_eq!(ws.program, before);
    }
}

#[tokio::test]
async fn python_literal_extension_rejects_same_mutation_shared_constructor_shadow() {
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        PermissionConfig::default(),
        Program::default(),
    );
    let mut ws = cfg.snapshot_working_set().await;
    let result = execute_native_tool("mutate", &json!({"source":"+module PySameMutationLiteral\n+fn list ()->Int\n +return 999\n+end\n+shared values:List<Int> = list(1)\n"}).to_string(), &cfg, &mut ws, &Program::default()).await;
    assert!(result.execution.has_errors);
    assert!(ws.program.modules.is_empty());
}

#[tokio::test]
async fn python_source_format_is_opt_in_and_refuses_unsupported_legacy_ast() {
    let program = build_program("+module PyLegacySource\n+shared value:Int = 1\n+fn shared_value ()->Int\n +return value\n+end").await;
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        PermissionConfig::default(),
        program.clone(),
    );
    let mut ws = cfg.snapshot_working_set().await;
    let default = execute_native_tool(
        "inspect",
        r#"{"kind":"source","target":"PyLegacySource.shared_value"}"#,
        &cfg,
        &mut ws,
        &program,
    )
    .await;
    assert!(!default.execution.has_errors && default.content.contains("+fn"));
    let python = execute_native_tool(
        "inspect",
        r#"{"kind":"source","target":"PyLegacySource.shared_value","format":"python"}"#,
        &cfg,
        &mut ws,
        &program,
    )
    .await;
    assert!(!python.execution.has_errors, "{}", python.content);
    assert!(python.content.contains("def shared_value"), "{}", python.content);
}

#[tokio::test]
async fn python_inspect_prints_types_and_match_functions() {
    let program = build_program("+module PyRound\n+type Point = {x:Int, y:Int}\n+type Expr = Literal(Int) | Add(Expr, Expr)\n+fn eval_expr (e:Expr)->Int\n  +match e\n  +case Literal(v)\n    +return v\n  +case Add(l, r)\n    +return 0\n  +case _\n    +return 0\n  +end\n+end\n").await;
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        PermissionConfig::default(),
        program.clone(),
    );
    let mut ws = cfg.snapshot_working_set().await;
    let function = execute_native_tool(
        "inspect",
        r#"{"kind":"source","target":"PyRound.eval_expr","format":"python"}"#,
        &cfg,
        &mut ws,
        &program,
    )
    .await;
    assert!(!function.execution.has_errors, "{}", function.content);
    assert!(function.content.contains("case Add("), "{}", function.content);
    let ty = execute_native_tool(
        "inspect",
        r#"{"kind":"source","target":"PyRound.Point","format":"python"}"#,
        &cfg,
        &mut ws,
        &program,
    )
    .await;
    assert!(!ty.execution.has_errors, "{}", ty.content);
    assert!(ty.content.contains("@dataclass"), "{}", ty.content);
    let union = execute_native_tool(
        "inspect",
        r#"{"kind":"source","target":"PyRound.Expr","format":"python"}"#,
        &cfg,
        &mut ws,
        &program,
    )
    .await;
    assert!(!union.execution.has_errors, "{}", union.content);
    assert!(union.content.contains("@union"), "{}", union.content);
}

#[tokio::test]
async fn python_telegrambot_like_inspect_edit_mutate_roundtrip() {
    let program = build_program("+module TgBot\n+shared bot_name:String = \"<redacted>\"\n+fn greet (name:String)->String\n  +return concat(bot_name, \": hi \", name)\n+fn classify (code:Int)->String\n  +match code\n  +case 0\n    +return \"zero\"\n  +case _\n    +return \"other\"\n  +end\n+end\n").await;
    let cfg = config_for("writer", AccessLevel::Full, PermissionConfig::default(), program.clone());
    let mut ws = cfg.snapshot_working_set().await;
    let inspected = execute_native_tool(
        "inspect",
        r#"{"kind":"source","target":"TgBot.greet","format":"python"}"#,
        &cfg,
        &mut ws,
        &program,
    )
    .await;
    assert!(!inspected.execution.has_errors, "{}", inspected.content);
    assert!(!inspected.execution.has_errors, "{}", inspected.content);
    let envelope: serde_json::Value = serde_json::from_str(&inspected.content).unwrap();
    let printed = envelope["query_results"].as_str().unwrap_or("").to_string();
    assert!(printed.contains("def greet"), "{printed}");
    // Re-mutate the printed function (plus its shared var) into a new module and test it.
    let source = format!("bot_name: str = \"<redacted>\"\n\n{printed}");
    let mutated = execute_native_tool(
        "mutate",
        &json!({"language":"python","module":"TgBot2","source":source}).to_string(),
        &cfg,
        &mut ws,
        &program,
    )
    .await;
    assert!(!mutated.execution.has_errors, "{}", mutated.content);
    let tested = execute_native_tool(
        "test",
        &json!({"language":"python","module":"TgBot2","source":"def test_greet():\n    assert greet('bob') == '<redacted>: hi bob'\n"}).to_string(),
        &cfg,
        &mut ws,
        &program,
    )
    .await;
    assert!(!tested.execution.has_errors, "{}", tested.content);
    assert!(!tested.execution.test_results.is_empty());
}
