//! #83 deliberately narrow admission, including implicit setup and stub boundaries.
use super::*;
use serde_json::{Value, json};

const TOTAL: &str = "+fn total (price:Int,quantity:Int,fee:Int)->Int\n +let product:Int = price*quantity\n +let subtotal:Int = product+fee\n +return subtotal\n+end\n";
const TESTS: &str = "def test_regular():\n    assert total(7, 3, 2) == 23\ndef test_zero():\n    assert total(0, 3, 2) == 2\n";

async fn fixture(module: &str, extra: &str) -> (AppConfig, WorkingSet, Program) {
    let program = build_program(&format!("+module {module}\n{TOTAL}\n{extra}")).await;
    let config = config_for(
        "writer",
        AccessLevel::Full,
        PermissionConfig::default(),
        program.clone(),
    );
    let ws = config.snapshot_working_set().await;
    (config, ws, program)
}

async fn invoke(
    config: &AppConfig,
    ws: &mut WorkingSet,
    baseline: &Program,
    name: &str,
    args: Value,
) -> Value {
    let result = execute_native_tool(name, &args.to_string(), config, ws, baseline).await;
    serde_json::from_str(&result.content).unwrap()
}

fn evaluation(module: &str) -> Value {
    json!({"language":"python","function":format!("{module}.total"),"input":"price=7, quantity=3, fee=2"})
}

#[tokio::test]
async fn python_scalar_desk_collision_allows_test_evaluate_and_source_with_root_stubs() {
    let module = "NtScalarRoot";
    let (config, mut ws, base) = fixture(module,
        "+module Desk\n+shared seed:Int = 2+3\n+fn restart ()->String [io,async]\n +return \"PRIVATE_RESTART_SENTINEL\"\n+end").await;
    // Root named evaluation/tests bypass the call-dispatch stub interceptor.
    // If a stub ran, the synthetic result would differ (or fail), never equal 23.
    let stub = crate::session::FunctionStub {
        function_name: format!("{module}.total"),
        patterns: vec![],
        response_expr: "Desk.restart()".into(),
    };
    ws.meta.function_stubs.push(stub.clone());
    let handle = crate::coroutine::CoroutineHandle::new_mock_with_stubs(vec![], vec![stub.clone()]);
    assert_eq!(
        handle
            .try_stub(&format!("{module}.total"), &[])
            .unwrap()
            .as_deref(),
        Some("Desk.restart()")
    );
    config.meta.lock().unwrap().function_stubs.push(stub);
    let tested = invoke(
        &config,
        &mut ws,
        &base,
        "test",
        json!({"language":"python","module":module,"source":TESTS}),
    )
    .await;
    assert_eq!(tested["success"], true, "{tested}");
    assert_eq!(tested["test_results"].as_array().unwrap().len(), 2);
    assert!(
        tested["test_results"]
            .as_array()
            .unwrap()
            .iter()
            .all(|t| t["pass"] == true)
    );
    let evaluated = invoke(&config, &mut ws, &base, "evaluate", evaluation(module)).await;
    assert_eq!(evaluated["success"], true, "{evaluated}");
    assert_eq!(
        evaluated["eval_results"],
        format!("eval {module}.total() = 23")
    );
    let source = invoke(
        &config,
        &mut ws,
        &base,
        "inspect",
        json!({"kind":"source","target":format!("{module}.total"),"format":"python"}),
    )
    .await;
    assert_eq!(source["success"], true, "{source}");
    assert!(
        source["query_results"]
            .as_str()
            .unwrap()
            .contains("return subtotal")
    );
    for result in [tested, evaluated, source] {
        assert!(!result.to_string().contains("PRIVATE_RESTART_SENTINEL"));
    }
}

#[tokio::test]
async fn python_scalar_shared_default_calls_refuse_without_execution_or_hidden_details() {
    let module = "NtScalarDefault";
    let (mut config, mut ws, base) = fixture(module,
        "+module NtScalarPrivate\n+fn restart ()->Int\n +return 991337\n+end\n+shared secret:Int = restart()\n").await;
    config.permission_config = Arc::new(
        toml::from_str(
            "[groups]\nsecret=['NtScalarPrivate']\n[model.writer]\nsecret='none'\nuser='write'\n",
        )
        .unwrap(),
    );
    let before = format!("{:?}", ws.program);
    for (name, args) in [
        ("evaluate", evaluation(module)),
        (
            "test",
            json!({"language":"python","module":module,"source":TESTS}),
        ),
    ] {
        let result = invoke(&config, &mut ws, &base, name, args).await;
        assert_eq!(result["success"], false);
        assert_eq!(
            result["output"],
            "cannot safely admit this Python entry under current runtime state"
        );
        assert_eq!(result["test_results"], json!([]));
        assert_eq!(result["eval_results"], Value::Null);
        assert!(!result.to_string().contains("991337"));
        assert!(!result.to_string().contains("NtScalarPrivate"));
        assert_eq!(before, format!("{:?}", ws.program));
        assert!(ws.runtime.shared_vars.is_empty());
    }
    // Printing is authority-filtered and does not evaluate even hidden defaults.
    let printed = invoke(
        &config,
        &mut ws,
        &base,
        "inspect",
        json!({"kind":"source","target":format!("{module}.total"),"format":"python"}),
    )
    .await;
    assert_eq!(printed["success"], true, "{printed}");
    let denied = invoke(
        &config,
        &mut ws,
        &base,
        "evaluate",
        json!({"language":"python","function":"NtScalarPrivate.restart"}),
    )
    .await;
    assert_eq!(denied["success"], false);
    assert!(
        denied["output"]
            .as_str()
            .unwrap()
            .contains("permission denied")
    );
    assert!(!denied.to_string().contains("991337"));
    ws.program
        .modules
        .retain(|module| module.name != "NtScalarPrivate");
    ws.program.rebuild_function_index();
    let without_private = invoke(
        &config,
        &mut ws,
        &base,
        "inspect",
        json!({"kind":"source","target":format!("{module}.total"),"format":"python"}),
    )
    .await;
    assert_eq!(
        printed, without_private,
        "hidden defaults cannot change printing"
    );
}

#[tokio::test]
async fn python_scalar_later_adapsis_calls_revoke_admission_and_lists_stay_checked() {
    let module = "NtScalarRevision";
    let (config, mut ws, base) = fixture(module,
        "+module NtScalarLegacy\n+fn restart ()->Int\n +return 999\n+end\n+fn list (x:Int)->Int\n +return 888\n+end\n").await;
    let good = invoke(
        &config,
        &mut ws,
        &base,
        "test",
        json!({"language":"python","module":module,"source":TESTS}),
    )
    .await;
    assert_eq!(good["success"], true, "{good}");
    for (name, args) in [
        (
            "evaluate",
            json!({"language":"python","function":format!("{module}.total"),"input":"[7], 3, 2"}),
        ),
        (
            "test",
            json!({"language":"python","module":module,"source":"def test_list():\n    assert total([7], 3, 2) == 23\n"}),
        ),
        (
            "test",
            json!({"language":"python","module":module,"source":"def test_list_expected():\n    assert total(7, 3, 2) == [23]\n"}),
        ),
    ] {
        let result = invoke(&config, &mut ws, &base, name, args).await;
        assert_eq!(result["success"], false, "{result}");
        assert_eq!(result["eval_results"], Value::Null);
    }
    // Simulate later legacy replacement on the SAME WorkingSet; no cached waiver.
    for body in ["+return restart()", "+return length(list(restart()))"] {
        let source = format!(
            "+module {module}\n+fn total (price:Int,quantity:Int,fee:Int)->Int\n {body}\n+end"
        );
        let replacement = build_program(&source).await;
        let function = replacement
            .get_function(&format!("{module}.total"))
            .unwrap()
            .clone();
        *ws.program
            .get_function_mut(&format!("{module}.total"))
            .unwrap() = function;
        let result = invoke(&config, &mut ws, &base, "evaluate", evaluation(module)).await;
        assert_eq!(result["success"], false, "{result}");
        assert!(
            result["output"]
                .as_str()
                .unwrap()
                .contains("shadows this builtin")
        );
        assert_eq!(result["eval_results"], Value::Null);
    }
}

#[tokio::test]
async fn python_scalar_unknown_shared_fields_control_flow_and_effects_keep_global_gate() {
    use crate::ast::{Effect, Expr, Literal, Statement, StatementKind};
    let module = "NtScalarExcluded";
    let (config, mut ws, base) = fixture(module,
        "+shared ambient:Int = 7\n+module NtScalarExcludedLegacy\n+fn restart ()->Int\n +return 999\n+end").await;
    let target = format!("{module}.total");
    let original = ws.program.get_function(&target).unwrap().clone();
    let last = original.body.last().unwrap().clone();
    let value = Expr::Literal(Literal::Int(23));
    let candidates = [
        (
            StatementKind::Return {
                value: Expr::Identifier("ambient".into()),
            },
            vec![],
        ),
        (
            StatementKind::Return {
                value: Expr::FieldAccess {
                    base: Box::new(Expr::Literal(Literal::String("abc".into()))),
                    field: "len".into(),
                },
            },
            vec![],
        ),
        (
            StatementKind::Branch {
                condition: Expr::Literal(Literal::Bool(true)),
                then_body: vec![last.clone()],
                else_body: vec![last.clone()],
            },
            vec![],
        ),
        (
            StatementKind::Return {
                value: value.clone(),
            },
            vec![Effect::Io],
        ),
        (StatementKind::Return { value }, vec![Effect::Async]),
    ];
    for (kind, effects) in candidates {
        let function = ws.program.get_function_mut(&target).unwrap();
        *function = original.clone();
        function.body = vec![Statement {
            id: last.id.clone(),
            kind,
        }];
        function.effects = effects;
        let result = invoke(&config, &mut ws, &base, "evaluate", evaluation(module)).await;
        assert_eq!(result["success"], false, "{result}");
        assert!(
            result["output"]
                .as_str()
                .unwrap()
                .contains("shadows this builtin")
        );
        assert_eq!(result["eval_results"], Value::Null);
    }
}
