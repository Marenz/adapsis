use super::*;
use serde_json::json;

async fn fixture(suffix: &str) -> (AppConfig, WorkingSet, Program, String, String) {
    let private = format!("NtPrivate{suffix}");
    let public = format!("NtPublic{suffix}");
    let source = format!(
        "+module {private}\n+type SecretType = {{value:String}}\n+fn secret ()->String\n +return \"PRIVATE_SENTINEL\"\n+end\n+fn {public} ()->String\n +return \"COLLISION_SENTINEL\"\n+end\n+module {public}\n+fn echo (x:String)->String\n +return x\n+end\n+fn async_echo (x:String)->String [io,async]\n +return x\n+end\n"
    );
    let program = build_program(&source).await;
    let permission: PermissionConfig = toml::from_str(&format!(
        r#"
        [groups]
        hidden = ["{private}"]
        [model.restricted]
        hidden = "none"
        user = "write"
    "#
    ))
    .unwrap();
    let cfg = config_for("restricted", AccessLevel::Full, permission, program.clone());
    let ws = cfg.snapshot_working_set().await;
    (cfg, ws, program, private, public)
}

#[tokio::test]
async fn native_bare_targets_and_module_collisions_never_resolve_hidden_source() {
    let (cfg, mut ws, baseline, _, public) = fixture("Bare").await;
    for target in ["secret", "SecretType", public.as_str()] {
        for kind in ["source", "deps"] {
            let result = execute_native_tool(
                "inspect",
                &json!({"kind":kind,"target":target}).to_string(),
                &cfg,
                &mut ws,
                &baseline,
            )
            .await;
            assert!(result.execution.has_errors);
            assert!(
                !result.content.contains("PRIVATE_SENTINEL")
                    && !result.content.contains("COLLISION_SENTINEL")
            );
        }
    }
    let result = execute_native_tool(
        "evaluate",
        r#"{"function":"secret"}"#,
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(result.execution.has_errors && !result.content.contains("PRIVATE_SENTINEL"));
}

#[tokio::test]
async fn native_calling_inputs_and_global_after_assertions_are_rejected() {
    let (cfg, mut ws, baseline, private, public) = fixture("Inputs").await;
    for (tool, args) in [
        (
            "evaluate",
            json!({"function":format!("{public}.async_echo"),"input":format!("{private}.secret()")}),
        ),
        (
            "test",
            json!({"source":format!("+test {public}.echo\n +with x={private}.secret() -> expect \"wrong\"\n")}),
        ),
        (
            "test",
            json!({"source":format!("+test {public}.echo\n +with x=\"x\" -> expect {private}.secret()\n")}),
        ),
        (
            "test",
            json!({"source":format!("+test {public}.echo\n +with x=\"x\" -> expect \"x\"\n +after modules contains \"absent\"\n")}),
        ),
    ] {
        let result = execute_native_tool(tool, &args.to_string(), &cfg, &mut ws, &baseline).await;
        assert!(result.execution.has_errors, "{tool}: {}", result.content);
        assert!(!result.content.contains("PRIVATE_SENTINEL"));
    }
}

#[tokio::test]
async fn native_authored_hidden_calls_and_local_name_collisions_are_denied_before_mutation() {
    let (cfg, mut ws, baseline, private, public) = fixture("Authored").await;
    for body in [
        format!("+fn probe ()->String\n +return {private}.secret()\n+end"),
        "+fn probe ()->String\n +return secret()\n+end\n+fn secret ()->String\n +return \"public\"\n+end".into(),
        format!("+shared leak:String = {private}.secret()"),
    ] {
        let args = json!({"source":format!("+module {public}\n{body}")});
        let result = execute_native_tool("mutate", &args.to_string(), &cfg, &mut ws, &baseline).await;
        assert!(result.execution.has_errors, "{}", result.content);
        assert!(ws.program.get_function(&format!("{public}.probe")).is_none());
    }
}

#[tokio::test]
async fn native_missing_qualified_targets_do_not_suggest_hidden_names() {
    let (cfg, mut ws, baseline, private, public) = fixture("Missing").await;
    for (tool, args) in [
        ("evaluate", json!({"function":format!("{public}.secret")})),
        (
            "test",
            json!({"source":format!("+test {public}.secret\n +with -> expect \"x\"\n")}),
        ),
    ] {
        let result = execute_native_tool(tool, &args.to_string(), &cfg, &mut ws, &baseline).await;
        assert!(result.execution.has_errors);
        assert!(!result.content.contains(&private));
        assert!(
            result
                .execution
                .mutation_results
                .iter()
                .all(|r| !r.message.contains(&private))
        );
    }
}

#[tokio::test]
async fn native_task_inspection_omits_hidden_rows_and_all_status_payloads() {
    let (mut cfg, mut ws, baseline, private, public) = fixture("Tasks").await;
    let tasks = [
        (
            1,
            format!("{private}.secret"),
            crate::coroutine::WaitReason::Completed("PRIVATE_SENTINEL".into()),
        ),
        (
            2,
            format!("{public}.echo"),
            crate::coroutine::WaitReason::ShellExec("SHELL_SECRET".into()),
        ),
        (
            3,
            format!("{public}.async_echo"),
            crate::coroutine::WaitReason::FileRead("/PRIVATE_PATH".into()),
        ),
    ]
    .into_iter()
    .map(|(id, function_name, status)| {
        (
            id,
            crate::coroutine::TaskInfo {
                id,
                function_name,
                status,
                started_at: "fixture".into(),
            },
        )
    })
    .collect();
    cfg.task_registry = Some(Arc::new(std::sync::Mutex::new(tasks)));
    let result =
        execute_native_tool("inspect", r#"{"kind":"tasks"}"#, &cfg, &mut ws, &baseline).await;
    assert!(!result.execution.has_errors);
    assert!(result.content.contains(&public) && result.content.contains("waiting"));
    for secret in [&private, "PRIVATE_SENTINEL", "SHELL_SECRET", "PRIVATE_PATH"] {
        assert!(!result.content.contains(secret), "{}", result.content);
    }
}

#[tokio::test]
async fn native_result_matchers_cannot_execute_shadowed_expected_functions() {
    let (cfg, mut ws, baseline, private, public) = fixture("Matcher").await;
    let ops = parser::parse(&format!(
        "+module {private}\n+fn Ok ()->Int\n +return 42\n+end\n+fn Err ()->Int\n +return 43\n+end"
    ))
    .unwrap();
    for op in ops {
        crate::validator::apply_and_validate(&mut ws.program, &op).unwrap();
    }
    assert!(ws.program.get_function("Ok").is_some());
    for expected in ["Ok", "Err", "Err(\"message\")"] {
        let args =
            json!({"source":format!("+test {public}.echo\n +with x=\"x\" -> expect {expected}\n")});
        let result = execute_native_tool("test", &args.to_string(), &cfg, &mut ws, &baseline).await;
        assert!(result.execution.has_errors);
        assert!(
            result.execution.test_results.is_empty(),
            "expected-expression code must never execute"
        );
        assert!(result.content.contains("literals"));
    }
    for matcher in ["contains(\"x\")", "starts_with(\"x\")"] {
        let args =
            json!({"source":format!("+test {public}.echo\n +with x=\"x\" -> expect {matcher}\n")});
        let result = execute_native_tool("test", &args.to_string(), &cfg, &mut ws, &baseline).await;
        assert!(!result.execution.has_errors, "{}", result.content);
    }
}
