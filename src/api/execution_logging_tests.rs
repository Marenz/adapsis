//! Real executor paths with inert watchers, a local mock provider, and a fake
//! opencode process that records argv then fails (so it cannot build or restart).
use super::*;

#[tokio::test]
async fn legacy_argument_logs_are_gated_before_mock_side_effects() {
    if !crate::logging::tests::child() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let exe = dir.path().join("opencode");
        std::fs::write(&exe,
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$ADAPSIS_LOG_TEST_ARGV\"\nexit 1\n").unwrap();
        std::fs::set_permissions(exe, std::fs::Permissions::from_mode(0o755)).unwrap();
        let args_file = dir.path().join("argv");
        for enabled in [false, true] {
            let captured = crate::logging::tests::capture_with_env(
                "api::tests::execution_logging_tests::legacy_argument_logs_are_gated_before_mock_side_effects",
                enabled, &[("PATH", dir.path().to_str().unwrap()),
                    ("ADAPSIS_LOG_TEST_ARGV", args_file.to_str().unwrap())]);
            for canary in ["CANARY_WATCH", "CANARY_AGENT", "CANARY_OPENCODE"] {
                assert_eq!(captured.contains(canary), enabled, "capture: {captured}");
            }
            assert!(std::fs::read_to_string(&args_file).unwrap().contains("CANARY_OPENCODE"));
        }
        return;
    }
    let mut config = test_config();
    let (url, requests, server) = start_scripted_llm(&["<code>!done</code>"]).await;
    config.llm_url = url;
    config.opencode_git_dir = std::env::var("PATH").unwrap();
    let mut ws = config.snapshot_working_set().await;
    let result = execute_code("!watch Missing.echo \"CANARY_WATCH\" 3600000\n\
        !agent fixture --scope full CANARY_AGENT",
        &config, &mut ws, None).await;
    assert!(result.agent_spawned);
    assert!(result.mutation_results.iter().any(|r| r.message.contains("Watching")));
    let result = execute_code("!opencode CANARY_OPENCODE", &config, &mut ws, None).await;
    assert!(result.mutation_results.iter().any(|r| r.message.contains("OpenCode failed")),
        "{:?}", result.mutation_results);
    assert!(!result.needs_opencode_restart);
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            if !requests.lock().await.is_empty() { break; }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    }).await.unwrap();
    assert!(requests.lock().await[0].to_string().contains("CANARY_AGENT"));
    server.abort();
}

#[tokio::test]
async fn shell_allowlist_refusal_keeps_command_only_in_caller_result() {
    if !crate::logging::tests::child() {
        for enabled in [false, true] {
            let output = crate::logging::tests::capture(
                "api::tests::execution_logging_tests::shell_allowlist_refusal_keeps_command_only_in_caller_result", enabled);
            assert_eq!(output.contains("CANARY_SHELL_PROGRAM"), enabled);
            assert!(output.contains("shell-policy-refused"));
        }
        return;
    }
    let (mut runtime, _rx) = crate::coroutine::Runtime::new();
    runtime.shell_policy = crate::coroutine::ShellPolicy::Allowlist(vec!["git".into()]);
    let (reply, receiver) = tokio::sync::oneshot::channel();
    runtime.handle_io(crate::coroutine::IoRequest::ShellExec {
        command: "CANARY_SHELL_PROGRAM argument".into(), reply,
    }).await;
    let error = receiver.await.unwrap().unwrap_err().to_string();
    assert!(error.contains("CANARY_SHELL_PROGRAM"));
    assert!(error.contains("allowlist"));
}

#[tokio::test]
async fn webhook_error_preserves_caller_detail_without_journal_leak() {
    if !crate::logging::tests::child() {
        for enabled in [false, true] {
            let output = crate::logging::tests::capture(
                "api::tests::execution_logging_tests::webhook_error_preserves_caller_detail_without_journal_leak", enabled);
            assert_eq!(output.contains("CANARY_WEBHOOK_ERROR"), enabled);
            assert!(output.contains("webhook-handler-error"));
        }
        return;
    }
    let config = test_config();
    let mut ws = config.snapshot_working_set().await;
    let setup = execute_code(
        "+module NtLogWebhook\n+fn bad(body:String)->String\n +return regex_replace(body, \"x\", \"text\")\n+end",
        &config, &mut ws, None).await;
    assert!(!setup.has_errors);
    config.write_back_working_set(&ws).await;
    config.runtime.write().unwrap().http_routes.push(crate::ast::HttpRoute {
        method: "POST".into(), path: "/logging-test".into(), handler_fn: "NtLogWebhook.bad".into(),
    });
    let response = super::super::adapsis_route_dispatch(State(config), axum::http::Method::POST,
        "/logging-test".parse().unwrap(), axum::body::Bytes::from_static(b"[CANARY_WEBHOOK_ERROR")).await;
    assert_eq!(response.status(), axum::http::StatusCode::INTERNAL_SERVER_ERROR);
    let body = axum::body::to_bytes(response.into_body(), 8000).await.unwrap();
    assert!(String::from_utf8_lossy(&body).contains("CANARY_WEBHOOK_ERROR"));
}
