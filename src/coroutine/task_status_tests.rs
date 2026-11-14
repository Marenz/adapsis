//! #112: task status rendering never prints an operation argument. Sentinels stand in for
//! credentials; nothing here touches the network or a real token.
use super::*;

fn render(reason: WaitReason) -> String {
    format!("{reason}")
}

const TELEGRAM: &str =
    "https://api.telegram.org/bot123456:SECRET-TOKEN_abc/getUpdates?offset=5&timeout=30";

#[test]
fn http_operations_show_scheme_and_host_only() {
    assert_eq!(
        render(WaitReason::HttpGet(TELEGRAM.into())),
        "http_get(https://api.telegram.org)"
    );
    assert_eq!(
        render(WaitReason::HttpPost(TELEGRAM.into())),
        "http_post(https://api.telegram.org)"
    );
    assert_eq!(
        render(WaitReason::HttpRequest("PATCH".into(), TELEGRAM.into())),
        "http_request(PATCH https://api.telegram.org)"
    );
    // Userinfo, port, query and fragment.
    assert_eq!(
        render(WaitReason::HttpGet(
            "https://user:pa55@host.example:8443/p?api_key=SECRET#frag".into()
        )),
        "http_get(https://host.example:8443)"
    );
    assert_eq!(
        render(WaitReason::HttpGet(
            "http://127.0.0.1:3002/api/status".into()
        )),
        "http_get(http://127.0.0.1:3002)"
    );
    assert_eq!(
        render(WaitReason::HttpGet("  HTTPS://Mixed.Case/x  ".into())),
        "http_get(https://mixed.case)"
    );
    assert_eq!(
        render(WaitReason::HttpGet("http://[::1]:9/x?k=SECRET".into())),
        "http_get(http://[::1]:9)"
    );
}

#[test]
fn unparsable_or_non_http_urls_are_never_echoed() {
    for raw in [
        "not a url with SECRET",
        "https://",
        "",
        "bot123:SECRET",
        "ftp://user:SECRET@host/x",
        "file:///etc/SECRET",
        "data:text/plain,SECRET",
        "javascript:SECRET",
        "https://user:SECRET@",
    ] {
        let text = render(WaitReason::HttpGet(raw.into()));
        assert_eq!(text, "http_get(<invalid url>)", "{raw}");
        assert!(!text.contains("SECRET"));
    }
}

#[test]
fn the_http_request_method_is_a_known_token_or_a_placeholder() {
    assert_eq!(
        render(WaitReason::HttpRequest("get".into(), "http://h/x".into())),
        "http_request(GET http://h)"
    );
    let hostile = render(WaitReason::HttpRequest(
        "GET SECRET".into(),
        "http://h/x".into(),
    ));
    assert_eq!(hostile, "http_request(? http://h)");
}

#[test]
fn non_http_operations_show_the_operation_name_only() {
    assert_eq!(
        render(WaitReason::FileRead("/home/u/SECRET_PATH".into())),
        "file_read"
    );
    assert_eq!(
        render(WaitReason::FileWrite("/home/u/SECRET_PATH".into())),
        "file_write"
    );
    assert_eq!(
        render(WaitReason::ShellExec(
            "curl -H 'Authorization: SECRET'".into()
        )),
        "shell"
    );
    assert_eq!(
        render(WaitReason::LlmTakeover("telegram:user:SECRET".into())),
        "llm_takeover"
    );
}

#[test]
fn terminal_states_carry_no_payload() {
    assert_eq!(
        render(WaitReason::Completed("return value SECRET".into())),
        "done"
    );
    assert_eq!(
        render(WaitReason::Failed(
            "error sending request for url (https://x/botSECRET)".into()
        )),
        "failed"
    );
}

#[test]
fn state_and_non_sensitive_progress_still_render() {
    assert_eq!(render(WaitReason::Running), "running");
    assert_eq!(render(WaitReason::Sleep(250)), "sleep(250ms)");
    assert_eq!(render(WaitReason::TcpListen(8080)), "tcp_listen(:8080)");
    assert_eq!(render(WaitReason::TcpAccept(3)), "tcp_accept(h3)");
    assert_eq!(render(WaitReason::TcpRead(4)), "tcp_read(h4)");
    assert_eq!(render(WaitReason::TcpWrite(5)), "tcp_write(h5)");
    assert_eq!(
        render(WaitReason::TcpConnect("db.internal".into(), 5432)),
        "tcp_connect(db.internal:5432)"
    );
    assert_eq!(render(WaitReason::LlmCall), "llm_call");
    assert_eq!(render(WaitReason::LlmAgent), "llm_agent");
    assert_eq!(render(WaitReason::StdinRead), "stdin_read");
}

#[test]
fn stored_values_stay_intact_for_callback_acknowledgement() {
    // Only rendering is redacted: the registry still holds the raw payloads.
    match WaitReason::Completed("value".into()) {
        WaitReason::Completed(value) => assert_eq!(value, "value"),
        _ => unreachable!(),
    }
    match WaitReason::HttpGet(TELEGRAM.into()) {
        WaitReason::HttpGet(url) => assert_eq!(url, TELEGRAM),
        _ => unreachable!(),
    }
}

fn registry_with(status: WaitReason) -> TaskRegistry {
    let registry: TaskRegistry = std::sync::Arc::new(std::sync::Mutex::new(HashMap::new()));
    registry.lock().unwrap().insert(
        7,
        TaskInfo {
            id: 7,
            function_name: "TelegramBot.poll_loop".into(),
            status,
            started_at: "2026-01-01T00:00:00Z".into(),
        },
    );
    registry
}

#[test]
fn query_tasks_builtin_and_snapshot_wait_reason_are_redacted() {
    let registry = registry_with(WaitReason::HttpGet(TELEGRAM.into()));
    let (tx, _) = mpsc::channel(1);
    let handle = CoroutineHandle {
        io_tx: tx,
        task_id: None,
        task_registry: Some(registry.clone()),
        snapshot_registry: None,
        mocks: Some(vec![]),
        stubs: None,
        turn: None,
        program_frozen: false,
    };
    let result = match handle.execute_await("query_tasks", &[]).unwrap() {
        Value::String(text) => text.as_ref().clone(),
        other => panic!("{other:?}"),
    };
    assert!(
        result.contains("task 7 [TelegramBot.poll_loop]")
            && result.contains("http_get(https://api.telegram.org)"),
        "{result}"
    );
    for secret in ["bot123456", "SECRET-TOKEN", "getUpdates", "offset"] {
        assert!(!result.contains(secret), "{secret}: {result}");
    }
    // The live snapshot copies the rendered status into `wait_reason`.
    let snapshots: TaskSnapshotRegistry = Default::default();
    let (tx, _) = mpsc::channel(1);
    let task = CoroutineHandle::new_with_task(tx, 7, registry, snapshots.clone());
    task.update_snapshot("TelegramBot.poll_loop", None, 1, vec![]);
    let reason = snapshots
        .lock()
        .unwrap()
        .get(&7)
        .unwrap()
        .wait_reason
        .clone();
    assert_eq!(reason, "http_get(https://api.telegram.org)");
}

#[test]
fn debug_is_the_redacted_display_for_every_variant() {
    let all = [
        WaitReason::Running,
        WaitReason::TcpListen(80),
        WaitReason::TcpAccept(1),
        WaitReason::TcpRead(2),
        WaitReason::TcpWrite(3),
        WaitReason::TcpConnect("db.internal".into(), 5432),
        WaitReason::FileRead("/home/u/SECRET_PATH".into()),
        WaitReason::FileWrite("/home/u/SECRET_PATH".into()),
        WaitReason::ShellExec("curl --token SECRET_CMD".into()),
        WaitReason::Sleep(5),
        WaitReason::LlmCall,
        WaitReason::LlmAgent,
        WaitReason::StdinRead,
        WaitReason::HttpGet(TELEGRAM.into()),
        WaitReason::HttpPost(TELEGRAM.into()),
        WaitReason::HttpRequest("GET".into(), TELEGRAM.into()),
        WaitReason::LlmTakeover("telegram:user:SECRET".into()),
        WaitReason::Completed("SECRET_RESULT".into()),
        WaitReason::Failed("failed for https://x/botSECRET_ERR".into()),
    ];
    for reason in all {
        let debug = format!("{reason:?}");
        assert_eq!(
            debug,
            format!("{reason}"),
            "Debug must be the redacted Display"
        );
        for secret in ["SECRET", "bot123456", "getUpdates", "offset", "/home/u"] {
            assert!(!debug.contains(secret), "{secret} leaked in {debug}");
        }
        // Containers that derive Debug over a WaitReason inherit the redaction.
        let info = TaskInfo {
            id: 1,
            function_name: "Mod.f".into(),
            status: reason.clone(),
            started_at: "t".into(),
        };
        let nested = format!("{info:?}") + &format!("{:#?}", info);
        for secret in ["SECRET", "bot123456", "getUpdates", "offset", "/home/u"] {
            assert!(!nested.contains(secret), "{secret} leaked in {nested}");
        }
    }
}
