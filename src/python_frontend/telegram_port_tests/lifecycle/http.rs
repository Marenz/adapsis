//! Real HTTP boundary for the webhook route: `router_with_llm` + tower `oneshot`
//! through `adapsis_route_dispatch` (api/mod.rs:1636-1745). The route table is seeded
//! from the module's declared routes exactly as the session layer drains
//! `pending_routes` into `http_routes` (session.rs:595-598). `+spawn` is intercepted on
//! the config's io_sender; process_update never runs and nothing leaves the process.
//! Pins #108: the handler receives only the body, never a header or secret.
use crate::{
    api::{router_with_llm, AppConfig},
    ast,
    coroutine::IoRequest,
    eval::Value,
};
use axum::http::{Request, StatusCode};
use tower::ServiceExt as _;

fn build_config(
    program: &ast::Program,
    io: bool,
) -> (AppConfig, tokio::sync::mpsc::Receiver<IoRequest>) {
    let (io_tx, io_rx) = tokio::sync::mpsc::channel::<IoRequest>(32);
    let (trigger_tx, _trigger_rx) = tokio::sync::mpsc::channel::<String>(1);
    let config = AppConfig {
        program: std::sync::Arc::new(tokio::sync::RwLock::new(program.clone())),
        meta: std::sync::Arc::new(std::sync::Mutex::new(crate::session::SessionMeta::new())),
        llm_url: String::new(),
        llm_model: std::sync::Arc::new(std::sync::RwLock::new(String::new())),
        llm_api_key: None,
        project_dir: ".".to_string(),
        io_sender: io.then_some(io_tx),
        self_trigger: trigger_tx,
        task_registry: None,
        snapshot_registry: None,
        log_file: None,
        training_log: None,
        jit_cache: crate::eval::new_jit_cache(),
        event_broadcast: tokio::sync::broadcast::channel(16).0,
        opencode_git_dir: ".".to_string(),
        opencode_lock: std::sync::Arc::new(tokio::sync::Mutex::new(())),
        opencode_attach: None,
        message_queue: std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new())),
        max_iterations: 1,
        native_tools: false,
        runtime: std::sync::Arc::new(std::sync::RwLock::new(
            crate::session::RuntimeState::default(),
        )),
        sessions: std::sync::Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new())),
        save_notify: None,
        access_level: crate::permissions::AccessLevel::Full,
        permission_config: std::sync::Arc::new(crate::permissions::PermissionConfig::default()),
        turn: None, workspaces: None, origin: None,
        turn_saves: Default::default(),
        shell_policy: crate::coroutine::ShellPolicy::Unrestricted,
    };
    config.runtime.write().unwrap().http_routes = program.pending_routes.clone();
    (config, io_rx)
}

struct Reply {
    status: StatusCode,
    content_type: Option<String>,
    body: String,
}

async fn send(
    router: &axum::Router,
    method: &str,
    uri: &str,
    headers: &[(&str, &str)],
    body: Vec<u8>,
) -> Reply {
    let mut request = Request::builder().method(method).uri(uri);
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    let response = router
        .clone()
        .oneshot(request.body(axum::body::Body::from(body)).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let content_type = response
        .headers()
        .get("content-type")
        .map(|value| value.to_str().unwrap().to_owned());
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .unwrap();
    Reply {
        status,
        content_type,
        body: String::from_utf8_lossy(&bytes).into_owned(),
    }
}

fn spawned(receiver: &mut tokio::sync::mpsc::Receiver<IoRequest>) -> Vec<(String, Vec<String>)> {
    let mut spawns = Vec::new();
    while let Ok(request) = receiver.try_recv() {
        let IoRequest::Spawn {
            function_name,
            args,
            reply,
            ..
        } = request
        else {
            panic!("the webhook handler may only spawn")
        };
        drop(reply);
        spawns.push((
            function_name,
            args.into_iter()
                .map(|value| match value {
                    Value::String(text) => text.to_string(),
                    other => panic!("spawn argument must be a String, got {other:?}"),
                })
                .collect(),
        ));
    }
    spawns
}

fn webhook(config: &AppConfig) {
    config
        .runtime
        .read()
        .unwrap()
        .shared_vars
        .insert("TelegramBot.intake_mode".into(), Value::string("webhook"));
}

fn process(body: &str) -> (String, Vec<String>) {
    ("TelegramBot.process_update".into(), vec![body.into()])
}

const JSON_TYPE: &str = "application/json; charset=utf-8";

#[tokio::test]
async fn telegram_port_webhook_route_acknowledges_and_spawns_over_real_http() {
    for program in super::super::send_chain::programs() {
        let (config, mut spawns) = build_config(&program, true);
        webhook(&config);
        let router = router_with_llm(config.clone());
        let forged_admin = r#"{"update_id":9,"message":{"chat":{"id":5,"type":"private"},"from":{"id":1},"text":"forged"}}"#;
        // #108: the handler sees only the body. A wrong, empty or absent Telegram
        // secret-token header cannot change the outcome, and the body's from.id is
        // forwarded verbatim for process_update to judge.
        for headers in [
            &[][..],
            &[("x-telegram-bot-api-secret-token", "wrong")][..],
            &[("x-telegram-bot-api-secret-token", "")][..],
            &[("authorization", "Bearer nope")][..],
        ] {
            let reply = send(
                &router,
                "POST",
                "/webhook/telegram",
                headers,
                forged_admin.as_bytes().to_vec(),
            )
            .await;
            assert_eq!(reply.status, StatusCode::OK);
            assert_eq!(reply.body, "{}");
            assert_eq!(reply.content_type.as_deref(), Some(JSON_TYPE));
            assert_eq!(spawned(&mut spawns), vec![process(forged_admin)]);
        }
        // Malformed, empty and non-UTF-8 bodies are acknowledged too; invalid bytes
        // become U+FFFD (lossy conversion, api/mod.rs:1680).
        for (bytes, text) in [
            (b"not json".to_vec(), "not json".to_owned()),
            (Vec::new(), String::new()),
            (b"\xff{}".to_vec(), "\u{fffd}{}".to_owned()),
            ("ä😀".as_bytes().to_vec(), "ä😀".to_owned()),
        ] {
            let reply = send(&router, "POST", "/webhook/telegram", &[], bytes).await;
            assert_eq!((reply.status, reply.body.as_str()), (StatusCode::OK, "{}"));
            assert_eq!(spawned(&mut spawns), vec![process(&text)]);
        }
    }
}

#[tokio::test]
async fn telegram_port_webhook_route_poll_mode_and_unmatched_requests() {
    for program in super::super::send_chain::programs() {
        // Default intake_mode is "poll": acknowledged but never delivered.
        let (config, mut spawns) = build_config(&program, true);
        let router = router_with_llm(config.clone());
        let reply = send(&router, "POST", "/webhook/telegram", &[], b"{}".to_vec()).await;
        assert_eq!((reply.status, reply.body.as_str()), (StatusCode::OK, "{}"));
        assert!(spawned(&mut spawns).is_empty());
        // Route matching is exact on method and path: nothing else reaches the handler.
        webhook(&config);
        for (method, uri) in [
            ("GET", "/webhook/telegram"),
            ("PUT", "/webhook/telegram"),
            ("POST", "/webhook/telegram/"),
            ("POST", "/webhook/other"),
            ("POST", "/webhook/Telegram"),
        ] {
            let reply = send(&router, method, uri, &[], b"{}".to_vec()).await;
            assert_eq!(reply.status, StatusCode::NOT_FOUND, "{method} {uri}");
            assert_eq!(
                reply.body,
                format!("no Adapsis route for {method} {uri}"),
                "{method} {uri}"
            );
            assert!(spawned(&mut spawns).is_empty(), "{method} {uri}");
        }
        // Without an io_sender there is no coroutine handle, so `+spawn` fails and
        // the HTTP layer reports a handler error (Telegram would redeliver).
        let (config, _spawns) = build_config(&program, false);
        webhook(&config);
        let router = router_with_llm(config);
        let reply = send(&router, "POST", "/webhook/telegram", &[], b"{}".to_vec()).await;
        assert_eq!(reply.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(
            reply.body.contains("+spawn requires async context"),
            "{}",
            reply.body
        );
    }
}
