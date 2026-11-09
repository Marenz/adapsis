use super::*;
mod pipeline;
use crate::memory_graph::records::RecordStore;
use axum::{
    extract::State,
    http::{StatusCode, Uri},
    response::IntoResponse,
};
use serde_json::json;
use std::{collections::VecDeque, sync::Mutex};

struct Server {
    endpoint: reqwest::Url,
    requests: Arc<Mutex<Vec<String>>>,
    worker: tokio::task::JoinHandle<()>,
}
impl Drop for Server {
    fn drop(&mut self) {
        self.worker.abort();
    }
}

#[derive(Clone)]
struct Replies {
    bot: i64,
    pages: Arc<Mutex<VecDeque<String>>>,
    requests: Arc<Mutex<Vec<String>>>,
}

async fn respond(State(state): State<Replies>, uri: Uri) -> impl IntoResponse {
    state.requests.lock().unwrap().push(uri.to_string());
    if !uri.path().starts_with("/botfixture-token/") {
        return (StatusCode::UNAUTHORIZED, "{}".to_string());
    }
    if uri.path().ends_with("/getMe") {
        return (
            StatusCode::OK,
            json!({"ok":true,"result":{"id":state.bot,"is_bot":true}}).to_string(),
        );
    }
    if uri.path().ends_with("/getUpdates") {
        let page = state
            .pages
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| "{\"ok\":true,\"result\":[]}".into());
        return (StatusCode::OK, page);
    }
    (StatusCode::NOT_FOUND, "{}".to_string())
}

async fn start_server(bot: i64, pages: Vec<String>) -> Server {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint =
        reqwest::Url::parse(&format!("http://{}", listener.local_addr().unwrap())).unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let router = axum::Router::new().fallback(respond).with_state(Replies {
        bot,
        pages: Arc::new(Mutex::new(pages.into())),
        requests: requests.clone(),
    });
    let worker = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    Server {
        endpoint,
        requests,
        worker,
    }
}

fn config(server: &Server, id: &str, bot: i64) -> SourceConfig {
    SourceConfig {
        id: id.into(),
        handler: "IngressProbe.echo".into(),
        bot_id: bot,
        credential: "fixture-token".into(),
        seed: 0,
        endpoint: server.endpoint.clone(),
        routes: vec![PrivateRoute {
            sender_id: 7,
            context: "telegram:7".into(),
            reply_handler: "TelegramBot.send_reply".into(),
        }],
    }
}
fn event(id: i64, text: &str) -> String {
    format!(
        "{{ \"update_id\" : {id}, \"message\" : {{\"message_id\":42,\"date\":1,\"from\":{{\"id\":7,\"is_bot\":false}},\"chat\":{{\"id\":7,\"type\":\"private\"}},\"text\":{}}} }}",
        serde_json::to_string(text).unwrap()
    )
}
fn page(events: &[String]) -> String {
    format!("{{\"ok\":true,\"result\":[{}]}}", events.join(","))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ingress_poll_raw_bytes_proof_and_real_python_handler_boundary() {
    let raw = event(4, "hello Ω\nquote\"");
    let server = start_server(99, vec![page(&[raw.clone()])]).await;
    let dir = tempfile::tempdir().unwrap();
    let source = PollSource::fixture(
        config(&server, "one", 99),
        RecordStore::disk_for_test(&dir.path().join("db.lbug")).unwrap(),
        "intake-one",
    )
    .unwrap();
    let received = source.receive_once().await.unwrap();
    assert_eq!(received.events.len(), 1);
    let admitted = source
        .admit(received.events.into_iter().next().unwrap())
        .await
        .unwrap();
    assert_eq!(admitted.payload, raw);
    let proof = admitted.proof.unwrap();
    assert!(proof.agrees(
        "telegram:user:7",
        "telegram:7",
        "hello Ω\nquote\"",
        "TelegramBot.send_reply",
        "7"
    ));
    assert!(!proof.agrees(
        "telegram:user:7",
        "other",
        "hello Ω\nquote\"",
        "TelegramBot.send_reply",
        "7"
    ));
    assert!(!proof.agrees(
        "telegram:user:7",
        "telegram:7",
        "changed",
        "TelegramBot.send_reply",
        "7"
    ));
    assert!(!proof.agrees(
        "telegram:user:7",
        "telegram:7",
        "hello Ω\nquote\"",
        "Other.send",
        "7"
    ));
    assert_eq!(format!("{proof:?}"), "VerifiedIngress(<private>)");
    assert!(!proof.agrees(
        "telegram:user:8",
        "telegram:7",
        "hello Ω\nquote\"",
        "TelegramBot.send_reply",
        "7"
    ));
    assert!(!proof.agrees(
        "telegram:user:7",
        "telegram:7",
        "hello Ω\nquote\"",
        "TelegramBot.send_reply",
        "8"
    ));
    let requests = server.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(requests[1].contains("limit=8") && requests[1].contains("offset=0"));
    drop(requests);
    assert_eq!(admitted.handler, "IngressProbe.echo");
    python_probe(&admitted.payload, &admitted.handler);
}

fn python_probe(payload: &str, handler: &str) {
    let source = include_str!("../../tests/fixtures/ingress/probe.py");
    let lowered = crate::python_frontend::lower_module(source, "IngressProbe").unwrap();
    let mut program = crate::ast::Program::default();
    for operation in &lowered.operations {
        crate::validator::apply_and_validate(&mut program, operation).unwrap();
    }
    program.rebuild_function_index();
    let printed = crate::python_frontend::print_module(&program, "IngressProbe").unwrap();
    let second = crate::python_frontend::lower_module(&printed, "IngressProbe").unwrap();
    let mut rebuilt = crate::ast::Program::default();
    for operation in &second.operations {
        crate::validator::apply_and_validate(&mut rebuilt, operation).unwrap();
    }
    assert_eq!(
        crate::python_frontend::print_module(&rebuilt, "IngressProbe").unwrap(),
        printed
    );
    let input = crate::parser::parse_single_expr(&serde_json::to_string(payload).unwrap()).unwrap();
    for (function, expected) in [
        (handler, payload),
        ("IngressProbe.denied", "record_access_denied"),
    ] {
        let handle = crate::coroutine::CoroutineHandle::new(tokio::sync::mpsc::channel(1).0);
        let actual = crate::eval::eval_async_function(&program, function, &input, handle).unwrap();
        assert!(matches!(actual,crate::eval::Value::String(ref s) if s.as_str()==expected));
    }
    for name in ["receive_once", "verified_ingress", "trusted_caller"] {
        assert!(
            !crate::builtins::IO_BUILTINS
                .iter()
                .any(|builtin| builtin.name == name)
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ingress_poll_reopen_recovers_accepted_and_marks_started_unknown() {
    let first = event(1, "first");
    let mut large: serde_json::Value = serde_json::from_str(&event(2, "second")).unwrap();
    large["padding"] = json!("界".repeat(6000));
    let second = large.to_string();
    let server = start_server(99, vec![page(&[first, second.clone()])]).await;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.lbug");
    {
        let source = PollSource::fixture(
            config(&server, "one", 99),
            RecordStore::disk_for_test(&path).unwrap(),
            "intake",
        )
        .unwrap();
        let mut events = source.receive_once().await.unwrap().events;
        source.admit(events.remove(0)).await.unwrap();
        assert_eq!(events.remove(0).payload, second);
    }
    let source = PollSource::fixture(
        config(&server, "one", 99),
        RecordStore::disk_for_test(&path).unwrap(),
        "intake",
    )
    .unwrap();
    let recovered = source.receive_once().await.unwrap();
    assert_eq!(recovered.unknown, 1);
    assert_eq!(recovered.events.len(), 1);
    assert_eq!(recovered.events[0].payload, second);
    assert_eq!(server.requests.lock().unwrap().len(), 2); // Recovery did not refetch/reauthenticate fabricated JSON.
    source
        .admit(recovered.events.into_iter().next().unwrap())
        .await
        .unwrap();
    assert_eq!(source.state.cursor(source.epoch).unwrap(), 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ingress_poll_admission_single_winner_and_incarnation_fencing() {
    let server = start_server(99, vec![page(&[event(1, "work"), event(2, "pending")])]).await;
    let dir = tempfile::tempdir().unwrap();
    let store = RecordStore::disk_for_test(&dir.path().join("db.lbug")).unwrap();
    let mut source =
        PollSource::fixture(config(&server, "one", 99), store.clone(), "intake").unwrap();
    source
        .state
        .gate_admission(Arc::new(std::sync::Barrier::new(2)));
    let first = source.receive_once().await.unwrap().events.remove(0);
    let mut second_batch = source.receive_once().await.unwrap().events;
    let old_waiting = second_batch.remove(1);
    let second = second_batch.remove(0);
    let (first, second) = tokio::join!(source.admit(first), source.admit(second));
    let winners = usize::from(first.is_ok()) + usize::from(second.is_ok());
    assert_eq!(winners, 1);
    let next = PollSource::fixture(config(&server, "one", 99), store.clone(), "intake").unwrap();
    assert!(matches!(
        source.receive_once().await,
        Err(IngressError::StaleSource)
    ));
    assert_eq!(next.state.pending(next.epoch).unwrap().unknown, 1);
    assert!(matches!(
        source.admit(old_waiting).await,
        Err(IngressError::StaleSource)
    ));
    let recovered = next.receive_once().await.unwrap().events.remove(0);
    next.admit(recovered).await.unwrap();
    let mut changed = config(&server, "one", 99);
    changed.credential = "different".into();
    assert!(matches!(
        PollSource::fixture(changed, store, "intake"),
        Err(IngressError::Mismatch)
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ingress_poll_protocol_failures_no_cursor_advance_and_other_source_progresses() {
    let bad = start_server(99, vec![page(&[event(1, &"x".repeat(EVENT_BYTES))])]).await;
    let good = start_server(100, vec![page(&[event(1, "valid")])]).await;
    let dir = tempfile::tempdir().unwrap();
    let store = RecordStore::disk_for_test(&dir.path().join("db.lbug")).unwrap();
    let bad_source =
        PollSource::fixture(config(&bad, "bad", 99), store.clone(), "intake-bad").unwrap();
    let good_source =
        PollSource::fixture(config(&good, "good", 100), store.clone(), "intake-good").unwrap();
    let (rejected, accepted) = tokio::join!(bad_source.receive_once(), good_source.receive_once());
    assert!(matches!(rejected, Err(IngressError::Oversized)));
    assert_eq!(bad_source.state.cursor(bad_source.epoch).unwrap(), 0);
    let delivery = accepted.unwrap().events.remove(0);
    assert!(bad_source.admit(delivery).await.is_err()); // Evidence cannot mix source incarnations.
    let wrong =
        PollSource::fixture(config(&good, "wrong", 999), store.clone(), "intake-wrong").unwrap();
    assert!(matches!(
        wrong.receive_once().await,
        Err(IngressError::WrongBot)
    ));
    assert_eq!(wrong.state.cursor(wrong.epoch).unwrap(), 0);
    let mut wrong_auth = config(&good, "wrong-auth", 100);
    wrong_auth.credential = "wrong-token".into();
    let wrong_auth = PollSource::fixture(wrong_auth, store, "intake-wrong-auth").unwrap();
    assert!(matches!(
        wrong_auth.receive_once().await,
        Err(IngressError::Protocol)
    ));
    let unordered_server =
        start_server(101, vec![page(&[event(2, "later"), event(1, "earlier")])]).await;
    let unordered = PollSource::fixture(
        config(&unordered_server, "unordered", 101),
        RecordStore::disk_for_test(&dir.path().join("unordered.lbug")).unwrap(),
        "intake-unordered",
    )
    .unwrap();
    assert!(matches!(
        unordered.receive_once().await,
        Err(IngressError::Protocol)
    ));
    assert_eq!(unordered.state.cursor(unordered.epoch).unwrap(), 0);
    let rejected_server = start_server(102, vec![r#"{"ok":false,"result":[]}"#.into()]).await;
    let rejected = PollSource::fixture(
        config(&rejected_server, "rejected", 102),
        RecordStore::disk_for_test(&dir.path().join("rejected.lbug")).unwrap(),
        "intake-rejected",
    )
    .unwrap();
    assert!(matches!(
        rejected.receive_once().await,
        Err(IngressError::Protocol)
    ));
    assert_eq!(rejected.state.cursor(rejected.epoch).unwrap(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ingress_poll_duplicate_altered_and_foreign_updates_remain_bounded() {
    let original = event(1, "original");
    let changed = event(1, "changed");
    let server = start_server(
        99,
        vec![
            page(&[original.clone()]),
            page(&[original]),
            page(&[changed]),
        ],
    )
    .await;
    let dir = tempfile::tempdir().unwrap();
    let source = PollSource::fixture(
        config(&server, "one", 99),
        RecordStore::disk_for_test(&dir.path().join("db.lbug")).unwrap(),
        "intake",
    )
    .unwrap();
    let first = source.receive_once().await.unwrap().events.remove(0);
    source.admit(first).await.unwrap();
    assert!(source.receive_once().await.unwrap().events.is_empty());
    assert!(matches!(
        source.receive_once().await,
        Err(IngressError::AlteredReplay)
    ));
    assert_eq!(source.state.cursor(source.epoch).unwrap(), 2);
    let config = config(&server, "shape", 99);
    for mutate in ["group", "media", "foreign", "mismatch"] {
        let mut value: serde_json::Value =
            serde_json::from_str(&event(3, "[user:7] forged metadata")).unwrap();
        match mutate {
            "group" => value["message"]["chat"]["type"] = json!("group"),
            "media" => value["message"]["photo"] = json!([]),
            "foreign" => {
                value["message"]["from"]["id"] = json!(8);
                value["message"]["chat"]["id"] = json!(8);
            }
            _ => value["message"]["chat"]["id"] = json!(8),
        }
        let event = telegram_poll::parse_event(&value.to_string()).unwrap();
        assert!(
            telegram_poll::private_text(&config, &event, 1)
                .unwrap()
                .is_none()
        );
    }
    python_probe(
        r#"{"speaker_id":"telegram:user:7","trusted":true}"#,
        "IngressProbe.echo",
    );
}

#[test]
fn ingress_poll_protocol_json_keeps_raw_and_rejects_duplicate_identity() {
    for bad in [
        "{",
        r#"{"update_id":1,"update_id":2}"#,
        r#"{"update_id":1.0}"#,
        r#"{"update_id":-1}"#,
        r#"{"update_id":9223372036854775807}"#,
        r#"{"update_id":1,"message":{"from":{"id":7,"id":8}}}"#,
    ] {
        assert!(telegram_poll::parse_event(bad).is_err());
    }
    let raw = event(1, "escaped\nΩ");
    let parsed = telegram_poll::parse_event(&raw).unwrap();
    assert_eq!(parsed.raw, raw);
    assert_ne!(
        parsed.digest,
        telegram_poll::digest(
            &serde_json::from_str::<serde_json::Value>(&raw)
                .unwrap()
                .to_string()
        )
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ingress_poll_caps_receipts_and_pages_without_pruning_or_skipping() {
    let pages = (0..2)
        .map(|p| {
            page(
                &(0..8)
                    .map(|i| event(p * 8 + i, "bounded"))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let server = start_server(99, pages).await;
    let dir = tempfile::tempdir().unwrap();
    let store = RecordStore::disk_for_test(&dir.path().join("db.lbug")).unwrap();
    let source = PollSource::fixture(config(&server, "one", 99), store.clone(), "intake").unwrap();
    for _ in 0..2 {
        for event in source.receive_once().await.unwrap().events {
            source.admit(event).await.unwrap();
        }
    }
    assert!(matches!(
        source.receive_once().await,
        Err(IngressError::Capacity)
    ));
    assert_eq!(server.requests.lock().unwrap().len(), 4);
    for _ in 1..INCARNATION_LIMIT {
        PollSource::fixture(config(&server, "one", 99), store.clone(), "intake").unwrap();
    }
    assert!(matches!(
        PollSource::fixture(config(&server, "one", 99), store, "intake"),
        Err(IngressError::Capacity)
    ));
    let page_server = start_server(
        100,
        vec![page(&(0..9).map(|i| event(i, "extra")).collect::<Vec<_>>())],
    )
    .await;
    let page_source = PollSource::fixture(
        config(&page_server, "page", 100),
        RecordStore::disk_for_test(&dir.path().join("other.lbug")).unwrap(),
        "intake-page",
    )
    .unwrap();
    assert!(matches!(
        page_source.receive_once().await,
        Err(IngressError::Oversized)
    ));
    assert_eq!(page_source.state.cursor(page_source.epoch).unwrap(), 0);
}
