//! Polling slice B: poll_once. `http_get`, `sleep` and `+spawn` are intercepted as
//! IoRequests on the normal async path; process_update never runs, nothing sleeps for
//! real and no network/file IO exists. Spawn gives no outcome (#104).
use crate::{
    ast,
    coroutine::{CoroutineHandle, IoRequest},
    eval::{self, Value},
    parser, shared_state,
};
use serde_json::{json, Value as Json};

const TESTS: [&str; 4] = [
    "telegram_port_poll_once_valid_pages_urls_and_sequence",
    "telegram_port_poll_once_rejected_bodies_back_off_without_dispatch",
    "telegram_port_poll_once_transport_errors_and_sleep_failure",
    "telegram_port_poll_once_skipped_updates_and_terminal_ids",
];

pub(in super::super) fn verify_repair(original: &ast::FunctionDecl, corrected: &ast::FunctionDecl) {
    use ast::{Expr, StatementKind as S, Type};
    assert!(original.params.is_empty() && corrected.params.is_empty());
    assert_eq!(original.return_type, corrected.return_type);
    let set = |effects: &[ast::Effect]| {
        let mut names: Vec<_> = effects.iter().map(|e| format!("{e:?}")).collect();
        names.sort();
        names
    };
    assert_eq!(set(&original.effects), set(&corrected.effects));
    let S::Match {
        expr: Expr::Call(old_call),
        ..
    } = &original.body[1].kind
    else {
        panic!("original match moved")
    };
    let S::Await { name, ty, call } = &corrected.body[1].kind else {
        panic!("explicit Result binding missing")
    };
    assert_eq!(name, "response");
    assert_eq!(*ty, Type::Result(Box::new(Type::String)));
    assert_eq!(call, old_call, "http_get call and URL argument preserved");
    let S::Match { arms, expr } = &corrected.body[2].kind else {
        panic!("corrected match moved")
    };
    assert_eq!(*expr, Expr::Identifier("response".into()));
    let ok = &arms[0].body;
    assert_eq!(ok.len(), 5, "guard, dispatch, guard, guard, return");
    for (index, text) in [
        (0, "poll rejected: invalid body"),
        (2, "poll rejected: no result"),
        (3, "poll rejected: no valid updates"),
    ] {
        let S::Branch {
            then_body,
            else_body,
            ..
        } = &ok[index].kind
        else {
            panic!("guard {index} missing")
        };
        assert!(else_body.is_empty());
        let S::Await { call, .. } = &then_body[0].kind else {
            panic!("backoff sleep missing")
        };
        assert_eq!(call.callee, "sleep");
        assert!(serde_json::to_string(&call.args).unwrap().contains("3000"));
        assert!(serde_json::to_string(&then_body[1]).unwrap().contains(text));
    }
    // Undo ONLY the ledgered transformations and prove the rest (URL, Err arm, the
    // original dispatch await and "ok" return) equals the frozen function modulo the
    // single local-call qualification.
    let mut restored = corrected.clone();
    restored.body.remove(1);
    let S::Match { expr, arms } = &mut restored.body[1].kind else {
        panic!()
    };
    *expr = Expr::Call(old_call.clone());
    let mut dispatch = ok[1].clone();
    let S::Await { name, ty, .. } = &mut dispatch.kind else {
        panic!("dispatch await moved")
    };
    *name = "_".into();
    *ty = Type::String;
    arms[0].body = vec![dispatch, ok[4].clone()];
    restored.effects = original.effects.clone();
    restored.doc = original.doc.clone();
    let locals = ["dispatch_updates".to_owned()].into_iter().collect();
    assert_eq!(
        super::super::qualification::verify(original, &restored, "TelegramBot", &locals),
        1
    );
    let manifest: Json = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/telegram-python-port/manifest.json"
    ))
    .unwrap();
    for test in TESTS {
        assert_eq!(
            manifest["modules"][0]["contract_tests"][test],
            json!(["poll_once"]),
            "{test}"
        );
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Ev {
    Get(i64),
    Sleep(i64),
    Spawn(String),
}

fn get(offset: i64) -> Ev {
    Ev::Get(offset)
}
fn spawn(update: &Json) -> Ev {
    Ev::Spawn(update.to_string())
}

type Reply = Result<String, &'static str>;

struct Polled {
    results: Vec<Result<String, String>>,
    events: Vec<Ev>,
    offset: i64,
}

fn collect(
    mut receiver: tokio::sync::mpsc::Receiver<IoRequest>,
    mut replies: std::collections::VecDeque<Reply>,
    sleep_fails: bool,
) -> Vec<Ev> {
    let mut events = Vec::new();
    while let Some(request) = receiver.blocking_recv() {
        match request {
            IoRequest::HttpGet { url, reply } => {
                let prefix = "<redacted>fixture-token/getUpdates?timeout=30&offset=";
                let offset = url
                    .strip_prefix(prefix)
                    .unwrap_or_else(|| panic!("unexpected getUpdates URL {url}"));
                events.push(get(offset.parse().expect("integer offset")));
                let next = replies.pop_front().expect("unexpected extra getUpdates");
                reply.send(next.map_err(anyhow::Error::msg)).unwrap();
            }
            IoRequest::Sleep { ms, reply } => {
                events.push(Ev::Sleep(ms as i64));
                reply
                    .send(if sleep_fails {
                        Err(anyhow::anyhow!("fixture sleep failure"))
                    } else {
                        Ok(())
                    })
                    .unwrap();
            }
            IoRequest::Spawn {
                function_name,
                args,
                reply,
            } => {
                assert_eq!(function_name, "TelegramBot.process_update");
                drop(reply);
                let [Value::String(update)] = args.as_slice() else {
                    panic!("spawn takes exactly one String")
                };
                events.push(Ev::Spawn(update.to_string()));
            }
            _ => panic!("poll_once may only get, sleep and spawn"),
        }
    }
    assert!(replies.is_empty(), "scripted reply was never requested");
    events
}

fn poll_n(program: &ast::Program, offset: i64, replies: Vec<Reply>, sleep_fails: bool) -> Polled {
    let _guard = super::super::routing::install(program, "1", "7");
    let runtime = shared_state::get_shared_runtime().unwrap();
    runtime
        .read()
        .unwrap()
        .shared_vars
        .insert("TelegramBot.poll_offset".into(), Value::Int(offset));
    let count = replies.len().max(1);
    let (sender, receiver) = tokio::sync::mpsc::channel(256);
    let queue = replies.into();
    let worker = std::thread::spawn(move || collect(receiver, queue, sleep_fails));
    let mut results = Vec::new();
    for _ in 0..count {
        results.push(
            eval::eval_async_function(
                program,
                "TelegramBot.poll_once",
                &parser::parse_single_expr("{}").unwrap(),
                CoroutineHandle::new(sender.clone()),
            )
            .map(|value| match value {
                Value::String(text) => text.to_string(),
                other => panic!("poll_once must return a String, got {other:?}"),
            })
            .map_err(|error| error.to_string()),
        );
        // A failed cycle ends the scripted sequence, as it would end the loop.
        if results.last().unwrap().is_err() {
            break;
        }
    }
    drop(sender);
    let events = worker.join().unwrap();
    let offset = match runtime
        .read()
        .unwrap()
        .shared_vars
        .get("TelegramBot.poll_offset")
    {
        Some(Value::Int(value)) => value,
        other => panic!("poll_offset must stay Int, got {other:?}"),
    };
    Polled {
        results,
        events,
        offset,
    }
}

fn once(program: &ast::Program, offset: i64, reply: Reply) -> Polled {
    poll_n(program, offset, vec![reply], false)
}

fn body(updates: &[Json]) -> Reply {
    Ok(json!({"ok": true, "result": updates}).to_string())
}

fn update(id: i64) -> Json {
    json!({"update_id": id, "message": {"text": format!("t{id} ä😀")}})
}

fn text(polled: &Polled) -> String {
    assert_eq!(polled.results.len(), 1);
    polled.results[0]
        .clone()
        .expect("expected a returned String")
}

fn nested(depth: usize) -> String {
    format!("{}{}", "[".repeat(depth), "]".repeat(depth))
}

#[test]
fn telegram_port_poll_once_valid_pages_urls_and_sequence() {
    for program in super::super::send_chain::programs() {
        assert!(
            crate::vm::compile_function(
                program.get_function("TelegramBot.poll_once").unwrap(),
                &program
            )
            .is_err(),
            "shared access is not direct-VM coverage"
        );
        // Exact URL per offset, including Int extremes; a genuine empty page is an
        // immediate "ok": no backoff, no spawn, offset untouched.
        for offset in [0, 7, i64::MAX] {
            let polled = once(&program, offset, body(&[]));
            assert_eq!(text(&polled), "ok");
            assert_eq!(polled.events, [get(offset)]);
            assert_eq!(polled.offset, offset);
        }
        let page = [update(5), update(6)];
        let polled = once(&program, 0, body(&page));
        assert_eq!(text(&polled), "ok");
        assert_eq!(
            polled.events,
            [get(0), spawn(&page[0]), spawn(&page[1])],
            "one request, spawns in array order, no sleep"
        );
        assert_eq!(polled.offset, 7);
        // The offset is read fresh at the start of every cycle: the second URL
        // carries the post-dispatch value from shared state.
        let sequence = poll_n(
            &program,
            0,
            vec![body(&page), body(&[]), body(&[update(40)])],
            false,
        );
        assert_eq!(
            sequence.results,
            vec![Ok("ok".to_owned()), Ok("ok".into()), Ok("ok".into())]
        );
        assert_eq!(
            sequence.events,
            [
                get(0),
                spawn(&page[0]),
                spawn(&page[1]),
                get(7),
                get(7),
                spawn(&update(40))
            ]
        );
        assert_eq!(sequence.offset, 41);
    }
}

#[test]
fn telegram_port_poll_once_rejected_bodies_back_off_without_dispatch() {
    for program in super::super::send_chain::programs() {
        // 200 responses that are valid JSON but carry no result array (HttpGet does
        // not check status, so Telegram 401/409 error JSON arrives as Ok).
        for raw in [
            "{}",
            r#"{"ok":false,"error_code":401,"description":"Unauthorized"}"#,
            r#"{"ok":false,"error_code":409,"description":"Conflict"}"#,
            r#"{"result":null}"#,
            r#"{"result":{}}"#,
            r#"{"result":"[]"}"#,
            r#"{"result":5}"#,
            r#"{"result":true}"#,
            "[]",
            "null",
            "5",
            r#""x""#,
        ] {
            let polled = once(&program, 3, Ok(raw.to_owned()));
            assert_eq!(text(&polled), "poll rejected: no result", "{raw}");
            assert_eq!(polled.events, [get(3), Ev::Sleep(3000)], "{raw}");
            assert_eq!(polled.offset, 3, "{raw}");
        }
        // Not JSON at all: json_valid is false, so dispatch (and its terminal parse
        // error) is never reached and nothing is spawned.
        let truncated = r#"{"result":[{"update_id":"#;
        let deep = nested(129);
        let huge = "[".repeat(100_000);
        for raw in [
            "",
            "   ",
            "<html>502 bad gateway</html>",
            truncated,
            "{} x",
            "\u{feff}{}",
            "{\0}",
            "1e400",
            deep.as_str(),
            huge.as_str(),
        ] {
            let polled = once(&program, 3, Ok(raw.to_owned()));
            assert_eq!(text(&polled), "poll rejected: invalid body");
            assert_eq!(polled.events, [get(3), Ev::Sleep(3000)]);
            assert_eq!(polled.offset, 3);
        }
    }
}

#[test]
fn telegram_port_poll_once_transport_errors_and_sleep_failure() {
    for program in super::super::send_chain::programs() {
        for error in ["connect lost", "", "Zoë 😀"] {
            let polled = once(&program, 9, Err(error));
            assert_eq!(text(&polled), format!("poll error: {error}"));
            assert_eq!(polled.events, [get(9), Ev::Sleep(3000)]);
            assert_eq!(polled.offset, 9);
        }
        // A failing sleep is not caught: the error leaves poll_once after exactly
        // one request and one sleep, with no retry (slice C owns loop survival).
        for reply in [
            Err("connect lost"),
            Ok("<html></html>".to_owned()),
            Ok("{}".to_owned()),
        ] {
            let polled = poll_n(&program, 9, vec![reply], true);
            assert_eq!(polled.events, [get(9), Ev::Sleep(3000)]);
            assert!(polled.results[0]
                .clone()
                .unwrap_err()
                .contains("fixture sleep failure"));
            assert_eq!(polled.offset, 9);
        }
    }
}

#[test]
fn telegram_port_poll_once_skipped_updates_and_terminal_ids() {
    for program in super::super::send_chain::programs() {
        // A mixed page is still "ok": valid ids advance the offset past the skipped
        // update; only the valid one is spawned.
        let mixed = [json!({}), update(21)];
        let polled = once(&program, 0, body(&mixed));
        assert_eq!(text(&polled), "ok");
        assert_eq!(polled.events, [get(0), spawn(&mixed[1])]);
        assert_eq!(polled.offset, 22);
        // Only skipped updates never advance the offset, so Telegram would redeliver
        // them immediately: poll_once backs off instead of hot-looping.
        for page in [
            vec![json!({})],
            vec![json!({"update_id": "5"}), json!(5), json!("x")],
            vec![json!({"update_id": -1}), json!({"update_id": null})],
        ] {
            let polled = once(&program, 6, body(&page));
            assert_eq!(text(&polled), "poll rejected: no valid updates");
            assert_eq!(polled.events, [get(6), Ev::Sleep(3000)]);
            assert_eq!(polled.offset, 6);
        }
        // Terminal characterization (not a fix): an id beyond i64 aborts dispatch
        // AFTER the earlier update was advanced and spawned; no sleep, no catch.
        let huge = r#"{"result":[{"update_id":4},{"update_id":18446744073709551615}]}"#;
        let polled = once(&program, 0, Ok(huge.to_owned()));
        assert!(polled.results[0]
            .clone()
            .unwrap_err()
            .contains("outside signed 64-bit range"));
        assert_eq!(polled.events, [get(0), spawn(&json!({"update_id": 4}))]);
        assert_eq!(polled.offset, 5);
    }
}
