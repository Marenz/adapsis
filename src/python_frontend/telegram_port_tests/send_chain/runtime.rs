use crate::{
    ast,
    coroutine::{CoroutineHandle, IoRequest},
    eval::{self, Value},
    parser,
    session::RuntimeState,
};
mod reply_closure;

#[derive(Clone, Copy, Debug)]
enum Reply {
    Body(&'static str),
    Transport(&'static str),
}

#[derive(Debug, PartialEq)]
enum Outcome {
    Tagged(String, String),
    Error(String),
}

#[derive(Debug, PartialEq)]
struct Attempt {
    text: String,
    html: bool,
}

struct StateGuard;
impl Drop for StateGuard {
    fn drop(&mut self) {
        eval::set_shared_runtime(None);
        eval::set_shared_program(None);
    }
}

fn respond(
    mut receiver: tokio::sync::mpsc::Receiver<IoRequest>,
    replies: Vec<Reply>,
    destination: i64,
) -> Vec<Attempt> {
    let mut replies = replies.into_iter();
    let mut attempts = Vec::new();
    while let Some(request) = receiver.blocking_recv() {
        let IoRequest::HttpPost {
            url,
            body,
            content_type,
            reply,
        } = request
        else {
            panic!("only intercepted HttpPost is permitted")
        };
        assert_eq!(url, "<redacted>fixture-token/sendMessage");
        assert_eq!(content_type, "application/json");
        let body: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(body["chat_id"], destination);
        let html = if let Some(mode) = body.get("parse_mode") {
            assert_eq!(mode, "HTML");
            true
        } else {
            false
        };
        assert_eq!(body.as_object().unwrap().len(), if html { 3 } else { 2 });
        attempts.push(Attempt {
            text: body["text"].as_str().unwrap().into(),
            html,
        });
        let result = match replies.next().expect("unexpected extra attempt") {
            Reply::Body(body) => Ok(body.into()),
            Reply::Transport(error) => Err(anyhow::anyhow!(error)),
        };
        reply.send(result).unwrap();
    }
    assert!(
        replies.next().is_none(),
        "expected response was never requested"
    );
    attempts
}

fn normal(
    program: &ast::Program,
    target: &str,
    input: &str,
    replies: Vec<Reply>,
) -> (Outcome, Vec<Attempt>) {
    normal_destination(program, target, input, replies, 7)
}

fn normal_destination(
    program: &ast::Program,
    target: &str,
    input: &str,
    replies: Vec<Reply>,
    destination: i64,
) -> (Outcome, Vec<Attempt>) {
    crate::intern::set_display_interner(&program.shared_interner);
    let state = RuntimeState::default();
    state.shared_vars.insert(
        "TelegramBot.telegram_token".into(),
        Value::string("fixture-token"),
    );
    eval::set_shared_runtime(Some(std::sync::Arc::new(std::sync::RwLock::new(state))));
    eval::set_shared_program(Some(std::sync::Arc::new(program.clone())));
    let _guard = StateGuard;
    let (sender, receiver) = tokio::sync::mpsc::channel(8);
    let responder = std::thread::spawn(move || respond(receiver, replies, destination));
    // The production normal async entry, not a direct-tree diagnostic route.
    let result = eval::eval_async_function(
        program,
        target,
        &parser::parse_single_expr(input).unwrap(),
        CoroutineHandle::new(sender),
    );
    let outcome = match result {
        Ok(Value::Union { variant, payload }) => {
            let [Value::String(text)] = payload.as_slice() else {
                panic!("unexpected union payload")
            };
            Outcome::Tagged(
                crate::intern::resolve_display(variant),
                text.as_ref().clone(),
            )
        }
        Err(error) => Outcome::Error(error.to_string()),
        Ok(other) => {
            panic!("app outcome must remain a Union, never an unwrapped Result/string: {other:?}")
        }
    };
    (outcome, responder.join().unwrap())
}

fn input(chunks: &[&str], index: i64) -> String {
    let values = chunks
        .iter()
        .map(|text| serde_json::to_string(text).unwrap())
        .collect::<Vec<_>>()
        .join(",");
    let forged_last = serde_json::to_string(r#"{"ok":true}"#).unwrap();
    format!("{{chat_id:7,parts:list({values}),i:{index},last:{forged_last}}}")
}

fn assert_tag(outcome: Outcome, name: &str, payload: &str) {
    assert_eq!(outcome, Outcome::Tagged(name.into(), payload.into()));
}

#[test]
fn telegram_port_send_leaf_strict_runtime_bool_and_union_contract_both_origins() {
    for program in super::programs() {
        for (body, tag) in [
            (r#"{"ok":true}"#, "TgSent"),
            (r#"{ "ok" : true }"#, "TgSent"),
            (r#"{"ok":false}"#, "TgRejected"),
            (r#"{ "ok" : false }"#, "TgRejected"),
            (r#"{"ok":"true"}"#, "TgUncertain"),
            (r#"{"ok":"false"}"#, "TgUncertain"),
            (r#"{"ok":1}"#, "TgUncertain"),
            (r#"{"ok":0}"#, "TgUncertain"),
            (r#"{"ok":null}"#, "TgUncertain"),
            (r#"{}"#, "TgUncertain"),
            (r#"{"ok":[false]}"#, "TgUncertain"),
            (r#"{"ok":{"nested":true}}"#, "TgUncertain"),
            (r#"{"description":"\"ok\":true"}"#, "TgUncertain"),
            ("true", "TgUncertain"),
        ] {
            let (outcome, attempts) = normal(
                &program,
                "TelegramBot.send_message_part",
                "{chat_id:7,text:\"short\",html:true}",
                vec![Reply::Body(body)],
            );
            assert_tag(outcome, tag, body);
            assert_eq!(
                attempts,
                [Attempt {
                    text: "short".into(),
                    html: true
                }]
            );
        }
        let error = "reply lost after sending: {\"ok\":true}";
        let (outcome, attempts) = normal(
            &program,
            "TelegramBot.send_message_part",
            "{chat_id:7,text:\"short\",html:false}",
            vec![Reply::Transport(error)],
        );
        assert_tag(outcome, "TgUncertain", error);
        assert_eq!(
            attempts,
            [Attempt {
                text: "short".into(),
                html: false
            }]
        );
    }
}

#[test]
fn telegram_port_send_chain_attempt_order_and_terminal_failures_both_origins() {
    const YES: &str = r#"{"ok":true,"id":1}"#;
    const NEXT: &str = r#"{ "ok": true, "id": 2 }"#;
    const NO: &str = r#"{ "ok" : false, "description":"bad entities" }"#;
    const AGAIN: &str = r#"{"ok":false,"description":"plain rejected"}"#;
    const NONBOOLEAN: &str = r#"{"ok":"false"}"#;
    const LOST: &str = "uncertain delivery; error text contains {\"ok\":true}";
    for program in super::programs() {
        for (chunks, replies, modes, tag, payload) in [
            (
                vec!["first"],
                vec![Reply::Body(YES)],
                vec![("first", true)],
                "TgSent",
                YES,
            ),
            (
                vec!["first", "second"],
                vec![Reply::Body(YES), Reply::Body(NEXT)],
                vec![("first", true), ("second", true)],
                "TgSent",
                NEXT,
            ),
            (
                vec!["first", "second"],
                vec![Reply::Body(NO), Reply::Body(YES), Reply::Body(NEXT)],
                vec![("first", true), ("first", false), ("second", true)],
                "TgSent",
                NEXT,
            ),
            (
                vec!["first", "second"],
                vec![Reply::Transport(LOST)],
                vec![("first", true)],
                "TgUncertain",
                LOST,
            ),
            (
                vec!["first", "second"],
                vec![Reply::Body(NO), Reply::Transport(LOST)],
                vec![("first", true), ("first", false)],
                "TgUncertain",
                LOST,
            ),
            (
                vec!["first", "second"],
                vec![Reply::Body(NO), Reply::Body(AGAIN)],
                vec![("first", true), ("first", false)],
                "TgRejected",
                AGAIN,
            ),
            (
                vec!["first", "second", "third"],
                vec![Reply::Body(YES), Reply::Transport(LOST)],
                vec![("first", true), ("second", true)],
                "TgUncertain",
                LOST,
            ),
            (
                vec!["first", "second", "third"],
                vec![Reply::Body(YES), Reply::Body(NO), Reply::Body(AGAIN)],
                vec![("first", true), ("second", true), ("second", false)],
                "TgRejected",
                AGAIN,
            ),
            (
                vec!["first", "second"],
                vec![Reply::Body(NO), Reply::Body(NONBOOLEAN)],
                vec![("first", true), ("first", false)],
                "TgUncertain",
                NONBOOLEAN,
            ),
        ] {
            let (outcome, attempts) = normal(
                &program,
                "TelegramBot.send_message_parts",
                &input(&chunks, 0),
                replies,
            );
            assert_tag(outcome, tag, payload);
            assert_eq!(
                attempts,
                modes
                    .into_iter()
                    .map(|(text, html)| Attempt {
                        text: text.into(),
                        html
                    })
                    .collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn telegram_port_send_chain_nonboolean_and_malformed_never_resend() {
    for program in super::programs() {
        for body in [
            r#"{"ok":"false"}"#,
            r#"{"ok":"true"}"#,
            r#"{"ok":0}"#,
            r#"{"ok":1}"#,
            r#"{"ok":null}"#,
            r#"{}"#,
            r#"{"description":"\"ok\":false"}"#,
        ] {
            let (outcome, attempts) = normal(
                &program,
                "TelegramBot.send_message_parts",
                &input(&["first", "second"], 0),
                vec![Reply::Body(body)],
            );
            assert_tag(outcome, "TgUncertain", body);
            assert_eq!(
                attempts,
                [Attempt {
                    text: "first".into(),
                    html: true
                }]
            );
        }
        for replies in [
            vec![Reply::Body("not json")],
            vec![Reply::Body(r#"{"ok":false}"#), Reply::Body("not json")],
        ] {
            let count = replies.len();
            let (outcome, attempts) = normal(
                &program,
                "TelegramBot.send_message_parts",
                &input(&["first", "second"], 0),
                replies,
            );
            assert!(
                matches!(outcome, Outcome::Error(ref error) if error.starts_with("json_get: invalid JSON:")),
                "{outcome:?}"
            );
            assert_eq!(attempts.len(), count);
            assert!(attempts.iter().all(|attempt| attempt.text == "first"));
            assert!(attempts[0].html);
            if count == 2 {
                assert!(!attempts[1].html);
            }
        }
    }
}

#[test]
fn telegram_port_send_chain_empty_and_out_of_range_never_manufacture_success() {
    for program in super::programs() {
        for (chunks, index) in [
            (vec![], 0),
            (vec!["short"], -1),
            (vec!["short"], 1),
            (vec!["short"], 99),
        ] {
            let (outcome, attempts) = normal(
                &program,
                "TelegramBot.send_message_parts",
                &input(&chunks, index),
                vec![],
            );
            assert_tag(outcome, "TgUncertain", "no chunk at index");
            assert!(attempts.is_empty());
        }
    }
}
