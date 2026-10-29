//! Callable outcome propagation only; generic callback acknowledgement (#104)
//! is intentionally not exercised or certified by these normal-async fixtures.
use super::{normal_destination, Attempt, Outcome, Reply};

fn input(reply: bool, destination: &str, text: &str) -> String {
    let text = serde_json::to_string(text).unwrap();
    if reply {
        format!(
            "{{chat_id_str:{},text:{text}}}",
            serde_json::to_string(destination).unwrap()
        )
    } else {
        format!("{{chat_id:{destination},text:{text}}}")
    }
}

#[test]
fn telegram_port_reply_closure_formats_and_propagates_without_caller_retry() {
    const YES: &str = r#"{ "ok": true, "result": {"message_id": 1} }"#;
    const NO: &str = r#"{"ok":false,"description":"rejected"}"#;
    const LOST: &str = "reply lost after sending: {\"ok\":true}";
    for program in super::super::programs() {
        for (target, reply) in [
            ("TelegramBot.send_message", false),
            ("TelegramBot.send_reply", true),
        ] {
            for (replies, tag, payload, modes) in [
                (vec![Reply::Body(YES)], "TgSent", YES, vec![true]),
                (
                    vec![Reply::Body(NO), Reply::Body(YES)],
                    "TgSent",
                    YES,
                    vec![true, false],
                ),
                (
                    vec![Reply::Transport(LOST)],
                    "TgUncertain",
                    LOST,
                    vec![true],
                ),
                (
                    vec![Reply::Body(NO), Reply::Body(NO)],
                    "TgRejected",
                    NO,
                    vec![true, false],
                ),
                (
                    vec![Reply::Body(NO), Reply::Transport(LOST)],
                    "TgUncertain",
                    LOST,
                    vec![true, false],
                ),
            ] {
                let (outcome, attempts) = normal_destination(
                    &program,
                    target,
                    &input(reply, "-123", "**hi**\n- ä"),
                    replies,
                    -123,
                );
                assert_eq!(outcome, Outcome::Tagged(tag.into(), payload.into()));
                assert_eq!(
                    attempts,
                    modes
                        .into_iter()
                        .map(|html| Attempt {
                            text: "<b>hi</b>\n• ä".into(),
                            html
                        })
                        .collect::<Vec<_>>()
                );
            }
            let (outcome, attempts) = normal_destination(
                &program,
                target,
                &input(reply, "-123", "short"),
                vec![Reply::Body("not json")],
                -123,
            );
            assert!(
                matches!(outcome, Outcome::Error(ref error) if error.starts_with("json_get: invalid JSON:"))
            );
            assert_eq!(
                attempts,
                [Attempt {
                    text: "short".into(),
                    html: true
                }]
            );
        }
    }
}

#[test]
fn telegram_port_reply_closure_empty_and_bad_destination_emit_no_requests() {
    for program in super::super::programs() {
        for (target, reply) in [
            ("TelegramBot.send_message", false),
            ("TelegramBot.send_reply", true),
        ] {
            let (outcome, attempts) =
                normal_destination(&program, target, &input(reply, "7", ""), vec![], 7);
            assert_eq!(
                outcome,
                Outcome::Tagged("TgUncertain".into(), "no chunk at index".into())
            );
            assert!(attempts.is_empty());
        }
        let (outcome, attempts) = normal_destination(
            &program,
            "TelegramBot.send_reply",
            &input(true, "not-a-chat", "short"),
            vec![],
            7,
        );
        assert!(
            matches!(outcome, Outcome::Error(ref error) if error == "to_int: cannot parse 'not-a-chat' as integer")
        );
        assert!(attempts.is_empty());
        let body = r#"{"ok":true}"#;
        let (outcome, attempts) = normal_destination(
            &program,
            "TelegramBot.send_reply",
            &input(true, "0007", "hi"),
            vec![Reply::Body(body)],
            7,
        );
        assert_eq!(outcome, Outcome::Tagged("TgSent".into(), body.into()));
        assert_eq!(
            attempts,
            [Attempt {
                text: "hi".into(),
                html: true
            }]
        );
    }
}
