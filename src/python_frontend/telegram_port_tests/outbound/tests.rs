use super::{
    check, outcome,
    runtime::{programs, run, Case},
    upload_cases,
};

#[test]
fn telegram_port_outbound_attachment_normal_async() {
    for program in programs() {
        for (upload, expected) in upload_cases() {
            // One test-only acquisition, followed by exactly one delivery upload.
            let c = Case {
                upload,
                ..Case::attachment()
            };
            check(run(&program, c), &expected);
            // Text already sent cannot hide subsequent upload failure or cause retry.
            let c = Case {
                text: "hello".into(),
                upload,
                posts: vec![Ok(r#"{"ok":true}"#)],
                events: vec!["acquire", "html", "upload"],
                ..Case::attachment()
            };
            check(run(&program, c), &expected);
        }
        for (posts, events, expected) in [
            (
                vec![Ok(r#"{"ok":false}"#), Ok(r#"{"ok":true}"#)],
                vec!["acquire", "html", "plain", "upload"],
                Ok(outcome("TgSent", r#"{"ok":true}"#)),
            ),
            (
                vec![Ok(r#"{"ok":false}"#), Ok(r#"{"ok":false}"#)],
                vec!["acquire", "html", "plain"],
                Ok(outcome("TgRejected", r#"{"ok":false}"#)),
            ),
            (
                vec![Err("text lost")],
                vec!["acquire", "html"],
                Ok(outcome("TgUncertain", "text lost")),
            ),
            (
                vec![Ok("not json")],
                vec!["acquire", "html"],
                Err("json_get: invalid JSON:"),
            ),
            (
                vec![Ok(r#"{"ok":"true"}"#)],
                vec!["acquire", "html"],
                Ok(outcome("TgUncertain", r#"{"ok":"true"}"#)),
            ),
        ] {
            let c = Case {
                text: "hello".into(),
                posts,
                events,
                ..Case::attachment()
            };
            check(run(&program, c), &expected);
        }
        for destination in ["invalid", "", "7x"] {
            // Not zero TOTAL IO: the driver acquired bytes before the original
            // callee parsed its destination. There are zero DELIVERY effects.
            let c = Case {
                destination: destination.into(),
                events: vec!["acquire"],
                ..Case::attachment()
            };
            assert!(run(&program, c)
                .unwrap_err()
                .starts_with("to_int: cannot parse"));
        }
        check(
            run(
                &program,
                Case {
                    destination: "0007".into(),
                    ..Case::attachment()
                },
            ),
            &Ok(outcome("TgSent", r#"{"ok":true}"#)),
        );
    }
}

#[test]
fn telegram_port_outbound_path_normal_async() {
    for program in programs() {
        for (upload, expected) in upload_cases() {
            check(
                run(
                    &program,
                    Case {
                        upload,
                        ..Case::path()
                    },
                ),
                &expected,
            );
        }
        for caption in [
            "",
            "a=b=c",
            "Grüße 😀\nnext",
            "100% %26chat_id=99",
            "chat_id=99",
        ] {
            check(
                run(
                    &program,
                    Case {
                        caption: caption.into(),
                        ..Case::path()
                    },
                ),
                &Ok(outcome("TgSent", r#"{"ok":true}"#)),
            );
        }
        for caption in [
            "&",
            "a&b",
            "hello&chat_id=99",
            "&caption=forged",
            "trailing&",
        ] {
            let c = Case {
                caption: caption.into(),
                events: vec![],
                ..Case::path()
            };
            check(
                run(&program, c),
                &Ok(outcome(
                    "TgUncertain",
                    "caption contains unsupported multipart field delimiter",
                )),
            );
        }
    }
}
