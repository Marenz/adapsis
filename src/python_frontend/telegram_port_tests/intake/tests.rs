use super::runtime::{run, Case};
use serde_json::{json, Value as Json};

fn no_reply() -> Json {
    json!({"tag":"TgNoImmediateReply","payload":["no immediate reply"]})
}
fn delivery(tag: &str, message: &str) -> Json {
    json!({"tag":"TgDeliveryAttempt","payload":[{"tag":tag,"payload":[message]}]})
}

#[test]
fn telegram_port_intake_raw_media_text_voice_normal_dispatch() {
    for program in super::super::send_chain::programs() {
        for kind in ["private", "group", "supergroup"] {
            for user in [1, 7] {
                for mode in ["text", "photo", "image-doc", "video", "video-doc", "voice"] {
                    let base = if mode == "text" {
                        Case::base(kind, user)
                    } else {
                        Case::media(kind, user, mode)
                    };
                    assert_eq!(run(&program, base.clone()).unwrap(), no_reply());
                    for (posts, events, tag, last) in [
                        (
                            vec![Ok(r#"{"ok":true}"#)],
                            vec!["html"],
                            "TgSent",
                            r#"{"ok":true}"#,
                        ),
                        (
                            vec![Ok(r#"{"ok":false}"#), Ok(r#"{"ok":true}"#)],
                            vec!["html", "plain"],
                            "TgSent",
                            r#"{"ok":true}"#,
                        ),
                        (
                            vec![Ok(r#"{"ok":false}"#), Ok(r#"{"ok":false}"#)],
                            vec!["html", "plain"],
                            "TgRejected",
                            r#"{"ok":false}"#,
                        ),
                        (
                            vec![Err("uncertain")],
                            vec!["html"],
                            "TgUncertain",
                            "uncertain",
                        ),
                    ] {
                        let mut c = base.clone();
                        c.reply = "reply";
                        c.posts = posts;
                        c.events.extend(events);
                        assert_eq!(run(&program, c).unwrap(), delivery(tag, last));
                    }
                }
            }
        }
    }
}

#[test]
fn telegram_port_intake_precedence_and_metadata() {
    for program in super::super::send_chain::programs() {
        let mut cases = Vec::new();
        for mode in ["photo", "image-doc", "video", "video-doc", "voice"] {
            let mut c = Case::media("group", 7, mode);
            c.payload["message"]["text"] = json!("text wins");
            c.text = "text wins".into();
            if mode == "voice" {
                c.events = vec!["takeover"];
                c.file_id = None;
            }
            cases.push(c);
        }
        for mode in ["photo", "video"] {
            let mut c = Case::media("private", 1, mode);
            c.payload["message"]
                .as_object_mut()
                .unwrap()
                .remove("caption");
            c.text = if mode == "photo" {
                ""
            } else {
                "[user sent a video; the original is available to video_stabilize()]"
            }
            .into();
            cases.push(c);
        }
        let mut c = Case::media("group", 7, "voice");
        c.payload["message"]["caption"] = json!("caption skips ASR");
        c.text = "caption skips ASR".into();
        c.events = vec!["takeover"];
        c.file_id = None;
        cases.push(c);
        let mut c = Case::media("group", 7, "photo");
        c.payload["message"]["video"] =
            json!({"file_id":"lower-video","file_name":"lower.webm","mime_type":"video/webm"});
        c.payload["message"]["voice"] = json!({"file_id":"lower-voice"});
        c.metadata["attachment_name"] = json!("lower.webm");
        c.metadata["attachment_mime"] = json!("video/webm");
        cases.push(c);
        let mut c = Case::media("private", 1, "video");
        c.payload["message"]["voice"] = json!({"file_id":"lower-voice"});
        cases.push(c);
        for mode in ["image-doc", "video-doc"] {
            let mut c = Case::media("group", 7, mode);
            c.payload["message"]["photo"] = json!([]);
            cases.push(c);
        }
        let mut c = Case::media("group", 7, "video");
        c.payload["message"]["document"] =
            json!({"file_id":"lower-doc","mime_type":"video/mp4","file_name":"lower.mp4"});
        cases.push(c);
        let mut c = Case::base("private", 1);
        c.payload["message"]
            .as_object_mut()
            .unwrap()
            .remove("message_id");
        c.metadata["message_id"] = json!("\"\"");
        c.payload["message"]["date"] = json!(9223372036854775_i64);
        c.metadata["created_at_ms"] = json!(9223372036854775000_i64);
        cases.push(c);
        let mut c = Case::base("group", 7);
        c.payload["message"]["photo"] = json!([]);
        cases.push(c);
        for text in ["null", "{}", "[]", "Grüße 😀\nnext"] {
            let mut c = Case::base("private", 7);
            c.payload["message"]["text"] = json!(text);
            c.text = text.into();
            cases.push(c);
        }
        for c in cases {
            assert_eq!(run(&program, c).unwrap(), no_reply());
        }
    }
}

fn set_path(root: &mut Json, path: &str, value: Json) {
    let mut current = root;
    let mut parts = path.split('.').peekable();
    while let Some(part) = parts.next() {
        if parts.peek().is_none() {
            current[part] = value;
            return;
        }
        if current.get(part).is_none() {
            current[part] = json!({});
        }
        current = &mut current[part];
    }
}

const CONSUMED_FIELDS: &[&str] = &[
    "message",
    "message.chat",
    "message.from",
    "message.chat.id",
    "message.from.id",
    "message.date",
    "message.message_id",
    "message.chat.type",
    "message.text",
    "message.caption",
    "message.from.first_name",
    "message.from.last_name",
    "message.from.username",
    "message.document.file_name",
    "message.document.mime_type",
    "message.document.file_id",
    "message.video.file_name",
    "message.video.mime_type",
    "message.video.file_id",
    "message.voice.file_id",
    "message.sticker.file_id",
    "message.sticker.emoji",
    "message.video_note.file_id",
    "message.audio.file_id",
    "message.animation.file_id",
    "message.contact.phone_number",
    "message.poll.question",
    "message.video.duration",
    "message.video.file_size",
    "message.document.file_size",
    "message.location.latitude",
];

fn invalid_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for path in CONSUMED_FIELDS {
        for value in [Json::Null, json!(true), json!([])] {
            let mut c = Case::base("private", 1);
            set_path(&mut c.payload, path, value);
            c.events.clear();
            cases.push(c);
        }
    }
    for path in [
        "message.chat.id",
        "message.from.id",
        "message.date",
        "message.message_id",
    ] {
        for value in [json!("1"), json!(1.0), json!({})] {
            let mut c = Case::base("group", 1);
            set_path(&mut c.payload, path, value);
            c.events.clear();
            cases.push(c);
        }
    }
    for media in [
        "photo",
        "video",
        "voice",
        "document",
        "sticker",
        "video_note",
        "audio",
        "animation",
        "contact",
        "location",
        "poll",
    ] {
        for value in [Json::Null, json!(false), json!(""), json!(1)] {
            let mut c = Case::base("private", 7);
            c.payload["message"][media] = value;
            c.events.clear();
            cases.push(c);
        }
    }
    for photo in [
        json!({}),
        json!([null]),
        json!([{}]),
        json!([{"file_id":""}]),
        json!([{"file_id":1}]),
        json!([{"file_id":false}]),
    ] {
        let mut c = Case::base("group", 1);
        c.payload["message"]["photo"] = photo;
        c.events.clear();
        cases.push(c);
    }
    for (path, value) in [
        ("message.from.id", json!(0)),
        ("message.chat.id", json!(0)),
        ("message.from.id", json!(-1)),
        ("message.date", json!(-1)),
        ("message.date", json!(9223372036854776_i64)),
        ("message.chat.type", json!("channel")),
        ("message.video.duration", json!(-1)),
        ("message.document.file_size", json!(1.5)),
    ] {
        let mut c = Case::base("private", 1);
        set_path(&mut c.payload, path, value);
        c.events.clear();
        cases.push(c);
    }
    for mode in ["video", "voice", "image-doc"] {
        let mut c = Case::media("private", 1, mode);
        let key = if mode == "image-doc" {
            "document"
        } else {
            mode
        };
        c.payload["message"][key]["file_id"] = json!("");
        c.events.clear();
        cases.push(c);
    }
    let mut c = Case::media("private", 1, "photo");
    c.payload["message"]["voice"] = json!({"file_id":false});
    c.events.clear();
    cases.push(c);
    cases
}

#[test]
fn telegram_port_intake_preflight_and_authorization_zero_io() {
    for program in super::super::send_chain::programs() {
        for c in invalid_cases() {
            assert_eq!(run(&program, c).unwrap()["tag"], "TgDeferred");
        }
        for mode in ["text", "photo", "video", "voice"] {
            let mut c = if mode == "text" {
                Case::base("group", 99)
            } else {
                Case::media("group", 99, mode)
            };
            c.events.clear();
            assert_eq!(
                run(&program, c).unwrap(),
                json!({"tag":"TgIgnored","payload":["unauthorized sender"]})
            );
        }
        let mut c = Case::base("private", 1);
        c.raw = Some("not json".into());
        c.events.clear();
        assert!(run(&program, c)
            .unwrap_err()
            .starts_with("json_kind: invalid JSON:"));
    }
}

#[test]
fn telegram_port_intake_fallback_kinds() {
    for program in super::super::send_chain::programs() {
        for (key, body, text) in [
            (
                "document",
                json!({"file_name":"notes.pdf"}),
                "user sent a file (notes.pdf) which I cannot read",
            ),
            (
                "document",
                json!({"file_id":"doc"}),
                "user sent a file which I cannot read",
            ),
            (
                "sticker",
                json!({"file_id":"st","emoji":"😀"}),
                "user sent a sticker 😀",
            ),
            (
                "video_note",
                json!({"file_id":"vn"}),
                "user sent a video note which I cannot watch",
            ),
            (
                "audio",
                json!({"file_id":"au","duration":1.5}),
                "user sent an audio file which I cannot hear",
            ),
            (
                "animation",
                json!({"file_id":"gif"}),
                "user sent a GIF which I cannot watch",
            ),
            (
                "contact",
                json!({"phone_number":"555"}),
                "user shared a contact",
            ),
            (
                "location",
                json!({"latitude":52.5}),
                "user shared a location",
            ),
            (
                "poll",
                json!({"question":"What?"}),
                "user sent a poll: What?",
            ),
        ] {
            let mut c = Case::base("private", 7);
            c.payload["message"]["text"] = json!("");
            if key == "document" && body.get("file_name").is_some() {
                c.metadata["attachment_name"] = json!("notes.pdf");
            }
            c.payload["message"][key] = body;
            c.text = format!("[{text}]");
            assert_eq!(run(&program, c).unwrap(), no_reply());
        }
        let mut c = Case::base("private", 1);
        c.payload["message"]["text"] = json!("");
        c.events.clear();
        assert_eq!(
            run(&program, c).unwrap(),
            json!({"tag":"TgIgnored","payload":["no recognisable content"]})
        );
    }
}

#[test]
fn telegram_port_intake_terminal_failures_and_guidance() {
    for program in super::super::send_chain::programs() {
        for mode in ["photo", "video", "voice"] {
            let base = Case::media("private", 7, mode);
            for (lookup, prefix) in [(true, "json_get: invalid JSON:"), (false, "binary failed")] {
                let mut c = base.clone();
                c.events = if lookup {
                    vec!["getFile"]
                } else {
                    vec!["getFile", "binary"]
                };
                if lookup {
                    c.lookup = Ok("bad json".into());
                } else {
                    c.binary_fail = true;
                }
                let error = run(&program, c).unwrap_err();
                assert!(error.starts_with(if mode == "voice" && lookup {
                    "json_kind: invalid JSON:"
                } else {
                    prefix
                }));
            }
            let mut c = base.clone();
            c.takeover_fail = true;
            assert_eq!(run(&program, c).unwrap_err(), "takeover failed");
            if mode != "voice" {
                guidance(&program, base, mode);
            }
        }
        for (upload, text) in [
            (Err("upload lost"), "[voice message - transcription failed]"),
            (
                Ok(r#"{"error":"bad"}"#.into()),
                "[voice message - transcription failed]",
            ),
            (Ok(r#"{"text":"null"}"#.into()), "null"),
        ] {
            let mut c = Case::media("group", 7, "voice");
            c.upload = upload;
            c.text = text.into();
            assert_eq!(run(&program, c).unwrap(), no_reply());
        }
        let mut c = Case::media("private", 7, "voice");
        c.upload = Ok("bad json".into());
        c.events.pop();
        assert!(run(&program, c)
            .unwrap_err()
            .starts_with("json_kind: invalid JSON:"));
        let mut c = Case::media("private", 7, "voice");
        c.lookup = Err("lost");
        c.events = vec!["getFile", "takeover"];
        c.text = "[voice message - getFile failed: lost]".into();
        assert_eq!(run(&program, c).unwrap(), no_reply());
    }
}

fn guidance(program: &crate::ast::Program, base: Case, mode: &str) {
    for lookup in [Err("lost"), Ok("{}".into())] {
        for (posts, labels, tag, last) in [
            (
                vec![Ok(r#"{"ok":true}"#)],
                vec!["html"],
                "TgSent",
                r#"{"ok":true}"#,
            ),
            (vec![Err("lost")], vec!["html"], "TgUncertain", "lost"),
            (
                vec![Ok(r#"{"ok":false}"#), Ok(r#"{"ok":false}"#)],
                vec!["html", "plain"],
                "TgRejected",
                r#"{"ok":false}"#,
            ),
        ] {
            let mut c = base.clone();
            c.lookup = lookup.clone();
            c.events = vec!["getFile"];
            c.events.extend(labels);
            c.posts = posts;
            c.post_text = format!("I could not download that {mode}. Please send it again.");
            assert_eq!(run(program, c).unwrap(), delivery(tag, last));
        }
    }
}
