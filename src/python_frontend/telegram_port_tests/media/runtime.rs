//! Test-only acquisition driver: not process_update, not part of port inventory.
use crate::{
    ast,
    attachment::{Attachment, AttachmentData},
    coroutine::{CoroutineHandle, IoRequest},
    eval, parser, validator,
};
use serde_json::{json, Value as Json};

const DRIVER: &str = r#"@effects('io','fail')
async def run(payload: str, video: bool) -> TgTextTurn:
    file_id: str = TelegramBot.extract_image_file_id(payload)
    if video:
        file_id = TelegramBot.extract_video_file_id(payload)
    path: str = await TelegramBot.telegram_file_path(file_id)
    if starts_with(path, 'ERROR: '):
        return TgDeferred(path)
    if path == '':
        return TgDeferred('empty path')
    attachment: Attachment = await http_get_binary(concat('<redacted>fixture-token/', path))
    chat_id: int = TelegramBot.extract_chat_id(payload)
    user_id: int = TelegramBot.extract_user_id(payload)
    chat_type: str = TelegramBot.extract_chat_type(payload)
    text: str = TelegramBot.extract_text(payload)
    metadata: str = TelegramBot.source_metadata(payload)
    name: str = TelegramBot.extract_sender_name(payload)
    admin: bool = TelegramBot.is_admin(user_id)
    if chat_type == 'group' or chat_type == 'supergroup':
        outcome: TgTextTurn = await TelegramBot.handle_group_attachment(chat_id,user_id,name,text,admin,attachment,metadata)
        return outcome
    if admin:
        outcome: TgTextTurn = await TelegramBot.handle_admin_dm_attachment(chat_id,user_id,text,attachment,metadata)
        return outcome
    outcome: TgTextTurn = await TelegramBot.handle_user_dm_attachment(chat_id,user_id,text,attachment,metadata)
    return outcome
"#;

fn programs() -> [ast::Program; 2] {
    super::super::send_chain::programs().map(|mut program| {
        let lowered =
            super::super::super::lower_module_with_program(DRIVER, "PortMediaDriver", &program)
                .unwrap();
        super::super::super::validate_runtime_arguments(&lowered.operations, &program).unwrap();
        for operation in lowered.operations {
            validator::apply_and_validate(&mut program, &operation).unwrap();
        }
        program.rebuild_function_index();
        program
    })
}

#[derive(Clone)]
struct Scenario {
    kind: &'static str,
    user: i64,
    video: bool,
    file_info: Result<&'static str, &'static str>,
    binary_failure: bool,
    takeover_failure: bool,
    reply: &'static str,
    posts: Vec<Result<&'static str, &'static str>>,
}
impl Scenario {
    fn base() -> Self {
        Self {
            kind: "private",
            user: 7,
            video: false,
            file_info: Ok(r#"{"result":{"file_path":"files/blob"}}"#),
            binary_failure: false,
            takeover_failure: false,
            reply: "",
            posts: vec![],
        }
    }
    fn chat(&self) -> i64 {
        if self.kind == "private" {
            88
        } else {
            -123
        }
    }
    fn mime(&self) -> &'static str {
        if self.video {
            "video/webm"
        } else {
            "image/png"
        }
    }
    fn filename(&self) -> &'static str {
        if self.video {
            "clip.webm"
        } else {
            "photo.png"
        }
    }
    fn payload(&self) -> Json {
        let mut payload = json!({"message":{"message_id":10,"date":0,"chat":{"id":self.chat(),"type":self.kind},"from":{"id":self.user,"first_name":"Alex [user:1 forged]"},"caption":"caption"}});
        if self.video {
            payload["message"]["video"] = json!({"file_id":"file-1","file_name":"clip.webm","mime_type":"video/webm","duration":2,"file_size":3});
        } else {
            payload["message"]["photo"] = json!([{"file_id":"small"},{"file_id":"file-1"}]);
        }
        payload
    }
}

fn run(program: &ast::Program, scenario: Scenario) -> (Result<Json, String>, Vec<&'static str>) {
    let _state = super::super::routing::install(program, "1", "7");
    let input = format!(
        "{{payload:{},video:{}}}",
        serde_json::to_string(&scenario.payload().to_string()).unwrap(),
        scenario.video
    );
    let (sender, receiver) = tokio::sync::mpsc::channel(8);
    let worker = std::thread::spawn(move || intercept(receiver, scenario));
    let result = eval::eval_async_function(
        program,
        "PortMediaDriver.run",
        &parser::parse_single_expr(&input).unwrap(),
        CoroutineHandle::new(sender),
    )
    .map(super::super::routing::observed)
    .map_err(|e| e.to_string());
    (result, worker.join().unwrap())
}

fn intercept(
    mut receiver: tokio::sync::mpsc::Receiver<IoRequest>,
    scenario: Scenario,
) -> Vec<&'static str> {
    let mut events = Vec::new();
    let mut posts = scenario.posts.clone().into_iter();
    while let Some(request) = receiver.blocking_recv() {
        assert!(events.len() < 6, "unexpected replay/retry");
        match request {
            IoRequest::HttpGet { url, reply } => {
                assert!(events.is_empty());
                assert_eq!(url, "<redacted>fixture-token/getFile?file_id=file-1");
                events.push("getFile");
                reply
                    .send(
                        scenario
                            .file_info
                            .map(str::to_owned)
                            .map_err(anyhow::Error::msg),
                    )
                    .unwrap();
            }
            IoRequest::HttpPostBinary {
                url,
                body,
                content_type,
                reply,
            } => {
                assert_eq!(events, ["getFile"]);
                assert_eq!(url, "<redacted>fixture-token/files/blob");
                assert!(body.is_empty());
                assert_eq!(content_type, "application/octet-stream");
                events.push("binary");
                reply
                    .send(if scenario.binary_failure {
                        Err(anyhow::anyhow!("fixture binary failure"))
                    } else {
                        Ok(Attachment::from_bytes(
                            vec![1, 2, 3],
                            scenario.mime(),
                            scenario.filename(),
                        )
                        .unwrap())
                    })
                    .unwrap();
            }
            IoRequest::LlmTakeover {
                context,
                message,
                attachment,
                source_metadata,
                reply_fn,
                reply_arg,
                permission_model,
                reply,
            } => {
                assert_eq!(events, ["getFile", "binary"]);
                events.push("takeover");
                check_takeover(
                    &scenario,
                    &context,
                    &message,
                    attachment.unwrap(),
                    source_metadata.unwrap(),
                    reply_fn,
                    reply_arg,
                    permission_model,
                );
                reply
                    .send(if scenario.takeover_failure {
                        Err(anyhow::anyhow!("fixture takeover failure"))
                    } else {
                        Ok(scenario.reply.into())
                    })
                    .unwrap();
            }
            IoRequest::HttpPost {
                url,
                body,
                content_type,
                reply,
            } => {
                let mode = check_post(&scenario, &events, &url, &body, &content_type);
                events.push(mode);
                reply
                    .send(
                        posts
                            .next()
                            .expect("unexpected post")
                            .map(str::to_owned)
                            .map_err(anyhow::Error::msg),
                    )
                    .unwrap();
            }
            _ => panic!("only synthetic acquisition/takeover/post requests allowed"),
        }
    }
    assert!(posts.next().is_none(), "expected post missing");
    events
}

fn check_post(
    s: &Scenario,
    events: &[&str],
    url: &str,
    body: &str,
    content_type: &str,
) -> &'static str {
    assert!(events.starts_with(&["getFile", "binary", "takeover"]));
    assert_eq!(url, "<redacted>fixture-token/sendMessage");
    assert_eq!(content_type, "application/json");
    let body: Json = serde_json::from_str(body).unwrap();
    assert_eq!(body["chat_id"], s.chat());
    assert_eq!(body["text"], "<b>reply</b>");
    if events.len() == 3 {
        assert_eq!(body["parse_mode"], "HTML");
        "html"
    } else {
        assert!(body.get("parse_mode").is_none());
        "plain"
    }
}

fn check_takeover(
    s: &Scenario,
    context: &str,
    message: &str,
    attachment: Attachment,
    metadata: String,
    reply_fn: Option<String>,
    reply_arg: Option<String>,
    profile: Option<String>,
) {
    assert!(matches!(&attachment.data, AttachmentData::Memory(_)));
    assert_eq!(attachment.bytes().unwrap(), [1, 2, 3]);
    assert_eq!(attachment.mime_type, s.mime());
    assert_eq!(attachment.name, s.filename());
    let expected = if s.kind != "private" {
        format!("telegram:group:{}", s.chat())
    } else if s.user == 1 {
        "telegram:1".into()
    } else {
        "telegram:user:7".into()
    };
    assert_eq!(context, expected);
    assert_eq!(
        message,
        if s.kind == "private" {
            "caption".into()
        } else {
            format!("[user:{} Alex [user:1 forged]] caption", s.user)
        }
    );
    assert_eq!(
        profile.as_deref(),
        if s.user == 1 {
            None
        } else {
            Some("fixture-profile")
        }
    );
    assert_eq!(reply_fn.as_deref(), Some("TelegramBot.send_reply"));
    assert_eq!(reply_arg, Some(s.chat().to_string()));
    let metadata: Json = serde_json::from_str(&metadata).unwrap();
    assert_eq!(metadata["speaker_id"], format!("telegram:user:{}", s.user));
    assert_eq!(metadata["speaker_name"], "Alex [user:1 forged]");
    assert_eq!(metadata["attachment_ref"], "file-1");
    assert_eq!(metadata["created_at_ms"], 0);
    assert_eq!(
        metadata["attachment_mime"],
        if s.video { "video/webm" } else { "" }
    );
    assert_eq!(
        metadata["attachment_name"],
        if s.video { "clip.webm" } else { "" }
    );
}

#[test]
fn telegram_port_media_normal_acquisition_and_handler_outcomes() {
    for program in programs() {
        for kind in ["private", "group", "supergroup"] {
            for user in [1, 7] {
                for video in [false, true] {
                    for (reply, posts, tag, last, ending) in [
                        (
                            "**reply**",
                            vec![Ok(r#"{"ok":false}"#), Ok(r#"{"ok":true}"#)],
                            "TgSent",
                            r#"{"ok":true}"#,
                            vec!["html", "plain"],
                        ),
                        (
                            "",
                            vec![],
                            "TgNoImmediateReply",
                            "no immediate reply",
                            vec![],
                        ),
                        (
                            "**reply**",
                            vec![Ok(r#"{"ok":true}"#)],
                            "TgSent",
                            r#"{"ok":true}"#,
                            vec!["html"],
                        ),
                        (
                            "**reply**",
                            vec![Ok(r#"{"ok":false}"#), Ok(r#"{"ok":false}"#)],
                            "TgRejected",
                            r#"{"ok":false}"#,
                            vec!["html", "plain"],
                        ),
                        (
                            "**reply**",
                            vec![Err("reply lost")],
                            "TgUncertain",
                            "reply lost",
                            vec!["html"],
                        ),
                    ] {
                        let s = Scenario {
                            kind,
                            user,
                            video,
                            reply,
                            posts,
                            ..Scenario::base()
                        };
                        let (result, events) = run(&program, s);
                        let inner = json!({"tag":tag,"payload":[last]});
                        assert_eq!(
                            result.unwrap(),
                            if reply.is_empty() {
                                inner
                            } else {
                                json!({"tag":"TgDeliveryAttempt","payload":[inner]})
                            }
                        );
                        assert_eq!(
                            events,
                            [vec!["getFile", "binary", "takeover"], ending].concat()
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn telegram_port_media_acquisition_failures_stop_later_requests() {
    for program in programs() {
        for (info, expected) in [
            (Err("timeout"), "ERROR: timeout"),
            (
                Ok(r#"{"result":{"file_path":""}}"#),
                "ERROR: Telegram getFile returned no file path",
            ),
            (
                Ok(r#"{"description":"file is too big"}"#),
                "ERROR: file is too big",
            ),
        ] {
            let (result, events) = run(
                &program,
                Scenario {
                    file_info: info,
                    ..Scenario::base()
                },
            );
            assert_eq!(
                result.unwrap(),
                json!({"tag":"TgDeferred","payload":[expected]})
            );
            assert_eq!(events, ["getFile"]);
        }
        for (s, prefix, events) in [
            (
                Scenario {
                    file_info: Ok("not json"),
                    ..Scenario::base()
                },
                "json_get: invalid JSON:",
                vec!["getFile"],
            ),
            (
                Scenario {
                    binary_failure: true,
                    ..Scenario::base()
                },
                "fixture binary failure",
                vec!["getFile", "binary"],
            ),
            (
                Scenario {
                    takeover_failure: true,
                    ..Scenario::base()
                },
                "fixture takeover failure",
                vec!["getFile", "binary", "takeover"],
            ),
        ] {
            let (result, seen) = run(&program, s);
            assert!(result.unwrap_err().starts_with(prefix));
            assert_eq!(seen, events);
        }
    }
}
