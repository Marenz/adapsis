use crate::{
    ast,
    attachment::{Attachment, AttachmentData},
    coroutine::{CoroutineHandle, IoRequest},
    eval, parser, validator,
};
use serde_json::{json, Value as Json};

const DRIVER: &str = r#"@effects('io','fail')
async def run(destination: str, text: str) -> TgSendAttempt:
    file: Attachment = await http_get_binary('fixture://outbound-bytes')
    result: TgSendAttempt = await TelegramBot.send_reply_with_attachment(destination, text, file)
    return result
"#;

pub(super) fn programs() -> [ast::Program; 2] {
    super::super::send_chain::programs().map(|mut p| {
        let lower =
            super::super::super::lower_module_with_program(DRIVER, "PortOutboundDriver", &p)
                .unwrap();
        for op in lower.operations {
            validator::apply_and_validate(&mut p, &op).unwrap();
        }
        p.rebuild_function_index();
        p
    })
}

#[derive(Clone)]
pub(super) struct Case {
    pub path: bool,
    pub destination: String,
    pub text: String,
    pub caption: String,
    pub upload: Result<&'static str, &'static str>,
    pub posts: Vec<Result<&'static str, &'static str>>,
    pub events: Vec<&'static str>,
}
impl Case {
    pub fn attachment() -> Self {
        Self {
            path: false,
            destination: "7".into(),
            text: "".into(),
            caption: "".into(),
            upload: Ok(r#"{"ok":true}"#),
            posts: vec![],
            events: vec!["acquire", "upload"],
        }
    }
    pub fn path() -> Self {
        Self {
            path: true,
            events: vec!["upload"],
            ..Self::attachment()
        }
    }
}

pub(super) fn run(program: &ast::Program, c: Case) -> Result<Json, String> {
    let _state = super::super::routing::install(program, "1", "7");
    let (function, input) = if c.path {
        (
            "TelegramBot.send_audio_file",
            format!(
                "{{chat_id:7,file_path:\"fixture-only-path\",caption:{}}}",
                json!(c.caption)
            ),
        )
    } else {
        (
            "PortOutboundDriver.run",
            format!(
                "{{destination:{},text:{}}}",
                json!(c.destination),
                json!(c.text)
            ),
        )
    };
    let (sender, receiver) = tokio::sync::mpsc::channel(5);
    let worker = std::thread::spawn(move || intercept(receiver, c));
    let result = eval::eval_async_function(
        program,
        function,
        &parser::parse_single_expr(&input).unwrap(),
        CoroutineHandle::new(sender),
    )
    .map(super::super::routing::observed)
    .map_err(|e| e.to_string());
    worker.join().unwrap();
    result
}

fn intercept(mut receiver: tokio::sync::mpsc::Receiver<IoRequest>, c: Case) {
    let mut seen = vec![];
    let mut posts = c.posts.clone().into_iter();
    while let Some(request) = receiver.blocking_recv() {
        let label = match request {
            IoRequest::HttpPostBinary {
                url,
                body,
                content_type,
                reply,
            } => {
                assert!(!c.path);
                assert_eq!(url, "fixture://outbound-bytes");
                assert!(body.is_empty());
                assert_eq!(content_type, "application/octet-stream");
                let a = Attachment::from_bytes(vec![4, 5, 6], "audio/ogg", "reply.ogg").unwrap();
                assert!(matches!(&a.data, AttachmentData::Memory(_)));
                reply.send(Ok(a)).unwrap();
                "acquire"
            }
            IoRequest::HttpPost {
                url,
                body,
                content_type,
                reply,
            } => {
                assert!(!c.path);
                assert_eq!(url, "<redacted>fixture-token/sendMessage");
                assert_eq!(content_type, "application/json");
                let body: Json = serde_json::from_str(&body).unwrap();
                assert_eq!(body["chat_id"], 7);
                assert_eq!(body["text"], c.text);
                let label = if body.get("parse_mode").is_some() {
                    assert_eq!(body["parse_mode"], "HTML");
                    "html"
                } else {
                    "plain"
                };
                reply
                    .send(
                        posts
                            .next()
                            .expect("unexpected text retry")
                            .map(str::to_owned)
                            .map_err(anyhow::Error::msg),
                    )
                    .unwrap();
                label
            }
            request @ (IoRequest::HttpUpload { .. } | IoRequest::HttpUploadBytes { .. }) => {
                upload(request, &c);
                "upload"
            }
            _ => panic!("unexpected IO; no Runtime loop/files/network"),
        };
        assert_eq!(
            c.events.get(seen.len()),
            Some(&label),
            "extra/reordered request"
        );
        seen.push(label);
    }
    assert_eq!(seen, c.events);
    assert!(posts.next().is_none());
}

fn upload(request: IoRequest, c: &Case) {
    let (url, field, extras, reply) = match request {
        IoRequest::HttpUpload {
            url,
            file_path,
            file_field,
            extra_fields,
            reply,
        } if c.path => {
            assert_eq!(file_path, "fixture-only-path");
            (url, file_field, extra_fields, reply)
        }
        IoRequest::HttpUploadBytes {
            url,
            bytes,
            file_name,
            mime_type,
            file_field,
            extra_fields,
            reply,
        } if !c.path => {
            assert_eq!(bytes, [4, 5, 6]);
            assert_eq!(file_name, "reply.ogg");
            assert_eq!(mime_type, "audio/ogg");
            (url, file_field, extra_fields, reply)
        }
        _ => panic!("wrong upload source overload"),
    };
    assert_eq!(url, "<redacted>fixture-token/sendDocument");
    assert_eq!(field, "document");
    let expected = if c.path {
        vec![
            ("chat_id".into(), "7".into()),
            ("caption".into(), c.caption.clone()),
        ]
    } else {
        vec![("chat_id".into(), "7".into())]
    };
    assert_eq!(extras, expected);
    reply
        .send(c.upload.map(str::to_owned).map_err(anyhow::Error::msg))
        .unwrap();
}
