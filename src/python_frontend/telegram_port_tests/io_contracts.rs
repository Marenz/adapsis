//! Production request dispatch with trusted in-memory replies. No Runtime IO loop,
//! sockets, files, sleeping, provider calls or credentials are involved.
use crate::{
    attachment::Attachment,
    coroutine::{CoroutineHandle, IoRequest},
    eval::Value,
};
use std::time::Duration;

fn exchange(
    op: &'static str,
    args: Vec<Value>,
    respond: impl FnOnce(IoRequest),
) -> anyhow::Result<Value> {
    let (sender, mut receiver) = tokio::sync::mpsc::channel(4);
    let (finished, result) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        let outcome = CoroutineHandle::new(sender).execute_await(op, &args);
        finished.send(outcome).unwrap();
    });
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
    let request = runtime
        .block_on(async { tokio::time::timeout(Duration::from_secs(3), receiver.recv()).await })
        .unwrap()
        .expect("expected exactly one request");
    respond(request);
    let value = result.recv_timeout(Duration::from_secs(3)).unwrap();
    worker.join().unwrap();
    assert!(
        receiver.try_recv().is_err(),
        "unexpected additional request"
    );
    value
}

fn strings(args: &[&str]) -> Vec<Value> {
    args.iter().map(|s| Value::string(*s)).collect()
}

fn response(fail: bool) -> anyhow::Result<String> {
    if fail {
        anyhow::bail!("fixture failure")
    } else {
        Ok("fixture body".into())
    }
}

fn wrapped(value: anyhow::Result<Value>, fail: bool) {
    match value.unwrap() {
        Value::Err(error) if fail => assert_eq!(error, "fixture failure"),
        Value::Ok(body) if !fail => {
            assert!(matches!(*body, Value::String(ref text) if text.as_str() == "fixture body"))
        }
        other => panic!("unexpected wrapper: {other}"),
    }
}

#[test]
fn telegram_port_file_and_sleep_contracts() {
    for fail in [false, true] {
        let read = exchange("read_file", strings(&["fixture-path"]), |request| {
            let IoRequest::FileRead { path, reply } = request else {
                panic!("expected FileRead")
            };
            assert_eq!(path, "fixture-path");
            reply.send(response(fail)).unwrap();
        });
        if fail {
            assert!(read.unwrap_err().to_string().contains("fixture failure"));
        } else {
            assert!(matches!(read.unwrap(), Value::String(ref s) if s.as_str() == "fixture body"));
        }
        let exists = exchange("file_exists", strings(&["fixture-path"]), |request| {
            let IoRequest::FileExists { path, reply } = request else {
                panic!("expected FileExists")
            };
            assert_eq!(path, "fixture-path");
            reply
                .send(if fail {
                    Err(anyhow::anyhow!("fixture failure"))
                } else {
                    Ok(true)
                })
                .unwrap();
        });
        if fail {
            assert!(exists.is_err());
        } else {
            assert!(matches!(exists.unwrap(), Value::Bool(true)));
        }
        let slept = exchange("sleep", vec![Value::Int(3000)], |request| {
            let IoRequest::Sleep { ms, reply } = request else {
                panic!("expected Sleep")
            };
            assert_eq!(ms, 3000);
            reply
                .send(if fail {
                    Err(anyhow::anyhow!("fixture failure"))
                } else {
                    Ok(())
                })
                .unwrap();
        });
        if fail {
            assert!(slept.is_err());
        } else {
            assert!(matches!(slept.unwrap(), Value::Int(0)));
        }
    }
}

#[test]
fn telegram_port_http_text_contracts() {
    for fail in [false, true] {
        wrapped(
            exchange("http_get", strings(&["fixture://get"]), |request| {
                let IoRequest::HttpGet { url, reply } = request else {
                    panic!("expected HttpGet")
                };
                assert_eq!(url, "fixture://get");
                reply.send(response(fail)).unwrap();
            }),
            fail,
        );
        for content_type in [None, Some("text/plain")] {
            let mut args = strings(&["fixture://post", "payload"]);
            if let Some(value) = content_type {
                args.push(Value::string(value));
            }
            wrapped(
                exchange("http_post", args, |request| {
                    let IoRequest::HttpPost {
                        url,
                        body,
                        content_type: actual,
                        reply,
                    } = request
                    else {
                        panic!("expected HttpPost")
                    };
                    assert_eq!(url, "fixture://post");
                    assert_eq!(body, "payload");
                    assert_eq!(actual, content_type.unwrap_or("application/json"));
                    reply.send(response(fail)).unwrap();
                }),
                fail,
            );
        }
    }
}

#[test]
fn telegram_port_binary_download_contract() {
    for fail in [false, true] {
        let result = exchange(
            "http_get_binary",
            strings(&["fixture://binary"]),
            |request| {
                let IoRequest::HttpPostBinary {
                    url,
                    body,
                    content_type,
                    reply,
                } = request
                else {
                    panic!("expected binary request")
                };
                assert_eq!(url, "fixture://binary");
                assert!(body.is_empty());
                assert_eq!(content_type, "application/octet-stream");
                reply
                    .send(if fail {
                        Err(anyhow::anyhow!("fixture failure"))
                    } else {
                        Attachment::from_bytes(vec![1, 2, 3], "image/png", "fixture.png")
                            .map_err(Into::into)
                    })
                    .unwrap();
            },
        );
        if fail {
            assert!(result.is_err());
        } else {
            let Value::Attachment(attachment) = result.unwrap() else {
                panic!("bare Attachment required")
            };
            assert_eq!(attachment.bytes().unwrap(), vec![1, 2, 3]);
            assert_eq!(attachment.mime_type, "image/png");
            assert_eq!(attachment.name, "fixture.png");
        }
    }
}

#[test]
fn telegram_port_upload_contracts() {
    for fail in [false, true] {
        for bytes in [false, true] {
            let file = if bytes {
                Value::Attachment(
                    Attachment::from_bytes(vec![1, 2, 3], "audio/ogg", "voice.ogg").unwrap(),
                )
            } else {
                Value::string("fixture-only-path")
            };
            wrapped(
                exchange(
                    "http_upload",
                    vec![
                        Value::string("fixture://upload"),
                        file,
                        Value::string("document"),
                        Value::string("chat_id=42&caption=hi"),
                    ],
                    |request| {
                        let (url, field, extras, reply) = match request {
                            IoRequest::HttpUploadBytes {
                                url,
                                bytes: data,
                                file_name,
                                mime_type,
                                file_field,
                                extra_fields,
                                reply,
                            } if bytes => {
                                assert_eq!(data, vec![1, 2, 3]);
                                assert_eq!(file_name, "voice.ogg");
                                assert_eq!(mime_type, "audio/ogg");
                                (url, file_field, extra_fields, reply)
                            }
                            IoRequest::HttpUpload {
                                url,
                                file_path,
                                file_field,
                                extra_fields,
                                reply,
                            } if !bytes => {
                                assert_eq!(file_path, "fixture-only-path");
                                (url, file_field, extra_fields, reply)
                            }
                            _ => panic!("wrong multipart request"),
                        };
                        assert_eq!(url, "fixture://upload");
                        assert_eq!(field, "document");
                        assert_eq!(
                            extras,
                            vec![
                                ("chat_id".into(), "42".into()),
                                ("caption".into(), "hi".into())
                            ]
                        );
                        reply.send(response(fail)).unwrap();
                    },
                ),
                fail,
            );
        }
    }
}

#[test]
fn telegram_port_takeover_overload_contracts() {
    for image in [false, true] {
        for fail in [false, true] {
            let mut args = strings(&["fixture:chat", "message", "Fixture.reply", "42", "limited"]);
            if image {
                args.push(Value::Attachment(
                    Attachment::from_bytes(vec![1, 2, 3], "image/png", "fixture.png").unwrap(),
                ));
            }
            args.push(Value::string("{\"speaker_id\":\"fixture:user\"}"));
            let result = exchange("llm_takeover", args, |request| {
                let IoRequest::LlmTakeover {
                    origin: None,
                    context,
                    message,
                    attachment,
                    source_metadata,
                    reply_fn,
                    reply_arg,
                    permission_model,
                    reply,
                } = request
                else {
                    panic!("expected LlmTakeover")
                };
                assert_eq!(context, "fixture:chat");
                assert_eq!(message, "message");
                assert_eq!(reply_fn.as_deref(), Some("Fixture.reply"));
                assert_eq!(reply_arg.as_deref(), Some("42"));
                assert_eq!(permission_model.as_deref(), Some("limited"));
                assert_eq!(
                    source_metadata.as_deref(),
                    Some("{\"speaker_id\":\"fixture:user\"}")
                );
                assert_eq!(attachment.is_some(), image);
                if let Some(attachment) = attachment {
                    assert_eq!(attachment.bytes().unwrap(), vec![1, 2, 3]);
                }
                reply.send(response(fail)).unwrap();
            });
            if fail {
                assert!(result.is_err());
            } else {
                assert!(
                    matches!(result.unwrap(), Value::String(ref text) if text.as_str() == "fixture body")
                );
            }
        }
    }
}

#[test]
fn telegram_port_json_get_actual_scalar_variants() {
    let program = crate::ast::Program::default();
    let mut env = crate::eval::Env::new_with_shared_interner(&program.shared_interner);
    for (json, kind, expected) in [
        ("{\"v\":\"123\"}", "String", "123"),
        ("{\"v\":123}", "Int", "123"),
        ("{\"v\":1.25}", "Float", "1.25"),
        ("{\"v\":true}", "Bool", "true"),
        ("{\"v\":null}", "String", "null"),
        ("{\"v\":{\"x\":1}}", "String", "{\"x\":1}"),
        ("{\"v\":[1,2]}", "String", "[1,2]"),
        ("{}", "String", ""),
    ] {
        let actual = crate::eval::eval_builtin_or_user(
            &program,
            "json_get",
            strings(&[json, "v"]),
            &mut env,
        )
        .unwrap();
        match actual {
            Value::String(text) => {
                assert_eq!(kind, "String");
                assert_eq!(text.as_str(), expected);
            }
            Value::Int(n) => {
                assert_eq!(kind, "Int");
                assert_eq!(n.to_string(), expected);
            }
            Value::Float(n) => {
                assert_eq!(kind, "Float");
                assert_eq!(n, 1.25);
            }
            Value::Bool(b) => {
                assert_eq!(kind, "Bool");
                assert!(b);
            }
            other => panic!("unexpected json_get result {other}"),
        }
    }
}
