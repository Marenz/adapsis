//! Reviewed #91 source repair, not equivalence to the malformed original.
//! Trusted Result values stay in Rust tests; Python literal policy is unchanged.
use crate::{
    ast,
    attachment::Attachment,
    coroutine::{CoroutineHandle, IoRequest},
    eval, parser,
};
use serde_json::json;

const FAILED: &str = "[voice message - transcription failed]";
const NO_FILE: &str = "[voice message - could not get file]";
const HELPER: &str = "TelegramBot.transcription_upload_response";

pub(super) fn verify_helper(original: &ast::FunctionDecl, corrected: &ast::FunctionDecl) {
    let mut restored = corrected.clone();
    let ast::StatementKind::Match { arms, .. } = &mut restored.body[0].kind else {
        panic!()
    };
    assert!(matches!(
        arms[0].body[0].kind,
        ast::StatementKind::Branch { .. }
    ));
    arms[0].body.remove(0);
    assert_eq!(
        super::qualification::verify(original, &restored, "TelegramBot", &Default::default()),
        0
    );
}

#[test]
fn telegram_port_voice_authored_repair_is_explicit() {
    let operations = parser::parse(super::FROZEN).unwrap();
    let parser::Operation::Module(module) = &operations[0] else {
        panic!()
    };
    let old = module
        .body
        .iter()
        .find_map(|op| match op {
            parser::Operation::Function(f) if f.name == "transcribe_voice" => Some(f),
            _ => None,
        })
        .unwrap();
    let manifest: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/telegram-python-port/manifest.json"
    ))
    .unwrap();
    assert_eq!(
        manifest["modules"][0]["semantic_repairs"]["transcribe_voice"]["kind"],
        "voice_control_flow_v1"
    );
    let ledger = &manifest["modules"][0];
    assert_eq!(
        ledger["trusted_result_contract_functions"],
        json!(["transcription_upload_response"])
    );
    assert_eq!(
        ledger["contract_tests"]["telegram_port_voice_helper_trusted_results"],
        json!(["transcription_upload_response"])
    );
    assert_eq!(
        ledger["contract_tests"]["telegram_port_voice_normal_async_contracts"],
        json!(["transcribe_voice", "transcription_upload_response"])
    );
    // Preserve the source defect as evidence; do NOT validate it and silently
    // normalize away the duplicate Err / out-of-scope e.
    assert!(super::FROZEN.contains(
        "    +case Err(e)\n    +return concat(\"[voice message - getFile failed: \", e, \"]\")"
    ));
    let corrected = parser::parse(super::ORIGIN).unwrap();
    let parser::Operation::Module(module) = &corrected[0] else {
        panic!()
    };
    let repaired = module
        .body
        .iter()
        .find_map(|op| match op {
            parser::Operation::Function(f) if f.name == "transcribe_voice" => Some(f),
            _ => None,
        })
        .unwrap();
    for declaration in [old, repaired] {
        assert_eq!(declaration.params.len(), 1);
        assert_eq!(declaration.params[0].name, "file_id");
        assert!(
            matches!(&declaration.params[0].ty, parser::TypeExpr::Named(name) if name == "String")
        );
        assert!(
            matches!(&declaration.return_type, parser::TypeExpr::Named(name) if name == "String")
        );
    }
    for program in super::send_chain::programs() {
        let fixed = program
            .get_function("TelegramBot.transcribe_voice")
            .unwrap();
        assert_eq!(fixed.params.len(), 1);
        assert_eq!(fixed.params[0].name, "file_id");
        assert_eq!(fixed.params[0].ty, ast::Type::String);
        assert_eq!(fixed.return_type, ast::Type::String);
        assert_eq!(fixed.doc, old.doc);
        assert_eq!(
            fixed.effects,
            [ast::Effect::Io, ast::Effect::Fail, ast::Effect::Async]
        );
    }
}

fn helper_cases() -> Vec<(eval::Value, Result<String, &'static str>)> {
    let mut cases = vec![(eval::Value::Err("upload failure".into()), Ok(FAILED.into()))];
    for text in ["hello", "Grüße 😀\nline two", "null", "{}", "[]", ""] {
        cases.push((
            eval::Value::Ok(Box::new(eval::Value::string(
                json!({"text":text}).to_string(),
            ))),
            Ok(if text.is_empty() { FAILED } else { text }.into()),
        ));
    }
    for body in [
        json!({}),
        json!({"error":"ASR failure"}),
        json!({"text":null}),
        json!({"text":true}),
        json!({"text":false}),
        json!({"text":1}),
        json!({"text":1.5}),
        json!({"text":[]}),
        json!({"text":{}}),
    ] {
        cases.push((
            eval::Value::Ok(Box::new(eval::Value::string(body.to_string()))),
            Ok(FAILED.into()),
        ));
    }
    cases.push((
        eval::Value::Ok(Box::new(eval::Value::string("not json"))),
        Err("json_kind: invalid JSON:"),
    ));
    cases
}

fn check_result(actual: anyhow::Result<eval::Value>, expected: &Result<String, &'static str>) {
    match expected {
        Ok(text) => {
            assert!(matches!(actual.unwrap(), eval::Value::String(s) if s.as_str() == text))
        }
        Err(prefix) => assert!(actual.unwrap_err().to_string().starts_with(prefix)),
    }
}

#[test]
fn telegram_port_voice_helper_trusted_results() {
    for program in super::send_chain::programs() {
        for (input, expected) in helper_cases() {
            check_result(
                super::super::logical_tier_tests::tree(&program, HELPER, vec![input.clone()]),
                &expected,
            );
            check_result(
                super::super::logical_tier_tests::vm_run(&program, HELPER, vec![input]),
                &expected,
            );
        }
    }
}

#[derive(Clone)]
struct Scenario {
    info: Result<String, &'static str>,
    path: String,
    binary_fail: bool,
    upload: Result<String, &'static str>,
}
impl Scenario {
    fn base() -> Self {
        Self {
            info: Ok(json!({"result":{"file_path":"voice/file.ogg"}}).to_string()),
            path: "voice/file.ogg".into(),
            binary_fail: false,
            upload: Ok(json!({"text":"hello"}).to_string()),
        }
    }
}

fn run(
    program: &ast::Program,
    scenario: Scenario,
) -> (anyhow::Result<eval::Value>, Vec<&'static str>) {
    let _state = super::routing::install(program, "1", "7");
    let (sender, receiver) = tokio::sync::mpsc::channel(4);
    let worker = std::thread::spawn(move || intercept(receiver, scenario));
    let result = eval::eval_async_function(
        program,
        "TelegramBot.transcribe_voice",
        &parser::parse_single_expr("{file_id:\"voice-1\"}").unwrap(),
        CoroutineHandle::new(sender),
    );
    (result, worker.join().unwrap())
}

fn intercept(
    mut receiver: tokio::sync::mpsc::Receiver<IoRequest>,
    s: Scenario,
) -> Vec<&'static str> {
    let mut events = Vec::new();
    while let Some(request) = receiver.blocking_recv() {
        assert!(events.len() < 3, "unexpected fourth request/retry");
        match request {
            IoRequest::HttpGet { url, reply } => {
                assert!(events.is_empty());
                assert_eq!(url, "<redacted>fixture-token/getFile?file_id=voice-1");
                events.push("getFile");
                reply
                    .send(s.info.clone().map_err(anyhow::Error::msg))
                    .unwrap();
            }
            IoRequest::HttpPostBinary {
                url,
                body,
                content_type,
                reply,
            } => {
                assert_eq!(events, ["getFile"]);
                assert_eq!(url, format!("<redacted>fixture-token/{}", s.path));
                assert!(body.is_empty());
                assert_eq!(content_type, "application/octet-stream");
                events.push("binary");
                reply
                    .send(if s.binary_fail {
                        Err(anyhow::anyhow!("fixture download failure"))
                    } else {
                        Ok(
                            Attachment::from_bytes(vec![1, 2, 3], "audio/ogg", "voice.ogg")
                                .unwrap(),
                        )
                    })
                    .unwrap();
            }
            IoRequest::HttpUploadBytes {
                url,
                bytes,
                file_name,
                mime_type,
                file_field,
                extra_fields,
                reply,
            } => {
                assert_eq!(events, ["getFile", "binary"]);
                assert_eq!(url, "http://127.0.0.1:8090/transcribe");
                assert_eq!(bytes, [1, 2, 3]);
                assert_eq!(file_name, "voice.ogg");
                assert_eq!(mime_type, "audio/ogg");
                assert_eq!(file_field, "file");
                assert!(extra_fields.is_empty());
                events.push("upload");
                reply
                    .send(s.upload.clone().map_err(anyhow::Error::msg))
                    .unwrap();
            }
            _ => panic!("unexpected request; no takeover, send, file or live IO permitted"),
        }
    }
    events
}

fn check(
    program: &ast::Program,
    s: Scenario,
    expected: Result<String, &'static str>,
    events: &[&str],
) {
    let (actual, seen) = run(program, s);
    check_result(actual, &expected);
    assert_eq!(seen, events);
}

#[test]
fn telegram_port_voice_normal_async_contracts() {
    for program in super::send_chain::programs() {
        // Every helper Result case must also traverse the real async call chain.
        for (response, expected) in helper_cases() {
            let upload = match response {
                eval::Value::Ok(value) => match *value {
                    eval::Value::String(s) => Ok(s.to_string()),
                    _ => panic!(),
                },
                eval::Value::Err(_) => Err("fixture upload failure"),
                _ => panic!(),
            };
            check(
                &program,
                Scenario {
                    upload,
                    ..Scenario::base()
                },
                expected,
                &["getFile", "binary", "upload"],
            );
        }
        lookup_cases(&program);
        check(
            &program,
            Scenario {
                binary_fail: true,
                ..Scenario::base()
            },
            Err("fixture download failure"),
            &["getFile", "binary"],
        );
    }
}

fn lookup_cases(program: &ast::Program) {
    check(
        program,
        Scenario {
            info: Err("lookup lost"),
            ..Scenario::base()
        },
        Ok("[voice message - getFile failed: lookup lost]".into()),
        &["getFile"],
    );
    check(
        program,
        Scenario {
            info: Ok("not json".into()),
            ..Scenario::base()
        },
        Err("json_kind: invalid JSON:"),
        &["getFile"],
    );
    for body in [
        json!({}),
        json!({"description":"file is too big"}),
        json!({"result":{"file_path":""}}),
        json!({"result":{"file_path":null}}),
        json!({"result":{"file_path":false}}),
        json!({"result":{"file_path":true}}),
        json!({"result":{"file_path":1}}),
        json!({"result":{"file_path":1.5}}),
        json!({"result":{"file_path":[]}}),
        json!({"result":{"file_path":{}}}),
    ] {
        check(
            program,
            Scenario {
                info: Ok(body.to_string()),
                ..Scenario::base()
            },
            Ok(NO_FILE.into()),
            &["getFile"],
        );
    }
    for path in ["null", "{}", "[]", "stimme/ä.ogg"] {
        check(
            program,
            Scenario {
                info: Ok(json!({"result":{"file_path":path}}).to_string()),
                path: path.into(),
                ..Scenario::base()
            },
            Ok("hello".into()),
            &["getFile", "binary", "upload"],
        );
    }
}
