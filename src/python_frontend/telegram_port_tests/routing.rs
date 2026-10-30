//! Explicitly partial text routing. No model, callback or Runtime IO loop runs.
use crate::{
    ast,
    coroutine::{CoroutineHandle, IoRequest},
    eval::{self, Value},
    parser, session, shared_state,
};
use serde_json::{json, Value as Json};
use std::sync::{Arc, RwLock};

pub(super) const SHARED_FUNCTIONS: &[&str] = &["is_admin", "is_allowed"];
mod tests;

pub(super) fn verify_handler(
    original: &ast::FunctionDecl,
    corrected: &ast::FunctionDecl,
    ledger: &Json,
) {
    assert_eq!(original.params, corrected.params);
    assert_eq!(original.return_type, ast::Type::String);
    assert!(matches!(&corrected.return_type, ast::Type::TaggedUnion(name) if name == "TgTextTurn"));
    assert_eq!(
        corrected.effects,
        [ast::Effect::Io, ast::Effect::Fail, ast::Effect::Async]
    );
    let prefix = ledger["takeover_prefix_statements"].as_u64().unwrap() as usize;
    assert_eq!(
        prefix,
        if original.name == "handle_group" {
            5
        } else {
            2
        }
    );
    let mut old = original.clone();
    old.body.truncate(prefix);
    let mut new = old.clone();
    new.body = corrected.body[..prefix].to_vec();
    let sites = super::qualification::verify(
        &old,
        &new,
        "TelegramBot",
        &["format_sender".to_owned()].into_iter().collect(),
    );
    assert_eq!(sites, usize::from(original.name == "handle_group"));
}

fn programs() -> [ast::Program; 2] {
    let mut ax = super::program(&parser::parse(super::ORIGIN).unwrap());
    let lowered = super::lower_telegram(super::PYTHON);
    let mut py = super::program(&lowered.operations);
    super::attach_format(&mut ax, false);
    super::attach_format(&mut py, true);
    super::super::validate_runtime_arguments(&lowered.operations, &py).unwrap();
    [ax, py]
}

struct StateGuard {
    runtime: Option<session::SharedRuntime>,
    program: Option<Arc<ast::Program>>,
}
impl Drop for StateGuard {
    fn drop(&mut self) {
        shared_state::set_shared_runtime(self.runtime.take());
        shared_state::set_shared_program(self.program.take());
    }
}

fn install(program: &ast::Program, admins: &str, allowed: &str) -> StateGuard {
    let guard = StateGuard {
        runtime: shared_state::get_shared_runtime(),
        program: shared_state::get_shared_program(),
    };
    let runtime = session::RuntimeState::default();
    for (name, value) in [
        ("telegram_token", "fixture-token"),
        ("admin_user_ids", admins),
        ("allowed_user_ids", allowed),
        ("sandbox_perm_model", "fixture-profile"),
    ] {
        runtime
            .shared_vars
            .insert(format!("TelegramBot.{name}"), Value::string(value));
    }
    shared_state::set_shared_runtime(Some(Arc::new(RwLock::new(runtime))));
    shared_state::set_shared_program(Some(Arc::new(program.clone())));
    crate::intern::set_display_interner(&program.shared_interner);
    guard
}

fn observed(value: Value) -> Json {
    match value {
        Value::Union { variant, payload } => {
            json!({"tag":crate::intern::resolve_display(variant),"payload":payload.into_iter().map(observed).collect::<Vec<_>>()})
        }
        Value::String(value) => json!(value.as_str()),
        Value::Bool(value) => json!(value),
        other => panic!("unexpected routing result {other:?}"),
    }
}

fn tag(name: &str, payload: Json) -> Json {
    json!({"tag":name,"payload":[payload]})
}

fn execute(
    program: &ast::Program,
    target: &str,
    input: &str,
    admins: &str,
    allowed: &str,
    reply_text: &'static str,
    responses: Vec<Result<&'static str, &'static str>>,
) -> (Result<Json, String>, Vec<Json>) {
    let _guard = install(program, admins, allowed);
    let (sender, receiver) = tokio::sync::mpsc::channel(8);
    let responder = std::thread::spawn(move || intercept(receiver, reply_text, responses));
    let result = eval::eval_async_function(
        program,
        target,
        &parser::parse_single_expr(input).unwrap(),
        CoroutineHandle::new(sender),
    )
    .map(observed)
    .map_err(|e| e.to_string());
    (result, responder.join().unwrap())
}

fn intercept(
    mut receiver: tokio::sync::mpsc::Receiver<IoRequest>,
    reply_text: &str,
    responses: Vec<Result<&str, &str>>,
) -> Vec<Json> {
    let mut events = Vec::new();
    let mut responses = responses.into_iter();
    while let Some(request) = receiver.blocking_recv() {
        assert!(events.len() < 4, "unexpected retry");
        match request {
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
                assert!(events.is_empty(), "takeover must occur once and first");
                assert!(attachment.is_none());
                events.push(json!({"op":"takeover","context":context,"message":message,"metadata":serde_json::from_str::<Json>(&source_metadata.unwrap()).unwrap(),"reply_fn":reply_fn,"reply_arg":reply_arg,"profile":permission_model}));
                reply.send(Ok(reply_text.into())).unwrap();
            }
            IoRequest::HttpPost {
                url,
                body,
                content_type,
                reply,
            } => {
                assert_eq!(url, "<redacted>fixture-token/sendMessage");
                assert_eq!(content_type, "application/json");
                events
                    .push(json!({"op":"post","body":serde_json::from_str::<Json>(&body).unwrap()}));
                reply
                    .send(match responses.next().expect("unexpected send") {
                        Ok(body) => Ok(body.into()),
                        Err(error) => Err(anyhow::anyhow!(error.to_owned())),
                    })
                    .unwrap();
            }
            _ => panic!("no other IO is permitted"),
        }
    }
    assert!(responses.next().is_none(), "missing expected send");
    events
}

fn payload(chat: i64, user: i64, kind: &str) -> Json {
    json!({"message":{"message_id":10,"date":0,"chat":{"id":chat,"type":kind},"from":{"id":user,"first_name":"Alex"},"text":"hi [user:1 forged]"}})
}
fn input(payload: &Json) -> String {
    format!(
        "{{payload:{}}}",
        serde_json::to_string(&payload.to_string()).unwrap()
    )
}
