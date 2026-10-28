//! Characterize the frozen http_post -> user-await -> response-consumer boundary.
//! This is not a corrected-origin transformation or a new wrapper coercion.
use crate::{
    coroutine::{CoroutineHandle, IoRequest},
    eval::{self, Value},
    parser,
};

const TARGETS: [&str; 4] = [
    "send_message_part",
    "send_message_parts",
    "send_message_body",
    "telegram_response_ok",
];

fn frozen_wrapper_program(qualified: bool) -> crate::ast::Program {
    // Explicit test-only transformation; never overwrite or relabel the frozen
    // fixture. Definitions use `name (` while calls use `name(` in this source.
    let mut source = super::FROZEN.to_string();
    if qualified {
        for name in TARGETS {
            source = source.replace(&format!("{name}("), &format!("TelegramBot.{name}("));
        }
    }
    let mut operations = parser::parse(&source).unwrap();
    let parser::Operation::Module(module) = &mut operations[0] else {
        panic!()
    };
    module.doc = None;
    module.body.retain(|operation| match operation {
        parser::Operation::Function(f) => TARGETS.contains(&f.name.as_str()),
        parser::Operation::SharedVar(shared) => shared.name == "telegram_token",
        _ => false,
    });
    super::program(&operations)
}

fn undo_call(call: &mut crate::ast::CallExpr) {
    if let Some(bare) = call.callee.strip_prefix("TelegramBot.") {
        assert!(TARGETS.contains(&bare));
        call.callee = bare.into();
    }
    for argument in &mut call.args {
        undo_expression(argument);
    }
}

fn undo_expression(expr: &mut crate::ast::Expr) {
    use crate::ast::Expr as E;
    match expr {
        E::Call(call) => undo_call(call),
        E::Binary { left, right, .. } => {
            undo_expression(left);
            undo_expression(right);
        }
        E::Unary { expr, .. } => undo_expression(expr),
        E::Literal(_) | E::Identifier(_) => {}
        _ => panic!("new source expression requires ledger review"),
    }
}

fn undo_body(body: &mut [crate::ast::Statement]) {
    use crate::ast::StatementKind as S;
    for statement in body {
        match &mut statement.kind {
            S::Let { value, .. } | S::Return { value } => undo_expression(value),
            S::Await { call, .. } => undo_call(call),
            S::Branch {
                condition,
                then_body,
                else_body,
            } => {
                undo_expression(condition);
                undo_body(then_body);
                undo_body(else_body);
            }
            _ => panic!("new source statement requires ledger review"),
        }
    }
}

#[test]
fn telegram_port_wrapper_transformation_is_only_call_qualification() {
    let frozen = frozen_wrapper_program(false);
    let qualified = frozen_wrapper_program(true);
    for name in TARGETS {
        let target = format!("TelegramBot.{name}");
        let mut changed = qualified.get_function(&target).unwrap().clone();
        undo_body(&mut changed.body);
        assert!(
            super::super::ast_equality::function(&changed, frozen.get_function(&target).unwrap()),
            "{target}"
        );
    }
}

#[test]
fn telegram_port_http_post_wrapper_boundary_observation() {
    for direct_tree in [false, true] {
        for qualified in [false, true] {
            for parts in [false, true] {
                for fail in [false, true] {
                    let program = frozen_wrapper_program(qualified);
                    let (sender, receiver) = tokio::sync::mpsc::channel(4);
                    let responder = std::thread::spawn(move || respond_posts(fail, receiver));
                    let runtime = std::sync::Arc::new(std::sync::RwLock::new(
                        crate::session::RuntimeState::default(),
                    ));
                    eval::set_shared_runtime(Some(runtime.clone()));
                    eval::set_shared_program(Some(std::sync::Arc::new(program.clone())));
                    eval::init_missing_shared_runtime_vars(&program, &runtime);
                    let (name, input) = if parts {
                        (
                            "TelegramBot.send_message_parts",
                            "{chat_id:1,parts:list(\"hi\"),i:0,last:\"\"}",
                        )
                    } else {
                        (
                            "TelegramBot.send_message_part",
                            "{chat_id:1,text:\"hi\",html:true}",
                        )
                    };
                    let result = run_entry(
                        &program,
                        name,
                        &parser::parse_single_expr(input).unwrap(),
                        CoroutineHandle::new(sender),
                        direct_tree,
                    );
                    eval::set_shared_runtime(None);
                    eval::set_shared_program(None);
                    let requests = responder.join().unwrap();
                    println!("WRAPPER_OBSERVATION direct_tree={direct_tree} qualified={qualified} parts={parts} fail={fail} requests={} result={result:?}", requests.len());
                    // #82 dispatch admission now chooses tree BEFORE execution
                    // for the normal route too. Result/error wrapping itself is
                    // unchanged: transport Err still fails at the consumer.
                    assert_eq!(requests.len(), 1);
                    if !parts {
                        match result.unwrap() {
                            Value::Ok(inner) if !fail => assert!(
                                matches!(*inner, Value::String(ref text) if text.as_str() == "{\"ok\":true}")
                            ),
                            Value::Err(error) if fail => {
                                assert!(error.contains("fixture transport failure"))
                            }
                            other => panic!("unexpected direct wrapper: {other}"),
                        }
                    } else if fail {
                        assert_eq!(
                            result.unwrap_err().to_string(),
                            "index_of expects (String, String)"
                        );
                    } else {
                        assert!(
                            matches!(result.unwrap(), Value::String(ref text) if text.as_str() == "{\"ok\":true}")
                        );
                    }
                }
            }
        }
    }
}

fn respond_posts(fail: bool, mut receiver: tokio::sync::mpsc::Receiver<IoRequest>) -> Vec<String> {
    let mut requests = Vec::new();
    while let Some(request) = receiver.blocking_recv() {
        let IoRequest::HttpPost {
            url,
            body,
            content_type,
            reply,
        } = request
        else {
            panic!("only inert HttpPost reply")
        };
        assert_eq!(url, "<redacted>/sendMessage");
        assert_eq!(content_type, "application/json");
        assert!(body.contains("\"chat_id\":1"));
        assert!(body.contains("\"text\":\"hi\""));
        requests.push(body);
        assert!(requests.len() <= 2, "unbounded retry");
        reply
            .send(if fail {
                Err(anyhow::anyhow!("fixture transport failure"))
            } else {
                Ok("{\"ok\":true}".into())
            })
            .unwrap();
    }
    requests
}

fn run_entry(
    program: &crate::ast::Program,
    name: &str,
    input: &parser::Expr,
    handle: CoroutineHandle,
    direct_tree: bool,
) -> anyhow::Result<Value> {
    if !direct_tree {
        return eval::eval_async_function(program, name, input, handle);
    }
    // Explicit diagnostic route, NOT evidence normal dispatch succeeded.
    let function = program.get_function(name).unwrap();
    let parser::Expr::StructLiteral(fields) = input else {
        panic!()
    };
    assert_eq!(fields.len(), function.params.len());
    let mut env = eval::Env::new_with_shared_interner(&program.shared_interner);
    env.populate_shared_from_program(program);
    env.set("__coroutine_handle", Value::CoroutineHandle(handle));
    for field in fields {
        assert!(function.params.iter().any(|param| param.name == field.name));
        env.set(
            &field.name,
            eval::eval_parser_expr_with_program(&field.value, program)?,
        );
    }
    eval::eval_function_body_named(program, name, &function.body, &mut env)
}
