use super::*;

fn guest() -> TurnIdentity {
    TurnIdentity { context: "telegram:user:42".into(), principal: "telegram:user:42".into(),
        may_write: false, guest_tools: Some(scoped::GUEST_TOOLS.iter().map(|s| s.to_string()).collect()) }
}

#[test]
fn scoped_tools_fail_closed_without_a_turn() {
    let (tx, _rx) = mpsc::channel(1);
    let handle = CoroutineHandle::new(tx);
    for op in ["conversation_history", "music_generate", "web_search", "web_read"] {
        let error = handle.execute_await(op, &[]).unwrap_err().to_string();
        assert!(error.contains("no conversational turn"), "{op}: {error}");
    }
}

#[test]
fn guest_blocks_raw_io_global_memory_and_foreign_destinations() {
    let (tx, _rx) = mpsc::channel(1);
    let handle = CoroutineHandle::new(tx).with_turn(Some(guest()));
    for op in ["shell_exec", "read_file", "http_get", "llm_takeover", "conversation_notify", "conversation_history", "memory_cypher", "llm_set_model"] {
        assert!(handle.execute_await(op, &[Value::string("x")]).unwrap_err().to_string().contains("permission denied"));
    }
    assert!(handle.execute_await("memory_remember", &[Value::string("note"), Value::string("global")]).is_err());
    assert!(handle.execute_await("music_generate", &[Value::string("song"), Value::Int(30), Value::string("lyrics"), Value::string("other-chat")]).is_err());
    assert!(handle.execute_await("music_generate", &[Value::string("song"), Value::Int(999)]).is_err());
    assert!(handle.execute_await("web_search", &[Value::Int(1)]).is_err());
    assert!(handle.execute_await("web_read", &[Value::string("")]).is_err());
}

#[test]
fn guest_code_is_checked_before_any_side_effects() {
    let program = crate::ast::Program::default();
    let tools = guest().guest_tools.unwrap();
    for code in [
        "!eval music_generate(\"piano\", 30, \"\")",
        "!eval web_search(\"Rust\")\n!done",
        "!eval memory_remember(\"I like jazz\")",
    ] {
        scoped::validate_guest_code(code, &tools, &program).unwrap();
    }
    for code in [
        "!eval web_search(read_file(\"/etc/passwd\"))", "!eval TelegramBot.init(\"a\", \"b\")",
        "!eval shell_exec(\"id\")", "!undo", "!agent hello", "?source TelegramBot",
        "+module Evil\n+fn x ()->Int\n  +return 1\n+end", "!eval web_search(\"ok\")\n!undo",
    ] {
        assert!(scoped::validate_guest_code(code, &tools, &program).is_err(), "accepted {code}");
    }
}

#[test]
fn guest_prompt_lists_only_granted_native_tools_with_call_signatures() {
    let prompt = scoped::prompt(&["music_generate".into(), "web_search".into()]);
    assert!(prompt.contains("music_generate(description, duration_seconds[, lyrics])"));
    assert!(prompt.contains("web_search(query)"));
    assert!(!prompt.contains("web_read(url)"));
    assert!(!prompt.contains("conversation_history("));
    assert!(!prompt.contains("shell_exec("));
}

#[test]
fn music_request_binds_delivery_to_the_actual_turn() {
    let (tx, mut rx) = mpsc::channel(1);
    let worker = std::thread::spawn(move || {
        match rx.blocking_recv().unwrap() {
            IoRequest::Scoped { request: scoped::Request::Music { context, caption, duration, lyrics }, reply } => {
                assert_eq!(context, "telegram:user:42");
                assert_eq!(caption, "piano"); assert_eq!(duration, 30); assert!(lyrics.is_empty());
                reply.send(Ok("started".into())).unwrap();
            }
            _ => panic!("unexpected request"),
        }
    });
    let handle = CoroutineHandle::new(tx).with_turn(Some(guest()));
    assert_eq!(handle.execute_await("music_generate", &[Value::string("piano"), Value::Int(30)]).unwrap().to_string(), "\"started\"");
    worker.join().unwrap();
}

#[test]
fn history_requires_admin_and_private_destination() {
    let (tx, _rx) = mpsc::channel(1);
    let mut identity = guest();
    identity.guest_tools = None;
    let handle = CoroutineHandle::new(tx.clone()).with_turn(Some(identity.clone()));
    assert!(handle.execute_await("conversation_history", &[Value::string("julia")]).is_err());
    identity.principal = crate::memory_graph::admin_principal().into();
    identity.context = "telegram:group:-1".into();
    let handle = CoroutineHandle::new(tx).with_turn(Some(identity));
    assert!(handle.execute_await("conversation_history", &[Value::string("julia")]).is_err());
}

#[test]
fn scoped_web_and_forget_requests_preserve_parameters() {
    for (op, argument) in [("web_search", "Rust language"), ("web_read", "https://rust-lang.org/"), ("memory_forget", "memory:42")] {
        let (tx, mut rx) = mpsc::channel(1);
        let worker = std::thread::spawn(move || {
            match rx.blocking_recv().unwrap() {
                IoRequest::Scoped { request, reply } => {
                    match request {
                        scoped::Request::Search { query } => assert_eq!(query, argument),
                        scoped::Request::Read { url } => assert_eq!(url, argument),
                        scoped::Request::Forget { principal, context, memory_id } => {
                            assert_eq!(principal, "telegram:user:42"); assert_eq!(context, principal); assert_eq!(memory_id, argument);
                        }
                        _ => panic!("wrong scoped operation"),
                    }
                    reply.send(Ok("ok".into())).unwrap();
                }
                _ => panic!("wrong IO request"),
            }
        });
        CoroutineHandle::new(tx).with_turn(Some(guest())).execute_await(op, &[Value::string(argument)]).unwrap();
        worker.join().unwrap();
    }
}
