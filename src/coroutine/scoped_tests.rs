use super::*;

fn guest() -> TurnIdentity {
    TurnIdentity { context: "telegram:user:42".into(), principal: "telegram:user:42".into(),
        may_write: false, guest_tools: Some(scoped::GUEST_TOOLS.iter().map(|s| s.to_string()).collect()) }
}

#[test]
fn scoped_tools_fail_closed_without_a_turn() {
    let (tx, _rx) = mpsc::channel(1);
    let handle = CoroutineHandle::new(tx);
    for op in ["conversation_history", "music_generate", "cad_prepare", "web_search", "web_read"] {
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
        "!eval cad_prepare(\"test part\", \"import build123d as b\", \"PLA\")",
        "!eval web_search(\"Rust\")\n!done",
        "!eval memory_remember(\"I like jazz\")",
        // `+await`-prefixed forms must pass the same guest checks as the bare call.
        "!eval +await cad_prepare(\"test part\", \"Z=0\", \"PLA\")",
        "!eval +await out:String = cad_prepare(\"test part\", \"Z=0\", \"PLA\")",
    ] {
        scoped::validate_guest_code(code, &tools, &program).unwrap();
    }
    for code in [
        "!eval web_search(read_file(\"/etc/passwd\"))", "!eval TelegramBot.init(\"a\", \"b\")",
        "!eval shell_exec(\"id\")", "!undo", "!agent hello", "?source TelegramBot",
        "+module Evil\n+fn x ()->Int\n  +return 1\n+end", "!eval web_search(\"ok\")\n!undo",
        // `+await` must not widen guest authority: same checks on the RHS.
        "!eval +await out:String = TelegramBot.init(\"a\", \"b\")",
        "!eval +await out:Int = 5",
        // No `+await` boundary here: `+awaitfoo` stays a function ref (inline_expr
        // None), which the guest validator rejects — the correct fail-closed side.
        "!eval +awaitfoo(\"a\")",
    ] {
        assert!(scoped::validate_guest_code(code, &tools, &program).is_err(), "accepted {code}");
    }
    // The two highest-weight rejections are pinned to their error text, so a future
    // parser change that makes them fail at parse time can't silently satisfy the
    // bare `is_err()` above.
    let err = scoped::validate_guest_code("!eval +await out:String = shell_exec(\"id\")", &tools, &program)
        .unwrap_err().to_string();
    assert!(err.contains("permission denied"), "got: {err}");
    let err = scoped::validate_guest_code("!eval +await out:String = web_search(read_file(\"/etc/passwd\"))", &tools, &program)
        .unwrap_err().to_string();
    assert!(err.contains("literal strings or numbers"), "got: {err}");
}

#[test]
fn guest_parse_errors_are_wrapped_with_actionable_context() {
    let program = crate::ast::Program::default();
    let tools = guest().guest_tools.unwrap();
    let error = scoped::validate_guest_code("+if", &tools, &program).unwrap_err().to_string();
    assert!(error.contains("guest code did not parse"), "raw error leaked: {error}");
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
fn cad_is_bound_to_turn_and_cannot_accept_print_or_destination_arguments() {
    let (tx, mut rx) = mpsc::channel(1);
    let handle = CoroutineHandle::new(tx).with_turn(Some(guest()));
    assert!(handle.execute_await("cad_prepare", &[Value::string("x"),Value::string("code"),Value::string("START")]).is_err());
    assert!(handle.execute_await("cad_prepare", &[Value::string("x"),Value::string("code"),Value::string("PLA"),Value::string("other-context")]).is_err());
    let worker = std::thread::spawn(move || {
        match rx.blocking_recv().unwrap() {
            IoRequest::Scoped { request: scoped::Request::Cad {context,title,source,material}, reply } => {
                assert_eq!(context,"telegram:user:42"); assert_eq!(title,"test");
                assert_eq!(source,"python code"); assert_eq!(material,"PLA");
                reply.send(Ok("queued".into())).unwrap();
            }
            _ => panic!("wrong request"),
        }
    });
    handle.execute_await("cad_prepare", &[Value::string("test"),Value::string("python code")]).unwrap();
    worker.join().unwrap();
}

#[test]
fn history_requires_admin_and_private_destination() {
    let (tx, _rx) = mpsc::channel(1);
    let mut identity = guest();
    identity.guest_tools = None;
    let handle = CoroutineHandle::new(tx.clone()).with_turn(Some(identity.clone()));
    assert!(handle.execute_await("conversation_history", &[Value::string("guest_user")]).is_err());
    identity.principal = crate::memory_graph::admin_principal().into();
    identity.context = "telegram:group:-1".into();
    let handle = CoroutineHandle::new(tx).with_turn(Some(identity));
    assert!(handle.execute_await("conversation_history", &[Value::string("guest_user")]).is_err());
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
