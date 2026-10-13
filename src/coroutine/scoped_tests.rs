use super::*;

fn guest() -> TurnIdentity {
    TurnIdentity { context: "telegram:user:42".into(), principal: "telegram:user:42".into(),
        may_write: false, guest_tools: Some(scoped::GUEST_TOOLS.iter().map(|s| s.to_string()).collect()) }
}

#[test]
fn scoped_tools_fail_closed_without_a_turn() {
    let (tx, _rx) = mpsc::channel(1);
    let handle = CoroutineHandle::new(tx);
    for op in ["conversation_history", "music_generate", "cad_prepare", "video_stabilize", "web_search", "web_read", "memory_inspect"] {
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
    assert!(handle.execute_await("video_stabilize", &[Value::string("/tmp/input.mp4")]).is_err());
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
        "!eval cad_prepare(\"referenced part\", \"match the photo\", \"PETG\", true)",
        "!eval web_search(\"Rust\")\n!done",
        "!eval video_stabilize()",
        "!eval memory_inspect()",
        "!eval memory_inspect(\"bicycle\")",
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
    assert!(err.contains("literal strings, numbers or booleans"), "got: {err}");
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
    let prompt = scoped::prompt(&["music_generate".into(), "video_stabilize".into(), "web_search".into()]);
    assert!(prompt.contains("music_generate(description, duration_seconds[, lyrics])"));
    assert!(prompt.contains("video_stabilize()"));
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
    assert!(handle.execute_await("cad_prepare", &[Value::string("x"),Value::string("code"),Value::string("PLA"),Value::Bool(true),Value::string("extra")]).is_err());
    let worker = std::thread::spawn(move || {
        for (expected_title, expected_material, expected_images) in
                [("test", "PLA", false), ("referenced", "PETG", true)] {
            match rx.blocking_recv().unwrap() {
                IoRequest::Scoped { request: scoped::Request::Cad { principal, context, title,
                        source, material, use_recent_images }, reply } => {
                    assert_eq!(principal,"telegram:user:42"); assert_eq!(context,"telegram:user:42");
                    assert_eq!(title,expected_title); assert_eq!(source,"python code");
                    assert_eq!(material,expected_material); assert_eq!(use_recent_images,expected_images);
                    reply.send(Ok("queued".into())).unwrap();
                }
                _ => panic!("wrong request"),
            }
        }
    });
    handle.execute_await("cad_prepare", &[Value::string("test"),Value::string("python code")]).unwrap();
    handle.execute_await("cad_prepare", &[Value::string("referenced"),Value::string("python code"),
        Value::string("PETG"),Value::Bool(true)]).unwrap();
    worker.join().unwrap();
}

#[test]
fn video_stabilize_is_bound_to_turn_and_accepts_no_paths_or_destination() {
    let (tx, mut rx) = mpsc::channel(1);
    let handle = CoroutineHandle::new(tx).with_turn(Some(guest()));
    assert!(handle.execute_await("video_stabilize", &[Value::string("/tmp/input.mp4")]).is_err());
    let worker = std::thread::spawn(move || {
        match rx.blocking_recv().unwrap() {
            IoRequest::Scoped {
                request: scoped::Request::VideoStabilize { principal, context },
                reply,
            } => {
                assert_eq!(principal, "telegram:user:42");
                assert_eq!(context, "telegram:user:42");
                reply.send(Ok("queued".into())).unwrap();
            }
            _ => panic!("wrong request"),
        }
    });
    assert_eq!(handle.execute_await("video_stabilize", &[]).unwrap().to_string(), "\"queued\"");
    worker.join().unwrap();
}

#[test]
fn guest_memory_inspection_is_bound_to_the_turn() {
    let (tx, mut rx) = mpsc::channel(1);
    let worker = std::thread::spawn(move || match rx.blocking_recv().unwrap() {
        IoRequest::MemoryInspect {
            context_id,
            principal_id,
            query,
            reply,
        } => {
            assert_eq!(context_id, "telegram:user:42");
            assert_eq!(principal_id, "telegram:user:42");
            assert_eq!(query, "metric units");
            reply.send(Ok("bounded inspection".into())).unwrap();
        }
        _ => panic!("wrong request"),
    });
    let result = CoroutineHandle::new(tx)
        .with_turn(Some(guest()))
        .execute_await("memory_inspect", &[Value::string("metric units")])
        .unwrap();
    assert_eq!(result.to_string(), "\"bounded inspection\"");
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
