//! Native `run` (#129): who may run what. The block is authorized as the function it is
//! lowered to, the Program is never written back, and no IO-boundary guard is bypassed.
use super::run_tests::*;
use super::*;
use serde_json::json;

fn turn(may_write: bool, guest_tools: Option<Vec<String>>) -> TurnIdentity {
    TurnIdentity {
        context: "telegram:group:-1".into(),
        principal: "telegram:user:100000103".into(),
        may_write,
        guest_tools,
    }
}

#[tokio::test]
async fn run_is_offered_exactly_where_mutate_is() {
    let (cfg, _ws, _baseline) = fixture("writer").await;
    let names: Vec<String> = tool_definitions(&cfg).iter().map(|d| d.name.clone()).collect();
    assert_eq!(names, ["inspect", "mutate", "test", "evaluate", "run", "finish"]);
    let definition = tool_definitions(&cfg).into_iter().find(|d| d.name == "run").unwrap();
    assert_eq!(definition.parameters["additionalProperties"], false);
    assert_eq!(definition.parameters["required"], json!(["source"]));
    let properties = definition.parameters["properties"].as_object().unwrap();
    assert_eq!(properties.keys().map(String::as_str).collect::<Vec<_>>(), ["language", "module", "source"]);
    assert!(native_tools_eligible(&cfg));
}

/// Speakers who cannot change code have `run` in its restricted form (DESIGN L3; the
/// details are in `run_restricted_tests.rs`). `mutate` stays refused for them by the
/// pinned snapshot, and a block that reaches for anything beyond pure builtins and the
/// speaker's own tools is refused at admission, before any IO.
#[tokio::test]
async fn restricted_speakers_cannot_mutate_and_their_blocks_cannot_reach_io() {
    let source = json!({"source": "text: str = await file_read(\"fixture://forged\")\nreturn text\n"}).to_string();
    let mutate = json!({"source": "+module NtRunForged\n+fn f ()->Int\n  +return 1\n+end\n"}).to_string();
    for (model, identity, reason) in [
        ("executor", None, "not write-capable"),
        ("writer", Some(turn(false, None)), "not write-capable"),
        ("writer", Some(turn(true, Some(vec![]))), "guest turns are not eligible"),
        ("writer", Some(turn(true, Some(vec!["run".into(), "file_read".into()]))), "guest turns are not eligible"),
    ] {
        let (mut cfg, mut ws, baseline) = fixture(model).await;
        cfg.turn = identity;
        let log = memory_io(&mut cfg, "must not be read");
        assert!(!native_tools_eligible(&cfg), "{model}");
        assert!(speaker_class(&cfg).restricted(), "{model}");
        let before = snapshot(&cfg, &ws);
        for (tool, arguments, fragment) in [
            ("mutate", &mutate, reason),
            ("run", &source, "`file_read` is not available in this conversation"),
        ] {
            let preflight = validate_native_tool_authorized(tool, arguments, &cfg).unwrap_err();
            assert!(preflight.contains(fragment), "{model} {tool}: {preflight}");
            let result = execute_native_tool(tool, arguments, &cfg, &mut ws, &baseline).await;
            assert!(result.execution.has_errors && result.content.contains(fragment), "{model} {tool}: {}", result.content);
        }
        assert!(log.lock().unwrap().is_empty(), "{model}: a refused turn performed IO");
        assert_nothing_persisted(&cfg, &ws, &baseline, &before).await;
    }
}

#[tokio::test]
async fn references_need_execute_and_the_scope_module_needs_write() {
    // `caller`: Execute on NtRunLib, none on NtRunSecret, Write on unlisted modules.
    let (mut cfg, mut ws, baseline) = fixture("caller").await;
    let log = memory_io(&mut cfg, "");
    let before = snapshot(&cfg, &ws);
    let allowed = run_source(&cfg, &mut ws, &baseline, "return NtRunLib.double(3)\n").await;
    assert_eq!(output(&allowed), "6", "{}", allowed.content);
    for (arguments, fragment) in [
        (json!({"source": "return NtRunSecret.token()\n"}), "permission denied for authored reference `NtRunSecret.token` (need Execute)"),
        (json!({"source": "t: str = NtRunSecret.token()\nreturn NtRunLib.double(1)\n"}), "permission denied for authored reference"),
        (json!({"source": "return double(3)\n", "module": "NtRunLib"}), "permission denied: cannot modify module 'NtRunLib' (need Write)"),
        (json!({"source": "return token()\n", "module": "NtRunSecret"}), "permission denied: cannot modify module 'NtRunSecret' (need Write)"),
    ] {
        // The batch preflight (no Program access) and dispatch agree.
        let preflight = validate_native_tool_authorized("run", &arguments.to_string(), &cfg).unwrap_err();
        assert!(preflight.contains(fragment), "{arguments}: {preflight}");
        let result = run(&cfg, &mut ws, &baseline, arguments.clone()).await;
        assert!(result.execution.has_errors && result.content.contains(fragment), "{arguments}: {}", result.content);
        assert!(!result.content.contains("hidden-token"));
    }
    assert!(log.lock().unwrap().is_empty());
    assert_nothing_persisted(&cfg, &ws, &baseline, &before).await;

    // `libonly`: Write on NtRunLib only. It is write-capable, so the tool is offered, but
    // an unscoped block needs what a new unlisted module would need.
    let (mut cfg, mut ws, baseline) = fixture("libonly").await;
    memory_io(&mut cfg, "");
    assert!(native_tools_eligible(&cfg));
    let unscoped = run_source(&cfg, &mut ws, &baseline, "return NtRunLib.double(3)\n").await;
    assert!(unscoped.execution.has_errors && unscoped.content.contains("run without `module` needs Write on unlisted (user) modules"), "{}", unscoped.content);
    assert!(!unscoped.content.contains(SCOPE));
    let scoped = run(&cfg, &mut ws, &baseline, json!({"source": "return double(3)\n", "module": "NtRunLib"})).await;
    assert_eq!(output(&scoped), "6", "{}", scoped.content);
    // The same model could not declare this function in a user module either.
    let declared = execute_native_tool(
        "mutate",
        &json!({"language":"python","module":"NtRunUserModule","source":"def f() -> int:\n    return NtRunLib.double(3)\n"}).to_string(),
        &cfg, &mut ws, &baseline,
    ).await;
    assert!(declared.execution.has_errors && declared.content.contains("need Write"), "{}", declared.content);
}

/// A block is admitted exactly when the same statements are admitted as a function body
/// submitted through `mutate` (given the annotations a function needs).
#[tokio::test]
async fn admission_agrees_with_mutate_for_the_same_statements() {
    let cases = [
        ("return NtRunLib.double(2)\n", "int"),
        ("return NtRunSecret.token()\n", "str"),
        ("return NtRunLib.missing(2)\n", "int"),
        ("return NtRunLib.double(\"two\")\n", "int"),
        ("text: str = await NtRunLib.read(\"fixture://x\")\nreturn text\n", "str"),
        ("text: str = NtRunLib.read(\"fixture://x\")\nreturn text\n", "str"),
        ("n: int = await NtRunLib.double(2)\nreturn n\n", "int"),
        ("n: int = await NtRunLib.read(\"fixture://x\")\nreturn n\n", "int"),
        ("r: Result[str] = await NtRunLib.read(\"fixture://x\")\nreturn 1\n", "int"),
        ("xs: list[int] = [1, \"two\"]\nreturn 1\n", "int"),
        ("return getattr(1, \"real\")\n", "int"),
        ("p: Point = Point(x=1)\nreturn 1\n", "int"),
    ];
    for model in ["writer", "caller"] {
        for (index, (statements, returns)) in cases.iter().enumerate() {
            let (mut cfg, mut ws, baseline) = fixture(model).await;
            memory_io(&mut cfg, "text");
            let body: String = statements.lines().map(|line| format!("    {line}\n")).collect();
            let function = format!("@effects('io', 'fail', 'mut')\nasync def probe() -> {returns}:\n{body}");
            let declared = execute_native_tool(
                "mutate",
                &json!({"language":"python","module":format!("NtRunAgree{index}"),"source":function}).to_string(),
                &cfg, &mut ws, &baseline,
            ).await;
            let (mut cfg, mut ws, baseline) = fixture(model).await;
            memory_io(&mut cfg, "text");
            let ran = run_source(&cfg, &mut ws, &baseline, statements).await;
            assert_eq!(
                ran.execution.has_errors, declared.execution.has_errors,
                "{model} case {index}:\nrun: {}\nmutate: {}", ran.content, declared.content
            );
            // Not vacuous: the table holds admitted and refused cases for both models.
            let admitted = matches!(index, 0 | 4) || (index == 1 && model == "writer");
            assert_eq!(!ran.execution.has_errors, admitted, "{model} case {index}: {}", ran.content);
        }
    }
}

#[tokio::test]
async fn program_changing_builtins_are_refused_before_any_effect() {
    let (mut cfg, mut ws, baseline) = fixture("writer").await;
    let log = memory_io(&mut cfg, "");
    let before = snapshot(&cfg, &ws);
    let builtins = crate::builtins::program_effects::program_changing_builtins();
    assert!(builtins.contains(&"mutate") && builtins.contains(&"library_reload"));
    for builtin in builtins {
        for source in [
            format!("written: str = await file_write(\"fixture://first\", \"x\")\ndone: str = await {builtin}(\"NtRunLib\")\nreturn done\n"),
            format!("ok: bool = True\nif ok:\n    done: str = await {builtin}(\"NtRunLib\")\nreturn 1\n"),
        ] {
            let arguments = json!({"source": source}).to_string();
            for refusal in [
                validate_native_tool("run", &arguments).unwrap_err(),
                validate_native_tool_authorized("run", &arguments, &cfg).unwrap_err(),
                execute_native_tool("run", &arguments, &cfg, &mut ws, &baseline).await.content,
            ] {
                assert!(refusal.contains(&format!("run cannot call `{builtin}`")), "{builtin}: {refusal}");
                assert!(refusal.contains("Use the mutate tool"), "{builtin}: {refusal}");
            }
        }
    }
    assert!(log.lock().unwrap().is_empty(), "admission refusal precedes every effect");
    assert_nothing_persisted(&cfg, &ws, &baseline, &before).await;
}

#[tokio::test]
async fn shared_state_changes_through_called_functions_persist_as_under_evaluate() {
    let (mut cfg, mut ws, baseline) = fixture("writer").await;
    memory_io(&mut cfg, "");
    let counter = |cfg: &AppConfig| cfg.runtime.read().unwrap().shared_vars.get("NtRunLib.counter").map(|v| v.to_string());
    let first = run_source(&cfg, &mut ws, &baseline, "return NtRunLib.bump()\n").await;
    assert_eq!(output(&first), "1", "{}", first.content);
    // The turn loop writes the working set back after every call; the change must survive.
    cfg.write_back_working_set(&ws).await;
    assert_eq!(counter(&cfg).as_deref(), Some("1"));
    let mut ws = cfg.snapshot_working_set().await;
    let second = run_source(&cfg, &mut ws, &baseline, "a: int = NtRunLib.bump()\nb: int = NtRunLib.bump()\nreturn a + b\n").await;
    assert_eq!(output(&second), "5", "{}", second.content);
    cfg.write_back_working_set(&ws).await;
    assert_eq!(counter(&cfg).as_deref(), Some("3"));
    // evaluate sees and continues the same store (its async path installs the live runtime;
    // the fixture has no HTTP handler that would install it for a synchronous evaluation).
    let mut ws = cfg.snapshot_working_set().await;
    let evaluated = execute_native_tool("evaluate", &json!({"function":"NtRunLib.bump_io"}).to_string(), &cfg, &mut ws, &baseline).await;
    assert!(evaluated.content.contains("= 4"), "{}", evaluated.content);
    cfg.write_back_working_set(&ws).await;
    // With `module`, the block is that module's code: it reads and writes the shared
    // variable by name, which is why the scope module needs Write.
    let mut ws = cfg.snapshot_working_set().await;
    let scoped = run(&cfg, &mut ws, &baseline, json!({"module":"NtRunLib","source":"counter = counter + 10\nreturn bump()\n"})).await;
    assert_eq!(output(&scoped), "15", "{}", scoped.content);
    cfg.write_back_working_set(&ws).await;
    assert_eq!(counter(&cfg).as_deref(), Some("15"));
    assert!(ws.program == baseline, "shared values are runtime state, not Program declarations");
}

/// The block's requests reach the same IO loop as every other evaluation, so the shell
/// policy decides there. `deny` on a harmless command: nothing runs even if the guard
/// were missing.
#[tokio::test]
async fn io_boundary_guards_apply_to_a_block_unchanged() {
    let (mut cfg, mut ws, baseline) = fixture("writer").await;
    let (sender, mut receiver) = tokio::sync::mpsc::channel(8);
    cfg.io_sender = Some(sender);
    let (mut runtime, _spawned) = crate::coroutine::Runtime::new();
    runtime.shell_policy = crate::coroutine::ShellPolicy::Denied;
    let serve = async { while let Some(request) = receiver.recv().await { runtime.handle_io(request).await; } };
    tokio::pin!(serve);
    for (source, failed, fragment) in [
        ("out: str = await shell_exec(\"echo run-shell-probe\")\nreturn out\n", true, "shell execution is disabled by policy"),
        ("r: Result[str] = await http_request(\"TRACE\", \"http://127.0.0.1:9/\", \"\", \"\")\nreturn r\n", false, "unsupported method"),
        ("r: Result[str] = await http_request(\"GET\", \"file:///etc/hostname\", \"\", \"\")\nreturn r\n", false, "Err("),
    ] {
        let result = tokio::select! {
            result = run_source(&cfg, &mut ws, &baseline, source) => result,
            _ = &mut serve => unreachable!(),
        };
        assert_eq!(result.execution.has_errors, failed, "{source}: {}", result.content);
        assert!(output(&result).contains(fragment), "{source}: {}", result.content);
        assert!(!output(&result).starts_with("run-shell-probe"), "the command must not have run: {}", result.content);
    }
}

/// A turn-bound builtin reads the identity from the handle the block runs with. A guest
/// identity is refused at the scoped IO boundary exactly as for any other evaluation; the
/// tool gate above is not the only line of defence.
#[tokio::test]
async fn the_scoped_io_boundary_still_decides_for_the_turn_identity() {
    let (mut cfg, _ws, _baseline) = fixture("writer").await;
    let log = memory_io(&mut cfg, "");
    let sender = cfg.io_sender.clone().unwrap();
    let program = cfg.program.read().await.clone();
    let lowered = crate::python_frontend::lower_block_with_program("seen: str = await memory_inspect()\nreturn seen\n", SCOPE, "run_block", &program).unwrap();
    let [crate::parser::Operation::Module(module)] = lowered.operations.as_slice() else { panic!() };
    let [crate::parser::Operation::Function(function)] = module.body.as_slice() else { panic!() };
    let declaration = crate::validator::detached_module_function(&program, SCOPE, function).unwrap();
    for (guest_tools, allowed) in [(None, true), (Some(vec![]), false)] {
        let handle = crate::coroutine::CoroutineHandle::new(sender.clone()).with_turn(Some(TurnIdentity {
            context: "fixture-run-context".into(),
            principal: "fixture-run-principal".into(),
            may_write: guest_tools.is_none(),
            guest_tools,
        }));
        let (program, body) = (program.clone(), declaration.body.clone());
        let outcome = tokio::task::spawn_blocking(move || {
            crate::eval::eval_detached_block(&program, &format!("{SCOPE}.run_block"), &body, handle, crate::eval::meter::RunContext::owner())
        })
        .await
        .unwrap();
        assert_eq!(outcome.is_ok(), allowed, "{outcome:?}");
    }
    assert_eq!(log.lock().unwrap().len(), 1, "the guest request never crossed the boundary");
}

#[tokio::test]
async fn the_scratch_scope_cannot_be_a_real_module() {
    let (mut cfg, mut ws, baseline) = fixture("writer").await;
    memory_io(&mut cfg, "");
    // No declaration path accepts the name.
    for arguments in [
        json!({"language":"python","module":SCOPE,"source":"def f() -> int:\n    return 1\n"}),
        json!({"source":format!("+module {SCOPE}\n+fn f ()->Int\n  +return 1\n+end\n")}),
    ] {
        let declared = execute_native_tool("mutate", &arguments.to_string(), &cfg, &mut ws, &baseline).await;
        assert!(declared.execution.has_errors && declared.content.contains("reserved for the run tool"), "{}", declared.content);
    }
    let mut program = Program::default();
    for operation in crate::parser::parse(&format!("+module {SCOPE}\n+fn f ()->Int\n  +return 1\n+end\n")).unwrap() {
        let error = crate::validator::apply_and_validate(&mut program, &operation).unwrap_err();
        assert!(error.to_string().contains("reserved for the run tool"), "{error}");
    }
    assert!(program.modules.is_empty());
    assert!(ws.program.modules.iter().all(|m| m.name != SCOPE));

    // A session file edited by hand could still carry one. Then the unscoped block is
    // refused rather than resolving that module's names as its own.
    let mut forged = baseline.clone();
    let mut impostor = forged.modules.iter().find(|m| m.name == "NtRunSecret").unwrap().clone();
    impostor.name = SCOPE.into();
    impostor.id = SCOPE.into();
    forged.modules.push(impostor);
    forged.rebuild_function_index();
    let mut cfg = config_for("writer", AccessLevel::Full, permissions(), forged.clone());
    memory_io(&mut cfg, "");
    let mut ws = cfg.snapshot_working_set().await;
    for source in ["return token()\n", "return 1\n"] {
        let result = run_source(&cfg, &mut ws, &forged, source).await;
        assert!(result.execution.has_errors, "{}", result.content);
        assert!(!result.content.contains("hidden-token"), "{}", result.content);
    }
    let named = run(&cfg, &mut ws, &forged, json!({"module":"NtRunLib","source":"return double(2)\n"})).await;
    assert_eq!(output(&named), "4", "{}", named.content);
}

#[tokio::test]
async fn the_block_name_never_shadows_a_real_function_of_its_scope() {
    let mut program = build_program("+module NtRunNamed\n+fn run_block ()->Int\n  +return 99\n+end\n+fn run_block_1 ()->Int\n  +return 98\n+end\n").await;
    program.require_modules = true;
    let mut cfg = config_for("writer", AccessLevel::Full, PermissionConfig::default(), program.clone());
    memory_io(&mut cfg, "");
    let mut ws = cfg.snapshot_working_set().await;
    let before = snapshot(&cfg, &ws);
    let result = run(&cfg, &mut ws, &program, json!({"module":"NtRunNamed","source":"return run_block() + run_block_1()\n"})).await;
    assert_eq!(output(&result), "197", "{}", result.content);
    assert_nothing_persisted(&cfg, &ws, &program, &before).await;
    let live = cfg.program.read().await;
    let module = live.modules.iter().find(|m| m.name == "NtRunNamed").unwrap();
    assert_eq!(module.functions.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(), ["run_block", "run_block_1"]);
}

/// The validator's own rules (the reserved `__` prefix of #124 among them) are reached
/// through the detached entry, on a declaration the Python frontend would never emit.
#[test]
fn the_detached_validator_entry_applies_the_reserved_name_rule_without_inserting() {
    let program = Program::default();
    let operations = crate::parser::parse("+module NtRunDetached\n+fn probe ()->Int [io,async]\n  +let stolen:Int = __coroutine_handle\n  +return 1\n+end\n").unwrap();
    let [crate::parser::Operation::Module(module)] = operations.as_slice() else { panic!() };
    let [crate::parser::Operation::Function(function)] = module.body.as_slice() else { panic!() };
    let error = crate::validator::detached_module_function(&program, "NtRunDetached", function).unwrap_err();
    assert!(error.to_string().contains("is reserved"), "{error}");
    let operations = crate::parser::parse("+module NtRunDetached\n+fn probe ()->Int\n  +await text:String = file_read(\"/x\")\n  +return 1\n+end\n").unwrap();
    let [crate::parser::Operation::Module(module)] = operations.as_slice() else { panic!() };
    let [crate::parser::Operation::Function(function)] = module.body.as_slice() else { panic!() };
    let error = crate::validator::detached_module_function(&program, "NtRunDetached", function).unwrap_err();
    assert!(error.to_string().contains("does not declare [async] effect"), "{error}");
    assert!(program.modules.is_empty() && program.functions.is_empty());
}
