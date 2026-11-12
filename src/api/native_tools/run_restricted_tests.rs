//! Restricted `run` at the tool boundary (DESIGN L3, #129): what guests and execute-only
//! speakers may put in a block, and the hostile-author cases. Every case must end in a
//! clean refusal, leave no partial state, and leave the next call working.
use super::run_tests::*;
use super::*;
use serde_json::json;

const GUEST_TOOLS: [&str; 3] = ["music_generate", "web_search", "memory_inspect"];

async fn viewer() -> (AppConfig, WorkingSet, Program, IoLog) {
    let (mut cfg, _ws, baseline) = fixture("viewer").await;
    cfg.turn = Some(TurnIdentity {
        context: "telegram:group:-7".into(),
        principal: "telegram:user:100000778".into(),
        may_write: false,
        guest_tools: None,
    });
    let log = memory_io(&mut cfg, "file text");
    let ws = cfg.snapshot_working_set().await;
    assert_eq!(speaker_class(&cfg), SpeakerClass::ExecuteOnly);
    (cfg, ws, baseline, log)
}

async fn guest() -> (AppConfig, WorkingSet, Program, IoLog) {
    let (mut cfg, _ws, baseline) = fixture("visitor").await;
    cfg.turn = Some(TurnIdentity {
        context: "telegram:user:100000777".into(),
        principal: "telegram:user:100000777".into(),
        may_write: false,
        guest_tools: Some(GUEST_TOOLS.iter().map(|tool| tool.to_string()).collect()),
    });
    let log = memory_io(&mut cfg, "file text");
    let ws = cfg.snapshot_working_set().await;
    assert_eq!(speaker_class(&cfg), SpeakerClass::Guest(GUEST_TOOLS.iter().map(|tool| tool.to_string()).collect()));
    (cfg, ws, baseline, log)
}

#[tokio::test]
async fn the_restricted_run_schema_offers_no_module_and_states_the_rule() {
    for (cfg, callable) in [(viewer().await.0, "functions of modules you may execute"), (guest().await.0, "music_generate, web_search, memory_inspect")] {
        let run = tool_definitions(&cfg).into_iter().find(|tool| tool.name == "run").unwrap();
        assert_eq!(run.parameters["properties"].as_object().unwrap().keys().collect::<Vec<_>>(), ["source"]);
        assert_eq!(run.parameters["additionalProperties"], false);
        assert!(run.description.contains(callable), "{}", run.description);
        assert!(run.description.contains("8 IO/tool calls, 60 seconds, values up to 1 MiB"), "{}", run.description);
    }
    let (cfg, ..) = guest().await;
    let names: Vec<String> = tool_definitions(&cfg).into_iter().map(|tool| tool.name).collect();
    assert_eq!(names, ["music_generate", "web_search", "memory_inspect", "run", "finish"]);
    for tool in tool_definitions(&cfg) {
        assert_eq!(tool.parameters["additionalProperties"], false, "{}", tool.name);
    }
    assert_eq!(guest_tools::names(), ["music_generate", "web_search", "web_read", "context_propose", "cad_prepare", "video_stabilize"]);
    for tool in crate::coroutine::scoped::GUEST_TOOLS {
        assert!(guest_tools::definition(tool).is_some() || speaker::is_memory_tool(tool), "guest tool `{tool}` has no native schema");
    }
}

#[tokio::test]
async fn an_execute_only_block_composes_pure_builtins_and_executable_functions() {
    let (cfg, mut ws, baseline, log) = viewer().await;
    let before = snapshot(&cfg, &ws);
    let source = "a: int = NtRunLib.double(4)\nparts: list[str] = split(\"x,y,z\", \",\")\ntext: str = await NtRunLib.read(\"fixture://via-callee\")\nreturn concat(to_string(a + len(parts)), \":\", text)\n";
    let result = run_source(&cfg, &mut ws, &baseline, source).await;
    assert_eq!(output(&result), "11:file text", "{}", result.content);
    // The callee is owner-written code: its own IO runs as under evaluate.
    assert_eq!(*log.lock().unwrap(), ["file_read fixture://via-callee"]);
    // A callee may use a builtin the block itself may not.
    let home = run_source(&cfg, &mut ws, &baseline, "return len(NtRunLib.home()) > 0\n").await;
    assert_eq!(output(&home), "true", "{}", home.content);
    assert_nothing_persisted(&cfg, &ws, &baseline, &before).await;
}

#[tokio::test]
async fn a_guest_block_composes_pure_builtins_and_its_own_tools() {
    let (cfg, mut ws, baseline, log) = guest().await;
    let before = snapshot(&cfg, &ws);
    let source = "first: str = await web_search(\"one\")\nsecond: str = await web_search(\"two\")\nreturn concat(first, \" + \", second)\n";
    let result = run_source(&cfg, &mut ws, &baseline, source).await;
    assert_eq!(output(&result), "scoped result + scoped result", "{}", result.content);
    assert_eq!(*log.lock().unwrap(), ["scoped Search { query: \"one\" }", "scoped Search { query: \"two\" }"]);
    assert_nothing_persisted(&cfg, &ws, &baseline, &before).await;
}

/// The rule, case by case. Each is refused at admission (so the batch preflight catches
/// it too) and nothing crosses the IO boundary.
#[tokio::test]
async fn a_block_may_call_only_what_its_speaker_can_already_call() {
    let cases: Vec<(&str, serde_json::Value, &str)> = vec![
        ("both", json!({"source": "return home_dir()\n"}), "`home_dir` is not available in this conversation"),
        ("both", json!({"source": "t: str = await file_read(\"/etc/passwd\")\nreturn t\n"}), "`file_read` is not available"),
        ("both", json!({"source": "t: str = await read_file(\"/etc/passwd\")\nreturn t\n"}), "`read_file` is not available"),
        ("both", json!({"source": "w: str = await file_write(\"/tmp/x\", \"y\")\nreturn w\n"}), "`file_write` is not available"),
        ("both", json!({"source": "o: str = await shell_exec(\"id\")\nreturn o\n"}), "`shell_exec` is not available"),
        ("both", json!({"source": "r: Result[str] = await http_get(\"http://127.0.0.1:3002/api/status\")\nreturn r\n"}), "`http_get` is not available"),
        ("both", json!({"source": "r: Result[str] = await http_request(\"GET\", \"http://10.0.0.4:3002/\", \"\", \"\")\nreturn r\n"}), "`http_request` is not available"),
        ("both", json!({"source": "m: str = await llm_set_model(\"other\")\nreturn m\n"}), "`llm_set_model` is not available"),
        ("both", json!({"source": "m: str = await llm_call(\"sys\", \"hi\")\nreturn m\n"}), "`llm_call` is not available"),
        ("both", json!({"source": "m: str = await conversation_notify(\"telegram:1\", \"hi\")\nreturn m\n"}), "`conversation_notify` is not available"),
        ("both", json!({"source": "m: str = await conversation_history(\"telegram:1\")\nreturn m\n"}), "`conversation_history` is not available"),
        ("both", json!({"source": "m: str = await memory_cypher(\"MATCH (n) RETURN n\")\nreturn m\n"}), "`memory_cypher` is not available"),
        ("both", json!({"source": "m: str = await query_symbols()\nreturn m\n"}), "`query_symbols` is not available"),
        ("both", json!({"source": "m: str = await runtime_errors(0)\nreturn m\n"}), "`runtime_errors` is not available"),
        ("both", json!({"source": "m: str = await mutate(\"+module X\")\nreturn m\n"}), "run cannot call `mutate`"),
        ("both", json!({"source": "spawn(NtRunLib.double(1))\nreturn 1\n"}), "`spawn` is not available"),
        ("both", json!({"source": "t: str = spawn(NtRunLib.double(1))\nreturn 1\n"}), "`spawn` is not available"),
        ("both", json!({"source": "source_add(\"timer(1000)\", \"tick\", \"NtRunLib.double\")\nreturn 1\n"}), "`source_add` is not available"),
        ("both", json!({"source": "event_emit(\"e\", 1)\nreturn 1\n"}), "`event_emit` is not available"),
        ("both", json!({"source": "return counter\n"}), "name `counter` is not bound in this block"),
        ("both", json!({"source": "return double(1)\n"}), "`double(...)` is not a builtin"),
        ("both", json!({"source": "return double(1)\n", "module": "NtRunLib"}), "a `module` scope is not available"),
        ("both", json!({"source": "counter = counter + 1\nreturn 1\n", "module": "NtRunLib"}), "a `module` scope is not available"),
        ("both", json!({"source": "return NtRunSecret.token()\n"}), "NtRunSecret.token"),
        ("guest", json!({"source": "return NtRunLib.double(1)\n"}), "function `NtRunLib.double` is not available"),
        ("guest", json!({"source": "p: str = await web_read(\"http://example.org\")\nreturn p\n"}), "`web_read` is not available"),
        ("guest", json!({"source": "p: str = await cad_prepare(\"t\", \"brief\")\nreturn p\n"}), "`cad_prepare` is not available"),
        ("viewer", json!({"source": "p: str = await web_search(\"x\")\nreturn p\n"}), "`web_search` is not available"),
        ("viewer", json!({"source": "p: str = await memory_inspect()\nreturn p\n"}), "`memory_inspect` is not available"),
    ];
    for class in ["viewer", "guest"] {
        let (cfg, mut ws, baseline, log) = if class == "viewer" { viewer().await } else { guest().await };
        let before = snapshot(&cfg, &ws);
        for (who, arguments, fragment) in cases.iter().filter(|(who, ..)| *who == "both" || *who == class) {
            let text = arguments.to_string();
            let preflight = validate_native_tool_authorized("run", &text, &cfg).expect_err(&format!("{class}/{who}: {text}"));
            assert!(preflight.contains(fragment), "{class}: {text}: {preflight}");
            let result = run(&cfg, &mut ws, &baseline, arguments.clone()).await;
            assert!(result.execution.has_errors && result.content.contains(fragment), "{class}: {text}: {}", result.content);
            assert!(!result.content.contains("hidden-token") && !result.content.contains(SCOPE), "{}", result.content);
        }
        assert!(log.lock().unwrap().is_empty(), "{class}: {:?}", log.lock().unwrap());
        assert_nothing_persisted(&cfg, &ws, &baseline, &before).await;
        // The next call works.
        let next = run_source(&cfg, &mut ws, &baseline, "return to_string(abs(0 - 3))\n").await;
        assert_eq!(output(&next), "3", "{class}: {}", next.content);
    }
}

/// The memory, time and call-count attacks, through the real tool path. The author of a
/// restricted block is untrusted; each attack ends in a `run stopped` refusal.
#[tokio::test]
async fn hostile_blocks_are_stopped_by_the_interpreter() {
    let doubling = |init: &str, grow: &str| format!("{init}\ni: int = 0\nwhile i < 200:\n    {grow}\n    i = i + 1\nreturn 1\n");
    let reads: String = (0..9).map(|n| format!("t{n}: str = await NtRunLib.read(\"fixture://{n}\")\n")).collect();
    let cases: Vec<(&str, String, &str)> = vec![
        ("string doubling", doubling("s: str = \"0123456789abcdef\"", "s = concat(s, s)"), "a value exceeds 1 MiB"),
        ("string doubling by +", doubling("s: str = \"0123456789abcdef\"", "s = s + s"), "a value exceeds 1 MiB"),
        ("list doubling", "xs: list[int] = [1, 2, 3, 4]\ni: int = 0\nwhile i < 200:\n    xs = push(xs, xs)\n    i = i + 1\nreturn 1\n".into(), "run stopped: a value"),
        ("nested map growth", "m: Map[str, int] = map(\"k\", 1)\ni: int = 0\nwhile i < 200:\n    m = map_set(m, to_string(i), m)\n    i = i + 1\nreturn 1\n".into(), "run stopped: a value"),
        ("many large values", "keep: str = \"\"\ni: int = 0\nwhile i < 500:\n    big: str = NtRunLib.half_megabyte()\n    keep = concat(big, to_string(i))\n    i = i + 1\nreturn 1\n".into(), "it produced more than 32 MiB of values in total"),
        ("nested loops", "xs: list[int] = []\ni: int = 0\nwhile i < 500:\n    xs = push(xs, i)\n    i = i + 1\nn: int = 0\nfor a in xs:\n    for b in xs:\n        n = n + 1\nreturn n\n".into(), "it executed more than 200000 steps"),
        ("burning time in pure code", "i: int = 0\nj: int = 0\nwhile i < 10000:\n    j = 0\n    while j < 10000:\n        j = j + 1\n    i = i + 1\nreturn i\n".into(), "it executed more than 200000 steps"),
        ("recursion through a callee", "return NtRunLib.recurse(1)\n".into(), "maximum call depth"),
        ("a ninth IO call", format!("{reads}return t8\n"), "`file_read` would be IO call 9 of this block"),
        ("rendering a huge value", "s: str = NtRunLib.half_megabyte()\nxs: list[str] = [s, s, s]\nreturn to_string(xs)\n".into(), "a value exceeds 1 MiB"),
    ];
    let (cfg, mut ws, baseline, log) = viewer().await;
    let before = snapshot(&cfg, &ws);
    for (label, source, fragment) in &cases {
        let started = std::time::Instant::now();
        let result = run_source(&cfg, &mut ws, &baseline, source).await;
        assert!(result.execution.has_errors, "{label}: {}", &result.content[..result.content.len().min(400)]);
        let text = output(&result);
        assert!(text.contains(fragment), "{label}: {}", &text[..text.len().min(400)]);
        assert!(text.len() < 2_000, "{label}: the refusal must not carry the value");
        assert!(started.elapsed() < std::time::Duration::from_secs(20), "{label}: {:?}", started.elapsed());
        let next = run_source(&cfg, &mut ws, &baseline, "return NtRunLib.double(2)\n").await;
        assert_eq!(output(&next), "4", "{label}: the next call must work: {}", next.content);
    }
    // IO done before a limit stopped the block stays done and is not repeated.
    assert_eq!(log.lock().unwrap().len(), 8, "{:?}", log.lock().unwrap());
    assert_nothing_persisted(&cfg, &ws, &baseline, &before).await;
}

#[tokio::test]
async fn a_guest_block_cannot_multiply_a_costly_tool() {
    let (cfg, mut ws, baseline, log) = guest().await;
    let twice = "a: str = await music_generate(\"calm\", 30)\nb: str = await music_generate(\"loud\", 30)\nreturn b\n";
    let result = run_source(&cfg, &mut ws, &baseline, twice).await;
    assert!(result.execution.has_errors, "{}", result.content);
    assert!(output(&result).contains("`music_generate` may be called 1 time(s) per block"), "{}", result.content);
    let in_loop = "i: int = 0\nwhile i < 5:\n    s: str = await music_generate(\"again\", 30)\n    i = i + 1\nreturn i\n";
    let result = run_source(&cfg, &mut ws, &baseline, in_loop).await;
    assert!(output(&result).contains("may be called 1 time(s) per block"), "{}", result.content);
    let jobs = log.lock().unwrap().iter().filter(|entry| entry.contains("Music")).count();
    assert_eq!(jobs, 2, "one job per block reached the boundary: {:?}", log.lock().unwrap());
    // The existing guest rule at the IO boundary still decides scope for memory tools.
    let (mut cfg, mut ws, baseline, _log) = guest().await;
    cfg.turn.as_mut().unwrap().guest_tools = Some(vec!["memory_remember".into()]);
    Arc::make_mut(&mut cfg.permission_config).guest_profiles.insert("visitor".into(), vec!["memory_remember".into()]);
    let global = run_source(&cfg, &mut ws, &baseline, "m: str = await memory_remember(\"note\", \"global\")\nreturn m\n").await;
    assert!(global.execution.has_errors && global.content.contains("guest memories must use context scope"), "{}", global.content);
    let direct = execute_native_tool("memory_remember", r#"{"content":"note","scope":"global"}"#, &cfg, &mut ws, &baseline).await;
    assert!(direct.execution.has_errors && direct.content.contains("guest memories must use context scope"), "{}", direct.content);
}

/// The single-value cap also protects the owner: the doubling loop that used to take the
/// process down ends in a refusal, and the owner keeps everything else (no step, IO or
/// time limit, any builtin, `module` scope).
#[tokio::test]
async fn the_owner_keeps_an_unrestricted_block_with_a_value_cap() {
    // Allocates up to the 64 MiB cap: isolated in a child process (see `meter_tests.rs`).
    if !crate::logging::tests::child() {
        crate::logging::tests::capture(
            "api::native_tools::tests::run_restricted_tests::the_owner_keeps_an_unrestricted_block_with_a_value_cap",
            false,
        );
        return;
    }
    let (mut cfg, mut ws, baseline) = fixture("writer").await;
    let log = memory_io(&mut cfg, "x");
    let doubling = "s: str = \"0123456789abcdef\"\ni: int = 0\nwhile i < 200:\n    s = concat(s, s)\n    i = i + 1\nreturn len(s)\n";
    let result = run_source(&cfg, &mut ws, &baseline, doubling).await;
    assert!(result.execution.has_errors, "{}", result.content);
    assert_eq!(output(&result), "run error: run stopped: a value exceeds 64 MiB (limit for a single value in this conversation)");
    let reads: String = (0..12).map(|n| format!("t{n}: str = await file_read(\"fixture://{n}\")\n")).collect();
    let many = run_source(&cfg, &mut ws, &baseline, &format!("{reads}h: str = home_dir()\nreturn len(h) > 0\n")).await;
    assert_eq!(output(&many), "true", "{}", many.content);
    assert_eq!(log.lock().unwrap().len(), 12);
    let steps = "i: int = 0\nj: int = 0\nwhile i < 60:\n    j = 0\n    while j < 5000:\n        j = j + 1\n    i = i + 1\nreturn i\n";
    assert_eq!(output(&run_source(&cfg, &mut ws, &baseline, steps).await), "60");
}

/// Class selection from the pinned snapshot (decision 4): every combination of the two
/// guest signals and the write/execute authority. Anything not certain is a guest with
/// an empty tool list.
#[tokio::test]
async fn the_speaker_class_is_derived_from_the_pinned_snapshot_and_fails_closed() {
    let identity = |may_write: bool, tools: Option<Vec<&str>>| {
        Some(TurnIdentity {
            context: "c".into(),
            principal: "p".into(),
            may_write,
            guest_tools: tools.map(|tools| tools.into_iter().map(str::to_string).collect()),
        })
    };
    let guest = |tools: &[&str]| SpeakerClass::Guest(tools.iter().map(|tool| tool.to_string()).collect());
    for (model, turn, expected) in [
        ("writer", None, SpeakerClass::Writer),
        ("writer", identity(true, None), SpeakerClass::Writer),
        ("writer", identity(false, None), SpeakerClass::ExecuteOnly),
        ("viewer", None, SpeakerClass::ExecuteOnly),
        ("viewer", identity(true, None), SpeakerClass::ExecuteOnly),
        ("executor", identity(false, None), SpeakerClass::ExecuteOnly),
        // Both guest signals agree: the intersection, in registry order, known tools only.
        ("visitor", identity(false, Some(vec!["web_search", "music_generate", "memory_inspect"])), guest(&["music_generate", "web_search", "memory_inspect"])),
        ("visitor", identity(false, Some(vec!["web_search", "shell_exec", "web_read"])), guest(&["web_search"])),
        ("visitor", identity(true, Some(vec![])), guest(&[])),
        // The signals disagree: a profile without a turn list, or a turn list without a profile.
        ("visitor", None, guest(&[])),
        ("visitor", identity(false, None), guest(&[])),
        ("writer", identity(true, Some(vec!["web_search"])), guest(&[])),
        ("viewer", identity(false, Some(vec!["web_search"])), guest(&[])),
        // No authority at all in any group.
        ("nothing", None, guest(&[])),
    ] {
        let (mut cfg, ..) = fixture(model).await;
        if model == "nothing" {
            let permissions = Arc::make_mut(&mut cfg.permission_config);
            let mut none = permissions.model["visitor"].clone();
            none.group_perms.values_mut().for_each(|level| *level = PermissionLevel::None);
            permissions.model.insert("nothing".into(), none);
        }
        cfg.turn = turn.clone();
        let class = speaker_class(&cfg);
        assert_eq!(class, expected, "{model} {turn:?}");
        assert_eq!(native_tools_eligible(&cfg), expected == SpeakerClass::Writer);
        let names: Vec<String> = tool_definitions(&cfg).into_iter().map(|tool| tool.name).collect();
        if class.restricted() {
            assert!(!names.iter().any(|name| name == "mutate" || name == "test"), "{model}: {names:?}");
        }
        if let SpeakerClass::Guest(tools) = &class {
            let mut listed: Vec<String> = tools.clone();
            listed.extend(["run".to_string(), "finish".to_string()]);
            assert_eq!(names, listed, "{model}");
        }
    }
}
