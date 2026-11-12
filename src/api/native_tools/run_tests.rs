//! Native `run` (#129): results, the nothing-persisted guarantee, limits and logging.
//! IO is served by in-memory `IoRequest` replies or a real `Runtime` on harmless inputs;
//! no live service is contacted. Authority tests live in `run_authority_tests.rs`.
use super::*;
use serde_json::json;
use std::sync::Mutex;

pub(super) const SCOPE: &str = crate::reserved_names::RUN_SCOPE;

pub(super) const LIBRARY: &str = "\
+module NtRunLib
+shared counter:Int = 0
+fn double (x:Int)->Int
  +return x * 2
+end
+fn bump ()->Int [mut]
  +set counter = counter + 1
  +return counter
+end
+fn bump_io ()->Int [mut,io,async]
  +set counter = counter + 1
  +return counter
+end
+fn nothing (x:Int)->Int
  +if x > 100
    +return x
  +end
+end
+fn big (x:Int)->Int
  +let a:Int = x + 1
  +let b:Int = a + 1
  +return b
+end
+fn read (path:String)->String [io,async]
  +await text:String = file_read(path)
  +return text
+end
+fn positive (x:Int)->Int [fail]
  +check is_positive x > 0 ~not_positive
  +return x
+end
+fn recurse (x:Int)->Int
  +return NtRunLib.recurse(x)
+end
+fn home ()->String
  +return home_dir()
+end
+fn half_megabyte ()->String
  +let s:String = \"0123456789abcdef\"
  +let i:Int = 0
  +while i < 15
    +set s = concat(s, s)
    +set i = i + 1
  +end
  +return s
+end
+module NtRunSecret
+fn token ()->String
  +return \"hidden-token\"
+end
";

pub(super) fn permissions() -> PermissionConfig {
    toml::from_str(
        r#"
        [groups]
        lib = ["NtRunLib"]
        secret = ["NtRunSecret"]

        [model.writer]
        lib = "write"
        secret = "write"
        user = "write"

        [model.libonly]
        lib = "write"
        secret = "none"
        user = "execute"

        [model.caller]
        lib = "execute"
        secret = "none"
        user = "write"

        [model.executor]
        lib = "execute"
        secret = "execute"
        user = "execute"

        [model.viewer]
        lib = "execute"
        secret = "none"
        user = "execute"

        [model.visitor]
        lib = "none"
        secret = "none"
        user = "none"

        [guest_profiles]
        visitor = ["web_search", "music_generate", "memory_inspect"]
        "#,
    )
    .unwrap()
}

pub(super) async fn fixture(model: &str) -> (AppConfig, WorkingSet, Program) {
    let mut program = build_program(LIBRARY).await;
    program.require_modules = true;
    let cfg = config_for(model, AccessLevel::Full, permissions(), program.clone());
    let ws = cfg.snapshot_working_set().await;
    (cfg, ws, program)
}

pub(super) async fn run(
    cfg: &AppConfig,
    ws: &mut WorkingSet,
    baseline: &Program,
    arguments: serde_json::Value,
) -> NativeToolResult {
    tokio::time::timeout(
        std::time::Duration::from_secs(20),
        execute_native_tool("run", &arguments.to_string(), cfg, ws, baseline),
    )
    .await
    .expect("run fixture timed out")
}

pub(super) async fn run_source(
    cfg: &AppConfig,
    ws: &mut WorkingSet,
    baseline: &Program,
    source: &str,
) -> NativeToolResult {
    run(cfg, ws, baseline, json!({"source": source})).await
}

pub(super) fn envelope(result: &NativeToolResult) -> serde_json::Value {
    serde_json::from_str(&result.content).unwrap()
}

pub(super) fn output(result: &NativeToolResult) -> String {
    envelope(result)["output"].as_str().unwrap().to_string()
}

pub(super) type IoLog = Arc<Mutex<Vec<String>>>;

/// In-memory IO: every request is recorded and answered deterministically. `file` is the
/// content of every file. An unexpected request kind is recorded and its reply dropped.
pub(super) fn memory_io(cfg: &mut AppConfig, file: &str) -> IoLog {
    use crate::coroutine::IoRequest::*;
    let (sender, mut receiver) = tokio::sync::mpsc::channel(8);
    cfg.io_sender = Some(sender);
    let log: IoLog = Arc::new(Mutex::new(Vec::new()));
    let seen = log.clone();
    let file = file.to_string();
    tokio::spawn(async move {
        while let Some(request) = receiver.recv().await {
            let record = |entry: String| seen.lock().unwrap().push(entry);
            match request {
                FileRead { path, reply } => {
                    record(format!("file_read {path}"));
                    let _ = if path.starts_with("missing://") {
                        reply.send(Err(anyhow::anyhow!("no such fixture {path}")))
                    } else {
                        reply.send(Ok(file.clone()))
                    };
                }
                FileWrite { path, data, reply } => {
                    record(format!("file_write {path} {data}"));
                    let _ = reply.send(Ok(()));
                }
                HttpGet { url, reply } => {
                    record(format!("http_get {url}"));
                    let _ = reply.send(Ok(format!("body of {url}")));
                }
                MemoryInspect { context_id, principal_id, query, reply } => {
                    record(format!("memory_inspect context={context_id} principal={principal_id} query={query}"));
                    let _ = reply.send(Ok("inspected".into()));
                }
                Scoped { request, reply } => {
                    record(format!("scoped {request:?}"));
                    let _ = reply.send(Ok("scoped result".into()));
                }
                _ => record("UNEXPECTED".into()),
            }
        }
    });
    log
}

fn same_declarations(left: &Program, right: &Program) -> bool {
    left == right
        && left.modules.len() == right.modules.len()
        && left.modules.iter().zip(&right.modules).all(|(a, b)| {
            a.functions.len() == b.functions.len()
                && a.functions.iter().zip(&b.functions).all(|(x, y)| Arc::ptr_eq(x, y))
        })
}

pub(super) struct Snapshot {
    live_meta: serde_json::Value,
    working_meta: serde_json::Value,
}

pub(super) fn snapshot(cfg: &AppConfig, ws: &WorkingSet) -> Snapshot {
    Snapshot {
        live_meta: serde_json::to_value(&*cfg.meta.lock().unwrap()).unwrap(),
        working_meta: serde_json::to_value(&ws.meta).unwrap(),
    }
}

/// Constraint 3 of #129 on one exit path: the Program is the same declarations (pointer
/// identical), before and after the caller's write-back; no mutation, replay or history
/// record exists; no scratch module, function or library file is visible.
pub(super) async fn assert_nothing_persisted(
    cfg: &AppConfig,
    ws: &WorkingSet,
    baseline: &Program,
    before: &Snapshot,
) {
    assert!(same_declarations(&ws.program, baseline), "working Program changed");
    assert_eq!(serde_json::to_value(&ws.meta).unwrap(), before.working_meta, "working meta changed");
    cfg.write_back_working_set(ws).await;
    let live = cfg.program.read().await;
    assert!(same_declarations(&live, baseline), "live Program changed");
    assert!(live.modules.iter().all(|m| m.name != SCOPE));
    assert!(live.get_function(&format!("{SCOPE}.run_block")).is_none());
    assert_eq!(live.get_function("run_block").is_some(), baseline.get_function("run_block").is_some());
    drop(live);
    let meta = cfg.meta.lock().unwrap();
    assert_eq!(serde_json::to_value(&*meta).unwrap(), before.live_meta, "live meta changed");
    assert_eq!(meta.revision, 0);
    assert!(meta.mutations.is_empty() && meta.sources.is_empty() && meta.history.is_empty());
    drop(meta);
    let directory = crate::library::library_dir();
    for extension in ["ax", "py"] {
        assert!(!directory.join(format!("{SCOPE}.{extension}")).exists());
    }
}

#[tokio::test]
async fn run_returns_a_value_and_persists_nothing() {
    let (mut cfg, mut ws, baseline) = fixture("writer").await;
    let log = memory_io(&mut cfg, "");
    let before = snapshot(&cfg, &ws);
    let result = run_source(&cfg, &mut ws, &baseline, "n: int = NtRunLib.double(21)\nreturn n\n").await;
    assert!(!result.execution.has_errors, "{}", result.content);
    let envelope = envelope(&result);
    assert_eq!(envelope["tool"], "run");
    assert_eq!(envelope["success"], true);
    assert!(envelope.get("returned").is_none(), "{envelope}");
    assert_eq!(envelope["output"], "42");
    assert_eq!(envelope["truncated"], false);
    assert_eq!(envelope["omitted_chars"], 0);
    assert_eq!(envelope["done_accepted"], false);
    // The text is carried once in the envelope; the executor result feeds the HTTP reply.
    assert_eq!(result.content.matches("42").count(), 1, "{}", result.content);
    assert_eq!(result.execution.mutation_results.len(), 1);
    assert_eq!(result.execution.mutation_results[0].message, "42");
    assert!(result.execution.mutation_results[0].success && !result.execution.done_accepted);
    assert!(log.lock().unwrap().is_empty(), "a pure block performs no IO");
    assert_nothing_persisted(&cfg, &ws, &baseline, &before).await;
}

#[tokio::test]
async fn run_renders_the_returned_value_as_written() {
    let (mut cfg, mut ws, baseline) = fixture("writer").await;
    memory_io(&mut cfg, "");
    for (source, expected) in [
        ("return \"plain text\\nsecond line\"\n", "plain text\nsecond line"),
        ("return \"\"\n", ""),
        ("return 1 + 2\n", "3"),
        ("flag: bool = 2 > 1\nreturn flag\n", "true"),
        ("xs: list[int] = [1, 2, 3]\nreturn xs\n", "[1, 2, 3]"),
        ("r: Result[int] = Ok(7)\nreturn r\n", "Ok(7)"),
        ("r: Result[int] = Err(\"reason\")\nreturn r\n", "Err(reason)"),
    ] {
        let result = run_source(&cfg, &mut ws, &baseline, source).await;
        assert!(!result.execution.has_errors, "{source}: {}", result.content);
        assert_eq!(output(&result), expected, "{source}");
    }
}

/// The evaluator has one "no value" value, so a block that does not return and a block
/// that returns that value are the same empty success. The envelope has no field claiming
/// to tell them apart.
#[tokio::test]
async fn run_without_a_value_is_an_empty_success() {
    let (mut cfg, mut ws, baseline) = fixture("writer").await;
    memory_io(&mut cfg, "");
    for source in [
        "n: int = NtRunLib.double(2)\n",
        "n: int = 1\nif n > 5:\n    return n\n",
        // The None literal is not in the native subset. The evaluator's None is its own
        // "no value" marker, produced by a function that falls off its end; returning
        // that from a block ends the block with the same "no value".
        "gone: int = NtRunLib.nothing(1)\nreturn gone\n",
        "return NtRunLib.nothing(1)\n",
    ] {
        let result = run_source(&cfg, &mut ws, &baseline, source).await;
        assert!(!result.execution.has_errors, "{source}: {}", result.content);
        let envelope = envelope(&result);
        assert_eq!(envelope["success"], true, "{source}");
        assert_eq!(envelope["output"], "", "{source}");
        assert!(envelope.get("returned").is_none(), "{source}");
        assert_eq!(result.execution.mutation_results[0].message, "run completed with no value", "{source}");
    }
}

#[tokio::test]
async fn run_awaits_io_builtins_and_user_functions_exactly_once() {
    let (mut cfg, mut ws, baseline) = fixture("writer").await;
    let log = memory_io(&mut cfg, "héllo");
    let before = snapshot(&cfg, &ws);
    let source = "text: str = await file_read(\"fixture://direct\")\nn: int = len(text)\nreturn concat(\"bytes: \", to_string(n))\n";
    let result = run_source(&cfg, &mut ws, &baseline, source).await;
    assert!(!result.execution.has_errors, "{}", result.content);
    assert_eq!(output(&result), "bytes: 6");
    let source = "via: str = await NtRunLib.read(\"fixture://callee\")\nwritten: str = await file_write(\"fixture://out\", via)\nreply: Result[str] = await http_get(\"fixture://http\")\nmatch reply:\n    case Ok(body):\n        return body\n    case Err(reason):\n        return reason\n";
    let result = run_source(&cfg, &mut ws, &baseline, source).await;
    assert!(!result.execution.has_errors, "{}", result.content);
    assert_eq!(output(&result), "body of fixture://http");
    assert_eq!(
        *log.lock().unwrap(),
        [
            "file_read fixture://direct",
            "file_read fixture://callee",
            "file_write fixture://out héllo",
            "http_get fixture://http",
        ]
    );
    assert_nothing_persisted(&cfg, &ws, &baseline, &before).await;
}

/// The same statements as a declared function under `evaluate` and as a `run` block issue
/// the same IO requests with the same turn identity: one execution path, one handle.
#[tokio::test]
async fn run_issues_the_same_io_requests_and_turn_identity_as_evaluate() {
    let mut program = build_program(LIBRARY).await;
    program.require_modules = false; // evaluate's test gate is not the subject here
    let mut cfg = config_for("writer", AccessLevel::Full, permissions(), program.clone());
    cfg.turn = Some(TurnIdentity {
        context: "fixture-run-context".into(),
        principal: "fixture-run-principal".into(),
        may_write: true,
        guest_tools: None,
    });
    let log = memory_io(&mut cfg, "file");
    let mut ws = cfg.snapshot_working_set().await;
    let statements = "    text: str = await file_read(\"fixture://parity\")\n    seen: str = await memory_inspect()\n    reply: Result[str] = await http_get(\"fixture://parity-http\")\n    return concat(text, seen)\n";
    let declared = execute_native_tool(
        "mutate",
        &json!({"language":"python","module":"NtRunParity","source":format!("@effects('io')\nasync def body() -> str:\n{statements}")}).to_string(),
        &cfg, &mut ws, &program,
    ).await;
    assert!(!declared.execution.has_errors, "{}", declared.content);
    cfg.write_back_working_set(&ws).await;
    let evaluated = execute_native_tool(
        "evaluate",
        &json!({"language":"python","function":"NtRunParity.body","input":""}).to_string(),
        &cfg, &mut ws, &program,
    ).await;
    assert!(!evaluated.execution.has_errors, "{}", evaluated.content);
    assert!(evaluated.content.contains("fileinspected"), "{}", evaluated.content);
    let from_evaluate: Vec<String> = std::mem::take(&mut *log.lock().unwrap());

    let block: String = statements.lines().map(|line| format!("{}\n", line.trim_start())).collect();
    let result = run_source(&cfg, &mut ws, &program, &block).await;
    assert!(!result.execution.has_errors, "{}", result.content);
    assert_eq!(output(&result), "fileinspected");
    let from_run = log.lock().unwrap().clone();
    assert_eq!(from_run, from_evaluate);
    assert_eq!(
        from_run,
        [
            "file_read fixture://parity",
            "memory_inspect context=fixture-run-context principal=fixture-run-principal query=",
            "http_get fixture://parity-http",
        ]
    );
}

/// The example of #129, end to end through a real IO loop on a temporary file.
#[tokio::test]
async fn run_reads_a_real_file_through_the_runtime_io_loop() {
    let (mut cfg, mut ws, baseline) = fixture("writer").await;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("x.txt");
    std::fs::write(&path, "twelve bytes").unwrap();
    let (sender, mut receiver) = tokio::sync::mpsc::channel(8);
    cfg.io_sender = Some(sender);
    let (runtime, _spawned) = crate::coroutine::Runtime::new();
    let source = format!(
        "text: str = await file_read({path:?})\nn: int = len(text)\nreturn concat(\"bytes: \", to_string(n))\n"
    );
    let before = snapshot(&cfg, &ws);
    let result = tokio::select! {
        result = run_source(&cfg, &mut ws, &baseline, &source) => result,
        _ = async { while let Some(request) = receiver.recv().await { runtime.handle_io(request).await; } } => unreachable!(),
    };
    assert!(!result.execution.has_errors, "{}", result.content);
    assert_eq!(output(&result), "bytes: 12");
    assert_nothing_persisted(&cfg, &ws, &baseline, &before).await;
}

#[tokio::test]
async fn run_reports_operational_and_language_failures_structurally() {
    let (mut cfg, mut ws, baseline) = fixture("writer").await;
    let log = memory_io(&mut cfg, "");
    let before = snapshot(&cfg, &ws);
    for (source, fragment) in [
        // Operational: the IO reply is an error.
        ("text: str = await file_read(\"missing://gone\")\nreturn text\n", "no such fixture"),
        // Operational: a runtime error in a builtin.
        ("xs: list[int] = [1]\nreturn get(xs, 5)\n", "run error"),
        // Language failure in a callee: a failed check is not a value here.
        ("n: int = NtRunLib.positive(0)\nreturn n\n", "not_positive"),
        // Language failure in the block itself.
        ("ok: bool = 1 > 2\ncheck(\"must_hold\", ok, \"did_not_hold\")\nreturn 1\n", "did_not_hold"),
    ] {
        let result = run_source(&cfg, &mut ws, &baseline, source).await;
        assert!(result.execution.has_errors, "{source}: {}", result.content);
        let envelope = envelope(&result);
        assert_eq!(envelope["success"], false, "{source}");
        assert_eq!(envelope["has_errors"], true, "{source}");
        assert!(output(&result).contains(fragment), "{source}: {}", result.content);
        assert!(!result.execution.mutation_results[0].success);
    }
    assert_eq!(*log.lock().unwrap(), ["file_read missing://gone"]);
    assert_nothing_persisted(&cfg, &ws, &baseline, &before).await;
}

#[tokio::test]
async fn run_refusals_change_nothing_and_perform_no_io() {
    let (mut cfg, mut ws, baseline) = fixture("writer").await;
    let log = memory_io(&mut cfg, "");
    let before = snapshot(&cfg, &ws);
    for (arguments, fragment) in [
        (json!({"source": ""}), "run `source` is empty"),
        (json!({"source": "   \n"}), "run `source` is empty"),
        (json!({}), "missing required field `source`"),
        (json!({"source": "return 1\n", "extra": 1}), "unexpected field `extra`"),
        (json!({"source": "return 1\n", "language": "adapsis"}), "Python only"),
        (json!({"source": "return 1\n", "module": "Nt.Run"}), "non-dunder identifier"),
        (json!({"source": "return 1\n", "module": "__x"}), "non-dunder identifier"),
        (json!({"source": "return 1\n", "module": SCOPE}), "reserved"),
        (json!({"source": "def f() -> int:\n    return 1\n"}), "mutate tool"),
        (json!({"source": "import os\n"}), "import is unsupported"),
        (json!({"source": "return eval(\"1\")\n"}), "run:1:"),
        (json!({"source": "text = await file_read(\"/x\")\nreturn text\n"}), "`text: str = await file_read(path)`"),
        (json!({"source": "await file_write(\"/x\", \"y\")\n"}), "`written: str = await file_write(path, text)`"),
        (json!({"source": "n = 1\nreturn n\n"}), "first assignment requires a type annotation"),
        (json!({"source": "return helper(1)\n"}), "Call user functions as Module.function"),
        (json!({"source": "return counter\n"}), "is not bound in this block"),
        (json!({"source": "return NtRunLib.missing(1)\n"}), "no authorized supported signature"),
        (json!({"source": "return NtRunLib.double(1, 2)\n"}), "do not match declared signature"),
        (json!({"source": "h: str = __coroutine_handle\nreturn h\n"}), "run:1:"),
    ] {
        let result = run(&cfg, &mut ws, &baseline, arguments.clone()).await;
        assert!(result.execution.has_errors, "{arguments}: {}", result.content);
        assert!(result.content.contains(fragment), "{arguments}: {}", result.content);
        assert!(!result.content.contains(SCOPE) || fragment == "reserved", "scope name leaked: {}", result.content);
    }
    assert!(log.lock().unwrap().is_empty(), "a refused block performs no IO");
    assert_nothing_persisted(&cfg, &ws, &baseline, &before).await;
}

#[tokio::test]
async fn the_block_is_not_visible_to_any_later_call() {
    let (mut cfg, mut ws, baseline) = fixture("writer").await;
    memory_io(&mut cfg, "");
    let first = run_source(&cfg, &mut ws, &baseline, "kept: int = 5\nreturn kept\n").await;
    assert_eq!(output(&first), "5");
    cfg.write_back_working_set(&ws).await;
    // No binding survives into the next block.
    let second = run_source(&cfg, &mut ws, &baseline, "return kept\n").await;
    assert!(second.execution.has_errors && second.content.contains("is not bound"), "{}", second.content);
    // No function or module became callable or inspectable.
    for target in [format!("{SCOPE}.run_block"), "NtRunLib.run_block".to_string()] {
        let evaluated = execute_native_tool("evaluate", &json!({"function": target}).to_string(), &cfg, &mut ws, &baseline).await;
        assert!(evaluated.execution.has_errors && evaluated.content.contains("was not found"), "{}", evaluated.content);
        let called = run_source(&cfg, &mut ws, &baseline, &format!("return {target}()\n")).await;
        assert!(called.execution.has_errors, "{}", called.content);
    }
    for arguments in [json!({"kind":"symbols"}), json!({"kind":"library"}), json!({"kind":"tasks"})] {
        let inspected = execute_native_tool("inspect", &arguments.to_string(), &cfg, &mut ws, &baseline).await;
        assert!(!inspected.execution.has_errors, "{}", inspected.content);
        assert!(!inspected.content.contains(SCOPE) && !inspected.content.contains("run_block"), "{}", inspected.content);
    }
    let scoped = execute_native_tool("inspect", &json!({"kind":"symbols","target":SCOPE}).to_string(), &cfg, &mut ws, &baseline).await;
    assert!(scoped.execution.has_errors, "{}", scoped.content);
}

#[tokio::test]
async fn run_has_no_test_gate_and_finish_rules_are_unchanged() {
    let (mut cfg, mut ws, baseline) = fixture("writer").await;
    memory_io(&mut cfg, "");
    // `big` has three statements and no tests: evaluate refuses it in AdapsisOS mode.
    let gated = execute_native_tool("evaluate", &json!({"function":"NtRunLib.big","input":"1"}).to_string(), &cfg, &mut ws, &baseline).await;
    assert!(gated.execution.has_errors && gated.content.contains("no passing tests"), "{}", gated.content);
    let result = run_source(&cfg, &mut ws, &baseline, "a: int = NtRunLib.big(1)\nb: int = a + 1\nc: int = b + 1\nd: int = c + 1\nreturn d\n").await;
    assert!(!result.execution.has_errors, "{}", result.content);
    assert_eq!(output(&result), "6");
    // A run adds no test obligation and satisfies none: finish is decided by the baseline.
    let finished = execute_native_tool("finish", "{}", &cfg, &mut ws, &baseline).await;
    assert!(finished.execution.done_accepted && !finished.execution.has_errors, "{}", finished.content);
}

#[tokio::test]
async fn run_output_is_capped_at_the_run_limit_with_an_explicit_notice() {
    assert_eq!(run::RUN_MAX_OUTPUT_CHARS, 100_000);
    assert!(run::RUN_MAX_OUTPUT_CHARS > MAX_OUTPUT_CHARS);
    for (length, omitted) in [(50_000usize, 0usize), (100_000, 0), (100_001, 1), (250_000, 150_000)] {
        let (mut cfg, mut ws, baseline) = fixture("writer").await;
        // Multi-byte characters: the cap counts characters, not bytes.
        memory_io(&mut cfg, &"é".repeat(length));
        let result = run_source(&cfg, &mut ws, &baseline, "text: str = await file_read(\"fixture://large\")\nreturn text\n").await;
        assert!(!result.execution.has_errors, "{length}");
        let envelope = envelope(&result);
        let text = envelope["output"].as_str().unwrap();
        assert_eq!(envelope["truncated"], omitted > 0, "{length}");
        assert_eq!(envelope["omitted_chars"], omitted, "{length}");
        if omitted == 0 {
            assert_eq!(text.chars().count(), length);
            assert!(!text.contains("truncated"));
        } else {
            let notice = format!("\n…[truncated: {omitted} of {length} characters omitted]");
            assert!(text.ends_with(&notice), "{length}: …{}", &text[text.len() - 80..]);
            assert_eq!(text.chars().count(), 100_000 + notice.chars().count());
        }
        assert!(text.starts_with("ééé"));
    }
}

#[test]
fn cap_output_keeps_the_boundary_exact() {
    let exact = "x".repeat(run::RUN_MAX_OUTPUT_CHARS);
    let capped = run::cap_output(&exact);
    assert_eq!((capped.text.len(), capped.omitted), (run::RUN_MAX_OUTPUT_CHARS, 0));
    let over = run::cap_output(&format!("{exact}yz"));
    assert_eq!(over.omitted, 2);
    assert!(over.text.ends_with("…[truncated: 2 of 100002 characters omitted]"));
    assert!(!over.text.contains('y'));
}

#[tokio::test]
async fn run_source_and_argument_caps_are_the_native_caps() {
    let (mut cfg, mut ws, baseline) = fixture("writer").await;
    memory_io(&mut cfg, "");
    let padding = "#".repeat(MAX_SOURCE_BYTES);
    let result = run_source(&cfg, &mut ws, &baseline, &format!("{padding}\nreturn 1\n")).await;
    assert!(result.execution.has_errors && result.content.contains("field `source` exceeds 32000 bytes"), "{}", result.content);
    let fits = format!("{}\nreturn 1\n", "#".repeat(MAX_SOURCE_BYTES - 20));
    assert_eq!(output(&run_source(&cfg, &mut ws, &baseline, &fits).await), "1");
    let huge = json!({"source": "return 1\n", "module": "M".repeat(MAX_ARGUMENTS_BYTES)});
    let result = run(&cfg, &mut ws, &baseline, huge).await;
    assert!(result.execution.has_errors && result.content.contains("arguments exceed 64000 bytes"), "{}", result.content);
}

#[tokio::test]
async fn the_interpreter_guards_bound_a_block() {
    let (mut cfg, mut ws, baseline) = fixture("writer").await;
    memory_io(&mut cfg, "");
    let looping = run_source(&cfg, &mut ws, &baseline, "n: int = 0\nwhile n >= 0:\n    n = n + 1\nreturn n\n").await;
    assert!(looping.execution.has_errors && looping.content.contains("while loop exceeded 10000 iterations"), "{}", looping.content);
    let recursing = run_source(&cfg, &mut ws, &baseline, "return NtRunLib.recurse(1)\n").await;
    assert!(recursing.execution.has_errors && recursing.content.contains("maximum call depth"), "{}", recursing.content);
    // The thread-local guards are balanced: the next block runs normally.
    assert_eq!(output(&run_source(&cfg, &mut ws, &baseline, "return NtRunLib.double(4)\n").await), "8");
}

#[tokio::test]
async fn run_requires_the_io_runtime() {
    let (cfg, mut ws, baseline) = fixture("writer").await;
    assert!(cfg.io_sender.is_none());
    let result = run_source(&cfg, &mut ws, &baseline, "return 1\n").await;
    assert!(result.execution.has_errors && result.content.contains("requires the coroutine IO runtime"), "{}", result.content);
}

/// #88: the block source and its result are content. The default sinks carry neither;
/// the opt-in sink carries the result exactly as evaluate's does.
#[tokio::test]
async fn run_source_and_result_stay_out_of_the_default_log_sinks() {
    if !crate::logging::tests::child() {
        for enabled in [false, true] {
            let captured = crate::logging::tests::capture(
                "api::native_tools::tests::run_tests::run_source_and_result_stay_out_of_the_default_log_sinks",
                enabled,
            );
            assert!(!captured.contains("CANARY_RUN_SOURCE") || enabled, "source logged by default: {captured}");
            for canary in ["CANARY_RUN_RESULT", "CANARY_RUN_ERROR"] {
                assert_eq!(captured.contains(canary), enabled, "{canary}: {captured}");
            }
            assert!(captured.contains("[run-error]"), "{captured}");
            if !enabled {
                assert!(captured.contains("[run-error] detail_bytes="), "{captured}");
            }
        }
        return;
    }
    let (mut cfg, mut ws, baseline) = fixture("writer").await;
    memory_io(&mut cfg, "CANARY_RUN_RESULT");
    let result = run_source(&cfg, &mut ws, &baseline, "marker: str = \"CANARY_RUN_SOURCE\"\ntext: str = await file_read(\"fixture://log\")\nreturn concat(text, marker)\n").await;
    assert_eq!(output(&result), "CANARY_RUN_RESULTCANARY_RUN_SOURCE");
    let failed = run_source(&cfg, &mut ws, &baseline, "marker: str = \"CANARY_RUN_SOURCE\"\ncheck(\"probe\", 1 > 2, \"CANARY_RUN_ERROR\")\nreturn marker\n").await;
    assert!(failed.execution.has_errors && failed.content.contains("CANARY_RUN_ERROR"), "{}", failed.content);
    let refused = run_source(&cfg, &mut ws, &baseline, "CANARY_RUN_SOURCE = 1\n").await;
    assert!(refused.execution.has_errors, "{}", refused.content);
}
