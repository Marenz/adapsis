//! Polling slice C: poll_loop (R6 option ii, #107). `http_get`, `sleep` and `+spawn`
//! are intercepted as IoRequests on the normal async path (the real tree spawn path:
//! no fake runtime), nothing sleeps for real and no network/file IO exists. A hook in
//! the interceptor writes the typed per-key `poll_running` slot, which the tree
//! re-reads on every access (Env::get -> materialize_shared_value -> read_shared_value,
//! eval/mod.rs:498-573), so the next while-condition evaluation observes it.
use crate::{
    ast,
    coroutine::{CoroutineHandle, IoRequest},
    eval::{self, Value},
    parser, session, shared_state, validator,
};
use serde_json::{json, Value as Json};

const TESTS: [&str; 4] = [
    "telegram_port_poll_loop_latch_stop_and_already_running",
    "telegram_port_poll_loop_generation_restart_and_successor",
    "telegram_port_poll_loop_outer_errors_keep_latch_and_spawn_failure_order",
    "telegram_port_poll_loop_original_shape_dies_at_tree_cap",
];
const EMPTY: &str = r#"{"ok":true,"result":[]}"#;

pub(in super::super) fn verify_repair(original: &ast::FunctionDecl, corrected: &ast::FunctionDecl) {
    use ast::{Expr, StatementKind as S};
    assert!(original.params.is_empty() && corrected.params.is_empty());
    assert_eq!(original.return_type, corrected.return_type);
    let set = |effects: &[ast::Effect]| {
        let mut names: Vec<_> = effects.iter().map(|e| format!("{e:?}")).collect();
        names.sort();
        names
    };
    assert_eq!(set(&original.effects), set(&corrected.effects));
    assert_eq!(corrected.body.len(), 6);
    assert!(matches!(&corrected.body[2].kind, S::Let { name, .. } if name == "cycles"));
    let S::While { condition, body } = &corrected.body[3].kind else {
        panic!("bounded loop moved")
    };
    let bound = serde_json::to_string(condition).unwrap();
    assert!(bound.contains("poll_running") && bound.contains("cycles") && bound.contains("1000"));
    assert!(matches!(&body[1].kind, S::Set { name, .. } if name == "cycles"));
    // Restart branch: clear the latch BEFORE queuing the successor, so a spawn that
    // cannot be queued leaves a recoverable latch; zero-arg qualified self spawn.
    let S::Branch {
        condition: restart_condition,
        then_body,
        else_body,
    } = &corrected.body[4].kind
    else {
        panic!("restart branch moved")
    };
    assert_eq!(*restart_condition, Expr::Identifier("poll_running".into()));
    assert!(else_body.is_empty() && then_body.len() == 3);
    assert!(matches!(&then_body[0].kind, S::Set { name, value }
        if name == "poll_running" && serde_json::to_string(value).unwrap().contains("false")));
    assert!(
        matches!(&then_body[1].kind, S::Spawn { call, binding: None }
        if call.callee == "TelegramBot.poll_loop" && call.args.is_empty())
    );
    assert!(serde_json::to_string(&then_body[2].kind)
        .unwrap()
        .contains("poll loop restarted"));
    // Undo ONLY the ledgered transformations; the rest must equal the frozen
    // function modulo the single qualified poll_once call.
    let mut restored = corrected.clone();
    restored.body.remove(4);
    restored.body.remove(2);
    let S::While { condition, body } = &mut restored.body[2].kind else {
        panic!()
    };
    *condition = Expr::Identifier("poll_running".into());
    body.pop();
    restored.effects = original.effects.clone();
    restored.doc = original.doc.clone();
    let locals = ["poll_once".to_owned()].into_iter().collect();
    assert_eq!(
        super::super::qualification::verify(original, &restored, "TelegramBot", &locals),
        1
    );
    let manifest: Json = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/telegram-python-port/manifest.json"
    ))
    .unwrap();
    for test in TESTS {
        assert_eq!(
            manifest["modules"][0]["contract_tests"][test],
            json!(["poll_loop"]),
            "{test}"
        );
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Ev {
    Get(i64),
    Sleep(i64),
    Spawn(String, Vec<String>),
}

type Reply = Result<String, &'static str>;

#[derive(Default)]
struct Script {
    /// 1-based getUpdates number -> scripted reply; anything else is an empty page.
    replies: Vec<(usize, Reply)>,
    /// Write poll_running=false while handling this getUpdates (a stop request).
    stop_at_get: Option<usize>,
    /// Drop the interceptor's receiver before answering this getUpdates, so the
    /// evaluator's next request fails through its real try_send path.
    close_before_get: Option<usize>,
    sleep_fails: bool,
}

struct Harness {
    _guard: super::super::routing::StateGuard,
    runtime: session::SharedRuntime,
}

impl Harness {
    fn new(program: &ast::Program) -> Self {
        let guard = super::super::routing::install(program, "1", "7");
        let runtime = shared_state::get_shared_runtime().unwrap();
        runtime
            .read()
            .unwrap()
            .shared_vars
            .insert("TelegramBot.poll_offset".into(), Value::Int(0));
        Self {
            _guard: guard,
            runtime,
        }
    }
    fn seed(&self, name: &str, value: Value) {
        self.runtime
            .read()
            .unwrap()
            .shared_vars
            .insert(format!("TelegramBot.{name}"), value);
    }
    fn flag(&self) -> bool {
        match self
            .runtime
            .read()
            .unwrap()
            .shared_vars
            .get("TelegramBot.poll_running")
        {
            Some(Value::Bool(value)) => value,
            other => panic!("poll_running must be a materialized Bool, got {other:?}"),
        }
    }
    fn offset(&self) -> i64 {
        match self
            .runtime
            .read()
            .unwrap()
            .shared_vars
            .get("TelegramBot.poll_offset")
        {
            Some(Value::Int(value)) => value,
            other => panic!("poll_offset must stay Int, got {other:?}"),
        }
    }
    fn run(&self, program: &ast::Program, script: Script) -> (Result<String, String>, Vec<Ev>) {
        let (sender, receiver) = tokio::sync::mpsc::channel(256);
        let runtime = self.runtime.clone();
        let worker = std::thread::spawn(move || collect(receiver, script, runtime));
        let result = eval::eval_async_function(
            program,
            "TelegramBot.poll_loop",
            &parser::parse_single_expr("{}").unwrap(),
            CoroutineHandle::new(sender),
        )
        .map(|value| match value {
            Value::String(text) => text.to_string(),
            other => panic!("poll_loop must return a String, got {other:?}"),
        })
        .map_err(|error| error.to_string());
        (result, worker.join().unwrap())
    }
}

fn collect(
    receiver: tokio::sync::mpsc::Receiver<IoRequest>,
    script: Script,
    runtime: session::SharedRuntime,
) -> Vec<Ev> {
    let mut receiver = Some(receiver);
    let mut events = Vec::new();
    let mut gets = 0;
    loop {
        let Some(request) = receiver.as_mut().and_then(|rx| rx.blocking_recv()) else {
            break;
        };
        match request {
            IoRequest::HttpGet { url, reply } => {
                gets += 1;
                let prefix = "<redacted>fixture-token/getUpdates?timeout=30&offset=";
                let offset = url
                    .strip_prefix(prefix)
                    .unwrap_or_else(|| panic!("unexpected getUpdates URL {url}"));
                events.push(Ev::Get(offset.parse().expect("integer offset")));
                if script.stop_at_get == Some(gets) {
                    runtime
                        .read()
                        .unwrap()
                        .shared_vars
                        .insert("TelegramBot.poll_running".into(), Value::Bool(false));
                }
                if script.close_before_get == Some(gets) {
                    receiver = None;
                }
                let scripted = script
                    .replies
                    .iter()
                    .find(|(number, _)| *number == gets)
                    .map(|(_, reply)| reply.clone())
                    .unwrap_or_else(|| Ok(EMPTY.to_owned()));
                reply.send(scripted.map_err(anyhow::Error::msg)).unwrap();
            }
            IoRequest::Sleep { ms, reply } => {
                events.push(Ev::Sleep(ms as i64));
                reply
                    .send(if script.sleep_fails {
                        Err(anyhow::anyhow!("fixture sleep failure"))
                    } else {
                        Ok(())
                    })
                    .unwrap();
            }
            IoRequest::Spawn {
                function_name,
                args,
                reply,
            } => {
                drop(reply);
                let args = args
                    .into_iter()
                    .map(|value| match value {
                        Value::String(text) => text.to_string(),
                        other => panic!("spawn argument must be a String, got {other:?}"),
                    })
                    .collect();
                events.push(Ev::Spawn(function_name, args));
            }
            _ => panic!("poll_loop may only get, sleep and spawn"),
        }
    }
    events
}

fn update(id: i64) -> Json {
    json!({"update_id": id, "message": {"text": format!("t{id} ä😀")}})
}
fn page(updates: &[Json]) -> Reply {
    Ok(json!({"ok": true, "result": updates}).to_string())
}
fn process(update: &Json) -> Ev {
    Ev::Spawn(
        "TelegramBot.process_update".into(),
        vec![update.to_string()],
    )
}
fn successor() -> Ev {
    Ev::Spawn("TelegramBot.poll_loop".into(), vec![])
}
fn ok(text: &str) -> Result<String, String> {
    Ok(text.to_owned())
}

#[test]
fn telegram_port_poll_loop_latch_stop_and_already_running() {
    for program in super::super::send_chain::programs() {
        assert!(
            crate::vm::compile_function(
                program.get_function("TelegramBot.poll_loop").unwrap(),
                &program
            )
            .is_err(),
            "shared access is not direct-VM coverage"
        );
        // The latch answers with ZERO IO and leaves the running loop's flag alone.
        let harness = Harness::new(&program);
        harness.seed("poll_running", Value::Bool(true));
        let (result, events) = harness.run(&program, Script::default());
        assert_eq!(result, ok("already running"));
        assert!(events.is_empty());
        assert!(harness.flag());
        // A stop request written during the third getUpdates ends the loop after that
        // cycle completes; no successor is spawned (stop is not resurrected).
        let harness = Harness::new(&program);
        let one = update(5);
        let script = Script {
            replies: vec![(2, page(std::slice::from_ref(&one)))],
            stop_at_get: Some(3),
            ..Script::default()
        };
        let (result, events) = harness.run(&program, script);
        assert_eq!(result, ok("poll loop stopped"));
        assert_eq!(
            events,
            [Ev::Get(0), Ev::Get(0), process(&one), Ev::Get(6)],
            "exactly three cycles, offset carried between them, no spawn of poll_loop"
        );
        assert!(!harness.flag());
        assert_eq!(harness.offset(), 6);
        // Stop-then-start under the same runtime works (flag is false): the new run
        // takes the latch again and its first URL carries the persisted offset.
        let (result, events) = harness.run(
            &program,
            Script {
                stop_at_get: Some(1),
                ..Script::default()
            },
        );
        assert_eq!(result, ok("poll loop stopped"));
        assert_eq!(events, [Ev::Get(6)]);
    }
}

#[test]
fn telegram_port_poll_loop_generation_restart_and_successor() {
    for program in super::super::send_chain::programs() {
        let harness = Harness::new(&program);
        let seven = update(7);
        let script = Script {
            replies: vec![(500, page(std::slice::from_ref(&seven)))],
            ..Script::default()
        };
        let (result, events) = harness.run(&program, script);
        // Exactly 1000 cycles, then ONE zero-arg qualified successor, latch cleared.
        assert_eq!(result, ok("poll loop restarted"));
        let mut expected = vec![Ev::Get(0); 500];
        expected.push(process(&seven));
        expected.extend(vec![Ev::Get(8); 500]);
        expected.push(successor());
        assert_eq!(events.len(), 1002);
        assert_eq!(events, expected);
        assert!(!harness.flag(), "latch cleared before the successor starts");
        assert_eq!(harness.offset(), 8);
        // The intercepted successor is simulated by invoking the same entry under the
        // same shared state: it takes the latch and continues at the saved offset.
        let (result, events) = harness.run(
            &program,
            Script {
                stop_at_get: Some(2),
                ..Script::default()
            },
        );
        assert_eq!(result, ok("poll loop stopped"));
        assert_eq!(events, [Ev::Get(8), Ev::Get(8)]);
        assert!(!harness.flag());
        // A concurrent starter in the hand-off window loses cleanly: while the flag is
        // set (successor already running) a second poll_loop() does nothing.
        harness.seed("poll_running", Value::Bool(true));
        let (result, events) = harness.run(&program, Script::default());
        assert_eq!(result, ok("already running"));
        assert!(events.is_empty());
    }
}

#[test]
fn telegram_port_poll_loop_outer_errors_keep_latch_and_spawn_failure_order() {
    for program in super::super::send_chain::programs() {
        // Outer errors cannot be caught in-language (#109): the latch stays set, so
        // every later poll_loop() answers "already running" until the flag is reset.
        let harness = Harness::new(&program);
        let script = Script {
            replies: vec![(1, Err("connect lost"))],
            sleep_fails: true,
            ..Script::default()
        };
        let (result, events) = harness.run(&program, script);
        assert!(result.unwrap_err().contains("fixture sleep failure"));
        assert_eq!(events, [Ev::Get(0), Ev::Sleep(3000)]);
        assert!(harness.flag(), "latch survives the outer error");
        let (result, events) = harness.run(&program, Script::default());
        assert_eq!(result, ok("already running"));
        assert!(events.is_empty());

        // An id beyond i64 stays terminal after the earlier update was spawned.
        let harness = Harness::new(&program);
        let huge = r#"{"result":[{"update_id":4},{"update_id":18446744073709551615}]}"#;
        let script = Script {
            replies: vec![(1, Ok(huge.to_owned()))],
            ..Script::default()
        };
        let (result, events) = harness.run(&program, script);
        assert!(result.unwrap_err().contains("outside signed 64-bit range"));
        assert_eq!(events, [Ev::Get(0), process(&json!({"update_id": 4}))]);
        assert!(harness.flag());
        assert_eq!(harness.offset(), 5);
        assert_eq!(
            harness.run(&program, Script::default()).0,
            ok("already running")
        );

        // Spawn-failure order, through the real tree spawn path: the receiver is
        // dropped before the 1000th answer, so the successor's try_send fails. The
        // latch was cleared first, so the state is recoverable, not stuck.
        let harness = Harness::new(&program);
        let script = Script {
            close_before_get: Some(1000),
            ..Script::default()
        };
        let (result, events) = harness.run(&program, script);
        assert!(result.unwrap_err().contains("+spawn failed to send"));
        assert_eq!(events, vec![Ev::Get(0); 1000]);
        assert!(
            !harness.flag(),
            "clear-then-spawn leaves the latch recoverable"
        );
        let (result, events) = harness.run(
            &program,
            Script {
                stop_at_get: Some(1),
                ..Script::default()
            },
        );
        assert_eq!(result, ok("poll loop stopped"));
        assert_eq!(events, [Ev::Get(0)]);
    }
}

/// Test-only evidence for R6, not an inventoried function: the frozen loop shape
/// (`while <shared flag>: await ...`) dies at the tree-walker's per-statement cap.
const CAP_PROBE: &str = "+module CapProbe\n+shared go:Bool = true\n+fn spin ()->String [io,async,mut]\n  +while go\n    +await _:Int = sleep(0)\n+end\n";

#[test]
fn telegram_port_poll_loop_original_shape_dies_at_tree_cap() {
    let [mut program, _] = super::super::send_chain::programs();
    for operation in parser::parse(CAP_PROBE).unwrap() {
        validator::apply_and_validate(&mut program, &operation).unwrap();
    }
    program.rebuild_function_index();
    let _guard = super::super::routing::install(&program, "1", "7");
    let (sender, mut receiver) = tokio::sync::mpsc::channel(256);
    let worker = std::thread::spawn(move || {
        let mut sleeps = 0;
        while let Some(request) = receiver.blocking_recv() {
            let IoRequest::Sleep { ms, reply } = request else {
                panic!("cap probe may only sleep")
            };
            assert_eq!(ms, 0);
            sleeps += 1;
            reply.send(Ok(())).unwrap();
        }
        sleeps
    });
    // The shared flag keeps the VM from compiling this, so (like a spawned task) it
    // runs on the tree-walker, which counts passes of this ONE while statement.
    let result = eval::eval_async_function(
        &program,
        "CapProbe.spin",
        &parser::parse_single_expr("{}").unwrap(),
        CoroutineHandle::new(sender),
    );
    let error = result.unwrap_err().to_string();
    assert!(
        error.contains("while loop exceeded 10000 iterations"),
        "{error}"
    );
    assert_eq!(
        worker.join().unwrap(),
        10_000,
        "body ran exactly 10000 times"
    );
}
