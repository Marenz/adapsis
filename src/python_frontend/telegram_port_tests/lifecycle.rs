//! Slice D1: module doc, token_path/allowed_path, `+startup` and the webhook `+route`.
//! Startup runs through the production call (main.rs:2149-2164): the stored lifecycle
//! body, a task-tracked handle, `__module_name`, `eval_function_body_named` with the
//! qualified name. File reads/existence checks and `+spawn` are intercepted as
//! IoRequests; no real file, token, network or Runtime IO loop is involved.
use crate::{
    ast,
    coroutine::{CoroutineHandle, IoRequest, Runtime},
    eval::{self, Value},
    session, shared_state,
};
use serde_json::{json, Value as Json};
use std::collections::BTreeSet;

mod http;

pub(super) fn expected_routes() -> Vec<ast::HttpRoute> {
    vec![ast::HttpRoute {
        method: "POST".into(),
        path: "/webhook/telegram".into(),
        handler_fn: "TelegramBot.handle_update".into(),
    }]
}

fn startup_function(block: &ast::LifecycleBlock) -> ast::FunctionDecl {
    ast::FunctionDecl {
        id: block.id.clone(),
        name: "startup".into(),
        params: vec![],
        return_type: ast::Type::String,
        effects: block.effects.clone(),
        body: block.body.clone(),
        tests: vec![],
        doc: None,
    }
}

fn effect_set(effects: &[ast::Effect]) -> Vec<String> {
    let mut names: Vec<_> = effects.iter().map(|e| format!("{e:?}")).collect();
    names.sort();
    names
}

/// Compare the corrected surface with the FROZEN module (selected subset): module
/// doc, all nine shared declarations, the startup block and the route. Startup differs
/// from the frozen block ONLY by two qualified calls and the effect spelling.
pub(super) fn verify_surface(frozen: &ast::Program, corrected: &ast::Program) {
    use ast::Effect::{Async, Io, Mut};
    let (old, new) = (&frozen.modules[0], &corrected.modules[0]);
    assert!(old
        .doc
        .as_deref()
        .is_some_and(|doc| doc.starts_with("Telegram bot: webhook OR long-poll intake")));
    assert_eq!(old.doc, new.doc);
    assert_eq!(old.shared_vars.len(), 9);
    assert_eq!(old.shared_vars, new.shared_vars);
    assert_eq!(old.routes, new.routes);
    assert_eq!(new.routes, expected_routes());
    assert_eq!(frozen.pending_routes, corrected.pending_routes);
    for module in [old, new] {
        assert!(module.shutdown.is_none());
        assert!(module.sources.is_empty() && module.event_decls.is_empty());
    }
    let (old_block, new_block) = (old.startup.as_ref().unwrap(), new.startup.as_ref().unwrap());
    // Python spells `async` last, so the corrected origin is [io,mut,async]; the
    // frozen block says [io,async,mut]. Only the order differs.
    assert_eq!(old_block.effects, [Io, Async, Mut]);
    assert_eq!(new_block.effects, [Io, Mut, Async]);
    assert_eq!(
        effect_set(&old_block.effects),
        effect_set(&new_block.effects)
    );
    let mut frozen_function = startup_function(old_block);
    frozen_function.effects = new_block.effects.clone();
    let locals: BTreeSet<String> = ["parse_id_list", "poll_loop"].map(String::from).into();
    let sites = super::qualification::verify(
        &frozen_function,
        &startup_function(new_block),
        "TelegramBot",
        &locals,
    );
    assert_eq!(
        sites, 2,
        "parse_id_list call and poll_loop spawn target qualified, nothing else changed"
    );
    // No semantic repair: ledger must say so and name the inherited observations.
    let manifest: Json = serde_json::from_str(include_str!(
        "../../../tests/fixtures/telegram-python-port/manifest.json"
    ))
    .unwrap();
    let surface = &manifest["modules"][0]["lifecycle_surface"];
    assert_eq!(surface["startup"]["semantic_repair"], json!(false));
    for test in [
        "telegram_port_startup_poll_mode_reads_files_trims_token_and_spawns",
        "telegram_port_startup_defaults_and_allowed_file_variants",
        "telegram_port_startup_non_poll_modes_never_spawn",
        "telegram_port_startup_failures_end_without_spawn_or_rollback",
        "telegram_port_startup_empty_token_still_spawns_and_rerun_respawns",
    ] {
        assert_eq!(
            surface["startup"]["tests"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|t| *t == test)
                .count(),
            1,
            "{test}"
        );
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Ev {
    Read(String),
    Exists(String),
    Spawn(String, Vec<String>),
}

#[derive(Clone, Copy)]
struct Script {
    token_path: &'static str,
    allowed_path: &'static str,
    token: Result<&'static str, &'static str>,
    exists: Result<bool, &'static str>,
    allowed: Result<&'static str, &'static str>,
    /// Drop the receiver right after the allowed-file answer so the evaluator's own
    /// `try_send` of the poll_loop spawn fails through the real tree path.
    close_before_spawn: bool,
}

const TOKEN_FILE: &str = "fixture/token.txt";
const ALLOWED_FILE: &str = "fixture/allowed.txt";

fn script() -> Script {
    Script {
        token_path: TOKEN_FILE,
        allowed_path: ALLOWED_FILE,
        token: Ok("SECRET-tök\n"),
        exists: Ok(true),
        allowed: Ok("# c\n42, 7\n\n9"),
        close_before_spawn: false,
    }
}

fn collect(mut receiver: tokio::sync::mpsc::Receiver<IoRequest>, s: Script) -> Vec<Ev> {
    let mut events = Vec::new();
    while let Some(request) = receiver.blocking_recv() {
        match request {
            IoRequest::FileRead { path, reply } => {
                let first = !events.iter().any(|e| matches!(e, Ev::Read(_)));
                let (expected, answer) = if first {
                    (s.token_path, s.token)
                } else {
                    (s.allowed_path, s.allowed)
                };
                assert_eq!(path, expected, "unexpected read order/path");
                events.push(Ev::Read(path));
                reply
                    .send(answer.map(str::to_owned).map_err(anyhow::Error::msg))
                    .unwrap();
                if !first && s.close_before_spawn {
                    return events;
                }
            }
            IoRequest::FileExists { path, reply } => {
                assert_eq!(path, s.allowed_path);
                events.push(Ev::Exists(path));
                reply.send(s.exists.map_err(anyhow::Error::msg)).unwrap();
            }
            IoRequest::Spawn {
                function_name,
                args,
                reply,
                ..
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
            _ => panic!("startup may only read, check existence and spawn"),
        }
    }
    events
}

struct Harness {
    _guard: super::routing::StateGuard,
    runtime: session::SharedRuntime,
}

impl Harness {
    fn new(program: &ast::Program) -> Self {
        let guard = super::routing::install(program, "1", "7");
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
    fn seeded(program: &ast::Program) -> Self {
        let harness = Self::new(program);
        harness.seed("token_path", TOKEN_FILE);
        harness.seed("allowed_path", ALLOWED_FILE);
        harness
    }
    fn seed(&self, name: &str, value: &str) {
        self.runtime
            .read()
            .unwrap()
            .shared_vars
            .insert(format!("TelegramBot.{name}"), Value::string(value));
    }
    fn text(&self, name: &str) -> Option<String> {
        match self
            .runtime
            .read()
            .unwrap()
            .shared_vars
            .get(&format!("TelegramBot.{name}"))
        {
            Some(Value::String(text)) => Some(text.to_string()),
            None => None,
            other => panic!("{name} must stay a String, got {other:?}"),
        }
    }
    fn running(&self) -> bool {
        matches!(
            self.runtime
                .read()
                .unwrap()
                .shared_vars
                .get("TelegramBot.poll_running"),
            Some(Value::Bool(true))
        )
    }
    fn run(&self, program: &ast::Program, s: Script) -> (Result<String, String>, Vec<Ev>) {
        let block = program.modules[0].startup.clone().expect("startup block");
        // Only the two registries are taken from a Runtime; its IO loop never runs
        // and requests go to our own channel.
        let (tasks, _io_rx) = Runtime::new();
        let (sender, receiver) = tokio::sync::mpsc::channel(256);
        let handle = CoroutineHandle::new_with_task(
            sender,
            0,
            tasks.task_registry.clone(),
            tasks.snapshot_registry.clone(),
        );
        let worker = std::thread::spawn(move || collect(receiver, s));
        let mut env = eval::Env::new_with_shared_interner(&program.shared_interner);
        env.set("__coroutine_handle", Value::CoroutineHandle(handle));
        env.set("__module_name", Value::string("TelegramBot"));
        let result =
            eval::eval_function_body_named(program, "TelegramBot.startup", &block.body, &mut env)
                .map(|value| match value {
                    Value::String(text) => text.to_string(),
                    other => panic!("startup must return a String, got {other:?}"),
                })
                .map_err(|error| error.to_string());
        drop(env);
        (result, worker.join().unwrap())
    }
}

fn spawn_poll_loop() -> Ev {
    Ev::Spawn("TelegramBot.poll_loop".into(), vec![])
}
fn read(path: &str) -> Ev {
    Ev::Read(path.into())
}
fn exists(path: &str) -> Ev {
    Ev::Exists(path.into())
}

#[test]
fn telegram_port_startup_poll_mode_reads_files_trims_token_and_spawns() {
    for program in super::send_chain::programs() {
        use ast::Effect::{Async, Io, Mut};
        let block = program.modules[0].startup.as_ref().unwrap();
        assert_eq!(block.effects, [Io, Mut, Async]);
        let harness = Harness::seeded(&program);
        let (result, events) = harness.run(&program, script());
        // "SECRET-tök" is 10 characters but 11 BYTES: len/length count bytes.
        assert_eq!("SECRET-tök".chars().count(), 10);
        assert_eq!("SECRET-tök".len(), 11);
        let result = result.unwrap();
        assert_eq!(result, "<redacted>11ch, admins=1, allowed=42,7,9");
        assert_eq!(
            events,
            [
                read(TOKEN_FILE),
                exists(ALLOWED_FILE),
                read(ALLOWED_FILE),
                spawn_poll_loop()
            ]
        );
        assert_eq!(
            harness.text("telegram_token").as_deref(),
            Some("SECRET-tök")
        );
        assert_eq!(harness.text("allowed_user_ids").as_deref(), Some("42,7,9"));
        assert_eq!(harness.text("admin_user_ids").as_deref(), Some("1"));
        assert!(!harness.running(), "startup never sets the poll latch");
        assert!(matches!(
            harness
                .runtime
                .read()
                .unwrap()
                .shared_vars
                .get("TelegramBot.poll_offset"),
            Some(Value::Int(0))
        ));
        // Token hygiene: only its LENGTH leaves startup; neither the text nor the
        // file names appear in the return value or in any spawn argument.
        for leaked in ["SECRET", "tök", TOKEN_FILE, ALLOWED_FILE] {
            assert!(!result.contains(leaked), "{leaked}");
            assert!(!format!("{:?}", events.last().unwrap()).contains(leaked));
        }
    }
}

#[test]
fn telegram_port_startup_defaults_and_allowed_file_variants() {
    for program in super::send_chain::programs() {
        // Unseeded paths are the declared (sanitized) defaults, which proves the new
        // shared declarations materialize; intake_mode defaults to poll.
        let harness = Harness::new(&program);
        let defaults = Script {
            token_path: "<redacted>",
            allowed_path: "<redacted>",
            token: Ok("abc"),
            exists: Ok(false),
            ..script()
        };
        let (result, events) = harness.run(&program, defaults);
        assert_eq!(result.unwrap(), "<redacted>3ch, admins=1, allowed=7");
        assert_eq!(
            events,
            [read("<redacted>"), exists("<redacted>"), spawn_poll_loop()]
        );
        // Missing allowed file: no second read, the existing list is left alone.
        let harness = Harness::seeded(&program);
        let missing = Script {
            exists: Ok(false),
            ..script()
        };
        let (result, events) = harness.run(&program, missing);
        assert_eq!(result.unwrap(), "<redacted>11ch, admins=1, allowed=7");
        assert_eq!(
            events,
            [read(TOKEN_FILE), exists(ALLOWED_FILE), spawn_poll_loop()]
        );
        assert_eq!(harness.text("allowed_user_ids").as_deref(), Some("7"));
        // An allowed file that yields nothing REPLACES the list with "" (parse_id_list
        // documents that the caller decides the fallback; startup supplies none).
        for raw in ["# only a comment\n\n", "  \n", ""] {
            let harness = Harness::seeded(&program);
            let empty = Script {
                allowed: Ok(raw),
                ..script()
            };
            let (result, _) = harness.run(&program, empty);
            assert_eq!(result.unwrap(), "<redacted>11ch, admins=1, allowed=");
            assert_eq!(harness.text("allowed_user_ids").as_deref(), Some(""));
        }
    }
}

#[test]
fn telegram_port_startup_non_poll_modes_never_spawn() {
    for program in super::send_chain::programs() {
        for mode in ["webhook", "", "Poll", "poll ", "polling"] {
            let harness = Harness::seeded(&program);
            harness.seed("intake_mode", mode);
            let (result, events) = harness.run(&program, script());
            // Same reads, same return string (both original branches are identical),
            // but the long-poll loop is never started.
            assert_eq!(
                result.unwrap(),
                "<redacted>11ch, admins=1, allowed=42,7,9",
                "mode {mode:?}"
            );
            assert_eq!(
                events,
                [read(TOKEN_FILE), exists(ALLOWED_FILE), read(ALLOWED_FILE)],
                "mode {mode:?}"
            );
        }
    }
}

#[test]
fn telegram_port_startup_failures_end_without_spawn_or_rollback() {
    for program in super::send_chain::programs() {
        // Outer errors cannot be caught in-language (#109 class). The caller
        // (main.rs:2161-2163) only eprintln!s them, so polling silently stays off.
        let harness = Harness::seeded(&program);
        let token_denied = Script {
            token: Err("token denied"),
            ..script()
        };
        let (result, events) = harness.run(&program, token_denied);
        assert!(result.unwrap_err().contains("token denied"));
        assert_eq!(events, [read(TOKEN_FILE)]);
        assert_eq!(
            harness.text("telegram_token").as_deref(),
            Some("fixture-token"),
            "a failed read must not wipe the existing token"
        );
        let harness = Harness::seeded(&program);
        let exists_failed = Script {
            exists: Err("stat failed"),
            ..script()
        };
        let (result, events) = harness.run(&program, exists_failed);
        assert!(result.unwrap_err().contains("stat failed"));
        assert_eq!(events, [read(TOKEN_FILE), exists(ALLOWED_FILE)]);
        assert_eq!(
            harness.text("telegram_token").as_deref(),
            Some("SECRET-tök"),
            "partial initialization is not rolled back"
        );
        let harness = Harness::seeded(&program);
        let allowed_failed = Script {
            allowed: Err("allowed denied"),
            ..script()
        };
        let (result, events) = harness.run(&program, allowed_failed);
        assert!(result.unwrap_err().contains("allowed denied"));
        assert_eq!(
            events,
            [read(TOKEN_FILE), exists(ALLOWED_FILE), read(ALLOWED_FILE)]
        );
        assert_eq!(harness.text("allowed_user_ids").as_deref(), Some("7"));
        // A poll_loop spawn that cannot be queued fails through the real tree
        // try_send path, after token and allow-list were already stored.
        let harness = Harness::seeded(&program);
        let closed = Script {
            close_before_spawn: true,
            ..script()
        };
        let (result, events) = harness.run(&program, closed);
        assert!(result.unwrap_err().contains("+spawn failed to send"));
        assert_eq!(
            events,
            [read(TOKEN_FILE), exists(ALLOWED_FILE), read(ALLOWED_FILE)]
        );
        assert_eq!(harness.text("allowed_user_ids").as_deref(), Some("42,7,9"));
    }
}

#[test]
fn telegram_port_startup_empty_token_still_spawns_and_rerun_respawns() {
    for program in super::send_chain::programs() {
        let harness = Harness::seeded(&program);
        let empty = Script {
            token: Ok("\n"),
            exists: Ok(false),
            ..script()
        };
        let (result, events) = harness.run(&program, empty);
        // An empty token is not rejected: the loop is still started (slice B's
        // backoff turns the resulting 404 cycles into a 3 s cadence, not a hot loop).
        assert_eq!(result.unwrap(), "<redacted>0ch, admins=1, allowed=7");
        assert_eq!(
            events,
            [read(TOKEN_FILE), exists(ALLOWED_FILE), spawn_poll_loop()]
        );
        assert_eq!(harness.text("telegram_token").as_deref(), Some(""));
        // Startup does not consult the latch: every run queues another poll_loop
        // (the spawned loop itself answers "already running" when one is active).
        let (first, first_events) = harness.run(&program, script());
        let (second, second_events) = harness.run(&program, script());
        assert_eq!(first, second);
        assert_eq!(first_events, second_events);
        assert_eq!(second_events.last(), Some(&spawn_poll_loop()));
    }
}

#[test]
fn telegram_port_route_declaration_and_production_dispatch_binding() {
    for program in super::send_chain::programs() {
        let routes = &program.modules[0].routes;
        assert_eq!(*routes, expected_routes());
        assert_eq!(program.pending_routes, expected_routes());
        let handler = routes[0].handler_fn.clone();
        let function = program
            .get_function(&handler)
            .expect("route handler exists");
        // Dispatch binds the raw body to the FIRST parameter, whatever its name.
        assert_eq!(function.params.len(), 1);
        assert_eq!(function.params[0].name, "payload");
        for (mode, body, spawned) in [
            (
                "webhook",
                "{\"update_id\":1,\"message\":{\"text\":\"ä😀\\n\"}}",
                true,
            ),
            ("webhook", "not json", true),
            ("webhook", "", true),
            ("webhook", "\u{feff}{}\n", true),
            ("poll", "{\"update_id\":1}", false),
        ] {
            let harness = Harness::new(&program);
            harness.seed("intake_mode", mode);
            let (response, spawns) = dispatch_replica(&program, &handler, body);
            assert_eq!(response.unwrap(), "{}", "{mode} {body:?}");
            let expected: Vec<Ev> = if spawned {
                vec![Ev::Spawn(
                    "TelegramBot.process_update".into(),
                    vec![body.into()],
                )]
            } else {
                vec![]
            };
            assert_eq!(spawns, expected, "{mode} {body:?}");
            drop(harness);
        }
    }
}

/// Mirror of api/mod.rs:1692-1715 (adapsis_route_dispatch): shared defaults, positional
/// body binding, a plain CoroutineHandle (no task id) and the qualified handler name.
fn dispatch_replica(
    program: &ast::Program,
    handler: &str,
    body: &str,
) -> (Result<String, String>, Vec<Ev>) {
    let function = program.get_function(handler).unwrap();
    let runtime = shared_state::get_shared_runtime().unwrap();
    eval::init_missing_shared_runtime_vars(program, &runtime);
    let (sender, mut receiver) = tokio::sync::mpsc::channel(32);
    let mut env = eval::Env::new_with_shared_interner(&program.shared_interner);
    env.populate_shared_from_program(program);
    env.set_shared_runtime(runtime);
    env.set(
        "__coroutine_handle",
        Value::CoroutineHandle(CoroutineHandle::new(sender)),
    );
    eval::bind_input_to_params(program, function, &Value::string(body), &mut env);
    let result = eval::eval_declared_function(program, handler, function, &mut env)
        .map(|value| match value {
            Value::String(text) => text.to_string(),
            other => panic!("route handler must return a String, got {other:?}"),
        })
        .map_err(|error| error.to_string());
    drop(env);
    let mut spawns = Vec::new();
    while let Ok(request) = receiver.try_recv() {
        let IoRequest::Spawn {
            function_name,
            args,
            reply,
            ..
        } = request
        else {
            panic!("a webhook handler may only spawn")
        };
        drop(reply);
        spawns.push(Ev::Spawn(
            function_name,
            args.into_iter()
                .map(|value| match value {
                    Value::String(text) => text.to_string(),
                    other => panic!("spawn argument must be a String, got {other:?}"),
                })
                .collect(),
        ));
    }
    (result, spawns)
}
