//! Native `run` (#129): the Program is frozen for everything a block executes. A called
//! function or spawned task that reaches a program-changing builtin gets an ordinary
//! builtin error at dispatch, before any state is touched; the same function under native
//! `evaluate` behaves as it always did.
use super::run_tests::{assert_nothing_persisted, envelope, output, run_source, snapshot};
use super::*;
use crate::builtins::program_effects::program_changing_builtins;
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Mutex;

struct Names {
    lib: String,
    other: String,
    disk: String,
    fresh: String,
}

fn names(tag: &str) -> Names {
    Names {
        lib: format!("NtFrz{tag}"),
        other: format!("NtFrz{tag}Other"),
        disk: format!("NtFrz{tag}Disk"),
        fresh: format!("NtFrz{tag}Fresh"),
    }
}

/// Arguments with which each classified builtin SUCCEEDS on this fixture when it is not
/// frozen (the evaluate test proves that), so a refusal is the only reason nothing changes.
/// A newly classified builtin fails here until it is given arguments.
fn arguments(builtin: &str, n: &Names) -> Vec<String> {
    match builtin {
        "mutate" => vec![format!("+module {}\n+shared slot:Int = 7\n+fn made ()->Int\n  +return 1\n+end", n.fresh)],
        "fn_remove" => vec![format!("{}.victim", n.lib)],
        "type_remove" => vec![format!("{}.Thing", n.lib)],
        "module_remove" => vec![n.other.clone()],
        "module_create" => vec![n.fresh.clone()],
        "fn_replace" => vec![format!("{}.victim.s1", n.lib), "  +return 2".into()],
        "move_symbols" => vec!["victim".into(), n.other.clone()],
        "library_reload" => vec![n.disk.clone()],
        other => panic!("program-changing builtin `{other}` has no freeze fixture; add one"),
    }
}

/// One `call_<builtin>` function per classified builtin, a spawner, and things to change.
fn library_source(n: &Names) -> String {
    let mut source = format!(
        "+module {lib}\n+shared marker:Int = 0\n+type Thing = {{x:Int}}\n+fn victim ()->Int\n  +return 1\n+end\n",
        lib = n.lib
    );
    for builtin in program_changing_builtins() {
        let count = arguments(builtin, n).len();
        let params = (0..count).map(|i| format!("a{i}:String")).collect::<Vec<_>>().join(", ");
        let args = (0..count).map(|i| format!("a{i}")).collect::<Vec<_>>().join(", ");
        source.push_str(&format!(
            "+fn call_{builtin} ({params})->String [io,async]\n  +await done:String = {builtin}({args})\n  +return done\n+end\n"
        ));
    }
    source.push_str(&format!(
        "+fn spawner (code:String)->Int [io,async]\n  +spawn {lib}.call_mutate(code)\n  +return 1\n+end\n+module {other}\n+fn keep ()->Int\n  +return 3\n+end\n",
        lib = n.lib,
        other = n.other
    ));
    source
}

/// What a spawned task did, keyed by nothing: (function, program_frozen, outcome).
type Tasks = Arc<Mutex<Vec<(String, bool, String)>>>;

/// File writes are recorded; a Spawn runs the task the way the daemon's IO loop does:
/// its own blocking thread, a private Program wrapper, and a handle built by
/// `CoroutineHandle::for_spawned_task` from the request's flag.
fn task_io(cfg: &mut AppConfig) -> (Tasks, Arc<Mutex<Vec<String>>>) {
    use crate::coroutine::{CoroutineHandle, IoRequest, TaskRegistry, TaskSnapshotRegistry};
    let (sender, mut receiver) = tokio::sync::mpsc::channel(8);
    cfg.io_sender = Some(sender.clone());
    let tasks = Tasks::default();
    let files: Arc<Mutex<Vec<String>>> = Default::default();
    let (seen_tasks, seen_files) = (tasks.clone(), files.clone());
    let (program, runtime, meta) = (cfg.program.clone(), cfg.runtime.clone(), cfg.meta.clone());
    tokio::spawn(async move {
        while let Some(request) = receiver.recv().await {
            match request {
                IoRequest::FileWrite { path, reply, .. } => {
                    seen_files.lock().unwrap().push(path);
                    let _ = reply.send(Ok(()));
                }
                IoRequest::Spawn { function_name, args, program_frozen, reply } => {
                    let _ = reply.send(Ok(1));
                    let (sender, seen, program) = (sender.clone(), seen_tasks.clone(), program.read().await.clone());
                    let (runtime, meta) = (runtime.clone(), meta.clone());
                    tokio::task::spawn_blocking(move || {
                        crate::eval::EvalContext::new_minimal(runtime, meta, &program, crate::eval::make_shared_program_mut(&program)).install();
                        let handle = CoroutineHandle::for_spawned_task(sender, 1, TaskRegistry::default(), TaskSnapshotRegistry::default(), program_frozen);
                        let function = program.get_function(&function_name).unwrap();
                        let mut env = crate::eval::Env::new_with_shared_interner(&program.shared_interner);
                        env.populate_shared_from_program(&program);
                        env.set("__coroutine_handle", crate::eval::Value::CoroutineHandle(handle));
                        for (param, value) in function.params.iter().zip(&args) {
                            env.set(&param.name, value.clone());
                        }
                        let outcome = match crate::eval::eval_declared_function(&program, &function_name, function, &mut env) {
                            Ok(value) => format!("ok: {value}"),
                            Err(error) => format!("error: {error}"),
                        };
                        seen.lock().unwrap().push((function_name, program_frozen, outcome));
                    });
                }
                _ => panic!("unexpected IO in the freeze fixture"),
            }
        }
    });
    (tasks, files)
}

async fn spawned(tasks: &Tasks, count: usize) -> Vec<(String, bool, String)> {
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            if tasks.lock().unwrap().len() >= count {
                return tasks.lock().unwrap().clone();
            }
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
    })
    .await
    .expect("the spawned task did not finish")
}

/// Fixture files of this tag in the (process-shared) library directory, with contents.
fn library_files(tag: &str) -> BTreeMap<String, String> {
    let prefix = format!("NtFrz{tag}");
    std::fs::read_dir(crate::library::library_dir())
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_name().to_string_lossy().starts_with(&prefix))
        .map(|entry| (entry.file_name().to_string_lossy().into_owned(), std::fs::read_to_string(entry.path()).unwrap()))
        .collect()
}

struct Fixture {
    cfg: AppConfig,
    ws: WorkingSet,
    baseline: Program,
    names: Names,
    library: crate::library::LibraryState,
    tasks: Tasks,
    files: Arc<Mutex<Vec<String>>>,
}

async fn fixture(tag: &str) -> Fixture {
    let names = names(tag);
    let mut baseline = build_program(&library_source(&names)).await;
    baseline.require_modules = true;
    // A module that exists on disk only: an unfrozen library_reload adopts it and records
    // its hash; that is the state a frozen one must leave alone.
    std::fs::create_dir_all(crate::library::library_dir()).unwrap();
    std::fs::write(
        crate::library::library_dir().join(format!("{}.ax", names.disk)),
        format!("+module {}\n+fn loaded ()->Int\n  +return 9\n+end\n", names.disk),
    )
    .unwrap();
    let mut cfg = config_for("writer", AccessLevel::Full, PermissionConfig::default(), baseline.clone());
    let library = crate::library::LibraryState::new();
    cfg.meta.lock().unwrap().library_state = Some(library.clone());
    crate::eval::init_missing_shared_runtime_vars(&baseline, &cfg.runtime);
    let (tasks, files) = task_io(&mut cfg);
    let ws = cfg.snapshot_working_set().await;
    Fixture { cfg, ws, baseline, names, library, tasks, files }
}

fn shared_keys(cfg: &AppConfig) -> Vec<String> {
    let mut keys: Vec<String> = cfg.runtime.read().unwrap().shared_vars.snapshot().into_keys().collect();
    keys.sort();
    keys
}

fn python_literal(text: &str) -> String {
    format!("{text:?}")
}

#[tokio::test]
async fn a_called_function_reaching_a_program_changing_builtin_is_refused_at_the_call() {
    let builtins = program_changing_builtins();
    assert_eq!(builtins.len(), 8, "{builtins:?}");
    for builtin in builtins {
        let mut f = fixture("Run").await;
        let before = snapshot(&f.cfg, &f.ws);
        let (shared_before, hashes_before, files_before) =
            (shared_keys(&f.cfg), f.library.file_hashes_for_tests(), library_files("Run"));
        assert_eq!(shared_before, [format!("{}.marker", f.names.lib)]);
        assert!(hashes_before.is_empty());
        let args = arguments(builtin, &f.names).iter().map(|a| python_literal(a)).collect::<Vec<_>>().join(", ");
        let source = format!(
            "written: str = await file_write(\"fixture://before\", \"kept\")\ndone: str = await {}.call_{builtin}({args})\nlate: str = await file_write(\"fixture://after\", \"never\")\nreturn done\n",
            f.names.lib
        );
        let result = run_source(&f.cfg, &mut f.ws, &f.baseline, &source).await;
        // The callee saw an ordinary builtin error; the block failed there.
        assert!(result.execution.has_errors, "{builtin}: {}", result.content);
        let text = output(&result);
        assert!(text.starts_with(&format!("run error: {builtin}: refused.")), "{builtin}: {text}");
        assert!(text.contains("a run block cannot change the program") && text.contains("Use the mutate tool"), "{builtin}: {text}");
        assert!(!text.contains("Program check"), "{builtin}: the backstop must not be what caught it: {text}");
        assert_eq!(envelope(&result)["success"], false);
        // IO before the refusal stays done; nothing after it ran.
        assert_eq!(*f.files.lock().unwrap(), ["fixture://before"], "{builtin}");
        // Nothing was touched: Program, mutation history, shared runtime, library state, files.
        assert_nothing_persisted(&f.cfg, &f.ws, &f.baseline, &before).await;
        assert_eq!(shared_keys(&f.cfg), shared_before, "{builtin}: a +shared slot was initialized");
        assert_eq!(f.library.file_hashes_for_tests(), hashes_before, "{builtin}: library hash state changed");
        assert!(f.library.errors.lock().unwrap().is_empty() && f.library.load_errors.lock().unwrap().is_empty(), "{builtin}");
        assert_eq!(library_files("Run"), files_before, "{builtin}: library directory changed");
        let live = f.cfg.program.read().await;
        assert!(live.modules.iter().all(|m| m.name != f.names.fresh && m.name != f.names.disk), "{builtin}");
        assert!(live.get_function(&format!("{}.victim", f.names.lib)).is_some(), "{builtin}");
        drop(live);
        // The frozen run left the next call a working, unchanged program.
        let next = run_source(&f.cfg, &mut f.ws, &f.baseline, &format!("return {}.victim()\n", f.names.lib)).await;
        assert_eq!(output(&next), "1", "{builtin}: {}", next.content);
    }
}

/// The contrast: the same callees with the same arguments under native `evaluate` are not
/// frozen. Each builtin succeeds and its Program change is written back, as before.
#[tokio::test]
async fn the_same_callees_under_evaluate_behave_as_before() {
    for builtin in program_changing_builtins() {
        let mut f = fixture("Eval").await;
        let input = arguments(builtin, &f.names).iter().map(|a| python_literal(a)).collect::<Vec<_>>().join(", ");
        let result = execute_native_tool(
            "evaluate",
            &json!({"language":"python","function":format!("{}.call_{builtin}", f.names.lib),"input":input}).to_string(),
            &f.cfg, &mut f.ws, &f.baseline,
        ).await;
        assert!(!result.execution.has_errors, "{builtin}: {}", result.content);
        assert!(!result.content.contains("refused"), "{builtin}: {}", result.content);
        f.cfg.write_back_working_set(&f.ws).await;
        let live = f.cfg.program.read().await;
        assert!(*live != f.baseline, "{builtin}: evaluate still writes the change back");
        let module = |name: &str| live.modules.iter().find(|m| m.name == name);
        match builtin {
            "mutate" => {
                assert!(live.get_function(&format!("{}.made", f.names.fresh)).is_some());
                assert!(shared_keys(&f.cfg).contains(&format!("{}.slot", f.names.fresh)));
            }
            "module_create" => assert!(module(&f.names.fresh).is_some()),
            "module_remove" => assert!(module(&f.names.other).is_none()),
            "fn_remove" => assert!(live.get_function(&format!("{}.victim", f.names.lib)).is_none()),
            "type_remove" => assert!(module(&f.names.lib).unwrap().types.is_empty()),
            "move_symbols" => assert!(live.get_function(&format!("{}.victim", f.names.other)).is_some()),
            "fn_replace" => assert!(!Arc::ptr_eq(
                &module(&f.names.lib).unwrap().functions[0],
                &f.baseline.modules.iter().find(|m| m.name == f.names.lib).unwrap().functions[0]
            )),
            "library_reload" => {
                assert!(live.get_function(&format!("{}.loaded", f.names.disk)).is_some());
                assert!(f.library.file_hashes_for_tests().contains_key(&f.names.disk), "the reload recorded its hash");
            }
            other => panic!("no expectation for `{other}`"),
        }
    }
}

/// The flag travels with the handle: into a task the block spawns itself, and through a
/// user await into a task a called function spawns. Under evaluate neither is frozen.
#[tokio::test]
async fn tasks_spawned_from_a_block_are_frozen_too() {
    let mut f = fixture("Spawn").await;
    let before = snapshot(&f.cfg, &f.ws);
    let code = python_literal(&format!("+module {}\n+shared slot:Int = 7", f.names.fresh));
    let lib = f.names.lib.clone();
    let task = format!("{lib}.call_mutate");

    let direct = run_source(&f.cfg, &mut f.ws, &f.baseline, &format!("spawn({task}({code}))\nreturn 1\n")).await;
    assert_eq!(output(&direct), "1", "{}", direct.content);
    let nested = run_source(&f.cfg, &mut f.ws, &f.baseline, &format!("n: int = await {lib}.spawner({code})\nreturn n\n")).await;
    assert_eq!(output(&nested), "1", "{}", nested.content);
    let seen = spawned(&f.tasks, 2).await;
    for (function, frozen, outcome) in &seen {
        assert_eq!(function, &task);
        assert!(*frozen, "the Spawn request of a block carries the freeze");
        assert!(outcome.starts_with("error: mutate: refused."), "{outcome}");
    }
    assert_nothing_persisted(&f.cfg, &f.ws, &f.baseline, &before).await;
    assert_eq!(shared_keys(&f.cfg), [format!("{lib}.marker")], "a frozen task initialized a +shared slot");
    assert!(f.library.file_hashes_for_tests().is_empty());

    // The same spawner under evaluate: the task is not frozen and its mutate succeeds on the
    // task's private Program copy, exactly as before this change.
    let evaluated = execute_native_tool(
        "evaluate",
        &json!({"language":"python","function":format!("{lib}.spawner"),"input":code}).to_string(),
        &f.cfg, &mut f.ws, &f.baseline,
    ).await;
    assert!(!evaluated.execution.has_errors, "{}", evaluated.content);
    let seen = spawned(&f.tasks, 3).await;
    let (_, frozen, outcome) = &seen[2];
    assert!(!frozen, "an evaluate-spawned task is not frozen");
    assert!(outcome.starts_with("ok: ") && outcome.contains("Applied 1 mutation"), "{outcome}");
}

/// The backstop behind the freeze: if a Program ever differs after a block (a future
/// program-changing path that was not classified), the run fails instead of hiding it.
#[test]
fn a_program_difference_after_a_block_fails_the_run() {
    let mut base = Program::default();
    for op in crate::parser::parse("+module NtFrzBackstop\n+fn f ()->Int\n  +return 1\n+end\n").unwrap() {
        crate::validator::apply_and_validate(&mut base, &op).unwrap();
    }
    assert!(run::same_program(&base, &base.clone()));
    // Structurally equal, but a different declaration object: still a difference.
    let mut reallocated = base.clone();
    let copy = (*reallocated.modules[0].functions[0]).clone();
    reallocated.modules[0].functions[0] = Arc::new(copy);
    assert!(reallocated == base && !run::same_program(&base, &reallocated));
    let mut grown = base.clone();
    for op in crate::parser::parse("+module NtFrzBackstopNew").unwrap() {
        crate::validator::apply_and_validate(&mut grown, &op).unwrap();
    }
    assert!(!run::same_program(&base, &grown));

    for (outcome, fragment) in [
        (Ok(crate::eval::Value::string("value")), "The block ended with:\nvalue"),
        (Ok(crate::eval::Value::None), "Report this as a runtime bug."),
    ] {
        let result = run::finish("run", outcome, false);
        assert!(result.execution.has_errors && !result.execution.mutation_results[0].success);
        let text = output(&result);
        assert!(text.starts_with("run failed its Program check"), "{text}");
        assert!(text.contains("was not written back") && text.contains("stays done"), "{text}");
        assert!(text.contains(fragment), "{text}");
        assert_eq!(envelope(&result)["success"], false);
    }
    let ok = run::finish("run", Ok(crate::eval::Value::Int(3)), true);
    assert!(!ok.execution.has_errors);
    assert_eq!(output(&ok), "3");
}
