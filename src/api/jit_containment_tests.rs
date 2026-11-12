//! #96 at the API boundary: no state lock is held while a JIT-eligible
//! function is evaluated, and a panicking JIT compile reaches neither the
//! caller nor the session locks.
use super::*;
use crate::eval::jit_seam;
use std::cell::Cell;
use std::rc::Rc;

fn config_with(source: &str) -> AppConfig {
    let config = test_config();
    let mut program = crate::ast::Program::default();
    for operation in crate::parser::parse(source).unwrap() {
        crate::validator::apply_and_validate(&mut program, &operation).unwrap();
    }
    program.rebuild_function_index();
    *config.program.try_write().unwrap() = program;
    config.meta.lock().unwrap().revision = 7;
    config
}

/// A compile hook that records whether the session locks are free at the
/// moment the evaluation runs, and then panics like Cranelift does.
fn probe_locks_then_panic(config: &AppConfig) -> (jit_seam::Reset, Rc<Cell<Option<(bool, bool)>>>) {
    let seen = Rc::new(Cell::new(None));
    let (program, meta, record) = (config.program.clone(), config.meta.clone(), seen.clone());
    let reset = jit_seam::hook(move || {
        record.set(Some((program.try_write().is_ok(), meta.try_lock().is_ok())));
        panic!("injected JIT compile panic (test)");
    });
    (reset, seen)
}

#[tokio::test]
async fn watch_evaluation_holds_no_state_lock_and_survives_a_jit_panic() {
    let config = config_with("+fn probe ()->Int\n  +return 11\n+end\n");
    let (_seam, seen) = probe_locks_then_panic(&config);
    let input = crate::parser::Expr::StructLiteral(vec![]);
    for _ in 0..3 {
        let result = execute::watch_evaluate(&config.program, &config.meta, &config.jit_cache, "probe", &input).await;
        assert_eq!(result, "11", "the watcher gets the VM result");
    }
    assert_eq!(seen.get(), Some((true, true)), "(program lock free, meta lock free) while the function was evaluated");
    assert_eq!((jit_seam::attempts(), jit_seam::journaled()), (1, 1), "the watched revision is compiled and journaled once");
    assert!(!config.meta.is_poisoned(), "the session metadata lock was poisoned");
    assert_eq!(config.meta.lock().unwrap().revision, 7);
    assert!(!config.jit_cache.is_poisoned());
    assert!(config.program.try_write().is_ok());
    // An ordinary evaluation error is still reported as text.
    let missing = execute::watch_evaluate(&config.program, &config.meta, &config.jit_cache, "absent", &input).await;
    assert!(missing.starts_with("error: function `absent` not found"), "{missing}");
}

#[tokio::test]
async fn watch_evaluation_uses_the_revision_of_the_program_it_snapshots() {
    let config = config_with("+fn probe ()->Int\n  +return 11\n+end\n");
    let input = crate::parser::Expr::StructLiteral(vec![]);
    assert_eq!(execute::watch_evaluate(&config.program, &config.meta, &config.jit_cache, "probe", &input).await, "11");
    assert!(matches!(&*config.jit_cache.lock().unwrap(), Some((7, slot)) if slot.is_compiled()));
}

#[tokio::test]
async fn api_eval_holds_no_state_lock_and_survives_a_jit_panic() {
    let config = config_with("+fn probe ()->Int\n  +return 11\n+end\n");
    let (_seam, seen) = probe_locks_then_panic(&config);
    for _ in 0..2 {
        let Json(response) = eval_fn(
            State(config.clone()),
            Json(EvalRequest { function: "probe".to_string(), input: String::new(), expression: None }),
        )
        .await;
        assert_eq!((response.result.as_str(), response.success, response.compiled), ("11", true, Some(false)));
    }
    assert_eq!(seen.get(), Some((true, true)), "(program lock free, meta lock free) while the function was evaluated");
    assert_eq!((jit_seam::attempts(), jit_seam::journaled()), (1, 1));
    assert!(!config.meta.is_poisoned() && !config.jit_cache.is_poisoned());
    // The evals were recorded: the metadata lock is fully usable afterwards.
    assert_eq!(config.meta.lock().unwrap().history.len(), 2);
}

// --- one snapshot of program and revision ----------------------------------

fn program_returning(value: i64) -> crate::ast::Program {
    let mut program = crate::ast::Program::default();
    for operation in crate::parser::parse(&format!("+fn probe ()->Int\n  +return {value}\n+end\n")).unwrap() {
        crate::validator::apply_and_validate(&mut program, &operation).unwrap();
    }
    program.rebuild_function_index();
    program
}

fn probe_value(program: &crate::ast::Program) -> String {
    let input = crate::parser::parse_single_expr("{}").unwrap();
    crate::eval::eval_call_with_input(program, "probe", &input).unwrap()
}

/// Queue a writer behind the program read the caller currently holds: it
/// replaces the program and moves the revision to 8 while it holds the write
/// lock. Returns once the writer is waiting for the lock.
fn queue_writer(config: &AppConfig) -> std::thread::JoinHandle<()> {
    let (config, new) = (config.clone(), program_returning(22));
    let (queued_tx, queued_rx) = std::sync::mpsc::sync_channel(1);
    let writer = std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
            let lock = config.program.write();
            tokio::pin!(lock);
            assert!(matches!(futures::poll!(&mut lock), std::task::Poll::Pending), "the reader does not hold the program");
            queued_tx.send(()).unwrap();
            let mut live = lock.await;
            *live = new;
            config.meta.lock().unwrap().revision = 8;
        });
    });
    queued_rx.recv_timeout(std::time::Duration::from_secs(60)).unwrap();
    writer
}

/// Wait until some other thread holds the program lock (true), or give up.
fn program_is_held(config: &AppConfig, wait: std::time::Duration) -> bool {
    let until = std::time::Instant::now() + wait;
    while config.program.try_write().is_ok() {
        if std::time::Instant::now() >= until {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    true
}

/// The reviewer's race (#96 review): `/api/eval` read the revision under one
/// program read and cloned the program under a second one; a writer queued in
/// between made the warm revision-7 module run on behalf of the revision-8
/// program, and the history recorded "revision 8 evaluated to 11".
///
/// With one snapshot the call evaluates either the program before the writer
/// or the one after it, and says which. Here the writer is queued behind the
/// call's read, so it is the one before: 11, recorded at revision 7.
#[test]
fn api_eval_evaluates_and_records_one_consistent_snapshot() {
    let config = test_config();
    *config.program.try_write().unwrap() = program_returning(11);
    config.meta.lock().unwrap().revision = 7;
    let input = crate::parser::parse_single_expr("{}").unwrap();
    let old = config.program.try_read().unwrap().clone();
    assert_eq!(
        crate::eval::eval_compiled_or_interpreted_cached(&old, "probe", &input, Some(&config.jit_cache), 7).unwrap(),
        ("11".into(), true),
        "a warm module for revision 7"
    );
    let eval = |config: AppConfig| {
        tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
            eval_fn(State(config), Json(EvalRequest { function: "probe".into(), input: String::new(), expression: None })).await.0
        })
    };
    let hold_meta = config.meta.lock().unwrap();
    let caller = {
        let config = config.clone();
        std::thread::spawn(move || eval(config))
    };
    assert!(program_is_held(&config, std::time::Duration::from_secs(60)), "the call never took its program read");
    // The call holds its program read and waits for the metadata.
    let writer = queue_writer(&config);
    drop(hold_meta);
    let response = caller.join().unwrap();
    writer.join().unwrap();
    assert_eq!(config.meta.lock().unwrap().revision, 8, "the writer ran");
    let recorded = |index: usize| match &config.meta.lock().unwrap().history[index] {
        crate::session::HistoryEntry::Eval { revision, result, .. } => (*revision, result.clone()),
        other => panic!("{other:?}"),
    };
    assert_eq!((response.result.as_str(), response.compiled), ("11", Some(true)));
    assert_eq!(recorded(0), (7, "11".to_string()), "the history names the revision that was evaluated");
    // The next call sees the writer's program and never the old module.
    let response = eval(config.clone());
    assert_eq!((response.result.as_str(), response.compiled), ("22", Some(true)));
    assert_eq!(recorded(1), (8, "22".to_string()));
}

/// `snapshot_working_set` cloned the program, released the read, and then
/// cloned the metadata: a writer in between gave a working set with the old
/// program under the new revision.
#[test]
fn working_set_snapshot_pairs_the_program_with_its_revision() {
    let config = test_config();
    *config.program.try_write().unwrap() = program_returning(11);
    config.meta.lock().unwrap().revision = 7;
    let mut hold_meta = config.meta.lock().unwrap();
    let caller = {
        let config = config.clone();
        std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(config.snapshot_working_set())
        })
    };
    let writer = if program_is_held(&config, std::time::Duration::from_millis(1500)) {
        // The snapshot holds the program while it waits for the metadata.
        let writer = queue_writer(&config);
        drop(hold_meta);
        Some(writer)
    } else {
        // The snapshot is not holding the program (the old code released it
        // before reading the metadata, or the thread has not started yet):
        // write now, as a writer that got in between would.
        *config.program.try_write().unwrap() = program_returning(22);
        hold_meta.revision = 8;
        drop(hold_meta);
        None
    };
    let session = caller.join().unwrap();
    if let Some(writer) = writer {
        writer.join().unwrap();
    }
    let pair = (probe_value(&session.program), session.meta.revision);
    assert!(pair == ("11".to_string(), 7) || pair == ("22".to_string(), 8), "program and revision of different moments: {pair:?}");
    assert_eq!(probe_value(session.base_program.as_ref().unwrap()), pair.0, "the CoW base is the same snapshot");
}
