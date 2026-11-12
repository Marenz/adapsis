//! #146: the result of a call that changed a module reports the real outcome
//! of the source save, a turn does not finish quietly with an unsaved edit, and
//! one line says what is unsaved. Real native path, mock provider, private
//! manifest-managed library. No autosave runs in these fixtures, so every file
//! change observed here was made by the awaited save of the call itself.
use super::*;
use crate::library::{LibraryState, UnsavedKind};

// Both files are in the printer's form, so a save that changes nothing rewrites nothing.
const PY_A: &str = "def value(x: int) -> int:\n    return (x + 1)\n";
const PY_B: &str = "def twice(x: int) -> int:\n    return x\n";
const MANIFEST: &str = r#"{"version":1,"modules":{"PyA":{"format":"python","dependencies":[]},"PyB":{"format":"python","dependencies":[]}}}"#;

fn python(id: &str, module: &str, source: &str) -> Value {
    tool(id, "mutate", json!({"language":"python", "module":module, "source":source}))
}

fn adapsis(id: &str, source: &str) -> Value {
    tool(id, "mutate", json!({"source":source}))
}

fn finish(id: &str) -> Value {
    tool(id, "finish", json!({}))
}

/// 130 chained operators: accepted by the `.ax` path, too deep to print.
fn too_deep(module: &str, function: &str) -> String {
    format!("+module {module}\n+fn {function} (x:Int)->Int\n  +return x{}\n+end\n", " + 1".repeat(130))
}

struct Managed {
    fixture: Fixture,
    dir: tempfile::TempDir,
    state: LibraryState,
}

impl Managed {
    async fn new(outputs: Vec<Value>) -> Self {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("library-sources.json"), MANIFEST).unwrap();
        std::fs::write(dir.path().join("PyA.py"), PY_A).unwrap();
        std::fs::write(dir.path().join("PyB.py"), PY_B).unwrap();
        let fixture = Fixture::new(outputs, "").await;
        let mut program = crate::ast::Program::default();
        let state = crate::library::load_library_from_dir_for_tests(&mut program, dir.path());
        state.require_authoritative_sources().unwrap();
        program.require_modules = true;
        *fixture.config.program.write().await = program;
        fixture.config.meta.lock().unwrap().library_state = Some(state.clone());
        Self { fixture, dir, state }
    }

    fn file(&self, name: &str) -> String {
        std::fs::read_to_string(self.dir.path().join(name)).unwrap()
    }

    /// The tool result the MODEL received for call `id` (from the next request).
    fn tool_result(&self, id: &str) -> String {
        let requests = self.fixture.requests.lock().unwrap();
        requests
            .iter()
            .flat_map(|request| request["messages"].as_array().cloned().unwrap_or_default())
            .find(|message| message["role"] == "tool" && message["tool_call_id"] == id)
            .map(|message| message["content"].as_str().unwrap_or_default().to_string())
            .unwrap_or_else(|| panic!("no tool result for call {id}"))
    }

    fn restart(&self) -> crate::ast::Program {
        let mut program = crate::ast::Program::default();
        crate::library::load_library_from_dir_for_tests(&mut program, self.dir.path()).require_authoritative_sources().unwrap();
        program
    }
}

fn lines(response: &AskResponse) -> Vec<String> {
    response.results.iter().map(|result| result.message.clone()).collect()
}

#[tokio::test]
async fn saved_result_names_the_file_and_the_file_already_holds_the_edit() {
    let m = Managed::new(vec![
        python("edit", "PyA", "def three() -> int:\n    return 3\n"),
        python("new", "PyNew", "def one() -> int:\n    return PyA.three()\n"),
        tool("test", "test", json!({"language":"python","module":"PyA","source":"def test_value() -> None:\n    assert value(1) == 2\n"})),
        finish("done"),
    ])
    .await;
    let response = m.fixture.run(false).await;
    assert_eq!(response.completed, Some(true), "{:?}", lines(&response));
    assert!(!response.has_errors, "{:?}", lines(&response));
    // What the model was told, per call.
    let edit = m.tool_result("edit");
    assert!(edit.contains("module `PyA` changed in memory") && edit.contains("saved to PyA.py"), "{edit}");
    assert!(!edit.contains("pending") && !edit.contains("not yet durable") && !edit.contains("NOT saved"), "{edit}");
    let new = m.tool_result("new");
    assert!(new.contains("saved to PyNew.py (new module, added to the library)"), "{new}");
    // Stored test cases are part of the file: `test` saves too.
    assert!(m.tool_result("test").contains("saved to PyA.py"));
    // No autosave exists here: the files were written by the calls themselves.
    assert!(m.file("PyA.py").contains("def three") && m.file("PyA.py").contains("@stored_test"));
    assert!(m.file("PyNew.py").contains("PyA.three()"));
    assert!(m.file("library-sources.json").contains("PyNew"));
    assert!(m.state.unsaved().is_empty() && !m.state.has_unconfirmed_saves());
    // The reply carries no runtime line when everything is saved.
    assert!(!response.reply.contains("[Runtime:"), "{}", response.reply);
    let restarted = m.restart();
    assert!(restarted.get_function("PyNew.one").is_some() && restarted.get_function("PyA.three").is_some());
}

#[tokio::test]
async fn failing_test_and_refused_mutation_store_nothing_and_run_no_save() {
    let m = Managed::new(vec![
        tool("bad-test", "test", json!({"language":"python","module":"PyA","source":"def test_value() -> None:\n    assert value(1) == 99\n"})),
        python("bad-edit", "PyA", "def broken() -> int:\n    return PyA.missing()\n"),
        finish("done"),
    ])
    .await;
    // A writer that ran would mark and clear `dirty`; hold every flush so a
    // flush attempt would show up as an unconfirmed save.
    let _hold = m.state.hold_flushes_for_tests();
    m.state.set_save_wait_for_tests(std::time::Duration::from_millis(100));
    let response = m.fixture.run(false).await;
    assert_eq!(response.completed, Some(true), "{:?}", lines(&response));
    for id in ["bad-test", "bad-edit"] {
        let result = m.tool_result(id);
        assert!(!result.contains("saved to") && !result.contains("NOT saved"), "{id}: {result}");
    }
    assert!(!m.state.has_unconfirmed_saves() && m.state.unsaved().is_empty());
    assert_eq!((m.file("PyA.py"), m.file("PyB.py")), (PY_A.to_string(), PY_B.to_string()));
}

/// One save per changing call: each result is true for the state after its
/// own call. Reasons covered: nested printed form, waits for another module.
#[tokio::test]
async fn batch_reports_each_call_and_a_repair_saves_what_waited() {
    let fresh = "+module PyA\n+fn fresh (x:Int)->Int\n  +return x\n+end\n";
    let m = Managed::new(vec![
        response("", vec![
            call("deep", "mutate", json!({"source": format!("{fresh}{}", too_deep("PyA", "long"))})),
            call("uses", "mutate", json!({"language":"python","module":"PyB","source":"def twice(x: int) -> int:\n    return PyA.fresh(x)\n"})),
            call("repair", "mutate", json!({"source":"+module PyA\n+fn long (x:Int)->Int\n  +return x\n+end\n"})),
        ]),
        finish("done"),
    ])
    .await;
    let response = m.fixture.run(false).await;
    assert_eq!(response.completed, Some(true), "{:?}", lines(&response));
    let deep = m.tool_result("deep");
    assert!(deep.contains("module `PyA` NOT saved: ") && deep.contains("printed form") && deep.contains("nesting limit of 250"), "{deep}");
    assert!(deep.contains("will be lost at the next restart unless you split the expression into typed local bindings or helper functions"), "{deep}");
    let uses = m.tool_result("uses");
    assert!(uses.contains("module `PyB` NOT saved: ") && uses.contains("unless you save or fix module `PyA` first"), "{uses}");
    assert!(uses.contains("Other modules still not saved: PyA (the printed form of an expression is nested too deeply)"), "{uses}");
    let repair = m.tool_result("repair");
    assert!(repair.contains("saved to PyA.py") && repair.contains("Also saved now: PyB.py"), "{repair}");
    assert!(!repair.contains("NOT saved"), "{repair}");
    // The record followed every step and is empty at the end; finish is quiet.
    assert!(m.state.unsaved().is_empty());
    assert!(!response.reply.contains("[Runtime:"));
    assert!(m.file("PyB.py").contains("PyA.fresh(x)") && m.file("PyA.py").contains("def fresh"));
    m.restart();
}

#[tokio::test]
async fn cycle_result_names_both_modules_and_how_to_break_it() {
    let m = Managed::new(vec![
        python("b", "PyB", "def twice(x: int) -> int:\n    return PyA.value(x)\n"),
        python("a", "PyA", "def back(x: int) -> int:\n    return PyB.twice(x)\n"),
        finish("first"),
        finish("second"),
    ])
    .await;
    let response = m.fixture.run(false).await;
    assert!(m.tool_result("b").contains("saved to PyB.py"));
    let a = m.tool_result("a");
    assert!(a.contains("module `PyA` NOT saved: ") && a.contains("call each other"), "{a}");
    assert!(a.contains("unless you break the cycle between `PyA` and `PyB`"), "{a}");
    assert_eq!(m.state.unsaved()["PyA"].kind, UnsavedKind::Cycle(vec!["PyB".into()]));
    assert_eq!(m.file("PyA.py"), PY_A, "the old source stays");
    // Fixable and changed in this turn: refused once, then accepted with the line.
    let first = m.tool_result("first");
    assert!(first.contains("Cannot accept finish yet: 1 module(s) changed in this turn are NOT saved as source"), "{first}");
    assert!(first.contains("A second finish is accepted as it is"), "{first}");
    assert_eq!(response.completed, Some(true));
    assert!(response.reply.ends_with("[Runtime: not saved to source: PyA (cycle with `PyB`). The change is live only until the next restart.]"), "{}", response.reply);
}

#[tokio::test]
async fn environment_causes_are_reported_without_a_retry_remedy_and_never_refuse_finish() {
    for cause in ["manifest changed", "directory read-only"] {
        use std::os::unix::fs::PermissionsExt;
        let m = Managed::new(vec![python("edit", "PyA", "def three() -> int:\n    return 3\n"), finish("done")]).await;
        match cause {
            "manifest changed" => std::fs::write(m.dir.path().join("library-sources.json"), format!("{MANIFEST} ")).unwrap(),
            _ => std::fs::set_permissions(m.dir.path(), std::fs::Permissions::from_mode(0o500)).unwrap(),
        }
        let response = m.fixture.run(false).await;
        std::fs::set_permissions(m.dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let edit = m.tool_result("edit");
        assert!(edit.contains("module `PyA` NOT saved: "), "{cause}: {edit}");
        assert!(edit.contains("tell the owner. Do not repeat the mutation."), "{cause}: {edit}");
        assert!(!edit.contains("unless you"), "{cause}: an environment cause has no remedy for the model: {edit}");
        let (kind, text) = match cause {
            "manifest changed" => (UnsavedKind::ChangedOnDisk, "changed on disk and nothing was overwritten"),
            _ => (UnsavedKind::WriteFailed, "This is a disk or permission error"),
        };
        assert_eq!(m.state.unsaved()["PyA"].kind, kind, "{cause}");
        assert!(edit.contains(text), "{cause}: {edit}");
        // Not the model's fault: finish is accepted at once, and the person is told.
        assert_eq!(response.completed, Some(true), "{cause}: {:?}", lines(&response));
        assert!(!m.tool_result_exists("second"));
        assert!(response.reply.contains("[Runtime: not saved to source: PyA ("), "{cause}: {}", response.reply);
        assert_eq!(m.file("PyA.py"), PY_A, "{cause}");
    }
}

impl Managed {
    fn tool_result_exists(&self, id: &str) -> bool {
        let requests = self.fixture.requests.lock().unwrap();
        requests.iter().flat_map(|request| request["messages"].as_array().cloned().unwrap_or_default()).any(|message| message["tool_call_id"] == id && message["role"] == "tool")
    }
}

/// The wait for the save happens under the conversation's turn lock only: while
/// the writer is held open (before it takes its own locks), the Program, the
/// runtime and the session metadata are all free.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn no_program_runtime_or_meta_lock_is_held_across_the_save_wait() {
    let m = Managed::new(vec![python("edit", "PyA", "def three() -> int:\n    return 3\n"), finish("done")]).await;
    let release = m.state.hold_flushes_for_tests();
    let config = m.fixture.config.clone();
    let turn = tokio::spawn(async move {
        llm_handlers::ask(State(config), Json(AskRequest { message: "edit".into(), caller: None })).await.0
    });
    // The entry point marks the module and then waits for the writer.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !m.state.has_unconfirmed_saves() {
        assert!(std::time::Instant::now() < deadline, "the save was never requested");
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert!(!turn.is_finished(), "the turn is waiting for the save");
    for _ in 0..20 {
        assert!(m.fixture.config.program.try_write().is_ok(), "Program lock is free");
        assert!(m.fixture.config.runtime.try_write().is_ok(), "runtime lock is free");
        assert!(m.fixture.config.meta.try_lock().is_ok(), "session metadata lock is free");
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert_eq!(m.file("PyA.py"), PY_A, "nothing is written while the writer is held");
    drop(release);
    let response = turn.await.unwrap();
    assert_eq!(response.completed, Some(true), "{:?}", lines(&response));
    assert!(m.tool_result("edit").contains("saved to PyA.py"));
    assert!(m.file("PyA.py").contains("def three"));
}

/// A writer that does not answer within the bound: the result says so, the
/// turn goes on, finish is accepted with the runtime line, and the record is
/// right once the writer has finished.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slow_writer_is_reported_as_not_confirmed_and_the_record_settles_later() {
    let m = Managed::new(vec![python("edit", "PyA", "def three() -> int:\n    return 3\n"), finish("done")]).await;
    m.state.set_save_wait_for_tests(std::time::Duration::from_millis(150));
    let release = m.state.hold_flushes_for_tests();
    let response = m.fixture.run(false).await;
    let edit = m.tool_result("edit");
    assert!(edit.contains("module `PyA` NOT saved: the save was not confirmed within 10 s and continues in the background."), "{edit}");
    assert!(edit.contains("Treat the edit as not saved until `inspect library` shows no unsaved line. Do not repeat the mutation."), "{edit}");
    assert_eq!(response.completed, Some(true), "an unconfirmed save never refuses finish: {:?}", lines(&response));
    assert!(response.reply.ends_with("[Runtime: not saved to source: PyA (save not confirmed yet). The change is live only until the next restart.]"), "{}", response.reply);
    assert_eq!(m.state.unsaved()["PyA"].kind, UnsavedKind::NotConfirmed);
    assert_eq!(m.file("PyA.py"), PY_A);
    // The writer was never cancelled: released, it saves and clears the record.
    drop(release);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !m.state.unsaved().is_empty() {
        assert!(std::time::Instant::now() < deadline, "the record never settled");
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(m.file("PyA.py").contains("def three"));
}

#[tokio::test]
async fn finish_is_refused_once_for_a_fixable_unsaved_edit_of_this_turn() {
    let m = Managed::new(vec![
        adapsis("deep", &too_deep("PyA", "long")),
        finish("first"),
        finish("second"),
    ])
    .await;
    let response = m.fixture.run(true).await;
    let first = m.tool_result("first");
    assert!(first.contains("Cannot accept finish yet") && first.contains("module `PyA` NOT saved") && first.contains("split the expression"), "{first}");
    assert!(first.contains("\"done_accepted\":false"), "{first}");
    // The second finish was accepted (its result ends the turn, so no later
    // request carries it).
    assert_eq!(response.completed, Some(true));
    assert_eq!(m.fixture.requests.lock().unwrap().len(), 3);
    // Streaming: the runtime line is sent as visible text.
    assert!(response.reply.contains("[Runtime: not saved to source: PyA (the printed form of an expression is nested too deeply). The change is live only until the next restart.]"), "{}", response.reply);
}

/// An unsaved module from an earlier turn is not this turn's debt: no refusal,
/// no runtime line in the reply. It is visible in the tail and in inspect.
#[tokio::test]
async fn older_unsaved_module_does_not_refuse_a_later_turn_but_is_shown() {
    let m = Managed::new(vec![
        adapsis("deep", &too_deep("PyA", "long")),
        finish("first"),
        finish("second"),
        // The next turn:
        tool("look", "inspect", json!({"kind":"library"})),
        finish("later"),
    ])
    .await;
    m.fixture.run(false).await;
    let before = m.fixture.requests.lock().unwrap().len();
    let response = m.fixture.run(false).await;
    assert_eq!(response.completed, Some(true), "{:?}", lines(&response));
    assert_eq!(m.fixture.requests.lock().unwrap().len(), before + 2, "inspect, then one accepted finish");
    assert!(!response.reply.contains("[Runtime:"), "{}", response.reply);
    let line = "Source NOT saved (live in memory only, lost at the next restart): PyA (the printed form of an expression is nested too deeply)";
    assert!(m.tool_result("look").contains(line), "{}", m.tool_result("look"));
    let request = m.fixture.requests.lock().unwrap()[before].to_string();
    assert_eq!(request.matches("Source NOT saved").count(), 1, "one line in the runtime tail of the new turn");
}

/// The cache prefix depends on it: with nothing unsaved the whole request is
/// byte-identical to one from a library that never had an unsaved module, and
/// with something unsaved it differs by exactly the one line.
#[tokio::test]
async fn tail_is_byte_identical_when_nothing_is_unsaved_and_one_line_otherwise() {
    async fn first_request(prepare: impl AsyncFnOnce(&Managed)) -> String {
        let m = Managed::new(vec![finish("done")]).await;
        prepare(&m).await;
        m.fixture.run(false).await;
        let request = m.fixture.requests.lock().unwrap()[0].to_string();
        request
    }
    let changed = std::collections::BTreeSet::from(["PyA".to_string()]);
    let clean = first_request(async |_| {}).await;
    // An unsaved entry that a later flush cleared leaves no trace.
    let cleared = first_request(async |m: &Managed| {
        let manifest = m.dir.path().join("library-sources.json");
        std::fs::write(&manifest, format!("{MANIFEST} ")).unwrap();
        assert!(m.fixture.config.save_modules(&m.state, changed.clone()).await.iter().any(|(line, saved)| !saved && line.contains("NOT saved")));
        assert!(m.state.unsaved().contains_key("PyA"));
        std::fs::write(&manifest, MANIFEST).unwrap();
        assert!(m.fixture.config.save_modules(&m.state, changed.clone()).await.iter().any(|(line, saved)| *saved && line == "saved to PyA.py"));
        assert!(m.state.unsaved().is_empty());
    })
    .await;
    assert_eq!(clean, cleared);
    let unsaved = first_request(async |m: &Managed| {
        std::fs::write(m.dir.path().join("library-sources.json"), format!("{MANIFEST} ")).unwrap();
        m.fixture.config.save_modules(&m.state, changed.clone()).await;
    })
    .await;
    assert_ne!(clean, unsaved);
    let line = "Source NOT saved (live in memory only, lost at the next restart): PyA (the file or the manifest changed on disk)\\n";
    assert_eq!(unsaved.matches(line).count(), 1, "{unsaved}");
    assert_eq!(unsaved.replacen(line, "", 1), clean, "the requests differ by exactly that line");
}

#[test]
fn tail_line_is_for_writers_only_and_names_only_readable_modules() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("library-sources.json"), MANIFEST).unwrap();
    std::fs::write(dir.path().join("PyA.py"), PY_A).unwrap();
    std::fs::write(dir.path().join("PyB.py"), PY_B).unwrap();
    let mut program = crate::ast::Program::default();
    let state = crate::library::load_library_from_dir_for_tests(&mut program, dir.path());
    let both = std::collections::BTreeSet::from(["PyA".to_string(), "PyB".to_string()]);
    // Both modules unsaved (policy failure after an external manifest change).
    std::fs::write(dir.path().join("library-sources.json"), format!("{MANIFEST} ")).unwrap();
    state.mark_dirty(&both);
    let live = std::sync::Arc::new(tokio::sync::RwLock::new(program));
    tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(crate::library::flush_live_sources(live, Some(state.clone()))).unwrap();
    assert_eq!(state.unsaved().len(), 2);
    let permissions: crate::permissions::PermissionConfig = toml::from_str(
        r#"
[groups]
hidden = ["PyB"]
[model.writer]
hidden = "write"
user = "write"
[model.partial]
hidden = "execute"
user = "write"
[model.reader]
hidden = "read"
user = "read"
"#,
    )
    .unwrap();
    let tail = |model: &str| super::super::source_save::unsaved_tail(Some(&state), &permissions, crate::permissions::AccessLevel::Full, model);
    assert!(tail("writer").contains("PyA (") && tail("writer").contains("PyB ("), "{}", tail("writer"));
    // May write `user` modules, may only execute PyB: PyB is not named.
    assert!(tail("partial").contains("PyA (") && !tail("partial").contains("PyB"), "{}", tail("partial"));
    // Cannot write anything: no line at all.
    assert_eq!(tail("reader"), "");
    // No manifest: no line.
    assert_eq!(super::super::source_save::unsaved_tail(Some(&LibraryState::new()), &permissions, crate::permissions::AccessLevel::Full, "writer"), "");
}

/// The record is the single source and does not go stale.
#[tokio::test]
async fn unsaved_record_is_cleared_by_a_saving_flush_by_removal_and_by_a_restart() {
    let m = Managed::new(vec![]).await;
    let save = |names: &[&str]| {
        let names: std::collections::BTreeSet<String> = names.iter().map(|name| name.to_string()).collect();
        let config = m.fixture.config.clone();
        let state = m.state.clone();
        async move { config.save_modules(&state, names).await }
    };
    let apply = |source: String| {
        let config = m.fixture.config.clone();
        async move {
            let mut program = config.program.write().await;
            for operation in crate::parser::parse(&source).unwrap() {
                crate::validator::apply_and_validate(&mut program, &operation).unwrap();
            }
        }
    };
    // 1. Cleared when a later flush saves the module.
    apply(too_deep("PyA", "long")).await;
    assert!(save(&["PyA"]).await[0].0.contains("NOT saved"));
    assert_eq!(m.state.unsaved()["PyA"].kind, UnsavedKind::Nesting);
    apply("+module PyA\n+fn long (x:Int)->Int\n  +return x\n+end\n".into()).await;
    assert_eq!(save(&["PyA"]).await[0], ("saved to PyA.py".to_string(), true));
    assert!(m.state.unsaved().is_empty());
    // 2. Cleared when the module is removed (an unlisted module that could not
    //    be enrolled; a listed module cannot be removed, #126).
    apply(too_deep("PyGone", "long")).await;
    assert!(save(&["PyGone"]).await[0].0.contains("module `PyGone` NOT saved"));
    assert!(m.state.unsaved().contains_key("PyGone"));
    m.fixture.config.program.write().await.modules.retain(|module| module.name != "PyGone");
    save(&[]).await;
    assert!(m.state.unsaved().is_empty(), "{:?}", m.state.unsaved());
    // 3. A restart starts with an empty record: the start loads the source,
    //    adopts a cached module, or drops it; nothing is "unsaved" afterwards.
    apply(too_deep("PyA", "deeper")).await;
    save(&["PyA"]).await;
    assert!(!m.state.unsaved().is_empty());
    let mut cached = m.fixture.config.program.read().await.clone();
    let restarted = crate::library::load_library_from_dir_for_tests(&mut cached, m.dir.path());
    restarted.require_authoritative_sources().unwrap();
    assert!(restarted.unsaved().is_empty() && !restarted.has_unconfirmed_saves());
    assert!(cached.get_function("PyA.deeper").is_none(), "the start loaded the saved source");
}

/// `/api/mutate`: the same outcome line, and an empty mutation runs no writer.
#[tokio::test]
async fn api_mutate_reports_the_save_and_an_empty_mutation_runs_no_writer() {
    let m = Managed::new(vec![]).await;
    let mutate = |source: &str| {
        let config = m.fixture.config.clone();
        let source = source.to_string();
        async move { crate::api::mutate(State(config), Json(crate::api::MutateRequest { source })).await.0 }
    };
    let answer = mutate("+module PyA\n+fn three ()->Int\n  +return 3\n+end\n").await;
    let messages: Vec<&str> = answer.results.iter().map(|result| result.message.as_str()).collect();
    assert!(messages.contains(&"module `PyA` changed in memory") && messages.contains(&"saved to PyA.py"), "{messages:?}");
    assert!(answer.results.iter().all(|result| result.success));
    assert!(m.file("PyA.py").contains("def three"), "written when the response exists");
    // A member the printer refuses (a bare call that binds to another module).
    let answer = mutate("+module PyB\n+fn bare (x:Int)->Int\n  +return value(x)\n+end\n").await;
    let text = answer.results.iter().map(|r| r.message.clone()).collect::<Vec<_>>().join("\n");
    assert!(text.contains("module `PyB` NOT saved: ") && text.contains("cannot be completely serialized as Python"), "{text}");
    assert!(text.contains("unless you change that member so it can be written as Python"), "{text}");
    assert_eq!(m.state.unsaved()["PyB"].kind, UnsavedKind::Unprintable);
    assert_eq!(m.file("PyB.py"), PY_B);
    let listing = crate::library::query_library(&*m.fixture.config.program.read().await, Some(&m.state));
    assert!(listing.contains("Source NOT saved (live in memory only, lost at the next restart): PyB (a member cannot be written as Python)"), "{listing}");
    let answer = mutate("!remove PyB.bare").await;
    assert!(answer.results.iter().any(|result| result.message == "saved to PyB.py"), "{:?}", answer.results.iter().map(|r| &r.message).collect::<Vec<_>>());
    let listing = crate::library::query_library(&*m.fixture.config.program.read().await, Some(&m.state));
    assert!(!listing.contains("Source NOT saved") && !listing.contains("pending"), "{listing}");
    let answer = mutate(&too_deep("PyA", "long")).await;
    assert!(answer.results.iter().any(|result| !result.success && result.message.contains("module `PyA` NOT saved")), "{:?}", answer.results.iter().map(|r| &r.message).collect::<Vec<_>>());
    // The preflight's save request: no result, no revision, and no writer run
    // (a writer run would hang on the held gate and mark the module).
    let _hold = m.state.hold_flushes_for_tests();
    let revision = m.fixture.config.meta.lock().unwrap().revision;
    let answer = tokio::time::timeout(std::time::Duration::from_secs(5), mutate("")).await.expect("an empty mutation does not wait for a writer");
    assert!(answer.results.is_empty() && answer.revision == revision);
    assert!(!m.state.has_unconfirmed_saves());
}

/// Legacy `<code>` turns: the block's results carry the save line, and `!done`
/// goes through the same gate.
#[tokio::test]
async fn legacy_code_blocks_report_the_save_and_done_uses_the_same_gate() {
    let code = |text: &str| response(&format!("<code>\n{text}\n</code>"), vec![]);
    let m = Managed::new(vec![
        code("+module PyB\n+fn three ()->Int\n  +return 3\n+end"),
        code(&too_deep("PyA", "long")),
        code("!done"),
        code("!done"),
    ])
    .await;
    let mut config = m.fixture.config.clone();
    config.native_tools = false;
    let response = llm_handlers::ask(State(config), Json(AskRequest { message: "edit".into(), caller: None })).await.0;
    let all = lines(&response).join("\n");
    assert!(all.contains("saved to PyB.py"), "{all}");
    assert!(all.contains("module `PyA` NOT saved: ") && all.contains("split the expression"), "{all}");
    assert_eq!(all.matches("Cannot accept finish yet").count(), 1, "refused once: {all}");
    assert!(response.reply.contains("[Runtime: not saved to source: PyA ("), "{}", response.reply);
    assert!(m.file("PyB.py").contains("def three"));
    // What the model was told after each block is in the next request.
    let requests = m.fixture.requests.lock().unwrap();
    assert!(requests[1].to_string().contains("saved to PyB.py"));
    assert!(requests[3].to_string().contains("Cannot accept finish yet"));
}

/// A speaker who changed nothing in this turn (a guest cannot mutate at all) is
/// never refused and gets no runtime line, whatever is unsaved.
#[tokio::test]
async fn done_without_changes_of_this_turn_is_accepted_quietly() {
    let m = Managed::new(vec![]).await;
    {
        let mut program = m.fixture.config.program.write().await;
        for operation in crate::parser::parse(&too_deep("PyA", "long")).unwrap() {
            crate::validator::apply_and_validate(&mut program, &operation).unwrap();
        }
    }
    m.fixture.config.save_modules(&m.state, std::collections::BTreeSet::from(["PyA".to_string()])).await;
    assert!(crate::library::fixable(&m.state.unsaved(), "PyA"));
    let baseline = m.fixture.config.program.read().await.clone();
    let mut ws = m.fixture.config.snapshot_working_set().await;
    let result = execute::execute_code_with_baseline("!done", &m.fixture.config, &mut ws, None, Some(&baseline)).await;
    assert!(result.done_accepted && !result.has_errors && result.unsaved_notice.is_none(), "{:?}", result.mutation_results.iter().map(|r| &r.message).collect::<Vec<_>>());
}

/// A mutation builtin inside an evaluation edits a copy that is merged after
/// the evaluation; the entry point reports the save.
#[tokio::test]
async fn evaluation_that_mutates_through_a_builtin_reports_the_save() {
    let m = Managed::new(vec![]).await;
    let (sender, _receiver) = tokio::sync::mpsc::channel::<crate::coroutine::IoRequest>(8);
    let mut config = m.fixture.config.clone();
    config.io_sender = Some(sender);
    let mut ws = config.snapshot_working_set().await;
    let code = r#"!eval mutate("+module PyB\n+fn four ()->Int\n  +return 4\n+end")"#;
    let result = execute::execute_code(code, &config, &mut ws, None).await;
    let all: Vec<&str> = result.mutation_results.iter().map(|r| r.message.as_str()).collect();
    assert!(!result.has_errors, "{all:?}");
    assert!(all.contains(&"saved to PyB.py"), "{all:?}");
    assert!(m.file("PyB.py").contains("def four"));
}

/// A route handler has nobody to tell: its save is started at once (not left
/// to the debounce) and not awaited by the response.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn route_handler_mutation_starts_the_save_without_waiting_for_it() {
    use tower::ServiceExt as _;
    let m = Managed::new(vec![]).await;
    let (sender, _receiver) = tokio::sync::mpsc::channel::<crate::coroutine::IoRequest>(8);
    let mut config = m.fixture.config.clone();
    config.io_sender = Some(sender);
    {
        let mut program = config.program.write().await;
        // The handler lives in the scratch module, which is never source.
        let handler = "+module RunToolScratch\n+fn handle (body:String)->String [io,async,mut]\n  +await done:String = mutate(\"+module PyB\\n+fn five ()->Int\\n  +return 5\\n+end\")\n  +return done\n+end\n";
        for operation in crate::parser::parse(handler).unwrap() {
            crate::validator::apply_and_validate(&mut program, &operation).unwrap();
        }
    }
    config.runtime.write().unwrap().http_routes.push(crate::ast::HttpRoute { method: "POST".into(), path: "/edit".into(), handler_fn: "RunToolScratch.handle".into() });
    // The writer is held: the response must arrive anyway.
    let release = m.state.hold_flushes_for_tests();
    let response = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        router_with_llm(config.clone()).oneshot(axum::http::Request::builder().method("POST").uri("/edit").body(axum::body::Body::from("x")).unwrap()),
    )
    .await
    .expect("the response does not wait for the save")
    .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    assert!(m.state.has_unconfirmed_saves(), "the save was started");
    assert!(!m.file("PyB.py").contains("def five"));
    drop(release);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !m.file("PyB.py").contains("def five") {
        assert!(std::time::Instant::now() < deadline, "the started save never wrote the file: {:?}", m.state.unsaved());
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(m.state.unsaved().is_empty());
}

/// Under a manifest a native Adapsis-source `mutate` cannot create a module
/// (#128): refused before any state change, no writer run, nothing on disk.
#[tokio::test]
async fn native_adapsis_mutate_cannot_create_a_module_and_runs_no_save() {
    let m = Managed::new(vec![adapsis("fresh", "+module PyAxFresh\n+fn one ()->Int\n  +return 1\n+end\n"), finish("done")]).await;
    let before = serde_json::to_string(&*m.fixture.config.program.read().await).unwrap();
    let revision = m.fixture.config.meta.lock().unwrap().revision;
    let _hold = m.state.hold_flushes_for_tests();
    let response = m.fixture.run(false).await;
    let fresh = m.tool_result("fresh");
    assert!(fresh.contains("`PyAxFresh` was not created") && fresh.contains("this library stores Python source; create the module with a Python mutation"), "{fresh}");
    assert!(!fresh.contains("saved to") && !fresh.contains("NOT saved"), "{fresh}");
    assert_eq!(response.completed, Some(true));
    assert_eq!(serde_json::to_string(&*m.fixture.config.program.read().await).unwrap(), before);
    assert_eq!(m.fixture.config.meta.lock().unwrap().revision, revision);
    assert!(!m.state.has_unconfirmed_saves() && m.state.unsaved().is_empty());
    assert_eq!(std::fs::read_dir(m.dir.path()).unwrap().count(), 3);
}

// ── Regressions from the release review of 1475885 (probes P1 to P7) ────────

fn apply_live(program: &mut crate::ast::Program, source: &str) {
    for operation in crate::parser::parse(source).unwrap() {
        crate::validator::apply_and_validate(program, &operation).unwrap();
    }
}

/// P1: "changed in this turn" means changed by this turn's OWN calls. A guest
/// whose turn overlaps another conversation's unsaveable edit is not refused,
/// is not told, and is never shown a module it cannot read.
#[tokio::test]
async fn another_conversations_unsaved_edit_neither_refuses_nor_reaches_a_guest() {
    let m = Managed::new(vec![]).await;
    let mut guest = m.fixture.config.clone().for_new_turn();
    guest.native_tools = false;
    guest.llm_model = std::sync::Arc::new(std::sync::RwLock::new("guest".to_string()));
    guest.permission_config = std::sync::Arc::new(toml::from_str("[model.guest]\nuser = \"none\"\n").unwrap());
    guest.turn = Some(crate::coroutine::TurnIdentity {
        context: "telegram:user:42".into(),
        principal: "telegram:user:42".into(),
        may_write: false,
        guest_tools: Some(vec![]),
    });
    assert_eq!(guest.unsaved_tail("guest"), "");
    let baseline = guest.program.read().await.clone();
    // Meanwhile the admin, in another conversation (its own turn), makes PyA unsaveable.
    let admin = m.fixture.config.clone().for_new_turn();
    apply_live(&mut *admin.program.write().await, &too_deep("PyA", "long"));
    admin.save_modules(&m.state, std::collections::BTreeSet::from(["PyA".to_string()])).await;
    assert!(crate::library::fixable(&m.state.unsaved(), "PyA"));
    for _ in 0..2 {
        let mut ws = guest.snapshot_working_set().await;
        let result = execute::execute_code_with_baseline("!done", &guest, &mut ws, None, Some(&baseline)).await;
        let text: Vec<&str> = result.mutation_results.iter().map(|r| r.message.as_str()).collect();
        assert!(result.done_accepted && !result.has_errors, "{text:?}");
        assert!(result.unsaved_notice.is_none(), "{:?}", result.unsaved_notice);
        assert!(text.iter().all(|line| !line.contains("PyA")), "{text:?}");
    }
    // The same holds for a writer in another conversation: not its modules.
    let other = m.fixture.config.clone().for_new_turn();
    let baseline = other.program.read().await.clone();
    let mut ws = other.snapshot_working_set().await;
    let result = execute::execute_code_with_baseline("!done", &other, &mut ws, None, Some(&baseline)).await;
    assert!(result.done_accepted && result.unsaved_notice.is_none());
}

/// A speaker who cannot write is never refused, and a module it cannot read is
/// never named, even when its own call changed that module (a function it may
/// execute mutates through a builtin).
#[tokio::test]
async fn execute_only_speaker_is_never_refused_and_never_shown_an_unreadable_module() {
    let m = Managed::new(vec![]).await;
    let (sender, _receiver) = tokio::sync::mpsc::channel::<crate::coroutine::IoRequest>(8);
    let mut config = m.fixture.config.clone().for_new_turn();
    config.io_sender = Some(sender);
    config.llm_model = std::sync::Arc::new(std::sync::RwLock::new("runner".to_string()));
    config.permission_config = std::sync::Arc::new(toml::from_str("[model.runner]\nuser = \"execute\"\n").unwrap());
    let baseline = config.program.read().await.clone();
    let mut ws = config.snapshot_working_set().await;
    let deep = too_deep("PyA", "long").replace('\n', "\\n");
    let result = execute::execute_code_with_baseline(&format!("!eval mutate(\"{deep}\")"), &config, &mut ws, None, Some(&baseline)).await;
    let text = result.mutation_results.iter().map(|r| r.message.clone()).collect::<Vec<_>>().join("\n");
    assert!(m.state.unsaved().contains_key("PyA"), "the call changed PyA and it could not be saved: {text}");
    assert!(text.contains("a module this call changed was NOT saved as source (not named: no read permission)"), "{text}");
    assert!(!text.contains("module `PyA` NOT saved") && !text.contains("split the expression"), "{text}");
    let mut ws = config.snapshot_working_set().await;
    let result = execute::execute_code_with_baseline("!done", &config, &mut ws, None, Some(&baseline)).await;
    assert!(result.done_accepted && !result.has_errors, "never refused");
    let notice = result.unsaved_notice.expect("the person is told that something is not saved");
    assert_eq!(notice, "[Runtime: not saved to source: 1 module(s) changed in this turn. The change is live only until the next restart.]");
}

/// P2 and P7: an I/O failure while enrolling a module or recording a new
/// dependency is an environment cause for exactly the affected modules. The
/// model is not told to change correct code, and finish is not refused.
#[tokio::test]
async fn disk_error_at_enrolment_or_dependency_update_is_an_environment_cause_per_module() {
    use std::os::unix::fs::PermissionsExt;
    for (case, module, source) in [
        ("new module", "PyNew", "def one() -> int:\n    return 1\n"),
        ("new dependency", "PyB", "def twice(x: int) -> int:\n    return PyA.value(x)\n"),
    ] {
        let m = Managed::new(vec![python("edit", module, source), finish("first"), finish("second")]).await;
        std::fs::set_permissions(m.dir.path(), std::fs::Permissions::from_mode(0o500)).unwrap();
        let response = m.fixture.run(false).await;
        std::fs::set_permissions(m.dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let edit = m.tool_result("edit");
        assert!(edit.contains(&format!("module `{module}` NOT saved: ")), "{case}: {edit}");
        assert!(edit.contains("This is a disk or permission error: tell the owner. Do not repeat the mutation."), "{case}: {edit}");
        assert!(!edit.contains("unless you") && !edit.contains("cannot be completely serialized") && !edit.contains("saved to"), "{case}: {edit}");
        assert_eq!(m.state.unsaved()[module].kind, UnsavedKind::WriteFailed, "{case}");
        assert!(!crate::library::fixable(&m.state.unsaved(), module), "{case}");
        // Accepted at the first finish, and the person is told.
        assert_eq!(response.completed, Some(true), "{case}");
        assert!(!m.tool_result_exists("first"), "{case}: the first finish ended the turn");
        assert!(response.reply.contains(&format!("[Runtime: not saved to source: {module} (disk or permission error)")), "{case}: {}", response.reply);
        assert_eq!(m.file("library-sources.json"), MANIFEST, "{case}");
        let listing = crate::library::query_library(&*m.fixture.config.program.read().await, Some(&m.state));
        assert!(listing.contains(&format!("Source NOT saved (live in memory only, lost at the next restart): {module} (disk or permission error)")), "{case}: {listing}");
        // Once the directory is writable again, the next save makes it durable.
        let lines = m.fixture.config.save_modules(&m.state, std::collections::BTreeSet::from([module.to_string()])).await;
        assert!(lines.iter().any(|(line, saved)| *saved && line.starts_with(&format!("saved to {module}.py"))), "{case}: {lines:?}");
        assert!(m.state.unsaved().is_empty(), "{case}");
    }
}

/// P3: nothing writes a snapshot taken before the save wait back after it.
/// What other conversations write during the wait survives.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn state_written_by_others_during_the_save_wait_is_not_reverted() {
    for native in [true, false] {
        let outputs = if native {
            vec![python("edit", "PyA", "def three() -> int:\n    return 3\n"), finish("done")]
        } else {
            vec![response("<code>\n+module PyA\n+fn three ()->Int\n  +return 3\n+end\n!done\n</code>", vec![])]
        };
        let m = Managed::new(outputs).await;
        m.fixture.config.runtime.write().unwrap().shared_vars.insert("TelegramBot.poll_offset".into(), crate::eval::Value::Int(100));
        let release = m.state.hold_flushes_for_tests();
        let mut config = m.fixture.config.clone();
        config.native_tools = native;
        let turn = tokio::spawn(async move {
            llm_handlers::ask(State(config), Json(AskRequest { message: "edit".into(), caller: None })).await.0
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !m.state.has_unconfirmed_saves() {
            assert!(std::time::Instant::now() < deadline);
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        // The poll loop advances its cursor; another chat stores a user message.
        m.fixture.config.runtime.write().unwrap().shared_vars.insert("TelegramBot.poll_offset".into(), crate::eval::Value::Int(101));
        m.fixture.config.meta.lock().unwrap().conversations.get_or_create("telegram:other").push_user("hello from another chat");
        drop(release);
        let response = turn.await.unwrap();
        assert_eq!(response.completed, if native { Some(true) } else { None }, "native={native}: {:?}", lines(&response));
        assert!(m.file("PyA.py").contains("def three"), "native={native}");
        let offset = m.fixture.config.runtime.read().unwrap().shared_vars.get("TelegramBot.poll_offset");
        assert!(matches!(offset, Some(crate::eval::Value::Int(101))), "native={native}: the cursor was reverted: {offset:?}");
        let other = m.fixture.config.meta.lock().unwrap().conversations.get("telegram:other").map(|c| c.messages.len());
        assert_eq!(other, Some(1), "native={native}: the other chat's message was reverted");
        // The turn's own state is there too.
        assert!(m.fixture.config.program.read().await.get_function("PyA.three").is_some());
    }
}

/// P4: the wait bound holds while a Program writer is queued behind the
/// writer's read guard; and a writer that FAILED is not said to continue.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn save_wait_bound_holds_with_a_queued_program_writer() {
    let m = Managed::new(vec![]).await;
    m.state.set_save_wait_for_tests(std::time::Duration::from_millis(100));
    let release = m.state.hold_flushes_for_tests();
    let program = m.fixture.config.program.clone();
    let held = program.clone().read_owned().await; // stand-in for the slow writer's read guard
    let queued = program.clone();
    let writer = tokio::spawn(async move {
        let _guard = queued.write().await;
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let changed = std::collections::BTreeSet::from(["PyA".to_string()]);
    let started = std::time::Instant::now();
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(3), m.fixture.config.save_modules(&m.state, changed)).await;
    let lines = outcome.expect("the bounded wait returned although a Program writer is queued");
    assert!(started.elapsed() < std::time::Duration::from_secs(1), "{:?}", started.elapsed());
    assert!(lines[0].0.contains("was not confirmed within 10 s and continues in the background"), "{lines:?}");
    drop(held);
    drop(release);
    writer.await.unwrap();
}

#[test]
fn a_failed_writer_is_not_reported_as_continuing() {
    let line = crate::library::Unsaved::writer_failed().result_line("PyA");
    assert!(line.contains("the source writer failed before it could save anything") && line.contains("Tell the owner. Do not repeat the mutation."), "{line}");
    assert!(!line.contains("continues in the background") && !line.contains("unless you"), "{line}");
}

/// P5: one refusal per turn, whatever else the blocks contain.
#[tokio::test]
async fn legacy_done_with_a_failing_operation_does_not_bring_the_refusal_back() {
    let code = |text: &str| response(&format!("<code>\n{text}\n</code>"), vec![]);
    let m = Managed::new(vec![
        code(&too_deep("PyA", "long")),
        code("!done"),
        code("!eval Nope.missing()\n!done"),
        code("!done"),
        code("!done"),
    ])
    .await;
    let mut config = m.fixture.config.clone();
    config.native_tools = false;
    config.max_iterations = 5;
    let response = llm_handlers::ask(State(config), Json(AskRequest { message: "edit".into(), caller: None })).await.0;
    let all = lines(&response).join("\n");
    assert_eq!(all.matches("Cannot accept finish yet").count(), 1, "{all}");
    assert_eq!(m.fixture.requests.lock().unwrap().len(), 4, "refused once, a block with an error, then accepted");
    assert!(response.reply.contains("[Runtime: not saved to source: PyA ("), "{}", response.reply);
}

/// P6: an edit and `!done` in ONE legacy block: the block's result says saved.
#[tokio::test]
async fn legacy_block_with_edit_and_done_reports_the_save() {
    let m = Managed::new(vec![response("<code>\n+module PyB\n+fn three ()->Int\n  +return 3\n+end\n!done\n</code>", vec![])]).await;
    let mut config = m.fixture.config.clone();
    config.native_tools = false;
    let response = llm_handlers::ask(State(config), Json(AskRequest { message: "edit".into(), caller: None })).await.0;
    assert!(lines(&response).contains(&"saved to PyB.py".to_string()), "{:?}", lines(&response));
    assert!(!response.has_errors && !response.reply.contains("[Runtime:"));
    assert!(m.file("PyB.py").contains("def three"));
    assert_eq!(m.fixture.requests.lock().unwrap().len(), 1, "the block finished the turn");
}

/// "Once per turn" is a real per-turn value: every turn has its own, so a
/// second turn with its own unsaveable edit is refused once again, and a
/// finish in one conversation does not wait for another one's unconfirmed save.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn each_turn_has_its_own_refusal_and_does_not_wait_for_another_turns_save() {
    let m = Managed::new(vec![
        adapsis("one", &too_deep("PyA", "long")),
        finish("first"),
        finish("second"),
        adapsis("two", &too_deep("PyB", "longer")),
        finish("third"),
        finish("fourth"),
    ])
    .await;
    m.fixture.run(false).await;
    let response = m.fixture.run(false).await;
    assert!(m.tool_result("first").contains("Cannot accept finish yet"));
    let third = m.tool_result("third");
    assert!(third.contains("Cannot accept finish yet: 1 module(s)") && third.contains("module `PyB` NOT saved"), "{third}");
    assert!(!third.contains("module `PyA` NOT saved"), "the earlier turn's module is not this turn's debt: {third}");
    assert_eq!(response.completed, Some(true));
    assert!(response.reply.ends_with("[Runtime: not saved to source: PyB (the printed form of an expression is nested too deeply). The change is live only until the next restart.]"), "{}", response.reply);
    // Another conversation's save is unconfirmed (its writer is held): a
    // finish here returns at once.
    let other = m.fixture.config.clone().for_new_turn();
    let _hold = m.state.hold_flushes_for_tests();
    m.state.mark_dirty(&std::collections::BTreeSet::from(["PyA".to_string()]));
    let baseline = other.program.read().await.clone();
    let mut ws = other.snapshot_working_set().await;
    let result = tokio::time::timeout(std::time::Duration::from_secs(2), execute::execute_code_with_baseline("!done", &other, &mut ws, None, Some(&baseline)))
        .await
        .expect("finish did not wait for another conversation's save");
    assert!(result.done_accepted && result.unsaved_notice.is_none());
}

/// The "committed" mark exists only together with the commit (one function,
/// private field), is consumed by exactly one write-back, and a call that did
/// not wait for a save leaves no mark behind.
#[tokio::test]
async fn committed_mark_is_set_only_by_the_commit_and_skips_exactly_one_write_back() {
    let m = Managed::new(vec![]).await;
    let config = m.fixture.config.clone().for_new_turn();
    let mut ws = config.snapshot_working_set().await;
    assert!(!ws.take_committed(), "a fresh working set is not committed");
    ws.meta.plan.push(crate::session::PlanStep { description: "from the working set".into(), status: crate::session::PlanStatus::Pending });
    config.commit_before_wait(&ws).await;
    assert_eq!(config.meta.lock().unwrap().plan.len(), 1, "the commit happened before the mark");
    // Someone else writes during the wait.
    config.meta.lock().unwrap().plan.clear();
    config.write_back_working_set(&ws).await;
    assert!(config.meta.lock().unwrap().plan.is_empty(), "the marked write-back was skipped");
    config.write_back_working_set(&ws).await;
    assert_eq!(config.meta.lock().unwrap().plan.len(), 1, "the mark is consumed: the next write-back is an ordinary one");
    // An executor call that changes no module waits for nothing and marks nothing.
    let baseline = config.program.read().await.clone();
    let mut ws = config.snapshot_working_set().await;
    execute::execute_code_with_baseline("?symbols", &config, &mut ws, None, Some(&baseline)).await;
    assert!(!ws.take_committed());
    // A mark left over from an earlier call does not survive into the next one.
    config.commit_before_wait(&ws).await;
    execute::execute_code_with_baseline("?symbols", &config, &mut ws, None, Some(&baseline)).await;
    assert!(!ws.take_committed(), "the executor clears a stale mark at its start");
}
