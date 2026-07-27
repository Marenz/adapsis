//! Parse/typecheck/test coverage for the `.ax.work` modules shipped in the
//! repository root.
//!
//! Before this existed, nothing in `cargo test` ever looked at those files, so a
//! syntax error or a broken `+test` block only surfaced when a live node loaded
//! the module. Issue #38 was three plumbing bugs of exactly that shape.
//!
//! Only pure (non-`[io,async]`) test cases are executed here — IO cases need a
//! coroutine runtime and mocks. Every function is still parsed and validated.

use crate::{ast, eval, parser, validator};

/// Modules shipped at the repository root that must always load.
const BUNDLED: &[&str] = &["TelegramBot.ax.work", "Wolfi.ax.work"];

fn load(path: &str) -> (ast::Program, Vec<parser::Operation>) {
    let source = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{path}: cannot read: {e}"));
    let operations = parser::parse(&source)
        .unwrap_or_else(|e| panic!("{path}: parse failed: {e}"));

    let mut program = ast::Program::default();
    for op in &operations {
        if matches!(
            op,
            parser::Operation::Test(_)
                | parser::Operation::Trace(_)
                | parser::Operation::Eval(_)
                | parser::Operation::Query(_)
                | parser::Operation::Plan(_)
                | parser::Operation::Roadmap(_)
                | parser::Operation::Mock { .. }
                | parser::Operation::Unmock
        ) {
            continue;
        }
        validator::apply_and_validate(&mut program, op)
            .unwrap_or_else(|e| panic!("{path}: validation failed: {e}"));
    }
    (program, operations)
}

/// Collect `+test` blocks from both top level and module bodies.
fn tests_of(operations: &[parser::Operation]) -> Vec<(String, &Vec<parser::TestCase>)> {
    let mut out = Vec::new();
    for op in operations {
        match op {
            parser::Operation::Test(t) => out.push((t.function_name.clone(), &t.cases)),
            parser::Operation::Module(m) => {
                for body_op in &m.body {
                    if let parser::Operation::Test(t) = body_op {
                        // Mirrors Session: already-qualified names are used as-is.
                        let name = if t.function_name.contains('.') {
                            t.function_name.clone()
                        } else {
                            format!("{}.{}", m.name, t.function_name)
                        };
                        out.push((name, &t.cases));
                    }
                }
            }
            _ => {}
        }
    }
    out
}

#[test]
fn bundled_modules_parse_and_validate() {
    for path in BUNDLED {
        let (program, _) = load(path);
        assert!(
            !program.modules.is_empty(),
            "{path}: produced no modules"
        );
    }
}

#[test]
fn bundled_module_pure_tests_pass() {
    let mut executed = 0usize;

    for path in BUNDLED {
        let (program, operations) = load(path);

        for (function_name, cases) in tests_of(&operations) {
            let Some(func) = program.get_function(&function_name) else {
                panic!("{path}: +test references unknown function `{function_name}`");
            };
            // IO/async cases need a coroutine runtime; skip them here.
            if func
                .effects
                .iter()
                .any(|e| matches!(e, ast::Effect::Io | ast::Effect::Async))
            {
                continue;
            }

            for case in cases {
                // `eval_test_case` signals the verdict through Result: Ok is a
                // pass, Err carries the mismatch or the evaluation error.
                if let Err(e) = eval::eval_test_case(&program, &function_name, case) {
                    panic!("{path}: {function_name}: {e}");
                }
                executed += 1;
            }
        }
    }

    assert!(
        executed > 0,
        "no pure test cases executed — the harness stopped covering anything"
    );
}

/// Issue #38 item 2: `Wolfi.remember` wrote to a hardcoded `/home/adapsis/...`
/// while `prompt::persona()` read from `$HOME/...`. Reader and writer were
/// split-brained by construction, so every remembered note was lost.
///
/// This asserts the writer's path and the reader's path are literally the same.
#[test]
fn wolfi_notes_path_matches_the_rust_persona_reader() {
    let (program, _) = load("Wolfi.ax.work");

    let func = program
        .get_function("Wolfi.notes_path")
        .expect("Wolfi.notes_path must exist");
    let mut env = eval::Env::new_with_shared_interner(&program.shared_interner);
    env.populate_shared_from_program(&program);
    let value = eval::eval_function_body_named(&program, "Wolfi.notes_path", &func.body, &mut env)
        .expect("Wolfi.notes_path must evaluate");
    let written = match value {
        eval::Value::String(s) => s.as_ref().clone(),
        other => panic!("Wolfi.notes_path returned {other:?}, expected a String"),
    };

    let read = crate::prompt::persona_notes_path()
        .expect("HOME must be set for this test")
        .to_string_lossy()
        .to_string();

    assert_eq!(
        written, read,
        "Wolfi writes notes to `{written}` but persona() reads them from `{read}` — \
         notes written by the model would be silently lost"
    );
}

/// The other half of issue #38 item 2: a note written to the notes path must
/// actually come back out of the assembled persona.
#[test]
fn persona_recalls_a_written_note() {
    let dir = std::env::temp_dir().join(format!("adapsis-persona-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let notes = dir.join("persona-notes.md");
    std::fs::write(&notes, "- Renate mag grossen Text\n").expect("write notes");

    let persona = crate::prompt::persona_from_paths(None, Some(&notes));
    assert!(
        persona.contains("Renate mag grossen Text"),
        "persona() dropped a written note; got:\n{persona}"
    );

    // Absent notes must not fabricate a notes section.
    let missing = dir.join("does-not-exist.md");
    let bare = crate::prompt::persona_from_paths(None, Some(&missing));
    assert!(
        !bare.contains("Renate mag grossen Text"),
        "persona() leaked notes from a previous read"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
