use crate::{ast, eval, validator};
use std::sync::Arc;
mod negatives;
mod support;
use support::*;

const MODULES: &[(&str, &str)] = &[
    ("TaskDomain", "domain.py"),
    ("TaskRecordCodec", "codec.py"),
    ("TaskDurable", "durable.py"),
];

fn install(program: &mut ast::Program, name: &str, source: &str) {
    let lowered = super::lower_module_with_program(source, name, program).unwrap();
    for operation in &lowered.operations {
        validator::apply_and_validate(program, operation).unwrap();
    }
    program.rebuild_function_index();
    super::validate_runtime_arguments(&lowered.operations, program).unwrap();
}

fn program() -> Arc<ast::Program> {
    let mut program = ast::Program::default();
    for (name, file) in MODULES {
        let source = std::fs::read_to_string(format!(
            "{}/examples/tasks/{file}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        install(&mut program, name, &source);
    }
    Arc::new(program)
}

#[test]
fn task_durable_exact_roundtrip_and_builtin_signatures() {
    let program = program();
    let mut rebuilt = ast::Program::default();
    for (name, _) in MODULES {
        let printed = super::print_module(&program, name).unwrap();
        install(&mut rebuilt, name, &printed);
        assert_eq!(super::print_module(&rebuilt, name).unwrap(), printed);
    }
    assert!(super::ast_equality::program(&program, &rebuilt));
    for call in [
        "records_read(1)",
        "records_commit('only-one')",
        "records_receipt('one','two')",
    ] {
        let source = format!(
            "@effects('io')\nasync def f() -> str:\n    r: Result[str] = await {call}\n    return 'unused'\n"
        );
        assert!(super::lower_module(&source, "Bad").is_err());
    }
    assert!(super::lower_module("@effects('io')\nasync def a() -> Result[str]:\n    r: Result[str] = await records_read('{}')\n    return r\n@effects('io')\nasync def b() -> str:\n    r: Result[str] = await a()\n    return 'no'\n", "Bad").is_err());
}

#[test]
fn task_durable_external_union_local_printer_blocker_is_characterized() {
    let mut program = ast::Program::default();
    install(
        &mut program,
        "ChoiceOwner",
        "@union\nclass Selection:\n    EmptySelection: tuple[()]\n    NamedSelection: tuple[str]\n",
    );
    install(
        &mut program,
        "ChoiceUser",
        "def example() -> str:\n    choice: Selection = EmptySelection\n    return 'ok'\n",
    );
    assert!(super::print_function(&program, "ChoiceUser.example").is_ok());
    let error = super::print_module(&program, "ChoiceUser").unwrap_err();
    assert!(
        error
            .message
            .contains("Python roundtrip changed the module")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn task_durable_real_concurrent_claims_are_atomic() {
    let directory = tempfile::tempdir().unwrap();
    let harness = Harness::open(&directory.path().join("records.lbug"));
    let program = program();
    initialize(program.clone(), &harness, "a", "run-a", "first").await;
    initialize(program.clone(), &harness, "b", "run-b", "second").await;
    harness.gate_next_two_reads();
    let (a, b) = tokio::join!(
        invoke(
            program.clone(),
            harness.handle(),
            "TaskDurable.act",
            &["a", "run-a", "shared", "acquire", "", "claim-a"]
        ),
        invoke(
            program.clone(),
            harness.handle(),
            "TaskDurable.act",
            &["b", "run-b", "shared", "acquire", "", "claim-b"]
        ),
    );
    assert_eq!(
        [&a.0, &b.0]
            .iter()
            .filter(|name| name.as_str() == "DurableCommitted")
            .count(),
        1
    );
    assert_eq!(
        [&a.0, &b.0]
            .iter()
            .filter(|name| name.as_str() == "DurableConflict")
            .count(),
        1
    );
    let (winner, loser) = if a.0 == "DurableCommitted" {
        ("a", "b")
    } else {
        ("b", "a")
    };
    assert_eq!(
        field(&row(&harness, &format!("task:{loser}")), "state"),
        "ready"
    );
    assert_eq!(
        field(&row(&harness, &format!("run:run-{loser}")), "state"),
        "pending"
    );
    assert_eq!(field(&row(&harness, "claim:shared"), "task"), winner);
    let snapshot = row(&harness, "claim:shared");
    let stale = invoke(
        program.clone(),
        harness.handle(),
        "TaskDurable.act",
        &[
            loser,
            &format!("run-{loser}"),
            "shared",
            "checkpoint",
            "",
            "wrong-release",
        ],
    )
    .await;
    assert_eq!(
        stale,
        ("DurableInvalid".into(), vec!["not_claim_holder".into()])
    );
    assert_eq!(row(&harness, "claim:shared"), snapshot);
    let wrong_run = invoke(
        program,
        harness.handle(),
        "TaskDurable.act",
        &[
            winner,
            &format!("run-{loser}"),
            "shared",
            "checkpoint",
            "",
            "wrong-run",
        ],
    )
    .await;
    assert_eq!(wrong_run.0, "DurableInvalid");
    assert_eq!(row(&harness, "claim:shared"), snapshot);
    harness.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn task_durable_independent_progress_pause_and_disk_reopen() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("records.lbug");
    let program = program();
    let title = "quote\" slash\\ newline\nUnicode Ω null {} []";
    let expected_a;
    let expected_b;
    {
        let harness = Harness::open(&path);
        initialize(program.clone(), &harness, "a", "run-a", title).await;
        initialize(program.clone(), &harness, "b", "run-b", "B").await;
        assert_eq!(field(&row(&harness, "task:a"), "title"), title);
        harness.gate_next_two_reads();
        let (a, b) = tokio::join!(
            invoke(
                program.clone(),
                harness.handle(),
                "TaskDurable.act",
                &["a", "run-a", "worktree-a", "acquire", "", "acquire-a"]
            ),
            invoke(
                program.clone(),
                harness.handle(),
                "TaskDurable.act",
                &["b", "run-b", "worktree-b", "acquire", "", "acquire-b"]
            ),
        );
        assert_eq!(a.0, "DurableCommitted");
        assert_eq!(b.0, "DurableCommitted");
        assert_eq!(
            invoke(
                program.clone(),
                harness.handle(),
                "TaskDurable.act",
                &["a", "run-a", "worktree-a", "pause", "", "pause-a"]
            )
            .await
            .0,
            "DurableCommitted"
        );
        assert_eq!(field(&row(&harness, "task:a"), "state"), "pause_requested");
        harness.gate_next_two_reads();
        let progress_args = ["b", "run-b", "worktree-b", "progress", title, "progress-b"];
        let (a, b) = tokio::join!(
            invoke(
                program.clone(),
                harness.handle(),
                "TaskDurable.act",
                &["a", "run-a", "worktree-a", "checkpoint", "", "checkpoint-a"]
            ),
            invoke(
                program.clone(),
                harness.handle(),
                "TaskDurable.act",
                &progress_args
            ),
        );
        assert_eq!(a.0, "DurableCommitted");
        assert_eq!(b.0, "DurableCommitted");
        expected_a = row(&harness, "task:a");
        expected_b = row(&harness, "run:run-b");
        assert_eq!(field(&expected_a, "state"), "paused");
        assert_eq!(field(&expected_b, "state"), "running");
        assert_eq!(field(&expected_b, "progress"), title);
        assert_eq!(field(&row(&harness, "claim:worktree-a"), "run"), "");
        assert_eq!(field(&row(&harness, "claim:worktree-b"), "run"), "run-b");
        harness.close().await;
    }
    let reopened = Harness::open(&path);
    assert_eq!(row(&reopened, "task:a"), expected_a);
    assert_eq!(row(&reopened, "run:run-b"), expected_b);
    assert_eq!(
        invoke(
            program.clone(),
            reopened.handle(),
            "TaskDurable.reconcile",
            &["checkpoint-a"]
        )
        .await
        .0,
        "DurableReceipt"
    );
    assert_eq!(
        invoke(
            program,
            reopened.handle(),
            "TaskDurable.reconcile",
            &["absent"]
        )
        .await
        .0,
        "DurableAbsent"
    );
    reopened.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn task_durable_primitive_receipt_replay_and_denial() {
    let directory = tempfile::tempdir().unwrap();
    let harness = Harness::open(&directory.path().join("records.lbug"));
    let batch = r#"{"version":1,"conditions":[{"key":"literal","expected":{"kind":"absent"}}],"puts":[{"key":"literal","fields":[{"name":"value","value":"null"}]}]}"#;
    let initial = raw(&harness, "records_commit", &["id", batch]).await;
    let repeated = raw(&harness, "records_commit", &["id", batch]).await;
    assert!(initial.matches(&repeated));
    assert_eq!(row(&harness, "literal").revision, "1");
    let mut update: serde_json::Value = serde_json::from_str(batch).unwrap();
    update["conditions"][0]["expected"] = serde_json::json!({"kind":"revision","revision":"1"});
    let update = update.to_string();
    let second = raw(&harness, "records_commit", &["id-2", &update]).await;
    let second_repeat = raw(&harness, "records_commit", &["id-2", &update]).await;
    assert!(second.matches(&second_repeat));
    let eval::Value::Ok(payload) = second else {
        panic!("second revision must commit")
    };
    let eval::Value::String(payload) = *payload else {
        panic!()
    };
    let receipt: serde_json::Value = serde_json::from_str(&payload).unwrap();
    assert_eq!(
        receipt,
        serde_json::json!({"format":"records_v1","receipt":{"version":1,"request_id":"id-2","revisions":[{"key":"literal","revision":"2"}]}})
    );
    assert_eq!(row(&harness, "literal").revision, "2");
    let changed = batch.replace("\"null\"", "\"[]\"");
    assert!(
        matches!(raw(&harness,"records_commit",&["id",&changed]).await,eval::Value::Err(e) if e=="record_request_id_mismatch")
    );
    let program = program();
    let no_grant = crate::coroutine::CoroutineHandle::new(harness.sender.clone())
        .with_turn(Some(harness.turn.clone()));
    assert_eq!(
        invoke(program.clone(), no_grant, "TaskDurable.reconcile", &["id"])
            .await
            .0,
        "DurableDenied"
    );
    let mut changed_turn = harness.turn.clone();
    changed_turn.principal = "foreign".into();
    assert_eq!(
        invoke(
            program,
            harness.handle().with_turn(Some(changed_turn)),
            "TaskDurable.reconcile",
            &["id"]
        )
        .await
        .0,
        "DurableDenied"
    );
    harness.close().await;
}
