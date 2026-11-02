use super::*;
use crate::coroutine::CoroutineHandle;
use serde_json::json;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn task_durable_lost_commit_reply_reconciles_without_retry() {
    let directory = tempfile::tempdir().unwrap();
    let harness = Harness::open(&directory.path().join("records.lbug"));
    let program = program();
    initialize(program.clone(), &harness, "a", "r", "title").await;
    harness.lose_next_commit_reply();
    let outcome = invoke(
        program.clone(),
        harness.handle(),
        "TaskDurable.act",
        &["a", "r", "resource", "acquire", "", "unknown"],
    )
    .await;
    assert_eq!(outcome.0, "DurableOutcomeUnknown");
    let before = row(&harness, "task:a");
    assert_eq!(field(&before, "state"), "active");
    assert_eq!(
        invoke(
            program,
            harness.handle(),
            "TaskDurable.reconcile",
            &["unknown"]
        )
        .await
        .0,
        "DurableReceipt"
    );
    assert_eq!(row(&harness, "task:a"), before);
    harness.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn task_durable_authority_ceiling_and_wire_contract() {
    let directory = tempfile::tempdir().unwrap();
    let harness = Harness::open(&directory.path().join("records.lbug"));
    let response = raw(
        &harness,
        "records_read",
        &[r#"{"version":1,"keys":["missing"]}"#],
    )
    .await;
    let eval::Value::Ok(value) = response else {
        panic!()
    };
    let eval::Value::String(text) = *value else {
        panic!()
    };
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&text).unwrap(),
        json!({"format":"records_v1","records":[null]})
    );
    let receipt = raw(&harness, "records_receipt", &["missing"]).await;
    let eval::Value::Ok(value) = receipt else {
        panic!()
    };
    let eval::Value::String(text) = *value else {
        panic!()
    };
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&text).unwrap(),
        json!({"format":"records_v1","receipt":null})
    );
    let read_only =
        harness
            .store
            .grant_for_test("fixture-namespace", &harness.turn, "read-only", false);
    let handle = CoroutineHandle::new(harness.sender.clone())
        .with_turn(Some(harness.turn.clone()))
        .with_records_for_test(harness.store.clone(), read_only);
    assert_eq!(
        invoke(program(), handle, "TaskDurable.reconcile", &["missing"])
            .await
            .0,
        "DurableDenied"
    );
    let mut guest = harness.turn.clone();
    guest.guest_tools = Some(vec!["records_read".into()]);
    let guest_handle = harness.handle().with_turn(Some(guest));
    assert!(
        guest_handle
            .execute_await("records_read", &[eval::Value::string("{}")])
            .is_err()
    );
    let mut other = harness.turn.clone();
    other.context = "different".into();
    assert_eq!(
        invoke(
            program(),
            harness.handle().with_turn(Some(other)),
            "TaskDurable.reconcile",
            &["missing"]
        )
        .await
        .0,
        "DurableDenied"
    );
    harness.close().await;
}

fn codec_check(program: &ast::Program, target: &str, body: &str) -> bool {
    let function = program.get_function(target).unwrap();
    let mut env = eval::Env::new_with_shared_interner(&program.shared_interner);
    env.set("body", eval::Value::string(body));
    matches!(
        eval::eval_function_body_named(program, target, &function.body, &mut env).unwrap(),
        eval::Value::Bool(true)
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn task_durable_stale_release_replay_never_releases_new_holder() {
    let directory = tempfile::tempdir().unwrap();
    let harness = Harness::open(&directory.path().join("records.lbug"));
    let program = program();
    initialize(program.clone(), &harness, "a", "run-a", "A").await;
    initialize(program.clone(), &harness, "b", "run-b", "B").await;
    for (action, id) in [
        ("acquire", "acquire-a"),
        ("pause", "pause-a"),
        ("checkpoint", "release-a"),
    ] {
        assert_eq!(
            invoke(
                program.clone(),
                harness.handle(),
                "TaskDurable.act",
                &["a", "run-a", "shared", action, "", id]
            )
            .await
            .0,
            "DurableCommitted"
        );
    }
    let old_release = harness.committed_batch("release-a");
    assert_eq!(
        invoke(
            program.clone(),
            harness.handle(),
            "TaskDurable.act",
            &["b", "run-b", "shared", "acquire", "", "acquire-b"]
        )
        .await
        .0,
        "DurableCommitted"
    );
    let before = row(&harness, "claim:shared");
    assert_eq!(field(&before, "run"), "run-b");
    assert!(matches!(
        raw(&harness, "records_commit", &["release-a", &old_release]).await,
        eval::Value::Ok(_)
    ));
    assert_eq!(row(&harness, "claim:shared"), before);
    assert!(
        matches!(raw(&harness,"records_commit",&["stale-new-id",&old_release]).await,eval::Value::Err(e) if e=="record_version_conflict")
    );
    let wrong = invoke(
        program,
        harness.handle(),
        "TaskDurable.act",
        &["a", "run-a", "shared", "checkpoint", "", "wrong-holder"],
    )
    .await;
    assert_eq!(
        wrong,
        ("DurableInvalid".into(), vec!["not_claim_holder".into()])
    );
    assert_eq!(row(&harness, "claim:shared"), before);
    harness.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn task_durable_rechecks_project_scope_and_escaped_ids() {
    let directory = tempfile::tempdir().unwrap();
    let harness = Harness::open(&directory.path().join("records.lbug"));
    let program = program();
    let task = "task\"\\\nΩ";
    let run = "run\"\\\nΩ";
    initialize(program.clone(), &harness, task, run, "{}").await;
    assert_eq!(field(&row(&harness, &format!("task:{task}")), "id"), task);
    let project = row(&harness, "project:project");
    let altered = json!({"version":1,"conditions":[{"key":project.key,"expected":{"kind":"revision","revision":project.revision}}],"puts":[{"key":"project:project","fields":[{"name":"id","value":"project"},{"name":"workspace","value":"foreign"}]}]});
    assert!(matches!(
        raw(
            &harness,
            "records_commit",
            &["alter-scope", &altered.to_string()]
        )
        .await,
        eval::Value::Ok(_)
    ));
    let result = invoke(
        program.clone(),
        harness.handle(),
        "TaskDurable.act",
        &[task, run, "resource", "acquire", "", "bad-scope"],
    )
    .await;
    assert_eq!(
        result,
        (
            "DurableInvalid".into(),
            vec!["cross_workspace_project".into()]
        )
    );
    assert_eq!(
        field(&row(&harness, &format!("task:{task}")), "state"),
        "ready"
    );
    assert_eq!(
        field(&row(&harness, &format!("run:{run}")), "state"),
        "pending"
    );
    let no_project = invoke(
        program,
        harness.handle(),
        "TaskDurable.initialize",
        &[
            "workspace",
            "",
            "projectless",
            "projectless-run",
            "[]",
            "projectless-init",
        ],
    )
    .await;
    assert_eq!(no_project.0, "DurableCommitted");
    assert_eq!(field(&row(&harness, "task:projectless"), "project"), "");
    harness.close().await;
}

fn codec_program() -> ast::Program {
    let mut program = (*program()).clone();
    install(
        &mut program,
        "CodecChecks",
        "def row_valid(body: str) -> bool:\n    match TaskRecordCodec.decode(body, ['x']):\n        case RecordsDecoded(rows):\n            row: RecordRow = get(rows, 0)\n            return TaskRecordCodec.fields_are(row, ['value'])\n        case RecordsMalformed():\n            return False\ndef receipt_valid(body: str) -> bool:\n    return TaskRecordCodec.receipt_matches(body, 'id')\n",
    );
    program
}

#[test]
fn task_durable_codec_guards_multiple_revisions_and_field_arrays() {
    let program = codec_program();
    let base = json!({"format":"records_v1","records":[{"key":"x","revision":"1","fields":[{"name":"value","value":"null"}]}]});
    assert!(codec_check(
        &program,
        "CodecChecks.row_valid",
        &base.to_string()
    ));
    for revision in ["1", "2", "3", "20", "128"] {
        let mut candidate = base.clone();
        candidate["records"][0]["revision"] = json!(revision);
        assert!(codec_check(
            &program,
            "CodecChecks.row_valid",
            &candidate.to_string()
        ));
        let receipt = json!({"format":"records_v1","receipt":{"request_id":"id","revisions":[{"key":"x","revision":revision}]}});
        assert!(codec_check(
            &program,
            "CodecChecks.receipt_valid",
            &receipt.to_string()
        ));
    }
    for revision in ["0", "02", "-1", "", "1x"] {
        let mut candidate = base.clone();
        candidate["records"][0]["revision"] = json!(revision);
        assert!(!codec_check(
            &program,
            "CodecChecks.row_valid",
            &candidate.to_string()
        ));
    }
    let mut bad = Vec::new();
    let mut v = base.clone();
    v["format"] = json!(true);
    bad.push(v);
    let mut v = base.clone();
    v["records"][0]["revision"] = json!(1);
    bad.push(v);
    let mut v = base.clone();
    v["records"][0]["key"] = json!("foreign");
    bad.push(v);
    let mut v = base.clone();
    v["records"][0]["fields"][0]["value"] = json!(null);
    bad.push(v);
    let mut v = base.clone();
    v["records"][0]["fields"]
        .as_array_mut()
        .unwrap()
        .push(json!({"name":"value","value":"duplicate"}));
    bad.push(v);
    let mut v = base.clone();
    v["records"][0]["fields"][0]["name"] = json!("unknown-domain-field");
    bad.push(v);
    for v in bad {
        assert!(!codec_check(
            &program,
            "CodecChecks.row_valid",
            &v.to_string()
        ));
    }
    assert!(!codec_check(&program, "CodecChecks.row_valid", "{"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn task_durable_forged_success_has_no_authority() {
    let program = codec_program();
    // Deliberately documented: envelope object-key duplication is not detectable.
    let forged = r#"{"format":"records_v1","receipt":{"request_id":"id","revisions":[{"key":"x","revision":"1"}]}}"#;
    assert!(codec_check(&program, "CodecChecks.receipt_valid", forged));
    let directory = tempfile::tempdir().unwrap();
    let harness = Harness::open(&directory.path().join("records.lbug"));
    let no_grant =
        CoroutineHandle::new(harness.sender.clone()).with_turn(Some(harness.turn.clone()));
    assert_eq!(
        invoke(
            Arc::new(program),
            no_grant,
            "TaskDurable.initialize",
            &["w", "", "t", "r", "title", "id"]
        )
        .await
        .0,
        "DurableDenied"
    );
    assert!(
        matches!(raw(&harness,"records_commit",&["id",forged]).await,eval::Value::Err(e) if e=="invalid_record_request")
    );
    assert_eq!(
        harness
            .store
            .receipt(Some(&harness.grant), Some(&harness.turn), "id")
            .unwrap(),
        None
    );
    let stored = harness
        .store
        .read(
            Some(&harness.grant),
            Some(&harness.turn),
            r#"{"version":1,"keys":["task:t","x"]}"#,
        )
        .unwrap();
    assert_eq!(stored, vec![None, None]);
    harness.close().await;
}
