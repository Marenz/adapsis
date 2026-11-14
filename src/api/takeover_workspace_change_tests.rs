//! W2: one real repository edit-and-test turn, no live Program mutation.
use super::*;

#[tokio::test]
async fn workspace_native_outcome_refuses_silent_finish_without_replaying_bash() {
    let repo = repo("outcome-integration");
    let harness = owner(&repo);
    let before = serde_json::to_string(&*harness.program.read().await).unwrap();
    let provider = provider(vec![
        tool("open", "workspace_open", json!({"repo":repo.name,"name":"outcome"})),
        tool("action", "bash", json!({"command":"printf 'one\\n' >> outcome-count.txt"})),
        response("", vec![call("no-outcome", "finish", json!({}))]),
        done("The repository action ran once; the larger task is not complete.", "reported"),
    ]).await;
    let reply = harness.turn(&provider, DM, ADMIN, "run the repository action").await;
    assert_eq!(reply, "The repository action ran once; the larger task is not complete.");
    assert_eq!(std::fs::read_to_string(repo.worktrees.join("outcome/outcome-count.txt")).unwrap(), "one\n");
    let requests = provider.requests.lock().unwrap();
    assert_eq!(requests.len(), 4);
    let refusal: Value = serde_json::from_str(&text_of(&requests[3], "no-outcome")).unwrap();
    assert_eq!(refusal["done_accepted"], false);
    assert_eq!(refusal["success"], false);
    assert_eq!(refusal["has_errors"], true);
    assert!(!refusal.to_string().contains("done accepted"));
    assert!(text_of(&requests[2], "action").contains("[exit status 0"));
    assert_eq!(serde_json::to_string(&*harness.program.read().await).unwrap(), before);
    assert_eq!(harness.meta.lock().unwrap().revision, 0, "no persistent helper module");
}

#[tokio::test]
async fn owner_edits_writes_and_runs_repository_tests_in_one_turn() {
    let repo = repo("changes");
    write(
        &repo.fixture.repository.checkout,
        "answer.txt",
        "answer=1\n",
    );
    commit_all(&repo.fixture.repository.checkout, "answer fixture");
    let harness = owner(&repo);
    let before = serde_json::to_string(&*harness.program.read().await).unwrap();
    let script =
        "set -eu\ntest \"$(cat answer.txt)\" = answer=2\nprintf 'REPOSITORY_TEST_PASSED\\n'\n";
    let provider = provider(vec![
        tool(
            "open",
            "workspace_open",
            json!({"repo":repo.name,"name":"change"}),
        ),
        response(
            "",
            vec![
                call("read", "read", json!({"path":"answer.txt"})),
                call(
                    "edit",
                    "edit",
                    json!({"path":"answer.txt","old":"answer=1","new":"answer=2"}),
                ),
                call("write", "write", json!({"path":"test.sh","content":script})),
            ],
        ),
        tool(
            "test",
            "bash",
            json!({"command":"bash test.sh","timeout_ms":10000}),
        ),
        tool("dirty", "workspace_close", json!({})),
        done("Edited and tested.", "f"),
    ])
    .await;
    assert_eq!(
        harness
            .turn(&provider, DM, ADMIN, "edit then run tests")
            .await,
        "Edited and tested."
    );
    let requests = provider.requests.lock().unwrap();
    assert_eq!(
        text_of(&requests[2], "edit"),
        "edited answer.txt: 1 replacement, line 1"
    );
    assert!(text_of(&requests[2], "write").starts_with("created test.sh ("));
    let result = text_of(&requests[3], "test");
    assert!(
        result.starts_with("bash in .\nREPOSITORY_TEST_PASSED\n[exit status 0"),
        "{result}"
    );
    assert!(
        text_of(&requests[4], "dirty").contains("uncommitted changes")
            && text_of(&requests[4], "dirty").contains("was not closed")
    );
    assert_eq!(
        std::fs::read_to_string(repo.worktrees.join("change/answer.txt")).unwrap(),
        "answer=2\n"
    );
    let ledger = binding(&repo, DM).unwrap().ledger;
    assert_eq!(ledger.len(), 2);
    assert_eq!(ledger.recorded("answer.txt").unwrap().size, 9);
    assert_eq!(
        serde_json::to_string(&*harness.program.read().await).unwrap(),
        before
    );
    assert_eq!(
        harness.meta.lock().unwrap().revision,
        0,
        "no Adapsis Program mutation/history"
    );
}

#[tokio::test]
async fn repository_change_refusals_and_failed_commands_are_structural_failures() {
    let repo = repo("change-errors");
    let harness = owner(&repo);
    let provider = provider(vec![
        tool("open","workspace_open",json!({"repo":repo.name,"name":"errors"})),
        tool("unread","edit",json!({"path":"README.md","old":"fixture","new":"changed"})),
        tool("unread-write","write",json!({"path":"README.md","content":"changed\n"})),
        tool("exit","bash",json!({"command":"printf 'intentional failure\\n'; exit 7"})),
        tool("push","bash",json!({"command":"git -C . -c user.name=Test push origin HEAD:refs/heads/main; touch must-not-exist"})),
        tool("read","read",json!({"path":"README.md"})),
        tool("outside","bash",json!({"command":"printf outside > README.md"})),
        tool("stale","edit",json!({"path":"README.md","old":"fixture","new":"changed"})),
        done("Refused and tested.","f"),
    ]).await;
    harness.turn(&provider, DM, ADMIN, "check errors").await;
    let requests = provider.requests.lock().unwrap();
    for (index, id) in [(2, "unread"), (3, "unread-write")] {
        assert!(text_of(&requests[index], id).contains("has not been read in this conversation"));
    }
    assert!(
        text_of(&requests[4], "exit")
            .starts_with("error: bash in .\nintentional failure\n[exit status 7")
    );
    assert!(text_of(&requests[5], "push").contains("push was not run: use an explicit remote"));
    assert!(
        !repo.worktrees.join("errors/must-not-exist").exists(),
        "guard refusal precedes every shell effect"
    );
    assert_eq!(
        std::fs::read_to_string(repo.worktrees.join("errors/README.md")).unwrap(),
        "outside"
    );
    assert!(text_of(&requests[8], "stale").contains("file changed since it was read"));
}

#[tokio::test]
async fn the_coroutine_shell_policy_also_gates_workspace_bash() {
    let repo = repo("denied-policy");
    let mut harness = owner(&repo);
    harness.shell_policy = crate::coroutine::ShellPolicy::Denied;
    let provider = provider(vec![
        tool(
            "open",
            "workspace_open",
            json!({"repo":repo.name,"name":"denied"}),
        ),
        tool(
            "bash",
            "bash",
            json!({"command":"printf bad > never-created"}),
        ),
        done("Denied.", "f"),
    ])
    .await;
    harness.turn(&provider, DM, ADMIN, "check policy").await;
    assert!(
        text_of(&provider.requests.lock().unwrap()[2], "bash")
            .starts_with("error: command not run:")
    );
    assert!(!repo.worktrees.join("denied/never-created").exists());
}

#[tokio::test]
async fn workspace_close_refuses_unverifiable_job_records_in_its_own_state_root() {
    let repo = repo("close-job");
    let workspace =
        crate::workspace::open(repo.site.registry.get(&repo.name).unwrap(), "job", None).unwrap();
    repo.site.bindings.claim(DM, &workspace, &|_| None).unwrap();
    let job = repo
        .site
        .bindings
        .root()
        .join(&repo.name)
        .join("job/jobs/j1");
    std::fs::create_dir_all(&job).unwrap();
    std::fs::write(job.join("job.json"), "not a valid job record").unwrap();
    let harness = owner(&repo);
    let provider = provider(vec![
        tool("close", "workspace_close", json!({})),
        done("Not closed.", "f"),
    ])
    .await;
    harness.turn(&provider, DM, ADMIN, "close").await;
    let text = text_of(&provider.requests.lock().unwrap()[1], "close");
    assert!(
        text.contains("unverifiable job records") && text.contains("was not closed"),
        "{text}"
    );
    assert!(workspace.path.exists() && binding(&repo, DM).is_some());
    assert!(
        job.join("job.json").exists(),
        "unverifiable state is not deleted"
    );
}

#[tokio::test]
async fn workspace_close_keeps_a_clean_worktree_while_an_existing_job_is_running() {
    if !crate::workspace::systemd::available() {
        return;
    }
    let repo = repo("close-running-job");
    let workspace =
        crate::workspace::open(repo.site.registry.get(&repo.name).unwrap(), "job", None).unwrap();
    repo.site.bindings.claim(DM, &workspace, &|_| None).unwrap();
    let jobs = crate::workspace::JobStore::at(
        repo.site.bindings.root(),
        crate::workspace::run::default_unit_prefix(),
    )
    .unwrap();
    let (store, ws) = (jobs.clone(), workspace.clone());
    let started = tokio::task::spawn_blocking(move || {
        let mut request = crate::workspace::JobRequest::new("sleep 60");
        request.max_runtime = Some(std::time::Duration::from_secs(20));
        store.start(
            &ws,
            &request,
            &crate::coroutine::ShellPolicy::Unrestricted,
            &crate::workspace::run::bash_argv,
        )
    })
    .await
    .unwrap()
    .unwrap();
    let harness = owner(&repo);
    let provider = provider(vec![
        tool("close", "workspace_close", json!({})),
        done("Job is running.", "f"),
    ])
    .await;
    harness.turn(&provider, DM, ADMIN, "close").await;
    let text = text_of(&provider.requests.lock().unwrap()[1], "close");
    let (store, ws, id) = (jobs, workspace.clone(), started.record.id);
    tokio::task::spawn_blocking(move || store.kill(&ws, &id))
        .await
        .unwrap()
        .unwrap();
    assert!(
        text.contains("has running jobs") && text.contains("was not closed"),
        "{text}"
    );
    assert!(workspace.path.exists() && binding(&repo, DM).is_some());
}

#[tokio::test]
async fn binary_command_output_keeps_its_failure_verdict_within_the_model_result_cap() {
    let repo = repo("binary-output");
    let harness = owner(&repo);
    let provider = provider(vec![
        tool("open","workspace_open",json!({"repo":repo.name,"name":"binary"})),
        tool("binary","bash",json!({"command":"/usr/bin/python3 -c 'import os; os.write(1, bytes([255])*25000); raise SystemExit(7)'","max_output_bytes":20000})),
        done("Failed as expected.","f"),
    ]).await;
    harness
        .turn(&provider, DM, ADMIN, "run a binary-output command")
        .await;
    let text = text_of(&provider.requests.lock().unwrap()[2], "binary");
    assert!(text.len() <= 20_512, "{}", text.len());
    assert!(text.contains("[exit status 7") && text.contains("display text shortened by runtime"));
    assert!(
        !text.contains("result cut by the runtime at the per-message limit"),
        "the exit diagnosis must not be cut away"
    );
}

/// Explicit, manual measurement (not a timing assertion or automatic policy
/// change): real owner turn with 50 reads, then equivalent read/hash work with
/// one ledger flush. Production still durably writes after each tool call.
#[tokio::test]
#[ignore = "manual 50-read ledger-fsync measurement"]
async fn measure_fifty_read_ledger_flushes_against_one_batch_flush() {
    let repo = repo("read-measure");
    for n in 0..50 {
        write(
            &repo.fixture.repository.checkout,
            &format!("data/{n}.txt"),
            "test content\n".repeat(100),
        );
    }
    commit_all(&repo.fixture.repository.checkout, "measurement files");
    let harness = owner(&repo);
    let calls = (0..50)
        .map(|n| {
            call(
                &format!("r{n}"),
                "read",
                json!({"path":format!("data/{n}.txt")}),
            )
        })
        .collect();
    let provider = provider(vec![
        tool(
            "open",
            "workspace_open",
            json!({"repo":repo.name,"name":"measure"}),
        ),
        response("", calls),
        done("Read.", "f"),
    ])
    .await;
    let started = std::time::Instant::now();
    assert_eq!(
        harness.turn(&provider, DM, ADMIN, "read fifty files").await,
        "Read."
    );
    let turn_time = started.elapsed();
    assert_eq!(
        repo.site.bindings.writes_for_tests(),
        51,
        "one claim, fifty durable read records"
    );
    let workspace =
        crate::workspace::find(repo.site.registry.get(&repo.name).unwrap(), "measure").unwrap();
    let before = repo.site.bindings.writes_for_tests();
    let mut ledger = crate::workspace::ReadLedger::default();
    let started = std::time::Instant::now();
    for n in 0..50 {
        ledger.note(
            &crate::workspace::read(&workspace, &format!("data/{n}.txt"), None, None).unwrap(),
        );
    }
    assert!(
        repo.site
            .bindings
            .save_ledger(DM, &workspace, &ledger)
            .unwrap()
    );
    let batch_time = started.elapsed();
    assert_eq!(repo.site.bindings.writes_for_tests() - before, 1);
    eprintln!(
        "READ50_MEASUREMENT turn_ms={:.3} batch_read_and_flush_ms={:.3} turn_durable_writes=51 batch_durable_writes=1 ledger_entries={}",
        turn_time.as_secs_f64() * 1000.0,
        batch_time.as_secs_f64() * 1000.0,
        ledger.len()
    );
}
