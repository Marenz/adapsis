//! Integration only: real workspace command + existing evaluated progress
//! callbacks, with the same shared turn/work locks and an explicitly gated
//! mock provider. No notice becomes model/history/memory/training content.
use super::*;
use crate::workspace::{self as ws, test_support::RepoFixture};
use axum::{
    body::Body,
    extract::ConnectInfo,
    http::{Request, StatusCode},
};
use tower::ServiceExt;

fn bound_harness(context: &str, name: &str) -> (RepoFixture, Arc<Harness>, ws::Workspace) {
    let repo = RepoFixture::new();
    let site = repo.site(name);
    let workspace = ws::open(site.registry.get(name).unwrap(), "ack", None).unwrap();
    site.bindings.claim(context, &workspace, &|_| None).unwrap();
    let mut harness = Harness::new(true);
    harness.workspaces = Some(site);
    (repo, Arc::new(harness), workspace)
}

fn spawn_turn(
    harness: &Arc<Harness>,
    io: &Arc<CallbackIo>,
    provider: &Arc<Provider>,
    context: &'static str,
    input: &'static str,
    destination: &'static str,
    function: &'static str,
) -> tokio::task::JoinHandle<String> {
    let (h, io, p) = (harness.clone(), io.clone(), provider.clone());
    tokio::spawn(async move {
        io.turn(
            &h,
            &p,
            context,
            ADMIN,
            input,
            (Some(function), Some(destination)),
        )
        .await
    })
}

async fn local_cancel(harness: &Harness, context: &str) {
    let mut config = crate::api::tests::test_config();
    config.workspaces = harness.workspaces.clone();
    let mut request = Request::builder()
        .method("POST")
        .uri("/api/workspace/cancel")
        .header("Content-Type", "application/json")
        .body(Body::from(json!({"context":context}).to_string()))
        .unwrap();
    request.extensions_mut().insert(ConnectInfo(
        "127.0.0.1:1234".parse::<std::net::SocketAddr>().unwrap(),
    ));
    let answer = crate::api::router_with_llm(config)
        .oneshot(request)
        .await
        .unwrap();
    assert_eq!(
        answer.status(),
        StatusCode::ACCEPTED,
        "only requests cancellation, not a stopped/delivered claim"
    );
}

fn assert_queued_and_change_stored_pair(
    harness: &Harness,
    io: &CallbackIo,
    provider: &Provider,
    context: &str,
) {
    assert_eq!(
        provider.requests.lock().unwrap().len(),
        1,
        "queued turn cannot generate before acquiring context lock"
    );
    assert!(
        !io.launches
            .lock()
            .unwrap()
            .iter()
            .any(|launch| launch.destination == "second-destination" && launch.message == STARTED)
    );
    assert!(
        !harness
            .meta
            .lock()
            .unwrap()
            .conversations
            .get(context)
            .unwrap()
            .messages
            .iter()
            .any(|message| message.content == "queued workspace turn")
    );
    // Callback pairing was pinned before the wait, not looked up from this metadata.
    harness
        .meta
        .lock()
        .unwrap()
        .conversations
        .get_mut(context)
        .unwrap()
        .reply_arg = Some("wrong-stored-destination".into());
}

fn assert_shared_turn_lock_held(context: &str) {
    assert!(
        crate::api::turn_lock::of(context).try_lock_owned().is_err(),
        "Progress acquired THE shared context lock"
    );
}

fn fixture_pid(workspace: &ws::Workspace) -> i32 {
    std::fs::read_to_string(workspace.path.join("ack.pid"))
        .unwrap()
        .parse()
        .unwrap()
}

fn assert_command_gone(pid: i32) {
    // SAFETY: observe the fixture command's existence only, never signal it.
    assert_ne!(
        unsafe { libc::kill(pid, 0) },
        0,
        "queued start cannot precede cleanup of the actual command"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn queued_progress_waits_for_the_shared_turn_lock_and_bash_cancel_cleanup() {
    if !crate::logging::tests::child() {
        // Child isolation avoids unrelated tests' same two valid admin-DM
        // locks; eligibility/shared-lock assertions remain real, without sleeps.
        crate::logging::tests::capture(
            "api::takeover_native::tests::progress::workspace::queued_progress_waits_for_the_shared_turn_lock_and_bash_cancel_cleanup",
            false,
        );
        return;
    }
    let context = "telegram:user:100000001";
    let (_repo, harness, workspace) = bound_harness(context, "ack-workspace-queue");
    let io = Arc::new(CallbackIo::new(&harness, false).await);
    let (provider, release) = held_sequence(vec![
        tool(
            "workspace-bash",
            "bash",
            json!({"command":"printf '%s' $$ > ack.pid; sleep 60","timeout_ms":10000}),
        ),
        output(true, 1),
    ])
    .await;
    let provider = Arc::new(provider);
    let first = spawn_turn(
        &harness,
        &io,
        &provider,
        context,
        "first workspace turn",
        "first-destination",
        REPLY,
    );
    until(|| provider.requests.lock().unwrap().len() == 1).await;
    release.add_permits(1);
    until(|| {
        std::fs::read_to_string(workspace.path.join("ack.pid")).is_ok_and(|text| !text.is_empty())
    })
    .await;
    let pid = fixture_pid(&workspace);
    let site = harness.workspaces.as_ref().unwrap();
    assert_shared_turn_lock_held(context);
    assert!(
        crate::api::turn_lock::work_now(site.bindings.root(), context).is_none(),
        "worker holds work lock"
    );
    let second = spawn_turn(
        &harness,
        &io,
        &provider,
        context,
        "queued workspace turn",
        "second-destination",
        REPLY,
    );
    until(|| {
        io.delivered
            .lock()
            .unwrap()
            .iter()
            .any(|text| text == &format!("second-destination|{QUEUED}"))
    })
    .await;
    assert_queued_and_change_stored_pair(&harness, &io, &provider, context);
    local_cancel(&harness, context).await;
    assert!(first.await.unwrap().contains("Foreground cancellation"));
    assert_command_gone(pid);
    until(|| {
        provider.requests.lock().unwrap().len() == 2
            && io
                .delivered
                .lock()
                .unwrap()
                .iter()
                .any(|text| text == &format!("second-destination|{STARTED}"))
    })
    .await;
    assert!(
        crate::api::turn_lock::work_now(site.bindings.root(), context).is_some(),
        "normal cancellation finished worker cleanup before releasing context"
    );
    release.add_permits(1);
    assert_eq!(second.await.unwrap(), "final answer");
    let queued: Vec<_> = io
        .launches
        .lock()
        .unwrap()
        .iter()
        .filter(|launch| launch.destination == "second-destination")
        .cloned()
        .collect();
    assert_eq!(
        queued
            .iter()
            .map(|launch| launch.message.as_str())
            .collect::<Vec<_>>(),
        [QUEUED, STARTED]
    );
    assert!(queued.iter().all(|launch| launch.function == REPLY));
    assert_clean(&harness, &io, &provider, context).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_failed_progress_callback_does_not_cancel_actual_workspace_work() {
    if !crate::logging::tests::child() {
        crate::logging::tests::capture(
            "api::takeover_native::tests::progress::workspace::a_failed_progress_callback_does_not_cancel_actual_workspace_work",
            false,
        );
        return;
    }
    let context = "telegram:100000001";
    let (_repo, harness, workspace) = bound_harness(context, "ack-workspace-failure");
    let io = Arc::new(CallbackIo::new(&harness, false).await);
    let (provider,release) = held_sequence(vec![
        tool("workspace-bash","bash",json!({"command":"printf '%s' $$ > ack.pid; while [ ! -f release ]; do sleep 0.01; done; printf completed","timeout_ms":10000})),output(true,1),
    ]).await;
    let provider = Arc::new(provider);
    let turn = spawn_turn(
        &harness,
        &io,
        &provider,
        context,
        "work despite callback failure",
        "failure-destination",
        "ProgressFixture.fail",
    );
    until(|| provider.requests.lock().unwrap().len() == 1).await;
    release.add_permits(1);
    until(|| {
        std::fs::read_to_string(workspace.path.join("ack.pid")).is_ok_and(|text| !text.is_empty())
            && io
                .registry
                .lock()
                .unwrap()
                .values()
                .any(|task| matches!(task.status, crate::coroutine::WaitReason::Failed(_)))
    })
    .await;
    assert!(
        !turn.is_finished(),
        "callback failure never finishes/cancels the actual command"
    );
    let pid = fixture_pid(&workspace);
    // SAFETY: signal zero observes this fixture command only.
    assert_eq!(
        unsafe { libc::kill(pid, 0) },
        0,
        "command still runs after the failed callback"
    );
    std::fs::write(workspace.path.join("release"), "continue").unwrap();
    until(|| provider.requests.lock().unwrap().len() == 2).await;
    let requests = provider.requests.lock().unwrap().clone();
    let result = requests[1]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["tool_call_id"] == "workspace-bash")
        .unwrap()["content"]
        .as_str()
        .unwrap();
    assert!(result.contains("completed\n[exit status 0"), "{result}");
    release.add_permits(1);
    assert_eq!(turn.await.unwrap(), "final answer");
    assert_eq!(
        io.launches.lock().unwrap().len(),
        1,
        "failed progress worker does not retry"
    );
    assert!(
        harness
            .workspaces
            .as_ref()
            .unwrap()
            .bindings
            .load(context)
            .unwrap()
            .is_some()
    );
    assert_clean(&harness, &io, &provider, context).await;
}
