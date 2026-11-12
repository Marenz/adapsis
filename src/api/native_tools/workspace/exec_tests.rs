//! One writer per workspace, under concurrency and dropped turns. These tests drive
//! the executor directly with conversation keys of their own, so their turn and work
//! locks are theirs alone. (The independent review's probes, kept as regressions.)
use super::*;
use crate::workspace::test_support::RepoFixture;
use std::sync::Barrier;
use std::time::{Duration, Instant};

fn work(site: &Arc<Site>, context: &str) -> Work {
    Work { site: site.clone(), context: context.into(), caps: ResultCaps::for_model("unknown/review"), bound: Bound::Unknown }
}

fn turn(site: &Arc<Site>, context: &str, wait: Duration) -> Turn {
    Turn { work: None, site: site.clone(), context: context.into(), caps: ResultCaps::for_model("unknown/review"), wait }
}

fn use_call(name: &str) -> Call {
    Call::Use { repo: "demo".into(), name: name.into() }
}

fn holders(site: &Arc<Site>, name: &str) -> Vec<String> {
    site.bindings.holders("demo", name).unwrap()
}

/// A `post-checkout` hook that blocks `git worktree add` until `release` exists.
fn blocking_hook(fixture: &RepoFixture) -> (std::path::PathBuf, std::path::PathBuf) {
    use std::os::unix::fs::PermissionsExt;
    let (ready, release) = (fixture.top.join("hook-ready"), fixture.top.join("hook-release"));
    let hook = fixture.repository.checkout.join(".git/hooks/post-checkout");
    std::fs::write(&hook, format!("#!/bin/sh\necho ready > '{}'\nwhile [ ! -f '{}' ]; do /usr/bin/sleep 0.01; done\n", ready.display(), release.display())).unwrap();
    std::fs::set_permissions(&hook, std::fs::Permissions::from_mode(0o700)).unwrap();
    (ready, release)
}

async fn until(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

#[test]
fn a_result_over_the_message_cap_is_cut_on_a_character_boundary_with_a_note() {
    assert_eq!(fit("short".into(), 100), "short");
    for cap in [60, 61, 62, 63, 200] {
        let cut = fit("ä".repeat(500), cap);
        assert!(cut.len() <= cap && cut.ends_with("[result cut by the runtime at the per-message limit]"), "{cap}: {}", cut.len());
    }
    // Smaller than the note itself: still bounded by the note, never a panic.
    assert!(fit("x".repeat(500), 0).starts_with('\n'));
    let outcome = Outcome::new("read", false, format!("error: first line\nsecond {}", "y".repeat(500)), 80);
    assert!(!outcome.success && outcome.label == "`read`: failed" && outcome.text.len() <= 80);
    assert_eq!(Outcome::new("search", true, "anything/secret-path: 3 matches".into(), 80).label, "`search`: ok");
}

/// Review probe: two conversations, each in its own turn, use the same UNBOUND workspace
/// at the same moment. Both stay in their turns until both calls are over: exactly one
/// gets the workspace, and the other is refused with nothing changed.
#[test]
fn two_conversations_using_one_unbound_workspace_leave_one_holder() {
    let fixture = RepoFixture::new();
    let site = fixture.site("demo");
    crate::workspace::open(site.registry.get("demo").unwrap(), "claim-race", None).unwrap();
    // Many other records, so both holder scans overlap if nothing serializes them.
    for index in 0..300 {
        site.bindings.put(&format!("padding-{index}"), &Binding::new("other", "other")).unwrap();
    }
    for trial in 0..12 {
        let (start, finish) = (Arc::new(Barrier::new(2)), Arc::new(Barrier::new(2)));
        let launch = |context: String| {
            let (site, start, finish) = (site.clone(), start.clone(), finish.clone());
            std::thread::spawn(move || {
                let _turn = turn_lock::of(&context).try_lock_owned().unwrap();
                let _work = turn_lock::work_now(site.bindings.root(), &context).unwrap();
                let mut work = work(&site, &context);
                start.wait();
                let result = work.run(use_call("claim-race"));
                finish.wait();
                (context, result)
            })
        };
        let threads = [launch(format!("exec-claim-a-{trial}")), launch(format!("exec-claim-b-{trial}"))];
        let results: Vec<_> = threads.into_iter().map(|thread| thread.join().unwrap()).collect();
        let winners: Vec<&String> = results.iter().filter(|(_, result)| result.is_ok()).map(|(context, _)| context).collect();
        assert_eq!(winners.len(), 1, "trial {trial}: {results:?}");
        assert_eq!(holders(&site, "claim-race"), [winners[0].clone()], "trial {trial}");
        for (context, result) in &results {
            if let Err(error) = result {
                assert!(error.to_string().contains("is in use by another conversation that is working right now; nothing was changed"), "{error}");
                assert_eq!(site.bindings.load(context).unwrap(), None);
            }
            site.bindings.release(context, None).unwrap();
        }
    }
}

/// Review probe: a turn is dropped while its blocking call runs (here `workspace_open`,
/// held in a git hook). The conversation stays busy until that call has finished: its
/// workspace cannot be taken meanwhile, a new turn of the same conversation waits for it
/// (and says so, or gives up after a bound), and the call finishes and binds.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_dropped_turn_keeps_its_conversation_busy_until_its_blocking_call_has_finished() {
    let fixture = RepoFixture::new();
    let site = fixture.site("demo");
    let (owner, taker) = ("exec-dropped-owner", "exec-dropped-taker");
    // The owner holds workspace `first` and has read nothing special.
    {
        let _turn = turn_lock::of(owner).lock_owned().await;
        let outcome = turn(&site, owner, WORK_WAIT_MAX).call("workspace_open", r#"{"repo":"demo","name":"first"}"#).await;
        assert!(outcome.success, "{}", outcome.text);
    }
    let (ready, release) = blocking_hook(&fixture);
    let task = {
        let site = site.clone();
        tokio::spawn(async move {
            let _turn = turn_lock::of(owner).lock_owned().await;
            turn(&site, owner, WORK_WAIT_MAX).call("workspace_open", r#"{"repo":"demo","name":"second"}"#).await
        })
    };
    until("the hook", || ready.exists()).await;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());

    // The turn is over, its blocking call is not: the conversation is not idle.
    assert!(turn_lock::of(owner).try_lock().is_ok(), "the turn lock was released with the dropped future");
    assert!(turn_lock::idle(site.bindings.root(), owner).is_none() && turn_lock::work_now(site.bindings.root(), owner).is_none(), "busy protection ended while the worker still runs");
    // Another conversation cannot take what the owner holds.
    let refused = {
        let _turn = turn_lock::of(taker).lock_owned().await;
        turn(&site, taker, WORK_WAIT_MAX).call("workspace_use", r#"{"repo":"demo","name":"first"}"#).await
    };
    assert_eq!(refused.text, "error: workspace demo/first is in use by another conversation that is working right now; nothing was changed. Try again when it has finished");
    assert_eq!(holders(&site, "first"), [owner]);
    // A new turn of the owner's conversation: its tail line does not read a binding
    // that is about to change, and a call gives up after its bound with nothing done.
    let line = super::tail_line_of(&site, owner).await;
    assert!(line.contains("an earlier workspace operation of this conversation is still finishing"), "{line}");
    let gave_up = turn(&site, owner, Duration::from_millis(200)).call("workspace_status", "{}").await;
    assert!(!gave_up.success);
    assert_eq!(
        gave_up.text,
        "error: an earlier workspace operation of this conversation is still running after 0 s, so `workspace_status` was not started and nothing was changed. Try again later; workspace_status shows the state once it has finished"
    );
    // A patient call waits, then reports the state after the earlier call, and says it waited.
    let waiting = {
        let site = site.clone();
        tokio::spawn(async move { turn(&site, owner, WORK_WAIT_MAX).call("workspace_status", "{}").await })
    };
    tokio::time::sleep(WORK_WAIT_NOTED + Duration::from_millis(300)).await;
    assert!(!waiting.is_finished());
    std::fs::write(&release, b"go").unwrap();
    let waited = waiting.await.unwrap();
    assert!(waited.success, "{}", waited.text);
    assert!(waited.text.starts_with("workspace second
branch: adapsis/second"), "the dropped call finished and bound: {}", waited.text);
    assert!(waited.text.contains("
[note: this call first waited 3 s for an earlier workspace operation of this conversation to finish]"), "{}", waited.text);
    // Finished and bound: one holder each, never two.
    assert_eq!(holders(&site, "second"), [owner]);
    assert!(holders(&site, "first").is_empty(), "the owner's one record moved to `second`");
    assert!(turn_lock::idle(site.bindings.root(), owner).is_some());
    // Now `first` is free for the other conversation.
    let taken = turn(&site, taker, WORK_WAIT_MAX).call("workspace_use", r#"{"repo":"demo","name":"first"}"#).await;
    assert!(taken.success && !taken.text.contains("another conversation"), "{}", taken.text);
}

/// A late writer does not bring a binding back: a conversation whose workspace was taken
/// (between its turns) cannot record a read on the strength of what it still has in memory.
#[test]
fn a_ledger_write_after_the_workspace_was_taken_changes_nothing() {
    let fixture = RepoFixture::new();
    let site = fixture.site("demo");
    crate::workspace::open(site.registry.get("demo").unwrap(), "shared", None).unwrap();
    let (first, second) = ("exec-late-first", "exec-late-second");
    let mut holder = work(&site, first);
    holder.run(use_call("shared")).unwrap();
    assert!(holder.run(Call::Read { path: "README.md".into(), offset: None, limit: None }).unwrap().starts_with("README.md: lines 1-1"));
    assert_eq!(site.bindings.load(first).unwrap().unwrap().ledger.len(), 1);
    // The other conversation takes it while the first is idle, and starts with nothing read.
    let taken = work(&site, second).run(use_call("shared")).unwrap();
    assert!(taken.contains("It was bound to another conversation, which no longer has it.") && taken.contains("Nothing counts as read here yet"), "{taken}");
    assert_eq!(holders(&site, "shared"), [second]);
    assert!(site.bindings.load(second).unwrap().unwrap().ledger.is_empty());
    // The first still has the workspace in memory (as a late blocking call would).
    crate::workspace::test_support::write(&fixture.repository.worktree_parent.join("shared"), "new.txt", "x\n");
    let late = holder.run(Call::Read { path: "new.txt".into(), offset: None, limit: None }).unwrap();
    assert!(late.ends_with("[note: this read could not be recorded; read the file again before changing it]"), "{late}");
    assert_eq!(holders(&site, "shared"), [second], "no second holder");
    assert_eq!(site.bindings.load(first).unwrap(), None);
    // And it stops working from memory: the next call finds it has no workspace.
    let after = holder.run(Call::Read { path: "README.md".into(), offset: None, limit: None }).unwrap_err().to_string();
    assert!(after.starts_with("this conversation has no workspace."), "{after}");
    // Taking it back works the ordinary way.
    assert!(work(&site, first).run(use_call("shared")).unwrap().contains("It was bound to another conversation"));
    assert_eq!(holders(&site, "shared"), [first]);
    // Using it twice is not a transfer and keeps the ledger.
    let mut again = work(&site, first);
    again.run(Call::Read { path: "README.md".into(), offset: None, limit: None }).unwrap();
    assert!(again.run(use_call("shared")).unwrap().starts_with("workspace demo/shared is already this conversation's workspace"));
    assert_eq!(site.bindings.load(first).unwrap().unwrap().ledger.len(), 1);
}

/// `workspace_open` and a busy other conversation. Before anything is created: refused.
/// When the other conversation binds the new workspace between its creation and the
/// opener's claim and is still working: the opener is told it is bound elsewhere. In
/// neither case is it told to whom (a conversation key can identify a person).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_opener_that_loses_the_new_workspace_is_not_told_to_whom() {
    let fixture = RepoFixture::new();
    let site = fixture.site("demo");
    let (opener, other) = ("exec-open-opener", "telegram:user:100000555");
    let other_turn = turn_lock::of(other).try_lock_owned().unwrap();
    // A record of that name left by an earlier workspace, its conversation in a turn.
    site.bindings.put(other, &Binding::new("demo", "wanted")).unwrap();
    let refused = work(&site, opener).run(Call::Open { repo: "demo".into(), name: "wanted".into(), base: None }).unwrap_err().to_string();
    assert_eq!(
        refused,
        "workspace demo/wanted was not opened: a conversation that is working right now holds a binding of that name; nothing was changed. Try again when it has finished, or choose another name"
    );
    assert!(!fixture.repository.worktree_parent.join("wanted").exists(), "nothing was created");
    site.bindings.release(other, None).unwrap();

    // The race: the opener is inside `git worktree add` when the other conversation,
    // in its turn, starts using the workspace that is just appearing.
    let (ready, release) = blocking_hook(&fixture);
    let opening = {
        let site = site.clone();
        std::thread::spawn(move || work(&site, opener).run(Call::Open { repo: "demo".into(), name: "wanted".into(), base: None }))
    };
    until("the hook", || ready.exists()).await;
    work(&site, other).run(use_call("wanted")).unwrap();
    std::fs::write(&release, b"go").unwrap();
    let lost = opening.join().unwrap().unwrap_err().to_string();
    assert_eq!(
        lost,
        "workspace demo/wanted was created, but it is bound elsewhere and not to this conversation. workspace_use can take it when the conversation that has it is idle"
    );
    for text in [&refused, &lost] {
        assert!(!text.contains("100000555") && !text.contains("telegram"), "{text}");
    }
    assert_eq!(holders(&site, "wanted"), [other]);
    assert_eq!(site.bindings.load(opener).unwrap(), None);
    // Once the other conversation is idle, the opener can take it, as the text says.
    drop(other_turn);
    assert!(work(&site, opener).run(use_call("wanted")).unwrap().contains("It was bound to another conversation"));
}
