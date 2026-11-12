//! Ordinary Git semantics from the independent W2 review. Every repository
//! and remote is a local tempfile fixture: no network, auth or live checkout.
use super::*;
use crate::workspace::{self, Workspace, git, test_support as fixtures};
use std::cell::Cell;
use std::path::PathBuf;
use std::time::Duration;

struct LocalPush {
    _fixture: fixtures::RepoFixture,
    remote: PathBuf,
    workspace: Workspace,
    launches: Cell<usize>,
}

impl LocalPush {
    fn new() -> Self {
        let fixture = fixtures::RepoFixture::new();
        let remote = fixture.top.join("remote.git");
        std::fs::create_dir(&remote).unwrap();
        fixtures::git(&remote, &["init", "--bare", "-q", "-b", "main"]);
        fixtures::git(
            &fixture.repository.checkout,
            &["remote", "add", "origin", remote.to_str().unwrap()],
        );
        fixtures::git(
            &fixture.repository.checkout,
            &["push", "origin", "refs/heads/main:refs/heads/main"],
        );
        let workspace = workspace::open(&fixture.repository, "task", None).unwrap();
        Self {
            _fixture: fixture,
            remote,
            workspace,
            launches: Cell::new(0),
        }
    }

    fn remote_ref(&self, name: &str) -> String {
        fixtures::git(&self.remote, &["rev-parse", name])
    }

    fn commit(&self, text: &str) -> String {
        fixtures::write(&self.workspace.path, "README.md", text);
        fixtures::commit_all(&self.workspace.path, text);
        fixtures::git(&self.workspace.path, &["rev-parse", "HEAD"])
    }

    /// Count real subprocess launches AFTER admission. A regression of the
    /// guard would actually perform the unsafe local push, as in the probes.
    fn push(&self, args: &[&str]) -> Result<git::Output> {
        let argv = args
            .iter()
            .map(|arg| format!("'{}'", arg.replace('\'', "'\\''")))
            .collect::<Vec<_>>()
            .join(" ");
        check(&format!("git push {argv}"), &self.workspace.base)?;
        self.launches.set(self.launches.get() + 1);
        let mut all = vec!["push"];
        all.extend_from_slice(args);
        git::run(
            &self.workspace.path,
            &all,
            "local push regression",
            Duration::from_secs(10),
        )
    }
}

#[test]
fn source_only_push_mapping_cannot_overwrite_protected_main() {
    let repo = LocalPush::new();
    let before = repo.remote_ref("refs/heads/main");
    let task = repo.commit("new task change\n");
    fixtures::git(
        &repo.workspace.path,
        &[
            "config",
            "remote.origin.push",
            "refs/heads/adapsis/task:refs/heads/main",
        ],
    );
    let decision = repo.push(&["origin", "adapsis/task"]);
    let main_changed = repo.remote_ref("refs/heads/main") != before;
    assert!(
        decision.is_err(),
        "source-only ref was admitted; protected main_changed={main_changed}"
    );
    assert!(!main_changed);
    assert_eq!(repo.launches.get(), 0, "mapping refused BEFORE launch");
    for spec in [
        "HEAD:main",
        "HEAD:refs/heads/main",
        ":refs/heads/adapsis/task",
        "HEAD:",
    ] {
        assert!(repo.push(&["origin", spec]).is_err(), "{spec}");
    }
    assert_eq!(repo.launches.get(), 0);
    // Explicit full destination overrides, rather than following, the same
    // still-configured remote.origin.push mapping.
    let allowed = repo
        .push(&["origin", "HEAD:refs/heads/adapsis/task"])
        .unwrap();
    assert!(allowed.success(), "{}", allowed.failure_text());
    assert_eq!(repo.remote_ref("refs/heads/adapsis/task"), task);
    assert_eq!(repo.remote_ref("refs/heads/main"), before);
    assert_eq!(repo.launches.get(), 1);
}

#[test]
fn plain_force_cannot_override_a_stale_lease_but_correct_lease_alone_works() {
    let repo = LocalPush::new();
    let initial = fixtures::git(&repo.workspace.path, &["rev-parse", "HEAD"]);
    let competing = repo.commit("remote competing change\n");
    fixtures::git(
        &repo.workspace.path,
        &["push", "origin", "HEAD:refs/heads/adapsis/task"],
    );
    // Diverge locally and deliberately leave the remote-tracking expectation
    // stale, exactly as the reviewer's real-Git probe did.
    fixtures::git(&repo.workspace.path, &["reset", "--hard", &initial]);
    fixtures::git(
        &repo.workspace.path,
        &["update-ref", "refs/remotes/origin/adapsis/task", &initial],
    );
    let local = repo.commit("local separate change\n");
    let destination = "HEAD:refs/heads/adapsis/task";
    for flags in [
        ["--force", "--force-with-lease"],
        ["--force-with-lease", "--force"],
        ["-f", "--force-with-lease"],
        ["--force-with-lease", "-uf"],
        ["-vf", "--force-with-lease"],
    ] {
        let decision = repo.push(&[flags[0], flags[1], "origin", destination]);
        let overwritten = repo.remote_ref("refs/heads/adapsis/task") != competing;
        assert!(
            decision.is_err(),
            "{flags:?}: stale_lease_overwritten={overwritten}"
        );
        assert!(!overwritten, "{flags:?}");
        assert_eq!(
            repo.launches.get(),
            0,
            "plain force must be refused BEFORE launch"
        );
    }
    // Admission allows lease-only; REAL Git rejects its stale expectation.
    let stale = repo
        .push(&["--force-with-lease", "origin", destination])
        .unwrap();
    assert!(
        !stale.success(),
        "Git must reject the stale lease without --force"
    );
    assert_eq!(repo.remote_ref("refs/heads/adapsis/task"), competing);
    let lease = format!("--force-with-lease=refs/heads/adapsis/task:{competing}");
    let correct = repo.push(&[&lease, "origin", destination]).unwrap();
    assert!(correct.success(), "{}", correct.failure_text());
    assert_eq!(repo.remote_ref("refs/heads/adapsis/task"), local);
    assert_eq!(repo.remote_ref("refs/heads/main"), initial);
    assert_eq!(
        repo.launches.get(),
        2,
        "only stale-lease and correct-lease controls were launched"
    );
}
