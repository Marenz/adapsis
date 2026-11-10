use std::os::unix::fs::PermissionsExt;
use std::time::{Duration, Instant};

use super::error::Error;
use super::lifecycle::{GitOptions, close, find, open, open_with, status, validate_base, validate_name};
use super::limits::{DIRTY_LIST_MAX, MAX_WORKTREES_PER_REPO, NAME_MAX_LEN};
use super::test_support::{RepoFixture, commit_all, git, write};

fn lifecycle_text(error: Error) -> String {
    match error {
        Error::Lifecycle(text) => text,
        other => panic!("expected a lifecycle refusal, got {other:?}"),
    }
}

#[test]
fn lifecycle_open_creates_branch_and_directory() {
    let fixture = RepoFixture::new();
    let workspace = open(&fixture.repository, "fix-parser", None).unwrap();
    assert_eq!(workspace.repo, "demo");
    assert_eq!(workspace.name, "fix-parser");
    assert_eq!(workspace.branch, "adapsis/fix-parser");
    assert_eq!(workspace.base, "refs/heads/main");
    assert_eq!(workspace.path, fixture.top.join("worktrees/fix-parser"));
    assert!(workspace.path.join("README.md").is_file());
    assert!(workspace.path.join(".git").is_file(), "a worktree has a .git file");
    let branches = git(&fixture.repository.checkout, &["branch", "--list", "adapsis/*", "--format=%(refname:short)"]);
    assert_eq!(branches, "adapsis/fix-parser");
    let current = status(&workspace).unwrap();
    assert_eq!(current.branch.as_deref(), Some("adapsis/fix-parser"));
    assert_eq!((current.ahead, current.behind, current.dirty_count), (Some(0), Some(0), 0));
    assert!(current.render().contains("clean: no uncommitted changes"), "{}", current.render());
}

#[test]
fn lifecycle_open_writes_exactly_one_config_key_and_git_drops_it_with_the_branch() {
    let fixture = RepoFixture::new();
    git(&fixture.repository.checkout, &["config", "--local", "user.name", "Owner"]);
    let before = fixture.config();
    let workspace = open(&fixture.repository, "one-key", None).unwrap();
    let after_open = fixture.config();
    let added: Vec<&String> = after_open.iter().filter(|line| !before.contains(line)).collect();
    assert_eq!(added, vec!["branch.adapsis/one-key.adapsis-base=refs/heads/main"]);
    assert!(before.iter().all(|line| after_open.contains(line)), "an existing key changed: {before:?} -> {after_open:?}");

    close(&workspace).unwrap();
    assert_eq!(fixture.config(), after_open, "close keeps the key while the branch exists");

    git(&fixture.repository.checkout, &["branch", "-D", "adapsis/one-key"]);
    assert_eq!(fixture.config(), before, "the key goes with the branch");
}

#[test]
fn lifecycle_stale_base_key_is_pruned_and_nothing_else_is_touched() {
    let fixture = RepoFixture::new();
    let checkout = &fixture.repository.checkout;
    git(checkout, &["config", "--local", "branch.adapsis/ghost.adapsis-base", "refs/heads/main"]);
    git(checkout, &["config", "--local", "branch.adapsis/ghost.description", "owner note"]);
    git(checkout, &["config", "--local", "branch.main.adapsis-base", "not ours"]);
    git(checkout, &["config", "--local", "branch.adapsis/Other_Name.adapsis-base", "not a workspace name"]);
    let before = fixture.config();
    let workspace = open(&fixture.repository, "real", None).unwrap();
    let after = fixture.config();
    let removed: Vec<&String> = before.iter().filter(|line| !after.contains(line)).collect();
    let added: Vec<&String> = after.iter().filter(|line| !before.contains(line)).collect();
    assert_eq!(removed, vec!["branch.adapsis/ghost.adapsis-base=refs/heads/main"]);
    assert_eq!(added, vec!["branch.adapsis/real.adapsis-base=refs/heads/main"]);
    // An absent key is tolerated: status and close still work.
    git(checkout, &["config", "--local", "--unset", "branch.adapsis/real.adapsis-base"]);
    assert_eq!(find(&fixture.repository, "real").unwrap().base, "main", "falls back to the repository default");
    close(&workspace).unwrap();
}

#[test]
fn lifecycle_find_tolerates_an_unusable_stored_base() {
    let fixture = RepoFixture::new();
    let opened = open(&fixture.repository, "stored", None).unwrap();
    assert_eq!(find(&fixture.repository, "stored").unwrap(), opened);
    git(&fixture.repository.checkout, &["config", "--local", "branch.adapsis/stored.adapsis-base", "--upload-pack=evil"]);
    let found = find(&fixture.repository, "stored").unwrap();
    assert_eq!(found.base, "main");
    assert_eq!(found.path, opened.path);
    // A stored base that no longer resolves gives an honest status.
    git(&fixture.repository.checkout, &["config", "--local", "branch.adapsis/stored.adapsis-base", "refs/heads/gone"]);
    let current = status(&find(&fixture.repository, "stored").unwrap()).unwrap();
    assert_eq!((current.ahead, current.behind), (None, None));
    assert!(current.render().contains("no longer resolves"), "{}", current.render());
}

#[test]
fn lifecycle_open_with_explicit_base_pins_a_commit() {
    let fixture = RepoFixture::new();
    let checkout = &fixture.repository.checkout;
    let first = git(checkout, &["rev-parse", "HEAD"]);
    write(checkout, "second.txt", "2\n");
    commit_all(checkout, "second");
    let workspace = open(&fixture.repository, "older", Some("HEAD~1")).unwrap();
    assert_eq!(workspace.base, first, "a relative revision is stored as the commit it named");
    assert!(!workspace.path.join("second.txt").exists());
    let current = status(&workspace).unwrap();
    assert_eq!((current.ahead, current.behind), (Some(0), Some(0)));
    assert_eq!(find(&fixture.repository, "older").unwrap().base, first);
}

#[test]
fn lifecycle_duplicate_name_is_refused() {
    let fixture = RepoFixture::new();
    let workspace = open(&fixture.repository, "twice", None).unwrap();
    let text = lifecycle_text(open(&fixture.repository, "twice", None).unwrap_err());
    assert!(text.contains("branch `adapsis/twice` already exists") && text.contains("find"), "{text}");
    close(&workspace).unwrap();
    let text = lifecycle_text(open(&fixture.repository, "twice", None).unwrap_err());
    assert!(text.contains("a closed workspace keeps its branch"), "{text}");
}

#[test]
fn lifecycle_existing_directory_is_never_overwritten() {
    let fixture = RepoFixture::new();
    let precious = write(&fixture.repository.worktree_parent, "taken/precious.txt", "keep\n");
    let text = lifecycle_text(open(&fixture.repository, "taken", None).unwrap_err());
    assert!(text.contains("already exists and is never overwritten"), "{text}");
    assert_eq!(std::fs::read_to_string(precious).unwrap(), "keep\n");
    assert_eq!(git(&fixture.repository.checkout, &["branch", "--list", "adapsis/*"]), "", "no branch is left behind");
}

#[test]
fn lifecycle_worktree_limit_counts_only_workspaces() {
    let fixture = RepoFixture::new();
    // A worktree the owner made by hand, in the same parent, does not count.
    std::fs::create_dir_all(&fixture.repository.worktree_parent).unwrap();
    let manual = fixture.repository.worktree_parent.join("manual");
    git(&fixture.repository.checkout, &["worktree", "add", "-q", "-b", "owner-branch", manual.to_str().unwrap()]);
    let mut opened = Vec::new();
    for index in 0..MAX_WORKTREES_PER_REPO {
        opened.push(open(&fixture.repository, &format!("ws-{index}"), None).unwrap());
    }
    let text = lifecycle_text(open(&fixture.repository, "one-too-many", None).unwrap_err());
    assert!(text.contains(&format!("already has {MAX_WORKTREES_PER_REPO} workspaces (limit {MAX_WORKTREES_PER_REPO})")), "{text}");
    assert!(text.contains("ws-0, ws-1, ws-2, ws-3") && text.contains("Close one first"), "{text}");
    assert!(!fixture.repository.worktree_parent.join("one-too-many").exists());
    // A workspace that switched to another branch still counts.
    git(&opened[0].path, &["checkout", "-q", "-b", "side"]);
    assert!(open(&fixture.repository, "still-too-many", None).is_err());
    close(&opened[1]).unwrap();
    open(&fixture.repository, "fits-again", None).unwrap();
}

#[test]
fn lifecycle_names_are_validated() {
    for bad in ["", "Bad", "a/b", "-x", "x-", "a b", "..", "a.b", "ä", "a\0b", &"x".repeat(NAME_MAX_LEN + 1)] {
        assert!(matches!(validate_name("workspace name", bad), Err(Error::InvalidName { .. })), "{bad:?} was accepted");
    }
    for good in ["a", "fix-129", "0day", &"x".repeat(NAME_MAX_LEN)] {
        validate_name("workspace name", good).unwrap();
    }
    let fixture = RepoFixture::new();
    let error = open(&fixture.repository, "../escape", None).unwrap_err();
    assert!(error.to_string().contains("may only contain lowercase letters, digits and dashes"), "{error}");
    assert!(!fixture.repository.worktree_parent.exists(), "nothing is created for a refused name");
}

#[test]
fn lifecycle_base_is_validated_and_must_be_a_commit() {
    for bad in ["", "-x", "--upload-pack=x", "a..b", "main; rm -rf /", "a b", "x:y", "main\n", &"a".repeat(201)] {
        assert!(matches!(validate_base(bad), Err(Error::InvalidName { .. })), "{bad:?} was accepted");
    }
    for good in ["main", "origin/main", "HEAD~3", "v1.0^{commit}", "@{u}", "refs/heads/main", &"a".repeat(40)] {
        validate_base(good).unwrap();
    }
    let fixture = RepoFixture::new();
    let text = lifecycle_text(open(&fixture.repository, "nobase", Some("no-such-branch")).unwrap_err());
    assert!(text.contains("base `no-such-branch` is not a commit"), "{text}");
    assert!(!fixture.repository.worktree_parent.join("nobase").exists());
}

#[test]
fn lifecycle_find_reports_unknown_workspaces() {
    let fixture = RepoFixture::new();
    let text = lifecycle_text(find(&fixture.repository, "absent").unwrap_err());
    assert!(text.contains("no workspace `absent`") && text.contains("none are open"), "{text}");
    open(&fixture.repository, "present", None).unwrap();
    let text = lifecycle_text(find(&fixture.repository, "absent").unwrap_err());
    assert!(text.contains("open: present"), "{text}");
    // The main checkout is never a workspace, whatever its directory is called.
    assert!(find(&fixture.repository, "checkout").is_err());
}

#[test]
fn lifecycle_status_reports_changes_and_distance_from_base() {
    let fixture = RepoFixture::new();
    let workspace = open(&fixture.repository, "busy", None).unwrap();
    write(&workspace.path, "README.md", "changed\n");
    write(&workspace.path, "new file.txt", "new\n");
    let current = status(&workspace).unwrap();
    assert_eq!(current.dirty_count, 2);
    assert_eq!(current.dirty, vec![" M README.md", "?? new file.txt"]);
    assert!(current.render().contains("uncommitted changes: 2 file(s)\n   M README.md\n  ?? new file.txt"), "{}", current.render());

    commit_all(&workspace.path, "work");
    write(&fixture.repository.checkout, "upstream.txt", "u\n");
    commit_all(&fixture.repository.checkout, "upstream");
    let current = status(&workspace).unwrap();
    assert_eq!((current.ahead, current.behind, current.dirty_count), (Some(1), Some(1), 0));
    assert!(current.render().contains("base: refs/heads/main (ahead 1, behind 1)"), "{}", current.render());

    git(&workspace.path, &["mv", "README.md", "READ.md"]);
    let current = status(&workspace).unwrap();
    assert_eq!(current.dirty, vec!["R  READ.md"], "a rename is one entry");

    git(&workspace.path, &["checkout", "-q", "--detach"]);
    let current = status(&workspace).unwrap();
    assert_eq!(current.branch, None);
    assert!(current.render().contains("detached HEAD (opened as adapsis/busy)"), "{}", current.render());
}

#[test]
fn lifecycle_close_refuses_a_dirty_worktree_with_the_file_list() {
    let fixture = RepoFixture::new();
    let workspace = open(&fixture.repository, "dirty", None).unwrap();
    write(&workspace.path, "README.md", "changed\n");
    write(&workspace.path, "scratch.txt", "x\n");
    let error = close(&workspace).unwrap_err();
    match &error {
        Error::Dirty { workspace, files, omitted } => {
            assert_eq!(workspace, "dirty");
            assert_eq!(files, &vec![" M README.md".to_string(), "?? scratch.txt".to_string()]);
            assert_eq!(*omitted, 0);
        }
        other => panic!("expected Dirty, got {other:?}"),
    }
    let text = error.to_string();
    assert!(text.contains("was not closed") && text.contains("\n   M README.md\n  ?? scratch.txt"), "{text}");
    assert_eq!(std::fs::read_to_string(workspace.path.join("scratch.txt")).unwrap(), "x\n", "nothing was removed");
}

#[test]
fn lifecycle_close_bounds_the_dirty_list() {
    let fixture = RepoFixture::new();
    let workspace = open(&fixture.repository, "many", None).unwrap();
    for index in 0..DIRTY_LIST_MAX + 7 {
        write(&workspace.path, &format!("untracked-{index:03}.txt"), "x\n");
    }
    match close(&workspace).unwrap_err() {
        Error::Dirty { files, omitted, .. } => assert_eq!((files.len(), omitted), (DIRTY_LIST_MAX, 7)),
        other => panic!("expected Dirty, got {other:?}"),
    }
}

#[test]
fn lifecycle_close_removes_a_clean_worktree_and_keeps_the_branch() {
    let fixture = RepoFixture::new();
    let workspace = open(&fixture.repository, "done", None).unwrap();
    write(&workspace.path, "work.txt", "result\n");
    commit_all(&workspace.path, "work");
    // Ignored build output does not block closing.
    write(&workspace.path, ".git-ignored-by-info", "x\n");
    let exclude = git(&workspace.path, &["rev-parse", "--git-path", "info/exclude"]);
    write(&workspace.path, &exclude, ".git-ignored-by-info\n");
    close(&workspace).unwrap();
    assert!(!workspace.path.exists());
    let log = git(&fixture.repository.checkout, &["log", "--format=%s", "-1", "adapsis/done"]);
    assert_eq!(log, "work", "the commit survives on the branch");
    assert!(find(&fixture.repository, "done").is_err());
    assert!(close(&workspace).is_err(), "closing twice is an error, not a panic");
}

#[test]
fn lifecycle_git_failure_is_an_error() {
    let fixture = RepoFixture::new();
    let mut not_a_repo = fixture.repository.clone();
    not_a_repo.checkout = fixture.top.join("plain");
    std::fs::create_dir_all(&not_a_repo.checkout).unwrap();
    let error = open(&not_a_repo, "x", None).unwrap_err();
    assert!(matches!(error, Error::Git { .. }), "{error:?}");
    assert!(error.to_string().to_lowercase().contains("not a git repository"), "{error}");

    let mut missing = fixture.repository.clone();
    missing.checkout = fixture.top.join("absent");
    let text = lifecycle_text(open(&missing, "x", None).unwrap_err());
    assert!(text.contains("does not exist or is not a directory") && text.contains("fix the workspace registry"), "{text}");
    assert!(find(&missing, "x").is_err());
}

#[test]
fn lifecycle_git_timeout_is_an_error_not_a_hang() {
    let fixture = RepoFixture::new();
    let hook = fixture.repository.checkout.join(".git/hooks/post-checkout");
    std::fs::write(&hook, "#!/bin/sh\nexec sleep 600\n").unwrap();
    std::fs::set_permissions(&hook, std::fs::Permissions::from_mode(0o755)).unwrap();
    let options = GitOptions { query: Duration::from_secs(120), worktree: Duration::from_millis(1500) };
    // Far below the hook's 600 s, far above the timeout: robust on a loaded machine.
    let started = Instant::now();
    let text = lifecycle_text(open_with(&fixture.repository, "slow", None, options).unwrap_err());
    assert!(started.elapsed() < Duration::from_secs(240), "took {:?}", started.elapsed());
    assert!(text.contains("did not finish within 1500 ms") && text.contains("may be partly created"), "{text}");
}
