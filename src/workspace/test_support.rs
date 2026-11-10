//! Fixtures for the workspace tests. Everything lives in a temp directory;
//! repositories are created with `git init` there. No test touches a real
//! checkout or the developer's git configuration (see `git::base_command`).

use std::path::{Path, PathBuf};
use std::time::Duration;

use tempfile::TempDir;

use super::git;
use super::lifecycle::Workspace;
use super::registry::Repository;

/// A temp directory with a workspace root `ws`, a sibling `outside` and a
/// read-only root `extra`. Paths are canonical.
pub struct Fixture {
    pub temp: TempDir,
    pub top: PathBuf,
    pub root: PathBuf,
    pub outside: PathBuf,
    pub extra: PathBuf,
    pub workspace: Workspace,
}

impl Fixture {
    /// A plain directory as workspace root (no git).
    pub fn plain() -> Self {
        Self::build(false)
    }

    /// The workspace root is a git repository with one commit.
    pub fn git() -> Self {
        Self::build(true)
    }

    fn build(with_git: bool) -> Self {
        let temp = tempfile::tempdir().expect("temp dir");
        let top = std::fs::canonicalize(temp.path()).expect("canonical temp dir");
        let (root, outside, extra) = (top.join("ws"), top.join("outside"), top.join("extra"));
        for directory in [&root, &outside, &extra] {
            std::fs::create_dir_all(directory).expect("fixture directory");
        }
        if with_git {
            init_repo(&root);
        }
        let repository = Repository {
            name: "fixture".to_string(),
            checkout: root.clone(),
            worktree_parent: top.join("worktrees"),
            base: "main".to_string(),
            read_roots: vec![extra.clone()],
            env: Default::default(),
        };
        let workspace = Workspace::at(&repository, "fixture", &root, "main").expect("workspace");
        Self { temp, top, root, outside, extra, workspace }
    }

    pub fn write(&self, relative: &str, content: impl AsRef<[u8]>) -> PathBuf {
        write(&self.root, relative, content)
    }
}

pub fn write(base: &Path, relative: &str, content: impl AsRef<[u8]>) -> PathBuf {
    let path = base.join(relative);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("parent directory");
    }
    std::fs::write(&path, content).expect("write fixture file");
    path
}

pub fn symlink(target: impl AsRef<Path>, link: impl AsRef<Path>) {
    std::os::unix::fs::symlink(target, link).expect("symlink");
}

/// Run git in `directory` with the hermetic environment; panics on failure.
pub fn git(directory: &Path, args: &[&str]) -> String {
    let mut command = git::base_command(directory);
    command.args(["-c", "user.name=Test", "-c", "user.email=test@example.invalid"]).args(args);
    let output = git::run_command(command, "test fixture", Duration::from_secs(60)).expect("git runs");
    assert!(output.success(), "git {args:?} failed: {}", output.failure_text());
    output.stdout_text()
}

/// `git init` with branch `main` and one commit.
pub fn init_repo(directory: &Path) {
    std::fs::create_dir_all(directory).expect("repository directory");
    git(directory, &["init", "-q", "-b", "main"]);
    write(directory, "README.md", "fixture\n");
    commit_all(directory, "initial");
}

pub fn commit_all(directory: &Path, message: &str) {
    git(directory, &["add", "-A"]);
    git(directory, &["commit", "-q", "-m", message]);
}

/// A registered repository in a temp directory: `checkout` is a fresh git
/// repository, `worktrees` its (not yet existing) worktree parent.
pub struct RepoFixture {
    pub temp: TempDir,
    pub top: PathBuf,
    pub repository: Repository,
}

impl RepoFixture {
    pub fn new() -> Self {
        let temp = tempfile::tempdir().expect("temp dir");
        let top = std::fs::canonicalize(temp.path()).expect("canonical temp dir");
        let checkout = top.join("checkout");
        init_repo(&checkout);
        let repository = Repository {
            name: "demo".to_string(),
            checkout,
            worktree_parent: top.join("worktrees"),
            base: "main".to_string(),
            read_roots: Vec::new(),
            env: Default::default(),
        };
        Self { temp, top, repository }
    }

    /// The checkout's local git configuration, sorted, one `key=value` per line.
    pub fn config(&self) -> Vec<String> {
        let mut lines: Vec<String> = git(&self.repository.checkout, &["config", "--local", "--list"]).lines().map(str::to_string).collect();
        lines.sort();
        lines
    }
}
