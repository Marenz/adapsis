//! Workspace lifecycle: a workspace is one `git worktree` of a registered
//! repository on its own branch `adapsis/<name>` (DESIGN T4).
//!
//! * `open` creates branch and worktree; it never forces and never reuses.
//! * `find` re-binds to a worktree that `open` created earlier (after a restart).
//! * `status` reports branch, ahead/behind of the base and uncommitted files.
//! * `close` removes a clean worktree and keeps the branch, so commits survive.
//!   It refuses while background jobs of the workspace are running.
//!
//! The base a workspace was opened from is the one fact that has to outlive
//! the process. It is kept in the checkout's git configuration as
//! `branch.adapsis/<name>.adapsis-base`, the only key this module ever writes.
//! git drops it together with the branch; a key whose branch is gone anyway is
//! removed on the next `open`/`close`, and an absent or unusable value falls
//! back to the repository's default base.

use std::collections::HashSet;
use std::ffi::OsStr;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use super::error::{Error, Result, shown};
use super::git;
use super::jobs::JobStore;
use super::limits::{
    BASE_REF_MAX_LEN, DIRTY_LIST_MAX, GIT_TIMEOUT, GIT_WORKTREE_TIMEOUT, MAX_WORKTREES_PER_REPO, NAME_MAX_LEN,
};
use super::registry::Repository;

/// Branch namespace of every workspace.
pub const BRANCH_PREFIX: &str = "adapsis/";
const BASE_KEY_SUFFIX: &str = ".adapsis-base";

/// Serializes `open` and `close` within this process, so two concurrent opens
/// cannot both pass the worktree limit.
static LIFECYCLE: Mutex<()> = Mutex::new(());

/// A bound workspace. `path` and `read_roots` are canonical (symlink-free);
/// the path rules in `resolve` rely on that.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Workspace {
    pub repo: String,
    pub name: String,
    pub path: PathBuf,
    /// The branch `open` created: `adapsis/<name>`.
    pub branch: String,
    /// The ref ahead/behind is measured against.
    pub base: String,
    /// The registered checkout the worktree belongs to.
    pub checkout: PathBuf,
    pub read_roots: Vec<PathBuf>,
    /// The repository's `env` table: added to every command's environment.
    pub env: Vec<(String, String)>,
}

impl Workspace {
    /// Bind to an existing directory. Read roots that do not exist are left
    /// out: a missing optional directory must not make the workspace unusable.
    pub fn at(repo: &Repository, name: &str, path: &Path, base: &str) -> Result<Self> {
        let path = std::fs::canonicalize(path)
            .map_err(|error| Error::Lifecycle(format!("workspace `{name}`: its directory cannot be used ({})", error.kind())))?;
        if !path.is_dir() {
            return Err(Error::Lifecycle(format!("workspace `{name}`: its path is not a directory")));
        }
        let read_roots = repo
            .read_roots
            .iter()
            .filter_map(|root| std::fs::canonicalize(root).ok())
            .filter(|root| root.is_dir() && root != Path::new("/"))
            .collect();
        Ok(Self {
            repo: repo.name.clone(),
            name: name.to_string(),
            path,
            branch: format!("{BRANCH_PREFIX}{name}"),
            base: base.to_string(),
            checkout: repo.checkout.clone(),
            read_roots,
            env: repo.env.iter().map(|(name, value)| (name.clone(), value.clone())).collect(),
        })
    }
}

/// Timeouts of the git calls; the defaults are the policy values.
#[derive(Debug, Clone, Copy)]
pub struct GitOptions {
    pub query: Duration,
    pub worktree: Duration,
}

impl Default for GitOptions {
    fn default() -> Self {
        Self { query: GIT_TIMEOUT, worktree: GIT_WORKTREE_TIMEOUT }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceStatus {
    pub workspace: String,
    /// The branch currently checked out; `None` when HEAD is detached.
    pub branch: Option<String>,
    pub expected_branch: String,
    pub base: String,
    /// Commits on HEAD that the base lacks / on the base that HEAD lacks.
    /// `None` when the base no longer resolves.
    pub ahead: Option<u64>,
    pub behind: Option<u64>,
    pub dirty_count: usize,
    /// `XY path` entries as git reports them, at most `DIRTY_LIST_MAX`.
    pub dirty: Vec<String>,
}

impl WorkspaceStatus {
    pub fn render(&self) -> String {
        let mut out = format!("workspace {}\n", self.workspace);
        match &self.branch {
            Some(branch) if *branch == self.expected_branch => out.push_str(&format!("branch: {branch}\n")),
            Some(branch) => out.push_str(&format!("branch: {branch} (opened as {})\n", self.expected_branch)),
            None => out.push_str(&format!("branch: detached HEAD (opened as {})\n", self.expected_branch)),
        }
        match (self.ahead, self.behind) {
            (Some(ahead), Some(behind)) => out.push_str(&format!("base: {} (ahead {ahead}, behind {behind})\n", self.base)),
            _ => out.push_str(&format!("base: {} (no longer resolves; ahead/behind unknown)\n", self.base)),
        }
        if self.dirty_count == 0 {
            out.push_str("clean: no uncommitted changes");
        } else {
            out.push_str(&format!("uncommitted changes: {} file(s)", self.dirty_count));
            for entry in &self.dirty {
                out.push_str(&format!("\n  {entry}"));
            }
            if self.dirty_count > self.dirty.len() {
                out.push_str(&format!("\n  … and {} more", self.dirty_count - self.dirty.len()));
            }
        }
        out
    }
}

/// Lowercase letters, digits and dashes; starts and ends with a letter or digit.
pub fn validate_name(what: &'static str, name: &str) -> Result<()> {
    let invalid = |reason: &str| Err(Error::InvalidName { what, reason: format!("`{}` {reason}", shown(name)) });
    if name.is_empty() {
        return invalid("is empty");
    }
    if name.len() > NAME_MAX_LEN {
        return invalid(&format!("is longer than {NAME_MAX_LEN} characters"));
    }
    if !name.bytes().all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-') {
        return invalid("may only contain lowercase letters, digits and dashes");
    }
    if name.starts_with('-') || name.ends_with('-') {
        return invalid("must start and end with a letter or digit");
    }
    Ok(())
}

/// A base is a single revision spelled with ref characters. It cannot be an
/// option (`-x`), a range (`a..b`) or a `rev:path`.
pub fn validate_base(base: &str) -> Result<()> {
    let invalid = |reason: &str| Err(Error::InvalidName { what: "base ref", reason: format!("`{}` {reason}", shown(base)) });
    if base.is_empty() {
        return invalid("is empty");
    }
    if base.len() > BASE_REF_MAX_LEN {
        return invalid(&format!("is longer than {BASE_REF_MAX_LEN} characters"));
    }
    if base.starts_with('-') {
        return invalid("must not start with a dash");
    }
    if base.contains("..") {
        return invalid("must name one revision, not a range");
    }
    if !base.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"/_.-~^@{}+".contains(&byte)) {
        return invalid("contains characters that are not allowed in a ref (allowed: letters, digits and / _ . - ~ ^ @ { } +)");
    }
    Ok(())
}

pub fn open(repo: &Repository, name: &str, base: Option<&str>) -> Result<Workspace> {
    open_with(repo, name, base, GitOptions::default())
}

pub fn open_with(repo: &Repository, name: &str, base: Option<&str>, options: GitOptions) -> Result<Workspace> {
    validate_name("workspace name", name)?;
    let base = base.unwrap_or(&repo.base);
    validate_base(base)?;
    let _guard = LIFECYCLE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let state = RepoState::read(repo, options)?;
    prune_keys_in(&repo.checkout, &state.branches, options);
    let branch = format!("{BRANCH_PREFIX}{name}");
    if state.branches.contains(&branch) {
        let hint = match state.worktree_of(repo, name) {
            Some(_) => "its worktree still exists: use that workspace instead of opening it again",
            None => "a closed workspace keeps its branch: choose another name, or delete the branch first",
        };
        return Err(Error::Lifecycle(format!(
            "workspace `{name}` cannot be opened: branch `{branch}` already exists in repository `{}` ({hint})",
            repo.name
        )));
    }
    let directory = repo.worktree_parent.join(name);
    if std::fs::symlink_metadata(&directory).is_ok() {
        return Err(Error::Lifecycle(format!(
            "workspace `{name}` cannot be opened: {} already exists and is never overwritten; choose another name",
            directory.display()
        )));
    }
    let managed = state.managed(repo);
    if managed.len() >= MAX_WORKTREES_PER_REPO {
        return Err(Error::Lifecycle(format!(
            "workspace `{name}` cannot be opened: repository `{}` already has {} workspaces (limit {MAX_WORKTREES_PER_REPO}): {}. Close one first",
            repo.name,
            managed.len(),
            managed.join(", ")
        )));
    }
    let stored_base = stable_base(repo, name, base, options)?;
    std::fs::create_dir_all(&repo.worktree_parent).map_err(|error| {
        Error::Lifecycle(format!("worktree parent {} cannot be created ({})", repo.worktree_parent.display(), error.kind()))
    })?;
    let args: [&OsStr; 7] = [
        OsStr::new("worktree"),
        OsStr::new("add"),
        OsStr::new("--no-track"),
        OsStr::new("-b"),
        OsStr::new(&branch),
        directory.as_os_str(),
        OsStr::new(base),
    ];
    git::run_ok(&repo.checkout, &args, "worktree add", options.worktree).map_err(|error| match error {
        Error::GitTimeout { action, millis } => Error::Lifecycle(format!(
            "git {action} did not finish within {millis} ms and was stopped; branch `{branch}` or its directory may be partly created (inspect with `git worktree list`)"
        )),
        other => other,
    })?;
    // Best effort: without the key, find() falls back to the default base.
    let _ = git::run(&repo.checkout, &["config", "--local", &base_key(name), &stored_base], "config", options.query);
    Workspace::at(repo, name, &directory, &stored_base)
}

/// Verify that `base` names a commit and turn it into a spelling that means
/// the same thing in every worktree and at every later time: the full ref name
/// when it is a ref (ahead/behind then follows the ref as it moves), otherwise
/// the commit id (`HEAD~1` would mean something else inside the new worktree).
fn stable_base(repo: &Repository, name: &str, base: &str, options: GitOptions) -> Result<String> {
    let commit = format!("{base}^{{commit}}");
    let verified = git::run(&repo.checkout, &["rev-parse", "--verify", "--quiet", "--end-of-options", &commit], "rev-parse", options.query)?;
    let id = verified.stdout_text();
    if !verified.success() || id.len() < 40 || !id.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(Error::Lifecycle(format!(
            "workspace `{name}` cannot be opened: base `{base}` is not a commit in repository `{}`",
            repo.name
        )));
    }
    // No `--end-of-options` here: rev-parse would echo it. `base` cannot be
    // an option, validate_base refused a leading dash.
    let symbolic = git::run(&repo.checkout, &["rev-parse", "--symbolic-full-name", base], "rev-parse", options.query)?;
    let full = symbolic.stdout_text();
    if symbolic.success() && full.starts_with("refs/") && validate_base(&full).is_ok() { Ok(full) } else { Ok(id) }
}

pub fn find(repo: &Repository, name: &str) -> Result<Workspace> {
    find_with(repo, name, GitOptions::default())
}

pub fn find_with(repo: &Repository, name: &str, options: GitOptions) -> Result<Workspace> {
    validate_name("workspace name", name)?;
    let state = RepoState::read(repo, options)?;
    let Some(entry) = state.worktree_of(repo, name) else {
        let managed = state.managed(repo);
        let known = if managed.is_empty() { "none are open".to_string() } else { format!("open: {}", managed.join(", ")) };
        return Err(Error::NotOpen(format!("no workspace `{name}` in repository `{}` ({known}); open it first", repo.name)));
    };
    if !entry.path.is_dir() {
        return Err(Error::Lifecycle(format!(
            "workspace `{name}` is registered with git but its directory is gone; run `git worktree prune` in the checkout, then open it under a new name"
        )));
    }
    let stored = git::run(&repo.checkout, &["config", "--local", "--get", &base_key(name)], "config", options.query)
        .ok()
        .filter(git::Output::success)
        .map(|output| output.stdout_text())
        .filter(|base| validate_base(base).is_ok());
    let base = stored.unwrap_or_else(|| repo.base.clone());
    Workspace::at(repo, name, &entry.path, &base)
}

/// Names of the open workspaces of `repo` (the ones that count against the
/// limit), in git's worktree order.
pub fn list(repo: &Repository) -> Result<Vec<String>> {
    Ok(RepoState::read(repo, GitOptions::default())?.managed(repo))
}

pub fn status(workspace: &Workspace) -> Result<WorkspaceStatus> {
    status_with(workspace, GitOptions::default())
}

pub fn status_with(workspace: &Workspace, options: GitOptions) -> Result<WorkspaceStatus> {
    if !workspace.path.is_dir() {
        return Err(Error::Lifecycle(format!(
            "workspace `{}`: its directory is gone (closed, or removed by hand; `git worktree prune` in the checkout clears a leftover entry)",
            workspace.name
        )));
    }
    let head = git::run(&workspace.path, &["symbolic-ref", "--short", "-q", "HEAD"], "symbolic-ref", options.query)?;
    let branch = Some(head.stdout_text()).filter(|name| head.success() && !name.is_empty());
    let listing = git::run_ok(
        &workspace.path,
        &["status", "--porcelain=v1", "-z", "--untracked-files=normal"],
        "status",
        options.query,
    )?;
    let all = dirty_entries(&listing.stdout);
    let dirty_count = all.len();
    let range = format!("{}...HEAD", workspace.base);
    let counts = if validate_base(&workspace.base).is_ok() {
        git::run(&workspace.path, &["rev-list", "--left-right", "--count", "--end-of-options", &range], "rev-list", options.query)
            .ok()
            .filter(git::Output::success)
            .and_then(|output| ahead_behind(&output.stdout_text()))
    } else {
        None
    };
    Ok(WorkspaceStatus {
        workspace: workspace.name.clone(),
        branch,
        expected_branch: workspace.branch.clone(),
        base: workspace.base.clone(),
        ahead: counts.map(|(_, ahead)| ahead),
        behind: counts.map(|(behind, _)| behind),
        dirty_count,
        dirty: all.into_iter().take(DIRTY_LIST_MAX).collect(),
    })
}

pub fn close(workspace: &Workspace) -> Result<()> {
    close_with(workspace, GitOptions::default())
}

/// Remove the worktree if it has no uncommitted changes. The branch stays.
/// Ignored files (build output) do not count as changes and are deleted.
pub fn close_with(workspace: &Workspace, options: GitOptions) -> Result<()> {
    close_with_store(workspace, options, &JobStore::standard())
}

/// The same close transaction for an explicitly configured private state root.
pub fn close_with_store(workspace: &Workspace, options: GitOptions, jobs: &JobStore) -> Result<()> {
    let _guard = LIFECYCLE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if std::fs::canonicalize(&workspace.checkout).is_ok_and(|checkout| checkout == workspace.path) {
        return Err(Error::Lifecycle(format!("workspace `{}` is the registered checkout itself and is never removed", workspace.name)));
    }
    let reservation = super::reservation::for_workspace(workspace);
    let _workspace_guard = reservation.lock().unwrap_or_else(|p| p.into_inner());
    jobs.close_guard_locked(workspace)?;
    let current = status_with(workspace, options)?;
    if current.dirty_count > 0 {
        return Err(Error::Dirty {
            workspace: workspace.name.clone(),
            omitted: current.dirty_count - current.dirty.len(),
            files: current.dirty,
        });
    }
    let args: [&OsStr; 3] = [OsStr::new("worktree"), OsStr::new("remove"), workspace.path.as_os_str()];
    git::run_ok(&workspace.checkout, &args, "worktree remove", options.worktree)?;
    jobs.forget_locked(workspace)?;
    if let Ok(branches) = adapsis_branches(&workspace.checkout, options) {
        prune_keys_in(&workspace.checkout, &branches, options);
    }
    Ok(())
}

/// What git knows about a repository's worktrees and `adapsis/*` branches.
struct RepoState {
    worktrees: Vec<WorktreeEntry>,
    branches: HashSet<String>,
}

struct WorktreeEntry {
    path: PathBuf,
    branch: Option<String>,
}

impl RepoState {
    fn read(repo: &Repository, options: GitOptions) -> Result<Self> {
        if !repo.checkout.is_dir() {
            return Err(Error::Lifecycle(format!(
                "repository `{}`: its checkout {} does not exist or is not a directory; fix the workspace registry",
                repo.name,
                repo.checkout.display()
            )));
        }
        let listing = git::run_ok(&repo.checkout, &["worktree", "list", "--porcelain", "-z"], "worktree list", options.query)?;
        Ok(Self { worktrees: worktree_entries(&listing.stdout), branches: adapsis_branches(&repo.checkout, options)? })
    }

    /// The worktree at `<parent>/<name>`, wherever its HEAD currently is.
    fn worktree_of(&self, repo: &Repository, name: &str) -> Option<&WorktreeEntry> {
        let wanted = real(&repo.worktree_parent.join(name));
        self.worktrees.iter().skip(1).find(|entry| real(&entry.path) == wanted)
    }

    /// Names of the worktrees that count against the limit: every linked
    /// worktree on an `adapsis/*` branch, and every directory `<parent>/<n>`
    /// whose branch `adapsis/<n>` exists (a workspace switched to another
    /// branch is still a workspace). Worktrees the owner made by hand do not
    /// count, even when they sit in the same parent directory.
    fn managed(&self, repo: &Repository) -> Vec<String> {
        let parent = real(&repo.worktree_parent);
        let mut names: Vec<String> = Vec::new();
        for entry in self.worktrees.iter().skip(1) {
            let by_branch = entry.branch.as_deref().and_then(|branch| branch.strip_prefix(BRANCH_PREFIX));
            let path = real(&entry.path);
            let by_place = path
                .file_name()
                .and_then(|name| name.to_str())
                .filter(|name| path.parent() == Some(parent.as_path()) && self.branches.contains(&format!("{BRANCH_PREFIX}{name}")));
            if let Some(name) = by_branch.or(by_place) {
                if !names.iter().any(|known| known == name) {
                    names.push(name.to_string());
                }
            }
        }
        names
    }
}

fn real(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Parse `git worktree list --porcelain -z`: NUL-terminated attributes, an
/// empty attribute between worktrees. The first worktree is the main checkout.
fn worktree_entries(output: &[u8]) -> Vec<WorktreeEntry> {
    let mut entries = Vec::new();
    let mut current: Option<WorktreeEntry> = None;
    for field in output.split(|byte| *byte == 0) {
        if field.is_empty() {
            entries.extend(current.take());
        } else if let Some(path) = field.strip_prefix(b"worktree ") {
            entries.extend(current.take());
            current = Some(WorktreeEntry { path: PathBuf::from(OsStr::from_bytes(path)), branch: None });
        } else if let (Some(branch), Some(entry)) = (field.strip_prefix(b"branch refs/heads/"), current.as_mut()) {
            entry.branch = Some(String::from_utf8_lossy(branch).into_owned());
        }
    }
    entries.extend(current);
    entries
}

fn adapsis_branches(checkout: &Path, options: GitOptions) -> Result<HashSet<String>> {
    let output = git::run_ok(checkout, &["for-each-ref", "--format=%(refname:short)", "refs/heads/adapsis/"], "for-each-ref", options.query)?;
    Ok(output.stdout_text().lines().map(str::to_string).collect())
}

fn base_key(name: &str) -> String {
    format!("branch.{BRANCH_PREFIX}{name}{BASE_KEY_SUFFIX}")
}

/// Remove `branch.adapsis/<name>.adapsis-base` keys whose branch is gone.
/// Best effort: a failure here never fails the operation that triggered it.
fn prune_keys_in(checkout: &Path, branches: &HashSet<String>, options: GitOptions) {
    let pattern = r"^branch\.adapsis/[a-z0-9-]+\.adapsis-base$";
    let Ok(listing) = git::run(checkout, &["config", "--local", "--name-only", "--get-regexp", pattern], "config", options.query) else {
        return;
    };
    if !listing.success() {
        return;
    }
    for key in listing.stdout_text().lines().take(4 * MAX_WORKTREES_PER_REPO + 64) {
        let name = key.strip_prefix("branch.").and_then(|rest| rest.strip_suffix(BASE_KEY_SUFFIX));
        let Some(branch) = name.filter(|branch| branch.starts_with(BRANCH_PREFIX)) else { continue };
        if validate_name("workspace name", &branch[BRANCH_PREFIX.len()..]).is_ok() && !branches.contains(branch) {
            let _ = git::run(checkout, &["config", "--local", "--unset", key], "config", options.query);
        }
    }
}

/// Parse `git status --porcelain=v1 -z` into `XY path` entries. A rename or
/// copy is followed by its source path as a separate field, which is skipped.
fn dirty_entries(output: &[u8]) -> Vec<String> {
    let mut entries = Vec::new();
    let mut fields = output.split(|byte| *byte == 0);
    while let Some(field) = fields.next() {
        if field.len() < 4 {
            continue;
        }
        let (code, path) = (&field[..2], &field[3..]);
        if code.contains(&b'R') || code.contains(&b'C') {
            fields.next();
        }
        let text: String = String::from_utf8_lossy(path).chars().map(|c| if c.is_control() { '\u{FFFD}' } else { c }).collect();
        entries.push(format!("{} {text}", String::from_utf8_lossy(code)));
    }
    entries
}

/// `git rev-list --left-right --count base...HEAD` prints `<behind>\t<ahead>`.
fn ahead_behind(text: &str) -> Option<(u64, u64)> {
    let mut parts = text.split_whitespace();
    let behind = parts.next()?.parse().ok()?;
    let ahead = parts.next()?.parse().ok()?;
    Some((behind, ahead))
}
