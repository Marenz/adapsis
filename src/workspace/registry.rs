//! The repository registry: which checkouts Adapsis may create worktrees of.
//!
//! ```toml
//! [repo.adapsis]
//! checkout = "/home/owner/Projects/adapsis"
//! worktree_parent = "/home/owner/Projects/adapsis-workspaces"
//! base = "origin/master"
//! read_roots = ["/home/owner/Projects/llm-gateway"]   # optional
//!
//! [repo.adapsis.env]                                   # optional
//! CARGO_TARGET_DIR = "/home/owner/.cache/adapsis-target"
//! ```
//!
//! The `env` table is added to the environment of every command run in a
//! workspace of that repository (after the fixed variables, before a
//! request's own additions). Its first use is a shared `CARGO_TARGET_DIR`,
//! because each worktree would otherwise build its own target directory of
//! several gigabytes. Whoever sets that should know the price: builds in two
//! worktrees then wait for each other on cargo's directory lock, and
//! switching between branches evicts and rebuilds each other's artifacts.
//! Values are used literally; there is no placeholder for the workspace name.
//!
//! Loading validates the *shape* of every entry and reports all problems at
//! once. Whether a checkout exists and is a git repository is checked when a
//! workspace is opened, so one unmounted disk does not disable every other
//! repository.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

use serde::Deserialize;

use super::error::{Error, Result};
use super::lifecycle::{validate_base, validate_name};
use super::limits::{READ_ROOTS_MAX, REGISTRY_ENV_MAX_VARS, REGISTRY_MAX_BYTES, REGISTRY_MAX_REPOSITORIES};
use super::run::check_env_pair;

/// Environment variable naming the registry file.
pub const REGISTRY_ENV: &str = "ADAPSIS_WORKSPACES_FILE";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Repository {
    pub name: String,
    /// The registered checkout; `git worktree add` runs here.
    pub checkout: PathBuf,
    /// Directory that receives one subdirectory per workspace.
    pub worktree_parent: PathBuf,
    /// Base ref of a workspace opened without an explicit base.
    pub base: String,
    /// Extra directories every workspace of this repository may read.
    pub read_roots: Vec<PathBuf>,
    /// Variables added to the environment of every command in a workspace.
    pub env: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Registry {
    repositories: BTreeMap<String, Repository>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRegistry {
    #[serde(default)]
    repo: BTreeMap<String, RawRepository>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRepository {
    checkout: String,
    worktree_parent: String,
    base: String,
    #[serde(default)]
    read_roots: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
}

impl Registry {
    /// Load the registry from its configured location. A missing file is an
    /// empty registry; an unreadable or invalid one is an error.
    pub fn load() -> Result<Self> {
        match registry_path(std::env::var_os(REGISTRY_ENV), std::env::var_os("HOME")) {
            Some(path) => Self::load_from(&path),
            None => Ok(Self::default()),
        }
    }

    pub fn load_from(path: &Path) -> Result<Self> {
        let metadata = match std::fs::metadata(path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(error) => {
                return Err(Error::Registry(format!("{} cannot be read: {}", path.display(), error.kind())));
            }
        };
        if !metadata.is_file() {
            return Err(Error::Registry(format!("{} is not a regular file", path.display())));
        }
        if metadata.len() > REGISTRY_MAX_BYTES {
            return Err(Error::Registry(format!(
                "{} is {} bytes; the limit is {REGISTRY_MAX_BYTES}",
                path.display(),
                metadata.len()
            )));
        }
        let text = std::fs::read_to_string(path)
            .map_err(|error| Error::Registry(format!("{} cannot be read: {}", path.display(), error.kind())))?;
        Self::from_toml(&text).map_err(|error| match error {
            Error::Registry(text) => Error::Registry(format!("{}: {text}", path.display())),
            other => other,
        })
    }

    pub fn from_toml(text: &str) -> Result<Self> {
        let raw: RawRegistry = toml::from_str(text).map_err(|error| {
            Error::Registry(format!(
                "not valid: {}. Expected `[repo.<name>]` tables with checkout, worktree_parent, base, optional read_roots and an optional env table of strings",
                error.message()
            ))
        })?;
        let mut problems = Vec::new();
        if raw.repo.len() > REGISTRY_MAX_REPOSITORIES {
            problems.push(format!("{} repositories declared; the limit is {REGISTRY_MAX_REPOSITORIES}", raw.repo.len()));
        }
        let mut repositories = BTreeMap::new();
        for (name, entry) in raw.repo {
            match repository(&name, entry) {
                Ok(repository) => {
                    repositories.insert(name, repository);
                }
                Err(mut found) => problems.append(&mut found),
            }
        }
        if problems.is_empty() { Ok(Self { repositories }) } else { Err(Error::Registry(problems.join("; "))) }
    }

    pub fn get(&self, name: &str) -> Result<&Repository> {
        self.repositories.get(name).ok_or_else(|| {
            let known = self.names().join(", ");
            let known = if known.is_empty() { "none are registered".to_string() } else { format!("registered: {known}") };
            Error::Lifecycle(format!("no repository `{}` in the workspace registry ({known})", super::error::shown(name)))
        })
    }

    pub fn names(&self) -> Vec<&str> {
        self.repositories.keys().map(String::as_str).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.repositories.is_empty()
    }
}

/// Where the registry lives: the override, else `~/.config/adapsis/workspaces.toml`.
///
/// Like `library::library_dir`, `cargo test` must never read the developer's
/// real file, so without the override a test build gets a per-process path
/// that does not exist.
pub fn registry_path(explicit: Option<OsString>, home: Option<OsString>) -> Option<PathBuf> {
    if let Some(path) = explicit.filter(|value| !value.is_empty()) {
        return Some(PathBuf::from(path));
    }
    #[cfg(test)]
    {
        let _ = home;
        Some(std::env::temp_dir().join(format!("adapsis-test-workspaces-{}.toml", std::process::id())))
    }
    #[cfg(not(test))]
    home.filter(|value| !value.is_empty())
        .map(|home| Path::new(&home).join(".config").join("adapsis").join("workspaces.toml"))
}

fn repository(name: &str, raw: RawRepository) -> std::result::Result<Repository, Vec<String>> {
    let mut problems = Vec::new();
    let mut note = |text: String| problems.push(format!("repo `{}`: {text}", super::error::shown(name)));
    if let Err(Error::InvalidName { reason, .. }) = validate_name("repository name", name) {
        note(format!("name {reason}"));
    }
    let checkout = absolute("checkout", &raw.checkout).map_err(&mut note).ok();
    let parent = absolute("worktree_parent", &raw.worktree_parent).map_err(&mut note).ok();
    if let (Some(checkout), Some(parent)) = (&checkout, &parent) {
        if parent.starts_with(checkout) {
            note("worktree_parent must not be the checkout or a directory inside it (the worktrees would show up as untracked files there)".into());
        }
    }
    if let Err(Error::InvalidName { reason, .. }) = validate_base(&raw.base) {
        note(format!("base {reason}"));
    }
    if raw.read_roots.len() > READ_ROOTS_MAX {
        note(format!("{} read_roots; the limit is {READ_ROOTS_MAX}", raw.read_roots.len()));
    }
    let mut read_roots = Vec::new();
    for (index, root) in raw.read_roots.iter().enumerate() {
        match absolute("read_roots entry", root) {
            Ok(path) if path == Path::new("/") => note(format!("read_roots[{index}] is `/`, which would make the whole filesystem readable")),
            Ok(path) => read_roots.push(path),
            Err(text) => note(format!("[{index}] {text}")),
        }
    }
    if raw.env.len() > REGISTRY_ENV_MAX_VARS {
        note(format!("{} env entries; the limit is {REGISTRY_ENV_MAX_VARS}", raw.env.len()));
    }
    for (variable, value) in &raw.env {
        if let Err(reason) = check_env_pair(variable, value) {
            note(format!("env: {reason}"));
        }
    }
    match (checkout, parent, problems.is_empty()) {
        (Some(checkout), Some(worktree_parent), true) => {
            Ok(Repository { name: name.to_string(), checkout, worktree_parent, base: raw.base, read_roots, env: raw.env })
        }
        _ => Err(problems),
    }
}

/// An absolute, already-normal path: no `~`, no `.`/`..`, no NUL.
fn absolute(field: &str, value: &str) -> std::result::Result<PathBuf, String> {
    if value.is_empty() {
        return Err(format!("{field} is empty; give an absolute path"));
    }
    if value.contains('\0') {
        return Err(format!("{field} contains a NUL byte"));
    }
    let path = Path::new(value);
    if !path.is_absolute() {
        return Err(format!("{field} `{value}` must be an absolute path (`~` is not expanded)"));
    }
    if path.components().any(|part| matches!(part, Component::ParentDir | Component::CurDir)) {
        return Err(format!("{field} `{value}` must not contain `.` or `..`"));
    }
    Ok(path.components().collect())
}
