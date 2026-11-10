//! Where background jobs live on disk, and the record that describes one.
//!
//! ```text
//! <root>/<repo>/<workspace>/jobs/last-id    highest job number ever used
//! <root>/<repo>/<workspace>/jobs/<id>/
//!   job.json       the record (this file)
//!   output.log     stdout and stderr together, the first HEAD bytes
//!   output.tail    the last TAIL bytes, only if the head filled up and the
//!                  stream ended normally
//!   exit           the command's exit status, written inside the unit
//!   kill           present once `job_kill` was asked for
//! ```
//!
//! `<root>` is `~/.local/share/adapsis/workspaces` (override
//! `ADAPSIS_WORKSPACE_STATE_DIR`; a per-process temp directory under
//! `cfg(test)`). The record holds the command text, so every directory is
//! created 0700 and every file 0600. Environment values are never stored,
//! only the names. A short-lived private `env` JSON file stages values for
//! the static adapter; it is never imported into a supervisory shell.
//!
//! The registry of jobs *is* this directory: nothing is kept in memory, so a
//! restarted process finds every job again. Whether a job still runs is never
//! read from the record; `jobs` asks systemd.

#[cfg(test)]
use std::io::Write;
#[cfg(test)]
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use super::error::{Error, Result};
use super::lifecycle::{Workspace, validate_name};
use super::state_io::Dir;

/// Version of the record format this code writes and understands.
pub const RECORD_VERSION: u32 = 1;
pub const STATE_DIR_ENV: &str = "ADAPSIS_WORKSPACE_STATE_DIR";
#[cfg(test)]
static WRITE_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Running,
    /// The command ended by itself; `exit_status` is set.
    Exited,
    /// systemd stopped it at its runtime limit.
    TimedOut,
    /// `job_kill` stopped it.
    Killed,
    /// It ended without an exit status while systemd still knew why
    /// (`end_reason`), or it could not be started.
    Failed,
    /// Its unit is gone and nothing recorded how it ended: killed hard, or
    /// the user session or machine restarted.
    Lost,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobRecord {
    pub version: u32,
    pub id: String,
    pub repo: String,
    pub workspace: String,
    /// The systemd unit the job runs in.
    pub unit: String,
    pub command: String,
    /// Workspace-relative directory the command runs in.
    pub cwd: String,
    pub started_at_ms: u64,
    pub max_runtime_s: u64,
    pub head_cap: u64,
    pub tail_cap: u64,
    /// Names of the variables the request added; values are not stored.
    pub env_names: Vec<String>,
    pub state: JobState,
    pub exit_status: Option<i32>,
    pub ended_at_ms: Option<u64>,
    /// systemd's word for an abnormal end (`oom-kill`, `signal`, ...).
    pub end_reason: Option<String>,
}

/// The default state root. Like `library::library_dir`, a test build never
/// points at the owner's real directory.
pub fn state_root() -> PathBuf {
    if let Some(path) = std::env::var_os(STATE_DIR_ENV).filter(|value| !value.is_empty()) {
        return PathBuf::from(path);
    }
    #[cfg(test)]
    {
        std::env::temp_dir().join(format!("adapsis-test-workspace-state-{}", std::process::id()))
    }
    #[cfg(not(test))]
    {
        let home = std::env::var_os("HOME")
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "/tmp".into());
        Path::new(&home).join(".local").join("share").join("adapsis").join("workspaces")
    }
}

/// `<root>/<repo>/<workspace>`; both names are checked, because they become
/// path components and part of a unit name.
pub fn workspace_dir(root: &Path, workspace: &Workspace) -> Result<PathBuf> {
    validate_name("repository name", &workspace.repo)?;
    validate_name("workspace name", &workspace.name)?;
    Ok(root.join(&workspace.repo).join(&workspace.name))
}

/// A job id is `j` and a number: `j1`, `j27`.
pub fn job_number(id: &str) -> Option<u32> {
    let digits = id.strip_prefix('j')?;
    if digits.is_empty() || digits.len() > 9 || digits.starts_with('0') || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

pub fn check_id(id: &str) -> Result<()> {
    match job_number(id) {
        Some(_) => Ok(()),
        None => Err(Error::Job(format!(
            "`{}` is not a job id; ids look like j1, j2 (see the job list)",
            super::error::shown(id)
        ))),
    }
}

#[cfg(test)]
pub fn create_private_dir(path: &Path) -> std::io::Result<()> {
    std::fs::DirBuilder::new().recursive(true).mode(0o700).create(path)
}

/// Write a file readable by the owner only, replacing it atomically.
#[cfg(test)]
pub fn write_private(path: &Path, content: &[u8]) -> std::io::Result<()> {
    let mut temporary = path.as_os_str().to_os_string();
    temporary.push(format!(
        ".{}-{}.tmp",
        std::process::id(),
        WRITE_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let temporary = PathBuf::from(temporary);
    let written = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temporary)?;
        file.write_all(content)?;
        file.sync_all()
    })();
    match written.and_then(|()| std::fs::rename(&temporary, path)) {
        Ok(()) => {
            if let Some(parent) = path.parent() {
                std::fs::File::open(parent)?.sync_all()?;
            }
            Ok(())
        }
        Err(error) => {
            let _ = std::fs::remove_file(&temporary);
            Err(error)
        }
    }
}

#[cfg(test)]
pub fn write_record(directory: &Path, record: &JobRecord) -> Result<()> {
    let dir = Dir::root(directory, false)
        .map_err(state_error)?
        .ok_or_else(|| Error::Job("job directory is missing".into()))?;
    write_record_at(&dir, record)
}

pub fn state_error(error: std::io::Error) -> Error {
    Error::Job(format!(
        "private job state refused/unavailable ({}); nothing may be assumed absent",
        error.kind()
    ))
}

pub fn write_record_at(directory: &Dir, record: &JobRecord) -> Result<()> {
    let text = serde_json::to_vec_pretty(record).map_err(|error| Error::Job(format!("job record could not be encoded: {error}")))?;
    directory
        .write("job.json", &text)
        .map_err(|error| Error::Job(format!("job record of {} could not be written ({})", record.id, error.kind())))
}

/// Read a record. A record of another version, or one that does not parse,
/// is reported with a reason and never interpreted.
#[cfg(test)]
pub fn read_record(directory: &Path) -> std::result::Result<JobRecord, String> {
    let dir = Dir::root(directory, false)
        .map_err(|e| e.to_string())?
        .ok_or("job directory is missing")?;
    read_record_at(&dir)
}

pub fn read_record_at(directory: &Dir) -> std::result::Result<JobRecord, String> {
    let bytes = directory
        .read("job.json", super::limits::JOB_RECORD_MAX_BYTES)
        .map_err(|error| format!("its record cannot be read ({})", error.kind()))?
        .ok_or("its record is missing")?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|_| "its record is not valid JSON".to_string())?;
    match value.get("version").and_then(serde_json::Value::as_u64) {
        Some(version) if version == u64::from(RECORD_VERSION) => {}
        Some(version) => {
            return Err(format!(
                "its record has format version {version}; this program understands {RECORD_VERSION}"
            ));
        }
        None => return Err("its record has no format version".to_string()),
    }
    let record: JobRecord = serde_json::from_value(value).map_err(|_| "its record does not have the expected fields".to_string())?;
    validate_record(&record)?;
    Ok(record)
}

fn validate_record(record: &JobRecord) -> std::result::Result<(), String> {
    use super::limits::*;
    use std::path::Component;
    let invalid = || "its v1 record violates semantic invariants".to_string();
    let running = record.state == JobState::Running;
    if job_number(&record.id).is_none()
        || validate_name("repo", &record.repo).is_err()
        || validate_name("workspace", &record.workspace).is_err()
        || record.head_cap != JOB_OUTPUT_HEAD_BYTES
        || record.tail_cap != JOB_OUTPUT_TAIL_BYTES
        || record.max_runtime_s < JOB_RUNTIME_MIN.as_secs()
        || record.max_runtime_s > JOB_RUNTIME_MAX.as_secs()
        || record.command.trim().is_empty()
        || record.command.len() > RUN_COMMAND_MAX_BYTES
        || record.command.contains('\0')
        || running != record.ended_at_ms.is_none()
        || record.ended_at_ms.is_some_and(|end| end < record.started_at_ms)
        || (running && (record.exit_status.is_some() || record.end_reason.is_some()))
        || (record.state == JobState::Exited && record.exit_status.is_none())
        || (matches!(record.state, JobState::Failed | JobState::Lost) && record.exit_status.is_some())
        || record.exit_status.is_some_and(|exit| !(0..=255).contains(&exit))
        || record.env_names.len() > RUN_ENV_MAX_VARS
        || record.env_names.iter().map(String::len).sum::<usize>() > RUN_ENV_MAX_BYTES
        || record.env_names.iter().any(|name| super::run::check_env_pair(name, "").is_err())
        || record.cwd.is_empty()
        || record.cwd.len() > PATH_MAX_BYTES
        || record.cwd.contains('\0')
        || Path::new(&record.cwd).is_absolute()
        || Path::new(&record.cwd)
            .components()
            .any(|c| !matches!(c, Component::Normal(_) | Component::CurDir))
    {
        return Err(invalid());
    }
    if record.end_reason.as_deref().is_some_and(|r| {
        ![
            "signal",
            "core-dump",
            "oom-kill",
            "resources",
            "protocol",
            "start-limit-hit",
            "exit-code",
            "unknown",
        ]
        .contains(&r)
    }) {
        return Err(invalid());
    }
    Ok(())
}

/// The job directories of a workspace, by ascending job number.
pub fn job_ids(jobs_dir: &Dir) -> std::io::Result<Vec<String>> {
    let names = jobs_dir.list()?;
    let mut numbered = Vec::new();
    for name in names {
        let metadata = jobs_dir.metadata(&name)?;
        if let Some(number) = name.to_str().and_then(job_number) {
            numbered.push((number, name.to_string_lossy().into_owned()));
        } else if metadata.is_some_and(|m| m.is_dir()) {
            return Err(std::io::Error::from(std::io::ErrorKind::InvalidData));
        }
    }
    numbered.sort();
    Ok(numbered.into_iter().map(|(_, name)| name).collect())
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_millis() as u64)
}

pub fn file_len(path: &Path) -> u64 {
    std::fs::metadata(path).map_or(0, |metadata| metadata.len())
}
