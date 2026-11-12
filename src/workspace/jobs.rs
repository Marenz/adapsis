//! Background jobs: commands that outlive the call that started them, and a
//! restart of this process.
//!
//! This program deploys itself, so a build or test run it starts must not die
//! with it. A job therefore runs in a transient *service* of the systemd user
//! manager: a sibling of Adapsis's own unit, not a child in its control
//! group. Everything about a job is on disk (`job_store`); a restarted
//! process finds it again by reading the directory and asking systemd.
//!
//! Inside the unit, `env -i` is the first thing executed (a service starts
//! with the user manager's environment, which this command must not see),
//! then a static `/usr/bin/python3 -I` adapter (`adapter::JOB`):
//!
//! * it loads bounded JSON environment data from a private nofollow file and
//!   deletes it; values are supplied only to the payload's Popen environment,
//! * runs the launcher's argument vector with stdout and stderr in one pipe,
//! * keeps the first `head` bytes in `output.log` and, once that is full, the
//!   last `tail` bytes in `output.tail` (written when the stream ends),
//! * writes the exit status to `exit`.
//!
//! The small adapter survives SIGTERM, so a job that is stopped politely (by
//! `job_kill` or by its runtime limit) still records its status and its tail.
//! A hard kill leaves neither; such a job is reported as what systemd says
//! happened, or as lost.
//!
//! Without a user manager jobs are refused. Nothing here falls back to a
//! process that would die with this one.
//!
//! The adapter is trusted static source, never interpolated from payloads.
//! Its Popen occurs in a small clean process, not by forking the large Rust
//! daemon. Missing Python/setup failures do not authorize replay.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::coroutine::shell_policy::ShellPolicy;

use super::error::{Error, Result};
use super::job_store::{
    JobRecord, JobState, RECORD_VERSION, check_id, job_ids, job_number, now_ms, read_record_at, state_error, state_root, workspace_dir,
    write_record_at,
};
use super::lifecycle::Workspace;
use super::limits::{
    JOB_OUTPUT_HEAD_BYTES, JOB_OUTPUT_TAIL_BYTES, JOB_RETENTION, JOB_RUNTIME_DEFAULT, JOB_RUNTIME_MAX, JOB_RUNTIME_MIN, JOB_STOP_GRACE,
    JOBS_KEPT_MAX, JOBS_RUNNING_MAX, RUN_KILL_GRACE,
};
use super::run::{Launcher, Prepared, bash_argv, default_unit_prefix, prepare};
use super::state_io::Dir;
use super::systemd;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobRequest {
    pub command: String,
    /// Directory to run in, relative to the workspace root.
    pub cwd: String,
    /// `None` is `JOB_RUNTIME_DEFAULT`; clamped to `JOB_RUNTIME_MIN..=MAX`.
    pub max_runtime: Option<Duration>,
    pub env: Vec<(String, String)>,
}

impl JobRequest {
    pub fn new(command: &str) -> Self {
        Self {
            command: command.to_string(),
            cwd: ".".to_string(),
            max_runtime: None,
            env: Vec::new(),
        }
    }
}

/// A job as it is now: its record, brought up to date, and its output sizes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobStatus {
    pub record: JobRecord,
    pub log_bytes: u64,
    /// Size of the tail file; non-zero only when the head cap was reached.
    pub tail_bytes: u64,
}

impl JobStatus {
    pub fn head_cap_reached(&self) -> bool {
        self.tail_bytes > 0 || self.log_bytes >= self.record.head_cap
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobEntry {
    Job(JobStatus),
    /// A job directory whose record this program cannot interpret. It is
    /// listed and otherwise left alone.
    Unreadable {
        id: String,
        reason: String,
    },
}

/// Where jobs are kept and how their units are named.
#[derive(Debug, Clone)]
pub struct JobStore {
    root: PathBuf,
    unit_prefix: String,
    /// Test hook: behave as if no user manager were reachable.
    pub(super) pretend_unavailable: Option<String>,
    #[cfg(test)]
    pub(super) start_gate: Option<std::sync::Arc<(std::sync::Barrier, std::sync::Barrier)>>,
}

impl JobStore {
    /// The store of this process: the standard state root and unit prefix.
    pub fn standard() -> Self {
        Self {
            root: state_root(),
            unit_prefix: default_unit_prefix().to_string(),
            pretend_unavailable: None,
            #[cfg(test)]
            start_gate: None,
        }
    }

    /// A store elsewhere. The prefix becomes part of unit names and must be
    /// made of lowercase letters, digits and dashes.
    pub fn at(root: &Path, unit_prefix: &str) -> Result<Self> {
        super::lifecycle::validate_name("unit prefix", unit_prefix)?;
        Ok(Self {
            root: root.to_path_buf(),
            unit_prefix: unit_prefix.to_string(),
            pretend_unavailable: None,
            #[cfg(test)]
            start_gate: None,
        })
    }

    fn jobs_dir(&self, workspace: &Workspace, create: bool) -> Result<Option<Dir>> {
        workspace_dir(&self.root, workspace)?;
        let Some(root) = Dir::root(&self.root, create).map_err(state_error)? else {
            return Ok(None);
        };
        let Some(repo) = root.child(workspace.repo.as_ref(), create).map_err(state_error)? else {
            return Ok(None);
        };
        let Some(ws) = repo.child(workspace.name.as_ref(), create).map_err(state_error)? else {
            return Ok(None);
        };
        ws.child("jobs".as_ref(), create).map_err(state_error)
    }

    fn job_dir(&self, workspace: &Workspace, id: &str) -> Result<Dir> {
        check_id(id)?;
        let parent = self.jobs_dir(workspace, false)?;
        if let Some(directory) = parent
            .as_ref()
            .map(|parent| parent.child(id.as_ref(), false))
            .transpose()
            .map_err(state_error)?
            .flatten()
        {
            Ok(directory)
        } else {
            let known = parent.as_ref().map(job_ids).transpose().map_err(state_error)?.unwrap_or_default();
            let known = if known.is_empty() {
                "it has no jobs".to_string()
            } else {
                format!("its jobs: {}", known.join(", "))
            };
            Err(Error::Job(format!("no job `{id}` in workspace `{}` ({known})", workspace.name)))
        }
    }

    fn availability(&self) -> std::result::Result<(), String> {
        match &self.pretend_unavailable {
            Some(reason) => Err(reason.clone()),
            None => systemd::availability(),
        }
    }

    /// Start a command as a background job.
    pub fn start(&self, workspace: &Workspace, request: &JobRequest, policy: &ShellPolicy, launcher: Launcher<'_>) -> Result<JobStatus> {
        if !Path::new(super::adapter::PYTHON).is_file() {
            return Err(Error::Job("workspace adapter requires /usr/bin/python3".into()));
        }
        let prepared = prepare(workspace, &request.command, &request.cwd, &request.env, policy, launcher)?;
        self.availability().map_err(|reason| {
            Error::Job(format!(
                "background jobs are not available: {reason}. Run the command in the foreground with a timeout instead"
            ))
        })?;
        let reservation = super::reservation::for_workspace(workspace);
        let _guard = reservation.lock().unwrap_or_else(|p| p.into_inner());
        if !workspace.path.is_dir() {
            return Err(Error::Job("workspace was closed before the job could start".into()));
        }
        self.make_room(workspace)?;
        #[cfg(test)]
        if let Some(gate) = &self.start_gate {
            gate.0.wait();
            gate.1.wait();
        }
        let jobs_dir = self
            .jobs_dir(workspace, true)?
            .ok_or_else(|| Error::Job("job directory is missing".into()))?;
        let (id, directory) = allocate(&jobs_dir)?;
        let record = JobRecord {
            version: RECORD_VERSION,
            unit: self.unit_name(workspace, &id),
            id,
            repo: workspace.repo.clone(),
            workspace: workspace.name.clone(),
            command: request.command.clone(),
            cwd: prepared.cwd.display.clone(),
            started_at_ms: now_ms(),
            max_runtime_s: request
                .max_runtime
                .unwrap_or(JOB_RUNTIME_DEFAULT)
                .clamp(JOB_RUNTIME_MIN, JOB_RUNTIME_MAX)
                .as_secs()
                .max(1),
            head_cap: JOB_OUTPUT_HEAD_BYTES,
            tail_cap: JOB_OUTPUT_TAIL_BYTES,
            env_names: request.env.iter().map(|(name, _)| name.clone()).collect(),
            state: JobState::Running,
            exit_status: None,
            ended_at_ms: None,
            end_reason: None,
        };
        let launched = launch(&directory, &self.root, workspace, &record, &prepared);
        if let Err(error) = launched {
            if directory.metadata("job.json".as_ref()).map_err(state_error)?.is_some() {
                // A lost client reply does not mean that the manager did not
                // start the unit. Keep its identity and private recovery data;
                // never start it again merely because the acknowledgement failed.
                return Err(Error::Job(format!(
                    "startup of job `{}` was not confirmed: {error}. Its record is preserved; inspect job_status before starting another",
                    record.id
                )));
            }
            jobs_dir.remove_tree(record.id.as_ref()).map_err(state_error)?;
            return Err(error);
        }
        status_of(&directory, record)
    }

    /// Enforce the per-workspace limits before a new job: drop finished jobs
    /// past their retention, refuse when too many run, make room for one
    /// more record.
    fn make_room(&self, workspace: &Workspace) -> Result<()> {
        let now = now_ms();
        let mut jobs: Vec<JobStatus> = Vec::new();
        let mut unreadable = 0;
        for entry in self.list_locked(workspace)? {
            let JobEntry::Job(job) = entry else {
                unreadable += 1;
                continue;
            };
            let expired = job
                .record
                .ended_at_ms
                .is_some_and(|ended| now.saturating_sub(ended) > JOB_RETENTION.as_millis() as u64);
            if job.record.state != JobState::Running && expired {
                self.remove(workspace, &job.record)?;
            } else {
                jobs.push(job);
            }
        }
        let running: Vec<&str> = jobs
            .iter()
            .filter(|job| job.record.state == JobState::Running)
            .map(|job| job.record.id.as_str())
            .collect();
        if running.len() >= JOBS_RUNNING_MAX {
            return Err(Error::Job(format!(
                "workspace `{}` already has {} running jobs ({}), the limit is {JOBS_RUNNING_MAX}; wait for one or stop it with job_kill",
                workspace.name,
                running.len(),
                running.join(", ")
            )));
        }
        jobs.sort_by_key(|job| job.record.started_at_ms);
        let mut kept = jobs.len() + unreadable;
        for job in jobs.iter().filter(|job| job.record.state != JobState::Running) {
            if kept < JOBS_KEPT_MAX {
                break;
            }
            self.remove(workspace, &job.record)?;
            kept -= 1;
        }
        if kept >= JOBS_KEPT_MAX {
            return Err(Error::Job(format!(
                "workspace `{}` has {unreadable} unreadable job records that cannot be pruned; inspect their private records before starting more jobs",
                workspace.name
            )));
        }
        Ok(())
    }

    fn remove(&self, workspace: &Workspace, record: &JobRecord) -> Result<()> {
        if let Some(parent) = self.jobs_dir(workspace, false)? {
            parent.remove_tree(record.id.as_ref()).map_err(state_error)?;
        }
        Ok(())
    }

    /// Component lengths disambiguate names such as `a-b`/`c` and `a`/`b-c`.
    /// A short digest also separates stores of the same workspace under two
    /// state roots (tests and an operator's isolated instance).
    pub(super) fn unit_name(&self, workspace: &Workspace, id: &str) -> String {
        let store = super::foreground::root_digest(&self.root);
        format!(
            "{}-job-{}-{}-{}-{}-{store}-{id}.service",
            self.unit_prefix,
            workspace.repo.len(),
            workspace.repo,
            workspace.name.len(),
            workspace.name
        )
    }

    /// Every job of the workspace, oldest first, each brought up to date.
    pub fn list(&self, workspace: &Workspace) -> Result<Vec<JobEntry>> {
        let reservation = super::reservation::for_workspace(workspace);
        let _guard = reservation.lock().unwrap_or_else(|p| p.into_inner());
        self.list_locked(workspace)
    }

    fn list_locked(&self, workspace: &Workspace) -> Result<Vec<JobEntry>> {
        let Some(jobs_dir) = self.jobs_dir(workspace, false)? else {
            return Ok(Vec::new());
        };
        let mut results = Vec::new();
        for id in job_ids(&jobs_dir).map_err(state_error)? {
            let directory = jobs_dir
                .child(id.as_ref(), false)
                .map_err(state_error)?
                .ok_or_else(|| Error::Job("job directory vanished during enumeration".into()))?;
            let observed = self
                .read_bound_record(workspace, &id, &directory)
                .and_then(|record| self.observe(&directory, record).map_err(|error| error.to_string()));
            let entry = match observed {
                Ok(record) => {
                    let expired = record
                        .ended_at_ms
                        .is_some_and(|ended| now_ms().saturating_sub(ended) > JOB_RETENTION.as_millis() as u64);
                    if record.state != JobState::Running && expired {
                        self.remove(workspace, &record)?;
                        continue;
                    }
                    JobEntry::Job(status_of(&directory, record)?)
                }
                Err(reason) => JobEntry::Unreadable { id, reason },
            };
            results.push(entry);
        }
        Ok(results)
    }

    pub fn status(&self, workspace: &Workspace, id: &str) -> Result<JobStatus> {
        let reservation = super::reservation::for_workspace(workspace);
        let _guard = reservation.lock().unwrap_or_else(|p| p.into_inner());
        self.status_locked(workspace, id)
    }

    fn status_locked(&self, workspace: &Workspace, id: &str) -> Result<JobStatus> {
        let directory = self.job_dir(workspace, id)?;
        let record = self
            .read_bound_record(workspace, id, &directory)
            .map_err(|reason| Error::Job(format!("job `{id}` cannot be used: {reason}")))?;
        status_of(&directory, self.observe(&directory, record)?)
    }

    /// Stop a running job: SIGTERM to everything in its unit, a grace
    /// period, then SIGKILL. A job that already ended is returned as it is.
    pub fn kill(&self, workspace: &Workspace, id: &str) -> Result<JobStatus> {
        let reservation = super::reservation::for_workspace(workspace);
        let _guard = reservation.lock().unwrap_or_else(|p| p.into_inner());
        let directory = self.job_dir(workspace, id)?;
        let current = self.status_locked(workspace, id)?;
        if current.record.state != JobState::Running {
            return Ok(current);
        }
        // The marker makes "killed" survive a restart between the signal and
        // the next look at the job.
        directory.write("kill", b"").map_err(state_error)?;
        let unit = current.record.unit.clone();
        if let Err(error) = systemd::kill(&unit, "SIGTERM") {
            // A refused/unconfirmed signal is not proof of a job_kill end.
            directory.unlink("kill".as_ref(), false).map_err(state_error)?;
            return Err(error);
        }
        if !wait_inactive(&unit, RUN_KILL_GRACE) {
            systemd::kill(&unit, "SIGKILL")?;
            wait_inactive(&unit, JOB_STOP_GRACE);
        }
        self.status_locked(workspace, id)
    }

    /// Refuse to close a workspace whose jobs still run.
    pub fn close_guard(&self, workspace: &Workspace) -> Result<()> {
        let reservation = super::reservation::for_workspace(workspace);
        let _guard = reservation.lock().unwrap_or_else(|p| p.into_inner());
        self.close_guard_locked(workspace)
    }

    pub(super) fn close_guard_locked(&self, workspace: &Workspace) -> Result<()> {
        let entries = self.list_locked(workspace)?;
        if let Some(dir) = self.jobs_dir(workspace, false)? {
            dir.validate_tree()?;
        }
        let unverifiable: Vec<&str> = entries
            .iter()
            .filter_map(|entry| match entry {
                JobEntry::Unreadable { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        if !unverifiable.is_empty() {
            return Err(Error::Job(format!(
                "workspace `{}` has unverifiable job records ({}) and was not closed; inspect their records or restore the user manager first",
                workspace.name,
                unverifiable.join(", ")
            )));
        }
        let running: Vec<String> = entries
            .into_iter()
            .filter_map(|entry| match entry {
                JobEntry::Job(job) if job.record.state == JobState::Running => Some(job.record.id),
                _ => None,
            })
            .collect();
        if running.is_empty() {
            Ok(())
        } else {
            Err(Error::Job(format!(
                "workspace `{}` has running jobs ({}) and was not closed; wait for them or stop them with job_kill",
                workspace.name,
                running.join(", ")
            )))
        }
    }

    /// Delete everything kept for a workspace that no longer exists.
    pub fn forget(&self, workspace: &Workspace) -> Result<()> {
        let reservation = super::reservation::for_workspace(workspace);
        let _guard = reservation.lock().unwrap_or_else(|p| p.into_inner());
        self.close_guard_locked(workspace)?;
        self.forget_locked(workspace)
    }

    pub(super) fn forget_locked(&self, workspace: &Workspace) -> Result<()> {
        workspace_dir(&self.root, workspace)?;
        if let Some(root) = Dir::root(&self.root, false).map_err(state_error)? {
            if let Some(repo) = root.child(workspace.repo.as_ref(), false).map_err(state_error)? {
                repo.remove_tree(workspace.name.as_ref()).map_err(state_error)?;
            }
        }
        Ok(())
    }

    pub(super) fn directory_of(&self, workspace: &Workspace, id: &str) -> Result<Dir> {
        self.job_dir(workspace, id)
    }

    fn read_bound_record(&self, workspace: &Workspace, id: &str, directory: &Dir) -> std::result::Result<JobRecord, String> {
        let record = read_record_at(directory)?;
        if record.id != id
            || record.repo != workspace.repo
            || record.workspace != workspace.name
            || record.unit != self.unit_name(workspace, id)
        {
            return Err("its record does not belong to this workspace/store/id".to_string());
        }
        Ok(record)
    }

    /// Bring a record up to date. A running job is asked about at systemd; a
    /// finished one is final. The first look after the end writes the
    /// outcome into the record and lets systemd forget the unit.
    fn observe(&self, directory: &Dir, mut record: JobRecord) -> Result<JobRecord> {
        if record.state != JobState::Running {
            return Ok(record);
        }
        self.availability().map_err(|reason| {
            Error::Job(format!(
                "job `{}` cannot be observed: {reason}; its last record is preserved, not evidence that it stopped",
                record.id
            ))
        })?;
        let properties = systemd::show(&record.unit, &["LoadState", "ActiveState", "Result"])?;
        if systemd::is_active(&properties) {
            return Ok(record);
        }
        let loaded = properties.get("LoadState").is_some_and(|state| state == "loaded");
        let result = properties.get("Result").filter(|_| loaded).map(String::as_str).unwrap_or_default();
        let exit_status = match directory.read("exit", 100).map_err(state_error)? {
            None => None,
            Some(bytes) => Some(
                std::str::from_utf8(&bytes)
                    .ok()
                    .and_then(|s| s.trim().parse::<i32>().ok())
                    .filter(|n| (0..=255).contains(n))
                    .ok_or_else(|| Error::Job("job exit record is malformed; state remains unconfirmed".into()))?,
            ),
        };
        record.state = if result == "timeout" {
            JobState::TimedOut
        } else if directory.metadata("kill".as_ref()).map_err(state_error)?.is_some() {
            JobState::Killed
        } else if exit_status.is_some() {
            JobState::Exited
        } else if loaded {
            JobState::Failed
        } else {
            JobState::Lost
        };
        record.exit_status = exit_status;
        record.end_reason = Some(result)
            .filter(|result| !matches!(*result, "" | "success" | "exit-code" | "timeout"))
            .map(str::to_string);
        let ended = directory
            .metadata("exit".as_ref())
            .map_err(state_error)?
            .and_then(|meta| meta.modified().ok());
        record.ended_at_ms = Some(
            ended
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or_else(now_ms, |elapsed| elapsed.as_millis() as u64),
        );
        // Only forget the unit once its outcome is safely on disk.
        if write_record_at(directory, &record).is_ok() && loaded {
            systemd::reset_failed(&record.unit);
        }
        // If startup failed before the wrapper could read the environment,
        // it need not remain on disk after a proven terminal observation.
        if directory.metadata("env".as_ref()).map_err(state_error)?.is_some() {
            directory.unlink("env".as_ref(), false).map_err(state_error)?;
        }
        Ok(record)
    }
}

fn status_of(directory: &Dir, record: JobRecord) -> Result<JobStatus> {
    let len = |name: &str| match directory.metadata(name.as_ref()).map_err(state_error)? {
        Some(meta) if !meta.is_file() => Err(Error::Job("job output is not a regular file".into())),
        meta => Ok(meta.map_or(0, |meta| meta.len())),
    };
    Ok(JobStatus {
        log_bytes: len("output.log")?,
        tail_bytes: len("output.tail")?,
        record,
    })
}

/// Reserve the next job id by creating its directory; the creation is the
/// lock, so two starters never get the same id. An id is never given out
/// twice in one workspace, even after its job was pruned: the highest number
/// ever used is kept in `last-id`, so `j7` always means the same job.
pub(super) fn allocate(jobs_dir: &Dir) -> Result<(String, Dir)> {
    let ids = job_ids(jobs_dir).map_err(state_error)?;
    let remembered = match jobs_dir.read("last-id", 32).map_err(state_error)? {
        None if ids.is_empty() => 0,
        None => {
            return Err(Error::Job(
                "last-id is missing from existing job history; allocation refused".into(),
            ));
        }
        Some(bytes) => std::str::from_utf8(&bytes)
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .filter(|n| *n < 999_999_999)
            .ok_or_else(|| Error::Job("existing last-id is malformed; allocation refused".into()))?,
    };
    let mut number = ids.last().and_then(|id| job_number(id)).unwrap_or(0).max(remembered);
    for _ in 0..32 {
        number = number
            .checked_add(1)
            .filter(|n| *n <= 999_999_999)
            .ok_or_else(|| Error::Job("job ids exhausted".into()))?;
        let id = format!("j{number}");
        match jobs_dir.create_child(&id) {
            Ok(directory) => {
                jobs_dir
                    .write("last-id", number.to_string().as_bytes())
                    .map_err(|error| Error::Job(format!("the job id counter could not be saved ({}); start refused", error.kind())))?;
                return Ok((id, directory));
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(Error::Job(format!("job directory could not be created ({})", error.kind()))),
        }
    }
    Err(Error::Job("no free job id could be reserved; try again".to_string()))
}

/// Write the job's files and start its unit.
fn launch(directory: &Dir, root: &Path, workspace: &Workspace, record: &JobRecord, prepared: &Prepared) -> Result<()> {
    use std::os::unix::fs::MetadataExt;
    let io_error =
        |what: &str, error: std::io::Error| Error::Job(format!("{what} of job {} could not be written ({})", record.id, error.kind()));
    let environment = super::run::encoded_environment(&prepared.environment)?;
    directory
        .write("env", &environment)
        .map_err(|error| io_error("the environment", error))?;
    directory
        .write("output.log", b"")
        .map_err(|error| io_error("the output file", error))?;
    write_record_at(directory, record)?;
    let mut command = systemd::client(systemd::SYSTEMD_RUN);
    command.args(["--user", "--quiet", "--expand-environment=no"]);
    command.arg(format!("--unit={}", record.unit));
    for property in [
        "KillMode=control-group".to_string(),
        format!("RuntimeMaxSec={}", record.max_runtime_s),
        format!("TimeoutStopSec={}", JOB_STOP_GRACE.as_secs()),
    ] {
        command.arg("-p").arg(property);
    }
    let mut working_directory = OsString::from("--working-directory=");
    working_directory.push(&prepared.cwd.real);
    command.arg(working_directory).arg("--");
    let meta = std::fs::metadata(directory.proc_path()).map_err(state_error)?;
    command.args(["/usr/bin/env", "-i", super::adapter::PYTHON, "-I", "-c", super::adapter::JOB]);
    command
        .arg(root)
        .args([workspace.repo.as_str(), workspace.name.as_str(), "jobs", record.id.as_str()]);
    command.args([
        meta.dev().to_string(),
        meta.ino().to_string(),
        record.head_cap.to_string(),
        record.tail_cap.to_string(),
    ]);
    command.args(&prepared.argv);
    let output = systemd::run_client(command, "job start")?;
    if output.success() {
        Ok(())
    } else {
        Err(Error::Systemd {
            action: "job start".to_string(),
            detail: output.failure_text(),
        })
    }
}

/// Poll until the unit is no longer active (true) or the time is up.
fn wait_inactive(unit: &str, limit: Duration) -> bool {
    let deadline = Instant::now() + limit;
    loop {
        match systemd::show(unit, &["ActiveState"]) {
            Ok(properties) if !systemd::is_active(&properties) => return true,
            _ => {}
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

// --- the process's own store ------------------------------------------------

/// Start a job under the process's shell policy, with the default launcher.
pub fn job_start(workspace: &Workspace, request: &JobRequest) -> Result<JobStatus> {
    JobStore::standard().start(workspace, request, &ShellPolicy::from_env(), &bash_argv)
}

pub fn job_list(workspace: &Workspace) -> Result<Vec<JobEntry>> {
    JobStore::standard().list(workspace)
}

pub fn job_status(workspace: &Workspace, id: &str) -> Result<JobStatus> {
    JobStore::standard().status(workspace, id)
}

pub fn job_kill(workspace: &Workspace, id: &str) -> Result<JobStatus> {
    JobStore::standard().kill(workspace, id)
}
