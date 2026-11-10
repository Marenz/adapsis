//! Tests of background jobs.
//!
//! No test touches a unit of a running Adapsis: every store here has its own
//! temp directory and a unit prefix `adapsis-wstest-<pid>-<n>`, and a guard
//! stops and forgets `<prefix>-job-*` when the test ends, whatever happened.
//! Tests that need a systemd user manager return early, with a note, where
//! there is none; the record format, the retention rules, the output reader
//! and the wrapper script are tested everywhere.

use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use crate::coroutine::shell_policy::ShellPolicy;

use super::error::Error;
use super::job_output::render_jobs;
use super::job_store::{JobRecord, JobState, RECORD_VERSION, job_number, read_record, write_record};
use super::jobs::{JobEntry, JobRequest, JobStatus, JobStore};
use super::lifecycle::{close, open};
use super::limits::{JOB_OUTPUT_HEAD_BYTES, JOB_OUTPUT_TAIL_BYTES, JOB_RETENTION, JOBS_KEPT_MAX, JOBS_RUNNING_MAX};
use super::run::bash_argv;
use super::systemd;
use super::test_support::{Fixture, RepoFixture};

static STORE_COUNTER: AtomicU32 = AtomicU32::new(0);

/// A workspace, a store of its own, and the cleanup of its units.
struct Jobs {
    fixture: Fixture,
    store: JobStore,
    root: PathBuf,
    prefix: String,
}

impl Jobs {
    fn new() -> Self {
        let fixture = Fixture::plain();
        let root = fixture.top.join("state");
        let prefix = format!("adapsis-wstest-{}-{}", std::process::id(), STORE_COUNTER.fetch_add(1, Ordering::Relaxed));
        let store = JobStore::at(&root, &prefix).unwrap();
        Self { fixture, store, root, prefix }
    }

    /// `None` (with a note) where there is no systemd user manager.
    fn with_systemd() -> Option<Self> {
        if systemd::available() {
            Some(Self::new())
        } else {
            eprintln!("skipped: no systemd user manager is reachable here");
            None
        }
    }

    fn start(&self, command: &str) -> JobStatus {
        self.try_start(&JobRequest::new(command)).unwrap()
    }

    fn try_start(&self, request: &JobRequest) -> Result<JobStatus, Error> {
        self.store.start(&self.fixture.workspace, request, &ShellPolicy::Unrestricted, &bash_argv)
    }

    fn status(&self, id: &str) -> JobStatus {
        self.store.status(&self.fixture.workspace, id).unwrap()
    }

    /// Wait (generously) until the job is no longer running.
    fn finished(&self, id: &str) -> JobStatus {
        let give_up = Instant::now() + Duration::from_secs(120);
        loop {
            let status = self.status(id);
            if status.record.state != JobState::Running || Instant::now() > give_up {
                return status;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
    }

    /// Wait until the job has written at least `bytes` of output.
    fn wrote(&self, id: &str, bytes: u64) {
        let give_up = Instant::now() + Duration::from_secs(120);
        while self.status(id).log_bytes < bytes && Instant::now() < give_up {
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn output(&self, id: &str, offset: Option<u64>, max: Option<usize>) -> super::job_output::JobOutput {
        self.store.output(&self.fixture.workspace, id, offset, max).unwrap()
    }

    fn job_dir(&self, id: &str) -> PathBuf {
        self.root.join("fixture/fixture/jobs").join(id)
    }

    fn jobs(&self) -> Vec<JobStatus> {
        self.store
            .list(&self.fixture.workspace)
            .unwrap()
            .into_iter()
            .filter_map(|entry| match entry {
                JobEntry::Job(status) => Some(status),
                JobEntry::Unreadable { .. } => None,
            })
            .collect()
    }
}

impl Drop for Jobs {
    fn drop(&mut self) {
        if systemd::available() {
            let pattern = format!("{}-job-*", self.prefix);
            let _ = systemd::systemctl(&["kill", "--signal=SIGKILL", &pattern], "cleanup");
            let _ = systemd::systemctl(&["stop", &pattern], "cleanup");
            let _ = systemd::systemctl(&["reset-failed", &pattern], "cleanup");
        }
    }
}

fn units_matching(pattern: &str) -> String {
    systemd::systemctl(&["list-units", "--all", "--no-legend", pattern], "list").map(|output| output.stdout_text()).unwrap_or_default()
}

fn mode(path: &Path) -> u32 {
    std::fs::metadata(path).unwrap().mode() & 0o7777
}

fn alive(pid: &str) -> bool {
    std::fs::read_to_string(format!("/proc/{}/stat", pid.trim())).is_ok_and(|stat| !stat.contains(") Z "))
}

fn gone(pid: &str) -> bool {
    (0..6000).any(|_| {
        let dead = !alive(pid);
        if !dead {
            std::thread::sleep(Duration::from_millis(10));
        }
        dead
    })
}

/// A finished job's record, written by hand: for the rules that need no systemd.
fn record(jobs: &Jobs, id: &str, state: JobState, ended_ms_ago: u64) -> JobRecord {
    let now = super::job_store::now_ms();
    let record = JobRecord {
        version: RECORD_VERSION,
        id: id.to_string(),
        repo: "fixture".to_string(),
        workspace: "fixture".to_string(),
        unit: jobs.store.unit_name(&jobs.fixture.workspace, id),
        command: format!("echo {id}"),
        cwd: ".".to_string(),
        started_at_ms: now.saturating_sub(ended_ms_ago + 1000),
        max_runtime_s: 3600,
        head_cap: JOB_OUTPUT_HEAD_BYTES,
        tail_cap: JOB_OUTPUT_TAIL_BYTES,
        env_names: Vec::new(),
        state,
        exit_status: (state == JobState::Exited).then_some(0),
        ended_at_ms: (state != JobState::Running).then_some(now.saturating_sub(ended_ms_ago)),
        end_reason: None,
    };
    let directory = jobs.job_dir(id);
    super::job_store::create_private_dir(&directory).unwrap();
    write_record(&directory, &record).unwrap();
    record
}

// --- everywhere: record, ids, reader, wrapper -----------------------------------

#[test]
fn jobs_record_format_is_versioned_and_private() {
    let jobs = Jobs::new();
    let written = record(&jobs, "j3", JobState::Exited, 5000);
    let directory = jobs.job_dir("j3");
    assert_eq!(read_record(&directory).unwrap(), written);
    assert_eq!(mode(&directory), 0o700);
    assert_eq!(mode(&directory.join("job.json")), 0o600);
    assert_eq!(mode(&jobs.root.join("fixture")), 0o700, "the whole tree is private");
    let json: serde_json::Value = serde_json::from_slice(&std::fs::read(directory.join("job.json")).unwrap()).unwrap();
    assert_eq!(json["version"], 1);
    assert_eq!(json["state"], "exited");
    let mut fields: Vec<&str> = json.as_object().unwrap().keys().map(String::as_str).collect();
    fields.sort();
    assert_eq!(
        fields,
        vec![
            "command", "cwd", "end_reason", "ended_at_ms", "env_names", "exit_status", "head_cap", "id", "max_runtime_s", "repo", "started_at_ms", "state",
            "tail_cap", "unit", "version", "workspace"
        ]
    );
    // Records this program does not understand are named, never interpreted.
    for (content, reason) in [
        ("{\"version\": 2, \"id\": \"j4\"}", "format version 2"),
        ("{\"id\": \"j4\"}", "no format version"),
        ("not json", "not valid JSON"),
        ("{\"version\": 1}", "expected fields"),
    ] {
        let other = jobs.job_dir("j4");
        super::job_store::create_private_dir(&other).unwrap();
        std::fs::write(other.join("job.json"), content).unwrap();
        assert!(read_record(&other).unwrap_err().contains(reason), "{content}");
        let entries = jobs.store.list(&jobs.fixture.workspace).unwrap();
        assert!(matches!(&entries[1], JobEntry::Unreadable { id, .. } if id == "j4"), "{entries:?}");
        assert!(render_jobs(&entries).contains("job j4: not usable") && render_jobs(&entries).contains("left untouched"), "{}", render_jobs(&entries));
        let error = jobs.store.status(&jobs.fixture.workspace, "j4").unwrap_err().to_string();
        assert!(error.contains("job `j4` cannot be used") && error.contains(reason), "{error}");
    }
    assert!(jobs.job_dir("j4").join("job.json").exists(), "an unreadable job is never removed by listing");
}

#[test]
fn jobs_ids_are_checked() {
    let jobs = Jobs::new();
    for (id, number) in [("j1", Some(1)), ("j27", Some(27)), ("j999999999", Some(999_999_999))] {
        assert_eq!(job_number(id), number);
    }
    for bad in ["", "j", "j0", "j01", "1", "J1", "j1 ", "j-1", "j1/..", "../j1", "j1\0", "j1234567890", "job1"] {
        assert_eq!(job_number(bad), None, "{bad:?}");
        let error = jobs.store.status(&jobs.fixture.workspace, bad).unwrap_err();
        assert!(matches!(error, Error::Job(_)) && error.to_string().contains("is not a job id"), "{bad:?}: {error}");
        assert!(jobs.store.output(&jobs.fixture.workspace, bad, None, None).is_err());
        assert!(jobs.store.kill(&jobs.fixture.workspace, bad).is_err());
    }
    let unknown = jobs.store.status(&jobs.fixture.workspace, "j5").unwrap_err().to_string();
    assert_eq!(unknown, "no job `j5` in workspace `fixture` (it has no jobs)");
    record(&jobs, "j2", JobState::Exited, 0);
    let unknown = jobs.store.status(&jobs.fixture.workspace, "j5").unwrap_err().to_string();
    assert_eq!(unknown, "no job `j5` in workspace `fixture` (its jobs: j2)");
    assert_eq!(render_jobs(&[]), "no jobs in this workspace");
    assert!(JobStore::at(&jobs.root, "Bad Prefix").is_err(), "a prefix becomes part of a unit name");
}

#[test]
fn jobs_without_a_user_manager_are_refused_and_old_records_are_unverifiable() {
    let mut jobs = Jobs::new();
    jobs.store.pretend_unavailable = Some("no user manager socket in /run/user/0".to_string());
    let error = jobs.try_start(&JobRequest::new("echo hi")).unwrap_err();
    assert_eq!(
        error.to_string(),
        "background jobs are not available: no user manager socket in /run/user/0. Run the command in the foreground with a timeout instead"
    );
    assert!(!jobs.root.join("fixture/fixture/jobs").exists(), "nothing is created for a refused job");
    // Failure to reach the manager does not prove that any job has stopped,
    // even when the command has written an exit file (its children may run).
    record(&jobs, "j1", JobState::Running, 0);
    record(&jobs, "j2", JobState::Running, 0);
    std::fs::write(jobs.job_dir("j2").join("exit"), "7\n").unwrap();
    for id in ["j1", "j2"] {
        let error = jobs.store.status(&jobs.fixture.workspace, id).unwrap_err().to_string();
        assert!(error.contains("cannot be observed") && error.contains("not evidence that it stopped"), "{error}");
        assert_eq!(read_record(&jobs.job_dir(id)).unwrap().state, JobState::Running);
    }
    assert!(jobs.store.list(&jobs.fixture.workspace).unwrap().iter().all(|entry| matches!(entry, JobEntry::Unreadable { .. })));
    let refusal = jobs.store.close_guard(&jobs.fixture.workspace).unwrap_err().to_string();
    assert!(refusal.contains("unverifiable job records (j1, j2)") && refusal.contains("was not closed"), "{refusal}");
}

#[test]
fn jobs_output_reader_follows_by_offset_without_splitting_characters() {
    let jobs = Jobs::new();
    record(&jobs, "j1", JobState::Exited, 0);
    let text = "zeile mit ümläuten und 日本語\n".repeat(200);
    std::fs::write(jobs.job_dir("j1").join("output.log"), &text).unwrap();
    // Read everything in small pieces; the pieces join to the original.
    let (mut offset, mut joined, mut rounds) = (0u64, String::new(), 0);
    loop {
        let piece = jobs.output("j1", Some(offset), Some(1024));
        assert!(!piece.text.contains('\u{FFFD}'), "a character was split at offset {offset}");
        assert_eq!(piece.offset, offset);
        joined.push_str(&piece.text);
        rounds += 1;
        if piece.next_offset == offset {
            break;
        }
        offset = piece.next_offset;
    }
    assert_eq!(joined, text);
    assert!(rounds > 5, "{rounds}");
    let first = jobs.output("j1", Some(0), Some(1024));
    assert!(first.render().ends_with(&format!("; output bytes 0-{} of {}; continue with offset={}]", first.next_offset, text.len(), first.next_offset)), "{}", first.render());
    // Without an offset: the end. Past the end: nothing, and the same offset back.
    let end = jobs.output("j1", None, Some(1024));
    assert!(text.ends_with(&end.text) && end.text.len() > 900 && end.next_offset == text.len() as u64, "{}", end.text.len());
    assert!(end.render().ends_with(&format!("; end of output at byte {}]", text.len())), "{}", end.render());
    let past = jobs.output("j1", Some(10_000_000), None);
    assert_eq!((past.text.as_str(), past.offset, past.next_offset), ("", text.len() as u64, text.len() as u64));
    // The limit has a floor and a ceiling.
    assert!(jobs.output("j1", Some(0), Some(1)).text.len() > 900);
    assert_eq!(jobs.output("j1", Some(0), Some(usize::MAX)).text, text);
    // A job that still runs: the reader does not guess at half a character.
    record(&jobs, "j2", JobState::Exited, 0);
    std::fs::write(jobs.job_dir("j2").join("output.log"), "abc\u{65e5}".as_bytes()[..5].to_vec()).unwrap();
    assert_eq!(jobs.output("j2", Some(0), None).text, "abc\u{FFFD}", "a finished job's last bytes are shown as they are");
    // No output file at all is an empty output, not an error.
    record(&jobs, "j3", JobState::Exited, 0);
    assert_eq!(jobs.output("j3", None, None).render(), "[job j3: exited with status 0 after 1 s; end of output at byte 0]");
}

#[test]
fn jobs_output_from_the_tail_file_says_the_middle_is_missing() {
    let jobs = Jobs::new();
    let written = record(&jobs, "j1", JobState::Exited, 0);
    // A v1 record's policy caps must remain the real configured caps.
    write_record(&jobs.job_dir("j1"), &written).unwrap();
    std::fs::write(jobs.job_dir("j1").join("output.log"), "h".repeat(2000)).unwrap();
    std::fs::write(jobs.job_dir("j1").join("output.tail"), format!("{}\nTHE VERDICT\n", "t".repeat(3000))).unwrap();
    let status = jobs.status("j1");
    assert!(status.head_cap_reached());
    assert!(status.render().contains("2000 bytes of output, cap of 16777216 reached"), "{}", status.render());
    let end = jobs.output("j1", None, Some(1024));
    assert!(end.from_tail && end.text.ends_with("\nTHE VERDICT\n") && end.text.len() == 1024);
    assert!(
        end.render().ends_with("THE VERDICT\n[job j1: exited with status 0 after 1 s. Its output exceeded 16777216 bytes: this is the end of it, the middle is missing; the beginning is at offset=0]"),
        "{}",
        end.render()
    );
    // The beginning is still readable by offset, and its end is marked as a cut.
    let head = jobs.output("j1", Some(1500), None);
    assert_eq!((head.text.len(), head.from_tail, head.next_offset), (500, false, 2000));
    assert!(head.render().contains("later output was dropped at the cap"), "{}", head.render());
}

/// Run the wrapper directly, as the unit would, without systemd.
fn run_wrapper(directory: &Path, head: u64, tail: u64, command: &str) -> std::process::ExitStatus {
    wrapper_command(directory, head, tail, command).status().unwrap()
}

fn wrapper_command(directory: &Path, head: u64, tail: u64, command: &str) -> Command {
    let bytes = std::fs::read(directory.join("env")).unwrap();
    let env: std::collections::BTreeMap<String, String> = bytes.split(|b| *b == 0).filter(|s| !s.is_empty())
        .map(|s| std::str::from_utf8(s).unwrap().split_once('=').unwrap())
        .map(|(k,v)| (k.to_string(), v.to_string())).collect();
    std::fs::write(directory.join("env"), serde_json::to_vec(&env).unwrap()).unwrap();
    super::job_store::write_private(&directory.join("output.log"), b"").unwrap();
    let meta = std::fs::metadata(directory).unwrap();
    let mut cmd = Command::new(super::adapter::PYTHON);
    cmd.env_clear().args(["-I", "-c", super::adapter::JOB]).arg(directory.ancestors().nth(4).unwrap())
        .args(["r", "w", "jobs", "j1"]).args([meta.dev().to_string(), meta.ino().to_string(), head.to_string(), tail.to_string()])
        .args(["/bin/bash", "-c", command]);
    cmd
}

#[test]
fn jobs_wrapper_records_output_head_tail_and_exit_status() {
    let temp = tempfile::tempdir().unwrap();
    let job_dir = temp.path().join("r/w/jobs/j1");
    std::fs::create_dir_all(&job_dir).unwrap();
    let directory = job_dir.as_path();
    let prepare = |environment: &[u8]| {
        for file in ["output.log", "output.tail", "exit"] {
            let _ = std::fs::remove_file(directory.join(file));
        }
        std::fs::write(directory.join("env"), environment).unwrap();
    };
    // Under the cap: everything in the log, no tail file, the status recorded.
    prepare(b"PATH=/usr/bin:/bin\0GREETING=hello world\nsecond line\0");
    let status = run_wrapper(directory, 1000, 100, "echo \"$GREETING\"; echo to-stderr >&2; exit 7");
    assert_eq!(status.code(), Some(7));
    assert_eq!(std::fs::read_to_string(directory.join("output.log")).unwrap(), "hello world\nsecond line\nto-stderr\n");
    assert_eq!(std::fs::read_to_string(directory.join("exit")).unwrap(), "7\n");
    assert!(!directory.join("output.tail").exists() && !directory.join("output.tail.tmp").exists() && !directory.join("exit.tmp").exists());
    assert!(!directory.join("env").exists(), "the environment file is deleted once it is loaded");
    assert_eq!(mode(&directory.join("exit")), 0o600);

    // Over the cap: exactly the head, the last bytes in the tail, nothing lost in between the two reads.
    prepare(b"PATH=/usr/bin:/bin\0");
    let status = run_wrapper(directory, 1000, 100, "printf 'HEAD-START '; head -c 50000 /dev/zero | tr '\\0' 'm'; printf ' TAIL-END\\n'");
    assert_eq!(status.code(), Some(0));
    let log = std::fs::read_to_string(directory.join("output.log")).unwrap();
    let tail = std::fs::read_to_string(directory.join("output.tail")).unwrap();
    assert_eq!((log.len(), tail.len()), (1000, 100));
    assert!(log.starts_with("HEAD-START mmm") && tail.ends_with("mmm TAIL-END\n"), "{tail}");
    assert_eq!(mode(&directory.join("output.tail")), 0o600);
    // Exactly at the cap there is no tail.
    prepare(b"PATH=/usr/bin:/bin\0");
    run_wrapper(directory, 1000, 100, "head -c 1000 /dev/zero | tr '\\0' 'x'");
    assert_eq!(std::fs::metadata(directory.join("output.log")).unwrap().len(), 1000);
    assert!(!directory.join("output.tail").exists());
    // The command's own umask is not touched by the wrapper's private files.
    prepare(b"PATH=/usr/bin:/bin\0");
    run_wrapper(directory, 1000, 100, "umask");
    let command_umask = std::fs::read_to_string(directory.join("output.log")).unwrap();
    assert_ne!(command_umask.trim(), "0077", "the wrapper leaked its umask into the command");
    // The command sees exactly the file's variables (plus what bash adds itself).
    prepare(b"PATH=/usr/bin:/bin\0ONLY=this\0");
    run_wrapper(directory, 10_000, 100, "env");
    let seen = std::fs::read_to_string(directory.join("output.log")).unwrap();
    let mut names: Vec<&str> = seen.lines().filter_map(|line| line.split('=').next()).filter(|name| !["PWD", "SHLVL", "OLDPWD", "_"].contains(name)).collect();
    names.sort();
    assert_eq!(names, vec!["ONLY", "PATH"], "{seen}");
}

#[test]
fn jobs_wrapper_survives_sigterm_to_record_status_and_tail() {
    use std::os::unix::process::CommandExt;
    let temp = tempfile::tempdir().unwrap();
    let job_dir = temp.path().join("r/w/jobs/j1");
    std::fs::create_dir_all(&job_dir).unwrap();
    let directory = job_dir.as_path();
    std::fs::write(directory.join("env"), b"PATH=/usr/bin:/bin\0").unwrap();
    let mut command = wrapper_command(directory, 200, 50, "head -c 1000 /dev/zero | tr '\\0' 'a'; echo ' LAST WORDS'; echo ready > ready; sleep 600");
    command.current_dir(directory).process_group(0);
    let mut child = command.spawn().unwrap();
    let give_up = Instant::now() + Duration::from_secs(120);
    while !directory.join("ready").exists() && Instant::now() < give_up {
        std::thread::sleep(Duration::from_millis(10));
    }
    // What systemd does on a polite stop: SIGTERM to every process of the unit.
    // SAFETY: kill(2) touches no memory; the group is the child this test started.
    unsafe { libc::kill(-(child.id() as i32), libc::SIGTERM) };
    let status = child.wait().unwrap();
    assert_eq!(status.code(), Some(143), "the wrapper ends with the command's status, 128 + SIGTERM");
    assert_eq!(std::fs::read_to_string(directory.join("exit")).unwrap(), "143\n");
    assert_eq!(std::fs::metadata(directory.join("output.log")).unwrap().len(), 200);
    assert!(std::fs::read_to_string(directory.join("output.tail")).unwrap().ends_with("aaa LAST WORDS\n"));
}

#[test]
fn jobs_limits_and_retention_without_systemd() {
    let mut jobs = Jobs::new();
    // The limits are checked before systemd is needed; refuse via the hook
    // after them so this runs everywhere.
    jobs.store.pretend_unavailable = Some("test".to_string());
    for index in 1..=JOBS_KEPT_MAX as u64 + 3 {
        record(&jobs, &format!("j{index}"), JobState::Exited, index * 1000);
    }
    record(&jobs, "j40", JobState::Exited, JOB_RETENTION.as_millis() as u64 + 60_000);
    assert_eq!(jobs.jobs().len(), JOBS_KEPT_MAX + 3, "listing prunes expired finished jobs");
    let _ = jobs.try_start(&JobRequest::new("echo hi"));
    assert_eq!(jobs.jobs().len(), JOBS_KEPT_MAX + 3, "a refused start changes nothing");
}

// --- with a systemd user manager -------------------------------------------------

#[test]
fn jobs_run_to_completion_and_keep_their_output() {
    let Some(jobs) = Jobs::with_systemd() else { return };
    jobs.fixture.write("sub/file.txt", "x\n");
    let mut request = JobRequest::new("echo \"in $(pwd)\"; ls; echo problem >&2; exit 3");
    request.cwd = "sub".to_string();
    let started = jobs.try_start(&request).unwrap();
    assert_eq!((started.record.id.as_str(), started.record.state, started.record.cwd.as_str()), ("j1", JobState::Running, "sub"));
    assert_eq!(started.record.unit, jobs.store.unit_name(&jobs.fixture.workspace, "j1"));
    let done = jobs.finished("j1");
    assert_eq!((done.record.state, done.record.exit_status, done.record.end_reason.clone()), (JobState::Exited, Some(3), None));
    assert!(done.record.ended_at_ms.unwrap() >= done.record.started_at_ms);
    let output = jobs.output("j1", Some(0), None);
    assert_eq!(output.text, format!("in {}\nfile.txt\nproblem\n", jobs.fixture.root.join("sub").display()));
    assert!(output.render().ends_with(&format!("; end of output at byte {}]", output.text.len())), "{}", output.render());
    assert!(done.render().starts_with("job j1: exited with status 3 after "), "{}", done.render());
    assert!(done.render().ends_with("cwd sub\n  $ echo \"in $(pwd)\"; ls; echo problem >&2; exit 3"), "{}", done.render());
    // On disk: private, complete, and without the environment file.
    let directory = jobs.job_dir("j1");
    assert_eq!(mode(&directory), 0o700);
    for file in ["job.json", "output.log", "exit"] {
        assert_eq!(mode(&directory.join(file)), 0o600, "{file}");
    }
    assert!(!directory.join("env").exists());
    assert_eq!(read_record(&directory).unwrap(), done.record, "the outcome is in the record");
    // A failed unit is forgotten by systemd once its outcome is recorded.
    assert_eq!(units_matching(&format!("{}-job-*", jobs.prefix)).trim(), "");
    // The next job gets the next id.
    assert_eq!(jobs.start("true").record.id, "j2");
    assert_eq!(jobs.finished("j2").record.exit_status, Some(0));
}

#[test]
fn jobs_are_found_again_by_a_new_store_and_do_not_belong_to_this_process() {
    let Some(jobs) = Jobs::with_systemd() else { return };
    let started = jobs.start("echo $$ > job.pid; echo begun; sleep 2; echo finished");
    // Not in this process's control group, and not its descendant: a restart
    // of this process (KillMode=control-group) cannot take the job with it.
    let own = std::fs::read_to_string("/proc/self/cgroup").unwrap();
    let own = own.lines().find_map(|line| line.strip_prefix("0::")).unwrap().to_string();
    let unit = systemd::show(&started.record.unit, &["ControlGroup", "MainPID"]).unwrap();
    let job_group = unit["ControlGroup"].clone();
    assert!(job_group.ends_with(&started.record.unit), "{job_group}");
    assert!(!job_group.starts_with(&format!("{own}/")) && job_group != own, "the job is inside our control group: {job_group} vs {own}");
    let give_up = Instant::now() + Duration::from_secs(60);
    while !jobs.fixture.root.join("job.pid").exists() && Instant::now() < give_up {
        std::thread::sleep(Duration::from_millis(10));
    }
    let mut pid: u32 = std::fs::read_to_string(jobs.fixture.root.join("job.pid")).unwrap().trim().parse().unwrap();
    let me = std::process::id();
    for _ in 0..64 {
        assert_ne!(pid, me, "the job descends from this process");
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
        let parent = stat.rsplit_once(')').and_then(|(_, rest)| rest.split_whitespace().nth(1)).and_then(|field| field.parse().ok()).unwrap_or(0);
        if parent <= 1 {
            break;
        }
        pid = parent;
    }
    jobs.wrote("j1", 6);
    // "After the restart": nothing but the directory is left of the first store.
    let prefix = jobs.prefix.clone();
    let reopened = JobStore::at(&jobs.root, &prefix).unwrap();
    let found = reopened.list(&jobs.fixture.workspace).unwrap();
    assert!(matches!(&found[..], [JobEntry::Job(job)] if job.record.id == "j1" && job.record.state == JobState::Running), "{found:?}");
    let early = reopened.output(&jobs.fixture.workspace, "j1", Some(0), None).unwrap();
    assert_eq!(early.text, "begun\n");
    assert!(early.render().contains("; output up to byte 6; ask again with offset=6 for what follows]"), "{}", early.render());
    let done = jobs.finished("j1");
    assert_eq!((done.record.state, done.record.exit_status), (JobState::Exited, Some(0)));
    let rest = reopened.output(&jobs.fixture.workspace, "j1", Some(early.next_offset), None).unwrap();
    assert_eq!(rest.text, "finished\n");
}

#[test]
fn jobs_do_not_see_the_user_managers_environment() {
    let Some(jobs) = Jobs::with_systemd() else { return };
    // The manager certainly has these; a service started by it inherits them.
    let manager = systemd::systemctl(&["show-environment"], "show-environment").unwrap().stdout_text();
    assert!(manager.lines().any(|line| line.starts_with("XDG_RUNTIME_DIR=")), "the test needs a variable the manager has: {manager}");
    let mut request = JobRequest::new("env");
    request.env = vec![("REQUESTED".to_string(), "a value with spaces\nand a second line".to_string())];
    jobs.try_start(&request).unwrap();
    assert_eq!(jobs.finished("j1").record.exit_status, Some(0));
    let seen = jobs.output("j1", Some(0), None).text;
    let mut names: Vec<&str> = seen.lines().filter(|line| line.contains('=')).filter_map(|line| line.split('=').next()).collect();
    names.retain(|name| !["PWD", "SHLVL", "OLDPWD", "_"].contains(name));
    names.sort();
    assert_eq!(names, vec!["GIT_TERMINAL_PROMPT", "HOME", "LANG", "PATH", "REQUESTED", "TERM"], "{seen}");
    for inherited in ["XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "INVOCATION_ID", "JOURNAL_STREAM", "MANAGERPID", "SYSTEMD_EXEC_PID"] {
        assert!(!seen.contains(inherited), "`{inherited}` reached the job: {seen}");
    }
    assert!(seen.contains("REQUESTED=a value with spaces\nand a second line\n"), "{seen}");
    // The record names the variable and nothing of its value; no file holds the value.
    let record = std::fs::read_to_string(jobs.job_dir("j1").join("job.json")).unwrap();
    assert!(record.contains("\"REQUESTED\"") && !record.contains("a value with spaces"), "{record}");
    assert!(!jobs.job_dir("j1").join("env").exists());
    // Unless it is requested: then the job gets the requested value, not the manager's.
    let mut asked = JobRequest::new("echo \"[$XDG_RUNTIME_DIR]\"");
    asked.env = vec![("XDG_RUNTIME_DIR".to_string(), "/requested/dir".to_string())];
    jobs.try_start(&asked).unwrap();
    jobs.finished("j2");
    assert_eq!(jobs.output("j2", Some(0), None).text, "[/requested/dir]\n");
}

#[test]
fn jobs_pass_the_command_text_through_unchanged() {
    let Some(jobs) = Jobs::with_systemd() else { return };
    jobs.start("X=inner; echo \"$$ ${X} $X %h %% $0 ${UNSET:-default}\" '$HOME ${HOME} %n'");
    assert_eq!(jobs.finished("j1").record.exit_status, Some(0));
    let line = jobs.output("j1", Some(0), None).text;
    let pid = line.split(' ').next().unwrap();
    assert!(pid.parse::<u32>().is_ok(), "`$$` did not reach bash: {line}");
    assert_eq!(&line[pid.len()..], " inner inner %h %% /bin/bash default $HOME ${HOME} %n\n");
}

#[test]
fn jobs_can_be_killed_with_everything_they_started() {
    let Some(jobs) = Jobs::with_systemd() else { return };
    jobs.start("setsid sleep 600 & echo $! > detached.pid; echo $$ > shell.pid; echo running; sleep 600");
    let give_up = Instant::now() + Duration::from_secs(60);
    while !jobs.fixture.root.join("shell.pid").exists() && Instant::now() < give_up {
        std::thread::sleep(Duration::from_millis(10));
    }
    jobs.wrote("j1", 8);
    let started = Instant::now();
    let killed = jobs.store.kill(&jobs.fixture.workspace, "j1").unwrap();
    assert!(started.elapsed() < Duration::from_secs(60), "took {:?}", started.elapsed());
    assert_eq!((killed.record.state, killed.record.exit_status), (JobState::Killed, Some(143)));
    assert!(killed.render().starts_with("job j1: stopped by job_kill after "), "{}", killed.render());
    for file in ["detached.pid", "shell.pid"] {
        let pid = std::fs::read_to_string(jobs.fixture.root.join(file)).unwrap();
        assert!(gone(&pid), "{file} {pid} survived job_kill");
    }
    assert_eq!(jobs.output("j1", Some(0), None).text, "running\n");
    // Killing again, or killing a job that ended by itself, changes nothing.
    assert_eq!(jobs.store.kill(&jobs.fixture.workspace, "j1").unwrap(), killed);
    jobs.start("exit 5");
    jobs.finished("j2");
    let ended = jobs.store.kill(&jobs.fixture.workspace, "j2").unwrap();
    assert_eq!((ended.record.state, ended.record.exit_status), (JobState::Exited, Some(5)));
    assert_eq!(units_matching(&format!("{}-job-*", jobs.prefix)).trim(), "");
}

#[test]
fn jobs_stubborn_command_is_killed_hard() {
    let Some(jobs) = Jobs::with_systemd() else { return };
    jobs.start("trap '' TERM; echo $$ > shell.pid; echo armed; while true; do sleep 600; done");
    let give_up = Instant::now() + Duration::from_secs(60);
    while !jobs.fixture.root.join("shell.pid").exists() && Instant::now() < give_up {
        std::thread::sleep(Duration::from_millis(10));
    }
    jobs.wrote("j1", 6);
    let killed = jobs.store.kill(&jobs.fixture.workspace, "j1").unwrap();
    // Killed hard, so the wrapper could not record a status; that it was
    // job_kill is known from the marker.
    assert_eq!((killed.record.state, killed.record.exit_status), (JobState::Killed, None));
    assert!(gone(&std::fs::read_to_string(jobs.fixture.root.join("shell.pid")).unwrap()));
}

#[test]
fn jobs_stop_at_their_runtime_limit() {
    let Some(jobs) = Jobs::with_systemd() else { return };
    let mut request = JobRequest::new("echo begun; setsid sleep 600 & echo $! > detached.pid; sleep 600");
    request.max_runtime = Some(Duration::from_millis(10));
    let started = jobs.try_start(&request).unwrap();
    assert_eq!(started.record.max_runtime_s, 1, "raised to the minimum");
    let done = jobs.finished("j1");
    assert_eq!(done.record.state, JobState::TimedOut);
    assert_eq!(done.record.exit_status, Some(143), "stopped politely first, so the wrapper recorded the status");
    assert!(done.render().starts_with("job j1: stopped at its runtime limit of 1 s; "), "{}", done.render());
    assert!(gone(&std::fs::read_to_string(jobs.fixture.root.join("detached.pid")).unwrap()));
    assert_eq!(jobs.output("j1", Some(0), None).text, "begun\n");
    // The limits of the limit.
    for (asked, effective) in [(None, 3600), (Some(Duration::from_secs(86_400)), 6 * 3600), (Some(Duration::from_secs(90)), 90)] {
        let mut request = JobRequest::new("true");
        request.max_runtime = asked;
        let job = jobs.try_start(&request).unwrap();
        assert_eq!(job.record.max_runtime_s, effective);
        jobs.finished(&job.record.id);
    }
}

#[test]
fn jobs_killed_from_outside_are_failed_or_lost() {
    let Some(jobs) = Jobs::with_systemd() else { return };
    // Killed hard by someone else while systemd still knows: failed, with its reason.
    let first = jobs.start("echo one; sleep 600");
    jobs.wrote("j1", 4);
    systemd::kill(&first.record.unit, "SIGKILL").unwrap();
    let failed = jobs.finished("j1");
    assert_eq!((failed.record.state, failed.record.exit_status, failed.record.end_reason.as_deref()), (JobState::Failed, None, Some("signal")));
    assert!(failed.render().starts_with("job j1: ended without an exit status (signal); "), "{}", failed.render());
    // The same, but systemd has forgotten the unit before anyone looked
    // (as after a restart of the user session): lost.
    let second = jobs.start("echo two; sleep 600");
    jobs.wrote("j2", 4);
    systemd::kill(&second.record.unit, "SIGKILL").unwrap();
    let give_up = Instant::now() + Duration::from_secs(60);
    while systemd::is_active(&systemd::show(&second.record.unit, &["ActiveState"]).unwrap()) && Instant::now() < give_up {
        std::thread::sleep(Duration::from_millis(20));
    }
    systemd::reset_failed(&second.record.unit);
    let lost = jobs.status("j2");
    assert_eq!((lost.record.state, lost.record.exit_status), (JobState::Lost, None));
    assert_eq!(jobs.output("j2", Some(0), None).text, "two\n", "what it wrote is still there");
}

#[test]
fn jobs_output_cap_keeps_head_and_tail() {
    let Some(jobs) = Jobs::with_systemd() else { return };
    let total = JOB_OUTPUT_HEAD_BYTES + 3_000_000;
    jobs.start(&format!("printf 'FIRST LINE\\n'; head -c {total} /dev/zero | tr '\\0' 'm'; printf '\\nTHE VERDICT\\n'"));
    let done = jobs.finished("j1");
    assert_eq!((done.record.state, done.record.exit_status), (JobState::Exited, Some(0)));
    assert_eq!((done.log_bytes, done.tail_bytes), (JOB_OUTPUT_HEAD_BYTES, JOB_OUTPUT_TAIL_BYTES));
    assert!(done.head_cap_reached());
    assert!(done.render().contains(&format!("{JOB_OUTPUT_HEAD_BYTES} bytes of output, cap of {JOB_OUTPUT_HEAD_BYTES} reached")), "{}", done.render());
    let end = jobs.output("j1", None, None);
    assert!(end.from_tail && end.text.ends_with("mmm\nTHE VERDICT\n"));
    assert!(end.render().contains("this is the end of it, the middle is missing"), "{}", &end.render()[end.render().len() - 200..]);
    let beginning = jobs.output("j1", Some(0), Some(1024));
    assert!(beginning.text.starts_with("FIRST LINE\nmmm") && !beginning.from_tail);
    assert_eq!(mode(&jobs.job_dir("j1").join("output.tail")), 0o600);
}

#[test]
fn jobs_limits_running_and_kept_records() {
    let Some(jobs) = Jobs::with_systemd() else { return };
    for _ in 0..JOBS_RUNNING_MAX {
        jobs.start("sleep 600");
    }
    let error = jobs.try_start(&JobRequest::new("echo third")).unwrap_err();
    assert_eq!(
        error.to_string(),
        "workspace `fixture` already has 2 running jobs (j1, j2), the limit is 2; wait for one or stop it with job_kill"
    );
    assert_eq!(jobs.jobs().len(), 2);
    let guard = jobs.store.close_guard(&jobs.fixture.workspace).unwrap_err();
    assert_eq!(guard.to_string(), "workspace `fixture` has running jobs (j1, j2) and was not closed; wait for them or stop them with job_kill");
    jobs.store.kill(&jobs.fixture.workspace, "j1").unwrap();
    // One running: more jobs fit. Finished ones are kept up to the limit, oldest first out.
    for index in 0..JOBS_KEPT_MAX {
        let job = jobs.start(&format!("echo job {index}"));
        jobs.finished(&job.record.id);
    }
    let ids: Vec<String> = jobs.jobs().iter().map(|job| job.record.id.clone()).collect();
    assert_eq!(ids.len(), JOBS_KEPT_MAX);
    assert!(ids.contains(&"j2".to_string()), "a running job is never pruned: {ids:?}");
    assert!(!ids.contains(&"j1".to_string()) && !ids.contains(&"j3".to_string()), "the oldest finished went first: {ids:?}");
    assert_eq!(ids.last().unwrap(), &format!("j{}", JOBS_RUNNING_MAX + JOBS_KEPT_MAX));
    // Age: a finished job past its retention goes at the next start, whatever the count.
    jobs.store.kill(&jobs.fixture.workspace, "j2").unwrap();
    let old_id = ids.last().unwrap().clone();
    let mut old = read_record(&jobs.job_dir(&old_id)).unwrap();
    old.ended_at_ms = Some(old.ended_at_ms.unwrap() - JOB_RETENTION.as_millis() as u64 - 1000);
    old.started_at_ms = old.ended_at_ms.unwrap().saturating_sub(1000);
    write_record(&jobs.job_dir(&old_id), &old).unwrap();
    let newest = jobs.start("true");
    jobs.finished(&newest.record.id);
    assert!(!jobs.job_dir(&old_id).exists(), "an expired job was kept");
    // Its id is not given out again: an id always means the same job.
    assert_eq!(newest.record.id, format!("j{}", JOBS_RUNNING_MAX + JOBS_KEPT_MAX + 1));
    assert!(matches!(jobs.store.status(&jobs.fixture.workspace, &old_id), Err(Error::Job(_))));
    jobs.store.close_guard(&jobs.fixture.workspace).unwrap();
}

#[test]
fn jobs_apply_the_same_checks_as_foreground_commands() {
    let Some(jobs) = Jobs::with_systemd() else { return };
    let refused = |request: JobRequest, policy: &ShellPolicy| jobs.store.start(&jobs.fixture.workspace, &request, policy, &bash_argv).unwrap_err();
    assert!(matches!(refused(JobRequest::new("touch ran; rm -rf /"), &ShellPolicy::Unrestricted), Error::CommandRefused(_)));
    assert!(matches!(refused(JobRequest::new("touch ran"), &ShellPolicy::Denied), Error::CommandRefused(_)));
    assert!(matches!(refused(JobRequest::new(""), &ShellPolicy::Unrestricted), Error::InvalidArgument(_)));
    let outside = JobRequest { cwd: "..".to_string(), ..JobRequest::new("touch ran") };
    assert_eq!(refused(outside, &ShellPolicy::Unrestricted).rule(), Some(super::error::Rule::ParentEscape));
    let bad_env = JobRequest { env: vec![("BAD NAME".to_string(), "x".to_string())], ..JobRequest::new("touch ran") };
    assert!(refused(bad_env, &ShellPolicy::Unrestricted).to_string().contains("is not valid"));
    assert!(!jobs.fixture.root.join("ran").exists() && !jobs.fixture.top.join("ran").exists());
    assert!(jobs.jobs().is_empty() && !jobs.job_dir("j1").exists(), "a refused job leaves no record");
    // The repository's environment applies to jobs as to foreground commands.
    let mut with_env = Jobs::with_systemd().unwrap();
    with_env.fixture.workspace.env = vec![("CARGO_TARGET_DIR".to_string(), "/shared/target".to_string())];
    with_env.start("echo \"$CARGO_TARGET_DIR\"");
    with_env.finished("j1");
    assert_eq!(with_env.output("j1", Some(0), None).text, "/shared/target\n");
}

#[test]
fn jobs_block_closing_their_workspace_until_they_are_done() {
    if !systemd::available() {
        eprintln!("skipped: no systemd user manager is reachable here");
        return;
    }
    // `close` uses the process's own store; under test that is a temp
    // directory and a unit prefix naming this test process.
    struct StopUnits(String);
    impl Drop for StopUnits {
        fn drop(&mut self) {
            let _ = systemd::systemctl(&["kill", "--signal=SIGKILL", &self.0], "cleanup");
            let _ = systemd::systemctl(&["reset-failed", &self.0], "cleanup");
        }
    }
    let _cleanup = StopUnits(format!("adapsis-wstest-{}-job-4-demo-10-jobs-close-*", std::process::id()));
    let repository = RepoFixture::new();
    let workspace = open(&repository.repository, "jobs-close", None).unwrap();
    let store = JobStore::standard();
    let job = store.start(&workspace, &JobRequest::new("sleep 600"), &ShellPolicy::Unrestricted, &bash_argv).unwrap();
    assert!(job.record.unit.starts_with(&format!("adapsis-wstest-{}-job-4-demo-10-jobs-close-", std::process::id())), "{}", job.record.unit);
    let error = close(&workspace).unwrap_err();
    assert_eq!(error.to_string(), "workspace `jobs-close` has running jobs (j1) and was not closed; wait for them or stop them with job_kill");
    assert!(workspace.path.is_dir(), "nothing was removed");
    store.kill(&workspace, "j1").unwrap();
    let state = super::job_store::workspace_dir(&super::job_store::state_root(), &workspace).unwrap();
    assert!(state.join("jobs/j1/job.json").is_file());
    close(&workspace).unwrap();
    assert!(!workspace.path.exists());
    assert!(!state.exists(), "the closed workspace's job records are removed with it");
}

#[test]
fn jobs_leave_no_unit_behind() {
    let Some(jobs) = Jobs::with_systemd() else { return };
    jobs.start("sleep 600");
    jobs.start("exit 9");
    jobs.finished("j2");
    let prefix = jobs.prefix.clone();
    assert!(units_matching(&format!("{prefix}-job-*")).contains("j1.service"));
    drop(jobs);
    let gone = (0..400).any(|_| {
        let left = units_matching(&format!("{prefix}-job-*"));
        if !left.trim().is_empty() {
            std::thread::sleep(Duration::from_millis(25));
        }
        left.trim().is_empty()
    });
    assert!(gone, "units left behind: {}", units_matching(&format!("{prefix}-job-*")));
}

/// Invoked as a separate test process by `jobs_survive_the_starter_being_killed`.
/// In an ordinary test run this is a no-op. It deliberately never returns,
/// so the parent proves a job survives losing the entire starting process.
#[test]
fn jobs_restart_child_helper() {
    let Some(root) = std::env::var_os("ADAPSIS_TEST_JOB_ROOT") else { return };
    let ws = PathBuf::from(std::env::var_os("ADAPSIS_TEST_JOB_WS").unwrap());
    let repo = super::registry::Repository {
        name: "fixture".into(), checkout: ws.clone(), worktree_parent: ws.with_file_name("worktrees"),
        base: "main".into(), read_roots: Vec::new(), env: Default::default(),
    };
    let workspace = super::lifecycle::Workspace::at(&repo, "fixture", &ws, "main").unwrap();
    let prefix = std::env::var("ADAPSIS_TEST_JOB_PREFIX").unwrap();
    let store = JobStore::at(Path::new(&root), &prefix).unwrap();
    let request = JobRequest::new("echo started; while [ ! -f finish-job ]; do sleep 0.05; done; echo survived");
    store.start(&workspace, &request, &ShellPolicy::Unrestricted, &bash_argv).unwrap();
    std::fs::write(ws.join("starter-ready"), b"ready").unwrap();
    loop { std::thread::sleep(Duration::from_secs(60)); }
}

#[test]
fn jobs_survive_the_starter_being_killed() {
    let Some(jobs) = Jobs::with_systemd() else { return };
    struct Starter(std::process::Child);
    impl Drop for Starter {
        fn drop(&mut self) { let _ = self.0.kill(); let _ = self.0.wait(); }
    }
    let child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "workspace::jobs_tests::jobs_restart_child_helper", "--nocapture"])
        .env("ADAPSIS_TEST_JOB_ROOT", &jobs.root)
        .env("ADAPSIS_TEST_JOB_WS", &jobs.fixture.root)
        .env("ADAPSIS_TEST_JOB_PREFIX", &jobs.prefix)
        .stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).spawn().unwrap();
    let mut child = Starter(child);
    let give_up = Instant::now() + Duration::from_secs(120);
    while !jobs.fixture.root.join("starter-ready").exists() && Instant::now() < give_up {
        assert!(child.0.try_wait().unwrap().is_none(), "the helper failed before starting the job");
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(jobs.fixture.root.join("starter-ready").exists());
    jobs.wrote("j1", 8);
    child.0.kill().unwrap();
    child.0.wait().unwrap();
    // A wholly new store reads the private record and asks the manager.
    let restored = JobStore::at(&jobs.root, &jobs.prefix).unwrap();
    assert_eq!(restored.status(&jobs.fixture.workspace, "j1").unwrap().record.state, JobState::Running);
    jobs.fixture.write("finish-job", "go");
    assert_eq!(jobs.finished("j1").record.exit_status, Some(0));
    assert_eq!(restored.output(&jobs.fixture.workspace, "j1", Some(0), None).unwrap().text, "started\nsurvived\n");
}

#[test]
fn jobs_reject_records_bound_to_somewhere_else() {
    let jobs = Jobs::new();
    let mut entry = record(&jobs, "j1", JobState::Running, 0);
    entry.unit = "unrelated-owner.service".into();
    write_record(&jobs.job_dir("j1"), &entry).unwrap();
    assert!(jobs.store.kill(&jobs.fixture.workspace, "j1").unwrap_err().to_string().contains("does not belong"));
    assert!(matches!(jobs.store.list(&jobs.fixture.workspace).unwrap()[0], JobEntry::Unreadable { .. }));
    assert_eq!(read_record(&jobs.job_dir("j1")).unwrap(), entry, "foreign records are untouched");
}

#[test]
fn jobs_unit_names_disambiguate_repositories_workspaces_and_stores() {
    let jobs = Jobs::new();
    let mut a = jobs.fixture.workspace.clone();
    a.repo = "a-b".into(); a.name = "c".into();
    let mut b = a.clone(); b.repo = "a".into(); b.name = "b-c".into();
    assert_ne!(jobs.store.unit_name(&a, "j1"), jobs.store.unit_name(&b, "j1"));
    let elsewhere = JobStore::at(&jobs.fixture.top.join("other-state"), &jobs.prefix).unwrap();
    assert_ne!(jobs.store.unit_name(&a, "j1"), elsewhere.unit_name(&a, "j1"));
}

#[test]
fn b1_job_payload_environment_cannot_mutate_supervisor_controls() {
    let Some(jobs) = Jobs::with_systemd() else { return };
    let startup = jobs.fixture.write("startup.sh", "echo startup >> startup-count\n");
    let mut request = JobRequest::new("printf '%s|%s|%s|%s' \"$dir\" \"$head_cap\" \"$tail_cap\" \"$status\"; echo");
    request.env = vec![("dir".into(), jobs.fixture.outside.to_str().unwrap().into()), ("head_cap".into(), "1".into()),
        ("tail_cap".into(), "2".into()), ("status".into(), "99".into()), ("IFS".into(), "=".into()),
        ("BASH_ENV".into(), startup.to_str().unwrap().into()), ("XDG_RUNTIME_DIR".into(), "ENV_VALUE_SENTINEL_772".into())];
    jobs.try_start(&request).unwrap();
    assert_eq!(jobs.finished("j1").record.exit_status, Some(0));
    assert_eq!(jobs.output("j1",Some(0),None).text, format!("{}|1|2|99\n",jobs.fixture.outside.display()));
    assert_eq!(std::fs::read_to_string(jobs.fixture.root.join("startup-count")).unwrap(), "startup\n");
    assert_eq!(std::fs::read_dir(&jobs.fixture.outside).unwrap().count(),0);
    let unit = jobs.status("j1").record.unit;
    let props = systemd::show(&unit,&["ExecStart"]).unwrap();
    assert!(!format!("{props:?}").contains("ENV_VALUE_SENTINEL_772"));
    // These names are all payload data, including shell-special names, when
    // the trusted launcher chooses a non-shell payload.
    let launcher = |_: &super::run::LaunchContext<'_>| vec!["/usr/bin/env".into()];
    let mut req = JobRequest::new("env");
    req.env = ["dir","head_cap","tail_cap","status","keep","xdg","name","IFS","BASHOPTS","SHELLOPTS","BASH_ENV"]
        .iter().map(|name| (name.to_string(),format!("value-{name}"))).collect();
    jobs.store.start(&jobs.fixture.workspace,&req,&ShellPolicy::Unrestricted,&launcher).unwrap();
    jobs.finished("j2");
    let output = jobs.output("j2",Some(0),None).text;
    for (name,value) in req.env { assert!(output.lines().any(|line| line==format!("{name}={value}")), "{name} was corrupted"); }
}

#[test]
fn b2_start_and_close_share_one_lifecycle_reservation() {
    if !systemd::available() { return; }
    let repo = RepoFixture::new();
    let ws = open(&repo.repository,"repair-race",None).unwrap();
    let gate = std::sync::Arc::new((std::sync::Barrier::new(2),std::sync::Barrier::new(2)));
    let mut store = JobStore::standard(); store.start_gate = Some(gate.clone());
    let worker_ws = ws.clone();
    let starter = std::thread::spawn(move || store.start(&worker_ws,&JobRequest::new("sleep 600"),&ShellPolicy::Unrestricted,&bash_argv));
    gate.0.wait(); // start holds the reservation after its empty-job snapshot.
    let (tx,rx) = std::sync::mpsc::channel();
    let closing_ws = ws.clone();
    let closer = std::thread::spawn(move || { tx.send(close(&closing_ws)).unwrap(); });
    assert!(matches!(rx.recv_timeout(Duration::from_millis(100)),Err(std::sync::mpsc::RecvTimeoutError::Timeout)));
    gate.1.wait();
    let started = starter.join().unwrap().unwrap();
    let closed = rx.recv_timeout(Duration::from_secs(30)).unwrap();
    closer.join().unwrap();
    let store = JobStore::standard();
    let retained = store.status(&ws,&started.record.id).unwrap();
    let _ = store.kill(&ws,&started.record.id);
    assert!(closed.unwrap_err().to_string().contains("running jobs"));
    assert!(ws.path.is_dir());
    assert_eq!(retained.record.state,JobState::Running);
}

#[test]
fn b3_enumeration_and_counter_errors_fail_closed_without_units() {
    let jobs = Jobs::new();
    let jobs_dir = jobs.root.join("fixture/fixture/jobs");
    std::fs::create_dir_all(jobs_dir.parent().unwrap()).unwrap();
    std::fs::write(&jobs_dir,b"not a directory").unwrap();
    assert!(jobs.store.list(&jobs.fixture.workspace).is_err());
    assert!(jobs.store.close_guard(&jobs.fixture.workspace).is_err());
    assert!(jobs.try_start(&JobRequest::new("touch ran")).is_err());
    assert!(!jobs.fixture.root.join("ran").exists());
    let injected = super::state_io::collect_names(vec![Ok("j1".into()),Err(std::io::Error::from(std::io::ErrorKind::PermissionDenied))].into_iter());
    assert!(injected.is_err(),"a per-entry error cannot disappear");
    std::fs::remove_file(&jobs_dir).unwrap(); std::fs::create_dir(&jobs_dir).unwrap();
    let dir = super::state_io::Dir::root(&jobs_dir,false).unwrap().unwrap();
    for bytes in [b"bogus".as_slice(),b"-1",b"4294967295",b"\xff"] {
        std::fs::write(jobs_dir.join("last-id"),bytes).unwrap();
        assert!(super::jobs::allocate(&dir).is_err());
        assert_eq!(std::fs::read(jobs_dir.join("last-id")).unwrap(),bytes);
        assert!(!jobs_dir.join("j1").exists());
    }
    std::fs::remove_file(jobs_dir.join("last-id")).unwrap(); std::fs::create_dir(jobs_dir.join("last-id")).unwrap();
    assert!(super::jobs::allocate(&dir).is_err());
}

#[test]
fn b3_actual_close_refuses_a_broken_jobs_tree() {
    let repo=RepoFixture::new();
    let ws=open(&repo.repository,"repair-enumeration",None).unwrap();
    let state=super::job_store::workspace_dir(&super::job_store::state_root(),&ws).unwrap();
    std::fs::create_dir_all(&state).unwrap();
    std::fs::write(state.join("jobs"),b"not-a-directory").unwrap();
    let refused=close(&ws);
    assert!(refused.is_err());
    assert!(ws.path.is_dir(),"close interpreted an enumeration error as no jobs");
    assert_eq!(std::fs::read(state.join("jobs")).unwrap(),b"not-a-directory");
    std::fs::remove_file(state.join("jobs")).unwrap();
    close(&ws).unwrap();
}

#[test]
fn b4_private_state_never_follows_links_at_any_depth() {
    for component in ["fixture", "fixture/fixture", "fixture/fixture/jobs", "fixture/fixture/jobs/j1"] {
        let jobs = Jobs::new();
        let link = jobs.root.join(component);
        std::fs::create_dir_all(link.parent().unwrap()).unwrap();
        let sentinel = jobs.fixture.outside.join("sentinel"); std::fs::write(&sentinel,b"keep").unwrap();
        super::test_support::symlink(&jobs.fixture.outside,&link);
        assert!(jobs.store.list(&jobs.fixture.workspace).is_err(),"{component}");
        assert!(jobs.store.forget(&jobs.fixture.workspace).is_err(),"{component}");
        assert!(jobs.try_start(&JobRequest::new("touch ran")).is_err());
        assert_eq!(std::fs::read(&sentinel).unwrap(),b"keep");
        assert!(!jobs.fixture.outside.join("jobs").exists());
    }
    for name in ["job.json","output.log","output.tail","env","last-id"] {
        let jobs=Jobs::new(); record(&jobs,"j1",JobState::Exited,0);
        let parent=if name=="last-id" { jobs.job_dir("j1").parent().unwrap().to_path_buf() } else { jobs.job_dir("j1") };
        let target=jobs.fixture.outside.join("sentinel"); std::fs::write(&target,b"keep").unwrap();
        let _=std::fs::remove_file(parent.join(name)); super::test_support::symlink(&target,parent.join(name));
        // Prune/forget either refuses during inspection or during nofollow
        // recursive removal; neither reads, overwrites nor deletes the target.
        assert!(jobs.store.forget(&jobs.fixture.workspace).is_err());
        assert_eq!(std::fs::read(&target).unwrap(),b"keep");
    }
    // Hold a directory, then replace its pathname with an outside symlink.
    let jobs=Jobs::new(); record(&jobs,"j1",JobState::Exited,0);
    let ws=jobs.root.join("fixture/fixture");
    let held=super::state_io::Dir::root(&ws,false).unwrap().unwrap();
    std::fs::rename(&ws,ws.with_file_name("parked")).unwrap();
    super::test_support::symlink(&jobs.fixture.outside,&ws);
    std::fs::write(jobs.fixture.outside.join("sentinel"),b"keep").unwrap();
    held.write("private",b"inside").unwrap();
    held.remove_tree("jobs".as_ref()).unwrap();
    assert!(!jobs.fixture.outside.join("private").exists());
    assert_eq!(std::fs::read(jobs.fixture.outside.join("sentinel")).unwrap(),b"keep");
}

#[test]
fn f3_type_correct_but_semantically_invalid_v1_records_are_preserved() {
    let jobs=Jobs::new();
    let valid=record(&jobs,"j1",JobState::Exited,0);
    let mutations: Vec<(&str,serde_json::Value)>=vec![
        ("head_cap",1.into()),("tail_cap",0.into()),("max_runtime_s",999999.into()),
        ("exit_status",serde_json::Value::Null),("ended_at_ms",0.into()),
        ("cwd","../outside".into()),("env_names",serde_json::json!(["BAD NAME"])),
        ("state","running".into()),("exit_status",999.into()),
    ];
    for (key,value) in mutations {
        let mut json=serde_json::to_value(&valid).unwrap(); json[key]=value;
        let bytes=serde_json::to_vec(&json).unwrap();
        std::fs::write(jobs.job_dir("j1").join("job.json"),&bytes).unwrap();
        assert!(read_record(&jobs.job_dir("j1")).unwrap_err().contains("semantic"),"{key}");
        assert!(matches!(jobs.store.list(&jobs.fixture.workspace).unwrap()[0],JobEntry::Unreadable{..}));
        assert!(jobs.store.close_guard(&jobs.fixture.workspace).is_err());
        assert_eq!(std::fs::read(jobs.job_dir("j1").join("job.json")).unwrap(),bytes);
    }
}
