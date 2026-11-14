//! Supervising one foreground command: wait for it, enforce the deadline and
//! the caller's cancel, and leave nothing behind.
//!
//! Two containments, one procedure:
//!
//! * **Scope.** The command runs in a transient systemd scope of the user
//!   manager. Everything it starts stays in the scope's control group,
//!   `setsid` or not, so signalling the unit reaches all of it.
//! * **Process group.** Without a user manager the command only has its own
//!   process group. A process that leaves the group (`setsid`) escapes the
//!   signals. The result says which containment was in effect.
//!
//! In both cases the command's exit is observed without reaping it
//! (`waitid` with `WNOWAIT`): while it is an unreaped zombie its pid, which is
//! also its process-group id, cannot be reused, so signalling the group is
//! safe. It is reaped only after everything else has been stopped.

use std::path::PathBuf;
use std::process::Child;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use std::os::unix::process::ExitStatusExt;

use super::git::signal_group;
use super::limits::{RUN_DRAIN_GRACE, RUN_KILL_GRACE, RUN_PIPE_GRACE};
use super::run::Exit;
use super::systemd;

/// A handle the caller keeps to stop a running command. Cloning gives
/// another handle to the same command; cancelling is idempotent and may
/// happen before the command has started.
#[derive(Debug, Clone, Default)]
pub struct Cancel {
    flag: Arc<AtomicBool>,
}

impl Cancel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
}

/// The scope a command was started in.
pub(super) struct Scope {
    #[cfg(test)]
    pub adapter_source: Option<&'static str>,
    pub unit: String,
    /// Created from inside the scope right before the command is executed:
    /// its existence proves that the scope was set up and the command ran.
    pub marker: std::fs::File,
}

impl Scope {
    pub fn adapter_source(&self) -> &'static str {
        #[cfg(test)]
        if let Some(source) = self.adapter_source {
            return source;
        }
        super::adapter::EXEC
    }
    pub fn established(&self) -> bool {
        self.marker.metadata().is_ok_and(|m| m.len() > 0)
    }
    pub fn handed_over(&self) -> bool {
        self.marker.metadata().is_ok_and(|m| m.len() > 1)
    }
}

pub(super) struct Ended {
    pub exit: Exit,
    pub leftover_killed: usize,
    pub cleanup_errors: Vec<String>,
}

/// Injectable unit boundary; tests fail operations after payload establishment.
pub trait UnitControl: Sync {
    fn kill(&self, unit: &str, signal: &str) -> std::result::Result<(), String>;
    fn members(&self, pid: u32, unit: &str) -> std::result::Result<usize, String>;
}
pub struct SystemdControl;
impl UnitControl for SystemdControl {
    fn kill(&self, unit: &str, signal: &str) -> std::result::Result<(), String> {
        match systemd::kill(unit, signal) {
            Ok(()) => Ok(()),
            Err(_) => match systemd::show(unit, &["ActiveState"]) {
                Ok(properties) if !systemd::is_active(&properties) => Ok(()),
                _ => Err("unit signal failed and stopped state could not be verified".into()),
            },
        }
    }
    fn members(&self, pid: u32, unit: &str) -> std::result::Result<usize, String> {
        match scope_cgroup(pid, unit) {
            Some(Cgroup::Gone) => Ok(0),
            Some(Cgroup::At(directory)) => {
                let listing = match std::fs::read_to_string(directory.join("cgroup.procs")) {
                    Ok(listing) => listing,
                    Err(_) => return inactive(unit),
                };
                Ok(listing
                    .lines()
                    .filter_map(|s| s.parse::<u32>().ok())
                    .filter(|p| *p != pid && is_alive(*p))
                    .count())
            }
            None => inactive(unit),
        }
    }
}
pub static SYSTEMD_CONTROL: SystemdControl = SystemdControl;

fn inactive(unit: &str) -> std::result::Result<usize, String> {
    match systemd::show(unit, &["ActiveState"]) {
        Ok(properties) if !systemd::is_active(&properties) => Ok(0),
        _ => Err("scope identity/inspection unavailable; stopped state unverified".into()),
    }
}

enum Waited {
    Exited,
    Deadline,
    Cancelled,
}

/// Wait for the command and stop whatever has to be stopped.
pub(super) fn supervise(
    mut child: Child,
    output_closed: &Receiver<()>,
    deadline: Instant,
    cancel: Option<&Cancel>,
    scope: Option<&Scope>,
    control: &dyn UnitControl,
) -> Ended {
    let pid = child.id();
    let contained = Contained { pid, scope, control };
    let mut cleanup_errors = Vec::new();
    let mut leftover_killed = 0;
    let mut closed = false;
    let waited = wait_exit(pid, deadline, cancel);
    match waited {
        Waited::Exited => {
            // The command is done. Give what it left behind a moment to
            // finish by itself, then stop it.
            let members = contained.members().unwrap_or_else(|e| {
                cleanup_errors.push(e);
                1
            });
            if members > 0 {
                closed = output_closed.recv_timeout(RUN_PIPE_GRACE).is_ok();
                leftover_killed = contained.settled_members().unwrap_or_else(|e| {
                    cleanup_errors.push(e);
                    members
                });
            }
        }
        Waited::Deadline | Waited::Cancelled => {
            if let Err(e) = contained.signal(libc::SIGTERM, "SIGTERM") {
                cleanup_errors.push(e);
            }
            wait_exit(pid, Instant::now() + RUN_KILL_GRACE, None);
        }
    }
    let stopped = !matches!(waited, Waited::Exited);
    if stopped || leftover_killed > 0 {
        if let Err(e) = contained.signal(libc::SIGKILL, "SIGKILL") {
            cleanup_errors.push(e);
        }
    }
    if stopped {
        // Belt and braces: the wait below must never block on a command
        // that somehow outlived the signals.
        let _ = child.kill();
    }
    // Confirm scope cleanup before reaping the leader: /proc/<pid>/cgroup is
    // still attributable while it is an unreaped zombie.
    if scope.is_some() && (stopped || leftover_killed > 0) {
        let until = Instant::now() + RUN_KILL_GRACE;
        loop {
            match contained.members() {
                Ok(0) => break,
                Err(e) => {
                    cleanup_errors.push(e);
                    break;
                }
                _ if Instant::now() >= until => {
                    cleanup_errors.push("scope still has live processes".into());
                    break;
                }
                _ => std::thread::sleep(Duration::from_millis(10)),
            }
        }
    }
    let status = child.wait();
    // Everything is gone, so the pipe closes as soon as the reader has
    // drained it. Only a process that got away and kept the pipe can hold it
    // open, hence the bound.
    if !closed {
        let _ = output_closed.recv_timeout(RUN_DRAIN_GRACE);
    }
    let exit = match (waited, status) {
        (Waited::Deadline, _) => Exit::TimedOut,
        (Waited::Cancelled, _) => Exit::Cancelled,
        (Waited::Exited, Ok(status)) => match (status.code(), status.signal()) {
            (Some(code), _) => Exit::Code(code),
            (None, Some(signal)) => Exit::Signal(signal),
            (None, None) => Exit::Code(-1),
        },
        (Waited::Exited, Err(_)) => Exit::Code(-1),
    };
    Ended {
        exit: if cleanup_errors.is_empty() { exit } else { Exit::CleanupUncertain },
        leftover_killed,
        cleanup_errors,
    }
}

/// Poll until the process has exited, the deadline passed or the caller
/// cancelled. The process is not reaped.
fn wait_exit(pid: u32, deadline: Instant, cancel: Option<&Cancel>) -> Waited {
    let mut pause = Duration::from_micros(500);
    loop {
        if has_exited(pid) {
            return Waited::Exited;
        }
        if cancel.is_some_and(Cancel::is_cancelled) {
            return Waited::Cancelled;
        }
        let now = Instant::now();
        if now >= deadline {
            return Waited::Deadline;
        }
        std::thread::sleep(pause.min(deadline - now));
        pause = (pause * 2).min(Duration::from_millis(20));
    }
}

pub(super) fn has_exited(pid: u32) -> bool {
    // SAFETY: an all-zero siginfo_t is a valid value for waitid to fill in.
    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
    // SAFETY: `info` is a valid, writable siginfo_t. WNOWAIT leaves the child
    // waitable, WNOHANG returns at once.
    let result = unsafe {
        libc::waitid(
            libc::P_PID,
            pid as libc::id_t,
            &mut info,
            libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
        )
    };
    // An error (no such child) means there is nothing left to wait for.
    // SAFETY: after a successful waitid the pid field is initialized; it is 0
    // when no child changed state.
    result != 0 || unsafe { info.si_pid() } != 0
}

/// The command (`pid`, leader of its process group) and what contains it.
struct Contained<'a> {
    pid: u32,
    scope: Option<&'a Scope>,
    control: &'a dyn UnitControl,
}

impl Contained<'_> {
    /// Signal everything the command started that we can reach.
    fn signal(&self, signal: libc::c_int, name: &str) -> std::result::Result<(), String> {
        signal_group(self.pid, signal);
        if let Some(scope) = self.scope {
            return self.control.kill(&scope.unit, name);
        }
        Ok(())
    }

    /// Live processes the command left behind, the command itself excluded.
    fn members(&self) -> std::result::Result<usize, String> {
        match self.scope {
            Some(scope) if scope.established() => self.control.members(self.pid, &scope.unit),
            _ => Ok(group_members(self.pid)),
        }
    }

    /// The members that are still there after a short settling time: a
    /// process that just closed the pipe by exiting may be visible for a
    /// moment longer.
    fn settled_members(&self) -> std::result::Result<usize, String> {
        let mut members = self.members()?;
        for _ in 0..10 {
            if members == 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
            members = self.members()?;
        }
        Ok(members)
    }
}

enum Cgroup {
    At(PathBuf),
    /// The control group was removed: nothing is left in it.
    Gone,
}

/// The control group of a scope, read from the command's own `/proc` entry.
/// The command joined the scope before it was executed, and an unreaped
/// zombie keeps its entry, so this works after the command has exited.
fn scope_cgroup(pid: u32, unit: &str) -> Option<Cgroup> {
    let text = std::fs::read_to_string(format!("/proc/{pid}/cgroup")).ok()?;
    let path = text.lines().find_map(|line| line.strip_prefix("0::"))?;
    if let Some(removed) = path.strip_suffix(" (deleted)") {
        return removed.ends_with(unit).then_some(Cgroup::Gone);
    }
    // Only trust a path that is this very scope; anything else (the scope was
    // not entered) falls back to the process group.
    path.ends_with(&format!("/{unit}"))
        .then(|| Cgroup::At(PathBuf::from(format!("/sys/fs/cgroup{path}"))))
}

fn is_alive(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .is_ok_and(|stat| !stat.rsplit_once(')').is_some_and(|(_, rest)| rest.trim_start().starts_with('Z')))
}

/// Live processes in the group led by `leader`, the leader itself excluded.
fn group_members(leader: u32) -> usize {
    let Ok(entries) = std::fs::read_dir("/proc") else { return 0 };
    let mut members = 0;
    for entry in entries.flatten() {
        let Some(pid) = entry.file_name().to_str().and_then(|name| name.parse::<u32>().ok()) else {
            continue;
        };
        if pid == leader {
            continue;
        }
        let Ok(stat) = std::fs::read_to_string(entry.path().join("stat")) else {
            continue;
        };
        // `pid (comm) state ppid pgrp ...`; comm may contain spaces and `)`.
        let Some(rest) = stat.rsplit_once(')').map(|(_, rest)| rest) else {
            continue;
        };
        let mut fields = rest.split_whitespace();
        let state = fields.next();
        let group = fields.nth(1).and_then(|field| field.parse::<u32>().ok());
        if group == Some(leader) && state != Some("Z") {
            members += 1;
        }
    }
    members
}
