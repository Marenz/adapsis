//! The systemd user manager, as far as the workspace tools need it.
//!
//! Commands are contained in transient units of the *user* manager: a scope
//! for a foreground command, a service for a background job. Both give the
//! command a control group of its own, which is what a timeout or a kill
//! acts on, so a process that calls `setsid` cannot get away, and which is
//! not inside this process's own control group, so a job survives a restart
//! of Adapsis.
//!
//! Only two programs are used, by absolute path, with a cleared environment:
//! `systemd-run` and `systemctl`. There is no D-Bus library. Every unit this
//! module is asked about carries a caller-chosen name prefix; nothing here
//! ever lists or touches units by a broader pattern.
//!
//! A user manager is not always there (a system service without lingering,
//! a container). `availability()` answers that, cached for a short time;
//! the callers decide what to do without it.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;
use std::time::Instant;

use super::error::{Error, Result};
use super::git;
use super::limits::{SYSTEMD_PROBE_TTL, SYSTEMD_TIMEOUT};

pub const SYSTEMD_RUN: &str = "/usr/bin/systemd-run";
pub const SYSTEMCTL: &str = "/usr/bin/systemctl";

/// PATH of the client programs themselves.
const CLIENT_PATH: &str = "/usr/bin:/bin";

static PROBE: Mutex<Option<(Instant, std::result::Result<(), String>)>> = Mutex::new(None);

/// The user's runtime directory, where the manager's bus socket lives:
/// `XDG_RUNTIME_DIR` if this process has it, else `/run/user/<uid>` if that
/// exists (a system service running as a user with lingering enabled).
pub fn runtime_dir() -> Option<PathBuf> {
    let from_environment = std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from).filter(|path| path.is_absolute() && path.is_dir());
    from_environment.or_else(|| {
        // SAFETY: getuid cannot fail and touches no memory.
        let uid = unsafe { libc::getuid() };
        Some(PathBuf::from(format!("/run/user/{uid}"))).filter(|path| path.is_dir())
    })
}

/// A client command with the fixed environment, before arguments are added.
pub fn client(program: &str) -> Command {
    let mut command = Command::new(program);
    command.env_clear();
    command.env("PATH", CLIENT_PATH);
    command.env("LC_ALL", "C");
    if let Some(directory) = runtime_dir() {
        command.env("XDG_RUNTIME_DIR", directory);
    }
    command
}

/// Whether a systemd user manager can be reached, and if not, why. The
/// answer is reused for `SYSTEMD_PROBE_TTL`.
pub fn availability() -> std::result::Result<(), String> {
    let mut cached = PROBE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some((taken, answer)) = cached.as_ref() {
        if taken.elapsed() < SYSTEMD_PROBE_TTL {
            return answer.clone();
        }
    }
    let answer = probe();
    *cached = Some((Instant::now(), answer.clone()));
    answer
}

pub fn available() -> bool {
    availability().is_ok()
}

fn probe() -> std::result::Result<(), String> {
    let hint = "a systemd user manager is needed (for a system service: `loginctl enable-linger <user>` and XDG_RUNTIME_DIR=/run/user/<uid>)";
    let Some(directory) = runtime_dir() else {
        return Err(format!("no user runtime directory (XDG_RUNTIME_DIR or /run/user/<uid>); {hint}"));
    };
    if !directory.join("bus").exists() && !directory.join("systemd/private").exists() {
        return Err(format!("no user manager socket in {}; {hint}", directory.display()));
    }
    let mut command = client(SYSTEMCTL);
    command.args(["--user", "show-environment"]);
    match git::run_command(command, "probe", SYSTEMD_TIMEOUT) {
        Ok(output) if output.success() => Ok(()),
        Ok(output) => Err(format!("the user manager does not answer ({}); {hint}", output.failure_text())),
        Err(_) => Err(format!("{SYSTEMCTL} could not be run; {hint}")),
    }
}

fn systemd_error(action: &str, detail: String) -> Error {
    Error::Systemd { action: action.to_string(), detail }
}

/// Run a prepared client command; `Ok` means it ran and exited.
pub fn run_client(command: Command, action: &str) -> Result<git::Output> {
    git::run_command(command, action, SYSTEMD_TIMEOUT).map_err(|error| match error {
        Error::Git { detail, .. } => systemd_error(action, detail),
        Error::GitTimeout { millis, .. } => systemd_error(action, format!("no answer within {millis} ms")),
        other => other,
    })
}

/// `systemctl --user <args>`; a non-zero exit is an error.
pub fn systemctl(args: &[&str], action: &str) -> Result<git::Output> {
    let mut command = client(SYSTEMCTL);
    command.arg("--user").args(args);
    let output = run_client(command, action)?;
    if output.success() { Ok(output) } else { Err(systemd_error(action, output.failure_text())) }
}

/// Selected properties of a unit. A unit that does not exist is not an
/// error: systemd answers `LoadState=not-found`.
pub fn show(unit: &str, properties: &[&str]) -> Result<HashMap<String, String>> {
    let mut args = vec!["show", unit];
    for property in properties {
        args.push("-p");
        args.push(property);
    }
    let output = systemctl(&args, "show")?;
    Ok(output.stdout_text().lines().filter_map(|line| line.split_once('=')).map(|(key, value)| (key.to_string(), value.to_string())).collect())
}

/// What `show` says about whether a unit is still doing something.
pub fn is_active(properties: &HashMap<String, String>) -> bool {
    matches!(properties.get("ActiveState").map(String::as_str), Some("active" | "activating" | "deactivating" | "reloading"))
}

/// Send a signal to every process of a unit. Best effort: the unit may be
/// gone already.
pub fn kill(unit: &str, signal: &str) -> Result<()> {
    systemctl(&["kill", &format!("--signal={signal}"), unit], "kill").map(|_| ())
}

/// Forget a failed unit, so it does not stay loaded.
pub fn reset_failed(unit: &str) {
    let _ = systemctl(&["reset-failed", unit], "reset-failed");
}
