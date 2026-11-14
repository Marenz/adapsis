use std::process::Command;
use std::time::{Duration, Instant};

use super::error::Error;
use super::git::{base_command, run, run_command, run_ok};
use super::limits::GIT_OUTPUT_MAX_BYTES;

fn shell(script: &str) -> Command {
    let mut command = Command::new("/bin/sh");
    command.args(["-c", script]);
    command
}

#[test]
fn git_runner_times_out_instead_of_hanging() {
    // The bounds are far apart on purpose: on a loaded machine the 200 ms
    // timeout may be noticed late, but never as late as the command's own end.
    let started = Instant::now();
    let error = run_command(shell("sleep 600"), "sleep", Duration::from_millis(200)).unwrap_err();
    assert!(matches!(error, Error::GitTimeout { ref action, millis: 200 } if action == "sleep"), "{error:?}");
    assert!(started.elapsed() < Duration::from_secs(120), "took {:?}", started.elapsed());
    assert!(error.to_string().contains("did not finish within 200 ms"), "{error}");
}

#[test]
fn git_runner_timeout_kills_the_whole_process_group() {
    let temp = tempfile::tempdir().unwrap();
    let pid_file = temp.path().join("pid");
    let script = format!("sh -c 'echo $$ > {}; exec sleep 600' & wait", pid_file.display());
    let error = run_command(shell(&script), "group", Duration::from_millis(1500)).unwrap_err();
    assert!(matches!(error, Error::GitTimeout { .. }), "{error:?}");
    let pid: String = std::fs::read_to_string(&pid_file).unwrap().trim().to_string();
    assert!(pid.parse::<u32>().is_ok(), "the grandchild never started: {pid:?}");
    let gone = (0..6000).any(|_| {
        let alive = std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|stat| !stat.contains(") Z "));
        if alive {
            std::thread::sleep(Duration::from_millis(10));
        }
        !alive
    });
    assert!(gone, "grandchild {pid} survived the timeout");
}

#[test]
fn git_runner_does_not_wait_forever_for_a_background_process_holding_the_pipes() {
    let started = Instant::now();
    let output = run_command(shell("sleep 600 & echo $!"), "background", Duration::from_secs(300)).unwrap();
    assert!(output.success());
    assert!(started.elapsed() < Duration::from_secs(120), "took {:?}", started.elapsed());
    // The background process is left alone by the runner; the test cleans up.
    let pid: i32 = output.stdout_text().trim().parse().expect("the script prints the background pid");
    // SAFETY: kill(2) touches no memory; the pid is the process this test started.
    unsafe { libc::kill(pid, libc::SIGKILL) };
}

#[test]
fn git_runner_caps_captured_output() {
    let output = run_command(shell("head -c 3000000 /dev/zero; echo tail >&2"), "flood", Duration::from_secs(30)).unwrap();
    assert!(output.success());
    assert_eq!(output.stdout.len(), GIT_OUTPUT_MAX_BYTES);
    assert_eq!(output.stderr, b"tail\n");
}

#[test]
fn git_runner_reports_a_program_that_cannot_start() {
    let error = run_command(Command::new("/nonexistent/git"), "start", Duration::from_secs(5)).unwrap_err();
    assert!(matches!(error, Error::Git { .. }), "{error:?}");
    assert!(error.to_string().contains("could not be started"), "{error}");
}

#[test]
fn git_failure_is_an_error_with_gits_own_explanation() {
    let temp = tempfile::tempdir().unwrap();
    let output = run(temp.path(), &["rev-parse", "--git-dir"], "rev-parse", Duration::from_secs(30)).unwrap();
    assert!(!output.success());
    let error = run_ok(temp.path(), &["rev-parse", "--git-dir"], "rev-parse", Duration::from_secs(30)).unwrap_err();
    assert!(error.to_string().starts_with("git rev-parse failed:"), "{error}");
    assert!(error.to_string().to_lowercase().contains("not a git repository"), "{error}");
}

#[test]
fn git_runs_with_a_fixed_environment() {
    let temp = tempfile::tempdir().unwrap();
    // An alias starting with `!` runs a command with git's environment.
    let mut command = base_command(temp.path());
    command.args(["-c", "alias.show-env=!env", "show-env"]);
    let output = run_command(command, "env", Duration::from_secs(30)).unwrap();
    assert!(output.success(), "{}", output.failure_text());
    let environment = output.stdout_text();
    assert!(environment.lines().any(|line| line == "GIT_TERMINAL_PROMPT=0"), "{environment}");
    // git puts its own exec directory in front of the PATH it was given.
    assert!(environment.lines().any(|line| line.starts_with("PATH=") && line.ends_with("/usr/local/bin:/usr/bin:/bin")), "{environment}");
    assert!(environment.lines().any(|line| line == "HOME=/nonexistent"), "tests do not use the developer's HOME: {environment}");
    assert!(environment.lines().any(|line| line == "LC_ALL=C"), "{environment}");
    // Nothing of the test process's own environment comes through: every
    // variable git's child sees is one we set or one git itself adds.
    for line in environment.lines() {
        let name = line.split('=').next().unwrap_or_default();
        assert!(name.starts_with("GIT_") || ["HOME", "LC_ALL", "PATH", "PWD", "OLDPWD", "SHLVL", "_"].contains(&name), "`{name}` leaked: {environment}");
    }
    assert!(std::env::vars_os().count() > 5, "the test process has an environment that could have leaked");
}
