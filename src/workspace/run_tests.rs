use std::ffi::OsString;
use std::time::{Duration, Instant};

use crate::coroutine::shell_policy::ShellPolicy;

use super::error::{Error, Rule};
use super::limits::{
    RUN_COMMAND_MAX_BYTES, RUN_ENV_MAX_VARS, RUN_OUTPUT_DEFAULT_BYTES, RUN_OUTPUT_MAX_BYTES, RUN_OUTPUT_MIN_BYTES, RUN_TIMEOUT_DEFAULT,
    RUN_TIMEOUT_MAX, RUN_TIMEOUT_MIN,
};
use super::contain::Cancel;
use super::run::{Contained, Containment, Exit, LaunchContext, Launcher, RunOptions, RunRequest, RunResult, bash_argv, environment, run_with};
use super::systemd;
use super::test_support::{Fixture, symlink, write};

fn request(command: &str) -> RunRequest {
    RunRequest::new(command)
}

/// Run with an explicit policy, so the tests do not depend on the
/// developer's `ADAPSIS_SHELL_POLICY`. Containment is automatic: a scope
/// where this machine has a user manager, the process group elsewhere.
fn run(fixture: &Fixture, request: &RunRequest) -> RunResult {
    run_with(&fixture.workspace, request, &RunOptions::new(&ShellPolicy::Unrestricted)).unwrap()
}

fn fails(fixture: &Fixture, request: &RunRequest) -> Error {
    run_with(&fixture.workspace, request, &RunOptions::new(&ShellPolicy::Unrestricted)).unwrap_err()
}

fn run_contained(fixture: &Fixture, request: &RunRequest, containment: Containment) -> RunResult {
    let options = RunOptions { containment, ..RunOptions::new(&ShellPolicy::Unrestricted) };
    run_with(&fixture.workspace, request, &options).unwrap()
}

fn run_launched(fixture: &Fixture, request: &RunRequest, launcher: Launcher<'_>) -> Result<RunResult, Error> {
    let options = RunOptions { launcher, ..RunOptions::new(&ShellPolicy::Unrestricted) };
    run_with(&fixture.workspace, request, &options)
}

/// Both containments where both exist; the tests that are about stopping
/// things run once for each.
fn containments() -> Vec<Containment> {
    if systemd::available() {
        vec![Containment::Scope, Containment::ProcessGroup]
    } else {
        eprintln!("note: no systemd user manager here; scope containment is not tested");
        vec![Containment::ProcessGroup]
    }
}

fn sh(fixture: &Fixture, command: &str) -> RunResult {
    run(fixture, &request(command))
}

fn alive(pid: &str) -> bool {
    std::fs::read_to_string(format!("/proc/{}/stat", pid.trim())).is_ok_and(|stat| !stat.contains(") Z "))
}

/// Wait (generously: the machine may be loaded) until a process is gone.
fn gone(pid: &str) -> bool {
    (0..6000).any(|_| {
        let dead = !alive(pid);
        if !dead {
            std::thread::sleep(Duration::from_millis(10));
        }
        dead
    })
}

#[test]
fn run_returns_output_and_exit_status() {
    let fixture = Fixture::plain();
    let ok = sh(&fixture, "echo hello");
    assert_eq!((ok.exit, ok.output.as_str(), ok.total_bytes, ok.dropped_bytes, ok.leftover_killed), (Exit::Code(0), "hello\n", 6, 0, 0));
    assert!(ok.success());
    assert!(ok.render().starts_with("hello\n[exit status 0 after "), "{}", ok.render());
    assert!(ok.render().ends_with(" s]"), "{}", ok.render());

    let failed = sh(&fixture, "echo before; exit 3");
    assert_eq!((failed.exit, failed.output.as_str(), failed.success()), (Exit::Code(3), "before\n", false));
    assert!(failed.render().starts_with("before\n[exit status 3 after "), "{}", failed.render());

    let silent = sh(&fixture, "true");
    assert!(silent.render().starts_with("(no output)\n[exit status 0 after "), "{}", silent.render());

    let missing = sh(&fixture, "definitely-not-a-command-xyz");
    assert_eq!(missing.exit, Exit::Code(127));
    assert!(missing.output.contains("command not found"), "{}", missing.output);

    let killed = sh(&fixture, "echo dying; kill -KILL $$");
    assert_eq!((killed.exit, killed.output.as_str()), (Exit::Signal(libc::SIGKILL), "dying\n"));
    assert!(killed.render().contains("[ended by signal 9 (SIGKILL) after "), "{}", killed.render());
}

#[test]
fn run_interleaves_stdout_and_stderr_in_order() {
    let fixture = Fixture::plain();
    let result = sh(&fixture, "echo one; echo two >&2; echo three; echo four >&2");
    assert_eq!(result.output, "one\ntwo\nthree\nfour\n");
    // stdin is empty and there is no terminal.
    let stdin = sh(&fixture, "cat; echo \"read $(wc -c < /dev/stdin) bytes\"; if [ -t 0 ] || [ -t 1 ] || [ -t 2 ]; then echo tty; else echo no tty; fi");
    assert_eq!(stdin.output, "read 0 bytes\nno tty\n");
}

#[test]
fn run_working_directory_goes_through_the_resolver() {
    let fixture = Fixture::plain();
    fixture.write("src/deep/file.txt", "x\n");
    write(&fixture.outside, "secret.txt", "SECRET\n");
    symlink("../outside", fixture.root.join("out"));
    symlink("src/deep", fixture.root.join("shortcut"));
    let root = fixture.root.to_str().unwrap();

    let default = sh(&fixture, "pwd");
    assert_eq!((default.output.trim(), default.cwd.as_str()), (root, "."));
    let mut nested = request("pwd; ls");
    nested.cwd = "src/deep".to_string();
    let result = run(&fixture, &nested);
    assert_eq!((result.output.as_str(), result.cwd.as_str()), (format!("{root}/src/deep\nfile.txt\n").as_str(), "src/deep"));
    for (cwd, shown) in [("", "."), ("shortcut", "src/deep"), (root, "."), ("src/deep/..", "src")] {
        let mut inside = request("true");
        inside.cwd = cwd.to_string();
        assert_eq!(run(&fixture, &inside).cwd, shown, "{cwd:?}");
    }

    let attempt = |cwd: &str| {
        let mut bad = request("touch ran-anyway");
        bad.cwd = cwd.to_string();
        fails(&fixture, &bad)
    };
    assert_eq!(attempt("..").rule(), Some(Rule::ParentEscape));
    assert_eq!(attempt("out").rule(), Some(Rule::SymlinkOutside));
    assert_eq!(attempt("/tmp").rule(), Some(Rule::OutsideWorkspace));
    assert_eq!(attempt(fixture.outside.to_str().unwrap()).rule(), Some(Rule::OutsideWorkspace));
    assert!(matches!(attempt("absent"), Error::NotFound { .. }));
    assert_eq!(attempt("src/deep/file.txt").to_string(), "`src/deep/file.txt` is not a directory to run in");
    let extra = attempt(fixture.extra.to_str().unwrap()).to_string();
    assert!(extra.contains("is in a read-only root; commands run inside the workspace"), "{extra}");
    for directory in [&fixture.root, &fixture.outside, &fixture.extra, &fixture.top] {
        assert!(!directory.join("ran-anyway").exists(), "a refused command ran in {}", directory.display());
    }
}

#[test]
fn run_environment_is_fixed_plus_explicit_additions() {
    let fixture = Fixture::plain();
    let mut with_env = request("env | sort");
    with_env.env = vec![("CARGO_TARGET_DIR".to_string(), "/tmp/target dir".to_string()), ("EMPTY".to_string(), String::new())];
    let result = run(&fixture, &with_env);
    let names: Vec<&str> = result.output.lines().filter_map(|line| line.split('=').next()).collect();
    // bash adds PWD, SHLVL, OLDPWD and `_` of its own.
    let ours = ["CARGO_TARGET_DIR", "EMPTY", "GIT_TERMINAL_PROMPT", "HOME", "LANG", "PATH", "TERM"];
    for name in &names {
        assert!(ours.contains(name) || ["PWD", "SHLVL", "OLDPWD", "_"].contains(name), "`{name}` leaked into the command: {}", result.output);
    }
    for name in ours {
        assert!(names.contains(&name), "`{name}` is missing: {}", result.output);
    }
    assert!(result.output.contains("CARGO_TARGET_DIR=/tmp/target dir\n"), "{}", result.output);
    assert!(result.output.contains("TERM=dumb\n") && result.output.contains("LANG=C.UTF-8\n") && result.output.contains("GIT_TERMINAL_PROMPT=0\n"));
    assert!(std::env::vars_os().count() > 7, "the test process has more variables than the command got");
}

#[test]
fn run_environment_builder_validates_and_overrides() {
    let home = Some(OsString::from("/home/owner"));
    let fixed = environment(&[], &[], home.clone()).unwrap();
    let get = |environment: &[(OsString, OsString)], name: &str| {
        environment.iter().find(|(key, _)| key == name).map(|(_, value)| value.to_string_lossy().into_owned())
    };
    assert_eq!(get(&fixed, "PATH").as_deref(), Some("/home/owner/.cargo/bin:/home/owner/.local/bin:/usr/local/bin:/usr/bin:/bin"));
    assert_eq!(get(&fixed, "HOME").as_deref(), Some("/home/owner"));
    assert_eq!(fixed.len(), 5);
    // Without a usable HOME the command still gets a sane environment.
    for bad_home in [None, Some(OsString::from("")), Some(OsString::from("relative/home"))] {
        let fallback = environment(&[], &[], bad_home).unwrap();
        assert_eq!(get(&fallback, "HOME").as_deref(), Some("/"));
        assert!(get(&fallback, "PATH").unwrap().ends_with("/usr/local/bin:/usr/bin:/bin"));
    }
    let replacing = vec![("PATH".to_string(), "/opt/tool/bin:/usr/bin".to_string()), ("A_1".to_string(), "x=y".to_string())];
    let replaced = environment(&[], &replacing, home.clone()).unwrap();
    assert_eq!(get(&replaced, "PATH").as_deref(), Some("/opt/tool/bin:/usr/bin"), "an explicit addition replaces the fixed value");
    assert_eq!(replaced.iter().filter(|(key, _)| key == "PATH").count(), 1);
    assert_eq!(get(&replaced, "A_1").as_deref(), Some("x=y"));

    for bad_name in ["", "1ABC", "A-B", "A B", "A=B", "ÄPFEL", "A\0"] {
        let bad = vec![(bad_name.to_string(), "v".to_string())];
        let error = environment(&[], &bad, home.clone()).unwrap_err();
        assert!(error.to_string().contains("is not valid"), "{bad_name:?}: {error}");
    }
    let nul = vec![("OK".to_string(), "a\0b".to_string())];
    assert!(environment(&[], &nul, home.clone()).unwrap_err().to_string().contains("NUL byte"));
    let many: Vec<(String, String)> = (0..=RUN_ENV_MAX_VARS).map(|index| (format!("V{index}"), String::new())).collect();
    assert!(environment(&[], &many, home.clone()).unwrap_err().to_string().contains("the limit is 64"));
    let large = vec![("BIG".to_string(), "x".repeat(40_000))];
    assert!(environment(&[], &large, home.clone()).unwrap_err().to_string().contains("the limit is 32768"));
    // The repository's table sits between the fixed variables and the request.
    let repository = vec![("CARGO_TARGET_DIR".to_string(), "/shared/target".to_string()), ("LANG".to_string(), "de_DE.UTF-8".to_string())];
    let request_env = vec![("CARGO_TARGET_DIR".to_string(), "/mine".to_string())];
    let layered = environment(&repository, &request_env, home.clone()).unwrap();
    assert_eq!(get(&layered, "LANG").as_deref(), Some("de_DE.UTF-8"), "the repository replaces a fixed value");
    assert_eq!(get(&layered, "CARGO_TARGET_DIR").as_deref(), Some("/mine"), "the request replaces the repository's value");
    assert_eq!(get(&environment(&repository, &[], home).unwrap(), "CARGO_TARGET_DIR").as_deref(), Some("/shared/target"));
}

#[test]
fn run_times_out_and_stops_the_whole_group() {
    for containment in containments() {
        let fixture = Fixture::plain();
        let mut slow = request("echo started; sleep 600 & echo $! > child.pid; echo $$ > shell.pid; wait");
        slow.timeout = Some(Duration::from_millis(1500));
        let started = Instant::now();
        let result = run_contained(&fixture, &slow, containment);
        assert!(started.elapsed() < Duration::from_secs(120), "took {:?}", started.elapsed());
        assert_eq!((result.exit, result.output.as_str(), result.timeout), (Exit::TimedOut, "started\n", Duration::from_millis(1500)));
        assert!(!result.success());
        let rendered = result.render();
        let timed_out = "started\n[timed out after 1.5 s; the command and everything it started were stopped. Set a longer timeout (up to 3600 s) if it needs one]";
        // The gap of the fallback is named exactly where it matters.
        match containment {
            Containment::ProcessGroup => {
                assert_eq!(result.containment, Contained::ProcessGroup);
                assert_eq!(rendered, format!("{timed_out}\n[note: no systemd user manager was reachable, so only the command's process group was stopped; a process that detached itself (setsid) may still be running]"));
            }
            _ => {
                assert_eq!(result.containment, Contained::Scope);
                assert_eq!(rendered, timed_out);
            }
        }
        for file in ["child.pid", "shell.pid"] {
            let pid = std::fs::read_to_string(fixture.root.join(file)).unwrap();
            assert!(gone(&pid), "{containment:?}: {file} {pid} survived the timeout");
        }
    }
}

#[test]
fn run_timeout_kills_a_command_that_ignores_sigterm() {
    let fixture = Fixture::plain();
    let mut stubborn = request("trap '' TERM; echo $$ > shell.pid; echo armed; while true; do sleep 600; done");
    stubborn.timeout = Some(Duration::from_millis(1000));
    let started = Instant::now();
    let result = run(&fixture, &stubborn);
    assert!(started.elapsed() < Duration::from_secs(120), "took {:?}", started.elapsed());
    assert_eq!((result.exit, result.output.as_str()), (Exit::TimedOut, "armed\n"));
    assert!(gone(&std::fs::read_to_string(fixture.root.join("shell.pid")).unwrap()));
}

#[test]
fn run_timeout_defaults_and_limits() {
    let fixture = Fixture::plain();
    assert_eq!(sh(&fixture, "true").timeout, RUN_TIMEOUT_DEFAULT);
    for (asked, effective) in [(Duration::ZERO, RUN_TIMEOUT_MIN), (Duration::from_secs(86_400), RUN_TIMEOUT_MAX), (Duration::from_secs(7), Duration::from_secs(7))] {
        let mut clamped = request("true");
        clamped.timeout = Some(asked);
        assert_eq!(run(&fixture, &clamped).timeout, effective);
    }
    assert_eq!((RUN_TIMEOUT_DEFAULT.as_secs(), RUN_TIMEOUT_MAX.as_secs()), (120, 3600));
}

#[test]
fn run_caps_output_keeping_head_and_tail() {
    let fixture = Fixture::plain();
    let result = sh(&fixture, "echo START; for i in $(seq 1 20000); do echo \"line $i of the output\"; done; echo END");
    let total = 6 + (1..=20000).map(|i| format!("line {i} of the output\n").len() as u64).sum::<u64>() + 4;
    assert_eq!(result.total_bytes, total);
    assert!(result.output.starts_with("START\nline 1 of the output\n"), "{}", &result.output[..60]);
    assert!(result.output.ends_with("line 20000 of the output\nEND\n"), "{}", &result.output[result.output.len() - 60..]);
    assert!(result.dropped_bytes > 0 && result.dropped_bytes < total);
    assert!(result.output.contains(&format!("\n[… {} bytes of output dropped here …]\n", result.dropped_bytes)));
    // What is shown plus what is dropped is everything; the shown part respects the limit.
    let marker = format!("[… {} bytes of output dropped here …]\n", result.dropped_bytes).len();
    let shown = result.output.len() - marker;
    assert!(shown <= RUN_OUTPUT_DEFAULT_BYTES + 1 && shown as u64 + result.dropped_bytes >= total, "{shown} shown, {} dropped", result.dropped_bytes);
    let rendered = result.render();
    assert!(rendered.contains(&format!("[note: the command wrote {total} bytes; {} from the middle are not shown.", result.dropped_bytes)), "{}", &rendered[rendered.len() - 300..]);

    // The limit is a request parameter with a floor and a ceiling.
    for (asked, limit) in [(Some(1), RUN_OUTPUT_MIN_BYTES), (Some(usize::MAX), RUN_OUTPUT_MAX_BYTES), (Some(5000), 5000)] {
        let mut limited = request("head -c 400000 /dev/zero | tr '\\0' 'x'");
        limited.max_output_bytes = asked;
        let result = run(&fixture, &limited);
        assert_eq!((result.total_bytes, result.dropped_bytes), (400_000, 400_000 - limit as u64), "{asked:?}");
    }
    // Under the limit nothing is dropped and nothing is marked.
    let small = sh(&fixture, "head -c 30000 /dev/zero | tr '\\0' 'y'");
    assert_eq!((small.output.len(), small.dropped_bytes), (30_000, 0));
    assert!(!small.render().contains("not shown"));
}

#[test]
fn run_output_cut_never_splits_a_character() {
    let fixture = Fixture::plain();
    // 3-byte characters, with an offset so the cuts fall inside characters.
    let mut wide = request("printf 'a'; for i in $(seq 1 3000); do printf '日本語'; done");
    wide.max_output_bytes = Some(2001);
    let result = run(&fixture, &wide);
    assert_eq!(result.total_bytes, 1 + 3000 * 9);
    assert!(!result.output.contains('\u{FFFD}'), "a character was cut in half");
    assert!(result.output.starts_with("a日本語") && result.output.ends_with("語"), "{}", result.output.len());
    // Both cuts fell inside a character, so a few more bytes than the plain difference are dropped.
    assert!(result.dropped_bytes > result.total_bytes - 2001 && result.dropped_bytes <= result.total_bytes - 2001 + 4, "{}", result.dropped_bytes);
    // Bytes that are no text at all are shown lossily, not as an error.
    let binary = sh(&fixture, "printf '\\377\\376ok\\n'");
    assert_eq!(binary.output, "\u{FFFD}\u{FFFD}ok\n");
}

#[test]
fn run_flood_until_timeout_stays_bounded() {
    let fixture = Fixture::plain();
    let mut flood = request("yes 'the same line again and again'");
    flood.timeout = Some(Duration::from_millis(700));
    let result = run(&fixture, &flood);
    assert_eq!(result.exit, Exit::TimedOut);
    assert!(result.total_bytes > 1_000_000, "{}", result.total_bytes);
    assert!(result.output.len() < RUN_OUTPUT_DEFAULT_BYTES + 200, "{}", result.output.len());
    assert!(result.dropped_bytes > 0 && result.dropped_bytes < result.total_bytes);
    assert!(result.output.starts_with("the same line again and again\n"));
}

#[test]
fn run_stops_background_processes_and_says_so() {
    let fixture = Fixture::plain();
    // One that keeps the output pipe, one that does not.
    let started = Instant::now();
    let result = sh(&fixture, "sleep 600 & echo $! > holding.pid; (sleep 600 > /dev/null 2>&1 &  echo $! > detached.pid; wait) > /dev/null 2>&1 & while [ ! -s detached.pid ]; do sleep 0.01; done; echo launched");
    assert!(started.elapsed() < Duration::from_secs(120), "took {:?}", started.elapsed());
    assert_eq!((result.exit, result.output.as_str()), (Exit::Code(0), "launched\n"));
    assert!(result.leftover_killed >= 2, "{}", result.leftover_killed);
    assert!(
        result.render().contains(&format!(
            "[note: {} process(es) the command left running in the background were stopped when it returned; start a background job for a command that has to keep running]",
            result.leftover_killed
        )),
        "{}",
        result.render()
    );
    for file in ["holding.pid", "detached.pid"] {
        let pid = std::fs::read_to_string(fixture.root.join(file)).unwrap();
        assert!(gone(&pid), "{file} {pid} outlived the command");
    }
    // A background job that finishes by itself is neither killed nor reported.
    let quick = sh(&fixture, "(sleep 0.05; echo late) & echo early");
    assert_eq!((quick.output.as_str(), quick.leftover_killed), ("early\nlate\n", 0));
    assert!(!quick.render().contains("background"));
}

#[test]
fn run_applies_the_shell_policy_before_starting_anything() {
    let fixture = Fixture::plain();
    let run_under = |policy: &ShellPolicy, command: &str| run_with(&fixture.workspace, &request(command), &RunOptions::new(policy));
    // The destructive-command guard holds under every policy.
    for policy in [ShellPolicy::Unrestricted, ShellPolicy::Allowlist(vec!["touch".to_string(), "rm".to_string()])] {
        for command in ["touch ran; rm -rf /", "touch ran; dd if=/dev/zero of=/dev/sda", "touch ran; :(){ :|:& };:"] {
            let error = run_under(&policy, command).unwrap_err();
            assert!(matches!(error, Error::CommandRefused(_)), "{command}: {error:?}");
            assert!(error.to_string().starts_with("command not run: refused as destructive/irreversible"), "{error}");
        }
    }
    assert!(!fixture.root.join("ran").exists(), "a refused command was started");
    // The guard is the shared one: the same commands, the same verdict.
    assert!(ShellPolicy::destructive_reason("rm -rf /").is_some() && ShellPolicy::destructive_reason("rm -rf ./build").is_none());
    assert_eq!(run_under(&ShellPolicy::Unrestricted, "mkdir build && rm -rf ./build && echo fine").unwrap().output, "fine\n");
    // An operator's deny or allowlist binds this tool as it binds shell_exec.
    let denied = run_under(&ShellPolicy::Denied, "echo hi").unwrap_err();
    assert_eq!(denied.to_string(), "command not run: shell execution is disabled by policy (ADAPSIS_SHELL_POLICY=deny)");
    let allow = ShellPolicy::Allowlist(vec!["echo".to_string()]);
    assert_eq!(run_under(&allow, "echo allowed").unwrap().output, "allowed\n");
    assert!(run_under(&allow, "ls").unwrap_err().to_string().contains("is not in the allowlist"));
}

#[test]
fn run_validates_the_command() {
    let fixture = Fixture::plain();
    for (command, expected) in [("", "the command is empty"), ("  \n\t ", "the command is empty"), ("echo a\0b", "NUL byte")] {
        let error = fails(&fixture, &request(command));
        assert!(matches!(error, Error::InvalidArgument(_)) && error.to_string().contains(expected), "{command:?}: {error}");
    }
    let long = format!("echo {}", "x".repeat(RUN_COMMAND_MAX_BYTES));
    assert!(fails(&fixture, &request(&long)).to_string().contains("the limit is 32768"));
    // A long but allowed command, with quotes and newlines, arrives intact.
    let script = format!("cat <<'EOF'\n{}\nit's \"quoted\" $HOME\nEOF", "y".repeat(20_000));
    let result = sh(&fixture, &script);
    assert_eq!(result.total_bytes, 20_001 + "it's \"quoted\" $HOME\n".len() as u64);
    assert!(result.output.ends_with("it's \"quoted\" $HOME\n"));
}

#[test]
fn run_launcher_is_a_seam() {
    let fixture = Fixture::plain();
    fixture.write("sub/marker.txt", "x\n");
    // A wrapper around the default argv, as a later slice would add one.
    let wrapped = |context: &LaunchContext<'_>| {
        let mut argv = vec![OsString::from("/usr/bin/env"), OsString::from("WRAPPED=yes"), OsString::from(format!("SEEN_CWD={}", context.cwd.display()))];
        argv.extend(bash_argv(context));
        argv
    };
    let mut in_sub = request("echo \"$WRAPPED $SEEN_CWD $(pwd)\"");
    in_sub.cwd = "sub".to_string();
    let result = run_launched(&fixture, &in_sub, &wrapped).unwrap();
    let sub = fixture.root.join("sub");
    assert_eq!(result.output, format!("yes {} {}\n", sub.display(), sub.display()));
    // The default launcher is bash from an absolute path with the command as one argument.
    let context = LaunchContext { command: "echo 'a b'; exit 4", cwd: &fixture.root, workspace: &fixture.workspace };
    let argv = bash_argv(&context);
    assert_eq!(argv.len(), 3);
    assert!(std::path::Path::new(&argv[0]).is_absolute() && argv[0].to_string_lossy().ends_with("/bash"), "{argv:?}");
    assert_eq!((argv[1].to_str(), argv[2].to_str()), (Some("-c"), Some("echo 'a b'; exit 4")));
    // A launcher must name its program absolutely (a bare name would be a PATH lookup and a fork).
    for bad in [vec![], vec![OsString::from("bash"), OsString::from("-c"), OsString::from("true")]] {
        let launcher = move |_: &LaunchContext<'_>| bad.clone();
        let error = run_launched(&fixture, &request("true"), &launcher).unwrap_err();
        assert!(error.to_string().contains("absolute program path"), "{error}");
    }
    let absent = |_: &LaunchContext<'_>| vec![OsString::from("/nonexistent/program")];
    // Without a scope the missing program is our own start failure; inside a
    // scope it is the scope's shell that cannot execute it.
    let options = RunOptions { launcher: &absent, containment: Containment::ProcessGroup, ..RunOptions::new(&ShellPolicy::Unrestricted) };
    let error = run_with(&fixture.workspace, &request("true"), &options).unwrap_err();
    assert_eq!(error.to_string(), "cannot start `/nonexistent/program`: entity not found");
    if systemd::available() {
        let in_scope = run_launched(&fixture, &request("true"), &absent).unwrap();
        assert!(matches!(in_scope.exit, Exit::Code(125 | 126 | 127)), "{:?}: {}", in_scope.exit, in_scope.output);
    }
}

#[test]
fn run_many_short_commands_leave_nothing_behind() {
    let fixture = Fixture::plain();
    let started = Instant::now();
    for index in 0..40 {
        let result = sh(&fixture, &format!("echo {index}"));
        assert_eq!((result.exit, result.output), (Exit::Code(0), format!("{index}\n")));
    }
    // 40 commands must not cost 40 grace periods.
    assert!(started.elapsed() < Duration::from_secs(60), "took {:?}", started.elapsed());
}

// --- containment, cancel, the repository environment --------------------------

fn kill_pid(pid: &str) {
    if let Ok(pid) = pid.trim().parse::<i32>() {
        // SAFETY: kill(2) touches no memory; the pid is a process this test started.
        unsafe { libc::kill(pid, libc::SIGKILL) };
    }
}

fn no_units_left(pattern: &str) -> bool {
    (0..200).any(|_| {
        let listed = systemd::systemctl(&["list-units", "--all", "--no-legend", pattern], "list").map(|output| output.stdout_text()).unwrap_or_default();
        if !listed.trim().is_empty() {
            std::thread::sleep(Duration::from_millis(25));
        }
        listed.trim().is_empty()
    })
}

#[test]
fn run_scope_contains_processes_that_leave_the_process_group() {
    if !systemd::available() {
        eprintln!("skipped: no systemd user manager");
        return;
    }
    let fixture = Fixture::plain();
    let prefix = format!("{}-contain-test", super::run::default_unit_prefix());
    let run_scoped = |request: &RunRequest| {
        let options = RunOptions { containment: Containment::Scope, unit_prefix: &prefix, ..RunOptions::new(&ShellPolicy::Unrestricted) };
        run_with(&fixture.workspace, request, &options).unwrap()
    };
    // Timeout: a detached child is stopped with the rest.
    let mut detached = request("setsid sleep 600 & echo $! > detached.pid; echo started; sleep 600");
    detached.timeout = Some(Duration::from_millis(1200));
    let result = run_scoped(&detached);
    assert_eq!((result.exit, result.output.as_str(), result.containment), (Exit::TimedOut, "started\n", Contained::Scope));
    assert!(gone(&std::fs::read_to_string(fixture.root.join("detached.pid")).unwrap()), "setsid escaped the scope");
    // Normal return: a detached leftover is found, stopped and counted.
    let result = run_scoped(&request("setsid sleep 600 > /dev/null 2>&1 & echo $! > leftover.pid; echo done"));
    assert_eq!((result.exit, result.output.as_str(), result.leftover_killed), (Exit::Code(0), "done\n", 1));
    assert!(gone(&std::fs::read_to_string(fixture.root.join("leftover.pid")).unwrap()));
    assert!(!result.render().contains("no systemd user manager"), "{}", result.render());
    // Auto means a scope here.
    assert_eq!(run(&fixture, &request("true")).containment, Contained::Scope);
    // Scopes are collected: no unit of this test process stays behind, failed or not.
    let pattern = format!("{prefix}-run-*");
    assert!(no_units_left(&pattern), "units left behind: {pattern}");
}

#[test]
fn run_process_group_fallback_has_the_stated_gap() {
    let fixture = Fixture::plain();
    let mut detached = request("setsid sleep 600 & echo $! > detached.pid; echo started; sleep 600");
    detached.timeout = Some(Duration::from_millis(800));
    let result = run_contained(&fixture, &detached, Containment::ProcessGroup);
    assert_eq!((result.exit, result.containment), (Exit::TimedOut, Contained::ProcessGroup));
    assert!(result.render().contains("a process that detached itself (setsid) may still be running"), "{}", result.render());
    // That note is true: this is exactly the process that got away.
    let pid = std::fs::read_to_string(fixture.root.join("detached.pid")).unwrap();
    assert!(alive(&pid), "the gap this fallback documents was not there");
    kill_pid(&pid);
    assert!(gone(&pid));
    // A command that needed no stopping gets no such note.
    let quiet = run_contained(&fixture, &request("echo fine"), Containment::ProcessGroup);
    assert_eq!(quiet.containment, Contained::ProcessGroup);
    assert!(!quiet.render().contains("note"), "{}", quiet.render());
}

#[test]
fn run_falls_back_when_the_scope_cannot_be_created_and_runs_the_command_once() {
    let fixture = Fixture::plain();
    // A unit name longer than systemd allows stands in for "systemd-run failed".
    let too_long = "x".repeat(300);
    let options = RunOptions { unit_prefix: &too_long, ..RunOptions::new(&ShellPolicy::Unrestricted) };
    let result = run_with(&fixture.workspace, &request("echo ran >> count.txt; echo hello"), &options).unwrap();
    assert_eq!((result.exit, result.output.as_str(), result.containment), (Exit::Code(0), "hello\n", Contained::ProcessGroup));
    assert_eq!(std::fs::read_to_string(fixture.root.join("count.txt")).unwrap(), "ran\n", "the command ran exactly once");
    if systemd::available() {
        // Asked for a scope explicitly, the same failure is an error and nothing runs.
        let strict = RunOptions { containment: Containment::Scope, ..options };
        let error = run_with(&fixture.workspace, &request("echo ran >> count.txt"), &strict).unwrap_err();
        assert!(matches!(error, Error::Systemd { .. }) && error.to_string().contains("the command was not run"), "{error}");
        assert_eq!(std::fs::read_to_string(fixture.root.join("count.txt")).unwrap(), "ran\n");
        // A prefix systemd would rewrite is refused before anything starts.
        let rewritten = RunOptions { unit_prefix: "not a/valid unit", ..RunOptions::new(&ShellPolicy::Unrestricted) };
        let error = run_with(&fixture.workspace, &request("true"), &rewritten).unwrap_err();
        assert!(error.to_string().contains("unit prefix"), "{error}");
    }
}

#[test]
fn run_passes_the_command_text_through_unchanged() {
    // systemd expands `$VAR` and `%` specifiers in some command lines; none of that may reach a command.
    for containment in containments() {
        let fixture = Fixture::plain();
        let mut literal = request("X=inner; echo \"$$ ${X} $X %h %% $0 ${UNSET:-default}\" '$HOME ${HOME} %n'");
        literal.env = vec![("UNRELATED".to_string(), "$PATH %u".to_string())];
        let result = run_contained(&fixture, &literal, containment);
        let line = result.output.trim_end();
        let pid = line.split(' ').next().unwrap();
        assert!(pid.parse::<u32>().is_ok(), "{containment:?}: `$$` did not reach bash: {line}");
        assert_eq!(&line[pid.len()..], " inner inner %h %% /bin/bash default $HOME ${HOME} %n", "{containment:?}");
        let value = run_contained(&fixture, &RunRequest { command: "printf '%s' \"$UNRELATED\"".to_string(), ..literal.clone() }, containment);
        assert_eq!(value.output, "$PATH %u", "{containment:?}: an environment value was expanded");
    }
}

#[test]
fn run_environment_is_exact_in_every_containment() {
    for containment in containments() {
        let fixture = Fixture::plain();
        let names = |output: &str| -> Vec<String> { output.lines().filter_map(|line| line.split('=').next()).map(str::to_string).collect() };
        let plain = run_contained(&fixture, &request("env"), containment);
        let mut seen = names(&plain.output);
        seen.sort();
        seen.retain(|name| !["PWD", "SHLVL", "OLDPWD", "_"].contains(&name.as_str()));
        assert_eq!(seen, vec!["GIT_TERMINAL_PROMPT", "HOME", "LANG", "PATH", "TERM"], "{containment:?}: {}", plain.output);
        // A variable the scope machinery itself uses can still be requested.
        let mut asked = request("echo \"$XDG_RUNTIME_DIR|$INVOCATION_ID\"");
        asked.env = vec![("XDG_RUNTIME_DIR".to_string(), "/custom/runtime".to_string())];
        assert_eq!(run_contained(&fixture, &asked, containment).output, "/custom/runtime|\n", "{containment:?}");
    }
}

#[test]
fn run_applies_the_repository_environment() {
    let mut fixture = Fixture::plain();
    fixture.workspace.env = vec![("CARGO_TARGET_DIR".to_string(), "/shared/target".to_string()), ("FROM_REGISTRY".to_string(), "yes".to_string())];
    let result = sh(&fixture, "echo \"$CARGO_TARGET_DIR $FROM_REGISTRY\"");
    assert_eq!(result.output, "/shared/target yes\n");
    let mut overriding = request("echo \"$CARGO_TARGET_DIR $FROM_REGISTRY\"");
    overriding.env = vec![("CARGO_TARGET_DIR".to_string(), "/mine".to_string())];
    assert_eq!(run(&fixture, &overriding).output, "/mine yes\n", "the request wins over the repository");
    // An invalid entry (a registry loaded by hand) is refused, not passed on.
    fixture.workspace.env = vec![("BAD NAME".to_string(), "x".to_string())];
    assert!(fails(&fixture, &request("true")).to_string().contains("is not valid"));
}

#[test]
fn run_can_be_cancelled_while_it_runs() {
    for containment in containments() {
        let fixture = Fixture::plain();
        let cancel = Cancel::new();
        let trigger = {
            let (cancel, marker) = (cancel.clone(), fixture.root.join("child.pid"));
            std::thread::spawn(move || {
                // Cancel once the command is demonstrably running.
                let give_up = Instant::now() + Duration::from_secs(120);
                while !marker.exists() && Instant::now() < give_up {
                    std::thread::sleep(Duration::from_millis(5));
                }
                std::thread::sleep(Duration::from_millis(100));
                cancel.cancel();
            })
        };
        let options = RunOptions { cancel: Some(&cancel), containment, ..RunOptions::new(&ShellPolicy::Unrestricted) };
        let started = Instant::now();
        let result = run_with(&fixture.workspace, &request("echo started; sleep 600 & echo $! > child.pid; wait; echo never"), &options).unwrap();
        trigger.join().unwrap();
        assert!(started.elapsed() < Duration::from_secs(120), "took {:?}", started.elapsed());
        assert_eq!((result.exit, result.output.as_str()), (Exit::Cancelled, "started\n"), "{containment:?}");
        assert!(!result.success());
        assert!(result.render().starts_with("started\n[cancelled after "), "{}", result.render());
        assert!(result.render().contains("the command and everything it started were stopped"));
        assert!(gone(&std::fs::read_to_string(fixture.root.join("child.pid")).unwrap()), "{containment:?}: a child survived the cancel");
        assert!(cancel.is_cancelled());

        // A handle that is already cancelled starts nothing at all.
        let options = RunOptions { cancel: Some(&cancel), containment, ..RunOptions::new(&ShellPolicy::Unrestricted) };
        let result = run_with(&fixture.workspace, &request("touch started-anyway"), &options).unwrap();
        assert_eq!((result.exit, result.total_bytes), (Exit::Cancelled, 0));
        assert!(!fixture.root.join("started-anyway").exists());
        // A fresh handle that is never used changes nothing.
        let unused = Cancel::new();
        let options = RunOptions { cancel: Some(&unused), containment, ..RunOptions::new(&ShellPolicy::Unrestricted) };
        assert_eq!(run_with(&fixture.workspace, &request("echo ok"), &options).unwrap().exit, Exit::Code(0));
    }
}
