//! `run`: one shell command in the workspace, bounded in time and output.
//!
//! This is the command runner behind the later `bash` tool. It gives the
//! command the owner's full authority (DESIGN N5: no execution isolation);
//! what it bounds is everything around the command:
//!
//! * **What is started.** One function builds the argument vector
//!   (`bash_argv`: `/bin/bash -c <command>`). It is a seam: `run_with` takes
//!   any launcher, so a later slice can wrap the command without touching the
//!   rest. The program is always an absolute path (see `git::git_program` for
//!   why a bare name would make every start a `fork` of this process).
//! * **Where.** The working directory goes through the resolver and must be
//!   a directory inside the workspace.
//! * **With what.** The environment is cleared; the command sees PATH, HOME,
//!   LANG, TERM=dumb, GIT_TERMINAL_PROMPT=0 and the request's own additions.
//!   Nothing of this process's environment (API keys, askpass helpers) is
//!   inherited. stdin is `/dev/null`.
//! * **Whether.** The shell policy is asked first (`ShellPolicy::check`,
//!   which contains the destructive-command guard). A refused command is
//!   never started.
//! * **How long.** A wall-clock timeout; then SIGTERM to the process group,
//!   a grace period, SIGKILL. A timeout is a result, with the output so far.
//! * **How much.** stdout and stderr share one pipe, so their order is the
//!   order of writing. The head and the tail of the output are kept and the
//!   result says how much was dropped in between.
//! * **What is left.** When the command returns, whatever it left running in
//!   its process group is stopped and counted in the result. Background jobs
//!   are a later slice; until then a server started with `&` does not outlive
//!   the call, and the model is told so.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::io::Read;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::time::{Duration, Instant};

use crate::coroutine::shell_policy::ShellPolicy;

use super::error::{Error, Result, shown};
use super::git::signal_group;
use super::lifecycle::Workspace;
use super::limits::{
    RUN_COMMAND_MAX_BYTES, RUN_DRAIN_GRACE, RUN_ENV_MAX_BYTES, RUN_ENV_MAX_VARS, RUN_KILL_GRACE, RUN_OUTPUT_DEFAULT_BYTES, RUN_OUTPUT_HEAD_PERCENT,
    RUN_OUTPUT_MAX_BYTES, RUN_OUTPUT_MIN_BYTES, RUN_PIPE_GRACE, RUN_TIMEOUT_DEFAULT, RUN_TIMEOUT_MAX, RUN_TIMEOUT_MIN,
};
use super::resolve::{Access, Area, Kind, resolve};

/// System directories of the command's PATH, after `$HOME/.cargo/bin` and
/// `$HOME/.local/bin`.
const SYSTEM_PATH: &str = "/usr/local/bin:/usr/bin:/bin";
const HOME_PATH: [&str; 2] = [".cargo/bin", ".local/bin"];
const BASH_CANDIDATES: [&str; 3] = ["/bin/bash", "/usr/bin/bash", "/usr/local/bin/bash"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunRequest {
    pub command: String,
    /// Directory to run in, relative to the workspace root; `.` is the root.
    pub cwd: String,
    /// `None` is `RUN_TIMEOUT_DEFAULT`; clamped to `RUN_TIMEOUT_MIN..=MAX`.
    pub timeout: Option<Duration>,
    /// Variables added to (or replacing) the fixed environment.
    pub env: Vec<(String, String)>,
    /// `None` is `RUN_OUTPUT_DEFAULT_BYTES`; clamped to `RUN_OUTPUT_MIN..=MAX`.
    pub max_output_bytes: Option<usize>,
}

impl RunRequest {
    pub fn new(command: &str) -> Self {
        Self { command: command.to_string(), cwd: ".".to_string(), timeout: None, env: Vec::new(), max_output_bytes: None }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    Code(i32),
    /// The command itself was ended by a signal it did not get from us.
    Signal(i32),
    /// The timeout passed; the process group was stopped.
    TimedOut,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunResult {
    pub exit: Exit,
    /// stdout and stderr in the order they were written, as text (invalid
    /// UTF-8 replaced). When output was dropped, a marker line stands in the
    /// middle.
    pub output: String,
    /// Bytes the command wrote, and how many of them are not in `output`.
    pub total_bytes: u64,
    pub dropped_bytes: u64,
    pub duration: Duration,
    /// The effective timeout.
    pub timeout: Duration,
    /// Workspace-relative directory the command ran in.
    pub cwd: String,
    /// Processes the command left running in its group and that were stopped
    /// when it returned.
    pub leftover_killed: usize,
}

/// What a launcher is told about the command it has to start.
#[derive(Debug, Clone, Copy)]
pub struct LaunchContext<'a> {
    pub command: &'a str,
    /// The real directory the command will run in.
    pub cwd: &'a Path,
    pub workspace: &'a Workspace,
}

/// Builds the argument vector for a command. `argv[0]` must be an absolute
/// path. The environment and working directory are applied by `run_with` to
/// the process this starts.
pub type Launcher<'a> = &'a (dyn Fn(&LaunchContext<'_>) -> Vec<OsString> + Sync);

/// The default launcher: `/bin/bash -c <command>`.
pub fn bash_argv(context: &LaunchContext<'_>) -> Vec<OsString> {
    vec![bash_program().as_os_str().to_os_string(), OsString::from("-c"), OsString::from(context.command)]
}

fn bash_program() -> &'static Path {
    static PROGRAM: OnceLock<PathBuf> = OnceLock::new();
    PROGRAM.get_or_init(|| {
        BASH_CANDIDATES.iter().map(PathBuf::from).find(|candidate| candidate.is_file()).unwrap_or_else(|| PathBuf::from(BASH_CANDIDATES[0]))
    })
}

/// Run a command under the process's shell policy (`ADAPSIS_SHELL_POLICY`).
pub fn run(workspace: &Workspace, request: &RunRequest) -> Result<RunResult> {
    run_with(workspace, request, &ShellPolicy::from_env(), &bash_argv)
}

pub fn run_with(workspace: &Workspace, request: &RunRequest, policy: &ShellPolicy, launcher: Launcher<'_>) -> Result<RunResult> {
    check_command(&request.command)?;
    policy.check(&request.command).map_err(Error::CommandRefused)?;
    let environment = environment(request, std::env::var_os("HOME"))?;
    let cwd = resolve(workspace, if request.cwd.is_empty() { "." } else { &request.cwd }, Access::Read)?;
    if cwd.kind != Kind::Directory {
        return Err(Error::WrongKind { path: cwd.supplied, expected: "a directory to run in" });
    }
    if cwd.area != Area::Root {
        return Err(Error::InvalidArgument(format!("cwd `{}` is in a read-only root; commands run inside the workspace", cwd.supplied)));
    }
    let argv = launcher(&LaunchContext { command: &request.command, cwd: &cwd.real, workspace });
    let Some((program, arguments)) = argv.split_first().filter(|(program, _)| Path::new(program).is_absolute()) else {
        return Err(Error::InvalidArgument("the launcher must return an argument vector that starts with an absolute program path".to_string()));
    };
    let timeout = request.timeout.unwrap_or(RUN_TIMEOUT_DEFAULT).clamp(RUN_TIMEOUT_MIN, RUN_TIMEOUT_MAX);
    let limit = request.max_output_bytes.unwrap_or(RUN_OUTPUT_DEFAULT_BYTES).clamp(RUN_OUTPUT_MIN_BYTES, RUN_OUTPUT_MAX_BYTES);

    let (reader, writer) = std::io::pipe().map_err(|error| Error::io("output pipe", "create", &error))?;
    let second = writer.try_clone().map_err(|error| Error::io("output pipe", "create", &error))?;
    let mut command = Command::new(program);
    command.args(arguments).env_clear().envs(environment).current_dir(&cwd.real);
    command.stdin(Stdio::null()).stdout(writer).stderr(second).process_group(0);
    let started = Instant::now();
    let spawned = command.spawn();
    // The command object holds our copies of the pipe's write end; they must
    // be closed or the reader never sees the end of the output.
    drop(command);
    let child = spawned.map_err(|error| Error::Io {
        path: shown(&program.to_string_lossy()),
        action: "start",
        reason: error.kind().to_string(),
    })?;
    let capture = Capture::start(reader, limit);
    let ended = supervise(child, &capture, started + timeout);
    let (output, total_bytes, dropped_bytes) = capture.finish();
    Ok(RunResult {
        exit: ended.exit,
        output,
        total_bytes,
        dropped_bytes,
        duration: started.elapsed(),
        timeout,
        cwd: cwd.display,
        leftover_killed: ended.leftover_killed,
    })
}

fn check_command(command: &str) -> Result<()> {
    if command.trim().is_empty() {
        return Err(Error::InvalidArgument("the command is empty".to_string()));
    }
    if command.contains('\0') {
        return Err(Error::InvalidArgument("the command contains a NUL byte".to_string()));
    }
    if command.len() > RUN_COMMAND_MAX_BYTES {
        return Err(Error::InvalidArgument(format!(
            "the command is {} bytes; the limit is {RUN_COMMAND_MAX_BYTES}. Put long scripts in a file and run that",
            command.len()
        )));
    }
    Ok(())
}

/// The complete environment of a command: the fixed variables, then the
/// request's additions (which may replace a fixed one). `home` is this
/// process's HOME; nothing else of its environment is consulted.
pub fn environment(request: &RunRequest, home: Option<OsString>) -> Result<Vec<(OsString, OsString)>> {
    let home = home.filter(|home| Path::new(home).is_absolute()).unwrap_or_else(|| OsString::from("/"));
    let mut path = OsString::new();
    for directory in HOME_PATH {
        path.push(Path::new(&home).join(directory));
        path.push(":");
    }
    path.push(SYSTEM_PATH);
    let mut environment: Vec<(OsString, OsString)> = vec![
        ("PATH".into(), path),
        ("HOME".into(), home),
        ("LANG".into(), "C.UTF-8".into()),
        ("TERM".into(), "dumb".into()),
        ("GIT_TERMINAL_PROMPT".into(), "0".into()),
    ];
    if request.env.len() > RUN_ENV_MAX_VARS {
        return Err(Error::InvalidArgument(format!("{} environment variables; the limit is {RUN_ENV_MAX_VARS}", request.env.len())));
    }
    let size: usize = request.env.iter().map(|(name, value)| name.len() + value.len()).sum();
    if size > RUN_ENV_MAX_BYTES {
        return Err(Error::InvalidArgument(format!("the environment additions are {size} bytes; the limit is {RUN_ENV_MAX_BYTES}")));
    }
    for (name, value) in &request.env {
        let mut characters = name.chars();
        let valid_start = characters.next().is_some_and(|first| first.is_ascii_alphabetic() || first == '_');
        if !valid_start || !characters.all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(Error::InvalidArgument(format!(
                "environment variable name `{}` is not valid (letters, digits and `_`, not starting with a digit)",
                shown(name)
            )));
        }
        if value.contains('\0') {
            return Err(Error::InvalidArgument(format!("the value of environment variable `{name}` contains a NUL byte")));
        }
        environment.retain(|(existing, _)| existing != name.as_str());
        environment.push((name.into(), value.into()));
    }
    Ok(environment)
}

struct Ended {
    exit: Exit,
    leftover_killed: usize,
}

/// Wait for the command, enforce the deadline, and leave nothing behind.
///
/// The exit is observed without reaping (`waitid` with `WNOWAIT`). While the
/// command is an unreaped zombie its pid, which is also the id of its process
/// group, cannot be given to another process, so signalling the group is
/// safe. Only after that is the command reaped.
fn supervise(mut child: std::process::Child, capture: &Capture, deadline: Instant) -> Ended {
    let pid = child.id();
    let mut timed_out = false;
    let mut leftover_killed = 0;
    let mut output_closed = false;
    if wait_exit(pid, deadline) {
        // The command is done. Give what it left behind a moment to finish
        // by itself, then stop it.
        if group_members(pid) > 0 {
            output_closed = capture.closed.recv_timeout(RUN_PIPE_GRACE).is_ok();
            leftover_killed = settled_members(pid);
        }
    } else {
        timed_out = true;
        signal_group(pid, libc::SIGTERM);
        wait_exit(pid, Instant::now() + RUN_KILL_GRACE);
    }
    if timed_out || leftover_killed > 0 {
        signal_group(pid, libc::SIGKILL);
    }
    if timed_out {
        // Belt and braces: the wait below must never block on a command that
        // somehow outlived the group signal.
        let _ = child.kill();
    }
    let status = child.wait();
    // Everything in the group is gone, so the pipe closes as soon as the
    // reader has drained it. Only a process that left the group and kept the
    // pipe can hold it open, hence the bound.
    if !output_closed {
        let _ = capture.closed.recv_timeout(RUN_DRAIN_GRACE);
    }
    let exit = match status {
        _ if timed_out => Exit::TimedOut,
        Ok(status) => match (status.code(), status.signal()) {
            (Some(code), _) => Exit::Code(code),
            (None, Some(signal)) => Exit::Signal(signal),
            (None, None) => Exit::Code(-1),
        },
        Err(_) => Exit::Code(-1),
    };
    Ended { exit, leftover_killed }
}

/// Poll until the process has exited (true) or the deadline passed (false).
/// The process is not reaped.
fn wait_exit(pid: u32, deadline: Instant) -> bool {
    let mut pause = Duration::from_micros(500);
    loop {
        if has_exited(pid) {
            return true;
        }
        let now = Instant::now();
        if now >= deadline {
            return false;
        }
        std::thread::sleep(pause.min(deadline - now));
        pause = (pause * 2).min(Duration::from_millis(20));
    }
}

fn has_exited(pid: u32) -> bool {
    // SAFETY: an all-zero siginfo_t is a valid value for waitid to fill in.
    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
    // SAFETY: `info` is a valid, writable siginfo_t. WNOWAIT leaves the child
    // waitable, WNOHANG returns at once.
    let result = unsafe { libc::waitid(libc::P_PID, pid as libc::id_t, &mut info, libc::WEXITED | libc::WNOHANG | libc::WNOWAIT) };
    // An error (no such child) means there is nothing left to wait for.
    // SAFETY: after a successful waitid the pid field is initialized; it is 0
    // when no child changed state.
    result != 0 || unsafe { info.si_pid() } != 0
}

/// The members that are still there after a short settling time: a process
/// that just closed the pipe by exiting may be visible for a moment longer.
fn settled_members(leader: u32) -> usize {
    let mut members = group_members(leader);
    for _ in 0..10 {
        if members == 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
        members = group_members(leader);
    }
    members
}

/// Live processes in the group led by `leader`, the leader itself excluded.
fn group_members(leader: u32) -> usize {
    let Ok(entries) = std::fs::read_dir("/proc") else { return 0 };
    let mut members = 0;
    for entry in entries.flatten() {
        let Some(pid) = entry.file_name().to_str().and_then(|name| name.parse::<u32>().ok()) else { continue };
        if pid == leader {
            continue;
        }
        let Ok(stat) = std::fs::read_to_string(entry.path().join("stat")) else { continue };
        // `pid (comm) state ppid pgrp ...`; comm may contain spaces and `)`.
        let Some(rest) = stat.rsplit_once(')').map(|(_, rest)| rest) else { continue };
        let mut fields = rest.split_whitespace();
        let state = fields.next();
        let group = fields.nth(1).and_then(|field| field.parse::<u32>().ok());
        if group == Some(leader) && state != Some("Z") {
            members += 1;
        }
    }
    members
}

/// The first and the last bytes of a stream, and how long it was.
struct HeadTail {
    head: Vec<u8>,
    tail: VecDeque<u8>,
    head_limit: usize,
    tail_limit: usize,
    total: u64,
}

impl HeadTail {
    fn new(limit: usize) -> Self {
        let head_limit = limit * RUN_OUTPUT_HEAD_PERCENT / 100;
        Self { head: Vec::new(), tail: VecDeque::new(), head_limit, tail_limit: limit - head_limit, total: 0 }
    }

    fn push(&mut self, chunk: &[u8]) {
        self.total += chunk.len() as u64;
        let to_head = chunk.len().min(self.head_limit - self.head.len().min(self.head_limit));
        self.head.extend_from_slice(&chunk[..to_head]);
        let rest = &chunk[to_head..];
        let rest = &rest[rest.len().saturating_sub(self.tail_limit)..];
        let overflow = (self.tail.len() + rest.len()).saturating_sub(self.tail_limit);
        self.tail.drain(..overflow.min(self.tail.len()));
        self.tail.extend(rest);
    }

    /// The kept output as text, the total length, and the dropped bytes.
    fn render(mut self) -> (String, u64, u64) {
        let tail = self.tail.make_contiguous();
        let kept = (self.head.len() + tail.len()) as u64;
        if self.total <= kept {
            self.head.extend_from_slice(tail);
            return (String::from_utf8_lossy(&self.head).into_owned(), self.total, 0);
        }
        // Do not show half a character on either side of the gap.
        let head_end = match std::str::from_utf8(&self.head) {
            Err(error) if error.error_len().is_none() => error.valid_up_to(),
            _ => self.head.len(),
        };
        let tail_start = tail.iter().take(3).take_while(|byte| (**byte & 0xC0) == 0x80).count();
        let dropped = self.total - head_end as u64 - (tail.len() - tail_start) as u64;
        let mut text = String::from_utf8_lossy(&self.head[..head_end]).into_owned();
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(&format!("[… {dropped} bytes of output dropped here …]\n"));
        text.push_str(&String::from_utf8_lossy(&tail[tail_start..]));
        (text, self.total, dropped)
    }
}

/// Reads the output pipe on its own thread into a bounded `HeadTail`.
struct Capture {
    kept: Arc<Mutex<HeadTail>>,
    closed: mpsc::Receiver<()>,
    limit: usize,
}

impl Capture {
    fn start(mut reader: impl Read + Send + 'static, limit: usize) -> Self {
        let kept = Arc::new(Mutex::new(HeadTail::new(limit)));
        let (sender, closed) = mpsc::channel();
        let shared = kept.clone();
        std::thread::spawn(move || {
            let mut chunk = vec![0u8; 64 * 1024];
            loop {
                match reader.read(&mut chunk) {
                    Ok(0) => break,
                    Ok(read) => shared.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(&chunk[..read.min(chunk.len())]),
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(_) => break,
                }
            }
            let _ = sender.send(());
        });
        Self { kept, closed, limit }
    }

    fn finish(self) -> (String, u64, u64) {
        let mut kept = self.kept.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        std::mem::replace(&mut *kept, HeadTail::new(self.limit)).render()
    }
}

impl RunResult {
    pub fn success(&self) -> bool {
        self.exit == Exit::Code(0)
    }

    /// The text the model sees: the output, then how the command ended.
    pub fn render(&self) -> String {
        let mut out = self.output.trim_end_matches('\n').to_string();
        if out.is_empty() {
            out.push_str("(no output)");
        }
        let seconds = self.duration.as_secs_f64();
        out.push('\n');
        out.push_str(&match self.exit {
            Exit::Code(code) => format!("[exit status {code} after {seconds:.1} s]"),
            Exit::Signal(signal) => format!("[ended by signal {signal}{} after {seconds:.1} s]", signal_name(signal)),
            Exit::TimedOut => format!(
                "[timed out after {} s; the command and everything it started were stopped. Set a longer timeout (up to {} s) if it needs one]",
                self.timeout.as_secs_f64(),
                RUN_TIMEOUT_MAX.as_secs()
            ),
        });
        if self.dropped_bytes > 0 {
            out.push_str(&format!(
                "\n[note: the command wrote {} bytes; {} from the middle are not shown. Redirect the output to a file and read or search that]",
                self.total_bytes, self.dropped_bytes
            ));
        }
        if self.leftover_killed > 0 {
            out.push_str(&format!(
                "\n[note: {} process(es) the command left running in the background were stopped when it returned; a command cannot leave a server running yet]",
                self.leftover_killed
            ));
        }
        out
    }
}

fn signal_name(signal: i32) -> &'static str {
    match signal {
        libc::SIGHUP => " (SIGHUP)",
        libc::SIGINT => " (SIGINT)",
        libc::SIGQUIT => " (SIGQUIT)",
        libc::SIGILL => " (SIGILL)",
        libc::SIGABRT => " (SIGABRT)",
        libc::SIGBUS => " (SIGBUS)",
        libc::SIGFPE => " (SIGFPE)",
        libc::SIGKILL => " (SIGKILL)",
        libc::SIGSEGV => " (SIGSEGV)",
        libc::SIGPIPE => " (SIGPIPE)",
        libc::SIGTERM => " (SIGTERM)",
        _ => "",
    }
}
