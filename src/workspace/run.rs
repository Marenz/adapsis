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
//! * **How long.** A wall-clock timeout, and a `Cancel` handle the caller
//!   may trigger; then SIGTERM, a grace period, SIGKILL. Either is a result,
//!   with the output so far.
//! * **How much.** stdout and stderr share one pipe, so their order is the
//!   order of writing. The head and the tail of the output are kept and the
//!   result says how much was dropped in between.
//! * **What is left.** When the command returns, whatever it left running is
//!   stopped and counted in the result. Use a background job (`jobs`) for a
//!   command that has to outlive the call.
//! * **Contained by what.** A transient systemd scope when a user manager is
//!   reachable (`contain`), so `setsid` does not escape any of the above;
//!   otherwise the command's process group, and the result says so.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::io::Read;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::time::{Duration, Instant};

use crate::coroutine::shell_policy::ShellPolicy;

use super::contain::{Cancel, Scope, supervise};
use super::error::{Error, Result, shown};
use super::lifecycle::Workspace;
use super::limits::{
    RUN_BACKSTOP_MARGIN, RUN_COMMAND_MAX_BYTES, RUN_ENV_MAX_BYTES, RUN_ENV_MAX_VARS, RUN_OUTPUT_DEFAULT_BYTES, RUN_OUTPUT_HEAD_PERCENT,
    RUN_OUTPUT_MAX_BYTES, RUN_OUTPUT_MIN_BYTES, RUN_TIMEOUT_DEFAULT, RUN_TIMEOUT_MAX, RUN_TIMEOUT_MIN,
};
use super::resolve::{Access, Area, Kind, Resolved, resolve};
use super::systemd;

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
        Self {
            command: command.to_string(),
            cwd: ".".to_string(),
            timeout: None,
            env: Vec::new(),
            max_output_bytes: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    /// The command ended, but unit cleanup could not be verified.
    CleanupUncertain,
    Code(i32),
    /// The command itself was ended by a signal it did not get from us.
    Signal(i32),
    /// The timeout passed; the command and what it started were stopped.
    TimedOut,
    /// The caller cancelled; the command and what it started were stopped.
    Cancelled,
}

/// How a command is to be contained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Containment {
    /// A systemd scope if a user manager is reachable, else the process group.
    Auto,
    /// A systemd scope; an error if no user manager is reachable.
    Scope,
    /// Only the process group, even where a scope would be possible.
    ProcessGroup,
}

/// How a command was contained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Contained {
    Scope,
    ProcessGroup,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunResult {
    pub cleanup_errors: Vec<String>,
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
    /// Processes the command left running and that were stopped when it
    /// returned.
    pub leftover_killed: usize,
    pub containment: Contained,
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
    vec![
        bash_program().as_os_str().to_os_string(),
        OsString::from("-c"),
        OsString::from(context.command),
    ]
}

pub(super) fn bash_program() -> &'static Path {
    static PROGRAM: OnceLock<PathBuf> = OnceLock::new();
    PROGRAM.get_or_init(|| {
        BASH_CANDIDATES
            .iter()
            .map(PathBuf::from)
            .find(|candidate| candidate.is_file())
            .unwrap_or_else(|| PathBuf::from(BASH_CANDIDATES[0]))
    })
}

/// Everything about a run that is not part of the request.
#[derive(Clone, Copy)]
pub struct RunOptions<'a> {
    #[cfg(test)]
    pub(super) adapter_source: Option<&'static str>,
    pub unit_control: &'a dyn super::contain::UnitControl,
    pub policy: &'a ShellPolicy,
    pub launcher: Launcher<'a>,
    /// Lets the caller stop the command while it runs.
    pub cancel: Option<&'a Cancel>,
    pub containment: Containment,
    /// Prefix of the scope's unit name (`<prefix>-run-<pid>-<n>.scope`).
    pub unit_prefix: &'a str,
}

impl<'a> RunOptions<'a> {
    /// The defaults: bash, automatic containment, no cancel handle.
    pub fn new(policy: &'a ShellPolicy) -> Self {
        Self {
            #[cfg(test)]
            adapter_source: None,
            unit_control: &super::contain::SYSTEMD_CONTROL,
            policy,
            launcher: &bash_argv,
            cancel: None,
            containment: Containment::Auto,
            unit_prefix: default_unit_prefix(),
        }
    }
}

/// Unit names start with `adapsis`; under `cfg(test)` with a prefix that
/// names the test process, so no test can touch a unit of a running Adapsis.
pub fn default_unit_prefix() -> &'static str {
    static PREFIX: OnceLock<String> = OnceLock::new();
    PREFIX.get_or_init(|| {
        if cfg!(test) {
            format!("adapsis-wstest-{}", std::process::id())
        } else {
            "adapsis".to_string()
        }
    })
}

/// Run a command under the process's shell policy (`ADAPSIS_SHELL_POLICY`).
pub fn run(workspace: &Workspace, request: &RunRequest) -> Result<RunResult> {
    run_with(workspace, request, &RunOptions::new(&ShellPolicy::from_env()))
}

pub fn run_with(workspace: &Workspace, request: &RunRequest, options: &RunOptions<'_>) -> Result<RunResult> {
    let prepared = prepare(
        workspace,
        &request.command,
        &request.cwd,
        &request.env,
        options.policy,
        options.launcher,
    )?;
    let timeout = request
        .timeout
        .unwrap_or(RUN_TIMEOUT_DEFAULT)
        .clamp(RUN_TIMEOUT_MIN, RUN_TIMEOUT_MAX);
    let limit = request
        .max_output_bytes
        .unwrap_or(RUN_OUTPUT_DEFAULT_BYTES)
        .clamp(RUN_OUTPUT_MIN_BYTES, RUN_OUTPUT_MAX_BYTES);
    let scope = match options.containment {
        Containment::ProcessGroup => false,
        Containment::Auto => systemd::available(),
        Containment::Scope => {
            systemd::availability().map_err(|reason| Error::Systemd {
                action: "scope".to_string(),
                detail: reason,
            })?;
            true
        }
    };
    if scope {
        // systemd would silently rewrite other characters, and the unit would
        // then not have the name we signal.
        let prefix = options.unit_prefix;
        if prefix.is_empty()
            || !prefix
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        {
            return Err(Error::InvalidArgument(
                "the unit prefix may only contain lowercase letters, digits and dashes".to_string(),
            ));
        }
        if !Path::new(super::adapter::PYTHON).is_file() {
            return Err(Error::InvalidArgument("workspace adapter requires /usr/bin/python3".into()));
        }
        let scope = new_scope(prefix)?;
        #[cfg(test)]
        let scope = Scope {
            adapter_source: options.adapter_source,
            ..scope
        };
        let attempt = execute(&prepared, timeout, limit, options.cancel, Some(scope), options.unit_control)?;
        // No marker: systemd-run could not set the scope up and never
        // executed the command, so running it now is the first time.
        if let Some(result) = attempt {
            return Ok(result);
        }
        if options.containment == Containment::Scope {
            return Err(Error::Systemd {
                action: "scope".to_string(),
                detail: "the scope could not be created; the command was not run".to_string(),
            });
        }
    }
    execute(&prepared, timeout, limit, options.cancel, None, options.unit_control)?
        .ok_or_else(|| Error::InvalidArgument("the command could not be started".to_string()))
}

/// A command that passed every check and is ready to be started.
pub(super) struct Prepared {
    /// The launcher's argument vector; `argv[0]` is an absolute path.
    pub argv: Vec<OsString>,
    pub environment: Vec<(OsString, OsString)>,
    pub cwd: Resolved,
}

/// The checks every command passes, foreground or job: the command text, the
/// shell policy, the environment, the working directory, the launcher.
pub(super) fn prepare(
    workspace: &Workspace,
    command: &str,
    cwd: &str,
    env: &[(String, String)],
    policy: &ShellPolicy,
    launcher: Launcher<'_>,
) -> Result<Prepared> {
    check_command(command)?;
    policy.check(command).map_err(Error::CommandRefused)?;
    let environment = environment(&workspace.env, env, std::env::var_os("HOME"))?;
    let cwd = resolve(workspace, if cwd.is_empty() { "." } else { cwd }, Access::Read)?;
    if cwd.kind != Kind::Directory {
        return Err(Error::WrongKind {
            path: cwd.supplied,
            expected: "a directory to run in",
        });
    }
    if cwd.area != Area::Root {
        return Err(Error::InvalidArgument(format!(
            "cwd `{}` is in a read-only root; commands run inside the workspace",
            cwd.supplied
        )));
    }
    let argv = launcher(&LaunchContext {
        command,
        cwd: &cwd.real,
        workspace,
    });
    if !argv.first().is_some_and(|program| Path::new(program).is_absolute()) {
        return Err(Error::InvalidArgument(
            "the launcher must return an argument vector that starts with an absolute program path".to_string(),
        ));
    }
    Ok(Prepared { argv, environment, cwd })
}

static SCOPE_COUNTER: AtomicU64 = AtomicU64::new(0);

fn new_scope(prefix: &str) -> Result<Scope> {
    let unit = format!(
        "{prefix}-run-{}-{}.scope",
        std::process::id(),
        SCOPE_COUNTER.fetch_add(1, Ordering::Relaxed)
    );
    Ok(Scope {
        #[cfg(test)]
        adapter_source: None,
        marker: handoff(b"")?,
        unit,
    })
}

pub(super) fn encoded_environment(environment: &[(OsString, OsString)]) -> Result<Vec<u8>> {
    let mut map = std::collections::BTreeMap::new();
    for (name, value) in environment {
        map.insert(
            name.to_str()
                .ok_or_else(|| Error::InvalidArgument("non-UTF-8 environment name".into()))?,
            value
                .to_str()
                .ok_or_else(|| Error::InvalidArgument("non-UTF-8 environment value".into()))?,
        );
    }
    let bytes = serde_json::to_vec(&map).map_err(|_| Error::InvalidArgument("environment encoding failed".into()))?;
    if bytes.len() > super::limits::RUN_ENV_ENCODED_MAX_BYTES {
        return Err(Error::InvalidArgument("encoded environment exceeds 512 KiB".into()));
    }
    Ok(bytes)
}

fn handoff(bytes: &[u8]) -> Result<std::fs::File> {
    use std::io::Write;
    use std::os::fd::FromRawFd;
    // SAFETY: static C string; CLOEXEC means no ambient descriptor inheritance.
    let fd = unsafe { libc::memfd_create(c"adapsis-handoff".as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(Error::io("adapter handoff", "create", &std::io::Error::last_os_error()));
    }
    // SAFETY: newly created descriptor, exclusively owned here.
    let mut file = unsafe { std::fs::File::from_raw_fd(fd) };
    use std::os::unix::fs::PermissionsExt;
    file.set_permissions(std::fs::Permissions::from_mode(0o600))
        .map_err(|e| Error::io("adapter handoff", "protect", &e))?;
    file.write_all(bytes).map_err(|e| Error::io("adapter handoff", "write", &e))?;
    Ok(file)
}

fn handoff_path(file: &std::fs::File) -> OsString {
    use std::os::fd::AsRawFd;
    format!("/proc/{}/fd/{}", std::process::id(), file.as_raw_fd()).into()
}

/// The program and arguments that start `prepared` inside `scope`:
/// `systemd-run --scope` registers the scope and executes the rest in it,
/// staying our child, so the exit status, the pipe and the process group are
/// what they would be without it. The static Python adapter receives only
/// explicit memfd paths, records establishment before payload startup, then
/// execves the payload with its decoded environment. Supervisor/client env is
/// never payload env; no env value is put into argv, even XDG_RUNTIME_DIR.
fn scope_argv(prepared: &Prepared, scope: &Scope, environment: &std::fs::File, timeout: Duration) -> Vec<OsString> {
    let backstop = (timeout + RUN_BACKSTOP_MARGIN).as_secs() + 1;
    let mut argv: Vec<OsString> = [
        systemd::SYSTEMD_RUN,
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "--expand-environment=no",
        &format!("--unit={}", scope.unit),
        "-p",
        "KillMode=control-group",
        "-p",
        &format!("RuntimeMaxSec={backstop}"),
        "--",
    ]
    .iter()
    .map(OsString::from)
    .collect();
    argv.extend([
        super::adapter::PYTHON.into(),
        "-I".into(),
        "-c".into(),
        scope.adapter_source().into(),
        handoff_path(environment),
        handoff_path(&scope.marker),
    ]);
    argv.extend(prepared.argv.iter().cloned());
    argv
}

/// Start the command and supervise it. `Ok(None)` means a scope was asked
/// for, could not be set up, and the command was never executed.
fn execute(
    prepared: &Prepared,
    timeout: Duration,
    limit: usize,
    cancel: Option<&Cancel>,
    scope: Option<Scope>,
    control: &dyn super::contain::UnitControl,
) -> Result<Option<RunResult>> {
    if cancel.is_some_and(Cancel::is_cancelled) {
        // Cancelled before it began: nothing is started.
        return Ok(Some(RunResult {
            cleanup_errors: Vec::new(),
            exit: Exit::Cancelled,
            output: String::new(),
            total_bytes: 0,
            dropped_bytes: 0,
            duration: Duration::ZERO,
            timeout,
            cwd: prepared.cwd.display.clone(),
            leftover_killed: 0,
            containment: if scope.is_some() {
                Contained::Scope
            } else {
                Contained::ProcessGroup
            },
        }));
    }
    let environment = if scope.is_some() {
        Some(handoff(&encoded_environment(&prepared.environment)?)?)
    } else {
        None
    };
    let argv = match (&scope, &environment) {
        (Some(scope), Some(env)) => scope_argv(prepared, scope, env, timeout),
        _ => prepared.argv.clone(),
    };
    let Some((program, arguments)) = argv.split_first() else {
        return Err(Error::InvalidArgument("the launcher returned an empty argument vector".to_string()));
    };
    let (reader, writer) = std::io::pipe().map_err(|error| Error::io("output pipe", "create", &error))?;
    let second = writer.try_clone().map_err(|error| Error::io("output pipe", "create", &error))?;
    let mut command = if scope.is_some() {
        systemd::client(systemd::SYSTEMD_RUN)
    } else {
        Command::new(program)
    };
    if scope.is_none() {
        command.env_clear().envs(prepared.environment.iter().cloned());
    }
    command.args(arguments).current_dir(&prepared.cwd.real);
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
    wait_handoff(scope.as_ref(), child.id(), started + timeout, cancel);
    let ended = supervise(child, &capture.closed, started + timeout, cancel, scope.as_ref(), control);
    // Cancel/deadline is not proof of handoff: a live adapter may still open
    // /proc/<parent>/fd/N. Retain ownership through stop/reap to prevent fd N
    // being reused for another request's environment (ABA).
    drop(environment);
    let (output, total_bytes, dropped_bytes) = capture.finish();
    if let Some(scope) = &scope {
        let established = scope.established();
        if !established {
            if !ended.cleanup_errors.is_empty() {
                return Err(Error::Job("scope setup/cleanup uncertain; command will not be replayed".into()));
            }
            return Ok(None);
        }
    }
    Ok(Some(RunResult {
        cleanup_errors: ended.cleanup_errors,
        exit: ended.exit,
        output,
        total_bytes,
        dropped_bytes,
        duration: started.elapsed(),
        timeout,
        cwd: prepared.cwd.display.clone(),
        leftover_killed: ended.leftover_killed,
        containment: if scope.is_some() {
            Contained::Scope
        } else {
            Contained::ProcessGroup
        },
    }))
}

fn wait_handoff(scope: Option<&Scope>, pid: u32, deadline: Instant, cancel: Option<&Cancel>) {
    if let Some(scope) = scope {
        while !scope.handed_over()
            && !super::contain::has_exited(pid)
            && Instant::now() < deadline
            && !cancel.is_some_and(Cancel::is_cancelled)
        {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
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
/// repository's `env` table from the registry, then the request's additions;
/// a later one replaces an earlier one of the same name. `home` is this
/// process's HOME; nothing else of its environment is consulted.
pub fn environment(
    repository: &[(String, String)],
    request: &[(String, String)],
    home: Option<OsString>,
) -> Result<Vec<(OsString, OsString)>> {
    let home = home
        .filter(|home| Path::new(home).is_absolute())
        .unwrap_or_else(|| OsString::from("/"));
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
    if request.len() > RUN_ENV_MAX_VARS {
        return Err(Error::InvalidArgument(format!(
            "{} environment variables; the limit is {RUN_ENV_MAX_VARS}",
            request.len()
        )));
    }
    let size: usize = request.iter().map(|(name, value)| name.len() + value.len()).sum();
    if size > RUN_ENV_MAX_BYTES {
        return Err(Error::InvalidArgument(format!(
            "the environment additions are {size} bytes; the limit is {RUN_ENV_MAX_BYTES}"
        )));
    }
    for (name, value) in repository.iter().chain(request) {
        check_env_pair(name, value).map_err(Error::InvalidArgument)?;
        environment.retain(|(existing, _)| existing != name.as_str());
        environment.push((name.into(), value.into()));
    }
    Ok(environment)
}

/// Whether a name and value can be an environment variable. The message
/// names the variable; the registry and the runner share this rule.
pub fn check_env_pair(name: &str, value: &str) -> std::result::Result<(), String> {
    let mut characters = name.chars();
    let valid_start = characters.next().is_some_and(|first| first.is_ascii_alphabetic() || first == '_');
    if !valid_start || !characters.all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(format!(
            "environment variable name `{}` is not valid (letters, digits and `_`, not starting with a digit)",
            shown(name)
        ));
    }
    if value.contains('\0') {
        return Err(format!("the value of environment variable `{name}` contains a NUL byte"));
    }
    Ok(())
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
        Self {
            head: Vec::new(),
            tail: VecDeque::new(),
            head_limit,
            tail_limit: limit - head_limit,
            total: 0,
        }
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
                    Ok(read) => shared
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .push(&chunk[..read.min(chunk.len())]),
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
            Exit::CleanupUncertain => "[command cleanup could not be confirmed; processes may still be running; do not replay the command]".into(),
            Exit::Code(code) => format!("[exit status {code} after {seconds:.1} s]"),
            Exit::Signal(signal) => format!("[ended by signal {signal}{} after {seconds:.1} s]", signal_name(signal)),
            Exit::TimedOut => format!(
                "[timed out after {} s; the command and everything it started were stopped. Set a longer timeout (up to {} s) if it needs one]",
                self.timeout.as_secs_f64(),
                RUN_TIMEOUT_MAX.as_secs()
            ),
            Exit::Cancelled => format!("[cancelled after {seconds:.1} s; the command and everything it started were stopped]"),
        });
        if !self.cleanup_errors.is_empty() {
            out.push_str(&format!("\n[cleanup errors: {}]", self.cleanup_errors.join("; ")));
        }
        if self.dropped_bytes > 0 {
            out.push_str(&format!(
                "\n[note: the command wrote {} bytes; {} from the middle are not shown. Redirect the output to a file and read or search that]",
                self.total_bytes, self.dropped_bytes
            ));
        }
        if self.leftover_killed > 0 && self.cleanup_errors.is_empty() {
            out.push_str(&format!(
                "\n[note: {} process(es) the command left running in the background were stopped when it returned; start a background job for a command that has to keep running]",
                self.leftover_killed
            ));
        }
        // Without a scope, "stopped" has a gap; say so where it matters.
        let stopped = self.leftover_killed > 0 || matches!(self.exit, Exit::TimedOut | Exit::Cancelled);
        if stopped && self.containment == Contained::ProcessGroup {
            out.push_str(
                "\n[note: no systemd user manager was reachable, so only the command's process group was stopped; a process that detached itself (setsid) may still be running]",
            );
        }
        out
    }

    /// Bound the DISPLAYED UTF-8 text as well as captured bytes. Lossy decoding
    /// can turn one binary byte into three UTF-8 bytes. Keep output head/tail
    /// AND the complete terminal/cleanup diagnosis; never cut off the verdict.
    pub fn render_within(&self, max_bytes: usize) -> String {
        let rendered = self.render();
        if rendered.len() <= max_bytes { return rendered; }
        let output = self.output.trim_end_matches('\n');
        let suffix = &rendered[output.len()..];
        let note = "\n[display text shortened by runtime; output head and tail kept]\n";
        let budget = max_bytes.saturating_sub(suffix.len() + note.len());
        let mut head = budget / 2;
        while !output.is_char_boundary(head) { head -= 1; }
        let mut tail = output.len().saturating_sub(budget - head);
        while !output.is_char_boundary(tail) { tail += 1; }
        format!("{}{note}{}{suffix}",&output[..head],&output[tail..])
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
