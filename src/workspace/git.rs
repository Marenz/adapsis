//! Running git: an argument vector, a fixed environment, a timeout.
//!
//! Nothing here goes through a shell, and nothing inherits the service's
//! environment (which holds API keys and askpass helpers). A git that does not
//! finish in time is killed together with everything it started (hooks,
//! credential helpers), because it runs in its own process group.

use std::ffi::OsStr;
use std::io::Read;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::time::{Duration, Instant};

use super::error::{Error, Result};
use super::limits::{GIT_OUTPUT_MAX_BYTES, GIT_PIPE_GRACE};

/// The PATH git's own helpers and hooks are looked up in.
const GIT_PATH: &str = "/usr/local/bin:/usr/bin:/bin";

/// The git executable: the first of the `GIT_PATH` directories that has one.
///
/// It is resolved to an absolute path on purpose. With a bare program name and
/// a changed `PATH`, `std::process::Command` cannot use `posix_spawn` and
/// falls back to `fork`, which copies the page tables of this whole process
/// and stalls every other thread meanwhile. In the daemon (and in the test
/// binary) that process is large; a few hundred such forks cost seconds.
fn git_program() -> &'static Path {
    static PROGRAM: OnceLock<PathBuf> = OnceLock::new();
    PROGRAM.get_or_init(|| {
        GIT_PATH
            .split(':')
            .map(|directory| Path::new(directory).join("git"))
            .find(|candidate| candidate.is_file())
            .unwrap_or_else(|| PathBuf::from("/usr/bin/git"))
    })
}

#[derive(Debug)]
pub struct Output {
    pub status: ExitStatus,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

impl Output {
    pub fn success(&self) -> bool {
        self.status.success()
    }

    pub fn code(&self) -> Option<i32> {
        self.status.code()
    }

    pub fn stdout_text(&self) -> String {
        String::from_utf8_lossy(&self.stdout).trim_end().to_string()
    }

    /// git's own explanation, shortened: the first lines of stderr.
    pub fn failure_text(&self) -> String {
        let text = String::from_utf8_lossy(&self.stderr);
        let lines: Vec<String> =
            text.lines().filter(|line| !line.trim().is_empty()).take(6).map(|line| line.chars().take(300).collect()).collect();
        if lines.is_empty() {
            match self.status.code() {
                Some(code) => format!("exit status {code}"),
                None => "terminated by a signal".to_string(),
            }
        } else {
            lines.join("\n")
        }
    }
}

/// A `git` command with the fixed environment, before arguments are added.
///
/// `HOME` is kept so the owner's git configuration (identity, safe.directory)
/// applies; under `cfg(test)` global and system configuration are disabled
/// instead, so the tests do not depend on the developer's settings.
pub fn base_command(directory: &Path) -> Command {
    let mut command = Command::new(git_program());
    command.env_clear();
    command.env("PATH", GIT_PATH);
    command.env("LC_ALL", "C");
    command.env("GIT_TERMINAL_PROMPT", "0");
    command.env("GIT_OPTIONAL_LOCKS", "0");
    #[cfg(not(test))]
    if let Some(home) = std::env::var_os("HOME") {
        command.env("HOME", home);
    }
    #[cfg(test)]
    {
        command.env("GIT_CONFIG_GLOBAL", "/dev/null");
        command.env("GIT_CONFIG_NOSYSTEM", "1");
        command.env("HOME", "/nonexistent");
    }
    command.current_dir(directory);
    command
}

/// Run `git <args>` in `directory`. `Ok` means git ran and exited; inspect
/// `Output::success`. `action` names the operation in error messages.
pub fn run<S: AsRef<OsStr>>(directory: &Path, args: &[S], action: &str, timeout: Duration) -> Result<Output> {
    let mut command = base_command(directory);
    command.args(args);
    run_command(command, action, timeout)
}

/// Like [`run`], but a non-zero exit is an error carrying git's explanation.
pub fn run_ok<S: AsRef<OsStr>>(directory: &Path, args: &[S], action: &str, timeout: Duration) -> Result<Output> {
    let output = run(directory, args, action, timeout)?;
    if output.success() {
        Ok(output)
    } else {
        Err(Error::Git { action: action.to_string(), detail: output.failure_text() })
    }
}

/// Spawn `command` in its own process group and wait for it with a deadline.
pub fn run_command(mut command: Command, action: &str, timeout: Duration) -> Result<Output> {
    command.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    command.process_group(0);
    let mut child = command.spawn().map_err(|error| Error::Git {
        action: action.to_string(),
        detail: format!("could not be started ({}); is git installed in {GIT_PATH}, and does the directory exist?", error.kind()),
    })?;
    let stdout = capture(child.stdout.take());
    let stderr = capture(child.stderr.take());
    let deadline = Instant::now() + timeout;
    let status = match wait_until(&mut child, deadline) {
        Some(status) => status,
        None => {
            kill_group(&mut child);
            return Err(Error::GitTimeout { action: action.to_string(), millis: timeout.as_millis() });
        }
    };
    // git has exited. Something it started (a hook's background process) may
    // still hold the pipes; give the readers a short grace period and then go
    // on with what they have. That process is not killed: git has been
    // reaped, so its pid no longer safely names the group, and the owner
    // running the same command by hand would have left the same process behind.
    let stdout_closed = stdout.closed.recv_timeout(GIT_PIPE_GRACE).is_ok();
    let _ = stderr.closed.recv_timeout(if stdout_closed { GIT_PIPE_GRACE } else { Duration::ZERO });
    Ok(Output { status, stdout: stdout.take(), stderr: stderr.take() })
}

/// One captured output stream: what was read so far, and a signal for
/// "the pipe is closed".
struct Capture {
    kept: Arc<Mutex<Vec<u8>>>,
    closed: mpsc::Receiver<()>,
}

impl Capture {
    fn take(&self) -> Vec<u8> {
        std::mem::take(&mut *self.kept.lock().unwrap_or_else(|poisoned| poisoned.into_inner()))
    }
}

/// Read a pipe on its own thread, keeping at most the output cap.
fn capture<R: Read + Send + 'static>(pipe: Option<R>) -> Capture {
    let kept = Arc::new(Mutex::new(Vec::new()));
    let (sender, closed) = mpsc::channel();
    let shared = kept.clone();
    std::thread::spawn(move || {
        if let Some(mut pipe) = pipe {
            let mut chunk = [0u8; 16 * 1024];
            loop {
                match pipe.read(&mut chunk) {
                    Ok(0) => break,
                    Ok(read) => {
                        let mut kept = shared.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                        let room = GIT_OUTPUT_MAX_BYTES.saturating_sub(kept.len());
                        kept.extend_from_slice(&chunk[..read.min(room).min(chunk.len())]);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(_) => break,
                }
            }
        }
        let _ = sender.send(());
    });
    Capture { kept, closed }
}

fn wait_until(child: &mut Child, deadline: Instant) -> Option<ExitStatus> {
    let mut pause = Duration::from_millis(2);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status),
            Ok(None) => {}
            Err(_) => return None,
        }
        let now = Instant::now();
        if now >= deadline {
            return None;
        }
        std::thread::sleep(pause.min(deadline - now));
        pause = (pause * 2).min(Duration::from_millis(50));
    }
}

fn kill_group(child: &mut Child) {
    signal_group(child.id(), libc::SIGKILL);
    let _ = child.kill();
    let _ = child.wait();
}

/// Signal the process group led by `pid` (a child started with
/// `process_group(0)`, so its pid is the group id). Only safe while that
/// child has not been reaped: until then its pid cannot be reused.
pub(super) fn signal_group(pid: u32, signal: libc::c_int) {
    if let Ok(pid) = i32::try_from(pid) {
        if pid > 1 {
            // SAFETY: kill(2) with a negative pid signals a process group and
            // touches no memory. The group id is our own child's pid.
            unsafe {
                libc::kill(-pid, signal);
            }
        }
    }
}
