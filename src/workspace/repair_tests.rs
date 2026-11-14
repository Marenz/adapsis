//! Discriminating regressions for the independent B1-B5/F1-F3 review.
use super::run::{Containment, Exit, RunOptions, RunRequest, run_with};
use super::test_support::Fixture;
use crate::coroutine::shell_policy::ShellPolicy;
use std::time::Duration;

#[test]
fn b1_scope_env_is_payload_only_and_bash_env_runs_once_without_replay() {
    for containment in [Containment::ProcessGroup, Containment::Scope] {
        if containment == Containment::Scope && !super::systemd::available() {
            continue;
        }
        let fixture = Fixture::plain();
        let startup = fixture.write("startup.sh", "echo startup >> startup-count\n");
        let mut req = RunRequest::new("echo payload >> payload-count; printf '%s|%s|%s' \"$keep\" \"$xdg\" \"$name\"");
        req.env = vec![
            ("keep".into(), "do-not-clear".into()),
            ("xdg".into(), "do-not-rewrite".into()),
            ("name".into(), "other".into()),
            ("XDG_RUNTIME_DIR".into(), "XDG_SENTINEL_9237".into()),
            ("BASH_ENV".into(), startup.to_str().unwrap().into()),
        ];
        let opts = RunOptions {
            containment,
            ..RunOptions::new(&ShellPolicy::Unrestricted)
        };
        let result = run_with(&fixture.workspace, &req, &opts).unwrap();
        assert!(result.success(), "{}", result.render());
        assert_eq!(result.output, "do-not-clear|do-not-rewrite|other");
        assert_eq!(std::fs::read_to_string(fixture.root.join("startup-count")).unwrap(), "startup\n");
        assert_eq!(std::fs::read_to_string(fixture.root.join("payload-count")).unwrap(), "payload\n");
        fixture.write("startup.sh", "echo effect >> effect-count\nexit 37\n");
        let result = run_with(&fixture.workspace, &req, &opts).unwrap();
        assert_eq!(result.exit, Exit::Code(37));
        assert_eq!(std::fs::read_to_string(fixture.root.join("effect-count")).unwrap(), "effect\n");
        assert_eq!(
            std::fs::read_to_string(fixture.root.join("payload-count")).unwrap(),
            "payload\n",
            "no replay/body after startup exit"
        );
        // Sentinel env values cannot appear in adapter/systemd arguments.
        let inspect = fixture.write("inspect.sh", "tr '\\0' '\\n' < /proc/$PPID/cmdline\n");
        let mut req = RunRequest::new("/bin/sh inspect.sh");
        req.env = vec![("XDG_RUNTIME_DIR".into(), "XDG_SENTINEL_9237".into())];
        let result = run_with(&fixture.workspace, &req, &opts).unwrap();
        assert!(!result.output.contains("XDG_SENTINEL_9237"));
        assert!(inspect.exists());
    }
}

#[test]
fn f1_expansion_is_checked_before_building_or_changing_ledger() {
    use super::edit::{Failure, edit, replace_bounded};
    assert_eq!(
        replace_bounded("xxxxxxxx", "x", &"z".repeat(1024), true, 100),
        Err(Failure::TooLarge)
    );
    let fixture = Fixture::plain();
    fixture.write("code.txt", "x".repeat(32768));
    let mut ledger = super::ledger::ReadLedger::default();
    ledger.note(&super::read::read(&fixture.workspace, "code.txt", None, None).unwrap());
    let saved = ledger.clone();
    let error = edit(&fixture.workspace, &mut ledger, "code.txt", "x", &"z".repeat(1024), true).unwrap_err();
    assert!(error.to_string().contains("result cap"));
    assert_eq!(ledger, saved);
    assert_eq!(std::fs::read(fixture.root.join("code.txt")).unwrap(), vec![b'x'; 32768]);
}

#[test]
fn f2_endless_no_newline_reader_stops_at_byte_ceiling() {
    use std::io::{Read, Result};
    struct Endless;
    impl Read for Endless {
        fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
            buf.fill(b'x');
            Ok(buf.len())
        }
    }
    let bounded = super::scan::Ceiling::new(Endless, 8192);
    let hit = bounded.hit.clone();
    let mut scanner = super::scan::LineScanner::new(bounded);
    let mut kept = Vec::new();
    assert_eq!(scanner.next_line(&mut kept, 32).unwrap().unwrap().len, 8192);
    assert_eq!(kept.len(), 32);
    assert!(scanner.next_line(&mut kept, 32).unwrap().is_none());
    assert!(hit.load(std::sync::atomic::Ordering::Relaxed));
}

#[test]
fn b5_cleanup_failures_are_not_success_or_stopped_claims() {
    use super::contain::{SYSTEMD_CONTROL, UnitControl};
    use std::sync::Mutex;
    struct Broken {
        unit: Mutex<Option<String>>,
        signal: bool,
        inspect: bool,
    }
    impl UnitControl for Broken {
        fn kill(&self, unit: &str, sig: &str) -> Result<(), String> {
            *self.unit.lock().unwrap() = Some(unit.into());
            if self.signal {
                Err("injected unit kill failure".into())
            } else {
                SYSTEMD_CONTROL.kill(unit, sig)
            }
        }
        fn members(&self, pid: u32, unit: &str) -> Result<usize, String> {
            *self.unit.lock().unwrap() = Some(unit.into());
            if self.inspect {
                Err("injected show/cgroup failure".into())
            } else {
                SYSTEMD_CONTROL.members(pid, unit)
            }
        }
    }
    if !super::systemd::available() {
        return;
    }
    for (signal, inspect) in [(true, false), (false, true), (true, true)] {
        let fixture = Fixture::plain();
        let broken = Broken {
            unit: Mutex::new(None),
            signal,
            inspect,
        };
        let options = RunOptions {
            unit_control: &broken,
            containment: Containment::Scope,
            ..RunOptions::new(&ShellPolicy::Unrestricted)
        };
        let mut req = RunRequest::new("echo once >> effects; setsid sleep 600 >/dev/null 2>&1 & echo done");
        req.timeout = Some(Duration::from_secs(3));
        let result = run_with(&fixture.workspace, &req, &options).unwrap();
        // Clean up through the real boundary even when the injected one failed.
        if let Some(unit) = broken.unit.lock().unwrap().as_ref() {
            let _ = super::systemd::kill(unit, "SIGKILL");
        }
        assert_eq!(result.exit, Exit::CleanupUncertain);
        assert!(!result.success());
        assert!(!result.cleanup_errors.is_empty());
        assert!(result.render().contains("do not replay"));
        assert!(!result.render().contains("were stopped"));
        assert_eq!(std::fs::read_to_string(fixture.root.join("effects")).unwrap(), "once\n");
    }
}

// Inert, static test adapter. Establishes the scope but pauses BEFORE opening
// the environment source, allowing teardown and fd-reuse attempts to be ordered
// deterministically. No payload command or uncontrolled environment is executed.
const PAUSED_ADAPTER: &str = r#"
import os, sys, time, signal, json
signal.signal(signal.SIGTERM, lambda *_: None)
root = sys.argv[3]
with open(sys.argv[2], 'wb') as marker: marker.write(b'1')
with open(root+'/adapter-count', 'a') as f: f.write('once\n')
with open(root+'/ready', 'w') as f: f.write(sys.argv[1])
while not os.path.exists(root+'/allow-read'): time.sleep(0.001)
try:
    with open(sys.argv[1], 'rb') as f: env = json.loads(f.read(524289))
    observed = env.get('RACE_VALUE','missing')
except Exception:
    observed = 'unavailable'
with open(root+'/consumed', 'w') as f: f.write(observed)
"#;

struct StopGate {
    entered: std::sync::mpsc::SyncSender<()>,
    release: std::sync::Mutex<std::sync::mpsc::Receiver<()>>,
}
impl super::contain::UnitControl for StopGate {
    fn kill(&self, unit: &str, signal: &str) -> Result<(), String> {
        if signal == "SIGTERM" {
            self.entered.send(()).map_err(|_| "test gate unavailable")?;
            self.release
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(30))
                .map_err(|_| "test gate timed out")?;
        }
        super::contain::UnitControl::kill(&super::contain::SYSTEMD_CONTROL, unit, signal)
    }
    fn members(&self, pid: u32, unit: &str) -> Result<usize, String> {
        super::contain::UnitControl::members(&super::contain::SYSTEMD_CONTROL, pid, unit)
    }
}

fn paused_runner(
    ws: super::Workspace,
    root: std::path::PathBuf,
    trigger: super::Cancel,
    control: std::sync::Arc<StopGate>,
    cancelled: bool,
) -> std::thread::JoinHandle<super::Result<super::RunResult>> {
    std::thread::spawn(move || {
        let launcher = |_: &super::run::LaunchContext<'_>| vec![root.as_os_str().to_os_string()];
        let options = RunOptions {
            adapter_source: Some(PAUSED_ADAPTER),
            cancel: Some(&trigger),
            unit_control: control.as_ref(),
            launcher: &launcher,
            ..RunOptions::new(&ShellPolicy::Unrestricted)
        };
        let mut request = RunRequest::new("inert lifetime test");
        request.env = vec![(
            "RACE_VALUE".into(),
            if cancelled { "ORIGINAL_CANCEL" } else { "ORIGINAL_DEADLINE" }.into(),
        )];
        request.timeout = Some(Duration::from_secs(if cancelled { 30 } else { 2 }));
        run_with(&ws, &request, &options)
    })
}

fn wait_test_text(path: &std::path::Path) -> String {
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        if let Ok(text) = std::fs::read_to_string(path) {
            if !text.is_empty() {
                return text;
            }
        }
        assert!(std::time::Instant::now() < deadline, "inert adapter did not reach test barrier");
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn other_request_memfds() -> Vec<std::fs::File> {
    use std::io::Write;
    use std::os::fd::FromRawFd;
    (0..64)
        .map(|_| {
            // SAFETY: static name; new CLOEXEC descriptor owned by this test.
            let fd = unsafe { libc::memfd_create(c"other-request".as_ptr(), libc::MFD_CLOEXEC) };
            assert!(fd >= 0);
            // SAFETY: newly created descriptor, exclusively owned here.
            let mut file = unsafe { std::fs::File::from_raw_fd(fd) };
            file.write_all(b"{\"RACE_VALUE\":\"OTHER_REQUEST\"}").unwrap();
            file
        })
        .collect()
}

fn handoff_lifetime_barrier(cancelled: bool) {
    use super::contain::Cancel;
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::MetadataExt;
    use std::sync::{Arc, Mutex, mpsc};
    if !super::systemd::available() {
        return;
    }
    let fixture = Fixture::plain();
    let cancel = Cancel::new();
    let (entered_tx, entered_rx) = mpsc::sync_channel(1);
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    let gate = Arc::new(StopGate {
        entered: entered_tx,
        release: Mutex::new(release_rx),
    });
    let runner = paused_runner(
        fixture.workspace.clone(),
        fixture.root.clone(),
        cancel.clone(),
        gate.clone(),
        cancelled,
    );
    let source = wait_test_text(&fixture.root.join("ready"));
    let target: i32 = source.rsplit('/').next().unwrap().parse().unwrap();
    let original = std::fs::metadata(&source).unwrap();
    if cancelled {
        cancel.cancel();
    }
    entered_rx.recv_timeout(Duration::from_secs(30)).unwrap(); // teardown is now held at the stop boundary.
    let pinned = std::fs::metadata(&source).is_ok_and(|m| m.dev() == original.dev() && m.ino() == original.ino());
    let sentinels = other_request_memfds();
    let reused = sentinels.iter().any(|file| file.as_raw_fd() == target);
    std::fs::write(fixture.root.join("allow-read"), b"go").unwrap();
    let consumed = wait_test_text(&fixture.root.join("consumed"));
    release_tx.send(()).unwrap();
    let result = runner.join().unwrap().unwrap(); // cleanup/reap completes before assertions.
    assert!(pinned && !reused, "environment fd was released before adapter termination/handoff");
    assert_eq!(consumed, if cancelled { "ORIGINAL_CANCEL" } else { "ORIGINAL_DEADLINE" });
    assert_eq!(
        std::fs::read_to_string(fixture.root.join("adapter-count")).unwrap(),
        "once\n",
        "adapter was replayed"
    );
    assert_eq!(result.exit, if cancelled { Exit::Cancelled } else { Exit::TimedOut });
    let mut ordinary = RunRequest::new("printf '%s' \"$RACE_VALUE\"");
    ordinary.env = vec![("RACE_VALUE".into(), "NORMAL_ENV_WORKS".into())];
    let normal = run_with(&fixture.workspace, &ordinary, &RunOptions::new(&ShellPolicy::Unrestricted)).unwrap();
    assert!(normal.success());
    assert_eq!(normal.output, "NORMAL_ENV_WORKS");
}

#[test]
fn memfd_lifetime_cancel_barrier_prevents_aba() {
    handoff_lifetime_barrier(true);
}

#[test]
fn memfd_lifetime_deadline_barrier_prevents_aba() {
    handoff_lifetime_barrier(false);
}
