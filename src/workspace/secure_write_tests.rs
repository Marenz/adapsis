use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::Path;

use super::error::{Error, Rule};
use super::ledger::Freshness;
use super::resolve::{Access, Kind, resolve};
use super::resolve_tests::tree;
use super::secure_open::Strategy;
use super::secure_write::{Replace, Stamp, TEMP_PREFIX, write_beneath};
use super::test_support::{Fixture, symlink, write};

const STRATEGIES: [Strategy; 3] = [Strategy::Auto, Strategy::Openat2, Strategy::Walk];

/// The process umask, read without changing it.
pub fn umask() -> u32 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status.lines().find(|line| line.starts_with("Umask:")).expect("the kernel reports the umask");
    u32::from_str_radix(line.trim_start_matches("Umask:").trim(), 8).unwrap()
}

fn mode(path: &Path) -> u32 {
    std::fs::symlink_metadata(path).unwrap().mode() & 0o7777
}

fn os_error(result: std::io::Result<super::secure_write::Written>) -> i32 {
    result.unwrap_err().raw_os_error().expect("an OS error code")
}

/// No temporary file may survive an operation, successful or not.
pub fn assert_no_temp_files(directory: &Path) {
    for entry in std::fs::read_dir(directory).unwrap().flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        assert!(!name.starts_with(TEMP_PREFIX), "left behind: {}", entry.path().display());
        if entry.file_type().unwrap().is_dir() {
            assert_no_temp_files(&entry.path());
        }
    }
}

/// Swap two directory entries atomically, so there is never a moment in
/// which the name is missing.
fn exchange(first: &Path, second: &Path) -> bool {
    let first = std::ffi::CString::new(first.to_str().unwrap()).unwrap();
    let second = std::ffi::CString::new(second.to_str().unwrap()).unwrap();
    // SAFETY: two valid NUL-terminated paths; RENAME_EXCHANGE is flag 2.
    unsafe { libc::syscall(libc::SYS_renameat2, libc::AT_FDCWD, first.as_ptr(), libc::AT_FDCWD, second.as_ptr(), 2u32) == 0 }
}

#[test]
fn secure_write_creates_a_file_and_its_directory_chain() {
    for strategy in STRATEGIES {
        let fixture = Fixture::plain();
        fixture.write("a/keep.txt", "keep\n");
        let written = write_beneath(&fixture.root, Path::new("a/b/c/file.txt"), b"hello\n", Replace::CreateNew, strategy).unwrap();
        let path = fixture.root.join("a/b/c/file.txt");
        assert_eq!(std::fs::read(&path).unwrap(), b"hello\n", "{strategy:?}");
        assert_eq!((written.size, written.created, written.mode), (6, true, 0o666 & !umask()), "{strategy:?}");
        assert_eq!(mode(&path), 0o666 & !umask(), "a new file gets 0666 minus the umask");
        assert_eq!(mode(&fixture.root.join("a/b")), 0o777 & !umask(), "a new directory gets 0777 minus the umask");
        let metadata = std::fs::metadata(&path).unwrap();
        assert_eq!(written.mtime_ns, metadata.mtime() * 1_000_000_000 + metadata.mtime_nsec());
        assert_eq!(std::fs::read_to_string(fixture.root.join("a/keep.txt")).unwrap(), "keep\n");
        assert_no_temp_files(&fixture.root);
        // Directly in the base, and empty content.
        write_beneath(&fixture.root, Path::new("top.txt"), b"", Replace::CreateNew, strategy).unwrap();
        assert_eq!(std::fs::read(fixture.root.join("top.txt")).unwrap(), b"");
    }
}

#[test]
fn secure_write_replaces_atomically_and_keeps_the_mode() {
    for strategy in STRATEGIES {
        let fixture = Fixture::plain();
        for file_mode in [0o750, 0o600, 0o644, 0o444] {
            let path = fixture.write("script.sh", "#!/bin/sh\necho old\n");
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(file_mode)).unwrap();
            let before = std::fs::metadata(&path).unwrap().ino();
            let written =
                write_beneath(&fixture.root, Path::new("script.sh"), b"#!/bin/sh\necho new\n", Replace::Overwrite { expect: None }, strategy).unwrap();
            assert_eq!(std::fs::read_to_string(&path).unwrap(), "#!/bin/sh\necho new\n");
            assert_eq!((mode(&path), written.mode, written.created), (file_mode, file_mode, false), "{strategy:?} {file_mode:o}");
            assert_ne!(std::fs::metadata(&path).unwrap().ino(), before, "replaced by a new file, not written in place");
            std::fs::remove_file(&path).unwrap();
        }
        assert_no_temp_files(&fixture.root);
    }
}

#[test]
fn secure_write_refuses_what_does_not_fit_the_request() {
    for strategy in STRATEGIES {
        let fixture = Fixture::plain();
        fixture.write("exists.txt", "original\n");
        fixture.write("dir/inner.txt", "x\n");
        let fifo = std::ffi::CString::new(fixture.root.join("pipe").to_str().unwrap()).unwrap();
        // SAFETY: a valid NUL-terminated path; mkfifo touches no memory.
        assert_eq!(unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) }, 0);
        let attempt = |path: &str, how| os_error(write_beneath(&fixture.root, Path::new(path), b"new\n", how, strategy));
        let overwrite = Replace::Overwrite { expect: None };
        assert_eq!(attempt("exists.txt", Replace::CreateNew), libc::EEXIST, "{strategy:?}");
        assert_eq!(std::fs::read_to_string(fixture.root.join("exists.txt")).unwrap(), "original\n");
        assert_eq!(attempt("absent.txt", overwrite), libc::ENOENT);
        assert!(!fixture.root.join("absent.txt").exists());
        assert_eq!(attempt("dir", overwrite), libc::EISDIR);
        assert_eq!(attempt("dir", Replace::CreateNew), libc::EISDIR);
        assert_eq!(attempt("pipe", overwrite), libc::ENXIO);
        assert_eq!(attempt("exists.txt/under/file", Replace::CreateNew), libc::ENOTDIR);
        for invalid in ["", "../x", "/etc/x", "dir/../x", "."] {
            let error = write_beneath(&fixture.root, Path::new(invalid), b"x", Replace::CreateNew, strategy).unwrap_err();
            assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput, "{strategy:?} {invalid:?}");
        }
        assert_no_temp_files(&fixture.root);
    }
}

#[test]
fn secure_write_overwrite_is_tied_to_the_file_that_was_read() {
    for strategy in STRATEGIES {
        let fixture = Fixture::plain();
        let path = fixture.write("code.rs", "fn a() {}\n");
        let stamp = Stamp::of(&std::fs::metadata(&path).unwrap());
        write_beneath(&fixture.root, Path::new("code.rs"), b"fn b() {}\n", Replace::Overwrite { expect: Some(stamp) }, strategy).unwrap();
        // The stamp is of the old file; the file is a different one now.
        let stale = write_beneath(&fixture.root, Path::new("code.rs"), b"fn c() {}\n", Replace::Overwrite { expect: Some(stamp) }, strategy);
        assert_eq!(os_error(stale), libc::ESTALE, "{strategy:?}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "fn b() {}\n");
        // Somebody else's in-place change is noticed as well.
        let stamp = Stamp::of(&std::fs::metadata(&path).unwrap());
        std::fs::write(&path, "fn somebody_else() {}\n").unwrap();
        let stale = write_beneath(&fixture.root, Path::new("code.rs"), b"fn c() {}\n", Replace::Overwrite { expect: Some(stamp) }, strategy);
        assert_eq!(os_error(stale), libc::ESTALE);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "fn somebody_else() {}\n");
        assert_no_temp_files(&fixture.root);
    }
}

#[test]
fn secure_write_never_goes_through_a_symlink() {
    for strategy in STRATEGIES {
        let fixture = tree();
        let attempt = |path: &str, how| os_error(write_beneath(&fixture.root, Path::new(path), b"OVERWRITTEN\n", how, strategy));
        let overwrite = Replace::Overwrite { expect: None };
        // The final component.
        for link in ["link_out", "link_in", "abs_in", "dangling", "loop1"] {
            assert_eq!(attempt(link, overwrite), libc::ELOOP, "{strategy:?} {link}");
            assert_eq!(attempt(link, Replace::CreateNew), libc::ELOOP, "{strategy:?} {link}");
            assert!(std::fs::symlink_metadata(fixture.root.join(link)).unwrap().file_type().is_symlink(), "the link itself is untouched");
        }
        // A directory on the way, to the outside or back inside, existing or dangling.
        for path in ["link_dir_out/new.txt", "link_dir_out/secret.txt", "link_dir_in/b.txt", "link_up/ws/a.txt", "dangling/sub/new.txt", "dir/link_root/a.txt"] {
            assert_eq!(attempt(path, Replace::CreateNew), libc::ELOOP, "{strategy:?} {path}");
            assert_eq!(attempt(path, overwrite), libc::ELOOP, "{strategy:?} {path}");
        }
        assert_eq!(std::fs::read_to_string(fixture.outside.join("secret.txt")).unwrap(), "SECRET\n");
        assert_eq!(std::fs::read_dir(&fixture.outside).unwrap().count(), 1, "nothing was created outside");
        assert!(!fixture.root.join("nowhere").exists(), "a dangling link's target is not created");
        assert_eq!(std::fs::read_to_string(fixture.root.join("a.txt")).unwrap(), "alpha\n");
        assert_eq!(std::fs::read_to_string(fixture.root.join("dir/b.txt")).unwrap(), "beta\n");
        assert_no_temp_files(&fixture.top);
    }
}

#[test]
fn secure_write_through_the_resolver_applies_every_write_rule() {
    let fixture = tree();
    let create = |path: &str| resolve(&fixture.workspace, path, Access::Write).and_then(|resolved| resolved.write_content(b"x\n", Replace::CreateNew));
    assert_eq!(create(".git/hooks/pre-commit").unwrap_err().rule(), Some(Rule::WriteGitDir));
    assert_eq!(create("dir/.git/config").unwrap_err().rule(), Some(Rule::WriteGitDir));
    assert_eq!(create("../outside/new.txt").unwrap_err().rule(), Some(Rule::ParentEscape));
    assert_eq!(create("link_dir_out/new.txt").unwrap_err().rule(), Some(Rule::SymlinkOutside));
    assert_eq!(create("link_out").unwrap_err().rule(), Some(Rule::WriteThroughSymlink));
    assert_eq!(create("link_extra_dir/new.txt").unwrap_err().rule(), Some(Rule::WriteReadOnlyRoot));
    assert!(matches!(create("."), Err(Error::WrongKind { .. })));
    assert!(matches!(create("dir"), Err(Error::WrongKind { .. })));
    assert!(matches!(create("a.txt"), Err(Error::NotFresh { state: Freshness::Unread, .. })), "create never replaces");
    // A path resolved for reading is not accepted for writing, whatever it is.
    let for_reading = resolve(&fixture.workspace, "a.txt", Access::Read).unwrap();
    let error = for_reading.write_content(b"x\n", Replace::Overwrite { expect: None }).unwrap_err();
    assert!(matches!(error, Error::InvalidArgument(_)), "{error:?}");
    // A symlinked parent that stays inside is fine: the file lands in the real directory.
    let through = resolve(&fixture.workspace, "link_dir_in/made.txt", Access::Write).unwrap();
    through.write_content(b"made\n", Replace::CreateNew).unwrap();
    assert_eq!(std::fs::read_to_string(fixture.root.join("dir/made.txt")).unwrap(), "made\n");
    assert_eq!(std::fs::read_dir(&fixture.outside).unwrap().count(), 1);
    assert_eq!(std::fs::read_dir(&fixture.extra).unwrap().count(), 3);
    assert_eq!(std::fs::read_to_string(fixture.root.join(".git/config")).unwrap(), "[core]\n");
}

#[test]
fn race_write_after_an_ancestor_became_a_symlink_is_refused() {
    for strategy in STRATEGIES {
        let fixture = tree();
        write(&fixture.outside, "b.txt", "SECRET\n");
        let existing = resolve(&fixture.workspace, "dir/b.txt", Access::Write).unwrap();
        let fresh = resolve(&fixture.workspace, "dir/sub/deeper/new.txt", Access::Write).unwrap();
        assert_eq!((existing.kind, fresh.kind), (Kind::File, Kind::Missing));
        // Between resolving and writing, `dir` becomes a link to the outside.
        std::fs::rename(fixture.root.join("dir"), fixture.root.join("dir-moved")).unwrap();
        symlink("../outside", fixture.root.join("dir"));
        let error = existing.write_content_with(b"OVERWRITTEN\n", Replace::Overwrite { expect: None }, strategy).unwrap_err();
        assert_eq!(error.rule(), Some(Rule::ChangedToSymlink), "{strategy:?}: {error}");
        let error = fresh.write_content_with(b"CREATED\n", Replace::CreateNew, strategy).unwrap_err();
        assert_eq!(error.rule(), Some(Rule::ChangedToSymlink), "{strategy:?}: {error}");
        assert_eq!(std::fs::read_to_string(fixture.outside.join("b.txt")).unwrap(), "SECRET\n");
        assert!(!fixture.outside.join("sub").exists(), "no directory was created outside");
        assert_eq!(std::fs::read_to_string(fixture.root.join("dir-moved/b.txt")).unwrap(), "beta\n");
        assert_no_temp_files(&fixture.top);
    }
}

#[test]
fn race_write_after_the_file_became_a_symlink_or_appeared_is_refused() {
    for strategy in STRATEGIES {
        let fixture = tree();
        let existing = resolve(&fixture.workspace, "a.txt", Access::Write).unwrap();
        let fresh = resolve(&fixture.workspace, "new.txt", Access::Write).unwrap();
        std::fs::remove_file(fixture.root.join("a.txt")).unwrap();
        symlink("../outside/secret.txt", fixture.root.join("a.txt"));
        symlink("../outside/created.txt", fixture.root.join("new.txt"));
        let error = existing.write_content_with(b"OVERWRITTEN\n", Replace::Overwrite { expect: None }, strategy).unwrap_err();
        assert_eq!(error.rule(), Some(Rule::ChangedToSymlink), "{strategy:?}: {error}");
        let error = fresh.write_content_with(b"CREATED\n", Replace::CreateNew, strategy).unwrap_err();
        assert_eq!(error.rule(), Some(Rule::ChangedToSymlink), "{strategy:?}: {error}");
        assert_eq!(std::fs::read_to_string(fixture.outside.join("secret.txt")).unwrap(), "SECRET\n");
        assert!(!fixture.outside.join("created.txt").exists());
        // A regular file that appears where a new one was to be created is kept.
        let appears = resolve(&fixture.workspace, "appears.txt", Access::Write).unwrap();
        fixture.write("appears.txt", "somebody else\n");
        let error = appears.write_content_with(b"mine\n", Replace::CreateNew, strategy).unwrap_err();
        assert!(matches!(error, Error::NotFresh { state: Freshness::Unread, .. }), "{strategy:?}: {error:?}");
        assert_eq!(std::fs::read_to_string(fixture.root.join("appears.txt")).unwrap(), "somebody else\n");
        // And one that vanishes is not silently re-created by an overwrite.
        let vanishes = resolve(&fixture.workspace, "dir/b.txt", Access::Write).unwrap();
        std::fs::remove_file(fixture.root.join("dir/b.txt")).unwrap();
        let error = vanishes.write_content_with(b"mine\n", Replace::Overwrite { expect: None }, strategy).unwrap_err();
        assert!(matches!(error, Error::NotFresh { state: Freshness::Missing, .. }), "{strategy:?}: {error:?}");
        assert_no_temp_files(&fixture.top);
    }
}

/// The write-side twin of the read race test: a directory that keeps
/// flipping between the real one and a symlink to the outside while files
/// are created and replaced in it. Guaranteed: nothing is ever created or
/// changed outside. Not guaranteed: which attempt succeeds.
#[test]
fn race_flipping_directory_never_receives_a_write_outside() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    for strategy in [Strategy::Openat2, Strategy::Walk] {
        let fixture = tree();
        write(&fixture.outside, "b.txt", "SECRET\n");
        symlink("../outside", fixture.root.join("park"));
        let (dir, park) = (fixture.root.join("dir"), fixture.root.join("park"));
        let stop = Arc::new(AtomicBool::new(false));
        let flips = Arc::new(AtomicU64::new(0));
        let flipper = {
            let (stop, flips) = (stop.clone(), flips.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    if exchange(&dir, &park) {
                        flips.fetch_add(1, Ordering::Relaxed);
                    }
                    std::thread::yield_now();
                }
            })
        };
        let give_up = Instant::now() + Duration::from_secs(120);
        let (mut rounds, mut landed, mut refused) = (0u64, 0u64, 0u64);
        while (rounds < 600 || landed < 20 || refused < 20 || flips.load(Ordering::Relaxed) < 600) && Instant::now() < give_up {
            rounds += 1;
            let name = format!("dir/made-{}.txt", rounds % 7);
            for how in [Replace::CreateNew, Replace::Overwrite { expect: None }] {
                for path in [name.as_str(), "dir/b.txt", "dir/deep/er/file.txt"] {
                    // Through the primitive directly (the path as it was once
                    // resolved) and through a fresh resolution.
                    match write_beneath(&fixture.root, Path::new(path), b"inside\n", how, strategy) {
                        Ok(_) => landed += 1,
                        Err(_) => refused += 1,
                    }
                    if let Ok(resolved) = resolve(&fixture.workspace, path, Access::Write) {
                        let _ = resolved.write_content_with(b"inside\n", how, strategy);
                    }
                }
            }
        }
        stop.store(true, Ordering::Relaxed);
        flipper.join().unwrap();
        assert!(flips.load(Ordering::Relaxed) >= 600 && landed >= 20 && refused >= 20, "{strategy:?}: {rounds} rounds, {landed} landed, {refused} refused");
        assert_eq!(std::fs::read_to_string(fixture.outside.join("b.txt")).unwrap(), "SECRET\n", "{strategy:?}");
        assert_eq!(std::fs::read_to_string(fixture.outside.join("secret.txt")).unwrap(), "SECRET\n");
        let outside: Vec<String> = std::fs::read_dir(&fixture.outside).unwrap().flatten().map(|entry| entry.file_name().to_string_lossy().into_owned()).collect();
        assert_eq!(outside.len(), 2, "{strategy:?}: something was created outside: {outside:?}");
    }
}

#[test]
fn secure_write_concurrent_creates_have_one_winner_with_whole_content() {
    for strategy in STRATEGIES {
        let fixture = Fixture::plain();
        let root = fixture.root.clone();
        let workers: Vec<_> = (0..8u8)
            .map(|worker| {
                let root = root.clone();
                std::thread::spawn(move || {
                    let content = vec![b'a' + worker; 200_000];
                    write_beneath(&root, Path::new("shared/once.txt"), &content, Replace::CreateNew, strategy).map(|_| worker)
                })
            })
            .collect();
        let outcomes: Vec<_> = workers.into_iter().map(|worker| worker.join().unwrap()).collect();
        let winners: Vec<u8> = outcomes.iter().filter_map(|outcome| outcome.as_ref().ok().copied()).collect();
        assert_eq!(winners.len(), 1, "{strategy:?}: {outcomes:?}");
        assert!(outcomes.iter().filter_map(|outcome| outcome.as_ref().err()).all(|error| error.raw_os_error() == Some(libc::EEXIST)));
        let content = std::fs::read(root.join("shared/once.txt")).unwrap();
        assert_eq!(content, vec![b'a' + winners[0]; 200_000], "the winner's content, whole");
        assert_no_temp_files(&root);
    }
}

#[test]
fn secure_write_readers_never_see_a_partial_file() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    let fixture = Fixture::plain();
    let path = fixture.write("big.txt", vec![b'A'; 300_000]);
    let stop = Arc::new(AtomicBool::new(false));
    let reader = {
        let (stop, path) = (stop.clone(), path.clone());
        std::thread::spawn(move || {
            let mut reads = 0u32;
            while !stop.load(Ordering::Relaxed) {
                let content = std::fs::read(&path).unwrap();
                assert_eq!(content.len(), 300_000, "a reader saw a truncated file");
                assert!(content.iter().all(|byte| *byte == content[0]), "a reader saw a mix of old and new content");
                reads += 1;
            }
            reads
        })
    };
    for round in 0..150u32 {
        let content = vec![if round % 2 == 0 { b'B' } else { b'A' }; 300_000];
        write_beneath(&fixture.root, Path::new("big.txt"), &content, Replace::Overwrite { expect: None }, Strategy::Auto).unwrap();
    }
    stop.store(true, Ordering::Relaxed);
    assert!(reader.join().unwrap() > 0);
    assert_no_temp_files(&fixture.root);
}
