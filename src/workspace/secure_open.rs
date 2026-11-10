//! Opening a resolved path without following symlinks.
//!
//! `resolve` returns a symlink-free location, but the tree can change between
//! resolving and opening. These functions open `relative` beneath `base` and
//! refuse every symlink on the way, so a component that was swapped for a
//! symlink in the meantime is an error, never a way out of the workspace.
//!
//! Two implementations with the same contract:
//!
//! * `openat2(2)` with `RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS` (Linux 5.6+),
//!   one system call.
//! * A walk of `openat(2)` calls with `O_NOFOLLOW`, one per component, for
//!   kernels and sandboxes without `openat2`. Each step holds the directory it
//!   came from open, so it is just as immune to renames.
//!
//! What neither can do: protect the path *to* `base` itself (the workspace
//! root's own ancestors are the owner's, not the model's), or stop a file
//! from being replaced by another regular file.

use std::ffi::{CString, OsStr};
use std::fs::{File, Metadata};
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::path::{Component, Path};
use std::sync::atomic::{AtomicBool, Ordering};

use super::error::{Error, Result, Rule};
use super::resolve::{Kind, Resolved};

const RESOLVE_NO_MAGICLINKS: u64 = 0x02;
const RESOLVE_NO_SYMLINKS: u64 = 0x04;
const RESOLVE_BENEATH: u64 = 0x08;

/// Reading a file: non-blocking so a FIFO cannot hang the open (regular files
/// ignore it), and never a controlling terminal.
const READ_FLAGS: libc::c_int = libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NONBLOCK | libc::O_NOCTTY | libc::O_NOFOLLOW;
/// A handle for `fstat` only; with `O_NOFOLLOW` it may name a symlink itself.
const STAT_FLAGS: libc::c_int = libc::O_PATH | libc::O_CLOEXEC | libc::O_NOFOLLOW;
const DIRECTORY_FLAGS: libc::c_int = libc::O_PATH | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW;

/// Set once `openat2` turned out to be unavailable in this process.
pub(super) static OPENAT2_UNAVAILABLE: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// `openat2`, falling back to the walk when the kernel lacks it.
    Auto,
    /// Only `openat2`; an error when it is unavailable. For tests.
    Openat2,
    /// Only the component walk. For tests and old kernels.
    Walk,
}

impl Resolved {
    /// Open the resolved regular file for reading.
    pub fn open_file(&self) -> Result<(File, Metadata)> {
        self.open_file_with(Strategy::Auto)
    }

    pub fn open_file_with(&self, strategy: Strategy) -> Result<(File, Metadata)> {
        match self.kind {
            Kind::Directory => return Err(Error::WrongKind { path: self.supplied.clone(), expected: "a file (it is a directory)" }),
            Kind::Missing => return Err(Error::NotFound { path: self.supplied.clone() }),
            Kind::File | Kind::Other => {}
        }
        let file = open_file_beneath(&self.base, &self.relative, strategy).map_err(|error| open_error(&self.supplied, &error))?;
        let metadata = file.metadata().map_err(|error| Error::io(&self.supplied, "inspect", &error))?;
        if !metadata.is_file() {
            return Err(Error::WrongKind { path: self.supplied.clone(), expected: "a regular file" });
        }
        Ok((file, metadata))
    }
}

/// Translate an open failure; `ELOOP` means a symlink appeared on the path.
pub fn open_error(supplied: &str, error: &io::Error) -> Error {
    if error.raw_os_error() == Some(libc::ELOOP) {
        Error::Refused { rule: Rule::ChangedToSymlink, path: supplied.to_string() }
    } else {
        Error::io(supplied, "open", error)
    }
}

/// Open `relative` beneath `base` for reading, refusing symlinks.
pub fn open_file_beneath(base: &Path, relative: &Path, strategy: Strategy) -> io::Result<File> {
    open_beneath(base, relative, READ_FLAGS, strategy).map(File::from)
}

/// `lstat` of `relative` beneath `base`, refusing symlinks in every component
/// but the last (a final symlink is described, not followed).
pub fn stat_beneath(base: &Path, relative: &Path, strategy: Strategy) -> io::Result<Metadata> {
    File::from(open_beneath(base, relative, STAT_FLAGS, strategy)?).metadata()
}

fn open_beneath(base: &Path, relative: &Path, flags: libc::c_int, strategy: Strategy) -> io::Result<OwnedFd> {
    let names = normal_components(relative)?;
    let base = open_raw(libc::AT_FDCWD, base.as_os_str(), libc::O_PATH | libc::O_DIRECTORY | libc::O_CLOEXEC)?;
    let try_openat2 = match strategy {
        Strategy::Walk => false,
        Strategy::Openat2 => true,
        Strategy::Auto => !OPENAT2_UNAVAILABLE.load(Ordering::Relaxed),
    };
    if try_openat2 {
        match openat2(&base, relative, flags, 0) {
            Err(error) if strategy == Strategy::Auto && openat2_unusable(&error) => {
                if error.raw_os_error() == Some(libc::ENOSYS) {
                    OPENAT2_UNAVAILABLE.store(true, Ordering::Relaxed);
                }
            }
            other => return other,
        }
    }
    walk(base, &names, flags)
}

/// Errors after which the walk is tried instead: the call does not exist, a
/// sandbox filters it, or the kernel gave up on a concurrent rename.
pub(super) fn openat2_unusable(error: &io::Error) -> bool {
    matches!(error.raw_os_error(), Some(libc::ENOSYS | libc::EPERM | libc::EINVAL | libc::E2BIG | libc::EAGAIN))
}

/// `relative` as plain names: non-empty, no `..`, no root.
fn normal_components(relative: &Path) -> io::Result<Vec<&OsStr>> {
    let mut names = Vec::new();
    for part in relative.components() {
        match part {
            Component::Normal(name) => names.push(name),
            _ => return Err(io::Error::from(io::ErrorKind::InvalidInput)),
        }
    }
    if names.is_empty() { Err(io::Error::from(io::ErrorKind::InvalidInput)) } else { Ok(names) }
}

/// Open one entry `name` of the open directory `directory` without following
/// a symlink, with the chosen strategy. `name` must be a single plain
/// component. `mode` is used only together with `O_CREAT`.
///
/// This is the step the write side is built from: every directory on the way
/// to a file is opened like this, relative to the descriptor of its parent.
pub(super) fn open_entry(
    directory: &OwnedFd,
    name: &OsStr,
    flags: libc::c_int,
    mode: libc::mode_t,
    strategy: Strategy,
) -> io::Result<OwnedFd> {
    let path = Path::new(name);
    if normal_components(path)?.len() != 1 {
        return Err(io::Error::from(io::ErrorKind::InvalidInput));
    }
    let try_openat2 = match strategy {
        Strategy::Walk => false,
        Strategy::Openat2 => true,
        Strategy::Auto => !OPENAT2_UNAVAILABLE.load(Ordering::Relaxed),
    };
    if try_openat2 {
        match openat2(directory, path, flags, mode) {
            Err(error) if strategy == Strategy::Auto && openat2_unusable(&error) => {
                if error.raw_os_error() == Some(libc::ENOSYS) {
                    OPENAT2_UNAVAILABLE.store(true, Ordering::Relaxed);
                }
            }
            other => return other,
        }
    }
    match open_raw_mode(directory.as_raw_fd(), name, flags | libc::O_NOFOLLOW, mode) {
        // O_DIRECTORY|O_NOFOLLOW reports a symlink as "not a directory".
        Err(error) if error.raw_os_error() == Some(libc::ENOTDIR) && is_symlink(directory, name) => {
            Err(io::Error::from_raw_os_error(libc::ELOOP))
        }
        other => other,
    }
}

/// Open a directory by path as the base of beneath-operations.
pub(super) fn open_base(base: &Path) -> io::Result<OwnedFd> {
    open_raw(libc::AT_FDCWD, base.as_os_str(), libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC)
}

pub(super) fn c_string(text: &OsStr) -> io::Result<CString> {
    CString::new(text.as_bytes()).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))
}

fn open_raw(directory: libc::c_int, name: &OsStr, flags: libc::c_int) -> io::Result<OwnedFd> {
    open_raw_mode(directory, name, flags, 0)
}

fn open_raw_mode(directory: libc::c_int, name: &OsStr, flags: libc::c_int, mode: libc::mode_t) -> io::Result<OwnedFd> {
    let name = c_string(name)?;
    loop {
        // SAFETY: `name` is a valid NUL-terminated string that outlives the
        // call; `directory` is AT_FDCWD or a descriptor we hold open.
        let fd = unsafe { libc::openat(directory, name.as_ptr(), flags, libc::c_uint::from(mode)) };
        if fd >= 0 {
            // SAFETY: the kernel just returned this descriptor to us alone.
            return Ok(unsafe { OwnedFd::from_raw_fd(fd) });
        }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

#[repr(C)]
struct OpenHow {
    flags: u64,
    mode: u64,
    resolve: u64,
}

fn openat2(base: &OwnedFd, relative: &Path, flags: libc::c_int, mode: libc::mode_t) -> io::Result<OwnedFd> {
    let path = c_string(relative.as_os_str())?;
    let how = OpenHow {
        flags: u64::try_from(flags).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?,
        mode: u64::from(mode),
        resolve: RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS,
    };
    loop {
        // SAFETY: `path` is a valid NUL-terminated string and `how` a properly
        // sized, initialized `open_how`; both outlive the call.
        let fd = unsafe {
            libc::syscall(libc::SYS_openat2, base.as_raw_fd(), path.as_ptr(), &how as *const OpenHow, std::mem::size_of::<OpenHow>())
        };
        if let Ok(fd) = libc::c_int::try_from(fd) {
            if fd >= 0 {
                // SAFETY: the kernel just returned this descriptor to us alone.
                return Ok(unsafe { OwnedFd::from_raw_fd(fd) });
            }
        }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

/// Open component by component. Every directory on the way is opened with
/// `O_NOFOLLOW | O_DIRECTORY`, which fails on a symlink; that failure is
/// reported as `ELOOP`, like `openat2` reports it.
fn walk(base: OwnedFd, names: &[&OsStr], final_flags: libc::c_int) -> io::Result<OwnedFd> {
    let mut directory = base;
    for (index, name) in names.iter().enumerate() {
        let last = index + 1 == names.len();
        match open_raw(directory.as_raw_fd(), name, if last { final_flags } else { DIRECTORY_FLAGS }) {
            Ok(next) => directory = next,
            Err(error) if !last && error.raw_os_error() == Some(libc::ENOTDIR) && is_symlink(&directory, name) => {
                return Err(io::Error::from_raw_os_error(libc::ELOOP));
            }
            Err(error) => return Err(error),
        }
    }
    Ok(directory)
}

fn is_symlink(directory: &OwnedFd, name: &OsStr) -> bool {
    open_raw(directory.as_raw_fd(), name, STAT_FLAGS)
        .and_then(|fd| File::from(fd).metadata())
        .is_ok_and(|metadata| metadata.file_type().is_symlink())
}
