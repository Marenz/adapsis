//! Writing a resolved path without following symlinks.
//!
//! The write side of `secure_open`. A file is never written in place and
//! never through a path: every directory on the way is opened relative to the
//! descriptor of its parent with symlinks refused (`secure_open::open_entry`,
//! `openat2` or the `O_NOFOLLOW` walk), missing directories are created with
//! `mkdirat` on those descriptors, the content goes to a temporary file in the
//! target's own directory, is flushed to disk, and is then moved over the
//! target with `renameat` on that same directory descriptor.
//!
//! What follows from that:
//!
//! * A component that became a symlink after resolution is an error
//!   (`ELOOP`); nothing is created or changed behind it.
//! * The directory the file lands in is the one that was opened. If it is
//!   renamed meanwhile the file still lands in that directory, wherever it is
//!   now, which is inside the workspace because it was reached without a
//!   symlink.
//! * A reader sees the old content or the new content, never a mix; a crash
//!   leaves the old file (and possibly a `.adapsis-tmp-*` file next to it).
//! * Creating never replaces: a file that appeared in the meantime is an
//!   error. Overwriting can be tied to the identity of the file the caller
//!   read (`Stamp`); a mismatch is an error. That check and the rename are two
//!   system calls, so a writer that slips in between them is overwritten.
//!
//! What it does not do: keep hard links (the target becomes a new inode),
//! extended attributes, ACLs or ownership. The mode bits are kept.

use std::ffi::OsStr;
use std::fs::{File, Metadata};
use std::io::{self, Write};
use std::os::fd::{AsRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path};
use std::sync::atomic::{AtomicU64, Ordering};

use super::error::{Error, Result, Rule};
use super::ledger::Freshness;
use super::resolve::{Access, Area, Resolved};
use super::secure_open::{Strategy, c_string, open_base, open_entry};

const DIRECTORY_FLAGS: libc::c_int = libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC;
const STAT_FLAGS: libc::c_int = libc::O_PATH | libc::O_CLOEXEC | libc::O_NOFOLLOW;
const CREATE_FLAGS: libc::c_int = libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC;
const RENAME_NOREPLACE: libc::c_uint = 1;

/// Prefix of the temporary files; one may be left behind by a crash.
pub const TEMP_PREFIX: &str = ".adapsis-tmp-";

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Identity of a file at one moment: which inode, and its size and time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stamp {
    pub device: u64,
    pub inode: u64,
    pub size: u64,
    pub mtime_ns: i64,
}

impl Stamp {
    pub fn of(metadata: &Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            size: metadata.len(),
            mtime_ns: metadata.mtime().saturating_mul(1_000_000_000).saturating_add(metadata.mtime_nsec()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Replace {
    /// The file must not exist; it gets mode 0666 minus the umask.
    CreateNew,
    /// The file must exist and be a regular file; its mode is kept. With
    /// `expect`, it must still be the file the caller looked at.
    Overwrite { expect: Option<Stamp> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Written {
    pub size: u64,
    pub mtime_ns: i64,
    /// Permission bits of the file as written.
    pub mode: u32,
    pub created: bool,
}

impl Resolved {
    /// Write `content` to a path that was resolved for writing.
    pub fn write_content(&self, content: &[u8], how: Replace) -> Result<Written> {
        self.write_content_with(content, how, Strategy::Auto)
    }

    pub fn write_content_with(&self, content: &[u8], how: Replace, strategy: Strategy) -> Result<Written> {
        if self.access != Access::Write {
            return Err(Error::InvalidArgument(format!("`{}` was not resolved for writing", self.supplied)));
        }
        // The resolver already decided these; a second look costs nothing.
        if self.area != Area::Root {
            return Err(Error::Refused { rule: Rule::WriteReadOnlyRoot, path: self.supplied.clone() });
        }
        if self.relative.components().any(|part| part.as_os_str().as_bytes().eq_ignore_ascii_case(b".git")) {
            return Err(Error::Refused { rule: Rule::WriteGitDir, path: self.supplied.clone() });
        }
        if self.relative.as_os_str().is_empty() {
            return Err(Error::WrongKind { path: self.supplied.clone(), expected: "a file (it is the workspace root)" });
        }
        write_beneath(&self.base, &self.relative, content, how, strategy).map_err(|error| write_error(&self.supplied, &error))
    }
}

/// Translate a failed write into the message a model can act on.
pub fn write_error(supplied: &str, error: &io::Error) -> Error {
    let path = supplied.to_string();
    match error.raw_os_error() {
        Some(libc::ELOOP) => Error::Refused { rule: Rule::ChangedToSymlink, path },
        Some(libc::EEXIST) => Error::NotFresh { path, state: Freshness::Unread },
        Some(libc::ESTALE) => Error::NotFresh { path, state: Freshness::Changed },
        Some(libc::ENOENT) => Error::NotFresh { path, state: Freshness::Missing },
        Some(libc::EISDIR) => Error::WrongKind { path, expected: "a file (it is a directory)" },
        Some(libc::ENOTDIR) => Error::WrongKind { path, expected: "a path through directories (one of its components is a file)" },
        Some(libc::ENXIO) => Error::WrongKind { path, expected: "a regular file" },
        _ => Error::io(supplied, "write", error),
    }
}

/// Write `content` to `relative` beneath `base`, creating missing directories.
///
/// Errors are reported as OS errors: `ELOOP` a symlink on the way or as the
/// target, `EEXIST` create-new found a file, `ENOENT` overwrite found none,
/// `ESTALE` the file is not the expected one, `EISDIR`/`ENXIO` the target is a
/// directory / not a regular file.
pub fn write_beneath(base: &Path, relative: &Path, content: &[u8], how: Replace, strategy: Strategy) -> io::Result<Written> {
    let names = plain_names(relative)?;
    let Some((name, parents)) = names.split_last() else {
        return Err(io::Error::from(io::ErrorKind::InvalidInput));
    };
    let mut directory = open_base(base)?;
    for parent in parents {
        directory = open_or_create_directory(&directory, parent, strategy)?;
    }
    let mode = existing_mode(&directory, name, how, strategy)?;
    let (temp_name, temp) = create_temp(&directory, mode, strategy)?;
    let outcome = fill_and_commit(&directory, &temp_name, temp, name, content, how, mode, strategy);
    if outcome.is_err() {
        remove(&directory, &temp_name);
    }
    outcome
}

fn plain_names(relative: &Path) -> io::Result<Vec<&OsStr>> {
    relative
        .components()
        .map(|part| match part {
            Component::Normal(name) => Ok(name),
            _ => Err(io::Error::from(io::ErrorKind::InvalidInput)),
        })
        .collect()
}

/// Open the directory `name` in `parent`, creating it if it does not exist.
/// A symlink at that name is refused, dangling or not.
fn open_or_create_directory(parent: &OwnedFd, name: &OsStr, strategy: Strategy) -> io::Result<OwnedFd> {
    match open_entry(parent, name, DIRECTORY_FLAGS, 0, strategy) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        other => return other,
    }
    let c_name = c_string(name)?;
    // SAFETY: `c_name` is a valid NUL-terminated string; `parent` is open.
    if unsafe { libc::mkdirat(parent.as_raw_fd(), c_name.as_ptr(), 0o777) } != 0 {
        let error = io::Error::last_os_error();
        // Somebody else created it first: fine, as long as it opens below.
        if error.raw_os_error() != Some(libc::EEXIST) {
            return Err(error);
        }
    }
    open_entry(parent, name, DIRECTORY_FLAGS, 0, strategy)
}

fn stat_entry(directory: &OwnedFd, name: &OsStr, strategy: Strategy) -> io::Result<Option<Metadata>> {
    match open_entry(directory, name, STAT_FLAGS, 0, strategy) {
        Ok(handle) => File::from(handle).metadata().map(Some),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

/// Check the target against what `how` requires. Returns the mode to keep
/// (`None` for a new file).
fn existing_mode(directory: &OwnedFd, name: &OsStr, how: Replace, strategy: Strategy) -> io::Result<Option<u32>> {
    let existing = stat_entry(directory, name, strategy)?;
    let code = match (&existing, how) {
        (Some(metadata), _) if metadata.file_type().is_symlink() => libc::ELOOP,
        (Some(metadata), _) if metadata.is_dir() => libc::EISDIR,
        (Some(metadata), _) if !metadata.is_file() => libc::ENXIO,
        (Some(_), Replace::CreateNew) => libc::EEXIST,
        (None, Replace::Overwrite { .. }) => libc::ENOENT,
        (Some(metadata), Replace::Overwrite { expect: Some(stamp) }) if Stamp::of(metadata) != stamp => libc::ESTALE,
        (None, Replace::CreateNew) => return Ok(None),
        (Some(metadata), Replace::Overwrite { .. }) => return Ok(Some(metadata.mode() & 0o7777)),
    };
    Err(io::Error::from_raw_os_error(code))
}

/// Create an exclusive temporary file in `directory`. A replacement starts
/// private (0600) and gets the old file's mode before it is moved in place; a
/// new file is created 0666 so the umask decides, as for any new file.
fn create_temp(directory: &OwnedFd, keep_mode: Option<u32>, strategy: Strategy) -> io::Result<(std::ffi::OsString, File)> {
    let mode: libc::mode_t = if keep_mode.is_some() { 0o600 } else { 0o666 };
    let mut last = io::Error::from(io::ErrorKind::AlreadyExists);
    for _ in 0..8 {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |elapsed| elapsed.subsec_nanos());
        let name = format!("{TEMP_PREFIX}{}-{}-{nanos:x}", std::process::id(), TEMP_COUNTER.fetch_add(1, Ordering::Relaxed));
        match open_entry(directory, OsStr::new(&name), CREATE_FLAGS, mode, strategy) {
            Ok(handle) => return Ok((name.into(), File::from(handle))),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => last = error,
            Err(error) => return Err(error),
        }
    }
    Err(last)
}

#[allow(clippy::too_many_arguments)]
fn fill_and_commit(
    directory: &OwnedFd,
    temp_name: &OsStr,
    mut temp: File,
    name: &OsStr,
    content: &[u8],
    how: Replace,
    keep_mode: Option<u32>,
    strategy: Strategy,
) -> io::Result<Written> {
    temp.write_all(content)?;
    if let Some(mode) = keep_mode {
        // SAFETY: fchmod on a descriptor we own; no memory is involved.
        if unsafe { libc::fchmod(temp.as_raw_fd(), mode as libc::mode_t) } != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    temp.sync_all()?;
    let metadata = temp.metadata()?;
    match how {
        Replace::CreateNew => rename_no_replace(directory, temp_name, name)?,
        Replace::Overwrite { expect } => {
            // The last look before the move: still a regular file, and still
            // the one the caller read.
            existing_mode(directory, name, Replace::Overwrite { expect }, strategy)?;
            rename(directory, temp_name, name, 0)?;
        }
    }
    // Make the new directory entry durable. A failure here does not undo the
    // write, so it is not reported as one.
    // SAFETY: fsync on a descriptor we hold open.
    unsafe { libc::fsync(directory.as_raw_fd()) };
    let stamp = Stamp::of(&metadata);
    Ok(Written { size: stamp.size, mtime_ns: stamp.mtime_ns, mode: metadata.mode() & 0o7777, created: how == Replace::CreateNew })
}

fn rename(directory: &OwnedFd, from: &OsStr, to: &OsStr, flags: libc::c_uint) -> io::Result<()> {
    let (from, to) = (c_string(from)?, c_string(to)?);
    let fd = directory.as_raw_fd();
    // SAFETY: both names are valid NUL-terminated strings; `fd` is open.
    let result = unsafe { libc::syscall(libc::SYS_renameat2, fd, from.as_ptr(), fd, to.as_ptr(), flags) };
    if result == 0 { Ok(()) } else { Err(io::Error::last_os_error()) }
}

/// Move the temporary file to `name` only if `name` does not exist. Uses
/// `RENAME_NOREPLACE`; where the filesystem lacks it, a hard link (which also
/// refuses an existing name) followed by removing the temporary name.
fn rename_no_replace(directory: &OwnedFd, temp_name: &OsStr, name: &OsStr) -> io::Result<()> {
    match rename(directory, temp_name, name, RENAME_NOREPLACE) {
        Err(error) if matches!(error.raw_os_error(), Some(libc::EINVAL | libc::ENOSYS)) => {}
        other => return other,
    }
    let (from, to) = (c_string(temp_name)?, c_string(name)?);
    let fd = directory.as_raw_fd();
    // SAFETY: both names are valid NUL-terminated strings; `fd` is open.
    if unsafe { libc::linkat(fd, from.as_ptr(), fd, to.as_ptr(), 0) } != 0 {
        return Err(io::Error::last_os_error());
    }
    remove(directory, temp_name);
    Ok(())
}

fn remove(directory: &OwnedFd, name: &OsStr) {
    if let Ok(name) = c_string(name) {
        // SAFETY: a valid NUL-terminated string and an open descriptor.
        unsafe { libc::unlinkat(directory.as_raw_fd(), name.as_ptr(), 0) };
    }
}
