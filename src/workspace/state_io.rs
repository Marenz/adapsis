//! Private job state, addressed through held directory descriptors. Only the
//! configured root's ancestors are trusted; every component below it is opened
//! with O_NOFOLLOW. Enumeration and removal fail closed on every error.
use std::ffi::{OsStr, OsString};
use std::fs::{File, Metadata};
use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::secure_open::{Strategy, c_string, open_entry};

const DIR_FLAGS: i32 = libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC;
static TEMP: AtomicU64 = AtomicU64::new(0);

pub struct Dir {
    fd: OwnedFd,
}

impl Dir {
    pub fn root(path: &Path, create: bool) -> io::Result<Option<Self>> {
        if !path.is_absolute() {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        if create {
            use std::os::unix::fs::DirBuilderExt;
            std::fs::DirBuilder::new().recursive(true).mode(0o700).create(path)?;
        }
        use std::os::unix::fs::OpenOptionsExt;
        match std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path)
        {
            Ok(file) => Ok(Some(Self { fd: file.into() })),
            Err(e) if !create && e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    pub fn child(&self, name: &OsStr, create: bool) -> io::Result<Option<Self>> {
        match open_entry(&self.fd, name, DIR_FLAGS, 0, Strategy::Walk) {
            Ok(fd) => return Ok(Some(Self { fd })),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        if !create {
            return Ok(None);
        }
        let name_c = c_string(name)?;
        // SAFETY: held directory and a checked single-component C string.
        if unsafe { libc::mkdirat(self.fd.as_raw_fd(), name_c.as_ptr(), 0o700) } != 0 {
            let e = io::Error::last_os_error();
            if e.kind() != io::ErrorKind::AlreadyExists {
                return Err(e);
            }
        }
        open_entry(&self.fd, name, DIR_FLAGS, 0, Strategy::Walk).map(|fd| Some(Self { fd }))
    }

    pub fn create_child(&self, name: &str) -> io::Result<Self> {
        let name_c = c_string(OsStr::new(name))?;
        // SAFETY: held directory and checked C string.
        if unsafe { libc::mkdirat(self.fd.as_raw_fd(), name_c.as_ptr(), 0o700) } != 0 {
            return Err(io::Error::last_os_error());
        }
        self.child(OsStr::new(name), false)?
            .ok_or_else(|| io::Error::from(io::ErrorKind::NotFound))
    }

    pub fn list(&self) -> io::Result<Vec<OsString>> {
        collect_names(std::fs::read_dir(self.proc_path())?.map(|entry| entry.map(|e| e.file_name())))
    }

    pub fn proc_path(&self) -> PathBuf {
        PathBuf::from(format!("/proc/self/fd/{}", self.fd.as_raw_fd()))
    }

    pub fn file(&self, name: &str) -> io::Result<File> {
        let fd = open_entry(
            &self.fd,
            OsStr::new(name),
            libc::O_RDONLY | libc::O_NONBLOCK | libc::O_CLOEXEC,
            0,
            Strategy::Walk,
        )?;
        let file = File::from(fd);
        if !file.metadata()?.is_file() {
            return Err(io::Error::from(io::ErrorKind::InvalidData));
        }
        Ok(file)
    }

    pub fn metadata(&self, name: &OsStr) -> io::Result<Option<Metadata>> {
        match open_entry(&self.fd, name, libc::O_PATH | libc::O_CLOEXEC | libc::O_NOFOLLOW, 0, Strategy::Walk) {
            Ok(fd) => {
                let meta = File::from(fd).metadata()?;
                if meta.file_type().is_symlink() {
                    return Err(io::Error::from_raw_os_error(libc::ELOOP));
                }
                Ok(Some(meta))
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    pub fn read(&self, name: &str, max: u64) -> io::Result<Option<Vec<u8>>> {
        let file = match self.file(name) {
            Ok(file) => file,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e),
        };
        let mut bytes = Vec::new();
        file.take(max + 1).read_to_end(&mut bytes)?;
        if bytes.len() as u64 > max {
            return Err(io::Error::from(io::ErrorKind::InvalidData));
        }
        Ok(Some(bytes))
    }

    pub fn write(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        self.metadata(OsStr::new(name))?;
        let temporary = format!(".tmp-{}-{}", std::process::id(), TEMP.fetch_add(1, Ordering::Relaxed));
        let fd = open_entry(
            &self.fd,
            OsStr::new(&temporary),
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC,
            0o600,
            Strategy::Walk,
        )?;
        let result = (|| {
            let mut file = File::from(fd);
            file.write_all(bytes)?;
            file.sync_all()?;
            self.metadata(OsStr::new(name))?;
            let (from, to) = (c_string(OsStr::new(&temporary))?, c_string(OsStr::new(name))?);
            // SAFETY: both names are single checked components of our held dir.
            if unsafe { libc::renameat(self.fd.as_raw_fd(), from.as_ptr(), self.fd.as_raw_fd(), to.as_ptr()) } != 0 {
                return Err(io::Error::last_os_error());
            }
            File::from(self.fd.try_clone()?).sync_all()
        })();
        if result.is_err() {
            let _ = self.unlink(OsStr::new(&temporary), false);
        }
        result
    }

    pub fn unlink(&self, name: &OsStr, directory: bool) -> io::Result<()> {
        let name = c_string(name)?;
        // SAFETY: name is a checked component below our held directory.
        if unsafe { libc::unlinkat(self.fd.as_raw_fd(), name.as_ptr(), if directory { libc::AT_REMOVEDIR } else { 0 }) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    pub fn remove_tree(&self, name: &OsStr) -> io::Result<()> {
        let Some(meta) = self.metadata(name)? else { return Ok(()) };
        if !meta.is_dir() {
            return self.unlink(name, false);
        }
        let child = self.child(name, false)?.ok_or_else(|| io::Error::from(io::ErrorKind::NotFound))?;
        child.validate_tree().map_err(|_| io::Error::from_raw_os_error(libc::ELOOP))?;
        for name in child.list()? {
            child.remove_tree(&name)?;
        }
        self.unlink(name, true)
    }

    pub fn validate_tree(&self) -> super::error::Result<()> {
        self.validate_depth(0)
    }

    fn validate_depth(&self, depth: usize) -> super::error::Result<()> {
        if depth >= super::limits::JOB_STATE_MAX_DEPTH {
            return Err(super::error::Error::Job(
                "job state exceeds 64 directory levels; inspection refused".into(),
            ));
        }
        for name in self.list().map_err(super::job_store::state_error)? {
            if let Some(meta) = self.metadata(&name).map_err(super::job_store::state_error)? {
                if meta.is_dir() {
                    let child = self
                        .child(&name, false)
                        .map_err(super::job_store::state_error)?
                        .ok_or_else(|| super::error::Error::Job("job state directory vanished".into()))?;
                    child.validate_depth(depth + 1)?;
                } else if !meta.is_file() {
                    return Err(super::error::Error::Job("job state contains a non-regular file".into()));
                }
            }
        }
        Ok(())
    }
}

pub(super) fn collect_names(entries: impl Iterator<Item = io::Result<OsString>>) -> io::Result<Vec<OsString>> {
    let mut names = Vec::new();
    for entry in entries {
        if names.len() >= super::limits::JOB_STATE_MAX_ENTRIES {
            return Err(io::Error::from(io::ErrorKind::InvalidData));
        }
        names.push(entry?);
    }
    Ok(names)
}
