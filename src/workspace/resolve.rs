//! Path resolution: the one place that decides whether a tool path is inside
//! the workspace.
//!
//! This prevents mistakes; it is not isolation (DESIGN N5). The code a model
//! runs has the owner's full authority anyway. What the rules guarantee is that
//! the *file tools* never read or write outside the workspace by accident, by
//! a confused `..`, or through a symlink someone checked in.
//!
//! The walk goes component by component from a canonical start, the way the
//! kernel would, but under our rules:
//!
//! * A relative path starts at the workspace root. An absolute path that
//!   spells the root is treated as relative to it; any other absolute path is
//!   walked from `/` and accepted only if it ends up inside.
//! * `..` is applied to the real location reached so far. After a `..` of the
//!   caller's own path the location must still be inside the workspace or a
//!   readable root, so no caller path can leave and come back.
//! * A symlink is followed like the kernel follows it. When a symlink that
//!   lives inside the workspace (or a readable root) has been fully followed,
//!   the location must again be inside one of them.
//! * Once a component does not exist, the rest is applied lexically (`..`
//!   removes a missing component); as soon as the walk is back on existing
//!   ground, the filesystem decides again. A missing tail is allowed for
//!   writes (a new file) and is "not found" for reads.
//! * Anything that goes wrong while the walk is outside the allowed areas is
//!   reported as the same refusal, so the answers do not describe the
//!   filesystem outside the workspace.
//!
//! The result is a symlink-free real path, split into the area it lies in and
//! the path beneath that area. `secure_open` opens exactly that, refusing
//! symlinks, so a component swapped for a symlink after resolution is not
//! followed.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::os::unix::ffi::OsStrExt;
use std::path::{Component, Path, PathBuf};

use super::error::{Error, Result, Rule, shown};
use super::lifecycle::Workspace;
use super::limits::{COMPONENT_MAX_BYTES, PATH_MAX_BYTES, SYMLINK_MAX_HOPS};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Area {
    /// Inside the workspace: readable and writable.
    Root,
    /// Inside a read-only extra root.
    ReadRoot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    File,
    Directory,
    /// Does not exist (yet). Only a write resolution returns this.
    Missing,
    /// Exists but is neither a regular file nor a directory.
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    /// The access the path was resolved for. A write primitive accepts only
    /// a `Write` resolution, so the write rules cannot be skipped.
    pub access: Access,
    /// The caller's spelling, made safe to echo in messages.
    pub supplied: String,
    /// The symlink-free absolute location.
    pub real: PathBuf,
    /// The area root `real` lies under, and the path beneath it.
    pub base: PathBuf,
    pub relative: PathBuf,
    pub area: Area,
    pub kind: Kind,
    /// How the path is shown to the model: relative to the workspace root
    /// (`.` for the root itself), or the full path in a read-only root.
    pub display: String,
}

pub fn resolve(workspace: &Workspace, supplied: &str, access: Access) -> Result<Resolved> {
    check_spelling(supplied)?;
    let path = Path::new(supplied);
    let (start, rest, absolute) = match path.strip_prefix(&workspace.path) {
        Ok(rest) => (workspace.path.clone(), rest, false),
        Err(_) if path.is_absolute() => (PathBuf::from("/"), path, true),
        Err(_) => (workspace.path.clone(), path, false),
    };
    let walk = Walk {
        workspace,
        supplied,
        access,
        absolute,
        current: start,
        kind: Kind::Directory,
        missing: Vec::new(),
        queue: tokens(rest).into(),
        hops: 0,
    };
    walk.run()
}

/// Show a real location the way `Resolved::display` does.
pub fn display_of(workspace: &Workspace, real: &Path) -> String {
    match real.strip_prefix(&workspace.path) {
        Ok(relative) if relative.as_os_str().is_empty() => ".".to_string(),
        Ok(relative) => relative.to_string_lossy().into_owned(),
        Err(_) => real.to_string_lossy().into_owned(),
    }
}

fn check_spelling(supplied: &str) -> Result<()> {
    let rule = if supplied.is_empty() {
        Rule::EmptyPath
    } else if supplied.contains('\0') {
        Rule::NulByte
    } else if supplied.len() > PATH_MAX_BYTES {
        Rule::PathTooLong
    } else if Path::new(supplied).components().any(|part| part.as_os_str().len() > COMPONENT_MAX_BYTES) {
        Rule::ComponentTooLong
    } else {
        return Ok(());
    };
    Err(Error::refused(rule, supplied))
}

enum Token {
    Name(OsString),
    Up,
    /// The end of a followed symlink's target. `check` is set when the link
    /// itself lay inside the allowed areas, so its target must too.
    LinkEnd { check: bool },
}

fn tokens(path: &Path) -> Vec<Token> {
    path.components()
        .filter_map(|part| match part {
            Component::Normal(name) => Some(Token::Name(name.to_os_string())),
            Component::ParentDir => Some(Token::Up),
            Component::CurDir | Component::RootDir | Component::Prefix(_) => None,
        })
        .collect()
}

struct Walk<'a> {
    workspace: &'a Workspace,
    supplied: &'a str,
    access: Access,
    /// The caller gave an absolute path that does not spell the root.
    absolute: bool,
    /// Real, existing, symlink-free location reached so far.
    current: PathBuf,
    kind: Kind,
    /// Components below `current` that do not exist.
    missing: Vec<OsString>,
    queue: VecDeque<Token>,
    hops: usize,
}

impl Walk<'_> {
    fn run(mut self) -> Result<Resolved> {
        while let Some(token) = self.queue.pop_front() {
            match token {
                Token::Name(name) => self.name(name)?,
                Token::Up => self.up()?,
                Token::LinkEnd { check } => {
                    if check && !self.allowed(&self.current) {
                        return Err(Error::refused(Rule::SymlinkOutside, self.supplied));
                    }
                }
            }
        }
        self.finish()
    }

    fn allowed(&self, path: &Path) -> bool {
        path.starts_with(&self.workspace.path) || self.workspace.read_roots.iter().any(|root| path.starts_with(root))
    }

    /// Inside a symlink's target (as opposed to the caller's own components).
    fn in_link(&self) -> bool {
        self.queue.iter().any(|token| matches!(token, Token::LinkEnd { .. }))
    }

    /// The one answer for everything that happens outside the allowed areas.
    fn outside(&self) -> Error {
        let rule = if self.in_link() && !self.absolute { Rule::SymlinkOutside } else { Rule::OutsideWorkspace };
        Error::refused(rule, self.supplied)
    }

    fn need_directory(&self) -> Result<()> {
        if self.kind == Kind::Directory {
            Ok(())
        } else if self.allowed(&self.current) {
            Err(Error::WrongKind { path: shown(self.supplied), expected: "a path through directories (one of its components is a file)" })
        } else {
            Err(self.outside())
        }
    }

    fn up(&mut self) -> Result<()> {
        if self.missing.pop().is_some() {
            return Ok(());
        }
        self.need_directory()?;
        let in_link = self.in_link();
        self.current.pop();
        if !in_link && !self.allowed(&self.current) {
            let rule = if self.absolute { Rule::OutsideWorkspace } else { Rule::ParentEscape };
            return Err(Error::refused(rule, self.supplied));
        }
        Ok(())
    }

    fn name(&mut self, name: OsString) -> Result<()> {
        if !self.missing.is_empty() {
            self.missing.push(name);
            return Ok(());
        }
        self.need_directory()?;
        let candidate = self.current.join(&name);
        match std::fs::symlink_metadata(&candidate) {
            Ok(metadata) if metadata.file_type().is_symlink() => self.symlink(&candidate),
            Ok(metadata) => {
                self.kind = if metadata.is_dir() {
                    Kind::Directory
                } else if metadata.is_file() {
                    Kind::File
                } else {
                    Kind::Other
                };
                self.current = candidate;
                Ok(())
            }
            Err(_) if !self.allowed(&self.current) => Err(self.outside()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                self.missing.push(name);
                Ok(())
            }
            Err(error) => Err(Error::io(self.supplied, "resolve", &error)),
        }
    }

    fn symlink(&mut self, link: &Path) -> Result<()> {
        let link_allowed = self.allowed(&self.current);
        let is_final = !self.queue.iter().any(|token| !matches!(token, Token::LinkEnd { .. }));
        if !link_allowed && (self.access == Access::Write && is_final) {
            return Err(self.outside());
        }
        if self.access == Access::Write && is_final {
            return Err(Error::refused(Rule::WriteThroughSymlink, self.supplied));
        }
        self.hops += 1;
        if self.hops > SYMLINK_MAX_HOPS {
            return Err(if link_allowed { Error::refused(Rule::SymlinkLoop, self.supplied) } else { self.outside() });
        }
        let target = match std::fs::read_link(link) {
            Ok(target) => target,
            Err(_) if !link_allowed => return Err(self.outside()),
            Err(error) => return Err(Error::io(self.supplied, "resolve", &error)),
        };
        if target.as_os_str().as_bytes().is_empty() {
            return Err(self.outside());
        }
        self.queue.push_front(Token::LinkEnd { check: link_allowed });
        for token in tokens(&target).into_iter().rev() {
            self.queue.push_front(token);
        }
        if target.is_absolute() {
            self.current = PathBuf::from("/");
        }
        self.kind = Kind::Directory;
        Ok(())
    }

    fn finish(self) -> Result<Resolved> {
        let mut real = self.current.clone();
        real.extend(&self.missing);
        let workspace = self.workspace;
        let (area, base) = if real.starts_with(&workspace.path) {
            (Area::Root, workspace.path.clone())
        } else if let Some(root) = workspace.read_roots.iter().find(|root| real.starts_with(root)) {
            (Area::ReadRoot, root.clone())
        } else {
            return Err(Error::refused(Rule::OutsideWorkspace, self.supplied));
        };
        let relative = real.strip_prefix(&base).map(Path::to_path_buf).unwrap_or_default();
        let kind = if self.missing.is_empty() { self.kind } else { Kind::Missing };
        match self.access {
            Access::Read if kind == Kind::Missing => return Err(Error::NotFound { path: shown(self.supplied) }),
            Access::Read => {}
            Access::Write if area != Area::Root => return Err(Error::refused(Rule::WriteReadOnlyRoot, self.supplied)),
            Access::Write if relative.components().any(|part| part.as_os_str().as_bytes().eq_ignore_ascii_case(b".git")) => {
                return Err(Error::refused(Rule::WriteGitDir, self.supplied));
            }
            Access::Write => {}
        }
        let display = display_of(workspace, &real);
        Ok(Resolved { access: self.access, supplied: shown(self.supplied), real, base, relative, area, kind, display })
    }
}
