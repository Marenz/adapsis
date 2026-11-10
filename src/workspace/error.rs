//! One error type for the workspace tools.
//!
//! Every message is written for the model that made the call: it names what
//! was refused and what to do instead. A message never contains more of a path
//! outside the workspace than the caller supplied: refusals echo the caller's
//! own spelling (shortened), never a resolved location or a symlink target.

use std::fmt;

use super::limits::ECHO_MAX_CHARS;

/// The path rule a refused path broke.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    EmptyPath,
    NulByte,
    PathTooLong,
    ComponentTooLong,
    /// The path names a location outside the workspace and its readable roots.
    OutsideWorkspace,
    /// A `..` in the path climbs out of the workspace.
    ParentEscape,
    /// A symlink on the path leads outside the workspace and its readable roots.
    SymlinkOutside,
    SymlinkLoop,
    /// A component turned into a symlink between resolving and opening.
    ChangedToSymlink,
    /// Write to a path whose final component is a symlink.
    WriteThroughSymlink,
    /// Write to git's own files.
    WriteGitDir,
    /// Write into a read-only extra root.
    WriteReadOnlyRoot,
}

impl Rule {
    pub fn describe(self) -> &'static str {
        match self {
            Rule::EmptyPath => "the path is empty",
            Rule::NulByte => "the path contains a NUL byte",
            Rule::PathTooLong => "the path is longer than 4096 bytes",
            Rule::ComponentTooLong => "a path component is longer than 255 bytes",
            Rule::OutsideWorkspace => "it is outside the workspace; use a path relative to the workspace root",
            Rule::ParentEscape => "its `..` leaves the workspace",
            Rule::SymlinkOutside => "a symlink on it leads outside the workspace and its readable roots",
            Rule::SymlinkLoop => "it passes through too many symlinks",
            Rule::ChangedToSymlink => "a component became a symlink while it was being opened; resolve it again",
            Rule::WriteThroughSymlink => "writes never go through a symlink as the final component; write to the real file",
            Rule::WriteGitDir => "writes under `.git` are not allowed; use the git tools",
            Rule::WriteReadOnlyRoot => "it is in a read-only extra root; only the workspace itself is writable",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The registry file is unreadable or invalid. The text lists every problem.
    Registry(String),
    /// A repository, workspace or ref name is not acceptable.
    InvalidName { what: &'static str, reason: String },
    InvalidArgument(String),
    InvalidPattern(String),
    /// A path was refused by a path rule. `path` is the caller's own spelling.
    Refused { rule: Rule, path: String },
    NotFound { path: String },
    /// The path exists but is not what the operation needs.
    WrongKind { path: String, expected: &'static str },
    Binary { path: String, size: u64 },
    /// A filesystem operation failed. `reason` is the error kind, never a path.
    Io { path: String, action: &'static str, reason: String },
    /// git exited with an error, or could not be started.
    Git { action: String, detail: String },
    GitTimeout { action: String, millis: u128 },
    /// A lifecycle rule refused the operation (exists, limit, unknown name).
    Lifecycle(String),
    /// `close` refused because the worktree has uncommitted changes.
    Dirty { workspace: String, files: Vec<String>, omitted: usize },
}

impl Error {
    pub fn refused(rule: Rule, supplied: &str) -> Self {
        Error::Refused { rule, path: shown(supplied) }
    }

    pub fn io(supplied: &str, action: &'static str, error: &std::io::Error) -> Self {
        if error.kind() == std::io::ErrorKind::NotFound {
            return Error::NotFound { path: shown(supplied) };
        }
        Error::Io { path: shown(supplied), action, reason: error.kind().to_string() }
    }

    /// The rule of a refusal, for callers and tests that branch on it.
    pub fn rule(&self) -> Option<Rule> {
        match self {
            Error::Refused { rule, .. } => Some(*rule),
            _ => None,
        }
    }
}

/// A caller-supplied string made safe to echo: bounded, no control characters.
pub fn shown(text: &str) -> String {
    let mut out: String = text
        .chars()
        .take(ECHO_MAX_CHARS)
        .map(|c| if c.is_control() { '\u{FFFD}' } else { c })
        .collect();
    if text.chars().nth(ECHO_MAX_CHARS).is_some() {
        out.push('…');
    }
    out
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Registry(text) => write!(f, "workspace registry: {text}"),
            Error::InvalidName { what, reason } => write!(f, "invalid {what}: {reason}"),
            Error::InvalidArgument(text) => write!(f, "invalid argument: {text}"),
            Error::InvalidPattern(text) => write!(f, "invalid pattern: {text}"),
            Error::Refused { rule, path } => write!(f, "path `{path}` refused: {}", rule.describe()),
            Error::NotFound { path } => write!(f, "`{path}` does not exist in the workspace"),
            Error::WrongKind { path, expected } => write!(f, "`{path}` is not {expected}"),
            Error::Binary { path, size } => {
                write!(f, "`{path}` is a binary file ({size} bytes); it cannot be shown as text")
            }
            Error::Io { path, action, reason } => write!(f, "cannot {action} `{path}`: {reason}"),
            Error::Git { action, detail } => write!(f, "git {action} failed: {detail}"),
            Error::GitTimeout { action, millis } => {
                write!(f, "git {action} did not finish within {millis} ms and was stopped")
            }
            Error::Lifecycle(text) => f.write_str(text),
            Error::Dirty { workspace, files, omitted } => {
                write!(f, "workspace `{workspace}` has uncommitted changes and was not closed; commit or discard them first:")?;
                for file in files {
                    write!(f, "\n  {file}")?;
                }
                if *omitted > 0 {
                    write!(f, "\n  … and {omitted} more")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
