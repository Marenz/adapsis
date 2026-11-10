//! Every bound of the workspace tools, in one place.
//!
//! These are POLICY VALUES, not derived facts: they were chosen so that one tool
//! result is large enough for a source file, a search result page or a file
//! listing, and small enough for a model context. Change them here, not at the
//! use site, and keep the tests that pin the behaviour at each bound.

use std::time::Duration;

// --- registry -------------------------------------------------------------

/// Largest registry file that is read at all.
pub const REGISTRY_MAX_BYTES: u64 = 256 * 1024;
/// Most repositories one registry may declare.
pub const REGISTRY_MAX_REPOSITORIES: usize = 64;
/// Most read-only extra roots per repository.
pub const READ_ROOTS_MAX: usize = 8;

// --- lifecycle ------------------------------------------------------------

/// Adapsis-managed worktrees per repository. Disk is tight on the machines
/// this runs on, and every worktree of this repository is a full checkout
/// plus, in practice, a build directory.
pub const MAX_WORKTREES_PER_REPO: usize = 4;
/// Longest repository or workspace name.
pub const NAME_MAX_LEN: usize = 48;
/// Longest base ref a caller may supply.
pub const BASE_REF_MAX_LEN: usize = 200;
/// Dirty files listed by a refused `close` before "and N more".
pub const DIRTY_LIST_MAX: usize = 50;

// --- git ------------------------------------------------------------------

/// Timeout of a git query (status, rev-parse, config, worktree list).
pub const GIT_TIMEOUT: Duration = Duration::from_secs(30);
/// Timeout of `git worktree add` / `remove`, which check out or delete a tree.
pub const GIT_WORKTREE_TIMEOUT: Duration = Duration::from_secs(180);
/// Captured bytes per git output stream; the rest is drained and dropped.
pub const GIT_OUTPUT_MAX_BYTES: usize = 1024 * 1024;
/// How long to wait for the output pipes after git itself has exited (a hook
/// may have left a background process holding them).
pub const GIT_PIPE_GRACE: Duration = Duration::from_secs(2);

// --- paths ----------------------------------------------------------------

/// Longest path a caller may supply, in bytes.
pub const PATH_MAX_BYTES: usize = 4096;
/// Longest single path component, in bytes (the common filesystem limit).
pub const COMPONENT_MAX_BYTES: usize = 255;
/// Symlinks followed while resolving one path.
pub const SYMLINK_MAX_HOPS: usize = 40;
/// Characters of a caller-supplied path echoed back in a message.
pub const ECHO_MAX_CHARS: usize = 200;

// --- read -----------------------------------------------------------------

/// Lines returned when the caller gives no limit, and the most per call.
pub const READ_MAX_LINES: usize = 2000;
/// Bytes of line text per call.
pub const READ_MAX_BYTES: usize = 100 * 1024;
/// Characters of one line before it is cut with a marker.
pub const LINE_MAX_CHARS: usize = 2000;
/// A NUL byte within this prefix makes a file binary.
pub const BINARY_SNIFF_BYTES: usize = 8 * 1024;
/// A file is scanned (hashed, lines counted) up to this size, which bounds the
/// time of one read. Of a larger file only this prefix can be read, and it
/// gets no content hash, so it cannot be proven unchanged for an edit.
pub const SCAN_MAX_BYTES: u64 = 64 * 1024 * 1024;
/// Files remembered by one read ledger; the oldest record is dropped.
pub const LEDGER_MAX_ENTRIES: usize = 512;

// --- write and edit ------------------------------------------------------

/// Largest content one `write` accepts. The content travels through a model's
/// output, so anything near this size is a runaway, not a source file.
pub const WRITE_MAX_BYTES: usize = 2 * 1024 * 1024;
/// Largest file `edit` loads (it holds the old and the new content in memory).
pub const EDIT_MAX_FILE_BYTES: u64 = 16 * 1024 * 1024;
/// Line numbers listed when the text to replace is ambiguous.
pub const AMBIGUOUS_LINES_SHOWN: usize = 10;

// --- run ------------------------------------------------------------------

/// Timeout of a command that gives none. Long enough for git, a linter or an
/// incremental `cargo check`; short enough that a command waiting for input
/// that never comes costs two minutes, not a turn.
pub const RUN_TIMEOUT_DEFAULT: Duration = Duration::from_secs(120);
/// Longest timeout a caller may ask for. A cold release build plus the full
/// test suite of this repository takes about a quarter of an hour; an hour
/// leaves room for a slower machine and nothing runs unattended for longer.
pub const RUN_TIMEOUT_MAX: Duration = Duration::from_secs(3600);
/// Shortest timeout; a smaller request is raised to it.
pub const RUN_TIMEOUT_MIN: Duration = Duration::from_millis(100);
/// Captured output when the caller gives no limit: about 8,000 tokens, enough
/// for a failing test with its backtrace and the summary.
pub const RUN_OUTPUT_DEFAULT_BYTES: usize = 32 * 1024;
/// Largest output a caller may ask for (a full test log); beyond it the
/// command should write to a file and the model should `read` or `search` it.
pub const RUN_OUTPUT_MAX_BYTES: usize = 256 * 1024;
/// Smallest output limit; a smaller request is raised to it.
pub const RUN_OUTPUT_MIN_BYTES: usize = 1024;
/// Share of the output limit kept from the start; the rest is kept from the
/// end. The end carries the verdict (test summary, last error), the start the
/// first error of a build, so the end gets the larger share.
pub const RUN_OUTPUT_HEAD_PERCENT: usize = 40;
/// Time between SIGTERM and SIGKILL when a command is stopped.
pub const RUN_KILL_GRACE: Duration = Duration::from_secs(2);
/// How long to wait for the output pipe to close after the command exited
/// (a background child may hold it), before everything left is stopped.
pub const RUN_PIPE_GRACE: Duration = Duration::from_millis(500);
/// How long to wait for the reader to drain the pipe once every process of
/// the group is gone. Reached only if a process left the group (setsid) and
/// kept the pipe open.
pub const RUN_DRAIN_GRACE: Duration = Duration::from_secs(5);
/// Longest command line, in bytes.
pub const RUN_COMMAND_MAX_BYTES: usize = 32 * 1024;
/// Most environment additions of one request, and their total size.
pub const RUN_ENV_MAX_VARS: usize = 64;
pub const RUN_ENV_MAX_BYTES: usize = 32 * 1024;

// --- search ---------------------------------------------------------------

/// Matches returned when the caller gives no limit.
pub const SEARCH_DEFAULT_LIMIT: usize = 200;
/// Largest limit a caller may ask for.
pub const SEARCH_MAX_LIMIT: usize = 2000;
/// Bytes of rendered search output.
pub const SEARCH_MAX_OUTPUT_BYTES: usize = 50 * 1024;
/// Longest pattern, in bytes.
pub const PATTERN_MAX_BYTES: usize = 1000;
/// Compiled size of a pattern (the regex crate's own guard).
pub const REGEX_SIZE_LIMIT: usize = 4 * 1024 * 1024;
/// Bytes of one line that are searched; the rest of a longer line is skipped.
pub const SEARCH_LINE_MAX_BYTES: usize = 64 * 1024;
/// Characters of one matching or context line that are shown.
pub const SEARCH_LINE_SHOWN_CHARS: usize = 500;
/// Most context lines on each side of a match.
pub const SEARCH_MAX_CONTEXT: usize = 20;
/// Wall-clock budget of one search or glob, after which it reports what it has.
pub const WALK_TIME_BUDGET: Duration = Duration::from_secs(10);
/// Directory entries visited by one search or glob.
pub const WALK_MAX_ENTRIES: u64 = 250_000;

// --- glob -----------------------------------------------------------------

/// Paths returned by one glob.
pub const GLOB_MAX_RESULTS: usize = 1000;
/// Bytes of rendered glob output (paths can be long).
pub const GLOB_MAX_OUTPUT_BYTES: usize = 50 * 1024;
