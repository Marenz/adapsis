//! Reading a job's output, and the text a model sees about jobs.
//!
//! Output is read by byte offset so a caller can follow a running job:
//! every answer carries `next_offset`, and asking again with it continues
//! exactly there. Without an offset the answer is the end of the output:
//! the tail file when the head cap was reached (the middle is then missing,
//! and the answer says so), otherwise the last bytes of the log.

use std::io::{Read, Seek, SeekFrom};

use super::error::{Error, Result};
use super::job_store::{JobState, now_ms};
use super::jobs::{JobEntry, JobStatus, JobStore};
use super::lifecycle::Workspace;
use super::limits::{RUN_OUTPUT_DEFAULT_BYTES, RUN_OUTPUT_MAX_BYTES, RUN_OUTPUT_MIN_BYTES};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobOutput {
    pub status: JobStatus,
    pub text: String,
    /// Byte range of `output.log` that `text` covers. For an answer from the
    /// tail file both are the log's size.
    pub offset: u64,
    pub next_offset: u64,
    /// The text is the tail file: the end of an output whose middle is lost.
    pub from_tail: bool,
}

impl JobStore {
    /// Read a job's output from `offset`, or its end when no offset is given.
    /// `max_bytes` defaults to `RUN_OUTPUT_DEFAULT_BYTES` and is clamped to
    /// `RUN_OUTPUT_MIN_BYTES..=RUN_OUTPUT_MAX_BYTES`.
    pub fn output(&self, workspace: &Workspace, id: &str, offset: Option<u64>, max_bytes: Option<usize>) -> Result<JobOutput> {
        let directory = self.directory_of(workspace, id)?;
        let status = self.status(workspace, id)?;
        let limit = max_bytes
            .unwrap_or(RUN_OUTPUT_DEFAULT_BYTES)
            .clamp(RUN_OUTPUT_MIN_BYTES, RUN_OUTPUT_MAX_BYTES) as u64;
        let read_error = |error: std::io::Error| Error::Job(format!("the output of job `{id}` cannot be read ({})", error.kind()));
        let finished = status.record.state != JobState::Running;
        match offset {
            None if status.tail_bytes > 0 => {
                let start = status.tail_bytes.saturating_sub(limit);
                let (text, _) = read_text(&directory, "output.tail", start, limit, true).map_err(read_error)?;
                Ok(JobOutput {
                    offset: status.log_bytes,
                    next_offset: status.log_bytes,
                    text,
                    from_tail: true,
                    status,
                })
            }
            _ => {
                let start = offset
                    .unwrap_or_else(|| status.log_bytes.saturating_sub(limit))
                    .min(status.log_bytes);
                let complete = finished && start + limit >= status.log_bytes;
                let (text, consumed) = read_text(&directory, "output.log", start, limit, complete).map_err(read_error)?;
                Ok(JobOutput {
                    offset: start,
                    next_offset: start + consumed,
                    text,
                    from_tail: false,
                    status,
                })
            }
        }
    }
}

/// Read up to `limit` bytes at `start` as text. A read that begins inside a
/// character skips to the next one. A read that ends inside a character
/// leaves it for the next read, unless `complete` says nothing more will
/// come. Returns the text and how many bytes of the file it accounts for.
fn read_text(directory: &super::state_io::Dir, name: &str, start: u64, limit: u64, complete: bool) -> std::io::Result<(String, u64)> {
    let mut file = match directory.file(name) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok((String::new(), 0)),
        Err(error) => return Err(error),
    };
    file.seek(SeekFrom::Start(start))?;
    let mut bytes = Vec::new();
    file.take(limit).read_to_end(&mut bytes)?;
    let skipped = if start > 0 {
        bytes.iter().take(3).take_while(|byte| (**byte & 0xC0) == 0x80).count()
    } else {
        0
    };
    let mut end = bytes.len();
    if !complete {
        if let Err(error) = std::str::from_utf8(&bytes[skipped..]) {
            if error.error_len().is_none() {
                end = skipped + error.valid_up_to();
            }
        }
    }
    Ok((String::from_utf8_lossy(&bytes[skipped..end]).into_owned(), end as u64))
}

fn seconds(milliseconds: u64) -> String {
    let total = milliseconds / 1000;
    match total {
        0..=119 => format!("{total} s"),
        120..=7199 => format!("{} min", total / 60),
        _ => format!("{} h", total / 3600),
    }
}

impl JobStatus {
    /// What happened to the job, in words.
    pub fn describe(&self) -> String {
        let record = &self.record;
        let ran = |until: u64| seconds(until.saturating_sub(record.started_at_ms));
        let ended = record.ended_at_ms.unwrap_or(record.started_at_ms);
        let reason = record
            .end_reason
            .as_deref()
            .map(|reason| format!(" ({reason})"))
            .unwrap_or_default();
        match record.state {
            JobState::Running => format!("running for {} (limit {})", ran(now_ms()), seconds(record.max_runtime_s * 1000)),
            JobState::Exited => format!("exited with status {} after {}", record.exit_status.unwrap_or(-1), ran(ended)),
            JobState::TimedOut => format!("stopped at its runtime limit of {}", seconds(record.max_runtime_s * 1000)),
            JobState::Killed => format!("stopped by job_kill after {}", ran(ended)),
            JobState::Failed => format!("ended without an exit status{reason}"),
            JobState::Lost => "lost: it ended without a trace (it was killed hard, or the user session or machine restarted)".to_string(),
        }
    }

    /// Two lines: what the job is and how it stands, then its command.
    pub fn render(&self) -> String {
        let record = &self.record;
        let command: String = record.command.lines().next().unwrap_or_default().chars().take(160).collect();
        let more = if record.command.lines().count() > 1 || record.command.chars().count() > 160 {
            " …"
        } else {
            ""
        };
        let capped = if self.head_cap_reached() {
            format!(", cap of {} reached", record.head_cap)
        } else {
            String::new()
        };
        format!(
            "job {}: {}; {} bytes of output{capped}; cwd {}\n  $ {command}{more}",
            record.id,
            self.describe(),
            self.log_bytes,
            record.cwd
        )
    }
}

impl JobEntry {
    pub fn render(&self) -> String {
        match self {
            JobEntry::Job(status) => status.render(),
            JobEntry::Unreadable { id, reason } => format!("job {id}: not usable, {reason}; it is left untouched"),
        }
    }
}

/// The job list as text.
pub fn render_jobs(entries: &[JobEntry]) -> String {
    if entries.is_empty() {
        "no jobs in this workspace".to_string()
    } else {
        entries.iter().map(JobEntry::render).collect::<Vec<_>>().join("\n")
    }
}

impl JobOutput {
    /// The output, then where it is from and how to go on.
    pub fn render(&self) -> String {
        let record = &self.status.record;
        let mut out = self.text.trim_end_matches('\n').to_string();
        if !out.is_empty() {
            out.push('\n');
        }
        let state = self.status.describe();
        let running = record.state == JobState::Running;
        if self.from_tail {
            out.push_str(&format!(
                "[job {}: {state}. Its output exceeded {} bytes: this is the end of it, the middle is missing; the beginning is at offset=0]",
                record.id, record.head_cap
            ));
        } else if self.next_offset < self.status.log_bytes {
            out.push_str(&format!(
                "[job {}: {state}; output bytes {}-{} of {}; continue with offset={}]",
                record.id, self.offset, self.next_offset, self.status.log_bytes, self.next_offset
            ));
        } else if running && self.status.head_cap_reached() {
            out.push_str(&format!(
                "[job {}: {state}; head output cap of {} bytes reached. Ask without an offset after it ends for a best-effort tail; the tail is lost on a hard kill]",
                record.id, record.head_cap
            ));
        } else if running {
            out.push_str(&format!(
                "[job {}: {state}; output up to byte {}; ask again with offset={} for what follows]",
                record.id, self.next_offset, self.next_offset
            ));
        } else {
            let lost = if self.status.head_cap_reached() {
                "; later output was dropped at the cap (without an offset you get the end, if it was kept)"
            } else {
                ""
            };
            out.push_str(&format!(
                "[job {}: {state}; end of output at byte {}{lost}]",
                record.id, self.next_offset
            ));
        }
        out
    }
}

/// Read a job's output from the process's own store.
pub fn job_output(workspace: &Workspace, id: &str, offset: Option<u64>, max_bytes: Option<usize>) -> Result<JobOutput> {
    JobStore::standard().output(workspace, id, offset, max_bytes)
}
