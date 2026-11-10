//! `search`: a regular expression over the text files of a tree.
//!
//! Matching is line by line (a pattern never spans lines), on bytes, with the
//! regex crate, whose matching time is linear in the input. The remaining
//! risks are bounded here: pattern length and compiled size, the searched
//! prefix of each line, the output size, and a wall-clock budget.

use std::collections::VecDeque;
use std::io::{self, Cursor, Read};
use std::path::Path;

use regex::bytes::{Regex, RegexBuilder};

use super::error::{Error, Result};
use super::lifecycle::Workspace;
use super::limits::{
    BINARY_SNIFF_BYTES, PATTERN_MAX_BYTES, REGEX_SIZE_LIMIT, SEARCH_DEFAULT_LIMIT, SEARCH_LINE_MAX_BYTES, SEARCH_LINE_SHOWN_CHARS,
    SEARCH_MAX_CONTEXT, SEARCH_MAX_LIMIT, SEARCH_MAX_OUTPUT_BYTES,
};
use super::resolve::{Access, Kind, Resolved, display_of, resolve};
use super::scan::{LineInfo, LineScanner, decode_line, looks_binary, read_head, strip_carriage_return};
use super::secure_open::{Strategy, open_file_beneath};
use super::walk::{Budget, GlobFilter, Stop, walker};

/// Output bytes kept free for the summary lines after the matches.
const TRAILER_RESERVE: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Matching lines: `path:line: text`.
    Content,
    /// Paths of files with at least one match.
    Files,
    /// Matching-line count per file: `path: N`.
    Count,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchRequest {
    pub pattern: String,
    /// File or directory to search; `.` is the workspace root.
    pub path: String,
    /// Only files matching this glob (see `walk::GlobFilter`).
    pub glob: Option<String>,
    pub mode: SearchMode,
    /// Lines shown before and after each match (content mode).
    pub context: usize,
    pub ignore_case: bool,
    /// Most matches (content) or files (files, count) returned.
    pub limit: usize,
}

impl SearchRequest {
    pub fn new(pattern: &str) -> Self {
        Self {
            pattern: pattern.to_string(),
            path: ".".to_string(),
            glob: None,
            mode: SearchMode::Content,
            context: 0,
            ignore_case: false,
            limit: SEARCH_DEFAULT_LIMIT,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchLine {
    Match { path: String, line: u64, text: String },
    Context { path: String, line: u64, text: String },
    /// Between two groups of lines that are not adjacent.
    Separator,
    File { path: String },
    Count { path: String, count: u64 },
}

impl SearchLine {
    fn render(&self) -> String {
        match self {
            SearchLine::Match { path, line, text } => format!("{path}:{line}: {text}"),
            SearchLine::Context { path, line, text } => format!("{path}-{line}- {text}"),
            SearchLine::Separator => "--".to_string(),
            SearchLine::File { path } => path.clone(),
            SearchLine::Count { path, count } => format!("{path}: {count}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchResult {
    /// Files whose scan reached the byte ceiling before observing EOF.
    pub scan_capped_files: u64,
    pub mode: SearchMode,
    pub lines: Vec<SearchLine>,
    /// Matches (content) or files (files, count) in `lines`.
    pub shown: usize,
    /// Further matches or files that were found but are not in `lines`.
    pub omitted: u64,
    /// The effective limit after clamping.
    pub limit: usize,
    /// The output size cap, not the limit, ended the listing.
    pub output_capped: bool,
    pub files_searched: u64,
    pub binary_skipped: u64,
    pub unreadable: u64,
    /// Lines longer than the searched prefix.
    pub long_lines: u64,
    /// Set when a budget ended the search; `omitted` is then a lower bound.
    pub stopped: Option<Stop>,
}

pub fn search(workspace: &Workspace, request: &SearchRequest) -> Result<SearchResult> {
    let regex = compile(&request.pattern, request.ignore_case)?;
    let path = if request.path.is_empty() { "." } else { &request.path };
    let start = resolve(workspace, path, Access::Read)?;
    let mut engine = Engine {
        workspace,
        regex,
        context: request.context.min(SEARCH_MAX_CONTEXT),
        output_bytes: 0,
        budget: Budget::new(),
        result: SearchResult {
            scan_capped_files: 0,
            mode: request.mode,
            lines: Vec::new(),
            shown: 0,
            omitted: 0,
            limit: request.limit.clamp(1, SEARCH_MAX_LIMIT),
            output_capped: false,
            files_searched: 0,
            binary_skipped: 0,
            unreadable: 0,
            long_lines: 0,
            stopped: None,
        },
    };
    let anchor = if start.kind == Kind::Directory { start.real.as_path() } else { start.base.as_path() };
    let filter = request.glob.as_deref().map(|glob| GlobFilter::new(anchor, glob)).transpose()?;
    match start.kind {
        Kind::Directory => engine.tree(&start, filter.as_ref()),
        Kind::File => engine.file(&start.base, &start.relative, &start.display),
        Kind::Missing | Kind::Other => {
            return Err(Error::WrongKind { path: start.supplied, expected: "a file or directory" });
        }
    }
    engine.result.stopped = engine.budget.stopped;
    Ok(engine.result)
}

fn compile(pattern: &str, ignore_case: bool) -> Result<Regex> {
    if pattern.is_empty() {
        return Err(Error::InvalidPattern("the search pattern is empty".to_string()));
    }
    if pattern.len() > PATTERN_MAX_BYTES {
        return Err(Error::InvalidPattern(format!(
            "the search pattern is {} bytes; the limit is {PATTERN_MAX_BYTES}",
            pattern.len()
        )));
    }
    RegexBuilder::new(pattern)
        .case_insensitive(ignore_case)
        .size_limit(REGEX_SIZE_LIMIT)
        .dfa_size_limit(REGEX_SIZE_LIMIT)
        .build()
        .map_err(|error| {
            let text: String = error.to_string().chars().take(600).collect();
            Error::InvalidPattern(format!("not a valid regular expression (Rust regex syntax; escape literal `(`, `{{`, `.`):\n{text}"))
        })
}

struct Engine<'a> {
    workspace: &'a Workspace,
    regex: Regex,
    context: usize,
    output_bytes: usize,
    budget: Budget,
    result: SearchResult,
}

/// Per-file state of a content search.
#[derive(Default)]
struct Group {
    /// The last lines that were neither matches nor shown as context.
    before: VecDeque<(u64, String)>,
    /// Context lines still owed after the last match.
    after: usize,
    /// Number of the last line of this file that was shown; 0 for none.
    last_shown: u64,
}

impl Engine<'_> {
    fn tree(&mut self, start: &Resolved, filter: Option<&GlobFilter>) {
        for entry in walker(&start.real) {
            if !self.budget.entry() {
                break;
            }
            let Ok(entry) = entry else {
                self.result.unreadable += 1;
                continue;
            };
            if !entry.file_type().is_some_and(|kind| kind.is_file()) {
                continue;
            }
            if filter.is_some_and(|filter| !filter.matches(entry.path())) {
                continue;
            }
            let Ok(relative) = entry.path().strip_prefix(&start.base) else { continue };
            let display = display_of(self.workspace, entry.path());
            self.file(&start.base, relative, &display);
        }
    }

    /// Search one file. It is opened beneath its area root without following
    /// symlinks, so a file swapped for a link during the walk is skipped.
    fn file(&mut self, base: &Path, relative: &Path, display: &str) {
        let Ok(mut file) = open_file_beneath(base, relative, Strategy::Auto) else {
            self.result.unreadable += 1;
            return;
        };
        if !file.metadata().is_ok_and(|metadata| metadata.is_file()) {
            return;
        }
        let Ok(head) = read_head(&mut file, BINARY_SNIFF_BYTES) else {
            self.result.unreadable += 1;
            return;
        };
        if looks_binary(&head) {
            self.result.binary_skipped += 1;
            return;
        }
        self.result.files_searched += 1;
        let bounded = super::scan::Ceiling::new(Cursor::new(head).chain(file), super::limits::SEARCH_SCAN_MAX_BYTES);
        let hit = bounded.hit.clone();
        let scanner = LineScanner::new(bounded);
        let outcome = match self.result.mode {
            SearchMode::Content => self.content(scanner, display),
            SearchMode::Files | SearchMode::Count => self.tally(scanner, display),
        };
        if outcome.is_err() {
            self.result.unreadable += 1;
        }
        if hit.load(std::sync::atomic::Ordering::Relaxed) { self.result.scan_capped_files += 1; }
    }

    /// Normalize a scanned line for matching: note an over-long line, drop
    /// the `\r` of a CRLF ending. Returns whether `kept` is the whole line.
    fn prepare(&mut self, kept: &mut Vec<u8>, info: LineInfo) -> bool {
        let whole = info.len == kept.len() as u64;
        if whole {
            strip_carriage_return(kept, info);
        } else {
            self.result.long_lines += 1;
        }
        whole
    }

    fn accepting(&self) -> bool {
        self.result.shown < self.result.limit && !self.result.output_capped
    }

    /// Add a line if the output cap allows it.
    fn push(&mut self, line: SearchLine) -> bool {
        let cost = line.render().len() + 1;
        if self.output_bytes + cost > SEARCH_MAX_OUTPUT_BYTES - TRAILER_RESERVE {
            self.result.output_capped = true;
            return false;
        }
        self.output_bytes += cost;
        self.result.lines.push(line);
        true
    }

    fn tally<R: Read>(&mut self, mut scanner: LineScanner<R>, display: &str) -> io::Result<()> {
        let files_only = self.result.mode == SearchMode::Files;
        let mut kept = Vec::new();
        let (mut number, mut count) = (0u64, 0u64);
        while let Some(info) = scanner.next_line(&mut kept, SEARCH_LINE_MAX_BYTES)? {
            number += 1;
            if number % 1024 == 0 && !self.budget.running() {
                break;
            }
            self.prepare(&mut kept, info);
            if self.regex.is_match(&kept) {
                count += 1;
                if files_only {
                    break;
                }
            }
        }
        if count > 0 {
            let path = display.to_string();
            let line = if files_only { SearchLine::File { path } } else { SearchLine::Count { path, count } };
            if self.accepting() && self.push(line) {
                self.result.shown += 1;
            } else {
                self.result.omitted += 1;
            }
        }
        Ok(())
    }

    fn content<R: Read>(&mut self, mut scanner: LineScanner<R>, display: &str) -> io::Result<()> {
        let mut group = Group::default();
        let mut kept = Vec::new();
        let mut number = 0u64;
        while let Some(info) = scanner.next_line(&mut kept, SEARCH_LINE_MAX_BYTES)? {
            number += 1;
            if number % 1024 == 0 && !self.budget.running() {
                break;
            }
            let whole = self.prepare(&mut kept, info);
            if self.regex.is_match(&kept) {
                if self.accepting() {
                    self.show_match(display, number, shown_text(&kept, whole), &mut group);
                } else {
                    self.result.omitted += 1;
                }
            } else if group.after > 0 && !self.result.output_capped {
                group.after -= 1;
                if self.push(SearchLine::Context { path: display.to_string(), line: number, text: shown_text(&kept, whole) }) {
                    group.last_shown = number;
                }
            } else if self.context > 0 && self.accepting() {
                group.before.push_back((number, shown_text(&kept, whole)));
                if group.before.len() > self.context {
                    group.before.pop_front();
                }
            }
        }
        Ok(())
    }

    /// Show one match with its leading context, separated from the previous
    /// group when the two are not adjacent.
    fn show_match(&mut self, display: &str, number: u64, text: String, group: &mut Group) {
        let first = group.before.front().map_or(number, |(line, _)| *line);
        if self.context > 0 && !self.result.lines.is_empty() && (group.last_shown == 0 || first > group.last_shown + 1) {
            self.push(SearchLine::Separator);
        }
        for (line, text) in group.before.drain(..) {
            self.push(SearchLine::Context { path: display.to_string(), line, text });
        }
        if self.push(SearchLine::Match { path: display.to_string(), line: number, text }) {
            self.result.shown += 1;
            group.last_shown = number;
            group.after = self.context;
        } else {
            self.result.omitted += 1;
        }
    }
}

fn shown_text(kept: &[u8], whole: bool) -> String {
    let decoded = decode_line(kept, whole, SEARCH_LINE_SHOWN_CHARS);
    if decoded.cut { format!("{} …[cut]", decoded.text) } else { decoded.text }
}

impl SearchResult {
    /// The text the model sees: one line per match, then what was left out.
    pub fn render(&self) -> String {
        let mut out: Vec<String> = self.lines.iter().map(SearchLine::render).collect();
        let unit = if self.mode == SearchMode::Content { "matches" } else { "files" };
        if self.scan_capped_files > 0 {
            out.push(format!("[partial search: {} file(s) reached the {} byte scan ceiling; later content was not searched]", self.scan_capped_files, super::limits::SEARCH_SCAN_MAX_BYTES));
        }
        if self.shown == 0 && self.omitted == 0 {
            out.push(format!("no {unit} found ({} text files searched)", self.files_searched));
        }
        if self.omitted > 0 {
            let bound = if self.stopped.is_some() { "at least " } else { "" };
            let cause = if self.output_capped { "output size cap".to_string() } else { format!("limit {}", self.limit) };
            out.push(format!("[{bound}{} more {unit} not shown ({cause}); narrow the pattern or the path]", self.omitted));
        }
        if let Some(stop) = self.stopped {
            out.push(format!("[note: {}]", stop.describe()));
        }
        if self.long_lines > 0 {
            out.push(format!(
                "[note: {} line(s) longer than {SEARCH_LINE_MAX_BYTES} bytes were searched only up to that length]",
                self.long_lines
            ));
        }
        if self.binary_skipped > 0 {
            out.push(format!("[note: {} binary file(s) skipped]", self.binary_skipped));
        }
        if self.unreadable > 0 {
            out.push(format!("[note: {} file(s) or directories could not be read]", self.unreadable));
        }
        out.join("\n")
    }
}
