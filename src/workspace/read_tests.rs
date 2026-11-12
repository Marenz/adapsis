use std::time::{Duration, Instant, SystemTime};

use sha2::{Digest, Sha256};

use super::error::{Error, Rule};
use super::ledger::{Freshness, ReadLedger};
use super::limits::{BINARY_SNIFF_BYTES, LEDGER_MAX_ENTRIES, LINE_MAX_CHARS, READ_MAX_BYTES, READ_MAX_LINES, SCAN_MAX_BYTES};
use super::read::{FileFacts, ReadResult, read, read_within};
use super::resolve::{Access, resolve};
use super::test_support::{Fixture, symlink, write};

fn numbered(count: usize) -> String {
    (1..=count).map(|number| format!("line {number}\n")).collect()
}

fn sha256_hex(content: &[u8]) -> String {
    Sha256::digest(content).iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
fn read_returns_numbered_lines_and_ranges() {
    let fixture = Fixture::plain();
    fixture.write("src/ten.txt", numbered(10));
    let all = read(&fixture.workspace, "src/ten.txt", None, None).unwrap();
    assert_eq!((all.lines.len(), all.total_lines, all.more, all.next_offset), (10, Some(10), false, None));
    assert_eq!(
        all.render(),
        "src/ten.txt: lines 1-10 of 10 (71 bytes)\n1: line 1\n2: line 2\n3: line 3\n4: line 4\n5: line 5\n6: line 6\n7: line 7\n8: line 8\n9: line 9\n10: line 10"
    );

    let part = read(&fixture.workspace, "src/ten.txt", Some(3), Some(4)).unwrap();
    assert_eq!(part.lines.iter().map(|line| line.number).collect::<Vec<_>>(), vec![3, 4, 5, 6]);
    assert_eq!((part.total_lines, part.more, part.next_offset), (Some(10), true, Some(7)));
    assert_eq!(
        part.render(),
        "src/ten.txt: lines 3-6 of 10 (71 bytes)\n3: line 3\n4: line 4\n5: line 5\n6: line 6\n[4 more lines; continue with offset=7]"
    );

    let tail = read(&fixture.workspace, "src/ten.txt", Some(10), Some(50)).unwrap();
    assert_eq!((tail.lines.len(), tail.more), (1, false));
    assert_eq!(read(&fixture.workspace, "src/ten.txt", Some(0), Some(0)).unwrap().lines.len(), 1, "0 is clamped to 1");
}

#[test]
fn read_offset_past_the_end_says_so() {
    let fixture = Fixture::plain();
    fixture.write("ten.txt", numbered(10));
    let past = read(&fixture.workspace, "ten.txt", Some(11), None).unwrap();
    assert!(past.lines.is_empty() && !past.more);
    assert_eq!(past.render(), "ten.txt: no lines at offset 11; the file has 10 lines (71 bytes)");
    assert!(read(&fixture.workspace, "ten.txt", Some(u64::MAX), Some(usize::MAX)).unwrap().lines.is_empty());
}

#[test]
fn read_caps_lines_per_call() {
    let fixture = Fixture::plain();
    fixture.write("big.txt", numbered(5000));
    for limit in [None, Some(99_999)] {
        let first = read(&fixture.workspace, "big.txt", None, limit).unwrap();
        assert_eq!((first.lines.len(), first.total_lines, first.more, first.next_offset), (READ_MAX_LINES, Some(5000), true, Some(2001)));
        assert!(first.render().ends_with("[3000 more lines; continue with offset=2001]"), "{}", &first.render()[first.render().len() - 80..]);
    }
    let last = read(&fixture.workspace, "big.txt", Some(4001), None).unwrap();
    assert_eq!((last.lines.len(), last.more, last.lines[999].text.as_str()), (1000, false, "line 5000"));
}

#[test]
fn read_caps_bytes_per_call() {
    let fixture = Fixture::plain();
    let line = "x".repeat(1000);
    fixture.write("wide.txt", format!("{line}\n").repeat(500));
    let first = read(&fixture.workspace, "wide.txt", None, None).unwrap();
    // The cap bounds the lines as rendered (`\nN: text`), not only their text.
    let rendered = |result: &ReadResult| result.render().len() - result.render().find('\n').unwrap();
    let bytes = rendered(&first) - first.render().lines().last().unwrap().len() - 1;
    assert!(bytes <= READ_MAX_BYTES && bytes > READ_MAX_BYTES - 1010, "{bytes}");
    assert_eq!(first.lines.len(), READ_MAX_BYTES / 1005);
    assert_eq!((first.more, first.next_offset, first.total_lines), (true, Some(first.lines.len() as u64 + 1), Some(500)));
    // Continuing from next_offset loses nothing.
    let second = read(&fixture.workspace, "wide.txt", first.next_offset, None).unwrap();
    assert_eq!(second.lines[0].number, first.lines.len() as u64 + 1);
}

/// A caller with a smaller result budget (a model with a small context) gets
/// whole lines within it, the same continuation, and the same file facts.
#[test]
fn read_within_a_smaller_byte_bound_cuts_at_a_line_and_continues() {
    let fixture = Fixture::plain();
    let content: String = (1..=3000).map(|n| format!("line {n} {}\n", "y".repeat(40))).collect();
    fixture.write("src.txt", &content);
    let whole = read(&fixture.workspace, "src.txt", None, None).unwrap();
    let mut offset = None;
    let mut seen = Vec::new();
    for _ in 0..200 {
        let part = read_within(&fixture.workspace, "src.txt", offset, None, 20_000).unwrap();
        let text = part.render();
        assert!(text.len() <= 20_000 + 200, "{}", text.len());
        assert_eq!(part.facts, whole.facts);
        seen.extend(part.lines.iter().map(|line| line.text.clone()));
        if !part.more {
            break;
        }
        assert!(text.ends_with(&format!("continue with offset={}]", part.next_offset.unwrap())), "{}", &text[text.len() - 80..]);
        offset = part.next_offset;
    }
    assert_eq!(seen.join("\n") + "\n", content);
    // The bound is clamped: never below one long line, never above the policy cap.
    let tiny = read_within(&fixture.workspace, "src.txt", None, None, 1).unwrap();
    assert!(tiny.lines.len() > 10 && tiny.render().len() <= super::limits::RESULT_MIN_BYTES + 200);
    let huge = read_within(&fixture.workspace, "src.txt", None, None, usize::MAX).unwrap();
    assert_eq!(huge.lines.len(), whole.lines.len());
    // A line cut with its marker is counted with the marker.
    fixture.write("long.txt", format!("{}\n", "z".repeat(50_000)).repeat(40));
    let long = read_within(&fixture.workspace, "long.txt", None, None, 20_000).unwrap();
    assert!(long.render().len() <= 20_000 + 200 && long.lines.len() >= 2, "{} {}", long.render().len(), long.lines.len());
}

#[test]
fn read_cuts_long_lines_with_a_marker() {
    let fixture = Fixture::plain();
    fixture.write("long.txt", format!("short\n{}\n{}\nend\n", "a".repeat(5000), "é".repeat(3000)));
    let result = read(&fixture.workspace, "long.txt", None, None).unwrap();
    assert_eq!(result.lines[0].cut_from_bytes, None);
    assert_eq!((result.lines[1].text.chars().count(), result.lines[1].cut_from_bytes), (LINE_MAX_CHARS, Some(5000)));
    assert_eq!((result.lines[2].text.chars().count(), result.lines[2].cut_from_bytes), (LINE_MAX_CHARS, Some(6000)));
    assert!(!result.lossy, "a character split by the cut is not invalid UTF-8");
    assert_eq!(result.lines[3].text, "end");
    assert!(result.render().contains(&format!("{} …[line cut after 2000 characters; it has 5000 bytes]\n3: ", "a".repeat(2000))));
    // A line longer than the kept prefix, cut in the middle of a character.
    fixture.write("split.txt", format!("x{}\n", "é".repeat(5000)));
    let split = read(&fixture.workspace, "split.txt", None, None).unwrap();
    assert_eq!((split.lines[0].text.chars().count(), split.lines[0].cut_from_bytes, split.lossy), (LINE_MAX_CHARS, Some(10_001), false));
    // Exactly at the limit is not cut.
    fixture.write("exact.txt", format!("{}\n", "b".repeat(LINE_MAX_CHARS)));
    assert_eq!(read(&fixture.workspace, "exact.txt", None, None).unwrap().lines[0].cut_from_bytes, None);
}

#[test]
fn read_single_huge_line_stays_bounded() {
    let fixture = Fixture::plain();
    fixture.write("huge.txt", "z".repeat(10 * 1024 * 1024));
    let started = Instant::now();
    let result = read(&fixture.workspace, "huge.txt", None, None).unwrap();
    assert!(started.elapsed() < Duration::from_secs(120), "took {:?}", started.elapsed());
    assert_eq!((result.lines.len(), result.total_lines, result.more), (1, Some(1), false));
    assert_eq!(result.lines[0].cut_from_bytes, Some(10 * 1024 * 1024));
    assert!(result.render().len() < 3000);
    assert!(!result.final_newline);
}

#[test]
fn read_refuses_binary_files_with_their_size() {
    let fixture = Fixture::plain();
    fixture.write("image.png", b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR");
    let error = read(&fixture.workspace, "image.png", None, None).unwrap_err();
    assert_eq!(error, Error::Binary { path: "image.png".to_string(), size: 16 });
    assert_eq!(error.to_string(), "`image.png` is a binary file (16 bytes); it cannot be shown as text");
    // The sniff window is the first 8 KiB: a NUL at its last byte counts, one after it does not.
    let mut content = vec![b'a'; BINARY_SNIFF_BYTES + 10];
    content[BINARY_SNIFF_BYTES - 1] = 0;
    fixture.write("edge.bin", &content);
    assert!(matches!(read(&fixture.workspace, "edge.bin", None, None), Err(Error::Binary { .. })));
    content[BINARY_SNIFF_BYTES - 1] = b'a';
    content[BINARY_SNIFF_BYTES] = 0;
    fixture.write("late.txt", &content);
    assert!(read(&fixture.workspace, "late.txt", None, None).is_ok());
}

#[test]
fn read_shows_invalid_utf8_lossily_with_a_notice() {
    let fixture = Fixture::plain();
    fixture.write("latin1.txt", b"caf\xe9\nplain\n\xff\xfe\n");
    let result = read(&fixture.workspace, "latin1.txt", None, None).unwrap();
    assert!(result.lossy);
    assert_eq!(result.lines[0].text, "caf\u{FFFD}");
    assert_eq!(result.lines[1].text, "plain");
    assert!(result.render().contains("[note: the file is not valid UTF-8; invalid bytes are shown as \u{FFFD}, so the text is not exact]"));
    let clean = read(&fixture.workspace, "latin1.txt", Some(2), Some(1)).unwrap();
    assert!(!clean.lossy, "only the shown lines are judged");
}

#[test]
fn read_empty_file() {
    let fixture = Fixture::plain();
    fixture.write("empty.txt", "");
    let result = read(&fixture.workspace, "empty.txt", None, None).unwrap();
    assert_eq!((result.lines.len(), result.total_lines, result.more), (0, Some(0), false));
    assert_eq!(result.render(), "empty.txt: empty file (0 bytes)");
    assert_eq!(result.facts.sha256.as_deref(), Some(sha256_hex(b"").as_str()));
}

#[test]
fn read_file_without_final_newline() {
    let fixture = Fixture::plain();
    fixture.write("open.txt", "a\nb");
    let result = read(&fixture.workspace, "open.txt", None, None).unwrap();
    assert_eq!((result.lines.len(), result.total_lines, result.final_newline), (2, Some(2), false));
    assert_eq!(result.render(), "open.txt: lines 1-2 of 2 (3 bytes)\n1: a\n2: b\n[note: no newline at the end of the file]");
    fixture.write("closed.txt", "a\nb\n");
    let closed = read(&fixture.workspace, "closed.txt", None, None).unwrap();
    assert!(closed.final_newline && !closed.render().contains("no newline"));
    fixture.write("blank.txt", "\n\n");
    assert_eq!(read(&fixture.workspace, "blank.txt", None, None).unwrap().total_lines, Some(2));
}

#[test]
fn read_crlf_lines() {
    let fixture = Fixture::plain();
    fixture.write("dos.txt", "one\r\ntwo\r\nthree");
    let result = read(&fixture.workspace, "dos.txt", None, None).unwrap();
    assert_eq!(result.lines.iter().map(|line| line.text.as_str()).collect::<Vec<_>>(), vec!["one", "two", "three"]);
    assert!(result.crlf);
    assert!(result.render().contains("[note: lines end in CRLF; the carriage returns are not shown]"));
    // A lone carriage return inside a line is content, not a line ending.
    fixture.write("cr.txt", "a\rb\n");
    let lone = read(&fixture.workspace, "cr.txt", None, None).unwrap();
    assert_eq!((lone.lines[0].text.as_str(), lone.crlf), ("a\rb", false));
}

#[test]
fn read_file_removed_between_calls() {
    let fixture = Fixture::plain();
    let path = fixture.write("gone.txt", "here\n");
    let mut ledger = ReadLedger::default();
    ledger.note(&read(&fixture.workspace, "gone.txt", None, None).unwrap());
    std::fs::remove_file(path).unwrap();
    let error = read(&fixture.workspace, "gone.txt", None, None).unwrap_err();
    assert_eq!(error.to_string(), "`gone.txt` does not exist in the workspace");
    assert_eq!(ledger.check(&fixture.workspace, "gone.txt").unwrap(), Freshness::Missing);
}

#[test]
fn read_refuses_what_is_not_a_readable_file() {
    let fixture = Fixture::plain();
    fixture.write("dir/file.txt", "x\n");
    write(&fixture.outside, "secret.txt", "SECRET\n");
    symlink("../outside/secret.txt", fixture.root.join("leak"));
    let directory = read(&fixture.workspace, "dir", None, None).unwrap_err();
    assert_eq!(directory.to_string(), "`dir` is not a file (it is a directory)");
    assert_eq!(read(&fixture.workspace, "leak", None, None).unwrap_err().rule(), Some(Rule::SymlinkOutside));
    assert_eq!(read(&fixture.workspace, "../outside/secret.txt", None, None).unwrap_err().rule(), Some(Rule::ParentEscape));
    assert!(matches!(read(&fixture.workspace, "absent.txt", None, None), Err(Error::NotFound { .. })));

    let fifo = std::ffi::CString::new(fixture.root.join("pipe").to_str().unwrap()).unwrap();
    // SAFETY: a valid NUL-terminated path; mkfifo touches no memory.
    assert_eq!(unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) }, 0);
    let started = Instant::now();
    assert_eq!(read(&fixture.workspace, "pipe", None, None).unwrap_err().to_string(), "`pipe` is not a regular file");
    assert!(started.elapsed() < Duration::from_secs(120), "opening a FIFO must not wait for a writer");

    use std::os::unix::fs::PermissionsExt;
    let locked = fixture.write("locked.txt", "x\n");
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
    if std::fs::read(&locked).is_err() {
        let error = read(&fixture.workspace, "locked.txt", None, None).unwrap_err();
        assert_eq!(error.to_string(), "cannot open `locked.txt`: permission denied");
    }
}

#[test]
fn read_reports_the_facts_an_edit_needs() {
    let fixture = Fixture::plain();
    let content = numbered(3000);
    let path = fixture.write("facts.txt", &content);
    // A partial read still describes the whole file.
    let result = read(&fixture.workspace, "facts.txt", Some(5), Some(2)).unwrap();
    let metadata = std::fs::metadata(&path).unwrap();
    let mtime = metadata.modified().unwrap().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos() as i64;
    assert_eq!(result.facts, FileFacts { size: content.len() as u64, mtime_ns: mtime, sha256: Some(sha256_hex(content.as_bytes())) });
    assert!(result.writable && !result.changed_during_read);
}

#[test]
fn read_through_an_extra_root_is_marked_read_only() {
    let fixture = Fixture::plain();
    write(&fixture.extra, "doc.md", "# extra\n");
    let absolute = fixture.extra.join("doc.md");
    let result = read(&fixture.workspace, absolute.to_str().unwrap(), None, None).unwrap();
    assert!(!result.writable);
    assert_eq!(result.path, absolute.to_str().unwrap());
    assert!(result.render().ends_with("[note: this file is in a read-only root]"), "{}", result.render());
    let mut ledger = ReadLedger::default();
    ledger.note(&result);
    assert!(ledger.is_empty(), "a file that can never be edited is not recorded");
}

#[test]
fn read_beyond_the_scan_limit_is_partial_and_unverifiable() {
    let fixture = Fixture::plain();
    let line = format!("{}\n", "k".repeat(1023));
    let lines = (SCAN_MAX_BYTES / 1024) as usize + 10;
    let path = fixture.root.join("enormous.log");
    {
        use std::io::Write;
        let mut file = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
        for _ in 0..lines {
            file.write_all(line.as_bytes()).unwrap();
        }
    }
    let result = read(&fixture.workspace, "enormous.log", Some(100), Some(2)).unwrap();
    assert_eq!((result.lines.len(), result.total_lines, result.more), (2, None, true));
    assert_eq!((result.facts.size, &result.facts.sha256), (lines as u64 * 1024, &None));
    let text = result.render();
    let header = format!("enormous.log: lines 100-101 ({} bytes; too large to count its lines)", lines * 1024);
    assert!(text.starts_with(&header), "{}", &text[..120]);
    assert!(text.contains("[more lines follow; continue with offset=102]"));
    assert!(text.contains("only that much of it can be read, and it cannot be edited"));
    let mut ledger = ReadLedger::default();
    ledger.note(&result);
    assert_eq!(ledger.check(&fixture.workspace, "enormous.log").unwrap(), Freshness::Unverifiable);
    // Past the scanned prefix there is nothing to return, and the result says so.
    let beyond = read(&fixture.workspace, "enormous.log", Some(lines as u64), None).unwrap();
    assert!(beyond.lines.is_empty() && !beyond.more);
    std::fs::remove_file(path).unwrap();
}

// --- the read ledger ---------------------------------------------------------

#[test]
fn ledger_tracks_read_and_unchanged() {
    let fixture = Fixture::plain();
    let path = fixture.write("dir/code.rs", "fn main() {}\n");
    let mut ledger = ReadLedger::default();
    assert_eq!(ledger.check(&fixture.workspace, "dir/code.rs").unwrap(), Freshness::Unread);
    assert_eq!(Freshness::Unread.refusal(), Some("the file has not been read in this conversation; read it first"));

    ledger.note(&read(&fixture.workspace, "dir/code.rs", None, None).unwrap());
    assert_eq!(ledger.check(&fixture.workspace, "dir/code.rs").unwrap(), Freshness::Unchanged);
    assert_eq!(Freshness::Unchanged.refusal(), None);
    // Another spelling of the same file shares the record.
    assert_eq!(ledger.check(&fixture.workspace, "./dir/../dir/code.rs").unwrap(), Freshness::Unchanged);
    assert_eq!(ledger.check(&fixture.workspace, fixture.root.join("dir/code.rs").to_str().unwrap()).unwrap(), Freshness::Unchanged);

    // A new timestamp with the same content is not a change.
    let file = std::fs::File::options().write(true).open(&path).unwrap();
    file.set_modified(SystemTime::now() + Duration::from_secs(3600)).unwrap();
    assert_eq!(ledger.check(&fixture.workspace, "dir/code.rs").unwrap(), Freshness::Unchanged);

    // Same size, different content, original timestamp restored: still a change.
    let recorded = ledger.recorded("dir/code.rs").unwrap().clone();
    std::fs::write(&path, "fn mian() {}\n").unwrap();
    let file = std::fs::File::options().write(true).open(&path).unwrap();
    file.set_modified(SystemTime::UNIX_EPOCH + Duration::from_nanos(recorded.mtime_ns as u64)).unwrap();
    assert_eq!(ledger.check(&fixture.workspace, "dir/code.rs").unwrap(), Freshness::Changed);
    assert!(Freshness::Changed.refusal().unwrap().contains("read it again"));

    // An edit tool records its own result and may then edit again.
    let resolved = resolve(&fixture.workspace, "dir/code.rs", Access::Write).unwrap();
    ledger.record(&resolved.display, super::read::file_facts(&resolved).unwrap());
    assert_eq!(ledger.check(&fixture.workspace, "dir/code.rs").unwrap(), Freshness::Unchanged);
    ledger.forget("dir/code.rs");
    assert_eq!(ledger.check(&fixture.workspace, "dir/code.rs").unwrap(), Freshness::Unread);
}

#[test]
fn ledger_check_applies_the_write_rules() {
    let fixture = Fixture::plain();
    fixture.write("a.txt", "a\n");
    fixture.write(".git/config", "[core]\n");
    symlink("a.txt", fixture.root.join("link"));
    let ledger = ReadLedger::default();
    assert_eq!(ledger.check(&fixture.workspace, "link").unwrap_err().rule(), Some(Rule::WriteThroughSymlink));
    assert_eq!(ledger.check(&fixture.workspace, ".git/config").unwrap_err().rule(), Some(Rule::WriteGitDir));
    assert_eq!(ledger.check(&fixture.workspace, "../x").unwrap_err().rule(), Some(Rule::ParentEscape));
    assert_eq!(ledger.check(&fixture.workspace, "new-file.txt").unwrap(), Freshness::Unread, "a file to be created");
}

#[test]
fn ledger_is_bounded_and_serializable() {
    let mut ledger = ReadLedger::default();
    let facts = |index: usize| FileFacts { size: index as u64, mtime_ns: 1, sha256: Some(format!("{index:064x}")) };
    for index in 0..LEDGER_MAX_ENTRIES + 25 {
        ledger.record(&format!("file-{index}.rs"), facts(index));
    }
    assert_eq!(ledger.len(), LEDGER_MAX_ENTRIES);
    assert!(ledger.recorded("file-0.rs").is_none() && ledger.recorded("file-24.rs").is_none(), "the oldest records go first");
    assert_eq!(ledger.recorded("file-25.rs"), Some(&facts(25)));
    // Re-recording refreshes a record's age.
    ledger.record("file-25.rs", facts(25));
    ledger.record("fresh.rs", facts(1));
    assert!(ledger.recorded("file-25.rs").is_some() && ledger.recorded("file-26.rs").is_none());

    let json = serde_json::to_string(&ledger).unwrap();
    let restored: ReadLedger = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, ledger);
    assert_eq!(serde_json::from_str::<ReadLedger>("{\"entries\":{},\"clock\":0}").unwrap(), ReadLedger::default());
}
