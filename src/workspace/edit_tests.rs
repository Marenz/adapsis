use std::os::unix::fs::{MetadataExt, PermissionsExt};

use super::edit::{Failure, edit, replace_exact};
use super::error::{Error, Rule};
use super::ledger::{Freshness, ReadLedger};
use super::limits::{AMBIGUOUS_LINES_SHOWN, EDIT_MAX_FILE_BYTES, SCAN_MAX_BYTES, WRITE_MAX_BYTES};
use super::read::read;
use super::secure_write_tests::{assert_no_temp_files, umask};
use super::test_support::{Fixture, symlink, write as write_file};
use super::write::write;

/// A fixture file that has been read into a fresh ledger.
fn read_file(fixture: &Fixture, path: &str, content: impl AsRef<[u8]>) -> ReadLedger {
    fixture.write(path, content);
    let mut ledger = ReadLedger::default();
    ledger.note(&read(&fixture.workspace, path, None, None).unwrap());
    ledger
}

fn content(fixture: &Fixture, path: &str) -> String {
    String::from_utf8(std::fs::read(fixture.root.join(path)).unwrap()).unwrap()
}

fn not_fresh(error: &Error) -> Freshness {
    match error {
        Error::NotFresh { state, .. } => *state,
        other => panic!("expected a freshness refusal, got {other:?}"),
    }
}

// --- the replacement itself (pure) -----------------------------------------

#[test]
fn replace_exact_replaces_one_occurrence_and_reports_its_lines() {
    let text = "one\ntwo\nthree\nfour\n";
    let single = replace_exact(text, "two", "2", false).unwrap();
    assert_eq!((single.text.as_str(), single.replacements, single.first_line, single.last_line), ("one\n2\nthree\nfour\n", 1, 2, 2));
    let grown = replace_exact(text, "two\nthree", "a\nb\nc\nd", false).unwrap();
    assert_eq!((grown.text.as_str(), grown.first_line, grown.last_line), ("one\na\nb\nc\nd\nfour\n", 2, 5));
    let removed = replace_exact(text, "two\n", "", false).unwrap();
    assert_eq!((removed.text.as_str(), removed.first_line, removed.last_line), ("one\nthree\nfour\n", 2, 2));
    let whole_line = replace_exact(text, "four\n", "4\n5\n", false).unwrap();
    assert_eq!((whole_line.text.as_str(), whole_line.first_line, whole_line.last_line), ("one\ntwo\nthree\n4\n5\n", 4, 5));
    assert_eq!(whole_line.kept_final_newline, None);
}

#[test]
fn replace_exact_requires_a_unique_match_unless_replace_all() {
    let text = "let x = 1;\nlet y = 1;\nlet z = x;\n";
    assert_eq!(replace_exact(text, "= 1;", "= 2;", false), Err(Failure::Ambiguous { count: 2, lines: vec![1, 2] }));
    let all = replace_exact(text, "= 1;", "= 2;", true).unwrap();
    assert_eq!((all.text.as_str(), all.replacements, all.first_line, all.last_line), ("let x = 2;\nlet y = 2;\nlet z = x;\n", 2, 1, 2));
    assert_eq!(replace_exact(text, "let w", "let v", false), Err(Failure::NoMatch { mixed_endings: false }));
    assert_eq!(replace_exact(text, "let w", "let v", true), Err(Failure::NoMatch { mixed_endings: false }));
    // Overlapping occurrences are occurrences: `aa` is not unique in `aaa`.
    assert_eq!(replace_exact("aaa\n", "aa", "b", false), Err(Failure::Ambiguous { count: 2, lines: vec![1, 1] }));
    assert_eq!(replace_exact("aaa\n", "aa", "b", true).unwrap().text, "ba\n");
    // Many occurrences: the count is exact, the listed lines are bounded.
    let many = "x\n".repeat(500);
    match replace_exact(&many, "x", "y", false) {
        Err(Failure::Ambiguous { count: 500, lines }) => assert_eq!(lines.len(), AMBIGUOUS_LINES_SHOWN),
        other => panic!("{other:?}"),
    }
    assert_eq!(replace_exact(&many, "x", "y", true).unwrap().replacements, 500);
    // Multi-byte text: offsets are bytes, the search never splits a character.
    let unicode = replace_exact("größe = 1\nGRÖSSE = 2\n", "ö", "oe", false).unwrap();
    assert_eq!(unicode.text, "groeße = 1\nGRÖSSE = 2\n");
    assert_eq!(replace_exact("ééé", "éé", "x", false), Err(Failure::Ambiguous { count: 2, lines: vec![1, 1] }));
}

#[test]
fn replace_exact_adapts_to_crlf_without_doubling_carriage_returns() {
    let dos = "one\r\ntwo\r\nthree\r\n";
    // The model sees and writes `\n`; the file keeps CRLF everywhere.
    let multi = replace_exact(dos, "one\ntwo", "1\n2\n2b", false).unwrap();
    assert_eq!(multi.text, "1\r\n2\r\n2b\r\nthree\r\n");
    let single = replace_exact(dos, "two", "zwei\ndrei", false).unwrap();
    assert_eq!(single.text, "one\r\nzwei\r\ndrei\r\nthree\r\n");
    // A caller that already wrote `\r\n` gets exactly that, never `\r\r\n`.
    let explicit = replace_exact(dos, "one\r\ntwo", "1\r\n2\nlast", false).unwrap();
    assert_eq!(explicit.text, "1\r\n2\r\nlast\r\nthree\r\n");
    assert!(!explicit.text.contains("\r\r"));
    let mixed_in_arguments = replace_exact(dos, "two\r\nthree\n", "x\n", false).unwrap();
    assert_eq!(mixed_in_arguments.text, "one\r\nx\r\n");
    // A file with LF only is never touched by this.
    let unix = replace_exact("one\ntwo\n", "one", "1\r\n1b", false).unwrap();
    assert_eq!(unix.text, "1\r\n1b\ntwo\n", "what the caller wrote, exactly");
    // Mixed endings: exact matching only, and a hint when that fails.
    let mixed = "one\r\ntwo\nthree\n";
    assert_eq!(replace_exact(mixed, "one\ntwo", "x", false), Err(Failure::NoMatch { mixed_endings: true }));
    assert_eq!(replace_exact(mixed, "two\nthree", "x\ny", false).unwrap().text, "one\r\nx\ny\n");
    assert_eq!(replace_exact(mixed, "one\r\ntwo", "x", false).unwrap().text, "x\nthree\n");
}

#[test]
fn replace_exact_keeps_the_final_newline_state() {
    // The file ends with a newline; the replacement would remove it.
    let kept = replace_exact("a\nb\n", "b\n", "c", false).unwrap();
    assert_eq!((kept.text.as_str(), kept.kept_final_newline), ("a\nc\n", Some(true)));
    // The file does not end with a newline; the replacement would add one (or several).
    let none = replace_exact("a\nb", "b", "c\n", false).unwrap();
    assert_eq!((none.text.as_str(), none.kept_final_newline), ("a\nc", Some(false)));
    let several = replace_exact("a\nb", "b", "c\n\n\n", false).unwrap();
    assert_eq!((several.text.as_str(), several.kept_final_newline), ("a\nc", Some(false)));
    // CRLF files get a CRLF back, and lose a whole CRLF.
    let dos = replace_exact("a\r\nb\r\n", "b\n", "c", false).unwrap();
    assert_eq!((dos.text.as_str(), dos.kept_final_newline), ("a\r\nc\r\n", Some(true)));
    let dos_none = replace_exact("a\r\nb", "b", "c\n", false).unwrap();
    assert_eq!((dos_none.text.as_str(), dos_none.kept_final_newline), ("a\r\nc", Some(false)));
    // Untouched when the replacement does not reach the end, or keeps the state.
    assert_eq!(replace_exact("a\nb\n", "a", "x", false).unwrap().kept_final_newline, None);
    assert_eq!(replace_exact("a\nb", "a", "x", false).unwrap().kept_final_newline, None);
    // Emptying a file is allowed; there is no state to keep.
    let emptied = replace_exact("only\n", "only\n", "", false).unwrap();
    assert_eq!((emptied.text.as_str(), emptied.kept_final_newline), ("", None));
    // A replacement that the rule turns into a no-op is reported as one.
    assert_eq!(replace_exact("a\nb\n", "b\n", "b", false), Err(Failure::Unchanged));
    assert_eq!(replace_exact("a\nb", "b", "b\n", false), Err(Failure::Unchanged));
}

// --- edit ------------------------------------------------------------------

#[test]
fn edit_replaces_text_in_a_file_that_was_read() {
    let fixture = Fixture::plain();
    let mut ledger = read_file(&fixture, "src/lib.rs", "fn main() {\n    println!(\"old\");\n}\n");
    let result = edit(&fixture.workspace, &mut ledger, "src/lib.rs", "println!(\"old\");", "println!(\"new\");\n    done();", false).unwrap();
    assert_eq!(content(&fixture, "src/lib.rs"), "fn main() {\n    println!(\"new\");\n    done();\n}\n");
    assert_eq!((result.path.as_str(), result.replacements, result.first_line, result.last_line), ("src/lib.rs", 1, 2, 3));
    assert_eq!(result.render(), "edited src/lib.rs: 1 replacement, lines 2-3");
    assert_no_temp_files(&fixture.root);
    // The ledger knows the new content: a second edit needs no new read...
    let second = edit(&fixture.workspace, &mut ledger, "./src/../src/lib.rs", "done();", "finished();", false).unwrap();
    assert_eq!(second.render(), "edited src/lib.rs: 1 replacement, line 3");
    // ...and a read shows exactly what the ledger holds.
    let facts = read(&fixture.workspace, "src/lib.rs", None, None).unwrap().facts;
    assert_eq!(ledger.recorded("src/lib.rs"), Some(&facts));
    let all = edit(&fixture.workspace, &mut ledger, "src/lib.rs", "    ", "\t", true).unwrap();
    assert_eq!(all.render(), "edited src/lib.rs: 2 replacements, lines 2-3");
    assert_eq!(content(&fixture, "src/lib.rs"), "fn main() {\n\tprintln!(\"new\");\n\tfinished();\n}\n");
}

#[test]
fn edit_refuses_a_file_that_is_not_provably_what_was_read() {
    let fixture = Fixture::plain();
    // Unread.
    fixture.write("unread.rs", "fn a() {}\n");
    let mut ledger = ReadLedger::default();
    let error = edit(&fixture.workspace, &mut ledger, "unread.rs", "a", "b", false).unwrap_err();
    assert_eq!(not_fresh(&error), Freshness::Unread);
    assert_eq!(error.to_string(), "`unread.rs` was not changed: the file has not been read in this conversation; read it first");
    assert_eq!(content(&fixture, "unread.rs"), "fn a() {}\n");

    // Changed since the read (same size, so only the content tells).
    let mut ledger = read_file(&fixture, "changed.rs", "fn a() {}\n");
    fixture.write("changed.rs", "fn z() {}\n");
    let error = edit(&fixture.workspace, &mut ledger, "changed.rs", "z", "b", false).unwrap_err();
    assert_eq!(not_fresh(&error), Freshness::Changed);
    assert_eq!(error.to_string(), "`changed.rs` was not changed: the file changed since it was read; read it again");
    assert_eq!(content(&fixture, "changed.rs"), "fn z() {}\n");
    // After reading again it works.
    ledger.note(&read(&fixture.workspace, "changed.rs", None, None).unwrap());
    edit(&fixture.workspace, &mut ledger, "changed.rs", "z", "b", false).unwrap();

    // Removed since the read.
    let mut ledger = read_file(&fixture, "gone.rs", "fn a() {}\n");
    std::fs::remove_file(fixture.root.join("gone.rs")).unwrap();
    let error = edit(&fixture.workspace, &mut ledger, "gone.rs", "a", "b", false).unwrap_err();
    assert_eq!(not_fresh(&error), Freshness::Missing);
    assert!(error.to_string().contains("no longer exists"), "{error}");
    assert!(!fixture.root.join("gone.rs").exists(), "edit never creates a file");
    // Never existed: plain not found.
    assert!(matches!(edit(&fixture.workspace, &mut ledger, "never.rs", "a", "b", false), Err(Error::NotFound { .. })));
}

#[test]
fn edit_refuses_an_unverifiable_file() {
    let fixture = Fixture::plain();
    // A file beyond the scan limit is read without a hash, so it can never be edited.
    let path = fixture.root.join("enormous.log");
    {
        use std::io::Write;
        let line = format!("{}\n", "k".repeat(1023));
        let mut file = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
        for _ in 0..(SCAN_MAX_BYTES / 1024) + 2 {
            file.write_all(line.as_bytes()).unwrap();
        }
    }
    let mut ledger = ReadLedger::default();
    ledger.note(&read(&fixture.workspace, "enormous.log", None, Some(1)).unwrap());
    let error = edit(&fixture.workspace, &mut ledger, "enormous.log", "k", "j", true).unwrap_err();
    assert_eq!(not_fresh(&error), Freshness::Unverifiable);
    let error = write(&fixture.workspace, &mut ledger, "enormous.log", "small\n").unwrap_err();
    assert_eq!(not_fresh(&error), Freshness::Unverifiable);
    assert!(error.to_string().contains("cannot be verified as unchanged; read it again"), "{error}");
    std::fs::remove_file(path).unwrap();
    // A file larger than the edit limit, but hashable, is refused by size.
    let big = fixture.root.join("big.txt");
    std::fs::write(&big, vec![b'x'; EDIT_MAX_FILE_BYTES as usize + 1]).unwrap();
    ledger.note(&read(&fixture.workspace, "big.txt", None, Some(1)).unwrap());
    let error = edit(&fixture.workspace, &mut ledger, "big.txt", "x", "y", true).unwrap_err();
    assert!(matches!(error, Error::InvalidArgument(_)) && error.to_string().contains("16777217 bytes"), "{error}");
    std::fs::remove_file(big).unwrap();
}

#[test]
fn edit_reports_absent_and_ambiguous_text() {
    let fixture = Fixture::plain();
    let original = "let x = 1;\nlet y = 1;\n";
    let mut ledger = read_file(&fixture, "code.rs", original);
    let error = edit(&fixture.workspace, &mut ledger, "code.rs", "let w", "let v", false).unwrap_err();
    assert_eq!(
        error.to_string(),
        "`code.rs` was not changed: the text to replace was not found. It must match the file exactly, including indentation and blank lines; read the lines again and copy them."
    );
    let error = edit(&fixture.workspace, &mut ledger, "code.rs", "= 1;", "= 2;", false).unwrap_err();
    assert_eq!(error, Error::Ambiguous { path: "code.rs".to_string(), count: 2, lines: vec![1, 2] });
    assert_eq!(
        error.to_string(),
        "`code.rs` was not changed: the text to replace occurs 2 times (starting at lines 1, 2). Add surrounding lines to make it unique, or set replace_all"
    );
    let identical = edit(&fixture.workspace, &mut ledger, "code.rs", "let x", "let x", false).unwrap_err();
    assert_eq!(identical.to_string(), "invalid argument: the old and the new text are identical; there is nothing to change");
    let empty = edit(&fixture.workspace, &mut ledger, "code.rs", "", "x", false).unwrap_err();
    assert!(empty.to_string().contains("the text to replace is empty") && empty.to_string().contains("use write"), "{empty}");
    assert_eq!(content(&fixture, "code.rs"), original, "no refusal changed the file");
    // A refused edit leaves the ledger valid: the next correct edit works.
    edit(&fixture.workspace, &mut ledger, "code.rs", "let x = 1;", "let x = 2;", false).unwrap();
    // The list of lines is bounded and says so.
    let mut ledger = read_file(&fixture, "many.txt", "x\n".repeat(40));
    let error = edit(&fixture.workspace, &mut ledger, "many.txt", "x", "y", false).unwrap_err().to_string();
    assert!(error.contains("occurs 40 times (starting at lines 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, …)"), "{error}");
}

#[test]
fn edit_preserves_crlf_and_the_final_newline() {
    let fixture = Fixture::plain();
    let mut ledger = read_file(&fixture, "dos.txt", "first\r\nsecond\r\nthird\r\n");
    edit(&fixture.workspace, &mut ledger, "dos.txt", "first\nsecond", "1\n2\n2b", false).unwrap();
    assert_eq!(content(&fixture, "dos.txt"), "1\r\n2\r\n2b\r\nthird\r\n");
    edit(&fixture.workspace, &mut ledger, "dos.txt", "2\r\n2b", "two\r\ntwo-b", false).unwrap();
    assert_eq!(content(&fixture, "dos.txt"), "1\r\ntwo\r\ntwo-b\r\nthird\r\n");
    let result = edit(&fixture.workspace, &mut ledger, "dos.txt", "third\n", "3", false).unwrap();
    assert_eq!(content(&fixture, "dos.txt"), "1\r\ntwo\r\ntwo-b\r\n3\r\n");
    assert_eq!(
        result.render(),
        "edited dos.txt: 1 replacement, line 4\n[note: edit keeps the file's final-newline state (it still ends with a newline); to change that on purpose, rewrite the file with write]"
    );

    let mut ledger = read_file(&fixture, "open.txt", "a\nb");
    let result = edit(&fixture.workspace, &mut ledger, "open.txt", "b", "b\nc\n", false).unwrap();
    assert_eq!(content(&fixture, "open.txt"), "a\nb\nc");
    assert!(result.render().contains("it still ends without a newline") && result.render().contains("rewrite the file with write"), "{}", result.render());
    // The way out the note names: write changes the state on purpose.
    write(&fixture.workspace, &mut ledger, "open.txt", "a\nb\nc\n").unwrap();
    assert_eq!(content(&fixture, "open.txt"), "a\nb\nc\n");
    edit(&fixture.workspace, &mut ledger, "open.txt", "c", "d", false).unwrap();
    assert_eq!(content(&fixture, "open.txt"), "a\nb\nd\n");

    let mut ledger = read_file(&fixture, "mixed.txt", "one\r\ntwo\nthree\n");
    let error = edit(&fixture.workspace, &mut ledger, "mixed.txt", "one\ntwo", "x", false).unwrap_err().to_string();
    assert!(error.contains("mixes CRLF and LF line endings"), "{error}");
    edit(&fixture.workspace, &mut ledger, "mixed.txt", "two", "2", false).unwrap();
    assert_eq!(content(&fixture, "mixed.txt"), "one\r\n2\nthree\n");
}

#[test]
fn edit_refuses_files_that_are_not_valid_utf8() {
    let fixture = Fixture::plain();
    let original: &[u8] = b"caf\xe9 = 1\nplain = 2\n";
    let mut ledger = read_file(&fixture, "latin1.txt", original);
    let error = edit(&fixture.workspace, &mut ledger, "latin1.txt", "plain", "simple", false).unwrap_err();
    assert_eq!(error.to_string(), "`latin1.txt` is not valid UTF-8 text; a file with other bytes cannot be edited exactly");
    assert_eq!(std::fs::read(fixture.root.join("latin1.txt")).unwrap(), original);
    // Valid UTF-8 beyond ASCII is edited byte-exactly.
    let mut ledger = read_file(&fixture, "utf8.txt", "größe = 1 # ✓\nname = \"日本語\"\n");
    edit(&fixture.workspace, &mut ledger, "utf8.txt", "日本語", "中文", false).unwrap();
    assert_eq!(content(&fixture, "utf8.txt"), "größe = 1 # ✓\nname = \"中文\"\n");
}

#[test]
fn edit_keeps_the_mode_and_follows_the_write_rules() {
    let fixture = Fixture::plain();
    let mut ledger = read_file(&fixture, "run.sh", "#!/bin/sh\necho one\n");
    let path = fixture.root.join("run.sh");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    edit(&fixture.workspace, &mut ledger, "run.sh", "one", "two", false).unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().mode() & 0o7777, 0o755);

    fixture.write(".git/config", "[core]\n");
    write_file(&fixture.outside, "secret.txt", "SECRET\n");
    write_file(&fixture.extra, "doc.md", "extra\n");
    symlink("run.sh", fixture.root.join("link"));
    symlink("../outside/secret.txt", fixture.root.join("leak"));
    let attempt = |ledger: &mut ReadLedger, path: &str| edit(&fixture.workspace, ledger, path, "e", "E", true).unwrap_err();
    assert_eq!(attempt(&mut ledger, "link").rule(), Some(Rule::WriteThroughSymlink));
    assert_eq!(attempt(&mut ledger, "leak").rule(), Some(Rule::WriteThroughSymlink));
    assert_eq!(attempt(&mut ledger, ".git/config").rule(), Some(Rule::WriteGitDir));
    assert_eq!(attempt(&mut ledger, "../outside/secret.txt").rule(), Some(Rule::ParentEscape));
    assert_eq!(attempt(&mut ledger, fixture.outside.join("secret.txt").to_str().unwrap()).rule(), Some(Rule::OutsideWorkspace));
    assert_eq!(attempt(&mut ledger, fixture.extra.join("doc.md").to_str().unwrap()).rule(), Some(Rule::WriteReadOnlyRoot));
    assert!(matches!(attempt(&mut ledger, "."), Error::WrongKind { .. }));
    assert_eq!(std::fs::read_to_string(fixture.outside.join("secret.txt")).unwrap(), "SECRET\n");
    assert_eq!(std::fs::read_to_string(fixture.extra.join("doc.md")).unwrap(), "extra\n");
    assert_eq!(content(&fixture, ".git/config"), "[core]\n");
    // Reading through a link records the real file, and the real file is what gets edited.
    ledger.note(&read(&fixture.workspace, "link", None, None).unwrap());
    edit(&fixture.workspace, &mut ledger, "run.sh", "two", "three", false).unwrap();
    assert_eq!(content(&fixture, "run.sh"), "#!/bin/sh\necho three\n");
    assert!(std::fs::symlink_metadata(fixture.root.join("link")).unwrap().file_type().is_symlink());
}

#[test]
fn edit_notices_a_change_that_keeps_size_and_time() {
    use std::time::{Duration, SystemTime};
    let fixture = Fixture::plain();
    let mut ledger = read_file(&fixture, "sneaky.rs", "fn a() {}\n");
    let path = fixture.root.join("sneaky.rs");
    let recorded = ledger.recorded("sneaky.rs").unwrap().clone();
    std::fs::write(&path, "fn b() {}\n").unwrap();
    let file = std::fs::File::options().write(true).open(&path).unwrap();
    file.set_modified(SystemTime::UNIX_EPOCH + Duration::from_nanos(recorded.mtime_ns as u64)).unwrap();
    let error = edit(&fixture.workspace, &mut ledger, "sneaky.rs", "fn", "pub fn", false).unwrap_err();
    assert_eq!(not_fresh(&error), Freshness::Changed, "the content hash decides, not the timestamp");
    // The opposite: a new timestamp with the same content is not a change.
    let mut ledger = read_file(&fixture, "touched.rs", "fn a() {}\n");
    let file = std::fs::File::options().write(true).open(fixture.root.join("touched.rs")).unwrap();
    file.set_modified(SystemTime::now() + Duration::from_secs(3600)).unwrap();
    edit(&fixture.workspace, &mut ledger, "touched.rs", "fn", "pub fn", false).unwrap();
    assert_eq!(content(&fixture, "touched.rs"), "pub fn a() {}\n");
}

// --- write -------------------------------------------------------------------

#[test]
fn write_creates_new_files_freely() {
    let fixture = Fixture::plain();
    let mut ledger = ReadLedger::default();
    let result = write(&fixture.workspace, &mut ledger, "src/new/module.rs", "pub fn hello() {}\n\n// end\n").unwrap();
    assert_eq!(result.render(), "created src/new/module.rs (26 bytes, 3 lines)");
    assert!(result.created);
    assert_eq!(content(&fixture, "src/new/module.rs"), "pub fn hello() {}\n\n// end\n");
    assert_eq!(std::fs::metadata(fixture.root.join("src/new/module.rs")).unwrap().mode() & 0o7777, 0o666 & !umask());
    assert_no_temp_files(&fixture.root);
    // What was written counts as read: it can be edited and overwritten at once.
    edit(&fixture.workspace, &mut ledger, "src/new/module.rs", "hello", "greet", false).unwrap();
    let again = write(&fixture.workspace, &mut ledger, "src/new/module.rs", "// replaced\n").unwrap();
    assert_eq!(again.render(), "replaced the content of src/new/module.rs (12 bytes, 1 lines)");
    assert_eq!(content(&fixture, "src/new/module.rs"), "// replaced\n");
    // Empty content and content without a final newline are written as given.
    write(&fixture.workspace, &mut ledger, "empty.txt", "").unwrap();
    write(&fixture.workspace, &mut ledger, "open.txt", "no newline").unwrap();
    assert_eq!((content(&fixture, "empty.txt").as_str(), content(&fixture, "open.txt").as_str()), ("", "no newline"));
}

#[test]
fn write_over_an_existing_file_needs_a_fresh_read() {
    let fixture = Fixture::plain();
    fixture.write("existing.rs", "fn a() {}\n");
    let mut ledger = ReadLedger::default();
    let error = write(&fixture.workspace, &mut ledger, "existing.rs", "fn b() {}\n").unwrap_err();
    assert_eq!(not_fresh(&error), Freshness::Unread);
    assert_eq!(content(&fixture, "existing.rs"), "fn a() {}\n");
    ledger.note(&read(&fixture.workspace, "existing.rs", None, None).unwrap());
    fixture.write("existing.rs", "fn other() {}\n");
    let error = write(&fixture.workspace, &mut ledger, "existing.rs", "fn b() {}\n").unwrap_err();
    assert_eq!(not_fresh(&error), Freshness::Changed);
    assert_eq!(content(&fixture, "existing.rs"), "fn other() {}\n");
    ledger.note(&read(&fixture.workspace, "existing.rs", None, None).unwrap());
    let path = fixture.root.join("existing.rs");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
    write(&fixture.workspace, &mut ledger, "existing.rs", "fn b() {}\n").unwrap();
    assert_eq!(content(&fixture, "existing.rs"), "fn b() {}\n");
    assert_eq!(std::fs::metadata(&path).unwrap().mode() & 0o7777, 0o640, "the mode of a replaced file is kept");
    // Read earlier, removed since: the file is simply created again.
    std::fs::remove_file(&path).unwrap();
    assert!(write(&fixture.workspace, &mut ledger, "existing.rs", "fn c() {}\n").unwrap().created);
}

#[test]
fn write_refuses_by_size_kind_and_path_rule() {
    let fixture = Fixture::plain();
    fixture.write("dir/file.txt", "x\n");
    fixture.write(".git/config", "[core]\n");
    write_file(&fixture.outside, "secret.txt", "SECRET\n");
    symlink("dir/file.txt", fixture.root.join("link"));
    symlink("../outside", fixture.root.join("out"));
    symlink("nowhere", fixture.root.join("dangling"));
    let mut ledger = ReadLedger::default();
    let mut attempt = |path: &str| write(&fixture.workspace, &mut ledger, path, "x\n").unwrap_err();
    assert!(matches!(attempt("dir"), Error::WrongKind { .. }));
    assert!(matches!(attempt("."), Error::WrongKind { .. }));
    assert!(matches!(attempt("dir/file.txt/under"), Error::WrongKind { .. }));
    assert_eq!(attempt("link").rule(), Some(Rule::WriteThroughSymlink));
    assert_eq!(attempt("dangling").rule(), Some(Rule::WriteThroughSymlink));
    assert_eq!(attempt("out/new.txt").rule(), Some(Rule::SymlinkOutside));
    assert_eq!(attempt("../outside/new.txt").rule(), Some(Rule::ParentEscape));
    assert_eq!(attempt("/tmp/adapsis-should-never-exist.txt").rule(), Some(Rule::OutsideWorkspace));
    assert_eq!(attempt(".git/hooks/post-checkout").rule(), Some(Rule::WriteGitDir));
    assert_eq!(attempt("sub/.git/config").rule(), Some(Rule::WriteGitDir));
    assert_eq!(attempt(fixture.extra.join("new.md").to_str().unwrap()).rule(), Some(Rule::WriteReadOnlyRoot));
    assert_eq!(std::fs::read_dir(&fixture.outside).unwrap().count(), 1);
    assert_eq!(std::fs::read_dir(&fixture.extra).unwrap().count(), 0);
    assert!(!fixture.root.join("nowhere").exists() && !fixture.root.join("sub").exists() && !fixture.root.join(".git/hooks").exists());

    let exactly = "y".repeat(WRITE_MAX_BYTES);
    write(&fixture.workspace, &mut ledger, "max.txt", &exactly).unwrap();
    let error = write(&fixture.workspace, &mut ledger, "too-big.txt", &format!("{exactly}y")).unwrap_err();
    assert!(error.to_string().contains("the content is 2097153 bytes; one write accepts at most 2097152"), "{error}");
    assert!(!fixture.root.join("too-big.txt").exists());
}

/// A line longer than the read cap is never shown completely, so nothing on it is
/// replaced, however the file was read; other lines of the same file stay editable.
#[test]
fn edit_refuses_text_on_a_line_read_cannot_show_completely() {
    use super::limits::LINE_MAX_CHARS;
    let fixture = Fixture::plain();
    let long = format!("head {} tail", "x".repeat(LINE_MAX_CHARS));
    let exact = "é".repeat(LINE_MAX_CHARS);
    fixture.write("a.txt", format!("first\n{long}\nthird\n{exact}\r\nlast\n"));
    let mut ledger = ReadLedger::default();
    let shown = read(&fixture.workspace, "a.txt", None, None).unwrap();
    assert!(shown.render().contains("1 line(s) are longer than 2000 characters and were cut. read cannot show the rest of such a line, and edit refuses"), "{}", shown.render());
    ledger.note(&shown);
    let before = std::fs::read(fixture.root.join("a.txt")).unwrap();
    for (old, new) in [("head ", "HEAD "), (" tail", " TAIL"), ("first\nhead", "first\nHEAD"), ("tail\nthird", "tail\nTHIRD")] {
        let error = edit(&fixture.workspace, &mut ledger, "a.txt", old, new, false).unwrap_err().to_string();
        assert_eq!(
            error,
            "invalid argument: `a.txt` was not changed: the text to replace touches line 2, which is longer than 2000 characters. read shows only the beginning of such a line, so the rest of it is not known here; change that line with a command",
        );
    }
    assert!(edit(&fixture.workspace, &mut ledger, "a.txt", "x", "y", true).unwrap_err().to_string().contains("touches line 2"));
    assert_eq!(std::fs::read(fixture.root.join("a.txt")).unwrap(), before, "nothing was written");
    // Lines around it, and a line of exactly the cap (CRLF not counted), are editable.
    edit(&fixture.workspace, &mut ledger, "a.txt", "first\n", "FIRST\n", false).unwrap();
    edit(&fixture.workspace, &mut ledger, "a.txt", "third", "THIRD", false).unwrap();
    edit(&fixture.workspace, &mut ledger, "a.txt", "éé", "ee", false).unwrap_err();
    edit(&fixture.workspace, &mut ledger, "a.txt", "last", "LAST", false).unwrap();
    let after = std::fs::read_to_string(fixture.root.join("a.txt")).unwrap();
    assert!(after.starts_with("FIRST\nhead x") && after.contains("\nTHIRD\n") && after.ends_with("\r\nLAST\n"), "{}", &after[after.len() - 20..]);
    // The model matters, not the unit: replace_exact on the bytes.
    assert_eq!(replace_exact(&format!("{}\n", "a".repeat(LINE_MAX_CHARS + 1)), "a\n", "b\n", false).unwrap_err(), Failure::CutLine { line: 1 });
    assert!(replace_exact(&format!("{}\n", "a".repeat(LINE_MAX_CHARS)), "a\n", "b\n", false).is_ok());
}
