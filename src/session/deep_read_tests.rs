//! The session reader: no recursion limit, a depth cap checked before
//! deserializing, and a stack sized for the cap. Every other JSON reader keeps
//! serde_json's default limit.
use super::*;
use crate::session::Session;

fn base_json() -> String {
    let mut session = Session::new();
    for operation in crate::parser::parse("+module DeepRead\n+fn f (x:Bool)->Bool\n  +return NOT x\n+end\n").unwrap() {
        crate::validator::apply_and_validate(&mut session.program, &operation).unwrap();
    }
    serde_json::to_string(&session).unwrap()
}

/// The deepest depth not above `limit` that `session_json` can produce (it
/// nests in steps of two containers).
fn reachable(limit: usize) -> usize {
    limit - (limit - json_depth(&base_json())) % 2
}

/// A valid session JSON whose deepest point is exactly `depth` containers: a
/// real one-function session, with its `NOT x` expression nested textually (no
/// deep structure is ever built on the test's own stack).
fn session_json(depth: usize) -> String {
    let text = base_json();
    let open = r#"{"Unary":{"op":"Not","expr":"#;
    let start = text.find(open).expect("the serialized `NOT x`");
    let inner_start = start + open.len();
    let inner_end = inner_start + text[inner_start..].find("}}").unwrap() + 1;
    let inner = &text[inner_start..inner_end];
    assert_eq!(&text[inner_end..inner_end + 2], "}}");
    let base = json_depth(&text);
    assert!(depth >= base && (depth - base) % 2 == 0, "depth {depth} is not reachable from {base} in steps of 2");
    let extra = (depth - base) / 2;
    let deep = format!("{}{inner}{}", open.repeat(extra + 1), "}}".repeat(extra + 1));
    let result = format!("{}{deep}{}", &text[..start], &text[inner_end + 2..]);
    assert_eq!(json_depth(&result), depth);
    result
}

fn write(json: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.json");
    std::fs::write(&path, json).unwrap();
    (dir, path)
}

fn on_stack<T: Send + 'static>(bytes: usize, work: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new().stack_size(bytes).spawn(work).unwrap().join().expect("no overflow, no panic")
}

#[test]
fn session_reader_depth_scan_counts_containers_and_ignores_strings() {
    assert_eq!(json_depth(""), 0);
    assert_eq!(json_depth("1"), 0);
    assert_eq!(json_depth("[]"), 1);
    assert_eq!(json_depth(r#"{"a":[{"b":[1,2]}],"c":{}}"#), 4);
    assert_eq!(json_depth(r#"{"text":"[[[[{{{{ \" ]]]] \\","more":"\\\"[[["}"#), 1);
    assert_eq!(json_depth(r#"["\\",[["\"]"]]]"#), 3);
    // Malformed text never panics or underflows; the deserializer diagnoses it.
    assert_eq!(json_depth("]]]}}}[[["), 3);
    assert_eq!(json_depth(r#"{"unterminated":"[[["#), 1);
    // Linear and without recursion: a million levels on a small stack.
    let deep = on_stack(256 * 1024, || json_depth(&"[".repeat(1_000_000)));
    assert_eq!(deep, 1_000_000);
}

#[test]
fn session_reader_reads_far_beyond_the_default_recursion_limit() {
    // Ordinary thread (2 MiB): `Session::load` moves the work to its own thread.
    on_stack(2 * 1024 * 1024, || {
        for depth in [129, 409, 1009, MAX_DEPTH] {
            let depth = reachable(depth);
            let json = session_json(depth);
            assert!(serde_json::from_str::<Session>(&json).is_err(), "the default limit refuses {depth} levels");
            let (_dir, path) = write(&json);
            let session = Session::load(&path).unwrap_or_else(|error| panic!("{depth}: {error:#}"));
            assert!(session.program.get_function("DeepRead.f").is_some());
            // Written again (pretty, as `save` does) and read again: identical.
            let again = serde_json::to_string(&session).unwrap();
            assert_eq!(again, json, "{depth}");
        }
    });
}

#[test]
fn session_reader_refuses_a_file_deeper_than_its_cap_cleanly() {
    let depth = reachable(MAX_DEPTH) + 2;
    let json = session_json(depth);
    let (_dir, path) = write(&json);
    let error = Session::load(&path).map(|_| ()).unwrap_err().to_string();
    assert!(error.contains(&format!("nested {depth} levels deep")) && error.contains("the reader accepts 4096"), "{error}");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), json, "the file is untouched");
    // Far deeper: still a clean error, from the linear scan.
    let (_dir, path) = write(&format!("{}{}", "[".repeat(3_000_000), "]".repeat(3_000_000)));
    assert!(Session::load(&path).map(|_| ()).unwrap_err().to_string().contains("nested 3000000 levels deep"));
}

#[test]
fn session_reader_reports_malformed_json_with_its_position() {
    let (_dir, path) = write("{\"program\": {\n  \"modules\": [oops]}}");
    let error = Session::load(&path).map(|_| ()).unwrap_err().to_string();
    assert!(error.contains("invalid session encoding at line 2, column"), "{error}");
    let (_dir, path) = write(&format!("{} trailing", base_json()));
    assert!(Session::load(&path).map(|_| ()).unwrap_err().to_string().contains("invalid session encoding"));
}

/// The reader thread has 64 MiB. Reading a session at the depth CAP on a thread
/// with one eighth of that proves a margin of at least 8 for the production
/// path (`from_str`; the per-level figure in the module comment was measured
/// through `from_reader`).
#[test]
fn session_reader_stack_has_a_margin_of_at_least_eight_at_the_depth_cap() {
    let depth = reachable(MAX_DEPTH);
    let json = session_json(depth);
    let functions = on_stack(STACK_BYTES / 8, move || {
        let session: Session = parse(&json).unwrap();
        session.program.modules[0].functions.len()
    });
    assert_eq!(functions, 1);
}

/// Stack use per JSON level of the derived `Session` reader, measured through
/// a reader hook (the numbers quoted in the module comment).
#[test]
fn session_reader_stack_use_per_level_is_within_the_documented_figure() {
    struct Hook<'a> {
        data: &'a [u8],
        at: usize,
        lowest: usize,
    }
    impl std::io::Read for Hook<'_> {
        #[inline(never)]
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            let marker = 0u8;
            self.lowest = self.lowest.min(std::hint::black_box(&marker) as *const u8 as usize);
            let count = buffer.len().min(self.data.len() - self.at);
            buffer[..count].copy_from_slice(&self.data[self.at..self.at + count]);
            self.at += count;
            Ok(count)
        }
    }
    let used = |depth: usize| {
        let json = session_json(depth);
        on_stack(STACK_BYTES, move || {
            let anchor = 0u8;
            let top = std::hint::black_box(&anchor) as *const u8 as usize;
            let mut hook = Hook { data: json.as_bytes(), at: 0, lowest: usize::MAX };
            let mut deserializer = serde_json::Deserializer::from_reader(&mut hook);
            deserializer.disable_recursion_limit();
            let session: Session = serde::Deserialize::deserialize(&mut deserializer).unwrap();
            drop(session);
            top - hook.lowest
        })
    };
    let base = json_depth(&base_json());
    let (shallow, deep) = (base + 200, base + 2200);
    let per_level = (used(deep) - used(shallow)) / (deep - shallow);
    eprintln!("SESSION-READER stack per JSON level (nested unary expression): {per_level} bytes; {MAX_DEPTH} levels need about {} KiB of {} KiB", per_level * MAX_DEPTH / 1024, STACK_BYTES / 1024);
    // Declarations (statements, blocks) measured up to 619 bytes per level.
    assert!(per_level > 0 && per_level <= 1024, "{per_level}");
    assert!(1024 * MAX_DEPTH * 8 <= STACK_BYTES, "a level of 1 KiB at the cap still leaves a factor of 8");
}

/// Only the session reader is unlimited. Untrusted JSON keeps the default: 127
/// nested containers are read, 128 are refused, by every other reader.
#[test]
fn every_other_json_reader_keeps_the_default_recursion_limit() {
    use crate::eval::Value;
    let nested = |levels: usize| format!("{}{}", "[".repeat(levels), "]".repeat(levels));
    let program = crate::ast::Program::default();
    let call = |name: &str, arguments: &[Value]| {
        let mut env = crate::eval::Env::new_with_shared_interner(&program.shared_interner);
        crate::eval::eval_builtin_or_user(&program, name, arguments.to_vec(), &mut env)
    };
    let (fits, deep) = (nested(127), nested(128));
    assert!(matches!(call("json_valid", &[Value::string(&fits)]), Ok(Value::Bool(true))));
    assert!(matches!(call("json_valid", &[Value::string(&deep)]), Ok(Value::Bool(false))));
    for (name, arguments) in [
        ("json_kind", vec![Value::string(&deep), Value::string("")]),
        ("json_get", vec![Value::string(&deep), Value::string("0")]),
        ("json_array_len", vec![Value::string(&deep)]),
    ] {
        let error = call(name, &arguments).map(|_| ()).unwrap_err().to_string();
        assert!(error.contains("invalid JSON") && error.contains("recursion limit"), "{name}: {error}");
        let mut shallow = arguments.clone();
        shallow[0] = Value::string(&fits);
        assert!(call(name, &shallow).is_ok(), "{name} reads 127 levels");
    }
    assert!(serde_json::from_str::<serde_json::Value>(&deep).is_err());
    assert!(serde_json::from_str::<crate::session::replay::TypedTestCase>(&deep).is_err());
    assert!(serde_json::from_slice::<serde_json::Value>(deep.as_bytes()).is_err());
}

/// The typed-record budget: iterative, exact at its limit.
#[test]
fn typed_record_depth_budget_is_exact_and_its_checker_does_not_recurse() {
    use crate::session::replay::limits;
    let nested = |levels: usize| {
        let mut value = serde_json::Value::Null;
        for _ in 0..levels {
            value = serde_json::Value::Array(vec![value]);
        }
        value
    };
    // Dropping a deep `serde_json::Value` recurses; that is not under test, so
    // every value is taken apart with a loop.
    fn dismantle(mut value: serde_json::Value) {
        while let serde_json::Value::Array(mut inner) = value {
            value = inner.pop().unwrap_or(serde_json::Value::Null);
        }
    }
    let checked = move |levels: usize| {
        let value = nested(levels);
        let outcome = limits::check_for_tests(&value).map_err(|error| error.to_string());
        dismantle(value);
        outcome
    };
    // 256 KiB stack, values far beyond the budget: the check itself is a loop.
    let (outcomes, far) = on_stack(256 * 1024, move || {
        let at = checked(limits::MAX_JSON_DEPTH).is_ok();
        let above = checked(limits::MAX_JSON_DEPTH + 1).unwrap_err();
        let far = checked(200_000).is_err();
        ((at, above), far)
    });
    assert!(outcomes.0, "{} levels are within the budget", limits::MAX_JSON_DEPTH);
    assert_eq!(outcomes.1, "typed persistence structural budget exceeded");
    assert!(far);
    assert_eq!((limits::MAX_JSON_DEPTH, MAX_DEPTH), (2048, 4096));
}
