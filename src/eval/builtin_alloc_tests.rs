//! What ONE pure builtin call may allocate and how long it may run (#129, review findings
//! F1 and F3). The meter admits arguments and measures results; this suite measures the
//! real heap use of each call in between, with the counting allocator, for every pure
//! builtin and hostile arguments at the argument cap.
//!
//! The suite runs in a child process: it allocates tens of megabytes, and large
//! allocations in the shared test process move later JIT mappings (#96).
use super::*;
use crate::run_limits::RunLimits;
use crate::test_alloc::peak;
use std::time::{Duration, Instant};

/// What one call may allocate beyond its (already existing) arguments, as a multiple of
/// the single-value cap. MEASURED worst case: 2.27, for `concat` (see the test output with
/// `--nocapture`). The parts: a copy of an argument that is shared (1x), growth slack of
/// a vector or string that is extended (up to 1x), the result (1x); regex compilation and
/// its caches stay below 1.5 MiB under the restricted limits whatever the cap.
const PEAK_MULTIPLE: usize = 3;
/// One call, wall clock. The regex work limit targets about one second on this machine;
/// the margin is for a loaded machine.
const CALL_TIME: Duration = Duration::from_secs(4);

fn cap() -> usize {
    RunLimits::restricted().value_bytes
}

fn context() -> meter::RunContext {
    meter::RunContext { limits: RunLimits::restricted(), policy: None, turn: meter::TurnBudget::new(None) }
}

fn text(piece: &str, bytes: usize) -> Value {
    Value::string(piece.repeat(bytes / piece.len()))
}

/// Values at the argument cap (each just under 1 MiB in the meter's accounting).
fn large() -> Vec<(&'static str, Value)> {
    let slot = meter::SLOT + meter::HEAP;
    let scalars = (cap() - 2 * slot) / meter::SLOT;
    vec![
        ("1 MiB of a", text("a", cap() - 2 * slot)),
        ("512 KiB text", text("0123456789abcdef", 512 * 1024)),
        ("400k lines", text("a\n", 800_000)),
        ("json one-entry objects", Value::string(format!("[{}0]", "{\"\":0},".repeat(130_000)))),
        ("json nested arrays", Value::string(format!("[{}0]", "[0],".repeat(230_000)))),
        ("900k quotes", text("\"", 900_000)),
        ("400k group references", text("$0", 800_000)),
        ("4,700 empty strings", Value::list(vec![Value::string(""); (cap() - 2 * slot) / slot])),
        ("5,900 ints", Value::list((0..scalars as i64).rev().map(Value::Int).collect())),
        ("5,900 floats", Value::list(vec![Value::Float(1e300); scalars])),
        ("100 strings of 10k", Value::list((0..100).map(|i| Value::string(format!("{i}{}", "x".repeat(10_000)))).collect())),
        ("2,900 map entries", Value::Map(Arc::new((0..(scalars / 2) as i64).map(|i| (Value::Int(i), Value::Int(i))).collect()))),
        ("5,900 set items", Value::Set(Arc::new((0..scalars as i64).map(Value::Int).collect()))),
    ]
}

fn small() -> Vec<(&'static str, Value)> {
    vec![
        ("empty", Value::string("")),
        ("a", Value::string("a")),
        ("comma", Value::string(",")),
        ("newline", Value::string("\n")),
        ("8 group references", Value::string("$0$0$0$0$0$0$0$0")),
        ("whole-text pattern", Value::string("^(?s).+$")),
        ("dot", Value::string(".")),
        ("unicode word", Value::string("\\w+")),
        ("counted pattern", Value::string("(?:a?){1000}a{1000}[^a]")),
        ("0", Value::Int(0)),
        ("1", Value::Int(1)),
        ("-1", Value::Int(-1)),
        ("max", Value::Int(i64::MAX)),
        ("min", Value::Int(i64::MIN)),
        ("2^40", Value::Int(1 << 40)),
        ("half", Value::Float(0.5)),
        ("nan", Value::Float(f64::NAN)),
        ("true", Value::Bool(true)),
        ("none", Value::None),
    ]
}

struct Worst {
    peak: usize,
    time: Duration,
    peak_case: String,
    time_case: String,
}

fn measure(name: &'static str, labels: &[&str], args: Vec<Value>, worst: &mut std::collections::BTreeMap<&'static str, Worst>) {
    let program = ast::Program::default();
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    // A fresh block per call: a tripped limit is sticky.
    let _meter = meter::install(context());
    let ((_, time), bytes) = peak(|| {
        let started = Instant::now();
        let outcome = eval_builtin_or_user(&program, name, args, &mut env);
        // The result (if any) is still alive here and counts.
        let elapsed = started.elapsed();
        (outcome.is_ok(), elapsed)
    });
    let entry = worst.entry(name).or_insert(Worst { peak: 0, time: Duration::ZERO, peak_case: String::new(), time_case: String::new() });
    if bytes > entry.peak {
        entry.peak = bytes;
        entry.peak_case = labels.join(" | ");
    }
    if time > entry.time {
        entry.time = time;
        entry.time_case = labels.join(" | ");
    }
}

#[test]
fn no_pure_builtin_call_allocates_or_runs_beyond_its_bounds() {
    const NAME: &str = "eval::builtin_alloc_tests::no_pure_builtin_call_allocates_or_runs_beyond_its_bounds";
    if !crate::logging::tests::child() {
        let captured = crate::logging::tests::capture(NAME, false);
        // Surface the measured table when run with --nocapture.
        for line in captured.lines().filter(|line| line.starts_with("ALLOC")) {
            println!("{line}");
        }
        return;
    }
    let (large, small) = (large(), small());
    let all: Vec<&(&str, Value)> = large.iter().chain(small.iter()).collect();
    // Arity three uses a reduced pool; every position still sees large and small values.
    let three: Vec<&(&str, Value)> = [0usize, 1, 6, 8].iter().map(|index| &large[*index])
        .chain([0usize, 1, 4, 5, 8, 9, 11, 14].iter().map(|index| &small[*index]))
        .collect();
    let mut worst = std::collections::BTreeMap::new();
    let mut calls = 0usize;
    for name in crate::builtins::pure::restricted_block_builtins() {
        measure(name, &[], vec![], &mut worst);
        for a in &all {
            measure(name, &[a.0], vec![a.1.clone()], &mut worst);
            for b in &all {
                measure(name, &[a.0, b.0], vec![a.1.clone(), b.1.clone()], &mut worst);
                calls += 1;
            }
        }
        for a in &three {
            for b in &three {
                for c in &three {
                    measure(name, &[a.0, b.0, c.0], vec![a.1.clone(), b.1.clone(), c.1.clone()], &mut worst);
                    calls += 1;
                }
            }
        }
    }
    assert!(calls > 150_000, "{calls}");
    let mut failures = Vec::new();
    for (name, entry) in &worst {
        println!(
            "ALLOC {name:16} peak {:>9} B = {:.2} x cap ({}) | slowest {:?} ({})",
            entry.peak, entry.peak as f64 / cap() as f64, entry.peak_case, entry.time, entry.time_case
        );
        if entry.peak > PEAK_MULTIPLE * cap() {
            failures.push(format!("{name} allocated {} bytes ({:.1} x cap) for ({})", entry.peak, entry.peak as f64 / cap() as f64, entry.peak_case));
        }
        if entry.time > CALL_TIME {
            failures.push(format!("{name} ran {:?} for ({})", entry.time, entry.time_case));
        }
    }
    assert!(failures.is_empty(), "a pure builtin exceeded its bounds; give it an output bound before it allocates (eval/meter.rs::reserve) and classify it in builtins/pure.rs::COSTS:\n{}", failures.join("\n"));
    assert_eq!(worst.len(), crate::builtins::pure::restricted_block_builtins().len());
}

/// The reviewer's amplifications (F1), each as ONE call: refused before the output is
/// built, with the allocation to show for it.
#[test]
fn amplifying_calls_are_refused_before_they_allocate() {
    const NAME: &str = "eval::builtin_alloc_tests::amplifying_calls_are_refused_before_they_allocate";
    if !crate::logging::tests::child() {
        crate::logging::tests::capture(NAME, false);
        return;
    }
    let separator = text("0123456789abcdef", 512 * 1024);
    let empty_parts = |count: usize| Value::list(vec![Value::string(""); count]);
    let references = |count: usize| Value::string("$0".repeat(count));
    let cases: Vec<(&str, &'static str, Vec<Value>)> = vec![
        ("join, 4,000 parts x 512 KiB", "join", vec![empty_parts(4_000), separator.clone()]),
        ("join, 30,000 parts x 512 KiB", "join", vec![empty_parts(4_700), text("0123456789abcdef", 1_000_000)]),
        ("regex_replace, every a becomes 512 KiB", "regex_replace", vec![Value::string("a"), separator.clone(), text("a", 4_000)]),
        ("regex_replace, one match expanded 4,000 times", "regex_replace", vec![Value::string("^(?s).+$"), references(4_000), separator.clone()]),
        ("regex_replace, named group expanded", "regex_replace", vec![Value::string("(?P<all>(?s).+)"), Value::string("${all}".repeat(4_000)), separator.clone()]),
        ("split by the empty string", "split", vec![separator.clone(), Value::string("")]),
        ("split into 400,000 lines", "split", vec![text("a\n", 800_000), Value::string("\n")]),
        ("concat of rendered floats", "concat", vec![Value::list(vec![Value::Float(1e300); 5_000]), Value::string("x")]),
        ("to_string of floats", "to_string", vec![Value::list(vec![Value::Float(-1e300); 5_900])]),
        ("join of floats", "join", vec![Value::list(vec![Value::Float(1e300); 5_900]), Value::string(", ")]),
        ("Err of floats", "Err", vec![Value::list(vec![Value::Float(1e300); 5_900])]),
        ("json: 130,000 one-entry objects", "json_valid", vec![Value::string(format!("[{}0]", "{\"\":0},".repeat(130_000)))]),
        ("json: nested arrays", "json_kind", vec![Value::string(format!("[{}0]", "[0],".repeat(230_000))), Value::string("0")]),
        ("json: get of the root", "json_get", vec![Value::string(format!("[{}0]", "0,".repeat(500_000))), Value::string("")]),
        ("json: array length", "json_array_len", vec![Value::string(format!("[{}0]", "{},".repeat(300_000)))]),
        ("map_entries of a full map", "map_entries", vec![Value::Map(Arc::new((0..2_900).map(|i| (Value::Int(i), Value::Int(i))).collect()))]),
        ("json_escape of quotes", "json_escape", vec![text("\"", 900_000)]),
        ("base64 of 1 MiB", "base64_encode", vec![text("a", 1_000_000)]),
    ];
    let program = ast::Program::default();
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    for (label, name, args) in cases {
        let _meter = meter::install(context());
        let (outcome, bytes) = peak(|| eval_builtin_or_user(&program, name, args, &mut env).map(|_| ()));
        let error = outcome.expect_err(label).to_string();
        assert_eq!(error, "run stopped: a value exceeds 1 MiB (limit for a single value in this conversation)", "{label}");
        println!("ALLOC refused {label}: peak {bytes} bytes");
        assert!(bytes <= 2 * cap(), "{label}: {bytes} bytes were allocated before the refusal");
    }
    // The owner's cap has the same guard: 200 parts x 512 KiB would be 100 MiB.
    let _meter = meter::install(meter::RunContext::owner());
    let (outcome, bytes) = peak(|| eval_builtin_or_user(&program, "join", vec![empty_parts(200), separator.clone()], &mut env).map(|_| ()));
    assert_eq!(outcome.unwrap_err().to_string(), "run stopped: a value exceeds 64 MiB (limit for a single value in this conversation)");
    assert!(bytes < cap(), "{bytes}");
}

/// The bounded paths give exactly the plain results when they admit the call.
#[test]
fn bounded_builtins_return_the_same_results_as_before() {
    let program = ast::Program::default();
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    let s = Value::string;
    let cases: Vec<(&str, Vec<Value>)> = vec![
        ("join", vec![Value::list(vec![s("a"), Value::Int(2), Value::Float(1.5), Value::list(vec![s("q")]), Value::None]), s(", ")]),
        ("join", vec![Value::list(vec![]), s("xyz")]),
        ("join", vec![Value::list(vec![s("only")]), s("xyz")]),
        ("concat", vec![s("a"), Value::Int(1), Value::list(vec![s("x"), Value::Bool(true)]), Value::Float(2.5)]),
        ("to_string", vec![Value::Map(Arc::new(vec![(s("k"), Value::Ok(Box::new(Value::Int(1))))]))]),
        ("split", vec![s("a,b,,c"), s(",")]),
        ("split", vec![s("héllo"), s("")]),
        ("split", vec![s(""), s("x")]),
        ("regex_replace", vec![s("([a-z]+)@([a-z]+)"), s("$2 at $1 ${1}x $$ $9"), s("ann@home, bob@work")]),
        ("regex_replace", vec![s("a*"), s("-"), s("baaac")]),
        ("regex_replace", vec![s(""), s("."), s("héy")]),
        ("regex_replace", vec![s("x"), s("y"), s("no match")]),
        ("regex_replace", vec![s("(?P<w>o)"), s("[$w]"), s("foo")]),
        ("regex_match", vec![s("^\\d+$"), s("12345")]),
        ("regex_match", vec![s("("), s("x")]),
        ("regex_match", vec![s("\\bwörd\\b"), s("ein wörd hier")]),
        ("json_get", vec![s("{\"a\":[1,{\"b\":null,\"c\":\"x\"}],\"n\":1.5}"), s("a.1")]),
        ("json_kind", vec![s("{\"a\":[1]}"), s("a.0")]),
        ("json_valid", vec![s("[1,2")]),
        ("json_array_len", vec![s("[1,2,3]")]),
        ("json_escape", vec![s("a\"b\\c\nd\te\r")]),
        ("base64_encode", vec![s("hello wörld")]),
        ("map_entries", vec![Value::Map(Arc::new(vec![(s("k"), Value::Int(1)), (Value::Int(2), s("v"))]))]),
        ("Err", vec![Value::list(vec![Value::Int(1)])]),
        ("error_suggest", vec![Value::list(vec![s("undefined variable")])]),
    ];
    for (name, args) in cases {
        let plain = eval_builtin_or_user(&program, name, args.clone(), &mut env).map(|value| value.to_string()).map_err(|error| error.to_string());
        for context in [context(), meter::RunContext::owner()] {
            let _meter = meter::install(context);
            let metered = eval_builtin_or_user(&program, name, args.clone(), &mut env).map(|value| value.to_string()).map_err(|error| error.to_string());
            assert_eq!(metered, plain, "{name}({args:?})");
        }
    }
    // An invalid pattern keeps its old answers in a block too.
    let _meter = meter::install(context());
    assert_eq!(eval_builtin_or_user(&program, "regex_match", vec![s("("), s("x")], &mut env).unwrap().to_string(), "false");
    assert!(eval_builtin_or_user(&program, "regex_replace", vec![s("("), s("x"), s("y")], &mut env).unwrap_err().to_string().contains("invalid pattern"));
}

/// The JSON tree bound is an upper bound of what parsing really allocates.
#[test]
fn the_json_tree_bound_covers_the_parsed_tree() {
    const NAME: &str = "eval::builtin_alloc_tests::the_json_tree_bound_covers_the_parsed_tree";
    if !crate::logging::tests::child() {
        crate::logging::tests::capture(NAME, false);
        return;
    }
    let families: Vec<(&str, String)> = vec![
        ("zeros", format!("[{}0]", "0,".repeat(200_000))),
        ("empty arrays", format!("[{}[]]", "[],".repeat(150_000))),
        ("empty objects", format!("[{}{{}}]", "{},".repeat(150_000))),
        ("one-entry objects", format!("[{}{{}}]", "{\"\":0},".repeat(60_000))),
        ("two-entry objects", format!("[{}{{}}]", "{\"a\":0,\"b\":0},".repeat(40_000))),
        ("wide object", format!("{{{}\"z\":0}}", (0..60_000).map(|i| format!("\"{i:x}\":0,")).collect::<String>())),
        ("nested pairs", format!("[{}[0]]", "[0],".repeat(120_000))),
        ("five-element arrays", format!("[{}[0]]", "[0,0,0,0,0],".repeat(50_000))),
        ("strings", format!("[{}\"\"]", "\"ab\",".repeat(100_000))),
        ("deep", format!("{}0{}", "[".repeat(100), "]".repeat(100))),
        ("ordinary", format!("[{}{{}}]", "{\"title\":\"A result\",\"url\":\"https://example.org/a/b\",\"snippet\":\"Some words here.\",\"rank\":3},".repeat(3_000))),
    ];
    for (label, text) in families {
        let bound = bounded::json_tree_bound(&text);
        let (valid, bytes) = peak(|| serde_json::from_str::<serde_json::Value>(&text).is_ok());
        assert!(valid, "{label}");
        println!("ALLOC json {label}: text {} B, parsed peak {bytes} B ({:.1}x), bound {bound} B ({:.1}x)", text.len(), bytes as f64 / text.len() as f64, bound as f64 / text.len() as f64);
        assert!(bytes <= bound, "{label}: parsing allocated {bytes} bytes, the bound says {bound}");
    }
}

/// Finding F3: with the work limit, the worst allowed regex call stays near a second
/// and the reviewer's 123-second call is refused before it starts.
#[test]
fn regex_calls_are_bounded_in_time_for_restricted_blocks() {
    let program = ast::Program::default();
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    let s = |value: &str| Value::string(value);
    let counted = |n: usize| Value::string(format!("(?:a?){{{n}}}a{{{n}}}[^a]"));
    let run = |name: &str, args: Vec<Value>, env: &mut Env| {
        let _meter = meter::install(context());
        let started = Instant::now();
        let outcome = eval_builtin_or_user(&program, name, args, env).map(|value| value.to_string()).map_err(|error| error.to_string());
        (outcome, started.elapsed())
    };
    // The reviewer's call (123 s) and smaller ones: refused in well under a second.
    for (n, text_bytes) in [(10_000, 512 * 1024), (1_000, 512 * 1024), (1_000, 64 * 1024), (5_000, 16 * 1024)] {
        let (outcome, time) = run("regex_match", vec![counted(n), text("a", text_bytes)], &mut env);
        let error = outcome.unwrap_err();
        assert!(error.contains("this pattern is too large for a text of") && error.contains("[A-Za-z0-9_]"), "{n}/{text_bytes}: {error}");
        assert!(time < Duration::from_millis(900), "n={n}: refusal took {time:?}");
        let (outcome, _) = run("regex_replace", vec![counted(n), s(""), text("a", text_bytes)], &mut env);
        assert!(outcome.unwrap_err().contains("too large for a text"));
    }
    // The worst calls the limit still admits (pattern size class x text length at the
    // limit): about a second each on this machine.
    let mut slowest = Duration::ZERO;
    for (n, text_bytes) in [(1_000, 16 * 1024), (500, 32 * 1024), (200, 64 * 1024), (20, 512 * 1024 - 512)] {
        let (outcome, time) = run("regex_match", vec![counted(n), text("a", text_bytes)], &mut env);
        println!("REGEX admitted n={n} text={text_bytes}: {outcome:?} in {time:?}");
        assert_eq!(outcome, Ok("false".to_string()), "n={n} text={text_bytes}");
        slowest = slowest.max(time);
    }
    assert!(slowest < CALL_TIME, "the slowest admitted regex call took {slowest:?}");
    // Ordinary patterns on large texts are admitted.
    for (pattern, text_bytes) in [("[a-z0-9._%+-]+@[a-z0-9.-]+\\.[a-z]{2,}", 1_000_000), ("\\d+", 400_000), ("\\w+", 32_000), ("(\\d{4})-(\\d{2})", 8_000)] {
        let (outcome, time) = run("regex_match", vec![s(pattern), text("lorem ipsum 42 ", text_bytes)], &mut env);
        assert!(outcome.is_ok(), "{pattern}: {outcome:?}");
        assert!(time < Duration::from_millis(900), "{pattern}: {time:?}");
    }
    // A pattern that rescans the text for every match is stopped between matches.
    let (outcome, time) = run("regex_replace", vec![s("a(?:.*z)?"), s("b"), text("a", 1_000_000)], &mut env);
    let error = outcome.unwrap_err();
    assert!(error.contains("regex_replace: stopped after 1 second(s)"), "{error}");
    assert!(time < CALL_TIME, "{time:?}");
    // The owner has no regex work limit: the same shape is admitted (kept small here).
    let _meter = meter::install(meter::RunContext::owner());
    assert!(eval_builtin_or_user(&program, "regex_match", vec![counted(300), text("a", 8 * 1024)], &mut env).is_ok());
}

/// Regex work is charged to the step budget, so repeating a call is bounded too.
#[test]
fn regex_work_drains_the_step_budget() {
    let program = ast::Program::default();
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    let _meter = meter::install(context());
    let pattern = Value::string("(?:a?){200}a{200}[^a]");
    let subject = text("a", 64 * 1024);
    let mut calls = 0;
    let error = loop {
        match eval_builtin_or_user(&program, "regex_match", vec![pattern.clone(), subject.clone()], &mut env) {
            Ok(_) => calls += 1,
            Err(error) => break error.to_string(),
        }
        assert!(calls < 100, "the step budget never ran out");
    };
    assert_eq!(error, "run stopped: it executed more than 200000 steps (limit for this conversation)");
    // Each call is charged its worst case (class 32,768 or 65,536 x 256 KiB of 8 GiB^2,
    // at 50,000 steps per second), so a handful of calls is the whole budget.
    assert!((2..=32).contains(&calls), "{calls} calls were admitted");
}

/// Argument bytes are charged to the step budget: linear builtins over large values
/// cannot be repeated 200,000 times.
#[test]
fn large_arguments_cost_steps() {
    let program = ast::Program::default();
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    let _meter = meter::install(context());
    let big = text("a", 1_000_000);
    let mut calls = 0u64;
    let error = loop {
        match eval_builtin_or_user(&program, "contains", vec![big.clone(), Value::string("zzz")], &mut env) {
            Ok(_) => calls += 1,
            Err(error) => break error.to_string(),
        }
    };
    assert!(error.contains("more than 200000 steps"), "{error}");
    // 1 + 1 MiB / 4096 = 245 steps per call.
    assert!((700..=900).contains(&calls), "{calls}");
}
