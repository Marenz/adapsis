//! Capture-state, escaped-JSON scratch and quadratic-set regressions from adv3.
use super::*;
use std::time::{Duration, Instant};

fn execute(name: &str, args: Vec<Value>, owner: bool) -> (Result<String>, usize, Duration) {
    let program = ast::Program::default();
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    let context = if owner { meter::RunContext::owner() } else {
        let limits = crate::run_limits::RunLimits::restricted();
        meter::RunContext { turn: meter::TurnBudget::for_limits(&limits), limits, policy: None }
    };
    let _guard = meter::install(context);
    let started = Instant::now();
    let (result, bytes) = crate::test_alloc::peak(|| eval_builtin_or_user(&program, name, args, &mut env).map(|v| v.to_string()));
    (result, bytes, started.elapsed())
}

#[test]
fn capture_scratch_and_first_search_are_bounded() {
    // The reviewer's first-search no-match family, including literal replacement:
    // Even find_iter may select capture-sized scratch: both paths must refuse
    // the heavy pattern BEFORE their first search, including literal replacement.
    for groups in [8usize, 9, 200, 400] {
        for repeats in [1usize, 5, 100, 500] {
            let pattern = format!("(?:{}){{{repeats}}}a{{{}}}[^a]", "(a?)".repeat(groups), groups * repeats);
            for len in [8192usize, 16384, 32768] {
                let (result, peak, elapsed) = execute("regex_match", vec![Value::string(&pattern), Value::string("a".repeat(len))], false);
                println!("BOOL_CAPTURE groups={groups} repeat={repeats} text={len}: peak={peak} time={elapsed:?} admitted={}", result.is_ok());
                assert!(peak < 3 * 1024 * 1024 && elapsed < Duration::from_secs(4));
                if groups > 8 {
                    let error = result.unwrap_err();
                    assert!(error.to_string().contains("capture tracking") || error.to_string().contains("too large"), "{error}");
                }
                for replacement in ["b", "$$", "$0", "$1", "${missing}"] {
                    let (result, peak, elapsed) = execute("regex_replace", vec![Value::string(&pattern), Value::string(replacement), Value::string("a".repeat(len))], false);
                    println!("CAPTURE groups={groups} repeat={repeats} text={len} repl={replacement}: peak={peak} time={elapsed:?} admitted={}", result.is_ok());
                    assert!(peak < 3 * 1024 * 1024, "{groups}/{repeats}/{len}/{replacement}: {peak}");
                    assert!(elapsed < Duration::from_secs(4), "{groups}/{repeats}/{len}/{replacement}: {elapsed:?}");
                    if groups > 8 {
                        let error = result.unwrap_err();
                        assert!(error.to_string().contains("capture tracking") || error.to_string().contains("too large"), "{error}");
                    }
                }
            }
        }
    }
    // Admission boundary: largest text the weighted work model allows for each
    // compiled-size class, at the capture ceiling and counted optional repetitions.
    let limits = crate::run_limits::RunLimits::restricted().regex.unwrap();
    let mut admitted = 0;
    for groups in [1usize, 4, 8] {
        for repeats in [5usize, 10, 20, 50, 100, 200, 300, 500] {
            let pattern = format!("(?:{}){{{repeats}}}a{{{}}}[^a]", "(a?)".repeat(groups), groups * repeats);
            let mut class = 1024usize;
            while regex::RegexBuilder::new(&pattern).size_limit(class).build().is_err() { class *= 2; }
            let slots = 2 * (groups + 1);
            if class * slots > limits.capture_state_bytes { continue; }
            let len = (limits.work / (class * slots) as u64) as usize;
            let len = len.min(1024 * 1024 - 512);
            let (result, peak, time) = execute("regex_match", vec![Value::string(&pattern), Value::string("a".repeat(len))], false);
            assert!(result.is_ok(), "{groups}/{repeats}/{class}/{len}: {result:?}");
            println!("BOOL_BOUNDARY groups={groups} repeats={repeats} class={class} len={len} peak={peak} time={time:?}");
            assert!(peak < 3 * 1024 * 1024 && time < Duration::from_secs(4));
            for replacement in ["b", "$0", "$1"] {
                let (result, peak, time) = execute("regex_replace", vec![Value::string(&pattern), Value::string(replacement), Value::string("a".repeat(len))], false);
                assert!(result.is_ok(), "{groups}/{repeats}/{class}/{len}: {result:?}");
                println!("CAPTURE_BOUNDARY groups={groups} repeats={repeats} class={class} len={len} peak={peak} time={time:?}");
                assert!(peak < 3 * 1024 * 1024 && time < Duration::from_secs(4));
                admitted += 1;
            }
        }
    }
    assert!(admitted >= 12);
}

/// Each condition independently refuses both public operations, not merely the
/// combined attack: group count alone, scratch alone, and weighted work alone.
#[test]
fn both_regex_builtins_discriminate_each_presearch_condition() {
    let class_of = |pattern: &str| {
        let mut class = 1024usize;
        while regex::RegexBuilder::new(pattern).size_limit(class).build().is_err() { class *= 2; }
        class
    };
    let invoke = |builtin: &str, pattern: &str, len: usize| {
        let mut args = vec![Value::string(pattern)];
        if builtin == "regex_replace" { args.push(Value::string("$1")); }
        args.push(Value::string("a".repeat(len)));
        execute(builtin, args, false).0
    };
    let limits = crate::run_limits::RunLimits::restricted().regex.unwrap();
    for builtin in ["regex_match", "regex_replace"] {
        let groups_only = "(a?)(a?)(a?)(a?)(a?)(a?)(a?)(a?)(a?)";
        assert!(class_of(groups_only) * 20 <= limits.capture_state_bytes);
        assert!(invoke(builtin, groups_only, 1).unwrap_err().to_string().contains("capture tracking"));
        let scratch_only = format!("(?:{}){{200}}a{{1600}}[^a]", "(a?)".repeat(8));
        let scratch = class_of(&scratch_only) * 18;
        assert!(scratch > limits.capture_state_bytes && (scratch as u64) < limits.work);
        assert!(invoke(builtin, &scratch_only, 1).unwrap_err().to_string().contains("capture tracking"));
        let weighted = format!("(?:{}){{20}}a{{80}}[^a]", "(a?)".repeat(4));
        let state = class_of(&weighted) * 10;
        assert!(state <= limits.capture_state_bytes);
        let len = (limits.work / state as u64) as usize;
        assert!(len < 1024 * 1024 - 512);
        assert!(invoke(builtin, &weighted, len).is_ok());
        assert!(invoke(builtin, &weighted, len + 1).unwrap_err().to_string().contains("capture tracking"));
    }
}

#[test]
fn replacement_dollars_preserve_exact_expansion_semantics() {
    for replacement in ["literal", "$$", "x$$y", "$", "x$", "$0", "$1", "$2", "$word", "${word}", "${1}x", "$$0", "$$$0", "$missing", "${missing}"] {
        let pattern = "(?P<word>a)(b?)";
        let text = "ab a cab";
        let expected = format!("\"{}\"", regex::Regex::new(pattern).unwrap().replace_all(text, replacement));
        for owner in [false, true] {
            let (actual, _, _) = execute("regex_replace", vec![Value::string(pattern), Value::string(replacement), Value::string(text)], owner);
            assert_eq!(actual.unwrap(), expected, "{replacement} owner={owner}");
        }
    }
    // Owner semantics are not restricted by the capture ceiling.
    let pattern = format!("{}b", "(a?)".repeat(20));
    let (result, _, _) = execute("regex_replace", vec![Value::string(pattern), Value::string("$0"), Value::string("aaab")], true);
    assert_eq!(result.unwrap(), "\"aaab\"");
}

#[test]
fn json_bound_includes_escaped_string_and_key_scratch() {
    for text in [
        format!("\"\\n{}\"", "a".repeat(500000)),
        format!("\"{}\\n\"", "a".repeat(500000)),
        format!("\"{}\"", "\\\"".repeat(400000)),
        format!("\"{}\"", "\\u00ff".repeat(150000)),
        format!("{{\"\\n{}\":0}}", "k".repeat(500000)),
        format!("{{\"{}\":0,\"{}\":1}}", "k".repeat(300000), "k".repeat(300000)),
    ] {
        let bound = bounded::json_tree_bound(&text);
        let (parsed, peak) = crate::test_alloc::peak(|| serde_json::from_str::<serde_json::Value>(&text));
        assert!(parsed.is_ok());
        assert!(peak <= bound, "text={} peak={peak} bound={bound}", text.len());
        let (result, metered_peak, _) = execute("json_valid", vec![Value::string(text)], false);
        assert!(result.unwrap_err().to_string().contains("a value exceeds 1 MiB"));
        assert!(metered_peak < 3 * 1024 * 1024);
    }
    for text in [format!("\"\\n{}\"", "a".repeat(100000)), format!("{{\"\\n{}\":0}}", "k".repeat(100000))] {
        let (result, peak, _) = execute("json_valid", vec![Value::string(&text)], false);
        assert_eq!(result.unwrap(), "true");
        assert!(peak <= bounded::json_tree_bound(&text));
    }
}

#[test]
fn set_construction_pays_for_pair_comparisons_before_scanning() {
    for count in [1000usize, 2000, 4000, 5900] {
        let (result, _, elapsed) = execute("set", (0..count as i64).map(Value::Int).collect(), false);
        assert!(result.unwrap_err().to_string().contains("more than 200000 steps"));
        assert!(elapsed < Duration::from_secs(1));
    }
    let (result, _, _) = execute("set", (0..100).map(Value::Int).collect(), false);
    assert!(result.is_ok());
    // Owner has no step budget: the representation and output are unchanged.
    let (result, _, _) = execute("set", (0..1000).map(Value::Int).collect(), true);
    assert!(result.is_ok());
}
