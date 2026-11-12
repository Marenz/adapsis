//! The review's reproductions (#129 restricted `run`, findings F1 to F4 and the turn
//! clock), through the real tool path. Each ends in a refusal and the next call works.
use super::run_tests::{envelope, fixture, memory_io, output, run_source, IoLog};
use super::*;
use serde_json::json;
use std::time::{Duration, Instant};

const EXTRA: &str = "\
+module NtHard
+fn deep (n:Int)->Int
  +if n > 0
    +return NtHard.deep(n - 1)
  +end
  +return 0
+end
+fn deep_io (n:Int)->Int [io,async]
  +if n > 0
    +await below:Int = NtHard.deep_io(n - 1)
    +return below
  +end
  +return 0
+end
+fn ping (n:Int)->Int
  +if n > 0
    +return NtHard.pong(n - 1)
  +end
  +return 0
+end
+fn pong (n:Int)->Int
  +if n > 0
    +return NtHard.ping(n - 1)
  +end
  +return 0
+end
+type Step = Go(Int) | Stop
+fn arms (n:Int)->Int
  +let items:List<Int> = list(n)
  +each items item:Int
    +let step:Step = Go(item)
    +match step
    +case Go(left)
      +if left > 0
        +return NtHard.arms(left - 1)
      +end
      +return 0
    +case Stop
      +return 0
    +end
  +end
  +return 0
+end
";

async fn speaker(model: &str, guest_tools: Option<&[&str]>) -> (AppConfig, WorkingSet, Program, IoLog) {
    let mut program = build_program(&format!("{}{EXTRA}", super::run_tests::LIBRARY)).await;
    program.require_modules = false; // evaluate's test gate is not the subject here
    let mut cfg = config_for(model, AccessLevel::Full, super::run_tests::permissions(), program.clone());
    let tools = guest_tools.map(|tools| tools.iter().map(|tool| tool.to_string()).collect::<Vec<_>>());
    if let Some(tools) = &tools {
        Arc::make_mut(&mut cfg.permission_config).guest_profiles.insert(model.to_string(), tools.clone());
    }
    cfg.turn = Some(TurnIdentity {
        context: "telegram:group:-7".into(),
        principal: "telegram:user:100000778".into(),
        may_write: model == "writer",
        guest_tools: tools,
    });
    let log = memory_io(&mut cfg, "file text");
    let ws = cfg.snapshot_working_set().await;
    (cfg, ws, program, log)
}

async fn guest() -> (AppConfig, WorkingSet, Program, IoLog) {
    let speaker = speaker("visitor", Some(&[])).await;
    assert_eq!(speaker_class(&speaker.0), SpeakerClass::Guest(vec![]));
    speaker
}

async fn viewer() -> (AppConfig, WorkingSet, Program, IoLog) {
    let speaker = speaker("viewer", None).await;
    assert_eq!(speaker_class(&speaker.0), SpeakerClass::ExecuteOnly);
    speaker
}

/// `doublings` doublings of a 16-byte string.
fn grow(name: &str, seed: &str, doublings: usize) -> String {
    format!("{name}: str = \"{seed}\"\n{name}_i: int = 0\nwhile {name}_i < {doublings}:\n    {name} = concat({name}, {name})\n    {name}_i = {name}_i + 1\n")
}

fn peak_rss_mb() -> usize {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status.lines().find(|line| line.starts_with("VmHWM:")).unwrap();
    line.split_whitespace().nth(1).unwrap().parse::<usize>().unwrap() / 1024
}

/// F1: one builtin call that would allocate K times its input. K = 30,000 used to abort
/// the process; K = 4,000 peaked at 2 GB before it was refused.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_amplifying_builtin_call_is_refused_before_it_allocates() {
    const NAME: &str = "api::native_tools::tests::run_hardening_tests::one_amplifying_builtin_call_is_refused_before_it_allocates";
    if !crate::logging::tests::child() {
        crate::logging::tests::capture(NAME, false);
        return;
    }
    let separator = grow("sep", "0123456789abcdef", 15); // 512 KiB
    let many_a = |doublings: usize| grow("a", "aaaaaaaaaaaaaaaa", doublings);
    let mut cases: Vec<(String, String)> = Vec::new();
    for k in [200usize, 4_000, 30_000] {
        cases.push((format!("join k={k}"), format!("{separator}{}parts: list[str] = split(substring(a, 0, {k}), \"a\")\nout: str = join(parts, sep)\nreturn len(out)\n", many_a(11))));
        cases.push((format!("regex k={k}"), format!("{separator}{}out: str = regex_replace(\"a\", sep, substring(a, 0, {k}))\nreturn len(out)\n", many_a(12))));
        cases.push((format!("regex group k={k}"), format!("{separator}rep: str = \"$0$0$0$0$0$0$0$0\"\nwhile len(rep) < {}:\n    rep = concat(rep, rep)\nout: str = regex_replace(\"^(?s).+$\", rep, sep)\nreturn len(out)\n", 2 * k)));
    }
    cases.push(("split by the empty string".into(), format!("{separator}parts: list[str] = split(sep, \"\")\nreturn len(parts)\n")));
    cases.push(("to_string of rendered floats".into(), "xs: list[float] = [1.0e300]\ni: int = 0\nwhile i < 5000:\n    xs = push(xs, 1.0e300)\n    i = i + 1\nreturn to_string(xs)\n".into()));
    let before = peak_rss_mb();
    for (cfg, mut ws, baseline, _log) in [guest().await, viewer().await] {
        for (label, source) in &cases {
            let started = Instant::now();
            let result = run_source(&cfg, &mut ws, &baseline, source).await;
            assert!(result.execution.has_errors, "{label}: {}", &result.content[..result.content.len().min(300)]);
            let text = output(&result);
            assert!(text.contains("run stopped: a value exceeds 1 MiB") || text.contains("it produced more than 32 MiB"), "{label}: {text}");
            assert!(started.elapsed() < Duration::from_secs(10), "{label}: {:?}", started.elapsed());
            let next = run_source(&cfg, &mut ws, &baseline, "return 1 + 1\n").await;
            assert_eq!(output(&next), "2", "{label}: the next call must work");
        }
    }
    // The whole series, including what used to need 2 GB and 15 GB, stays small.
    let grown = peak_rss_mb().saturating_sub(before);
    println!("peak RSS grew by {grown} MB over the amplification series");
    assert!(grown < 200, "peak RSS grew by {grown} MB");
}

/// F3: one regex call that ran 123 seconds against the 60-second clock.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_long_regex_call_is_refused_before_it_starts() {
    for (cfg, mut ws, baseline, _log) in [guest().await, viewer().await] {
        let text = grow("text", "aaaaaaaaaaaaaaaa", 15);
        for call in ["hit: bool = regex_match(pat, text)\nreturn hit\n", "out: str = regex_replace(pat, \"\", text)\nreturn len(out)\n"] {
            let source = format!("{text}pat: str = \"(?:a?){{10000}}a{{10000}}[^a]\"\n{call}");
            let started = Instant::now();
            let result = run_source(&cfg, &mut ws, &baseline, &source).await;
            assert!(result.execution.has_errors, "{}", result.content);
            assert!(output(&result).contains("this pattern is too large for a text of 524288 bytes"), "{}", result.content);
            assert!(started.elapsed() < Duration::from_secs(5), "{:?}", started.elapsed());
        }
        // Repeating an admitted heavy call drains the step budget instead of the clock.
        let source = format!("{}pat: str = \"(?:a?){{200}}a{{200}}[^a]\"\nn: int = 0\nwhile n < 1000:\n    hit: bool = regex_match(pat, text)\n    n = n + 1\nreturn n\n", grow("text", "aaaaaaaaaaaaaaaa", 12));
        let started = Instant::now();
        let result = run_source(&cfg, &mut ws, &baseline, &source).await;
        assert!(output(&result).contains("it executed more than 200000 steps"), "{}", result.content);
        assert!(started.elapsed() < Duration::from_secs(10), "{:?}", started.elapsed());
        let ordinary = run_source(&cfg, &mut ws, &baseline, "return regex_replace(\"(\\\\d+)-(\\\\d+)\", \"$2/$1\", \"10-20 and 30-40\")\n").await;
        assert_eq!(output(&ordinary), "20/10 and 40/30", "{}", ordinary.content);
    }
}

/// adv3: first capture search used 130 MB / 40 s before any between-match clock.
#[tokio::test]
async fn capture_heavy_guest_replacement_is_refused_before_first_search() {
    let (cfg, mut ws, baseline, _) = guest().await;
    for groups in [200usize, 400] {
        let pattern = format!("(?:{}){{5}}a{{{}}}[^a]", "(a?)".repeat(groups), groups * 5);
        let source = format!("{}return regex_match({pattern:?}, text)\n", grow("text", "aaaaaaaaaaaaaaaa", 10));
        let started = Instant::now();
        let result = run_source(&cfg, &mut ws, &baseline, &source).await;
        assert!(result.execution.has_errors && output(&result).contains("regex_match: capture tracking is too expensive"), "{}", result.content);
        assert!(started.elapsed() < Duration::from_secs(2));
        assert_eq!(output(&run_source(&cfg, &mut ws, &baseline, "return 1\n").await), "1");
        for replacement in ["b", "$$", "$0", "$1", "${missing}"] {
            let source = format!("{}return regex_replace({pattern:?}, {replacement:?}, text)\n", grow("text", "aaaaaaaaaaaaaaaa", 10));
            let started = Instant::now();
            let result = run_source(&cfg, &mut ws, &baseline, &source).await;
            assert!(result.execution.has_errors && output(&result).contains("capture tracking is too expensive"), "{}", result.content);
            assert!(started.elapsed() < Duration::from_secs(2));
            assert_eq!(output(&run_source(&cfg, &mut ws, &baseline, "return 1\n").await), "1");
        }
    }
}

/// F2: the nesting guard, from guest and execute-only blocks. 1,500 unary minus signs
/// used to abort the process.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deeply_nested_blocks_are_refused_for_restricted_speakers() {
    // On a worker-sized stack, as in production.
    tokio::spawn(async {
        for (cfg, mut ws, baseline, log) in [guest().await, viewer().await] {
            for depth in [1_500usize, 3_000] {
                for (label, source) in [
                    ("unary chain", format!("x: int = {}1\nreturn x\n", "-".repeat(depth))),
                    ("parentheses", format!("x: int = {}1{}\nreturn x\n", "(".repeat(depth), ")".repeat(depth))),
                    ("nested calls", format!("x: int = {}1{}\nreturn x\n", "abs(".repeat(depth), ")".repeat(depth))),
                    ("nested lists", format!("x: int = len({}1{})\nreturn x\n", "[".repeat(depth), "]".repeat(depth))),
                    ("operator chain", format!("x: int = 1{}\nreturn x\n", " + 1".repeat(depth))),
                    ("not chain", format!("x: bool = {}True\nreturn x\n", "not ".repeat(depth))),
                ] {
                    assert!(source.len() <= 32_000, "{label} at {depth}");
                    let arguments = json!({"source": source}).to_string();
                    let preflight = validate_native_tool_authorized("run", &arguments, &cfg).unwrap_err();
                    assert!(preflight.contains("nesting too deep: more than 100 levels"), "{label} at {depth}: {preflight}");
                    let result = run_source(&cfg, &mut ws, &baseline, &source).await;
                    assert!(result.execution.has_errors, "{label} at {depth}");
                    let text = output(&result);
                    assert!(text.contains("nesting too deep: more than 100 levels") && text.starts_with("run:1:"), "{label} at {depth}: {text}");
                    let next = run_source(&cfg, &mut ws, &baseline, "return abs(0 - 2)\n").await;
                    assert_eq!(output(&next), "2", "{label} at {depth}: the next call must work");
                }
            }
            // Exactly at the limit a block lowers and runs (nothing is stored, so the
            // typed-persistence depth budget that limits `mutate` on this branch does not apply).
            let limit = crate::python_frontend::MAX_NESTING;
            for (source, expected) in [
                (format!("x: int = {}1{}\nreturn x\n", "(".repeat(limit - 1), ")".repeat(limit - 1)), "1".to_string()),
                (format!("x: int = 1{}\nreturn x\n", " + 1".repeat(limit - 1)), limit.to_string()),
                (format!("x: int = {}0 - 3{}\nreturn x\n", "abs(".repeat(limit - 2), ")".repeat(limit - 2)), "3".to_string()),
            ] {
                let result = run_source(&cfg, &mut ws, &baseline, &source).await;
                assert_eq!(output(&result), expected, "{}", result.content);
            }
            assert!(log.lock().unwrap().is_empty());
        }
    })
    .await
    .unwrap();
}

/// F4: deep recursion in a called function. Depth 200 used to overflow the stack before
/// the call-depth guard (256) could fire; now 256 nested calls run and the next is a
/// clean error, for every class and through `evaluate` as well.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_call_depth_guard_fires_before_the_stack_runs_out() {
    tokio::spawn(async {
        for model in ["viewer", "writer"] {
            let (cfg, mut ws, baseline, _log) = speaker(model, None).await;
            // `return f(255)` is 256 nested calls, the deepest that is allowed; `arms` also
            // calls a builtin at its innermost level, which is one more.
            for (function, deepest) in [("NtHard.deep", 255), ("NtHard.ping", 255), ("NtHard.arms", 254)] {
                let result = run_source(&cfg, &mut ws, &baseline, &format!("return {function}({deepest})\n")).await;
                assert_eq!(output(&result), "0", "{model} {function}: {}", result.content);
                for depth in [256, 257, 400, 5_000] {
                    let result = run_source(&cfg, &mut ws, &baseline, &format!("return {function}({depth})\n")).await;
                    assert!(result.execution.has_errors, "{model} {function}({depth})");
                    assert!(output(&result).contains("maximum call depth (256) exceeded"), "{model} {function}({depth}): {}", result.content);
                }
                let next = run_source(&cfg, &mut ws, &baseline, &format!("return {function}(3)\n")).await;
                assert_eq!(output(&next), "0", "{model} {function}: the next call must work");
            }
            // Through an awaited user function, in a block and under evaluate (the same
            // evaluation thread).
            let awaited = run_source(&cfg, &mut ws, &baseline, "n: int = await NtHard.deep_io(254)\nreturn n\n").await;
            assert_eq!(output(&awaited), "0", "{model}: {}", awaited.content);
            let too_deep = run_source(&cfg, &mut ws, &baseline, "n: int = await NtHard.deep_io(300)\nreturn n\n").await;
            assert!(output(&too_deep).contains("maximum call depth (256) exceeded"), "{model}: {}", too_deep.content);
            // `evaluate` of an async function uses the same evaluation thread: the depth
            // guard fires there too.
            for (depth, deep) in [(250, false), (300, true), (5_000, true)] {
                let result = execute_native_tool("evaluate", &json!({"language":"python","function":"NtHard.deep_io","input":depth.to_string()}).to_string(), &cfg, &mut ws, &baseline).await;
                assert_eq!(result.execution.has_errors, deep, "{model} evaluate deep_io({depth}): {}", result.content);
                assert_eq!(result.content.contains("maximum call depth (256) exceeded"), deep, "{}", result.content);
            }
            // `evaluate` of a synchronous function runs on the caller's own (small) stack,
            // compiled where possible. Whatever tier runs it, the process survives: a
            // value, the depth guard, or the stack check.
            for function in ["NtHard.deep", "NtHard.ping", "NtHard.arms"] {
                for depth in [200, 250, 5_000] {
                    let result = execute_native_tool("evaluate", &json!({"language":"python","function":function,"input":depth.to_string()}).to_string(), &cfg, &mut ws, &baseline).await;
                    assert!(
                        !result.execution.has_errors || result.content.contains("maximum call depth (256) exceeded") || result.content.contains("call stack exhausted"),
                        "{model} evaluate {function}({depth}): {}", result.content
                    );
                    println!("evaluate {function}({depth}) as {model}: {}", &result.content[..result.content.len().min(160)]);
                }
            }
        }
    })
    .await
    .unwrap();
}

async fn in_turn(cfg: &AppConfig, ws: &mut WorkingSet, baseline: &Program, budget: &Arc<TurnBudget>, source: &str) -> NativeToolResult {
    execute_native_tool_in_turn("run", &json!({"source": source}).to_string(), cfg, ws, baseline, budget).await
}

/// The blocks of one turn share a wall-clock budget; a pure block draws nothing from the
/// tool-call budget, so this is what bounds a turn of pure blocks.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_blocks_of_one_turn_share_a_time_budget() {
    let (cfg, mut ws, baseline, _log) = viewer().await;
    let class = speaker_class(&cfg);
    assert_eq!(class.limits().turn_block_time, Some(Duration::from_secs(180)));
    // A turn budget with 400 ms of block time, otherwise the class's own limits.
    let mut limits = class.limits();
    limits.turn_block_time = Some(Duration::from_millis(400));
    let budget = TurnBudget::for_limits(&limits);
    let quick = in_turn(&cfg, &mut ws, &baseline, &budget, "return 1\n").await;
    assert_eq!(output(&quick), "1");
    // Blocks of pure work (120,000 steps, some tens of milliseconds each) until the
    // turn's time is used up.
    let spin = "i: int = 0\nj: int = 0\nwhile i < 60:\n    j = 0\n    while j < 1000:\n        j = j + 1\n    i = i + 1\nreturn i\n";
    let mut refused = None;
    for round in 0..400 {
        let result = in_turn(&cfg, &mut ws, &baseline, &budget, spin).await;
        if result.execution.has_errors {
            refused = Some((round, output(&result)));
            break;
        }
    }
    let (round, text) = refused.expect("the turn's block time never ran out");
    assert!(round >= 1, "the first block must fit");
    assert_eq!(text, "run error: run stopped: the blocks of this turn ran longer than 0 seconds in total (limit for one turn in this conversation)");
    assert!(budget.block_time_used() >= Duration::from_millis(400));
    // Once spent, every further block of the turn is refused at its first step.
    let after = in_turn(&cfg, &mut ws, &baseline, &budget, "return 1\n").await;
    assert!(output(&after).contains("the blocks of this turn ran longer than"), "{}", after.content);
    assert_eq!(envelope(&after)["success"], false);
    // A new turn starts with a fresh budget; the owner has none.
    let fresh = run_source(&cfg, &mut ws, &baseline, "return 1\n").await;
    assert_eq!(output(&fresh), "1");
    assert_eq!(SpeakerClass::Writer.limits().turn_block_time, None);
}
