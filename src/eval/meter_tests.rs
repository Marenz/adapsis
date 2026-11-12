//! The run-block meter in isolation: these tests execute statement bodies directly, with
//! NO admission in front, so they show what the interpreter itself enforces against a
//! hostile block. Admission-level tests are in `api/native_tools/run_restricted_tests.rs`.
use super::super::*;
use super::*;
use crate::run_limits::RunLimits;
use std::time::Duration;

const LIBRARY: &str = "\
+module MeterLib
+fn home ()->String
  +return home_dir()
+end
+fn pure (x:Int)->Int
  +return abs(x)
+end
+fn deep (n:Int)->String
  +if n > 0
    +return MeterLib.deep(n - 1)
  +end
  +return home_dir()
+end
+fn fails (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~not_positive
  +return x
+end
+fn read (path:String)->String [io,async]
  +await text:String = file_read(path)
  +return text
+end
+fn spawner ()->Int [io,async]
  +spawn MeterLib.pure(1)
  +return 1
+end
+fn big ()->String
  +let s:String = \"0123456789abcdef\"
  +let i:Int = 0
  +while i < 14
    +set s = concat(s, s)
    +set i = i + 1
  +end
  +return s
+end
";

fn program(block: &str) -> (ast::Program, Vec<ast::Statement>) {
    let source = format!("{LIBRARY}+module MeterBlock\n+fn block ()->String [io,async,fail,mut]\n{block}+end\n");
    let mut program = ast::Program::default();
    for op in crate::parser::parse(&source).unwrap_or_else(|error| panic!("{error}\n{source}")) {
        crate::validator::apply_and_validate(&mut program, &op).unwrap_or_else(|error| panic!("{error}\n{source}"));
    }
    program.rebuild_function_index();
    let body = program.get_function("MeterBlock.block").unwrap().body.clone();
    (program, body)
}

fn restricted(functions: &[&str], tools: &[&str]) -> RunContext {
    RunContext {
        limits: RunLimits::restricted(),
        policy: Some(BlockPolicy {
            tools: tools.iter().map(|tool| tool.to_string()).collect(),
            functions: functions.iter().map(|function| function.to_string()).collect(),
        }),
        turn: TurnBudget::new(RunLimits::restricted().turn_tool_calls),
    }
}

/// Every `file_read` and tool call is answered by a mock; nothing touches the host.
fn handle() -> crate::coroutine::CoroutineHandle {
    let mock = |operation: &str| crate::session::IoMock {
        operation: operation.into(),
        patterns: vec![],
        response: "mocked".into(),
    };
    crate::coroutine::CoroutineHandle::new_mock(vec![mock("file_read"), mock("web_search"), mock("music_generate")])
}

fn run(block: &str, context: RunContext) -> Result<Value> {
    let (program, body) = program(block);
    let outcome = eval_detached_block(&program, "RunToolScratch.run_block", &body, handle(), context);
    assert!(!active(), "the meter must be gone when the block ends");
    outcome
}

fn refused(block: &str, context: RunContext) -> String {
    run(block, context).expect_err("the block must be stopped").to_string()
}

#[test]
fn no_meter_exists_outside_a_block_and_it_is_cleared_on_every_exit() {
    assert!(!active());
    assert!(step().is_ok() && io_call("file_read").is_ok() && check_builtin("file_read").is_ok());
    assert!(enter_callee().is_none() && frame_is_block().is_none());
    assert_eq!(run("  +return \"ok\"\n", RunContext::owner()).unwrap().to_string(), "\"ok\"");
    assert!(run("  +let xs:List<Int> = list(1)\n  +return get(xs, 5)\n", RunContext::owner()).is_err());
    // A panic inside the block unwinds through the guard.
    let (program, body) = program("  +return \"ok\"\n");
    let unwound = std::panic::catch_unwind(|| {
        let _meter = install(RunContext::owner());
        assert!(active());
        let _ = (&program, &body);
        panic!("fixture panic inside a block");
    });
    assert!(unwound.is_err());
    assert!(!active(), "a panic must not leave a meter on a reused thread");
    assert!(frame_is_block().is_none());
}

/// Decision 5: the block's frame is marked when the block is entered, and every declared
/// function pushes its own marker. The policy follows the marker, not a depth count.
#[test]
fn the_call_policy_applies_to_the_block_frame_and_only_to_it() {
    let allowed = ["MeterLib.home", "MeterLib.pure", "MeterLib.deep", "MeterLib.fails"];
    // A non-pure builtin issued by the block itself is refused...
    let error = refused("  +return home_dir()\n", restricted(&allowed, &[]));
    assert!(error.contains("`home_dir` is not available in this conversation"), "{error}");
    // ...while a callee is owner-written code and may call it, as under evaluate.
    for block in [
        "  +return MeterLib.home()\n",
        // A recursive callee: every level is a callee frame.
        "  +return MeterLib.deep(40)\n",
    ] {
        let value = run(block, restricted(&allowed, &[])).unwrap();
        assert!(matches!(value, Value::String(_)), "{block}: {value}");
    }
    // After a callee returns (normally, through recursion, through a pure builtin, or
    // through a caught failure), the block frame is the block frame again.
    for prefix in [
        "  +let h:String = MeterLib.home()\n",
        "  +let h:String = MeterLib.deep(25)\n",
        "  +let p:Int = MeterLib.pure(0 - 4)\n  +let q:Int = abs(p)\n",
        "  +call r:Result<Int> = MeterLib.fails(0)\n",
    ] {
        let error = refused(&format!("{prefix}  +return home_dir()\n"), restricted(&allowed, &[]));
        assert!(error.contains("`home_dir` is not available"), "{prefix}: {error}");
    }
    // A callee calling back into a pure builtin does not turn that builtin's frame into
    // anything else, and the block may use pure builtins and constructors itself.
    let value = run("  +let p:Int = MeterLib.pure(0 - 4)\n  +let r:Result<Int> = Ok(p)\n  +return to_string(abs(p))\n", restricted(&allowed, &[])).unwrap();
    assert_eq!(value.to_string(), "\"4\"");
    // The markers themselves, observed directly.
    let _meter = install(restricted(&[], &[]));
    assert_eq!(frame_is_block(), Some(true));
    {
        let _outer = enter_callee();
        assert_eq!(frame_is_block(), Some(false));
        {
            let _inner = enter_callee();
            assert_eq!(frame_is_block(), Some(false));
        }
        assert_eq!(frame_is_block(), Some(false));
        assert!(check_builtin("home_dir").is_ok(), "a callee frame is not policed");
    }
    assert_eq!(frame_is_block(), Some(true));
    assert!(check_builtin("home_dir").is_err());
}

#[test]
fn the_block_frame_may_call_only_admitted_functions_and_its_own_tools() {
    let error = refused("  +return MeterLib.home()\n", restricted(&["MeterLib.pure"], &[]));
    assert!(error.contains("function `MeterLib.home` is not available in this conversation"), "{error}");
    // Guests: no user function at all.
    let error = refused("  +let p:Int = MeterLib.pure(1)\n  +return \"x\"\n", restricted(&[], &["web_search"]));
    assert!(error.contains("function `MeterLib.pure`"), "{error}");
    // A tool on the list is callable; any other IO builtin is not, awaited or not.
    assert_eq!(run("  +await hits:String = web_search(\"q\")\n  +return hits\n", restricted(&[], &["web_search"])).unwrap().to_string(), "\"mocked\"");
    for block in [
        "  +await text:String = file_read(\"/etc/hostname\")\n  +return text\n",
        "  +let text:String = file_read(\"/etc/hostname\")\n  +return text\n",
        "  +await out:String = shell_exec(\"id\")\n  +return out\n",
        "  +await done:String = mutate(\"+module X\")\n  +return done\n",
        "  +await set:String = llm_set_model(\"other\")\n  +return set\n",
        "  +await song:String = music_generate(\"x\", 30)\n  +return song\n",
    ] {
        let error = refused(block, restricted(&[], &["web_search"]));
        assert!(error.contains("is not available in this conversation"), "{block}: {error}");
        assert!(error.contains("your tools (web_search)"), "{error}");
    }
    // Statement forms that start tasks or register handlers.
    for (block, what) in [
        ("  +spawn MeterLib.pure(1)\n  +return \"x\"\n", "spawn"),
        ("  +source timer tick 60000\n  +return \"x\"\n", "a source declaration"),
    ] {
        let error = refused(block, restricted(&["MeterLib.pure"], &[]));
        assert!(error.contains(&format!("run: {what} is not available")), "{block}: {error}");
    }
    // A function the block may call can still spawn, as under evaluate: the policy lets
    // it through to the transport (which this mock handle does not have).
    let error = refused("  +await n:Int = MeterLib.spawner()\n  +return \"x\"\n", restricted(&["MeterLib.spawner"], &[]));
    assert!(error.contains("+spawn failed to send"), "{error}");
    // An owner block has no policy: its own spawn reaches the transport too.
    assert!(run("  +let h:String = home_dir()\n  +return MeterLib.home()\n", RunContext::owner()).is_ok());
    let error = refused("  +spawn MeterLib.pure(1)\n  +return \"x\"\n", RunContext::owner());
    assert!(error.contains("+spawn failed to send"), "{error}");
}

#[test]
fn steps_wall_clock_and_io_calls_are_limited() {
    let spin = "  +let i:Int = 0\n  +let j:Int = 0\n  +while i < 9000\n    +set j = 0\n    +while j < 9000\n      +set j = j + 1\n    +end\n    +set i = i + 1\n  +end\n  +return \"finished\"\n";
    let error = refused(spin, restricted(&[], &[]));
    assert_eq!(error, "run stopped: it executed more than 200000 steps (limit for this conversation)");
    // A loop with an empty body still counts: the iteration itself is a step.
    let each = "  +let xs:List<Int> = list(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)\n  +each xs x:Int\n  +end\n  +return \"finished\"\n";
    let mut tiny = restricted(&[], &[]);
    tiny.limits.steps = Some(8);
    assert!(refused(each, tiny).contains("more than 8 steps"));
    // Wall clock: checked between steps, so a pure loop cannot outlast it.
    let mut hurried = restricted(&[], &[]);
    hurried.limits.steps = None;
    hurried.limits.wall_clock = Some(Duration::from_millis(30));
    let started = std::time::Instant::now();
    let error = refused(&spin.replace("9000", "10000"), hurried);
    assert_eq!(error, "run stopped: it ran longer than 0 seconds (limit for this conversation)");
    assert!(started.elapsed() < Duration::from_secs(5));
    // IO calls made by called functions count; the ninth is refused before it is sent.
    let reads: String = (0..9).map(|n| format!("  +await t{n}:String = MeterLib.read(\"f{n}\")\n")).collect();
    let error = refused(&format!("{reads}  +return t8\n"), restricted(&["MeterLib.read"], &[]));
    assert_eq!(error, "run stopped: `file_read` would be IO call 9 of this block; the limit for this conversation is 8 per block");
    assert!(run(&format!("{}  +return t7\n", &reads[..reads.len() - reads.lines().last().unwrap().len() - 1]), restricted(&["MeterLib.read"], &[])).is_ok());
    // Costly tools: once per block each.
    let error = refused("  +await a:String = music_generate(\"x\", 30)\n  +await b:String = music_generate(\"y\", 30)\n  +return b\n", restricted(&[], &["music_generate"]));
    assert_eq!(error, "run stopped: `music_generate` may be called 1 time(s) per block in this conversation");
    // The owner has none of these limits.
    // 490,000 iterations, well past the restricted step limit.
    assert_eq!(run(&spin.replace("9000", "700"), RunContext::owner()).unwrap().to_string(), "\"finished\"");
    assert!(run(&format!("{reads}  +return t8\n"), RunContext::owner()).is_ok());
}

#[test]
fn the_turn_budget_is_shared_and_exhaustion_sends_nothing() {
    let turn = TurnBudget::new(Some(3));
    let context = |turn: &std::sync::Arc<TurnBudget>| RunContext { turn: turn.clone(), ..restricted(&[], &["web_search"]) };
    let two = "  +await a:String = web_search(\"1\")\n  +await b:String = web_search(\"2\")\n  +return b\n";
    assert!(run(two, context(&turn)).is_ok());
    assert_eq!(turn.used(), 2);
    let error = refused(two, context(&turn));
    assert!(error.starts_with("run stopped: `web_search` refused: this turn has used its 3 tool calls."), "{error}");
    assert!(turn.take("web_search").is_err());
    assert!(TurnBudget::new(None).take("anything").is_ok());
}

/// Once a limit trips, the block cannot continue even where an error is swallowed.
#[test]
fn a_tripped_limit_is_sticky() {
    let mut context = restricted(&[], &[]);
    context.limits.steps = Some(2);
    let _meter = install(context);
    assert!(step().is_ok() && step().is_ok());
    let first = step().unwrap_err().to_string();
    assert_eq!(first, "run stopped: it executed more than 2 steps (limit for this conversation)");
    for _ in 0..3 {
        assert_eq!(step().unwrap_err().to_string(), first);
        assert_eq!(io_call("web_search").unwrap_err().to_string(), first);
    }
}

/// Tests that really allocate up to the owner's 64 MiB cap run in a child process. Large
/// allocations in the shared test process move later JIT mappings far enough away to
/// trigger the known cranelift relocation panic (#96) in unrelated JIT tests.
fn in_child(test: &str) -> bool {
    if crate::logging::tests::child() {
        return true;
    }
    crate::logging::tests::capture(&format!("eval::meter::tests::{test}"), false);
    false
}

fn doubling(kind: &str) -> String {
    let (init, grow) = match kind {
        "string" => ("  +let v:String = \"0123456789abcdef\"\n", "    +set v = concat(v, v)\n"),
        "list" => ("  +let v:List<Int> = list(1, 2, 3, 4)\n", "    +set v = list(v, v)\n"),
        "append-self" => ("  +let v:List<String> = list(\"x\")\n", "    +set v = push(v, v)\n"),
        "map" => ("  +let v:Map<String, Int> = map(\"k\", 1)\n", "    +set v = map_set(v, concat(\"k\", to_string(i)), v)\n"),
        "struct" => ("  +let v:List<Int> = list(1)\n", "    +set v = list({a: v, b: v})\n"),
        "result" => ("  +let v:List<Int> = list(1)\n", "    +set v = list(Ok(v), Ok(v))\n"),
        other => panic!("{other}"),
    };
    format!("{init}  +let i:Int = 0\n  +while i < 100\n{grow}    +set i = i + 1\n  +end\n  +return to_string(len(v))\n")
}

/// The memory attacks: each grows by sharing, so real memory stays tiny while the value
/// "as held or rendered" explodes. All are stopped by the single-value cap, for the owner
/// too, long before the process is at risk.
#[test]
fn growth_by_doubling_is_stopped_by_the_single_value_cap_for_every_class() {
    if !in_child("growth_by_doubling_is_stopped_by_the_single_value_cap_for_every_class") {
        return;
    }
    for kind in ["string", "list", "append-self", "map", "struct", "result"] {
        for (label, context, limit) in [
            ("restricted", restricted(&[], &[]), "1 MiB"),
            ("owner", RunContext::owner(), "64 MiB"),
        ] {
            let started = std::time::Instant::now();
            let error = refused(&doubling(kind), context);
            let too_large = format!("run stopped: a value exceeds {limit} (limit for a single value in this conversation)");
            let too_deep = "run stopped: a value is nested deeper than 64 levels";
            assert!(error == too_large || error == too_deep, "{kind}/{label}: {error}");
            assert!(started.elapsed() < Duration::from_secs(20), "{kind}/{label} took {:?}", started.elapsed());
        }
    }
}

#[test]
fn a_list_of_large_values_is_stopped_by_the_single_value_cap_under_the_owner_profile() {
    if !in_child("a_list_of_large_values_is_stopped_by_the_single_value_cap_under_the_owner_profile") {
        return;
    }
    // 16 values of 16 MiB each, all distinct allocations: 256 MiB if it were allowed.
    let block = "  +let base:String = \"0123456789abcdef\"\n  +let i:Int = 0\n  +while i < 20\n    +set base = concat(base, base)\n    +set i = i + 1\n  +end\n  +let acc:List<String> = list()\n  +let n:Int = 0\n  +while n < 16\n    +set acc = push(acc, concat(base, to_string(n)))\n    +set n = n + 1\n  +end\n  +return to_string(len(acc))\n";
    let error = refused(block, RunContext::owner());
    assert_eq!(error, "run stopped: a value exceeds 64 MiB (limit for a single value in this conversation)");
    // Three of them fit.
    assert_eq!(run(&block.replace("n < 16", "n < 3"), RunContext::owner()).unwrap().to_string(), "\"3\"");
}

#[test]
fn restricted_blocks_cannot_hold_many_large_values_or_take_one_from_a_callee() {
    // Many separate values just under the single-value cap: the cumulative cap stops it.
    let block = "  +let base:String = MeterLib.big()\n  +let keep:String = \"\"\n  +let i:Int = 0\n  +while i < 400\n    +set keep = concat(base, to_string(i))\n    +set i = i + 1\n  +end\n  +return \"finished\"\n";
    let error = refused(block, restricted(&["MeterLib.big"], &[]));
    assert_eq!(error, "run stopped: it produced more than 32 MiB of values in total (limit for this conversation)");
    assert_eq!(run(block, RunContext::owner()).unwrap().to_string(), "\"finished\"");
    // Arguments are admitted before a builtin runs, so nothing renders an oversized value
    // and a combining builtin cannot be handed more than twice the cap.
    let mut small = restricted(&["MeterLib.big"], &[]);
    small.limits.value_bytes = 100_000;
    let error = refused("  +let base:String = MeterLib.big()\n  +return to_string(len(base))\n", small);
    assert!(error.contains("a value exceeds 100000 bytes"), "{error}");
    let mut half = restricted(&["MeterLib.big"], &[]);
    half.limits.value_bytes = 400_000;
    let error = refused("  +let base:String = MeterLib.big()\n  +let ok:Bool = starts_with(base, base)\n  +return concat(base, base, base, base, base, base, base, base)\n", half);
    assert!(error.contains("a value exceeds 400000 bytes"), "{error}");
}

/// An independent, cache-free computation of the size model.
fn exact(value: &Value) -> usize {
    match value {
        Value::Float(_) => 64,
        Value::String(text) => 32 + text.len(),
        Value::Err(text) => 32 + text.len(),
        Value::Ok(inner) => 32 + exact(inner),
        Value::Union { payload, .. } => 32 + payload.iter().map(exact).sum::<usize>(),
        Value::List(items) | Value::Set(items) => 32 + items.iter().map(exact).sum::<usize>(),
        Value::Map(entries) => 32 + entries.iter().map(|(key, item)| exact(key) + exact(item)).sum::<usize>(),
        Value::Struct(_, fields) => 32 + fields.values().map(|item| 32 + exact(item)).sum::<usize>(),
        _ => 32,
    }
}

/// Soundness of the remembered sizes: whatever a builtin returns, the size the meter holds
/// for it is never below its real size. A pseudo-random mix of every container builtin.
#[test]
fn remembered_sizes_never_undercount() {
    let program = ast::Program::default();
    let _meter = install(RunContext::owner());
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = |bound: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % bound as u64) as usize
    };
    let mut pool: Vec<Value> = vec![
        Value::Int(1),
        Value::string("seed"),
        Value::list(vec![]),
        eval_builtin_or_user(&program, "map", vec![], &mut env).unwrap(),
        eval_builtin_or_user(&program, "set", vec![], &mut env).unwrap(),
    ];
    let mut checked = 0;
    for round in 0..4000 {
        let pick = |pool: &Vec<Value>, index: usize| pool[index % pool.len()].clone();
        let (a, b, c) = (pick(&pool, next(1000)), pick(&pool, next(1000)), pick(&pool, next(1000)));
        let (name, args): (&str, Vec<Value>) = match next(16) {
            0 => ("push", vec![a, b]),
            1 => ("insert", vec![a, Value::Int(0), b]),
            2 => ("list", vec![a, b, c]),
            3 => ("map_set", vec![a, Value::string(format!("k{}", round % 7)), b]),
            4 => ("set_add", vec![a, b]),
            5 => ("map", vec![Value::string("k"), a, Value::string("j"), b]),
            6 => ("reverse", vec![a]),
            7 => ("slice", vec![a, Value::Int(0), Value::Int(2)]),
            8 => ("map_entries", vec![a]),
            9 => ("map_values", vec![a]),
            10 => ("set_to_list", vec![a]),
            11 => ("concat", vec![Value::string("x"), Value::string(format!("{round}"))]),
            12 => ("Ok", vec![a]),
            13 => ("get", vec![a, Value::Int(0)]),
            14 => ("map_keys", vec![a]),
            _ => ("split", vec![Value::string("a b c d"), Value::string(" ")]),
        };
        let Ok(result) = eval_builtin_or_user(&program, name, args, &mut env) else { continue };
        if let Some(recorded) = recorded_bytes(&result) {
            assert!(recorded >= exact(&result), "{name}: recorded {recorded} < exact {}", exact(&result));
            checked += 1;
        }
        if pool.len() < 64 {
            pool.push(result);
        } else {
            let slot = next(64);
            pool[slot] = result;
        }
    }
    assert!(checked > 1000, "only {checked} container results were compared");
}

/// Decision 3: a loop that appends step by step must not re-walk the whole value at every
/// step. Deterministic: count the nodes the meter visited, not the time it took.
#[test]
fn appending_step_by_step_is_measured_in_constant_work_per_step() {
    for (label, init, step_source, steps) in [
        ("list", "  +let v:List<Int> = list()\n", "    +set v = push(v, i)\n", 3000u64),
        ("map", "  +let v:Map<String, Int> = map()\n", "    +set v = map_set(v, to_string(i), i)\n", 1500),
        ("string", "  +let v:String = \"\"\n", "    +set v = concat(v, \"ab\")\n", 3000),
    ] {
        let block = format!("{init}  +let i:Int = 0\n  +while i < {steps}\n{step_source}    +set i = i + 1\n  +end\n  +return to_string(len(v))\n");
        let (program, body) = program(&block);
        let _meter = install(RunContext::owner());
        let mut env = invocation::async_tree_env(&program, handle());
        let value = eval_function_body_named(&program, "RunToolScratch.run_block", &body, &mut env).unwrap();
        assert!(matches!(value, Value::String(_)));
        let walked = walked_nodes();
        // A re-walk per step would visit about steps*steps/2 nodes (4.5 million for 3000).
        assert!(walked < steps * 12, "{label}: {walked} nodes for {steps} steps");
    }
}

/// The same property as wall-clock time, on a loop the interpreter itself runs in linear
/// time: 100,000 append steps under the owner profile.
#[test]
fn a_hundred_thousand_step_append_loop_finishes_quickly_under_the_owner_profile() {
    let block = "  +let s:String = \"\"\n  +let i:Int = 0\n  +let j:Int = 0\n  +while i < 10\n    +set j = 0\n    +while j < 10000\n      +set s = concat(s, \"ab\")\n      +set j = j + 1\n    +end\n    +set i = i + 1\n  +end\n  +return to_string(len(s))\n";
    let started = std::time::Instant::now();
    let value = run(block, RunContext::owner()).unwrap();
    assert_eq!(value.to_string(), "\"200000\"");
    assert!(started.elapsed() < Duration::from_secs(20), "{:?}", started.elapsed());
}

/// The blocks of one turn share a wall-clock budget: a block runs against its own limit
/// or what is left of the turn's, whichever is shorter, and says which one stopped it.
#[test]
fn the_turn_block_time_budget_is_shared_by_the_blocks_of_a_turn() {
    let spin = "  +let i:Int = 0\n  +let j:Int = 0\n  +while i < 10000\n    +set j = 0\n    +while j < 10000\n      +set j = j + 1\n    +end\n    +set i = i + 1\n  +end\n  +return \"finished\"\n";
    let mut limits = RunLimits::restricted();
    limits.steps = None;
    limits.turn_block_time = Some(Duration::from_millis(120));
    let turn = TurnBudget::for_limits(&limits);
    let context = |limits: &RunLimits| RunContext { limits: limits.clone(), policy: None, turn: turn.clone() };
    // A short block uses a little of the turn's time and is not refused.
    assert_eq!(run("  +return \"quick\"\n", context(&limits)).unwrap().to_string(), "\"quick\"");
    assert!(turn.block_time_used() < Duration::from_millis(100));
    // The turn's remaining time (120 ms) is shorter than the block's own limit (60 s).
    let started = std::time::Instant::now();
    let error = refused(spin, context(&limits));
    assert_eq!(error, "run stopped: the blocks of this turn ran longer than 0 seconds in total (limit for one turn in this conversation)");
    assert!(started.elapsed() < Duration::from_secs(5));
    assert!(turn.block_time_used() >= Duration::from_millis(120));
    // Spent: the next block of the turn stops at its first step, before any work.
    let error = refused("  +return \"never\"\n", context(&limits));
    assert!(error.contains("the blocks of this turn ran longer than"), "{error}");
    // The block's own limit binds when it is the shorter one, and says so.
    let mut short = RunLimits::restricted();
    short.steps = None;
    short.wall_clock = Some(Duration::from_millis(30));
    let fresh = TurnBudget::for_limits(&short);
    let error = refused(spin, RunContext { limits: short, policy: None, turn: fresh.clone() });
    assert_eq!(error, "run stopped: it ran longer than 0 seconds (limit for this conversation)");
    assert!(fresh.block_time_used() >= Duration::from_millis(30));
    // The owner has neither clock.
    assert_eq!((RunLimits::owner().wall_clock, RunLimits::owner().turn_block_time), (None, None));
}

/// The accounting is real memory: one slot per value, so a list of n scalars is n slots.
#[test]
fn sizes_are_real_bytes() {
    assert_eq!(SLOT, std::mem::size_of::<Value>());
    let _meter = install(RunContext::owner());
    let program = ast::Program::default();
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    let list = eval_builtin_or_user(&program, "list", vec![Value::Int(1), Value::Int(2), Value::string("abc")], &mut env).unwrap();
    assert_eq!(recorded_bytes(&list), Some((SLOT + HEAP) + 2 * SLOT + (SLOT + HEAP + 3)));
    let (_, allocated) = crate::test_alloc::peak(|| Value::list((0..1000).map(Value::Int).collect()));
    assert!(allocated >= 1000 * SLOT && allocated <= 1000 * SLOT + 256, "{allocated}");
}
