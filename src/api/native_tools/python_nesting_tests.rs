//! #133: untrusted Python source of any nesting depth must never abort the
//! process. Every case runs on a 2 MiB stack (a tokio worker / blocking thread),
//! through the real native tool dispatch.
use super::*;
use serde_json::json;

/// Run `scenario` on a 2 MiB thread with its own runtime, like a worker.
fn on_worker_stack<F, Fut>(scenario: F)
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()>,
{
    std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(scenario());
        })
        .unwrap()
        .join()
        .expect("the process survived and the scenario passed");
}

async fn setup(module: &str) -> (AppConfig, WorkingSet, Program) {
    let program = build_program(&format!("+module {module}Seed\n+fn one ()->Int\n +return 1\n+end")).await;
    let cfg = config_for("writer", AccessLevel::Full, PermissionConfig::default(), program.clone());
    let ws = cfg.snapshot_working_set().await;
    (cfg, ws, program)
}

async fn tool(name: &str, arguments: serde_json::Value, cfg: &AppConfig, ws: &mut WorkingSet, baseline: &Program) -> NativeToolResult {
    execute_native_tool(name, &arguments.to_string(), cfg, ws, baseline).await
}

/// DIVERGENCE from the release line's copy of this file (6136115), confined to the two
/// "at the limit" tests: this branch stores a typed replay record for every Python
/// mutation (#122), and that record's own depth budget (96 JSON levels,
/// `session/replay/limits.rs`) is reached before 100 source levels. So the deepest source
/// `mutate` can STORE here is below `MAX_NESTING`; it is found by stepping down from the
/// nesting limit. The nesting guard itself is unchanged and exact (asserted one level
/// above `MAX_NESTING` below, and at exactly `MAX_NESTING` through `run`, which stores
/// nothing, in `run_hardening_tests.rs`).
const STORAGE_BUDGET: &str = "typed persistence structural budget exceeded";

fn refusal(result: &NativeToolResult) -> String {
    assert!(result.execution.has_errors, "expected a refusal: {}", result.content);
    result.content.clone()
}

/// (name, function source with `depth` levels, test source, evaluate input, expected value)
type Case = (&'static str, fn(usize) -> String, fn(usize) -> String, &'static str, fn(usize) -> String);

const CASES: &[Case] = &[
    ("operators", |n| format!("def f(x: int) -> int:\n    return x{}\n", " + 1".repeat(n - 1)),
        |n| format!("def test_f() -> None:\n    assert f(1) == {n}\n"), "1", |n| n.to_string()),
    ("calls", |n| format!("def g(x: int) -> int:\n    return x + 1\n\ndef f(x: int) -> int:\n    return {}x{}\n", "g(".repeat(n - 1), ")".repeat(n - 1)),
        |n| format!("def test_f() -> None:\n    assert f(1) == {n}\n"), "1", |n| n.to_string()),
    ("parentheses", |n| format!("def f(x: int) -> int:\n    return {}x{}\n", "(".repeat(n - 1), ")".repeat(n - 1)),
        |_| "def test_f() -> None:\n    assert f(7) == 7\n".into(), "7", |_| "7".into()),
    ("not", |n| format!("def f(x: bool) -> bool:\n    return {}x\n", "not ".repeat(2 * ((n - 1) / 2))),
        |_| "def test_f() -> None:\n    assert f(True) == True\n".into(), "True", |_| "true".into()),
    ("blocks", |n| {
        let mut source = String::from("def f(x: bool) -> int:\n");
        for level in 0..n - 1 {
            source.push_str(&format!("{}if x:\n", " ".repeat(level + 1)));
        }
        source.push_str(&format!("{}return 7\n return 0\n", " ".repeat(n)));
        source
    }, |_| "def test_f() -> None:\n    assert f(True) == 7\n".into(), "True", |_| "7".into()),
];

/// At the limit: the deepest accepted source is lowered, stored, tested and
/// evaluated correctly on a worker-sized stack.
#[test]
fn python_nesting_at_the_limit_runs_through_mutate_test_and_evaluate() {
    on_worker_stack(|| async {
        for (name, source, test, input, expected) in CASES {
            let module = format!("PyNestOk{}", name.replace(' ', ""));
            let (cfg, mut ws, baseline) = setup(&module).await;
            let nesting_limit = crate::python_frontend::MAX_NESTING;
            let mut limit = nesting_limit;
            let result = loop {
                let result = tool("mutate", json!({"language":"python","module":module,"source":source(limit)}), &cfg, &mut ws, &baseline).await;
                if !(result.execution.has_errors && result.content.contains(STORAGE_BUDGET)) {
                    break result;
                }
                limit -= 1;
            };
            println!("storable depth for {name}: {limit}");
            assert!(limit >= 16, "{name}: storable depth {limit}");
            assert!(!result.execution.has_errors, "{name} mutate: {}", result.content);
            let result = tool("test", json!({"language":"python","module":module,"source":test(limit)}), &cfg, &mut ws, &baseline).await;
            assert!(!result.execution.has_errors, "{name} test: {}", result.content);
            assert!(result.execution.test_results.iter().all(|t| t.pass) && !result.execution.test_results.is_empty(), "{name}: {}", result.content);
            let result = tool("evaluate", json!({"language":"python","function":format!("{module}.f"),"input":input}), &cfg, &mut ws, &baseline).await;
            assert!(!result.execution.has_errors, "{name} evaluate: {}", result.content);
            assert!(result.content.contains(&expected(limit)), "{name}: {}", result.content);
            // One level more is refused, with the position, and nothing is stored.
            let before = format!("{:?}", ws.program.get_function(&format!("{module}.f")).map(|function| function as *const _));
            let text = refusal(&tool("mutate", json!({"language":"python","module":module,"source":source(nesting_limit + 1)}), &cfg, &mut ws, &baseline).await);
            assert!(text.contains("nesting too deep: more than 100 levels") && text.contains(&format!("{module}:")), "{name}: {text}");
            assert_eq!(format!("{:?}", ws.program.get_function(&format!("{module}.f")).map(|function| function as *const _)), before, "{name}: stored function untouched");
        }
    });
}

/// Far beyond the limit, at the depths that used to overflow lowering (about
/// 700 to 1,300 levels on this stack) and as deep as a native tool call can
/// carry (`source` is capped at 32,000 bytes): refused before parsing, in every
/// tool. Depths beyond the tool cap are covered at the frontend entry points
/// (`python_frontend::nesting::tests`).
#[test]
fn python_nesting_far_beyond_the_limit_is_refused_without_overflow() {
    on_worker_stack(|| async {
        let (cfg, mut ws, baseline) = setup("PyNestDeep").await;
        let mut refused = 0;
        for depth in [150, 700, 1_300, 3_000, 7_000, 10_000] {
            let mut sources: Vec<String> = CASES.iter().map(|(_, source, _, _, _)| source(depth)).collect();
            // Shapes that nest only in the Python tree.
            sources.extend([
                format!("def f(x: int) -> int:\n    return x{}\n", ".a".repeat(depth)),
                format!("def f(x: int) -> int:\n    return {}x\n", "-".repeat(depth)),
                format!("def f(x: int) -> int:\n    return {}x\n", "lambda: ".repeat(depth)),
                format!("def f(x: int) -> int:\n    return {}x\n", "x if x else ".repeat(depth)),
                // (Small f-strings get the frontend's ordinary "unsupported" refusal.)
                format!("def f(x: int) -> int:\n    return f\"{{x{}}}\"\n", " + 1".repeat(depth.max(600))),
                format!("def f(x: {}int{}) -> int:\n    return 1\n", "list[".repeat(depth), "]".repeat(depth)),
                format!("def f(x: int) -> int:\n    match x:\n        case {}y{}:\n            return 1\n    return 0\n", "[".repeat(depth), "]".repeat(depth)),
            ]);
            for source in sources.into_iter().filter(|source| source.len() <= 32_000) {
                let text = refusal(&tool("mutate", json!({"language":"python","module":"PyNestDeep","source":source}), &cfg, &mut ws, &baseline).await);
                let shown: String = text.chars().take(300).collect();
                assert!(text.contains("nesting too deep") || text.contains("f-string replacement fields too large"), "{depth}: {shown}");
                // An actionable refusal: where, the limit, and what to do.
                assert!(text.contains("PyNestDeep:") && text.contains("limit") && (text.contains("split the expression") || text.contains("concat")), "{depth}: {shown}");
                refused += 1;
            }
            // Test sources and evaluate inputs go through the same guard.
            let seed = tool("mutate", json!({"language":"python","module":"PyNestDeep","source":"def f(x: int) -> int:\n    return x\n"}), &cfg, &mut ws, &baseline).await;
            assert!(!seed.execution.has_errors, "{}", seed.content);
            for expression in [format!("{}1", "-".repeat(depth)), format!("{}1{}", "[".repeat(depth), "]".repeat(depth)), format!("1{}", " + 1".repeat(depth.min(7_000)))] {
                let test = format!("def test_f() -> None:\n    assert f({expression}) == 1\n");
                let result = tool("test", json!({"language":"python","module":"PyNestDeep","source":test}), &cfg, &mut ws, &baseline).await;
                assert!(refusal(&result).contains("nesting too deep"), "test at {depth}: {}", result.content.chars().take(300).collect::<String>());
                assert!(result.execution.test_results.is_empty(), "no test ran");
                let result = tool("evaluate", json!({"language":"python","function":"PyNestDeep.f","input":expression}), &cfg, &mut ws, &baseline).await;
                assert!(refusal(&result).contains("nesting too deep"), "evaluate at {depth}: {}", result.content.chars().take(300).collect::<String>());
                refused += 2;
            }
        }
        assert!(refused > 90, "{refused} refusals exercised");
        // The module is unharmed and still usable after all the refusals.
        let result = tool("evaluate", json!({"language":"python","function":"PyNestDeep.f","input":"41"}), &cfg, &mut ws, &baseline).await;
        assert!(!result.execution.has_errors && result.content.contains("41"), "{}", result.content);
    });
}

/// An `elif` chain nests in the runtime too (MEASURED: evaluating about 660
/// branches overflows this stack). The longest accepted chain is lowered,
/// tested and evaluated; chains that used to abort the process are refused.
#[test]
fn python_elif_chains_run_at_the_limit_and_are_refused_beyond_it() {
    on_worker_stack(|| async {
        let (cfg, mut ws, baseline) = setup("PyNestElif").await;
        let chain = |branches: usize| {
            let mut source = String::from("def f(x: int) -> int:\n    if x == 0:\n        return 0\n");
            for branch in 1..=branches {
                source.push_str(&format!("    elif x == {branch}:\n        return {}\n", branch * 2));
            }
            source.push_str("    return -1\n");
            source
        };
        // The function body, 98 branches and `==` (or the branch's block): 100 levels.
        // (On this branch the storable chain is shorter; see STORAGE_BUDGET.)
        let mut stored = 98;
        let result = loop {
            let result = tool("mutate", json!({"language":"python","module":"PyNestElif","source":chain(stored)}), &cfg, &mut ws, &baseline).await;
            if !(result.execution.has_errors && result.content.contains(STORAGE_BUDGET)) {
                break result;
            }
            stored -= 1;
        };
        println!("storable elif branches: {stored}");
        assert!(stored >= 16, "storable elif chain: {stored}");
        assert!(!result.execution.has_errors, "{}", result.content.chars().take(400).collect::<String>());
        let test = format!("def test_f() -> None:\n    assert f({stored}) == {}\n\ndef test_f_default() -> None:\n    assert f(5000) == -1\n", stored * 2);
        let result = tool("test", json!({"language":"python","module":"PyNestElif","source":test}), &cfg, &mut ws, &baseline).await;
        assert!(!result.execution.has_errors, "{}", result.content.chars().take(400).collect::<String>());
        let result = tool("evaluate", json!({"language":"python","function":"PyNestElif.f","input":(stored - 2).to_string()}), &cfg, &mut ws, &baseline).await;
        assert!(!result.execution.has_errors && result.content.contains(&((stored - 2) * 2).to_string()), "{}", result.content.chars().take(400).collect::<String>());
        for branches in [99, 400, 700] {
            let source = chain(branches);
            assert!(source.len() <= 32_000);
            let text = refusal(&tool("mutate", json!({"language":"python","module":"PyNestElif","source":source}), &cfg, &mut ws, &baseline).await);
            assert!(text.contains("nesting too deep") && text.contains("elif branches") && text.contains("PyNestElif:"), "{branches}: {}", text.chars().take(300).collect::<String>());
        }
        // The stored function is still the one that runs.
        let result = tool("evaluate", json!({"language":"python","function":"PyNestElif.f","input":"1"}), &cfg, &mut ws, &baseline).await;
        assert!(!result.execution.has_errors && result.content.contains('2'), "{}", result.content);
    });
}
