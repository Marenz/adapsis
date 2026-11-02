//! G1 through the production native path: `.len`/`.is_empty` on str/list and
//! `.is_ok`/`.is_err`/`.unwrap` on a Result apply, test, evaluate and inspect; out-of-gate
//! uses are refused without touching the program; an async function that reads and then
//! uses `.len` gets the tree result.
use super::python_catalog_tests::{inspect, mutate, permissions};
use super::*;
use serde_json::json;

const SOURCE: &str = "def size(text: str) -> int:\n    return text.len\n\ndef blank(text: str) -> bool:\n    return text.is_empty\n\ndef count(items: list[int]) -> int:\n    return items.len\n\ndef good(r: Result[int]) -> bool:\n    return r.is_ok\n\ndef bad(r: Result[int]) -> bool:\n    return r.is_err\n\ndef value_or_zero(r: Result[int]) -> int:\n    if r.is_ok:\n        return r.unwrap\n    return 0\n";

async fn evaluate(
    cfg: &AppConfig,
    ws: &mut WorkingSet,
    baseline: &Program,
    function: &str,
    input: &str,
) -> String {
    let before = serde_json::to_value(&ws.program).unwrap();
    let result = execute_native_tool(
        "evaluate",
        &json!({"language":"python","function":function,"input":input}).to_string(),
        cfg,
        ws,
        baseline,
    )
    .await;
    assert_eq!(
        before,
        serde_json::to_value(&ws.program).unwrap(),
        "evaluate is read-only"
    );
    assert!(!result.execution.has_errors, "{}", result.content);
    let content: serde_json::Value = serde_json::from_str(&result.content).unwrap();
    content["eval_results"].as_str().unwrap().to_owned()
}

#[tokio::test]
async fn native_python_pseudo_fields_mutate_test_evaluate_and_inspect() {
    let baseline =
        build_program("+module NtPseudoOther\n+fn one ()->Int\n  +return 1\n+end\n").await;
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        permissions(None),
        baseline.clone(),
    );
    let mut ws = cfg.snapshot_working_set().await;
    let created = mutate(&cfg, &mut ws, &baseline, "NtPseudo", SOURCE).await;
    assert!(!created.execution.has_errors, "{}", created.content);
    cfg.write_back_working_set(&ws).await;
    for (index, assertion) in [
        "size('héllo') == 6",
        "blank('') == True",
        "count([1, 2, 3]) == 3",
    ]
    .into_iter()
    .enumerate()
    {
        let tested = execute_native_tool(
            "test",
            &json!({"language":"python","module":"NtPseudo","source":format!("def test_case_{index}():\n    assert {assertion}\n")}).to_string(),
            &cfg,
            &mut ws,
            &baseline,
        )
        .await;
        assert!(
            !tested.execution.has_errors,
            "{assertion}: {}",
            tested.content
        );
        assert!(!tested.execution.test_results.is_empty());
    }
    assert!(
        evaluate(&cfg, &mut ws, &baseline, "NtPseudo.size", "text='héllo'")
            .await
            .contains("= 6")
    );
    assert!(
        evaluate(&cfg, &mut ws, &baseline, "NtPseudo.count", "items=[1, 2]")
            .await
            .contains("= 2")
    );
    // The runtime's transparent Result handling is the reference for these three.
    let ok = evaluate(&cfg, &mut ws, &baseline, "NtPseudo.value_or_zero", "r=3").await;
    assert!(ok.contains("= 3"), "{ok}");
    // Inspection prints the stored functions back as property syntax.
    let size = inspect(&cfg, &mut ws, &baseline, "NtPseudo.size").await;
    assert!(size.contains("return text.len"), "{size}");
    let unwrap = inspect(&cfg, &mut ws, &baseline, "NtPseudo.value_or_zero").await;
    assert!(
        unwrap.contains("if r.is_ok:") && unwrap.contains("return r.unwrap"),
        "{unwrap}"
    );
    // An `.ax` source function with the same shape is inspectable too.
    let ax = execute_native_tool(
        "mutate",
        &json!({"source":"+module NtPseudoAx\n+fn size (s:String)->Int\n  +return s.len\n+end\n"})
            .to_string(),
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(!ax.execution.has_errors, "{}", ax.content);
    let printed = inspect(&cfg, &mut ws, &baseline, "NtPseudoAx.size").await;
    assert!(
        printed.contains("def size(s: str) -> int:") && printed.contains("return s.len"),
        "{printed}"
    );
}

#[tokio::test]
async fn native_python_refuses_pseudo_fields_outside_the_type_gate_without_side_effects() {
    let baseline =
        build_program("+module NtPseudoOther\n+fn one ()->Int\n  +return 1\n+end\n").await;
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        permissions(None),
        baseline.clone(),
    );
    let mut ws = cfg.snapshot_working_set().await;
    for (index, (source, reason)) in [
        (
            "def f(r: Result[int]) -> int:\n    return r.len\n",
            "not supported on a Result",
        ),
        (
            "def f(m: Map[str, int]) -> int:\n    return m.len\n",
            "not supported on Map/Set",
        ),
        (
            "def f(s: Set[int]) -> bool:\n    return s.is_empty\n",
            "not supported on Map/Set",
        ),
        (
            "def f(n: int) -> int:\n    return n.len\n",
            "supported non-struct properties",
        ),
        (
            "def f(r: Result[int]) -> str:\n    return r.error\n",
            "supported non-struct properties",
        ),
        (
            "def f(o: Option[int]) -> bool:\n    return o.is_some\n",
            "supported non-struct properties",
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let before = serde_json::to_value(&ws.program).unwrap();
        let rejected = mutate(
            &cfg,
            &mut ws,
            &baseline,
            &format!("NtPseudoRefused{index}"),
            source,
        )
        .await;
        assert!(
            rejected.execution.has_errors,
            "{source}\n{}",
            rejected.content
        );
        assert!(
            rejected.content.contains(reason),
            "{source}\n{}",
            rejected.content
        );
        assert_eq!(
            before,
            serde_json::to_value(&ws.program).unwrap(),
            "{source}"
        );
    }
}

#[tokio::test]
async fn native_async_function_using_len_after_an_await_gives_the_tree_result() {
    let baseline =
        build_program("+module NtPseudoOther\n+fn one ()->Int\n  +return 1\n+end\n").await;
    let mut cfg = config_for(
        "writer",
        AccessLevel::Full,
        permissions(None),
        baseline.clone(),
    );
    let mut ws = cfg.snapshot_working_set().await;
    let source = "@effects('io')\nasync def read_len(path: str) -> int:\n    text: str = await read_file(path)\n    return text.len\n";
    let created = mutate(&cfg, &mut ws, &baseline, "NtPseudoAsync", source).await;
    assert!(!created.execution.has_errors, "{}", created.content);
    // Tests run on the tree with a mocked read.
    ws.meta.io_mocks.push(crate::session::IoMock {
        operation: "read_file".into(),
        patterns: vec!["fixture".into()],
        response: "héllo".into(),
    });
    let tested = execute_native_tool(
        "test",
        &json!({"language":"python","module":"NtPseudoAsync","source":"def test_case_0():\n    assert read_len('fixture') == 6\n"}).to_string(),
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(!tested.execution.has_errors, "{}", tested.content);
    assert!(!tested.execution.test_results.is_empty());
    // Normal evaluate dispatch through the IO loop with a synthetic reply, exactly once.
    // The VM would perform the read and then fail in GetField (#114); the field admission
    // sends the function to the tree before any IO.
    cfg.write_back_working_set(&ws).await;
    let (sender, mut receiver) = tokio::sync::mpsc::channel(4);
    cfg.io_sender = Some(sender);
    let responder = tokio::spawn(async move {
        match receiver.recv().await.unwrap() {
            crate::coroutine::IoRequest::FileRead { path, reply } => {
                assert_eq!(path, "fixture://pseudo");
                reply.send(Ok("héllo".to_string())).unwrap();
            }
            _ => panic!("unexpected IO"),
        }
        receiver
    });
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        execute_native_tool(
            "evaluate",
            &json!({"language":"python","function":"NtPseudoAsync.read_len","input":"path='fixture://pseudo'"}).to_string(),
            &cfg,
            &mut ws,
            &baseline,
        ),
    )
    .await
    .unwrap();
    assert!(!result.execution.has_errors, "{}", result.content);
    assert!(result.content.contains("= 6"), "{}", result.content);
    let mut receiver = responder.await.unwrap();
    assert!(receiver.try_recv().is_err(), "IO was repeated");
}
