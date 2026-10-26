//! Actual native admission under user=none: values are not namespaces and
//! constructor roles are server-derived, target-module-only evidence.
use super::python_catalog_tests::{inspect, mutate, permissions};
use super::*;
use serde_json::json;

async fn fixture(_tag: &str) -> (AppConfig, WorkingSet, Program) {
    let program =
        build_program("+module NtReadmeSites\n+fn idle ()->Int\n +return 0\n+end\n").await;
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        restricted("NtReadmeExample", ""),
        program.clone(),
    );
    let ws = cfg.snapshot_working_set().await;
    (cfg, ws, program)
}

/// #95 A/B/C: keyboard/post-fix behaviour must stay discriminating.
/// A: colliding legacy program keeps unrelated functions evaluable.
/// B: a Python function colliding with a distant variant is refused at
///    declaration time, so the unreachable-all state cannot be entered.
/// C: a reachable ambiguous call is still refused; unrelated replays of the
///    old global predicate would have refused A's evaluate/inspect/test.
#[tokio::test]
async fn python_reference_95_distant_variant_collision_does_not_disable_unrelated_python() {
    let baseline = build_program(
        "+module Nt95Distant\n+type Legacy = LegacyLeaf(Int) | LegacyEmpty\n+module Nt95Unrelated\n+fn answer ()->Int\n +return 42\n+end\n+fn why ()->Int\n +return 7\n+end\n",
    )
    .await;
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        toml::from_str(
            "[groups]\ndistant = ['Nt95Distant']\n[model.writer]\ndistant = 'read'\nuser = 'write'\n",
        )
        .unwrap(),
        baseline.clone(),
    );
    let mut ws = cfg.snapshot_working_set().await;

    // A: legacy collision exists elsewhere, unrelated function still evaluates.
    let evaluated = execute_native_tool(
        "evaluate",
        r#"{"language":"python","function":"Nt95Unrelated.answer","input":""}"#,
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(
        !evaluated.execution.has_errors
            && evaluated
                .execution
                .mutation_results
                .iter()
                .any(|r| r.success && r.message.ends_with("= 42")),
        "A unrelated evaluate: {}",
        evaluated.content
    );
    let inspected = execute_native_tool(
        "inspect",
        r#"{"kind":"source","target":"Nt95Unrelated.answer","format":"python"}"#,
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(
        !inspected.execution.has_errors,
        "A unrelated inspect: {}",
        inspected.content
    );
    let tested = execute_native_tool(
        "test",
        &serde_json::json!({"language":"python","module":"Nt95Unrelated","source":"def test_answer():\n    assert answer() == 42\n"}).to_string(),
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(
        !tested.execution.has_errors,
        "A unrelated test: {}",
        tested.content
    );

    // B: Python declaring a function named like a distant variant is refused.
    let before = serde_json::to_value(&ws.program).unwrap();
    let collision = mutate(
        &cfg,
        &mut ws,
        &baseline,
        "Nt95Collide",
        "def LegacyLeaf(x: int) -> int:\n    return x\n",
    )
    .await;
    assert!(
        collision.execution.has_errors
            && collision
                .content
                .contains("collides with a union variant name"),
        "B declaration must be refused: {}",
        collision.content
    );
    assert_eq!(
        serde_json::to_value(&ws.program).unwrap(),
        before,
        "B no mutation"
    );
    assert!(
        !collision.content.contains("Nt95Distant"),
        "refusal must not name the hidden module"
    );

    // C: a reachable ambiguous spelling is still refused, and the program is
    // still usable afterwards (the deadlock this issue tracks is gone).
    let ambiguous = mutate(
        &cfg,
        &mut ws,
        &baseline,
        "Nt95Unrelated",
        "def LegacyLeaf(x: int) -> int:\n    return x\n",
    )
    .await;
    assert!(
        ambiguous.execution.has_errors,
        "C must refuse: {}",
        ambiguous.content
    );
    let after = execute_native_tool(
        "evaluate",
        r#"{"language":"python","function":"Nt95Unrelated.answer","input":""}"#,
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(
        !after.execution.has_errors,
        "C program must remain usable: {}",
        after.content
    );
}

/// #95: the runtime recheck is entry-scoped. A hand-built legacy program whose
/// call graph from `certified` never touches the ambiguous spelling must still
/// execute, while an entry that does reach it is refused.
#[tokio::test]
async fn python_reference_95_reachable_graph_scope_is_entry_local() {
    let baseline = build_program(
        "+module Nt95Graph\n+type Legacy = LegacyLeaf(Int) | LegacyEmpty\n+module Nt95GraphOther\n+fn LegacyLeaf (x:Int)->Int\n +return x+1\n+end\n+fn ident (x:Int)->Int\n +return x\n+end\n+fn certified ()->Int\n +return 5\n+end\n+fn poisoned ()->Int\n +return LegacyLeaf(1)\n+end\n+fn nested ()->Int\n +return ident(LegacyLeaf(1))\n+end\n",
    )
    .await;
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        restricted("Nt95GraphOther", ""),
        baseline.clone(),
    );
    let mut ws = cfg.snapshot_working_set().await;
    let clean = execute_native_tool(
        "evaluate",
        r#"{"language":"python","function":"Nt95GraphOther.certified","input":""}"#,
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(
        !clean.execution.has_errors,
        "clean entry: {}",
        clean.content
    );
    let poisoned = execute_native_tool(
        "evaluate",
        r#"{"language":"python","function":"Nt95GraphOther.poisoned","input":""}"#,
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(
        poisoned.execution.has_errors && poisoned.content.contains("constructor role"),
        "reachable ambiguous entry must refuse: {}",
        poisoned.content
    );
    // F1: the collision reached only through a NESTED call argument must be
    // detected too; otherwise evaluate and test silently disagrees.
    let nested = execute_native_tool(
        "evaluate",
        r#"{"language":"python","function":"Nt95GraphOther.nested","input":""}"#,
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(
        nested.execution.has_errors && nested.content.contains("constructor role"),
        "nested-argument ambiguous entry must refuse: {}",
        nested.content
    );
}
fn restricted(owner: &str, extra: &str) -> PermissionConfig {
    toml::from_str(&format!("[groups]\nown = ['{owner}']\n{extra}\n[model.writer]\nown = 'write'\nuser = 'none'\nexternal = 'execute'\n")).unwrap()
}

#[tokio::test]
async fn python_reference_local_constructors_roundtrip_test_and_evaluate() {
    let baseline = Program::default();
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        restricted("NtRoleConstruct", ""),
        baseline.clone(),
    );
    let mut ws = cfg.snapshot_working_set().await;
    let source = "@union\nclass Expr:\n    Leaf: tuple[int]\n    Empty: tuple[()]\ndef score() -> int:\n    value: Expr = Leaf(7)\n    match value:\n        case Leaf(number):\n            return number\n        case Empty():\n            return 0\ndef empty_score() -> int:\n    value: Expr = Empty\n    match value:\n        case Leaf(number):\n            return number\n        case Empty():\n            return 0\n";
    let created = mutate(&cfg, &mut ws, &baseline, "NtRoleConstruct", source).await;
    assert!(!created.execution.has_errors, "{}", created.content);
    cfg.write_back_working_set(&ws).await;
    let printed = inspect(&cfg, &mut ws, &baseline, "NtRoleConstruct.score").await;
    assert!(
        printed.contains("Leaf(7)") && !printed.contains("NtRoleConstruct.Leaf("),
        "{printed}"
    );
    let resubmitted = mutate(&cfg, &mut ws, &baseline, "NtRoleConstruct", &printed).await;
    assert!(!resubmitted.execution.has_errors, "{}", resubmitted.content);
    cfg.write_back_working_set(&ws).await;
    assert_eq!(
        inspect(&cfg, &mut ws, &baseline, "NtRoleConstruct.score").await,
        printed
    );
    let tested = execute_native_tool("test", &json!({"language":"python","module":"NtRoleConstruct","source":"def test_score():\n    assert score() == 7\ndef test_empty_score():\n    assert empty_score() == 0\n"}).to_string(), &cfg, &mut ws, &baseline).await;
    assert!(!tested.execution.has_errors, "{}", tested.content);
    for (function, expected) in [("score", "7"), ("empty_score", "0")] {
        let result = execute_native_tool("evaluate", &json!({"language":"python","function":format!("NtRoleConstruct.{function}"),"input":""}).to_string(), &cfg, &mut ws, &baseline).await;
        assert!(
            !result.execution.has_errors
                && result.execution.mutation_results.iter().any(|r| r
                    .message
                    .rsplit_once(" = ")
                    .is_some_and(|(_, value)| value == expected)),
            "{}",
            result.content
        );
    }
    for expression in ["Leaf()", "Leaf(1, 2)", "Leaf('wrong')"] {
        let before = serde_json::to_value(&ws.program).unwrap();
        let result = mutate(
            &cfg,
            &mut ws,
            &baseline,
            "NtRoleConstruct",
            &format!("def bad() -> Expr:\n    return {expression}\n"),
        )
        .await;
        assert!(result.execution.has_errors, "{}", result.content);
        assert_eq!(serde_json::to_value(&ws.program).unwrap(), before);
    }
}

#[tokio::test]
async fn python_reference_lexical_fields_need_no_phantom_module_permission() {
    let baseline = build_program("+module NtRoleFields\n+type Child = x:Int\n+type Point = child:Child\n+module p\n+fn child ()->Int\n +return 987654\n+end\n").await;
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        restricted("NtRoleFields", ""),
        baseline.clone(),
    );
    let mut ws = cfg.snapshot_working_set().await;
    let source = "def read(p: Point) -> int:\n    return p.child.x\n";
    let result = mutate(&cfg, &mut ws, &baseline, "NtRoleFields", source).await;
    assert!(!result.execution.has_errors, "{}", result.content);
    cfg.write_back_working_set(&ws).await;
    let printed = inspect(&cfg, &mut ws, &baseline, "NtRoleFields.read").await;
    assert!(printed.contains("p.child.x") && !printed.contains("987654"));
    let result = mutate(&cfg, &mut ws, &baseline, "NtRoleFields", &printed).await;
    assert!(!result.execution.has_errors, "{}", result.content);
    let result = execute_native_tool(
        "evaluate",
        &json!({"function":"NtRoleFields.read","input":"{child:{x:9}}"}).to_string(),
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(
        !result.execution.has_errors && result.content.contains("= 9"),
        "{}",
        result.content
    );
    let scoped = "@union\nclass Box:\n    Wrap: tuple[Point]\n    Empty: tuple[()]\ndef capture(p: Point) -> int:\n    value: Box = Wrap(p)\n    match value:\n        case Wrap(item):\n            return item.child.x\n        case Empty():\n            return 0\n@effects('mut')\ndef total(points: list[Point]) -> int:\n    result: int = 0\n    for item in points:\n        result = result + item.child.x\n    return result\n";
    let result = mutate(&cfg, &mut ws, &baseline, "NtRoleFields", scoped).await;
    assert!(!result.execution.has_errors, "{}", result.content);
    cfg.write_back_working_set(&ws).await;
    let tests = "+test NtRoleFields.capture\n +with {child:{x:9}} -> expect 9\n+test NtRoleFields.total\n +with list({child:{x:9}}) -> expect 9\n";
    let tested = execute_native_tool(
        "test",
        &json!({"source":tests}).to_string(),
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(!tested.execution.has_errors, "{}", tested.content);
    for body in [
        "def bad(p: Point) -> int:\n    return p.child()\n",
        "def bad() -> int:\n    return p.child.x\n    p: Point = Point(child=Child(x=1))\n",
        "def bad(flag: bool) -> int:\n    if flag:\n        p: Point = Point(child=Child(x=1))\n    return p.child.x\n",
        "def bad(p: Point) -> int:\n    return p.missing\n",
    ] {
        let before = serde_json::to_value(&ws.program).unwrap();
        let result = mutate(&cfg, &mut ws, &baseline, "NtRoleFields", body).await;
        assert!(
            result.execution.has_errors && !result.content.contains("987654"),
            "{}",
            result.content
        );
        assert_eq!(serde_json::to_value(&ws.program).unwrap(), before);
    }
}

#[tokio::test]
async fn python_reference_hidden_function_collisions_are_neutral_and_external_variants_refused() {
    let source = "@union\nclass Expr:\n    Leaf: tuple[int]\n    Empty: tuple[()]\ndef f() -> Expr:\n    return Leaf(1)\n";
    let mut errors = Vec::new();
    for signature in ["secretAlpha:Int", "secretBeta:String, other:Bool"] {
        let baseline = build_program(&format!(
            "+module NtRoleHidden\n+fn Leaf ({signature})->Int\n +return 987654\n+end\n"
        ))
        .await;
        let cfg = config_for(
            "writer",
            AccessLevel::Full,
            restricted("NtRoleCollision", ""),
            baseline.clone(),
        );
        let mut ws = cfg.snapshot_working_set().await;
        let result = mutate(&cfg, &mut ws, &baseline, "NtRoleCollision", source).await;
        assert!(
            result.execution.has_errors
                && (result.content.contains("constructor role")
                    || result
                        .content
                        .contains("collides with a union variant name")
                    || result
                        .content
                        .contains("union variant name collides with a function")),
            "{}",
            result.content
        );
        assert!(
            !result.content.contains("secretAlpha")
                && !result.content.contains("secretBeta")
                && !result.content.contains("987654")
        );
        assert_eq!(ws.program, baseline);
        errors.push(result.content);
    }
    assert_eq!(errors[0], errors[1]);
    let baseline =
        build_program("+module NtRoleExternal\n+type Foreign = ForeignLeaf(Int) | ForeignEmpty\n")
            .await;
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        restricted("NtRoleConsumer", "external = ['NtRoleExternal']"),
        baseline.clone(),
    );
    let mut ws = cfg.snapshot_working_set().await;
    for value in ["ForeignLeaf(1)", "ForeignEmpty"] {
        let result = mutate(
            &cfg,
            &mut ws,
            &baseline,
            "NtRoleConsumer",
            &format!("def f() -> Foreign:\n    return {value}\n"),
        )
        .await;
        assert!(
            result.execution.has_errors
                && result
                    .content
                    .contains("external bare variants are unsupported"),
            "{}",
            result.content
        );
        assert_eq!(ws.program, baseline);
    }
}

#[tokio::test]
async fn python_reference_canonical_value_scope_never_exempts_calls_or_escaped_bindings() {
    let baseline = build_program("+module NtRoleWalk\n+type Point = x:Int\n+module Denied\n+fn secret ()->Int\n +return 987654\n+end\n").await;
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        restricted("NtRoleWalk", ""),
        baseline.clone(),
    );
    for body in [
        "+fn f (Denied:Point)->Int\n +return Denied.secret()\n+end\n",
        "+fn f ()->Int\n +return p.x\n +let p:Point = {x:1}\n+end\n",
        "+fn f (flag:Bool)->Int\n +if flag\n  +let p:Point = {x:1}\n +end\n +return p.x\n+end\n",
        "+fn f ()->Int\n +each list({x:1}) p:Point\n  +check valid true ~err_valid\n +end\n +return p.x\n+end\n",
    ] {
        let operations = crate::parser::parse(&format!("+module NtRoleWalk\n{body}")).unwrap();
        let error = super::super::authored::check_operations(
            &operations,
            &cfg,
            "writer",
            Some((&baseline, true)),
            None,
        )
        .unwrap_err();
        assert!(error.contains("permission denied"), "{error}");
    }
}

#[tokio::test]
async fn python_reference_execution_rechecks_later_constructor_function_collision() {
    let baseline = build_program("+module NtRoleLater\n+type Expr = Leaf(Int) | Empty\n+fn score ()->Int\n +let value:Expr = Leaf(7)\n +match value\n +case Leaf(n)\n  +return n\n +case Empty\n  +return 0\n +end\n+end\n").await;
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        restricted("NtRoleLater", ""),
        baseline.clone(),
    );
    let mut ws = cfg.snapshot_working_set().await;
    let hidden = build_program(
        "+module NtRoleLaterHidden\n+fn Leaf (secret:Int)->Int\n +return 987654\n+end\n",
    )
    .await;
    ws.program.modules.extend(hidden.modules);
    ws.program.rebuild_function_index();
    cfg.write_back_working_set(&ws).await;
    let before = serde_json::to_value(&ws.program).unwrap();
    let result = execute_native_tool(
        "evaluate",
        &json!({"language":"python","function":"NtRoleLater.score","input":""}).to_string(),
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(
        result.execution.has_errors && result.content.contains("constructor role"),
        "{}",
        result.content
    );
    assert!(!result.content.contains("secret") && !result.content.contains("987654"));
    assert_eq!(serde_json::to_value(&ws.program).unwrap(), before);
}

#[tokio::test]
async fn python_reference_readme_await_match_example_is_exactly_supported() {
    let (mut cfg, mut ws, baseline) = fixture("ReadmeExample").await;
    let (sender, mut receiver) = tokio::sync::mpsc::channel(4);
    cfg.io_sender = Some(sender);
    let responder = tokio::spawn(async move {
        let crate::coroutine::IoRequest::HttpGet { url, reply } = receiver.recv().await.unwrap()
        else {
            panic!("only inert HTTP fixture allowed")
        };
        assert_eq!(url, "https://example.invalid/data");
        reply.send(Ok("readme-value".into())).unwrap();
        receiver
    });
    // Copied verbatim from examples/python/README.md (#94 recommended form).
    let source = "@effects(\"io\")\nasync def fetch_ok(url: str) -> str:\n    response: Result[str] = await http_get(url)\n    match response:\n        case Ok(value):\n            return value\n        case Err(error):\n            return error\n";
    let created = mutate(&cfg, &mut ws, &baseline, "NtReadmeExample", source).await;
    assert!(!created.execution.has_errors, "{}", created.content);
    cfg.write_back_working_set(&ws).await;
    let result = tokio::time::timeout(std::time::Duration::from_secs(10), execute_native_tool("evaluate", r#"{"language":"python","function":"NtReadmeExample.fetch_ok","input":"'https://example.invalid/data'"}"#, &cfg, &mut ws, &baseline)).await.unwrap();
    assert!(
        !result.execution.has_errors && result.content.contains("readme-value"),
        "{}",
        result.content
    );
    let mut receiver = responder.await.unwrap();
    assert!(
        receiver.try_recv().is_err(),
        "documented example replayed IO"
    );
}

/// F2(a): a single mutate submitting BOTH a union and a colliding function must
/// be refused. Comparing only the pre-existing program catalog missed this.
#[tokio::test]
async fn python_reference_95_same_submission_union_function_collision_is_refused() {
    let baseline = Program::default();
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        permissions(None),
        baseline.clone(),
    );
    let mut ws = cfg.snapshot_working_set().await;
    let source = "@union\nclass Tok:\n    Leaf: tuple[()]\n    Other: tuple[()]\ndef Leaf() -> Tok:\n    return Other\n";
    let result = mutate(&cfg, &mut ws, &baseline, "Nt95SameBatch", source).await;
    assert!(
        result.execution.has_errors,
        "same-submission function/variant collision must refuse: {}",
        result.content
    );
    assert_eq!(ws.program, baseline);
}

/// F2(b): the reverse direction — declaring a variant whose name matches an
/// EXISTING function must also be refused.
#[tokio::test]
async fn python_reference_95_variant_named_after_existing_function_is_refused() {
    let baseline = build_program(
        "+module Nt95Sym\n+type Seed = A(Int) | B\n+module Nt95SymUser\n+fn Leaf ()->Int\n +return 1\n+end\n",
    )
    .await;
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        permissions(None),
        baseline.clone(),
    );
    let mut ws = cfg.snapshot_working_set().await;
    let source = "@union\nclass Tok:\n    Leaf: tuple[()]\n    Other: tuple[()]\n";
    let result = mutate(&cfg, &mut ws, &baseline, "Nt95SymUser", source).await;
    assert!(
        result.execution.has_errors
            && result
                .content
                .contains("variant name collides with a function"),
        "variant named after an existing function must refuse: {}",
        result.content
    );
    assert_eq!(ws.program, baseline);
}

/// Reviewer F2 follow-up: a variant in a module the caller cannot read must still
/// block a same-spelling Python function at declaration time. Without the full
/// catalog in the union, this admitted a function that permanently bricked every
/// bare-reachable entry (evaluate/test refused by invisible state).
#[tokio::test]
async fn python_reference_95_hidden_variant_name_blocks_function_declaration() {
    let baseline =
        build_program("+module Nt95HiddenVariant\n+type Hidden = Empty | Filled(Int)\n").await;
    // `user = write` keeps the writer model write-capable; only the hidden
    // module is at `none`.
    let permissions = toml::from_str(
        "[groups]\nhidden = ['Nt95HiddenVariant']\n[model.writer]\nhidden = 'none'\nuser = 'write'\n",
    )
    .unwrap();
    let cfg = config_for("writer", AccessLevel::Full, permissions, baseline.clone());
    let mut ws = cfg.snapshot_working_set().await;
    let before = serde_json::to_value(&ws.program).unwrap();
    let result = mutate(
        &cfg,
        &mut ws,
        &baseline,
        "Nt95HiddenConsumer",
        "def Empty() -> int:\n    return 1\n",
    )
    .await;
    assert!(
        result.execution.has_errors,
        "hidden variant spelling must block declaration: {}",
        result.content
    );
    assert!(
        result
            .content
            .contains("collides with a union variant name"),
        "{}",
        result.content
    );
    assert_eq!(
        serde_json::to_value(&ws.program).unwrap(),
        before,
        "no mutation"
    );
    assert!(
        !result.content.contains("Nt95HiddenVariant"),
        "refusal must not name the hidden module: {}",
        result.content
    );
}
