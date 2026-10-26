//! Owner-aware catalog regressions through the actual native dispatch pipeline.
use super::*;
use serde_json::json;

pub(super) fn permissions(hidden: Option<&str>) -> PermissionConfig {
    let source = match hidden {
        Some(module) => format!(
            "[groups]\nprivate = ['{module}']\n[model.writer]\nprivate = 'none'\nuser = 'write'\n"
        ),
        None => "[model.writer]\nuser = 'write'\n".into(),
    };
    toml::from_str(&source).unwrap()
}

pub(super) async fn mutate(
    cfg: &AppConfig,
    ws: &mut WorkingSet,
    baseline: &Program,
    module: &str,
    source: &str,
) -> super::super::NativeToolResult {
    execute_native_tool(
        "mutate",
        &json!({"language":"python","module":module,"source":source}).to_string(),
        cfg,
        ws,
        baseline,
    )
    .await
}

pub(super) async fn inspect(cfg: &AppConfig, ws: &mut WorkingSet, baseline: &Program, target: &str) -> String {
    let result = execute_native_tool(
        "inspect",
        &json!({"kind":"source","format":"python","target":target}).to_string(),
        cfg,
        ws,
        baseline,
    )
    .await;
    assert!(!result.execution.has_errors, "{}", result.content);
    serde_json::from_str::<serde_json::Value>(&result.content).unwrap()["query_results"]
        .as_str()
        .unwrap()
        .to_owned()
}

pub(super) async fn roundtrip(
    cfg: &AppConfig,
    ws: &mut WorkingSet,
    baseline: &Program,
    module: &str,
    source: &str,
) {
    let result = mutate(cfg, ws, baseline, module, source).await;
    assert!(!result.execution.has_errors, "{}", result.content);
    cfg.write_back_working_set(ws).await;
    let printed = inspect(cfg, ws, baseline, &format!("{module}.read")).await;
    let before = ws
        .program
        .get_function(&format!("{module}.read"))
        .unwrap()
        .clone();
    let result = mutate(cfg, ws, baseline, module, &printed).await;
    assert!(!result.execution.has_errors, "{}", result.content);
    cfg.write_back_working_set(ws).await;
    assert_eq!(
        inspect(cfg, ws, baseline, &format!("{module}.read")).await,
        printed
    );
    let after = ws.program.get_function(&format!("{module}.read")).unwrap();
    assert_eq!(after.return_type, before.return_type);
    assert_eq!(after.params[0].ty, before.params[0].ty);
}

#[tokio::test]
async fn python_catalog_same_named_structs_use_declaration_owner_in_both_orders() {
    for reverse in [false, true] {
        let a = format!("NtCatalogStructA{}", u8::from(reverse));
        let b = format!("NtCatalogStructB{}", u8::from(reverse));
        let first = format!("+module {a}\n+type Point = x:Int\n");
        let second = format!("+module {b}\n+type Point = x:String\n");
        let baseline = build_program(&if reverse {
            format!("{second}{first}")
        } else {
            format!("{first}{second}")
        })
        .await;
        let cfg = config_for(
            "writer",
            AccessLevel::Full,
            permissions(None),
            baseline.clone(),
        );
        let mut ws = cfg.snapshot_working_set().await;
        for (module, py_type, input, expected, ast_type) in [
            (&a, "int", "{x:7}", "7", crate::ast::Type::Int),
            (
                &b,
                "str",
                "{x:\"text\"}",
                "\"text\"",
                crate::ast::Type::String,
            ),
        ] {
            let printed_type = inspect(&cfg, &mut ws, &baseline, &format!("{module}.Point")).await;
            assert!(
                printed_type.contains(&format!("x: {py_type}")),
                "{printed_type}"
            );
            roundtrip(
                &cfg,
                &mut ws,
                &baseline,
                module,
                &format!("def read(p: Point) -> {py_type}:\n    return p.x\n"),
            )
            .await;
            let function = ws.program.get_function(&format!("{module}.read")).unwrap();
            assert_eq!(function.return_type, ast_type);
            assert_eq!(
                function.params[0].ty,
                crate::ast::Type::Struct("Point".into()),
                "internal owner must not leak into persisted AST"
            );
            assert_eq!(
                crate::eval::eval_call_with_input(
                    &ws.program,
                    &format!("{module}.read"),
                    &crate::parser::parse_test_input(1, input).unwrap()
                )
                .unwrap(),
                expected
            );
        }
    }
}

#[tokio::test]
async fn python_catalog_recursive_same_named_unions_keep_payload_owner() {
    for reverse in [false, true] {
        let a = format!("NtCatalogUnionA{}", u8::from(reverse));
        let b = format!("NtCatalogUnionB{}", u8::from(reverse));
        let first = format!("+module {a}\n+type Expr = Leaf(Int) | Wrap(Expr)\n");
        let second = format!("+module {b}\n+type Expr = Leaf(String) | Wrap(Expr)\n");
        let baseline = build_program(&if reverse {
            format!("{second}{first}")
        } else {
            format!("{first}{second}")
        })
        .await;
        let cfg = config_for(
            "writer",
            AccessLevel::Full,
            permissions(None),
            baseline.clone(),
        );
        let mut ws = cfg.snapshot_working_set().await;
        for (module, leaf, wrap, py_type, zero, input, expected) in [
            (&a, "Leaf", "Wrap", "int", "0", "Wrap(Leaf(8))", "8"),
            (
                &b,
                "Leaf",
                "Wrap",
                "str",
                "''",
                "Wrap(Leaf(\"word\"))",
                "\"word\"",
            ),
        ] {
            let type_source = inspect(&cfg, &mut ws, &baseline, &format!("{module}.Expr")).await;
            assert!(
                type_source.contains(&format!("{leaf}: tuple[{py_type}]")),
                "{type_source}"
            );
            let source = format!(
                "def read(value: Expr) -> {py_type}:\n    match value:\n        case {wrap}({leaf}(item)):\n            return item\n        case _:\n            return {zero}\n"
            );
            roundtrip(&cfg, &mut ws, &baseline, module, &source).await;
            let function = ws.program.get_function(&format!("{module}.read")).unwrap();
            assert_eq!(
                function.params[0].ty,
                crate::ast::Type::TaggedUnion("Expr".into())
            );
            assert_eq!(
                function.return_type,
                if py_type == "int" {
                    crate::ast::Type::Int
                } else {
                    crate::ast::Type::String
                }
            );
            assert_eq!(
                crate::eval::eval_call_with_input(
                    &ws.program,
                    &format!("{module}.read"),
                    &crate::parser::parse_test_input(1, input).unwrap()
                )
                .unwrap(),
                expected
            );
        }
    }
}

#[tokio::test]
async fn python_catalog_external_signature_keeps_callee_owner_not_caller() {
    let baseline = build_program("+module NtCatalogSignatureA\n+type Point = x:Int\n+fn take (p:Point)->Int\n +return p.x\n+end\n+fn make ()->Point\n +return {x:1}\n+end\n+module NtCatalogSignatureB\n+type Point = x:String\n").await;
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        permissions(None),
        baseline.clone(),
    );
    let mut ws = cfg.snapshot_working_set().await;
    for source in [
        "def wrong(p: Point) -> int:\n    return NtCatalogSignatureA.take(p)\n",
        "def wrong() -> Point:\n    return NtCatalogSignatureA.make()\n",
    ] {
        let before = ws.program.clone();
        let result = mutate(&cfg, &mut ws, &baseline, "NtCatalogSignatureB", source).await;
        assert!(result.execution.has_errors, "{}", result.content);
        assert_eq!(ws.program, before);
    }
}

#[tokio::test]
async fn python_catalog_denied_competitors_do_not_change_source_or_diagnostics() {
    let mut observed = Vec::new();
    for field_type in ["String", "Bool"] {
        let baseline = build_program(&format!("+module NtCatalogVisible\n+type Point = x:Int\n+fn read (p:Point)->Int\n +return p.x\n+end\n+module NtCatalogDenied\n+type Point = secret:{field_type}\n")).await;
        let cfg = config_for(
            "writer",
            AccessLevel::Full,
            permissions(Some("NtCatalogDenied")),
            baseline.clone(),
        );
        let mut ws = cfg.snapshot_working_set().await;
        let source = inspect(&cfg, &mut ws, &baseline, "NtCatalogVisible.read").await;
        let error = mutate(
            &cfg,
            &mut ws,
            &baseline,
            "NtCatalogVisible",
            "def wrong(p: Point) -> str:\n    return p.x\n",
        )
        .await;
        assert!(
            error.execution.has_errors && !error.content.contains("secret"),
            "{}",
            error.content
        );
        observed.push((source, error.content));
    }
    assert_eq!(observed[0], observed[1]);
}

#[tokio::test]
async fn python_catalog_ambiguous_fallback_is_rejected_independently_of_order() {
    let mut errors = Vec::new();
    for reverse in [false, true] {
        let a = "+module NtCatalogFallbackA\n+type Point = x:Int\n";
        let b = "+module NtCatalogFallbackB\n+type Point = x:String\n";
        let baseline = build_program(&if reverse {
            format!("{b}{a}")
        } else {
            format!("{a}{b}")
        })
        .await;
        let cfg = config_for(
            "writer",
            AccessLevel::Full,
            permissions(None),
            baseline.clone(),
        );
        let mut ws = cfg.snapshot_working_set().await;
        let result = mutate(
            &cfg,
            &mut ws,
            &baseline,
            "NtCatalogFallbackC",
            "def read(p: Point) -> int:\n    return p.x\n",
        )
        .await;
        assert!(
            result.execution.has_errors && result.content.contains("ambiguous type"),
            "{}",
            result.content
        );
        assert_eq!(ws.program, baseline);
        errors.push(result.content);
    }
    assert_eq!(errors[0], errors[1]);
}
