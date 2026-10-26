//! Native canaries for filtered-to-full type binding, before executor validation.
use super::python_catalog_tests::{inspect, mutate, permissions, roundtrip};
use super::*;

const REFUSAL: &str = "Python type binding cannot be preserved safely by canonical validation";
const CONSTRUCT: &str = "def f() -> int:\n    p: Point = Point(x=1)\n    return 0\n";

async fn refuses_before_executor(
    program: Program,
    hidden: &str,
    consumer: &str,
    source: &str,
) -> String {
    let before = serde_json::to_value(&program).unwrap();
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        permissions(Some(hidden)),
        program.clone(),
    );
    let mut ws = cfg.snapshot_working_set().await;
    let arguments =
        serde_json::json!({"language":"python","module":consumer,"source":source}).to_string();
    match super::super::prepare_native_tool("mutate", &arguments, &cfg, &ws.program) {
        Err(error) => assert!(error.contains(REFUSAL), "{error}"),
        Ok(_) => panic!("admission must refuse before execute_code can validate a hidden schema"),
    }
    let result = mutate(&cfg, &mut ws, &program, consumer, source).await;
    assert!(
        result.execution.has_errors && result.content.contains(REFUSAL),
        "{}",
        result.content
    );
    // denied_result represents admission errors as synthetic failed mutations.
    assert!(
        result
            .execution
            .mutation_results
            .iter()
            .all(|r| !r.success && r.message.contains(REFUSAL))
    );
    assert_eq!(serde_json::to_value(&ws.program).unwrap(), before);
    assert_eq!(
        serde_json::to_value(&*cfg.program.read().await).unwrap(),
        before
    );
    result.content
}

#[tokio::test]
async fn python_binding_guard_hidden_struct_fallback_never_reaches_schema_validation() {
    let mut diagnostics = Vec::new();
    for reverse in [false, true] {
        for (field, ty) in [
            ("secretAlphaCanary", "String"),
            ("secretBetaCanary", "Bool"),
        ] {
            let hidden = format!("+module NtBindingHidden\n+type Point = {field}:{ty}\n");
            let allowed = "+module NtBindingAllowed\n+type Point = x:Int\n";
            let program = build_program(&if reverse {
                format!("{allowed}{hidden}")
            } else {
                format!("{hidden}{allowed}")
            })
            .await;
            let result =
                refuses_before_executor(program, "NtBindingHidden", "NtBindingConsumer", CONSTRUCT)
                    .await;
            assert!(
                !result.contains(field) && !result.contains("NtBindingHidden"),
                "{result}"
            );
            diagnostics.push(result);
        }
    }
    assert!(
        diagnostics.iter().all(|d| d == &diagnostics[0]),
        "hidden fields/order altered refusal"
    );
}

#[tokio::test]
async fn python_binding_guard_local_struct_cannot_be_reclassified_by_foreign_union() {
    let mut diagnostics = Vec::new();
    for reverse in [false, true] {
        for variant in ["SecretVariantAlpha", "SecretVariantBeta"] {
            let hidden = format!(
                "+module NtBindingUnionHidden\n+type Point = {variant}(String) | EndCanary\n"
            );
            let local = "+module NtBindingUnionLocal\n+type Point = x:Int\n";
            let program = build_program(&if reverse {
                format!("{local}{hidden}")
            } else {
                format!("{hidden}{local}")
            })
            .await;
            let result = refuses_before_executor(
                program,
                "NtBindingUnionHidden",
                "NtBindingUnionLocal",
                CONSTRUCT,
            )
            .await;
            assert!(!result.contains(variant), "{result}");
            diagnostics.push(result);
        }
    }
    assert!(diagnostics.iter().all(|d| d == &diagnostics[0]));
}

#[tokio::test]
async fn python_binding_guard_keeps_safe_local_top_level_and_unique_fallback() {
    let cases = [
        (
            "local",
            "+module NtBindingSafeHidden\n+type Point = privateCanary:String\n+module NtBindingSafeLocal\n+type Point = x:Int\n",
            "NtBindingSafeLocal",
        ),
        (
            "top",
            "+type Point = x:Int\n+module NtBindingSafeHidden\n+type Point = privateCanary:String\n",
            "NtBindingSafeTop",
        ),
        (
            "fallback",
            "+module NtBindingSafeAllowed\n+type Point = x:Int\n+module NtBindingSafeHidden\n+type Unrelated = privateCanary:String\n",
            "NtBindingSafeFallback",
        ),
    ];
    for (label, source, consumer) in cases {
        let program = build_program(source).await;
        let cfg = config_for(
            "writer",
            AccessLevel::Full,
            permissions(Some("NtBindingSafeHidden")),
            program.clone(),
        );
        let mut ws = cfg.snapshot_working_set().await;
        let result = mutate(&cfg, &mut ws, &program, consumer, CONSTRUCT).await;
        assert!(!result.execution.has_errors, "{label}: {}", result.content);
        let f = ws.program.get_function(&format!("{consumer}.f")).unwrap();
        let crate::ast::StatementKind::Let { ty, .. } = &f.body[0].kind else {
            panic!("expected binding")
        };
        assert_eq!(ty, &crate::ast::Type::Struct("Point".into()));
        assert_eq!(
            crate::eval::eval_call_with_input(
                &ws.program,
                &format!("{consumer}.f"),
                &crate::parser::parse_test_input(1, "").unwrap()
            )
            .unwrap(),
            "0"
        );
    }
}

#[tokio::test]
async fn python_binding_guard_respects_type_install_order_without_executing_earlier_mutations() {
    let program =
        build_program("+module NtBindingOrderHidden\n+type Point = secretOrderCanary:String\n")
            .await;
    let source = "def f() -> int:\n    p: Point = Point(x=1)\n    return 0\n@dataclass\nclass Point:\n    x: int\n";
    let result = refuses_before_executor(
        program,
        "NtBindingOrderHidden",
        "NtBindingOrderConsumer",
        source,
    )
    .await;
    assert!(!result.contains("secretOrderCanary"));
    // Same owner/kind is insufficient when lowering saw a future replacement.
    let program =
        build_program("+module NtBindingOrderConsumer\n+type Point = previousShapeCanary:String\n")
            .await;
    let result = refuses_before_executor(
        program,
        "NtBindingOrderHidden",
        "NtBindingOrderConsumer",
        source,
    )
    .await;
    assert!(!result.contains("previousShapeCanary"));
}

#[tokio::test]
async fn python_binding_guard_recursive_same_named_structs_keep_field_and_signature_owner() {
    for reverse in [false, true] {
        let a = format!("NtBindingRecursiveA{}", u8::from(reverse));
        let b = format!("NtBindingRecursiveB{}", u8::from(reverse));
        let first = format!(
            "+module {a}\n+type Node = value:Int, next:Option<Node>\n+fn take (p:Node)->Int\n +return p.value\n+end\n"
        );
        let second = format!(
            "+module {b}\n+type Node = value:String, next:Option<Node>\n+fn take (p:Node)->String\n +return p.value\n+end\n"
        );
        let program = build_program(&if reverse {
            format!("{second}{first}")
        } else {
            format!("{first}{second}")
        })
        .await;
        let cfg = config_for(
            "writer",
            AccessLevel::Full,
            permissions(None),
            program.clone(),
        );
        let mut ws = cfg.snapshot_working_set().await;
        for (owner, ty, input, expected) in [
            (&a, "int", "{value:3,next:None}", "3"),
            (&b, "str", "{value:\"text\",next:None}", "\"text\""),
        ] {
            let type_source = inspect(&cfg, &mut ws, &program, &format!("{owner}.Node")).await;
            assert!(
                type_source.contains(&format!("value: {ty}"))
                    && type_source.contains("Option[Node]"),
                "{type_source}"
            );
            roundtrip(
                &cfg,
                &mut ws,
                &program,
                owner,
                &format!("def read(p: Node) -> {ty}:\n    return {owner}.take(p)\n"),
            )
            .await;
            assert_eq!(
                crate::eval::eval_call_with_input(
                    &ws.program,
                    &format!("{owner}.read"),
                    &crate::parser::parse_test_input(1, input).unwrap()
                )
                .unwrap(),
                expected
            );
        }
        let result = mutate(
            &cfg,
            &mut ws,
            &program,
            &b,
            &format!("def wrong(p: Node) -> int:\n    return {a}.take(p)\n"),
        )
        .await;
        assert!(result.execution.has_errors, "{}", result.content);
    }
}
