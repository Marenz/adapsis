//! Slice 3: shared declarations are structurally preflighted and their defaults
//! are data-only and type-checked; no runtime signature or execution is used.
use super::python_catalog_tests::permissions;
use super::*;
use serde_json::json;

async fn writer() -> (AppConfig, WorkingSet, Program) {
    let baseline = Program::default();
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        permissions(None),
        baseline.clone(),
    );
    let ws = cfg.snapshot_working_set().await;
    (cfg, ws, baseline)
}

async fn attempt(
    cfg: &AppConfig,
    ws: &mut WorkingSet,
    baseline: &Program,
    module: &str,
    source: &str,
) -> NativeToolResult {
    execute_native_tool(
        "mutate",
        &json!({"language":"python","module":module,"source":source}).to_string(),
        cfg,
        ws,
        baseline,
    )
    .await
}

#[tokio::test]
async fn python_shared_positive_scalars_and_lists() {
    let (cfg, mut ws, baseline) = writer().await;
    // Native literal policy admits scalars and checked list construction. Record
    // defaults (Point(...)/Leaf(...)) remain outside that gate by existing design;
    // the frontend accepts them, the native tool does not, so they are not
    // asserted here.
    let source = "count: int = 0\nlabel: str = 'ready'\nactive: bool = True\nratio: float = 1.5\nempty: list[int] = []\nnested: list[int] = [1, 2, 3]\n";
    let created = attempt(&cfg, &mut ws, &baseline, "NtSharedPositive", source).await;
    assert!(!created.execution.has_errors, "{}", created.content);
    let module = &ws.program.modules[0];
    let names: Vec<_> = module.shared_vars.iter().map(|v| v.name.as_str()).collect();
    for expected in ["count", "label", "active", "ratio", "empty", "nested"] {
        assert!(names.contains(&expected), "{expected} missing");
    }
    let count = module
        .shared_vars
        .iter()
        .find(|v| v.name == "count")
        .unwrap();
    assert_eq!(count.ty, crate::ast::Type::Int);
}

#[tokio::test]
async fn python_shared_type_and_data_only_defaults_are_rejected_without_mutation() {
    let (cfg, mut ws, baseline) = writer().await;
    // Each case names the gate that actually rejects it, so a future gate
    // reordering cannot silently mislabel coverage.
    let cases = [
        ("count: int = 'wrong'\n", "type mismatch"),
        ("values: list[int] = [1, 'two']\n", "one element type"),
        (
            "values: list[int] = [other.call()]\n",
            "literal, list or record/variant",
        ),
        (
            "@dataclass\nclass Point:\n    x: int\ndef p() -> int:\n    return 1\npoint: Point = Point(x=other.call())\n",
            "literal, list or record/variant",
        ),
        (
            "@dataclass\nclass Point:\n    x: int\npoint: Point = Point(y=1)\n",
            "unknown field",
        ),
        (
            "@dataclass\nclass Point:\n    x: int\npoint: Point = Point()\n",
            "missing field",
        ),
        (
            "@union\nclass Kind:\n    Leaf: tuple[int]\n    Empty: tuple[()]\nkind: Kind = Leaf(1, 2)\n",
            "expects 1 payload",
        ),
        ("values: list[int] = []\nvalues: str = 'dup'\n", "duplicate"),
        ("value: int\n", "default value"),
        ("value = 1\n", "unsupported module statement"),
        ("f = f'x{1}'\n", "unsupported"),
    ];
    for (source, message) in cases {
        let before = serde_json::to_value(&ws.program).unwrap();
        let result = attempt(&cfg, &mut ws, &baseline, "NtSharedReject", source).await;
        assert!(
            result.execution.has_errors && result.content.contains(message),
            "{message:?} not reported for {source:?}: {}",
            result.content
        );
        assert_eq!(
            serde_json::to_value(&ws.program).unwrap(),
            before,
            "{source:?} mutated"
        );
    }
}

/// Union payload type mismatches are checked by the frontend data-only walker.
/// The native literal gate admits only `list(...)` calls, so a `Leaf(...)` default
/// is refused natively before payload typing can be observed through that path.
#[tokio::test]
async fn python_shared_variant_payload_type_is_checked() {
    let (cfg, mut ws, baseline) = writer().await;
    let before = serde_json::to_value(&ws.program).unwrap();
    let source = "@union\nclass Kind:\n    Leaf: tuple[int]\n    Empty: tuple[()]\nkind: Kind = Leaf('wrong')\n";
    let result = attempt(&cfg, &mut ws, &baseline, "NtSharedVariantType", source).await;
    assert!(result.execution.has_errors, "{}", result.content);
    assert_eq!(serde_json::to_value(&ws.program).unwrap(), before);
}

#[tokio::test]
async fn python_shared_unknown_is_not_proof_of_default_type() {
    let (cfg, mut ws, baseline) = writer().await;
    let before = serde_json::to_value(&ws.program).unwrap();
    let source = "count: int = Other.value()\n";
    let result = attempt(&cfg, &mut ws, &baseline, "NtSharedUnknown", source).await;
    assert!(result.execution.has_errors, "{}", result.content);
    assert_eq!(serde_json::to_value(&ws.program).unwrap(), before);
}

#[tokio::test]
async fn python_shared_batch_preflight_rejects_whole_source_before_any_effect() {
    let (cfg, mut ws, baseline) = writer().await;
    let source = "@dataclass\nclass Good:\n    x: int\nbroken: list[int] = [1, nested.call()]\n";
    let before = serde_json::to_value(&ws.program).unwrap();
    let result = attempt(&cfg, &mut ws, &baseline, "NtSharedBatch", source).await;
    assert!(result.execution.has_errors, "{}", result.content);
    assert_eq!(serde_json::to_value(&ws.program).unwrap(), before);
    assert!(ws.program.get_function("NtSharedBatch.Good").is_none());
}

/// F3: typed RECORD shared defaults are natively admitted end-to-end (the
/// native literal gate permits StructLiteral); only VARIANT constructor defaults
/// are refused by that gate while the frontend accepts them.
#[tokio::test]
async fn python_shared_typed_record_default_is_natively_admitted() {
    let (cfg, mut ws, baseline) = writer().await;
    let source =
        "@dataclass\nclass Point:\n    x: int\n    y: int\norigin: Point = Point(x=0, y=0)\n";
    let created = attempt(&cfg, &mut ws, &baseline, "NtSharedRecord", source).await;
    assert!(
        !created.execution.has_errors,
        "record default must be admitted: {}",
        created.content
    );
    let module = &ws.program.modules[0];
    let origin = module
        .shared_vars
        .iter()
        .find(|v| v.name == "origin")
        .expect("origin shared var");
    assert_eq!(origin.ty, crate::ast::Type::Struct("Point".into()));
    assert!(
        matches!(&origin.default, crate::ast::Expr::StructInit { .. }),
        "default must persist as a record: {:?}",
        origin.default
    );
}

/// Admitted variant defaults must not smuggle computation: only literal payload
/// data is allowed, and the frontend walker's arity/type checks still apply.
#[tokio::test]
async fn python_shared_variant_default_cannot_smuggle_calls() {
    let (cfg, mut ws, baseline) = writer().await;
    let cases = [
        (
            "@union\nclass Kind:\n    Leaf: tuple[int]\n    Empty: tuple[()]\nkind: Kind = Leaf(compute())\n",
            "literal, list or record/variant",
        ),
        (
            "@union\nclass Kind:\n    Leaf: tuple[int]\n    Empty: tuple[()]\nkind: Kind = Leaf(1, 2)\n",
            "expects 1 payload",
        ),
        (
            "@union\nclass Kind:\n    Leaf: tuple[int]\n    Empty: tuple[()]\nkind: Kind = Leaf('wrong')\n",
            "expects Int",
        ),
        (
            "@union\nclass Kind:\n    Leaf: tuple[int]\n    Empty: tuple[()]\nkind: Kind = Missing(1)\n",
            "literal, list or record/variant",
        ),
        (
            "@union\nclass Kind:\n    Leaf: tuple[int]\n    Empty: tuple[()]\nkind: Kind = Leaf(other.call())\n",
            "literal, list or record/variant",
        ),
        // A variant default must not route around the #95 declaration guards:
        // a function anywhere named Leaf still blocks a function named Leaf, and
        // a variant-default-bearing submission cannot declare such a function.
        (
            "@union\nclass Kind:\n    Leaf: tuple[int]\n    Empty: tuple[()]\nkind: Kind = Leaf(1)\ndef Leaf() -> int:\n    return 1\n",
            "collides",
        ),
    ];
    for (source, message) in cases {
        let before = serde_json::to_value(&ws.program).unwrap();
        let result = attempt(&cfg, &mut ws, &baseline, "NtSharedSmuggle", source).await;
        assert!(
            result.execution.has_errors,
            "{message:?} not refused: {}",
            result.content
        );
        assert!(
            result.content.contains(message),
            "{message:?} not reported: {}",
            result.content
        );
        assert_eq!(serde_json::to_value(&ws.program).unwrap(), before);
    }
}

/// The narrower gate applies only to shared defaults. Test inputs and exact
/// expectations must still reject variant constructors.
#[tokio::test]
async fn python_shared_variant_gate_does_not_widen_test_expectations() {
    let (cfg, mut ws, baseline) = writer().await;
    let result = execute_native_tool(
        "test",
        &serde_json::json!({"language":"python","module":"NtSharedNoWiden","source":"def test_x():\n    assert value() == Leaf(1)\n"}).to_string(),
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(result.execution.has_errors, "{}", result.content);
}

/// Claim (1) verification: the widened gate must admit through the REAL
/// `execute_native_tool("mutate")` chain (finalize → re-auth → runtime_targets),
/// and the persisted AST must show the union type and the constructor default.
/// `Leaf(7)` (call-shaped) and `Empty` (bare zero-payload variant) are both
/// asserted separately.
#[tokio::test]
async fn python_shared_variant_defaults_are_really_admitted_end_to_end() {
    let (cfg, mut ws, baseline) = writer().await;
    let source = "@union\nclass Kind:\n    Leaf: tuple[int]\n    Empty: tuple[()]\nleaf: Kind = Leaf(7)\nempty: Kind = Empty\n";
    let created = attempt(&cfg, &mut ws, &baseline, "NtSharedVariantReal", source).await;
    assert!(
        !created.execution.has_errors,
        "variant defaults must be admitted: {}",
        created.content
    );

    let module = &ws.program.modules[0];
    let leaf = module
        .shared_vars
        .iter()
        .find(|v| v.name == "leaf")
        .expect("leaf shared var");
    assert_eq!(leaf.ty, crate::ast::Type::TaggedUnion("Kind".into()));
    assert!(
        matches!(&leaf.default, crate::ast::Expr::Call(call) if call.callee == "Leaf"
            && matches!(call.args.as_slice(), [crate::ast::Expr::Literal(crate::ast::Literal::Int(7))])),
        "leaf default must persist as Leaf(7): {:?}",
        leaf.default
    );
    let empty = module
        .shared_vars
        .iter()
        .find(|v| v.name == "empty")
        .expect("empty shared var");
    assert_eq!(empty.ty, crate::ast::Type::TaggedUnion("Kind".into()));
    assert!(
        matches!(&empty.default, crate::ast::Expr::Identifier(name) if name == "Empty"),
        "zero-payload default must persist as the bare Empty ident: {:?}",
        empty.default
    );
}

/// Residual refusals: a call-shaped default under a STRUCT annotation, under a
/// non-union/non-Named annotation, plus a foreign/unknown constructor and a
/// constructor belonging to a DIFFERENT union.
#[tokio::test]
async fn python_shared_variant_gate_only_admits_the_annotations_own_union() {
    let (cfg, mut ws, baseline) = writer().await;
    let cases = [
        (
            // Struct annotation: a call-shaped default is not a variant position.
            "@dataclass\nclass Point:\n    x: int\npoint: Point = Point(x=1)\n",
            true,
        ),
        (
            // List annotation: no call default.
            "values: list[int] = list(1, 2)\n",
            false,
        ),
        (
            // Primitive annotation under a variant spelling.
            "count: int = Leaf(1)\n",
            false,
        ),
        (
            // A constructor of ANOTHER union in the same module.
            "@union\nclass A:\n    First: tuple[()]\n    Second: tuple[()]\n@union\nclass B:\n    Third: tuple[int]\n    Fourth: tuple[()]\nvalue: A = Third(1)\n",
            false,
        ),
        (
            // Unknown constructor entirely.
            "@union\nclass Kind:\n    Leaf: tuple[int]\n    Empty: tuple[()]\nkind: Kind = Missing(1)\n",
            false,
        ),
    ];
    for (source, admitted) in cases {
        let before = serde_json::to_value(&ws.program).unwrap();
        let result = attempt(&cfg, &mut ws, &baseline, "NtSharedVariantScope", source).await;
        if admitted {
            // A struct default is admitted; this entry documents that the variant
            // path is not what admits it.
            assert!(!result.execution.has_errors, "{}", result.content);
        } else {
            assert!(
                result.execution.has_errors,
                "must refuse: {}",
                result.content
            );
            assert_eq!(serde_json::to_value(&ws.program).unwrap(), before);
        }
    }
}
