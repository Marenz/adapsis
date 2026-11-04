//! Construct malformed IR directly: no parser or Python lowerer can sanitize it
//! before the guard sees it. The guard itself performs no execution/lookup.
use super::*;

fn function(body: Vec<p::Operation>) -> p::Operation {
    p::Operation::Function(p::FunctionDecl {
        name: "value".into(),
        params: vec![],
        return_type: p::TypeExpr::Named("Int".into()),
        effects: vec![],
        body,
        doc: None,
    })
}

fn returns(expr: p::Expr) -> p::Operation {
    p::Operation::Return(p::ReturnDecl { expr })
}

fn module(body: Vec<p::Operation>) -> p::Operation {
    p::Operation::Module(p::ModuleDecl {
        name: "Example".into(),
        body,
        doc: None,
    })
}

fn named_call(name: &str, args: Vec<p::Expr>) -> p::Expr {
    p::Expr::Call {
        callee: Box::new(p::Expr::Ident(name.into())),
        args,
    }
}

fn reject(op: p::Operation, reason: &str) {
    let error = validate(&[op]).unwrap_err();
    assert!(error.contains(reason), "{error}");
}

#[test]
fn python_ir_cast_rejected_even_though_runtime_conversion_erases_it() {
    let cast = p::Expr::Cast {
        expr: Box::new(p::Expr::Int(3)),
        ty: p::TypeExpr::Named("String".into()),
    };
    let statement = returns(cast);
    assert!(crate::validator::convert_statement_op(&statement).is_ok());
    reject(
        module(vec![function(vec![statement])]),
        "would erase the cast",
    );
}

#[test]
fn python_ir_placement_and_dynamic_calls_fail_without_reparse() {
    reject(returns(p::Expr::Int(1)), "placement");
    reject(module(vec![module(vec![])]), "placement");
    reject(function(vec![function(vec![])]), "function body");
    reject(module(vec![p::Operation::Done]), "placement");
    let dynamic = p::Expr::Call {
        callee: Box::new(named_call("factory", vec![])),
        args: vec![],
    };
    reject(function(vec![returns(dynamic)]), "dynamic call target");
    reject(
        function(vec![returns(named_call("safe\n!done", vec![]))]),
        "identifier",
    );
    assert!(validate(&[module(vec![]), function(vec![])]).is_err());
}

#[test]
fn python_ir_effects_generic_arity_and_numeric_shapes_are_explicit() {
    let p::Operation::Function(mut f) = function(vec![returns(p::Expr::Int(1))]) else {
        panic!()
    };
    for effects in [vec!["network".into()], vec!["io".into(), "io".into()]] {
        f.effects = effects;
        reject(p::Operation::Function(f.clone()), "effect");
    }
    f.effects.clear();
    for (name, args) in [
        ("Map", vec![p::TypeExpr::Named("Int".into())]),
        ("Future", vec![]),
    ] {
        f.return_type = p::TypeExpr::Generic {
            name: name.into(),
            args,
        };
        reject(p::Operation::Function(f.clone()), "generic");
    }
    for expr in [
        p::Expr::Int(-1),
        p::Expr::Float(-0.0),
        p::Expr::Float(f64::NAN),
    ] {
        reject(function(vec![returns(expr)]), "numeric constant");
    }
    let signed = p::Expr::Unary {
        op: p::UnaryOp::Neg,
        expr: Box::new(p::Expr::Int(1)),
    };
    assert!(validate(&[function(vec![returns(signed)])]).is_ok());
}

#[test]
fn python_ir_lifecycle_metadata_cannot_be_silently_discarded() {
    let p::Operation::Function(mut f) = function(vec![returns(p::Expr::String("ok".into()))])
    else {
        panic!()
    };
    f.name = "startup".into();
    f.return_type = p::TypeExpr::Named("String".into());
    f.effects = vec!["io".into(), "async".into()];
    assert!(validate(&[module(vec![p::Operation::Startup(f.clone())])]).is_ok());
    f.doc = Some("would be lost".into());
    reject(module(vec![p::Operation::Startup(f.clone())]), "metadata");
    f.doc = None;
    reject(module(vec![p::Operation::Shutdown(f)]), "metadata");
}

#[test]
fn python_ir_nested_pattern_placeholders_and_capture_identity_are_checked() {
    let mut arm = p::MatchArmDecl {
        variant: "Ok".into(),
        bindings: vec!["__pat_0".into()],
        patterns: Some(vec![p::MatchPatternDecl::Variant {
            variant: "Leaf".into(),
            sub_patterns: vec![p::MatchPatternDecl::Binding("x".into())],
        }]),
        body: vec![returns(p::Expr::Int(1))],
    };
    assert!(values::arm(&arm).is_ok());
    let mut empty_variant = arm.clone();
    empty_variant.patterns = Some(vec![p::MatchPatternDecl::Variant {
        variant: "Leaf".into(),
        sub_patterns: vec![],
    }]);
    assert!(
        values::arm(&empty_variant)
            .unwrap_err()
            .contains("pattern shape")
    );
    arm.bindings[0] = "unexpected".into();
    assert!(values::arm(&arm).unwrap_err().contains("placeholder"));
    arm.variant = "_".into();
    assert!(values::arm(&arm).unwrap_err().contains("wildcard"));
    arm.variant = "Ok".into();
    arm.patterns = None;
    arm.bindings = vec!["x".into(), "x".into()];
    assert!(values::arm(&arm).unwrap_err().contains("duplicate"));
}

#[test]
fn python_ir_source_event_fragments_cannot_inject_operations() {
    for kind in [
        "timer(1)\n!done",
        "Example.event\n+module Hidden",
        "timer(1))",
        "channel;!done",
        "timer(1;!done)",
        "timer(1 2)",
        "timer(1) ignored)",
    ] {
        reject(
            function(vec![p::Operation::SourceAdd(p::SourceAddDecl {
                kind_text: kind.into(),
                alias: "pulse".into(),
                handler: "tick".into(),
            })]),
            "operation",
        );
    }
    for payload_type in [
        "String)\n!done",
        "Map<Int>",
        "Result<>",
        "Int garbage",
        "String\r",
    ] {
        assert!(
            values::payload_type(payload_type).is_err(),
            "{payload_type:?}"
        );
    }
    assert!(values::payload_type("Map<String, List<Int>>").is_ok());
    reject(
        module(vec![p::Operation::ModuleSource(p::ModuleSourceDecl {
            name: "pulse".into(),
            source_type: "timer".into(),
            handler: "tick".into(),
            config: vec![("interval".into(), "1\n+module Hidden".into())],
        })]),
        "single-token",
    );
    let string = "literal\n!done\n+module Inert\t\"quoted\"";
    assert!(validate(&[function(vec![returns(p::Expr::String(string.into()))])]).is_ok());
}

fn assert_legacy_oracle(operations: &[p::Operation]) {
    let emitted = super::super::emit::operations(operations).unwrap();
    let reparsed = p::parse(&emitted).unwrap();
    assert_eq!(format!("{operations:?}"), format!("{reparsed:?}"));
    let mut direct = crate::ast::Program::default();
    let mut legacy = crate::ast::Program::default();
    for op in operations {
        crate::validator::apply_and_validate(&mut direct, op).unwrap();
    }
    for op in &reparsed {
        crate::validator::apply_and_validate(&mut legacy, op).unwrap();
    }
    assert_eq!(
        serde_json::to_value(direct).unwrap(),
        serde_json::to_value(legacy).unwrap()
    );
}

#[test]
fn python_ir_timer_literal_semicolon_add_and_replace_preserve_legacy_ast() {
    let kind = "timer(len(\"a;b\") + 1)";
    for operation in [
        p::Operation::SourceAdd(p::SourceAddDecl {
            kind_text: kind.into(),
            alias: "pulse".into(),
            handler: "tick".into(),
        }),
        p::Operation::SourceReplace(p::SourceReplaceDecl {
            kind_text: kind.into(),
            alias: "pulse".into(),
            handler: "tick".into(),
        }),
    ] {
        let p::Operation::Function(mut tick) = function(vec![returns(p::Expr::Int(1))]) else {
            panic!()
        };
        tick.name = "tick".into();
        let p::Operation::Function(mut setup) = function(vec![operation, returns(p::Expr::Int(1))])
        else {
            panic!()
        };
        setup.name = "setup".into();
        setup.effects = vec!["io".into(), "async".into()];
        let operations = [module(vec![
            p::Operation::Function(tick),
            p::Operation::Function(setup),
        ])];
        assert_legacy_oracle(&operations); // The old path accepts this exact IR.
        validate(&operations).unwrap();
    }
}

#[test]
fn python_ir_timer_literal_semicolon_actual_python_setup_preserves_legacy_ast() {
    let source = "def tick() -> str:\n    return 'ok'\n\n@effects('io')\nasync def setup() -> str:\n    source_add('timer(len(\"a;b\") + 1)', 'pulse', 'tick')\n    source_replace('pulse', 'timer(len(\"a;b\") + 1)', 'tick')\n    return 'started'\n";
    let lowered = super::super::lower_module(source, "IrTimerLiteral").unwrap();
    assert_legacy_oracle(&lowered.operations);
    assert!(
        lowered
            .legacy_export().unwrap()
            .contains("timer(len(\"a;b\") + 1)")
    );
}

#[test]
fn python_ir_test_data_and_matcher_shapes_do_not_admit_hidden_calls() {
    let mut test = p::TestMutation {
        function_name: "Example.value".into(),
        cases: vec![p::TestCase {
            input: p::Expr::Int(1),
            expected: p::Expr::String("prefix".into()),
            matcher: Some(p::TestMatcher::StartsWith("prefix".into())),
            after_checks: vec![],
        }],
    };
    assert!(validate(&[p::Operation::Test(test.clone())]).is_ok());
    test.cases[0].expected = p::Expr::String("different".into());
    reject(p::Operation::Test(test.clone()), "matcher/expected");
    test.cases[0].matcher = None;
    test.cases[0].input = named_call("Hidden.peek", vec![]);
    reject(p::Operation::Test(test.clone()), "test data");
    test.cases[0].input = named_call("list", vec![p::Expr::Int(1), p::Expr::Int(2)]);
    assert!(validate(&[p::Operation::Test(test)]).is_ok());
}

#[test]
fn python_ir_supported_frontend_operations_keep_the_legacy_oracle() {
    for (source, module) in [
        (include_str!("../../../examples/python/basic.py"), "IrBasic"),
        (include_str!("../../../examples/python/flow.py"), "IrFlow"),
        (
            "@source('beat', type='timer', interval='1000')\ndef tick() -> str:\n    return 'ok'\n\n@effects('io')\nasync def startup() -> None:\n    source_add('timer(1000)', 'pulse', 'tick')\n    event_register('message', 'String')\n    event_emit('message', 'hi')\n    return 'started'\n",
            "IrLifecycle",
        ),
    ] {
        let lowered = super::super::lower_module(source, module).unwrap();
        validate(&lowered.operations).unwrap();
        let parsed = p::parse(&lowered.legacy_export().unwrap()).unwrap();
        assert_eq!(format!("{:?}", lowered.operations), format!("{parsed:?}"));
        let mut direct = crate::ast::Program::default();
        let mut legacy = crate::ast::Program::default();
        for op in &lowered.operations {
            crate::validator::apply_and_validate(&mut direct, op).unwrap();
        }
        for op in &parsed {
            crate::validator::apply_and_validate(&mut legacy, op).unwrap();
        }
        assert_eq!(
            serde_json::to_value(direct).unwrap(),
            serde_json::to_value(legacy).unwrap()
        );
    }
}

#[test]
fn python_ir_new_guard_errors_are_honestly_submission_scoped() {
    let source = "# heading\n\n@route('POST\\n!done', '/x')\ndef handle(body: str) -> str:\n    return body\n";
    let error = super::super::lower_module(source, "IrLocation").unwrap_err();
    assert_eq!((error.line, error.column), (1, 1));
    assert!(
        error
            .message
            .starts_with("Python operation IR: operation 0:")
    );
    assert!(error.message.contains("route method"));
}
