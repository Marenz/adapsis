//! Printer unit tests: every accepted shape roundtrips Debug-identical through
//! `lower_tests`, and every refusal names its reason.
use super::{lower_tests, tests_print};
use crate::parser as p;

fn parse_block(source: &str) -> p::TestMutation {
    let operations = p::parse(&format!(
        "+module M\n+fn f ()->Int\n  +return 1\n+end\n{source}"
    ))
    .unwrap();
    let p::Operation::Module(module) = &operations[0] else {
        panic!()
    };
    module
        .body
        .iter()
        .find_map(|op| match op {
            p::Operation::Test(test) => Some(test.clone()),
            _ => None,
        })
        .expect("test block")
}

fn roundtrip(source: &str) -> String {
    let test = parse_block(source);
    let text = tests_print::print_block(&test).unwrap_or_else(|e| panic!("{source}: {e}"));
    let lowered = lower_tests(&text, "M").unwrap_or_else(|e| panic!("{text}\n{e}"));
    let cases: Vec<_> = lowered
        .operations
        .into_iter()
        .flat_map(|op| match op {
            p::Operation::Test(t) => {
                assert_eq!(t.function_name, format!("M.{}", test.function_name));
                t.cases
            }
            _ => panic!(),
        })
        .collect();
    assert_eq!(format!("{:?}", test.cases), format!("{cases:?}"), "{text}");
    let again = tests_print::print_block(&p::TestMutation {
        function_name: test.function_name.clone(),
        cases,
    })
    .unwrap();
    assert_eq!(text, again, "second print");
    text
}

fn refused(source: &str) -> String {
    tests_print::print_block(&parse_block(source)).unwrap_err()
}

#[test]
fn printer_roundtrips_every_literal_kind_and_argument_shape() {
    roundtrip("+test f\n  +with 15 -> expect \"fizz\"\n  +with -3 -> expect 0\n  +with true -> expect false\n");
    roundtrip("+test f\n  +with 1.5 -> expect -2.25\n  +with 0.0 -> expect 1.0\n  +with 100.125 -> expect 0.1\n");
    roundtrip("+test f\n  +with \"a \\\"q\\\" \\\\ \\n\\t ä😀\" -> expect \"\"\n");
    roundtrip("+test f\n  +with list(1, 2) -> expect list()\n  +with list(\"a\", \"b\") -> expect list(\"c\")\n");
    roundtrip(
        "+test f\n  +with x=1 -> expect 2\n  +with a=\"s\" b=3 c=true d=list(1) -> expect \"r\"\n",
    );
    roundtrip("+test f\n  +with {a: 1, b: \"x\"} -> expect 2\n");
    roundtrip("+test f\n  +with 1 2 3 -> expect 6\n");
    roundtrip("+test f\n  +with {_0: 1, _1: 2} -> expect 3\n");
}

#[test]
fn printer_roundtrips_matchers_err_forms_and_after_checks() {
    roundtrip("+test f\n  +with 1 -> expect contains(\"sub\")\n  +with 2 -> expect starts_with(\"pre\")\n");
    roundtrip("+test f\n  +with 1 -> expect Ok\n  +with 2 -> expect Err(err_empty_name)\n");
    roundtrip("+test f\n  +with 1 -> expect Err(\"boom message\")\n");
    let text = roundtrip("+test f\n  +with 1 -> expect 2\n    +after routes contains \"/chat\"\n");
    assert!(
        text.contains("assert_after(\"routes\", \"contains\", \"/chat\")"),
        "{text}"
    );
}

#[test]
fn printer_refuses_everything_the_test_data_policy_refuses_with_a_reason() {
    for (source, needle) in [
        ("+test f\n  +with Ok(\"x\") -> expect 1\n", "call to Ok"),
        ("+test f\n  +with Some(1) -> expect 1\n", "call to Some"),
        (
            "+test f\n  +with 1 -> expect concat(\"a\", \"b\")\n",
            "call to concat",
        ),
        (
            "+test f\n  +with {a: {b: 1}} -> expect 1\n",
            "record literal",
        ),
        ("+test f\n  +with 1 -> expect {a: 1}\n", "record literal"),
        ("+test f\n  +with x=y -> expect 1\n", "identifier y"),
        ("+test f\n  +with 1 -> expect None\n", "identifier None"),
    ] {
        let reason = refused(source);
        assert!(reason.starts_with("case 1: "), "{source}: {reason}");
        assert!(reason.contains(needle), "{source}: {reason}");
    }
    // A control character inside a string is refused by the quoting helper.
    let mut control = parse_block("+test f\n  +with \"x\" -> expect 1\n");
    control.cases[0].input = p::Expr::String("a\u{1}b".into());
    let reason = tests_print::print_block(&control).unwrap_err();
    assert!(
        reason.starts_with("case 1: ") && reason.contains("control character"),
        "{reason}"
    );
    // The second case is named when only it fails.
    let reason = refused("+test f\n  +with 1 -> expect 1\n  +with Some(1) -> expect 2\n");
    assert!(reason.starts_with("case 2: "), "{reason}");
}

#[test]
fn printer_refuses_unsafe_names_and_unrepresentable_numbers() {
    let mut test = parse_block("+test f\n  +with x=1 -> expect 1\n");
    test.function_name = "bad__name".into();
    assert!(tests_print::print_block(&test).is_err());
    test.function_name = "f".into();
    test.cases[0].input = p::Expr::StructLiteral(vec![
        p::FieldValue {
            name: "x".into(),
            value: p::Expr::Int(1),
        },
        p::FieldValue {
            name: "x".into(),
            value: p::Expr::Int(2),
        },
    ]);
    assert!(tests_print::print_block(&test)
        .unwrap_err()
        .contains("unique safe keyword"));
    test.cases[0].input = p::Expr::StructLiteral(vec![
        p::FieldValue {
            name: "_7".into(),
            value: p::Expr::Int(1),
        },
        p::FieldValue {
            name: "y".into(),
            value: p::Expr::Int(2),
        },
    ]);
    assert!(tests_print::print_block(&test)
        .unwrap_err()
        .contains("reserved for positional"));
    // A lone `_0` field cannot be produced by Python (one positional value is bare).
    test.cases[0].input = p::Expr::StructLiteral(vec![p::FieldValue {
        name: "_0".into(),
        value: p::Expr::Int(1),
    }]);
    assert!(tests_print::print_block(&test).is_err());
    test.cases[0].input = p::Expr::Int(i64::MIN);
    assert!(tests_print::print_block(&test)
        .unwrap_err()
        .contains("-9223372036854775808"));
    test.cases[0].input = p::Expr::Float(f64::NAN);
    assert!(tests_print::print_block(&test)
        .unwrap_err()
        .contains("non-finite"));
    test.cases[0].input = p::Expr::Int(1);
    test.cases[0].matcher = Some(p::TestMatcher::Contains("a".into()));
    test.cases[0].expected = p::Expr::String("different".into());
    assert!(tests_print::print_block(&test)
        .unwrap_err()
        .contains("matcher text differs"));
    test.cases.clear();
    assert!(tests_print::print_block(&test)
        .unwrap_err()
        .contains("no cases"));
}

#[test]
fn printed_text_is_stable_and_readable() {
    let text = roundtrip(
        "+test f\n  +with a=1 b=list(\"x\") -> expect \"ok\"\n  +with 7 -> expect Err(err_x)\n",
    );
    assert_eq!(
        text,
        "def test_case_1() -> None:\n    assert f(a=1, b=[\"x\"]) == \"ok\"\n\ndef test_case_2() -> None:\n    assert f(7) == Err(err_x)\n"
    );
}

#[test]
fn keyword_shorthand_negative_constants_are_refused_with_the_111_reason() {
    for source in [
        "+with a=-1 -> expect 1",
        "+with a=-1.5 -> expect 1",
        "+with a=-0.0 -> expect 1",
        "+with a=1 b=-2 -> expect 1",
    ] {
        let reason = refused(&format!("+test f\n  {source}\n"));
        assert!(
            reason.starts_with("case 1: ") && reason.contains("(#111)"),
            "{source}: {reason}"
        );
    }
    // A hand-built negative constant anywhere (list item, expectation, double sign) is
    // refused the same way: no Python spelling lowers back to it.
    let mut test = parse_block("+test f\n  +with 1 -> expect 1\n");
    test.cases[0].expected = p::Expr::Int(-1);
    assert!(tests_print::print_block(&test)
        .unwrap_err()
        .contains("(#111)"));
    test.cases[0].expected = p::Expr::Int(1);
    test.cases[0].input = p::Expr::Call {
        callee: Box::new(p::Expr::Ident("list".into())),
        args: vec![p::Expr::Float(-2.0)],
    };
    assert!(tests_print::print_block(&test)
        .unwrap_err()
        .contains("(#111)"));
    test.cases[0].input = p::Expr::Unary {
        op: p::UnaryOp::Neg,
        expr: Box::new(p::Expr::Int(-1)),
    };
    assert!(tests_print::print_block(&test)
        .unwrap_err()
        .contains("(#111)"));
}

#[test]
fn every_other_negative_spelling_still_roundtrips_identically() {
    for source in [
        "+with {a: -1} -> expect -2",
        "+with -3 -> expect -2.25",
        "+with {_0: -1, _1: 2} -> expect 0",
        "+with a=list(-1, 2) -> expect 0",
        "+with {a: -1.5, b: -2} -> expect 0",
    ] {
        roundtrip(&format!("+test f\n  {source}\n"));
    }
}

#[test]
fn module_less_test_lowering_keeps_the_runtime_target_string() {
    let text = "def test_case_1() -> None:\n    assert f(1) == 1\n\ndef test_case_2() -> None:\n    assert M.f(2) == 2\n\ndef test_case_3() -> None:\n    assert contains(f(3), \"x\")\n";
    let lowered = super::lower_tests_toplevel(text).unwrap();
    let names: Vec<_> = lowered
        .operations
        .iter()
        .map(|op| match op {
            p::Operation::Test(t) => (t.function_name.clone(), t.cases.len()),
            _ => panic!(),
        })
        .collect();
    assert_eq!(names, [("f".into(), 2), ("M.f".into(), 1)]);
    assert!(
        lowered.canonical_adapsis.contains("+test f")
            && !lowered.canonical_adapsis.contains("+module")
    );
    // Module lowering is unchanged: bare targets are qualified, dotted ones kept.
    let lowered = lower_tests(text, "M").unwrap();
    let names: Vec<_> = lowered
        .operations
        .iter()
        .map(|op| match op {
            p::Operation::Test(t) => (t.function_name.clone(), t.cases.len()),
            _ => panic!(),
        })
        .collect();
    assert_eq!(names, [("M.f".into(), 3)]);
    // The module-name check still guards the public entry point.
    assert!(lower_tests(text, "").is_err());
}
