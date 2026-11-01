//! G2: `input` is a legal BINDING name (parameter, local, loop target, match capture,
//! value, field-access base) and stays forbidden in every other role. The frontend never
//! executes Python, so a bound `input` can only shadow the builtin; the hazard is a
//! callable `input(..)`, a function/type/member/shared/annotation/attribute named
//! `input`, and those remain refused.
use super::{lower_module, lower_module_with_program, lower_tests, print_function};
use crate::{ast, parser, validator};

fn program(ax: &str) -> ast::Program {
    let mut program = ast::Program::default();
    for op in parser::parse(ax).unwrap() {
        validator::apply_and_validate(&mut program, &op).unwrap();
    }
    program.rebuild_function_index();
    program
}

/// Lowered through BOTH entry points: the program-free one and the authority-filtered
/// native one (which runs the surface preflight first).
fn both(source: &str, ax: &str) -> [Result<String, String>; 2] {
    let free = lower_module(source, "InputProbe")
        .map(|l| l.canonical_adapsis)
        .map_err(|e| e.to_string());
    let native = lower_module_with_program(source, "InputProbe", &program(ax))
        .map(|l| l.canonical_adapsis)
        .map_err(|e| e.to_string());
    [free, native]
}

#[test]
fn input_is_accepted_as_parameter_local_loop_target_capture_and_value() {
    let sources = [
        // parameter + value use
        "def f(input: str) -> str:\n    return input\n",
        // local binding
        "def f(text: str) -> str:\n    input: str = concat(text, '!')\n    return input\n",
        // loop target
        "def f(items: list[int]) -> int:\n    total: int = 0\n    for input in items:\n        total = total + input\n    return total\n",
        // match capture
        "def f(r: Result[int]) -> int:\n    match r:\n        case Ok(input):\n            return input\n        case Err(e):\n            return 0\n",
        // field-access base
        "@dataclass\nclass P:\n    x: int\n\ndef f(input: P) -> int:\n    return input.x\n",
    ];
    for source in sources {
        for result in both(source, "") {
            let ax = result.unwrap_or_else(|e| panic!("{source}\n{e}"));
            assert!(ax.contains("input"), "{ax}");
        }
    }
}

#[test]
fn a_callable_input_is_still_refused_bound_and_unbound() {
    // Unbound: the callee rule applies.
    for result in both("def f(x: str) -> str:\n    return input(x)\n", "") {
        let message = result.unwrap_err();
        assert!(
            message.contains("callable") || message.contains("no authorized supported signature"),
            "{message}"
        );
    }
    // Bound: a parameter is a value, never a callee.
    for result in both("def f(input: str) -> str:\n    return input(input)\n", "") {
        let message = result.unwrap_err();
        assert!(
            message.contains("callable") || message.contains("no authorized supported signature"),
            "{message}"
        );
    }
}

#[test]
fn an_existing_ax_function_named_input_is_never_a_python_callee() {
    // Legacy `.ax` may declare `+fn input`; the Python surface still must not call it,
    // qualified or (inside its own module) bare. Only the callee rule can refuse this:
    // the signature exists, so "unknown callable" would not.
    let ax = "+module Other\n+fn input (x:Int)->Int\n  +return x\n+end\n";
    let qualified = lower_module_with_program(
        "def f(x: int) -> int:\n    return Other.input(x)\n",
        "InputProbe",
        &program(ax),
    );
    assert!(qualified.is_err(), "{qualified:?}");
    let bare = lower_module_with_program(
        "def g(x: int) -> int:\n    return input(x)\n",
        "Other",
        &program(ax),
    );
    assert!(bare.is_err(), "{bare:?}");
}

#[test]
fn input_stays_forbidden_as_function_type_member_shared_annotation_and_attribute() {
    let refused = [
        (
            "def input(x: int) -> int:\n    return x\n",
            "function name input",
        ),
        (
            "@dataclass\nclass input:\n    x: int\n",
            "reserved/unsafe type name",
        ),
        (
            "@dataclass\nclass P:\n    input: int\n",
            "unsupported member name",
        ),
        ("input: int = 0\n", "unsupported shared variable name"),
        (
            "def f(x: input) -> int:\n    return 0\n",
            "unsupported type name",
        ),
    ];
    for (source, needle) in refused {
        for result in both(source, "") {
            let message = result.unwrap_err();
            assert!(message.contains(needle), "{source}\n{message}");
        }
    }
    // Attribute NAME (`p.input`): refused by the preflight even when the struct is known.
    let message = lower_module_with_program(
        "def f(p: P) -> int:\n    return p.input\n",
        "InputProbe",
        &program("+type P = x:Int\n"),
    )
    .unwrap_err()
    .to_string();
    assert!(
        message.contains("unsupported field access identifier"),
        "{message}"
    );
}

#[test]
fn other_forbidden_names_remain_forbidden_as_bindings() {
    // The exemption is `input` only: every other Python escape name is unchanged.
    for name in [
        "eval", "exec", "open", "getattr", "vars", "compile", "import", "x__y",
    ] {
        for source in [
            format!("def f({name}: str) -> str:\n    return {name}\n"),
            format!("def f(t: str) -> str:\n    {name}: str = t\n    return {name}\n"),
        ] {
            for result in both(&source, "") {
                assert!(result.is_err(), "{name}: {source}");
            }
        }
    }
}

#[test]
fn input_keyword_in_tests_was_and_stays_accepted() {
    let lowered = lower_tests("def test_case_1():\n    assert f(input='x') == 1\n", "M").unwrap();
    assert!(
        lowered.canonical_adapsis.contains("{input: \"x\"}"),
        "{}",
        lowered.canonical_adapsis
    );
}

#[test]
fn an_ax_function_with_an_input_parameter_prints_and_roundtrips() {
    let p = program(
        "+module P\n+fn size (input:String)->Int\n  +let n:Int = len(input)\n  +return n\n+end\n+fn echo (text:String)->String\n  +let input:String = concat(text, \"!\")\n  +return input\n+end\n",
    );
    for (target, needle) in [
        ("P.size", "def size(input: str) -> int:"),
        ("P.echo", "input: str = concat(text, \"!\")"),
    ] {
        let text = print_function(&p, target).unwrap_or_else(|e| panic!("{target}: {e}"));
        assert!(text.contains(needle), "{text}");
    }
}
