//! Bounded regressions for the parity admission audit. These tests distinguish
//! branch-local reuse from escape/shadowing and known len errors from Unknown.
use super::*;
use crate::{ast, parser, validator};

fn program(source: &str) -> ast::Program {
    let mut program = ast::Program::default();
    for op in lower_module(source, "ScopeProbe").unwrap().operations {
        validator::apply_and_validate(&mut program, &op).unwrap();
    }
    program
}

fn reject(source: &str, message: &str) {
    let error = lower_module(source, "ScopeProbe").unwrap_err();
    assert!(error.message.contains(message), "{error}");
}

#[test]
fn python_len_rejects_known_invalid_arguments_and_arity() {
    for function in ["len", "length"] {
        for (annotation, value) in [
            ("int", "1"),
            ("float", "1.5"),
            ("bool", "True"),
            ("Result[int]", "Ok(1)"),
            ("Option[int]", "Some(1)"),
        ] {
            reject(
                &format!(
                    "def f() -> int:\n    x: {annotation} = {value}\n    return {function}(x)\n"
                ),
                "unsupported or unresolved",
            );
        }
        for arguments in ["", "[], []"] {
            reject(
                &format!("def f() -> int:\n    return {function}({arguments})\n"),
                "one argument",
            );
        }
        reject(
            &format!(
                "@dataclass\nclass Point:\n    x: int\ndef f(p: Point) -> int:\n    return {function}(p)\n"
            ),
            "unsupported or unresolved",
        );
    }
}

#[test]
fn python_len_preserves_known_supported_types_and_rejects_unresolved_results() {
    for annotation in ["list[int]", "str", "Map[str, int]", "Set[int]"] {
        let source = format!("def f(x: {annotation}) -> int:\n    return len(x)\n");
        assert!(lower_module(&source, "ScopeProbe").is_ok(), "{source}");
    }
    let source = "def f() -> int:\n    return len(Other.value())\n";
    reject(source, "statically known");
}

#[test]
fn python_sibling_blocks_reuse_names_without_type_or_value_leakage() {
    let source = "def f(flag: bool) -> int:\n    if flag:\n        local: int = 7\n        return local\n    else:\n        local: str = 'abc'\n        return len(local)\n";
    let program = program(source);
    for (input, expected) in [("true", "7"), ("false", "3")] {
        let input = parser::parse_test_input(1, input).unwrap();
        assert_eq!(
            crate::eval::eval_call_with_input(&program, "ScopeProbe.f", &input).unwrap(),
            expected
        );
    }
    assert!(print_function(&program, "ScopeProbe.f").is_ok());
}

#[test]
fn python_match_arm_locals_and_captures_are_scoped() {
    let source = "def f(value: Result[int]) -> int:\n    match value:\n        case Ok(item):\n            local: int = item\n            return local\n        case Err(item):\n            local: str = item\n            return len(local)\ndef success() -> int:\n    return f(Ok(7))\ndef failure() -> int:\n    return f(Err('abc'))\n";
    let program = program(source);
    for (target, expected) in [("ScopeProbe.success", "7"), ("ScopeProbe.failure", "3")] {
        let input = parser::parse_test_input(1, "").unwrap();
        assert_eq!(
            crate::eval::eval_call_with_input(&program, target, &input).unwrap(),
            expected
        );
    }
    assert!(print_function(&program, "ScopeProbe.f").is_ok());
    reject(
        "def f(value: Result[int]) -> int:\n    match value:\n        case Ok(item):\n            pass\n        case Err(item):\n            pass\n    return item\n",
        "out-of-block",
    );
}

#[test]
fn python_await_reuses_sibling_name_without_relaxing_effect_or_type_checks() {
    let source = "@effects('io')\nasync def f(flag: bool, path: str) -> str:\n    if flag:\n        text: str = await read_file(path)\n        return text\n    else:\n        text: str = await read_file(path)\n        return text\n";
    let lowered = lower_module(source, "ScopeProbe").unwrap();
    assert_eq!(
        lowered
            .canonical_adapsis
            .matches("+await text:String")
            .count(),
        2
    );
    reject(&source.replace("@effects('io')\n", ""), "IO await requires");
    reject(&source.replace("async def", "def"), "await requires async");
    reject(
        "@effects('io')\nasync def f(path: str) -> str:\n    text: int = 0\n    text: str = await read_file(path)\n    return text\n",
        "cannot change its type",
    );
}

#[test]
fn python_scoped_reuse_retains_escape_callable_and_iteration_guards() {
    reject(
        "def f(flag: bool) -> int:\n    if flag:\n        local: int = 2\n    return local\n",
        "out-of-block",
    );
    reject(
        "def f(values: list[int]) -> int:\n    item: int = 1\n    for item in values:\n        pass\n    return item\n",
        "shadows/reuses",
    );
    reject(
        "def f(flag: bool) -> int:\n    if flag:\n        len: int = 1\n    return 0\n",
        "shadows/reuses",
    );
}
