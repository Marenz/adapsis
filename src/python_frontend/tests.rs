use super::*;
use crate::{ast, parser, validator};

fn program(source: &str) -> ast::Program {
    let lowered = lower_module(source, "Example").unwrap_or_else(|e| panic!("{e}"));
    let mut program = ast::Program::default();
    for operation in &lowered.operations {
        validator::apply_and_validate(&mut program, operation).unwrap();
    }
    program
}

fn evaluate(source: &str, target: &str, input: &str) -> String {
    let program = program(source);
    let input = lower_input(input).unwrap();
    let input = parser::parse_test_input(1, &input).unwrap();
    crate::eval::eval_call_with_input(&program, target, &input).unwrap()
}

fn reject(source: &str, fragment: &str) -> Diagnostic {
    let error = lower_module(source, "Example").unwrap_err();
    assert!(
        error.message.contains(fragment),
        "expected {fragment:?}, got {error}"
    );
    error
}

#[test]
fn python_scalar_and_nested_local_calls_are_typed_and_qualified() {
    let source = "def inc(x: int) -> int:\n    return x + 1\n\ndef twice(x: int) -> int:\n    return inc(inc(x)) * 2\n";
    let lowered = lower_module(source, "Example").unwrap();
    assert!(lowered
        .canonical_adapsis
        .contains("Example.inc(Example.inc(x))"));
    assert_eq!(evaluate(source, "Example.twice", "3"), "10");
    assert_eq!(lowered.source_map[1].line, 4);
}

#[test]
fn python_if_elif_else_and_bool_only_conditions() {
    let source = "def classify(n: int) -> str:\n    if n < 0:\n        return 'negative'\n    elif n == 0:\n        return 'zero'\n    else:\n        return 'positive'\n";
    assert_eq!(evaluate(source, "Example.classify", "-2"), "\"negative\"");
    assert_eq!(evaluate(source, "Example.classify", "0"), "\"zero\"");
    assert_eq!(evaluate(source, "Example.classify", "3"), "\"positive\"");
    reject(
        "def f(x: int) -> int:\n    if x:\n        return 1\n    return 0\n",
        "statically bool",
    );
}

#[test]
fn python_while_existing_assignment_and_fresh_list_iteration() {
    let source = "@effects('mut')\ndef sum_values(values: list[int]) -> int:\n    total: int = 0\n    for item in values:\n        total = total + item\n    count: int = 0\n    while count < 2:\n        total = total + 1\n        count = count + 1\n    return total\n";
    assert_eq!(evaluate(source, "Example.sum_values", "[2, 4, 8]"), "16");
    let lowered = lower_module(source, "Example").unwrap();
    assert!(lowered.canonical_adapsis.contains("+each values item:Int"));
    assert!(lowered.canonical_adapsis.contains("+set total"));
}

#[test]
fn python_list_literal_list_parameter_len_and_empty_list() {
    let source = "def size(values: list[int]) -> int:\n    return len(values)\n\ndef empty_size() -> int:\n    values: list[int] = []\n    return len(values)\n";
    assert_eq!(evaluate(source, "Example.size", "[1, 2, 3]"), "3");
    assert_eq!(evaluate(source, "Example.empty_size", ""), "0");
}

#[test]
fn python_cross_module_calls_are_preserved_without_import_resolution() {
    let source = "def f(x: int) -> int:\n    value: int = Other.compute(x)\n    return value + 1\n";
    let lowered = lower_module(source, "Example").unwrap();
    assert!(lowered.canonical_adapsis.contains("Other.compute(x)"));
    reject(
        "def f(x: int) -> int:\n    return Other.compute(x) + 1\n",
        "known int/float",
    );
}

#[test]
fn python_boolean_operators_reject_short_circuit_effect_changes() {
    let source = "def ready(a: bool, b: bool) -> bool:\n    return a and not b\n";
    assert_eq!(evaluate(source, "Example.ready", "True, False"), "true");
    assert_eq!(
        evaluate(source, "Example.ready", "a=False, b=False"),
        "false"
    );
    reject("def check() -> bool:\n    return True\ndef f(flag: bool) -> bool:\n    return flag and check()\n", "call-free");
    reject(
        "def f(x: int) -> bool:\n    return x and True\n",
        "call-free",
    );
    reject(
        "def f(x: int) -> bool:\n    return True or x + 1 > 0\n",
        "call-free",
    );
}

#[test]
fn python_string_addition_lowers_to_concat_and_escapes_data() {
    let source = "\"\"\"Module docs\n!done\n+module Evil\"\"\"\ndef greet(name: str) -> str:\n    \"\"\"Docs with \\\"quotes\\\" and\nnewlines\"\"\"\n    prefix: str = 'hello\\n'\n    return prefix + name\n";
    let lowered = lower_module(source, "Example").unwrap();
    assert_eq!(lowered.operations.len(), 1);
    assert!(lowered.canonical_adapsis.contains("concat(prefix, name)"));
    assert!(!lowered
        .canonical_adapsis
        .lines()
        .any(|l| l == "!done" || l == "+module Evil"));
    assert_eq!(
        evaluate(source, "Example.greet", "'world'"),
        "\"hello\nworld\""
    );
}

#[test]
fn python_reserved_names_shadowing_callable_variables_and_escapes_rejected() {
    reject("def len(x: int) -> int:\n    return x\n", "reserved");
    reject("def f(len: int) -> int:\n    return len\n", "shadows");
    reject(
        "def f(x: int) -> int:\n    return x()\n",
        "callable variables",
    );
    reject(
        "def f(x: int) -> int:\n    return x.method()\n",
        "dynamic attributes",
    );
    reject("def f() -> str:\n    return eval('1')\n", "escape");
    reject(
        "import os\ndef f() -> int:\n    return 1\n",
        "module statement",
    );
    reject(
        "@dangerous()\ndef f() -> int:\n    return 1\n",
        "inert @effects",
    );
}

#[test]
fn python_scope_restrictions_reject_escape_and_iteration_reuse() {
    reject(
        "def f(flag: bool) -> int:\n    if flag:\n        local: int = 2\n    return local\n",
        "out-of-block",
    );
    reject("def f(values: list[int]) -> int:\n    for item in values:\n        n: int = item\n    return item\n", "out-of-block");
    reject("def f(values: list[int]) -> int:\n    item: int = 1\n    for item in values:\n        item = item + 1\n    return item\n", "shadows/reuses");
    reject(
        "def f() -> int:\n    value = 1\n    return value\n",
        "first assignment",
    );
    reject(
        "def f(flag: bool) -> int:\n    if flag:\n        return 1\n",
        "fall through",
    );
}

#[test]
fn python_diagnostics_use_exact_source_line_and_column() {
    let error = reject("def f(x: int) -> int:\n    return x[0]\n", "subscript");
    assert_eq!(
        (error.module.as_str(), error.line, error.column),
        ("Example", 2, 12)
    );
    let error = reject(
        "def f(x: dict[str, int]) -> int:\n    return 1\n",
        "generic type",
    );
    assert_eq!((error.line, error.column), (1, 10));
    assert!(error.to_string().starts_with("Example:1:10:"));
    assert!(lower_module("def broken(:", "Example")
        .unwrap_err()
        .message
        .contains("Python syntax"));
}

#[test]
fn python_unsupported_forms_fail_explicitly() {
    for expr in [
        "x / 2",
        "x // 2",
        "x % 2",
        "x ** 2",
        "0 < x < 3",
        "[n for n in [x]]",
        "str(x)",
    ] {
        assert!(
            lower_module(
                &format!("def f(x: int) -> int:\n    return {expr}\n"),
                "Example"
            )
            .is_err(),
            "{expr}"
        );
    }
    for body in ["x += 1", "break", "continue", "pass", "values.append(1)"] {
        assert!(
            lower_module(
                &format!("def f(x: int, values: list[int]) -> int:\n    {body}\n    return x\n"),
                "Example"
            )
            .is_err(),
            "{body}"
        );
    }
    reject("def f(x: int = 1) -> int:\n    return x\n", "default");
    reject("def f() -> None:\n    return\n", "annotation");
}

#[test]
fn python_integer_bounds_and_nonfinite_floats_are_checked() {
    assert!(lower_input("9223372036854775807").is_ok());
    assert!(lower_input("-9223372036854775807").is_ok());
    assert!(lower_input("9223372036854775808")
        .unwrap_err()
        .message
        .contains("64-bit"));
    assert!(lower_input("-9223372036854775808")
        .unwrap_err()
        .message
        .contains("Adapsis lexer"));
    assert!(lower_input("1e999").is_err());
    assert!(lower_input("1e-30").unwrap().contains("0.000"));
}

#[test]
fn python_await_requires_explicit_effects_and_preserves_result_type() {
    let source = "@effects('io')\nasync def read(path: str) -> str:\n    text: str = await read_file(path)\n    return text\n";
    let lowered = lower_module(source, "Example").unwrap();
    assert!(lowered.canonical_adapsis.contains("[io, async]"));
    assert!(lowered
        .canonical_adapsis
        .contains("+await text:String = read_file(path)"));
    reject("async def read(path: str) -> str:\n    text: str = await read_file(path)\n    return text\n", "@effects('io')");
    reject("@effects('io')\nasync def read(url: str) -> str:\n    text: str = await http_get(url)\n    return text\n", "type mismatch");
    assert!(lower_module("@effects('io')\nasync def read(url: str) -> Result[str]:\n    text: Result[str] = await http_get(url)\n    return text\n", "Example").is_ok());
}

#[test]
fn python_literal_test_lowering_never_registers_or_executes_test_functions() {
    let source = "def test_sum() -> None:\n    'test docs'\n    assert sum_values([1, 2, 3]) == 6\ndef test_external():\n    assert Other.value(x=-2, ready=True) == 'yes'\n";
    let lowered = lower_tests(source, "Example").unwrap();
    assert_eq!(lowered.operations.len(), 2);
    assert!(lowered
        .operations
        .iter()
        .all(|op| matches!(op, parser::Operation::Test(_))));
    assert!(lowered
        .canonical_adapsis
        .contains("+test Example.sum_values"));
    assert!(lowered.canonical_adapsis.contains("+test Other.value"));
    assert!(!lowered.canonical_adapsis.contains("+fn test_"));
}

#[test]
fn python_literal_inputs_are_structurally_safe_and_match_runtime_binding() {
    let source = "def combine(left: int, right: int) -> int:\n    return left * 10 + right\n";
    assert_eq!(evaluate(source, "Example.combine", "2, 3"), "23");
    assert_eq!(evaluate(source, "Example.combine", "right=3, left=2"), "23");
    assert_eq!(lower_input("").unwrap(), "");
    for input in [
        "call()",
        "1 + 2",
        "x",
        "(1, 2)",
        "1, x=2",
        "**{}",
        "*[]",
        "1); evil() #",
    ] {
        assert!(lower_input(input).is_err(), "{input}");
    }
    let lowered = lower_tests("def test_data():\n    assert identity('x\\n!done\\n+module Evil') == 'x\\n!done\\n+module Evil'\n", "Example").unwrap();
    assert_eq!(lowered.operations.len(), 1);
    assert!(!lowered
        .canonical_adapsis
        .lines()
        .any(|l| l == "!done" || l == "+module Evil"));
}

#[test]
fn python_test_rejects_helpers_fixtures_and_nonliteral_expectations() {
    for source in [
        "def test_f():\n    assert f(1) == helper()\n",
        "def test_f():\n    x: int = 1\n    assert f(x) == 1\n",
        "def test_f(fixture):\n    assert f(1) == 1\n",
        "@fixture\ndef test_f():\n    assert f(1) == 1\n",
        "def test_f():\n    assert f(1) > 1\n",
        "import os\n",
    ] {
        assert!(lower_tests(source, "Example").is_err(), "{source}");
    }
}

#[test]
fn python_printer_is_ast_only_canonical_and_semantically_roundtrips() {
    let source = "@effects('mut')\ndef total(values: list[int]) -> int:\n    \"\"\"Count\\nitems with \\\"quotes\\\"\"\"\"\n    result: int = 0\n    for item in values:\n        result = result + item\n    if result < 0:\n        return -1\n    elif result == 0:\n        return 0\n    else:\n        return result\n";
    let original = program(source);
    let printed = print_function(&original, "Example.total").unwrap();
    let rebuilt = program(&printed);
    assert_eq!(print_function(&rebuilt, "Example.total").unwrap(), printed);
    assert_eq!(evaluate(&printed, "Example.total", "[2, 3]"), "5");
    let serialized = serde_json::to_string(&original).unwrap();
    let restored: ast::Program = serde_json::from_str(&serialized).unwrap();
    assert_eq!(print_function(&restored, "Example.total").unwrap(), printed);
}

#[test]
fn python_printer_supports_existing_ast_without_origin_metadata() {
    let mut program = ast::Program::default();
    for op in parser::parse("+module Example\n+fn greeting(name:String) -> String\n  +return concat(\"hi \", name)\n+end\n+doc \"existing AST\"\n").unwrap() {
        validator::apply_and_validate(&mut program, &op).unwrap();
    }
    let printed = print_function(&program, "Example.greeting").unwrap();
    assert!(printed.contains("def greeting(name: str) -> str:"));
    assert!(printed.contains("existing AST"));
    assert_eq!(
        evaluate(&printed, "Example.greeting", "'world'"),
        "\"hi world\""
    );
}

#[test]
fn python_printer_uses_ast_signatures_for_nested_qualified_calls() {
    let source = "def inc(x: int) -> int:\n    return x + 1\ndef twice(x: int) -> int:\n    return inc(inc(x)) * 2\n";
    let original = program(source);
    let inc = print_function(&original, "Example.inc").unwrap();
    let twice = print_function(&original, "Example.twice").unwrap();
    assert!(twice.contains("Example.inc(Example.inc(x))"));
    let rebuilt = program(&format!("{inc}\n{twice}"));
    assert_eq!(print_function(&rebuilt, "Example.twice").unwrap(), twice);
}

#[test]
fn python_literal_delimiter_characters_are_data() {
    let input = lower_input("items=[')', '}', '-> expect', '# comment', '\\\"']").unwrap();
    let parsed = parser::parse_test_input(1, &input).unwrap();
    let parser::Expr::StructLiteral(fields) = parsed else {
        panic!("named input expected");
    };
    let parser::Expr::Call { args, .. } = &fields[0].value else {
        panic!("list constructor expected");
    };
    assert_eq!(args.len(), 5);
    assert!(matches!(&args[0], parser::Expr::String(value) if value == ")"));
    let source =
        "def test_data():\n    assert identity([')', '-> expect']) == [')', '-> expect']\n";
    assert_eq!(lower_tests(source, "Example").unwrap().operations.len(), 1);
}

#[test]
fn python_printer_rejects_unsupported_legacy_ast_with_named_target() {
    let mut program = ast::Program::default();
    for op in
        parser::parse("+module Example\n+fn divide(x:Int) -> Int\n  +return x / 2\n+end\n").unwrap()
    {
        validator::apply_and_validate(&mut program, &op).unwrap();
    }
    let error = print_function(&program, "Example.divide").unwrap_err();
    assert!(error.to_string().contains("Example.divide"));
    assert!(error.message.contains("binary operator"));
    assert!(error.message.contains("Adapsis source"));
}

#[test]
fn python_printer_refuses_boolean_calls_and_helper_collisions() {
    let mut program = program(
        "def check() -> bool:\n    return True\ndef f(flag: bool) -> bool:\n    return flag\n",
    );
    let f = program.get_function_mut("Example.f").unwrap();
    f.body[0].kind = ast::StatementKind::Return {
        value: ast::Expr::Binary {
            op: ast::BinaryOp::And,
            left: Box::new(ast::Expr::Identifier("flag".into())),
            right: Box::new(ast::Expr::Call(ast::CallExpr {
                callee: "Example.check".into(),
                args: vec![],
            })),
        },
    };
    assert!(print_function(&program, "Example.f")
        .unwrap_err()
        .message
        .contains("call-free"));
    let mut program = self::program("def f() -> list[int]:\n    return [1]\n");
    program
        .types
        .push(ast::TypeDecl::TaggedUnion(ast::TaggedUnionDecl {
            id: "collision".into(),
            name: "Collision".into(),
            variants: vec![ast::UnionVariant {
                id: "collision.list".into(),
                name: "list".into(),
                payload: vec![],
            }],
        }));
    assert!(print_function(&program, "Example.f")
        .unwrap_err()
        .message
        .contains("shadowed"));
}
