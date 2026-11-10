//! Ad-hoc statement blocks (#129): the same statement rules as a function body, lowered
//! without a synthetic `def`, so positions are those of the submitted text.
use super::*;
use crate::{ast, parser as p, validator};

const SCOPE: &str = "BlockScope";

fn library() -> ast::Program {
    let source = "counter: int = 0\n\ndef double(x: int) -> int:\n    return x * 2\n\n@effects('io')\nasync def fetch(url: str) -> Result[str]:\n    response: Result[str] = await http_get(url)\n    return response\n";
    let mut program = ast::Program::default();
    for op in lower_module(source, "BlockLib").unwrap().operations {
        validator::apply_and_validate(&mut program, &op).unwrap();
    }
    program
}

fn block_in(source: &str, scope: &str, program: &ast::Program) -> Result<p::FunctionDecl, Diagnostic> {
    preflight_block(source, scope)?;
    let lowered = lower_block_with_program(source, scope, "run_block", program)?;
    let [p::Operation::Module(module)] = lowered.operations.as_slice() else {
        panic!("a block lowers to exactly one module: {:?}", lowered.operations);
    };
    assert_eq!(module.name, scope);
    let [p::Operation::Function(function)] = module.body.as_slice() else {
        panic!("a block lowers to exactly one function: {:?}", module.body);
    };
    Ok(function.clone())
}

fn block(source: &str) -> p::FunctionDecl {
    block_in(source, SCOPE, &ast::Program::default()).unwrap_or_else(|error| panic!("{error}"))
}

fn refused(source: &str) -> Diagnostic {
    block_in(source, SCOPE, &library()).expect_err("block must be refused")
}

fn returns(function: &p::FunctionDecl) -> String {
    format!("{:?}", function.return_type)
}

#[test]
fn module_level_return_and_await_parse_and_lower_without_wrapping() {
    let function = block("text: str = await file_read(\"/tmp/x.txt\")\nn: int = len(text)\nreturn concat(\"bytes: \", to_string(n))\n");
    assert_eq!(function.name, "run_block");
    assert!(function.params.is_empty());
    assert_eq!(function.effects, ["io", "fail", "mut", "async"]);
    assert_eq!(function.body.len(), 3);
    assert!(matches!(&function.body[0], p::Operation::Await(decl) if decl.name == "text"));
    assert!(matches!(&function.body[1], p::Operation::Let(decl) if decl.name == "n"));
    assert!(matches!(&function.body[2], p::Operation::Return(_)));
    assert!(returns(&function).contains("String"), "{}", returns(&function));
}

/// The block body must be the SAME operations a function body with these statements
/// lowers to: one set of statement rules, not a second dialect.
#[test]
fn block_body_equals_the_same_statements_inside_an_async_function() {
    let statements = [
        "total: int = 0",
        "items: list[int] = [1, 2, 3]",
        "for item in items:",
        "    total = total + item",
        "if total > 5:",
        "    label: str = \"big\"",
        "    return concat(label, to_string(total))",
        "response: Result[str] = await http_get(\"http://127.0.0.1:9/x\")",
        "match response:",
        "    case Ok(body):",
        "        return body",
        "    case Err(reason):",
        "        return reason",
    ];
    let block_source = statements.join("\n") + "\n";
    let function_source = format!(
        "@effects('io', 'fail', 'mut')\nasync def wrapped() -> str:\n{}\n",
        statements.iter().map(|line| format!("    {line}")).collect::<Vec<_>>().join("\n")
    );
    let wrapped = lower_module(&function_source, SCOPE).unwrap();
    let [p::Operation::Module(module)] = wrapped.operations.as_slice() else { panic!() };
    let [p::Operation::Function(expected)] = module.body.as_slice() else { panic!() };
    let actual = block(&block_source);
    assert_eq!(format!("{:?}", actual.body), format!("{:?}", expected.body));
    assert_eq!(returns(&actual), returns(expected));
}

#[test]
fn the_first_typed_return_fixes_the_result_type() {
    assert!(returns(&block("return 1 + 2\n")).contains("Int"));
    assert!(returns(&block("flag: bool = True\nreturn flag\n")).contains("Bool"));
    let result = returns(&block("r: Result[str] = await http_get(\"http://h/\")\nreturn r\n"));
    assert!(result.contains("Result") && result.contains("String"), "{result}");
    // An uncatalogued builtin result is unresolved: it neither fixes nor contradicts the
    // type, and an unresolved block result is declared as text.
    let unknown = block("xs: list[int] = [1]\nif len(xs) > 5:\n    return get(xs, 0)\nreturn 7\n");
    assert!(returns(&unknown).contains("Int"), "{}", returns(&unknown));
    assert!(returns(&block("xs: list[int] = [1]\nreturn get(xs, 0)\n")).contains("String"));
}

#[test]
fn later_returns_must_match_and_plain_values_are_not_lifted_into_result() {
    let error = refused("flag: bool = True\nif flag:\n    return 1\nreturn \"text\"\n");
    assert!(error.message.contains("return type mismatch"), "{error}");
    assert_eq!((error.line, error.column), (4, 8), "{error}");
    // A declared function may return plain T under Result[T] because invocation wraps
    // it. A block has no invocation boundary, so the same mix is a mismatch.
    let error = refused("r: Result[str] = await http_get(\"http://h/\")\nif r.is_ok:\n    return r\nreturn \"plain\"\n");
    assert!(error.message.contains("return type mismatch"), "{error}");
}

#[test]
fn falling_off_the_end_is_allowed() {
    let no_return = block("n: int = 1\n");
    assert_eq!(no_return.body.len(), 1);
    assert!(returns(&no_return).contains("String"));
    let partial = block("n: int = 1\nif n > 3:\n    return n\n");
    assert!(returns(&partial).contains("Int"));
}

#[test]
fn qualified_user_calls_resolve_in_any_scope_and_bare_ones_only_in_their_module() {
    let program = library();
    let qualified = block_in("return BlockLib.double(21)\n", SCOPE, &program).unwrap();
    assert!(format!("{:?}", qualified.body).contains("double"));
    let awaited = block_in("r: Result[str] = await BlockLib.fetch(\"http://h/\")\nreturn r\n", SCOPE, &program).unwrap();
    assert!(matches!(&awaited.body[0], p::Operation::Await(_)));

    let bare = block_in("return double(21)\n", "BlockLib", &program).unwrap();
    assert_eq!(format!("{:?}", bare.body), format!("{:?}", qualified.body), "a bare local call is qualified by lowering");
    let shared = block_in("counter = counter + 1\nreturn counter\n", "BlockLib", &program).unwrap();
    assert!(matches!(&shared.body[0], p::Operation::Set(decl) if decl.name == "counter"));

    let error = block_in("return double(21)\n", SCOPE, &program).unwrap_err();
    assert!(error.message.contains("has no authorized supported signature") || error.message.contains("unknown local callable"), "{error}");
    let error = block_in("return counter\n", SCOPE, &program).unwrap_err();
    assert!(error.message.contains("unbound"), "{error}");
}

#[test]
fn the_block_cannot_call_itself_or_an_unknown_function() {
    let error = block_in("return run_block()\n", "BlockLib", &library()).unwrap_err();
    assert!(error.message.contains("run_block"), "{error}");
    let error = refused("return BlockLib.missing(1)\n");
    assert!(error.message.contains("no authorized supported signature"), "{error}");
}

#[test]
fn declarations_imports_and_empty_blocks_are_refused_with_the_fix() {
    for source in [
        "def helper() -> int:\n    return 1\nreturn helper()\n",
        "async def helper() -> int:\n    return 1\n",
        "@dataclass\nclass Point:\n    x: int\n",
    ] {
        let error = refused(source);
        assert!(error.message.contains("mutate tool"), "{error}");
    }
    for source in ["import os\nreturn 1\n", "from os import path\nreturn 1\n"] {
        assert!(refused(source).message.contains("import is unsupported"));
    }
    for source in ["", "\n\n", "# only a comment\n", "pass\n"] {
        assert!(refused(source).message.contains("empty"), "{source:?}");
    }
}

#[test]
fn python_execution_escapes_and_reserved_names_are_refused() {
    for source in [
        "return eval(\"1\")\n",
        "return exec(\"1\")\n",
        "x: str = __import__(\"os\")\nreturn x\n",
        "return open(\"/etc/passwd\")\n",
        "h: str = __coroutine_handle\nreturn h\n",
        "__coroutine_handle: int = 1\nreturn 1\n",
        "x: int = 1\nreturn x.__class__\n",
        "return getattr(1, \"real\")\n",
        "f: int = lambda: 1\nreturn f\n",
        "xs: list[int] = [n for n in [1]]\nreturn xs\n",
        "try:\n    x: int = 1\nexcept Exception:\n    pass\n",
        "with open(\"x\") as f:\n    pass\n",
        "global counter\nreturn 1\n",
    ] {
        refused(source);
    }
}

#[test]
fn the_function_body_binding_and_await_rules_apply_unchanged() {
    let error = refused("n = len(\"abc\")\nreturn n\n");
    assert!(error.message.contains("first assignment requires a type annotation"), "{error}");
    assert_eq!((error.line, error.column), (1, 1), "{error}");
    let error = refused("text: str = file_read(\"/tmp/x\")\nreturn text\n");
    assert!(error.message.contains("requires an annotated await binding"), "{error}");
    let error = refused("n: int = await len(\"abc\")\nreturn n\n");
    assert!(error.message.contains("synchronous calls are not awaitable"), "{error}");
    let error = refused("n: int = \"text\"\nreturn n\n");
    assert!(error.message.contains("type mismatch"), "{error}");
    let error = refused("BlockLib.double(2)\n");
    assert!(error.message.contains("expression statement"), "{error}");
    let error = refused("return\n");
    assert!(error.message.contains("bare return"), "{error}");
}

/// The two shapes a model writes first get one text each, naming the corrected form. The
/// texts are the shared frontend's: a function body submitted through `mutate` (structural
/// preflight) and a module lowered without preflight get exactly the same refusal.
#[test]
fn unannotated_and_bare_awaits_are_refused_with_the_annotated_form() {
    let error = refused("text = await file_read(\"/tmp/x\")\nreturn text\n");
    assert_eq!(error.message, surface::UNANNOTATED_AWAIT);
    assert!(error.message.contains("`text: str = await file_read(path)`"), "{error}");
    assert_eq!((error.line, error.column), (1, 1), "{error}");
    let error = refused("ok: bool = True\nif ok:\n    await file_write(\"/tmp/x\", \"data\")\n");
    assert_eq!(error.message, surface::BARE_AWAIT);
    assert!(error.message.contains("`written: str = await file_write(path, text)`"), "{error}");
    assert_eq!((error.line, error.column), (3, 5), "{error}");
    for (body, expected, position) in [
        ("    text = await file_read(\"/tmp/x\")\n    return text\n", surface::UNANNOTATED_AWAIT, (3, 5)),
        ("    text: str = \"\"\n    text = await file_read(\"/tmp/x\")\n    return text\n", surface::UNANNOTATED_AWAIT, (4, 5)),
        ("    await file_write(\"/tmp/x\", \"data\")\n    return \"done\"\n", surface::BARE_AWAIT, (3, 5)),
    ] {
        let function = format!("@effects('io')\nasync def probe() -> str:\n{body}");
        for error in [
            preflight_module(&function, SCOPE).err().expect("mutate preflight must refuse"),
            lower_module(&function, SCOPE).expect_err("lowering must refuse"),
        ] {
            assert_eq!(error.message, expected, "{function}");
            assert_eq!((error.line, error.column), position, "{function}");
        }
    }
    // Rebinding with the annotation repeated, as the text says, is admitted.
    lower_module("@effects('io')\nasync def probe() -> str:\n    text: str = \"\"\n    text: str = await file_read(\"/tmp/x\")\n    return text\n", SCOPE).unwrap();
    for nested in [
        "xs: list[int] = [1]\nfor x in xs:\n    t = await file_read(\"/x\")\n",
        "go: bool = True\nwhile go:\n    t = await file_read(\"/x\")\n",
        "r: Result[str] = await http_get(\"http://h/\")\nmatch r:\n    case Ok(body):\n        t = await file_read(body)\n    case Err(reason):\n        return reason\n",
    ] {
        assert_eq!(refused(nested).message, surface::UNANNOTATED_AWAIT, "{nested}");
    }
    // The annotated forms the refusals recommend are admitted.
    block("text: str = await file_read(\"/tmp/x\")\nwritten: str = await file_write(\"/tmp/y\", text)\nreturn written\n");
}

#[test]
fn diagnostics_carry_the_position_in_the_submitted_text() {
    let source = "first: int = 1\nsecond: str = \"\"\"line one\nline two\"\"\"\n    \nif first > 0:\n    third: int = first + missing\nreturn first\n";
    let error = refused(source);
    assert!(error.message.contains("missing"), "{error}");
    assert_eq!((error.line, error.column), (6, 26), "{error}");
    assert_eq!(error.module, SCOPE);
    let syntax = refused("ok: int = 1\nbroken: int = (\n");
    assert!(syntax.message.contains("Python syntax"), "{syntax}");
    assert!(syntax.line >= 2, "{syntax}");
}

#[test]
fn multi_line_strings_are_preserved_byte_for_byte() {
    let function = block("text: str = \"\"\"a\n  b\n\tc\"\"\"\nreturn text\n");
    let p::Operation::Let(binding) = &function.body[0] else { panic!() };
    assert!(matches!(&binding.expr, p::Expr::String(value) if value == "a\n  b\n\tc"), "{:?}", binding.expr);
}

#[test]
fn preflight_reports_references_without_a_program() {
    let preflight = preflight_block("a: int = BlockLib.double(2)\nb: str = await file_read(\"/x\")\nc: str = await mutate(\"+module X\")\nreturn helper(a)\n", SCOPE).unwrap();
    assert_eq!(preflight.references, ["BlockLib.double"]);
    assert_eq!(preflight.local_calls, ["helper"]);
    assert!(preflight.declared_functions.is_empty() && preflight.declared_types.is_empty());
    for name in ["file_read", "mutate", "helper"] {
        assert!(preflight.bare_calls.iter().any(|call| call == name), "{name}: {:?}", preflight.bare_calls);
    }
}
