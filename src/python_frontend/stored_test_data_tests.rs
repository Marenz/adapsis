//! Trusted `@stored_test` data (#128 part 2): pure calls and constructors in
//! persisted `.py` test cases, while native/CLI literal policy is unchanged.
use super::{lower_python_module, print_python_module};
use crate::{ast, parser, validator};

/// Builds a module from `.ax` (declarations plus stored `+test` cases).
fn program(source: &str) -> ast::Program {
    let mut program = ast::Program::default();
    let operations = parser::parse(source).unwrap();
    for operation in &operations {
        if !matches!(operation, parser::Operation::Test(_)) {
            validator::apply_and_validate(&mut program, operation).unwrap();
        }
    }
    program.rebuild_function_index();
    for operation in &operations {
        let parser::Operation::Module(module) = operation else { continue };
        for child in &module.body {
            if let parser::Operation::Test(test) = child {
                let target = format!("{}.{}", module.name, test.function_name);
                crate::session::store_test(&mut program, &target, &test.cases);
            }
        }
    }
    program.pending_routes.clear();
    program
}

fn without(program: &ast::Program, module: &str) -> ast::Program {
    let mut other = program.clone();
    other.modules.retain(|m| m.name != module);
    other.rebuild_function_index();
    other
}

/// Print, lower, rebuild with the lowered (typed) cases, and run them.
fn roundtrip_and_run(program: &ast::Program, module: &str) -> String {
    let text = print_python_module(program, module).unwrap_or_else(|e| panic!("{e}"));
    let lowered = lower_python_module(&text, module, &without(program, module)).unwrap();
    let mut rebuilt = without(program, module);
    for operation in &lowered.declarations.operations {
        validator::apply_and_validate(&mut rebuilt, operation).unwrap();
    }
    rebuilt.rebuild_function_index();
    for test in &lowered.tests {
        crate::session::store_test_payload(&mut rebuilt, &test.function_name, &test.cases, true);
        for case in &test.cases {
            crate::eval::eval_test_case(&rebuilt, &test.function_name, case)
                .unwrap_or_else(|e| panic!("{}: {e}", test.function_name));
        }
    }
    assert_eq!(print_python_module(&rebuilt, module).unwrap(), text, "typed reprint is byte-identical");
    text
}

#[test]
fn concat_and_pure_helper_calls_in_stored_test_data_print_roundtrip_and_run() {
    let p = program(concat!(
        "+module SdA\n",
        "+fn sample (n:Int)->String\n  +return concat(\"s\", \"x\")\n+end\n",
        "+fn wrap (s:String)->String\n  +return concat(\"<\", s)\n+end\n",
        "+test wrap\n",
        "  +with concat(\"a\", \"b\") -> expect \"<ab\"\n",
        "  +with sample(2) -> expect concat(\"<\", \"sx\")\n",
    ));
    let text = roundtrip_and_run(&p, "SdA");
    assert!(text.contains("wrap(concat(\"a\", \"b\")) == \"<ab\""), "{text}");
    // A bare user helper keeps its exact spelling: the runtime resolves test
    // data exactly as the original stored case did.
    assert!(text.contains("wrap(sample(2)) == concat(\"<\", \"sx\")"), "{text}");
}

#[test]
fn ok_err_some_none_constructors_work_in_stored_test_data() {
    let p = program(concat!(
        "+module SdB\n",
        "+fn ok (x:Int)->Result<Int>\n  +return Ok(x)\n+end\n",
        "+fn opt (x:Option<Int>)->Int\n  +match x\n  +case Some(v)\n    +return v\n  +case None\n    +return 0\n  +end\n+end\n",
        "+test ok\n  +with 1 -> expect Ok(1)\n",
        "+test opt\n  +with Some(4) -> expect 4\n  +with None -> expect 0\n",
    ));
    let text = roundtrip_and_run(&p, "SdB");
    assert!(text.contains("ok(1) == Ok(1)"), "{text}");
    assert!(text.contains("opt(Some(4)) == 4"), "{text}");
    assert!(text.contains("opt(None) == 0"), "{text}");
}

const IO_MODULE: &str = concat!(
    "+module SdC\n",
    "+fn fetch ()->String [io,async]\n  +await body:String = file_read(\"x\")\n  +return body\n+end\n",
    "+fn echo (s:String)->String\n  +return s\n+end\n",
);

#[test]
fn effectful_call_in_stored_test_data_is_refused_at_save() {
    let p = program(&format!("{IO_MODULE}+test echo\n  +with fetch() -> expect \"x\"\n"));
    let error = print_python_module(&p, "SdC").unwrap_err();
    assert!(
        error.message.contains("call to fetch declares the io effect"),
        "{}",
        error.message
    );
    let p = program(&format!("{IO_MODULE}+test echo\n  +with file_read(\"x\") -> expect \"x\"\n"));
    let error = print_python_module(&p, "SdC").unwrap_err();
    assert!(error.message.contains("call to file_read is an IO builtin"), "{}", error.message);
}

#[test]
fn effectful_call_in_stored_test_data_is_refused_at_load() {
    let source = "@effects(\"io\")\nasync def fetch() -> str:\n    body: str = await file_read(\"x\")\n    return body\n\ndef echo(s: str) -> str:\n    return s\n\n@stored_test(\"SdD.echo\")\ndef test_case_1() -> None:\n    assert SdD.echo(fetch()) == \"x\"\n";
    let error = lower_python_module(source, "SdD", &ast::Program::default()).unwrap_err();
    assert!(error.message.contains("call to fetch declares the io effect"), "{}", error.message);
    let unknown = source.replace("SdD.echo(fetch())", "SdD.echo(missing_helper(1))");
    let error = lower_python_module(&unknown, "SdD", &ast::Program::default()).unwrap_err();
    assert!(error.message.contains("is not a known function"), "{}", error.message);
}

#[test]
fn native_and_cli_literal_policy_is_unchanged() {
    // The ordinary test lowerer (native test tool / CLI) still refuses calls,
    // None and constructors as test data.
    for source in [
        "def test_a() -> None:\n    assert wrap(concat(\"a\", \"b\")) == \"<ab\"\n",
        "def test_a() -> None:\n    assert ok(1) == Ok(1)\n",
        "def test_a() -> None:\n    assert opt(None) == 0\n",
    ] {
        let error = super::lower_tests(source, "Native").unwrap_err();
        assert!(
            error.message.contains("literal required") || error.message.contains("unsupported literal"),
            "{}",
            error.message
        );
    }
    // The literal printer (native inspection) also refuses calls.
    let case = parser::TestCase {
        input: parser::parse_test_input(0, "concat(\"a\", \"b\")").unwrap(),
        expected: parser::Expr::String("ab".into()),
        matcher: None,
        after_checks: vec![],
    };
    let error = super::tests_print::print_case("Native.f", &case).unwrap_err();
    assert!(error.contains("call to concat is not a Python test-data literal"), "{error}");
    // `@stored_test` itself is still refused by the ordinary lowerers.
    let stored = "@stored_test(\"Native.f\")\ndef test_case_1() -> None:\n    assert Native.f(1) == 1\n";
    assert!(super::lower_tests(stored, "Native").is_err());
}
