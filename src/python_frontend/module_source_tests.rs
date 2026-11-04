use super::{lower_python_module, print_python_module};
use crate::{ast, parser, validator};
use std::sync::Arc;

fn build(source: &str, module: &str) -> ast::Program {
    let lowered = super::lower_module(source, module).unwrap_or_else(|e| panic!("{e}"));
    let mut program = ast::Program::default();
    for operation in &lowered.operations {
        validator::apply_and_validate(&mut program, operation).unwrap();
    }
    program.pending_routes.clear();
    program
}

fn store_tests(program: &mut ast::Program, module: &str, source: &str) {
    let lowered = super::lower_tests(source, module).unwrap_or_else(|e| panic!("{e}"));
    for operation in lowered.operations {
        if let parser::Operation::Test(test) = operation {
            crate::session::store_test(program, &test.function_name, &test.cases);
        }
    }
}

fn full_source() -> &'static str {
    r#"'Full module with @stored_test("Example.inc") inside docs'

@dataclass
class Point:
    x: int
    y: int

count: int = 0

@route("POST", "/hook")
def handle(body: str) -> str:
    return body

@source("heartbeat", type="timer", interval="60000")
def on_heartbeat() -> str:
    return "beat"

@effects("io")
async def startup() -> None:
    source_list()
    return "started"

@effects("io")
async def shutdown() -> None:
    return "stopped"

def inc(x: int) -> int:
    return x + 1

def greet(name: str) -> str:
    return name

def origin() -> Point:
    return Point(x=0, y=1)

def test_case_1() -> int:
    return 0
"#
}

fn full_program() -> ast::Program {
    let mut program = build(full_source(), "Example");
    store_tests(
        &mut program,
        "Example",
        "def test_inc_1() -> None:\n    assert inc(1) == 2\n\ndef test_inc_2() -> None:\n    assert inc(2) == 3\n\ndef test_greet() -> None:\n    assert greet(\"a\") == \"a\"\n",
    );
    program
}

#[test]
fn full_module_and_stored_tests_roundtrip_through_the_checked_api() {
    let program = full_program();
    let text = print_python_module(&program, "Example").unwrap();
    assert!(text.contains("Full module with @stored_test"));
    assert!(text.contains("@dataclass"));
    assert!(text.contains("@route(\"POST\", \"/hook\")"));
    assert!(text.contains("@source(\"heartbeat\", type=\"timer\", interval=\"60000\")"));
    assert!(text.contains("async def startup() -> None:"));
    assert!(text.contains("@stored_test(\"Example.inc\")"));
    assert!(text.contains("@stored_test(\"Example.greet\")"));
    assert!(text.contains("def test_case_1() -> int:"));
    assert!(text.contains("def test_case_2() -> None:"));
    assert!(text.contains("def test_case_3() -> None:"));
    assert!(text.contains("def test_case_4() -> None:"));

    let lowered = lower_python_module(&text, "Example", &ast::Program::default()).unwrap();
    let targets: Vec<_> = lowered
        .tests
        .iter()
        .map(|test| test.function_name.as_str())
        .collect();
    assert_eq!(targets, vec!["Example.inc", "Example.greet"]);
    assert_eq!(lowered.tests[0].cases.len(), 2);
    assert_eq!(lowered.tests[1].cases.len(), 1);

    let mut rebuilt = ast::Program::default();
    for operation in &lowered.declarations.operations {
        validator::apply_and_validate(&mut rebuilt, operation).unwrap();
    }
    rebuilt.pending_routes.clear();
    for test in &lowered.tests {
        crate::session::store_test(&mut rebuilt, &test.function_name, &test.cases);
    }
    let again = print_python_module(&rebuilt, "Example").unwrap();
    assert_eq!(text, again);
}

#[test]
fn load_catalog_rejects_stale_same_name_and_top_level_members() {
    let program = full_program();
    let text = print_python_module(&program, "Example").unwrap();
    assert!(lower_python_module(&text, "Example", &program).is_err());

    let mut top_level = ast::Program::default();
    top_level.types.push(program.modules[0].types[0].clone());
    let mut save_with_top_level = program.clone();
    save_with_top_level
        .types
        .push(program.modules[0].types[0].clone());
    let error = print_python_module(&save_with_top_level, "Example").unwrap_err();
    assert!(error.to_string().contains("top-level"));
    let error = lower_python_module(&text, "Example", &top_level).unwrap_err();
    assert!(error.to_string().contains("top-level"));
}

#[test]
fn save_and_load_reject_pending_routes() {
    let mut program = full_program();
    program.pending_routes.push(ast::HttpRoute {
        method: "GET".into(),
        path: "/pending".into(),
        handler_fn: "Example.handle".into(),
    });
    assert!(print_python_module(&program, "Example").is_err());
    let text = "def f() -> int:\n    return 1\n";
    assert!(lower_python_module(text, "Example", &program).is_err());
}

#[test]
fn foreign_type_with_one_exact_owner_roundtrips() {
    let dependency = build(
        "@dataclass\nclass Point:\n    x: int\n    y: int\n\ndef make(x: int, y: int) -> Point:\n    return Point(x=x, y=y)\n",
        "Types",
    );
    let current = super::lower_module_with_program(
        "def origin() -> Point:\n    return Types.make(1, 2)\n",
        "Uses",
        &dependency,
    )
    .unwrap();
    let mut save = dependency.clone();
    for operation in &current.operations {
        validator::apply_and_validate(&mut save, operation).unwrap();
    }
    let text = print_python_module(&save, "Uses").unwrap();
    assert!(text.contains("-> Point"));
    let lowered = lower_python_module(&text, "Uses", &dependency).unwrap();
    let mut rebuilt = dependency;
    for operation in &lowered.declarations.operations {
        validator::apply_and_validate(&mut rebuilt, operation).unwrap();
    }
    assert!(rebuilt.get_function("Uses.origin").is_some());
}

#[test]
fn foreign_type_with_ambiguous_owner_is_refused() {
    let first = build("@dataclass\nclass Point:\n    x: int\n", "First");
    let second = build("@dataclass\nclass Point:\n    y: int\n", "Second");
    let current =
        super::lower_module("def origin(x: Point) -> int:\n    return 0\n", "Uses").unwrap();
    let mut save = ast::Program::default();
    save.modules.push(first.modules[0].clone());
    save.modules.push(second.modules[0].clone());
    for operation in &current.operations {
        validator::apply_and_validate(&mut save, operation).unwrap();
    }
    let error = print_python_module(&save, "Uses").unwrap_err();
    assert!(error.to_string().contains("ambiguous"), "{error}");
    let mut closure = save.clone();
    closure.modules.retain(|module| module.name != "Uses");
    closure.rebuild_function_index();
    let error = lower_python_module(
        "def origin(x: Point) -> int:\n    return 0\n",
        "Uses",
        &closure,
    )
    .unwrap_err();
    assert!(error.to_string().contains("ambiguous"), "{error}");
}

#[test]
fn unknown_type_and_missing_qualified_callee_are_refused() {
    let unknown = build("def f(x: Missing) -> int:\n    return 0\n", "Uses");
    let error = print_python_module(&unknown, "Uses").unwrap_err();
    assert!(error.to_string().contains("Missing"), "{error}");

    let missing = build("def f() -> int:\n    return Other.missing(1)\n", "Uses");
    let error = print_python_module(&missing, "Uses").unwrap_err();
    assert!(error.to_string().contains("Other.missing"), "{error}");
}

#[test]
fn malformed_stored_test_cases_refuse_the_whole_save() {
    let mut program = full_program();
    let function = program
        .modules
        .iter_mut()
        .find(|m| m.name == "Example")
        .unwrap()
        .functions
        .iter_mut()
        .find(|f| f.name == "inc")
        .unwrap();
    Arc::make_mut(function).tests[0].input = "Ok(\"x\")".into();
    let error = print_python_module(&program, "Example").unwrap_err();
    assert!(error.to_string().contains("call to Ok"), "{error}");

    let mut program = full_program();
    let function = program
        .modules
        .iter_mut()
        .find(|m| m.name == "Example")
        .unwrap()
        .functions
        .iter_mut()
        .find(|f| f.name == "inc")
        .unwrap();
    Arc::make_mut(function).tests[0].input = "\"unterminated".into();
    assert!(print_python_module(&program, "Example").is_err());
}

#[test]
fn duplicate_or_dangling_members_are_refused_not_dropped() {
    let base = full_program();
    let mut nested = base.clone();
    nested.modules[0].modules.push(base.modules[0].clone());
    assert!(print_python_module(&nested, "Example").is_err());

    let mut event = base.clone();
    event.modules[0].event_decls.push(ast::EventDecl {
        name: "tick".into(),
        payload_type: ast::Type::String,
    });
    assert!(print_python_module(&event, "Example").is_err());

    let mut dangling_route = base.clone();
    dangling_route.modules[0].routes.push(ast::HttpRoute {
        method: "GET".into(),
        path: "/dangling".into(),
        handler_fn: "Example.missing".into(),
    });
    assert!(print_python_module(&dangling_route, "Example").is_err());

    let mut duplicate_route = base.clone();
    duplicate_route.modules[0].routes.push(ast::HttpRoute {
        method: "GET".into(),
        path: "/second".into(),
        handler_fn: "Example.handle".into(),
    });
    assert!(print_python_module(&duplicate_route, "Example").is_err());

    let mut dangling_source = base.clone();
    dangling_source.modules[0].sources.push(ast::SourceDecl {
        name: "other".into(),
        source_type: "timer".into(),
        config: vec![],
        handler: "missing".into(),
    });
    assert!(print_python_module(&dangling_source, "Example").is_err());

    let mut duplicate_source = base.clone();
    let existing = duplicate_source.modules[0].sources[0].clone();
    duplicate_source.modules[0].sources.push(existing);
    assert!(print_python_module(&duplicate_source, "Example").is_err());
}

#[test]
fn lower_rejects_bad_decorator_targets_and_duplicate_test_names() {
    let source = "def f() -> int:\n    return 1\n\n@stored_test(\"Example.missing\")\ndef test_case_1() -> None:\n    assert f() == 1\n";
    assert!(lower_python_module(source, "Example", &ast::Program::default()).is_err());

    let source = "def f() -> int:\n    return 1\n\n@stored_test(\"Example.f\")\ndef test_case_1() -> None:\n    assert f() == 1\n\n@stored_test(\"Example.f\")\ndef test_case_1() -> None:\n    assert f() == 2\n";
    let error = lower_python_module(source, "Example", &ast::Program::default()).unwrap_err();
    assert!(
        error.to_string().contains("duplicate stored test"),
        "{error}"
    );

    let source = "def f() -> int:\n    return 1\n\n@stored_test(\"Example.f\")\ndef test_case_1() -> int:\n    return 0\n";
    assert!(lower_python_module(source, "Example", &ast::Program::default()).is_err());
}

#[test]
fn decorator_text_inside_docstrings_and_literals_does_not_partition() {
    let source = "\"@stored_test(\\\"Example.f\\\") only text\"\n\ndef f() -> int:\n    return 1\n\ndef g() -> str:\n    return \"@stored_test(\\\"Example.f\\\")\"\n";
    let lowered = lower_python_module(source, "Example", &ast::Program::default()).unwrap();
    assert!(lowered.tests.is_empty());
    let mut program = ast::Program::default();
    for operation in &lowered.declarations.operations {
        validator::apply_and_validate(&mut program, operation).unwrap();
    }
    let text = print_python_module(&program, "Example").unwrap();
    assert!(text.contains("@stored_test(\\\"Example.f\\\")"));
}

#[test]
fn stripping_tests_preserves_module_error_line_positions() {
    let source = "def f() -> int:\n    return 1\n\n@stored_test(\"Example.f\")\ndef test_case_1() -> None:\n    assert f() == 1\n\ndef eval() -> int:\n    return 1\n";
    let error = lower_python_module(source, "Example", &ast::Program::default()).unwrap_err();
    assert_eq!(error.line, 8, "{error}");
}

#[test]
fn duplicate_raw_route_keys_are_refused_before_validator_merging() {
    let source = "@route(\"GET\", \"/same\")\ndef a() -> str:\n    return \"a\"\n\n@route(\"GET\", \"/same\")\ndef b() -> str:\n    return \"b\"\n";
    let error = lower_python_module(source, "Example", &ast::Program::default()).unwrap_err();
    assert!(error.to_string().contains("duplicate route"), "{error}");
}

#[test]
fn duplicate_raw_source_aliases_are_refused() {
    let source = "@source(\"alias\", type=\"timer\", interval=\"1\")\ndef a() -> str:\n    return \"a\"\n\n@source(\"alias\", type=\"timer\", interval=\"2\")\ndef b() -> str:\n    return \"b\"\n";
    let error = lower_python_module(source, "Example", &ast::Program::default()).unwrap_err();
    assert!(error.to_string().contains("duplicate source"), "{error}");
}

#[test]
fn stripping_tests_preserves_original_utf8_byte_offsets_and_spans() {
    let source = "def f() -> int:\n    return 1\n\n@stored_test(\"Example.f\")\ndef test_case_1() -> None:\n    assert f() == 1\n    assert_after(\"routes\", \"contains\", \"Ω😀\")\n\ndef later() -> int:\n    return 2\n";
    let expected = source.find("def later").unwrap();
    let lowered = lower_python_module(source, "Example", &ast::Program::default()).unwrap();
    let span = lowered
        .declarations
        .source_map
        .iter()
        .find(|span| span.start == expected)
        .unwrap_or_else(|| {
            panic!(
                "later function span start {:?} did not preserve original byte offset {expected}",
                lowered.declarations.source_map
            )
        });
    let slice = source.get(span.start..span.end).unwrap_or_else(|| {
        panic!("span {span:?} is not valid UTF-8 boundaries in original source")
    });
    assert!(slice.starts_with("def later"), "{span:?} -> {slice:?}");
}
