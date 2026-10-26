//! Roundtrip and negative tests for Task 3: module-level declarations —
//! `+shared`, `@route`, `@source`, startup/shutdown, source/event helpers, and
//! the whole-module printer — plus the keyword-variant/Option residual fixes.
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

fn reject(source: &str, fragment: &str) -> Diagnostic {
    let error = lower_module(source, "Example").unwrap_err();
    assert!(
        error.message.contains(fragment),
        "expected {fragment:?}, got {error}"
    );
    error
}

#[test]
fn shared_var_read_and_write_lower_to_bare_ident_and_set() {
    let source = "count: int = 0\n\n@effects('mut')\ndef incr() -> int:\n    count = count + 1\n    return count\n";
    let lowered = lower_module(source, "Example").unwrap();
    assert!(lowered.canonical_adapsis.contains("+shared count:Int = 0"));
    assert!(lowered.canonical_adapsis.contains("+set count = (count + 1)"));
    let original = program(source);
    let printed = printer::print_module(&original, "Example").unwrap();
    assert!(printed.contains("count: int = 0"));
    assert!(printed.contains("count = (count + 1)"));
    let rebuilt = program(&printed);
    assert_eq!(
        printer::print_module(&rebuilt, "Example").unwrap(),
        printed
    );
}

#[test]
fn shared_var_defaults_reject_function_calls() {
    reject("x: int = incr()\n\ndef incr() -> int:\n    return 1\n", "literal, list or record/variant constructor");
}

#[test]
fn route_decorator_roundtrips() {
    let source = "@route(\"POST\", \"/webhook\")\ndef handle(body: str) -> str:\n    return body\n";
    let lowered = lower_module(source, "Example").unwrap();
    assert!(lowered.canonical_adapsis.contains("+route POST \"/webhook\" -> handle"));
    let original = program(source);
    let printed = printer::print_module(&original, "Example").unwrap();
    assert!(printed.contains("@route(\"POST\", \"/webhook\")"));
    let rebuilt = program(&printed);
    assert_eq!(
        printer::print_module(&rebuilt, "Example").unwrap(),
        printed
    );
}

#[test]
fn source_decorator_and_inbody_helpers_roundtrip() {
    let source = "@source(\"heartbeat\", type=\"timer\", interval=\"60000\")\ndef on_heartbeat() -> str:\n    return \"beat\"\n\n@effects(\"io\")\nasync def setup() -> str:\n    source_add(\"timer(300000)\", \"poll\", \"on_tick\")\n    source_add(\"channel\", \"inbox\", \"on_message\")\n    source_remove(\"poll\")\n    source_replace(\"inbox\", \"timer(60000)\", \"on_message\")\n    source_list()\n    event_register(\"new_message\", \"String\")\n    event_emit(\"new_message\", \"hi\")\n    return \"ok\"\n";
    let lowered = lower_module(source, "Example").unwrap();
    assert!(lowered.canonical_adapsis.contains("+source heartbeat timer interval=60000 -> on_heartbeat"));
    assert!(lowered.canonical_adapsis.contains("+source add timer(300000) as poll -> on_tick"));
    assert!(lowered.canonical_adapsis.contains("+event register new_message(String)"));
    assert!(lowered.canonical_adapsis.contains("+event emit new_message \"hi\""));
    let original = program(source);
    let printed = printer::print_module(&original, "Example").unwrap();
    assert!(printed.contains("@source(\"heartbeat\", type=\"timer\", interval=\"60000\")"));
    assert!(printed.contains("source_add(\"timer(300000)\", \"poll\", \"on_tick\")"));
    assert!(printed.contains("event_register(\"new_message\", \"String\")"));
    let rebuilt = program(&printed);
    assert_eq!(
        printer::print_module(&rebuilt, "Example").unwrap(),
        printed
    );
}

#[test]
fn startup_shutdown_roundtrip() {
    let source = "@effects(\"io\")\nasync def startup() -> None:\n    source_add(\"timer(300000)\", \"poll\", \"on_tick\")\n    return \"started\"\n\n@effects(\"io\")\nasync def shutdown() -> None:\n    source_remove(\"poll\")\n    return \"stopped\"\n\ndef on_tick() -> str:\n    return \"tick\"\n";
    let lowered = lower_module(source, "Example").unwrap();
    assert!(lowered.canonical_adapsis.contains("+startup [io, async]"));
    assert!(lowered.canonical_adapsis.contains("+shutdown [io, async]"));
    let original = program(source);
    let printed = printer::print_module(&original, "Example").unwrap();
    assert!(printed.contains("async def startup() -> None:"));
    assert!(printed.contains("async def shutdown() -> None:"));
    let rebuilt = program(&printed);
    assert_eq!(
        printer::print_module(&rebuilt, "Example").unwrap(),
        printed
    );
}

#[test]
fn keyword_variant_option_roundtrip() {
    let source = "@union\nclass Option:\n    Some: tuple[int]\n    None_: tuple[()]\n\ndef unwrap_or(o: Option, default: int) -> int:\n    match o:\n        case Some(val):\n            return val\n        case None:\n            return default\n";
    let lowered = lower_module(source, "Example").unwrap();
    assert!(lowered.canonical_adapsis.contains("+type Option = Some(Int) | None"));
    assert!(lowered.canonical_adapsis.contains("+case None"));
    let original = program(source);
    let printed = printer::print_type(&original, "Example.Option").unwrap();
    assert!(printed.contains("None_: tuple[()]"));
    let rebuilt = program(&printed);
    assert_eq!(printer::print_type(&rebuilt, "Example.Option").unwrap(), printed);
    let fn_printed = printer::print_function(&original, "Example.unwrap_or").unwrap();
    assert!(fn_printed.contains("case None:"));
}

#[test]
fn option_subscript_rejected_when_option_type_declared() {
    reject(
        "@union\nclass Option:\n    Some: tuple[int]\n    None_: tuple[()]\n\ndef f(o: Option[int]) -> int:\n    return 0\n",
        "ambiguous",
    );
}

#[test]
fn route_and_source_reject_bad_decorator_args() {
    reject("@route(\"POST\")\ndef f() -> str:\n    return \"\"\n", "two string arguments");
    reject("@source(\"h\")\ndef f() -> str:\n    return \"\"\n", "requires a type=");
    reject("@route(\"POST\", \"/x\", \"extra\")\ndef f() -> str:\n    return \"\"\n", "two string arguments");
}

#[test]
fn source_event_helpers_reject_wrong_arity() {
    reject(
        "@effects(\"io\")\nasync def f() -> str:\n    source_add(\"timer(1)\")\n    return \"\"\n",
        "source_add(kind, alias, handler)",
    );
    reject(
        "@effects(\"io\")\nasync def f() -> str:\n    source_list(1)\n    return \"\"\n",
        "takes no arguments",
    );
    reject(
        "@effects(\"io\")\nasync def f() -> str:\n    event_emit(\"x\")\n    return \"\"\n",
        "event_emit(name, value)",
    );
}

#[test]
fn service_lifecycle_full_module_roundtrip() {
    let source = "\"Polling service\"\n\n@source(\"heartbeat\", type=\"timer\", interval=\"60000\")\ndef on_heartbeat() -> str:\n    return \"beat\"\n\n@effects(\"io\")\nasync def startup() -> None:\n    source_add(\"timer(300000)\", \"poll\", \"on_tick\")\n    return \"started\"\n\n@effects(\"io\")\nasync def shutdown() -> None:\n    source_remove(\"poll\")\n    return \"stopped\"\n\ndef on_tick() -> str:\n    return \"tick\"\n\ndef get_interval() -> int:\n    return 300000\n";
    let original = program(source);
    let printed = printer::print_module(&original, "Example").unwrap();
    let rebuilt = program(&printed);
    assert_eq!(
        printer::print_module(&rebuilt, "Example").unwrap(),
        printed
    );
}
