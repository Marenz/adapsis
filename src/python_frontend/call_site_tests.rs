//! Exercise the canonical dispatch visitor directly: source-lowering refusal
//! must not conceal a blind executable subtree in the second-line check.
use crate::{ast, parser, python_frontend, validator};

fn program() -> ast::Program {
    let source = "+module VisitTargets\n+fn sync (x:Int)->Int\n +return x\n+end\n+fn task (x:Int)->Int [async]\n +return x\n+end\n+fn result (x:Int)->Result<Int> [async]\n +return Ok(x)\n+end\n+fn io (x:Int)->Int [io,async]\n +return x\n+end\n+fn text ()->String [async]\n +return \"unused\"\n+end\n";
    let mut program = ast::Program::default();
    for op in parser::parse(source).unwrap() {
        validator::apply_and_validate(&mut program, &op).unwrap();
    }
    program.rebuild_function_index();
    program
}

fn check(body: &str, effects: &str) -> Result<(), String> {
    let source = format!("+module VisitCaller\n+fn probe ()->Int {effects}\n{body}\n+end\n");
    let operations = parser::parse(&source).unwrap();
    python_frontend::validate_runtime_arguments(&operations, &program())
}

fn reject(body: &str, effects: &str, message: &str) {
    let error = check(body, effects).unwrap_err();
    assert!(error.contains(message), "expected {message:?}: {error}");
}

#[test]
fn python_call_visitor_checks_match_subject_all_arms_check_and_spawn() {
    for body in [
        "  +match VisitTargets.task(1)\n  +case _\n    +return 0\n  +end",
        "  +match Ok(1)\n  +case Ok(value)\n    +return VisitTargets.task(value)\n  +case Err(error)\n    +return 0\n  +end",
        "  +check ready VisitTargets.task(1)==1 ~err_ready\n  +return 0",
        "  +spawn VisitTargets.sync(VisitTargets.task(1))\n  +return 0",
    ] {
        reject(body, "[io,async]", "annotated await");
    }
    for body in [
        "  +match Ok(1)\n  +case Ok(value)\n    +return 0\n  +case Err(error)\n    +return VisitTargets.sync()\n  +end",
        "  +check ready VisitTargets.sync()==1 ~err_ready\n  +return 0",
        "  +spawn VisitTargets.sync()\n  +return 0",
    ] {
        reject(body, "[io,async]", "exactly 1 arguments");
    }
}

#[test]
fn python_call_visitor_root_exceptions_do_not_reach_arguments_or_user_matches() {
    reject(
        "  +match VisitTargets.result(1)\n  +case _\n    +return 0\n  +end",
        "[io,async]",
        "annotated await",
    );
    reject(
        "  +match http_get(VisitTargets.text())\n  +case _\n    +return 0\n  +end",
        "[io,async]",
        "annotated await",
    );
    reject(
        "  +match Ok(http_get(\"fixture://no-io\"))\n  +case _\n    +return 0\n  +end",
        "[io,async]",
        "annotated await",
    );
    reject(
        "  +await value:Int = VisitTargets.result(1)\n  +return value",
        "[async]",
        "#82",
    );
    reject(
        "  +await value:Int = VisitTargets.task(VisitTargets.task(1))\n  +return value",
        "[async]",
        "annotated await",
    );
}

#[test]
fn python_call_visitor_enforces_caller_effects_and_spawn_target_kind() {
    let matched = "  +match http_get(\"fixture://match\")\n  +case _\n    +return 0\n  +end";
    assert!(check(matched, "[io,async]").is_ok());
    reject(matched, "[async]", "@effects('io')");
    reject(matched, "[io]", "async def");
    reject(
        "  +spawn VisitTargets.task(1)\n  +return 0",
        "",
        "spawn requires async def",
    );
    reject(
        "  +spawn VisitTargets.io(1)\n  +return 0",
        "[async]",
        "@effects('io')",
    );
    reject(
        "  +spawn http_get(\"fixture://never-spawn\")\n  +return 0",
        "[io,async]",
        "qualified user-function",
    );
    for body in [
        "  +spawn VisitTargets.sync(1)\n  +return 0",
        "  +spawn VisitTargets.task(1)\n  +return 0",
        "  +spawn VisitTargets.result(1)\n  +return 0",
        "  +await value:Int = VisitTargets.task(1)\n  +return value",
    ] {
        assert!(check(body, "[async]").is_ok(), "{body}");
    }
}

#[test]
fn python_call_visitor_checks_lifecycle_bodies_and_rejects_unchecked_operations() {
    for lifecycle in ["startup", "shutdown"] {
        let source = format!(
            "+module VisitLife\n+{lifecycle} [io,async]\n  +check ready VisitTargets.sync()==1 ~err_ready\n+end\n"
        );
        let error = python_frontend::validate_runtime_arguments(
            &parser::parse(&source).unwrap(),
            &program(),
        )
        .unwrap_err();
        assert!(error.contains("exactly 1 arguments"), "{error}");
    }
    reject(
        "  +source add timer(VisitTargets.task(1)) as poll -> VisitTargets.sync\n  +return 0",
        "[io,async]",
        "unsupported executable operation",
    );
    let operations = parser::parse("!done").unwrap();
    assert!(python_frontend::validate_runtime_arguments(&operations, &program()).is_err());
}

#[test]
fn python_call_visitor_io_match_rejects_same_submission_variant_shadowing() {
    let source = "+module VisitShadow\n+type Collision = http_get(String) | Other\n+fn probe ()->Int [io,async]\n  +match http_get(\"fixture://never-called\")\n  +case _\n    +return 0\n  +end\n+end\n";
    let error =
        python_frontend::validate_runtime_arguments(&parser::parse(source).unwrap(), &program())
            .unwrap_err();
    assert!(error.contains("unshadowed runtime builtin"), "{error}");
}

#[test]
fn python_call_visitor_io_match_rechecks_snapshot_function_and_variant_shadowing() {
    let probe = parser::parse("+module VisitSnapshotProbe\n+fn probe ()->Int [io,async]\n  +match http_get(\"fixture://not-called\")\n  +case _\n    +return 0\n  +end\n+end\n").unwrap();
    for declaration in [
        "+fn http_get (url:String)->String\n +return url\n+end\n",
        "+type Collision = http_get(String) | Other\n",
    ] {
        let mut program = program();
        for op in parser::parse(&format!("+module VisitExistingShadow\n{declaration}")).unwrap() {
            validator::apply_and_validate(&mut program, &op).unwrap();
        }
        let error = python_frontend::validate_runtime_arguments(&probe, &program).unwrap_err();
        assert!(error.contains("unshadowed runtime builtin"), "{error}");
    }
}
