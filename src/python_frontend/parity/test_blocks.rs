//! Stored `+test` blocks. The validator never stores them in the Program, so the
//! comparison is on `parser::TestCase` values (Debug is a complete structural form of
//! input, expectation, matcher and after-checks), exactly like the port's stored-test
//! gate. No test-data policy is broadened: a block with any case the restricted Python
//! test syntax cannot express stays Unsupported with the first offending case named.
use super::{Check, Status};
use crate::{ast, parser, python_frontend::tests_print};

pub(super) fn check(program: &ast::Program, scope: &str, test: &parser::TestMutation) -> Check {
    let bare = &test.function_name;
    let expected = if scope.is_empty() {
        toplevel_target(program, bare)?
    } else {
        module_target(program, scope, bare)?
    };
    roundtrip(test, scope, &expected)
}

/// MODULE-LESS block (owner ""). The runtime runs `+test name` against `name` verbatim,
/// resolved by `Program::get_function` (top-level function first, otherwise the first
/// module defining it). The module-less Python lowering keeps a bare target bare and a
/// dotted target dotted, so the lowered string IS the runtime string: exact by
/// construction, and the roundtrip below requires that equality. Only existence is
/// checked here.
fn toplevel_target(program: &ast::Program, bare: &str) -> Result<String, (Status, String)> {
    if program.get_function(bare).is_none() {
        return Err((
            Status::InvalidProgram,
            format!("test targets undeclared function `{bare}`"),
        ));
    }
    Ok(bare.to_owned())
}

/// MODULE block. The session runs a module-body test against the bare name whenever ANY
/// function with that name exists (session.rs:636-641, Program::get_function is global),
/// otherwise against `Module.name`. The Python lowering always targets `Module.name`; the
/// block is exact only if both name the same function.
fn module_target(
    program: &ast::Program,
    scope: &str,
    bare: &str,
) -> Result<String, (Status, String)> {
    let session = if program.get_function(bare).is_some() {
        bare.to_owned()
    } else {
        format!("{scope}.{bare}")
    };
    let lowered_target = if bare.contains('.') {
        bare.to_owned()
    } else {
        format!("{scope}.{bare}")
    };
    let Some(session_function) = program.get_function(&session) else {
        return Err((
            Status::InvalidProgram,
            format!("test targets undeclared function `{bare}`"),
        ));
    };
    match program.get_function(&lowered_target) {
        Some(function) if std::ptr::eq(function, session_function) => Ok(lowered_target),
        Some(_) => Err((
            Status::Unsupported,
            format!(
                "bare test target `{bare}` runs {session}, but the module-qualified Python lowering targets {lowered_target} (#110)"
            ),
        )),
        None => Err((
            Status::Unsupported,
            format!("module-qualified test target {lowered_target} does not exist"),
        )),
    }
}

/// Print -> lower -> Debug-equal cases -> second print byte-identical -> second lowering
/// equal, requiring every lowered declaration to target `expected`.
fn roundtrip(test: &parser::TestMutation, scope: &str, expected: &str) -> Check {
    let bare = &test.function_name;
    let text = tests_print::print_block(test).map_err(|e| (Status::Unsupported, e))?;
    let cases = lower(&text, scope, expected)?;
    for (index, (original, lowered)) in test.cases.iter().zip(&cases).enumerate() {
        if format!("{original:?}") != format!("{lowered:?}") {
            return Err((
                Status::AstMismatch,
                format!("case {} changed through the Python roundtrip", index + 1),
            ));
        }
    }
    let again = tests_print::print_block(&parser::TestMutation {
        function_name: bare.clone(),
        cases: cases.clone(),
    })
    .map_err(|e| (Status::Unsupported, e))?;
    if again != text {
        return Err((
            Status::NonIdempotent,
            "second print differs byte-for-byte".into(),
        ));
    }
    let relowered = lower(&again, scope, expected)?;
    if format!("{cases:?}") != format!("{relowered:?}") {
        return Err((
            Status::AstMismatch,
            "second lowering changed the cases".into(),
        ));
    }
    Ok(())
}

/// Lower the printed block into one grouped operation targeting `expected`.
pub(super) fn lower(
    text: &str,
    scope: &str,
    expected: &str,
) -> Result<Vec<parser::TestCase>, (Status, String)> {
    let lowered = if scope.is_empty() {
        crate::python_frontend::lower_tests_toplevel(text)
    } else {
        crate::python_frontend::lower_tests(text, scope)
    }
    .map_err(|e| (Status::Unsupported, e.to_string()))?;
    if lowered.operations.len() != 1 {
        return Err((
            Status::AstMismatch,
            "expected one grouped test target".into(),
        ));
    }
    let mut cases = Vec::new();
    for operation in lowered.operations {
        let parser::Operation::Test(test) = operation else {
            return Err((Status::AstMismatch, "lowering produced a non-test".into()));
        };
        if test.function_name != expected || test.cases.is_empty() {
            return Err((
                Status::AstMismatch,
                format!(
                    "lowered target {} differs from {expected}",
                    test.function_name
                ),
            ));
        }
        cases.extend(test.cases);
    }
    if lowered.source_map.len() != cases.len()
        || lowered
            .source_map
            .iter()
            .any(|span| span.operation_index != 0)
    {
        return Err((
            Status::AstMismatch,
            "test definition/case mapping differs".into(),
        ));
    }
    Ok(cases)
}
