//! Stored `+test` blocks. The validator never stores them in the Program, so the
//! comparison is on `parser::TestCase` values (Debug is a complete structural form of
//! input, expectation, matcher and after-checks), exactly like the port's stored-test
//! gate. No test-data policy is broadened: a block with any case the restricted Python
//! test syntax cannot express stays Unsupported with the first offending case named.
use super::{Check, Status};
use crate::{ast, parser, python_frontend::tests_print};

pub(super) fn check(program: &ast::Program, scope: &str, test: &parser::TestMutation) -> Check {
    if scope.is_empty() {
        return Err((
            Status::Unsupported,
            "top-level stored test: the Python scope surface for module-less declarations is pending".into(),
        ));
    }
    // The session runs a module-body test against the bare name whenever ANY function
    // with that name exists (session.rs:636-641, Program::get_function is global),
    // otherwise against `Module.name`. The Python lowering always targets
    // `Module.name`; the block is exact only if both name the same function.
    let bare = &test.function_name;
    let session = if program.get_function(bare).is_some() {
        bare.clone()
    } else {
        format!("{scope}.{bare}")
    };
    let lowered_target = if bare.contains('.') {
        bare.clone()
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
        Some(function) if std::ptr::eq(function, session_function) => {}
        Some(_) => {
            return Err((
                Status::Unsupported,
                format!(
                    "bare test target `{bare}` runs {session}, but the module-qualified Python lowering targets {lowered_target}"
                ),
            ))
        }
        None => {
            return Err((
                Status::Unsupported,
                format!("module-qualified test target {lowered_target} does not exist"),
            ))
        }
    }
    let text = tests_print::print_block(test).map_err(|e| (Status::Unsupported, e))?;
    let cases = lower(&text, scope, bare)?;
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
    let relowered = lower(&again, scope, bare)?;
    if format!("{cases:?}") != format!("{relowered:?}") {
        return Err((
            Status::AstMismatch,
            "second lowering changed the cases".into(),
        ));
    }
    Ok(())
}

/// Lower the printed block and return its cases, requiring one declaration per case
/// all targeting the module-qualified function.
fn lower(text: &str, scope: &str, bare: &str) -> Result<Vec<parser::TestCase>, (Status, String)> {
    let lowered = crate::python_frontend::lower_tests(text, scope)
        .map_err(|e| (Status::Unsupported, e.to_string()))?;
    let expected = if bare.contains('.') {
        bare.to_owned()
    } else {
        format!("{scope}.{bare}")
    };
    let mut cases = Vec::new();
    for operation in lowered.operations {
        let parser::Operation::Test(test) = operation else {
            return Err((Status::AstMismatch, "lowering produced a non-test".into()));
        };
        if test.function_name != expected || test.cases.len() != 1 {
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
    Ok(cases)
}
