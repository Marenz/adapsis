//! Module-less (owner "") declarations for the strict harness. Each check applies the
//! declaration to a program that does NOT already contain it, so the validator's own
//! duplicate rule stays in force, and compares through the test-only top-level surface.
use super::{equality, same_text, Check, Status};
use crate::{ast, python_frontend};

fn rest_without(program: &ast::Program, name: &str) -> ast::Program {
    let mut rest = program.clone();
    rest.types.retain(|t| t.name() != name);
    rest
}

/// Lower `text` into a fresh program holding everything except the type itself and
/// return that program (the type, if lowering succeeded, is the one appended).
fn rebuilt(text: &str, rest: &ast::Program) -> Result<ast::Program, (Status, String)> {
    let lowered = python_frontend::lower_toplevel(text, rest)
        .map_err(|e| (Status::Unsupported, e.to_string()))?;
    let mut program = rest.clone();
    super::build_into(&mut program, &lowered.operations).map_err(|e| (Status::Unsupported, e))?;
    Ok(program)
}

pub(super) fn check_type(program: &ast::Program, name: &str) -> Check {
    let original = program.types.iter().find(|t| t.name() == name).ok_or((
        Status::InvalidProgram,
        "type missing after validation".into(),
    ))?;
    let text = python_frontend::print_toplevel_type(program, name)
        .map_err(|e| (Status::Unsupported, e.to_string()))?;
    let rest = rest_without(program, name);
    let fresh = rebuilt(&text, &rest)?;
    let candidate = fresh
        .types
        .iter()
        .find(|t| t.name() == name)
        .ok_or((Status::AstMismatch, "type missing after lowering".into()))?;
    if !equality::ty(original, candidate) {
        return Err((Status::AstMismatch, "type changed".into()));
    }
    // Second print from the re-lowered declaration, then a second lowering.
    let mut second_catalog = program.clone();
    *second_catalog
        .types
        .iter_mut()
        .find(|t| t.name() == name)
        .unwrap() = candidate.clone();
    let second = python_frontend::print_toplevel_type(&second_catalog, name)
        .map_err(|e| (Status::Unsupported, e.to_string()))?;
    same_text(&text, &second)?;
    let again = rebuilt(&second, &rest)?;
    let candidate = again.types.iter().find(|t| t.name() == name).unwrap();
    if !equality::ty(original, candidate) {
        return Err((Status::AstMismatch, "second lowering changed type".into()));
    }
    Ok(())
}
