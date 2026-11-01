//! Module-less (owner "") declarations for the strict harness. Each check applies the
//! declaration to a program that does NOT already contain it, so the validator's own
//! duplicate rule stays in force, and compares through the test-only top-level surface.
use super::{equality, same_text, Check, Status};
use crate::{ast, python_frontend};

fn rest_without_type(program: &ast::Program, name: &str) -> ast::Program {
    let mut rest = program.clone();
    rest.types.retain(|t| t.name() != name);
    rest
}

fn rest_without_function(program: &ast::Program, name: &str) -> ast::Program {
    let mut rest = program.clone();
    rest.functions.retain(|f| f.name != name);
    rest.rebuild_function_index();
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
    let rest = rest_without_type(program, name);
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

pub(super) fn check_function(program: &ast::Program, name: &str) -> Check {
    let original = program.functions.iter().find(|f| f.name == name).ok_or((
        Status::InvalidProgram,
        "top-level function missing after validation".into(),
    ))?;
    let text = python_frontend::print_toplevel_function(program, name)
        .map_err(|e| (Status::Unsupported, e.to_string()))?;
    let rest = rest_without_function(program, name);
    let fresh = rebuilt(&text, &rest)?;
    let candidate = fresh.functions.iter().find(|f| f.name == name).ok_or((
        Status::AstMismatch,
        "function missing after re-lowering".into(),
    ))?;
    if !equality::function(original, candidate) {
        return Err((
            Status::AstMismatch,
            format!(
                "{} changed (only typed NodeIds ignored)",
                equality::function_difference(original, candidate)
            ),
        ));
    }
    // Second print from the re-lowered declaration, then a second lowering.
    let mut second_catalog = program.clone();
    for slot in second_catalog
        .functions
        .iter_mut()
        .filter(|f| f.name == name)
    {
        *slot = candidate.clone();
    }
    second_catalog.rebuild_function_index();
    let second = python_frontend::print_toplevel_function(&second_catalog, name)
        .map_err(|e| (Status::Unsupported, e.to_string()))?;
    same_text(&text, &second)?;
    let again = rebuilt(&second, &rest)?;
    let candidate = again.functions.iter().find(|f| f.name == name).unwrap();
    if !equality::function(original, candidate) {
        return Err((
            Status::AstMismatch,
            "second lowering changed function".into(),
        ));
    }
    Ok(())
}

/// The whole module-less surface as ONE Python submission: types in program order, then
/// functions in program order (so recursion, mutual recursion and forward references
/// resolve inside a single lowering). Empty when the program has none.
pub(super) fn print_all(program: &ast::Program) -> Result<String, (Status, String)> {
    let mut parts = Vec::new();
    for ty in &program.types {
        parts.push(
            python_frontend::print_toplevel_type(program, ty.name())
                .map_err(|e| (Status::Unsupported, e.to_string()))?,
        );
    }
    for function in &program.functions {
        parts.push(
            python_frontend::print_toplevel_function(program, &function.name)
                .map_err(|e| (Status::Unsupported, e.to_string()))?,
        );
    }
    Ok(parts.join("\n"))
}

/// Print and lower all module-less declarations against the program WITHOUT them (the
/// validator's own duplicate rule stays in force). Returns the type operations, the
/// function operations (to be applied after the modules) and the printed text.
pub(super) fn lower_all(
    program: &ast::Program,
) -> Result<
    (
        Vec<crate::parser::Operation>,
        Vec<crate::parser::Operation>,
        String,
    ),
    (Status, String),
> {
    let text = print_all(program)?;
    if text.is_empty() {
        return Ok((Vec::new(), Vec::new(), text));
    }
    let mut rest = program.clone();
    rest.types.clear();
    rest.functions.clear();
    rest.rebuild_function_index();
    let lowered = python_frontend::lower_toplevel(&text, &rest)
        .map_err(|e| (Status::Unsupported, e.to_string()))?;
    let (types, functions) = lowered
        .operations
        .into_iter()
        .partition(|op| matches!(op, crate::parser::Operation::Type(_)));
    Ok((types, functions, text))
}
