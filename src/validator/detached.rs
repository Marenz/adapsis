//! Conversion and validation of ONE module-owned function without inserting it into a
//! Program. `apply_module` uses it for every declared function; the native `run` tool
//! (#129) uses it for an ad-hoc block that is checked and executed but never declared.
use super::*;

/// Convert and validate ONE function as a member of `module`, without inserting it:
/// reserved-name refusal, union-type resolution, struct-literal and effect validation.
/// `module_types` are the module's types as of this declaration's position.
pub(super) fn checked_module_function(
    program: &ast::Program,
    module_types: &[ast::TypeDecl],
    module: &str,
    decl: &parser::FunctionDecl,
) -> Result<ast::FunctionDecl> {
    let mut converted = convert_function(decl)?;
    let union_names = {
        let mut names = collect_union_names(program);
        for td in module_types {
            if let ast::TypeDecl::TaggedUnion(u) = td {
                names.insert(u.name.clone());
            }
        }
        names
    };
    if !union_names.is_empty() {
        resolve_type(&mut converted.return_type, &union_names);
        for param in &mut converted.params {
            resolve_type(&mut param.ty, &union_names);
        }
        resolve_union_types_in_stmts(&mut converted.body, &union_names);
    }
    validate_function_struct_literals(program, &converted, Some(module))?;
    validate_function_effects(program, &converted, Some(module))?;
    Ok(converted)
}

/// The checks a function declared inside `module` passes, for a declaration that is never
/// applied (the native `run` tool's ad-hoc block, #129). `program` is not changed, so no
/// module is created and nothing becomes visible to a later lookup.
pub fn detached_module_function(
    program: &ast::Program,
    module: &str,
    decl: &parser::FunctionDecl,
) -> Result<ast::FunctionDecl> {
    let module_types = program
        .modules
        .iter()
        .find(|candidate| candidate.name == module)
        .map(|found| found.types.as_slice())
        .unwrap_or(&[]);
    checked_module_function(program, module_types, module, decl)
}
