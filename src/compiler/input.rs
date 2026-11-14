//! What the JIT compiles from, and therefore what a compiled module is valid for.
//!
//! [`compile`](super::compile) and its admission passes look into a `Program`
//! through these lookups, and nothing else:
//!
//! | lookup | made by | answered from |
//! |---|---|---|
//! | `program.functions` (iteration) | `compile` (every top-level function is emitted), `failure_admission::check_program` (roots) | top-level functions |
//! | `Program::get_function(name)` | `failure_admission` (callee graph), `string_length_admission` (callee return and parameter types), `truthiness` (callee signature) | top-level functions; functions of top-level modules, by `Module.name` or, for a bare name no top-level function has, the first module in order that declares it |
//! | `Program::is_union_variant(name)` | `truthiness` | tagged-union declarations, top-level and of every module at any depth |
//! | `get_struct_field_types(program, name)` | `flatten_type` (signatures, parameter binding), `string_length_admission` | struct declarations, top-level and of top-level modules |
//! | `contains_result(type, program)` | `failure_admission` | type declarations, top-level and of every module at any depth |
//!
//! Module functions are never *emitted*, but they are *read*: a callee name
//! resolves into a module, and admission follows it there.
//!
//! [`CompileInput::of`] therefore builds a program with **all functions**
//! (top-level and of every module at any depth, the same `Arc`s) and **all
//! type declarations**, in the same module structure (names and order, on
//! which name resolution depends). Everything else a module holds (shared
//! variables, lifecycle blocks, routes, sources, events, docs) is left out.
//! The JIT dispatch compiles *that program*, not the live one, so the
//! compiler cannot come to depend on a part of the program that
//! [`CompileInput::matches`] does not compare. The derived lookup tables
//! (function and module indexes, union-variant set, interner) are rebuilt
//! from exactly these parts.
//!
//! The cache keeps the input next to the module compiled from it; a later
//! call may use the module only if the live program still has the same input.
//! A session revision number is not such a proof: writers do not maintain it
//! atomically with the program, some write-backs do not change it at all, and
//! a rewind reuses numbers.
use crate::ast;
use std::sync::Arc;

pub struct CompileInput {
    program: ast::Program,
}

impl CompileInput {
    pub fn of(live: &ast::Program) -> Self {
        let mut program = ast::Program::default();
        // The same `Arc`s as the live program: held here, so an address can
        // never be reused for another function while this input exists.
        program.functions = live.functions.clone();
        program.types = live.types.clone();
        program.modules = live.modules.iter().map(functions_and_types).collect();
        program.rebuild_function_index();
        Self { program }
    }

    /// The program to hand to `compile`.
    pub fn program(&self) -> &ast::Program {
        &self.program
    }

    /// Does `live` still have exactly this input? Functions are compared by
    /// identity at every depth (the write-back keeps the `Arc` of every
    /// function it does not change, see `Program::merge_changed_from`); a
    /// function rebuilt with equal content counts as different, which only
    /// costs a recompile. Type declarations have no `Arc` and are compared by
    /// value. Module names and order are part of the comparison.
    pub fn matches(&self, live: &ast::Program) -> bool {
        let compiled = &self.program;
        same_functions(&compiled.functions, &live.functions) && compiled.types == live.types && same_modules(&compiled.modules, &live.modules)
    }
}

/// A module reduced to its functions and type declarations, at every depth.
fn functions_and_types(module: &ast::Module) -> ast::Module {
    ast::Module {
        id: module.id.clone(),
        name: module.name.clone(),
        doc: None,
        types: module.types.clone(),
        functions: module.functions.clone(),
        modules: module.modules.iter().map(functions_and_types).collect(),
        shared_vars: Vec::new(),
        startup: None,
        shutdown: None,
        sources: Vec::new(),
        event_decls: Vec::new(),
        routes: Vec::new(),
        fn_index: Default::default(),
    }
}

fn same_functions(compiled: &[Arc<ast::FunctionDecl>], live: &[Arc<ast::FunctionDecl>]) -> bool {
    compiled.len() == live.len() && compiled.iter().zip(live).all(|(old, new)| Arc::ptr_eq(old, new))
}

fn same_modules(compiled: &[ast::Module], live: &[ast::Module]) -> bool {
    compiled.len() == live.len()
        && compiled.iter().zip(live).all(|(old, new)| {
            old.name == new.name
                && same_functions(&old.functions, &new.functions)
                && old.types == new.types
                && same_modules(&old.modules, &new.modules)
        })
}
