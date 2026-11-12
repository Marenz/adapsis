//! A deliberately non-executing Python syntax frontend for the Adapsis AST.
//! No Python interpreter, imports, source sidecars, or authority decisions live here.

#[allow(dead_code)] // Explicit legacy exporter/oracle; normal Python paths do not call it.
mod emit;
mod ir_surface;
pub(crate) use ir_surface::validate as validate_operation_ir;
mod ast_equality;
mod expressions;
mod lower;
mod nesting;
#[cfg(test)]
mod parity;
mod module_source;
mod printer;
mod qualification;
pub(crate) use qualification::declaration_count;
mod runtime_arguments;
mod call_sites;
mod builtin_signatures;
mod surface;
mod pseudo_fields;
mod result_returns;
mod shared_default;
mod references;
mod tests_lower;
#[cfg(test)]
mod top_level;
#[cfg(test)]
mod top_level_tests;
#[cfg(test)]
mod top_level_function_tests;
#[cfg(test)]
mod input_binding_tests;
#[cfg(test)]
mod pseudo_field_tests;
mod tests_print;
#[cfg(test)]
mod tests_print_tests;
#[cfg(test)]
mod module_source_tests;
#[cfg(test)]
mod qualification_tests;
#[cfg(test)]
mod stored_test_data_tests;
mod types;
mod type_bindings;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod types_tests;
#[cfg(test)]
mod match_tests;
#[cfg(test)]
mod module_tests;
#[cfg(test)]
mod review_tests;
#[cfg(test)]
mod task4_tests;
#[cfg(test)]
mod task5_tests;
#[cfg(test)]
mod admission_regression_tests;
#[cfg(test)]
mod runtime_semantics_tests;
#[cfg(test)]
mod call_site_tests;
#[cfg(test)]
mod overflow_tests;
#[cfg(test)]
mod builtin_signature_tests;
#[cfg(test)]
mod json_kind_tests;
#[cfg(test)]
mod json_valid_tests;
#[cfg(test)]
mod logical_tier_tests;
#[cfg(test)]
mod condition_tier_tests;
#[cfg(test)]
mod telegram_port_tests;
#[cfg(test)]
mod task_domain_tests;
#[cfg(test)]
mod block_tests;

use rustpython_parser::{ast as py, ast::Ranged, Parse};

pub use printer::print_function;
#[cfg(test)]
pub use printer::print_module;
pub use printer::print_type;
pub use module_source::{lower_python_module, print_python_module, print_python_module_with_ledger, LoweredPythonModule, QualificationLedger};
pub use runtime_arguments::validate_runtime_arguments;
pub use runtime_arguments::{validate_await_contract_from, validate_entry_await_contract};
pub use references::{NativeReferences, resolve_native_references, retain_native_dependencies, check_constructor_environment, check_constructor_environment_from};
pub use lower::Preflight as NativePreflight;
pub(crate) use references::COLLISION as CONSTRUCTOR_ROLE_ERROR;
pub(crate) use call_sites::runtime_shadowed;
pub use type_bindings::check_native_type_bindings;
#[cfg(test)]
pub(crate) use nesting::MAX_NESTING;
#[allow(unused_imports)] // Kept as the explicit legacy input export adapter.
pub use tests_lower::lower_input;
pub use tests_lower::{lower_input_expr, lower_tests};
#[cfg(test)]
pub(crate) use tests_lower::lower_tests_toplevel;
/// Test-only: module-less (top-level) lowering/printing. See `top_level.rs`.
#[cfg(test)]
pub(crate) fn lower_toplevel(source: &str, program: &crate::ast::Program) -> Result<LoweredSource, Diagnostic> {
    top_level::lower(source, program)
}
#[cfg(test)]
pub(crate) use printer::{print_toplevel_function, print_toplevel_type};

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub module: String,
    pub line: usize,
    pub column: usize,
    pub message: String,
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}:{}: {}",
            self.module, self.line, self.column, self.message
        )
    }
}

impl std::error::Error for Diagnostic {}

#[derive(Debug, Clone)]
#[allow(dead_code)] // Source diagnostics; consumed by tests and future IDE tooling.
pub struct SourceSpan {
    pub operation_index: usize,
    pub start: usize,
    pub end: usize,
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // `source_map` is diagnostic metadata.
pub struct LoweredSource {
    pub operations: Vec<crate::parser::Operation>,
    pub source_map: Vec<SourceSpan>,
}

impl LoweredSource {
    /// Optional legacy export and transition oracle, never a typed execution API.
    #[allow(dead_code)]
    pub fn legacy_export(&self) -> Result<String, Diagnostic> {
        let exported = emit::operations(&self.operations).map_err(|message| Diagnostic {
            module: "<export>".into(), line: 1, column: 1, message,
        })?;
        let parsed = crate::parser::parse(&exported).map_err(|_| Diagnostic {
            module: "<export>".into(), line: 1, column: 1, message: "legacy export cannot be reparsed".into(),
        })?;
        if format!("{:?}", self.operations) != format!("{parsed:?}") {
            return Err(Diagnostic { module: "<export>".into(), line: 1, column: 1, message: "legacy export structural mismatch".into() });
        }
        Ok(exported)
    }
}

pub fn lower_module(source: &str, module: &str) -> Result<LoweredSource, Diagnostic> {
    lower::module(source, module)
}

/// Program-free whole-source structural preflight. Returned references must be
/// authorized before a caller supplies any runtime signatures to final lowering.
pub fn preflight_module(source: &str, module: &str) -> Result<lower::Preflight, Diagnostic> {
    lower::preflight(source, module)
}

/// The caller must pass only authority-filtered runtime signatures. Native
/// dispatch does this after checking all direct source references.
pub fn lower_module_with_program(source: &str, module: &str, program: &crate::ast::Program) -> Result<LoweredSource, Diagnostic> {
    let signatures = scope_signatures(program, module);
    let types = types::catalog_from_program(program);
    let shared = types::module_shared_vars(program, module);
    lower::module_with_signatures(source, module, signatures, &types, &shared, true)
}

/// Qualified signatures of the supplied program plus bare spellings for `module`'s own
/// functions: the name resolution a declaration inside `module` gets.
fn scope_signatures(program: &crate::ast::Program, module: &str) -> std::collections::HashMap<String, expressions::Signature> {
    let mut signatures = std::collections::HashMap::new();
    printer::collect_signatures(&program.modules, "", &mut signatures);
    let prefix = format!("{module}.");
    let local: Vec<_> = signatures.iter().filter_map(|(name, signature)| {
        name.strip_prefix(&prefix).filter(|name| !name.contains('.')).map(|name| (name.to_string(), signature.clone()))
    }).collect();
    signatures.extend(local);
    signatures
}

/// Program-free structural preflight of an ad-hoc statement block (#129). `scope` is the
/// module whose names resolve bare; the same authorization contract as
/// [`preflight_module`] applies to the returned references.
pub fn preflight_block(source: &str, scope: &str) -> Result<lower::Preflight, Diagnostic> {
    lower::block::preflight(source, scope)
}

/// Lower an ad-hoc statement block as the body of ONE parameterless function `function`
/// inside `scope`. The caller must pass only authority-filtered runtime signatures, and
/// must never apply the returned declaration to a Program: it exists to be checked and
/// executed once.
pub fn lower_block_with_program(source: &str, scope: &str, function: &str, program: &crate::ast::Program) -> Result<LoweredSource, Diagnostic> {
    let types = types::catalog_from_program(program);
    let shared = types::module_shared_vars(program, scope);
    lower::block::lower(source, scope, function, scope_signatures(program, scope), &types, &shared)
}

struct Context<'a> {
    source: &'a str,
    module: &'a str,
}

impl<'a> Context<'a> {
    fn error(&self, offset: usize, message: impl Into<String>) -> Diagnostic {
        let offset = offset.min(self.source.len());
        let prefix = self.source.get(..offset).unwrap_or("");
        Diagnostic {
            module: if self.module.is_empty() {
                "<top-level>".to_owned()
            } else {
                self.module.to_owned()
            },
            line: prefix.bytes().filter(|b| *b == b'\n').count() + 1,
            column: prefix.rsplit('\n').next().unwrap_or("").chars().count() + 1,
            message: message.into(),
        }
    }

    fn at(&self, node: &impl Ranged, message: impl Into<String>) -> Diagnostic {
        self.error(u32::from(node.start()) as usize, message)
    }

    /// The ONLY place Python source reaches the parser. The nesting guard runs
    /// first: the parser's tree, lowering and everything behind it recurse on
    /// nesting, and a stack overflow aborts the process (#133).
    fn parse(&self) -> Result<py::Suite, Diagnostic> {
        nesting::check(self.source).map_err(|deep| self.error(deep.offset, deep.message))?;
        py::Suite::parse(self.source, self.module).map_err(|e| {
            self.error(
                u32::from(e.offset) as usize,
                format!("Python syntax: {}", e.error),
            )
        })
    }

    fn span(&self, node: &impl Ranged, operation_index: usize) -> SourceSpan {
        let start = u32::from(node.start()) as usize;
        let location = self.error(start, "");
        SourceSpan {
            operation_index,
            start,
            end: u32::from(node.end()) as usize,
            line: location.line,
            column: location.column,
        }
    }

    fn finish(
        &self,
        operations: Vec<crate::parser::Operation>,
        source_map: Vec<SourceSpan>,
    ) -> Result<LoweredSource, Diagnostic> {
        // Submission-level location only: current spans do not identify nested IR
        // nodes. The guard supplies an operation path, never fabricated .ax lines.
        ir_surface::validate(&operations)
            .map_err(|error| self.error(0, format!("Python operation IR: {error}")))?;
        Ok(LoweredSource {
            operations,
            source_map,
        })
    }
}

fn identifier(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !name.contains("__")
        && !matches!(name, "true" | "false" | "AND" | "OR" | "NOT" | "as")
        && !matches!(name, "False" | "None" | "True" | "and" | "assert" | "async" | "await"
            | "break" | "class" | "continue" | "def" | "del" | "elif" | "else" | "except"
            | "finally" | "for" | "from" | "global" | "if" | "import" | "in" | "is" | "lambda"
            | "nonlocal" | "not" | "or" | "pass" | "raise" | "return" | "try" | "while" | "with" | "yield")
}

fn reserved_positional_name(name: &str) -> bool {
    name.strip_prefix('_').is_some_and(|suffix| !suffix.is_empty() && suffix.chars().all(|character| character.is_ascii_digit()))
}

/// Names that may not be BOUND (parameter, local, match capture, value or field-access
/// base): the full `forbidden` list minus `input`. `input` is Python's stdin builtin, not
/// an execution escape, and is a perfectly ordinary Adapsis identifier; as a value name
/// it can only shadow the builtin, which the non-executing frontend never calls. It stays
/// in `forbidden` for every other role: a callee (`input(x)` must never lower), function,
/// type, member, shared-variable, annotation and attribute names.
fn forbidden_binding(name: &str) -> bool {
    name != "input" && forbidden(name)
}

fn forbidden(name: &str) -> bool {
    matches!(
        name,
        "eval"
            | "exec"
            | "open"
            | "getattr"
            | "setattr"
            | "delattr"
            | "globals"
            | "locals"
            | "compile"
            | "vars"
            | "input"
            | "import"
    ) || name.contains("__")
}

/// Frontend-only reserved helper names. They may not be shadowed by a user
/// function, parameter or local binding, but they are not `forbidden` (which
/// `direct_name`/`identifier` reject as unsafe).
pub(super) fn reserved_helper_name(name: &str) -> bool {
    matches!(
        name,
        "propagate"
            | "check"
            | "spawn"
            | "source_add"
            | "source_remove"
            | "source_replace"
            | "source_list"
            | "event_register"
            | "event_emit"
            | "is_ok"
            | "is_err"
            | "assert_after"
    )
}

fn docstring(stmt: &py::Stmt) -> Option<&str> {
    match stmt {
        py::Stmt::Expr(e) => match e.value.as_ref() {
            py::Expr::Constant(c) => match &c.value {
                py::Constant::Str(s) => Some(s),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }
}
