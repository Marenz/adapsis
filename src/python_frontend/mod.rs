//! A deliberately non-executing Python syntax frontend for the Adapsis AST.
//! No Python interpreter, imports, source sidecars, or authority decisions live here.

mod emit;
mod expressions;
mod lower;
mod printer;
mod runtime_arguments;
mod surface;
mod tests_lower;

#[cfg(test)]
mod tests;

use rustpython_parser::{ast as py, ast::Ranged, Parse};

pub use printer::print_function;
pub use runtime_arguments::validate_runtime_arguments;
pub use tests_lower::{lower_input, lower_tests};

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
pub struct SourceSpan {
    pub operation_index: usize,
    pub start: usize,
    pub end: usize,
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone)]
pub struct LoweredSource {
    pub operations: Vec<crate::parser::Operation>,
    pub canonical_adapsis: String,
    pub source_map: Vec<SourceSpan>,
}

pub fn lower_module(source: &str, module: &str) -> Result<LoweredSource, Diagnostic> {
    lower::module(source, module)
}

/// Program-free whole-source structural preflight. Returned references must be
/// authorized before a caller supplies any runtime signatures to final lowering.
pub fn preflight_module(source: &str, module: &str) -> Result<Vec<String>, Diagnostic> {
    lower::preflight(source, module)
}

/// The caller must pass only authority-filtered runtime signatures. Native
/// dispatch does this after checking all direct source references.
pub fn lower_module_with_program(source: &str, module: &str, program: &crate::ast::Program) -> Result<LoweredSource, Diagnostic> {
    let mut signatures = std::collections::HashMap::new();
    printer::collect_signatures(&program.modules, "", &mut signatures);
    let prefix = format!("{module}.");
    let local: Vec<_> = signatures.iter().filter_map(|(name, signature)| {
        name.strip_prefix(&prefix).filter(|name| !name.contains('.')).map(|name| (name.to_string(), signature.clone()))
    }).collect();
    signatures.extend(local);
    lower::module_with_signatures(source, module, signatures, true)
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
            module: self.module.to_owned(),
            line: prefix.bytes().filter(|b| *b == b'\n').count() + 1,
            column: prefix.rsplit('\n').next().unwrap_or("").chars().count() + 1,
            message: message.into(),
        }
    }

    fn at(&self, node: &impl Ranged, message: impl Into<String>) -> Diagnostic {
        self.error(u32::from(node.start()) as usize, message)
    }

    fn parse(&self) -> Result<py::Suite, Diagnostic> {
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
        let canonical_adapsis = emit::operations(&operations).map_err(|e| self.error(0, e))?;
        let reparsed = crate::parser::parse(&canonical_adapsis).map_err(|e| {
            self.error(
                0,
                format!("canonical Adapsis cannot represent this source: {e}"),
            )
        })?;
        // Parser operations have no NodeIds. Their Debug form is a complete structural
        // representation, so this verifies docs, names, types and all expression nodes,
        // not merely that the generated source happens to parse.
        if format!("{operations:?}") != format!("{reparsed:?}") {
            return Err(self.error(
                0,
                "canonical Adapsis structural roundtrip mismatch; source rejected",
            ));
        }
        Ok(LoweredSource {
            operations,
            canonical_adapsis,
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
