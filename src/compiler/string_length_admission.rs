//! JIT safety refusal: a string used where its length is needed, in a form
//! whose length the code generator does not know.
//!
//! The generated code passes a string as (pointer, length) and takes the
//! length of every string operand from "the last string expression compiled"
//! (`_last_str_len`). Only three expression forms set that slot:
//!
//! * a string literal,
//! * `concat(...)`,
//! * a call to a user function that returns `String`.
//!
//! Any other string-valued expression (a variable, a struct field, anything
//! added later) yields a pointer and leaves the slot as it was, so the operand
//! gets the length of some other string. `concat("hello ", name)` read `name`
//! with length 6; `+return p.label` returned an empty string. The result is
//! garbage, not an error.
//!
//! One rule, as an allowlist: **wherever a string's length is consumed, the
//! operand must be one of the three forms above.** A function that breaks the
//! rule is refused before any code is emitted and runs on the VM. The
//! length-consuming positions are:
//!
//! 1. the value returned by a function whose return type is `String`;
//! 2. every argument of `concat`;
//! 3. both operands of a string `==` / `!=`. One of them must also be a
//!    literal: the code generator compares by content only when it sees a
//!    literal (`is_string_ast_expr`), otherwise it compares two pointers;
//! 4. an argument passed to a `String` parameter of a user function.
//!
//! Lift the rule only together with real length tracking in the generator.
use crate::{
    ast,
    vm::await_admission::{expression_work, statement_work, Work},
};
use std::collections::HashMap;

pub(super) const REFUSAL: &str = "jit: a string is used where the native code generator does not know its length";

/// Does `function` consume the length of a string expression that does not
/// provide one? Only this function's body is inspected; callees are judged
/// when the admission walk visits them.
pub(super) fn consumes_unknown_string_length(function: &ast::FunctionDecl, program: &ast::Program) -> bool {
    let scope = Scope { program, variables: declared_variables(function) };
    let returns_string = function.return_type == ast::Type::String;
    let mut work: Vec<Work<'_>> = function.body.iter().map(Work::Statement).collect();
    while let Some(item) = work.pop() {
        match item {
            Work::Statement(statement) => {
                if let ast::StatementKind::Return { value } = &statement.kind {
                    if returns_string && !scope.provides_length(value) {
                        return true;
                    }
                }
                statement_work(&statement.kind, &mut work);
            }
            Work::Expression(expression) => {
                if let ast::Expr::Binary { left, op: ast::BinaryOp::Equal | ast::BinaryOp::NotEqual, right } = expression {
                    if scope.is_bad_comparison(left, right) {
                        return true;
                    }
                }
                expression_work(expression, &mut work);
            }
            Work::Call(call, _) => {
                if scope.has_bad_argument(call) {
                    return true;
                }
                work.extend(call.args.iter().map(Work::Expression));
            }
            Work::Function(_) | Work::Source(_) => {}
        }
    }
    false
}

/// Declared types of the function's variables: parameters and every `+let`
/// and `+call` binding of the whole body. A name declared more than once
/// keeps all its types; it counts as a string if any of them is `String`.
fn declared_variables(function: &ast::FunctionDecl) -> HashMap<&str, Vec<&ast::Type>> {
    let mut variables: HashMap<&str, Vec<&ast::Type>> = HashMap::new();
    for param in &function.params {
        variables.entry(param.name.as_str()).or_default().push(&param.ty);
    }
    let mut work: Vec<Work<'_>> = function.body.iter().map(Work::Statement).collect();
    while let Some(item) = work.pop() {
        if let Work::Statement(statement) = item {
            match &statement.kind {
                ast::StatementKind::Let { name, ty, .. } => variables.entry(name.as_str()).or_default().push(ty),
                ast::StatementKind::Call { binding: Some(binding), .. } => {
                    variables.entry(binding.name.as_str()).or_default().push(&binding.ty)
                }
                _ => {}
            }
            // Only nested statements matter here; expressions declare nothing.
            let mut nested = Vec::new();
            statement_work(&statement.kind, &mut nested);
            work.extend(nested.into_iter().filter(|item| matches!(item, Work::Statement(_))));
        }
    }
    variables
}

struct Scope<'a> {
    program: &'a ast::Program,
    variables: HashMap<&'a str, Vec<&'a ast::Type>>,
}

impl Scope<'_> {
    /// The allowlist: the expression forms after which the generated code
    /// knows the string's length.
    ///
    /// A call to something that is not a user function (a builtin other than
    /// `concat`) is let through: the code generator has no lowering for it and
    /// fails with its own "undefined function" error, so it never emits code.
    fn provides_length(&self, expression: &ast::Expr) -> bool {
        match expression {
            ast::Expr::Literal(ast::Literal::String(_)) => true,
            ast::Expr::Call(call) => self.is_string_call(call) || self.program.get_function(&call.callee).is_none(),
            _ => false,
        }
    }

    /// `concat(...)` or a call to a user function that returns `String`.
    fn is_string_call(&self, call: &ast::CallExpr) -> bool {
        call.callee == "concat" || self.program.get_function(&call.callee).is_some_and(|callee| callee.return_type == ast::Type::String)
    }

    /// The declared type of a variable or of a struct field reached from one.
    fn types_of(&self, expression: &ast::Expr) -> Vec<ast::Type> {
        match expression {
            ast::Expr::Identifier(name) => self.variables.get(name.as_str()).into_iter().flatten().map(|ty| (*ty).clone()).collect(),
            ast::Expr::FieldAccess { base, field } => self
                .types_of(base)
                .into_iter()
                .filter_map(|base_type| match base_type {
                    ast::Type::Struct(name) => super::get_struct_field_types(self.program, &name)?
                        .into_iter()
                        .find(|(field_name, _)| field_name == field)
                        .map(|(_, field_type)| field_type),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Is the expression a string, as far as the declarations say, or would
    /// the code generator treat it as one?
    fn is_string(&self, expression: &ast::Expr) -> bool {
        matches!(expression, ast::Expr::Call(call) if self.is_string_call(call))
            || super::is_string_ast_expr(expression)
            || self.types_of(expression).contains(&ast::Type::String)
    }

    /// `==` / `!=` with a string on either side: both sides must provide
    /// their length and one must be a literal (position 3).
    fn is_bad_comparison(&self, left: &ast::Expr, right: &ast::Expr) -> bool {
        if !self.is_string(left) && !self.is_string(right) {
            return false;
        }
        let literal = |expression: &ast::Expr| matches!(expression, ast::Expr::Literal(ast::Literal::String(_)));
        !(self.provides_length(left) && self.provides_length(right) && (literal(left) || literal(right)))
    }

    /// `concat` arguments (position 2) and arguments for `String` parameters
    /// of a user function (position 4).
    fn has_bad_argument(&self, call: &ast::CallExpr) -> bool {
        if call.callee == "concat" {
            return call.args.iter().any(|argument| !self.provides_length(argument));
        }
        let Some(callee) = self.program.get_function(&call.callee) else { return false };
        callee
            .params
            .iter()
            .zip(&call.args)
            .any(|(param, argument)| param.ty == ast::Type::String && !self.provides_length(argument))
    }
}

#[cfg(test)]
#[path = "string_length_admission_tests.rs"]
mod tests;
