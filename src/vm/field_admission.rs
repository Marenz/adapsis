//! #114: the VM's `GetField` only reads `Value::Struct`, while the tree walker also gives
//! `FieldAccess` meaning on String/List (`len`, `is_empty`), on Result values (`is_ok`,
//! `is_err`, `unwrap`, `error`, `unwrap_err`) and transparently through `Ok(struct)`.
//! Compiling such a function for the VM is a trap: the failure only shows up at run time,
//! and after IO the async path cannot fall back (`io_performed`).
//!
//! This preflight walks the same reachable graph as `await_admission` and refuses a
//! function BEFORE any bytecode runs unless every `FieldAccess` base is statically
//! proven to be a `Value::Struct`. It is deliberately conservative: it may route a
//! struct-only function to the tree walker, it never admits an unproven base. The tree
//! walker is the reference, so VM execution semantics and the JIT are untouched.
//!
//! Proof rules (declared types are accepted only where no value flow can have produced
//! an `Ok(..)` wrapper, see the `viaok` divergence in #82):
//! - parameters and `+each` bindings with a declared `Struct` type (parameters are sound
//!   because every in-graph call site must pass a proven struct, and the root call's
//!   arguments are shape-checked by `check_arguments` before execution);
//! - `+let` with a declared `Struct` type whose VALUE is itself proven;
//! - struct literals, proven identifiers, chained fields declared `Struct`, and plain
//!   user-call expressions with a declared non-Result `Struct` return type.
//! - NOT proven: every `+call`/`+spawn`/`+await` binding, match-arm captures (which also
//!   poison a same-named parameter/local), shared variables, builtin calls, and any name
//!   whose bindings disagree.
use super::await_admission::{expression_work, statement_work, Work};
use crate::ast;
use std::collections::{HashMap, HashSet};

const REQUIRES_TREE: &str =
    "vm: pseudo-field or non-struct field access requires tree evaluation before execution";
const NON_STRUCT_ARGUMENT: &str =
    "vm: struct-typed parameter received a non-struct value; requires tree evaluation before execution";

/// Entry guard for the ROOT call. A `Struct` parameter counts as proof inside the graph
/// (call sites are checked statically, below), but the entry arguments come from outside
/// (`!eval f p=Ok({..})`, route input, tests), where the tree walker's transparent
/// `Ok(struct)` access would succeed and the VM's `GetField` would fail after IO.
pub(super) fn check_arguments(
    compiled: &super::CompiledFunction,
    arguments: &[crate::eval::Value],
) -> anyhow::Result<()> {
    let mismatch = compiled
        .struct_params
        .iter()
        .zip(arguments)
        .any(|(is_struct, value)| *is_struct && !matches!(value, crate::eval::Value::Struct(..)));
    if mismatch {
        anyhow::bail!(NON_STRUCT_ARGUMENT);
    }
    Ok(())
}

pub(super) fn check(function: &ast::FunctionDecl, program: &ast::Program) -> anyhow::Result<()> {
    if analyze(function, program) {
        anyhow::bail!(REQUIRES_TREE);
    }
    Ok(())
}

/// A proven struct, with its declaration when it is uniquely known.
type Proof<'a> = Option<&'a ast::StructDecl>;

#[derive(Clone, Copy)]
enum State<'a> {
    Struct(Proof<'a>),
    Poison,
}

/// Declarations by bare name; a name declared more than once (any kind) is ambiguous.
struct Declarations<'a> {
    by_name: HashMap<&'a str, (usize, Option<&'a ast::StructDecl>)>,
}

impl<'a> Declarations<'a> {
    fn collect(program: &'a ast::Program) -> Self {
        let mut by_name: HashMap<&str, (usize, Option<&ast::StructDecl>)> = HashMap::new();
        let mut modules: Vec<&ast::Module> = program.modules.iter().collect();
        let mut types: Vec<&ast::TypeDecl> = program.types.iter().collect();
        while let Some(module) = modules.pop() {
            types.extend(&module.types);
            modules.extend(&module.modules);
        }
        for declaration in types {
            let entry = by_name.entry(declaration.name()).or_insert((0, None));
            entry.0 += 1;
            if let ast::TypeDecl::Struct(decl) = declaration {
                entry.1 = Some(decl);
            }
        }
        Self { by_name }
    }

    fn unique_struct(&self, name: &str) -> Proof<'a> {
        match self.by_name.get(name) {
            Some((1, Some(decl))) => Some(decl),
            _ => None,
        }
    }
}

/// True when the reachable graph needs the tree walker.
fn analyze<'a>(root: &'a ast::FunctionDecl, program: &'a ast::Program) -> bool {
    let declarations = Declarations::collect(program);
    // Pointer identity only during this immutable snapshot; never dereferenced,
    // persisted or reused across Program revisions.
    let mut visited: HashSet<*const ast::FunctionDecl> = HashSet::new();
    let mut functions = vec![root];
    while let Some(function) = functions.pop() {
        if !visited.insert(function as *const _) {
            continue;
        }
        let bindings = Bindings::build(function, program, &declarations);
        let mut work: Vec<Work> = function.body.iter().map(Work::Statement).collect();
        while let Some(item) = work.pop() {
            match item {
                Work::Function(function) => functions.push(function),
                Work::Statement(statement) => statement_work(&statement.kind, &mut work),
                Work::Expression(expression) => {
                    if let ast::Expr::FieldAccess { base, .. } = expression {
                        if bindings.proof(base).is_none() {
                            return true;
                        }
                    }
                    expression_work(expression, &mut work);
                }
                Work::Call(call, _) => {
                    // The same lookup the tree and the VM use; no caller-module-first.
                    if let Some(callee) = program.get_function(&call.callee) {
                        // A callee's `Struct` parameters are trusted as proof, so every
                        // call site must pass a proven struct (a `+call` binding may hold
                        // `Ok(struct)` in the VM while the tree has unwrapped it).
                        let unproven = callee.params.iter().zip(&call.args).any(|(param, arg)| {
                            matches!(param.ty, ast::Type::Struct(_))
                                && bindings.proof(arg).is_none()
                        });
                        if unproven {
                            return true;
                        }
                        functions.push(callee);
                    }
                    work.extend(call.args.iter().map(Work::Expression));
                }
                Work::Source(source) => {
                    if let ast::SourceType::Timer(expression) = source {
                        work.push(Work::Expression(expression));
                    }
                }
            }
        }
    }
    false
}

/// Flow-insensitive binding states of one function, to a fixpoint (poison only grows).
struct Bindings<'a, 'd> {
    states: HashMap<&'a str, State<'a>>,
    program: &'a ast::Program,
    declarations: &'d Declarations<'a>,
}

impl<'a, 'd> Bindings<'a, 'd> {
    fn build(
        function: &'a ast::FunctionDecl,
        program: &'a ast::Program,
        declarations: &'d Declarations<'a>,
    ) -> Self {
        let mut poisoned: HashSet<&'a str> = HashSet::new();
        loop {
            let mut this = Bindings {
                states: HashMap::new(),
                program,
                declarations,
            };
            for name in &poisoned {
                this.states.insert(name, State::Poison);
            }
            for param in &function.params {
                this.bind_declared(&param.name, &param.ty);
            }
            this.bind_body(&function.body);
            let now: HashSet<&str> = this
                .states
                .iter()
                .filter(|(_, state)| matches!(state, State::Poison))
                .map(|(name, _)| *name)
                .collect();
            if now.len() == poisoned.len() {
                return this;
            }
            poisoned = now;
        }
    }

    fn bind_body(&mut self, body: &'a [ast::Statement]) {
        use ast::StatementKind as S;
        // Source order, nested bodies inline (iterative).
        let mut stack: Vec<&'a ast::Statement> = body.iter().rev().collect();
        while let Some(statement) = stack.pop() {
            let mut nested: Vec<&'a [ast::Statement]> = Vec::new();
            match &statement.kind {
                S::Let { name, ty, value } => match ty {
                    ast::Type::Struct(_) if self.proof(value).is_some() => {
                        self.bind_declared(name, ty)
                    }
                    _ => self.poison(name),
                },
                S::Call { binding, .. } | S::Spawn { binding, .. } => {
                    if let Some(binding) = binding {
                        self.poison(&binding.name);
                    }
                }
                S::Await { name, .. } => self.poison(name),
                S::Each { binding, body, .. } => {
                    self.bind_declared(&binding.name, &binding.ty);
                    nested.push(body);
                }
                S::Set { name, value } => {
                    let bound = matches!(self.states.get(name.as_str()), Some(State::Struct(_)));
                    if !(bound && self.proof(value).is_some()) {
                        self.poison(name);
                    }
                }
                S::Branch {
                    then_body,
                    else_body,
                    ..
                } => {
                    nested.extend([then_body.as_slice(), else_body.as_slice()]);
                }
                S::While { body, .. } => nested.push(body),
                S::Match { arms, .. } => {
                    for arm in arms {
                        arm.bindings.iter().for_each(|name| self.poison(name));
                        for pattern in arm.patterns.iter().flatten() {
                            self.poison_pattern(pattern);
                        }
                        nested.push(&arm.body);
                    }
                }
                S::Return { .. }
                | S::Check { .. }
                | S::Yield { .. }
                | S::Source(_)
                | S::Event(_) => {}
            }
            for body in nested.into_iter().rev() {
                stack.extend(body.iter().rev());
            }
        }
    }

    fn poison_pattern(&mut self, pattern: &'a ast::MatchPattern) {
        let mut stack = vec![pattern];
        while let Some(pattern) = stack.pop() {
            match pattern {
                ast::MatchPattern::Binding(name) => self.poison(name),
                ast::MatchPattern::Variant { sub_patterns, .. } => stack.extend(sub_patterns),
                ast::MatchPattern::Literal(_) => {}
            }
        }
    }

    fn poison(&mut self, name: &'a str) {
        self.states.insert(name, State::Poison);
    }

    /// Bind `name` to a declared type: only a `Struct` type can be proven, and a second
    /// binding must agree with the first.
    fn bind_declared(&mut self, name: &'a str, ty: &'a ast::Type) {
        let ast::Type::Struct(type_name) = ty else {
            return self.poison(name);
        };
        let decl = self.declarations.unique_struct(type_name);
        let next = match self.states.get(name) {
            None => State::Struct(decl),
            Some(State::Struct(existing)) if same(*existing, decl) => State::Struct(decl),
            Some(_) => State::Poison,
        };
        self.states.insert(name, next);
    }

    /// `Some(decl)` when `expr` is statically a `Value::Struct`. Iterative over field chains.
    fn proof(&self, expr: &'a ast::Expr) -> Option<Proof<'a>> {
        let mut fields: Vec<&str> = Vec::new();
        let mut root = expr;
        while let ast::Expr::FieldAccess { base, field } = root {
            fields.push(field);
            root = base;
        }
        let mut current: Proof<'a> = match root {
            ast::Expr::StructInit { ty, .. } => self.declarations.unique_struct(ty),
            ast::Expr::Identifier(name) => match self.states.get(name.as_str()) {
                Some(State::Struct(decl)) => *decl,
                _ => return None,
            },
            ast::Expr::Call(call) => {
                // A plain user call with a declared, non-Result Struct return type.
                let function = self.program.get_function(&call.callee)?;
                match &function.return_type {
                    ast::Type::Struct(name) => self.declarations.unique_struct(name),
                    _ => return None,
                }
            }
            _ => return None,
        };
        // Each chained field must be declared, and declared `Struct`.
        for field in fields.into_iter().rev() {
            let declared = current?.fields.iter().find(|f| f.name == field)?;
            let ast::Type::Struct(name) = &declared.ty else {
                return None;
            };
            current = self.declarations.unique_struct(name);
        }
        Some(current)
    }
}

fn same(a: Proof<'_>, b: Proof<'_>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => std::ptr::eq(a, b),
        _ => false,
    }
}

#[cfg(test)]
mod tests;
