//! Ledgered "equal modulo proven qualification" for persisted Python source.
//!
//! The persistence printer may print a bare local call `f` inside module `M`
//! as `M.f` only when runtime resolution of the bare name is proven to be M's
//! own declaration (`printer::Printer::proven_local`). Reparsing then yields
//! a qualified callee. This module applies exactly that rewrite, for exactly
//! the ledgered names, to the ORIGINAL module so the round-trip gate can stay a
//! strict AST equality: any other difference still fails.
use std::collections::BTreeSet;

use crate::ast as a;

/// One rewritten call site: the enclosing member (`function`, `<shared>`,
/// `<startup>`, `<shutdown>`) and the bare callee name.
pub type Site = (String, String);

/// Returns the original module with every bare call to a ledgered name
/// qualified as `Module.name`, plus every rewritten call site in AST order.
pub(super) fn qualify_module(
    module: &a::Module,
    names: &BTreeSet<String>,
) -> (a::Module, Vec<Site>) {
    let mut module = module.clone();
    let mut sites = Vec::new();
    if names.is_empty() {
        return (module, sites);
    }
    let owner = module.name.clone();
    let mut q = Qualifier {
        owner: &owner,
        names,
        sites: &mut sites,
        member: String::new(),
    };
    q.member = "<shared>".into();
    for shared in &mut module.shared_vars {
        q.expr(&mut shared.default);
    }
    for (member, block) in [
        ("<startup>", &mut module.startup),
        ("<shutdown>", &mut module.shutdown),
    ] {
        if let Some(block) = block {
            q.member = member.into();
            q.block(&mut block.body);
        }
    }
    for function in &mut module.functions {
        let mut changed = function.as_ref().clone();
        let before = q.sites.len();
        q.member = changed.name.clone();
        q.block(&mut changed.body);
        if q.sites.len() != before {
            *function = std::sync::Arc::new(changed);
        }
    }
    (module, sites)
}

/// Number of declarations of bare `name` the runtime could pick (top-level or
/// any module). More than one means the proven binding is order-dependent.
pub(crate) fn declaration_count(program: &a::Program, name: &str) -> usize {
    program.functions.iter().filter(|f| f.name == name).count()
        + program
            .modules
            .iter()
            .filter(|m| m.functions.iter().any(|f| f.name == name))
            .count()
}

struct Qualifier<'a> {
    owner: &'a str,
    names: &'a BTreeSet<String>,
    sites: &'a mut Vec<Site>,
    member: String,
}

impl Qualifier<'_> {
    fn call(&mut self, call: &mut a::CallExpr) {
        if !call.callee.contains('.') && self.names.contains(&call.callee) {
            self.sites.push((self.member.clone(), call.callee.clone()));
            call.callee = format!("{}.{}", self.owner, call.callee);
        }
        for arg in &mut call.args {
            self.expr(arg);
        }
    }

    fn block(&mut self, body: &mut [a::Statement]) {
        for statement in body {
            self.statement(statement);
        }
    }

    fn statement(&mut self, statement: &mut a::Statement) {
        use a::StatementKind::*;
        match &mut statement.kind {
            Let { value, .. } | Set { value, .. } | Return { value } | Yield { value } => {
                self.expr(value)
            }
            Call { call, .. } | Await { call, .. } | Spawn { call, .. } => self.call(call),
            Check { condition, .. } => self.expr(condition),
            Branch {
                condition,
                then_body,
                else_body,
            } => {
                self.expr(condition);
                self.block(then_body);
                self.block(else_body);
            }
            Each { iterator, body, .. } => {
                self.expr(iterator);
                self.block(body);
            }
            While { condition, body } => {
                self.expr(condition);
                self.block(body);
            }
            Match { expr, arms } => {
                self.expr(expr);
                for arm in arms {
                    self.block(&mut arm.body);
                }
            }
            Source(
                a::SourceOp::Add { source_type, .. } | a::SourceOp::Replace { source_type, .. },
            ) => {
                if let a::SourceType::Timer(interval) = source_type {
                    self.expr(interval);
                }
            }
            Source(_) => {}
            Event(a::EventOp::Emit { value, .. }) => self.expr(value),
            Event(_) => {}
        }
    }

    fn expr(&mut self, expr: &mut a::Expr) {
        match expr {
            a::Expr::Literal(_) | a::Expr::Identifier(_) => {}
            a::Expr::FieldAccess { base, .. } => self.expr(base),
            a::Expr::Call(call) => self.call(call),
            a::Expr::Binary { left, right, .. } => {
                self.expr(left);
                self.expr(right);
            }
            a::Expr::Unary { expr, .. } => self.expr(expr),
            a::Expr::StructInit { fields, .. } => {
                for field in fields {
                    self.expr(&mut field.value);
                }
            }
        }
    }
}
