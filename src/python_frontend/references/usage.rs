//! Sequential source-only value scopes. Syntax admission runs before this pass;
//! no runtime signature or hidden declaration is consulted here.
use crate::python_frontend::{Context, Diagnostic};
use rustpython_parser::ast as py;
use std::collections::HashSet;

#[derive(Default)]
pub(in crate::python_frontend) struct Usage {
    pub values: HashSet<String>,
    pub calls: HashSet<String>,
    pub namespaces: HashSet<String>,
}

pub(in crate::python_frontend) fn collect(
    cx: &Context<'_>,
    suite: &[py::Stmt],
) -> Result<Usage, Diagnostic> {
    let shared: HashSet<String> = suite
        .iter()
        .filter_map(|stmt| match stmt {
            py::Stmt::AnnAssign(a) => match a.target.as_ref() {
                py::Expr::Name(n) => Some(n.id.to_string()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    let mut usage = Usage::default();
    for stmt in suite {
        let (args, body) = match stmt {
            py::Stmt::FunctionDef(f) => (&f.args, &f.body),
            py::Stmt::AsyncFunctionDef(f) => (&f.args, &f.body),
            _ => continue, // Shared default validation is a separate slice.
        };
        let mut scope = shared.clone();
        scope.extend(args.args.iter().map(|a| a.def.arg.to_string()));
        usage.block(cx, body, &mut scope)?;
    }
    Ok(usage)
}

impl Usage {
    fn block(
        &mut self,
        cx: &Context<'_>,
        body: &[py::Stmt],
        scope: &mut HashSet<String>,
    ) -> Result<(), Diagnostic> {
        for stmt in body {
            match stmt {
                py::Stmt::AnnAssign(a) => {
                    if let Some(v) = &a.value {
                        self.expr(cx, v, scope)?;
                    }
                    bind(&a.target, scope);
                }
                py::Stmt::Assign(a) => {
                    self.expr(cx, &a.value, scope)?;
                    for t in &a.targets {
                        bind(t, scope);
                    }
                }
                py::Stmt::Return(r) => {
                    if let Some(v) = &r.value {
                        self.expr(cx, v, scope)?;
                    }
                }
                py::Stmt::Expr(e) => self.expr(cx, &e.value, scope)?,
                py::Stmt::If(v) => {
                    self.expr(cx, &v.test, scope)?;
                    self.block(cx, &v.body, &mut scope.clone())?;
                    self.block(cx, &v.orelse, &mut scope.clone())?;
                }
                py::Stmt::While(v) => {
                    self.expr(cx, &v.test, scope)?;
                    self.block(cx, &v.body, &mut scope.clone())?;
                }
                py::Stmt::For(v) => {
                    self.expr(cx, &v.iter, scope)?;
                    let mut inner = scope.clone();
                    bind(&v.target, &mut inner);
                    self.block(cx, &v.body, &mut inner)?;
                }
                py::Stmt::Match(v) => {
                    self.expr(cx, &v.subject, scope)?;
                    for case in &v.cases {
                        let mut inner = scope.clone();
                        captures(&case.pattern, &mut inner);
                        self.block(cx, &case.body, &mut inner)?;
                    }
                }
                _ => {} // Closed shape validation is owned by surface.rs.
            }
        }
        Ok(())
    }

    fn expr(
        &mut self,
        cx: &Context<'_>,
        e: &py::Expr,
        scope: &HashSet<String>,
    ) -> Result<(), Diagnostic> {
        match e {
            py::Expr::Name(n) if !scope.contains(n.id.as_str()) => {
                self.values.insert(n.id.to_string());
            }
            py::Expr::Constant(c) if matches!(c.value, py::Constant::None) => {
                self.values.insert("None".into());
            }
            py::Expr::Call(c) => {
                let name = path(&c.func)
                    .ok_or_else(|| cx.at(c, "Python dynamic call targets are unsupported"))?;
                let root = name.split('.').next().unwrap();
                if scope.contains(root) {
                    return Err(cx.at(
                        c,
                        "Python callable variables/dynamic attributes are unsupported",
                    ));
                }
                if !name.contains('.')
                    && c.keywords.is_empty()
                    && !crate::python_frontend::reserved_helper_name(&name)
                {
                    self.calls.insert(name);
                }
                for arg in &c.args {
                    self.expr(cx, arg, scope)?;
                }
                for arg in &c.keywords {
                    self.expr(cx, &arg.value, scope)?;
                }
            }
            py::Expr::Attribute(a) => {
                let name = path(e).ok_or_else(|| cx.at(a, "unsupported field access base"))?;
                if !scope.contains(name.split('.').next().unwrap()) {
                    self.namespaces.insert(name);
                }
            }
            py::Expr::Await(a) => self.expr(cx, &a.value, scope)?,
            py::Expr::List(l) => {
                for v in &l.elts {
                    self.expr(cx, v, scope)?;
                }
            }
            py::Expr::BinOp(b) => {
                self.expr(cx, &b.left, scope)?;
                self.expr(cx, &b.right, scope)?;
            }
            py::Expr::UnaryOp(u) => self.expr(cx, &u.operand, scope)?,
            py::Expr::BoolOp(b) => {
                for v in &b.values {
                    self.expr(cx, v, scope)?;
                }
            }
            py::Expr::Compare(c) => {
                self.expr(cx, &c.left, scope)?;
                for v in &c.comparators {
                    self.expr(cx, v, scope)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

fn bind(e: &py::Expr, scope: &mut HashSet<String>) {
    if let py::Expr::Name(n) = e {
        scope.insert(n.id.to_string());
    }
}
fn captures(p: &py::Pattern, scope: &mut HashSet<String>) {
    match p {
        py::Pattern::MatchAs(a) => {
            if let Some(n) = &a.name {
                scope.insert(n.to_string());
            }
            if let Some(p) = &a.pattern {
                captures(p, scope);
            }
        }
        py::Pattern::MatchClass(c) => {
            for p in &c.patterns {
                captures(p, scope);
            }
        }
        _ => {}
    }
}
fn path(e: &py::Expr) -> Option<String> {
    match e {
        py::Expr::Name(n) => Some(n.id.to_string()),
        py::Expr::Attribute(a) => Some(format!("{}.{}", path(&a.value)?, a.attr)),
        _ => None,
    }
}
