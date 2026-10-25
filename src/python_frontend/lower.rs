use super::expressions::{Expressions, Signature, Ty, annotation, compatible};
use super::{Context, Diagnostic, LoweredSource};
use crate::parser as p;
use rustpython_parser::ast as py;
use std::collections::{HashMap, HashSet};

pub(super) fn module(source: &str, module: &str) -> Result<LoweredSource, Diagnostic> {
    module_with_signatures(source, module, HashMap::new(), false)
}

pub(super) fn module_with_signatures(
    source: &str,
    module: &str,
    mut signatures: HashMap<String, Signature>,
    require_resolved: bool,
) -> Result<LoweredSource, Diagnostic> {
    let cx = Context { source, module };
    if !super::identifier(module) {
        return Err(cx.error(
            0,
            "module must be a simple ASCII identifier without dunders",
        ));
    }
    let suite = cx.parse()?;
    let mut defs = Vec::new();
    let mut doc = None;
    for (i, stmt) in suite.iter().enumerate() {
        if i == 0 {
            if let Some(text) = super::docstring(stmt) {
                doc = Some(text.to_owned());
                continue;
            }
        }
        defs.push(Definition::from_stmt(&cx, stmt)?);
    }
    if defs.is_empty() {
        return Err(cx.error(
            0,
            "module submission requires at least one annotated function",
        ));
    }
    let mut declared = HashSet::new();
    for def in &defs {
        let signature = def.signature(&cx)?;
        if !declared.insert(def.name.to_owned()) {
            return Err(cx.at(
                def.stmt,
                format!("duplicate function {}; submit it only once", def.name),
            ));
        }
        signatures.insert(format!("{module}.{}", def.name), signature.clone());
        signatures.insert(def.name.to_owned(), signature);
    }
    let mut body = Vec::new();
    if require_resolved {
        for reference in preflight(source, module)? {
            if !signatures.contains_key(&reference) {
                return Err(cx.error(0, format!("Python callable `{reference}` has no authorized supported signature; declare the target first")));
            }
        }
    }
    for def in &defs {
        body.push(p::Operation::Function(def.lower(&cx, &signatures)?));
    }
    let operations = vec![p::Operation::Module(p::ModuleDecl {
        name: module.into(),
        body,
        doc,
    })];
    let source_map = suite.iter().map(|stmt| cx.span(stmt, 0)).collect();
    cx.finish(operations, source_map)
}

struct Definition<'a> {
    stmt: &'a py::Stmt,
    name: &'a str,
    args: &'a py::Arguments,
    returns: Option<&'a py::Expr>,
    body: &'a [py::Stmt],
    decorators: &'a [py::Expr],
    asynchronous: bool,
}

pub(super) fn preflight(source: &str, module: &str) -> Result<Vec<String>, Diagnostic> {
    let cx = Context { source, module };
    if !super::identifier(module) {
        return Err(cx.error(
            0,
            "module must be a simple ASCII identifier without dunders",
        ));
    }
    let suite = cx.parse()?;
    let mut names = HashSet::new();
    let mut references = Vec::new();
    for (index, stmt) in suite.iter().enumerate() {
        if index == 0 && super::docstring(stmt).is_some() {
            continue;
        }
        let def = Definition::from_stmt(&cx, stmt)?;
        def.signature(&cx)?;
        if !names.insert(def.name) {
            return Err(cx.at(stmt, "duplicate function declaration"));
        }
        super::surface::statements(&cx, def.body, true, &mut references)?;
    }
    if names.is_empty() {
        return Err(cx.error(
            0,
            "module submission requires at least one annotated function",
        ));
    }
    references.sort();
    references.dedup();
    Ok(references)
}

impl<'a> Definition<'a> {
    fn from_stmt(cx: &Context<'_>, stmt: &'a py::Stmt) -> Result<Self, Diagnostic> {
        let value = match stmt {
            py::Stmt::FunctionDef(f) if f.type_params.is_empty() && f.type_comment.is_none() => Self {
                stmt, name: f.name.as_str(), args: &f.args, returns: f.returns.as_deref(),
                body: &f.body, decorators: &f.decorator_list, asynchronous: false,
            },
            py::Stmt::AsyncFunctionDef(f) if f.type_params.is_empty() && f.type_comment.is_none() => Self {
                stmt, name: f.name.as_str(), args: &f.args, returns: f.returns.as_deref(),
                body: &f.body, decorators: &f.decorator_list, asynchronous: true,
            },
            _ => return Err(cx.at(stmt, "unsupported module statement; only an initial docstring and annotated def/async def declarations are allowed")),
        };
        if !super::identifier(value.name)
            || super::forbidden(value.name)
            || crate::builtins::is_builtin(value.name)
            || matches!(
                value.name,
                "effects" | "int" | "float" | "bool" | "str" | "Result"
            )
        {
            return Err(cx.at(
                stmt,
                format!(
                    "reserved/unsafe function name {}; builtin and syntax names cannot be shadowed",
                    value.name
                ),
            ));
        }
        Ok(value)
    }

    fn signature(&self, cx: &Context<'_>) -> Result<Signature, Diagnostic> {
        ordinary_args(cx, self.stmt, self.args)?;
        let mut params = Vec::new();
        for arg in &self.args.args {
            let ann =
                arg.def.annotation.as_deref().ok_or_else(|| {
                    cx.at(&arg.def, "parameter requires an explicit type annotation")
                })?;
            params.push(annotation(cx, ann)?);
        }
        let returns = self.returns.ok_or_else(|| {
            cx.at(
                self.stmt,
                "function requires an explicit return annotation; None/unit is not supported",
            )
        })?;
        let result = annotation(cx, returns)?;
        let effects = effects(cx, self.decorators, self.asynchronous)?;
        Ok(Signature {
            params,
            result,
            effects,
        })
    }

    fn lower(
        &self,
        cx: &Context<'_>,
        signatures: &HashMap<String, Signature>,
    ) -> Result<p::FunctionDecl, Diagnostic> {
        let sig = &signatures[self.name];
        let mut lower = Lower {
            cx,
            signatures,
            env: HashMap::new(),
            seen: HashSet::new(),
            result: sig.result.clone(),
            effects: &sig.effects,
        };
        let mut params = Vec::new();
        for (arg, ty) in self.args.args.iter().zip(&sig.params) {
            lower.bind(arg.def.arg.as_str(), ty.clone(), &arg.def)?;
            params.push(p::Param {
                name: arg.def.arg.to_string(),
                ty: ty.parser(),
            });
        }
        let (doc, stmts) = match self.body.first().and_then(super::docstring) {
            Some(text) => (Some(text.to_owned()), &self.body[1..]),
            None => (None, self.body),
        };
        let body = lower.block(stmts)?;
        if !returns(&body) {
            return Err(cx.at(self.stmt, "function may fall through to Python None; provide an explicit value return on every path"));
        }
        Ok(p::FunctionDecl {
            name: self.name.into(),
            params,
            return_type: sig.result.parser(),
            effects: sig.effects.clone(),
            body,
            doc,
        })
    }
}

pub(super) fn ordinary_args(
    cx: &Context<'_>,
    node: &impl py::Ranged,
    args: &py::Arguments,
) -> Result<(), Diagnostic> {
    if let Some(arg) = args
        .args
        .iter()
        .find(|arg| super::reserved_positional_name(arg.def.arg.as_str()))
    {
        return Err(cx.at(
            &arg.def,
            "parameter names _0, _1, ... are reserved for runtime positional arguments",
        ));
    }
    if let Some(arg) = args.args.iter().find(|arg| {
        !super::identifier(arg.def.arg.as_str()) || super::forbidden(arg.def.arg.as_str())
    }) {
        return Err(cx.at(&arg.def, "unsupported/unsafe parameter identifier"));
    }
    if !args.posonlyargs.is_empty()
        || !args.kwonlyargs.is_empty()
        || args.vararg.is_some()
        || args.kwarg.is_some()
        || args.args.iter().any(|a| a.default.is_some())
    {
        return Err(cx.at(
            node,
            "default, positional-only, keyword-only, *args and **kwargs parameters are unsupported",
        ));
    }
    Ok(())
}

fn effects(
    cx: &Context<'_>,
    decorators: &[py::Expr],
    asynchronous: bool,
) -> Result<Vec<String>, Diagnostic> {
    let mut out = Vec::new();
    for decorator in decorators {
        let py::Expr::Call(c) = decorator else {
            return Err(cx.at(
                decorator,
                "only inert @effects('io', 'fail', 'mut') decorators are supported",
            ));
        };
        if !matches!(c.func.as_ref(), py::Expr::Name(n) if n.id.as_str() == "effects")
            || !c.keywords.is_empty()
        {
            return Err(cx.at(
                decorator,
                "only inert @effects('io', 'fail', 'mut') decorators are supported",
            ));
        }
        for arg in &c.args {
            let py::Expr::Constant(v) = arg else {
                return Err(cx.at(arg, "effect must be a literal string"));
            };
            let py::Constant::Str(effect) = &v.value else {
                return Err(cx.at(arg, "effect must be a literal string"));
            };
            if !matches!(effect.as_str(), "io" | "fail" | "mut") || out.contains(effect) {
                return Err(cx.at(arg, "unsupported or duplicate effect; allowed: io, fail, mut (async comes from async def)"));
            }
            out.push(effect.clone());
        }
    }
    if asynchronous {
        out.push("async".into());
    }
    Ok(out)
}

struct Lower<'a, 'b> {
    cx: &'a Context<'b>,
    signatures: &'a HashMap<String, Signature>,
    env: HashMap<String, Ty>,
    seen: HashSet<String>,
    result: Ty,
    effects: &'a [String],
}

impl Lower<'_, '_> {
    fn expressions(&self) -> Expressions<'_, '_> {
        Expressions {
            cx: self.cx,
            env: &self.env,
            signatures: self.signatures,
        }
    }

    fn bind(&mut self, name: &str, ty: Ty, node: &impl py::Ranged) -> Result<(), Diagnostic> {
        if !super::identifier(name)
            || super::forbidden(name)
            || crate::builtins::is_builtin(name)
            || self.signatures.contains_key(name)
            || name == self.cx.module
            || self.seen.contains(name)
            || matches!(
                name,
                "int" | "float" | "bool" | "str" | "Result" | "effects"
            )
        {
            return Err(self.cx.at(node, format!("binding {name} shadows/reuses a reserved, callable, or previously scoped name; choose a fresh name")));
        }
        self.seen.insert(name.to_owned());
        self.env.insert(name.to_owned(), ty);
        Ok(())
    }

    fn block(&mut self, body: &[py::Stmt]) -> Result<Vec<p::Operation>, Diagnostic> {
        let original = self.env.clone();
        let result = body.iter().map(|s| self.statement(s)).collect();
        self.env = original;
        result
    }

    fn statement(&mut self, s: &py::Stmt) -> Result<p::Operation, Diagnostic> {
        match s {
            py::Stmt::AnnAssign(a) => self.annotated(a),
            py::Stmt::Assign(a) => self.assign(a),
            py::Stmt::Return(r) => {
                let value = r.value.as_deref().ok_or_else(|| self.cx.at(s, "bare return is unsupported; runtime has no None/unit value"))?;
                let (expr, ty) = self.expressions().lower(value)?;
                self.check_type(&self.result, &ty, value)?;
                Ok(p::Operation::Return(p::ReturnDecl { expr }))
            }
            py::Stmt::If(v) => self.if_statement(v),
            py::Stmt::While(v) => {
                if !v.orelse.is_empty() { return Err(self.cx.at(s, "while-else is unsupported")); }
                let condition = self.condition(&v.test)?;
                let body = self.block(&v.body)?;
                Ok(p::Operation::While(p::WhileDecl { condition, body }))
            }
            py::Stmt::For(v) => self.each(v),
            _ => Err(self.cx.at(s, format!("unsupported statement {}; use annotated bindings, assignment, return, if, while, or fresh-target for", statement_kind(s)))),
        }
    }

    fn annotated(&mut self, a: &py::StmtAnnAssign) -> Result<p::Operation, Diagnostic> {
        let py::Expr::Name(n) = a.target.as_ref() else {
            return Err(self
                .cx
                .at(a, "annotated binding target must be a simple name"));
        };
        let ty = annotation(self.cx, &a.annotation)?;
        let value = a
            .value
            .as_deref()
            .ok_or_else(|| self.cx.at(a, "annotated binding requires an initial value"))?;
        let awaited = matches!(value, py::Expr::Await(_));
        let (expr, actual) = if let py::Expr::Await(v) = value {
            if !self.effects.iter().any(|effect| effect == "async") {
                return Err(self.cx.at(value, "await requires async def"));
            }
            let py::Expr::Call(call) = v.value.as_ref() else {
                return Err(self.cx.at(value, "await requires a direct function call"));
            };
            if self.expressions().requires_io(call)?
                && !self.effects.iter().any(|effect| effect == "io")
            {
                return Err(self.cx.at(value, "IO await requires @effects('io')"));
            }
            self.expressions().call(call, true)?
        } else {
            self.expressions().lower(value)?
        };
        self.check_type(&ty, &actual, value)?;
        if let Some(existing) = self.env.get(n.id.as_str()) {
            if *existing != ty {
                return Err(self
                    .cx
                    .at(a, "existing binding annotation cannot change its type"));
            }
            if awaited {
                return Ok(p::Operation::Await(p::AwaitDecl {
                    name: n.id.to_string(),
                    ty: ty.parser(),
                    call: expr,
                }));
            }
            return Ok(p::Operation::Set(p::SetDecl {
                name: n.id.to_string(),
                expr,
            }));
        }
        self.bind(n.id.as_str(), ty.clone(), a.target.as_ref())?;
        if awaited {
            Ok(p::Operation::Await(p::AwaitDecl {
                name: n.id.to_string(),
                ty: ty.parser(),
                call: expr,
            }))
        } else {
            Ok(p::Operation::Let(p::LetDecl {
                name: n.id.to_string(),
                ty: ty.parser(),
                expr,
            }))
        }
    }

    fn assign(&mut self, a: &py::StmtAssign) -> Result<p::Operation, Diagnostic> {
        if a.targets.len() != 1 || a.type_comment.is_some() {
            return Err(self.cx.at(
                a,
                "multiple targets/type comments are unsupported; annotate each new binding",
            ));
        }
        let py::Expr::Name(n) = &a.targets[0] else {
            return Err(self.cx.at(a, "assignment target must be a simple existing name; list mutation/unpacking is unsupported"));
        };
        let expected = self.env.get(n.id.as_str()).ok_or_else(|| {
            self.cx.at(
                a,
                "first assignment requires a type annotation; block-local names cannot escape",
            )
        })?;
        let (expr, actual) = self.expressions().lower(&a.value)?;
        self.check_type(expected, &actual, a.value.as_ref())?;
        Ok(p::Operation::Set(p::SetDecl {
            name: n.id.to_string(),
            expr,
        }))
    }

    fn check_type(
        &self,
        expected: &Ty,
        actual: &Ty,
        node: &impl py::Ranged,
    ) -> Result<(), Diagnostic> {
        if !compatible(expected, actual) {
            return Err(self.cx.at(node, format!("type mismatch: expected {expected:?}, got {actual:?}; no implicit Python coercions")));
        }
        Ok(())
    }

    fn condition(&self, expr: &py::Expr) -> Result<p::Expr, Diagnostic> {
        let (expr_out, ty) = self.expressions().lower(expr)?;
        if ty != Ty::Bool {
            return Err(self.cx.at(
                expr,
                "condition must be statically bool; Python truthiness is unsupported",
            ));
        }
        Ok(expr_out)
    }

    fn if_statement(&mut self, v: &py::StmtIf) -> Result<p::Operation, Diagnostic> {
        let condition = self.condition(&v.test)?;
        let then_body = self.block(&v.body)?;
        let mut rest = v.orelse.as_slice();
        let mut elif_branches = Vec::new();
        while let [py::Stmt::If(next)] = rest {
            let condition = self.condition(&next.test)?;
            let body = self.block(&next.body)?;
            elif_branches.push((condition, body));
            rest = &next.orelse;
        }
        let else_body = self.block(rest)?;
        Ok(p::Operation::If(p::IfDecl {
            condition,
            then_body,
            elif_branches,
            else_body,
        }))
    }

    fn each(&mut self, v: &py::StmtFor) -> Result<p::Operation, Diagnostic> {
        if !v.orelse.is_empty() || v.type_comment.is_some() {
            return Err(self.cx.at(v, "for-else/type comments are unsupported"));
        }
        let py::Expr::Name(n) = v.target.as_ref() else {
            return Err(self.cx.at(
                v,
                "for requires one fresh simple item name; unpacking is unsupported",
            ));
        };
        let (collection, ty) = self.expressions().lower(&v.iter)?;
        let Ty::List(item) = ty else {
            return Err(self
                .cx
                .at(v.iter.as_ref(), "for collection must be statically list[T]"));
        };
        if *item == Ty::Unknown {
            return Err(self.cx.at(
                v.iter.as_ref(),
                "empty list iteration needs an annotated list binding first",
            ));
        }
        self.bind(n.id.as_str(), (*item).clone(), v.target.as_ref())?;
        let body = self.block(&v.body)?;
        self.env.remove(n.id.as_str());
        Ok(p::Operation::Each(p::EachDecl {
            collection,
            item: n.id.to_string(),
            item_type: item.parser(),
            body,
        }))
    }
}

fn returns(body: &[p::Operation]) -> bool {
    body.iter().any(|op| match op {
        p::Operation::Return(_) => true,
        p::Operation::If(v) => {
            returns(&v.then_body)
                && !v.else_body.is_empty()
                && returns(&v.else_body)
                && v.elif_branches.iter().all(|(_, body)| returns(body))
        }
        _ => false,
    })
}

fn statement_kind(s: &py::Stmt) -> &'static str {
    match s {
        py::Stmt::Break(_) => "break",
        py::Stmt::Continue(_) => "continue",
        py::Stmt::AugAssign(_) => "augmented assignment",
        py::Stmt::Import(_) | py::Stmt::ImportFrom(_) => "import",
        py::Stmt::FunctionDef(_) | py::Stmt::AsyncFunctionDef(_) => "nested definition",
        py::Stmt::ClassDef(_) => "class",
        py::Stmt::Expr(_) => "expression statement",
        py::Stmt::Pass(_) => "pass",
        py::Stmt::Try(_) | py::Stmt::TryStar(_) => "try",
        py::Stmt::With(_) | py::Stmt::AsyncWith(_) => "with",
        py::Stmt::Assert(_) => "assert outside test lowering",
        _ => "form",
    }
}
