use super::{Context, Diagnostic};
use crate::parser as p;
use rustpython_parser::ast as py;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Ty {
    Int,
    Float,
    Bool,
    Str,
    List(Box<Ty>),
    Result(Box<Ty>),
    Unknown,
}

impl Ty {
    pub(super) fn parser(&self) -> p::TypeExpr {
        match self {
            Self::List(t) | Self::Result(t) => p::TypeExpr::Generic {
                name: if matches!(self, Self::List(_)) {
                    "List"
                } else {
                    "Result"
                }
                .into(),
                args: vec![t.parser()],
            },
            _ => p::TypeExpr::Named(
                match self {
                    Self::Int => "Int",
                    Self::Float => "Float",
                    Self::Bool => "Bool",
                    Self::Str => "String",
                    _ => "Unknown",
                }
                .into(),
            ),
        }
    }
    fn numeric(&self) -> bool {
        matches!(self, Self::Int | Self::Float)
    }
}

#[derive(Clone)]
pub(super) struct Signature {
    pub params: Vec<Ty>,
    pub result: Ty,
    pub effects: Vec<String>,
}

pub(super) fn annotation(cx: &Context<'_>, e: &py::Expr) -> Result<Ty, Diagnostic> {
    match e {
        py::Expr::Name(n) => match n.id.as_str() {
            "int" => Ok(Ty::Int),
            "float" => Ok(Ty::Float),
            "bool" => Ok(Ty::Bool),
            "str" => Ok(Ty::Str),
            _ => Err(cx.at(
                e,
                "unsupported type; use int, float, bool, str, list[T], or Result[T]",
            )),
        },
        py::Expr::Subscript(s) => {
            let py::Expr::Name(n) = s.value.as_ref() else {
                return Err(cx.at(e, "unsupported generic type; use list[T] or Result[T]"));
            };
            if !matches!(n.id.as_str(), "list" | "Result") {
                return Err(cx.at(e, "unsupported generic type; use list[T] or Result[T]"));
            }
            let inner = Box::new(annotation(cx, &s.slice)?);
            match n.id.as_str() {
                "list" => Ok(Ty::List(inner)),
                "Result" => Ok(Ty::Result(inner)),
                _ => Err(cx.at(e, "unsupported generic type; use list[T] or Result[T]")),
            }
        }
        _ => Err(cx.at(e, "unsupported annotation; explicit subset type required")),
    }
}

pub(super) fn compatible(expected: &Ty, actual: &Ty) -> bool {
    match (expected, actual) {
        (_, Ty::Unknown) => true,
        (Ty::List(a), Ty::List(b)) => compatible(a, b),
        _ => expected == actual,
    }
}

pub(super) struct Expressions<'a, 'b> {
    pub cx: &'a Context<'b>,
    pub env: &'a HashMap<String, Ty>,
    pub signatures: &'a HashMap<String, Signature>,
}

impl Expressions<'_, '_> {
    pub fn lower(&self, e: &py::Expr) -> Result<(p::Expr, Ty), Diagnostic> {
        match e {
            py::Expr::Constant(_) | py::Expr::List(_) => self.literal_or_list(e),
            py::Expr::Name(n) => self.env.get(n.id.as_str()).cloned()
                .map(|ty| (p::Expr::Ident(n.id.to_string()), ty))
                .ok_or_else(|| self.cx.at(e, format!("unbound or out-of-block name {}; annotate new local bindings", n.id))),
            py::Expr::Call(c) => self.call(c, false),
            py::Expr::BinOp(b) => self.binary(b),
            py::Expr::Compare(c) => self.compare(c),
            py::Expr::BoolOp(b) => self.boolean(b),
            py::Expr::UnaryOp(u) => self.unary(u),
            _ => Err(self.cx.at(e, format!("unsupported expression {}; use typed names, literals, direct calls or supported operators", expression_kind(e)))),
        }
    }

    fn literal_or_list(&self, e: &py::Expr) -> Result<(p::Expr, Ty), Diagnostic> {
        if let py::Expr::List(l) = e {
            let mut item_ty = Ty::Unknown;
            let mut args = Vec::new();
            for item in &l.elts {
                let (value, ty) = self.lower(item)?;
                if ty == Ty::Unknown {
                    return Err(self.cx.at(item, "list items need a known static type; bind the call to an annotated local first"));
                }
                if item_ty != Ty::Unknown && item_ty != ty {
                    return Err(self.cx.at(
                        item,
                        "list literal items must have one explicit element type",
                    ));
                }
                item_ty = ty;
                args.push(value);
            }
            return Ok((call("list", args), Ty::List(Box::new(item_ty))));
        }
        literal(self.cx, e)
    }

    pub fn call(&self, c: &py::ExprCall, awaited: bool) -> Result<(p::Expr, Ty), Diagnostic> {
        if !c.keywords.is_empty() {
            return Err(self.cx.at(
                c,
                "keyword arguments in function bodies are unsupported; use positional arguments",
            ));
        }
        let name = direct_name(self.cx, &c.func, self.env)?;
        let resolved = if !name.contains('.') && self.signatures.contains_key(&name) {
            format!("{}.{}", self.cx.module, name)
        } else {
            name.clone()
        };
        if !name.contains('.')
            && !self.signatures.contains_key(&name)
            && !crate::builtins::is_builtin(&name)
        {
            return Err(self.cx.at(c, format!("unknown local callable {name}; declare it in this submission or use Module.function")));
        }
        reject_unsafe_builtin(self.cx, c, &name)?;
        let sig = self
            .signatures
            .get(&name)
            .or_else(|| self.signatures.get(&resolved))
            .or_else(|| {
                resolved
                    .strip_prefix(&format!("{}.", self.cx.module))
                    .and_then(|n| self.signatures.get(n))
            });
        let is_io = crate::builtins::is_io_builtin(&name)
            || sig.is_some_and(|s| s.effects.iter().any(|e| e == "io" || e == "async"));
        if awaited
            && !crate::builtins::is_io_builtin(&name)
            && (crate::builtins::is_builtin(&name)
                || sig.is_some_and(|s| !s.effects.iter().any(|e| e == "async")))
        {
            return Err(self.cx.at(c, "await requires an async function or runtime IO builtin; synchronous calls are not awaitable"));
        }
        if awaited
            && !crate::builtins::is_io_builtin(&name)
            && sig.is_some_and(|signature| matches!(signature.result, Ty::Result(_)))
        {
            return Err(self.cx.at(c, "awaiting Result-returning user functions is unsupported until runtime Result unwrapping is reconciled (#82); direct runtime IO builtin awaits remain supported"));
        }
        if is_io && !awaited {
            return Err(self.cx.at(c, "IO/async call requires an annotated await binding in an async function with @effects('io')"));
        }
        let values = c
            .args
            .iter()
            .map(|a| self.lower(a))
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(s) = sig {
            if s.params.len() != values.len()
                || s.params
                    .iter()
                    .zip(&values)
                    .any(|(a, (_, b))| !compatible(a, b))
            {
                return Err(self.cx.at(
                    c,
                    format!("arguments do not match declared signature of {resolved}"),
                ));
            }
        }
        let result = if matches!(name.as_str(), "len" | "length") {
            if values.len() != 1 || !matches!(values[0].1, Ty::List(_)) {
                return Err(self.cx.at(
                    c,
                    "len requires one statically typed list argument in the Python subset",
                ));
            }
            Ty::Int
        } else if name == "concat" {
            if values.len() < 2 || values.iter().any(|(_, t)| *t != Ty::Str) {
                return Err(self.cx.at(c, "concat requires at least two statically str arguments; implicit coercion is unsupported"));
            }
            Ty::Str
        } else if matches!(name.as_str(), "read_file" | "file_read") {
            Ty::Str
        } else if name == "http_get" {
            Ty::Result(Box::new(Ty::Str))
        } else {
            sig.map(|s| s.result.clone()).unwrap_or(Ty::Unknown)
        };
        Ok((
            call(&resolved, values.into_iter().map(|(e, _)| e).collect()),
            result,
        ))
    }

    pub fn requires_io(&self, call: &py::ExprCall) -> Result<bool, Diagnostic> {
        let name = direct_name(self.cx, &call.func, self.env)?;
        let signature = self
            .signatures
            .get(&name)
            .or_else(|| self.signatures.get(&format!("{}.{}", self.cx.module, name)));
        Ok(crate::builtins::is_io_builtin(&name)
            || signature
                .is_some_and(|signature| signature.effects.iter().any(|effect| effect == "io")))
    }

    fn binary(&self, b: &py::ExprBinOp) -> Result<(p::Expr, Ty), Diagnostic> {
        let (left, lt) = self.lower(&b.left)?;
        let (right, rt) = self.lower(&b.right)?;
        if b.op == py::Operator::Add && lt == Ty::Str && rt == Ty::Str {
            return Ok((call("concat", vec![left, right]), Ty::Str));
        }
        if !lt.numeric() || lt != rt {
            return Err(self.cx.at(b, "arithmetic requires matching known int/float types (str + str is also supported); annotate unknown call results first"));
        }
        let op = match b.op {
            py::Operator::Add => p::BinaryOp::Add, py::Operator::Sub => p::BinaryOp::Sub,
            py::Operator::Mult => p::BinaryOp::Mul,
            _ => return Err(self.cx.at(b, "unsupported arithmetic operator; only +, -, * preserve this runtime's subset semantics")),
        };
        Ok((
            p::Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            },
            lt,
        ))
    }

    fn compare(&self, c: &py::ExprCompare) -> Result<(p::Expr, Ty), Diagnostic> {
        if c.ops.len() != 1 || c.comparators.len() != 1 {
            return Err(self.cx.at(
                c,
                "chained comparisons are unsupported; use separate bool bindings",
            ));
        }
        let (left, lt) = self.lower(&c.left)?;
        let (right, rt) = self.lower(&c.comparators[0])?;
        let equality = matches!(c.ops[0], py::CmpOp::Eq | py::CmpOp::NotEq);
        if lt != rt || !(lt.numeric() || (equality && matches!(lt, Ty::Bool | Ty::Str))) {
            return Err(self.cx.at(
                c,
                "comparison requires matching known primitive types; ordering is numeric only",
            ));
        }
        let op = match c.ops[0] {
            py::CmpOp::Eq => p::BinaryOp::Eq,
            py::CmpOp::NotEq => p::BinaryOp::Neq,
            py::CmpOp::Gt => p::BinaryOp::Gt,
            py::CmpOp::Lt => p::BinaryOp::Lt,
            py::CmpOp::GtE => p::BinaryOp::Gte,
            py::CmpOp::LtE => p::BinaryOp::Lte,
            _ => {
                return Err(self.cx.at(
                    c,
                    "unsupported comparison; identity/membership operators are not supported",
                ));
            }
        };
        Ok((
            p::Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            },
            Ty::Bool,
        ))
    }

    fn boolean(&self, b: &py::ExprBoolOp) -> Result<(p::Expr, Ty), Diagnostic> {
        let mut values = Vec::new();
        for value in &b.values {
            let (expr, ty) = self.lower(value)?;
            if ty != Ty::Bool || !total_boolean_operand(&expr) {
                return Err(self.cx.at(value, "and/or require total, call-free bool operands; runtime eager evaluation cannot emulate Python short-circuit calls"));
            }
            values.push(expr);
        }
        let mut values = values.into_iter();
        let mut expr = values
            .next()
            .ok_or_else(|| self.cx.at(b, "empty boolean expression"))?;
        for rhs in values {
            expr = p::Expr::Binary {
                op: if b.op == py::BoolOp::And {
                    p::BinaryOp::And
                } else {
                    p::BinaryOp::Or
                },
                left: Box::new(expr),
                right: Box::new(rhs),
            };
        }
        Ok((expr, Ty::Bool))
    }

    fn unary(&self, u: &py::ExprUnaryOp) -> Result<(p::Expr, Ty), Diagnostic> {
        if matches!(u.op, py::UnaryOp::USub | py::UnaryOp::UAdd)
            && matches!(u.operand.as_ref(), py::Expr::Constant(_))
        {
            return literal(self.cx, &py::Expr::UnaryOp(u.clone()));
        }
        let (expr, ty) = self.lower(&u.operand)?;
        let op = match u.op {
            py::UnaryOp::Not if ty == Ty::Bool => p::UnaryOp::Not,
            py::UnaryOp::USub if ty.numeric() => p::UnaryOp::Neg,
            _ => return Err(self.cx.at(u, "unsupported unary operation; not requires bool and minus requires a known numeric type")),
        };
        Ok((
            p::Expr::Unary {
                op,
                expr: Box::new(expr),
            },
            ty,
        ))
    }
}

pub(super) fn total_boolean_operand(e: &p::Expr) -> bool {
    match e {
        p::Expr::Bool(_)
        | p::Expr::Int(_)
        | p::Expr::Float(_)
        | p::Expr::String(_)
        | p::Expr::Ident(_) => true,
        p::Expr::Unary {
            op: p::UnaryOp::Not,
            expr,
        } => total_boolean_operand(expr),
        p::Expr::Unary {
            op: p::UnaryOp::Neg,
            expr,
        } => {
            matches!(expr.as_ref(), p::Expr::Int(n) if *n >= 0)
                || matches!(expr.as_ref(), p::Expr::Float(n) if n.is_finite())
        }
        // Integer arithmetic/negation can overflow, so it is not total either.
        p::Expr::Binary { op, left, right }
            if matches!(
                op,
                p::BinaryOp::Eq
                    | p::BinaryOp::Neq
                    | p::BinaryOp::Gt
                    | p::BinaryOp::Lt
                    | p::BinaryOp::Gte
                    | p::BinaryOp::Lte
                    | p::BinaryOp::And
                    | p::BinaryOp::Or
            ) =>
        {
            total_boolean_operand(left) && total_boolean_operand(right)
        }
        _ => false,
    }
}

pub(super) fn direct_name(
    cx: &Context<'_>,
    e: &py::Expr,
    env: &HashMap<String, Ty>,
) -> Result<String, Diagnostic> {
    let name =
        match e {
            py::Expr::Name(n) if !env.contains_key(n.id.as_str()) => n.id.to_string(),
            py::Expr::Attribute(a) => match a.value.as_ref() {
                py::Expr::Name(n) if !env.contains_key(n.id.as_str()) => {
                    format!("{}.{}", n.id, a.attr)
                }
                _ => return Err(cx.at(
                    e,
                    "dynamic attributes/callable variables are unsupported; use Module.function",
                )),
            },
            _ => return Err(cx.at(
                e,
                "callable variables and dynamic calls are unsupported; use a direct function name",
            )),
        };
    if name
        .split('.')
        .any(|n| !super::identifier(n) || super::forbidden(n))
    {
        return Err(cx.at(e, "unsafe/unsupported callable identifier; dunders and Python execution escapes are forbidden"));
    }
    Ok(name)
}

pub(super) fn reject_unsafe_builtin(
    cx: &Context<'_>,
    c: &py::ExprCall,
    name: &str,
) -> Result<(), Diagnostic> {
    if matches!(name, "int" | "float" | "bool" | "str" | "list") {
        return Err(cx.at(
            c,
            "Python constructors/coercions are unsupported; use typed values and list literals",
        ));
    }
    let list_builtin = crate::builtins::BUILTINS
        .iter()
        .find(|b| b.name == name || b.aliases.contains(&name))
        .is_some_and(|b| matches!(b.category, crate::builtins::BuiltinCategory::List));
    if list_builtin && !matches!(name, "len" | "length") {
        return Err(cx.at(
            c,
            "list operations are read-only in this subset; use literals, iteration, or len",
        ));
    }
    Ok(())
}

pub(super) fn call(name: &str, args: Vec<p::Expr>) -> p::Expr {
    let callee = match name.split_once('.') {
        Some((module, member)) => p::Expr::FieldAccess {
            base: Box::new(p::Expr::Ident(module.into())),
            field: member.into(),
        },
        None => p::Expr::Ident(name.into()),
    };
    p::Expr::Call {
        callee: Box::new(callee),
        args,
    }
}

pub(super) fn literal(cx: &Context<'_>, e: &py::Expr) -> Result<(p::Expr, Ty), Diagnostic> {
    match e {
        py::Expr::Constant(c) => match &c.value {
            py::Constant::Bool(v) => Ok((p::Expr::Bool(*v), Ty::Bool)),
            py::Constant::Str(s) => {
                super::emit::quote(s).map_err(|m| cx.at(e, m))?;
                Ok((p::Expr::String(s.clone()), Ty::Str))
            }
            py::Constant::Int(n) => n.to_string().parse::<i64>().map(|n| (p::Expr::Int(n), Ty::Int))
                .map_err(|_| cx.at(e, "integer literal exceeds signed 64-bit runtime range")),
            py::Constant::Float(n) if n.is_finite() => Ok((p::Expr::Float(*n), Ty::Float)),
            _ => Err(cx.at(e, "unsupported literal; only finite int/float, bool, str and lists are supported")),
        },
        py::Expr::UnaryOp(u) if matches!(u.op, py::UnaryOp::USub | py::UnaryOp::UAdd) => {
            if !matches!(u.operand.as_ref(), py::Expr::Constant(_)) { return Err(cx.at(e, "signed literal requires one numeric constant")); }
            if u.op == py::UnaryOp::USub && matches!(u.operand.as_ref(), py::Expr::Constant(c) if matches!(&c.value, py::Constant::Int(n) if n.to_string() == "9223372036854775808")) {
                return Err(cx.at(e, "-9223372036854775808 cannot roundtrip through the Adapsis lexer; supported integer literals are -9223372036854775807 through 9223372036854775807"));
            }
            let (value, ty) = literal(cx, &u.operand)?;
            if !ty.numeric() { return Err(cx.at(e, "sign is only supported on numeric literals")); }
            if u.op == py::UnaryOp::UAdd { return Ok((value, ty)); }
            Ok((p::Expr::Unary { op: p::UnaryOp::Neg, expr: Box::new(value) }, ty))
        }
        py::Expr::List(l) => {
            let mut ty = Ty::Unknown;
            let mut items = Vec::new();
            for item in &l.elts {
                let (value, next) = literal(cx, item)?;
                if ty != Ty::Unknown && ty != next { return Err(cx.at(item, "literal list items must have the same type")); }
                ty = next; items.push(value);
            }
            Ok((call("list", items), Ty::List(Box::new(ty))))
        }
        _ => Err(cx.at(e, "literal required; calls, operators, names and comprehensions cannot be test/evaluate data")),
    }
}

fn expression_kind(e: &py::Expr) -> &'static str {
    match e {
        py::Expr::Attribute(_) => "attribute access",
        py::Expr::Subscript(_) => "subscript",
        py::Expr::Await(_) => "await outside annotated binding",
        py::Expr::Lambda(_) => "lambda",
        py::Expr::ListComp(_) => "list comprehension",
        py::Expr::JoinedStr(_) => "f-string",
        py::Expr::IfExp(_) => "conditional expression",
        py::Expr::Dict(_) => "dict",
        py::Expr::Tuple(_) => "tuple",
        py::Expr::NamedExpr(_) => "assignment expression",
        _ => "form",
    }
}
