//! Python inspection is derived exclusively from the live runtime AST.
use super::{
    Diagnostic,
    expressions::{Signature, Ty},
};
use crate::{ast as a, parser as p};
use std::collections::HashMap;

pub fn print_function(program: &a::Program, target: &str) -> Result<String, Diagnostic> {
    let error = |message: String| Diagnostic {
        module: target.into(),
        line: 1,
        column: 1,
        message: format!(
            "cannot print {target} as Python: {message}; inspect Adapsis source instead"
        ),
    };
    let Some((module, member)) = target.split_once('.') else {
        return Err(error(
            "a qualified Module.function target is required".into(),
        ));
    };
    if !super::identifier(module) || !super::identifier(member) {
        return Err(error("unsupported target identifier".into()));
    }
    let f = program
        .get_function(target)
        .ok_or_else(|| error("function not found".into()))?;
    let printer = Printer { program };
    let text = printer.function(f).map_err(&error)?;
    let mut signatures = HashMap::new();
    collect_signatures(&program.modules, "", &mut signatures);
    // This type environment comes only from the supplied AST. Native callers
    // pass authority-filtered signatures, as they do for final mutation lowering.
    let lowered = super::lower::module_with_signatures(&text, module, signatures, true)
        .map_err(|e| error(e.message))?;
    let [p::Operation::Module(m)] = lowered.operations.as_slice() else {
        return Err(error("internal module roundtrip mismatch".into()));
    };
    let [p::Operation::Function(lowered)] = m.body.as_slice() else {
        return Err(error("internal function roundtrip mismatch".into()));
    };
    let reconstructed = lowered
        .body
        .iter()
        .map(crate::validator::convert_statement_op)
        .collect::<anyhow::Result<Vec<_>>>()
        .map_err(|e| error(e.to_string()))?;
    if semantic_body(&f.body).map_err(&error)? != semantic_body(&reconstructed).map_err(&error)? {
        return Err(error(
            "unsupported AST semantics (Python roundtrip changed a statement)".into(),
        ));
    }
    if !same_signature(f, lowered).map_err(&error)? {
        return Err(error(
            "Python roundtrip changed the function signature, effects or documentation".into(),
        ));
    }
    Ok(text)
}

fn same_signature(original: &a::FunctionDecl, lowered: &p::FunctionDecl) -> Result<bool, String> {
    let params = original
        .params
        .iter()
        .map(|param| {
            Ok(p::Param {
                name: param.name.clone(),
                ty: static_type(&param.ty)?.parser(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let result = static_type(&original.return_type)?.parser();
    let mut original_effects = effect_names(&original.effects)?;
    let mut lowered_effects = lowered.effects.clone();
    original_effects.sort();
    lowered_effects.sort();
    Ok(original.name == lowered.name
        && original.doc == lowered.doc
        && format!("{params:?}") == format!("{:?}", lowered.params)
        && format!("{result:?}") == format!("{:?}", lowered.return_type)
        && original_effects == lowered_effects)
}

pub(super) fn collect_signatures(
    modules: &[a::Module],
    prefix: &str,
    out: &mut HashMap<String, Signature>,
) {
    for module in modules {
        let path = if prefix.is_empty() {
            module.name.clone()
        } else {
            format!("{prefix}.{}", module.name)
        };
        for f in &module.functions {
            if let (Ok(params), Ok(result), Ok(effects)) = (
                f.params
                    .iter()
                    .map(|p| static_type(&p.ty))
                    .collect::<Result<Vec<_>, _>>(),
                static_type(&f.return_type),
                effect_names(&f.effects),
            ) {
                out.insert(
                    format!("{path}.{}", f.name),
                    Signature {
                        params,
                        result,
                        effects,
                    },
                );
            }
        }
        collect_signatures(&module.modules, &path, out);
    }
}

fn semantic_body(body: &[a::Statement]) -> Result<serde_json::Value, String> {
    let mut value = serde_json::to_value(body).map_err(|e| e.to_string())?;
    strip_ids(&mut value);
    Ok(value)
}

fn strip_ids(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(fields) => {
            fields.remove("id");
            for value in fields.values_mut() {
                strip_ids(value);
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                strip_ids(value);
            }
        }
        _ => {}
    }
}

fn static_type(ty: &a::Type) -> Result<Ty, String> {
    Ok(match ty {
        a::Type::Int => Ty::Int,
        a::Type::Float => Ty::Float,
        a::Type::Bool => Ty::Bool,
        a::Type::String => Ty::Str,
        a::Type::List(t) => Ty::List(Box::new(static_type(t)?)),
        a::Type::Result(t) => Ty::Result(Box::new(static_type(t)?)),
        _ => return Err(format!("unsupported type {ty:?}")),
    })
}

fn type_text(ty: &a::Type) -> Result<String, String> {
    Ok(match ty {
        a::Type::Int => "int".into(),
        a::Type::Float => "float".into(),
        a::Type::Bool => "bool".into(),
        a::Type::String => "str".into(),
        a::Type::List(t) => format!("list[{}]", type_text(t)?),
        a::Type::Result(t) => format!("Result[{}]", type_text(t)?),
        _ => return Err(format!("unsupported type {ty:?}")),
    })
}

fn effect_names(effects: &[a::Effect]) -> Result<Vec<String>, String> {
    effects
        .iter()
        .map(|e| {
            Ok(match e {
                a::Effect::Io => "io",
                a::Effect::Async => "async",
                a::Effect::Fail => "fail",
                a::Effect::Mut => "mut",
                _ => return Err(format!("unsupported effect {e:?}")),
            }
            .into())
        })
        .collect()
}

struct Printer<'a> {
    program: &'a a::Program,
}

impl Printer<'_> {
    fn function(&self, f: &a::FunctionDecl) -> Result<String, String> {
        let mut out = String::new();
        let effects = effect_names(&f.effects)?;
        let decorated = effects
            .iter()
            .filter(|e| e.as_str() != "async")
            .map(|e| super::emit::quote(e))
            .collect::<Result<Vec<_>, _>>()?;
        if !decorated.is_empty() {
            out.push_str(&format!("@effects({})\n", decorated.join(", ")));
        }
        let asynchronous = if effects.iter().any(|e| e == "async") {
            "async "
        } else {
            ""
        };
        let params = f
            .params
            .iter()
            .map(|p| {
                Ok(format!(
                    "{}: {}",
                    super::emit::name(&p.name)?,
                    type_text(&p.ty)?
                ))
            })
            .collect::<Result<Vec<_>, String>>()?
            .join(", ");
        out.push_str(&format!(
            "{asynchronous}def {}({params}) -> {}:\n",
            super::emit::name(&f.name)?,
            type_text(&f.return_type)?
        ));
        if let Some(doc) = &f.doc {
            line(&mut out, 1, super::emit::quote(doc)?);
        }
        self.block(&f.body, 1, &mut out)?;
        Ok(out)
    }

    fn block(&self, body: &[a::Statement], indent: usize, out: &mut String) -> Result<(), String> {
        if body.is_empty() {
            return Err("unsupported empty statement block".into());
        }
        for stmt in body {
            self.statement(&stmt.kind, indent, out)?;
        }
        Ok(())
    }

    fn statement(&self, stmt: &a::StatementKind, i: usize, out: &mut String) -> Result<(), String> {
        use a::StatementKind::*;
        match stmt {
            Let { name, ty, value } => line(out, i, format!("{}: {} = {}", super::emit::name(name)?, type_text(ty)?, self.expr(value)?)),
            Set { name, value } => line(out, i, format!("{} = {}", super::emit::name(name)?, self.expr(value)?)),
            Return { value } => line(out, i, format!("return {}", self.expr(value)?)),
            Await { name, ty, call } => line(out, i, format!("{}: {} = await {}", super::emit::name(name)?, type_text(ty)?, self.call(call)?)),
            Branch { condition, then_body, else_body } => {
                line(out, i, format!("if {}:", self.expr(condition)?));
                self.block(then_body, i + 1, out)?;
                if !else_body.is_empty() { line(out, i, "else:"); self.block(else_body, i + 1, out)?; }
            }
            While { condition, body } => {
                line(out, i, format!("while {}:", self.expr(condition)?));
                self.block(body, i + 1, out)?;
            }
            Each { iterator, binding, body } => {
                line(out, i, format!("for {} in {}:", super::emit::name(&binding.name)?, self.expr(iterator)?));
                self.block(body, i + 1, out)?;
            }
            Call { .. } => return Err("unsupported +call statement (error propagation cannot be represented as Python assignment)".into()),
            Match { .. } => return Err("unsupported match statement".into()),
            Spawn { .. } => return Err("unsupported spawn statement".into()),
            Yield { .. } => return Err("unsupported yield statement".into()),
            Check { .. } => return Err("unsupported check statement".into()),
            Source(_) | Event(_) => return Err("unsupported source/event statement".into()),
        }
        Ok(())
    }

    fn expr(&self, e: &a::Expr) -> Result<String, String> {
        use a::Expr::*;
        match e {
            Literal(a::Literal::Int(n)) => Ok(n.to_string()),
            Literal(a::Literal::Float(n)) if n.is_finite() => {
                let s = n.to_string();
                Ok(if s.contains('.') { s } else { format!("{s}.0") })
            }
            Literal(a::Literal::Float(_)) => Err("unsupported nonfinite float".into()),
            Literal(a::Literal::Bool(v)) => Ok(if *v { "True" } else { "False" }.into()),
            Literal(a::Literal::String(s)) => super::emit::quote(s),
            Identifier(n) => super::emit::name(n),
            Call(c) => self.call(c),
            Binary { op, left, right } => Ok(format!(
                "({} {} {})",
                self.expr(left)?,
                binary(op)?,
                self.expr(right)?
            )),
            Unary { op, expr } => Ok(format!(
                "({}{})",
                match op {
                    a::UnaryOp::Not => "not ",
                    a::UnaryOp::Neg => "-",
                },
                self.expr(expr)?
            )),
            FieldAccess { .. } => Err("unsupported field access".into()),
            StructInit { .. } => Err("unsupported struct initialization".into()),
        }
    }

    fn call(&self, c: &a::CallExpr) -> Result<String, String> {
        let name = super::emit::name(&c.callee)?;
        if !name.contains('.') {
            if !crate::builtins::is_builtin(&name) {
                return Err(format!(
                    "unsupported unqualified user call {name}; original lookup cannot be guessed"
                ));
            }
            if shadowed(self.program, &name) {
                return Err(format!(
                    "builtin/helper {name} is shadowed by a runtime function or union variant"
                ));
            }
        }
        let args = c
            .args
            .iter()
            .map(|e| self.expr(e))
            .collect::<Result<Vec<_>, _>>()?;
        if name == "list" {
            return Ok(format!("[{}]", args.join(", ")));
        }
        Ok(format!("{name}({})", args.join(", ")))
    }
}

fn shadowed(program: &a::Program, name: &str) -> bool {
    program.functions.iter().any(|f| f.name == name)
        || variants(&program.types, name)
        || modules_shadow(&program.modules, name)
}

fn modules_shadow(modules: &[a::Module], name: &str) -> bool {
    modules.iter().any(|m| {
        m.functions.iter().any(|f| f.name == name)
            || variants(&m.types, name)
            || modules_shadow(&m.modules, name)
    })
}

fn variants(types: &[a::TypeDecl], name: &str) -> bool {
    types.iter().any(
        |t| matches!(t, a::TypeDecl::TaggedUnion(u) if u.variants.iter().any(|v| v.name == name)),
    )
}

fn binary(op: &a::BinaryOp) -> Result<&'static str, String> {
    Ok(match op {
        a::BinaryOp::Add => "+",
        a::BinaryOp::Sub => "-",
        a::BinaryOp::Mul => "*",
        a::BinaryOp::Equal => "==",
        a::BinaryOp::NotEqual => "!=",
        a::BinaryOp::GreaterThan => ">",
        a::BinaryOp::LessThan => "<",
        a::BinaryOp::GreaterThanOrEqual => ">=",
        a::BinaryOp::LessThanOrEqual => "<=",
        a::BinaryOp::And => "and",
        a::BinaryOp::Or => "or",
        _ => return Err(format!("unsupported binary operator {op:?}")),
    })
}

fn line(out: &mut String, indent: usize, text: impl AsRef<str>) {
    out.push_str(&"    ".repeat(indent));
    out.push_str(text.as_ref());
    out.push('\n');
}
