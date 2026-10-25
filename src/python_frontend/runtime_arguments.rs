//! Python call fidelity at the actual dispatch snapshot, never transitive body
//! authorization. Legacy Adapsis's permissive argument binding remains intact.
use crate::{
    ast::Program,
    parser::{Expr, Operation},
};
use std::collections::HashMap;

pub fn validate_runtime_arguments(ops: &[Operation], program: &Program) -> Result<(), String> {
    let mut declared = HashMap::new();
    declarations(ops, None, &mut declared);
    let context = Calls { program, declared };
    context.operations(ops)
}

#[derive(Clone)]
struct Signature {
    params: Vec<String>,
    asynchronous: bool,
    returns_result: bool,
}

fn declarations(ops: &[Operation], module: Option<&str>, out: &mut HashMap<String, Signature>) {
    for op in ops {
        match op {
            Operation::Module(decl) => declarations(&decl.body, Some(&decl.name), out),
            Operation::Function(decl) => {
                if let Some(module) = module {
                    out.insert(
                        format!("{module}.{}", decl.name),
                        Signature {
                            params: decl.params.iter().map(|p| p.name.clone()).collect(),
                            asynchronous: decl.effects.iter().any(|effect| effect == "async"),
                            returns_result: matches!(&decl.return_type, crate::parser::TypeExpr::Generic { name, .. } if name == "Result"),
                        },
                    );
                }
            }
            _ => {}
        }
    }
}

struct Calls<'a> {
    program: &'a Program,
    declared: HashMap<String, Signature>,
}

impl Calls<'_> {
    fn signature(&self, target: &str) -> Result<Signature, String> {
        let signature = self.declared.get(target).cloned().or_else(|| {
            self.program.get_function(target).map(|function| Signature {
                params: function.params.iter().map(|p| p.name.clone()).collect(),
                asynchronous: function.effects.contains(&crate::ast::Effect::Async),
                returns_result: matches!(function.return_type, crate::ast::Type::Result(_)),
            })
        }).ok_or_else(|| format!("Python call target `{target}` was not found; declare it before submitting its caller"))?;
        if signature
            .params
            .iter()
            .any(|name| super::reserved_positional_name(name))
        {
            return Err(format!(
                "Python calls cannot target `{target}`: parameter names _0, _1, ... are reserved"
            ));
        }
        Ok(signature)
    }

    fn positional(&self, target: &str, count: usize) -> Result<(), String> {
        let params = self.signature(target)?.params;
        if params.len() != count {
            return Err(format!(
                "Python call `{target}` expects exactly {} arguments, received {count}",
                params.len()
            ));
        }
        Ok(())
    }

    fn input(&self, target: &str, input: &Expr) -> Result<(), String> {
        let Expr::StructLiteral(fields) = input else {
            return self.positional(target, 1);
        };
        let params = self.signature(target)?.params;
        let positional = fields
            .iter()
            .all(|field| super::reserved_positional_name(&field.name));
        if positional {
            if fields.len() != params.len()
                || fields
                    .iter()
                    .enumerate()
                    .any(|(index, field)| field.name != format!("_{index}"))
            {
                return Err(format!(
                    "Python call `{target}` expects exactly {} positional arguments, received {}",
                    params.len(),
                    fields.len()
                ));
            }
        } else if fields.len() != params.len()
            || fields.iter().any(|field| !params.contains(&field.name))
            || params
                .iter()
                .any(|param| fields.iter().filter(|field| &field.name == param).count() != 1)
        {
            return Err(format!(
                "Python call `{target}` requires exactly these keyword names: {}",
                params.join(", ")
            ));
        }
        Ok(())
    }

    fn expression(&self, expr: &Expr) -> Result<(), String> {
        self.expression_mode(expr, false)
    }

    fn expression_mode(&self, expr: &Expr, awaited: bool) -> Result<(), String> {
        match expr {
            Expr::Call { callee, args } => {
                let target =
                    qualified(callee).ok_or("Python dynamic call targets are unsupported")?;
                let asynchronous = if target.contains('.') {
                    self.positional(&target, args.len())?;
                    let signature = self.signature(&target)?;
                    if awaited && signature.returns_result {
                        return Err(format!(
                            "Python await of Result-returning user function `{target}` is unsupported until runtime unwrapping is reconciled (#82)"
                        ));
                    }
                    signature.asynchronous
                } else {
                    crate::builtins::is_io_builtin(&target)
                };
                if awaited != asynchronous {
                    return Err(format!(
                        "Python call `{target}` {}",
                        if awaited {
                            "is not awaitable; await requires an async function or runtime IO builtin"
                        } else {
                            "requires an annotated await binding; unawaited async values are unsupported"
                        }
                    ));
                }
                args.iter().try_for_each(|arg| self.expression(arg))
            }
            Expr::Binary { left, right, .. } => {
                self.expression(left)?;
                self.expression(right)
            }
            Expr::Unary { expr, .. } | Expr::Cast { expr, .. } => self.expression(expr),
            Expr::FieldAccess { base, .. } => self.expression(base),
            Expr::StructLiteral(fields) => fields
                .iter()
                .try_for_each(|field| self.expression(&field.value)),
            _ => Ok(()),
        }
    }

    fn operations(&self, ops: &[Operation]) -> Result<(), String> {
        for op in ops {
            match op {
                Operation::Module(decl) => self.operations(&decl.body)?,
                Operation::Function(decl) => self.operations(&decl.body)?,
                Operation::Test(test) => {
                    for case in &test.cases {
                        self.input(&test.function_name, &case.input)?;
                    }
                }
                Operation::Eval(eval) => self.input(&eval.function_name, &eval.input)?,
                Operation::Let(value) => self.expression(&value.expr)?,
                Operation::Set(value) => self.expression(&value.expr)?,
                Operation::Call(value) => self.expression(&value.expr)?,
                Operation::Return(value) => self.expression(&value.expr)?,
                Operation::Await(value) => self.expression_mode(&value.call, true)?,
                Operation::If(branch) => {
                    self.expression(&branch.condition)?;
                    self.operations(&branch.then_body)?;
                    self.operations(&branch.else_body)?;
                    for (condition, body) in &branch.elif_branches {
                        self.expression(condition)?;
                        self.operations(body)?;
                    }
                }
                Operation::While(loop_) => {
                    self.expression(&loop_.condition)?;
                    self.operations(&loop_.body)?;
                }
                Operation::Each(loop_) => {
                    self.expression(&loop_.collection)?;
                    self.operations(&loop_.body)?;
                }
                _ => {}
            }
        }
        Ok(())
    }
}

fn qualified(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Ident(name) => Some(name.clone()),
        Expr::FieldAccess { base, field } => Some(format!("{}.{field}", qualified(base)?)),
        _ => None,
    }
}
