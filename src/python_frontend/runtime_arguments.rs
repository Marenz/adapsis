//! Python call fidelity at the actual dispatch snapshot, never transitive body
//! authorization. Legacy Adapsis's permissive argument binding remains intact.
use super::call_sites::{self, Effects, Site, Target, UserSignature};
use crate::{
    ast::Program,
    parser::{Expr, Operation},
};
use std::collections::HashMap;
use std::collections::HashSet;

/// Native Python entry preflight, not transitive authorization or payload validation.
/// Inspect only the reachable graph in this snapshot, with tree resolution and no cache.
pub fn validate_entry_await_contract(program: &Program, entry: &str) -> Result<(), String> {
    use crate::ast::{StatementKind, Type};
    use crate::vm::await_admission::{expression_work, statement_work, Work};
    let root = program.get_function(entry).ok_or("Python entry was not found")?;
    let mut visited = HashSet::new();
    let mut work = vec![Work::Function(root)];
    while let Some(item) = work.pop() {
        match item {
            Work::Function(function) => {
                if visited.insert(function as *const _) {
                    work.extend(function.body.iter().map(Work::Statement));
                }
            }
            Work::Statement(statement) => {
                if let StatementKind::Await { ty, call, .. } = &statement.kind {
                    if let Some(callee) = program.get_function(&call.callee) {
                        if matches!(ty, Type::Result(_)) != matches!(callee.return_type, Type::Result(_)) {
                            // Reachable internals may be hidden: disclose neither names nor types.
                            return Err("Python entry has an incompatible user-await Result binding; reconcile its current Result/non-Result contract before execution".into());
                        }
                    }
                }
                statement_work(&statement.kind, &mut work);
            }
            Work::Expression(expression) => expression_work(expression, &mut work),
            Work::Call(call, _) => {
                if let Some(callee) = program.get_function(&call.callee) {
                    work.push(Work::Function(callee));
                }
                work.extend(call.args.iter().map(Work::Expression));
            }
            Work::Source(crate::ast::SourceType::Timer(expr)) => work.push(Work::Expression(expr)),
            Work::Source(_) => {},
        }
    }
    Ok(())
}

pub fn validate_runtime_arguments(ops: &[Operation], program: &Program) -> Result<(), String> {
    let mut declared = HashMap::new();
    let mut shadowing = HashSet::new();
    declarations(ops, None, &mut declared, &mut shadowing);
    let context = Calls {
        program,
        declared,
        shadowing,
    };
    context.operations(ops, Effects::default())
}

#[derive(Clone)]
struct Signature {
    params: Vec<String>,
    effects: Effects,
    returns_result: bool,
}

fn declarations(
    ops: &[Operation],
    module: Option<&str>,
    out: &mut HashMap<String, Signature>,
    shadowing: &mut HashSet<String>,
) {
    for op in ops {
        match op {
            Operation::Module(decl) => declarations(&decl.body, Some(&decl.name), out, shadowing),
            Operation::Function(decl) => {
                shadowing.insert(decl.name.clone());
                if let Some(module) = module {
                    out.insert(
                        format!("{module}.{}", decl.name),
                        Signature {
                            params: decl.params.iter().map(|p| p.name.clone()).collect(),
                            effects: Effects::from_names(&decl.effects),
                            returns_result: matches!(&decl.return_type, crate::parser::TypeExpr::Generic { name, .. } if name == "Result"),
                        },
                    );
                }
            }
            Operation::Type(decl) => {
                if let crate::parser::TypeBody::Union(variants) = &decl.body {
                    shadowing.extend(variants.iter().map(|variant| variant.name.clone()));
                }
            }
            _ => {}
        }
    }
}

struct Calls<'a> {
    program: &'a Program,
    declared: HashMap<String, Signature>,
    shadowing: HashSet<String>,
}

impl Calls<'_> {
    fn signature(&self, target: &str) -> Result<Signature, String> {
        let signature = self.declared.get(target).cloned().or_else(|| {
            self.program.get_function(target).map(|function| Signature {
                params: function.params.iter().map(|p| p.name.clone()).collect(),
                effects: Effects { asynchronous: function.effects.contains(&crate::ast::Effect::Async), io: function.effects.contains(&crate::ast::Effect::Io) },
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

    fn expression(&self, expr: &Expr, caller: Effects) -> Result<(), String> {
        self.expression_mode(expr, Site::Ordinary, caller)
    }

    fn expression_mode(&self, expr: &Expr, site: Site, caller: Effects) -> Result<(), String> {
        if matches!(site, Site::AwaitBinding | Site::SpawnTarget)
            && !matches!(expr, Expr::Call { .. })
        {
            return Err("Python await/spawn requires a direct function call".into());
        }
        match expr {
            Expr::Call { callee, args } => {
                let target =
                    qualified(callee).ok_or("Python dynamic call targets are unsupported")?;
                let kind = if target.contains('.') {
                    self.positional(&target, args.len())?;
                    let signature = self.signature(&target)?;
                    Target::User(Some(UserSignature {
                        effects: signature.effects,
                    }))
                } else {
                    Target::BuiltinOrConstructor {
                        io: crate::builtins::is_io_builtin(&target),
                    }
                };
                call_sites::check(site, kind, caller)
                    .map_err(|message| format!("Python call `{target}`: {message}"))?;
                if matches!(site, Site::DirectBuiltinMatch | Site::AwaitBinding)
                    && matches!(kind, Target::BuiltinOrConstructor { io: true })
                    && (self.shadowing.contains(&target)
                        || call_sites::runtime_shadowed(self.program, &target))
                {
                    return Err(format!(
                        "Python IO call `{target}` requires an unshadowed runtime builtin"
                    ));
                }
                args.iter().try_for_each(|arg| self.expression(arg, caller))
            }
            Expr::Binary { left, right, .. } => {
                self.expression(left, caller)?;
                self.expression(right, caller)
            }
            Expr::Unary { expr, .. } | Expr::Cast { expr, .. } => self.expression(expr, caller),
            Expr::FieldAccess { base, .. } => self.expression(base, caller),
            Expr::StructLiteral(fields) => fields
                .iter()
                .try_for_each(|field| self.expression(&field.value, caller)),
            Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) | Expr::String(_) | Expr::Ident(_) => {
                Ok(())
            }
        }
    }

    fn operations(&self, ops: &[Operation], caller: Effects) -> Result<(), String> {
        for op in ops {
            use Operation::*;
            match op {
                Module(decl) => self.operations(&decl.body, Effects::default())?,
                Function(decl) | Startup(decl) | Shutdown(decl) => {
                    self.operations(&decl.body, Effects::from_names(&decl.effects))?
                }
                Test(test) => {
                    for case in &test.cases {
                        self.input(&test.function_name, &case.input)?;
                        self.expression(&case.input, Effects::default())?;
                        self.expression(&case.expected, Effects::default())?;
                    }
                }
                Eval(eval) => {
                    self.input(&eval.function_name, &eval.input)?;
                    self.expression(&eval.input, Effects::default())?;
                }
                Let(value) => self.expression(&value.expr, caller)?,
                Set(value) => self.expression(&value.expr, caller)?,
                Call(value) => self.expression(&value.expr, caller)?,
                Check(value) => self.expression(&value.expr, caller)?,
                Return(value) => self.expression(&value.expr, caller)?,
                Await(value) => {
                    self.expression_mode(&value.call, Site::AwaitBinding, caller)?;
                    // Runtime-signature defense: a later .ax edit must not turn a stored
                    // plain-T await into an implicit Result unwrap. Payload type checking
                    // remains in strict lowering; this visitor pins the outer contract.
                    if let Expr::Call { callee, .. } = &value.call {
                        if let Some(target) = qualified(callee).filter(|name| name.contains('.')) {
                            let returns_result = self.signature(&target)?.returns_result;
                            let binds_result = matches!(&value.ty, crate::parser::TypeExpr::Generic { name, .. } if name == "Result");
                            if returns_result != binds_result {
                                return Err(if returns_result {
                                    format!("Python await of `{target}` preserves Result; use a Result binding, not plain T")
                                } else {
                                    format!("Python await of `{target}` returns a non-Result; a Result binding requires a Result-returning user function")
                                });
                            }
                        }
                    }
                }
                Spawn(value) => self.expression_mode(&value.call, Site::SpawnTarget, caller)?,
                Match(value) => {
                    self.expression_mode(&value.expr, Site::DirectBuiltinMatch, caller)?;
                    for arm in &value.arms {
                        self.operations(&arm.body, caller)?;
                    }
                }
                If(branch) => {
                    self.expression(&branch.condition, caller)?;
                    self.operations(&branch.then_body, caller)?;
                    self.operations(&branch.else_body, caller)?;
                    for (condition, body) in &branch.elif_branches {
                        self.expression(condition, caller)?;
                        self.operations(body, caller)?;
                    }
                }
                While(loop_) => {
                    self.expression(&loop_.condition, caller)?;
                    self.operations(&loop_.body, caller)?;
                }
                Each(loop_) => {
                    self.expression(&loop_.collection, caller)?;
                    self.operations(&loop_.body, caller)?;
                }
                SharedVar(value) => self.expression(&value.default, Effects::default())?,
                EventEmit(value) => self.expression(&value.value, caller)?,
                // These contain metadata, not direct executable expressions.
                Type(_) | Doc { .. } | Route { .. } | ModuleSource(_) | EventRegister(_) => {}
                // Encoded source timer/handler execution and legacy commands
                // have no checked Python call-site surface in this validator.
                SourceAdd(_)
                | SourceReplace(_)
                | SourceRemove(_)
                | SourceList
                | Branch(_)
                | Replace(_)
                | Trace(_)
                | Move { .. }
                | Plan(_)
                | Roadmap(_)
                | Remove(_)
                | RemoveRoute { .. }
                | Watch { .. }
                | Undo
                | Agent { .. }
                | OpenCode(_)
                | Message { .. }
                | Mock { .. }
                | Unmock
                | Stub { .. }
                | Unstub
                | Done
                | Query(_)
                | Sandbox(_) => {
                    return Err("unsupported executable operation in Python call validation".into());
                }
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
