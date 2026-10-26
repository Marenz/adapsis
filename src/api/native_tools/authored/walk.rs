//! Lexical value paths are data; CALL targets always retain namespace checks.
use super::*;
use std::collections::HashSet;

pub(super) struct Checker<'a> {
    pub config: &'a AppConfig,
    pub model: &'a str,
    pub program: Option<Runtime<'a>>,
    pub roles: Option<&'a crate::python_frontend::NativeReferences>,
}

impl Checker<'_> {
    /// The declaration's OWN annotation resolves to a union in the authorized
    /// catalog; return that union's constructor spellings. The default's callee
    /// must be one of THEM, so a foreign, unknown or builtin-spelled constructor
    /// can never be admitted. Builtin spellings are allowed when they genuinely
    /// are a declared variant of this annotation.
    fn annotation_variants(&self, decl: &crate::parser::SharedVarDecl) -> Option<Vec<String>> {
        let crate::parser::TypeExpr::Named(name) = &decl.ty else {
            return None;
        };
        let roles = self.roles?;
        roles.union_variants(name)
    }

    fn reference(&self, target: &str, module: &str, call: bool) -> Result<(), String> {
        if self.roles.is_some_and(|r| r.shadows_builtin(target)) {
            return Err(crate::python_frontend::CONSTRUCTOR_ROLE_ERROR.into());
        }
        super::reference(
            target,
            self.config,
            self.model,
            self.program,
            call && self.roles.is_some_and(|r| r.constructor(module, target)),
        )
    }

    fn expression(&self, expr: &Expr, module: &str, scope: &HashSet<String>) -> Result<(), String> {
        match expr {
            Expr::Call { callee, args } => {
                let target =
                    name(callee).ok_or("native mutation does not support dynamic call targets")?;
                // Never apply the value-field exemption to a call target.
                self.reference(&target, module, true)?;
                for arg in args {
                    self.expression(arg, module, scope)?;
                }
            }
            Expr::FieldAccess { base, .. } => {
                if let Some(target) = name(expr) {
                    let root = target.split('.').next().unwrap();
                    if !scope.contains(root) {
                        self.reference(&target, module, false)?;
                    }
                }
                self.expression(base, module, scope)?;
            }
            Expr::Ident(value)
                if self.roles.is_some()
                    && !scope.contains(value)
                    && !matches!(value.as_str(), "true" | "false") =>
            {
                if !self.roles.is_some_and(|r| r.constructor(module, value)) {
                    return Err("Python value reference is unbound or is not a supported target-module constructor".into());
                }
            }
            Expr::Binary { left, right, .. } => {
                self.expression(left, module, scope)?;
                self.expression(right, module, scope)?;
            }
            Expr::Unary { expr, .. } | Expr::Cast { expr, .. } => {
                self.expression(expr, module, scope)?
            }
            Expr::StructLiteral(fields) => {
                for field in fields {
                    self.expression(&field.value, module, scope)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub fn operations(
        &self,
        ops: &[Operation],
        module: &str,
        scope: &mut HashSet<String>,
    ) -> Result<(), String> {
        for op in ops {
            match op {
                Operation::Module(decl) => {
                    let mut shared = self.roles.map(|r| r.shared(&decl.name)).unwrap_or_default();
                    shared.extend(decl.body.iter().filter_map(|op| match op {
                        Operation::SharedVar(v) => Some(v.name.clone()),
                        _ => None,
                    }));
                    self.operations(&decl.body, &decl.name, &mut shared)?;
                }
                Operation::Function(decl) => {
                    let mut local = scope.clone();
                    local.extend(decl.params.iter().map(|p| p.name.clone()));
                    self.operations(&decl.body, module, &mut local)?;
                }
                Operation::SharedVar(decl) => {
                    // The ONE position where a variant constructor is admitted:
                    // a data-only default whose callee is a variant of its own
                    // declared union. `variants` is empty otherwise, so every
                    // other call stays a strict literal error.
                    let variants = self.annotation_variants(decl).unwrap_or_default();
                    match self.program {
                        Some((program, _)) => super::super::literals::check_runtime_variant(
                            &decl.default,
                            program,
                            &variants,
                        )?,
                        // Authorization-only pass (no program yet): validate the
                        // ADMITTED SHAPE without the list-shadow lookup. The
                        // executor pass repeats this with the real program.
                        None => admitted_variant_shape(&decl.default, &variants)?,
                    }
                }
                Operation::Type(_) | Operation::Doc { .. } => {}
                Operation::Let(v) => {
                    self.expression(&v.expr, module, scope)?;
                    scope.insert(v.name.clone());
                }
                Operation::Call(v) => {
                    self.expression(&v.expr, module, scope)?;
                    scope.insert(v.name.clone());
                }
                Operation::Await(v) => {
                    self.expression(&v.call, module, scope)?;
                    scope.insert(v.name.clone());
                }
                Operation::Spawn(v) => {
                    self.expression(&v.call, module, scope)?;
                    if let Some((name, _)) = &v.binding {
                        scope.insert(name.clone());
                    }
                }
                Operation::Set(v) => {
                    self.expression(&v.expr, module, scope)?;
                    if v.name.contains('.') {
                        self.reference(&v.name, module, false)?;
                    }
                }
                Operation::Check(v) => self.expression(&v.expr, module, scope)?,
                Operation::Return(v) => self.expression(&v.expr, module, scope)?,
                Operation::If(v) => {
                    self.expression(&v.condition, module, scope)?;
                    self.operations(&v.then_body, module, &mut scope.clone())?;
                    self.operations(&v.else_body, module, &mut scope.clone())?;
                    for (condition, body) in &v.elif_branches {
                        self.expression(condition, module, scope)?;
                        self.operations(body, module, &mut scope.clone())?;
                    }
                }
                Operation::While(v) => {
                    self.expression(&v.condition, module, scope)?;
                    self.operations(&v.body, module, &mut scope.clone())?;
                }
                Operation::Each(v) => {
                    self.expression(&v.collection, module, scope)?;
                    let mut local = scope.clone();
                    local.insert(v.item.clone());
                    self.operations(&v.body, module, &mut local)?;
                }
                Operation::Match(v) => {
                    self.expression(&v.expr, module, scope)?;
                    for arm in &v.arms {
                        let mut local = scope.clone();
                        local.extend(arm.bindings.iter().cloned());
                        if let Some(patterns) = &arm.patterns {
                            for p in patterns {
                                captures(p, &mut local);
                            }
                        }
                        self.operations(&arm.body, module, &mut local)?;
                    }
                }
                Operation::Branch(v) => {
                    self.expression(&v.expr, module, scope)?;
                    for arm in &v.arms {
                        self.expression(&arm.pattern, module, scope)?;
                        self.reference(&arm.target, module, false)?;
                    }
                }
                _ => {
                    return Err(format!(
                        "native mutation does not support authored `{}` inside functions",
                        super::super::describe_op(op)
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Shape-only form of `literals::check_runtime_variant`, used when no program is
/// available (authorization pass). Mirrors it minus `list` shadowing.
fn admitted_variant_shape(expr: &Expr, variants: &[String]) -> Result<(), String> {
    match expr {
        Expr::Call { callee, args } if matches!(callee.as_ref(), Expr::Ident(name) if variants.iter().any(|v| v == name)) => {
            args.iter()
                .try_for_each(|arg| admitted_variant_shape(arg, variants))
        }
        // Zero-payload variant default: the bare constructor spelling.
        Expr::Ident(name) if variants.iter().any(|v| v == name) => Ok(()),
        other => literal(other),
    }
}

fn captures(pattern: &crate::parser::MatchPatternDecl, scope: &mut HashSet<String>) {
    match pattern {
        crate::parser::MatchPatternDecl::Binding(name) => {
            scope.insert(name.clone());
        }
        crate::parser::MatchPatternDecl::Variant { sub_patterns, .. } => {
            for p in sub_patterns {
                captures(p, scope);
            }
        }
        _ => {}
    }
}
