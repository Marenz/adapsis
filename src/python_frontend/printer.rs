//! Python inspection is derived exclusively from the live runtime AST.
use super::{
    Diagnostic,
    expressions::{Signature, Ty},
    types::TypeCatalog,
};
use crate::{ast as a, parser as p};
use std::collections::HashMap;
use super::call_sites::runtime_shadowed as shadowed;

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
    let mut printer = Printer::new(program);
    printer.set_module(module);
    let text = printer.function(f).map_err(&error)?;
    let mut signatures = HashMap::new();
    collect_signatures(&program.modules, "", &mut signatures);
    let types = printer.catalog();
    // This type environment comes only from the supplied AST. Native callers
    // pass authority-filtered signatures, as they do for final mutation lowering.
    let shared = super::types::module_shared_vars(program, module);
    let lowered = super::lower::module_with_signatures(&text, module, signatures, &types, &shared, true)
        .map_err(|e| error(e.message))?;
    let reconstructed = reconstruct_function(program, module, member, &lowered.operations)
        .map_err(&error)?;
    if !super::ast_equality::function(f, &reconstructed) {
        return Err(error(
            "Python roundtrip is not exact (only typed NodeIds may differ; callee spelling, effects, documentation and tests must be preserved)".into(),
        ));
    }
    Ok(text)
}

fn reconstruct_function(
    program: &a::Program,
    module: &str,
    member: &str,
    operations: &[p::Operation],
) -> Result<a::FunctionDecl, String> {
    // Model actual declaration replacement in the supplied authority-filtered
    // program. The validator deliberately retains attached tests on replacement;
    // compare that complete result, not a synthetic fresh function with tests
    // removed. The corpus separately checks fresh source/test reconstruction.
    let mut fresh = program.clone();
    for operation in operations {
        crate::validator::apply_and_validate(&mut fresh, operation).map_err(|e| e.to_string())?;
    }
    fresh.get_function(&format!("{module}.{member}")).cloned()
        .ok_or_else(|| "function missing after Python re-lowering".to_string())
}

pub fn print_type(program: &a::Program, target: &str) -> Result<String, Diagnostic> {
    let error = |message: String| Diagnostic {
        module: target.into(),
        line: 1,
        column: 1,
        message: format!(
            "cannot print {target} as Python: {message}; inspect Adapsis source instead"
        ),
    };
    let (module, decl) = find_type(program, target).ok_or_else(|| error("type not found".into()))?;
    let printer = Printer::new(program);
    let text = printer.type_decl(&decl).map_err(&error)?;
    let mut signatures = HashMap::new();
    collect_signatures(&program.modules, "", &mut signatures);
    let types = printer.catalog();
    let shared = super::types::module_shared_vars(program, &module);
    let lowered = super::lower::module_with_signatures(&text, &module, signatures, &types, &shared, true)
        .map_err(|e| error(e.message))?;
    let [p::Operation::Module(m)] = lowered.operations.as_slice() else {
        return Err(error("internal module roundtrip mismatch".into()));
    };
    let [p::Operation::Type(lowered_type)] = m.body.as_slice() else {
        return Err(error("internal type roundtrip mismatch".into()));
    };
    let converted = crate::validator::convert_type_decl(lowered_type)
        .map_err(|e| error(e.to_string()))?;
    if !super::ast_equality::ty(&decl, &converted) {
        return Err(error(
            "unsupported AST semantics (Python roundtrip changed the type)".into(),
        ));
    }
    Ok(text)
}

pub fn print_module(program: &a::Program, module_name: &str) -> Result<String, Diagnostic> {
    let error = |message: String| Diagnostic {
        module: module_name.into(),
        line: 1,
        column: 1,
        message: format!(
            "cannot print module {module_name} as Python: {message}; inspect Adapsis source instead"
        ),
    };
    let module = program
        .modules
        .iter()
        .find(|m| m.name == module_name)
        .ok_or_else(|| error("module not found".into()))?
        .clone();
    let mut printer = Printer::new(program);
    printer.set_module(module_name);
    let text = printer.module(&module).map_err(&error)?;
    let mut signatures = HashMap::new();
    collect_signatures(&program.modules, "", &mut signatures);
    let types = printer.catalog();
    let shared = super::types::module_shared_vars(program, module_name);
    let lowered = super::lower::module_with_signatures(&text, module_name, signatures, &types, &shared, true)
        .map_err(|e| error(e.message))?;
    let mut fresh = a::Program::default();
    for op in &lowered.operations {
        crate::validator::apply_and_validate(&mut fresh, op).map_err(|e| error(e.to_string()))?;
    }
    let rebuilt = fresh
        .modules
        .iter()
        .find(|m| m.name == module_name)
        .ok_or_else(|| error("internal module roundtrip mismatch".into()))?;
    if !super::ast_equality::module(&module, rebuilt) {
        return Err(error(
            "unsupported AST semantics (Python roundtrip changed the module)".into(),
        ));
    }
    Ok(text)
}

/// Resolve a type target (`TypeName` or `Module.TypeName`) to its owning module
/// name and declaration. Bare names search every module (recursively) and the
/// top-level type list.
fn find_type(program: &a::Program, target: &str) -> Option<(String, a::TypeDecl)> {
    if let Some((module, name)) = target.split_once('.') {
        for m in &program.modules {
            if m.name == module {
                return find_type_in_module(&[m.clone()], name).map(|td| (module.to_string(), td));
            }
        }
        return None;
    }
    for td in &program.types {
        if td.name() == target {
            return Some((target.to_string(), td.clone()));
        }
    }
    find_type_in_module(&program.modules, target).map(|td| (td.name().to_string(), td))
}

fn find_type_in_module(modules: &[a::Module], name: &str) -> Option<a::TypeDecl> {
    for module in modules {
        for td in &module.types {
            if td.name() == name {
                return Some(td.clone());
            }
        }
        if let Some(td) = find_type_in_module(&module.modules, name) {
            return Some(td);
        }
    }
    None
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
                    .map(|p| Ok::<_, String>(super::types::ast_type_to_ty_in(&p.ty, &path)))
                    .collect::<Result<Vec<_>, _>>(),
                Ok::<_, String>(super::types::ast_type_to_ty_in(&f.return_type, &path)),
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

fn type_text(ty: &a::Type) -> Result<String, String> {
    Ok(super::types::type_to_python(&super::types::ast_type_to_ty(ty)))
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
                a::Effect::Rand => "rand",
                a::Effect::Yield => "yield",
                a::Effect::Parallel => "parallel",
                a::Effect::Unsafe => "unsafe",
            }
            .into())
        })
        .collect()
}

struct Printer<'a> {
    program: &'a a::Program,
    types: TypeCatalog,
    env: HashMap<String, a::Type>,
    return_type: Option<a::Type>,
    current_module: String,
}

impl<'a> Printer<'a> {
    fn new(program: &'a a::Program) -> Self {
        Printer {
            program,
            types: super::types::catalog_from_program(program),
            env: HashMap::new(),
            return_type: None,
            current_module: String::new(),
        }
    }

    fn catalog(&self) -> TypeCatalog {
        self.types.clone()
    }

    fn set_module(&mut self, module: &str) {
        self.current_module = module.to_string();
    }

    fn function(&mut self, f: &a::FunctionDecl) -> Result<String, String> {
        self.env.clear();
        self.return_type = Some(f.return_type.clone());
        for param in &f.params {
            self.env.insert(param.name.clone(), param.ty.clone());
        }
        collect_env(&f.body, &mut self.env);
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

    fn type_decl(&self, td: &a::TypeDecl) -> Result<String, String> {
        let mut out = String::new();
        match td {
            a::TypeDecl::Struct(s) => {
                out.push_str("@dataclass\n");
                out.push_str(&format!("class {}:\n", super::emit::name(&s.name)?));
                for field in &s.fields {
                    line(
                        &mut out,
                        1,
                        format!(
                            "{}: {}",
                            super::emit::name(&field.name)?,
                            type_text(&field.ty)?
                        ),
                    );
                }
            }
            a::TypeDecl::TaggedUnion(u) => {
                out.push_str("@union\n");
                out.push_str(&format!("class {}:\n", super::emit::name(&u.name)?));
                for variant in &u.variants {
                    if variant.payload.is_empty() {
                        line(
                            &mut out,
                            1,
                            format!("{}: tuple[()]", variant_name_python(&variant.name)?),
                        );
                    } else {
                        let payload = variant
                            .payload
                            .iter()
                            .map(type_text)
                            .collect::<Result<Vec<_>, String>>()?
                            .join(", ");
                        line(
                            &mut out,
                            1,
                            format!(
                                "{}: tuple[{payload}]",
                                variant_name_python(&variant.name)?
                            ),
                        );
                    }
                }
            }
        }
        Ok(out)
    }

    fn module(&mut self, m: &a::Module) -> Result<String, String> {
        self.env.clear();
        self.return_type = None;
        for sv in &m.shared_vars {
            self.env.insert(sv.name.clone(), sv.ty.clone());
        }
        let mut out = String::new();
        if let Some(doc) = &m.doc {
            line(&mut out, 0, super::emit::quote(doc)?);
        }
        for ty in &m.types {
            out.push_str(&self.type_decl(ty)?);
        }
        for sv in &m.shared_vars {
            line(
                &mut out,
                0,
                format!(
                    "{}: {} = {}",
                    super::emit::name(&sv.name)?,
                    type_text(&sv.ty)?,
                    self.expr(&sv.default, None)?
                ),
            );
        }
        if let Some(startup) = &m.startup {
            out.push_str(&self.lifecycle_block("startup", startup)?);
        }
        if let Some(shutdown) = &m.shutdown {
            out.push_str(&self.lifecycle_block("shutdown", shutdown)?);
        }
        for f in &m.functions {
            let qualified = format!("{}.{}", m.name, f.name);
            if let Some(route) = m.routes.iter().find(|r| r.handler_fn == qualified) {
                line(
                    &mut out,
                    0,
                    format!(
                        "@route({}, {})",
                        super::emit::quote(&route.method)?,
                        super::emit::quote(&route.path)?
                    ),
                );
            }
            if let Some(source) = m.sources.iter().find(|s| s.handler == f.name) {
                line(&mut out, 0, self.source_decorator(source)?);
            }
            out.push_str(&self.function(f)?);
        }
        Ok(out)
    }

    fn lifecycle_block(&self, name: &str, block: &a::LifecycleBlock) -> Result<String, String> {
        let mut out = String::new();
        let effects = effect_names(&block.effects)?;
        let decorated = effects
            .iter()
            .filter(|e| e.as_str() != "async")
            .map(|e| super::emit::quote(e))
            .collect::<Result<Vec<_>, _>>()?;
        if !decorated.is_empty() {
            out.push_str(&format!("@effects({})\n", decorated.join(", ")));
        }
        let asynchronous = if effects.iter().any(|e| e == "async") { "async " } else { "" };
        out.push_str(&format!("{asynchronous}def {name}() -> None:\n"));
        self.block(&block.body, 1, &mut out)?;
        Ok(out)
    }

    fn source_decorator(&self, source: &a::SourceDecl) -> Result<String, String> {
        let mut config = String::new();
        if !source.source_type.is_empty() {
            config.push_str(&format!("type={}", super::emit::quote(&source.source_type)?));
        }
        for (key, value) in &source.config {
            config.push_str(&format!(", {key}={}", super::emit::quote(value)?));
        }
        Ok(format!(
            "@source({}, {config})",
            super::emit::quote(&source.name)?
        ))
    }

    fn block(&self, body: &[a::Statement], indent: usize, out: &mut String) -> Result<(), String> {
        if body.is_empty() {
            line(out, indent, "pass");
            return Ok(());
        }
        for stmt in body {
            self.statement(&stmt.kind, indent, out)?;
        }
        Ok(())
    }

    fn statement(&self, stmt: &a::StatementKind, i: usize, out: &mut String) -> Result<(), String> {
        use a::StatementKind::*;
        match stmt {
            Let { name, ty, value } => line(
                out,
                i,
                format!(
                    "{}: {} = {}",
                    super::emit::name(name)?,
                    type_text(ty)?,
                    self.expr(value, Some(ty))?
                ),
            ),
            Set { name, value } => line(
                out,
                i,
                format!(
                    "{} = {}",
                    super::emit::name(name)?,
                    self.expr(value, self.env.get(name))?
                ),
            ),
            Return { value } => line(
                out,
                i,
                format!(
                    "return {}",
                    self.expr(value, self.return_type.as_ref())?
                ),
            ),
            Await { name, ty, call } => line(
                out,
                i,
                format!(
                    "{}: {} = await {}",
                    super::emit::name(name)?,
                    type_text(ty)?,
                    self.call(call, None)?
                ),
            ),
            Branch { condition, then_body, else_body } => {
                line(out, i, format!("if {}:", self.expr(condition, None)?));
                self.block(then_body, i + 1, out)?;
                if !else_body.is_empty() { line(out, i, "else:"); self.block(else_body, i + 1, out)?; }
            }
            While { condition, body } => {
                line(out, i, format!("while {}:", self.expr(condition, None)?));
                self.block(body, i + 1, out)?;
            }
            Each { iterator, binding, body } => {
                line(out, i, format!("for {} in {}:", super::emit::name(&binding.name)?, self.expr(iterator, None)?));
                self.block(body, i + 1, out)?;
            }
            Call { .. } | Spawn { .. } | Check { .. } => self.call_check_spawn(stmt, i, out)?,
            Match { expr, arms } => {
                self.check_match_printable(expr, arms)?;
                line(out, i, format!("match {}:", self.expr(expr, None)?));
                for arm in arms {
                    line(out, i + 1, format!("case {}:", self.case_pattern(arm)?));
                    self.block(&arm.body, i + 2, out)?;
                }
            }
            Yield { .. } => return Err("unsupported yield statement".into()),
            Source(op) => self.source_statement(op, i, out)?,
            Event(op) => self.event_statement(op, i, out)?,
        }
        Ok(())
    }

    fn call_check_spawn(
        &self,
        stmt: &a::StatementKind,
        i: usize,
        out: &mut String,
    ) -> Result<(), String> {
        match stmt {
            a::StatementKind::Call { binding, call } => {
                let Some(binding) = binding else {
                    return Err("+call statement without a binding cannot be printed".into());
                };
                line(
                    out,
                    i,
                    format!(
                        "{}: {} = propagate({})",
                        super::emit::name(&binding.name)?,
                        type_text(&binding.ty)?,
                        self.call(call, None)?
                    ),
                );
            }
            a::StatementKind::Spawn { call, binding } => match binding {
                Some(binding) => line(
                    out,
                    i,
                    format!(
                        "{}: {} = spawn({})",
                        super::emit::name(&binding.name)?,
                        type_text(&binding.ty)?,
                        self.call(call, None)?
                    ),
                ),
                None => line(out, i, format!("spawn({})", self.call(call, None)?)),
            },
            a::StatementKind::Check { label, condition, on_fail } => line(
                out,
                i,
                format!(
                    "check({}, {}, {})",
                    super::emit::quote(label)?,
                    self.expr(condition, None)?,
                    super::emit::quote(on_fail)?
                ),
            ),
            _ => unreachable!("call_check_spawn only handles call/spawn/check"),
        }
        Ok(())
    }

    fn source_statement(
        &self,
        op: &a::SourceOp,
        i: usize,
        out: &mut String,
    ) -> Result<(), String> {
        match op {
            a::SourceOp::Add { source_type, alias, handler } => line(
                out,
                i,
                format!(
                    "source_add({}, {}, {})",
                    super::emit::quote(&self.source_type_text(source_type)?)?,
                    super::emit::quote(alias)?,
                    super::emit::quote(handler)?
                ),
            ),
            a::SourceOp::Remove { alias } => line(
                out,
                i,
                format!("source_remove({})", super::emit::quote(alias)?),
            ),
            a::SourceOp::Replace { alias, source_type, handler } => line(
                out,
                i,
                format!(
                    "source_replace({}, {}, {})",
                    super::emit::quote(alias)?,
                    super::emit::quote(&self.source_type_text(source_type)?)?,
                    super::emit::quote(handler)?
                ),
            ),
            a::SourceOp::List => line(out, i, "source_list()"),
        }
        Ok(())
    }

    fn event_statement(
        &self,
        op: &a::EventOp,
        i: usize,
        out: &mut String,
    ) -> Result<(), String> {
        match op {
            a::EventOp::Register { name, payload_type } => line(
                out,
                i,
                format!(
                    "event_register({}, {})",
                    super::emit::quote(name)?,
                    super::emit::quote(payload_type)?
                ),
            ),
            a::EventOp::Emit { name, value } => line(
                out,
                i,
                format!(
                    "event_emit({}, {})",
                    super::emit::quote(name)?,
                    self.expr(value, None)?
                ),
            ),
        }
        Ok(())
    }

    fn source_type_text(&self, ty: &a::SourceType) -> Result<String, String> {
        match ty {
            a::SourceType::Timer(expr) => Ok(format!("timer({})", self.expr(expr, None)?)),
            a::SourceType::Channel => Ok("channel".into()),
            a::SourceType::Event(module, event) => Ok(format!("{module}.{event}")),
        }
    }

    /// A `.ax` match is only printable as Python `match` when it is exhaustive:
    /// it has a `_` arm, or (when the subject's declared type is known) it covers
    /// every case. Non-exhaustive matches bail at runtime in Adapsis, a semantic
    /// Python cannot express, so they are refused.
    fn check_match_printable(&self, expr: &a::Expr, arms: &[a::MatchArm]) -> Result<(), String> {
        if arms.iter().any(|arm| arm.variant == "_") {
            return Ok(());
        }
        let a::Expr::Identifier(name) = expr else {
            return Ok(());
        };
        let Some(subject_ty) = self.env.get(name) else {
            return Ok(());
        };
        let covered: std::collections::HashSet<&str> =
            arms.iter().map(|arm| arm.variant.as_str()).collect();
        let required: Vec<String> = match subject_ty {
            a::Type::Result(_) => vec!["Ok".into(), "Err".into()],
            a::Type::Option(_) => vec!["Some".into(), "None".into()],
            a::Type::TaggedUnion(name) => match super::types::resolve_name(&self.types, &self.current_module, name)?
                .as_ref().and_then(|key| self.types.get(key)) {
                Some(super::types::TypeInfo::Union { variants, .. }) => {
                    variants.iter().map(|(name, _)| name.clone()).collect()
                }
                _ => return Ok(()),
            },
            _ => return Ok(()),
        };
        let missing: Vec<&str> = required
            .iter()
            .filter(|name| !covered.contains(name.as_str()))
            .map(|name| name.as_str())
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        Err(format!(
            "non-exhaustive match has no Python form (missing case(s): {}); inspect Adapsis source instead",
            missing.join(", ")
        ))
    }

    fn expr(&self, e: &a::Expr, hint: Option<&a::Type>) -> Result<String, String> {
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
            Identifier(n) => super::emit::bare_name(n),
            Call(c) => self.call(c, hint),
            Binary { op: a::BinaryOp::Div, left, right } => Ok(format!(
                "div({}, {})",
                self.expr(left, None)?,
                self.expr(right, None)?
            )),
            Binary { op: a::BinaryOp::Mod, left, right } => Ok(format!(
                "rem({}, {})",
                self.expr(left, None)?,
                self.expr(right, None)?
            )),
            Binary { op, left, right } => Ok(format!(
                "({} {} {})",
                self.expr(left, None)?,
                binary(op)?,
                self.expr(right, None)?
            )),
            Unary { op, expr } => Ok(format!(
                "({}{})",
                match op {
                    a::UnaryOp::Not => "not ",
                    a::UnaryOp::Neg => "-",
                },
                self.expr(expr, None)?
            )),
            FieldAccess { base, field } => Ok(format!(
                "{}.{}",
                self.expr(base, None)?,
                super::emit::name(field)?
            )),
            StructInit { ty, fields } => self.struct_init(ty, fields, hint),
        }
    }

    fn struct_init(
        &self,
        ty: &str,
        fields: &[a::StructFieldValue],
        hint: Option<&a::Type>,
    ) -> Result<String, String> {
        let name = if !ty.is_empty() {
            ty.to_string()
        } else if let Some(name) = hint.and_then(struct_name) {
            name.to_string()
        } else {
            return Err(
                "struct literal has no type name and no enclosing struct/return type context".into(),
            );
        };
        let key = super::types::resolve_name(&self.types, &self.current_module, &name)?
            .ok_or_else(|| format!("struct type {} is not declared in the supplied catalog", super::types::bare_name(&name)))?;
        let bare = super::types::bare_name(&name);
        let printable = super::types::resolve_name(&self.types, &self.current_module, bare)?;
        if printable.as_ref() != Some(&key) {
            return Err(format!("struct type {bare} belongs to another declaration owner; its bare constructor has no exact form in this module"));
        }
        let mut args = Vec::new();
        for field in fields {
            let field_ty = super::types::struct_field_type(&self.types, &Ty::Named(key.clone()), &field.name)?;
            let value = match field_ty {
                Some(ty) => self.expr(&field.value, Some(&super::types::ty_to_ast(&ty)))?,
                None => self.expr(&field.value, None)?,
            };
            args.push(format!(
                "{}={}",
                super::emit::name(&field.name)?,
                value
            ));
        }
        Ok(format!("{}({})", super::emit::name(bare)?, args.join(", ")))
    }

    fn call(&self, c: &a::CallExpr, hint: Option<&a::Type>) -> Result<String, String> {
        let name = super::emit::name(&c.callee)?;
        if !name.contains('.') {
            match super::types::variant_payloads(&self.types, &name, &self.current_module) {
                Ok(Some((_union, payload))) => {
                // A declared union variant. Variant priority resolves `Some(x)` to
                // the variant even though `Some` is also a builtin, so constructor
                // builtins are printable; a non-constructor builtin shadowed by a
                // variant (e.g. `list`) is ambiguous and refused.
                if crate::builtins::is_builtin(&name) && !constructor_builtin(&name) {
                    return Err(format!(
                        "builtin/helper {name} is shadowed by a runtime function or union variant"
                    ));
                }
                let args = c
                    .args
                    .iter()
                    .zip(payload.into_iter().map(|ty| super::types::ty_to_ast(&ty)).chain(std::iter::repeat(a::Type::String)))
                    .map(|(e, ty)| self.expr(e, Some(&ty)))
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok(format!(
                    "{}({})",
                    super::emit::bare_name(&name)?,
                    args.join(", ")
                ));
                }
                Ok(None) => {}
                Err(message) => return Err(message),
            }
            if !crate::builtins::is_builtin(&name) {
                return Err(format!("unqualified user call {name} has no exact Python roundtrip surface; qualification could change runtime resolution"));
            }
            if shadowed(self.program, &name) {
                return Err(format!(
                    "builtin/helper {name} is shadowed by a runtime function or union variant"
                ));
            }
            if constructor_builtin(&name) {
                // Ok/Err/Some unwrap the enclosing Result/Option hint for their
                // single payload, so `Ok(Point(x=1))` prints the struct name.
                let inner = match (name.as_str(), hint) {
                    ("Ok" | "Err", Some(a::Type::Result(inner))) => Some(inner.as_ref()),
                    ("Some", Some(a::Type::Option(inner))) => Some(inner.as_ref()),
                    _ => None,
                };
                let args = c
                    .args
                    .iter()
                    .map(|e| self.expr(e, inner))
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok(format!("{name}({})", args.join(", ")));
            }
        }
        let hints = self.param_type_hints(&name, c.args.len())?;
        let args = c
            .args
            .iter()
            .zip(hints)
            .map(|(e, hint)| self.expr(e, hint.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        if name == "list" {
            return Ok(format!("[{}]", args.join(", ")));
        }
        Ok(format!("{name}({})", args.join(", ")))
    }

    /// Struct/union parameter types for a callee's positions, so a struct-literal
    /// argument can be printed with the right constructor name.
    fn param_type_hints(&self, callee: &str, arg_count: usize) -> Result<Vec<Option<a::Type>>, String> {
        let mut hints = Vec::new();
        if let Some((owner, _)) = callee.rsplit_once('.') {
            if let Some(function) = self.program.get_function(callee) {
                hints = function.params.iter().map(|param| {
                    let ty = super::types::ast_type_to_ty_in(&param.ty, owner);
                    super::types::resolve_ty(&self.types, &ty).map(|ty| Some(super::types::ty_to_ast(&ty)))
                }).collect::<Result<Vec<_>, _>>()?;
            }
        }
        hints.resize(arg_count, None);
        Ok(hints)
    }

    fn case_pattern(&self, arm: &a::MatchArm) -> Result<String, String> {
        if arm.variant == "_" {
            return Ok("_".into());
        }
        if arm.variant == "None" && arm.bindings.is_empty() && arm.patterns.is_none() {
            return Ok("None".into());
        }
        match &arm.patterns {
            None => {
                if arm.bindings.is_empty() {
                    Ok(format!("{}()", super::emit::bare_name(&arm.variant)?))
                } else {
                    Ok(format!(
                        "{}({})",
                        super::emit::bare_name(&arm.variant)?,
                        arm.bindings
                            .iter()
                            .map(|b| super::emit::bare_name(b))
                            .collect::<Result<Vec<_>, _>>()?
                            .join(", ")
                    ))
                }
            }
            Some(patterns) => Ok(format!(
                "{}({})",
                super::emit::bare_name(&arm.variant)?,
                patterns
                    .iter()
                    .map(|p| self.pattern(p))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(", ")
            )),
        }
    }

    fn pattern(&self, pattern: &a::MatchPattern) -> Result<String, String> {
        match pattern {
            a::MatchPattern::Binding(name) => super::emit::bare_name(name),
            a::MatchPattern::Literal(a::Literal::Int(n)) => {
                if *n == i64::MIN {
                    return Err("i64::MIN literal pattern cannot be represented in Python".into());
                }
                Ok(n.to_string())
            }
            a::MatchPattern::Literal(a::Literal::Bool(b)) => Ok(if *b { "True" } else { "False" }.into()),
            a::MatchPattern::Literal(a::Literal::String(s)) => super::emit::quote(s),
            a::MatchPattern::Literal(_) => Err("unsupported literal pattern".into()),
            a::MatchPattern::Variant {
                variant,
                sub_patterns,
            } => {
                if sub_patterns.is_empty() {
                    Ok(format!("{}()", super::emit::bare_name(variant)?))
                } else {
                    Ok(format!(
                        "{}({})",
                        super::emit::bare_name(variant)?,
                        sub_patterns
                            .iter()
                            .map(|p| self.pattern(p))
                            .collect::<Result<Vec<_>, _>>()?
                            .join(", ")
                    ))
                }
            }
        }
    }
}

fn struct_name(ty: &a::Type) -> Option<&str> {
    match ty {
        a::Type::Struct(name) | a::Type::TaggedUnion(name) => Some(name.as_str()),
        _ => None,
    }
}

/// The Result/Option constructor builtins are legitimately usable as union
/// variant names (`Some` appears in `Maybe`/`Option`); variant priority makes
/// `Some(x)` the variant, so they are exempt from the variant-shadow refusal.
fn constructor_builtin(name: &str) -> bool {
    matches!(name, "Some" | "Ok" | "Err")
}

/// Python spelling of a union variant name: Python keywords use the PEP 8
/// trailing-underscore convention (`None` → `None_`). A variant literally named
/// `None_` (only reachable from `.ax`) cannot be represented and is refused.
fn variant_name_python(value: &str) -> Result<String, String> {
    match value {
        "None" => Ok("None_".into()),
        "True" => Ok("True_".into()),
        "False" => Ok("False_".into()),
        "None_" | "True_" | "False_" => Err(format!(
            "variant name {value:?} collides with the keyword trailing-underscore convention"
        )),
        _ => super::emit::bare_name(value),
    }
}

fn collect_env(body: &[a::Statement], env: &mut HashMap<String, a::Type>) {
    for stmt in body {
        match &stmt.kind {
            a::StatementKind::Let { name, ty, .. }
            | a::StatementKind::Await { name, ty, .. } => {
                env.insert(name.clone(), ty.clone());
            }
            a::StatementKind::Each { binding, body, .. } => {
                env.insert(binding.name.clone(), binding.ty.clone());
                collect_env(body, env);
            }
            a::StatementKind::Branch { then_body, else_body, .. } => {
                collect_env(then_body, env);
                collect_env(else_body, env);
            }
            a::StatementKind::While { body, .. } => collect_env(body, env),
            a::StatementKind::Match { arms, .. } => {
                for arm in arms {
                    collect_env(&arm.body, env);
                }
            }
            _ => {}
        }
    }
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
