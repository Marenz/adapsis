use super::call_sites::Site;
use super::expressions::{Expressions, Signature, Ty, annotation, compatible};
use super::types::TypeCatalog;
use super::{Context, Diagnostic, LoweredSource};
use crate::parser as p;
use rustpython_parser::ast as py;
use std::collections::{HashMap, HashSet};

pub(super) fn module(source: &str, module: &str) -> Result<LoweredSource, Diagnostic> {
    module_with_signatures(
        source,
        module,
        HashMap::new(),
        &HashMap::new(),
        &HashMap::new(),
        false,
    )
}

pub(super) fn module_with_signatures(
    source: &str,
    module: &str,
    signatures: HashMap<String, Signature>,
    types: &TypeCatalog,
    program_shared: &HashMap<String, Ty>,
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
    let (doc, items) = collect_module_items(&cx, &suite, types)?;
    let catalog = collect_module_catalog(&cx, &items, signatures, types, program_shared)?;
    if require_resolved {
        let references = preflight(source, module)?;
        for reference in references.references {
            if !catalog.signatures.contains_key(&reference) {
                return Err(cx.error(0, format!("Python callable `{reference}` has no authorized supported signature; declare the target first")));
            }
        }
        for name in references.local_calls {
            let target = format!("{module}.{name}");
            if !catalog.signatures.contains_key(&target)
                && super::types::variant_payloads(&catalog.types, &name, module)
                    .map_err(|message| cx.error(0, message))?
                    .is_none()
            {
                return Err(cx.error(0, format!("Python callable `{target}` has no authorized supported signature; declare the target first")));
            }
        }
    }
    validate_shared_default_types(&cx, &items, &catalog)?;
    let body = lower_module_items(&cx, &items, &catalog)?;
    let operations = vec![p::Operation::Module(p::ModuleDecl {
        name: module.into(),
        body,
        doc,
    })];
    let source_map = suite.iter().map(|stmt| cx.span(stmt, 0)).collect();
    cx.finish(operations, source_map)
}

fn collect_module_items<'a>(
    cx: &Context<'_>,
    suite: &'a [py::Stmt],
    types: &TypeCatalog,
) -> Result<(Option<String>, Vec<ModuleItem<'a>>), Diagnostic> {
    let mut items = Vec::new();
    let mut doc = None;
    for (i, stmt) in suite.iter().enumerate() {
        if i == 0 {
            if let Some(text) = super::docstring(stmt) {
                doc = Some(text.to_owned());
                continue;
            }
        }
        items.push(ModuleItem::from_stmt(cx, stmt, types)?);
    }
    if items.is_empty() {
        return Err(cx.error(
            0,
            "module submission requires at least one annotated function, type or shared declaration",
        ));
    }
    Ok((doc, items))
}

struct ModuleCatalog {
    signatures: HashMap<String, Signature>,
    types: TypeCatalog,
    shared: HashMap<String, Ty>,
}

fn collect_module_catalog(
    cx: &Context<'_>,
    items: &[ModuleItem<'_>],
    mut signatures: HashMap<String, Signature>,
    types: &TypeCatalog,
    program_shared: &HashMap<String, Ty>,
) -> Result<ModuleCatalog, Diagnostic> {
    // Preserve the two-pass declaration/signature order, including current
    // catalog limitations. Owner-qualified type resolution is a separate step.
    let mut declared = HashSet::new();
    let mut local_types: TypeCatalog = types.clone();
    let mut shared_vars: HashMap<String, Ty> = program_shared.clone();
    for item in items {
        match item {
            ModuleItem::Type(decl) => {
                if !declared.insert(decl.name.clone()) {
                    return Err(cx.error(
                        0,
                        format!(
                            "duplicate type or function {}; submit each declaration once",
                            decl.name
                        ),
                    ));
                }
                insert_parser_type(decl, cx.module, &mut local_types);
            }
            ModuleItem::Shared(assign) => {
                let py::Expr::Name(n) = assign.target.as_ref() else {
                    return Err(cx.at(*assign, "shared variables require a simple name annotation"));
                };
                if !declared.insert(n.id.to_string()) {
                    return Err(cx.at(*assign, format!("duplicate shared var {}", n.id)));
                }
                let ty = super::expressions::annotation(cx, &assign.annotation, &local_types)?;
                shared_vars.insert(n.id.to_string(), ty);
            }
            _ => {}
        }
    }
    // Phase 2: signatures (functions + lifecycle blocks), with the full catalog.
    for item in items {
        match item {
            ModuleItem::Function(def) | ModuleItem::Startup(def) | ModuleItem::Shutdown(def) => {
                let signature = def.signature(cx, &local_types)?;
                if !declared.insert(def.name.to_owned()) {
                    return Err(cx.at(
                        def.stmt,
                        format!("duplicate function {}; submit it only once", def.name),
                    ));
                }
                signatures.insert(format!("{}.{}", cx.module, def.name), signature.clone());
                signatures.insert(def.name.to_owned(), signature);
            }
            _ => {}
        }
    }
    Ok(ModuleCatalog {
        signatures,
        types: local_types,
        shared: shared_vars,
    })
}

/// Slice 3: a shared default must match its declared annotation, checked against
/// the complete local type catalog (not a pre-catalog partial view).
fn validate_shared_default_types(
    cx: &Context<'_>,
    items: &[ModuleItem<'_>],
    catalog: &ModuleCatalog,
) -> Result<(), Diagnostic> {
    for item in items {
        let ModuleItem::Shared(assign) = item else {
            continue;
        };
        let Some(default) = assign.value.as_deref() else {
            return Err(cx.at(*assign, "shared variable requires a default value"));
        };
        let expected = super::expressions::annotation(cx, &assign.annotation, &catalog.types)?;
        let expected = super::types::resolve_ty(&catalog.types, &expected)
            .map_err(|message| cx.error(0, message))?;
        super::shared_default::check(cx, default, &expected, &catalog.types)?;
    }
    Ok(())
}

fn shared_decl(
    cx: &Context<'_>,
    a: &py::StmtAnnAssign,
    types: &TypeCatalog,
) -> Result<p::SharedVarDecl, Diagnostic> {
    let py::Expr::Name(n) = a.target.as_ref() else {
        return Err(cx.at(a, "shared variables require a simple name annotation"));
    };
    if !super::identifier(n.id.as_str()) || super::forbidden(n.id.as_str()) {
        return Err(cx.at(a, "unsupported shared variable name"));
    }
    let ty = annotation(cx, &a.annotation, types)?;
    let ty = super::types::resolve_ty(types, &ty).map_err(|message| cx.at(a, message))?;
    let default = a
        .value
        .as_deref()
        .ok_or_else(|| cx.at(a, "shared variable requires a default value"))?;
    let env = HashMap::new();
    let signatures = HashMap::new();
    let shared_vars = HashMap::new();
    let expressions = Expressions {
        cx,
        env: &env,
        signatures: &signatures,
        types,
        shared_vars: &shared_vars,
        effects: &[],
    };
    let (expr, _ty) = expressions.lower_shared_default(default)?;
    Ok(p::SharedVarDecl {
        name: n.id.to_string(),
        ty: ty.parser(),
        default: expr,
    })
}

fn lower_module_items(
    cx: &Context<'_>,
    items: &[ModuleItem<'_>],
    catalog: &ModuleCatalog,
) -> Result<Vec<p::Operation>, Diagnostic> {
    let mut body = Vec::new();
    for item in items {
        match item {
            ModuleItem::Function(def) => {
                if let Some((method, path)) = &def.route {
                    body.push(p::Operation::Route {
                        method: method.clone(),
                        path: path.clone(),
                        handler_fn: def.name.to_string(),
                    });
                }
                if let Some(source) = &def.source {
                    body.push(p::Operation::ModuleSource(source.clone()));
                }
                body.push(p::Operation::Function(def.lower(
                    cx,
                    &catalog.signatures,
                    &catalog.types,
                    &catalog.shared,
                )?));
            }
            ModuleItem::Type(decl) => body.push(p::Operation::Type(decl.clone())),
            ModuleItem::Shared(assign) => {
                body.push(p::Operation::SharedVar(shared_decl(
                    cx,
                    assign,
                    &catalog.types,
                )?));
            }
            ModuleItem::Startup(def) => body.push(p::Operation::Startup(def.lower_lifecycle(
                cx,
                &catalog.signatures,
                &catalog.types,
                &catalog.shared,
            )?)),
            ModuleItem::Shutdown(def) => body.push(p::Operation::Shutdown(def.lower_lifecycle(
                cx,
                &catalog.signatures,
                &catalog.types,
                &catalog.shared,
            )?)),
        }
    }
    Ok(body)
}

enum ModuleItem<'a> {
    Function(Definition<'a>),
    Type(p::TypeDecl),
    /// Name, annotation and default node. Both are resolved in the catalog
    /// phase, once every submitted type exists (#slice3).
    Shared(&'a py::StmtAnnAssign),
    Startup(Definition<'a>),
    Shutdown(Definition<'a>),
}

impl<'a> ModuleItem<'a> {
    fn from_stmt(
        cx: &Context<'_>,
        stmt: &'a py::Stmt,
        types: &TypeCatalog,
    ) -> Result<Self, Diagnostic> {
        if let py::Stmt::ClassDef(class) = stmt {
            return Ok(ModuleItem::Type(super::types::lower_type_decl(cx, class)?));
        }
        if let py::Stmt::AnnAssign(assign) = stmt {
            return Ok(ModuleItem::Shared(assign));
        }
        let def = Definition::from_stmt(cx, stmt)?;
        match def.name {
            "startup" => Ok(ModuleItem::Startup(def)),
            "shutdown" => Ok(ModuleItem::Shutdown(def)),
            _ => Ok(ModuleItem::Function(def)),
        }
    }
}

/// Record a submitted type declaration into the effective catalog so later
/// functions in the same module can construct/access it. `Ty` payloads are
/// rebuilt from the parser `TypeExpr`; this never consults runtime state.
pub(super) fn insert_parser_type(decl: &p::TypeDecl, module: &str, out: &mut TypeCatalog) {
    let info = match &decl.body {
        p::TypeBody::Struct(fields) => super::types::TypeInfo::Struct {
            fields: fields
                .iter()
                .map(|f| {
                    (
                        f.name.clone(),
                        super::types::with_owner(type_expr_to_ty(&f.ty), module),
                    )
                })
                .collect(),
        },
        p::TypeBody::Union(variants) => super::types::TypeInfo::Union {
            module: module.to_string(),
            variants: variants
                .iter()
                .map(|v| {
                    (
                        v.name.clone(),
                        v.payload
                            .iter()
                            .map(|ty| super::types::with_owner(type_expr_to_ty(ty), module))
                            .collect(),
                    )
                })
                .collect(),
        },
        p::TypeBody::Alias(_) => super::types::TypeInfo::Struct { fields: Vec::new() },
    };
    out.insert(super::types::owned_name(module, &decl.name), info);
}

fn type_expr_to_ty(ty: &p::TypeExpr) -> Ty {
    match ty {
        p::TypeExpr::Named(name) => match name.as_str() {
            "Int" => Ty::Int,
            "Float" => Ty::Float,
            "Bool" => Ty::Bool,
            "String" => Ty::Str,
            "Byte" => Ty::Byte,
            other => Ty::Named(other.to_string()),
        },
        p::TypeExpr::Generic { name, args } => match name.as_str() {
            "List" if args.len() == 1 => Ty::List(Box::new(type_expr_to_ty(&args[0]))),
            "Set" if args.len() == 1 => Ty::Set(Box::new(type_expr_to_ty(&args[0]))),
            "Option" if args.len() == 1 => Ty::Option(Box::new(type_expr_to_ty(&args[0]))),
            "Result" if args.len() == 1 => Ty::Result(Box::new(type_expr_to_ty(&args[0]))),
            "Map" if args.len() == 2 => Ty::Map(
                Box::new(type_expr_to_ty(&args[0])),
                Box::new(type_expr_to_ty(&args[1])),
            ),
            _ => Ty::Unknown,
        },
        p::TypeExpr::Struct(_) => Ty::Unknown,
    }
}

struct Definition<'a> {
    stmt: &'a py::Stmt,
    name: &'a str,
    args: &'a py::Arguments,
    returns: Option<&'a py::Expr>,
    body: &'a [py::Stmt],
    decorators: &'a [py::Expr],
    asynchronous: bool,
    /// `@route("METHOD", "/path")` decorator: (method, path).
    route: Option<(String, String)>,
    /// `@source("name", type=..., k=v)` decorator.
    source: Option<p::ModuleSourceDecl>,
}

pub struct Preflight {
    pub references: Vec<String>,
    pub type_references: Vec<String>,
    pub local_calls: Vec<String>,
    pub declared_types: Vec<p::TypeDecl>,
    pub declared_functions: Vec<String>,
    pub value_references: Vec<String>,
    pub bare_calls: Vec<String>,
}

pub(super) fn preflight(source: &str, module: &str) -> Result<Preflight, Diagnostic> {
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
    let mut type_references = Vec::new();
    let mut local_calls = Vec::new();
    let mut declared_types = Vec::new();
    let mut declared_functions = Vec::new();
    for (index, stmt) in suite.iter().enumerate() {
        if index == 0 && super::docstring(stmt).is_some() {
            continue;
        }
        if let py::Stmt::ClassDef(class) = stmt {
            let decl = super::types::lower_type_decl(&cx, class)?;
            if !names.insert(decl.name.clone()) {
                return Err(cx.at(stmt, "duplicate type declaration"));
            }
            declared_types.push(decl);
            continue;
        }
        if let py::Stmt::AnnAssign(assign) = stmt {
            let py::Expr::Name(n) = assign.target.as_ref() else {
                return Err(cx.at(stmt, "shared variables require a simple name annotation"));
            };
            if !names.insert(n.id.to_string()) {
                return Err(cx.at(stmt, "duplicate shared variable declaration"));
            }
            if assign.value.is_none() {
                return Err(cx.at(stmt, "shared variable requires a default value"));
            }
            let ty = super::expressions::annotation(&cx, &assign.annotation, &Default::default())?;
            super::expressions::collect_ty_names(&ty, &mut type_references);
            continue;
        }
        let def = Definition::from_stmt(&cx, stmt)?;
        let signature = def.signature(&cx, &Default::default())?;
        for param in &signature.params {
            super::expressions::collect_ty_names(param, &mut type_references);
        }
        super::expressions::collect_ty_names(&signature.result, &mut type_references);
        if !names.insert(def.name.to_string()) {
            return Err(cx.at(stmt, "duplicate function declaration"));
        }
        declared_functions.push(def.name.to_string());
        super::surface::statements(&cx, def.body, true, &mut references, &mut type_references)?;
    }
    if names.is_empty() {
        return Err(cx.error(
            0,
            "module submission requires at least one annotated function, type or shared declaration",
        ));
    }
    // Bare candidates carry no fabricated function role. Classification against
    // existing target-module declarations happens only after authority checks.
    references.retain(|reference| {
        if let Some(name) = reference.strip_prefix("@local:") {
            local_calls.push(name.to_string());
            false
        } else {
            true
        }
    });
    local_calls.sort();
    local_calls.dedup();
    let usage = super::references::usage::collect(&cx, &suite)?;
    references.extend(usage.namespaces);
    let mut value_references: Vec<_> = usage.values.into_iter().collect();
    let mut bare_calls: Vec<_> = usage.calls.into_iter().collect();
    value_references.sort();
    bare_calls.sort();
    references.sort();
    references.dedup();
    type_references.sort();
    type_references.dedup();
    Ok(Preflight {
        references,
        type_references,
        local_calls,
        declared_types,
        declared_functions,
        value_references,
        bare_calls,
    })
}

impl<'a> Definition<'a> {
    fn from_stmt(cx: &Context<'_>, stmt: &'a py::Stmt) -> Result<Self, Diagnostic> {
        let (asynchronous, name, args, returns, body, decorator_list) = match stmt {
            py::Stmt::FunctionDef(f) if f.type_params.is_empty() && f.type_comment.is_none() => {
                (false, f.name.as_str(), &f.args, f.returns.as_deref(), &f.body, &f.decorator_list)
            }
            py::Stmt::AsyncFunctionDef(f)
                if f.type_params.is_empty() && f.type_comment.is_none() =>
            {
                (true, f.name.as_str(), &f.args, f.returns.as_deref(), &f.body, &f.decorator_list)
            }
            _ => return Err(cx.at(stmt, "unsupported module statement; only an initial docstring, annotated def/async def declarations and @dataclass/@union class declarations are allowed")),
        };
        if !super::identifier(name)
            || super::forbidden(name)
            || crate::builtins::is_builtin(name)
            || super::reserved_helper_name(name)
            || matches!(
                name,
                "effects" | "int" | "float" | "bool" | "str" | "Result"
            )
        {
            return Err(cx.at(
                stmt,
                format!(
                    "reserved/unsafe function name {name}; builtin and syntax names cannot be shadowed"
                ),
            ));
        }
        let decorations = decorators(cx, decorator_list, asynchronous)?;
        Ok(Definition {
            stmt,
            name,
            args,
            returns,
            body,
            decorators: decorator_list,
            asynchronous,
            route: decorations.route,
            source: decorations.source.map(|mut source| {
                source.handler = name.to_string();
                source
            }),
        })
    }

    fn signature(&self, cx: &Context<'_>, types: &TypeCatalog) -> Result<Signature, Diagnostic> {
        ordinary_args(cx, self.stmt, self.args)?;
        let mut params = Vec::new();
        for arg in &self.args.args {
            let ann =
                arg.def.annotation.as_deref().ok_or_else(|| {
                    cx.at(&arg.def, "parameter requires an explicit type annotation")
                })?;
            params.push(annotation(cx, ann, types)?);
        }
        let result = if is_lifecycle(self.name) {
            if !self.args.args.is_empty() {
                return Err(cx.at(self.stmt, "startup/shutdown blocks take no parameters"));
            }
            match self.returns {
                Some(r) if is_none_annotation(r) => Ty::Str,
                Some(_) => {
                    return Err(cx.at(
                        self.stmt,
                        "startup/shutdown blocks cannot declare a return type; use -> None",
                    ));
                }
                None => Ty::Str,
            }
        } else {
            let returns = self.returns.ok_or_else(|| {
                cx.at(
                    self.stmt,
                    "function requires an explicit return annotation; None/unit is not supported",
                )
            })?;
            annotation(cx, returns, types)?
        };
        let effects = decorators(cx, self.decorators, self.asynchronous)?.effects;
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
        types: &TypeCatalog,
        shared_vars: &HashMap<String, Ty>,
    ) -> Result<p::FunctionDecl, Diagnostic> {
        let sig = &signatures[self.name];
        let mut lower = Lower {
            cx,
            signatures,
            types,
            shared_vars,
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

    fn lower_lifecycle(
        &self,
        cx: &Context<'_>,
        signatures: &HashMap<String, Signature>,
        types: &TypeCatalog,
        shared_vars: &HashMap<String, Ty>,
    ) -> Result<p::FunctionDecl, Diagnostic> {
        let sig = &signatures[self.name];
        let mut lower = Lower {
            cx,
            signatures,
            types,
            shared_vars,
            env: HashMap::new(),
            seen: HashSet::new(),
            result: sig.result.clone(),
            effects: &sig.effects,
        };
        let (doc, stmts) = match self.body.first().and_then(super::docstring) {
            Some(text) => (Some(text.to_owned()), &self.body[1..]),
            None => (None, self.body),
        };
        let body = lower.block(stmts)?;
        Ok(p::FunctionDecl {
            name: self.name.into(),
            params: vec![],
            return_type: p::TypeExpr::Named("String".into()),
            effects: sig.effects.clone(),
            body,
            doc,
        })
    }
}

fn is_lifecycle(name: &str) -> bool {
    matches!(name, "startup" | "shutdown")
}

fn is_effect_name(name: &str) -> bool {
    matches!(
        name,
        "io" | "fail" | "mut" | "rand" | "yield" | "parallel" | "unsafe"
    )
}

fn is_none_annotation(r: &py::Expr) -> bool {
    matches!(r, py::Expr::Constant(c) if matches!(c.value, py::Constant::None))
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

struct Decorators {
    effects: Vec<String>,
    route: Option<(String, String)>,
    source: Option<p::ModuleSourceDecl>,
}

/// Parse a function's decorators into effects plus optional `@route`/`@source`
/// declarations. Only `@effects`, `@route` and `@source` are allowed, and each
/// argument must be a literal string.
fn decorators(
    cx: &Context<'_>,
    decorators: &[py::Expr],
    asynchronous: bool,
) -> Result<Decorators, Diagnostic> {
    let mut effects = Vec::new();
    let mut route = None;
    let mut source = None;
    for decorator in decorators {
        let py::Expr::Call(c) = decorator else {
            return Err(cx.at(
                decorator,
                "unsupported decorator; use @effects, @route, or @source",
            ));
        };
        let py::Expr::Name(n) = c.func.as_ref() else {
            return Err(cx.at(decorator, "unsupported decorator call target"));
        };
        match n.id.as_str() {
            "effects" => {
                if !c.keywords.is_empty() {
                    return Err(cx.at(decorator, "@effects takes positional string arguments only"));
                }
                for arg in &c.args {
                    let effect = literal_string(cx, arg)?;
                    if !is_effect_name(&effect) || effects.contains(&effect) {
                        return Err(cx.at(arg, "unsupported or duplicate effect; allowed: io, fail, mut, rand, yield, parallel, unsafe (async comes from async def)"));
                    }
                    effects.push(effect);
                }
            }
            "route" => {
                if route.is_some() {
                    return Err(cx.at(decorator, "duplicate @route decorator"));
                }
                if !c.keywords.is_empty() || c.args.len() != 2 {
                    return Err(cx.at(decorator, "@route requires exactly two string arguments: method and path"));
                }
                route = Some((literal_string(cx, &c.args[0])?.to_uppercase(), literal_string(cx, &c.args[1])?));
            }
            "source" => {
                if source.is_some() {
                    return Err(cx.at(decorator, "duplicate @source decorator"));
                }
                if c.args.len() != 1 {
                    return Err(cx.at(decorator, "@source requires one positional name argument"));
                }
                let name = literal_string(cx, &c.args[0])?;
                let mut source_type = String::new();
                let mut config = Vec::new();
                for keyword in &c.keywords {
                    let Some(key) = &keyword.arg else {
                        return Err(cx.at(keyword, "** expansion is unsupported in @source"));
                    };
                    let value = literal_string(cx, &keyword.value)?;
                    if key == "type" {
                        source_type = value;
                    } else {
                        config.push((key.to_string(), value));
                    }
                }
                if source_type.is_empty() {
                    return Err(cx.at(decorator, "@source requires a type= keyword"));
                }
                source = Some(p::ModuleSourceDecl {
                    name,
                    source_type,
                    config,
                    handler: String::new(),
                });
            }
            other => return Err(cx.at(decorator, format!("unsupported decorator @{other}; only inert @effects, @route and @source are supported"))),
        }
    }
    if asynchronous {
        effects.push("async".into());
    }
    Ok(Decorators {
        effects,
        route,
        source,
    })
}

fn literal_string(cx: &Context<'_>, e: &py::Expr) -> Result<String, Diagnostic> {
    match e {
        py::Expr::Constant(c) => match &c.value {
            py::Constant::Str(s) => Ok(s.clone()),
            _ => Err(cx.at(e, "expected a literal string")),
        },
        _ => Err(cx.at(e, "expected a literal string")),
    }
}

fn literal_arg(cx: &Context<'_>, e: &py::Expr) -> Result<String, Diagnostic> {
    literal_string(cx, e)
}

struct Lower<'a, 'b> {
    cx: &'a Context<'b>,
    signatures: &'a HashMap<String, Signature>,
    types: &'a TypeCatalog,
    shared_vars: &'a HashMap<String, Ty>,
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
            types: self.types,
            shared_vars: self.shared_vars,
            effects: self.effects,
        }
    }

    fn bind(&mut self, name: &str, ty: Ty, node: &impl py::Ranged) -> Result<(), Diagnostic> {
        if name == "_" {
            // Discard binding: allowed repeatedly, never resolved in scope.
            return Ok(());
        }
        if !super::identifier(name)
            || super::forbidden(name)
            || crate::builtins::is_builtin(name)
            || super::reserved_helper_name(name)
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
        // A declaration in one branch/arm must not reserve its name in siblings.
        // Keep outer reservations so loop targets still cannot shadow bindings.
        let original_seen = self.seen.clone();
        let result = body
            .iter()
            .filter(|s| !matches!(s, py::Stmt::Pass(_)))
            .map(|s| self.statement(s))
            .collect();
        self.env = original;
        self.seen = original_seen;
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
            py::Stmt::Match(v) => self.match_statement(v),
            py::Stmt::Expr(v) => self.source_event_statement(v),
            _ => Err(self.cx.at(s, format!("unsupported statement {}; use annotated bindings, assignment, return, if, while, or fresh-target for", statement_kind(s)))),
        }
    }

    fn annotated(&mut self, a: &py::StmtAnnAssign) -> Result<p::Operation, Diagnostic> {
        let py::Expr::Name(n) = a.target.as_ref() else {
            return Err(self
                .cx
                .at(a, "annotated binding target must be a simple name"));
        };
        let ty = annotation(self.cx, &a.annotation, self.types)?;
        let value = a
            .value
            .as_deref()
            .ok_or_else(|| self.cx.at(a, "annotated binding requires an initial value"))?;
        if let Some(call) = self.try_propagate(value)? {
            let (expr, actual) = self.expressions().call(call, Site::Ordinary)?;
            self.check_type(&ty, &actual, value)?;
            if self.env.contains_key(n.id.as_str()) {
                return Err(self.cx.at(a, "propagate binding reuses an existing name"));
            }
            self.bind(n.id.as_str(), ty.clone(), a.target.as_ref())?;
            return Ok(p::Operation::Call(p::CallDecl {
                name: n.id.to_string(),
                ty: ty.parser(),
                expr,
            }));
        }
        if let Some(call) = self.try_spawn(value)? {
            let (expr, _actual) = self.expressions().call(call, Site::SpawnTarget)?;
            if self.env.contains_key(n.id.as_str()) {
                return Err(self.cx.at(a, "spawn binding reuses an existing name"));
            }
            self.bind(n.id.as_str(), ty.clone(), a.target.as_ref())?;
            return Ok(p::Operation::Spawn(p::SpawnDecl {
                call: expr,
                binding: Some((n.id.to_string(), ty.parser())),
            }));
        }
        let awaited = matches!(value, py::Expr::Await(_));
        let (expr, actual) = if let py::Expr::Await(v) = value {
            let py::Expr::Call(call) = v.value.as_ref() else {
                return Err(self.cx.at(value, "await requires a direct function call"));
            };
            self.expressions().call(call, Site::AwaitBinding)?
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
        let expected = self
            .env
            .get(n.id.as_str())
            .or_else(|| self.shared_vars.get(n.id.as_str()))
            .ok_or_else(|| {
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

    /// A bare expression statement must call a reserved source/event helper.
    fn source_event_statement(&self, e: &py::StmtExpr) -> Result<p::Operation, Diagnostic> {
        let py::Expr::Call(c) = e.value.as_ref() else {
            return Err(self
                .cx
                .at(e, "expression statements must call a source/event helper"));
        };
        if !c.keywords.is_empty() {
            return Err(self
                .cx
                .at(e, "source/event helpers take positional arguments"));
        }
        let name = super::expressions::direct_name(self.cx, &c.func, &self.env)?;
        match name.as_str() {
            "source_add" => {
                if c.args.len() != 3 {
                    return Err(self.cx.at(e, "source_add(kind, alias, handler) expects 3 string arguments"));
                }
                Ok(p::Operation::SourceAdd(p::SourceAddDecl {
                    kind_text: literal_arg(self.cx, &c.args[0])?,
                    alias: literal_arg(self.cx, &c.args[1])?,
                    handler: literal_arg(self.cx, &c.args[2])?,
                }))
            }
            "source_remove" => {
                if c.args.len() != 1 {
                    return Err(self.cx.at(e, "source_remove(alias) expects 1 string argument"));
                }
                Ok(p::Operation::SourceRemove(literal_arg(self.cx, &c.args[0])?))
            }
            "source_replace" => {
                if c.args.len() != 3 {
                    return Err(self.cx.at(e, "source_replace(alias, kind, handler) expects 3 string arguments"));
                }
                Ok(p::Operation::SourceReplace(p::SourceReplaceDecl {
                    alias: literal_arg(self.cx, &c.args[0])?,
                    kind_text: literal_arg(self.cx, &c.args[1])?,
                    handler: literal_arg(self.cx, &c.args[2])?,
                }))
            }
            "source_list" => {
                if !c.args.is_empty() {
                    return Err(self.cx.at(e, "source_list() takes no arguments"));
                }
                Ok(p::Operation::SourceList)
            }
            "event_register" => {
                if c.args.len() != 2 {
                    return Err(self.cx.at(e, "event_register(name, payload_type) expects 2 string arguments"));
                }
                Ok(p::Operation::EventRegister(p::EventRegisterDecl {
                    name: literal_arg(self.cx, &c.args[0])?,
                    payload_type: literal_arg(self.cx, &c.args[1])?,
                }))
            }
            "event_emit" => {
                if c.args.len() != 2 {
                    return Err(self.cx.at(e, "event_emit(name, value) expects 2 arguments"));
                }
                let name = literal_arg(self.cx, &c.args[0])?;
                let (value, _ty) = self.expressions().lower(&c.args[1])?;
                Ok(p::Operation::EventEmit(p::EventEmitDecl { name, value }))
            }
            "check" => {
                if c.args.len() != 3 {
                    return Err(self.cx.at(e, "check(label, condition, err_label) expects 3 arguments"));
                }
                let label = literal_arg(self.cx, &c.args[0])?;
                let (condition, cond_ty) = self.expressions().lower(&c.args[1])?;
                if cond_ty != Ty::Bool {
                    return Err(self.cx.at(e, "check condition must be statically bool"));
                }
                let err_label = literal_arg(self.cx, &c.args[2])?;
                Ok(p::Operation::Check(p::CheckDecl {
                    name: label,
                    expr: condition,
                    err_label,
                }))
            }
            "spawn" => {
                let (expr, _ty) = self.spawn_call_arg(e, c)?;
                Ok(p::Operation::Spawn(p::SpawnDecl {
                    call: expr,
                    binding: None,
                }))
            }
            other => Err(self.cx.at(
                e,
                format!("unsupported expression statement {other}; only source/event helpers are allowed"),
            )),
        }
    }

    /// Lower the single function-call argument of `spawn(...)` / `propagate(...)`.
    fn spawn_call_arg(
        &self,
        e: &py::StmtExpr,
        c: &py::ExprCall,
    ) -> Result<(p::Expr, Ty), Diagnostic> {
        if !c.keywords.is_empty() || c.args.len() != 1 {
            return Err(self.cx.at(e, "spawn requires a single function call"));
        }
        let py::Expr::Call(inner) = &c.args[0] else {
            return Err(self.cx.at(e, "spawn requires a single function call"));
        };
        self.expressions().call(inner, Site::SpawnTarget)
    }

    /// `name: T = propagate(f(x))` lowers to `+call name:T = f(x)`. The helper
    /// wraps exactly one direct function call.
    fn try_propagate<'v>(
        &self,
        value: &'v py::Expr,
    ) -> Result<Option<&'v py::ExprCall>, Diagnostic> {
        let py::Expr::Call(c) = value else {
            return Ok(None);
        };
        let py::Expr::Name(n) = c.func.as_ref() else {
            return Ok(None);
        };
        if n.id.as_str() != "propagate" {
            return Ok(None);
        }
        if !c.keywords.is_empty() || c.args.len() != 1 {
            return Err(self
                .cx
                .at(value, "propagate requires a single function call"));
        }
        let py::Expr::Call(inner) = &c.args[0] else {
            return Err(self
                .cx
                .at(value, "propagate requires a single function call"));
        };
        Ok(Some(inner))
    }

    /// `name: T = spawn(f(x))` lowers to `+spawn name:T = f(x)`.
    fn try_spawn<'v>(&self, value: &'v py::Expr) -> Result<Option<&'v py::ExprCall>, Diagnostic> {
        let py::Expr::Call(c) = value else {
            return Ok(None);
        };
        let py::Expr::Name(n) = c.func.as_ref() else {
            return Ok(None);
        };
        if n.id.as_str() != "spawn" {
            return Ok(None);
        }
        if !c.keywords.is_empty() || c.args.len() != 1 {
            return Err(self.cx.at(value, "spawn requires a single function call"));
        }
        let py::Expr::Call(inner) = &c.args[0] else {
            return Err(self.cx.at(value, "spawn requires a single function call"));
        };
        Ok(Some(inner))
    }

    fn match_statement(&mut self, m: &py::StmtMatch) -> Result<p::Operation, Diagnostic> {
        let (expr, subject_ty) = self.expressions().lower_match_subject(&m.subject)?;
        let mut arms = Vec::new();
        for case in &m.cases {
            arms.push(self.match_case(case, &subject_ty)?);
        }
        if arms.is_empty() {
            return Err(self.cx.at(m, "match requires at least one case arm"));
        }
        check_exhaustiveness(self.cx, m, &subject_ty, &arms, self.types)?;
        Ok(p::Operation::Match(p::MatchDecl { expr, arms }))
    }

    fn match_case(
        &mut self,
        case: &py::MatchCase,
        subject_ty: &Ty,
    ) -> Result<p::MatchArmDecl, Diagnostic> {
        if case.guard.is_some() {
            return Err(self.cx.at(
                &case.pattern,
                "match guards (case ... if ...) are unsupported",
            ));
        }
        let (variant, bindings, patterns) = lower_arm_pattern(self.cx, self.types, &case.pattern)?;
        let arity = match &patterns {
            Some(patterns) => patterns.len(),
            None => bindings.len(),
        };
        check_case_membership(
            self.cx,
            &case.pattern,
            subject_ty,
            &variant,
            arity,
            self.types,
        )?;
        let mut to_bind = resolve_binding_types(subject_ty, &variant, &bindings, self.types);
        if let Some(patterns) = &patterns {
            let payload = variant_payload_types(subject_ty, &variant, self.types);
            for (index, pattern) in patterns.iter().enumerate() {
                let sub_ty = payload.get(index).cloned().unwrap_or(Ty::Unknown);
                resolve_nested_pattern_types(pattern, &sub_ty, self.types, &mut to_bind);
            }
        }
        let saved = self.env.clone();
        for (name, ty) in to_bind {
            self.env.insert(name, ty);
        }
        let body = self.block(&case.body);
        self.env = saved;
        let body = body?;
        Ok(p::MatchArmDecl {
            variant,
            bindings,
            patterns,
            body,
        })
    }
}

/// Reject case arms whose variant does not belong to the match subject's type or
/// whose payload arity mismatches. Undecidable subjects (unknown types) pass.
fn check_case_membership(
    cx: &Context<'_>,
    node: &impl py::Ranged,
    subject: &Ty,
    variant: &str,
    arity: usize,
    types: &TypeCatalog,
) -> Result<(), Diagnostic> {
    if variant == "_" {
        return Ok(());
    }
    match subject {
        Ty::Result(_) => match variant {
            "Ok" | "Err" if arity == 1 => Ok(()),
            "Ok" | "Err" => Err(cx.at(
                node,
                format!("case {variant} expects exactly 1 payload, got {arity}"),
            )),
            _ => Err(cx.at(
                node,
                format!(
                    "case variant {variant} does not belong to a Result subject; use Ok or Err"
                ),
            )),
        },
        Ty::Option(_) => match variant {
            "Some" if arity == 1 => Ok(()),
            "Some" => Err(cx.at(node, "case Some expects exactly 1 payload")),
            "None" if arity == 0 => Ok(()),
            "None" => Err(cx.at(node, "case None takes no payload")),
            _ => Err(cx.at(
                node,
                format!(
                    "case variant {variant} does not belong to an Option subject; use Some or None"
                ),
            )),
        },
        Ty::Named(union) => {
            match types.get(union) {
                Some(super::types::TypeInfo::Union { variants, .. }) => {
                    match variants.iter().find(|(name, _)| name == variant) {
                    Some((_, payload)) if payload.len() == arity => Ok(()),
                    Some((_, payload)) => Err(cx.at(
                        node,
                        format!("variant {variant} expects {} payload(s), got {arity}", payload.len()),
                    )),
                    None => Err(cx.at(
                        node,
                        format!("case variant {variant} is not a declared variant of the subject type"),
                    )),
                }
                }
                _ => Ok(()),
            }
        }
        _ => Ok(()),
    }
}

/// Require exhaustiveness where the subject type is decidable: a `_` arm, or all
/// declared union variants, or Ok+Err for Result, or Some+None for Option.
fn check_exhaustiveness(
    cx: &Context<'_>,
    node: &impl py::Ranged,
    subject: &Ty,
    arms: &[p::MatchArmDecl],
    types: &TypeCatalog,
) -> Result<(), Diagnostic> {
    if arms.iter().any(|arm| arm.variant == "_") {
        return Ok(());
    }
    let covered: std::collections::HashSet<&str> =
        arms.iter().map(|arm| arm.variant.as_str()).collect();
    let required: Vec<String> = match subject {
        Ty::Result(_) => vec!["Ok".into(), "Err".into()],
        Ty::Option(_) => vec!["Some".into(), "None".into()],
        Ty::Named(union) => match types.get(union) {
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
    Err(cx.at(
        node,
        format!(
            "non-exhaustive match; missing case(s): {} — add them or a wildcard _",
            missing.join(", ")
        ),
    ))
}

/// Recursively collect captured names inside nested patterns, resolving each
/// captured name's static type from the enclosing variant payload position.
fn resolve_nested_pattern_types(
    pattern: &p::MatchPatternDecl,
    subject_ty: &Ty,
    types: &TypeCatalog,
    out: &mut Vec<(String, Ty)>,
) {
    match pattern {
        p::MatchPatternDecl::Binding(name) => {
            if name != "_" {
                out.push((name.clone(), subject_ty.clone()));
            }
        }
        p::MatchPatternDecl::Variant {
            variant,
            sub_patterns,
        } => {
            let payload = variant_payload_types(subject_ty, variant, types);
            for (index, sub) in sub_patterns.iter().enumerate() {
                let sub_ty = payload.get(index).cloned().unwrap_or(Ty::Unknown);
                resolve_nested_pattern_types(sub, &sub_ty, types, out);
            }
        }
        _ => {}
    }
}

/// The payload types of a variant/constructor name within a subject type.
fn variant_payload_types(subject: &Ty, variant: &str, types: &TypeCatalog) -> Vec<Ty> {
    match subject {
        Ty::Result(inner) if variant == "Ok" => vec![(**inner).clone()],
        Ty::Result(_) if variant == "Err" => vec![Ty::Str],
        Ty::Option(inner) if variant == "Some" => vec![(**inner).clone()],
        Ty::Named(union) => types
            .get(union)
            .and_then(|info| match info {
                super::types::TypeInfo::Union { variants, .. } => variants
                    .iter()
                    .find(|(name, _)| name == variant)
                    .map(|(_, payload)| payload.clone()),
                super::types::TypeInfo::Struct { .. } => None,
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// Resolve the static types of a case arm's captured bindings from the match
/// subject's type. `Ok`/`Err`/`Some` payloads, and declared variant payloads,
/// supply the type per binding position; nested-pattern placeholders (`__pat_N`)
/// and `_` are skipped. Unknown positions fall back to `Ty::Unknown`.
fn resolve_binding_types(
    subject: &Ty,
    variant: &str,
    bindings: &[String],
    types: &TypeCatalog,
) -> Vec<(String, Ty)> {
    let payload: Vec<Ty> = variant_payload_types(subject, variant, types);
    bindings
        .iter()
        .enumerate()
        .filter(|(_, name)| name.as_str() != "_" && !name.starts_with("__pat_"))
        .map(|(index, name)| {
            let ty = payload.get(index).cloned().unwrap_or(Ty::Unknown);
            (name.clone(), ty)
        })
        .collect()
}

fn is_known_arm_target(
    cx: &Context<'_>,
    types: &TypeCatalog,
    name: &str,
) -> Result<bool, Diagnostic> {
    if matches!(name, "Ok" | "Err" | "Some" | "None") {
        return Ok(true);
    }
    match super::types::variant_payloads(types, name, &cx.module) {
        Ok(Some(_)) => Ok(true),
        Ok(None) => Ok(false),
        Err(message) => Err(cx.error(0, message)),
    }
}

/// Lower a Python `case` pattern to an Adapsis `(variant, bindings, patterns)`
/// arm. Only declared variants, `Ok`/`Err`/`Some`, `None`, and `_` are allowed;
/// guards, `|`, as-patterns, sequences, mappings and star patterns are rejected.
fn lower_arm_pattern(
    cx: &Context<'_>,
    types: &TypeCatalog,
    pattern: &py::Pattern,
) -> Result<(String, Vec<String>, Option<Vec<p::MatchPatternDecl>>), Diagnostic> {
    match pattern {
        py::Pattern::MatchClass(class) => {
            let py::Expr::Name(cls) = class.cls.as_ref() else {
                return Err(cx.at(
                    class,
                    "case class patterns require a bare variant/constructor name",
                ));
            };
            let variant = cls.id.as_str();
            if !is_known_arm_target(cx, types, variant)? {
                return Err(cx.at(
                    class,
                    format!(
                        "unknown case variant {variant}; only declared variants and Ok/Err/Some/None are allowed"
                    ),
                ));
            }
            if !class.kwd_attrs.is_empty() || !class.kwd_patterns.is_empty() {
                return Err(cx.at(class, "keyword patterns are unsupported in case arms"));
            }
            let mut sub_patterns = Vec::new();
            for sub in &class.patterns {
                sub_patterns.push(lower_sub_pattern(cx, types, sub)?);
            }
            let mut flat_bindings = Vec::new();
            let mut has_nested = false;
            for sub in &sub_patterns {
                match sub {
                    p::MatchPatternDecl::Binding(name) => flat_bindings.push(name.clone()),
                    _ => {
                        has_nested = true;
                        flat_bindings.push(format!("__pat_{}", flat_bindings.len()));
                    }
                }
            }
            let patterns = if has_nested { Some(sub_patterns) } else { None };
            Ok((variant.to_string(), flat_bindings, patterns))
        }
        py::Pattern::MatchAs(aspat) if aspat.pattern.is_none() && aspat.name.is_none() => {
            Ok(("_".to_string(), vec![], None))
        }
        py::Pattern::MatchSingleton(single) if matches!(single.value, py::Constant::None) => {
            Ok(("None".to_string(), vec![], None))
        }
        py::Pattern::MatchAs(aspat) if aspat.pattern.is_none() => Err(cx.at(
            pattern,
            "capture patterns are unsupported in case arms; use _ to ignore",
        )),
        py::Pattern::MatchAs(_) => Err(cx.at(
            pattern,
            "as-patterns are unsupported; match the value directly",
        )),
        py::Pattern::MatchValue(_) | py::Pattern::MatchSingleton(_) => Err(cx.at(
            pattern,
            "top-level literal case patterns are unsupported; Adapsis matches union variants",
        )),
        py::Pattern::MatchOr(_) => Err(cx.at(pattern, "'|' alternative patterns are unsupported")),
        py::Pattern::MatchSequence(_) => Err(cx.at(pattern, "sequence patterns are unsupported")),
        py::Pattern::MatchMapping(_) => Err(cx.at(pattern, "mapping patterns are unsupported")),
        py::Pattern::MatchStar(_) => Err(cx.at(pattern, "star patterns are unsupported")),
    }
}

fn lower_sub_pattern(
    cx: &Context<'_>,
    types: &TypeCatalog,
    pattern: &py::Pattern,
) -> Result<p::MatchPatternDecl, Diagnostic> {
    match pattern {
        py::Pattern::MatchValue(value) => literal_pattern(cx, &value.value),
        py::Pattern::MatchSingleton(single) => match &single.value {
            py::Constant::Bool(b) => Ok(p::MatchPatternDecl::LiteralBool(*b)),
            py::Constant::None => Err(cx.at(
                pattern,
                "nested None patterns are unsupported; bind with a name or ignore with _",
            )),
            _ => Err(cx.at(pattern, "unsupported singleton pattern")),
        },
        py::Pattern::MatchAs(aspat) if aspat.pattern.is_none() => Ok(p::MatchPatternDecl::Binding(
            aspat.name.as_deref().unwrap_or("_").to_string(),
        )),
        py::Pattern::MatchAs(aspat) if aspat.pattern.is_some() && aspat.name.is_some() => Err(cx
            .at(
                pattern,
                "as-patterns are unsupported; match the value directly",
            )),
        py::Pattern::MatchClass(class) => {
            let py::Expr::Name(cls) = class.cls.as_ref() else {
                return Err(cx.at(class, "nested variant patterns require a bare variant name"));
            };
            let variant = cls.id.as_str();
            if !is_known_arm_target(cx, types, variant)? {
                return Err(cx.at(class, format!("unknown nested variant {variant}")));
            }
            if !class.kwd_attrs.is_empty() || !class.kwd_patterns.is_empty() {
                return Err(cx.at(class, "keyword patterns are unsupported in case arms"));
            }
            if class.patterns.is_empty() {
                return Err(cx.at(
                    class,
                    "nested zero-payload variant patterns are unsupported; Adapsis cannot express them distinctly from a binding — use a name or _",
                ));
            }
            let sub_patterns = class
                .patterns
                .iter()
                .map(|sub| lower_sub_pattern(cx, types, sub))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(p::MatchPatternDecl::Variant {
                variant: variant.to_string(),
                sub_patterns,
            })
        }
        py::Pattern::MatchOr(_) => Err(cx.at(pattern, "'|' alternative patterns are unsupported")),
        py::Pattern::MatchSequence(_) => Err(cx.at(pattern, "sequence patterns are unsupported")),
        py::Pattern::MatchMapping(_) => Err(cx.at(pattern, "mapping patterns are unsupported")),
        py::Pattern::MatchStar(_) => Err(cx.at(pattern, "star patterns are unsupported")),
        py::Pattern::MatchAs(_) => Err(cx.at(pattern, "unsupported pattern")),
    }
}

fn literal_pattern(cx: &Context<'_>, value: &py::Expr) -> Result<p::MatchPatternDecl, Diagnostic> {
    match value {
        py::Expr::Constant(c) => match &c.value {
            py::Constant::Int(n) => n
                .to_string()
                .parse::<i64>()
                .map(p::MatchPatternDecl::LiteralInt)
                .map_err(|_| cx.at(value, "integer pattern exceeds signed 64-bit range")),
            py::Constant::Str(s) => Ok(p::MatchPatternDecl::LiteralString(s.clone())),
            py::Constant::Bool(b) => Ok(p::MatchPatternDecl::LiteralBool(*b)),
            _ => Err(cx.at(
                value,
                "unsupported literal pattern; use int, string, or bool",
            )),
        },
        py::Expr::UnaryOp(u) if matches!(u.op, py::UnaryOp::USub) => {
            let py::Expr::Constant(c) = u.operand.as_ref() else {
                return Err(cx.at(
                    value,
                    "negative literal pattern requires an integer constant",
                ));
            };
            let py::Constant::Int(n) = &c.value else {
                return Err(cx.at(
                    value,
                    "negative literal pattern requires an integer constant",
                ));
            };
            n.to_string()
                .parse::<i64>()
                .map(|v| p::MatchPatternDecl::LiteralInt(-v))
                .map_err(|_| cx.at(value, "integer pattern exceeds signed 64-bit range"))
        }
        _ => Err(cx.at(
            value,
            "unsupported literal pattern; use int, string, or bool",
        )),
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
        p::Operation::Match(v) => {
            // A non-exhaustive match bails at runtime rather than falling through
            // to None, so only an arm that itself does not return can fall through.
            !v.arms.is_empty() && v.arms.iter().all(|arm| returns(&arm.body))
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
