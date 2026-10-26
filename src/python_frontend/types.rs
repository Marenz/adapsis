//! Struct/union type declarations, the type catalog, and the ast <-> `Ty` mapping.
//!
//! Types are declared in Python as `@dataclass` (struct) or `@union` (tagged
//! union) classes and referenced elsewhere by bare name. The catalog records
//! declared struct field names/types and union variant names/payloads so the
//! lowerer can distinguish a struct constructor `Point(x=1)` from a function
//! call, and so the printer can resolve a struct literal's type name when the
//! anonymous `{..}` form left `StructInit.ty` empty.
use super::expressions::{Ty, annotation};
use super::{Context, Diagnostic};
use crate::ast as a;
use crate::parser as p;
use rustpython_parser::ast as py;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub(super) enum TypeInfo {
    Struct {
        fields: Vec<(String, Ty)>,
    },
    Union {
        module: String,
        variants: Vec<(String, Vec<Ty>)>,
    },
}

pub(super) type TypeCatalog = HashMap<String, TypeInfo>;

/// Catalog keys and Named Ty values carry declaration ownership internally.
/// Python annotations and persisted parser/AST types keep their original name.
pub(super) fn owned_name(owner: &str, name: &str) -> String {
    if owner.is_empty() || name.contains('.') {
        name.into()
    } else {
        format!("{owner}.{name}")
    }
}

pub(super) fn bare_name(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or(name)
}

pub(super) fn with_owner(ty: Ty, owner: &str) -> Ty {
    match ty {
        Ty::Named(name) => Ty::Named(owned_name(owner, &name)),
        Ty::List(t) => Ty::List(Box::new(with_owner(*t, owner))),
        Ty::Set(t) => Ty::Set(Box::new(with_owner(*t, owner))),
        Ty::Option(t) => Ty::Option(Box::new(with_owner(*t, owner))),
        Ty::Result(t) => Ty::Result(Box::new(with_owner(*t, owner))),
        Ty::Map(k, v) => Ty::Map(
            Box::new(with_owner(*k, owner)),
            Box::new(with_owner(*v, owner)),
        ),
        other => other,
    }
}

pub(super) fn resolve_name(
    types: &TypeCatalog,
    owner: &str,
    name: &str,
) -> Result<Option<String>, String> {
    let key = owned_name(owner, name);
    if types.contains_key(&key) {
        return Ok(Some(key));
    }
    let bare = bare_name(name);
    if types.contains_key(bare) {
        return Ok(Some(bare.into()));
    }
    let mut matches = types.keys().filter(|key| bare_name(key) == bare);
    let first = matches.next().cloned();
    if matches.next().is_some() {
        return Err(format!(
            "ambiguous type `{bare}` outside its declaration owner; an exact local type is required"
        ));
    }
    Ok(first)
}

pub(super) fn resolve_ty(types: &TypeCatalog, ty: &Ty) -> Result<Ty, String> {
    Ok(match ty {
        Ty::Named(name) => {
            let (owner, bare) = name.rsplit_once('.').unwrap_or(("", name));
            Ty::Named(resolve_name(types, owner, bare)?.unwrap_or_else(|| bare.into()))
        }
        Ty::List(t) => Ty::List(Box::new(resolve_ty(types, t)?)),
        Ty::Set(t) => Ty::Set(Box::new(resolve_ty(types, t)?)),
        Ty::Option(t) => Ty::Option(Box::new(resolve_ty(types, t)?)),
        Ty::Result(t) => Ty::Result(Box::new(resolve_ty(types, t)?)),
        Ty::Map(k, v) => Ty::Map(
            Box::new(resolve_ty(types, k)?),
            Box::new(resolve_ty(types, v)?),
        ),
        other => other.clone(),
    })
}

impl TypeInfo {
    pub(super) fn field_type(&self, field: &str) -> Option<Ty> {
        match self {
            TypeInfo::Struct { fields } => fields
                .iter()
                .find(|(name, _)| name == field)
                .map(|(_, ty)| ty.clone()),
            TypeInfo::Union { .. } => None,
        }
    }
}

/// Resolve the type of `field` on a value of type `ty`, if `ty` names a struct.
pub(super) fn struct_field_type(
    types: &TypeCatalog,
    ty: &Ty,
    field: &str,
) -> Result<Option<Ty>, String> {
    let resolved = resolve_ty(types, ty)?;
    match resolved {
        Ty::Named(name) => types
            .get(&name)
            .and_then(|info| info.field_type(field))
            .map(|ty| resolve_ty(types, &ty))
            .transpose(),
        _ => Ok(None),
    }
}

/// Look up a union variant by its bare name, returning its owning union name and
/// declared payload types. Variant names are global (bare identifiers) in
/// Adapsis. Resolution is the current module's definition first, then a
/// unique fallback. Multiple candidates are ambiguous even when their arities
/// match: payload types and declaring owners still differ.
pub(super) fn variant_payloads(
    types: &TypeCatalog,
    name: &str,
    current_module: &str,
) -> Result<Option<(String, Vec<Ty>)>, String> {
    let mut matches: Vec<(&String, &String, &Vec<Ty>)> = Vec::new();
    for (union_name, info) in types {
        if let TypeInfo::Union { module, variants } = info {
            if let Some((_, payload)) = variants.iter().find(|(variant, _)| variant == name) {
                matches.push((union_name, module, payload));
            }
        }
    }
    if matches.is_empty() {
        return Ok(None);
    }
    let local: Vec<_> = matches
        .iter()
        .filter(|(_, module, _)| module.as_str() == current_module)
        .collect();
    let candidates: Vec<_> = if local.is_empty() {
        matches.iter().collect()
    } else {
        local
    };
    if candidates.len() != 1 {
        return Err(format!(
            "ambiguous union variant `{name}`; an unambiguous declaration owner is required"
        ));
    }
    let (union, _, payload) = candidates[0];
    let payload = payload
        .iter()
        .map(|ty| resolve_ty(types, ty))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some((union.to_string(), payload)))
}

/// Total mapping from the runtime AST type to the frontend `Ty`.
pub(super) fn ast_type_to_ty(ty: &a::Type) -> Ty {
    ast_type_to_ty_in(ty, "")
}

pub(super) fn ast_type_to_ty_in(ty: &a::Type, owner: &str) -> Ty {
    match ty {
        a::Type::Int => Ty::Int,
        a::Type::Float => Ty::Float,
        a::Type::Bool => Ty::Bool,
        a::Type::String => Ty::Str,
        a::Type::Byte => Ty::Byte,
        a::Type::List(t) => Ty::List(Box::new(ast_type_to_ty_in(t, owner))),
        a::Type::Map(k, v) => Ty::Map(
            Box::new(ast_type_to_ty_in(k, owner)),
            Box::new(ast_type_to_ty_in(v, owner)),
        ),
        a::Type::Set(t) => Ty::Set(Box::new(ast_type_to_ty_in(t, owner))),
        a::Type::Option(t) => Ty::Option(Box::new(ast_type_to_ty_in(t, owner))),
        a::Type::Result(t) => Ty::Result(Box::new(ast_type_to_ty_in(t, owner))),
        a::Type::Struct(name) | a::Type::TaggedUnion(name) => Ty::Named(owned_name(owner, name)),
    }
}

/// Printer-only type hints retain internal owner-qualified names. They are never
/// installed as runtime AST types: rendered annotations and Ty::parser strip
/// the internal owner after checking that the selected declaration is exact.
pub(super) fn ty_to_ast(ty: &Ty) -> a::Type {
    match ty {
        Ty::Int => a::Type::Int,
        Ty::Float => a::Type::Float,
        Ty::Bool => a::Type::Bool,
        Ty::Str => a::Type::String,
        Ty::Byte => a::Type::Byte,
        Ty::List(t) => a::Type::List(Box::new(ty_to_ast(t))),
        Ty::Map(k, v) => a::Type::Map(Box::new(ty_to_ast(k)), Box::new(ty_to_ast(v))),
        Ty::Set(t) => a::Type::Set(Box::new(ty_to_ast(t))),
        Ty::Option(t) => a::Type::Option(Box::new(ty_to_ast(t))),
        Ty::Result(t) => a::Type::Result(Box::new(ty_to_ast(t))),
        Ty::Named(name) => a::Type::Struct(name.clone()),
        Ty::Unknown => a::Type::String,
    }
}

/// Render a frontend `Ty` as Python annotation text.
pub(super) fn type_to_python(ty: &Ty) -> String {
    match ty {
        Ty::Int => "int".into(),
        Ty::Float => "float".into(),
        Ty::Bool => "bool".into(),
        Ty::Str => "str".into(),
        Ty::Byte => "Byte".into(),
        Ty::List(t) => format!("list[{}]", type_to_python(t)),
        Ty::Map(k, v) => format!("Map[{}, {}]", type_to_python(k), type_to_python(v)),
        Ty::Set(t) => format!("Set[{}]", type_to_python(t)),
        Ty::Option(t) => format!("Option[{}]", type_to_python(t)),
        Ty::Result(t) => format!("Result[{}]", type_to_python(t)),
        Ty::Named(name) => bare_name(name).into(),
        Ty::Unknown => "Unknown".into(),
    }
}

/// Collect declared types from a program's modules (recursive) into a catalog.
pub(super) fn collect_types(modules: &[a::Module], out: &mut TypeCatalog) {
    collect_types_in(modules, "", out);
}

fn collect_types_in(modules: &[a::Module], parent: &str, out: &mut TypeCatalog) {
    for module in modules {
        let owner = owned_name(parent, &module.name);
        for ty in &module.types {
            insert_ast_type(ty, &owner, out);
        }
        collect_types_in(&module.modules, &owner, out);
    }
}

/// Build the full catalog for a program: top-level types plus every module's.
pub(super) fn catalog_from_program(program: &a::Program) -> TypeCatalog {
    let mut out = TypeCatalog::new();
    for ty in &program.types {
        insert_ast_type(ty, "", &mut out);
    }
    collect_types(&program.modules, &mut out);
    out
}

/// The declared shared variables of a module, name → type, for resolving bare
/// reads/writes when a single function or whole module is printed in isolation.
pub(super) fn module_shared_vars(
    program: &a::Program,
    module: &str,
) -> std::collections::HashMap<String, Ty> {
    program
        .modules
        .iter()
        .find(|m| m.name == module)
        .map(|m| {
            m.shared_vars
                .iter()
                .map(|sv| (sv.name.clone(), ast_type_to_ty_in(&sv.ty, module)))
                .collect()
        })
        .unwrap_or_default()
}

fn insert_ast_type(ty: &a::TypeDecl, module: &str, out: &mut TypeCatalog) {
    match ty {
        a::TypeDecl::Struct(s) => {
            let fields = s
                .fields
                .iter()
                .map(|f| (f.name.clone(), ast_type_to_ty_in(&f.ty, module)))
                .collect();
            out.insert(owned_name(module, &s.name), TypeInfo::Struct { fields });
        }
        a::TypeDecl::TaggedUnion(u) => {
            let variants = u
                .variants
                .iter()
                .map(|v| {
                    (
                        v.name.clone(),
                        v.payload
                            .iter()
                            .map(|t| ast_type_to_ty_in(t, module))
                            .collect(),
                    )
                })
                .collect();
            out.insert(
                owned_name(module, &u.name),
                TypeInfo::Union {
                    module: module.to_string(),
                    variants,
                },
            );
        }
    }
}

enum TypeKind {
    Struct,
    Union,
}

/// Lower a `@dataclass` or `@union` class into a parser `TypeDecl`.
pub(super) fn lower_type_decl(
    cx: &Context<'_>,
    class: &py::StmtClassDef,
) -> Result<p::TypeDecl, Diagnostic> {
    let kind = class_kind(cx, class)?;
    let name = class.name.as_str();
    if !super::identifier(name)
        || super::forbidden(name)
        || crate::builtins::is_builtin(name)
        || syntax_name(name)
    {
        return Err(cx.at(
            class,
            format!(
                "reserved/unsafe type name {name}; type names cannot shadow builtins or syntax names"
            ),
        ));
    }
    if !class.bases.is_empty() || !class.keywords.is_empty() || !class.type_params.is_empty() {
        return Err(cx.at(
            class,
            "type declarations cannot have base classes, keyword arguments or type parameters",
        ));
    }
    let body = match kind {
        TypeKind::Struct => p::TypeBody::Struct(lower_struct_fields(cx, class)?),
        TypeKind::Union => p::TypeBody::Union(lower_union_variants(cx, class)?),
    };
    Ok(p::TypeDecl {
        name: name.to_string(),
        body,
    })
}

fn class_kind(cx: &Context<'_>, class: &py::StmtClassDef) -> Result<TypeKind, Diagnostic> {
    if class.decorator_list.len() != 1 {
        return Err(cx.at(
            class,
            "type classes require exactly one decorator: @dataclass (struct) or @union (tagged union)",
        ));
    }
    let decorator = &class.decorator_list[0];
    let py::Expr::Name(n) = decorator else {
        return Err(cx.at(
            decorator,
            "type class decorators must be bare @dataclass or @union",
        ));
    };
    match n.id.as_str() {
        "dataclass" => Ok(TypeKind::Struct),
        "union" => Ok(TypeKind::Union),
        other => Err(cx.at(
            decorator,
            format!("unsupported type decorator @{other}; use @dataclass or @union"),
        )),
    }
}

fn lower_struct_fields(
    cx: &Context<'_>,
    class: &py::StmtClassDef,
) -> Result<Vec<p::FieldType>, Diagnostic> {
    let mut fields = Vec::new();
    for stmt in &class.body {
        let py::Stmt::AnnAssign(a) = stmt else {
            return Err(cx.at(
                stmt,
                "struct bodies may only contain annotated field declarations",
            ));
        };
        let py::Expr::Name(n) = a.target.as_ref() else {
            return Err(cx.at(stmt, "struct fields require a simple name annotation"));
        };
        if a.value.is_some() {
            return Err(cx.at(stmt, "struct fields cannot have default values"));
        }
        validate_member_name(cx, stmt, n.id.as_str())?;
        let ty = annotation(cx, &a.annotation, &TypeCatalog::new())?;
        fields.push(p::FieldType {
            name: n.id.to_string(),
            ty: ty.parser(),
        });
    }
    if fields.is_empty() {
        return Err(cx.at(class, "struct requires at least one field"));
    }
    Ok(fields)
}

fn lower_union_variants(
    cx: &Context<'_>,
    class: &py::StmtClassDef,
) -> Result<Vec<p::VariantType>, Diagnostic> {
    let mut variants = Vec::new();
    for stmt in &class.body {
        let py::Stmt::AnnAssign(a) = stmt else {
            return Err(cx.at(
                stmt,
                "union bodies may only contain variant payload annotations (tuple[...])",
            ));
        };
        let py::Expr::Name(n) = a.target.as_ref() else {
            return Err(cx.at(stmt, "union variants require a simple name annotation"));
        };
        if a.value.is_some() {
            return Err(cx.at(stmt, "union variant annotations cannot have a value"));
        }
        let raw = n.id.as_str();
        let variant_name = keyword_variant(raw).unwrap_or(raw);
        validate_variant_name(cx, stmt, variant_name)?;
        let payload = payload_types(cx, &a.annotation)?;
        variants.push(p::VariantType {
            name: variant_name.to_string(),
            payload: payload.into_iter().map(|t| t.parser()).collect(),
        });
    }
    if variants.len() < 2 {
        return Err(cx.at(
            class,
            "union requires at least two variants; a single-variant union cannot roundtrip to Adapsis",
        ));
    }
    Ok(variants)
}

fn validate_member_name(cx: &Context<'_>, stmt: &py::Stmt, name: &str) -> Result<(), Diagnostic> {
    if !super::identifier(name)
        || super::forbidden(name)
        || name == "_"
        || super::reserved_positional_name(name)
        || syntax_name(name)
    {
        return Err(cx.at(
            stmt,
            format!("unsupported member name {name:?}; choose a simple safe identifier"),
        ));
    }
    Ok(())
}

/// Union variant names additionally allow Python keywords (`None`, `True`,
/// `False`) which arrive via the trailing-underscore convention. Struct fields
/// never do.
fn validate_variant_name(cx: &Context<'_>, stmt: &py::Stmt, name: &str) -> Result<(), Diagnostic> {
    if matches!(name, "None" | "True" | "False") {
        return Ok(());
    }
    validate_member_name(cx, stmt, name)
}

/// A struct field or union variant payload reference must not reuse a name that
/// the frontend treats as syntax. Builtin names are allowed here (`Some`, `Ok`,
/// `Err`, `None` are legitimate union variants) — only the type *name* itself is
/// held to the stricter builtin-collision rule above.
fn syntax_name(name: &str) -> bool {
    matches!(
        name,
        "int"
            | "float"
            | "bool"
            | "str"
            | "Byte"
            | "list"
            | "Result"
            | "Map"
            | "Set"
            | "tuple"
            | "effects"
            | "Unknown"
    )
}

/// PEP 8 trailing-underscore convention for Python-keyword variant names:
/// `None_` declares the `None` variant (and `True_`/`False_` if ever needed).
fn keyword_variant(name: &str) -> Option<&str> {
    let base = name.strip_suffix('_')?;
    if matches!(base, "None" | "True" | "False") {
        Some(base)
    } else {
        None
    }
}

/// Parse a union variant payload annotation `tuple[...]` into payload types.
/// `tuple[()]` is the no-payload form; `tuple[T]`, `tuple[T1, T2]` and
/// string forward references (`tuple["Expr", "Expr"]`) are also accepted.
fn payload_types(cx: &Context<'_>, ann: &py::Expr) -> Result<Vec<Ty>, Diagnostic> {
    let py::Expr::Subscript(s) = ann else {
        return Err(cx.at(
            ann,
            "union variant payload must be tuple[...]; use tuple[()] for no payload",
        ));
    };
    let py::Expr::Name(n) = s.value.as_ref() else {
        return Err(cx.at(ann, "union variant payload must be tuple[...]"));
    };
    if n.id.as_str() != "tuple" {
        return Err(cx.at(ann, "union variant payload must be tuple[...]"));
    }
    match s.slice.as_ref() {
        py::Expr::Tuple(tuple)
            if tuple.elts.len() == 1
                && matches!(tuple.elts[0], py::Expr::Tuple(ref inner) if inner.elts.is_empty()) =>
        {
            Ok(Vec::new())
        }
        py::Expr::Tuple(tuple) => tuple
            .elts
            .iter()
            .map(|element| annotation(cx, element, &TypeCatalog::new()))
            .collect(),
        single => Ok(vec![annotation(cx, single, &TypeCatalog::new())?]),
    }
}
