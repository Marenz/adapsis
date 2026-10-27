use super::call_sites::{self, Effects, Site, Target, UserSignature};
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
    Byte,
    List(Box<Ty>),
    Map(Box<Ty>, Box<Ty>),
    Set(Box<Ty>),
    Option(Box<Ty>),
    Result(Box<Ty>),
    /// Named struct/union identity, owner-qualified internally when resolved.
    /// Parser/Python emission retains only the original declared type name.
    Named(String),
    Unknown,
}

impl Ty {
    pub(super) fn parser(&self) -> p::TypeExpr {
        match self {
            Self::List(t) => p::TypeExpr::Generic {
                name: "List".into(),
                args: vec![t.parser()],
            },
            Self::Map(k, v) => p::TypeExpr::Generic {
                name: "Map".into(),
                args: vec![k.parser(), v.parser()],
            },
            Self::Set(t) => p::TypeExpr::Generic {
                name: "Set".into(),
                args: vec![t.parser()],
            },
            Self::Option(t) => p::TypeExpr::Generic {
                name: "Option".into(),
                args: vec![t.parser()],
            },
            Self::Result(t) => p::TypeExpr::Generic {
                name: "Result".into(),
                args: vec![t.parser()],
            },
            Self::Named(n) => p::TypeExpr::Named(super::types::bare_name(n).into()),
            Self::Int => p::TypeExpr::Named("Int".into()),
            Self::Float => p::TypeExpr::Named("Float".into()),
            Self::Bool => p::TypeExpr::Named("Bool".into()),
            Self::Str => p::TypeExpr::Named("String".into()),
            Self::Byte => p::TypeExpr::Named("Byte".into()),
            Self::Unknown => p::TypeExpr::Named("Unknown".into()),
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

pub(super) fn annotation(
    cx: &Context<'_>,
    e: &py::Expr,
    types: &super::types::TypeCatalog,
) -> Result<Ty, Diagnostic> {
    match e {
        py::Expr::Name(n) => named(cx, e, n.id.as_str(), types),
        py::Expr::Subscript(s) => {
            let py::Expr::Name(n) = s.value.as_ref() else {
                return Err(cx.at(
                    e,
                    "unsupported generic type; use list[T], Result[T], Map[K, V], Set[T], or Option[T]",
                ));
            };
            match n.id.as_str() {
                "list" | "Result" | "Set" | "Option" => {
                    if n.id.as_str() == "Option" && types.contains_key(&super::types::owned_name(cx.module, "Option")) {
                        return Err(cx.at(
                            e,
                            "Option[...] is ambiguous in this module because a type named Option is declared; rename the type or use it bare",
                        ));
                    }
                    let inner = Box::new(annotation(cx, &s.slice, types)?);
                    match n.id.as_str() {
                        "list" => Ok(Ty::List(inner)),
                        "Result" => Ok(Ty::Result(inner)),
                        "Set" => Ok(Ty::Set(inner)),
                        "Option" => Ok(Ty::Option(inner)),
                        _ => unreachable!("matched arm above"),
                    }
                }
                "Map" => {
                    let py::Expr::Tuple(tuple) = s.slice.as_ref() else {
                        return Err(cx.at(
                            e,
                            "Map requires exactly two type arguments: Map[K, V]",
                        ));
                    };
                    if tuple.elts.len() != 2 {
                        return Err(cx.at(
                            e,
                            "Map requires exactly two type arguments: Map[K, V]",
                        ));
                    }
                    let key = Box::new(annotation(cx, &tuple.elts[0], types)?);
                    let value = Box::new(annotation(cx, &tuple.elts[1], types)?);
                    Ok(Ty::Map(key, value))
                }
                _ => Err(cx.at(
                    e,
                    "unsupported generic type; use list[T], Result[T], Map[K, V], Set[T], or Option[T]",
                )),
            }
        }
        py::Expr::Constant(c) => match &c.value {
            py::Constant::Str(name) => named(cx, e, name, types),
            _ => Err(cx.at(e, "unsupported annotation; explicit subset type required")),
        },
        _ => Err(cx.at(e, "unsupported annotation; explicit subset type required")),
    }
}

/// A bare type name is either a primitive (`int`, `float`, `bool`, `str`, `Byte`)
/// or a reference to a declared struct/union type. String forward references
/// (`"Point"`) also land here, which is idiomatic Python for recursive types.
fn named(
    cx: &Context<'_>,
    e: &py::Expr,
    name: &str,
    types: &super::types::TypeCatalog,
) -> Result<Ty, Diagnostic> {
    match name {
        "int" => Ok(Ty::Int),
        "float" => Ok(Ty::Float),
        "bool" => Ok(Ty::Bool),
        "str" => Ok(Ty::Str),
        "Byte" => Ok(Ty::Byte),
        other => {
            if !super::identifier(other) || super::forbidden(other) {
                return Err(cx.at(
                    e,
                    format!("unsupported type name {other:?}; expected a primitive or a declared struct/union type"),
                ));
            }
            let key = super::types::resolve_name(types, cx.module, other)
                .map_err(|message| cx.at(e, message))?
                .unwrap_or_else(|| other.into());
            Ok(Ty::Named(key))
        }
    }
}

pub(super) fn compatible(expected: &Ty, actual: &Ty) -> bool {
    match (expected, actual) {
        (_, Ty::Unknown) => true,
        (Ty::List(a), Ty::List(b)) => compatible(a, b),
        (Ty::Map(ak, av), Ty::Map(bk, bv)) => compatible(ak, bk) && compatible(av, bv),
        (Ty::Set(a), Ty::Set(b))
        | (Ty::Option(a), Ty::Option(b))
        | (Ty::Result(a), Ty::Result(b)) => compatible(a, b),
        _ => expected == actual,
    }
}

/// Collect the named struct/union type references within a `Ty`, recursively.
pub(super) fn collect_ty_names(ty: &Ty, out: &mut Vec<String>) {
    match ty {
        Ty::Named(name) => out.push(super::types::bare_name(name).into()),
        Ty::List(t) | Ty::Set(t) | Ty::Option(t) | Ty::Result(t) => collect_ty_names(t, out),
        Ty::Map(k, v) => {
            collect_ty_names(k, out);
            collect_ty_names(v, out);
        }
        _ => {}
    }
}

pub(super) struct Expressions<'a, 'b> {
    pub cx: &'a Context<'b>,
    pub env: &'a HashMap<String, Ty>,
    pub signatures: &'a HashMap<String, Signature>,
    pub types: &'a super::types::TypeCatalog,
    pub shared_vars: &'a HashMap<String, Ty>,
    pub effects: &'a [String],
}

impl Expressions<'_, '_> {
    pub fn lower(&self, e: &py::Expr) -> Result<(p::Expr, Ty), Diagnostic> {
        match e {
            py::Expr::Constant(c) if matches!(&c.value, py::Constant::None) => self.none_value(),
            py::Expr::Constant(_) | py::Expr::List(_) => self.literal_or_list(e),
            py::Expr::Name(n) => self.name(n.id.as_str(), e),
            py::Expr::Call(c) => {
                if let Some(div_mod) = self.try_div_mod(c)? {
                    return Ok(div_mod);
                }
                if let Some(constructed) = self.try_struct_constructor(c)? {
                    return Ok(constructed);
                }
                if let Some(constructed) = self.try_variant_constructor(c)? {
                    return Ok(constructed);
                }
                self.call(c, Site::Ordinary)
            }
            py::Expr::Attribute(a) => self.field_access(a),
            py::Expr::BinOp(b) => self.binary(b),
            py::Expr::Compare(c) => self.compare(c),
            py::Expr::BoolOp(b) => self.boolean(b),
            py::Expr::UnaryOp(u) => self.unary(u),
            _ => Err(self.cx.at(e, format!("unsupported expression {}; use typed names, literals, direct calls or supported operators", expression_kind(e)))),
        }
    }

    fn name(&self, name: &str, e: &py::Expr) -> Result<(p::Expr, Ty), Diagnostic> {
        if let Some(ty) = self.env.get(name).cloned() {
            return Ok((
                p::Expr::Ident(name.to_string()),
                super::types::resolve_ty(self.types, &ty)
                    .map_err(|message| self.cx.at(e, message))?,
            ));
        }
        if let Some(ty) = self.shared_vars.get(name).cloned() {
            return Ok((
                p::Expr::Ident(name.to_string()),
                super::types::resolve_ty(self.types, &ty)
                    .map_err(|message| self.cx.at(e, message))?,
            ));
        }
        match super::types::variant_payloads(self.types, name, &self.cx.module) {
            Ok(Some((union, payload))) => {
                if payload.is_empty() {
                    return Ok((p::Expr::Ident(name.to_string()), Ty::Named(union)));
                }
                return Err(self.cx.at(
                    e,
                    format!(
                        "variant {name} requires {} payload argument(s)",
                        payload.len()
                    ),
                ));
            }
            Ok(None) => {}
            Err(message) => return Err(self.cx.error(0, message)),
        }
        Err(self.cx.at(
            e,
            format!("unbound or out-of-block name {name}; annotate new local bindings"),
        ))
    }

    /// Shared-var defaults are value-only: literals, list literals, struct
    /// construction and union-variant construction. Function/IO calls are refused.
    /// Shared defaults are data-only. This walker never delegates to ordinary
    /// expression lowering, so a nested call inside a list or record cannot slip
    /// past an outer-node literal check (#slice3).
    pub fn lower_shared_default(&self, e: &py::Expr) -> Result<(p::Expr, Ty), Diagnostic> {
        match e {
            py::Expr::Constant(_) => literal(self.cx, e),
            py::Expr::UnaryOp(u) if matches!(u.op, py::UnaryOp::USub | py::UnaryOp::UAdd) => {
                literal(self.cx, e)
            }
            py::Expr::List(list) => {
                let mut item_ty = Ty::Unknown;
                let mut args = Vec::new();
                for item in &list.elts {
                    let (value, ty) = self.lower_shared_default(item)?;
                    if ty == Ty::Unknown {
                        return Err(self.cx.at(item, "shared list items need a known data type"));
                    }
                    if item_ty != Ty::Unknown && item_ty != ty {
                        return Err(self.cx.at(item, "shared list items must have one element type"));
                    }
                    item_ty = ty;
                    args.push(value);
                }
                Ok((call("list", args), Ty::List(Box::new(item_ty))))
            }
            py::Expr::Name(n) => self.name(n.id.as_str(), e),
            py::Expr::Call(c) => self.shared_constructor(c, e),
            _ => Err(self.cx.at(
                e,
                "shared var default must be a literal, list or record/variant constructor; calls and references are unsupported",
            )),
        }
    }

    fn shared_constructor(
        &self,
        c: &py::ExprCall,
        e: &py::Expr,
    ) -> Result<(p::Expr, Ty), Diagnostic> {
        // Zero-argument calls are ambiguous between a struct with no fields
        // (invalid) and a zero-payload variant. Prefer a declared struct so the
        // field-shape diagnostics fire, then fall back to variants.
        if !c.keywords.is_empty() {
            return self.struct_default(c);
        }
        if let py::Expr::Name(n) = c.func.as_ref() {
            let declared_struct =
                super::types::resolve_name(self.types, self.cx.module, n.id.as_str())
                    .ok()
                    .flatten()
                    .is_some_and(|key| {
                        matches!(
                            self.types.get(&key),
                            Some(super::types::TypeInfo::Struct { .. })
                        )
                    });
            if declared_struct {
                return self.struct_default(c);
            }
        }
        self.variant_default(c, e)
    }

    /// Struct defaults re-check each field value as data rather than reusing
    /// ordinary expression lowering (which would accept calls in field values).
    fn struct_default(&self, c: &py::ExprCall) -> Result<(p::Expr, Ty), Diagnostic> {
        let py::Expr::Name(n) = c.func.as_ref() else {
            return Err(self
                .cx
                .at(c, "shared struct default requires a bare type name"));
        };
        let Some(key) = super::types::resolve_name(self.types, self.cx.module, n.id.as_str())
            .map_err(|message| self.cx.at(c, message))?
        else {
            return Err(self.cx.at(
                c,
                "shared struct default requires a declared local or authorized type",
            ));
        };
        let Some(super::types::TypeInfo::Struct { fields }) = self.types.get(&key).cloned() else {
            return Err(self
                .cx
                .at(c, "shared struct default requires a struct type"));
        };
        let mut provided = std::collections::HashSet::new();
        let mut values = Vec::new();
        for keyword in &c.keywords {
            let Some(field) = &keyword.arg else {
                return Err(self.cx.at(
                    keyword,
                    "shared struct default does not support ** expansion",
                ));
            };
            let Some((_, expected)) = fields.iter().find(|(name, _)| name == field.as_str()) else {
                return Err(self.cx.at(
                    keyword,
                    format!("unknown field {field:?} for shared default"),
                ));
            };
            if !provided.insert(field.to_string()) {
                return Err(self.cx.at(
                    keyword,
                    format!("duplicate field {field:?} in shared default"),
                ));
            }
            let expected = super::types::resolve_ty(self.types, expected)
                .map_err(|message| self.cx.at(keyword, message))?;
            let (value, ty) = self.lower_shared_default(&keyword.value)?;
            if !compatible(&expected, &ty) {
                return Err(self.cx.at(
                    &keyword.value,
                    format!("shared default field {field} expects {expected:?}, got {ty:?}"),
                ));
            }
            values.push(p::FieldValue {
                name: field.to_string(),
                value,
            });
        }
        let missing: Vec<&str> = fields
            .iter()
            .map(|(name, _)| name.as_str())
            .filter(|name| !provided.contains(*name))
            .collect();
        if !missing.is_empty() {
            return Err(self.cx.at(
                c,
                format!(
                    "shared struct default is missing field(s): {}",
                    missing.join(", ")
                ),
            ));
        }
        Ok((p::Expr::StructLiteral(values), Ty::Named(key)))
    }

    fn variant_default(&self, c: &py::ExprCall, e: &py::Expr) -> Result<(p::Expr, Ty), Diagnostic> {
        let py::Expr::Name(n) = c.func.as_ref() else {
            return Err(self.cx.at(
                e,
                "shared var default must be a literal, list or record/variant constructor; calls and references are unsupported",
            ));
        };
        let name = n.id.as_str();
        let (union, payload) = match super::types::variant_payloads(self.types, name, self.cx.module)
            .map_err(|message| self.cx.error(0, message))?
        {
            Some(found) => found,
            None => {
                return Err(self.cx.at(
                    e,
                    "shared var default must be a literal, list or record/variant constructor; function calls are unsupported",
                ))
            }
        };
        if c.args.len() != payload.len() {
            return Err(self.cx.at(
                c,
                format!(
                    "variant {name} expects {} payload argument(s), got {}",
                    payload.len(),
                    c.args.len()
                ),
            ));
        }
        let mut values = Vec::new();
        for (arg, expected) in c.args.iter().zip(&payload) {
            let expected = super::types::resolve_ty(self.types, expected)
                .map_err(|message| self.cx.at(arg, message))?;
            let (value, ty) = self.lower_shared_default(arg)?;
            if !compatible(&expected, &ty) {
                return Err(self.cx.at(
                    arg,
                    format!("payload of {name} expects {expected:?}, got {ty:?}"),
                ));
            }
            values.push(value);
        }
        Ok((call(name, values), Ty::Named(union)))
    }

    /// Python's `None` literal names the Adapsis `None` variant (there is no
    /// unit type). It is accepted structurally even when no `None` variant is
    /// declared — the runtime resolves it like any bare identifier.
    fn none_value(&self) -> Result<(p::Expr, Ty), Diagnostic> {
        let ty = match super::types::variant_payloads(self.types, "None", &self.cx.module) {
            Ok(Some((union, payload))) if payload.is_empty() => Ty::Named(union),
            Ok(_) => Ty::Unknown,
            Err(message) => return Err(self.cx.error(0, message)),
        };
        Ok((p::Expr::Ident("None".to_string()), ty))
    }

    /// `div(a, b)` / `rem(a, b)` are the reserved helpers for the runtime's
    /// integer division and remainder operators (Python's `/` and `%` differ in
    /// semantics). They lower to `BinaryOp::Div`/`BinaryOp::Mod`, not a call.
    fn try_div_mod(&self, c: &py::ExprCall) -> Result<Option<(p::Expr, Ty)>, Diagnostic> {
        if !c.keywords.is_empty() {
            return Ok(None);
        }
        let name = match c.func.as_ref() {
            py::Expr::Name(n) if !self.env.contains_key(n.id.as_str()) => n.id.as_str(),
            _ => return Ok(None),
        };
        let op = match name {
            "div" => p::BinaryOp::Div,
            "rem" => p::BinaryOp::Mod,
            _ => return Ok(None),
        };
        if c.args.len() != 2 {
            return Err(self
                .cx
                .at(c, format!("{name} expects exactly 2 numeric arguments")));
        }
        let (left, left_ty) = self.lower(&c.args[0])?;
        let (right, right_ty) = self.lower(&c.args[1])?;
        if !left_ty.numeric() || !right_ty.numeric() {
            return Err(self.cx.at(c, format!("{name} expects numeric arguments")));
        }
        let ty = if left_ty == Ty::Float || right_ty == Ty::Float {
            Ty::Float
        } else {
            Ty::Int
        };
        Ok(Some((
            p::Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            },
            ty,
        )))
    }

    /// `Point(x=1, y=2)` is a struct construction when `Point` is a declared
    /// struct and every argument is a keyword. Function calls keep the existing
    /// positional-only rule, so this check must run before `call`.
    fn try_struct_constructor(
        &self,
        c: &py::ExprCall,
    ) -> Result<Option<(p::Expr, Ty)>, Diagnostic> {
        let name = match c.func.as_ref() {
            py::Expr::Name(n) if !self.env.contains_key(n.id.as_str()) => n.id.as_str(),
            _ => return Ok(None),
        };
        let Some(key) = super::types::resolve_name(self.types, self.cx.module, name)
            .map_err(|message| self.cx.at(c, message))?
        else {
            return Ok(None);
        };
        let Some(super::types::TypeInfo::Struct { fields }) = self.types.get(&key).cloned() else {
            return Ok(None);
        };
        if self.signatures.contains_key(name)
            || self
                .signatures
                .contains_key(&format!("{}.{}", self.cx.module, name))
        {
            return Err(self.cx.at(
                c,
                format!("{name} is declared as both a struct and a function; rename one of them"),
            ));
        }
        if !c.keywords.is_empty() && !c.args.is_empty() {
            return Err(self.cx.at(
                c,
                "struct construction takes either all keyword fields or none; mixed positional/keyword is unsupported",
            ));
        }
        if !c.keywords.is_empty() {
            let mut provided = std::collections::HashSet::new();
            let mut values = Vec::new();
            for keyword in &c.keywords {
                let Some(field) = &keyword.arg else {
                    return Err(self.cx.at(
                        keyword,
                        "** expansion is unsupported in struct construction",
                    ));
                };
                if !fields.iter().any(|(n, _)| n == field.as_str()) {
                    return Err(self.cx.at(
                        keyword,
                        format!("unknown field {field:?} for struct {name}"),
                    ));
                }
                if !provided.insert(field.to_string()) {
                    return Err(self.cx.at(
                        keyword,
                        format!("duplicate field {field:?} for struct {name}"),
                    ));
                }
                let (value, ty) = self.lower(&keyword.value)?;
                if let Some((_, expected)) = fields.iter().find(|(n, _)| n == field.as_str()) {
                    let expected = super::types::resolve_ty(self.types, expected)
                        .map_err(|message| self.cx.at(&keyword.value, message))?;
                    if !compatible(&expected, &ty) {
                        return Err(self.cx.at(
                            &keyword.value,
                            format!("field {field} of {name} expects {expected:?}, got {ty:?}"),
                        ));
                    }
                }
                values.push(p::FieldValue {
                    name: field.to_string(),
                    value,
                });
            }
            let missing: Vec<&str> = fields
                .iter()
                .map(|(n, _)| n.as_str())
                .filter(|n| !provided.contains(*n))
                .collect();
            if !missing.is_empty() {
                return Err(self.cx.at(
                    c,
                    format!("struct {name} is missing field(s): {}", missing.join(", ")),
                ));
            }
            return Ok(Some((p::Expr::StructLiteral(values), Ty::Named(key))));
        }
        Err(self.cx.at(
            c,
            format!("struct {name} requires keyword fields, e.g. {name}(x=1)"),
        ))
    }

    /// `Cons(head, tail)` is a union variant construction when `Cons` is a
    /// declared variant. Builtins (`Some`, `Ok`, `Err`) are handled by `call`,
    /// so this only runs for non-builtin variant names.
    fn try_variant_constructor(
        &self,
        c: &py::ExprCall,
    ) -> Result<Option<(p::Expr, Ty)>, Diagnostic> {
        if !c.keywords.is_empty() {
            return Ok(None);
        }
        let name = match c.func.as_ref() {
            py::Expr::Name(n) if !self.env.contains_key(n.id.as_str()) => n.id.as_str(),
            _ => return Ok(None),
        };
        if crate::builtins::is_builtin(name) {
            return Ok(None);
        }
        if self.signatures.contains_key(name)
            || self
                .signatures
                .contains_key(&format!("{}.{}", self.cx.module, name))
        {
            if self.types.values().any(|info| matches!(info, super::types::TypeInfo::Union { module, variants } if module == self.cx.module && variants.iter().any(|(variant, _)| variant == name))) {
                return Err(self.cx.at(c, "ambiguous local function/constructor role"));
            }
            return Ok(None);
        }
        let (union, payload) =
            match super::types::variant_payloads(self.types, name, &self.cx.module) {
                Ok(Some(found)) => found,
                Ok(None) => return Ok(None),
                Err(message) => return Err(self.cx.error(0, message)),
            };
        if c.args.len() != payload.len() {
            return Err(self.cx.at(
                c,
                format!(
                    "variant {name} expects {} payload argument(s), got {}",
                    payload.len(),
                    c.args.len()
                ),
            ));
        }
        let values = c
            .args
            .iter()
            .map(|arg| self.lower(arg))
            .collect::<Result<Vec<_>, _>>()?;
        for (expected, (_, actual)) in payload.iter().zip(&values) {
            if !compatible(expected, actual) {
                return Err(self.cx.at(
                    c,
                    format!("variant {name} payload expects {expected:?}, got {actual:?}"),
                ));
            }
        }
        Ok(Some((
            call(name, values.into_iter().map(|(e, _)| e).collect()),
            Ty::Named(union),
        )))
    }

    fn field_access(&self, a: &py::ExprAttribute) -> Result<(p::Expr, Ty), Diagnostic> {
        let (base, base_ty) = self.field_base(a.value.as_ref())?;
        if !matches!(base_ty, Ty::Named(_)) {
            return Err(self
                .cx
                .at(a, "field access requires a struct-typed base name"));
        }
        let field_ty = super::types::struct_field_type(self.types, &base_ty, a.attr.as_str())
            .map_err(|message| self.cx.at(a, message))?
            .ok_or_else(|| {
                self.cx.at(
                    a,
                    format!(
                        "unknown field {} on {base_ty:?}; use a field declared on this struct",
                        a.attr
                    ),
                )
            })?;
        Ok((
            p::Expr::FieldAccess {
                base: Box::new(base),
                field: a.attr.to_string(),
            },
            field_ty,
        ))
    }

    fn field_base(&self, e: &py::Expr) -> Result<(p::Expr, Ty), Diagnostic> {
        match e {
            py::Expr::Name(n) if self.env.contains_key(n.id.as_str()) => Ok((
                p::Expr::Ident(n.id.to_string()),
                self.env[n.id.as_str()].clone(),
            )),
            py::Expr::Attribute(inner) => self.field_access(inner),
            _ => Err(self.cx.at(
                e,
                "field access requires a struct-typed name; dynamic attributes are unsupported",
            )),
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

    pub fn lower_match_subject(&self, e: &py::Expr) -> Result<(p::Expr, Ty), Diagnostic> {
        if let py::Expr::Call(c) = e {
            if let Some(constructed) = self.try_struct_constructor(c)? {
                return Ok(constructed);
            }
            if let Some(constructed) = self.try_variant_constructor(c)? {
                return Ok(constructed);
            }
            return self.call(c, Site::DirectBuiltinMatch);
        }
        self.lower(e)
    }

    pub fn call(&self, c: &py::ExprCall, site: Site) -> Result<(p::Expr, Ty), Diagnostic> {
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
        let target = if resolved.contains('.') {
            Target::User(sig.map(|s| UserSignature {
                effects: Effects::from_names(&s.effects),
                returns_result: matches!(s.result, Ty::Result(_)),
            }))
        } else {
            Target::BuiltinOrConstructor {
                io: crate::builtins::is_io_builtin(&resolved),
            }
        };
        call_sites::check(site, target, Effects::from_names(self.effects))
            .map_err(|message| self.cx.at(c, message))?;
        if matches!(site, Site::DirectBuiltinMatch | Site::AwaitBinding)
            && matches!(target, Target::BuiltinOrConstructor { io: true })
            && self.types.values().any(|info| matches!(info, super::types::TypeInfo::Union { variants, .. } if variants.iter().any(|(variant, _)| variant == &resolved))) {
            return Err(self.cx.at(c, "runtime IO builtin is shadowed by a declared variant; this call requires an unshadowed IO builtin"));
        }
        let values = c
            .args
            .iter()
            .map(|a| self.lower(a))
            .collect::<Result<Vec<_>, _>>()?;
        let result = self.typed_call_result(c, &name, &resolved, sig, &values)?;
        Ok((
            call(&resolved, values.into_iter().map(|(e, _)| e).collect()),
            result,
        ))
    }

    fn typed_call_result(
        &self,
        c: &py::ExprCall,
        name: &str,
        resolved: &str,
        sig: Option<&Signature>,
        values: &[(p::Expr, Ty)],
    ) -> Result<Ty, Diagnostic> {
        if let Some(s) = sig {
            let expected = s
                .params
                .iter()
                .map(|ty| super::types::resolve_ty(self.types, ty))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|message| self.cx.at(c, message))?;
            if s.params.len() != values.len()
                || expected
                    .iter()
                    .zip(values)
                    .any(|(a, (_, b))| !compatible(a, b))
            {
                return Err(self.cx.at(
                    c,
                    format!("arguments do not match declared signature of {resolved}"),
                ));
            }
        }
        // Catalogued builtins: arity and (where known) element types come from
        // one authoritative table; anything uncatalogued keeps `Ty::Unknown`
        // unless a declared signature applies. `concat`/`http_get`/`read_file`
        // are handled explicitly below because their typing is special.
        let catalog_entry = if sig.is_none() {
            super::builtin_signatures::lookup(name)
        } else {
            None
        };
        let result = if let Some(entry) = catalog_entry {
            if entry.params.len() != values.len() {
                // `len` keeps its long-standing wording; other builtins use the
                // generic catalog message.
                return Err(self.cx.at(
                    c,
                    if matches!(name, "len" | "length") {
                        "len expects one argument".to_string()
                    } else {
                        format!(
                            "builtin {name} expects exactly {} argument(s), received {}",
                            entry.params.len(),
                            values.len()
                        )
                    },
                ));
            }
            // `Ty::Unknown` in the catalog means "no type constraint" (the
            // runtime accepts any value of the right shape), so those positions
            // check arity only.
            for (expected, (_, actual)) in entry.params.iter().zip(values.iter()) {
                if *expected != Ty::Unknown && !compatible(expected, actual) {
                    return Err(self.cx.at(
                        c,
                        format!("builtin {name} argument expects {expected:?}, got {actual:?}"),
                    ));
                }
            }
            // `len` is the one catalogued builtin whose runtime rejects known
            // non-collection types, so its element type is validated explicitly.
            if name == "len" || name == "length" {
                if !matches!(
                    &values[0].1,
                    Ty::List(_) | Ty::Str | Ty::Map(_, _) | Ty::Set(_)
                ) {
                    return Err(self.cx.at(c, "len expects a statically known list, str, Map, or Set; unsupported or unresolved argument type — annotate an unresolved call result first"));
                }
            }
            entry.result.clone()
        } else if name == "concat" {
            if values.len() < 2 || values.iter().any(|(_, t)| *t != Ty::Str) {
                return Err(self.cx.at(c, "concat requires at least two str-compatible arguments; implicit coercion is unsupported"));
            }
            Ty::Str
        } else if matches!(name, "read_file" | "file_read") {
            Ty::Str
        } else if name == "http_get" {
            Ty::Result(Box::new(Ty::Str))
        } else {
            match sig {
                Some(s) => super::types::resolve_ty(self.types, &s.result)
                    .map_err(|message| self.cx.at(c, message))?,
                None => Ty::Unknown,
            }
        };
        Ok(result)
    }

    fn binary(&self, b: &py::ExprBinOp) -> Result<(p::Expr, Ty), Diagnostic> {
        let (left, lt) = self.lower(&b.left)?;
        let (right, rt) = self.lower(&b.right)?;
        if b.op == py::Operator::Add && lt == Ty::Str && rt == Ty::Str {
            return Ok((call("concat", vec![left, right]), Ty::Str));
        }
        if !lt.numeric() || lt != rt {
            return Err(self.cx.at(b, "arithmetic requires matching known int/float types; annotate unresolved call results first"));
        }
        let op = match b.op {
            py::Operator::Add => p::BinaryOp::Add, py::Operator::Sub => p::BinaryOp::Sub,
            py::Operator::Mult => p::BinaryOp::Mul,
            _ => return Err(self.cx.at(b, "unsupported arithmetic operator; only +, -, * preserve this runtime's subset semantics")),
        };
        let result = lt;
        Ok((
            p::Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            },
            result,
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
            if matches!(u.op, py::UnaryOp::Not) {
                Ty::Bool
            } else {
                ty
            },
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
    // These spellings are Python's own constructors/coercions. `str` is also an
    // Adapsis runtime alias for `to_string`, but inside Python source it must keep
    // Python's coercion meaning, so it is rejected here rather than silently
    // rebinding to a different semantics. Use `to_string(x)` explicitly.
    if matches!(name, "int" | "float" | "bool" | "str" | "list") {
        return Err(cx.at(
            c,
            "Python constructors/coercions are unsupported; use typed values, list literals, and to_string(x)",
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
