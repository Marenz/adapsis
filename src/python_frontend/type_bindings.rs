//! Native-only pre-executor proof for erasing internal type ownership.
//! The full program supplies ONLY owner/name/kind/order metadata. Schema payloads
//! come exclusively from the already authority-filtered lowering catalog.
mod metadata;

use super::{
    expressions::Ty,
    types::{self, TypeCatalog, TypeInfo},
};
use crate::{ast, parser as p};
use metadata::{Binding, Kind, Metadata};
use std::collections::{HashMap, HashSet};

const REFUSAL: &str = "Python type binding cannot be preserved safely by canonical validation; use an unambiguous authorized local declaration";

#[derive(Clone, Copy)]
enum Mode {
    Function,
    Raw,
    Reference,
    Existing(Kind),
}

struct Guard {
    expected: TypeCatalog,
    revisions: HashMap<String, usize>,
    full: Metadata,
    revision: usize,
}

/// Call only after module/reference authorization and filtered lowering, before
/// any canonical execution. Hidden schema contents never enter an error message.
pub fn check_native_type_bindings(
    operations: &[p::Operation],
    module: &str,
    type_references: &[String],
    references: &[String],
    authorized: &ast::Program,
    full: &ast::Program,
) -> Result<(), String> {
    let mut guard = Guard {
        expected: types::catalog_from_program(authorized),
        revisions: HashMap::new(),
        full: Metadata::from_program(full),
        revision: 0,
    };
    let mut submitted = HashSet::new();
    guard.collect_expected(operations, "", &mut submitted);
    guard.revision = 0;
    let checked = (|| {
        guard.operations(operations, "", Mode::Raw)?;
        // Stored declarations permit forward references; check their dependency
        // closure at the end. Function bindings above are checked at their
        // actual position in the canonical operation sequence.
        for key in guard.revisions.keys() {
            let (owner, name) = key.rsplit_once('.').unwrap_or(("", key));
            guard.named(owner, name, Mode::Reference, &mut HashSet::new())?;
        }
        // Constructors can erase their name into anonymous record expressions;
        // include explicit type references even without an annotated binding.
        for name in type_references {
            if types::resolve_name(&guard.expected, module, name)
                .map_err(|_| ())?
                .is_some()
            {
                guard.named(module, name, Mode::Reference, &mut HashSet::new())?;
            }
        }
        // Existing callee signatures carry their DECLARING owner. Inspect no
        // trusted function bodies and do not substitute the caller's context.
        for target in references.iter().filter(|t| !submitted.contains(*t)) {
            if let (Some((owner, _)), Some(function)) =
                (target.rsplit_once('.'), authorized.get_function(target))
            {
                for param in &function.params {
                    guard.ast_type(owner, &param.ty)?;
                }
                guard.ast_type(owner, &function.return_type)?;
            }
        }
        Ok::<_, ()>(())
    })();
    checked.map_err(|_| REFUSAL.to_owned())
}

impl Guard {
    fn collect_expected(
        &mut self,
        operations: &[p::Operation],
        owner: &str,
        submitted: &mut HashSet<String>,
    ) {
        for op in operations {
            match op {
                p::Operation::Module(m) => {
                    self.collect_expected(&m.body, &types::owned_name(owner, &m.name), submitted)
                }
                p::Operation::Type(t) => {
                    self.revision += 1;
                    super::lower::insert_parser_type(t, owner, &mut self.expected);
                    self.revisions
                        .insert(types::owned_name(owner, &t.name), self.revision);
                }
                p::Operation::Function(f) => {
                    submitted.insert(types::owned_name(owner, &f.name));
                }
                _ => {}
            }
        }
    }

    fn named(
        &self,
        owner: &str,
        name: &str,
        mode: Mode,
        visiting: &mut HashSet<String>,
    ) -> Result<(), ()> {
        let selected = types::resolve_name(&self.expected, owner, name).map_err(|_| ())?;
        let actual = self.full.select(owner, types::bare_name(name))?;
        let Some(key) = selected else {
            return if actual.is_none() { Ok(()) } else { Err(()) };
        };
        let info = self.expected.get(&key).ok_or(())?;
        let kind = match info {
            TypeInfo::Struct { .. } => Kind::Struct,
            TypeInfo::Union { .. } => Kind::Union,
        };
        let expected = Binding {
            key: key.clone(),
            kind,
            revision: *self.revisions.get(&key).unwrap_or(&0),
        };
        // A declaration submitted in this same mutation is not yet part of full
        // state; the canonical executor will install exactly this revision.
        let submitted_now = expected.revision > 0;
        if submitted_now {
            if actual.is_some_and(|found| found != expected) {
                return Err(());
            }
        } else if actual.as_ref() != Some(&expected) {
            return Err(());
        }
        let canonical_kind = match mode {
            Mode::Function => Some(self.full.function_kind(types::bare_name(name))),
            Mode::Raw => Some(Kind::Struct),
            Mode::Existing(kind) => Some(kind),
            Mode::Reference => None,
        };
        if canonical_kind.is_some_and(|actual| actual != kind) {
            return Err(());
        }
        if !visiting.insert(key) {
            return Ok(());
        }
        // Recursive fields/payloads resolve in the type declaration's owner,
        // independent of the module that happens to use this type.
        match info {
            TypeInfo::Struct { fields } => {
                for (_, ty) in fields {
                    self.dependency(ty, visiting)?;
                }
            }
            TypeInfo::Union { variants, .. } => {
                for (_, payload) in variants {
                    for ty in payload {
                        self.dependency(ty, visiting)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn dependency(&self, ty: &Ty, visiting: &mut HashSet<String>) -> Result<(), ()> {
        match ty {
            Ty::Named(name) => {
                let (owner, name) = name.rsplit_once('.').unwrap_or(("", name));
                self.named(owner, name, Mode::Reference, visiting)
            }
            Ty::List(t) | Ty::Set(t) | Ty::Option(t) | Ty::Result(t) => {
                self.dependency(t, visiting)
            }
            Ty::Map(k, v) => {
                self.dependency(k, visiting)?;
                self.dependency(v, visiting)
            }
            _ => Ok(()),
        }
    }

    fn parser_type(&self, owner: &str, ty: &p::TypeExpr, mode: Mode) -> Result<(), ()> {
        match ty {
            p::TypeExpr::Named(name)
                if matches!(name.as_str(), "Int" | "Float" | "Bool" | "String" | "Byte") =>
            {
                Ok(())
            }
            p::TypeExpr::Named(name) => self.named(owner, name, mode, &mut HashSet::new()),
            p::TypeExpr::Generic { args, .. } => {
                for ty in args {
                    self.parser_type(owner, ty, mode)?;
                }
                Ok(())
            }
            p::TypeExpr::Struct(fields) => {
                for field in fields {
                    self.parser_type(owner, &field.ty, mode)?;
                }
                Ok(())
            }
        }
    }

    fn ast_type(&self, owner: &str, ty: &ast::Type) -> Result<(), ()> {
        match ty {
            ast::Type::Struct(name) => self.named(
                owner,
                name,
                Mode::Existing(Kind::Struct),
                &mut HashSet::new(),
            ),
            ast::Type::TaggedUnion(name) => self.named(
                owner,
                name,
                Mode::Existing(Kind::Union),
                &mut HashSet::new(),
            ),
            ast::Type::List(t)
            | ast::Type::Set(t)
            | ast::Type::Option(t)
            | ast::Type::Result(t) => self.ast_type(owner, t),
            ast::Type::Map(k, v) => {
                self.ast_type(owner, k)?;
                self.ast_type(owner, v)
            }
            _ => Ok(()),
        }
    }

    fn operations(
        &mut self,
        operations: &[p::Operation],
        owner: &str,
        mode: Mode,
    ) -> Result<(), ()> {
        for op in operations {
            use p::Operation::*;
            match op {
                Module(m) => {
                    self.operations(&m.body, &types::owned_name(owner, &m.name), Mode::Raw)?
                }
                Type(t) => {
                    self.revision += 1;
                    self.full.install(owner, t, self.revision)?;
                }
                Function(f) => {
                    for param in &f.params {
                        self.parser_type(owner, &param.ty, Mode::Function)?;
                    }
                    self.parser_type(owner, &f.return_type, Mode::Function)?;
                    self.operations(&f.body, owner, Mode::Function)?;
                }
                Startup(f) | Shutdown(f) => self.operations(&f.body, owner, Mode::Raw)?,
                // A shared declaration may hold a struct OR a union; its kind is
                // already proven against the authorized catalog above.
                SharedVar(v) => self.parser_type(owner, &v.ty, Mode::Reference)?,
                Let(v) => self.parser_type(owner, &v.ty, mode)?,
                Call(v) => self.parser_type(owner, &v.ty, mode)?,
                Await(v) => self.parser_type(owner, &v.ty, mode)?,
                Spawn(v) => {
                    if let Some((_, ty)) = &v.binding {
                        self.parser_type(owner, ty, Mode::Raw)?;
                    }
                }
                Each(v) => {
                    self.parser_type(owner, &v.item_type, mode)?;
                    self.operations(&v.body, owner, mode)?;
                }
                If(v) => {
                    self.operations(&v.then_body, owner, mode)?;
                    for (_, body) in &v.elif_branches {
                        self.operations(body, owner, mode)?;
                    }
                    self.operations(&v.else_body, owner, mode)?;
                }
                While(v) => self.operations(&v.body, owner, mode)?,
                Match(v) => {
                    for arm in &v.arms {
                        self.operations(&arm.body, owner, mode)?;
                    }
                }
                _ => {} // No type-bearing positions; ordinary native guards still apply.
            }
        }
        Ok(())
    }
}
