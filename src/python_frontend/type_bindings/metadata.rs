//! Full-state binding metadata, deliberately excluding fields/variant payloads.
//! This conservatively models validator::struct_field_names precedence and
//! collect_union_names reclassification; ambiguous external fallback is refused
//! even if legacy first-module resolution would happen to choose the safe owner.
use super::super::types;
use crate::{ast, parser};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    Struct,
    Union,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Binding {
    pub key: String,
    pub kind: Kind,
    pub revision: usize,
}

struct Entry {
    owner: String,
    name: String,
    binding: Binding,
    global_union: bool,
}

pub(super) struct Metadata {
    entries: Vec<Entry>,
}

impl Metadata {
    pub fn from_program(program: &ast::Program) -> Self {
        let mut out = Self {
            entries: Vec::new(),
        };
        for ty in &program.types {
            out.existing("", ty, true);
        }
        out.modules(&program.modules, "", true);
        out
    }

    fn modules(&mut self, modules: &[ast::Module], parent: &str, root: bool) {
        for module in modules {
            let owner = types::owned_name(parent, &module.name);
            for ty in &module.types {
                self.existing(&owner, ty, root);
            }
            self.modules(&module.modules, &owner, false);
        }
    }

    fn existing(&mut self, owner: &str, ty: &ast::TypeDecl, global: bool) {
        let kind = match ty {
            ast::TypeDecl::Struct(_) => Kind::Struct,
            ast::TypeDecl::TaggedUnion(_) => Kind::Union,
        };
        self.entries.push(Entry {
            owner: owner.into(),
            name: ty.name().into(),
            binding: Binding {
                key: types::owned_name(owner, ty.name()),
                kind,
                revision: 0,
            },
            global_union: global && kind == Kind::Union,
        });
    }

    pub fn install(
        &mut self,
        owner: &str,
        ty: &parser::TypeDecl,
        revision: usize,
    ) -> Result<(), ()> {
        let kind = match &ty.body {
            parser::TypeBody::Struct(_) => Kind::Struct,
            parser::TypeBody::Union(_) => Kind::Union,
            _ => return Err(()),
        };
        let entry = Entry {
            owner: owner.into(),
            name: ty.name.clone(),
            binding: Binding {
                key: types::owned_name(owner, &ty.name),
                kind,
                revision,
            },
            global_union: !owner.contains('.') && kind == Kind::Union,
        };
        if let Some(slot) = self
            .entries
            .iter_mut()
            .find(|e| e.binding.key == entry.binding.key)
        {
            *slot = entry;
        } else {
            self.entries.push(entry);
        }
        Ok(())
    }

    pub fn function_kind(&self, name: &str) -> Kind {
        if self
            .entries
            .iter()
            .any(|e| e.name == name && e.global_union)
        {
            Kind::Union
        } else {
            Kind::Struct
        }
    }

    pub fn select(&self, owner: &str, name: &str) -> Result<Option<Binding>, ()> {
        let exact: Vec<_> = self
            .entries
            .iter()
            .filter(|e| e.owner == owner && e.name == name)
            .collect();
        if exact.len() > 1 {
            return Err(());
        }
        if let Some(entry) = exact.first() {
            return Ok(Some(entry.binding.clone()));
        }
        // Legacy local lookup searches descendant modules before top-level types.
        // A filtered catalog excludes descendants, so do not silently skip them.
        let nested: Vec<_> = self
            .entries
            .iter()
            .filter(|e| {
                !owner.is_empty() && e.owner.starts_with(&format!("{owner}.")) && e.name == name
            })
            .collect();
        if !nested.is_empty() {
            return Err(());
        }
        let top: Vec<_> = self
            .entries
            .iter()
            .filter(|e| e.owner.is_empty() && e.name == name)
            .collect();
        if top.len() > 1 {
            return Err(());
        }
        if let Some(entry) = top.first() {
            return Ok(Some(entry.binding.clone()));
        }
        let candidates: Vec<_> = self.entries.iter().filter(|e| e.name == name).collect();
        match candidates.as_slice() {
            [] => Ok(None),
            [entry] => Ok(Some(entry.binding.clone())),
            _ => Err(()),
        }
    }
}
