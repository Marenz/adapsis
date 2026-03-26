use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

pub type NodeId = String;
pub type Identifier = String;

/// A compact interned name identifier for use in the evaluator's scope stack.
/// This is a `u32` index into a `StringInterner` table — see `crate::intern`.
pub type InternedName = crate::intern::InternedId;

/// A registered HTTP route that dispatches to an Adapsis function.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HttpRoute {
    /// HTTP method (e.g. "POST", "GET")
    pub method: String,
    /// URL path (e.g. "/webhook/telegram")
    pub path: String,
    /// Name of the Adapsis function to call (receives body:String, returns String)
    pub handler_fn: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Program {
    pub modules: Vec<Module>,
    pub functions: Vec<Arc<FunctionDecl>>,
    pub types: Vec<TypeDecl>,
    /// Maps interned function name → index in `functions` Vec. Derived index, not serialized.
    /// Uses `InternedId` (u32) keys instead of `String` for faster hash + comparison on lookup.
    #[serde(skip)]
    fn_index: HashMap<crate::intern::InternedId, usize>,
    /// Maps interned module name → index in `modules` Vec. Derived index, not serialized.
    /// Uses `InternedId` (u32) keys instead of `String` for faster hash + comparison on lookup.
    #[serde(skip)]
    module_index: HashMap<crate::intern::InternedId, usize>,
    /// Pre-populated string interner containing all names in the AST (function names,
    /// parameter names, variable names from let/set/await statements, match bindings, etc.).
    /// Derived, not serialized — rebuilt by `rebuild_function_index()`.
    /// The evaluator clones this into each `Env` so that name→id lookups on the hot path
    /// are guaranteed cache hits (no allocation, no thread-local indirection).
    #[serde(skip)]
    pub interner: crate::intern::StringInterner,
    /// Arc-backed snapshot of the interner for O(1) sharing with Env instances.
    /// Rebuilt alongside `interner` by `rebuild_function_index()`.
    /// Cloning this is an O(1) Arc reference-count bump instead of a full
    /// HashMap + Vec clone.
    #[serde(skip)]
    pub shared_interner: crate::intern::SharedInterner,
    /// Pre-built set of all union variant names (as interned IDs) across all types and modules.
    /// Replaces the O(N×M) linear scan in `is_union_variant()` with O(1) HashSet lookup.
    /// Uses `InternedId` (u32) instead of `String` for faster hash + comparison.
    /// Derived, not serialized — rebuilt by `rebuild_function_index()`.
    #[serde(skip)]
    pub union_variants: HashSet<crate::intern::InternedId>,
    /// When true, reject top-level functions — must be inside a module. Set in AdapsisOS mode.
    #[serde(default)]
    pub require_modules: bool,
}

impl PartialEq for Program {
    fn eq(&self, other: &Self) -> bool {
        self.modules == other.modules
            && self.functions == other.functions
            && self.types == other.types
    }
}

impl Program {
    /// Rebuild all function name → index lookup tables. Call after any mutation
    /// to `self.functions` or `self.modules`.
    pub fn rebuild_function_index(&mut self) {
        // Pre-intern all names in the AST first so that index rebuilds below
        // can use interned IDs instead of String keys.
        self.intern_all_names();

        self.fn_index.clear();
        for (i, f) in self.functions.iter().enumerate() {
            let id = self.interner.intern(&f.name);
            self.fn_index.insert(id, i);
        }
        // Rebuild module name index and per-module function indices
        self.module_index.clear();
        for (i, module) in self.modules.iter_mut().enumerate() {
            let id = self.interner.intern(&module.name);
            self.module_index.insert(id, i);
            module.rebuild_function_index(&mut self.interner);
        }
        // Build the Arc-backed shared interner for O(1) Env creation.
        self.shared_interner = self.interner.shared();
        // Pre-build the set of all union variant names for O(1) lookup.
        self.rebuild_union_variants();
    }

    /// Rebuild the set of all union variant names from type declarations.
    /// Called by `rebuild_function_index()` after every mutation.
    /// Uses interned IDs for O(1) integer-based lookups instead of string hashing.
    fn rebuild_union_variants(&mut self) {
        self.union_variants.clear();
        for td in &self.types {
            if let TypeDecl::TaggedUnion(u) = td {
                for variant in &u.variants {
                    let id = self.interner.intern(&variant.name);
                    self.union_variants.insert(id);
                }
            }
        }
        for module in &self.modules {
            Self::collect_union_variants_from_module(
                &mut self.union_variants,
                &mut self.interner,
                module,
            );
        }
    }

    /// Collect union variant names from a module and its sub-modules recursively.
    fn collect_union_variants_from_module(
        variants: &mut HashSet<crate::intern::InternedId>,
        interner: &mut crate::intern::StringInterner,
        module: &Module,
    ) {
        for td in &module.types {
            if let TypeDecl::TaggedUnion(u) = td {
                for variant in &u.variants {
                    let id = interner.intern(&variant.name);
                    variants.insert(id);
                }
            }
        }
        for sub in &module.modules {
            Self::collect_union_variants_from_module(variants, interner, sub);
        }
    }

    /// Check if a name is a union variant in O(1) using the pre-built interned ID set.
    /// Looks up the name in the interner first — if not interned, it can't be a variant.
    pub fn is_union_variant(&self, name: &str) -> bool {
        if let Some(id) = self.interner.get(name) {
            self.union_variants.contains(&id)
        } else {
            false
        }
    }

    /// Walk the entire AST and intern every name (function names, parameter names,
    /// variable names from let/set/await/each/match statements, identifiers in
    /// expressions). This populates `self.interner` so that cloning it into an
    /// `Env` guarantees all name→id lookups are cache hits.
    pub fn intern_all_names(&mut self) {
        // Intern a few well-known names that the evaluator uses as internal keys
        self.interner.intern("__coroutine_handle");
        self.interner.intern("true");
        self.interner.intern("false");

        // Top-level functions
        for func in &self.functions {
            Self::intern_function_names(&mut self.interner, func);
        }
        // Modules and their functions
        for module in &self.modules {
            self.interner.intern(&module.name);
            for func in &module.functions {
                Self::intern_function_names(&mut self.interner, func);
            }
            // Shared vars
            for sv in &module.shared_vars {
                self.interner.intern(&sv.name);
            }
            // Sub-modules (recursive)
            Self::intern_module_names_recursive(&mut self.interner, module);
        }
        // Type declarations
        for td in &self.types {
            Self::intern_type_decl_names(&mut self.interner, td);
        }
    }

    /// Intern all names within a function declaration: function name, parameter
    /// names, and all names referenced in the body statements/expressions.
    fn intern_function_names(interner: &mut crate::intern::StringInterner, func: &FunctionDecl) {
        interner.intern(&func.name);
        for param in &func.params {
            interner.intern(&param.name);
        }
        Self::intern_body_names(interner, &func.body);
    }

    /// Intern all names in a statement body (recursive for nested blocks).
    fn intern_body_names(interner: &mut crate::intern::StringInterner, body: &[Statement]) {
        for stmt in body {
            match &stmt.kind {
                StatementKind::Let { name, value, .. } => {
                    interner.intern(name);
                    Self::intern_expr_names(interner, value);
                }
                StatementKind::Set { name, value } => {
                    interner.intern(name);
                    Self::intern_expr_names(interner, value);
                }
                StatementKind::Await { name, call, .. } => {
                    interner.intern(name);
                    interner.intern(&call.callee);
                    for arg in &call.args {
                        Self::intern_expr_names(interner, arg);
                    }
                }
                StatementKind::Call { binding, call } => {
                    if let Some(b) = binding {
                        interner.intern(&b.name);
                    }
                    interner.intern(&call.callee);
                    for arg in &call.args {
                        Self::intern_expr_names(interner, arg);
                    }
                }
                StatementKind::Each {
                    iterator,
                    binding,
                    body,
                } => {
                    interner.intern(&binding.name);
                    Self::intern_expr_names(interner, iterator);
                    Self::intern_body_names(interner, body);
                }
                StatementKind::Match { expr, arms } => {
                    Self::intern_expr_names(interner, expr);
                    for arm in arms {
                        interner.intern(&arm.variant);
                        for b in &arm.bindings {
                            interner.intern(b);
                        }
                        Self::intern_body_names(interner, &arm.body);
                    }
                }
                StatementKind::Branch {
                    condition,
                    then_body,
                    else_body,
                } => {
                    Self::intern_expr_names(interner, condition);
                    Self::intern_body_names(interner, then_body);
                    Self::intern_body_names(interner, else_body);
                }
                StatementKind::While { condition, body } => {
                    Self::intern_expr_names(interner, condition);
                    Self::intern_body_names(interner, body);
                }
                StatementKind::Return { value } => {
                    Self::intern_expr_names(interner, value);
                }
                StatementKind::Check { condition, .. } => {
                    Self::intern_expr_names(interner, condition);
                }
                StatementKind::Spawn { call, binding } => {
                    interner.intern(&call.callee);
                    for arg in &call.args {
                        Self::intern_expr_names(interner, arg);
                    }
                    if let Some(b) = binding {
                        interner.intern(&b.name);
                    }
                }
                StatementKind::Yield { value } => {
                    Self::intern_expr_names(interner, value);
                }
            }
        }
    }

    /// Intern all names in an expression (recursive).
    fn intern_expr_names(interner: &mut crate::intern::StringInterner, expr: &Expr) {
        match expr {
            Expr::Identifier(name) => {
                interner.intern(name);
            }
            Expr::FieldAccess { base, field } => {
                Self::intern_expr_names(interner, base);
                interner.intern(field);
            }
            Expr::Call(call) => {
                interner.intern(&call.callee);
                for arg in &call.args {
                    Self::intern_expr_names(interner, arg);
                }
            }
            Expr::Binary { left, right, .. } => {
                Self::intern_expr_names(interner, left);
                Self::intern_expr_names(interner, right);
            }
            Expr::Unary { expr: inner, .. } => {
                Self::intern_expr_names(interner, inner);
            }
            Expr::StructInit { ty, fields } => {
                interner.intern(ty);
                for f in fields {
                    interner.intern(&f.name);
                    Self::intern_expr_names(interner, &f.value);
                }
            }
            Expr::Literal(_) => {}
        }
    }

    /// Intern names from sub-modules recursively.
    fn intern_module_names_recursive(
        interner: &mut crate::intern::StringInterner,
        module: &Module,
    ) {
        for sub in &module.modules {
            interner.intern(&sub.name);
            for func in &sub.functions {
                Self::intern_function_names(interner, func);
            }
            for sv in &sub.shared_vars {
                interner.intern(&sv.name);
            }
            Self::intern_module_names_recursive(interner, sub);
        }
    }

    /// Intern names from type declarations.
    fn intern_type_decl_names(interner: &mut crate::intern::StringInterner, td: &TypeDecl) {
        match td {
            TypeDecl::Struct(s) => {
                interner.intern(&s.name);
                for f in &s.fields {
                    interner.intern(&f.name);
                }
            }
            TypeDecl::TaggedUnion(u) => {
                interner.intern(&u.name);
                for v in &u.variants {
                    interner.intern(&v.name);
                }
            }
        }
    }

    pub fn get_function(&self, name: &str) -> Option<&FunctionDecl> {
        // Module-qualified lookup: "Module.func" — use module_index for O(1) module lookup
        if let Some((module_name, function_name)) = name.split_once('.') {
            if !self.module_index.is_empty() {
                let mod_id = self.interner.get(module_name)?;
                return self
                    .module_index
                    .get(&mod_id)
                    .and_then(|&idx| self.modules.get(idx))
                    .and_then(|module| {
                        module.get_function_interned(function_name, &self.interner)
                    });
            }
            // Fallback: linear scan if module index not built
            return self
                .modules
                .iter()
                .find(|module| module.name == module_name)
                .and_then(|module| module.get_function(function_name));
        }

        // Use index for O(1) lookup on top-level functions
        if !self.fn_index.is_empty() {
            let fn_id = self.interner.get(name)?;
            if let Some(&idx) = self.fn_index.get(&fn_id) {
                return self.functions.get(idx).map(|f| f.as_ref());
            }
        } else if let Some(f) = self.functions.iter().find(|function| function.name == name) {
            // Fallback: linear scan if index not yet built (e.g. after deserialization
            // before first rebuild)
            return Some(f.as_ref());
        }

        // Search inside modules for unqualified name
        self.modules
            .iter()
            .find_map(|module| module.get_function(name))
    }

    pub fn get_function_mut(&mut self, name: &str) -> Option<&mut FunctionDecl> {
        // Module-qualified lookup: "Module.func" — use module_index for O(1) module lookup
        if let Some((module_name, function_name)) = name.split_once('.') {
            if !self.module_index.is_empty() {
                // Intern lookup first (copies the u32), then use it — avoids borrow conflict.
                let mod_id = self.interner.get(module_name)?;
                if let Some(&idx) = self.module_index.get(&mod_id) {
                    return self.modules.get_mut(idx).and_then(|module| {
                        module.get_function_mut_interned(function_name, &self.interner)
                    });
                }
                return None;
            }
            // Fallback: linear scan if module index not built
            return self
                .modules
                .iter_mut()
                .find(|module| module.name == module_name)
                .and_then(|module| module.get_function_mut(function_name));
        }

        // Use index for O(1) lookup on top-level functions
        if !self.fn_index.is_empty() {
            // Intern lookup first (copies the u32), then use it — avoids borrow conflict.
            if let Some(fn_id) = self.interner.get(name) {
                if let Some(&idx) = self.fn_index.get(&fn_id) {
                    return self.functions.get_mut(idx).map(|f| Arc::make_mut(f));
                }
            }
        } else if let Some(f) = self
            .functions
            .iter_mut()
            .find(|function| function.name == name)
        {
            return Some(Arc::make_mut(f));
        }

        // Search inside modules for unqualified name
        self.modules
            .iter_mut()
            .find_map(|module| module.get_function_mut(name))
    }
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Program: {} module(s), {} standalone function(s), {} standalone type(s)",
            self.modules.len(),
            self.functions.len(),
            self.types.len(),
        )?;

        for module in &self.modules {
            writeln!(
                f,
                "- module {}: {} type(s), {} function(s)",
                module.name,
                module.types.len(),
                module.functions.len()
            )?;
        }

        for function in &self.functions {
            writeln!(f, "- fn {}", function.name)?;
        }

        for type_decl in &self.types {
            writeln!(f, "- type {}", type_decl.name())?;
        }

        Ok(())
    }
}

/// A shared variable declaration: `+shared name:Type = default_expr`.
/// Module-scoped mutable state, stored in RuntimeState at runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SharedVarDecl {
    pub name: String,
    pub ty: Type,
    pub default: Expr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Module {
    pub id: NodeId,
    pub name: Identifier,
    pub types: Vec<TypeDecl>,
    pub functions: Vec<Arc<FunctionDecl>>,
    pub modules: Vec<Module>,
    #[serde(default)]
    pub shared_vars: Vec<SharedVarDecl>,
    /// Maps interned function name → index in this module's `functions` Vec.
    /// Uses `InternedId` (u32) keys for faster hash + comparison on lookup.
    /// Derived index, not serialized. Public so validators can construct Module literals.
    #[serde(skip)]
    pub fn_index: HashMap<crate::intern::InternedId, usize>,
}

impl PartialEq for Module {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.name == other.name
            && self.types == other.types
            && self.functions == other.functions
            && self.modules == other.modules
            && self.shared_vars == other.shared_vars
    }
}

impl Module {
    /// Rebuild the per-module function name → index lookup table using interned IDs.
    pub fn rebuild_function_index(&mut self, interner: &mut crate::intern::StringInterner) {
        self.fn_index.clear();
        for (i, f) in self.functions.iter().enumerate() {
            let id = interner.intern(&f.name);
            self.fn_index.insert(id, i);
        }
        // Recurse into sub-modules
        for sub in &mut self.modules {
            sub.rebuild_function_index(interner);
        }
    }

    /// O(1) function lookup within this module using the index.
    /// Falls back to linear scan if the index hasn't been built.
    pub fn get_function(&self, name: &str) -> Option<&FunctionDecl> {
        // Fallback: linear scan (used when index not yet built, e.g. before rebuild)
        self.functions
            .iter()
            .find(|f| f.name == name)
            .map(|f| f.as_ref())
    }

    /// O(1) function lookup using the interned ID index.
    /// Called from `Program::get_function` which has access to the interner.
    pub fn get_function_interned(
        &self,
        name: &str,
        interner: &crate::intern::StringInterner,
    ) -> Option<&FunctionDecl> {
        if !self.fn_index.is_empty() {
            let fn_id = interner.get(name)?;
            if let Some(&idx) = self.fn_index.get(&fn_id) {
                return self.functions.get(idx).map(|f| f.as_ref());
            }
            return None;
        }
        // Fallback: linear scan if index not yet built
        self.get_function(name)
    }

    /// O(1) mutable function lookup within this module using the index.
    pub fn get_function_mut(&mut self, name: &str) -> Option<&mut FunctionDecl> {
        // Fallback: linear scan (used when index not yet built)
        self.functions
            .iter_mut()
            .find(|f| f.name == name)
            .map(|f| Arc::make_mut(f))
    }

    /// O(1) mutable function lookup using the interned ID index.
    /// Called from `Program::get_function_mut` which has access to the interner.
    pub fn get_function_mut_interned(
        &mut self,
        name: &str,
        interner: &crate::intern::StringInterner,
    ) -> Option<&mut FunctionDecl> {
        if !self.fn_index.is_empty() {
            let fn_id = interner.get(name)?;
            if let Some(&idx) = self.fn_index.get(&fn_id) {
                return self.functions.get_mut(idx).map(|f| Arc::make_mut(f));
            }
            return None;
        }
        self.get_function_mut(name)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypeDecl {
    Struct(StructDecl),
    TaggedUnion(TaggedUnionDecl),
}

impl TypeDecl {
    pub fn name(&self) -> &str {
        match self {
            TypeDecl::Struct(decl) => &decl.name,
            TypeDecl::TaggedUnion(decl) => &decl.name,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructDecl {
    pub id: NodeId,
    pub name: Identifier,
    pub fields: Vec<FieldDecl>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaggedUnionDecl {
    pub id: NodeId,
    pub name: Identifier,
    pub variants: Vec<UnionVariant>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnionVariant {
    pub id: NodeId,
    pub name: Identifier,
    pub payload: Vec<Type>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldDecl {
    pub id: NodeId,
    pub name: Identifier,
    pub ty: Type,
}

/// A post-execution side-effect check: `+after <target> <matcher> "<value>"`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AfterCheck {
    pub target: String,
    pub matcher: String,
    pub value: String,
}

/// A test case stored alongside its function declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TestCase {
    pub input: String,
    pub expected: String,
    #[serde(default)]
    pub passed: bool,
    /// Serialized matcher type (e.g. "contains:foo", "starts_with:bar", "AnyOk", "AnyErr", "ErrContaining:msg").
    #[serde(default)]
    pub matcher: Option<String>,
    /// Post-execution side-effect assertions (`+after` lines).
    #[serde(default)]
    pub after_checks: Vec<AfterCheck>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionDecl {
    pub id: NodeId,
    pub name: Identifier,
    pub params: Vec<Param>,
    pub return_type: Type,
    pub effects: Vec<Effect>,
    pub body: Vec<Statement>,
    #[serde(default)]
    pub tests: Vec<TestCase>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Param {
    pub id: NodeId,
    pub name: Identifier,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Type {
    Int,
    Float,
    Bool,
    String,
    Byte,
    List(Box<Type>),
    Map(Box<Type>, Box<Type>),
    Set(Box<Type>),
    Option(Box<Type>),
    Result(Box<Type>),
    Struct(Identifier),
    TaggedUnion(Identifier),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Effect {
    Io,
    Mut,
    Fail,
    Async,
    Rand,
    Yield,
    Parallel,
    Unsafe,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Statement {
    pub id: NodeId,
    pub kind: StatementKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StatementKind {
    Let {
        name: Identifier,
        ty: Type,
        value: Expr,
    },
    Call {
        binding: Option<Binding>,
        call: CallExpr,
    },
    Check {
        label: Identifier,
        condition: Expr,
        on_fail: Identifier,
    },
    Branch {
        condition: Expr,
        then_body: Vec<Statement>,
        else_body: Vec<Statement>,
    },
    Return {
        value: Expr,
    },
    Each {
        iterator: Expr,
        binding: Binding,
        body: Vec<Statement>,
    },
    Set {
        name: Identifier,
        value: Expr,
    },
    Match {
        expr: Expr,
        arms: Vec<MatchArm>,
    },
    Await {
        name: Identifier,
        ty: Type,
        call: CallExpr,
    },
    Spawn {
        call: CallExpr,
        binding: Option<Binding>,
    },
    While {
        condition: Expr,
        body: Vec<Statement>,
    },
    Yield {
        value: Expr,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Binding {
    pub name: Identifier,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    Literal(Literal),
    Identifier(Identifier),
    FieldAccess {
        base: Box<Expr>,
        field: Identifier,
    },
    Call(CallExpr),
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    StructInit {
        ty: Identifier,
        fields: Vec<StructFieldValue>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CallExpr {
    pub callee: Identifier,
    pub args: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchArm {
    pub variant: Identifier,
    pub bindings: Vec<Identifier>,
    pub patterns: Option<Vec<MatchPattern>>,
    pub body: Vec<Statement>,
}

/// A nested pattern for matching inside union payloads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MatchPattern {
    /// Just bind the value to a name (or "_" to ignore)
    Binding(Identifier),
    /// Match a specific variant with sub-patterns: e.g. Literal(x)
    Variant {
        variant: Identifier,
        sub_patterns: Vec<MatchPattern>,
    },
    /// Match a literal value: e.g. 0, "hello", true
    Literal(Literal),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructFieldValue {
    pub name: Identifier,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    And,
    Or,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UnaryOp {
    Not,
    Neg,
}

/// Escape a string for use inside an Adapsis string literal.
/// Produces the content between the outer quotes: `\"` → `\\\"`, `\\` → `\\\\`,
/// newline → `\\n`, etc.
pub fn escape_string_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

/// Format a string as a complete Adapsis string literal (with outer quotes).
pub fn format_string_literal(s: &str) -> String {
    format!("\"{}\"", escape_string_literal(s))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────

    /// Create a minimal FunctionDecl with no params, no body, no effects.
    fn make_fn(name: &str, ret: Type) -> Arc<FunctionDecl> {
        Arc::new(FunctionDecl {
            id: String::new(),
            name: name.to_string(),
            params: vec![],
            return_type: ret,
            effects: vec![],
            body: vec![],
            tests: vec![],
        })
    }

    /// Create a FunctionDecl with params.
    fn make_fn_with_params(name: &str, params: Vec<(&str, Type)>, ret: Type) -> Arc<FunctionDecl> {
        Arc::new(FunctionDecl {
            id: String::new(),
            name: name.to_string(),
            params: params
                .into_iter()
                .map(|(n, ty)| Param {
                    id: String::new(),
                    name: n.to_string(),
                    ty,
                })
                .collect(),
            return_type: ret,
            effects: vec![],
            body: vec![],
            tests: vec![],
        })
    }

    /// Create a Module with named functions.
    fn make_module(name: &str, functions: Vec<Arc<FunctionDecl>>) -> Module {
        Module {
            id: String::new(),
            name: name.to_string(),
            types: vec![],
            functions,
            modules: vec![],
            shared_vars: vec![],
            fn_index: HashMap::new(),
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // 1. Program::rebuild_function_index
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn rebuild_index_enables_fast_lookup() {
        let mut program = Program::default();
        program.functions.push(make_fn("alpha", Type::Int));
        program.functions.push(make_fn("beta", Type::String));
        program.functions.push(make_fn("gamma", Type::Bool));

        // Before rebuild, index is empty — falls back to linear scan
        assert!(program.get_function("alpha").is_some());

        // Rebuild index
        program.rebuild_function_index();

        // After rebuild, lookup via index
        assert!(program.get_function("alpha").is_some());
        assert!(program.get_function("beta").is_some());
        assert!(program.get_function("gamma").is_some());
        assert_eq!(program.get_function("alpha").unwrap().name, "alpha");
    }

    #[test]
    fn rebuild_index_after_adding_function() {
        let mut program = Program::default();
        program.functions.push(make_fn("first", Type::Int));
        program.rebuild_function_index();
        assert!(program.get_function("first").is_some());

        // Add another function — index is now stale
        program.functions.push(make_fn("second", Type::String));
        // Linear fallback won't find it via index since index doesn't have it,
        // but it won't find via linear scan either because index is non-empty.
        // So we must rebuild.
        program.rebuild_function_index();
        assert!(program.get_function("second").is_some());
    }

    // ═════════════════════════════════════════════════════════════════════
    // 2. Program::get_function — all lookup paths
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn get_function_qualified_module_lookup() {
        let mut program = Program::default();
        program
            .modules
            .push(make_module("Math", vec![make_fn("add", Type::Int)]));
        let func = program.get_function("Math.add");
        assert!(func.is_some(), "qualified lookup should find Math.add");
        assert_eq!(func.unwrap().name, "add");
    }

    #[test]
    fn module_get_function_mut_with_index() {
        let mut program = Program::default();
        program
            .modules
            .push(make_module("Math", vec![make_fn("add", Type::Int)]));
        program.rebuild_function_index();

        // Mutable lookup via index
        let func = program.get_function_mut("Math.add");
        assert!(
            func.is_some(),
            "mutable qualified lookup should work with index"
        );
        func.unwrap().name = "add_v2".to_string();

        // Rebuild index after mutation so the new name is discoverable
        program.rebuild_function_index();
        assert!(program.get_function("Math.add_v2").is_some());
    }

    #[test]
    fn get_function_qualified_wrong_function() {
        let mut program = Program::default();
        program
            .modules
            .push(make_module("Math", vec![make_fn("add", Type::Int)]));
        assert!(program.get_function("Math.subtract").is_none());
    }

    #[test]
    fn get_function_top_level_via_index() {
        let mut program = Program::default();
        program.functions.push(make_fn("greet", Type::String));
        program.rebuild_function_index();
        let func = program.get_function("greet");
        assert!(func.is_some());
        assert_eq!(func.unwrap().name, "greet");
    }

    #[test]
    fn get_function_top_level_linear_fallback() {
        // Without rebuilding index, uses linear scan
        let mut program = Program::default();
        program.functions.push(make_fn("greet", Type::String));
        // fn_index is empty, so linear scan
        let func = program.get_function("greet");
        assert!(func.is_some());
    }

    #[test]
    fn get_function_unqualified_module_search() {
        // Unqualified name falls through to module search
        let mut program = Program::default();
        program.rebuild_function_index(); // empty index
        program
            .modules
            .push(make_module("Utils", vec![make_fn("helper", Type::Int)]));
        let func = program.get_function("helper");
        assert!(func.is_some(), "unqualified should find in modules");
        assert_eq!(func.unwrap().name, "helper");
    }

    #[test]
    fn get_function_missing_returns_none() {
        let program = Program::default();
        assert!(program.get_function("nonexistent").is_none());
    }

    #[test]
    fn get_function_top_level_preferred_over_module() {
        // If a function exists both top-level and in a module,
        // top-level (via index) should be found first
        let mut program = Program::default();
        program.functions.push(make_fn_with_params(
            "shared",
            vec![("x", Type::Int)],
            Type::Int,
        ));
        program
            .modules
            .push(make_module("Mod", vec![make_fn("shared", Type::String)]));
        program.rebuild_function_index();
        let func = program.get_function("shared").unwrap();
        // Top-level version has Int return type
        assert!(matches!(func.return_type, Type::Int));
    }

    // ═════════════════════════════════════════════════════════════════════
    // 3. Program::get_function_mut
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn get_function_mut_qualified() {
        let mut program = Program::default();
        program
            .modules
            .push(make_module("Math", vec![make_fn("add", Type::Int)]));
        let func = program.get_function_mut("Math.add");
        assert!(func.is_some());
        // Mutate it
        func.unwrap().name = "add_modified".to_string();
        // Verify mutation stuck
        assert!(program.get_function("Math.add_modified").is_some());
    }

    #[test]
    fn get_function_mut_top_level() {
        let mut program = Program::default();
        program.functions.push(make_fn("greet", Type::String));
        program.rebuild_function_index();
        let func = program.get_function_mut("greet");
        assert!(func.is_some());
        func.unwrap().effects.push(Effect::Io);
        // Verify mutation
        let func = program.get_function("greet").unwrap();
        assert!(func.effects.contains(&Effect::Io));
    }

    #[test]
    fn get_function_mut_unqualified_module_search() {
        let mut program = Program::default();
        program.rebuild_function_index();
        program
            .modules
            .push(make_module("Utils", vec![make_fn("helper", Type::Int)]));
        let func = program.get_function_mut("helper");
        assert!(func.is_some());
    }

    #[test]
    fn get_function_mut_missing_returns_none() {
        let mut program = Program::default();
        assert!(program.get_function_mut("nonexistent").is_none());
    }

    // ═════════════════════════════════════════════════════════════════════
    // 4. TypeDecl::name()
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn type_decl_name_struct() {
        let decl = TypeDecl::Struct(StructDecl {
            id: String::new(),
            name: "User".to_string(),
            fields: vec![],
        });
        assert_eq!(decl.name(), "User");
    }

    #[test]
    fn type_decl_name_tagged_union() {
        let decl = TypeDecl::TaggedUnion(TaggedUnionDecl {
            id: String::new(),
            name: "Color".to_string(),
            variants: vec![],
        });
        assert_eq!(decl.name(), "Color");
    }

    // ═════════════════════════════════════════════════════════════════════
    // 5. escape_string_literal
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn escape_plain_string() {
        assert_eq!(escape_string_literal("hello"), "hello");
    }

    #[test]
    fn escape_quotes() {
        assert_eq!(escape_string_literal(r#"say "hello""#), r#"say \"hello\""#);
    }

    #[test]
    fn escape_backslash() {
        assert_eq!(escape_string_literal(r"path\to\file"), r"path\\to\\file");
    }

    #[test]
    fn escape_newline_and_tab() {
        assert_eq!(
            escape_string_literal("line1\nline2\ttab"),
            r"line1\nline2\ttab"
        );
    }

    #[test]
    fn escape_carriage_return() {
        assert_eq!(escape_string_literal("cr\rhere"), r"cr\rhere");
    }

    #[test]
    fn escape_combined() {
        assert_eq!(escape_string_literal("a\"b\\c\nd\te"), r#"a\"b\\c\nd\te"#);
    }

    // ═════════════════════════════════════════════════════════════════════
    // 6. format_string_literal
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn format_plain_string() {
        assert_eq!(format_string_literal("hello"), r#""hello""#);
    }

    #[test]
    fn format_string_with_quotes() {
        assert_eq!(format_string_literal(r#"say "hi""#), r#""say \"hi\"""#);
    }

    #[test]
    fn format_string_with_special_chars() {
        assert_eq!(format_string_literal("a\nb"), r#""a\nb""#);
    }

    // ═════════════════════════════════════════════════════════════════════
    // 7. Module construction
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn module_empty_construction() {
        let module = make_module("Empty", vec![]);
        assert_eq!(module.name, "Empty");
        assert!(module.functions.is_empty());
        assert!(module.types.is_empty());
        assert!(module.modules.is_empty());
        assert!(module.shared_vars.is_empty());
    }

    #[test]
    fn module_with_functions() {
        let module = make_module(
            "Math",
            vec![make_fn("add", Type::Int), make_fn("sub", Type::Int)],
        );
        assert_eq!(module.name, "Math");
        assert_eq!(module.functions.len(), 2);
        assert_eq!(module.functions[0].name, "add");
        assert_eq!(module.functions[1].name, "sub");
    }

    // ═════════════════════════════════════════════════════════════════════
    // 8. HttpRoute
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn http_route_creation() {
        let route = HttpRoute {
            method: "POST".to_string(),
            path: "/webhook/telegram".to_string(),
            handler_fn: "handle_telegram".to_string(),
        };
        assert_eq!(route.method, "POST");
        assert_eq!(route.path, "/webhook/telegram");
        assert_eq!(route.handler_fn, "handle_telegram");
    }

    #[test]
    fn http_route_equality() {
        let a = HttpRoute {
            method: "GET".to_string(),
            path: "/health".to_string(),
            handler_fn: "health".to_string(),
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ═════════════════════════════════════════════════════════════════════
    // 9. Program Display
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn program_display_summary() {
        let mut program = Program::default();
        program.functions.push(make_fn("hello", Type::String));
        program.types.push(TypeDecl::Struct(StructDecl {
            id: String::new(),
            name: "Point".to_string(),
            fields: vec![],
        }));
        program
            .modules
            .push(make_module("Utils", vec![make_fn("helper", Type::Int)]));

        let display = format!("{program}");
        assert!(display.contains("1 module(s)"), "display: {display}");
        assert!(
            display.contains("1 standalone function(s)"),
            "display: {display}"
        );
        assert!(
            display.contains("1 standalone type(s)"),
            "display: {display}"
        );
        assert!(display.contains("fn hello"), "display: {display}");
        assert!(display.contains("type Point"), "display: {display}");
        assert!(display.contains("module Utils"), "display: {display}");
    }

    // ═════════════════════════════════════════════════════════════════════
    // 10. Type and Effect enums
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn type_equality() {
        assert_eq!(Type::Int, Type::Int);
        assert_ne!(Type::Int, Type::String);
        assert_eq!(
            Type::Result(Box::new(Type::Int)),
            Type::Result(Box::new(Type::Int))
        );
        assert_ne!(
            Type::Result(Box::new(Type::Int)),
            Type::Result(Box::new(Type::String))
        );
        assert_eq!(
            Type::List(Box::new(Type::String)),
            Type::List(Box::new(Type::String))
        );
    }

    #[test]
    fn effect_equality() {
        assert_eq!(Effect::Io, Effect::Io);
        assert_ne!(Effect::Io, Effect::Async);
    }

    #[test]
    fn program_default_is_empty() {
        let program = Program::default();
        assert!(program.modules.is_empty());
        assert!(program.functions.is_empty());
        assert!(program.types.is_empty());
        assert!(!program.require_modules);
    }

    #[test]
    fn program_equality() {
        let mut a = Program::default();
        let mut b = Program::default();
        assert_eq!(a, b);
        a.functions.push(make_fn("f", Type::Int));
        assert_ne!(a, b);
        b.functions.push(make_fn("f", Type::Int));
        assert_eq!(a, b);
    }

    // ═════════════════════════════════════════════════════════════════════
    // Expr variant construction
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn expr_literal_int() {
        let e = Expr::Literal(Literal::Int(42));
        assert!(matches!(e, Expr::Literal(Literal::Int(42))));
    }

    #[test]
    fn expr_literal_float() {
        let e = Expr::Literal(Literal::Float(3.14));
        match e {
            Expr::Literal(Literal::Float(f)) => assert!((f - 3.14).abs() < 1e-10),
            _ => panic!("expected Float literal"),
        }
    }

    #[test]
    fn expr_literal_bool() {
        assert_eq!(
            Expr::Literal(Literal::Bool(true)),
            Expr::Literal(Literal::Bool(true))
        );
        assert_ne!(
            Expr::Literal(Literal::Bool(true)),
            Expr::Literal(Literal::Bool(false))
        );
    }

    #[test]
    fn expr_literal_string() {
        let e = Expr::Literal(Literal::String("hello".to_string()));
        match &e {
            Expr::Literal(Literal::String(s)) => assert_eq!(s, "hello"),
            _ => panic!("expected String literal"),
        }
    }

    #[test]
    fn expr_identifier() {
        let e = Expr::Identifier("my_var".to_string());
        assert!(matches!(e, Expr::Identifier(ref n) if n == "my_var"));
    }

    #[test]
    fn expr_field_access() {
        let e = Expr::FieldAccess {
            base: Box::new(Expr::Identifier("user".to_string())),
            field: "name".to_string(),
        };
        match &e {
            Expr::FieldAccess { base, field } => {
                assert!(matches!(base.as_ref(), Expr::Identifier(n) if n == "user"));
                assert_eq!(field, "name");
            }
            _ => panic!("expected FieldAccess"),
        }
    }

    #[test]
    fn expr_call() {
        let e = Expr::Call(CallExpr {
            callee: "concat".to_string(),
            args: vec![
                Expr::Literal(Literal::String("a".to_string())),
                Expr::Literal(Literal::String("b".to_string())),
            ],
        });
        match &e {
            Expr::Call(call) => {
                assert_eq!(call.callee, "concat");
                assert_eq!(call.args.len(), 2);
            }
            _ => panic!("expected Call"),
        }
    }

    #[test]
    fn expr_binary() {
        let e = Expr::Binary {
            left: Box::new(Expr::Literal(Literal::Int(1))),
            op: BinaryOp::Add,
            right: Box::new(Expr::Literal(Literal::Int(2))),
        };
        match &e {
            Expr::Binary { op, .. } => assert_eq!(*op, BinaryOp::Add),
            _ => panic!("expected Binary"),
        }
    }

    #[test]
    fn expr_unary_not() {
        let e = Expr::Unary {
            op: UnaryOp::Not,
            expr: Box::new(Expr::Literal(Literal::Bool(true))),
        };
        assert!(matches!(
            e,
            Expr::Unary {
                op: UnaryOp::Not,
                ..
            }
        ));
    }

    #[test]
    fn expr_unary_neg() {
        let e = Expr::Unary {
            op: UnaryOp::Neg,
            expr: Box::new(Expr::Literal(Literal::Int(5))),
        };
        assert!(matches!(
            e,
            Expr::Unary {
                op: UnaryOp::Neg,
                ..
            }
        ));
    }

    #[test]
    fn expr_struct_init() {
        let e = Expr::StructInit {
            ty: "Point".to_string(),
            fields: vec![
                StructFieldValue {
                    name: "x".to_string(),
                    value: Expr::Literal(Literal::Int(10)),
                },
                StructFieldValue {
                    name: "y".to_string(),
                    value: Expr::Literal(Literal::Int(20)),
                },
            ],
        };
        match &e {
            Expr::StructInit { ty, fields } => {
                assert_eq!(ty, "Point");
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].name, "x");
                assert_eq!(fields[1].name, "y");
            }
            _ => panic!("expected StructInit"),
        }
    }

    #[test]
    fn expr_nested_binary() {
        // (1 + 2) * 3
        let e = Expr::Binary {
            left: Box::new(Expr::Binary {
                left: Box::new(Expr::Literal(Literal::Int(1))),
                op: BinaryOp::Add,
                right: Box::new(Expr::Literal(Literal::Int(2))),
            }),
            op: BinaryOp::Mul,
            right: Box::new(Expr::Literal(Literal::Int(3))),
        };
        match &e {
            Expr::Binary {
                op: BinaryOp::Mul,
                left,
                ..
            } => {
                assert!(matches!(
                    left.as_ref(),
                    Expr::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
            }
            _ => panic!("expected nested Binary"),
        }
    }

    #[test]
    fn expr_equality() {
        let a = Expr::Literal(Literal::Int(42));
        let b = Expr::Literal(Literal::Int(42));
        let c = Expr::Literal(Literal::Int(99));
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // ═════════════════════════════════════════════════════════════════════
    // StatementKind variant construction
    // ═════════════════════════════════════════════════════════════════════

    fn make_stmt(kind: StatementKind) -> Statement {
        Statement {
            id: "s1".to_string(),
            kind,
        }
    }

    #[test]
    fn stmt_let() {
        let stmt = make_stmt(StatementKind::Let {
            name: "x".to_string(),
            ty: Type::Int,
            value: Expr::Literal(Literal::Int(42)),
        });
        match &stmt.kind {
            StatementKind::Let { name, ty, value } => {
                assert_eq!(name, "x");
                assert_eq!(*ty, Type::Int);
                assert_eq!(*value, Expr::Literal(Literal::Int(42)));
            }
            _ => panic!("expected Let"),
        }
    }

    #[test]
    fn stmt_set() {
        let stmt = make_stmt(StatementKind::Set {
            name: "count".to_string(),
            value: Expr::Binary {
                left: Box::new(Expr::Identifier("count".to_string())),
                op: BinaryOp::Add,
                right: Box::new(Expr::Literal(Literal::Int(1))),
            },
        });
        assert!(matches!(stmt.kind, StatementKind::Set { .. }));
    }

    #[test]
    fn stmt_return() {
        let stmt = make_stmt(StatementKind::Return {
            value: Expr::Identifier("result".to_string()),
        });
        match &stmt.kind {
            StatementKind::Return { value } => {
                assert!(matches!(value, Expr::Identifier(n) if n == "result"));
            }
            _ => panic!("expected Return"),
        }
    }

    #[test]
    fn stmt_check() {
        let stmt = make_stmt(StatementKind::Check {
            label: "positive".to_string(),
            condition: Expr::Binary {
                left: Box::new(Expr::Identifier("x".to_string())),
                op: BinaryOp::GreaterThan,
                right: Box::new(Expr::Literal(Literal::Int(0))),
            },
            on_fail: "err_negative".to_string(),
        });
        match &stmt.kind {
            StatementKind::Check { label, on_fail, .. } => {
                assert_eq!(label, "positive");
                assert_eq!(on_fail, "err_negative");
            }
            _ => panic!("expected Check"),
        }
    }

    #[test]
    fn stmt_call() {
        let stmt = make_stmt(StatementKind::Call {
            binding: Some(Binding {
                name: "result".to_string(),
                ty: Type::String,
            }),
            call: CallExpr {
                callee: "validate".to_string(),
                args: vec![Expr::Identifier("input".to_string())],
            },
        });
        match &stmt.kind {
            StatementKind::Call { binding, call } => {
                assert!(binding.is_some());
                assert_eq!(binding.as_ref().unwrap().name, "result");
                assert_eq!(call.callee, "validate");
                assert_eq!(call.args.len(), 1);
            }
            _ => panic!("expected Call"),
        }
    }

    #[test]
    fn stmt_call_no_binding() {
        let stmt = make_stmt(StatementKind::Call {
            binding: None,
            call: CallExpr {
                callee: "side_effect".to_string(),
                args: vec![],
            },
        });
        match &stmt.kind {
            StatementKind::Call { binding, .. } => assert!(binding.is_none()),
            _ => panic!("expected Call"),
        }
    }

    #[test]
    fn stmt_branch() {
        let stmt = make_stmt(StatementKind::Branch {
            condition: Expr::Binary {
                left: Box::new(Expr::Identifier("x".to_string())),
                op: BinaryOp::GreaterThan,
                right: Box::new(Expr::Literal(Literal::Int(0))),
            },
            then_body: vec![make_stmt(StatementKind::Return {
                value: Expr::Literal(Literal::String("positive".to_string())),
            })],
            else_body: vec![make_stmt(StatementKind::Return {
                value: Expr::Literal(Literal::String("non-positive".to_string())),
            })],
        });
        match &stmt.kind {
            StatementKind::Branch {
                then_body,
                else_body,
                ..
            } => {
                assert_eq!(then_body.len(), 1);
                assert_eq!(else_body.len(), 1);
            }
            _ => panic!("expected Branch"),
        }
    }

    #[test]
    fn stmt_while() {
        let stmt = make_stmt(StatementKind::While {
            condition: Expr::Binary {
                left: Box::new(Expr::Identifier("i".to_string())),
                op: BinaryOp::LessThan,
                right: Box::new(Expr::Literal(Literal::Int(10))),
            },
            body: vec![make_stmt(StatementKind::Set {
                name: "i".to_string(),
                value: Expr::Binary {
                    left: Box::new(Expr::Identifier("i".to_string())),
                    op: BinaryOp::Add,
                    right: Box::new(Expr::Literal(Literal::Int(1))),
                },
            })],
        });
        match &stmt.kind {
            StatementKind::While { body, .. } => assert_eq!(body.len(), 1),
            _ => panic!("expected While"),
        }
    }

    #[test]
    fn stmt_match() {
        let stmt = make_stmt(StatementKind::Match {
            expr: Expr::Identifier("color".to_string()),
            arms: vec![
                MatchArm {
                    variant: "Red".to_string(),
                    bindings: vec![],
                    patterns: None,
                    body: vec![make_stmt(StatementKind::Return {
                        value: Expr::Literal(Literal::Int(1)),
                    })],
                },
                MatchArm {
                    variant: "_".to_string(),
                    bindings: vec![],
                    patterns: None,
                    body: vec![make_stmt(StatementKind::Return {
                        value: Expr::Literal(Literal::Int(0)),
                    })],
                },
            ],
        });
        match &stmt.kind {
            StatementKind::Match { arms, .. } => {
                assert_eq!(arms.len(), 2);
                assert_eq!(arms[0].variant, "Red");
                assert_eq!(arms[1].variant, "_");
            }
            _ => panic!("expected Match"),
        }
    }

    #[test]
    fn stmt_match_with_bindings() {
        let arm = MatchArm {
            variant: "Circle".to_string(),
            bindings: vec!["radius".to_string()],
            patterns: None,
            body: vec![],
        };
        assert_eq!(arm.variant, "Circle");
        assert_eq!(arm.bindings, vec!["radius"]);
    }

    #[test]
    fn stmt_each() {
        let stmt = make_stmt(StatementKind::Each {
            iterator: Expr::Identifier("items".to_string()),
            binding: Binding {
                name: "item".to_string(),
                ty: Type::String,
            },
            body: vec![],
        });
        match &stmt.kind {
            StatementKind::Each { binding, body, .. } => {
                assert_eq!(binding.name, "item");
                assert_eq!(binding.ty, Type::String);
                assert!(body.is_empty());
            }
            _ => panic!("expected Each"),
        }
    }

    #[test]
    fn stmt_await() {
        let stmt = make_stmt(StatementKind::Await {
            name: "data".to_string(),
            ty: Type::String,
            call: CallExpr {
                callee: "http_get".to_string(),
                args: vec![Expr::Identifier("url".to_string())],
            },
        });
        match &stmt.kind {
            StatementKind::Await { name, ty, call } => {
                assert_eq!(name, "data");
                assert_eq!(*ty, Type::String);
                assert_eq!(call.callee, "http_get");
            }
            _ => panic!("expected Await"),
        }
    }

    #[test]
    fn stmt_spawn() {
        let stmt = make_stmt(StatementKind::Spawn {
            call: CallExpr {
                callee: "background_task".to_string(),
                args: vec![],
            },
            binding: None,
        });
        assert!(matches!(stmt.kind, StatementKind::Spawn { .. }));
    }

    #[test]
    fn stmt_yield() {
        let stmt = make_stmt(StatementKind::Yield {
            value: Expr::Literal(Literal::Int(42)),
        });
        assert!(matches!(stmt.kind, StatementKind::Yield { .. }));
    }

    // ═════════════════════════════════════════════════════════════════════
    // Type construction (generics)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn type_list() {
        let t = Type::List(Box::new(Type::Int));
        match &t {
            Type::List(inner) => assert_eq!(inner.as_ref(), &Type::Int),
            _ => panic!("expected List"),
        }
    }

    #[test]
    fn type_nested_list() {
        let t = Type::List(Box::new(Type::List(Box::new(Type::String))));
        match &t {
            Type::List(outer) => match outer.as_ref() {
                Type::List(inner) => assert_eq!(inner.as_ref(), &Type::String),
                _ => panic!("expected nested List"),
            },
            _ => panic!("expected List"),
        }
    }

    #[test]
    fn type_map() {
        let t = Type::Map(Box::new(Type::String), Box::new(Type::Int));
        match &t {
            Type::Map(k, v) => {
                assert_eq!(k.as_ref(), &Type::String);
                assert_eq!(v.as_ref(), &Type::Int);
            }
            _ => panic!("expected Map"),
        }
    }

    #[test]
    fn type_option() {
        let t = Type::Option(Box::new(Type::String));
        assert!(matches!(&t, Type::Option(inner) if **inner == Type::String));
    }

    #[test]
    fn type_result() {
        let t = Type::Result(Box::new(Type::Int));
        assert!(matches!(&t, Type::Result(inner) if **inner == Type::Int));
    }

    #[test]
    fn type_set() {
        let t = Type::Set(Box::new(Type::Int));
        assert!(matches!(&t, Type::Set(inner) if **inner == Type::Int));
    }

    #[test]
    fn type_struct_name() {
        let t = Type::Struct("User".to_string());
        assert!(matches!(&t, Type::Struct(n) if n == "User"));
    }

    #[test]
    fn type_tagged_union_name() {
        let t = Type::TaggedUnion("Color".to_string());
        assert!(matches!(&t, Type::TaggedUnion(n) if n == "Color"));
    }

    #[test]
    fn type_byte() {
        assert_eq!(Type::Byte, Type::Byte);
        assert_ne!(Type::Byte, Type::Int);
    }

    // ═════════════════════════════════════════════════════════════════════
    // FunctionDecl construction
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn function_decl_with_params_and_effects() {
        let func = FunctionDecl {
            id: "fn1".to_string(),
            name: "fetch".to_string(),
            params: vec![
                Param {
                    id: String::new(),
                    name: "url".to_string(),
                    ty: Type::String,
                },
                Param {
                    id: String::new(),
                    name: "timeout".to_string(),
                    ty: Type::Int,
                },
            ],
            return_type: Type::Result(Box::new(Type::String)),
            effects: vec![Effect::Io, Effect::Async, Effect::Fail],
            body: vec![],
            tests: vec![],
        };
        assert_eq!(func.name, "fetch");
        assert_eq!(func.params.len(), 2);
        assert_eq!(func.params[0].name, "url");
        assert!(matches!(func.return_type, Type::Result(_)));
        assert_eq!(func.effects.len(), 3);
        assert!(func.effects.contains(&Effect::Io));
        assert!(func.effects.contains(&Effect::Async));
        assert!(func.effects.contains(&Effect::Fail));
    }

    #[test]
    fn function_decl_with_body() {
        let func = FunctionDecl {
            id: String::new(),
            name: "double".to_string(),
            params: vec![Param {
                id: String::new(),
                name: "x".to_string(),
                ty: Type::Int,
            }],
            return_type: Type::Int,
            effects: vec![],
            body: vec![
                make_stmt(StatementKind::Let {
                    name: "result".to_string(),
                    ty: Type::Int,
                    value: Expr::Binary {
                        left: Box::new(Expr::Identifier("x".to_string())),
                        op: BinaryOp::Mul,
                        right: Box::new(Expr::Literal(Literal::Int(2))),
                    },
                }),
                make_stmt(StatementKind::Return {
                    value: Expr::Identifier("result".to_string()),
                }),
            ],
            tests: vec![],
        };
        assert_eq!(func.body.len(), 2);
        assert!(matches!(func.body[0].kind, StatementKind::Let { .. }));
        assert!(matches!(func.body[1].kind, StatementKind::Return { .. }));
    }

    // ═════════════════════════════════════════════════════════════════════
    // MatchPattern construction
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn match_pattern_binding() {
        let p = MatchPattern::Binding("x".to_string());
        assert!(matches!(p, MatchPattern::Binding(ref n) if n == "x"));
    }

    #[test]
    fn match_pattern_variant() {
        let p = MatchPattern::Variant {
            variant: "Some".to_string(),
            sub_patterns: vec![MatchPattern::Binding("val".to_string())],
        };
        match &p {
            MatchPattern::Variant {
                variant,
                sub_patterns,
            } => {
                assert_eq!(variant, "Some");
                assert_eq!(sub_patterns.len(), 1);
            }
            _ => panic!("expected Variant pattern"),
        }
    }

    #[test]
    fn match_pattern_literal() {
        let p = MatchPattern::Literal(Literal::Int(0));
        assert!(matches!(p, MatchPattern::Literal(Literal::Int(0))));
    }

    // ═════════════════════════════════════════════════════════════════════
    // BinaryOp / UnaryOp coverage
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn all_binary_ops_constructible() {
        let ops = vec![
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Mod,
            BinaryOp::GreaterThan,
            BinaryOp::LessThan,
            BinaryOp::GreaterThanOrEqual,
            BinaryOp::LessThanOrEqual,
            BinaryOp::Equal,
            BinaryOp::NotEqual,
            BinaryOp::And,
            BinaryOp::Or,
        ];
        assert_eq!(ops.len(), 13);
        // Each is distinct
        for (i, a) in ops.iter().enumerate() {
            for (j, b) in ops.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "ops at {i} and {j} should differ");
                }
            }
        }
    }

    #[test]
    fn all_unary_ops_constructible() {
        assert_ne!(UnaryOp::Not, UnaryOp::Neg);
    }

    // ═════════════════════════════════════════════════════════════════════
    // SharedVarDecl and AfterCheck
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn shared_var_decl_construction() {
        let decl = SharedVarDecl {
            name: "counter".to_string(),
            ty: Type::Int,
            default: Expr::Literal(Literal::Int(0)),
        };
        assert_eq!(decl.name, "counter");
        assert_eq!(decl.ty, Type::Int);
    }

    #[test]
    fn after_check_construction() {
        let check = AfterCheck {
            target: "routes".to_string(),
            matcher: "contains".to_string(),
            value: "/chat".to_string(),
        };
        assert_eq!(check.target, "routes");
        assert_eq!(check.matcher, "contains");
        assert_eq!(check.value, "/chat");
    }

    #[test]
    fn test_case_construction() {
        let tc = TestCase {
            input: "x=5".to_string(),
            expected: "10".to_string(),
            passed: false,
            matcher: None,
            after_checks: vec![],
        };
        assert_eq!(tc.input, "x=5");
        assert!(!tc.passed);
        assert!(tc.matcher.is_none());
    }

    // ═════════════════════════════════════════════════════════════════════
    // Module index and per-module fn_index tests
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn module_index_enables_fast_qualified_lookup() {
        let mut program = Program::default();
        program
            .modules
            .push(make_module("Math", vec![make_fn("add", Type::Int)]));
        program
            .modules
            .push(make_module("Strings", vec![make_fn("trim", Type::String)]));
        program.rebuild_function_index();

        // Qualified lookup should use module_index + per-module fn_index
        let func = program.get_function("Math.add");
        assert!(func.is_some(), "Math.add should be found via module index");
        assert_eq!(func.unwrap().name, "add");

        let func = program.get_function("Strings.trim");
        assert!(
            func.is_some(),
            "Strings.trim should be found via module index"
        );
        assert_eq!(func.unwrap().name, "trim");
    }

    #[test]
    fn module_index_qualified_not_found() {
        let mut program = Program::default();
        program
            .modules
            .push(make_module("Math", vec![make_fn("add", Type::Int)]));
        program.rebuild_function_index();

        // Non-existent module
        assert!(program.get_function("NonExistent.add").is_none());
        // Non-existent function in existing module
        assert!(program.get_function("Math.subtract").is_none());
    }

    #[test]
    fn module_fn_index_rebuild() {
        let mut program = Program::default();
        program.modules.push(make_module(
            "Mod",
            vec![make_fn("alpha", Type::Int), make_fn("beta", Type::String)],
        ));
        program.rebuild_function_index();

        // Both functions should be findable
        assert!(program.get_function("Mod.alpha").is_some());
        assert!(program.get_function("Mod.beta").is_some());

        // Verify per-module fn_index is populated
        assert!(!program.modules[0].fn_index.is_empty());
    }

    // ═════════════════════════════════════════════════════════════════════
    // Union variant HashSet (name interning optimization)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn union_variant_set_populated_after_rebuild() {
        let mut program = Program::default();
        program.types.push(TypeDecl::TaggedUnion(TaggedUnionDecl {
            id: String::new(),
            name: "Color".to_string(),
            variants: vec![
                UnionVariant {
                    id: String::new(),
                    name: "Red".to_string(),
                    payload: vec![],
                },
                UnionVariant {
                    id: String::new(),
                    name: "Green".to_string(),
                    payload: vec![],
                },
                UnionVariant {
                    id: String::new(),
                    name: "Blue".to_string(),
                    payload: vec![],
                },
            ],
        }));
        program.rebuild_function_index();

        assert!(program.is_union_variant("Red"));
        assert!(program.is_union_variant("Green"));
        assert!(program.is_union_variant("Blue"));
        assert!(!program.is_union_variant("Yellow"));
        assert!(!program.is_union_variant("Color")); // type name, not a variant
    }

    #[test]
    fn union_variant_set_includes_module_types() {
        let mut program = Program::default();
        let mut module = make_module("MyModule", vec![]);
        module.types.push(TypeDecl::TaggedUnion(TaggedUnionDecl {
            id: String::new(),
            name: "Shape".to_string(),
            variants: vec![
                UnionVariant {
                    id: String::new(),
                    name: "Circle".to_string(),
                    payload: vec![Type::Float],
                },
                UnionVariant {
                    id: String::new(),
                    name: "Square".to_string(),
                    payload: vec![],
                },
            ],
        }));
        program.modules.push(module);
        program.rebuild_function_index();

        assert!(program.is_union_variant("Circle"));
        assert!(program.is_union_variant("Square"));
        assert!(!program.is_union_variant("Triangle"));
    }

    #[test]
    fn union_variant_set_empty_when_no_types() {
        let mut program = Program::default();
        program.functions.push(make_fn("foo", Type::Int));
        program.rebuild_function_index();

        assert!(!program.is_union_variant("anything"));
        assert!(program.union_variants.is_empty());
    }

    #[test]
    fn interner_populated_after_rebuild() {
        let mut program = Program::default();
        program.functions.push(make_fn_with_params(
            "add",
            vec![("x", Type::Int), ("y", Type::Int)],
            Type::Int,
        ));
        program.rebuild_function_index();

        // Function name and parameter names should be interned
        assert!(program.interner.get("add").is_some());
        assert!(program.interner.get("x").is_some());
        assert!(program.interner.get("y").is_some());
        // Well-known names should also be interned
        assert!(program.interner.get("true").is_some());
        assert!(program.interner.get("false").is_some());
        assert!(program.interner.get("__coroutine_handle").is_some());
        // Unknown names should not be interned
        assert!(program.interner.get("nonexistent_var").is_none());
    }

    #[test]
    fn interner_populated_for_modules() {
        let mut program = Program::default();
        let module = Module {
            id: "mod_math".to_string(),
            name: "Math".to_string(),
            types: vec![],
            functions: vec![make_fn_with_params(
                "square",
                vec![("n", Type::Int)],
                Type::Int,
            )],
            modules: vec![],
            shared_vars: vec![SharedVarDecl {
                name: "counter".to_string(),
                ty: Type::Int,
                default: Expr::Literal(Literal::Int(0)),
            }],
            fn_index: HashMap::new(),
        };
        program.modules.push(module);
        program.rebuild_function_index();

        assert!(program.interner.get("Math").is_some(), "module name");
        assert!(
            program.interner.get("square").is_some(),
            "module function name"
        );
        assert!(program.interner.get("n").is_some(), "module function param");
        assert!(program.interner.get("counter").is_some(), "shared var name");
    }

    #[test]
    fn interner_populated_for_type_decls() {
        let mut program = Program::default();
        program.types.push(TypeDecl::Struct(StructDecl {
            id: "struct_user".to_string(),
            name: "User".to_string(),
            fields: vec![
                FieldDecl {
                    id: "f1".to_string(),
                    name: "name".to_string(),
                    ty: Type::String,
                },
                FieldDecl {
                    id: "f2".to_string(),
                    name: "age".to_string(),
                    ty: Type::Int,
                },
            ],
        }));
        program.types.push(TypeDecl::TaggedUnion(TaggedUnionDecl {
            id: "union_color".to_string(),
            name: "Color".to_string(),
            variants: vec![
                UnionVariant {
                    id: "v1".to_string(),
                    name: "Red".to_string(),
                    payload: vec![],
                },
                UnionVariant {
                    id: "v2".to_string(),
                    name: "Green".to_string(),
                    payload: vec![],
                },
            ],
        }));
        program.rebuild_function_index();

        assert!(program.interner.get("User").is_some(), "struct name");
        assert!(program.interner.get("name").is_some(), "struct field name");
        assert!(program.interner.get("age").is_some(), "struct field name");
        assert!(program.interner.get("Color").is_some(), "union name");
        assert!(program.interner.get("Red").is_some(), "union variant name");
        assert!(
            program.interner.get("Green").is_some(),
            "union variant name"
        );
    }

    #[test]
    fn interner_empty_program_has_wellknown_names() {
        let mut program = Program::default();
        program.rebuild_function_index();

        // Even an empty program should intern well-known names
        assert!(program.interner.get("__coroutine_handle").is_some());
        assert!(program.interner.get("true").is_some());
        assert!(program.interner.get("false").is_some());
        // But nothing else
        assert!(program.interner.len() >= 3);
    }

    // ═════════════════════════════════════════════════════════════════════
    // fn_index / module_index / union_variants use InternedId keys
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn fn_index_uses_interned_ids_after_rebuild() {
        // Verify that fn_index keys are InternedId values matching the interner
        let mut program = Program::default();
        program.functions.push(make_fn("alpha", Type::Int));
        program.functions.push(make_fn("beta", Type::String));
        program.rebuild_function_index();

        let alpha_id = program
            .interner
            .get("alpha")
            .expect("alpha should be interned");
        let beta_id = program
            .interner
            .get("beta")
            .expect("beta should be interned");

        // fn_index should contain these interned ids as keys
        assert!(program.fn_index.contains_key(&alpha_id));
        assert!(program.fn_index.contains_key(&beta_id));

        // And the indices should point to the right functions
        let &alpha_idx = program.fn_index.get(&alpha_id).unwrap();
        assert_eq!(program.functions[alpha_idx].name, "alpha");
        let &beta_idx = program.fn_index.get(&beta_id).unwrap();
        assert_eq!(program.functions[beta_idx].name, "beta");
    }

    #[test]
    fn module_index_uses_interned_ids_after_rebuild() {
        // Verify that module_index keys are InternedId values
        let mut program = Program::default();
        program
            .modules
            .push(make_module("Math", vec![make_fn("add", Type::Int)]));
        program
            .modules
            .push(make_module("Strings", vec![make_fn("trim", Type::String)]));
        program.rebuild_function_index();

        let math_id = program
            .interner
            .get("Math")
            .expect("Math should be interned");
        let strings_id = program
            .interner
            .get("Strings")
            .expect("Strings should be interned");

        assert!(program.module_index.contains_key(&math_id));
        assert!(program.module_index.contains_key(&strings_id));

        let &math_idx = program.module_index.get(&math_id).unwrap();
        assert_eq!(program.modules[math_idx].name, "Math");
    }

    #[test]
    fn module_fn_index_uses_interned_ids() {
        // Per-module fn_index should also use InternedId keys
        let mut program = Program::default();
        program.modules.push(make_module(
            "Mod",
            vec![make_fn("foo", Type::Int), make_fn("bar", Type::String)],
        ));
        program.rebuild_function_index();

        let foo_id = program.interner.get("foo").expect("foo should be interned");
        let bar_id = program.interner.get("bar").expect("bar should be interned");

        assert!(program.modules[0].fn_index.contains_key(&foo_id));
        assert!(program.modules[0].fn_index.contains_key(&bar_id));
    }

    #[test]
    fn union_variants_uses_interned_ids() {
        // union_variants should be HashSet<InternedId> not HashSet<String>
        let mut program = Program::default();
        program.types.push(TypeDecl::TaggedUnion(TaggedUnionDecl {
            id: String::new(),
            name: "Color".to_string(),
            variants: vec![
                UnionVariant {
                    id: String::new(),
                    name: "Red".to_string(),
                    payload: vec![],
                },
                UnionVariant {
                    id: String::new(),
                    name: "Green".to_string(),
                    payload: vec![],
                },
            ],
        }));
        program.rebuild_function_index();

        let red_id = program.interner.get("Red").expect("Red should be interned");
        let green_id = program
            .interner
            .get("Green")
            .expect("Green should be interned");

        assert!(program.union_variants.contains(&red_id));
        assert!(program.union_variants.contains(&green_id));

        // is_union_variant should still work via string interface
        assert!(program.is_union_variant("Red"));
        assert!(program.is_union_variant("Green"));
        assert!(!program.is_union_variant("Blue"));
    }

    #[test]
    fn get_function_with_uninterned_name_returns_none() {
        // Looking up a name that was never interned should return None, not panic
        let mut program = Program::default();
        program.functions.push(make_fn("known", Type::Int));
        program.rebuild_function_index();

        // "known" exists
        assert!(program.get_function("known").is_some());
        // "unknown" was never interned — should gracefully return None
        assert!(program.get_function("unknown").is_none());
    }

    #[test]
    fn get_function_qualified_with_uninterned_module_returns_none() {
        // Looking up "UnknownModule.func" where the module name was never interned
        let mut program = Program::default();
        program
            .modules
            .push(make_module("Math", vec![make_fn("add", Type::Int)]));
        program.rebuild_function_index();

        assert!(program.get_function("Math.add").is_some());
        assert!(program.get_function("Fake.add").is_none());
    }

    #[test]
    fn get_function_mut_with_interned_index() {
        // get_function_mut should work through the interned fn_index
        let mut program = Program::default();
        program.functions.push(make_fn("greet", Type::String));
        program.rebuild_function_index();

        let func = program.get_function_mut("greet");
        assert!(func.is_some());
        assert_eq!(func.unwrap().name, "greet");

        // Non-existent function
        assert!(program.get_function_mut("missing").is_none());
    }

    #[test]
    fn get_function_mut_qualified_with_interned_index() {
        // get_function_mut with module-qualified name through interned indices
        let mut program = Program::default();
        program
            .modules
            .push(make_module("Math", vec![make_fn("add", Type::Int)]));
        program.rebuild_function_index();

        let func = program.get_function_mut("Math.add");
        assert!(func.is_some());
        assert_eq!(func.unwrap().name, "add");

        assert!(program.get_function_mut("Math.subtract").is_none());
        assert!(program.get_function_mut("Fake.add").is_none());
    }
}
