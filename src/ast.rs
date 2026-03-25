use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

pub type NodeId = String;
pub type Identifier = String;

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
    /// Maps function name → index in `functions` Vec. Derived index, not serialized.
    #[serde(skip)]
    fn_index: HashMap<String, usize>,
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
    /// Rebuild the function name → index lookup table. Call after any mutation
    /// to `self.functions`.
    pub fn rebuild_function_index(&mut self) {
        self.fn_index.clear();
        for (i, f) in self.functions.iter().enumerate() {
            self.fn_index.insert(f.name.clone(), i);
        }
    }

    pub fn get_function(&self, name: &str) -> Option<&FunctionDecl> {
        // Module-qualified lookup: "Module.func"
        if let Some((module_name, function_name)) = name.split_once('.') {
            return self
                .modules
                .iter()
                .find(|module| module.name == module_name)
                .and_then(|module| {
                    module
                        .functions
                        .iter()
                        .find(|function| function.name == function_name)
                        .map(|f| f.as_ref())
                });
        }

        // Use index for O(1) lookup on top-level functions
        if !self.fn_index.is_empty() {
            if let Some(&idx) = self.fn_index.get(name) {
                return self.functions.get(idx).map(|f| f.as_ref());
            }
        } else if let Some(f) = self.functions.iter().find(|function| function.name == name) {
            // Fallback: linear scan if index not yet built (e.g. after deserialization
            // before first rebuild)
            return Some(f.as_ref());
        }

        // Search inside modules for unqualified name
        self.modules.iter().find_map(|module| {
            module
                .functions
                .iter()
                .find(|function| function.name == name)
                .map(|f| f.as_ref())
        })
    }

    pub fn get_function_mut(&mut self, name: &str) -> Option<&mut FunctionDecl> {
        // Module-qualified lookup: "Module.func"
        if let Some((module_name, function_name)) = name.split_once('.') {
            return self
                .modules
                .iter_mut()
                .find(|module| module.name == module_name)
                .and_then(|module| {
                    module
                        .functions
                        .iter_mut()
                        .find(|function| function.name == function_name)
                        .map(|f| Arc::make_mut(f))
                });
        }

        // Use index for O(1) lookup on top-level functions
        if !self.fn_index.is_empty() {
            if let Some(&idx) = self.fn_index.get(name) {
                return self.functions.get_mut(idx).map(|f| Arc::make_mut(f));
            }
        } else if let Some(f) = self
            .functions
            .iter_mut()
            .find(|function| function.name == name)
        {
            return Some(Arc::make_mut(f));
        }

        // Search inside modules for unqualified name
        self.modules.iter_mut().find_map(|module| {
            module
                .functions
                .iter_mut()
                .find(|function| function.name == name)
                .map(|f| Arc::make_mut(f))
        })
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Module {
    pub id: NodeId,
    pub name: Identifier,
    pub types: Vec<TypeDecl>,
    pub functions: Vec<Arc<FunctionDecl>>,
    pub modules: Vec<Module>,
    #[serde(default)]
    pub shared_vars: Vec<SharedVarDecl>,
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
    fn get_function_qualified_wrong_module() {
        let mut program = Program::default();
        program
            .modules
            .push(make_module("Math", vec![make_fn("add", Type::Int)]));
        assert!(program.get_function("Utils.add").is_none());
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
}
