use serde::{Deserialize, Serialize};
use std::fmt;

pub type NodeId = String;
pub type Identifier = String;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Program {
    pub modules: Vec<Module>,
    pub functions: Vec<FunctionDecl>,
    pub types: Vec<TypeDecl>,
}

impl Program {
    pub fn get_function(&self, name: &str) -> Option<&FunctionDecl> {
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
                });
        }

        self.functions
            .iter()
            .find(|function| function.name == name)
            .or_else(|| {
                self.modules.iter().find_map(|module| {
                    module
                        .functions
                        .iter()
                        .find(|function| function.name == name)
                })
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
            self.types.len()
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Module {
    pub id: NodeId,
    pub name: Identifier,
    pub types: Vec<TypeDecl>,
    pub functions: Vec<FunctionDecl>,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionDecl {
    pub id: NodeId,
    pub name: Identifier,
    pub params: Vec<Param>,
    pub return_type: Type,
    pub effects: Vec<Effect>,
    pub body: Vec<Statement>,
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
