//! Event system for streaming AST mutations to connected clients.

use serde::Serialize;
use tokio::sync::broadcast;

/// Events streamed to the browser over WebSocket.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ForgeEvent {
    /// A new iteration of the feedback loop started
    IterationStart {
        iteration: usize,
        max_iterations: usize,
    },
    /// LLM is thinking (reasoning content)
    Thinking { text: String },
    /// LLM emitted code
    Code { text: String },
    /// A mutation was applied successfully
    MutationOk { message: String },
    /// A mutation failed
    MutationError { message: String },
    /// Type check warning
    TypeWarning { message: String },
    /// A test passed
    TestPass {
        function: String,
        index: usize,
        message: String,
    },
    /// A test failed
    TestFail {
        function: String,
        index: usize,
        message: String,
    },
    /// A trace step
    TraceStep {
        stmt_id: String,
        description: String,
        result: String,
        status: String,
    },
    /// Query result
    QueryResult { query: String, response: String },
    /// Current program state
    ProgramState { summary: String },
    /// The feedback loop completed
    Done,
    /// Full program state as JSON (for rendering)
    ProgramSnapshot {
        modules: Vec<ModuleSnapshot>,
        functions: Vec<FunctionSnapshot>,
        types: Vec<TypeSnapshot>,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct ModuleSnapshot {
    pub name: String,
    pub functions: Vec<FunctionSnapshot>,
    pub types: Vec<TypeSnapshot>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionSnapshot {
    pub name: String,
    pub params: Vec<ParamSnapshot>,
    pub return_type: String,
    pub effects: Vec<String>,
    pub statements: Vec<StatementSnapshot>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ParamSnapshot {
    pub name: String,
    pub ty: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct StatementSnapshot {
    pub id: String,
    pub kind: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct TypeSnapshot {
    pub name: String,
    pub kind: String, // "struct" or "union"
    pub fields: Vec<FieldSnapshot>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FieldSnapshot {
    pub name: String,
    pub ty: String,
}

/// Event bus for broadcasting to multiple WebSocket clients.
#[derive(Clone)]
pub struct EventBus {
    sender: broadcast::Sender<ForgeEvent>,
}

impl EventBus {
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(256);
        Self { sender }
    }

    pub fn send(&self, event: ForgeEvent) {
        // Ignore errors (no receivers connected)
        let _ = self.sender.send(event);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<ForgeEvent> {
        self.sender.subscribe()
    }
}

/// Build a ProgramSnapshot from the AST.
pub fn snapshot_program(program: &crate::ast::Program) -> ForgeEvent {
    let modules = program
        .modules
        .iter()
        .map(|m| ModuleSnapshot {
            name: m.name.clone(),
            functions: m.functions.iter().map(snapshot_function).collect(),
            types: m.types.iter().map(snapshot_type).collect(),
        })
        .collect();

    let functions = program.functions.iter().map(snapshot_function).collect();
    let types = program.types.iter().map(snapshot_type).collect();

    ForgeEvent::ProgramSnapshot {
        modules,
        functions,
        types,
    }
}

fn snapshot_function(func: &crate::ast::FunctionDecl) -> FunctionSnapshot {
    FunctionSnapshot {
        name: func.name.clone(),
        params: func
            .params
            .iter()
            .map(|p| ParamSnapshot {
                name: p.name.clone(),
                ty: format!("{:?}", p.ty),
            })
            .collect(),
        return_type: format!("{:?}", func.return_type),
        effects: func.effects.iter().map(|e| format!("{e:?}")).collect(),
        statements: func
            .body
            .iter()
            .map(|s| StatementSnapshot {
                id: s.id.clone(),
                kind: match &s.kind {
                    crate::ast::StatementKind::Let { name, .. } => format!("let {name}"),
                    crate::ast::StatementKind::Call { call, .. } => {
                        format!("call {}", call.callee)
                    }
                    crate::ast::StatementKind::Check { label, .. } => format!("check {label}"),
                    crate::ast::StatementKind::Branch { .. } => "branch".to_string(),
                    crate::ast::StatementKind::Return { .. } => "return".to_string(),
                    crate::ast::StatementKind::Each { .. } => "each".to_string(),
                    crate::ast::StatementKind::Set { name, .. } => format!("set {name}"),
                    crate::ast::StatementKind::While { .. } => "while".to_string(),
                    crate::ast::StatementKind::Await { name, .. } => format!("await {name}"),
                    crate::ast::StatementKind::Spawn { call } => format!("spawn {}", call.callee),
                    crate::ast::StatementKind::Yield { .. } => "yield".to_string(),
                },
                description: format!("{:?}", s.kind),
            })
            .collect(),
    }
}

fn snapshot_type(td: &crate::ast::TypeDecl) -> TypeSnapshot {
    match td {
        crate::ast::TypeDecl::Struct(s) => TypeSnapshot {
            name: s.name.clone(),
            kind: "struct".to_string(),
            fields: s
                .fields
                .iter()
                .map(|f| FieldSnapshot {
                    name: f.name.clone(),
                    ty: format!("{:?}", f.ty),
                })
                .collect(),
        },
        crate::ast::TypeDecl::TaggedUnion(u) => TypeSnapshot {
            name: u.name.clone(),
            kind: "union".to_string(),
            fields: u
                .variants
                .iter()
                .map(|v| FieldSnapshot {
                    name: v.name.clone(),
                    ty: v
                        .payload
                        .as_ref()
                        .map(|t| format!("{t:?}"))
                        .unwrap_or_default(),
                })
                .collect(),
        },
    }
}
