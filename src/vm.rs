//! Bytecode VM data structures for Adapsis.
//!
//! This module defines the instruction set, execution state, and compiled
//! function representation for the bytecode virtual machine (issue #20).
//! The VM reads functions via `Arc<FunctionDecl>` and executes them using
//! a stack-based bytecode interpreter, replacing the tree-walking evaluator
//! for hot paths.

use crate::eval::Value;

/// A single bytecode instruction.
#[derive(Debug, Clone)]
pub enum Op {
    // ── Literals / constants ─────────────────────────────────────────
    /// Push an integer constant onto the stack.
    PushInt(i64),
    /// Push a float constant onto the stack.
    PushFloat(f64),
    /// Push a boolean constant onto the stack.
    PushBool(bool),
    /// Push a string constant onto the stack.
    PushString(String),
    /// Push the unit value (empty struct / void).
    PushUnit,

    // ── Local variable access ────────────────────────────────────────
    /// Load a local variable by slot index onto the stack.
    LoadLocal(usize),
    /// Pop the top of stack and store into a local variable slot.
    StoreLocal(usize),

    // ── Arithmetic ───────────────────────────────────────────────────
    /// Pop two values, push their sum.
    Add,
    /// Pop two values, push their difference (second - first).
    Sub,
    /// Pop two values, push their product.
    Mul,
    /// Pop two values, push their quotient (second / first).
    Div,
    /// Pop two values, push their remainder (second % first).
    Mod,

    // ── Comparison ───────────────────────────────────────────────────
    /// Pop two values, push true if equal.
    Eq,
    /// Pop two values, push true if not equal.
    Neq,
    /// Pop two values, push true if second < first.
    Lt,
    /// Pop two values, push true if second <= first.
    Lte,
    /// Pop two values, push true if second > first.
    Gt,
    /// Pop two values, push true if second >= first.
    Gte,

    // ── Boolean logic ────────────────────────────────────────────────
    /// Pop two booleans, push logical AND.
    And,
    /// Pop two booleans, push logical OR.
    Or,
    /// Pop one boolean, push logical NOT.
    Not,
    /// Pop one numeric value, push its negation.
    Negate,

    // ── Function calls ───────────────────────────────────────────────
    /// Call a user-defined function by name. Pops `arg_count` values from
    /// the stack as arguments (right-to-left). Pushes the return value.
    Call(String, usize),
    /// Call a builtin function by name with `arg_count` arguments.
    CallBuiltin(String, usize),
    /// Await an async IO operation by name with `arg_count` arguments.
    /// Requires a coroutine handle in the environment.
    AwaitIo(String, usize),

    // ── Control flow ─────────────────────────────────────────────────
    /// Return from the current function. Pops the return value from
    /// the stack.
    Return,
    /// Unconditional jump to instruction at the given offset.
    Jump(usize),
    /// Pop a boolean; if true, jump to the given offset.
    BranchIf(usize),
    /// Pop a boolean; if false, jump to the given offset.
    BranchIfNot(usize),

    // ── Stack manipulation ───────────────────────────────────────────
    /// Discard the top value on the stack.
    Pop,
    /// Duplicate the top value on the stack.
    Dup,

    // ── Struct / composite construction ──────────────────────────────
    /// Pop N values (one per field name), construct a struct.
    /// Field names are in declaration order; values are popped in reverse.
    MakeStruct(Vec<String>),
    /// Pop a struct value, push the value of the named field.
    GetField(String),
    /// Pop N values, construct a list.
    MakeList(usize),

    // ── Result / Option constructors ─────────────────────────────────
    /// Pop one value, wrap in Ok.
    PushOk,
    /// Pop one value (string), wrap in Err.
    PushErr,
    /// Pop one value, wrap in Some (= Ok internally).
    PushSome,
    /// Push None.
    PushNone,

    // ── Union / variant operations ───────────────────────────────────
    /// Construct a union variant with `payload_count` values from the stack.
    PushVariant(String, usize),
    /// Pattern match: pop a value, check if it's the named variant with
    /// the expected binding count. If it matches, push the bindings
    /// (payload values) then push `true`. If not, push `false`.
    MatchVariant(String, usize),
}

/// A call-stack frame, saved when entering a function call.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Instruction pointer to return to in the caller's bytecode.
    pub return_ip: usize,
    /// The caller's bytecode (swapped out during the call).
    pub return_bytecode: Vec<Op>,
    /// Base index into `VmState::locals` for the caller's local variables.
    pub locals_base: usize,
}

/// Runtime state of the bytecode VM during execution.
#[derive(Debug, Clone)]
pub struct VmState {
    /// Name of the currently executing function (for diagnostics).
    pub function_name: String,
    /// Instruction pointer — index into `bytecode`.
    pub ip: usize,
    /// The bytecode being executed.
    pub bytecode: Vec<Op>,
    /// Flat array of local variables (partitioned by call frames via `locals_base`).
    pub locals: Vec<Value>,
    /// Operand stack.
    pub stack: Vec<Value>,
    /// Call stack (saved frames for nested function calls).
    pub call_stack: Vec<Frame>,
}

/// A compiled function ready for VM execution.
#[derive(Debug, Clone)]
pub struct CompiledFunction {
    /// Function name (may be module-qualified, e.g. "Math.add").
    pub name: String,
    /// The bytecode instructions for this function.
    pub bytecode: Vec<Op>,
    /// Number of local variable slots required (includes parameters).
    pub local_count: usize,
    /// Parameter names in declaration order (for binding call arguments).
    pub param_names: Vec<String>,
    /// All local variable names in slot order (for debug/inspect).
    pub local_names: Vec<String>,
}
