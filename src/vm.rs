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
    /// The caller's bytecode (shared Arc, no clone on call/return).
    pub return_bytecode: std::sync::Arc<Vec<Op>>,
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
    /// The bytecode being executed (shared, never cloned on call).
    pub bytecode: std::sync::Arc<Vec<Op>>,
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
    /// The bytecode instructions for this function (shared, cheap to clone).
    pub bytecode: std::sync::Arc<Vec<Op>>,
    /// Number of local variable slots required (includes parameters).
    pub local_count: usize,
    /// Parameter names in declaration order (for binding call arguments).
    pub param_names: Vec<String>,
    /// All local variable names in slot order (for debug/inspect).
    pub local_names: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════════
// Bytecode Compiler: AST FunctionDecl → CompiledFunction
// ═══════════════════════════════════════════════════════════════════════

use anyhow::{bail, Result};
use std::collections::HashMap;

use crate::ast;

/// Compile an AST function declaration into bytecode.
pub fn compile_function(
    func: &ast::FunctionDecl,
    program: &ast::Program,
) -> Result<CompiledFunction> {
    let mut compiler = Compiler::new(program);

    // Allocate parameter slots first (slot 0..N-1)
    for param in &func.params {
        compiler.alloc_local(&param.name);
    }

    compiler.compile_body(&func.body)?;

    // If the function body doesn't end with a Return, emit a default PushUnit + Return
    if !matches!(compiler.bytecode.last(), Some(Op::Return)) {
        compiler.emit(Op::PushUnit);
        compiler.emit(Op::Return);
    }

    Ok(CompiledFunction {
        name: func.name.clone(),
        bytecode: std::sync::Arc::new(compiler.bytecode),
        local_count: compiler.local_count,
        param_names: func.params.iter().map(|p| p.name.clone()).collect(),
        local_names: compiler.local_names.clone(),
    })
}

/// Internal compiler state.
struct Compiler<'a> {
    bytecode: Vec<Op>,
    locals: HashMap<String, usize>,
    local_names: Vec<String>,
    local_count: usize,
    program: &'a ast::Program,
}

impl<'a> Compiler<'a> {
    fn new(program: &'a ast::Program) -> Self {
        Self {
            bytecode: Vec::new(),
            locals: HashMap::new(),
            local_names: Vec::new(),
            local_count: 0,
            program,
        }
    }

    fn emit(&mut self, op: Op) {
        self.bytecode.push(op);
    }

    /// Current bytecode offset (for jump targets).
    fn here(&self) -> usize {
        self.bytecode.len()
    }

    /// Patch a jump/branch instruction at `addr` to point to `target`.
    fn patch(&mut self, addr: usize, target: usize) {
        match &mut self.bytecode[addr] {
            Op::Jump(t) | Op::BranchIf(t) | Op::BranchIfNot(t) => {
                *t = target;
            }
            _ => panic!("patch: instruction at {addr} is not a jump/branch"),
        }
    }

    /// Allocate a new local variable slot. Returns the slot index.
    fn alloc_local(&mut self, name: &str) -> usize {
        let slot = self.local_count;
        self.locals.insert(name.to_string(), slot);
        self.local_names.push(name.to_string());
        self.local_count += 1;
        slot
    }

    /// Look up an existing local variable by name.
    fn resolve_local(&self, name: &str) -> Option<usize> {
        self.locals.get(name).copied()
    }

    // ── Body / Statement compilation ─────────────────────────────────

    fn compile_body(&mut self, stmts: &[ast::Statement]) -> Result<()> {
        for stmt in stmts {
            self.compile_statement(stmt)?;
        }
        Ok(())
    }

    fn compile_statement(&mut self, stmt: &ast::Statement) -> Result<()> {
        match &stmt.kind {
            ast::StatementKind::Let { name, value, .. } => {
                self.compile_expr(value)?;
                let slot = self.alloc_local(name);
                self.emit(Op::StoreLocal(slot));
            }

            ast::StatementKind::Set { name, value } => {
                self.compile_expr(value)?;
                match self.resolve_local(name) {
                    Some(slot) => self.emit(Op::StoreLocal(slot)),
                    None => bail!("compile: undefined variable `{name}` in set"),
                }
            }

            ast::StatementKind::Return { value } => {
                self.compile_expr(value)?;
                self.emit(Op::Return);
            }

            ast::StatementKind::Call { binding, call } => {
                self.compile_call_expr(call)?;
                if let Some(binding) = binding {
                    let slot = self.alloc_local(&binding.name);
                    self.emit(Op::StoreLocal(slot));
                } else {
                    self.emit(Op::Pop);
                }
            }

            ast::StatementKind::Check {
                condition, on_fail, ..
            } => {
                // Compile condition; if false, push Err(on_fail) + return
                self.compile_expr(condition)?;
                let fail_jump = self.here();
                self.emit(Op::BranchIfNot(0)); // placeholder

                // Condition was true — continue (no-op)
                let past_fail = self.here();
                self.emit(Op::Jump(0)); // placeholder — jump past fail block

                // Fail path
                let fail_target = self.here();
                self.patch(fail_jump, fail_target);
                self.emit(Op::PushString(on_fail.clone()));
                self.emit(Op::PushErr);
                self.emit(Op::Return);

                // Continue path
                let continue_target = self.here();
                self.patch(past_fail, continue_target);
            }

            ast::StatementKind::Branch {
                condition,
                then_body,
                else_body,
            } => {
                self.compile_expr(condition)?;
                let else_jump = self.here();
                self.emit(Op::BranchIfNot(0)); // placeholder

                // Then branch
                self.compile_body(then_body)?;
                let past_else = self.here();
                if !else_body.is_empty() {
                    self.emit(Op::Jump(0)); // placeholder
                }

                // Else branch
                let else_target = self.here();
                self.patch(else_jump, else_target);
                if !else_body.is_empty() {
                    self.compile_body(else_body)?;
                    let end_target = self.here();
                    self.patch(past_else, end_target);
                }
            }

            ast::StatementKind::While { condition, body } => {
                let loop_start = self.here();
                self.compile_expr(condition)?;
                let exit_jump = self.here();
                self.emit(Op::BranchIfNot(0)); // placeholder

                self.compile_body(body)?;
                self.emit(Op::Jump(loop_start));

                let loop_end = self.here();
                self.patch(exit_jump, loop_end);
            }

            ast::StatementKind::Each {
                iterator,
                binding,
                body,
            } => {
                // Compile: let __list = <iterator>
                self.compile_expr(iterator)?;
                let list_slot = self.alloc_local("__each_list");
                self.emit(Op::StoreLocal(list_slot));

                // let __idx = 0
                self.emit(Op::PushInt(0));
                let idx_slot = self.alloc_local("__each_idx");
                self.emit(Op::StoreLocal(idx_slot));

                // let __len = len(__list)
                self.emit(Op::LoadLocal(list_slot));
                self.emit(Op::CallBuiltin("len".to_string(), 1));
                let len_slot = self.alloc_local("__each_len");
                self.emit(Op::StoreLocal(len_slot));

                // Loop start: if __idx >= __len, break
                let loop_start = self.here();
                self.emit(Op::LoadLocal(idx_slot));
                self.emit(Op::LoadLocal(len_slot));
                self.emit(Op::Lt); // idx < len
                let exit_jump = self.here();
                self.emit(Op::BranchIfNot(0)); // placeholder

                // let <binding> = get(__list, __idx)
                self.emit(Op::LoadLocal(list_slot));
                self.emit(Op::LoadLocal(idx_slot));
                self.emit(Op::CallBuiltin("get".to_string(), 2));
                let item_slot = self.alloc_local(&binding.name);
                self.emit(Op::StoreLocal(item_slot));

                // Compile body
                self.compile_body(body)?;

                // __idx = __idx + 1
                self.emit(Op::LoadLocal(idx_slot));
                self.emit(Op::PushInt(1));
                self.emit(Op::Add);
                self.emit(Op::StoreLocal(idx_slot));

                self.emit(Op::Jump(loop_start));

                let loop_end = self.here();
                self.patch(exit_jump, loop_end);
            }

            ast::StatementKind::Match { expr, arms } => {
                self.compile_expr(expr)?;
                // Store the match subject in a temp local
                let subject_slot = self.alloc_local("__match_subject");
                self.emit(Op::StoreLocal(subject_slot));

                let mut end_jumps = Vec::new();

                for arm in arms {
                    if arm.variant == "_" {
                        // Wildcard — always matches
                        self.compile_body(&arm.body)?;
                        // No need to jump past — this is the last arm
                        break;
                    }

                    // MatchVariant: load subject, check variant, push bindings + true/false
                    self.emit(Op::LoadLocal(subject_slot));
                    self.emit(Op::MatchVariant(arm.variant.clone(), arm.bindings.len()));
                    let skip_jump = self.here();
                    self.emit(Op::BranchIfNot(0)); // placeholder

                    // Store bindings into locals
                    // MatchVariant pushes bindings in order, then true.
                    // After BranchIfNot consumed the true, bindings are on stack.
                    for binding_name in arm.bindings.iter().rev() {
                        let slot = self.alloc_local(binding_name);
                        self.emit(Op::StoreLocal(slot));
                    }

                    self.compile_body(&arm.body)?;
                    let end_jump = self.here();
                    self.emit(Op::Jump(0)); // placeholder — jump to end of match
                    end_jumps.push(end_jump);

                    let next_arm = self.here();
                    self.patch(skip_jump, next_arm);
                }

                // Patch all end jumps
                let match_end = self.here();
                for j in end_jumps {
                    self.patch(j, match_end);
                }
            }

            ast::StatementKind::Await { name, call, .. } => {
                // Compile args
                for arg in &call.args {
                    self.compile_expr(arg)?;
                }
                self.emit(Op::AwaitIo(call.callee.clone(), call.args.len()));
                let slot = self.alloc_local(name);
                self.emit(Op::StoreLocal(slot));
            }

            ast::StatementKind::Spawn { call, binding } => {
                self.compile_call_expr(call)?;
                if let Some(binding) = binding {
                    let slot = self.alloc_local(&binding.name);
                    self.emit(Op::StoreLocal(slot));
                } else {
                    self.emit(Op::Pop);
                }
            }

            ast::StatementKind::Yield { value } => {
                self.compile_expr(value)?;
                // Yield is conceptually a return for generators — use Return for now
                self.emit(Op::Return);
            }
        }
        Ok(())
    }

    // ── Expression compilation ───────────────────────────────────────

    fn compile_expr(&mut self, expr: &ast::Expr) -> Result<()> {
        match expr {
            ast::Expr::Literal(lit) => match lit {
                ast::Literal::Int(n) => self.emit(Op::PushInt(*n)),
                ast::Literal::Float(f) => self.emit(Op::PushFloat(*f)),
                ast::Literal::Bool(b) => self.emit(Op::PushBool(*b)),
                ast::Literal::String(s) => self.emit(Op::PushString(s.clone())),
            },

            ast::Expr::Identifier(name) => {
                match self.resolve_local(name) {
                    Some(slot) => self.emit(Op::LoadLocal(slot)),
                    None => {
                        // Could be a zero-arg union constructor or builtin
                        if name == "true" {
                            self.emit(Op::PushBool(true));
                        } else if name == "false" {
                            self.emit(Op::PushBool(false));
                        } else if name == "None" {
                            self.emit(Op::PushNone);
                        } else {
                            // Treat as zero-arg variant constructor
                            self.emit(Op::PushVariant(name.clone(), 0));
                        }
                    }
                }
            }

            ast::Expr::FieldAccess { base, field } => {
                self.compile_expr(base)?;
                self.emit(Op::GetField(field.clone()));
            }

            ast::Expr::Call(call) => {
                self.compile_call_expr(call)?;
            }

            ast::Expr::Binary { left, op, right } => {
                self.compile_expr(left)?;
                self.compile_expr(right)?;
                let vm_op = match op {
                    ast::BinaryOp::Add => Op::Add,
                    ast::BinaryOp::Sub => Op::Sub,
                    ast::BinaryOp::Mul => Op::Mul,
                    ast::BinaryOp::Div => Op::Div,
                    ast::BinaryOp::Mod => Op::Mod,
                    ast::BinaryOp::Equal => Op::Eq,
                    ast::BinaryOp::NotEqual => Op::Neq,
                    ast::BinaryOp::LessThan => Op::Lt,
                    ast::BinaryOp::LessThanOrEqual => Op::Lte,
                    ast::BinaryOp::GreaterThan => Op::Gt,
                    ast::BinaryOp::GreaterThanOrEqual => Op::Gte,
                    ast::BinaryOp::And => Op::And,
                    ast::BinaryOp::Or => Op::Or,
                };
                self.emit(vm_op);
            }

            ast::Expr::Unary { op, expr } => {
                self.compile_expr(expr)?;
                match op {
                    ast::UnaryOp::Not => self.emit(Op::Not),
                    ast::UnaryOp::Neg => self.emit(Op::Negate),
                }
            }

            ast::Expr::StructInit { ty, fields } => {
                let field_names: Vec<String> = fields.iter().map(|f| f.name.clone()).collect();
                for field in fields {
                    self.compile_expr(&field.value)?;
                }
                if ty.is_empty() {
                    self.emit(Op::MakeStruct(field_names));
                } else {
                    self.emit(Op::MakeStruct(field_names));
                }
            }
        }
        Ok(())
    }

    /// Compile a function/builtin call expression.
    fn compile_call_expr(&mut self, call: &ast::CallExpr) -> Result<()> {
        let callee = &call.callee;
        let argc = call.args.len();

        // Special-case Result/Option constructors
        match callee.as_str() {
            "Ok" => {
                if argc == 1 {
                    self.compile_expr(&call.args[0])?;
                    self.emit(Op::PushOk);
                } else {
                    self.emit(Op::PushUnit);
                    self.emit(Op::PushOk);
                }
                return Ok(());
            }
            "Err" => {
                if argc == 1 {
                    self.compile_expr(&call.args[0])?;
                } else {
                    self.emit(Op::PushString("unknown".to_string()));
                }
                self.emit(Op::PushErr);
                return Ok(());
            }
            "Some" => {
                if argc == 1 {
                    self.compile_expr(&call.args[0])?;
                    self.emit(Op::PushSome);
                } else {
                    self.emit(Op::PushUnit);
                    self.emit(Op::PushSome);
                }
                return Ok(());
            }
            _ => {}
        }

        // Compile arguments
        for arg in &call.args {
            self.compile_expr(arg)?;
        }

        // Check if it's a user-defined function or a builtin
        if self.program.get_function(callee).is_some() {
            self.emit(Op::Call(callee.clone(), argc));
        } else {
            // Treat as builtin (includes builtins + union constructors)
            self.emit(Op::CallBuiltin(callee.clone(), argc));
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// VM Interpreter: execute compiled bytecode
// ═══════════════════════════════════════════════════════════════════════

/// Result of VM execution — either a completed value or an async suspension.
#[derive(Debug, Clone)]
pub enum VmResult {
    /// Execution completed with a return value.
    Done(Value),
    /// Execution suspended for an async IO operation.
    /// The caller should perform the IO, then resume by providing the result.
    Await {
        op_name: String,
        args: Vec<Value>,
        /// The saved VM state, ready to resume after IO completes.
        vm_state: VmState,
    },
}

/// Execute a compiled function with the given arguments.
///
/// For synchronous functions, returns `VmResult::Done(value)`.
/// For functions that hit an `AwaitIo` instruction, returns `VmResult::Await`
/// with the suspended state so the caller can perform IO and resume.
/// Compilation cache: avoids recompiling the same function on every call.
pub type CompileCache = std::collections::HashMap<String, CompiledFunction>;

pub fn execute(
    compiled: &CompiledFunction,
    args: Vec<Value>,
    program: &ast::Program,
) -> Result<VmResult> {
    let mut cache = CompileCache::new();
    cache.insert(compiled.name.clone(), compiled.clone());
    execute_cached(compiled, args, program, &mut cache)
}

fn execute_cached(
    compiled: &CompiledFunction,
    args: Vec<Value>,
    program: &ast::Program,
    cache: &mut CompileCache,
) -> Result<VmResult> {
    let mut state = VmState {
        function_name: compiled.name.clone(),
        ip: 0,
        bytecode: std::sync::Arc::clone(&compiled.bytecode),
        locals: vec![Value::None; compiled.local_count.max(args.len())],
        stack: Vec::with_capacity(32),
        call_stack: Vec::new(),
    };

    // Load arguments into local slots
    for (i, arg) in args.into_iter().enumerate() {
        if i < state.locals.len() {
            state.locals[i] = arg;
        }
    }

    run_loop(state, program, cache)
}

/// Resume a suspended VM after an async IO operation completes.
/// Pushes the IO result onto the stack and continues execution.
pub fn resume(mut vm_state: VmState, io_result: Value, program: &ast::Program) -> Result<VmResult> {
    vm_state.stack.push(io_result);
    let mut cache = CompileCache::new();
    run_loop(vm_state, program, &mut cache)
}

/// Execute a compiled function with async IO support.
/// When the VM suspends for IO, calls `io_handler` to perform the operation,
/// then resumes. Loops until execution completes.
pub fn execute_with_io(
    compiled: &CompiledFunction,
    args: Vec<Value>,
    program: &ast::Program,
    io_handler: &dyn Fn(&str, &[Value]) -> Result<Value>,
) -> Result<Value> {
    let mut cache = CompileCache::new();
    cache.insert(compiled.name.clone(), compiled.clone());
    let mut result = execute_cached(compiled, args, program, &mut cache)?;
    loop {
        match result {
            VmResult::Done(val) => return Ok(val),
            VmResult::Await {
                op_name,
                args,
                vm_state,
            } => {
                let io_result = io_handler(&op_name, &args)?;
                let mut vm_state = vm_state;
                vm_state.stack.push(io_result);
                result = run_loop(vm_state, program, &mut cache)?;
            }
        }
    }
}

/// The main execution loop. Processes instructions until Return or AwaitIo.
fn run_loop(
    mut state: VmState,
    program: &ast::Program,
    cache: &mut CompileCache,
) -> Result<VmResult> {
    loop {
        if state.ip >= state.bytecode.len() {
            // Fell off the end — return unit
            return Ok(VmResult::Done(Value::None));
        }

        let op = state.bytecode[state.ip].clone();
        state.ip += 1;

        match op {
            // ── Literals ─────────────────────────────────────────────
            Op::PushInt(n) => state.stack.push(Value::Int(n)),
            Op::PushFloat(f) => state.stack.push(Value::Float(f)),
            Op::PushBool(b) => state.stack.push(Value::Bool(b)),
            Op::PushString(s) => state.stack.push(Value::String(s)),
            Op::PushUnit => state.stack.push(Value::None),

            // ── Locals ───────────────────────────────────────────────
            Op::LoadLocal(slot) => {
                let base = state.call_stack.last().map(|f| f.locals_base).unwrap_or(0);
                let idx = base + slot;
                let val = state.locals.get(idx).cloned().unwrap_or(Value::None);
                state.stack.push(val);
            }
            Op::StoreLocal(slot) => {
                let base = state.call_stack.last().map(|f| f.locals_base).unwrap_or(0);
                let idx = base + slot;
                let val = pop_stack(&mut state.stack)?;
                // Grow locals if needed
                while state.locals.len() <= idx {
                    state.locals.push(Value::None);
                }
                state.locals[idx] = val;
            }

            // ── Arithmetic ───────────────────────────────────────────
            Op::Add => binary_arith(&mut state.stack, |a, b| a + b, |a, b| a + b)?,
            Op::Sub => binary_arith(&mut state.stack, |a, b| a - b, |a, b| a - b)?,
            Op::Mul => binary_arith(&mut state.stack, |a, b| a * b, |a, b| a * b)?,
            Op::Div => {
                let rhs = pop_stack(&mut state.stack)?;
                let lhs = pop_stack(&mut state.stack)?;
                match (&lhs, &rhs) {
                    (Value::Int(a), Value::Int(b)) => {
                        if *b == 0 {
                            bail!("division by zero");
                        }
                        state.stack.push(Value::Int(a / b));
                    }
                    (Value::Float(a), Value::Float(b)) => {
                        state.stack.push(Value::Float(a / b));
                    }
                    (Value::Int(a), Value::Float(b)) => {
                        state.stack.push(Value::Float(*a as f64 / b));
                    }
                    (Value::Float(a), Value::Int(b)) => {
                        if *b == 0 {
                            bail!("division by zero");
                        }
                        state.stack.push(Value::Float(a / *b as f64));
                    }
                    _ => bail!("Div: unsupported types {lhs} / {rhs}"),
                }
            }
            Op::Mod => {
                let rhs = pop_stack(&mut state.stack)?;
                let lhs = pop_stack(&mut state.stack)?;
                match (&lhs, &rhs) {
                    (Value::Int(a), Value::Int(b)) => {
                        if *b == 0 {
                            bail!("modulo by zero");
                        }
                        state.stack.push(Value::Int(a % b));
                    }
                    _ => bail!("Mod: unsupported types"),
                }
            }

            // ── Comparison ───────────────────────────────────────────
            Op::Eq => binary_cmp(
                &mut state.stack,
                |a, b| a == b,
                |a, b| (a - b).abs() < f64::EPSILON,
            )?,
            Op::Neq => binary_cmp(
                &mut state.stack,
                |a, b| a != b,
                |a, b| (a - b).abs() >= f64::EPSILON,
            )?,
            Op::Lt => binary_cmp(&mut state.stack, |a, b| a < b, |a, b| a < b)?,
            Op::Lte => binary_cmp(&mut state.stack, |a, b| a <= b, |a, b| a <= b)?,
            Op::Gt => binary_cmp(&mut state.stack, |a, b| a > b, |a, b| a > b)?,
            Op::Gte => binary_cmp(&mut state.stack, |a, b| a >= b, |a, b| a >= b)?,

            // ── Boolean logic ────────────────────────────────────────
            Op::And => {
                let b = pop_bool(&mut state.stack)?;
                let a = pop_bool(&mut state.stack)?;
                state.stack.push(Value::Bool(a && b));
            }
            Op::Or => {
                let b = pop_bool(&mut state.stack)?;
                let a = pop_bool(&mut state.stack)?;
                state.stack.push(Value::Bool(a || b));
            }
            Op::Not => {
                let v = pop_bool(&mut state.stack)?;
                state.stack.push(Value::Bool(!v));
            }
            Op::Negate => {
                let v = pop_stack(&mut state.stack)?;
                match v {
                    Value::Int(n) => state.stack.push(Value::Int(-n)),
                    Value::Float(f) => state.stack.push(Value::Float(-f)),
                    _ => bail!("Negate: unsupported type {v}"),
                }
            }

            // ── Function calls ───────────────────────────────────────
            Op::Call(name, argc) => {
                let mut call_args = Vec::with_capacity(argc);
                for _ in 0..argc {
                    call_args.push(pop_stack(&mut state.stack)?);
                }
                call_args.reverse(); // arguments were pushed left-to-right

                // Compile callee (cached — only compiles once per function)
                if !cache.contains_key(&name) {
                    let func = program
                        .get_function(&name)
                        .ok_or_else(|| anyhow::anyhow!("vm: function `{name}` not found"))?;
                    let compiled = compile_function(func, program)?;
                    cache.insert(name.clone(), compiled);
                }
                let callee = cache.get(&name).unwrap();

                // Save current frame
                let locals_base = state.locals.len();
                let callee_bytecode = std::sync::Arc::clone(&callee.bytecode);
                state.call_stack.push(Frame {
                    return_ip: state.ip,
                    return_bytecode: std::mem::replace(&mut state.bytecode, callee_bytecode),
                    locals_base,
                });

                // Allocate locals for callee and load arguments
                state.locals.resize(
                    locals_base + callee.local_count.max(call_args.len()),
                    Value::None,
                );
                for (i, arg) in call_args.into_iter().enumerate() {
                    state.locals[locals_base + i] = arg;
                }
                state.ip = 0;
            }
            Op::CallBuiltin(name, argc) => {
                let mut call_args = Vec::with_capacity(argc);
                for _ in 0..argc {
                    call_args.push(pop_stack(&mut state.stack)?);
                }
                call_args.reverse();

                let mut env = crate::eval::Env::new();
                let result =
                    crate::eval::eval_builtin_or_user(program, &name, call_args, &mut env)?;
                state.stack.push(result);
            }
            Op::AwaitIo(op_name, argc) => {
                let mut io_args = Vec::with_capacity(argc);
                for _ in 0..argc {
                    io_args.push(pop_stack(&mut state.stack)?);
                }
                io_args.reverse();

                // Suspend execution — caller must perform IO and resume
                return Ok(VmResult::Await {
                    op_name,
                    args: io_args,
                    vm_state: state.clone(),
                });
            }

            // ── Control flow ─────────────────────────────────────────
            Op::Return => {
                let val = pop_stack(&mut state.stack)?;
                if state.call_stack.is_empty() {
                    return Ok(VmResult::Done(val));
                }
                // Return to caller frame
                let frame = state.call_stack.pop().unwrap();
                state.ip = frame.return_ip;
                state.bytecode = frame.return_bytecode;
                state.locals.truncate(frame.locals_base);
                state.stack.push(val);
            }
            Op::Jump(target) => {
                state.ip = target;
            }
            Op::BranchIf(target) => {
                let cond = pop_bool(&mut state.stack)?;
                if cond {
                    state.ip = target;
                }
            }
            Op::BranchIfNot(target) => {
                let cond = pop_bool(&mut state.stack)?;
                if !cond {
                    state.ip = target;
                }
            }

            // ── Stack manipulation ───────────────────────────────────
            Op::Pop => {
                pop_stack(&mut state.stack)?;
            }
            Op::Dup => {
                let val = state
                    .stack
                    .last()
                    .ok_or_else(|| anyhow::anyhow!("Dup: empty stack"))?
                    .clone();
                state.stack.push(val);
            }

            // ── Struct / composite ───────────────────────────────────
            Op::MakeStruct(ref field_names) => {
                let mut fields = HashMap::new();
                for name in field_names.iter().rev() {
                    let val = pop_stack(&mut state.stack)?;
                    fields.insert(name.clone(), val);
                }
                state.stack.push(Value::Struct(String::new(), fields));
            }
            Op::GetField(ref field) => {
                let val = pop_stack(&mut state.stack)?;
                match val {
                    Value::Struct(_, ref map) => {
                        let field_val = map
                            .get(field)
                            .ok_or_else(|| anyhow::anyhow!("field `{field}` not found on struct"))?
                            .clone();
                        state.stack.push(field_val);
                    }
                    _ => bail!("GetField: expected struct, got {val}"),
                }
            }
            Op::MakeList(count) => {
                let mut items = Vec::with_capacity(count);
                for _ in 0..count {
                    items.push(pop_stack(&mut state.stack)?);
                }
                items.reverse();
                state.stack.push(Value::List(items));
            }

            // ── Result / Option constructors ─────────────────────────
            Op::PushOk => {
                let val = pop_stack(&mut state.stack)?;
                state.stack.push(Value::Ok(Box::new(val)));
            }
            Op::PushErr => {
                let val = pop_stack(&mut state.stack)?;
                let msg = match val {
                    Value::String(s) => s,
                    other => format!("{other}"),
                };
                state.stack.push(Value::Err(msg));
            }
            Op::PushSome => {
                let val = pop_stack(&mut state.stack)?;
                state.stack.push(Value::Ok(Box::new(val)));
            }
            Op::PushNone => {
                state.stack.push(Value::None);
            }

            // ── Union / variant ops ──────────────────────────────────
            Op::PushVariant(ref variant_name, count) => {
                let mut payload = Vec::with_capacity(count);
                for _ in 0..count {
                    payload.push(pop_stack(&mut state.stack)?);
                }
                payload.reverse();
                state.stack.push(Value::Union {
                    variant: variant_name.clone(),
                    payload,
                });
            }
            Op::MatchVariant(ref variant_name, binding_count) => {
                let val = pop_stack(&mut state.stack)?;
                match val {
                    Value::Union {
                        ref variant,
                        ref payload,
                    } if variant == variant_name => {
                        // Push bindings onto stack (in order), then push true
                        for i in 0..binding_count.min(payload.len()) {
                            state.stack.push(payload[i].clone());
                        }
                        state.stack.push(Value::Bool(true));
                    }
                    _ => {
                        // No match — push false (no bindings)
                        state.stack.push(Value::Bool(false));
                    }
                }
            }
        }
    }
}

// ── Stack helpers ────────────────────────────────────────────────────

fn pop_stack(stack: &mut Vec<Value>) -> Result<Value> {
    stack
        .pop()
        .ok_or_else(|| anyhow::anyhow!("vm: stack underflow"))
}

fn pop_bool(stack: &mut Vec<Value>) -> Result<bool> {
    match pop_stack(stack)? {
        Value::Bool(b) => Ok(b),
        Value::Int(n) => Ok(n != 0), // truthy: non-zero
        other => bail!("vm: expected Bool, got {other}"),
    }
}

/// Binary arithmetic on Int or Float values.
fn binary_arith(
    stack: &mut Vec<Value>,
    int_op: impl Fn(i64, i64) -> i64,
    float_op: impl Fn(f64, f64) -> f64,
) -> Result<()> {
    let rhs = pop_stack(stack)?;
    let lhs = pop_stack(stack)?;
    let result = match (&lhs, &rhs) {
        (Value::Int(a), Value::Int(b)) => Value::Int(int_op(*a, *b)),
        (Value::Float(a), Value::Float(b)) => Value::Float(float_op(*a, *b)),
        (Value::Int(a), Value::Float(b)) => Value::Float(float_op(*a as f64, *b)),
        (Value::Float(a), Value::Int(b)) => Value::Float(float_op(*a, *b as f64)),
        // String concatenation for Add
        (Value::String(a), Value::String(b)) => Value::String(format!("{a}{b}")),
        (Value::String(a), other) => Value::String(format!("{a}{other}")),
        (other, Value::String(b)) => Value::String(format!("{other}{b}")),
        _ => bail!("arithmetic: unsupported types {lhs} and {rhs}"),
    };
    stack.push(result);
    Ok(())
}

/// Binary comparison on Int, Float, or String values.
fn binary_cmp(
    stack: &mut Vec<Value>,
    int_cmp: impl Fn(i64, i64) -> bool,
    float_cmp: impl Fn(f64, f64) -> bool,
) -> Result<()> {
    let rhs = pop_stack(stack)?;
    let lhs = pop_stack(stack)?;
    let result = match (&lhs, &rhs) {
        (Value::Int(a), Value::Int(b)) => int_cmp(*a, *b),
        (Value::Float(a), Value::Float(b)) => float_cmp(*a, *b),
        (Value::Int(a), Value::Float(b)) => float_cmp(*a as f64, *b),
        (Value::Float(a), Value::Int(b)) => float_cmp(*a, *b as f64),
        (Value::Bool(a), Value::Bool(b)) => int_cmp(*a as i64, *b as i64),
        (Value::String(a), Value::String(b)) => int_cmp(a.cmp(b) as i64, 0),
        _ => bail!("comparison: unsupported types {lhs} and {rhs}"),
    };
    stack.push(Value::Bool(result));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parser, validator};
    use std::sync::Arc;

    /// Parse and validate Adapsis source into a Program.
    fn build_program(source: &str) -> ast::Program {
        let ops = parser::parse(source).expect("parse failed");
        let mut program = ast::Program::default();
        for op in &ops {
            match op {
                parser::Operation::Test(_) | parser::Operation::Eval(_) => {}
                _ => {
                    validator::apply_and_validate(&mut program, op).expect("validation failed");
                }
            }
        }
        program.rebuild_function_index();
        program
    }

    /// Compile a function from source, return the CompiledFunction.
    fn compile_from_source(source: &str, fn_name: &str) -> CompiledFunction {
        let program = build_program(source);
        let func = program
            .get_function(fn_name)
            .unwrap_or_else(|| panic!("function `{fn_name}` not found"));
        compile_function(func, &program).expect("compilation failed")
    }

    #[test]
    fn compile_simple_let_return() {
        let compiled = compile_from_source(
            "\
+fn double (x:Int)->Int
  +let result:Int = x * 2
  +return result
",
            "double",
        );

        assert_eq!(compiled.name, "double");
        assert_eq!(compiled.param_names, vec!["x"]);
        assert!(compiled.local_count >= 2); // x + result
        assert!(compiled.local_names.contains(&"x".to_string()));
        assert!(compiled.local_names.contains(&"result".to_string()));

        // Should have: LoadLocal(x), PushInt(2), Mul, StoreLocal(result), LoadLocal(result), Return
        assert!(compiled.bytecode.iter().any(|op| matches!(op, Op::Mul)));
        assert!(compiled.bytecode.iter().any(|op| matches!(op, Op::Return)));
        assert!(compiled
            .bytecode
            .iter()
            .any(|op| matches!(op, Op::PushInt(2))));
    }

    #[test]
    fn compile_branch_if_else() {
        let compiled = compile_from_source(
            "\
+fn abs_val (x:Int)->Int
  +if x >= 0
    +return x
  +else
    +let neg:Int = 0 - x
    +return neg
  +end
",
            "abs_val",
        );

        assert_eq!(compiled.name, "abs_val");
        // Should have Gte, BranchIfNot, Return, Jump, Return
        let has_gte = compiled.bytecode.iter().any(|op| matches!(op, Op::Gte));
        let has_branch = compiled
            .bytecode
            .iter()
            .any(|op| matches!(op, Op::BranchIfNot(_)));
        let return_count = compiled
            .bytecode
            .iter()
            .filter(|op| matches!(op, Op::Return))
            .count();
        assert!(has_gte, "should have Gte for x >= 0");
        assert!(has_branch, "should have BranchIfNot for if condition");
        assert!(
            return_count >= 2,
            "should have at least 2 returns (then + else)"
        );
    }

    #[test]
    fn compile_check_statement() {
        let compiled = compile_from_source(
            "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x
",
            "validate",
        );

        assert_eq!(compiled.name, "validate");
        // Check compiles to: condition, BranchIfNot(fail), Jump(past_fail), PushString, PushErr, Return
        let has_gt = compiled.bytecode.iter().any(|op| matches!(op, Op::Gt));
        let has_push_err = compiled.bytecode.iter().any(|op| matches!(op, Op::PushErr));
        let has_err_label = compiled
            .bytecode
            .iter()
            .any(|op| matches!(op, Op::PushString(s) if s == "err_negative"));
        assert!(has_gt, "should have Gt for x > 0");
        assert!(has_push_err, "should have PushErr for fail path");
        assert!(has_err_label, "should have err_negative label");
    }

    #[test]
    fn compile_while_loop() {
        let compiled = compile_from_source(
            "\
+fn count_to (n:Int)->Int
  +let i:Int = 0
  +while i < n
    +set i = i + 1
  +end
  +return i
",
            "count_to",
        );

        assert_eq!(compiled.name, "count_to");
        // Should have: PushInt(0), StoreLocal, Lt, BranchIfNot, Add, PushInt(1), Jump, Return
        let has_lt = compiled.bytecode.iter().any(|op| matches!(op, Op::Lt));
        let has_jump = compiled.bytecode.iter().any(|op| matches!(op, Op::Jump(_)));
        let has_branch = compiled
            .bytecode
            .iter()
            .any(|op| matches!(op, Op::BranchIfNot(_)));
        assert!(has_lt, "should have Lt for i < n");
        assert!(has_jump, "should have Jump back to loop start");
        assert!(has_branch, "should have BranchIfNot for loop exit");

        // Verify Jump target is before BranchIfNot (loop back)
        let jump_pos = compiled
            .bytecode
            .iter()
            .position(|op| matches!(op, Op::Jump(_)))
            .unwrap();
        if let Op::Jump(target) = &compiled.bytecode[jump_pos] {
            assert!(*target < jump_pos, "Jump should go backward to loop start");
        }
    }

    #[test]
    fn compile_builtin_call() {
        let compiled = compile_from_source(
            r#"
+fn greet (name:String)->String
  +let msg:String = concat("hello ", name)
  +return msg
"#,
            "greet",
        );

        assert_eq!(compiled.name, "greet");
        // concat is a builtin — should emit CallBuiltin
        let has_call_builtin = compiled
            .bytecode
            .iter()
            .any(|op| matches!(op, Op::CallBuiltin(name, 2) if name == "concat"));
        assert!(has_call_builtin, "should emit CallBuiltin(\"concat\", 2)");
    }

    #[test]
    fn compile_user_function_call() {
        let compiled = compile_from_source(
            "\
+fn double (x:Int)->Int
  +return x * 2

+fn quadruple (x:Int)->Int
  +call d:Int = double(x)
  +call r:Int = double(d)
  +return r
",
            "quadruple",
        );

        assert_eq!(compiled.name, "quadruple");
        // double is a user function — should emit Call, not CallBuiltin
        let call_count = compiled
            .bytecode
            .iter()
            .filter(|op| matches!(op, Op::Call(name, 1) if name == "double"))
            .count();
        assert_eq!(
            call_count, 2,
            "should have 2 Call(\"double\", 1) instructions"
        );
    }

    #[test]
    fn compile_struct_init_and_field_access() {
        let compiled = compile_from_source(
            "\
+type Point = x:Int, y:Int

+fn get_x (p:Point)->Int
  +return p.x
",
            "get_x",
        );

        let has_get_field = compiled
            .bytecode
            .iter()
            .any(|op| matches!(op, Op::GetField(f) if f == "x"));
        assert!(has_get_field, "should emit GetField(\"x\")");
    }

    #[test]
    fn compile_match_on_union() {
        let compiled = compile_from_source(
            "\
+type Color = Red | Green | Blue

+fn is_red (c:Color)->Bool
  +match c
  +case Red
    +return true
  +case _
    +return false
  +end
",
            "is_red",
        );

        assert_eq!(compiled.name, "is_red");
        let has_match = compiled
            .bytecode
            .iter()
            .any(|op| matches!(op, Op::MatchVariant(v, _) if v == "Red"));
        assert!(has_match, "should emit MatchVariant(\"Red\", 0)");
    }

    #[test]
    fn compile_each_loop() {
        let compiled = compile_from_source(
            "\
+fn sum_list (items:List<Int>)->Int
  +let total:Int = 0
  +each items item:Int
    +set total = total + item
  +end
  +return total
",
            "sum_list",
        );

        assert_eq!(compiled.name, "sum_list");
        // Each compiles to: store list, init idx, call len, loop with get + body
        let has_len_call = compiled
            .bytecode
            .iter()
            .any(|op| matches!(op, Op::CallBuiltin(name, 1) if name == "len"));
        let has_get_call = compiled
            .bytecode
            .iter()
            .any(|op| matches!(op, Op::CallBuiltin(name, 2) if name == "get"));
        assert!(has_len_call, "should call builtin len for loop bound");
        assert!(has_get_call, "should call builtin get for element access");
    }

    #[test]
    fn compile_await_io() {
        let compiled = compile_from_source(
            "\
+fn fetch (url:String)->String [io,async]
  +await data:String = http_get(url)
  +return data
",
            "fetch",
        );

        let has_await = compiled
            .bytecode
            .iter()
            .any(|op| matches!(op, Op::AwaitIo(name, 1) if name == "http_get"));
        assert!(has_await, "should emit AwaitIo(\"http_get\", 1)");
    }

    #[test]
    fn compile_empty_function_gets_default_return() {
        let compiled = compile_from_source(
            "\
+fn noop ()->Int
+end
",
            "noop",
        );

        // Empty body should get PushUnit + Return
        assert!(matches!(compiled.bytecode.last(), Some(Op::Return)));
        assert!(compiled.bytecode.len() >= 2);
    }

    #[test]
    fn compile_ok_err_constructors() {
        let compiled = compile_from_source(
            "\
+fn wrap (x:Int)->Result<Int> [fail]
  +check pos x > 0 ~err_neg
  +return x
",
            "wrap",
        );

        // The check's fail path should have PushErr
        let has_push_err = compiled.bytecode.iter().any(|op| matches!(op, Op::PushErr));
        assert!(has_push_err);
    }

    #[test]
    fn compile_params_allocated_as_first_slots() {
        let compiled = compile_from_source(
            "\
+fn add (a:Int, b:Int)->Int
  +return a + b
",
            "add",
        );

        assert_eq!(compiled.param_names, vec!["a", "b"]);
        assert_eq!(compiled.local_names[0], "a");
        assert_eq!(compiled.local_names[1], "b");
        // LoadLocal(0) for a, LoadLocal(1) for b
        assert!(compiled
            .bytecode
            .iter()
            .any(|op| matches!(op, Op::LoadLocal(0))));
        assert!(compiled
            .bytecode
            .iter()
            .any(|op| matches!(op, Op::LoadLocal(1))));
    }

    // ═════════════════════════════════════════════════════════════════════
    // VM Execution tests
    // ═════════════════════════════════════════════════════════════════════

    /// Compile and execute a function, returning the result Value.
    fn exec_fn(source: &str, fn_name: &str, args: Vec<Value>) -> Value {
        let program = build_program(source);
        let func = program
            .get_function(fn_name)
            .unwrap_or_else(|| panic!("function `{fn_name}` not found"));
        let compiled = compile_function(func, &program).expect("compilation failed");
        match execute(&compiled, args, &program).expect("execution failed") {
            VmResult::Done(val) => val,
            VmResult::Await { .. } => panic!("unexpected Await result"),
        }
    }

    #[test]
    fn exec_double() {
        let val = exec_fn(
            "\
+fn double (x:Int)->Int
  +let result:Int = x * 2
  +return result
",
            "double",
            vec![Value::Int(5)],
        );
        assert!(matches!(val, Value::Int(10)), "expected 10, got {val}");
    }

    #[test]
    fn exec_add() {
        let val = exec_fn(
            "\
+fn add (a:Int, b:Int)->Int
  +return a + b
",
            "add",
            vec![Value::Int(3), Value::Int(4)],
        );
        assert!(matches!(val, Value::Int(7)), "expected 7, got {val}");
    }

    #[test]
    fn exec_arithmetic_compound() {
        let val = exec_fn(
            "\
+fn calc (a:Int, b:Int)->Int
  +let sum:Int = a + b
  +let product:Int = a * b
  +let result:Int = sum + product
  +return result
",
            "calc",
            vec![Value::Int(3), Value::Int(4)],
        );
        // sum=7, product=12, result=19
        assert!(matches!(val, Value::Int(19)), "expected 19, got {val}");
    }

    #[test]
    fn exec_if_else_positive() {
        let source = "\
+fn classify (x:Int)->Int
  +if x >= 0
    +return 1
  +else
    +return 0
  +end
";
        let val = exec_fn(source, "classify", vec![Value::Int(5)]);
        assert!(
            matches!(val, Value::Int(1)),
            "expected 1 for positive, got {val}"
        );

        let val = exec_fn(source, "classify", vec![Value::Int(-3)]);
        assert!(
            matches!(val, Value::Int(0)),
            "expected 0 for negative, got {val}"
        );
    }

    #[test]
    fn exec_while_loop() {
        let val = exec_fn(
            "\
+fn count_to (n:Int)->Int
  +let i:Int = 0
  +while i < n
    +set i = i + 1
  +end
  +return i
",
            "count_to",
            vec![Value::Int(5)],
        );
        assert!(matches!(val, Value::Int(5)), "expected 5, got {val}");
    }

    #[test]
    fn exec_while_loop_sum() {
        let val = exec_fn(
            "\
+fn sum_to (n:Int)->Int
  +let total:Int = 0
  +let i:Int = 1
  +while i <= n
    +set total = total + i
    +set i = i + 1
  +end
  +return total
",
            "sum_to",
            vec![Value::Int(10)],
        );
        assert!(matches!(val, Value::Int(55)), "expected 55, got {val}");
    }

    #[test]
    fn exec_check_pass() {
        let val = exec_fn(
            "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x
",
            "validate",
            vec![Value::Int(42)],
        );
        // Check passed — returns the raw value (not wrapped in Ok by VM)
        assert!(matches!(val, Value::Int(42)), "expected 42, got {val}");
    }

    #[test]
    fn exec_check_fail() {
        let val = exec_fn(
            "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x
",
            "validate",
            vec![Value::Int(-1)],
        );
        assert!(
            matches!(val, Value::Err(ref msg) if msg == "err_negative"),
            "expected Err(err_negative), got {val}"
        );
    }

    #[test]
    fn exec_user_function_call() {
        let val = exec_fn(
            "\
+fn double (x:Int)->Int
  +return x * 2

+fn quadruple (x:Int)->Int
  +call d:Int = double(x)
  +call r:Int = double(d)
  +return r
",
            "quadruple",
            vec![Value::Int(5)],
        );
        assert!(matches!(val, Value::Int(20)), "expected 20, got {val}");
    }

    #[test]
    fn exec_builtin_concat() {
        let val = exec_fn(
            r#"
+fn greet (name:String)->String
  +let msg:String = concat("hello ", name)
  +return msg
"#,
            "greet",
            vec![Value::String("world".to_string())],
        );
        match val {
            Value::String(s) => assert_eq!(s, "hello world"),
            _ => panic!("expected String, got {val}"),
        }
    }

    #[test]
    fn exec_builtin_len() {
        let val = exec_fn(
            r#"
+fn str_len (s:String)->Int
  +let n:Int = len(s)
  +return n
"#,
            "str_len",
            vec![Value::String("hello".to_string())],
        );
        assert!(matches!(val, Value::Int(5)), "expected 5, got {val}");
    }

    #[test]
    fn exec_comparison_ops() {
        let source = "\
+fn check_gt (a:Int, b:Int)->Int
  +if a > b
    +return 1
  +else
    +return 0
  +end
";
        assert!(matches!(
            exec_fn(source, "check_gt", vec![Value::Int(5), Value::Int(3)]),
            Value::Int(1)
        ));
        assert!(matches!(
            exec_fn(source, "check_gt", vec![Value::Int(1), Value::Int(3)]),
            Value::Int(0)
        ));
    }

    #[test]
    fn exec_boolean_logic() {
        let val = exec_fn(
            "\
+fn both (a:Bool, b:Bool)->Bool
  +if a AND b
    +return true
  +else
    +return false
  +end
",
            "both",
            vec![Value::Bool(true), Value::Bool(true)],
        );
        assert!(matches!(val, Value::Bool(true)));

        let val = exec_fn(
            "\
+fn both (a:Bool, b:Bool)->Bool
  +if a AND b
    +return true
  +else
    +return false
  +end
",
            "both",
            vec![Value::Bool(true), Value::Bool(false)],
        );
        assert!(matches!(val, Value::Bool(false)));
    }

    #[test]
    fn exec_constant_return() {
        let val = exec_fn(
            "\
+fn answer ()->Int
  +return 42
",
            "answer",
            vec![],
        );
        assert!(matches!(val, Value::Int(42)));
    }

    #[test]
    fn exec_float_arithmetic() {
        let val = exec_fn(
            "\
+fn add_f (a:Float, b:Float)->Float
  +return a + b
",
            "add_f",
            vec![Value::Float(1.5), Value::Float(2.5)],
        );
        match val {
            Value::Float(f) => assert!((f - 4.0).abs() < 1e-10),
            _ => panic!("expected Float, got {val}"),
        }
    }

    #[test]
    fn exec_await_io_suspends() {
        let program = build_program(
            "\
+fn fetch (url:String)->String [io,async]
  +await data:String = http_get(url)
  +return data
",
        );
        let func = program.get_function("fetch").unwrap();
        let compiled = compile_function(func, &program).unwrap();
        let result = execute(
            &compiled,
            vec![Value::String("http://example.com".to_string())],
            &program,
        )
        .unwrap();
        match result {
            VmResult::Await { op_name, args, .. } => {
                assert_eq!(op_name, "http_get");
                assert_eq!(args.len(), 1);
            }
            VmResult::Done(_) => panic!("expected Await, got Done"),
        }
    }
}
