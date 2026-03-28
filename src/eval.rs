use anyhow::{anyhow, bail, Result};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::ast;
use crate::compiler::CompiledProgram;
use crate::intern::{self, InternedId, SharedInterner, StringInterner};
use crate::parser;
use crate::vm;

/// Cache for JIT-compiled programs, keyed by session revision.
/// When the revision matches, the compiled module is reused instead of recompiling.
pub type JitCache = Arc<Mutex<Option<(usize, CompiledProgram)>>>;

/// Create a new empty JIT cache.
pub fn new_jit_cache() -> JitCache {
    Arc::new(Mutex::new(None))
}

/// A runtime value during evaluation.
///
/// **Performance notes**: Struct field keys, struct type names, and union variant
/// names use `InternedId` (u32) instead of `String` for fast hash + comparison.
/// `List` uses `Arc<Vec<Value>>` and `Struct` uses `Arc<HashMap<InternedId, Value>>`
/// — compound values are read far more often than they are mutated.
///
/// `Arc` (rather than `Rc`) is required because `Value`s are sent across thread
/// boundaries via `tokio::task::spawn_blocking` in the async evaluation paths.
///
/// To resolve interned IDs back to strings (e.g. for Display), use the
/// thread-local display interner installed by `intern::set_display_interner()`.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum Value {
    CoroutineHandle(crate::coroutine::CoroutineHandle),
    TaskHandle(crate::coroutine::TaskId),
    Union {
        variant: InternedId,
        payload: Vec<Value>,
    },
    Int(i64),
    Float(f64),
    Bool(bool),
    String(Arc<String>),
    Struct(InternedId, Arc<HashMap<InternedId, Value>>),
    List(Arc<Vec<Value>>),
    Ok(Box<Value>),
    Err(String),
    None,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(v) => write!(f, "{v}"),
            Value::Float(v) => write!(f, "{v}"),
            Value::Bool(v) => write!(f, "{v}"),
            Value::String(v) => write!(f, "\"{v}\""),
            Value::Struct(name_id, fields) => {
                let name = intern::resolve_display(*name_id);
                write!(f, "{name}{{")?;
                for (i, (k_id, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    let k = intern::resolve_display(*k_id);
                    write!(f, "{k}: {v}")?;
                }
                write!(f, "}}")
            }
            Value::List(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{item}")?;
                }
                write!(f, "]")
            }
            Value::Ok(v) => write!(f, "Ok({v})"),
            Value::Err(msg) => write!(f, "Err({msg})"),
            Value::None => write!(f, "None"),
            Value::Union { variant, payload } => {
                let variant_name = intern::resolve_display(*variant);
                if payload.is_empty() {
                    write!(f, "{variant_name}")
                } else {
                    write!(f, "{variant_name}(")?;
                    for (i, v) in payload.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{v}")?;
                    }
                    write!(f, ")")
                }
            }
            Value::CoroutineHandle(_) => write!(f, "<coroutine>"),
            Value::TaskHandle(id) => write!(f, "<task:{id}>"),
        }
    }
}

impl Value {
    /// Convenience constructor: wrap a string in `Arc`.
    #[inline]
    pub fn string(s: impl Into<String>) -> Self {
        Value::String(Arc::new(s.into()))
    }

    /// Convenience constructor: wrap a vec in `Arc`.
    #[inline]
    pub fn list(items: Vec<Value>) -> Self {
        Value::List(Arc::new(items))
    }

    /// Convenience constructor: wrap a struct name + field map in `Arc`.
    /// Accepts string keys and interns them via the thread-local display interner.
    #[inline]
    pub fn strct(name: impl AsRef<str>, fields: HashMap<String, Value>) -> Self {
        let name_id = intern::intern_display(name.as_ref());
        let interned_fields: HashMap<InternedId, Value> = fields
            .into_iter()
            .map(|(k, v)| (intern::intern_display(&k), v))
            .collect();
        Value::Struct(name_id, Arc::new(interned_fields))
    }

    /// Convenience constructor: create a struct from pre-interned field keys.
    /// This is the fast path — no string interning overhead.
    #[inline]
    pub fn strct_interned(name_id: InternedId, fields: HashMap<InternedId, Value>) -> Self {
        Value::Struct(name_id, Arc::new(fields))
    }

    /// Borrow the inner string slice, if this is a `String` variant.
    #[inline]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Borrow the inner slice, if this is a `List` variant.
    #[inline]
    pub fn as_list(&self) -> Option<&[Value]> {
        match self {
            Value::List(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Get a mutable reference to the inner `Vec`, cloning the `Arc` if needed (CoW).
    #[inline]
    pub fn as_list_mut(&mut self) -> Option<&mut Vec<Value>> {
        match self {
            Value::List(v) => Some(Arc::make_mut(v)),
            _ => None,
        }
    }

    /// Look up a field on a struct value by string name, resolving via the
    /// thread-local display interner.
    pub fn get_field(&self, field: &str) -> Option<&Value> {
        match self {
            Value::Struct(_, fields) => {
                let id = intern::intern_display(field);
                fields.get(&id)
            }
            _ => None,
        }
    }

    fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Int(n) => *n != 0,
            Value::String(s) => !s.is_empty(),
            Value::None => false,
            Value::Err(_) => false,
            _ => true,
        }
    }

    /// Check structural equality for test assertions.
    fn matches(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => (a - b).abs() < f64::EPSILON,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Ok(a), Value::Ok(b)) => {
                // Ok(None) is a wildcard — matches any Ok value
                matches!(b.as_ref(), Value::None) || a.matches(b)
            }
            (Value::Err(a), Value::Err(b)) => a == b,
            (Value::None, Value::None) => true,
            (Value::Struct(n1, f1), Value::Struct(n2, f2)) => {
                // Allow empty-string name (interned) to match any struct name.
                // InternedId comparison is O(1) — just integer equality.
                let empty_id = intern::intern_display("");
                (n1 == &empty_id || n2 == &empty_id || n1 == n2)
                    && f1.len() == f2.len()
                    && f1
                        .iter()
                        .all(|(k, v)| f2.get(k).is_some_and(|v2| v.matches(v2)))
            }
            (Value::List(a), Value::List(b)) => {
                a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.matches(y))
            }
            (
                Value::Union {
                    variant: v1,
                    payload: p1,
                },
                Value::Union {
                    variant: v2,
                    payload: p2,
                },
            ) => v1 == v2 && p1.len() == p2.len() && p1.iter().zip(p2).all(|(a, b)| a.matches(b)),
            _ => false,
        }
    }
}

/// Type alias for the scope stack. Uses `SmallVec` with inline capacity of 4
/// to avoid heap allocation for the common case (root scope + function body +
/// 1-2 nested blocks like if/while/each).
type ScopeStack = SmallVec<[HashMap<InternedId, Value>; 4]>;

/// Evaluation environment with a scope stack.
/// Each scope is a HashMap of variable bindings keyed by interned `u32` ids.
/// `+let` defines into the top scope, `+set` mutates the nearest scope that
/// already contains the variable, and lookups walk the stack from top to bottom.
///
/// Variable names are interned via a `SharedInterner` (Arc-backed) so that
/// lookups use fast `u32` hashing/comparison instead of string operations.
/// Creating a new Env from a Program's interner is O(1) (Arc clone) instead
/// of cloning the entire HashMap + Vec.
pub struct Env {
    scopes: ScopeStack,
    /// Shared runtime state for +shared variable access.
    shared_runtime: Option<crate::session::SharedRuntime>,
    /// Local cache of shared vars (key = "Module.name") for borrow-friendly reads.
    /// These remain String-keyed since they use compound module-qualified names
    /// and are accessed far less frequently than local variables.
    shared_cache: HashMap<String, Value>,
    /// Arc-backed string interner — cloning is O(1) reference-count bump.
    /// Seeded from `Program::shared_interner()` when available so that all
    /// name→id lookups on the hot path are guaranteed cache hits.
    /// Wrapped in `RefCell` so that `&self` methods (like `get`) can still intern
    /// previously-unseen names without requiring `&mut self`. The `RefCell`
    /// overhead is minimal since `SharedInterner::get()` is a direct Arc read.
    interner: std::cell::RefCell<SharedInterner>,
}

impl Env {
    pub fn new() -> Self {
        // Seed from the thread-local interner so existing interned ids stay consistent
        let interner = STRING_INTERNER.with(|si| si.borrow().shared());
        let mut env = Self {
            scopes: smallvec::smallvec![HashMap::new()],
            shared_runtime: None,
            shared_cache: HashMap::new(),
            interner: std::cell::RefCell::new(interner),
        };
        // Auto-pick up thread-local SharedRuntime if set
        SHARED_RUNTIME.with(|rt| {
            if let Some(rt) = rt.borrow().as_ref() {
                env.set_runtime(rt.clone());
            }
        });
        env
    }

    /// Create an Env seeded with a pre-populated interner from a Program.
    /// This is the fast path: cloning a `SharedInterner` is O(1) (Arc clone),
    /// and the Program's interner already contains all names in the AST, so
    /// every `intern_name()` call during evaluation is a cache hit.
    pub fn new_with_interner(interner: &StringInterner) -> Self {
        let mut env = Self {
            scopes: smallvec::smallvec![HashMap::new()],
            shared_runtime: None,
            shared_cache: HashMap::new(),
            interner: std::cell::RefCell::new(interner.shared()),
        };
        // Auto-pick up thread-local SharedRuntime if set
        SHARED_RUNTIME.with(|rt| {
            if let Some(rt) = rt.borrow().as_ref() {
                env.set_runtime(rt.clone());
            }
        });
        env
    }

    /// Create an Env from a pre-built SharedInterner (O(1) Arc clone).
    /// This is the fastest path for creating Envs during function calls —
    /// avoids even the `StringInterner::shared()` conversion.
    pub fn new_with_shared_interner(interner: &SharedInterner) -> Self {
        let mut env = Self {
            scopes: smallvec::smallvec![HashMap::new()],
            shared_runtime: None,
            shared_cache: HashMap::new(),
            interner: std::cell::RefCell::new(interner.clone()),
        };
        // Auto-pick up thread-local SharedRuntime if set
        SHARED_RUNTIME.with(|rt| {
            if let Some(rt) = rt.borrow().as_ref() {
                env.set_runtime(rt.clone());
            }
        });
        env
    }

    /// Intern a variable name string, returning its compact u32 id for use as
    /// a scope key. Uses the env-local shared interner.
    /// Fast path: read-only probe through Arc — since the interner is pre-seeded
    /// with all AST names, this is almost always a cache hit.
    #[inline]
    fn intern_name(&self, name: &str) -> InternedId {
        // Fast path: read-only probe — avoids RefCell write lock overhead
        if let Some(id) = self.interner.borrow().get(name) {
            return id;
        }
        // Slow path: name not yet interned (rare when seeded from Program).
        // SharedInterner::intern uses copy-on-write via Arc::make_mut.
        self.interner.borrow_mut().intern(name)
    }

    /// Resolve an interned id back to its string. Used for error messages
    /// and debug display.
    #[inline]
    fn resolve_name(&self, id: InternedId) -> String {
        self.interner
            .borrow()
            .resolve(id)
            .unwrap_or("<unknown>")
            .to_string()
    }

    /// Attach shared runtime state for +shared variable access.
    pub fn set_runtime(&mut self, rt: crate::session::SharedRuntime) {
        // Pre-populate the local cache from the runtime's current shared_vars.
        if let Ok(state) = rt.read() {
            for (key, val) in &state.shared_vars {
                self.shared_cache.insert(key.clone(), val.clone());
            }
        }
        self.shared_runtime = Some(rt);
    }

    /// Populate the shared_cache directly from the program's module shared var
    /// declarations. This is a fallback for when SharedRuntime is not available
    /// (e.g. in tests or CLI mode). Evaluates each default expression.
    pub fn populate_shared_from_program(&mut self, program: &ast::Program) {
        for module in &program.modules {
            for sv in &module.shared_vars {
                let key = format!("{}.{}", module.name, sv.name);
                if !self.shared_cache.contains_key(&key) {
                    let value = eval_expr_standalone(program, &sv.default)
                        .unwrap_or(Value::Int(0));
                    self.shared_cache.insert(key, value);
                }
            }
        }
    }

    /// Inherit shared variable state from a parent Env.
    /// Used when creating a child Env for a nested function call so that
    /// shared variables remain accessible even without a SharedRuntime
    /// (e.g. during sync tests or CLI eval).
    fn inherit_shared_from(&mut self, parent: &Env) {
        if self.shared_cache.is_empty() && !parent.shared_cache.is_empty() {
            self.shared_cache.clone_from(&parent.shared_cache);
        }
        if self.shared_runtime.is_none() {
            if let Some(rt) = &parent.shared_runtime {
                self.shared_runtime = Some(rt.clone());
            }
        }
    }

    /// Push a new empty scope onto the stack.
    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    /// Pop the top scope. Panics if only the root scope remains.
    fn pop_scope(&mut self) {
        debug_assert!(self.scopes.len() > 1, "cannot pop the root scope");
        self.scopes.pop();
    }

    /// Define a new variable in the current (top) scope.
    /// Used by `+let` and parameter binding.
    pub fn set(&mut self, name: &str, value: Value) {
        let id = self.intern_name(name);
        self.scopes
            .last_mut()
            .expect("scope stack empty")
            .insert(id, value);
    }

    /// Define a variable in the current scope using a pre-interned id.
    /// This is the fast path — no string→id conversion needed.
    #[inline]
    pub fn set_id(&mut self, id: InternedId, value: Value) {
        self.scopes
            .last_mut()
            .expect("scope stack empty")
            .insert(id, value);
    }

    fn current_module_name(&self) -> Option<String> {
        FN_NAME_STACK.with(|s| {
            let stack = s.borrow();
            stack
                .last()
                .and_then(|fn_name| fn_name.split_once('.').map(|(m, _)| m.to_string()))
        })
    }

    fn shared_key(&self, name: &str) -> Option<String> {
        self.current_module_name()
            .map(|module| format!("{module}.{name}"))
    }

    fn read_shared_value(&mut self, key: &str) -> Option<Value> {
        if let Some(rt) = &self.shared_runtime {
            if let Ok(state) = rt.read() {
                if let Some(value) = state.shared_vars.get(key) {
                    let value = value.clone();
                    self.shared_cache.insert(key.to_string(), value.clone());
                    return Some(value);
                }
            }
        }
        self.shared_cache.get(key).cloned()
    }

    fn materialize_shared_value(&mut self, name: &str) -> Option<(String, Value)> {
        let key = self.shared_key(name)?;
        if let Some(value) = self.read_shared_value(&key) {
            return Some((key, value));
        }

        let program = get_shared_program()?;
        let module_name = self.current_module_name()?;
        let module = program.modules.iter().find(|m| m.name == module_name)?;
        let shared = module.shared_vars.iter().find(|sv| sv.name == name)?;
        let value = eval_expr_standalone(&program, &shared.default).unwrap_or(Value::Int(0));

        if let Some(rt) = &self.shared_runtime {
            if let Ok(mut state) = rt.write() {
                state.shared_vars.entry(key.clone()).or_insert_with(|| value.clone());
            }
        }
        self.shared_cache.insert(key.clone(), value.clone());
        Some((key, value))
    }

    /// Mutate an existing variable: walk scopes top-to-bottom, update the
    /// first scope that contains `name`. If not found in local scopes, check
    /// shared vars (keyed by "Module.name"). If still not found, insert into
    /// the top scope (same as `set`). Used by `+set`.
    fn set_existing(&mut self, name: &str, value: Value) {
        let id = self.intern_name(name);
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(&id) {
                scope.insert(id, value);
                return;
            }
        }
        if let Some((key, _)) = self.materialize_shared_value(name) {
            if let Some(rt) = &self.shared_runtime {
                if let Ok(mut state) = rt.write() {
                    state.shared_vars.insert(key.clone(), value.clone());
                }
            }
            self.shared_cache.insert(key, value);
            return;
        }
        // Not found anywhere — define in current scope
        self.set(name, value);
    }

    /// Look up a variable by walking scopes from top to bottom.
    /// Falls back to shared vars cache (keyed by "Module.name") if not found locally.
    fn get(&mut self, name: &str) -> Result<Value> {
        let id = self.intern_name(name);
        for scope in self.scopes.iter().rev() {
            if let Some(val) = scope.get(&id) {
                return Ok(val.clone());
            }
        }
        if let Some((_, value)) = self.materialize_shared_value(name) {
            return Ok(value);
        }
        Err(anyhow!("undefined variable `{name}`"))
    }

    /// Look up a variable using a pre-interned id. Fast path — no string interning.
    #[inline]
    fn get_id(&self, id: InternedId) -> Option<&Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(val) = scope.get(&id) {
                return Some(val);
            }
        }
        None
    }

    /// Raw lookup (returns Option) — used for special variables like __coroutine_handle.
    fn get_raw(&self, name: &str) -> Option<&Value> {
        let id = self.intern_name(name);
        for scope in self.scopes.iter().rev() {
            if let Some(val) = scope.get(&id) {
                return Some(val);
            }
        }
        None
    }

    /// Flatten all visible bindings into display-string pairs for task inspection.
    /// Skips internal names (prefixed with `__`) and deduplicates across scopes
    /// (inner scopes shadow outer ones).
    pub fn snapshot_bindings(&self) -> Vec<(String, String)> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        // Walk from top (innermost) to bottom (outermost) so shadowed names are skipped.
        for scope in self.scopes.iter().rev() {
            for (id, val) in scope {
                let name = self.resolve_name(*id);
                if name.starts_with("__") {
                    continue;
                }
                if seen.insert(*id) {
                    result.push((name, format!("{val}")));
                }
            }
        }
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }
}

/// Try to evaluate using the JIT compiler, falling back to the interpreter.
/// Returns (result_string, was_compiled).
///
/// If `cache` is provided along with a `revision`, the compiled module is reused
/// when the revision hasn't changed, avoiding recompilation on every eval call.
/// Find similar function names for "did you mean" suggestions.
pub fn suggest_similar(program: &ast::Program, name: &str) -> String {
    let bare = name.rsplit('.').next().unwrap_or(name);
    let mut candidates: Vec<String> = Vec::new();
    for m in &program.modules {
        for f in &m.functions {
            let qname = format!("{}.{}", m.name, f.name);
            if f.name == bare || f.name.contains(bare) || bare.contains(&f.name) {
                candidates.push(qname);
            }
        }
    }
    for f in &program.functions {
        if f.name.contains(bare) || bare.contains(&f.name) {
            candidates.push(f.name.clone());
        }
    }
    if candidates.is_empty() { String::new() }
    else { format!(". Did you mean: {}?", candidates.join(", ")) }
}

pub fn eval_compiled_or_interpreted(
    program: &ast::Program,
    function_name: &str,
    input: &parser::Expr,
) -> Result<(String, bool)> {
    eval_compiled_or_interpreted_cached(program, function_name, input, None, 0)
}

/// Like `eval_compiled_or_interpreted`, but with an optional JIT cache.
pub fn eval_compiled_or_interpreted_cached(
    program: &ast::Program,
    function_name: &str,
    input: &parser::Expr,
    cache: Option<&JitCache>,
    revision: usize,
) -> Result<(String, bool)> {
    // Install the program's interner for Value::Display resolution.
    intern::set_display_interner(&program.shared_interner);
    // If it's a builtin, use eval_call_with_input directly
    if program.get_function(function_name).is_none() && crate::builtins::is_builtin(function_name) {
        let result = eval_call_with_input(program, function_name, input)?;
        return Ok((result, false));
    }

    let func = program
        .get_function(function_name)
        .ok_or_else(|| anyhow!("function `{function_name}` not found{}", crate::eval::suggest_similar(program, function_name)))?;

    // Try compiled path
    let returns_string = matches!(&func.return_type, ast::Type::String);
    if crate::compiler::is_compilable_function(func) {
        if let Ok(args) = input_to_i64_args(input, func) {
            // Try to use cached compiled module
            if let Some(jit_cache) = cache {
                if let Ok(mut guard) = jit_cache.lock() {
                    // Check if cache is valid for this revision
                    let needs_compile = match &*guard {
                        Some((cached_rev, _)) => *cached_rev != revision,
                        None => true,
                    };

                    if needs_compile {
                        // Compile and store in cache
                        if let Ok(compiled) = crate::compiler::compile(program) {
                            *guard = Some((revision, compiled));
                        }
                    }

                    // Try to use the cached compiled module
                    if let Some((_, ref mut compiled)) = *guard {
                        if returns_string {
                            if let Ok(result) = compiled.call_string(function_name, &args) {
                                return Ok((format!("\"{result}\""), true));
                            }
                        } else {
                            if let Ok(result) = compiled.call_i64(function_name, &args) {
                                return Ok((format!("{result}"), true));
                            }
                        }
                    }
                }
            } else {
                // No cache — compile fresh each time (original behavior)
                if let Ok(mut compiled) = crate::compiler::compile(program) {
                    if returns_string {
                        if let Ok(result) = compiled.call_string(function_name, &args) {
                            return Ok((format!("\"{result}\""), true));
                        }
                    } else {
                        if let Ok(result) = compiled.call_i64(function_name, &args) {
                            return Ok((format!("{result}"), true));
                        }
                    }
                }
            }
        }
    }

    // Try bytecode VM (middle tier — more complete than JIT, faster than tree-walk)
    // For async functions, the VM will hit AwaitIo and fall through to the interpreter.
    if let Ok(result) = try_vm_execute(program, func, input) {
        return Ok((result, false));
    }

    // Fall back to interpreter
    let result = eval_call_with_input(program, function_name, input)?;
    Ok((result, false))
}

/// Try to execute a function via the bytecode VM. Returns the formatted result
/// string on success, or an error to fall through to the tree-walker.
fn try_vm_execute(
    program: &ast::Program,
    func: &ast::FunctionDecl,
    input: &parser::Expr,
) -> Result<String> {
    // Convert input expression to VM argument values
    let vm_args = input_to_vm_args(input, func)?;

    // Compile to bytecode
    let compiled = vm::compile_function(func, program)?;

    // Execute with IO support — if the function hits AwaitIo, bail to
    // let the tree-walker handle it (it has proper coroutine support).
    let result = vm::execute_with_io(&compiled, vm_args, program, &|op_name, _args| {
        bail!("VM: sync path cannot perform async IO ({op_name})")
    })?;
    Ok(format!("{result}"))
}

/// Convert a parser expression to a Vec<Value> suitable for VM execution.
fn input_to_vm_args(input: &parser::Expr, func: &ast::FunctionDecl) -> Result<Vec<Value>> {
    if func.params.is_empty() {
        return Ok(vec![]);
    }

    let input_val = eval_parser_expr_standalone(input)?;

    match input_val {
        // Struct literal with named fields → extract in param order
        Value::Struct(_, ref fields) => {
            let mut args = Vec::new();
            for param in &func.params {
                let pid = intern::intern_display(&param.name);
                if let Some(val) = fields.get(&pid) {
                    args.push(val.clone());
                } else {
                    bail!("missing argument for parameter `{}`", param.name);
                }
            }
            Ok(args)
        }
        // Empty struct (no args provided) and params exist → error
        Value::None if !func.params.is_empty() => {
            bail!("no arguments provided for {} parameter(s)", func.params.len());
        }
        // Single value with single param → use directly
        _ if func.params.len() == 1 => Ok(vec![input_val]),
        _ => bail!("cannot convert input to VM args for {} parameters", func.params.len()),
    }
}

/// Try to convert a parser expression to i64 args for the compiler.
fn input_to_i64_args(input: &parser::Expr, func: &ast::FunctionDecl) -> Result<Vec<i64>> {
    match input {
        parser::Expr::Int(n) => Ok(vec![*n]),
        parser::Expr::Bool(b) => Ok(vec![*b as i64]),
        parser::Expr::StructLiteral(fields) => {
            // Match field names to param order
            let mut args = Vec::new();
            for param in &func.params {
                let field = fields
                    .iter()
                    .find(|f| f.name == param.name)
                    .ok_or_else(|| anyhow!("missing field `{}`", param.name))?;
                match &field.value {
                    parser::Expr::Int(n) => args.push(*n),
                    parser::Expr::Bool(b) => args.push(*b as i64),
                    parser::Expr::Float(f) => args.push(f.to_bits() as i64), // bit-cast
                    _ => bail!("unsupported arg type for compiled call"),
                }
            }
            Ok(args)
        }
        _ => bail!("unsupported input format for compiled call"),
    }
}

/// Evaluate a function call with given input, returning the result as a displayable string.
pub fn eval_call_with_input(
    program: &ast::Program,
    function_name: &str,
    input: &parser::Expr,
) -> Result<String> {
    // Install the program's interner for Value::Display resolution.
    intern::set_display_interner(&program.shared_interner);
    set_shared_program(Some(std::sync::Arc::new(program.clone())));
    if let Some(rt) = get_shared_runtime() {
        init_missing_shared_runtime_vars(program, &rt);
    }
    // Try user-defined function first
    if let Some(func) = program.get_function(function_name) {
        let input_val = eval_parser_expr_standalone(input)?;
        let mut env = Env::new_with_shared_interner(&program.shared_interner);
        env.populate_shared_from_program(program);
        bind_input_to_params(program, func, &input_val, &mut env);

        let returns_result = matches!(&func.return_type, ast::Type::Result(_));

        let qualified = program.qualify_function_name(function_name);
        FN_NAME_STACK.with(|s| s.borrow_mut().push(qualified));
        let body_result = eval_function_body(program, &func.body, &mut env);
        FN_NAME_STACK.with(|s| s.borrow_mut().pop());
        return match body_result {
            Ok(val) => {
                let result = if returns_result {
                    match &val {
                        Value::Ok(_) | Value::Err(_) => val,
                        _ => Value::Ok(Box::new(val)),
                    }
                } else {
                    val
                };
                Ok(format!("{result}"))
            }
            Err(e) => Ok(format!("Err({e})")),
        };
    }

    // Try as a builtin function
    if crate::builtins::is_builtin(function_name) {
        // Get args in order from the parser expression (not via HashMap which loses order)
        let args = match input {
            parser::Expr::StructLiteral(fields) => {
                // Preserve field order from the parser
                fields
                    .iter()
                    .map(|f| eval_parser_expr_standalone(&f.value))
                    .collect::<Result<Vec<_>>>()?
            }
            _ => {
                let input_val = eval_parser_expr_standalone(input)?;
                match &input_val {
                    Value::None => vec![],
                    other => vec![other.clone()],
                }
            }
        };
        let mut env = Env::new_with_shared_interner(&program.shared_interner);
        let call = ast::CallExpr {
            callee: function_name.to_string(),
            args: vec![], // not used — we call eval_call_inner directly
        };
        // Build a fake call and evaluate using the builtin path
        match eval_call_inner_with_args(program, function_name, args, &mut env) {
            Ok(val) => Ok(format!("{val}")),
            Err(e) => Ok(format!("Err({e})")),
        }
    } else {
        bail!("function `{function_name}` not found (not a user function or builtin)")
    }
}

/// Evaluate an inline expression directly (not a function call).
/// Used for `!eval 1 + 2`, `!eval concat("a", "b")`, etc.
pub fn eval_inline_expr(program: &ast::Program, expr: &parser::Expr) -> Result<Value> {
    // Install the program's interner for Value::Display resolution.
    intern::set_display_interner(&program.shared_interner);
    set_shared_program(Some(std::sync::Arc::new(program.clone())));
    if let Some(rt) = get_shared_runtime() {
        init_missing_shared_runtime_vars(program, &rt);
    }
    eval_parser_expr_with_program(expr, program)
}

/// Evaluate an inline expression with IO support via a coroutine handle.
/// When the expression contains IO builtin calls (shell_exec, http_get, etc.),
/// they are executed through the coroutine runtime automatically.
/// Must be called from within a `spawn_blocking` context (not a tokio async task).
pub fn eval_inline_expr_with_io(
    program: &ast::Program,
    expr: &parser::Expr,
    io_sender: tokio::sync::mpsc::Sender<crate::coroutine::IoRequest>,
) -> Result<Value> {
    set_shared_program(Some(std::sync::Arc::new(program.clone())));
    if let Some(rt) = get_shared_runtime() {
        init_missing_shared_runtime_vars(program, &rt);
    }
    let handle = crate::coroutine::CoroutineHandle::new(io_sender);
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    env.populate_shared_from_program(program);
    env.set("__coroutine_handle", Value::CoroutineHandle(handle));
    eval_parser_expr_with_env(expr, program, &mut env)
}

/// Check if a parser expression contains any IO builtin calls.
/// Used to determine if inline `!eval` needs async execution.
pub fn expr_contains_io_builtin(expr: &parser::Expr) -> bool {
    match expr {
        parser::Expr::Call { callee, args } => {
            let name = parser_callee_name(callee);
            if crate::builtins::is_io_builtin(&name) {
                return true;
            }
            // Check if any argument sub-expressions contain IO builtins
            args.iter().any(expr_contains_io_builtin)
        }
        parser::Expr::Binary { left, right, .. } => {
            expr_contains_io_builtin(left) || expr_contains_io_builtin(right)
        }
        parser::Expr::Unary { expr: inner, .. } => {
            expr_contains_io_builtin(inner)
        }
        parser::Expr::FieldAccess { base, .. } => {
            expr_contains_io_builtin(base)
        }
        parser::Expr::StructLiteral(fields) => {
            fields.iter().any(|f| expr_contains_io_builtin(&f.value))
        }
        parser::Expr::Cast { expr: inner, .. } => {
            expr_contains_io_builtin(inner)
        }
        parser::Expr::Int(_)
        | parser::Expr::Float(_)
        | parser::Expr::Bool(_)
        | parser::Expr::String(_)
        | parser::Expr::Ident(_) => false,
    }
}

/// Evaluate a test case against a function in the program.
/// Bind test input values to function parameters.
/// Handles three cases:
/// 1. Single param → bind directly
/// 2. Multi-param, all fields match param names → bind each field to its param
/// 3. Multi-param with struct params → distribute fields based on type definitions
pub fn bind_input_to_params(
    program: &ast::Program,
    func: &ast::FunctionDecl,
    input: &Value,
    env: &mut Env,
) {
    match (input, func.params.len()) {
        (_, 0) => {} // no params
        (Value::Struct(_, fields), n) if n == 1 => {
            // Single struct-typed param — pass the whole struct
            env.set(&func.params[0].name, input.clone());
            // Also expose fields directly for field access (e.g., input.name)
            for (k_id, v) in fields.iter() {
                let k = intern::resolve_display(*k_id);
                env.set(&k, v.clone());
            }
        }
        (Value::Struct(_, fields), _) => {
            // Multi-param function with struct input (key=value pairs)
            // First, check if all fields directly match param names
            let all_match = func
                .params
                .iter()
                .all(|p| {
                    let pid = intern::intern_display(&p.name);
                    fields.contains_key(&pid)
                });

            if all_match {
                // Direct match: a=3 b=4 for (a:Int, b:Int)
                for param in &func.params {
                    let pid = intern::intern_display(&param.name);
                    if let Some(val) = fields.get(&pid) {
                        env.set(&param.name, val.clone());
                    }
                }
            } else {
                // Check for positional fields (_0, _1, ...) from space-separated args
                let is_positional = fields.keys().any(|k| {
                    let s = intern::resolve_display(*k);
                    s.starts_with('_') && s[1..].parse::<usize>().is_ok()
                });
                if is_positional && fields.len() == func.params.len() {
                    for (i, param) in func.params.iter().enumerate() {
                        let pos_id = intern::intern_display(&format!("_{i}"));
                        if let Some(val) = fields.get(&pos_id) {
                            env.set(&param.name, val.clone());
                        }
                    }
                } else {
                    // Smart distribution: fields may belong to struct-typed params
                    // For each param, check if it's a struct type and collect matching fields
                    let mut used_fields: std::collections::HashSet<&str> =
                        std::collections::HashSet::new();

                    for param in &func.params {
                        // Check if this param matches a field directly
                        let pid = intern::intern_display(&param.name);
                        if let Some(val) = fields.get(&pid) {
                            env.set(&param.name, val.clone());
                            used_fields.insert(&param.name);
                            continue;
                        }

                        // Check if this param is a struct type — look up its field names
                        if let ast::Type::Struct(type_name) = &param.ty {
                            if let Some(type_fields) = get_struct_fields(program, type_name) {
                                // Collect input fields that match this struct's fields
                                let mut struct_fields: HashMap<InternedId, Value> = HashMap::new();
                                for (tf_name, _) in &type_fields {
                                    let tf_id = intern::intern_display(tf_name);
                                    if let Some(val) = fields.get(&tf_id) {
                                        struct_fields.insert(tf_id, val.clone());
                                    }
                                }
                                if !struct_fields.is_empty() {
                                    let struct_val = Value::strct_interned(
                                        intern::intern_display(type_name),
                                        struct_fields,
                                    );
                                    env.set(&param.name, struct_val);
                                }
                            }
                        }
                    }
                }
            }
        }
        (_, 1) => {
            env.set(&func.params[0].name, input.clone());
        }
        _ => {
            // Multiple params but non-struct input — bind first param
            env.set(&func.params[0].name, input.clone());
        }
    }
}

/// Check if a name is a union variant. Uses the pre-built HashSet on Program
/// for O(1) lookup instead of scanning all type declarations.
#[inline]
fn is_union_variant(program: &ast::Program, name: &str) -> bool {
    program.is_union_variant(name)
}

/// Check if nested patterns match the payload values, binding variables on success.
fn match_nested_patterns(
    program: &ast::Program,
    patterns: &[ast::MatchPattern],
    values: &[Value],
    env: &mut Env,
) -> Result<bool> {
    for (pattern, value) in patterns.iter().zip(values.iter()) {
        if !match_single_pattern(program, pattern, value, env)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Match a single pattern against a value, binding variables on success.
fn match_single_pattern(
    program: &ast::Program,
    pattern: &ast::MatchPattern,
    value: &Value,
    env: &mut Env,
) -> Result<bool> {
    match pattern {
        ast::MatchPattern::Binding(name) => {
            if name != "_" {
                env.set(name, value.clone());
            }
            Ok(true)
        }
        ast::MatchPattern::Literal(lit) => {
            let expected = match lit {
                ast::Literal::Int(n) => Value::Int(*n),
                ast::Literal::Float(f) => Value::Float(*f),
                ast::Literal::Bool(b) => Value::Bool(*b),
                ast::Literal::String(s) => Value::string(s.clone()),
            };
            Ok(value.matches(&expected))
        }
        ast::MatchPattern::Variant {
            variant,
            sub_patterns,
        } => {
            if let Value::Union {
                variant: v,
                payload,
            } = value
            {
                if intern::resolve_display(*v) == *variant {
                    match_nested_patterns(program, sub_patterns, payload, env)
                } else {
                    Ok(false)
                }
            } else {
                Ok(false)
            }
        }
    }
}

/// Look up struct field names from the program's type declarations.
fn get_struct_fields(program: &ast::Program, type_name: &str) -> Option<Vec<(String, ast::Type)>> {
    for td in &program.types {
        if let ast::TypeDecl::Struct(s) = td {
            if s.name == type_name {
                return Some(
                    s.fields
                        .iter()
                        .map(|f| (f.name.clone(), f.ty.clone()))
                        .collect(),
                );
            }
        }
    }
    // Also check modules
    for module in &program.modules {
        for td in &module.types {
            if let ast::TypeDecl::Struct(s) = td {
                if s.name == type_name {
                    return Some(
                        s.fields
                            .iter()
                            .map(|f| (f.name.clone(), f.ty.clone()))
                            .collect(),
                    );
                }
            }
        }
    }
    None
}

pub fn eval_test_case(
    program: &ast::Program,
    function_name: &str,
    case: &parser::TestCase,
) -> Result<String> {
    eval_test_case_with_mocks(program, function_name, case, &[], &[])
}

pub fn init_missing_shared_runtime_vars(
    program: &ast::Program,
    runtime: &crate::session::SharedRuntime,
) {
    let mut missing = Vec::new();
    if let Ok(state) = runtime.read() {
        for module in &program.modules {
            for shared in &module.shared_vars {
                let key = format!("{}.{}", module.name, shared.name);
                if !state.shared_vars.contains_key(&key) {
                    let value = eval_expr_standalone(program, &shared.default).unwrap_or(Value::Int(0));
                    missing.push((key, value));
                }
            }
        }
    }

    if missing.is_empty() {
        return;
    }

    if let Ok(mut state) = runtime.write() {
        for (key, value) in missing {
            state.shared_vars.entry(key).or_insert(value);
        }
    }
}

pub fn eval_test_case_with_mocks(
    program: &ast::Program,
    function_name: &str,
    case: &parser::TestCase,
    mocks: &[crate::session::IoMock],
    http_routes: &[ast::HttpRoute],
) -> Result<String> {
    // Install the program's interner for Value::Display resolution.
    intern::set_display_interner(&program.shared_interner);
    set_shared_program(Some(std::sync::Arc::new(program.clone())));
    if let Some(rt) = get_shared_runtime() {
        init_missing_shared_runtime_vars(program, &rt);
    }
    let func = program
        .get_function(function_name)
        .ok_or_else(|| anyhow!("function `{function_name}` not found{}", crate::eval::suggest_similar(program, function_name)))?;

    let has_async = func
        .effects
        .iter()
        .any(|e| matches!(e, ast::Effect::Async | ast::Effect::Io));

    if has_async {
        // Async function: spin up a temporary coroutine runtime so +await
        // operations execute through real IO (with mock fallback if provided).
        // This avoids "requires async context" and "no mock for ..." errors.
        return eval_test_case_with_runtime(program, function_name, case, mocks, http_routes);
    }

    let input = eval_parser_expr_with_program(&case.input, program)?;
    let expected = eval_parser_expr_with_program(&case.expected, program)?;

    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    env.populate_shared_from_program(program);
    bind_input_to_params(program, func, &input, &mut env);

    // Execute function body (use named variant so FN_NAME_STACK has the qualified name
    // for shared variable resolution in Env::get())
    let result = eval_function_body_named(program, function_name, &func.body, &mut env);

    let msg = check_test_result(result, &func.return_type, &input, &expected, case.matcher.as_ref())?;

    // Check +after assertions (sync path — program is immutable here, so
    // after_checks on routes/modules work; tasks/mocks not applicable in sync)
    for after in &case.after_checks {
        check_after(program, after, None, http_routes)?;
    }

    Ok(msg)
}

/// Create a forked RuntimeState for test isolation.
/// Populates shared_vars from program defaults and includes the given HTTP routes.
/// Returns a SharedRuntime (Arc<RwLock<RuntimeState>>) suitable for set_shared_runtime.
fn fork_runtime_for_test(
    program: &ast::Program,
    http_routes: &[ast::HttpRoute],
) -> Option<crate::session::SharedRuntime> {
    let mut shared_vars = HashMap::new();
    for module in &program.modules {
        for sv in &module.shared_vars {
            let key = format!("{}.{}", module.name, sv.name);
            let value = eval_expr_standalone(program, &sv.default).unwrap_or(Value::Int(0));
            shared_vars.insert(key, value);
        }
    }
    let forked = crate::session::RuntimeState {
        http_routes: http_routes.to_vec(),
        shared_vars,
        agent_mailbox: std::collections::HashMap::new(),
        pending_commands: Vec::new(),
        library_errors: Vec::new(),
        library_load_errors: Vec::new(),
        failure_history: crate::session::FailureHistory::default(),
    };
    Some(std::sync::Arc::new(std::sync::RwLock::new(forked)))
}

/// Run a test case for an async function by spinning up a temporary coroutine
/// runtime. Mocks are checked first; unmatched IO operations execute for real.
fn eval_test_case_with_runtime(
    program: &ast::Program,
    function_name: &str,
    case: &parser::TestCase,
    mocks: &[crate::session::IoMock],
    http_routes: &[ast::HttpRoute],
) -> Result<String> {
    let func = program
        .get_function(function_name)
        .ok_or_else(|| anyhow!("function `{function_name}` not found{}", crate::eval::suggest_similar(program, function_name)))?;

    let input = eval_parser_expr_with_program(&case.input, program)?;
    let expected = eval_parser_expr_with_program(&case.expected, program)?;
    let return_type = func.return_type.clone();
    let program = program.clone();
    let fn_name = function_name.to_string();
    let mocks = mocks.to_vec();
    let matcher = case.matcher.clone();
    let after_checks = case.after_checks.clone();
    let routes = http_routes.to_vec();
    let forked_runtime = fork_runtime_for_test(&program, http_routes);
    // Create an isolated SessionMeta for test execution.
    // Pre-populate io_mocks so mock_set/mock_clear builtins work in tests.
    let forked_meta: crate::session::SharedMeta = {
        let mut m = crate::session::SessionMeta::new();
        m.io_mocks = mocks.to_vec();
        std::sync::Arc::new(std::sync::Mutex::new(m))
    };

    // Spin up a temporary tokio runtime + coroutine IO loop on a dedicated
    // thread.  This works whether or not the caller is already inside a tokio
    // runtime (nested block_on is not allowed, so we always use a fresh thread).
    std::thread::spawn(move || {
        // Use a forked RuntimeState for test isolation — shared vars are fresh
        // copies from program defaults, not the live runtime.
        let forked_rt_clone = forked_runtime.clone();
        set_shared_runtime(forked_runtime);
        set_shared_meta(Some(forked_meta.clone()));
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|e| anyhow!("failed to create async test runtime: {e}"))?;

        rt.block_on(async {
            let (runtime, mut io_rx) = crate::coroutine::Runtime::new();
            let runtime = std::sync::Arc::new(runtime);
            let io_sender = runtime.io_sender();

            // Spawn the IO handler loop
            let rt_handle = runtime.clone();
            let io_loop = tokio::spawn(async move {
                while let Some(request) = io_rx.recv().await {
                    let rt = rt_handle.clone();
                    tokio::spawn(async move {
                        rt.handle_io(request).await;
                    });
                }
            });

            // Run the evaluation in a blocking task so blocking_send/blocking_recv
            // don't stall the tokio executor.
            let forked_rt_for_blocking = forked_rt_clone;
            let forked_meta_for_blocking = forked_meta.clone();
            let eval_result = tokio::task::spawn_blocking(move || {
                set_shared_runtime(forked_rt_for_blocking);
                set_shared_meta(Some(forked_meta_for_blocking));
                // Set program snapshot so query builtins (symbols_list, source_get,
                // etc.) work inside test execution.
                set_shared_program(Some(std::sync::Arc::new(program.clone())));
                // Install the display interner on this worker thread so that
                // bind_input_to_params can resolve InternedIds ↔ strings
                // (intern::resolve_display / intern::intern_display use a
                // thread-local interner that is empty on freshly spawned threads).
                intern::set_display_interner(&program.shared_interner);
                let func = program
                    .get_function(&fn_name)
                    .ok_or_else(|| anyhow!("function `{fn_name}` not found"))?;

                let mut env = Env::new_with_shared_interner(&program.shared_interner);
                env.populate_shared_from_program(&program);

                // Tests always use mock-only handles.  Unmocked IO operations
                // fail with "no mock for ..." instead of executing real IO,
                // which would deadlock when the test calls back into the same
                // server (e.g. http_get to /api/status while the session lock
                // is held).
                let handle = crate::coroutine::CoroutineHandle::new_mock(mocks);
                env.set("__coroutine_handle", Value::CoroutineHandle(handle));

                bind_input_to_params(&program, func, &input, &mut env);
                // Use eval_function_body_named so FN_NAME_STACK is populated
                // with the qualified function name — required for +shared
                // variable resolution in Env::get()/set_existing().
                let result = eval_function_body_named(&program, &fn_name, &func.body, &mut env);
                let msg = check_test_result(result, &return_type, &input, &expected, matcher.as_ref())?;

                // Check +after assertions
                for after in &after_checks {
                    check_after(&program, after, None, &routes)?;
                }

                Ok::<String, anyhow::Error>(msg)
            })
            .await
            .map_err(|e| anyhow!("async test task panicked: {e}"))??;

            // Shut down the IO loop
            io_loop.abort();

            Ok(eval_result)
        })
    })
    .join()
    .map_err(|_| anyhow!("async test thread panicked"))?
}

/// Run a test case through the real coroutine runtime (for async functions with
/// real IO). Falls back to mock-only execution for sync functions.
/// Must be called from within a tokio runtime.
pub async fn eval_test_case_async(
    program: &ast::Program,
    function_name: &str,
    case: &parser::TestCase,
    mocks: &[crate::session::IoMock],
    _io_sender: tokio::sync::mpsc::Sender<crate::coroutine::IoRequest>,
    http_routes: &[ast::HttpRoute],
) -> Result<String> {
    let func = program
        .get_function(function_name)
        .ok_or_else(|| anyhow!("function `{function_name}` not found{}", crate::eval::suggest_similar(program, function_name)))?;

    let has_async = func
        .effects
        .iter()
        .any(|e| matches!(e, ast::Effect::Async | ast::Effect::Io));

    // If the function isn't async, delegate to the sync path
    if !has_async {
        return eval_test_case_with_mocks(program, function_name, case, mocks, http_routes);
    }

    // Install the display interner on the calling thread so
    // eval_parser_expr_with_program can resolve/intern names correctly
    // (it uses intern::intern_display which needs the thread-local interner).
    intern::set_display_interner(&program.shared_interner);

    let input = eval_parser_expr_with_program(&case.input, program)?;
    let expected = eval_parser_expr_with_program(&case.expected, program)?;

    // Clone what we need for the blocking task
    let program = program.clone();
    let fn_name = function_name.to_string();
    let return_type = func.return_type.clone();
    let mocks = mocks.to_vec();
    let matcher = case.matcher.clone();
    let after_checks = case.after_checks.clone();
    let routes = http_routes.to_vec();
    let forked_rt = fork_runtime_for_test(&program, http_routes);
    let forked_meta_async: crate::session::SharedMeta = {
        let mut m = crate::session::SessionMeta::new();
        m.io_mocks = mocks.to_vec();
        std::sync::Arc::new(std::sync::Mutex::new(m))
    };

    let eval_result = tokio::task::spawn_blocking(move || {
        // Use a forked RuntimeState for test isolation — shared vars are fresh
        // copies from program defaults, not the live runtime.
        set_shared_runtime(forked_rt);
        set_shared_meta(Some(forked_meta_async));
        // Set program snapshot so query builtins (symbols_list, source_get,
        // etc.) work inside async test execution.
        set_shared_program(Some(std::sync::Arc::new(program.clone())));
        // Install the display interner on this worker thread so that
        // bind_input_to_params can resolve InternedIds ↔ strings
        // (intern::resolve_display / intern::intern_display use a
        // thread-local interner that is empty on freshly spawned threads).
        intern::set_display_interner(&program.shared_interner);

        let func = program
            .get_function(&fn_name)
            .ok_or_else(|| anyhow!("function `{fn_name}` not found"))?;

        let mut env = Env::new_with_shared_interner(&program.shared_interner);
        env.populate_shared_from_program(&program);

        // Tests always use mock-only handles — see comment in
        // eval_test_case_with_runtime for rationale.
        let handle = crate::coroutine::CoroutineHandle::new_mock(mocks);
        env.set("__coroutine_handle", Value::CoroutineHandle(handle));

        bind_input_to_params(&program, func, &input, &mut env);

        // Use eval_function_body_named so FN_NAME_STACK is populated
        // with the qualified function name — required for +shared
        // variable resolution in Env::get()/set_existing().
        let result = eval_function_body_named(&program, &fn_name, &func.body, &mut env);
        let msg = check_test_result(result, &return_type, &input, &expected, matcher.as_ref())?;

        // Check +after assertions
        for after in &after_checks {
            check_after(&program, after, None, &routes)?;
        }

        Ok::<String, anyhow::Error>(msg)
    })
    .await
    .map_err(|e| anyhow!("async test task panicked: {e}"))??;

    Ok(eval_result)
}

/// Compare the eval result against expected, handling Result<T> wrapping.
/// When a `TestMatcher` is present, it overrides exact comparison.
fn check_test_result(
    result: Result<Value>,
    return_type: &ast::Type,
    input: &Value,
    expected: &Value,
    matcher: Option<&parser::TestMatcher>,
) -> Result<String> {
    let returns_result = matches!(return_type, ast::Type::Result(_));

    let actual = match result {
        Ok(val) => {
            if returns_result {
                // Wrap in Ok if not already a result type
                match &val {
                    Value::Ok(_) | Value::Err(_) => val,
                    _ => Value::Ok(Box::new(val)),
                }
            } else {
                val
            }
        }
        Err(err) => {
            // Check failures produce Err with the label
            let err_str = err.to_string();
            Value::Err(err_str.into())
        }
    };

    // If a matcher is present, use it instead of exact comparison
    if let Some(m) = matcher {
        // For string matchers (contains/starts_with), extract the raw string
        // content when the value is a String or Ok(String), avoiding the
        // Display quotes that wrap String values.
        let raw_text = match &actual {
            Value::String(s) => s.as_ref().clone(),
            Value::Ok(inner) => match inner.as_ref() {
                Value::String(s) => s.as_ref().clone(),
                other => format!("{other}"),
            },
            other => format!("{other}"),
        };
        let matched = match m {
            parser::TestMatcher::Contains(s) => {
                raw_text.contains(s.as_str())
            }
            parser::TestMatcher::StartsWith(s) => {
                raw_text.starts_with(s.as_str())
            }
            parser::TestMatcher::AnyOk => {
                matches!(&actual, Value::Ok(_))
            }
            parser::TestMatcher::AnyErr => {
                matches!(&actual, Value::Err(_))
            }
            parser::TestMatcher::ErrContaining(s) => {
                matches!(&actual, Value::Err(msg) if msg.contains(s.as_str()))
            }
        };
        if matched {
            Ok(format!("input={input} => {actual} (matcher: {m:?})"))
        } else {
            bail!("input={input} => {actual}, matcher {m:?} did not match")
        }
    } else {
        // Exact comparison
        if actual.matches(expected) {
            Ok(format!("input={input} => {actual} (expected {expected})"))
        } else {
            bail!("input={input} => {actual}, expected {expected}")
        }
    }
}

/// Check a single `+after` assertion against the current program state.
///
/// `mocks` is provided when the test has access to session-level IO mocks.
fn check_after(
    program: &ast::Program,
    after: &parser::AfterCheck,
    mocks: Option<&[crate::session::IoMock]>,
    http_routes: &[ast::HttpRoute],
) -> Result<()> {
    match after.target.as_str() {
        "routes" => {
            match after.matcher.as_str() {
                "contains" => {
                    let found = http_routes.iter().any(|r| {
                        r.path.contains(&after.value) || r.handler_fn.contains(&after.value)
                    });
                    if !found {
                        bail!("+after routes contains \"{}\": no matching route found (routes: {:?})",
                            after.value,
                            http_routes.iter().map(|r| format!("{} {} -> {}", r.method, r.path, r.handler_fn)).collect::<Vec<_>>());
                    }
                }
                other => bail!("+after routes: unknown matcher `{other}` (expected `contains`)"),
            }
        }
        "modules" => {
            match after.matcher.as_str() {
                "contains" => {
                    let found = program.modules.iter().any(|m| m.name.contains(&after.value));
                    if !found {
                        bail!("+after modules contains \"{}\": no matching module found (modules: {:?})",
                            after.value,
                            program.modules.iter().map(|m| &m.name).collect::<Vec<_>>());
                    }
                }
                other => bail!("+after modules: unknown matcher `{other}` (expected `contains`)"),
            }
        }
        "mocks" => {
            match after.matcher.as_str() {
                "contains" => {
                    if let Some(mocks) = mocks {
                        let found = mocks.iter().any(|m| {
                            m.operation.contains(&after.value)
                                || m.patterns.iter().any(|p| p.contains(&after.value))
                        });
                        if !found {
                            bail!("+after mocks contains \"{}\": no matching mock found", after.value);
                        }
                    } else {
                        bail!("+after mocks: mock state not available in this test context");
                    }
                }
                other => bail!("+after mocks: unknown matcher `{other}` (expected `contains`)"),
            }
        }
        "tasks" => {
            // Tasks are runtime-level and not directly inspectable from the program.
            // For now, this is a no-op with a warning.
            eprintln!("[test] +after tasks check skipped (tasks require live runtime context)");
        }
        other => bail!("+after: unknown target `{other}` (expected routes, modules, mocks, or tasks)"),
    }
    Ok(())
}

/// Public entry point for running a function body with an env.
pub fn eval_function_body_pub(
    program: &ast::Program,
    body: &[ast::Statement],
    env: &mut Env,
) -> Result<Value> {
    eval_function_body(program, body, env)
}

/// Evaluate an AST expression in isolation (no function context).
/// Used for evaluating +shared variable default values like `0`, `""`, `true`.
pub fn eval_expr_standalone(program: &ast::Program, expr: &ast::Expr) -> Result<Value> {
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    eval_ast_expr(program, expr, &mut env)
}

/// Public entry point that also sets the top-level function name for snapshot tracking.
/// Use this for spawned tasks so `?inspect task N` shows the correct function name.
pub fn eval_function_body_named(
    program: &ast::Program,
    function_name: &str,
    body: &[ast::Statement],
    env: &mut Env,
) -> Result<Value> {
    FN_NAME_STACK.with(|s| s.borrow_mut().push(function_name.to_string()));
    let result = eval_function_body(program, body, env);
    FN_NAME_STACK.with(|s| s.borrow_mut().pop());
    result
}

fn eval_function_body(
    program: &ast::Program,
    body: &[ast::Statement],
    env: &mut Env,
) -> Result<Value> {
    for stmt in body {
        // Update task snapshot if this is a tracked coroutine task.
        if let Some(Value::CoroutineHandle(handle)) = env.get_raw("__coroutine_handle") {
            if handle.task_id.is_some() {
                let fn_name = FN_NAME_STACK.with(|s| {
                    s.borrow().last().cloned().unwrap_or_else(|| "<top>".to_string())
                });
                let depth = FN_NAME_STACK.with(|s| s.borrow().len());
                handle.update_snapshot(&fn_name, Some(stmt.id.clone()), depth, env);
            }
        }
        match &stmt.kind {
            ast::StatementKind::Let { name, value, .. } => {
                let val = eval_ast_expr(program, value, env)?;
                env.set(name, val);
            }
            ast::StatementKind::Call { binding, call } => {
                let wants_result = binding
                    .as_ref()
                    .is_some_and(|b| matches!(&b.ty, ast::Type::Result(_)));

                if wants_result {
                    // Binding is Result<T> — catch errors instead of propagating
                    match eval_call(program, call, env) {
                        Ok(val) => {
                            if let Some(binding) = binding {
                                // Wrap success in Ok
                                let wrapped = match &val {
                                    Value::Ok(_) | Value::Err(_) => val,
                                    _ => Value::Ok(Box::new(val)),
                                };
                                env.set(&binding.name, wrapped);
                            }
                        }
                        Err(e) => {
                            if let Some(binding) = binding {
                                env.set(&binding.name, Value::Err(e.to_string().into()));
                            }
                        }
                    }
                } else {
                    // Binding is plain T — errors propagate (like Rust's ?)
                    let val = eval_call(program, call, env)?;
                    // Unwrap Ok() if the function returned a Result but the caller wants plain T
                    let val = match val {
                        Value::Ok(inner) => *inner,
                        other => other,
                    };
                    if let Some(binding) = binding {
                        env.set(&binding.name, val);
                    }
                }
            }
            ast::StatementKind::Check {
                condition, on_fail, ..
            } => {
                let val = eval_ast_expr(program, condition, env)?;
                if !val.is_truthy() {
                    return Err(anyhow!("{on_fail}"));
                }
            }
            ast::StatementKind::Return { value } => {
                return eval_ast_expr(program, value, env);
            }
            ast::StatementKind::Branch {
                condition,
                then_body,
                else_body,
            } => {
                let val = eval_ast_expr(program, condition, env)?;
                let branch = if val.is_truthy() {
                    then_body
                } else {
                    else_body
                };
                env.push_scope();
                let result = eval_function_body(program, branch, env);
                env.pop_scope();
                let result = result?;
                // Only propagate if the branch did an explicit return (non-None)
                if !matches!(result, Value::None) {
                    return Ok(result);
                }
            }
            ast::StatementKind::Each {
                iterator,
                binding,
                body: each_body,
            } => {
                let iter_val = eval_ast_expr(program, iterator, env)?;
                match iter_val {
                    Value::List(items) => {
                        for item in items.iter().cloned() {
                            env.push_scope();
                            env.set(&binding.name, item);
                            let result = eval_function_body(program, each_body, env);
                            env.pop_scope();
                            match result {
                                Ok(val) => {
                                    // Propagate explicit +return from inside +each
                                    if !matches!(val, Value::None) {
                                        return Ok(val);
                                    }
                                }
                                Err(e) => return Err(e),
                            }
                        }
                    }
                    _ => bail!("each: expected list, got {}", iter_val),
                }
            }
            ast::StatementKind::Set { name, value } => {
                let val = eval_ast_expr(program, value, env)?;
                env.set_existing(name, val);
            }
            ast::StatementKind::Match { expr, arms } => {
                let val = eval_ast_expr(program, expr, env)?;
                match &val {
                    Value::Union { variant, payload } => {
                        let mut matched = false;
                        let variant_str = intern::resolve_display(*variant);
                        for arm in arms {
                            if arm.variant == variant_str {
                                env.push_scope();
                                // Exact variant match — bind payload values
                                for (i, binding) in arm.bindings.iter().enumerate() {
                                    if binding != "_" {
                                        if let Some(pval) = payload.get(i) {
                                            env.set(binding, pval.clone());
                                        }
                                    }
                                }
                                // Check nested pattern guards
                                if let Some(ref patterns) = arm.patterns {
                                    if !match_nested_patterns(program, patterns, payload, env)? {
                                        env.pop_scope();
                                        continue; // nested pattern didn't match, try next arm
                                    }
                                }
                                let result = eval_function_body(program, &arm.body, env);
                                env.pop_scope();
                                let result = result?;
                                if !matches!(result, Value::None) {
                                    return Ok(result);
                                }
                                matched = true;
                                break;
                            } else if arm.variant == "_" {
                                env.push_scope();
                                // Wildcard/default case — bind the whole value if there's a binding
                                if arm.bindings.len() == 1 && arm.bindings[0] != "_" {
                                    env.set(&arm.bindings[0], val.clone());
                                }
                                let result = eval_function_body(program, &arm.body, env);
                                env.pop_scope();
                                let result = result?;
                                if !matches!(result, Value::None) {
                                    return Ok(result);
                                }
                                matched = true;
                                break;
                            }
                        }
                        if !matched {
                            bail!("+match: no arm matched variant `{variant_str}`");
                        }
                    }
                    // Treat Ok/Err as union variants for pattern matching
                    Value::Ok(inner) => {
                        let as_union = Value::Union {
                            variant: intern::intern_display("Ok"),
                            payload: vec![inner.as_ref().clone()],
                        };
                        let mut matched = false;
                        for arm in arms {
                            if arm.variant == "Ok" {
                                env.push_scope();
                                if let Some(binding) = arm.bindings.first() {
                                    if binding != "_" { env.set(binding, inner.as_ref().clone()); }
                                }
                                let result = eval_function_body(program, &arm.body, env);
                                env.pop_scope();
                                let result = result?;
                                if !matches!(result, Value::None) { return Ok(result); }
                                matched = true;
                                break;
                            } else if arm.variant == "_" {
                                env.push_scope();
                                if arm.bindings.len() == 1 && arm.bindings[0] != "_" {
                                    env.set(&arm.bindings[0], as_union.clone());
                                }
                                let result = eval_function_body(program, &arm.body, env);
                                env.pop_scope();
                                let result = result?;
                                if !matches!(result, Value::None) { return Ok(result); }
                                matched = true;
                                break;
                            }
                        }
                        if !matched { bail!("+match: no arm matched Ok"); }
                    }
                    Value::Err(msg) => {
                        let as_union = Value::Union {
                            variant: intern::intern_display("Err"),
                            payload: vec![Value::string(msg.clone())],
                        };
                        let mut matched = false;
                        for arm in arms {
                            if arm.variant == "Err" {
                                env.push_scope();
                                if let Some(binding) = arm.bindings.first() {
                                    if binding != "_" { env.set(binding, Value::string(msg.clone())); }
                                }
                                let result = eval_function_body(program, &arm.body, env);
                                env.pop_scope();
                                let result = result?;
                                if !matches!(result, Value::None) { return Ok(result); }
                                matched = true;
                                break;
                            } else if arm.variant == "_" {
                                env.push_scope();
                                if arm.bindings.len() == 1 && arm.bindings[0] != "_" {
                                    env.set(&arm.bindings[0], as_union.clone());
                                }
                                let result = eval_function_body(program, &arm.body, env);
                                env.pop_scope();
                                let result = result?;
                                if !matches!(result, Value::None) { return Ok(result); }
                                matched = true;
                                break;
                            }
                        }
                        if !matched { bail!("+match: no arm matched Err"); }
                    }
                    // Also match on None (Option type)
                    Value::None => {
                        let mut matched = false;
                        for arm in arms {
                            if arm.variant == "None" || arm.variant == "_" {
                                env.push_scope();
                                let result = eval_function_body(program, &arm.body, env);
                                env.pop_scope();
                                let result = result?;
                                if !matches!(result, Value::None) { return Ok(result); }
                                matched = true;
                                break;
                            }
                        }
                        if !matched { bail!("+match: no arm matched None"); }
                    }
                    _ => bail!("+match expects a union, Ok/Err, or None value, got {val}"),
                }
            }
            ast::StatementKind::Await { name, call, .. } => {
                // First check if the callee is a user-defined Adapsis function
                // If so, call it normally (it will use +await internally for IO)
                if program.get_function(&call.callee).is_some() {
                    let val = eval_call(program, call, env)?;
                    let val = match val {
                        Value::Ok(inner) => *inner,
                        other => other,
                    };
                    env.set(name, val);
                } else {
                    // Builtin IO operation
                    let handle = match env.get_raw("__coroutine_handle") {
                        Some(Value::CoroutineHandle(h)) => h.clone(),
                        _ => bail!("+await requires async context — use 'adapsis run-async'"),
                    };
                    let args: Vec<Value> = call
                        .args
                        .iter()
                        .map(|a| eval_ast_expr(program, a, env))
                        .collect::<Result<Vec<_>>>()?;
                    // Ensure query builtins can access the program AST via thread-local.
                    // Check both query_* prefixed names and their aliases (symbols_list,
                    // source_get, callers_get, callees_get, deps_get, routes_list).
                    let needs_program_read = call.callee.starts_with("query_")
                        || matches!(call.callee.as_str(),
                            "symbols_list" | "source_get" | "callers_get"
                            | "callees_get" | "deps_get" | "routes_list"
                            | "route_list" | "test_run");
                    if needs_program_read {
                        set_shared_program(Some(std::sync::Arc::new(program.clone())));
                    }
                    // Ensure mutation builtins can write to the program AST via thread-local.
                    let is_mutation_builtin = matches!(call.callee.as_str(),
                        "mutate" | "fn_remove" | "type_remove" | "module_remove"
                        | "module_create" | "fn_replace" | "move_symbols");
                    if is_mutation_builtin {
                        // Create a mutable wrapper if not already set
                        if get_shared_program_mut().is_none() {
                            set_shared_program_mut(Some(std::sync::Arc::new(
                                std::sync::RwLock::new(program.clone()),
                            )));
                        }
                    }
                    let result = handle.execute_await(&call.callee, &args)?;
                    env.set(name, result);
                }
            }
            ast::StatementKind::Spawn { call, binding } => {
                let handle = match env.get_raw("__coroutine_handle") {
                    Some(Value::CoroutineHandle(h)) => h.clone(),
                    _ => bail!("+spawn requires async context"),
                };
                let args: Vec<Value> = call
                    .args
                    .iter()
                    .map(|a| eval_ast_expr(program, a, env))
                    .collect::<Result<Vec<_>>>()?;
                let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
                let io_tx = handle.io_sender();
                let _ = io_tx.blocking_send(crate::coroutine::IoRequest::Spawn {
                    function_name: call.callee.clone(),
                    args,
                    reply: reply_tx,
                });
                // If there's a binding, wait for the task ID
                if let Some(b) = binding {
                    let task_id = reply_rx
                        .blocking_recv()
                        .map_err(|e| anyhow::anyhow!("spawn reply failed: {e}"))??;
                    env.set(&b.name, Value::TaskHandle(task_id));
                }
            }
            ast::StatementKind::While { condition, body } => {
                let mut iterations = 0;
                loop {
                    let cond = eval_ast_expr(program, condition, env)?;
                    if !cond.is_truthy() {
                        break;
                    }
                    iterations += 1;
                    if iterations > 10000 {
                        bail!("while loop exceeded 10000 iterations — possible infinite loop");
                    }
                    env.push_scope();
                    let result = eval_function_body(program, body, env);
                    env.pop_scope();
                    match result {
                        Ok(val) => {
                            // If the body returned a value (via +return), propagate it
                            if !matches!(val, Value::None) {
                                return Ok(val);
                            }
                        }
                        Err(e) => return Err(e),
                    }
                }
            }
            ast::StatementKind::Yield { value } => {
                let _val = eval_ast_expr(program, value, env)?;
            }
            // Source and event statements — runtime stubs (Phase 1: parse/validate only)
            ast::StatementKind::SourceAdd { .. }
            | ast::StatementKind::SourceRemove { .. }
            | ast::StatementKind::SourceReplace { .. }
            | ast::StatementKind::EventRegister { .. } => {
                // No-op: runtime behavior will be implemented in Phase 2
            }
            ast::StatementKind::EventEmit { value, .. } => {
                let _val = eval_ast_expr(program, value, env)?;
                // No-op: event dispatch will be implemented in Phase 2
            }
        }
    }

    // If no explicit return, return None
    Ok(Value::None)
}

std::thread_local! {
    static CALL_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Stack of function names for the current call chain (used by task snapshots).
    static FN_NAME_STACK: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
    /// Thread-local SharedRuntime so newly-created Env instances automatically
    /// have access to +shared variables without explicit plumbing.
    static SHARED_RUNTIME: std::cell::RefCell<Option<crate::session::SharedRuntime>> = std::cell::RefCell::new(None);
    /// Thread-local SharedMeta so roadmap/plan/mock builtins in coroutine.rs
    /// access the same data as handle_roadmap() in session.rs. No more syncing.
    static SHARED_META: std::cell::RefCell<Option<crate::session::SharedMeta>> = std::cell::RefCell::new(None);
    /// Thread-local SSE broadcast sender so IO builtins can emit API events.
    static SHARED_EVENT_BROADCAST: std::cell::RefCell<Option<tokio::sync::broadcast::Sender<String>>> = std::cell::RefCell::new(None);
    /// Thread-local Program snapshot for query builtins (query_symbols, query_source, etc.)
    /// that need access to the AST from within coroutine IO dispatch.
    static SHARED_PROGRAM: std::cell::RefCell<Option<std::sync::Arc<crate::ast::Program>>> = std::cell::RefCell::new(None);
    /// Thread-local mutable Program reference for mutation builtins (mutate, fn_remove, etc.)
    /// that need write access to the AST from within coroutine IO dispatch.
    static SHARED_PROGRAM_MUT: std::cell::RefCell<Option<std::sync::Arc<std::sync::RwLock<crate::ast::Program>>>> = std::cell::RefCell::new(None);
    /// Thread-local string interner for variable name interning.
    /// Env uses this to convert string names to u32 ids for fast scope lookups.
    static STRING_INTERNER: std::cell::RefCell<StringInterner> = std::cell::RefCell::new(StringInterner::new());
}

/// Set the thread-local SharedRuntime for +shared variable access.
/// Call this before any eval functions that need shared variable support.
pub fn set_shared_runtime(rt: Option<crate::session::SharedRuntime>) {
    SHARED_RUNTIME.with(|s| *s.borrow_mut() = rt);
}

/// Get the thread-local SharedRuntime (if set). Used by coroutine.rs for
/// roadmap builtins that need access to runtime state.
pub fn get_shared_runtime() -> Option<crate::session::SharedRuntime> {
    SHARED_RUNTIME.with(|s| s.borrow().clone())
}

/// Set the thread-local SharedMeta for roadmap/plan/mock builtin access.
/// Call this alongside set_shared_runtime before spawn_blocking eval tasks.
pub fn set_shared_meta(meta: Option<crate::session::SharedMeta>) {
    SHARED_META.with(|s| *s.borrow_mut() = meta);
}

/// Get the thread-local SharedMeta (if set). Used by coroutine.rs for
/// roadmap_add/roadmap_done/plan_set/plan_done/plan_fail builtins.
pub fn get_shared_meta() -> Option<crate::session::SharedMeta> {
    SHARED_META.with(|s| s.borrow().clone())
}

pub fn set_shared_event_broadcast(sender: Option<tokio::sync::broadcast::Sender<String>>) {
    SHARED_EVENT_BROADCAST.with(|s| *s.borrow_mut() = sender);
}

pub fn get_shared_event_broadcast() -> Option<tokio::sync::broadcast::Sender<String>> {
    SHARED_EVENT_BROADCAST.with(|s| s.borrow().clone())
}

/// Set the thread-local Program snapshot for query builtins.
/// Call this alongside set_shared_runtime before spawn_blocking eval tasks.
pub fn set_shared_program(program: Option<std::sync::Arc<crate::ast::Program>>) {
    SHARED_PROGRAM.with(|s| *s.borrow_mut() = program);
}

/// Get the thread-local Program snapshot (if set). Used by coroutine.rs for
/// query builtins (query_symbols, query_source, etc.) that need AST access.
pub fn get_shared_program() -> Option<std::sync::Arc<crate::ast::Program>> {
    SHARED_PROGRAM.with(|s| s.borrow().clone())
}

/// Set the thread-local mutable Program reference for mutation builtins.
/// Call this alongside set_shared_runtime before spawn_blocking eval tasks.
pub fn set_shared_program_mut(program: Option<std::sync::Arc<std::sync::RwLock<crate::ast::Program>>>) {
    SHARED_PROGRAM_MUT.with(|s| *s.borrow_mut() = program);
}

/// Get the thread-local mutable Program reference (if set). Used by coroutine.rs for
/// mutation builtins (mutate, fn_remove, type_remove, module_remove) that need write access.
pub fn get_shared_program_mut() -> Option<std::sync::Arc<std::sync::RwLock<crate::ast::Program>>> {
    SHARED_PROGRAM_MUT.with(|s| s.borrow().clone())
}

/// Create a shared mutable program wrapper for use in `spawn_blocking` contexts.
/// Returns the `Arc<RwLock<Program>>` so the caller can read back mutations after
/// the blocking task completes. Pass `arc.clone()` to `set_shared_program_mut()`
/// inside the `spawn_blocking` closure, then call `read_back_program_mutations()`
/// after the task returns.
pub fn make_shared_program_mut(program: &crate::ast::Program) -> std::sync::Arc<std::sync::RwLock<crate::ast::Program>> {
    std::sync::Arc::new(std::sync::RwLock::new(program.clone()))
}

/// Read back a potentially-mutated program from the shared mutable wrapper.
/// Returns `Some(program)` if the lock can be acquired, `None` on lock error.
/// The caller should compare with the original program to detect mutations.
pub fn read_back_program_mutations(program_mut: &std::sync::Arc<std::sync::RwLock<crate::ast::Program>>) -> Option<crate::ast::Program> {
    program_mut.read().ok().map(|p| p.clone())
}

const MAX_CALL_DEPTH: usize = 256;

fn eval_call(program: &ast::Program, call: &ast::CallExpr, env: &mut Env) -> Result<Value> {
    let depth = CALL_DEPTH.with(|d| {
        let v = d.get();
        d.set(v + 1);
        v
    });
    if depth >= MAX_CALL_DEPTH {
        CALL_DEPTH.with(|d| d.set(0));
        bail!(
            "maximum call depth ({MAX_CALL_DEPTH}) exceeded — possible infinite recursion in `{}`",
            call.callee
        );
    }

    let result = eval_call_inner(program, call, env);
    CALL_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    result
}

/// Evaluate a builtin call with pre-evaluated args. Public for !eval on builtins.
pub fn eval_call_inner_with_args(
    program: &ast::Program,
    callee: &str,
    args: Vec<Value>,
    env: &mut Env,
) -> Result<Value> {
    eval_builtin_or_user(program, callee, args, env)
}

fn eval_call_inner(program: &ast::Program, call: &ast::CallExpr, env: &mut Env) -> Result<Value> {
    let args: Vec<Value> = call
        .args
        .iter()
        .map(|a| eval_ast_expr(program, a, env))
        .collect::<Result<Vec<_>>>()?;

    eval_builtin_or_user(program, &call.callee, args, env)
}

pub fn eval_builtin_or_user(
    program: &ast::Program,
    callee: &str,
    args: Vec<Value>,
    env: &mut Env,
) -> Result<Value> {
    // User-defined union variants take priority over builtins
    // (e.g., user defines Maybe = Some(Int) | None — "Some" should create Union, not Ok)
    if is_union_variant(program, callee) {
        return Ok(Value::Union {
            variant: intern::intern_display(callee),
            payload: args,
        });
    }

    // Check for built-in functions
    match callee {
        "Ok" => {
            if args.len() == 1 {
                Ok(Value::Ok(Box::new(args.into_iter().next().unwrap())))
            } else {
                Ok(Value::Ok(Box::new(Value::None)))
            }
        }
        "Err" => {
            if args.len() == 1 {
                match &args[0] {
                    Value::String(s) => Ok(Value::Err(s.as_ref().clone().into())),
                    other => Ok(Value::Err(format!("{other}").into())),
                }
            } else {
                Ok(Value::Err("unknown".into()))
            }
        }
        "Some" => {
            if args.len() == 1 {
                Ok(Value::Ok(Box::new(args.into_iter().next().unwrap())))
            } else {
                Ok(Value::Ok(Box::new(Value::None)))
            }
        }
        "len" | "length" => {
            if args.len() != 1 {
                bail!("len() expects 1 argument");
            }
            match &args[0] {
                Value::String(s) => Ok(Value::Int(s.len() as i64)),
                Value::List(l) => Ok(Value::Int(l.len() as i64)),
                _ => bail!("len() expects string or list"),
            }
        }
        "error_suggest" | "failure_suggest" => {
            if args.len() != 1 {
                bail!("error_suggest() expects 1 argument");
            }
            let message = match &args[0] {
                Value::String(s) => s.as_ref().clone(),
                other => format!("{other}"),
            };
            let lower = message.to_lowercase();
            let suggestion = if lower.contains("undefined variable") {
                "Check variable spelling. Variables must be declared with +let or +call before use. Function parameters use the exact names from the signature.".to_string()
            } else if lower.contains("expected `,") || lower.contains("expected `}") {
                "Check struct literal syntax. Use {field: value, field2: value2}. Make sure string values with special chars are properly escaped.".to_string()
            } else if lower.contains("out of range") {
                "Statement index is 1-based. Use ?source to check how many statements the function has.".to_string()
            } else if lower.contains("len() expects string or list") {
                "The len() builtin only works on String and List types. Convert your value first or use a different approach.".to_string()
            } else if lower.contains("missing effect") || lower.contains("requires effect") {
                "Add the missing effect to your function signature, e.g. [io,async] or [fail].".to_string()
            } else {
                String::new()
            };
            Ok(Value::string(suggestion))
        }
        "concat" => {
            // Pre-calculate total length to avoid repeated reallocations.
            // For non-String values we format them into a small buffer first
            // so we can measure before the final allocation.
            let mut parts: Vec<std::borrow::Cow<'_, str>> = Vec::with_capacity(args.len());
            let mut total_len = 0usize;
            for arg in &args {
                match arg {
                    Value::String(s) => {
                        total_len += s.len();
                        parts.push(std::borrow::Cow::Borrowed(s.as_ref()));
                    }
                    other => {
                        let formatted = format!("{other}");
                        total_len += formatted.len();
                        parts.push(std::borrow::Cow::Owned(formatted));
                    }
                }
            }
            let mut result = String::with_capacity(total_len);
            for part in &parts {
                result.push_str(part);
            }
            Ok(Value::string(result))
        }
        "to_string" | "str" => {
            if args.len() != 1 {
                bail!("to_string() expects 1 argument");
            }
            Ok(Value::string(format!("{}", args[0])))
        }
        "char_at" => {
            if args.len() != 2 {
                bail!("char_at(s, i) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::String(s), Value::Int(i)) => {
                    let i = *i as usize;
                    if i < s.len() {
                        Ok(Value::string(s[i..i + 1].to_string()))
                    } else {
                        bail!("char_at: index {i} out of bounds (len {})", s.len())
                    }
                }
                _ => bail!("char_at expects (String, Int)"),
            }
        }
        "substring" | "substr" => {
            if args.len() != 3 {
                bail!("substring(s, start, end) expects 3 arguments");
            }
            match (&args[0], &args[1], &args[2]) {
                (Value::String(s), Value::Int(start), Value::Int(end)) => {
                    let start = *start as usize;
                    let end = (*end as usize).min(s.len());
                    if start <= end && start <= s.len() {
                        Ok(Value::string(s[start..end].to_string()))
                    } else {
                        bail!(
                            "substring: invalid range {}..{} (len {})",
                            start,
                            end,
                            s.len()
                        )
                    }
                }
                _ => bail!("substring expects (String, Int, Int)"),
            }
        }
        "starts_with" => {
            if args.len() != 2 {
                bail!("starts_with(s, prefix) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::String(s), Value::String(prefix)) => {
                    Ok(Value::Bool(s.starts_with(prefix.as_ref())))
                }
                _ => bail!("starts_with expects (String, String)"),
            }
        }
        "ends_with" => {
            if args.len() != 2 {
                bail!("ends_with(s, suffix) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::String(s), Value::String(suffix)) => {
                    Ok(Value::Bool(s.ends_with(suffix.as_ref())))
                }
                _ => bail!("ends_with expects (String, String)"),
            }
        }
        "contains" => {
            if args.len() != 2 {
                bail!("contains(s, substr) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::String(s), Value::String(sub)) => Ok(Value::Bool(s.contains(sub.as_ref()))),
                _ => bail!("contains expects (String, String)"),
            }
        }
        "regex_match" => {
            if args.len() != 2 {
                bail!("regex_match(pattern, text) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::String(pattern), Value::String(text)) => match regex::Regex::new(pattern) {
                    Ok(re) => Ok(Value::Bool(re.is_match(text))),
                    Err(_) => Ok(Value::Bool(false)),
                },
                _ => bail!("regex_match expects (String, String)"),
            }
        }
        "regex_replace" => {
            if args.len() != 3 {
                bail!("regex_replace(pattern, replacement, text) expects 3 arguments");
            }
            match (&args[0], &args[1], &args[2]) {
                (Value::String(pattern), Value::String(replacement), Value::String(text)) => {
                    match regex::Regex::new(pattern) {
                        Ok(re) => Ok(Value::string(
                            re.replace_all(text, replacement.as_ref()).into_owned(),
                        )),
                        Err(e) => bail!("regex_replace: invalid pattern '{}': {}", pattern, e),
                    }
                }
                _ => bail!("regex_replace expects (String, String, String)"),
            }
        }
        "index_of" => {
            if args.len() != 2 {
                bail!("index_of(s, substr) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::String(s), Value::String(sub)) => match s.find(sub.as_ref()) {
                    Some(i) => Ok(Value::Int(i as i64)),
                    None => Ok(Value::Int(-1)),
                },
                _ => bail!("index_of expects (String, String)"),
            }
        }
        "split" => {
            if args.len() != 2 {
                bail!("split(s, delim) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::String(s), Value::String(delim)) => {
                    let parts: Vec<Value> = s
                        .split(delim.as_ref())
                        .map(|p| Value::string(p.to_string()))
                        .collect();
                    Ok(Value::list(parts))
                }
                _ => bail!("split expects (String, String)"),
            }
        }
        "trim" => {
            if args.len() != 1 {
                bail!("trim(s) expects 1 argument");
            }
            match &args[0] {
                Value::String(s) => Ok(Value::string(s.trim().to_string())),
                _ => bail!("trim expects String"),
            }
        }
        "json_get" => {
            if args.len() != 2 {
                bail!("json_get(json, key_path) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::String(json_str), Value::String(path)) => {
                    let parsed: serde_json::Value = serde_json::from_str(json_str)
                        .map_err(|e| anyhow!("json_get: invalid JSON: {e}"))?;
                    let mut current = &parsed;
                    for key in path.split('.') {
                        if key.is_empty() {
                            continue;
                        }
                        if let Ok(idx) = key.parse::<usize>() {
                            current = current
                                .get(idx)
                                .ok_or_else(|| anyhow!("json_get: index {idx} not found"))?;
                        } else {
                            current = current
                                .get(key)
                                .ok_or_else(|| anyhow!("json_get: key '{key}' not found"))?;
                        }
                    }
                    // Return the value as a string, stripping quotes from string values
                    match current {
                        serde_json::Value::String(s) => Ok(Value::string(s.clone())),
                        serde_json::Value::Number(n) => {
                            if let Some(i) = n.as_i64() {
                                Ok(Value::Int(i))
                            } else if let Some(f) = n.as_f64() {
                                Ok(Value::Float(f))
                            } else {
                                Ok(Value::string(n.to_string()))
                            }
                        }
                        serde_json::Value::Bool(b) => Ok(Value::Bool(*b)),
                        serde_json::Value::Null => Ok(Value::string("null")),
                        // Arrays and objects are returned as JSON strings
                        other => Ok(Value::string(other.to_string())),
                    }
                }
                _ => bail!("json_get expects (String, String)"),
            }
        }
        "json_array_len" => {
            if args.len() != 1 {
                bail!("json_array_len(json) expects 1 argument");
            }
            match &args[0] {
                Value::String(json_str) => {
                    let parsed: serde_json::Value = serde_json::from_str(json_str)
                        .map_err(|e| anyhow!("json_array_len: invalid JSON: {e}"))?;
                    match &parsed {
                        serde_json::Value::Array(arr) => Ok(Value::Int(arr.len() as i64)),
                        _ => bail!("json_array_len: expected JSON array, got {}", parsed),
                    }
                }
                _ => bail!("json_array_len expects String"),
            }
        }
        "json_escape" => {
            if args.len() != 1 {
                bail!("json_escape(s) expects 1 argument");
            }
            match &args[0] {
                Value::String(s) => {
                    let mut escaped = String::with_capacity(s.len());
                    for ch in s.chars() {
                        match ch {
                            '\\' => escaped.push_str("\\\\"),
                            '"' => escaped.push_str("\\\""),
                            '\n' => escaped.push_str("\\n"),
                            '\r' => escaped.push_str("\\r"),
                            '\t' => escaped.push_str("\\t"),
                            c => escaped.push(c),
                        }
                    }
                    Ok(Value::string(escaped))
                }
                _ => bail!("json_escape expects String"),
            }
        }
        "base64_encode" => {
            if args.len() != 1 {
                bail!("base64_encode(s) expects 1 argument");
            }
            match &args[0] {
                Value::String(s) => {
                    use base64::Engine;
                    let encoded = base64::engine::general_purpose::STANDARD.encode(s.as_bytes());
                    Ok(Value::string(encoded))
                }
                _ => bail!("base64_encode expects String"),
            }
        }
        // List operations
        "list" => {
            // list() creates empty list, list(a, b, c) creates list with items
            Ok(Value::list(args))
        }
        "push" => {
            if args.len() != 2 {
                bail!("push(list, item) expects 2 arguments");
            }
            // Move args out to avoid cloning the list — args is owned Vec<Value>.
            let mut drain = args.into_iter();
            let list_val = drain.next().unwrap();
            let item = drain.next().unwrap();
            match list_val {
                Value::List(mut items) => {
                    Arc::make_mut(&mut items).push(item);
                    Ok(Value::List(items))
                }
                _ => bail!("push expects (List, value)"),
            }
        }
        "get" => {
            if args.len() != 2 {
                bail!("get(list, index) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::List(items), Value::Int(i)) => {
                    let i = *i as usize;
                    if i < items.len() {
                        Ok(items[i].clone())
                    } else {
                        bail!("get: index {i} out of bounds (len {})", items.len())
                    }
                }
                _ => bail!("get expects (List, Int)"),
            }
        }
        "join" => {
            if args.len() != 2 {
                bail!("join(list, delim) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::List(items), Value::String(delim)) => {
                    let parts: Vec<String> = items.iter().map(|v| format!("{v}")).collect();
                    Ok(Value::string(parts.join(delim.as_ref())))
                }
                _ => bail!("join expects (List, String)"),
            }
        }
        "abs" => {
            if args.len() != 1 {
                bail!("abs() expects 1 argument");
            }
            match &args[0] {
                Value::Int(n) => Ok(Value::Int(n.abs())),
                Value::Float(n) => Ok(Value::Float(n.abs())),
                _ => bail!("abs() expects number"),
            }
        }
        "sqrt" => {
            if args.len() != 1 {
                bail!("sqrt() expects 1 argument");
            }
            match &args[0] {
                Value::Int(n) => Ok(Value::Float((*n as f64).sqrt())),
                Value::Float(n) => Ok(Value::Float(n.sqrt())),
                _ => bail!("sqrt() expects number"),
            }
        }
        "pow" => {
            if args.len() != 2 {
                bail!("pow() expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.pow(*b as u32))),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a.powf(*b))),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float((*a as f64).powf(*b))),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a.powi(*b as i32))),
                _ => bail!("pow() expects numbers"),
            }
        }
        "floor" => {
            if args.len() != 1 {
                bail!("floor() expects 1 argument");
            }
            match &args[0] {
                Value::Float(n) => Ok(Value::Int(n.floor() as i64)),
                Value::Int(n) => Ok(Value::Int(*n)),
                _ => bail!("floor() expects number"),
            }
        }
        "to_int" | "parse_int" | "int" => {
            if args.len() != 1 {
                bail!("to_int() expects 1 argument");
            }
            match &args[0] {
                Value::Int(n) => Ok(Value::Int(*n)),
                Value::Float(n) => Ok(Value::Int(*n as i64)),
                Value::Bool(b) => Ok(Value::Int(*b as i64)),
                Value::String(s) => match s.parse::<i64>() {
                    Ok(n) => Ok(Value::Int(n)),
                    Err(_) => bail!("to_int: cannot parse '{s}' as integer"),
                },
                _ => bail!("to_int() expects a value convertible to Int"),
            }
        }
        "digit_value" => {
            // digit_value("5") -> 5, digit_value("a") -> -1
            if args.len() != 1 {
                bail!("digit_value() expects 1 argument");
            }
            match &args[0] {
                Value::String(s) => {
                    if s.len() == 1 {
                        let ch = s.chars().next().unwrap();
                        if ch.is_ascii_digit() {
                            Ok(Value::Int((ch as i64) - ('0' as i64)))
                        } else {
                            Ok(Value::Int(-1))
                        }
                    } else {
                        Ok(Value::Int(-1))
                    }
                }
                _ => bail!("digit_value expects String"),
            }
        }
        "is_digit_char" => {
            if args.len() != 1 {
                bail!("is_digit_char() expects 1 argument");
            }
            match &args[0] {
                Value::String(s) => Ok(Value::Bool(
                    s.len() == 1 && s.chars().next().unwrap().is_ascii_digit(),
                )),
                _ => Ok(Value::Bool(false)),
            }
        }
        // Bitwise operations
        "bit_and" => {
            if args.len() != 2 {
                bail!("bit_and(a, b) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a & b)),
                _ => bail!("bit_and expects (Int, Int)"),
            }
        }
        "bit_or" => {
            if args.len() != 2 {
                bail!("bit_or(a, b) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a | b)),
                _ => bail!("bit_or expects (Int, Int)"),
            }
        }
        "bit_xor" => {
            if args.len() != 2 {
                bail!("bit_xor(a, b) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a ^ b)),
                _ => bail!("bit_xor expects (Int, Int)"),
            }
        }
        "bit_not" => {
            if args.len() != 1 {
                bail!("bit_not(a) expects 1 argument");
            }
            match &args[0] {
                Value::Int(a) => Ok(Value::Int(!a)),
                _ => bail!("bit_not expects Int"),
            }
        }
        "bit_shl" | "shl" => {
            if args.len() != 2 {
                bail!("bit_shl(a, n) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::Int(a), Value::Int(n)) => Ok(Value::Int(a << n)),
                _ => bail!("bit_shl expects (Int, Int)"),
            }
        }
        "bit_shr" | "shr" => {
            if args.len() != 2 {
                bail!("bit_shr(a, n) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::Int(a), Value::Int(n)) => Ok(Value::Int(a >> n)),
                _ => bail!("bit_shr expects (Int, Int)"),
            }
        }
        "left_rotate" | "rotl" => {
            if args.len() != 2 {
                bail!("left_rotate(val, n) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::Int(val), Value::Int(n)) => {
                    // 32-bit left rotation
                    let v = *val as u32;
                    let n = (*n as u32) % 32;
                    Ok(Value::Int(v.rotate_left(n) as i64))
                }
                _ => bail!("left_rotate expects (Int, Int)"),
            }
        }
        "to_hex" => {
            if args.len() != 1 {
                bail!("to_hex(n) expects 1 argument");
            }
            match &args[0] {
                Value::Int(n) => Ok(Value::string(format!("{:08x}", *n as u32))),
                _ => bail!("to_hex expects Int"),
            }
        }
        "char_code" | "ord" => {
            if args.len() != 1 {
                bail!("char_code(ch) expects 1 argument");
            }
            match &args[0] {
                Value::String(s) => {
                    if s.len() == 1 {
                        Ok(Value::Int(s.bytes().next().unwrap() as i64))
                    } else {
                        bail!("char_code expects single character string")
                    }
                }
                _ => bail!("char_code expects String"),
            }
        }
        "from_char_code" | "chr" => {
            if args.len() != 1 {
                bail!("from_char_code(n) expects 1 argument");
            }
            match &args[0] {
                Value::Int(n) => Ok(Value::string(String::from(*n as u8 as char))),
                _ => bail!("from_char_code expects Int"),
            }
        }
        "u32_wrap" => {
            // Wrap to unsigned 32-bit
            if args.len() != 1 {
                bail!("u32_wrap(n) expects 1 argument");
            }
            match &args[0] {
                Value::Int(n) => Ok(Value::Int((*n as u32) as i64)),
                _ => bail!("u32_wrap expects Int"),
            }
        }
        "max" => {
            if args.len() != 2 {
                bail!("max() expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(*a.max(b))),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a.max(*b))),
                _ => bail!("max() expects two numbers of same type"),
            }
        }
        "min" => {
            if args.len() != 2 {
                bail!("min() expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(*a.min(b))),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a.min(*b))),
                _ => bail!("min() expects two numbers of same type"),
            }
        }
        _ => {
            // Try to find the function in the program and call it
            if let Some(func) = program.get_function(callee) {
                let mut call_env = Env::new_with_shared_interner(&program.shared_interner);
                // Inherit shared variable state from caller so nested calls
                // can access +shared vars even without a SharedRuntime.
                call_env.inherit_shared_from(env);
                for (param, arg) in func.params.iter().zip(args) {
                    call_env.set(&param.name, arg);
                }
                // Propagate coroutine handle to called functions
                if let Some(handle) = env.get_raw("__coroutine_handle") {
                    call_env.set("__coroutine_handle", handle.clone());
                }
                // Qualify the function name so shared variable resolution can
                // derive the correct module prefix from FN_NAME_STACK.
                // Without this, intra-module calls like `B()` push bare "B"
                // instead of "Module.B", making shared vars invisible.
                let qualified = program.qualify_function_name(callee);
                FN_NAME_STACK.with(|s| s.borrow_mut().push(qualified));
                let result = eval_function_body(program, &func.body, &mut call_env);
                FN_NAME_STACK.with(|s| s.borrow_mut().pop());
                result
            } else {
                // Check if it's a union variant constructor
                if is_union_variant(program, callee) {
                    return Ok(Value::Union {
                        variant: intern::intern_display(callee),
                        payload: args,
                    });
                }
                // Check if it's an IO builtin — execute via coroutine handle if available
                if crate::builtins::is_io_builtin(callee) {
                    if let Some(Value::CoroutineHandle(handle)) = env.get_raw("__coroutine_handle") {
                        let handle = handle.clone();
                        return handle.execute_await(callee, &args);
                    }
                    bail!(
                        "`{callee}` is an async IO operation, use: +await result:String = {callee}({})",
                        args.iter().enumerate().map(|(i, _)| format!("arg{i}")).collect::<Vec<_>>().join(", ")
                    )
                }
                bail!(
                    "undefined function `{}` (called with {} args)",
                    callee,
                    args.len()
                )
            }
        }
    }
}

fn eval_ast_expr(program: &ast::Program, expr: &ast::Expr, env: &mut Env) -> Result<Value> {
    match expr {
        ast::Expr::Literal(lit) => match lit {
            ast::Literal::Int(v) => Ok(Value::Int(*v)),
            ast::Literal::Float(v) => Ok(Value::Float(*v)),
            ast::Literal::Bool(v) => Ok(Value::Bool(*v)),
            ast::Literal::String(v) => Ok(Value::string(v.clone())),
        },
        ast::Expr::Identifier(name) => {
            match name.as_str() {
                "true" => Ok(Value::Bool(true)),
                "false" => Ok(Value::Bool(false)),
                _ => {
                    // Try variable first
                    if let Ok(val) = env.get(name) {
                        Ok(val)
                    } else if is_union_variant(program, name) {
                        // No-payload union variant
                        Ok(Value::Union {
                            variant: intern::intern_display(name),
                            payload: vec![],
                        })
                    } else {
                        bail!("undefined variable `{name}`")
                    }
                }
            }
        }
        ast::Expr::FieldAccess { base, field } => {
            let base_val = eval_ast_expr(program, base, env)?;
            let field_id = intern::intern_display(field);
            match &base_val {
                Value::Struct(_, fields) => fields
                    .get(&field_id)
                    .cloned()
                    .ok_or_else(|| anyhow!("field `{field}` not found on {base_val}")),
                Value::Ok(inner) => {
                    // Special methods on Result values
                    match field.as_str() {
                        "is_ok" => Ok(Value::Bool(true)),
                        "is_err" => Ok(Value::Bool(false)),
                        "unwrap" => Ok(inner.as_ref().clone()),
                        _ => {
                            // Transparent field access on Ok values
                            match inner.as_ref() {
                                Value::Struct(_, fields) => fields
                                    .get(&field_id)
                                    .cloned()
                                    .ok_or_else(|| anyhow!("field `{field}` not found")),
                                _ => bail!("cannot access field `{field}` on {base_val}"),
                            }
                        }
                    }
                }
                Value::Err(e) => match field.as_str() {
                    "is_ok" => Ok(Value::Bool(false)),
                    "is_err" => Ok(Value::Bool(true)),
                    "error" | "unwrap_err" => Ok(Value::string(e.to_string())),
                    "unwrap" => bail!("unwrap on Err({e})"),
                    _ => bail!("cannot access field `{field}` on Err({e})"),
                },
                // Special methods
                _ => match field.as_str() {
                    "len" => match &base_val {
                        Value::String(s) => Ok(Value::Int(s.len() as i64)),
                        Value::List(l) => Ok(Value::Int(l.len() as i64)),
                        _ => bail!("`.len` not supported on {base_val}"),
                    },
                    "is_ok" => match &base_val {
                        Value::Ok(_) => Ok(Value::Bool(true)),
                        Value::Err(_) => Ok(Value::Bool(false)),
                        _ => Ok(Value::Bool(true)),
                    },
                    "is_err" => match &base_val {
                        Value::Err(_) => Ok(Value::Bool(true)),
                        _ => Ok(Value::Bool(false)),
                    },
                    "is_empty" => match &base_val {
                        Value::String(s) => Ok(Value::Bool(s.is_empty())),
                        Value::List(l) => Ok(Value::Bool(l.is_empty())),
                        _ => bail!("`.is_empty` not supported on {base_val}"),
                    },
                    "unwrap" => match &base_val {
                        Value::Ok(v) => Ok(v.as_ref().clone()),
                        Value::Err(e) => bail!("unwrap on Err({e})"),
                        other => Ok(other.clone()),
                    },
                    _ => bail!("cannot access field `{field}` on {base_val}"),
                },
            }
        }
        ast::Expr::Call(call) => eval_call(program, call, env),
        ast::Expr::Binary { left, op, right } => {
            // Short-circuit AND and OR
            if matches!(op, ast::BinaryOp::And) {
                let lhs = eval_ast_expr(program, left, env)?;
                if !lhs.is_truthy() {
                    return Ok(Value::Bool(false));
                }
                let rhs = eval_ast_expr(program, right, env)?;
                return Ok(Value::Bool(rhs.is_truthy()));
            }
            if matches!(op, ast::BinaryOp::Or) {
                let lhs = eval_ast_expr(program, left, env)?;
                if lhs.is_truthy() {
                    return Ok(Value::Bool(true));
                }
                let rhs = eval_ast_expr(program, right, env)?;
                return Ok(Value::Bool(rhs.is_truthy()));
            }
            let lhs = eval_ast_expr(program, left, env)?;
            let rhs = eval_ast_expr(program, right, env)?;
            eval_binary_op(&lhs, op, &rhs)
        }
        ast::Expr::Unary { op, expr } => {
            let val = eval_ast_expr(program, expr, env)?;
            match op {
                ast::UnaryOp::Not => Ok(Value::Bool(!val.is_truthy())),
                ast::UnaryOp::Neg => match val {
                    Value::Int(n) => Ok(Value::Int(-n)),
                    Value::Float(n) => Ok(Value::Float(-n)),
                    _ => bail!("cannot negate {val}"),
                },
            }
        }
        ast::Expr::StructInit { ty, fields } => {
            let mut field_map: HashMap<InternedId, Value> = HashMap::new();
            for f in fields {
                let val = eval_ast_expr(program, &f.value, env)?;
                field_map.insert(intern::intern_display(&f.name), val);
            }
            Ok(Value::strct_interned(intern::intern_display(ty), field_map))
        }
    }
}

fn eval_binary_op(lhs: &Value, op: &ast::BinaryOp, rhs: &Value) -> Result<Value> {
    match op {
        ast::BinaryOp::Add => match (lhs, rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 + b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a + *b as f64)),
            _ => bail!("cannot add {lhs} + {rhs}"),
        },
        ast::BinaryOp::Sub => match (lhs, rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 - b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a - *b as f64)),
            _ => bail!("cannot subtract {lhs} - {rhs}"),
        },
        ast::BinaryOp::Mul => match (lhs, rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 * b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a * *b as f64)),
            _ => bail!("cannot multiply {lhs} * {rhs}"),
        },
        ast::BinaryOp::Div => match (lhs, rhs) {
            (Value::Int(a), Value::Int(b)) => {
                if *b == 0 {
                    bail!("division by zero")
                }
                Ok(Value::Int(a / b))
            }
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 / b)),
            (Value::Float(a), Value::Int(b)) => {
                if *b == 0 {
                    bail!("division by zero")
                }
                Ok(Value::Float(a / *b as f64))
            }
            _ => bail!("cannot divide {lhs} / {rhs}"),
        },
        ast::BinaryOp::Mod => match (lhs, rhs) {
            (Value::Int(a), Value::Int(b)) => {
                if *b == 0 {
                    bail!("modulo by zero")
                }
                Ok(Value::Int(a % b))
            }
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a % b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 % b)),
            (Value::Float(a), Value::Int(b)) => {
                if *b == 0 {
                    bail!("modulo by zero")
                }
                Ok(Value::Float(a % *b as f64))
            }
            _ => bail!("cannot modulo {lhs} % {rhs}"),
        },
        ast::BinaryOp::And => Ok(Value::Bool(lhs.is_truthy() && rhs.is_truthy())),
        ast::BinaryOp::Or => Ok(Value::Bool(lhs.is_truthy() || rhs.is_truthy())),
        ast::BinaryOp::Equal => Ok(Value::Bool(lhs.matches(rhs))),
        ast::BinaryOp::NotEqual => Ok(Value::Bool(!lhs.matches(rhs))),
        ast::BinaryOp::GreaterThan => match (lhs, rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a > b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a > b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Bool((*a as f64) > *b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Bool(*a > *b as f64)),
            _ => bail!("cannot compare {lhs} > {rhs}"),
        },
        ast::BinaryOp::LessThan => match (lhs, rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a < b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a < b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Bool((*a as f64) < *b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Bool(*a < *b as f64)),
            _ => bail!("cannot compare {lhs} < {rhs}"),
        },
        ast::BinaryOp::GreaterThanOrEqual => match (lhs, rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a >= b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a >= b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Bool((*a as f64) >= *b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Bool(*a >= *b as f64)),
            _ => bail!("cannot compare {lhs} >= {rhs}"),
        },
        ast::BinaryOp::LessThanOrEqual => match (lhs, rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a <= b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a <= b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Bool((*a as f64) <= *b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Bool(*a <= *b as f64)),
            _ => bail!("cannot compare {lhs} <= {rhs}"),
        },
    }
}

/// Evaluate a parser::Expr directly (for test case inputs/expected values).
/// Evaluate a parser expression with access to the program (can call user functions).
pub fn eval_parser_expr_with_program(expr: &parser::Expr, program: &ast::Program) -> Result<Value> {
    match expr {
        parser::Expr::Ident(name) => {
            // Check if this is a user-defined union variant first
            if is_union_variant(program, name) {
                return Ok(Value::Union {
                    variant: intern::intern_display(name),
                    payload: vec![],
                });
            }
            // Check if this is a zero-arg user function (e.g. initial_context_state)
            if let Some(func) = program.get_function(name) {
                if func.params.is_empty() {
                    // Reject functions with side effects in test expressions
                    let has_side_effects = func.effects.iter().any(|e| {
                        matches!(e, ast::Effect::Io | ast::Effect::Async | ast::Effect::Mut | ast::Effect::Unsafe)
                    });
                    if has_side_effects {
                        bail!(
                            "cannot call `{name}` in test expression: function has side effects {:?} — \
                             use !mock and an async test wrapper instead",
                            func.effects
                        );
                    }
                    let mut env = Env::new_with_shared_interner(&program.shared_interner);
                    env.populate_shared_from_program(program);
                    let qualified = program.qualify_function_name(name);
                    FN_NAME_STACK.with(|s| s.borrow_mut().push(qualified));
                    let result = eval_function_body(program, &func.body, &mut env);
                    FN_NAME_STACK.with(|s| s.borrow_mut().pop());
                    return result;
                }
            }
            // Fall through to standalone handling
            eval_parser_expr_standalone(expr)
        }
        parser::Expr::Call { callee, args } => {
            let name = parser_callee_name(callee);
            // Check if it's a user-defined union constructor
            if is_union_variant(program, &name) {
                let payload = args
                    .iter()
                    .map(|a| eval_parser_expr_with_program(a, program))
                    .collect::<Result<Vec<_>>>()?;
                return Ok(Value::Union {
                    variant: intern::intern_display(&name),
                    payload,
                });
            }
            // Try as user function
            if let Some(func) = program.get_function(&name) {
                // Reject functions with side effects in test expressions
                let has_side_effects = func.effects.iter().any(|e| {
                    matches!(e, ast::Effect::Io | ast::Effect::Async | ast::Effect::Mut | ast::Effect::Unsafe)
                });
                if has_side_effects {
                    bail!(
                        "cannot call `{name}` in test expression: function has side effects {:?} — \
                         use !mock and an async test wrapper instead",
                        func.effects
                    );
                }
                let eval_args: Vec<Value> = args
                    .iter()
                    .map(|a| eval_parser_expr_with_program(a, program))
                    .collect::<Result<Vec<_>>>()?;
                let mut env = Env::new_with_shared_interner(&program.shared_interner);
                env.populate_shared_from_program(program);
                for (param, arg) in func.params.iter().zip(eval_args) {
                    env.set(&param.name, arg);
                }
                let qualified = program.qualify_function_name(&name);
                FN_NAME_STACK.with(|s| s.borrow_mut().push(qualified));
                let result = eval_function_body(program, &func.body, &mut env);
                FN_NAME_STACK.with(|s| s.borrow_mut().pop());
                return result;
            }
            // Try as builtin function (concat, len, to_string, etc.)
            // Skip Ok/Err/Some/None — these are handled by eval_parser_expr_standalone
            // with special parser-level semantics (e.g. Err(bare_ident) as error labels).
            if crate::builtins::is_builtin(&name)
                && !matches!(name.as_str(), "Ok" | "Err" | "Some" | "None")
            {
                let eval_args: Vec<Value> = args
                    .iter()
                    .map(|a| eval_parser_expr_with_program(a, program))
                    .collect::<Result<Vec<_>>>()?;
                let mut env = Env::new_with_shared_interner(&program.shared_interner);
                return eval_builtin_or_user(program, &name, eval_args, &mut env);
            }
            // Fall through to standalone (handles union constructors, Ok, Err)
            eval_parser_expr_standalone(expr)
        }
        // StructLiteral needs program access so field values can call user functions
        parser::Expr::StructLiteral(fields) => {
            let empty_id = intern::intern_display("");
            let mut field_map: HashMap<InternedId, Value> = HashMap::new();
            for f in fields {
                let val = eval_parser_expr_with_program(&f.value, program)?;
                field_map.insert(intern::intern_display(&f.name), val);
            }
            Ok(Value::strct_interned(empty_id, field_map))
        }
        // Unary expressions need program access for their inner expression
        parser::Expr::Unary { op, expr: inner } => {
            let val = eval_parser_expr_with_program(inner, program)?;
            match op {
                parser::UnaryOp::Not => Ok(Value::Bool(!val.is_truthy())),
                parser::UnaryOp::Neg => match val {
                    Value::Int(n) => Ok(Value::Int(-n)),
                    Value::Float(n) => Ok(Value::Float(-n)),
                    _ => bail!("cannot negate {val}"),
                },
            }
        }
        // Binary expressions need program access for both sides
        parser::Expr::Binary { left, op, right } => {
            let l = eval_parser_expr_with_program(left, program)?;
            let r = eval_parser_expr_with_program(right, program)?;
            let ast_op = match op {
                parser::BinaryOp::Add => ast::BinaryOp::Add,
                parser::BinaryOp::Sub => ast::BinaryOp::Sub,
                parser::BinaryOp::Mul => ast::BinaryOp::Mul,
                parser::BinaryOp::Div => ast::BinaryOp::Div,
                parser::BinaryOp::Mod => ast::BinaryOp::Mod,
                parser::BinaryOp::Eq => ast::BinaryOp::Equal,
                parser::BinaryOp::Neq => ast::BinaryOp::NotEqual,
                parser::BinaryOp::Gt => ast::BinaryOp::GreaterThan,
                parser::BinaryOp::Lt => ast::BinaryOp::LessThan,
                parser::BinaryOp::Gte => ast::BinaryOp::GreaterThanOrEqual,
                parser::BinaryOp::Lte => ast::BinaryOp::LessThanOrEqual,
                parser::BinaryOp::And => ast::BinaryOp::And,
                parser::BinaryOp::Or => ast::BinaryOp::Or,
            };
            eval_binary_op(&l, &ast_op, &r)
        }
        // Everything else delegates to standalone
        _ => eval_parser_expr_standalone(expr),
    }
}

/// Like `eval_parser_expr_with_program`, but threads an environment through
/// so that `__coroutine_handle` is available to IO builtin calls.
/// Used by `eval_inline_expr_with_io` for `!eval shell_exec(...)` etc.
fn eval_parser_expr_with_env(
    expr: &parser::Expr,
    program: &ast::Program,
    env: &mut Env,
) -> Result<Value> {
    match expr {
        parser::Expr::Ident(name) => {
            // Check env first (for variables bound by the caller)
            if let Ok(val) = env.get(name) {
                return Ok(val);
            }
            if is_union_variant(program, name) {
                return Ok(Value::Union {
                    variant: intern::intern_display(name),
                    payload: vec![],
                });
            }
            if let Some(func) = program.get_function(name) {
                if func.params.is_empty() {
                    let mut call_env = Env::new_with_shared_interner(&program.shared_interner);
                    call_env.inherit_shared_from(env);
                    if let Some(handle) = env.get_raw("__coroutine_handle") {
                        call_env.set("__coroutine_handle", handle.clone());
                    }
                    let qualified = program.qualify_function_name(name);
                    FN_NAME_STACK.with(|s| s.borrow_mut().push(qualified));
                    let result = eval_function_body(program, &func.body, &mut call_env);
                    FN_NAME_STACK.with(|s| s.borrow_mut().pop());
                    return result;
                }
            }
            eval_parser_expr_standalone(expr)
        }
        parser::Expr::Call { callee, args } => {
            let name = parser_callee_name(callee);
            if is_union_variant(program, &name) {
                let payload = args
                    .iter()
                    .map(|a| eval_parser_expr_with_env(a, program, env))
                    .collect::<Result<Vec<_>>>()?;
                return Ok(Value::Union {
                    variant: intern::intern_display(&name),
                    payload,
                });
            }
            if let Some(func) = program.get_function(&name) {
                let eval_args: Vec<Value> = args
                    .iter()
                    .map(|a| eval_parser_expr_with_env(a, program, env))
                    .collect::<Result<Vec<_>>>()?;
                let mut call_env = Env::new_with_shared_interner(&program.shared_interner);
                call_env.inherit_shared_from(env);
                for (param, arg) in func.params.iter().zip(eval_args) {
                    call_env.set(&param.name, arg);
                }
                if let Some(handle) = env.get_raw("__coroutine_handle") {
                    call_env.set("__coroutine_handle", handle.clone());
                }
                let qualified = program.qualify_function_name(&name);
                FN_NAME_STACK.with(|s| s.borrow_mut().push(qualified));
                let result = eval_function_body(program, &func.body, &mut call_env);
                FN_NAME_STACK.with(|s| s.borrow_mut().pop());
                return result;
            }
            if crate::builtins::is_builtin(&name)
                && !matches!(name.as_str(), "Ok" | "Err" | "Some" | "None")
            {
                let eval_args: Vec<Value> = args
                    .iter()
                    .map(|a| eval_parser_expr_with_env(a, program, env))
                    .collect::<Result<Vec<_>>>()?;
                return eval_builtin_or_user(program, &name, eval_args, env);
            }
            eval_parser_expr_standalone(expr)
        }
        parser::Expr::StructLiteral(fields) => {
            let empty_id = intern::intern_display("");
            let mut field_map: HashMap<InternedId, Value> = HashMap::new();
            for f in fields {
                let val = eval_parser_expr_with_env(&f.value, program, env)?;
                field_map.insert(intern::intern_display(&f.name), val);
            }
            Ok(Value::strct_interned(empty_id, field_map))
        }
        parser::Expr::Unary { op, expr: inner } => {
            let val = eval_parser_expr_with_env(inner, program, env)?;
            match op {
                parser::UnaryOp::Not => Ok(Value::Bool(!val.is_truthy())),
                parser::UnaryOp::Neg => match val {
                    Value::Int(n) => Ok(Value::Int(-n)),
                    Value::Float(n) => Ok(Value::Float(-n)),
                    _ => bail!("cannot negate {val}"),
                },
            }
        }
        parser::Expr::Binary { left, op, right } => {
            let l = eval_parser_expr_with_env(left, program, env)?;
            let r = eval_parser_expr_with_env(right, program, env)?;
            let ast_op = match op {
                parser::BinaryOp::Add => ast::BinaryOp::Add,
                parser::BinaryOp::Sub => ast::BinaryOp::Sub,
                parser::BinaryOp::Mul => ast::BinaryOp::Mul,
                parser::BinaryOp::Div => ast::BinaryOp::Div,
                parser::BinaryOp::Mod => ast::BinaryOp::Mod,
                parser::BinaryOp::Eq => ast::BinaryOp::Equal,
                parser::BinaryOp::Neq => ast::BinaryOp::NotEqual,
                parser::BinaryOp::Gt => ast::BinaryOp::GreaterThan,
                parser::BinaryOp::Lt => ast::BinaryOp::LessThan,
                parser::BinaryOp::Gte => ast::BinaryOp::GreaterThanOrEqual,
                parser::BinaryOp::Lte => ast::BinaryOp::LessThanOrEqual,
                parser::BinaryOp::And => ast::BinaryOp::And,
                parser::BinaryOp::Or => ast::BinaryOp::Or,
            };
            eval_binary_op(&l, &ast_op, &r)
        }
        _ => eval_parser_expr_standalone(expr),
    }
}

pub fn eval_parser_expr_standalone(expr: &parser::Expr) -> Result<Value> {
    match expr {
        parser::Expr::Int(v) => Ok(Value::Int(*v)),
        parser::Expr::Float(v) => Ok(Value::Float(*v)),
        parser::Expr::Bool(v) => Ok(Value::Bool(*v)),
        parser::Expr::String(v) => Ok(Value::string(v.clone())),
        parser::Expr::Ident(name) => {
            match name.as_str() {
                "true" => Ok(Value::Bool(true)),
                "false" => Ok(Value::Bool(false)),
                _ => {
                    // Uppercase names: union variants (including user-defined None, Some, Ok)
                    // Lowercase names: error labels for Err matching
                    if name.chars().next().is_some_and(|c| c.is_uppercase()) {
                        // Special cases for Result/Option builtins (used in test expectations)
                        match name.as_str() {
                            "Ok" => Ok(Value::Ok(Box::new(Value::None))),
                            "None" => Ok(Value::Union {
                                variant: intern::intern_display("None"),
                                payload: vec![],
                            }),
                            _ => Ok(Value::Union {
                                variant: intern::intern_display(name),
                                payload: vec![],
                            }),
                        }
                    } else {
                        Ok(Value::Err(name.clone()))
                    }
                }
            }
        }
        parser::Expr::StructLiteral(fields) => {
            let empty_id = intern::intern_display("");
            let mut field_map: HashMap<InternedId, Value> = HashMap::new();
            for f in fields {
                let val = eval_parser_expr_standalone(&f.value)?;
                field_map.insert(intern::intern_display(&f.name), val);
            }
            Ok(Value::strct_interned(empty_id, field_map))
        }
        parser::Expr::Call { callee, args } => {
            // Handle Ok(...) and Err(...) constructors
            let name = parser_callee_name(callee);
            match name.as_str() {
                "Ok" => {
                    if args.len() == 1 {
                        let inner = eval_parser_expr_standalone(&args[0])?;
                        Ok(Value::Ok(Box::new(inner)))
                    } else {
                        Ok(Value::Ok(Box::new(Value::None)))
                    }
                }
                "Err" => {
                    if args.len() == 1 {
                        match &args[0] {
                            parser::Expr::Ident(label) => Ok(Value::Err(label.clone())),
                            other => {
                                let val = eval_parser_expr_standalone(other)?;
                                Ok(Value::Err(format!("{val}").into()))
                            }
                        }
                    } else {
                        Ok(Value::Err("unknown".into()))
                    }
                }
                "list" => {
                    let items = args
                        .iter()
                        .map(eval_parser_expr_standalone)
                        .collect::<Result<Vec<_>>>()?;
                    Ok(Value::list(items))
                }
                _ => {
                    // Treat as union variant constructor
                    let payload = args
                        .iter()
                        .map(eval_parser_expr_standalone)
                        .collect::<Result<Vec<_>>>()?;
                    Ok(Value::Union {
                        variant: intern::intern_display(&name),
                        payload,
                    })
                }
            }
        }
        parser::Expr::FieldAccess { base, field } => {
            // For test expectations like result.name — just create a path identifier
            let base_name = parser_callee_name(base);
            Ok(Value::Err(format!("{base_name}.{field}").into()))
        }
        parser::Expr::Unary { op, expr } => {
            let val = eval_parser_expr_standalone(expr)?;
            match op {
                parser::UnaryOp::Not => Ok(Value::Bool(!val.is_truthy())),
                parser::UnaryOp::Neg => match val {
                    Value::Int(n) => Ok(Value::Int(-n)),
                    Value::Float(n) => Ok(Value::Float(-n)),
                    _ => bail!("cannot negate {val}"),
                },
            }
        }
        _ => bail!("cannot evaluate expression {:?} in test case", expr),
    }
}

fn parser_callee_name(expr: &parser::Expr) -> String {
    match expr {
        parser::Expr::Ident(name) => name.clone(),
        parser::Expr::FieldAccess { base, field } => {
            format!("{}.{}", parser_callee_name(base), field)
        }
        _ => format!("{:?}", expr),
    }
}

/// A single step in an execution trace.
#[derive(Debug, Clone)]
pub struct TraceStep {
    pub stmt_id: String,
    pub description: String,
    pub result: String,
    pub status: TraceStatus,
}

#[derive(Debug, Clone)]
pub enum TraceStatus {
    Pass,
    Fail,
    Return,
}

impl std::fmt::Display for TraceStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let marker = match self.status {
            TraceStatus::Pass => "pass",
            TraceStatus::Fail => "FAIL",
            TraceStatus::Return => "return",
        };
        write!(
            f,
            "{}: {} | {} | {}",
            self.stmt_id, self.description, self.result, marker
        )
    }
}

/// Execute a function with tracing, returning each step.
pub fn trace_function(
    program: &ast::Program,
    function_name: &str,
    input: &parser::Expr,
) -> Result<Vec<TraceStep>> {
    let func = program
        .get_function(function_name)
        .ok_or_else(|| anyhow!("function `{function_name}` not found{}", crate::eval::suggest_similar(program, function_name)))?;

    let input_val = eval_parser_expr_standalone(input)?;
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    bind_input_to_params(program, func, &input_val, &mut env);

    let mut steps = vec![];
    trace_body(program, &func.body, &mut env, &mut steps);
    Ok(steps)
}

fn trace_body(
    program: &ast::Program,
    body: &[ast::Statement],
    env: &mut Env,
    steps: &mut Vec<TraceStep>,
) -> Option<Value> {
    for stmt in body {
        match &stmt.kind {
            ast::StatementKind::Let { name, value, ty } => {
                match eval_ast_expr(program, value, env) {
                    Ok(val) => {
                        steps.push(TraceStep {
                            stmt_id: stmt.id.clone(),
                            description: format!("let {name}:{ty:?}"),
                            result: format!("{val}"),
                            status: TraceStatus::Pass,
                        });
                        env.set(name, val);
                    }
                    Err(e) => {
                        steps.push(TraceStep {
                            stmt_id: stmt.id.clone(),
                            description: format!("let {name}:{ty:?}"),
                            result: format!("ERROR: {e}"),
                            status: TraceStatus::Fail,
                        });
                        return Some(Value::Err(e.to_string().into()));
                    }
                }
            }
            ast::StatementKind::Call { binding, call } => match eval_call(program, call, env) {
                Ok(val) => {
                    let desc = if let Some(b) = binding {
                        format!("call {}:{:?} = {}()", b.name, b.ty, call.callee)
                    } else {
                        format!("call {}()", call.callee)
                    };
                    steps.push(TraceStep {
                        stmt_id: stmt.id.clone(),
                        description: desc,
                        result: format!("{val}"),
                        status: TraceStatus::Pass,
                    });
                    if let Some(b) = binding {
                        env.set(&b.name, val);
                    }
                }
                Err(e) => {
                    steps.push(TraceStep {
                        stmt_id: stmt.id.clone(),
                        description: format!("call {}()", call.callee),
                        result: format!("ERROR: {e}"),
                        status: TraceStatus::Fail,
                    });
                        return Some(Value::Err(e.to_string().into()));
                }
            },
            ast::StatementKind::Check {
                label,
                condition,
                on_fail,
            } => match eval_ast_expr(program, condition, env) {
                Ok(val) => {
                    let is_true = val.is_truthy();
                    steps.push(TraceStep {
                        stmt_id: stmt.id.clone(),
                        description: format!("check {label}"),
                        result: format!("{val} = {is_true}"),
                        status: if is_true {
                            TraceStatus::Pass
                        } else {
                            TraceStatus::Fail
                        },
                    });
                    if !is_true {
                        return Some(Value::Err(on_fail.clone()));
                    }
                }
                Err(e) => {
                    steps.push(TraceStep {
                        stmt_id: stmt.id.clone(),
                        description: format!("check {label}"),
                        result: format!("ERROR: {e}"),
                        status: TraceStatus::Fail,
                    });
                        return Some(Value::Err(e.to_string().into()));
                }
            },
            ast::StatementKind::Return { value } => match eval_ast_expr(program, value, env) {
                Ok(val) => {
                    steps.push(TraceStep {
                        stmt_id: stmt.id.clone(),
                        description: "return".to_string(),
                        result: format!("{val}"),
                        status: TraceStatus::Return,
                    });
                    return Some(val);
                }
                Err(e) => {
                    steps.push(TraceStep {
                        stmt_id: stmt.id.clone(),
                        description: "return".to_string(),
                        result: format!("ERROR: {e}"),
                        status: TraceStatus::Fail,
                    });
                    return Some(Value::Err(e.to_string().into()));
                }
            },
            _ => {
                // Branch, Each, Yield — simplified tracing
                steps.push(TraceStep {
                    stmt_id: stmt.id.clone(),
                    description: format!("{:?}", std::mem::discriminant(&stmt.kind)),
                    result: "...".to_string(),
                    status: TraceStatus::Pass,
                });
            }
        }
    }
    None
}

/// Try to execute a function via the bytecode VM.
/// Returns `Some(Ok(val))` on success, `None` if the VM can't handle it
/// (compilation error or async suspension), allowing the tree-walker to
/// take over as fallback. Optionally accepts a CoroutineHandle for async IO.
pub fn try_vm_eval(
    func: &ast::FunctionDecl,
    args: &[Value],
    program: &ast::Program,
) -> Option<Result<Value>> {
    let compiled = match vm::compile_function(func, program) {
        Ok(c) => c,
        Err(_) => return None, // VM can't compile → fall back
    };
    // Use execute_with_io — for sync callers, IO operations cause fallback
    match vm::execute_with_io(&compiled, args.to_vec(), program, &|op, _| {
        bail!("VM: async IO ({op}) not available in sync context")
    }) {
        Ok(val) => Some(Ok(val)),
        Err(_) => None, // any error → fall back
    }
}

/// Try to execute a function via the bytecode VM with async IO support.
/// The io_handler performs IO operations when the VM suspends.
pub fn try_vm_eval_async(
    func: &ast::FunctionDecl,
    args: &[Value],
    program: &ast::Program,
    io_handler: &dyn Fn(&str, &[Value]) -> Result<Value>,
) -> Option<Result<Value>> {
    let compiled = match vm::compile_function(func, program) {
        Ok(c) => c,
        Err(_) => return None,
    };
    match vm::execute_with_io(&compiled, args.to_vec(), program, io_handler) {
        Ok(val) => Some(Ok(val)),
        Err(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parser, session::IoMock, validator};

    /// Helper: parse Adapsis source and build a program from it.
    fn build_program(source: &str) -> ast::Program {
        let ops = parser::parse(source).expect("parse failed");
        let mut program = ast::Program::default();
        for op in &ops {
            match op {
                parser::Operation::Test(_) | parser::Operation::Eval(_) => {}
                _ => {
                    validator::apply_and_validate(&mut program, op)
                        .expect("validation failed");
                }
            }
        }
        program.rebuild_function_index();
        program
    }

    /// Helper: extract test cases from parsed source.
    fn extract_test_cases(source: &str) -> Vec<(String, parser::TestCase)> {
        let ops = parser::parse(source).expect("parse failed");
        let mut cases = Vec::new();
        for op in &ops {
            if let parser::Operation::Test(test) = op {
                for case in &test.cases {
                    cases.push((test.function_name.clone(), case.clone()));
                }
            }
            // Also extract tests embedded inside module bodies
            if let parser::Operation::Module(m) = op {
                for body_op in &m.body {
                    if let parser::Operation::Test(test) = body_op {
                        for case in &test.cases {
                            cases.push((test.function_name.clone(), case.clone()));
                        }
                    }
                }
            }
        }
        cases
    }

    /// Helper: extract mocks from parsed source.
    fn extract_mocks(source: &str) -> Vec<IoMock> {
        let ops = parser::parse(source).expect("parse failed");
        let mut mocks = Vec::new();
        for op in ops {
            if let parser::Operation::Mock { operation, patterns, response } = op {
                mocks.push(IoMock { operation, patterns, response });
            }
        }
        mocks
    }

    // ── Sync function tests ───────────────────────────────────────────

    #[test]
    fn test_sync_function_passes() {
        let source = "\
+fn add (a:Int, b:Int)->Int
  +let sum:Int = a + b
  +return sum

!test add
  +with a=2 b=3 -> expect 5
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);

        let (fn_name, case) = &cases[0];
        let result = eval_test_case(&program, fn_name, case);
        assert!(result.is_ok(), "sync test should pass: {:?}", result);
        assert!(result.unwrap().contains("expected 5"));
    }

    #[test]
    fn test_sync_function_fails_on_wrong_expected() {
        let source = "\
+fn add (a:Int, b:Int)->Int
  +let sum:Int = a + b
  +return sum

!test add
  +with a=2 b=3 -> expect 99
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case(&program, fn_name, case);
        assert!(result.is_err(), "should fail when expected is wrong");
    }

    // ── Async function tests (mock-only path) ────────────────────────

    #[test]
    fn test_async_function_with_mock_http_get() {
        let source = "\
+fn fetch_data (url:String)->String [io,async]
  +await resp:String = http_get(url)
  +return resp
";
        let program = build_program(source);

        let test_source = "\
!test fetch_data
  +with url=\"https://example.com\" -> expect \"hello\"
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];

        let mocks = vec![IoMock {
            operation: "http_get".to_string(),
            patterns: vec!["example.com".to_string()],
            response: "hello".to_string(),
        }];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
        assert!(result.is_ok(), "async test with mock should pass: {:?}", result);
    }

    #[test]
    fn test_async_function_without_mock_errors_on_unmocked_io() {
        // Tests always use mock-only handles to prevent deadlocks from
        // self-referential HTTP calls.  Unmocked IO should fail, not execute.
        let source = "\
+fn delayed_value ()->String [io,async]
  +await _:String = sleep(1)
  +return \"done\"
";
        let program = build_program(source);

        let test_source = "\
!test delayed_value
  +with -> expect \"done\"
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];

        // No mocks — should fail with "no mock for sleep" instead of running real IO
        let result = eval_test_case(&program, fn_name, case);
        assert!(result.is_err(), "async test without mocks should error: {:?}", result);
        let err = result.unwrap_err().to_string();
        assert!(err.contains("no mock"), "error should mention missing mock: {err}");
    }

    #[test]
    fn test_async_function_with_io_effect_gets_handle() {
        let source = "\
+fn fetch_data (url:String)->String [io,async]
  +await resp:String = http_get(url)
  +return resp
";
        let program = build_program(source);

        let test_source = "\
!test fetch_data
  +with url=\"https://example.com\" -> expect \"world\"
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];

        let mocks = vec![IoMock {
            operation: "http_get".to_string(),
            patterns: vec!["example.com".to_string()],
            response: "world".to_string(),
        }];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
        assert!(result.is_ok(), "io test with mock should pass: {:?}", result);
    }

    #[test]
    fn test_async_function_await_sleep_with_mock() {
        // +await on `sleep` is a builtin IO op — should be intercepted by mock
        let source = "\
+fn delayed_value (ms:Int)->String [io,async]
  +await _:String = sleep(ms)
  +return \"done\"
";
        let program = build_program(source);

        let test_source = "\
!test delayed_value
  +with ms=1000 -> expect \"done\"
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];

        // Mock sleep so it returns immediately without real delay
        let mocks = vec![IoMock {
            operation: "sleep".to_string(),
            patterns: vec!["1000".to_string()],
            response: "".to_string(),
        }];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
        assert!(result.is_ok(), "sleep mock test should pass: {:?}", result);
    }

    #[test]
    fn test_async_function_with_mock_inbox_read() {
        let source = "\
+fn drain_inbox ()->String [io,async]
  +await resp:String = inbox_read()
  +return resp
";
        let program = build_program(source);

        let test_source = "\
!test drain_inbox
  +with -> expect \"[\\\"mocked\\\"]\"
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];

        let mocks = vec![IoMock {
            operation: "inbox_read".to_string(),
            patterns: vec!["".to_string()],
            response: "[\"mocked\"]".to_string(),
        }];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
        assert!(result.is_ok(), "inbox_read mock test should pass: {:?}", result);
    }

    #[test]
    fn test_async_function_nested_await_propagates_handle() {
        // An async function that calls another user-defined async function
        // which itself does +await on a builtin — handle must propagate
        let source = "\
+fn inner_fetch (url:String)->String [io,async]
  +await resp:String = http_get(url)
  +return resp

+fn outer_fetch (url:String)->String [io,async]
  +await data:String = inner_fetch(url)
  +return data
";
        let program = build_program(source);

        let test_source = "\
!test outer_fetch
  +with url=\"https://api.test.com\" -> expect \"nested_ok\"
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];

        let mocks = vec![IoMock {
            operation: "http_get".to_string(),
            patterns: vec!["api.test.com".to_string()],
            response: "nested_ok".to_string(),
        }];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
        assert!(result.is_ok(), "nested async call should pass: {:?}", result);
    }

    // ── Mock JSON response tests ─────────────────────────────────────

    #[test]
    fn test_mock_http_get_json_consumed_by_json_get() {
        // Mock returns JSON, function uses json_get to extract a field
        let source = "\
+fn get_name (url:String)->String [io,async]
  +await body:String = http_get(url)
  +let name:String = json_get(body, \"name\")
  +return name
";
        let program = build_program(source);

        let test_source = "\
!test get_name
  +with url=\"https://api.example.com/user\" -> expect \"alice\"
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];

        // The mock response is valid JSON — note: IoMock.response is the
        // *decoded* string (no backslash escapes).
        let mocks = vec![IoMock {
            operation: "http_get".to_string(),
            patterns: vec!["api.example.com".to_string()],
            response: r#"{"name":"alice","age":30}"#.to_string(),
        }];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
        assert!(result.is_ok(), "json_get on mock JSON should pass: {:?}", result);
    }

    #[test]
    fn test_mock_http_get_json_consumed_by_json_array_len() {
        // Mock returns a JSON array, function uses json_array_len
        let source = "\
+fn count_items (url:String)->Int [io,async]
  +await body:String = http_get(url)
  +let count:Int = json_array_len(body)
  +return count
";
        let program = build_program(source);

        let test_source = "\
!test count_items
  +with url=\"https://api.example.com/items\" -> expect 3
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];

        let mocks = vec![IoMock {
            operation: "http_get".to_string(),
            patterns: vec!["api.example.com".to_string()],
            response: r#"[1,2,3]"#.to_string(),
        }];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
        assert!(result.is_ok(), "json_array_len on mock JSON should pass: {:?}", result);
    }

    #[test]
    fn test_mock_escape_decoding_via_parser() {
        // Verify that !mock strings are properly unescaped by the parser.
        // The Adapsis source text: !mock http_get "api.test" -> "{\"ok\":true,\"items\":[1,2]}"
        let source = "!mock http_get \"api.test\" -> \"{\\\"ok\\\":true,\\\"items\\\":[1,2]}\"";
        let mocks = extract_mocks(source);
        assert_eq!(mocks.len(), 1);
        assert_eq!(mocks[0].patterns, vec!["api.test"]);
        // After unescape, response should be valid JSON without backslashes
        assert_eq!(mocks[0].response, r#"{"ok":true,"items":[1,2]}"#);
        // Verify it's valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&mocks[0].response)
            .expect("mock response should be valid JSON");
        assert_eq!(parsed["ok"], true);
    }

    #[test]
    fn test_mock_json_end_to_end_with_parser_escaping() {
        // End-to-end: !mock with escaped JSON → function uses json_get
        let fn_source = "\
+fn check_status (url:String)->String [io,async]
  +await body:String = http_get(url)
  +let status:String = json_get(body, \"status\")
  +return status
";
        let program = build_program(fn_source);

        // This is how it would appear in a .ax file — escaped quotes
        // Adapsis source: !mock http_get "api.svc" -> "{\"status\":\"healthy\",\"uptime\":99}"
        let mock_source = "!mock http_get \"api.svc\" -> \"{\\\"status\\\":\\\"healthy\\\",\\\"uptime\\\":99}\"";
        let mocks = extract_mocks(mock_source);

        let test_source = "\
!test check_status
  +with url=\"https://api.svc.local/health\" -> expect \"healthy\"
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];

        let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
        assert!(result.is_ok(), "end-to-end mock JSON test should pass: {:?}", result);
    }

    // ── Orchestrator-style integration tests ────────────────────────
    // These simulate the exact flow: parse source with +fn, !mock, !test;
    // collect mocks; build program; run eval_test_case_with_mocks.

    /// Helper: simulate orchestrator flow — parse full source, collect mocks,
    /// build program, run each test case with mocks.
    fn run_orchestrator_style(source: &str) -> Vec<Result<String>> {
        let ops = parser::parse(source).expect("parse failed");
        let mut program = ast::Program::default();
        let mut test_ops = Vec::new();
        let mut io_mocks: Vec<IoMock> = Vec::new();

        for op in &ops {
            match op {
                parser::Operation::Test(test) => test_ops.push(test.clone()),
                parser::Operation::Module(m) => {
                    // Extract tests embedded inside module bodies
                    for body_op in &m.body {
                        if let parser::Operation::Test(test) = body_op {
                            test_ops.push(test.clone());
                        }
                    }
                    let _ = validator::apply_and_validate(&mut program, op);
                }
                parser::Operation::Mock { operation, patterns, response } => {
                    io_mocks.push(IoMock {
                        operation: operation.clone(),
                        patterns: patterns.clone(),
                        response: response.clone(),
                    });
                }
                parser::Operation::Unmock => { io_mocks.clear(); }
                parser::Operation::Eval(_) => {}
                _ => { let _ = validator::apply_and_validate(&mut program, op); }
            }
        }
        program.rebuild_function_index();

        let mut results = Vec::new();
        for test in &test_ops {
            for case in &test.cases {
                results.push(eval_test_case_with_mocks(
                    &program, &test.function_name, case, &io_mocks, &[],
                ));
            }
        }
        results
    }

    #[test]
    fn test_orchestrator_async_http_get_with_mock() {
        // Simulates LLM output containing +fn [async], !mock, !test
        let source = "\
+fn fetch_data (url:String)->String [io,async]
  +await resp:String = http_get(url)
  +return resp

!mock http_get \"example.com\" -> \"hello world\"

!test fetch_data
  +with url=\"https://example.com/api\" -> expect \"hello world\"
";
        let results = run_orchestrator_style(source);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok(), "orchestrator async http_get test should pass: {:?}", results[0]);
    }

    #[test]
    fn test_orchestrator_async_sleep_with_mock() {
        let source = "\
+fn delayed (ms:Int)->String [io,async]
  +await _:String = sleep(ms)
  +return \"done\"

!mock sleep \"500\" -> \"\"

!test delayed
  +with ms=500 -> expect \"done\"
";
        let results = run_orchestrator_style(source);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok(), "orchestrator async sleep test should pass: {:?}", results[0]);
    }

    #[test]
    fn test_orchestrator_nested_async_with_mock() {
        // wrapper -> inner_fetch -> http_get (all async, handle must propagate)
        let source = "\
+fn inner_fetch (url:String)->String [io,async]
  +await resp:String = http_get(url)
  +return resp

+fn wrapper (url:String)->String [io,async]
  +await data:String = inner_fetch(url)
  +return data

!mock http_get \"api.test\" -> \"nested result\"

!test wrapper
  +with url=\"https://api.test/v1\" -> expect \"nested result\"
";
        let results = run_orchestrator_style(source);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok(), "orchestrator nested async test should pass: {:?}", results[0]);
    }

    #[test]
    fn test_orchestrator_mock_json_escape_json_get_json_array_len() {
        // Proves: !mock with escaped JSON → parser decodes → json_get + json_array_len work
        // Adapsis source: !mock http_get "x" -> "{\"ok\":true,\"result\":[]}"
        let fn_source = "\
+fn check (url:String)->Int [io,async]
  +await body:String = http_get(url)
  +let arr:String = json_get(body, \"result\")
  +let count:Int = json_array_len(arr)
  +return count
";
        let mock_source = "!mock http_get \"x\" -> \"{\\\"ok\\\":true,\\\"result\\\":[]}\"";
        let test_source = "\
!test check
  +with url=\"x\" -> expect 0
";
        let full_source = format!("{fn_source}\n{mock_source}\n\n{test_source}");
        let results = run_orchestrator_style(&full_source);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok(), "mock JSON escape + json_get + json_array_len should pass: {:?}", results[0]);
    }

    // ── Async eval_test_case_async tests ─────────────────────────────

    #[tokio::test]
    async fn test_async_eval_test_case_with_mocks() {
        let source = "\
+fn fetch_data (url:String)->String [io,async]
  +await resp:String = http_get(url)
  +return resp
";
        let program = build_program(source);

        let test_source = "\
!test fetch_data
  +with url=\"https://example.com\" -> expect \"async_hello\"
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];

        let mocks = vec![IoMock {
            operation: "http_get".to_string(),
            patterns: vec!["example.com".to_string()],
            response: "async_hello".to_string(),
        }];

        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let result = eval_test_case_async(&program, fn_name, case, &mocks, tx, &[]).await;
        assert!(result.is_ok(), "async test case should pass: {:?}", result);
    }

    #[tokio::test]
    async fn test_session_apply_async_runs_async_tests_with_mocks() {
        // Simulate the full session flow: define an async function,
        // register mocks, then run !test — all through apply_async.
        let mut session = crate::session::Session::new();

        // Define async function
        let define_source = "\
+fn fetch_data (url:String)->String [io,async]
  +await resp:String = http_get(url)
  +return resp
";
        let results = session.apply_async(define_source, None).await;
        assert!(results.is_ok(), "define should succeed: {:?}", results);

        // Register mock
        let mock_source = "!mock http_get \"example.com\" -> \"mocked_response\"";
        let results = session.apply_async(mock_source, None).await;
        assert!(results.is_ok(), "mock should succeed: {:?}", results);

        // Run test — async function with mock, no io_sender needed (mock-only)
        let test_source = "\
!test fetch_data
  +with url=\"https://example.com/api\" -> expect \"mocked_response\"
";
        let results = session.apply_async(test_source, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].1, "async test with mock should pass: {:?}", results[0]);
        assert!(results[0].0.contains("PASS"), "result should be PASS: {:?}", results[0]);
    }

    #[tokio::test]
    async fn test_session_apply_async_nested_async_with_mocks() {
        // Nested async: wrapper -> inner_fetch -> http_get
        let mut session = crate::session::Session::new();

        let source = "\
+fn inner_fetch (url:String)->String [io,async]
  +await resp:String = http_get(url)
  +return resp

+fn wrapper (url:String)->String [io,async]
  +call data:String = inner_fetch(url)
  +return data
";
        let _ = session.apply_async(source, None).await;

        let mock_source = "!mock http_get \"api.test\" -> \"nested_result\"";
        let _ = session.apply_async(mock_source, None).await;

        let test_source = "\
!test wrapper
  +with url=\"https://api.test/v1\" -> expect \"nested_result\"
";
        let results = session.apply_async(test_source, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].1, "nested async test should pass: {:?}", results[0]);
    }

    #[tokio::test]
    async fn test_async_eval_delegates_sync_to_sync_path() {
        let source = "\
+fn add (a:Int, b:Int)->Int
  +let sum:Int = a + b
  +return sum
";
        let program = build_program(source);

        let test_source = "\
!test add
  +with a=10 b=20 -> expect 30
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];

        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let result = eval_test_case_async(&program, fn_name, case, &[], tx, &[]).await;
        assert!(result.is_ok(), "sync function via async path should pass: {:?}", result);
    }

    // ── UTF-8 regression tests ───────────────────────────────────────

    #[test]
    fn test_json_get_preserves_utf8() {
        // Verify json_get returns multi-byte UTF-8 characters intact
        // Use a wrapper that builds the JSON internally to avoid parser quoting issues
        let source = r#"
+fn get_cafe ()->String
  +let json:String = "{\"name\":\"café\"}"
  +return json_get(json, "name")
"#;
        let program = build_program(source);
        let input = parser::parse("!eval get_cafe")
            .unwrap()
            .into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .unwrap();
        let result = eval_call_with_input(&program, &input.function_name, &input.input).unwrap();
        assert_eq!(result, r#""café""#, "json_get should preserve UTF-8 chars");
    }

    #[test]
    fn test_json_escape_preserves_utf8() {
        // Verify json_escape passes multi-byte UTF-8 through unchanged
        let source = r#"
+fn escape_it (s:String)->String
  +return json_escape(s)
"#;
        let program = build_program(source);
        let input = parser::parse(r#"!eval escape_it s="café élève naïve""#)
            .unwrap()
            .into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .unwrap();
        let result = eval_call_with_input(&program, &input.function_name, &input.input).unwrap();
        assert_eq!(result, r#""café élève naïve""#, "json_escape should preserve UTF-8 chars");
    }

    #[test]
    fn test_value_display_utf8_string() {
        // Verify Value::String Display preserves multi-byte UTF-8
        let val = Value::string("café ☕ 日本語");
        let displayed = format!("{val}");
        assert_eq!(displayed, r#""café ☕ 日本語""#, "Value display should preserve UTF-8");
    }

    #[test]
    fn test_concat_preserves_utf8() {
        let source = r#"
+fn greet (name:String)->String
  +return concat("Bonjour, ", name)
"#;
        let program = build_program(source);
        let input = parser::parse(r#"!eval greet name="André""#)
            .unwrap()
            .into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .unwrap();
        let result = eval_call_with_input(&program, &input.function_name, &input.input).unwrap();
        assert_eq!(result, r#""Bonjour, André""#, "concat should preserve UTF-8");
    }

    #[test]
    fn test_unicode_string_literal_roundtrip() {
        // Full Unicode string literal: parse → eval → display must preserve
        // all multi-byte characters exactly.
        let source = "
+fn probe ()->String
  +return \"hé — 你好 ✓ ★\"
";
        let program = build_program(source);
        let input = parser::parse("!eval probe")
            .unwrap()
            .into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .unwrap();
        let result = eval_call_with_input(&program, &input.function_name, &input.input).unwrap();
        assert_eq!(result, "\"hé — 你好 ✓ ★\"",
            "Unicode string literal must survive parse/eval/display without mojibake");
        // Verify actual byte representation
        let inner = &result[1..result.len()-1]; // strip quotes
        assert_eq!(inner.as_bytes(), "hé — 你好 ✓ ★".as_bytes(),
            "UTF-8 byte representation must match");
    }

    #[test]
    fn test_unicode_json_serialization_roundtrip() {
        // Build JSON containing Unicode via json_escape, then extract via json_get.
        // This simulates what send_message_body does with Unicode text.
        let source = r#"
+fn build_json (text:String)->String
  +let escaped:String = json_escape(text)
  +let body:String = concat("{\"text\":\"", concat(escaped, "\"}"))
  +return body

+fn extract_text (json:String)->String
  +return json_get(json, "text")

+fn roundtrip (text:String)->String
  +let json:String = build_json(text)
  +return extract_text(json)
"#;
        let program = build_program(source);
        let input = parser::parse("!eval roundtrip text=\"café — 你好 ✓ ★\"")
            .unwrap()
            .into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .unwrap();
        let result = eval_call_with_input(&program, &input.function_name, &input.input).unwrap();
        assert_eq!(result, "\"café — 你好 ✓ ★\"",
            "Unicode text must survive json_escape → JSON embedding → json_get roundtrip");
    }

    #[tokio::test]
    async fn test_mocked_http_utf8_roundtrip() {
        // Simulate an http_get returning UTF-8 text via mock, then extracting it.
        let mut session = crate::session::Session::new();

        let source = "
+fn fetch_text (url:String)->String [io,async]
  +await resp:String = http_get(url)
  +return json_get(resp, \"text\")
";
        let _ = session.apply_async(source, None).await;

        // Mock returns JSON with Unicode content
        let mock_source = "!mock http_get \"unicode-test\" -> \"{\\\"text\\\":\\\"café — 你好 ✓\\\"}\"";
        let _ = session.apply_async(mock_source, None).await;

        let test_source = "
!test fetch_text
  +with url=\"https://unicode-test.example.com\" -> expect \"café — 你好 ✓\"
";
        let results = session.apply_async(test_source, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].1, "mocked HTTP UTF-8 test should pass: {:?}", results[0]);
        assert!(results[0].0.contains("PASS"),
            "mocked HTTP returning UTF-8 JSON should round-trip: {:?}", results[0]);
    }

    #[tokio::test]
    async fn test_mocked_llm_utf8_roundtrip() {
        // Simulate llm_call returning UTF-8 text via mock.
        let mut session = crate::session::Session::new();

        let source = "
+fn ask_llm (prompt:String)->String [io,async]
  +await reply:String = llm_call(prompt, \"echo\")
  +return reply
";
        let _ = session.apply_async(source, None).await;

        // Mock llm_call to return Unicode text
        let mock_source = "!mock llm_call \"test\" -> \"café — 你好 ✓ ★\"";
        let _ = session.apply_async(mock_source, None).await;

        let test_source = "
!test ask_llm
  +with prompt=\"test prompt\" -> expect \"café — 你好 ✓ ★\"
";
        let results = session.apply_async(test_source, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].1, "mocked LLM UTF-8 test should pass: {:?}", results[0]);
    }

    // ── Pure function calls in test parameters ────────────────────────

    #[test]
    fn test_zero_arg_function_as_test_param_value() {
        // A function call with () used as a test parameter value
        // should call the function and use its return value
        let source = "\
+type Config = {host:String, port:Int}

+fn make_default ()->Config
  +let c:Config = {host: \"localhost\", port: 8080}
  +return c

+fn get_host (c:Config)->String
  +return c.host

!test get_host
  +with c=make_default() -> expect \"localhost\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_ok(), "zero-arg function call as param value should work: {:?}", result);
        assert!(result.unwrap().contains("expected \"localhost\""));
    }

    #[test]
    fn test_bare_function_name_as_test_param_value() {
        // A bare function name (no parens) for a zero-arg function should
        // be called, not turned into Err("make_default")
        let source = "\
+type Config = {host:String, port:Int}

+fn make_default ()->Config
  +let c:Config = {host: \"localhost\", port: 8080}
  +return c

+fn get_host (c:Config)->String
  +return c.host

!test get_host
  +with c=make_default -> expect \"localhost\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_ok(), "bare function name as param value should call function: {:?}", result);
        assert!(result.unwrap().contains("expected \"localhost\""));
    }

    #[test]
    fn test_function_call_with_args_in_test_param() {
        // A function call with arguments in a test parameter should work
        let source = "\
+type Config = {host:String, port:Int}

+fn make_config (h:String, p:Int)->Config
  +let c:Config = {host: h, port: p}
  +return c

+fn get_port (c:Config)->Int
  +return c.port

!test get_port
  +with c=make_config(\"example.com\", 3000) -> expect 3000
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_ok(), "function call with args as param value: {:?}", result);
        assert!(result.unwrap().contains("expected 3000"));
    }

    #[test]
    fn test_function_call_in_expected_value() {
        // Function calls should also work on the expected side of ->
        let source = "\
+type Config = {host:String, port:Int}

+fn make_default ()->Config
  +let c:Config = {host: \"localhost\", port: 8080}
  +return c

+fn identity (c:Config)->Config
  +return c

!test identity
  +with c=make_default() -> expect make_default()
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_ok(), "function call in expected value: {:?}", result);
        assert!(result.unwrap().contains("expected"));
    }

    // ── Positional (space-separated) args in !eval and +with ──────────

    #[test]
    fn test_eval_positional_multiple_strings() {
        let source = "\
+fn concat_two (a:String, b:String)->String
  +let result:String = concat(a, b)
  +return result
";
        let program = build_program(source);
        let eval_source = r#"!eval concat_two "hello" "world""#;
        let ops = parser::parse(eval_source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        let result = eval_compiled_or_interpreted(&program, &ev.function_name, &ev.input);
        assert!(result.is_ok(), "positional string args should work: {:?}", result);
        assert_eq!(result.unwrap().0, "\"helloworld\"");
    }

    #[test]
    fn test_eval_positional_mixed_types() {
        let source = "\
+fn show (a:String, b:Int)->String
  +let result:String = concat(a, to_string(b))
  +return result
";
        let program = build_program(source);
        let eval_source = r#"!eval show "count:" 42"#;
        let ops = parser::parse(eval_source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        let result = eval_compiled_or_interpreted(&program, &ev.function_name, &ev.input);
        assert!(result.is_ok(), "mixed positional args: {:?}", result);
        assert_eq!(result.unwrap().0, "\"count:42\"");
    }

    #[test]
    fn test_eval_positional_ints() {
        let source = "\
+fn add (a:Int, b:Int)->Int
  +let result:Int = a + b
  +return result
";
        let program = build_program(source);
        let eval_source = "!eval add 3 4";
        let ops = parser::parse(eval_source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        let result = eval_compiled_or_interpreted(&program, &ev.function_name, &ev.input);
        assert!(result.is_ok(), "positional int args: {:?}", result);
        assert_eq!(result.unwrap().0, "7");
    }

    #[test]
    fn test_with_positional_strings() {
        let source = "\
+fn concat_two (a:String, b:String)->String
  +let result:String = concat(a, b)
  +return result

!test concat_two
  +with \"hello\" \"world\" -> expect \"helloworld\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_ok(), "+with positional strings: {:?}", result);
        assert!(result.unwrap().contains("expected \"helloworld\""));
    }

    #[test]
    fn test_eval_builtin_positional_strings() {
        // Test that positional args also work for builtins
        let program = ast::Program::default();
        let eval_source = r#"!eval concat "foo" "bar""#;
        let ops = parser::parse(eval_source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        let result = eval_compiled_or_interpreted(&program, &ev.function_name, &ev.input);
        assert!(result.is_ok(), "builtin positional strings: {:?}", result);
        assert_eq!(result.unwrap().0, "\"foobar\"");
    }

    // ── Inline expression eval (!eval <expr>) ─────────────────────────

    #[test]
    fn test_eval_inline_arithmetic() {
        let program = ast::Program::default();
        let source = "!eval 1 + 2";
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some(), "should be inline expression");
        let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
        assert!(result.is_ok(), "inline 1+2: {:?}", result);
        assert_eq!(format!("{}", result.unwrap()), "3");
    }

    #[test]
    fn test_eval_inline_multiply_add() {
        let program = ast::Program::default();
        let source = "!eval 3 * 4 + 1";
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
        assert!(result.is_ok(), "inline 3*4+1: {:?}", result);
        assert_eq!(format!("{}", result.unwrap()), "13");
    }

    #[test]
    fn test_eval_inline_string_literal() {
        let program = ast::Program::default();
        let source = r#"!eval "hello""#;
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
        assert!(result.is_ok(), "inline string: {:?}", result);
        assert_eq!(format!("{}", result.unwrap()), "\"hello\"");
    }

    #[test]
    fn test_eval_inline_numeric_literal() {
        let program = ast::Program::default();
        let source = "!eval 42";
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
        assert!(result.is_ok(), "inline 42: {:?}", result);
        assert_eq!(format!("{}", result.unwrap()), "42");
    }

    #[test]
    fn test_eval_inline_boolean() {
        let program = ast::Program::default();
        let source = "!eval true";
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
        assert!(result.is_ok(), "inline true: {:?}", result);
        assert_eq!(format!("{}", result.unwrap()), "true");
    }

    #[test]
    fn test_eval_inline_comparison() {
        let program = ast::Program::default();
        let source = "!eval 3 > 2";
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
        assert!(result.is_ok(), "inline 3>2: {:?}", result);
        assert_eq!(format!("{}", result.unwrap()), "true");
    }

    #[test]
    fn test_eval_inline_concat_call() {
        let program = ast::Program::default();
        let source = r#"!eval concat("hello", " ", "world")"#;
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
        assert!(result.is_ok(), "inline concat: {:?}", result);
        assert_eq!(format!("{}", result.unwrap()), "\"hello world\"");
    }

    #[test]
    fn test_eval_inline_len_call() {
        let program = ast::Program::default();
        let source = r#"!eval len("hello")"#;
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
        assert!(result.is_ok(), "inline len: {:?}", result);
        assert_eq!(format!("{}", result.unwrap()), "5");
    }

    #[test]
    fn test_error_suggest_matches_known_pattern() {
        let program = ast::Program::default();
        let mut env = Env::new();
        let result = eval_builtin_or_user(
            &program,
            "error_suggest",
            vec![Value::string("undefined variable `user_id`")],
            &mut env,
        ).unwrap();
        assert_eq!(
            format!("{result}"),
            r#""Check variable spelling. Variables must be declared with +let or +call before use. Function parameters use the exact names from the signature.""#
        );
    }

    #[test]
    fn test_error_suggest_unknown_pattern_returns_empty_string() {
        let program = ast::Program::default();
        let mut env = Env::new();
        let result = eval_builtin_or_user(
            &program,
            "error_suggest",
            vec![Value::string("weird custom failure")],
            &mut env,
        ).unwrap();
        assert_eq!(format!("{result}"), "\"\"");
    }

    #[test]
    fn test_eval_inline_nested_calls() {
        let program = ast::Program::default();
        let source = r#"!eval len(concat("a", "b"))"#;
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
        assert!(result.is_ok(), "inline nested calls: {:?}", result);
        assert_eq!(format!("{}", result.unwrap()), "2");
    }

    #[test]
    fn test_eval_inline_struct_literal() {
        let program = ast::Program::default();
        let source = r#"!eval {name: "alice", age: 25}"#;
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
        assert!(result.is_ok(), "inline struct: {:?}", result);
        let val = result.unwrap();
        // Struct should contain the fields
        match val {
            Value::Struct(_, fields) => {
                let name_id = intern::intern_display("name");
                let age_id = intern::intern_display("age");
                assert!(matches!(fields.get(&name_id), Some(Value::String(s)) if s.as_str() == "alice"), "expected name=alice");
                assert!(matches!(fields.get(&age_id), Some(Value::Int(25))), "expected age=25");
            }
            _ => panic!("expected struct, got {val}"),
        }
    }

    #[test]
    fn test_eval_inline_list_creation() {
        let program = ast::Program::default();
        let source = "!eval list(1, 2, 3)";
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
        assert!(result.is_ok(), "inline list: {:?}", result);
        assert_eq!(format!("{}", result.unwrap()), "[1, 2, 3]");
    }

    #[test]
    fn test_eval_inline_user_function_call() {
        // Inline expression calling a user-defined function
        let source = "\
+fn double (x:Int)->Int
  +let result:Int = x * 2
  +return result
";
        let program = build_program(source);
        let eval_source = "!eval double(5)";
        let ops = parser::parse(eval_source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
        assert!(result.is_ok(), "inline user fn call: {:?}", result);
        assert_eq!(format!("{}", result.unwrap()), "10");
    }

    #[test]
    fn test_eval_func_name_still_works() {
        // Existing !eval func_name syntax should still work
        let source = "\
+fn greet ()->String
  +return \"hello\"
";
        let program = build_program(source);
        let eval_source = "!eval greet";
        let ops = parser::parse(eval_source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_none(), "bare function name should not be inline");
        let result = eval_compiled_or_interpreted(&program, &ev.function_name, &ev.input);
        assert!(result.is_ok(), "bare function name: {:?}", result);
        assert_eq!(result.unwrap().0, "\"hello\"");
    }

    #[test]
    fn test_eval_func_with_args_still_works() {
        // Existing !eval func_name arg1 arg2 syntax should still work
        let source = "\
+fn add (a:Int, b:Int)->Int
  +let result:Int = a + b
  +return result
";
        let program = build_program(source);
        let eval_source = "!eval add 3 4";
        let ops = parser::parse(eval_source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_none(), "func + args should not be inline");
        let result = eval_compiled_or_interpreted(&program, &ev.function_name, &ev.input);
        assert!(result.is_ok(), "func with args: {:?}", result);
        assert_eq!(result.unwrap().0, "7");
    }

    // ── expr_contains_io_builtin detection ─────────────────────────────

    #[test]
    fn test_expr_contains_io_builtin_detects_direct_call() {
        // shell_exec("echo hello") should be detected as IO
        let source = r#"!eval shell_exec("echo hello")"#;
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        assert!(
            expr_contains_io_builtin(ev.inline_expr.as_ref().unwrap()),
            "shell_exec should be detected as IO builtin"
        );
    }

    #[test]
    fn test_expr_contains_io_builtin_detects_nested_call() {
        // concat("result: ", shell_exec("echo hi")) — IO in args
        let source = r#"!eval concat("result: ", shell_exec("echo hi"))"#;
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        assert!(
            expr_contains_io_builtin(ev.inline_expr.as_ref().unwrap()),
            "nested shell_exec in concat args should be detected"
        );
    }

    #[test]
    fn test_expr_contains_io_builtin_false_for_sync() {
        // concat("a", "b") is NOT an IO builtin
        let source = r#"!eval concat("a", "b")"#;
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        assert!(
            !expr_contains_io_builtin(ev.inline_expr.as_ref().unwrap()),
            "concat should NOT be detected as IO builtin"
        );
    }

    #[test]
    fn test_expr_contains_io_builtin_false_for_arithmetic() {
        let source = "!eval 1 + 2";
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        assert!(
            !expr_contains_io_builtin(ev.inline_expr.as_ref().unwrap()),
            "arithmetic should NOT be detected as IO builtin"
        );
    }

    #[test]
    fn test_expr_contains_io_builtin_detects_http_get() {
        let source = r#"!eval http_get("http://example.com")"#;
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        assert!(
            expr_contains_io_builtin(ev.inline_expr.as_ref().unwrap()),
            "http_get should be detected as IO builtin"
        );
    }

    #[test]
    fn test_eval_inline_io_without_handle_still_errors() {
        // When no coroutine handle is available, IO builtins should still error
        let program = ast::Program::default();
        let source = r#"!eval shell_exec("echo hello")"#;
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
        assert!(result.is_err(), "IO builtin without handle should error");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("async IO operation"),
            "error should mention async IO: {err}"
        );
    }

    #[test]
    fn test_eval_inline_io_with_coroutine_handle_via_mock() {
        // When a coroutine handle IS available (via eval_inline_expr_with_io),
        // IO builtins should execute through it.
        // We use a full tokio runtime + coroutine Runtime to test this end-to-end.
        let program = ast::Program::default();
        let source = r#"!eval println("test message")"#;
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        let expr = ev.inline_expr.unwrap();

        // Spin up a real tokio runtime with coroutine IO loop
        let result = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async {
                let (runtime, mut io_rx) = crate::coroutine::Runtime::new();
                let runtime = std::sync::Arc::new(runtime);
                let io_sender = runtime.io_sender();

                // Spawn IO loop to handle requests (same pattern as eval_test_case_with_mocks)
                let rt_handle = runtime.clone();
                let io_loop = tokio::spawn(async move {
                    while let Some(request) = io_rx.recv().await {
                        let rt = rt_handle.clone();
                        tokio::spawn(async move {
                            rt.handle_io(request).await;
                        });
                    }
                });

                let eval_result = tokio::task::spawn_blocking(move || {
                    eval_inline_expr_with_io(&program, &expr, io_sender)
                }).await.unwrap();

                // Shut down the IO loop
                io_loop.abort();

                eval_result
            })
        }).join().unwrap();

        // println returns "" (empty string) on success
        assert!(result.is_ok(), "println via IO handle should succeed: {:?}", result);
    }

    #[test]
    fn test_eval_inline_io_builtin_in_user_function_context() {
        // eval_builtin_or_user should execute IO builtins when __coroutine_handle
        // is present in the env, rather than rejecting them
        let program = ast::Program::default();
        let mut env = Env::new();

        // Create a mock coroutine handle (will error on actual IO, but won't
        // give the "is an async IO operation" rejection)
        let handle = crate::coroutine::CoroutineHandle::new_mock(vec![
            crate::session::IoMock {
                operation: "println".to_string(),
                patterns: vec![],
                response: "".to_string(),
            },
        ]);
        env.set("__coroutine_handle", Value::CoroutineHandle(handle));

        let result = eval_builtin_or_user(
            &program,
            "println",
            vec![Value::string("hello")],
            &mut env,
        );
        // With mock, println should succeed (mocked response)
        assert!(result.is_ok(), "IO builtin with coroutine handle should not reject: {:?}", result);
    }

    // ── Side-effect checks for function calls in test params ──────────

    #[test]
    fn test_impure_function_rejected_in_test_param_bare() {
        // A function with [io,async] effects used as a bare name in a test
        // param should produce a clear error, not a confusing runtime failure
        let source = "\
+type State = {count:Int, name:String}

+fn fetch_state ()->State [io,async]
  +await data:String = http_get(\"http://example.com\")
  +let s:State = {count: 0, name: data}
  +return s

+fn get_name (state:State)->String
  +return state.name

!test get_name
  +with state=fetch_state -> expect \"default\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_err(), "impure function should be rejected: {:?}", result);
        let err = result.unwrap_err().to_string();
        assert!(err.contains("side effects"), "error should mention side effects: {err}");
        assert!(err.contains("fetch_state"), "error should name the function: {err}");
    }

    #[test]
    fn test_impure_function_rejected_in_test_param_call() {
        // Same but with parens: state=fetch_state()
        let source = "\
+type State = {count:Int, name:String}

+fn fetch_state ()->State [io,async]
  +await data:String = http_get(\"http://example.com\")
  +let s:State = {count: 0, name: data}
  +return s

+fn get_name (state:State)->String
  +return state.name

!test get_name
  +with state=fetch_state() -> expect \"default\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_err(), "impure function call should be rejected: {:?}", result);
        let err = result.unwrap_err().to_string();
        assert!(err.contains("side effects"), "error should mention side effects: {err}");
    }

    #[test]
    fn test_fail_effect_allowed_in_test_param() {
        // [fail] is not a side effect — it should be allowed
        let source = "\
+type Config = {host:String, port:Int}

+fn validated_config (host:String, port:Int)->Config [fail]
  +check valid_port port > 0 ~err_invalid_port
  +let c:Config = {host: host, port: port}
  +return c

+fn get_host (cfg:Config)->String
  +return cfg.host

!test get_host
  +with cfg=validated_config(\"localhost\", 8080) -> expect \"localhost\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_ok(), "[fail] function should be allowed in test params: {:?}", result);
        assert!(result.unwrap().contains("expected \"localhost\""));
    }

    // ── Escaped quotes in test value strings ──────────────────────────

    #[test]
    fn test_escaped_quotes_in_key_value_string() {
        // Strings with escaped quotes in key=value test params should work
        let source = r#"
+fn identity (s:String)->String
  +return s

!test identity
  +with s="hello\"world" -> expect "hello\"world"
"#;
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_ok(), "escaped quotes in key=value string: {:?}", result);
    }

    #[test]
    fn test_escaped_quotes_multiple_string_args() {
        // Multiple string args with escaped quotes should parse correctly
        let source = r#"
+fn concat_two (a:String, b:String)->String
  +let result:String = concat(a, b)
  +return result

!test concat_two
  +with a="he\"llo" b="wo\"rld" -> expect "he\"llowo\"rld"
"#;
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_ok(), "escaped quotes in multiple args: {:?}", result);
    }

    #[test]
    fn test_newline_and_tab_escapes_in_test_value() {
        let source = r#"
+fn identity (s:String)->String
  +return s

!test identity
  +with s="line1\nline2" -> expect "line1\nline2"
"#;
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_ok(), "newline escape in test value: {:?}", result);
    }

    // ── IO builtin error message tests ──────────────────────────────

    #[test]
    fn test_io_builtin_without_await_gives_helpful_error() {
        // Calling an IO builtin like http_get without +await should give a
        // specific error, not "undefined function"
        let source = "\
+fn broken (url:String)->String
  +let resp:String = http_get(url)
  +return resp
";
        let program = build_program(source);
        let test_source = "\
!test broken
  +with url=\"http://example.com\" -> expect \"\"
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_err(), "should fail when IO builtin used without +await");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("async IO operation"), "error should mention async IO: {err}");
        assert!(err.contains("+await"), "error should suggest +await: {err}");
        assert!(!err.contains("undefined function"), "should NOT say undefined function: {err}");
    }

    // ── +each scoping tests ──────────────────────────────────────────

    #[test]
    fn test_each_return_propagation() {
        // +return inside +each should propagate to the enclosing function
        let source = "\
+type Message = {role:String, content:String}

+fn find_role (messages:List<Message>)->String
  +each messages msg:Message
    +return msg.role
  +end
  +return \"none\"

!test find_role
  +with messages=list({role: \"user\", content: \"hi\"}) -> expect \"user\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_ok(), "+return inside +each should propagate: {:?}", result);
    }

    #[test]
    fn test_each_field_access_and_let() {
        // +let with field access on the loop variable should work
        let source = "\
+type Message = {role:String, content:String}

+fn build_text (messages:List<Message>)->String
  +let result:String = \"\"
  +each messages msg:Message
    +let role:String = msg.role
    +set result = concat(result, role, \" \")
  +end
  +return result

!test build_text
  +with messages=list({role: \"user\", content: \"hi\"}, {role: \"bot\", content: \"hello\"}) -> expect \"user bot \"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_ok(), "+let with field access in +each: {:?}", result);
    }

    // ── format_expr round-trip tests ──────────────────────────────────

    #[test]
    fn test_format_expr_struct_with_list_roundtrip() {
        // Bug: format_expr produced `messages=list()last_id=0` (missing separator)
        // for struct fields containing function calls, causing reparse failure.

        let source = "\
+type State = {messages:List<String>, last_id:Int}

+fn process (s:State)->Int
  +return s.last_id
";
        let program = build_program(source);

        // Simulate a stored test with struct input containing list()
        let test_source = "\
!test process
  +with s={messages: list(), last_id: 0} -> expect 0
";
        let cases = extract_test_cases(test_source);
        assert_eq!(cases.len(), 1);
        let (fn_name, case) = &cases[0];

        // First, verify the test passes directly
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_ok(), "direct test should pass: {:?}", result);

        // Now simulate the store/reparse cycle that invalidate_and_retest does
        let input_str = crate::session::format_expr_pub(&case.input);
        let expected_str = crate::session::format_expr_pub(&case.expected);

        // Reconstruct test source (same as invalidate_and_retest)
        let reconstructed = format!("!test process\n  +with {} -> expect {}\n", input_str, expected_str);

        // The reconstructed source must parse successfully
        let reparse = parser::parse(&reconstructed);
        assert!(reparse.is_ok(), "reconstructed test should parse: {:?} from source: {}", reparse.err(), reconstructed);

        // And the reparsed test should pass
        let reparsed_cases = extract_test_cases(&reconstructed);
        assert_eq!(reparsed_cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &reparsed_cases[0].0, &reparsed_cases[0].1, &[], &[]);
        assert!(result.is_ok(), "reparsed test should pass: {:?}", result);
    }

    // ── Test matchers ────────────────────────────────────────────────

    #[test]
    fn test_contains_matcher() {
        let source = "\
+fn greet (name:String)->String
  +return concat(\"hello \", name, \"!\")

!test greet
  +with name=\"world\" -> expect contains(\"hello\")
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let (fn_name, case) = &cases[0];
        assert!(case.matcher.is_some(), "should have a matcher");
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_ok(), "contains matcher should pass: {:?}", result);
    }

    #[test]
    fn test_contains_matcher_fails_when_not_present() {
        let source = "\
+fn greet (name:String)->String
  +return concat(\"hello \", name, \"!\")

!test greet
  +with name=\"world\" -> expect contains(\"goodbye\")
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_err(), "contains matcher should fail when substring absent");
    }

    #[test]
    fn test_starts_with_matcher() {
        let source = "\
+fn greet (name:String)->String
  +return concat(\"hello \", name)

!test greet
  +with name=\"world\" -> expect starts_with(\"hello\")
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_ok(), "starts_with matcher should pass: {:?}", result);
    }

    #[test]
    fn test_starts_with_matcher_fails() {
        let source = "\
+fn greet (name:String)->String
  +return concat(\"hello \", name)

!test greet
  +with name=\"world\" -> expect starts_with(\"goodbye\")
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_err(), "starts_with matcher should fail on wrong prefix");
    }

    #[test]
    fn test_any_ok_matcher() {
        let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x

!test validate
  +with x=5 -> expect Ok
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        assert!(matches!(case.matcher, Some(parser::TestMatcher::AnyOk)));
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_ok(), "AnyOk matcher should pass for Ok result: {:?}", result);
    }

    #[test]
    fn test_any_ok_matcher_fails_on_err() {
        let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x

!test validate
  +with x=-1 -> expect Ok
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_err(), "AnyOk matcher should fail on Err result");
    }

    #[test]
    fn test_any_err_matcher() {
        let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x

!test validate
  +with x=-1 -> expect Err
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        assert!(matches!(case.matcher, Some(parser::TestMatcher::AnyErr)));
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_ok(), "AnyErr matcher should pass for Err result: {:?}", result);
    }

    #[test]
    fn test_any_err_matcher_fails_on_ok() {
        let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x

!test validate
  +with x=5 -> expect Err
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_err(), "AnyErr matcher should fail on Ok result");
    }

    #[test]
    fn test_err_containing_matcher() {
        let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x

!test validate
  +with x=-1 -> expect Err(\"err_negative\")
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        assert!(matches!(case.matcher, Some(parser::TestMatcher::ErrContaining(_))));
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_ok(), "ErrContaining matcher should pass: {:?}", result);
    }

    #[test]
    fn test_err_containing_matcher_fails_on_wrong_msg() {
        let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x

!test validate
  +with x=-1 -> expect Err(\"err_something_else\")
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_err(), "ErrContaining should fail on wrong message");
    }

    // ── +after checks ────────────────────────────────────────────────

    #[test]
    fn test_after_routes_contains_pass() {
        // Build a program with a route, then test with +after routes contains
        let source = "\
+fn handler (body:String)->String
  +return \"ok\"

+route POST \"/chat\" -> handler

!test handler
  +with body=\"hello\" -> expect \"ok\"
  +after routes contains \"/chat\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let (fn_name, case) = &cases[0];
        assert_eq!(case.after_checks.len(), 1);
        assert_eq!(case.after_checks[0].target, "routes");
        assert_eq!(case.after_checks[0].value, "/chat");

        // Routes are now in RuntimeState, pass them explicitly
        let routes = vec![crate::ast::HttpRoute {
            method: "POST".to_string(),
            path: "/chat".to_string(),
            handler_fn: "handler".to_string(),
        }];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &routes);
        assert!(result.is_ok(), "+after routes should pass: {:?}", result);
    }

    #[test]
    fn test_after_routes_contains_fail() {
        // Route doesn't exist — +after should fail
        let source = "\
+fn handler (body:String)->String
  +return \"ok\"

!test handler
  +with body=\"hello\" -> expect \"ok\"
  +after routes contains \"/nonexistent\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_err(), "+after routes should fail when route missing");
    }

    #[test]
    fn test_after_modules_contains_pass() {
        let source = "\
!module TestMod

+fn helper ()->Int
  +return 42

!test helper
  +with -> expect 42
  +after modules contains \"TestMod\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_ok(), "+after modules should pass: {:?}", result);
    }

    #[test]
    fn test_after_modules_contains_fail() {
        let source = "\
+fn helper ()->Int
  +return 42

!test helper
  +with -> expect 42
  +after modules contains \"NonExistent\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_err(), "+after modules should fail when module missing");
    }

    #[test]
    fn test_matchers_combined_with_after() {
        // Combine a matcher with +after checks
        let source = "\
+fn handler (body:String)->String
  +return concat(\"processed: \", body)

+route GET \"/api\" -> handler

!test handler
  +with body=\"test\" -> expect contains(\"processed\")
  +after routes contains \"/api\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        assert!(case.matcher.is_some());
        assert_eq!(case.after_checks.len(), 1);
        let routes = vec![crate::ast::HttpRoute {
            method: "GET".to_string(),
            path: "/api".to_string(),
            handler_fn: "handler".to_string(),
        }];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &routes);
        assert!(result.is_ok(), "matcher + after should pass: {:?}", result);
    }

    #[test]
    fn test_multiple_after_checks() {
        let source = "\
!module MyMod

+fn handler (body:String)->String
  +return \"ok\"

+route POST \"/webhook\" -> handler

!test handler
  +with body=\"x\" -> expect \"ok\"
  +after routes contains \"/webhook\"
  +after modules contains \"MyMod\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        assert_eq!(case.after_checks.len(), 2);
        let routes = vec![crate::ast::HttpRoute {
            method: "POST".to_string(),
            path: "/webhook".to_string(),
            handler_fn: "handler".to_string(),
        }];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &routes);
        assert!(result.is_ok(), "multiple +after checks should pass: {:?}", result);
    }

    #[test]
    fn test_exact_err_still_works_without_matcher() {
        // Err(err_label) without quotes should still work as exact match (no matcher)
        let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x

!test validate
  +with x=-1 -> expect Err(err_negative)
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        // No matcher — this is the old exact match behavior
        assert!(case.matcher.is_none(), "Err(bare_ident) should not create a matcher");
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_ok(), "exact Err match should still work: {:?}", result);
    }

    // ═════════════════════════════════════════════════════════════════════
    // Inline expression helpers for builtin/arithmetic tests
    // ═════════════════════════════════════════════════════════════════════

    /// Evaluate an inline expression string and return the formatted result.
    fn eval_expr_str(program: &ast::Program, expr_text: &str) -> String {
        let expr = parser::parse_expr_pub(0, expr_text)
            .unwrap_or_else(|e| panic!("parse failed for `{expr_text}`: {e}"));
        let val = eval_inline_expr(program, &expr)
            .unwrap_or_else(|e| panic!("eval failed for `{expr_text}`: {e}"));
        format!("{val}")
    }

    /// Evaluate and return the raw Value (for type-specific assertions).
    fn eval_expr_val(program: &ast::Program, expr_text: &str) -> Value {
        let expr = parser::parse_expr_pub(0, expr_text)
            .unwrap_or_else(|e| panic!("parse failed for `{expr_text}`: {e}"));
        eval_inline_expr(program, &expr)
            .unwrap_or_else(|e| panic!("eval failed for `{expr_text}`: {e}"))
    }

    /// Evaluate a full program and return the result of eval'ing a function.
    /// Uses interpreter-only path to avoid JIT crashes in test context.
    fn eval_fn_result(source: &str, fn_name: &str, input: &str) -> String {
        let program = build_program(source);
        let input_expr = if input.is_empty() {
            parser::Expr::StructLiteral(vec![])
        } else {
            parser::parse_test_input(0, input).expect("parse input failed")
        };
        eval_call_with_input(&program, fn_name, &input_expr)
            .unwrap_or_else(|e| panic!("eval `{fn_name}` failed: {e}"))
    }

    // ═════════════════════════════════════════════════════════════════════
    // Arithmetic operations (+, -, *, /, %)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn arith_subtraction() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, "10 - 3"), "7");
    }

    #[test]
    fn arith_division() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, "10 / 3"), "3");
    }

    #[test]
    fn arith_modulo() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, "10 % 3"), "1");
    }

    #[test]
    fn arith_negative_result() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, "3 - 10"), "-7");
    }

    #[test]
    fn arith_compound_precedence() {
        let p = ast::Program::default();
        // (2 * 3) + (10 / 5) - 1 = 6 + 2 - 1 = 7
        assert_eq!(eval_expr_str(&p, "2 * 3 + 10 / 5 - 1"), "7");
    }

    #[test]
    fn arith_float_basic() {
        let p = ast::Program::default();
        let val = eval_expr_val(&p, "3.5 + 1.5");
        match val {
            Value::Float(f) => assert!((f - 5.0).abs() < 1e-10),
            _ => panic!("expected Float, got {val}"),
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // Comparison and boolean logic
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn cmp_less_than() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, "2 < 5"), "true");
        assert_eq!(eval_expr_str(&p, "5 < 2"), "false");
    }

    #[test]
    fn cmp_gte_lte() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, "5 >= 5"), "true");
        assert_eq!(eval_expr_str(&p, "5 <= 5"), "true");
        assert_eq!(eval_expr_str(&p, "4 >= 5"), "false");
        assert_eq!(eval_expr_str(&p, "6 <= 5"), "false");
    }

    #[test]
    fn cmp_eq_neq() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, "42 == 42"), "true");
        assert_eq!(eval_expr_str(&p, "42 != 42"), "false");
        assert_eq!(eval_expr_str(&p, "42 != 43"), "true");
    }

    #[test]
    fn bool_and_or_not() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, "true AND true"), "true");
        assert_eq!(eval_expr_str(&p, "true AND false"), "false");
        assert_eq!(eval_expr_str(&p, "false OR true"), "true");
        assert_eq!(eval_expr_str(&p, "false OR false"), "false");
        assert_eq!(eval_expr_str(&p, "NOT true"), "false");
        assert_eq!(eval_expr_str(&p, "NOT false"), "true");
    }

    // ═════════════════════════════════════════════════════════════════════
    // Builtin function calls
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn builtin_split() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, r#"split("a,b,c", ",")"#), r#"["a", "b", "c"]"#);
    }

    #[test]
    fn builtin_join() {
        let p = ast::Program::default();
        // join uses Display format for list items — strings include quotes
        assert_eq!(eval_expr_str(&p, "join(list(1, 2, 3), \"-\")"), r#""1-2-3""#);
    }

    #[test]
    fn builtin_trim() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, r#"trim("  hello  ")"#), r#""hello""#);
    }

    #[test]
    fn builtin_to_string() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, "to_string(42)"), r#""42""#);
        assert_eq!(eval_expr_str(&p, "to_string(true)"), r#""true""#);
    }

    #[test]
    fn builtin_to_int() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, r#"to_int("42")"#), "42");
        assert_eq!(eval_expr_str(&p, "to_int(3.7)"), "3");
        assert_eq!(eval_expr_str(&p, "to_int(true)"), "1");
    }

    #[test]
    fn builtin_push_and_get() {
        let p = ast::Program::default();
        // push returns a new list
        assert_eq!(eval_expr_str(&p, "push(list(1, 2), 3)"), "[1, 2, 3]");
        // get returns element at index
        assert_eq!(eval_expr_str(&p, "get(list(10, 20, 30), 1)"), "20");
    }

    #[test]
    fn builtin_char_at() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, r#"char_at("hello", 0)"#), r#""h""#);
        assert_eq!(eval_expr_str(&p, r#"char_at("hello", 4)"#), r#""o""#);
    }

    #[test]
    fn builtin_substring() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, r#"substring("hello world", 0, 5)"#), r#""hello""#);
        assert_eq!(eval_expr_str(&p, r#"substring("hello world", 6, 11)"#), r#""world""#);
    }

    #[test]
    fn builtin_starts_with_ends_with() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, r#"starts_with("hello", "hel")"#), "true");
        assert_eq!(eval_expr_str(&p, r#"starts_with("hello", "xyz")"#), "false");
        assert_eq!(eval_expr_str(&p, r#"ends_with("hello", "llo")"#), "true");
        assert_eq!(eval_expr_str(&p, r#"ends_with("hello", "xyz")"#), "false");
    }

    #[test]
    fn builtin_contains() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, r#"contains("hello world", "world")"#), "true");
        assert_eq!(eval_expr_str(&p, r#"contains("hello world", "xyz")"#), "false");
    }

    #[test]
    fn builtin_index_of() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, r#"index_of("hello", "ll")"#), "2");
        assert_eq!(eval_expr_str(&p, r#"index_of("hello", "xyz")"#), "-1");
    }

    #[test]
    fn builtin_abs() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, "abs(5)"), "5");
    }

    #[test]
    fn builtin_min_max() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, "min(3, 7)"), "3");
        assert_eq!(eval_expr_str(&p, "max(3, 7)"), "7");
    }

    #[test]
    fn builtin_floor() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, "floor(3.7)"), "3");
        assert_eq!(eval_expr_str(&p, "floor(3.2)"), "3");
    }

    #[test]
    fn builtin_pow() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, "pow(2, 10)"), "1024");
    }

    #[test]
    fn builtin_len_on_list() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, "len(list(1, 2, 3))"), "3");
        assert_eq!(eval_expr_str(&p, "len(list())"), "0");
    }

    #[test]
    fn builtin_base64_encode() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, r#"base64_encode("hello")"#), r#""aGVsbG8=""#);
    }

    #[test]
    fn builtin_digit_value() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, r#"digit_value("5")"#), "5");
        assert_eq!(eval_expr_str(&p, r#"digit_value("a")"#), "-1");
    }

    #[test]
    fn builtin_is_digit_char() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, r#"is_digit_char("7")"#), "true");
        assert_eq!(eval_expr_str(&p, r#"is_digit_char("x")"#), "false");
    }

    #[test]
    fn builtin_char_code_and_from_char_code() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, r#"char_code("A")"#), "65");
        assert_eq!(eval_expr_str(&p, "from_char_code(65)"), r#""A""#);
    }

    #[test]
    fn builtin_bitwise_ops() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, "bit_and(12, 10)"), "8"); // 1100 & 1010 = 1000
        assert_eq!(eval_expr_str(&p, "bit_or(12, 10)"), "14"); // 1100 | 1010 = 1110
        assert_eq!(eval_expr_str(&p, "bit_xor(12, 10)"), "6"); // 1100 ^ 1010 = 0110
        assert_eq!(eval_expr_str(&p, "bit_shl(1, 3)"), "8");   // 1 << 3 = 8
        assert_eq!(eval_expr_str(&p, "bit_shr(8, 2)"), "2");   // 8 >> 2 = 2
    }

    // ═════════════════════════════════════════════════════════════════════
    // Let bindings and variable lookup
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn let_binding_and_return() {
        let result = eval_fn_result("\
+fn double (x:Int)->Int
  +let result:Int = x * 2
  +return result
", "double", "5");
        assert_eq!(result, "10");
    }

    #[test]
    fn let_multiple_bindings() {
        let result = eval_fn_result("\
+fn compute (a:Int, b:Int)->Int
  +let sum:Int = a + b
  +let product:Int = a * b
  +let result:Int = sum + product
  +return result
", "compute", "a=3 b=4");
        // sum=7, product=12, result=19
        assert_eq!(result, "19");
    }

    #[test]
    fn set_mutation() {
        let result = eval_fn_result("\
+fn count_up ()->Int
  +let i:Int = 0
  +set i = i + 1
  +set i = i + 1
  +set i = i + 1
  +return i
", "count_up", "");
        assert_eq!(result, "3");
    }

    // ═════════════════════════════════════════════════════════════════════
    // If/elif/else control flow
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn if_then_branch() {
        let source = "\
+fn classify (x:Int)->String
  +if x > 0
    +return \"positive\"
  +else
    +return \"non-positive\"
  +end
";
        assert_eq!(eval_fn_result(source, "classify", "5"), "\"positive\"");
        assert_eq!(eval_fn_result(source, "classify", "0"), "\"non-positive\"");
    }

    #[test]
    fn if_elif_else() {
        let source = "\
+fn describe (x:Int)->String
  +if x > 0
    +return \"positive\"
  +elif x == 0
    +return \"zero\"
  +else
    +return \"negative\"
  +end
";
        assert_eq!(eval_fn_result(source, "describe", "5"), "\"positive\"");
        assert_eq!(eval_fn_result(source, "describe", "0"), "\"zero\"");
        assert_eq!(eval_fn_result(source, "describe", "-3"), "\"negative\"");
    }

    #[test]
    fn nested_if() {
        let source = "\
+fn size (x:Int)->String
  +if x > 0
    +if x > 100
      +return \"big\"
    +else
      +return \"small\"
    +end
  +else
    +return \"non-positive\"
  +end
";
        assert_eq!(eval_fn_result(source, "size", "200"), "\"big\"");
        assert_eq!(eval_fn_result(source, "size", "50"), "\"small\"");
        assert_eq!(eval_fn_result(source, "size", "-1"), "\"non-positive\"");
    }

    // ═════════════════════════════════════════════════════════════════════
    // Pattern matching on union types
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn match_on_union() {
        let source = "\
+type Shape = Circle(Float) | Rect(Float, Float) | Point

+fn area (s:Shape)->Float
  +match s
  +case Circle(r)
    +return r * r * 3.14
  +case Rect(w, h)
    +return w * h
  +case Point
    +return 0.0
  +end
";
        let program = build_program(source);
        // Test with Circle(5.0)
        let input = parser::parse_test_input(0, "Circle(5.0)").unwrap();
        let result = eval_call_with_input(&program, "area", &input).unwrap();
        let f: f64 = result.parse().unwrap();
        assert!((f - 78.5).abs() < 0.01, "Circle area: {result}");

        // Test with Rect(3.0, 4.0)
        let input = parser::parse_test_input(0, "Rect(3.0, 4.0)").unwrap();
        let result = eval_call_with_input(&program, "area", &input).unwrap();
        let f: f64 = result.parse().unwrap();
        assert!((f - 12.0).abs() < 0.01, "Rect area: {result}");
    }

    #[test]
    fn match_wildcard() {
        let source = "\
+type Color = Red | Green | Blue

+fn is_red (c:Color)->Bool
  +match c
  +case Red
    +return true
  +case _
    +return false
  +end
";
        let program = build_program(source);
        let input = parser::parse_test_input(0, "Red").unwrap();
        let result = eval_call_with_input(&program, "is_red", &input).unwrap();
        assert_eq!(result, "true");

        let input = parser::parse_test_input(0, "Blue").unwrap();
        let result = eval_call_with_input(&program, "is_red", &input).unwrap();
        assert_eq!(result, "false");
    }

    #[test]
    fn match_recursive_type() {
        let source = "\
+type Expr = Literal(Int) | Add(Expr, Expr)

+fn eval_expr (e:Expr)->Int
  +match e
  +case Literal(val)
    +return val
  +case Add(left, right)
    +let l:Int = eval_expr(left)
    +let r:Int = eval_expr(right)
    +return l + r
  +end
";
        let program = build_program(source);
        // Add(Literal(2), Literal(3)) should be 5
        let input = parser::parse_test_input(0, "Add(Literal(2), Literal(3))").unwrap();
        let result = eval_call_with_input(&program, "eval_expr", &input).unwrap();
        assert_eq!(result, "5");

        // Add(Literal(1), Add(Literal(2), Literal(3))) should be 6
        let input = parser::parse_test_input(0, "Add(Literal(1), Add(Literal(2), Literal(3)))").unwrap();
        let result = eval_call_with_input(&program, "eval_expr", &input).unwrap();
        assert_eq!(result, "6");
    }

    // ═════════════════════════════════════════════════════════════════════
    // Check statements (success and failure)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn check_passes() {
        let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +check small x < 100 ~err_too_large
  +return x
";
        // [fail] functions wrap successful returns in Ok
        assert_eq!(eval_fn_result(source, "validate", "50"), "Ok(50)");
    }

    #[test]
    fn check_fails_first() {
        let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x
";
        let result = eval_fn_result(source, "validate", "-5");
        assert!(result.contains("Err") && result.contains("err_negative"),
            "expected Err(err_negative), got: {result}");
    }

    #[test]
    fn check_fails_second() {
        let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +check small x < 100 ~err_too_large
  +return x
";
        let result = eval_fn_result(source, "validate", "200");
        assert!(result.contains("Err") && result.contains("err_too_large"),
            "expected Err(err_too_large), got: {result}");
    }

    // ═════════════════════════════════════════════════════════════════════
    // Function calls with error propagation ([fail] effect)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn function_call_chain() {
        let source = "\
+fn double (x:Int)->Int
  +let r:Int = x * 2
  +return r

+fn quadruple (x:Int)->Int
  +let d:Int = double(x)
  +let r:Int = double(d)
  +return r
";
        assert_eq!(eval_fn_result(source, "quadruple", "5"), "20");
    }

    #[test]
    fn function_call_with_check_propagation() {
        // When a [fail] function calls another [fail] function that fails,
        // the error should propagate via the +call binding
        let source = "\
+fn validate_positive (x:Int)->Result<Int> [fail]
  +check pos x > 0 ~err_negative
  +return x

+fn process (x:Int)->Result<Int> [fail]
  +call val:Int = validate_positive(x)
  +let result:Int = val * 2
  +return result
";
        assert_eq!(eval_fn_result(source, "process", "5"), "Ok(10)");
        let result = eval_fn_result(source, "process", "-1");
        assert!(result.contains("Err"), "expected Err propagation, got: {result}");
    }

    // ═════════════════════════════════════════════════════════════════════
    // Struct construction and field access
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn struct_create_and_access() {
        let source = "\
+type Point = x:Int, y:Int

+fn get_x (p:Point)->Int
  +return p.x

+fn make_point ()->Point
  +let p:Point = {x: 10, y: 20}
  +return p
";
        let result = eval_fn_result(source, "make_point", "");
        // Field order in HashMap is not deterministic, check both fields present
        assert!(result.contains("x: 10"), "expected x: 10 in {result}");
        assert!(result.contains("y: 20"), "expected y: 20 in {result}");
        assert_eq!(eval_fn_result(source, "get_x", "x=5 y=10"), "5");
    }

    #[test]
    fn struct_field_access_in_expression() {
        let source = "\
+type Rect = width:Int, height:Int

+fn area (r:Rect)->Int
  +let a:Int = r.width * r.height
  +return a
";
        assert_eq!(eval_fn_result(source, "area", "width=3 height=4"), "12");
    }

    #[test]
    fn struct_nested_field_access() {
        let source = "\
+type Inner = value:Int
+type Outer = inner:Inner, label:String

+fn get_value (o:Outer)->Int
  +return o.inner.value
";
        assert_eq!(eval_fn_result(source, "get_value", r#"inner={value: 42} label="test""#), "42");
    }

    // ═════════════════════════════════════════════════════════════════════
    // While loops
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn while_loop_counter() {
        let source = "\
+fn count_to (n:Int)->Int
  +let i:Int = 0
  +while i < n
    +set i = i + 1
  +end
  +return i
";
        assert_eq!(eval_fn_result(source, "count_to", "5"), "5");
        assert_eq!(eval_fn_result(source, "count_to", "0"), "0");
    }

    #[test]
    fn while_loop_accumulator() {
        let source = "\
+fn sum_to (n:Int)->Int
  +let total:Int = 0
  +let i:Int = 1
  +while i <= n
    +set total = total + i
    +set i = i + 1
  +end
  +return total
";
        assert_eq!(eval_fn_result(source, "sum_to", "10"), "55");
    }

    // ═════════════════════════════════════════════════════════════════════
    // String concat with mixed types
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn concat_mixed_types() {
        let p = ast::Program::default();
        // concat coerces non-strings to string via Display
        assert_eq!(eval_expr_str(&p, r#"concat("count: ", 42)"#), r#""count: 42""#);
        assert_eq!(eval_expr_str(&p, r#"concat("flag: ", true)"#), r#""flag: true""#);
    }

    // ═════════════════════════════════════════════════════════════════════
    // Error cases for builtins
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn builtin_get_out_of_bounds() {
        let p = ast::Program::default();
        let expr = parser::parse_expr_pub(0, "get(list(1, 2), 5)").unwrap();
        let result = eval_inline_expr(&p, &expr);
        assert!(result.is_err(), "expected error for out-of-bounds get");
    }

    #[test]
    fn builtin_char_at_out_of_bounds() {
        let p = ast::Program::default();
        let expr = parser::parse_expr_pub(0, r#"char_at("hi", 10)"#).unwrap();
        let result = eval_inline_expr(&p, &expr);
        assert!(result.is_err(), "expected error for out-of-bounds char_at");
    }

    #[test]
    fn builtin_to_int_invalid() {
        let p = ast::Program::default();
        let expr = parser::parse_expr_pub(0, r#"to_int("not_a_number")"#).unwrap();
        let result = eval_inline_expr(&p, &expr);
        assert!(result.is_err(), "expected error for invalid to_int");
    }

    // ═════════════════════════════════════════════════════════════════════
    // Missing builtins: bit_not, left_rotate, to_hex, u32_wrap, sqrt
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn builtin_bit_not() {
        let p = ast::Program::default();
        // bit_not(0) = -1 (all bits flipped in two's complement)
        assert_eq!(eval_expr_str(&p, "bit_not(0)"), "-1");
        // bit_not(-1) = 0
        assert_eq!(eval_expr_str(&p, "bit_not(-1)"), "0");
    }

    #[test]
    fn builtin_left_rotate() {
        let p = ast::Program::default();
        // 1 rotated left by 4 = 16 (32-bit rotation)
        assert_eq!(eval_expr_str(&p, "left_rotate(1, 4)"), "16");
        // rotl alias works too
        assert_eq!(eval_expr_str(&p, "rotl(1, 4)"), "16");
    }

    #[test]
    fn builtin_to_hex() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, "to_hex(255)"), r#""000000ff""#);
        assert_eq!(eval_expr_str(&p, "to_hex(0)"), r#""00000000""#);
    }

    #[test]
    fn builtin_u32_wrap() {
        let p = ast::Program::default();
        // Large value wraps to 32-bit unsigned
        assert_eq!(eval_expr_str(&p, "u32_wrap(256)"), "256");
        // Negative wraps around
        let val = eval_expr_val(&p, "u32_wrap(-1)");
        match val {
            Value::Int(n) => assert_eq!(n, 4294967295), // u32::MAX
            _ => panic!("expected Int, got {val}"),
        }
    }

    #[test]
    fn builtin_sqrt() {
        let p = ast::Program::default();
        let val = eval_expr_val(&p, "sqrt(25)");
        match val {
            Value::Float(f) => assert!((f - 5.0).abs() < 1e-10),
            _ => panic!("expected Float, got {val}"),
        }
        let val = eval_expr_val(&p, "sqrt(2.0)");
        match val {
            Value::Float(f) => assert!((f - 1.41421356).abs() < 1e-5),
            _ => panic!("expected Float, got {val}"),
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // Ok/Err/Some constructors
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn constructor_ok() {
        let p = ast::Program::default();
        let val = eval_expr_val(&p, "Ok(42)");
        match val {
            Value::Ok(inner) => {
                assert!(matches!(*inner, Value::Int(42)));
            }
            _ => panic!("expected Ok(42), got {val}"),
        }
    }

    #[test]
    fn constructor_err() {
        let p = ast::Program::default();
        let val = eval_expr_val(&p, r#"Err("not found")"#);
        match val {
            Value::Err(msg) => assert_eq!(&*msg, "\"not found\""),
            _ => panic!("expected Err, got {val}"),
        }
    }

    #[test]
    fn constructor_some() {
        let p = ast::Program::default();
        // Some(5) via parser expression path creates a Union variant
        let val = eval_expr_val(&p, "Some(5)");
        // The parser expr path treats Some as a generic union constructor
        let display = format!("{val}");
        assert!(display.contains("5"), "expected Some containing 5, got {display}");
    }

    #[test]
    fn constructor_ok_no_args() {
        let p = ast::Program::default();
        // Ok() with no args wraps None
        let val = eval_expr_val(&p, "Ok()");
        assert!(matches!(val, Value::Ok(ref inner) if matches!(**inner, Value::None)),
            "expected Ok(None), got {val}");
    }

    // ═════════════════════════════════════════════════════════════════════
    // regex_match and regex_replace
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn builtin_regex_match() {
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, r#"regex_match("^[0-9]+$", "12345")"#), "true");
        assert_eq!(eval_expr_str(&p, r#"regex_match("^[0-9]+$", "abc")"#), "false");
        assert_eq!(eval_expr_str(&p, r#"regex_match("[a-z]+", "Hello World")"#), "true");
    }

    #[test]
    fn builtin_regex_replace() {
        let p = ast::Program::default();
        assert_eq!(
            eval_expr_str(&p, r#"regex_replace("[0-9]+", "NUM", "foo123bar456")"#),
            r#""fooNUMbarNUM""#
        );
        assert_eq!(
            eval_expr_str(&p, r#"regex_replace("\\s+", "-", "hello   world")"#),
            r#""hello-world""#
        );
    }

    // ═════════════════════════════════════════════════════════════════════
    // Result/Option pattern matching
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn match_on_result_ok() {
        let source = "\
+fn unwrap_or_default (r:Result<Int>)->Int
  +match r
  +case Ok(val)
    +return val
  +case Err(msg)
    +return 0
  +end
";
        // Pass Ok(42)
        assert_eq!(eval_fn_result(source, "unwrap_or_default", "Ok(42)"), "42");
    }

    #[test]
    fn match_on_result_err() {
        let source = "\
+fn unwrap_or_default (r:Result<Int>)->Int
  +match r
  +case Ok(val)
    +return val
  +case Err(msg)
    +return 0
  +end
";
        // Pass Err("fail")
        assert_eq!(eval_fn_result(source, "unwrap_or_default", r#"Err("fail")"#), "0");
    }

    #[test]
    fn match_on_option_some_none() {
        let source = "\
+type Maybe = Just(Int) | Nothing

+fn get_or_zero (m:Maybe)->Int
  +match m
  +case Just(val)
    +return val
  +case Nothing
    +return 0
  +end
";
        assert_eq!(eval_fn_result(source, "get_or_zero", "Just(99)"), "99");
        assert_eq!(eval_fn_result(source, "get_or_zero", "Nothing"), "0");
    }

    // ═════════════════════════════════════════════════════════════════════
    // Effect checking — pure function cannot call IO
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn pure_function_rejects_io_call() {
        // A pure function (no effects) calling an IO builtin produces an error
        // result (the function returns an Err value, not a hard crash)
        let result = eval_fn_result("\
+fn bad ()->String
  +return http_get(\"http://example.com\")
", "bad", "");
        // The eval catches the IO-without-await error and returns it
        assert!(result.contains("async IO operation") || result.contains("http_get"),
            "expected IO rejection message, got: {result}");
    }

    #[test]
    fn fail_effect_wraps_result() {
        // [fail] functions wrap their return in Ok on success, Err on check failure
        let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x
";
        let result = eval_fn_result(source, "validate", "10");
        assert!(result.starts_with("Ok("), "expected Ok wrapping, got: {result}");
        let result = eval_fn_result(source, "validate", "-1");
        assert!(result.starts_with("Err("), "expected Err wrapping, got: {result}");
    }

    // ═════════════════════════════════════════════════════════════════════
    // Each loop over lists
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn each_loop_accumulate() {
        let source = "\
+fn sum_list (items:List<Int>)->Int
  +let total:Int = 0
  +each items item:Int
    +set total = total + item
  +end
  +return total
";
        assert_eq!(eval_fn_result(source, "sum_list", "list(1, 2, 3, 4, 5)"), "15");
    }

    #[test]
    fn each_loop_transform() {
        let source = "\
+fn double_all (items:List<Int>)->List<Int>
  +let result:List<Int> = list()
  +each items item:Int
    +set result = push(result, item * 2)
  +end
  +return result
";
        assert_eq!(eval_fn_result(source, "double_all", "list(1, 2, 3)"), "[2, 4, 6]");
    }

    #[test]
    fn each_loop_empty_list() {
        let source = "\
+fn sum_list (items:List<Int>)->Int
  +let total:Int = 0
  +each items item:Int
    +set total = total + item
  +end
  +return total
";
        assert_eq!(eval_fn_result(source, "sum_list", "list()"), "0");
    }

    // ═════════════════════════════════════════════════════════════════════
    // While loop with early return
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn while_loop_factorial() {
        let source = "\
+fn factorial (n:Int)->Int
  +let result:Int = 1
  +let i:Int = 1
  +while i <= n
    +set result = result * i
    +set i = i + 1
  +end
  +return result
";
        assert_eq!(eval_fn_result(source, "factorial", "5"), "120");
        assert_eq!(eval_fn_result(source, "factorial", "0"), "1");
        assert_eq!(eval_fn_result(source, "factorial", "1"), "1");
    }

    // ═════════════════════════════════════════════════════════════════════
    // If/elif/else chains (additional cases)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn if_multiple_elif() {
        let source = "\
+fn grade (score:Int)->String
  +if score >= 90
    +return \"A\"
  +elif score >= 80
    +return \"B\"
  +elif score >= 70
    +return \"C\"
  +elif score >= 60
    +return \"D\"
  +else
    +return \"F\"
  +end
";
        assert_eq!(eval_fn_result(source, "grade", "95"), "\"A\"");
        assert_eq!(eval_fn_result(source, "grade", "85"), "\"B\"");
        assert_eq!(eval_fn_result(source, "grade", "75"), "\"C\"");
        assert_eq!(eval_fn_result(source, "grade", "65"), "\"D\"");
        assert_eq!(eval_fn_result(source, "grade", "50"), "\"F\"");
    }

    #[test]
    fn if_with_complex_condition() {
        let source = "\
+fn in_range (x:Int)->Bool
  +if x >= 10 AND x <= 20
    +return true
  +else
    +return false
  +end
";
        assert_eq!(eval_fn_result(source, "in_range", "15"), "true");
        assert_eq!(eval_fn_result(source, "in_range", "5"), "false");
        assert_eq!(eval_fn_result(source, "in_range", "25"), "false");
    }

    // ═════════════════════════════════════════════════════════════════════
    // Check statement (additional edge cases)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn check_with_compound_condition() {
        let source = "\
+fn validate_range (x:Int)->Result<Int> [fail]
  +check in_range x >= 0 AND x <= 100 ~err_out_of_range
  +return x
";
        assert_eq!(eval_fn_result(source, "validate_range", "50"), "Ok(50)");
        let result = eval_fn_result(source, "validate_range", "-1");
        assert!(result.contains("err_out_of_range"), "expected err_out_of_range, got: {result}");
        let result = eval_fn_result(source, "validate_range", "200");
        assert!(result.contains("err_out_of_range"), "expected err_out_of_range, got: {result}");
    }

    #[test]
    fn check_multiple_sequential() {
        // Multiple checks, each with different error labels
        let source = "\
+fn validate_user (name:String, age:Int)->Result<String> [fail]
  +check has_name len(name) > 0 ~err_empty_name
  +check valid_age age >= 0 AND age <= 150 ~err_bad_age
  +return concat(name, \" is valid\")
";
        assert_eq!(
            eval_fn_result(source, "validate_user", r#"name="alice" age=25"#),
            r#"Ok("alice is valid")"#
        );
        let result = eval_fn_result(source, "validate_user", r#"name="" age=25"#);
        assert!(result.contains("err_empty_name"), "got: {result}");
        let result = eval_fn_result(source, "validate_user", r#"name="bob" age=-5"#);
        assert!(result.contains("err_bad_age"), "got: {result}");
    }

    // ═════════════════════════════════════════════════════════════════════
    // Match with nested variant bindings
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn match_nested_variant_bindings() {
        let source = "\
+type Tree = Leaf(Int) | Node(Tree, Tree)

+fn tree_sum (t:Tree)->Int
  +match t
  +case Leaf(val)
    +return val
  +case Node(left, right)
    +let l:Int = tree_sum(left)
    +let r:Int = tree_sum(right)
    +return l + r
  +end
";
        // Leaf(5) = 5
        assert_eq!(eval_fn_result(source, "tree_sum", "Leaf(5)"), "5");
        // Node(Leaf(3), Leaf(7)) = 10
        assert_eq!(eval_fn_result(source, "tree_sum", "Node(Leaf(3), Leaf(7))"), "10");
        // Node(Node(Leaf(1), Leaf(2)), Leaf(3)) = 6
        assert_eq!(eval_fn_result(source, "tree_sum", "Node(Node(Leaf(1), Leaf(2)), Leaf(3))"), "6");
    }

    // ═════════════════════════════════════════════════════════════════════
    // Architecture: fork_runtime_for_test isolation
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn fork_runtime_creates_isolated_shared_vars() {
        let source = "\
!module Counter
+shared count:Int = 0
+fn get_count ()->Int
  +return count
";
        let program = build_program(source);
        let routes = vec![];
        let forked = fork_runtime_for_test(&program, &routes);
        assert!(forked.is_some());
        let rt = forked.unwrap();
        let state = rt.read().unwrap();
        // Should have Counter.count initialized to 0
        assert_eq!(
            state.shared_vars.get("Counter.count").map(|v| format!("{v}")),
            Some("0".to_string()),
            "forked runtime should have Counter.count=0"
        );
    }

    #[test]
    fn fork_runtime_mutation_does_not_affect_original() {
        let source = "\
!module State
+shared value:Int = 10
+fn get ()->Int
  +return value
";
        let program = build_program(source);
        let routes = vec![];

        let forked1 = fork_runtime_for_test(&program, &routes).unwrap();
        let forked2 = fork_runtime_for_test(&program, &routes).unwrap();

        // Mutate forked1
        {
            let mut state = forked1.write().unwrap();
            state.shared_vars.insert("State.value".to_string(), Value::Int(99));
        }

        // forked2 should still have original value
        {
            let state = forked2.read().unwrap();
            assert!(matches!(
                state.shared_vars.get("State.value"),
                Some(Value::Int(10))
            ), "forked2 should be unaffected by forked1 mutation");
        }
    }

    #[test]
    fn fork_runtime_includes_http_routes() {
        let program = ast::Program::default();
        let routes = vec![ast::HttpRoute {
            method: "POST".to_string(),
            path: "/webhook".to_string(),
            handler_fn: "handle".to_string(),
        }];
        let forked = fork_runtime_for_test(&program, &routes).unwrap();
        let state = forked.read().unwrap();
        assert_eq!(state.http_routes.len(), 1);
        assert_eq!(state.http_routes[0].path, "/webhook");
    }

    #[test]
    fn fork_runtime_empty_program_no_shared_vars() {
        let program = ast::Program::default();
        let forked = fork_runtime_for_test(&program, &[]).unwrap();
        let state = forked.read().unwrap();
        assert!(state.shared_vars.is_empty());
        assert!(state.http_routes.is_empty());
    }

    // ═════════════════════════════════════════════════════════════════════
    // Architecture: +shared variable behavior
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn shared_var_populated_in_env() {
        let source = "\
!module Config
+shared debug:Bool = false
+shared max_retries:Int = 3
+fn get_retries ()->Int
  +return max_retries
";
        let program = build_program(source);
        let mut env = Env::new();
        env.populate_shared_from_program(&program);
        // Should have both shared vars in the cache
        assert!(matches!(
            env.shared_cache.get("Config.debug"),
            Some(Value::Bool(false))
        ));
        assert!(matches!(
            env.shared_cache.get("Config.max_retries"),
            Some(Value::Int(3))
        ));
    }

    #[test]
    fn shared_var_accessible_via_qualified_name() {
        // Shared vars are stored as "Module.name" keys
        let source = "\
!module App
+shared counter:Int = 42
+fn get ()->Int
  +return 0
";
        let program = build_program(source);
        let mut env = Env::new();
        env.populate_shared_from_program(&program);
        // Direct cache lookup with qualified key
        assert!(matches!(
            env.shared_cache.get("App.counter"),
            Some(Value::Int(42))
        ));
    }

    #[test]
    fn shared_var_multiple_modules() {
        let source = "\
!module A
+shared x:Int = 1
+fn get_x ()->Int
  +return x

!module B
+shared y:Int = 2
+fn get_y ()->Int
  +return y
";
        let program = build_program(source);
        let mut env = Env::new();
        env.populate_shared_from_program(&program);
        assert!(matches!(env.shared_cache.get("A.x"), Some(Value::Int(1))));
        assert!(matches!(env.shared_cache.get("B.y"), Some(Value::Int(2))));
    }

    #[test]
    #[ignore] // Run with: cargo test --release bench_vm_vs_treewalker -- --ignored --nocapture
    fn bench_vm_vs_treewalker() {
        use std::time::Instant;

        let source = r#"
!module Bench
+fn fib (n:Int)->Int
  +if n <= 1
    +return n
  +else
    +return fib(n - 1) + fib(n - 2)
  +end
+end

+fn sum_to (n:Int)->Int
  +let total:Int = 0
  +let i:Int = 0
  +while i <= n
    +set total = total + i
    +set i = i + 1
  +end
  +return total
+end

+fn collatz_steps (n:Int)->Int
  +let steps:Int = 0
  +let current:Int = n
  +while current != 1
    +if current % 2 == 0
      +set current = current / 2
    +else
      +set current = current * 3 + 1
    +end
    +set steps = steps + 1
  +end
  +return steps
+end

+fn nested_loops (n:Int)->Int
  +let total:Int = 0
  +let i:Int = 0
  +while i < n
    +let j:Int = 0
    +while j < n
      +set total = total + i * j
      +set j = j + 1
    +end
    +set i = i + 1
  +end
  +return total
+end
"#;

        let program = build_program(source);

        struct Bench {
            name: &'static str,
            input: &'static str,
            expected: i64,
        }

        let benches = vec![
            Bench { name: "Bench.fib", input: "25", expected: 75025 },
            Bench { name: "Bench.sum_to", input: "9000", expected: 40504500 },
            Bench { name: "Bench.collatz_steps", input: "27", expected: 111 },
            Bench { name: "Bench.nested_loops", input: "100", expected: 24502500 },
        ];

        println!("\n=== Adapsis VM vs Tree-Walker Benchmark ===\n");
        println!("{:<25} {:>12} {:>12} {:>10}", "Function", "Tree-Walk", "VM", "Speedup");
        println!("{}", "-".repeat(62));

        for b in &benches {
            let func = program.get_function(b.name).unwrap();
            let input_expr = crate::parser::parse_expr_pub(0, b.input).unwrap();

            // Tree-walker: warmup + measure
            let _ = eval_call_with_input(&program, b.name, &input_expr);
            let start = Instant::now();
            let tw_result = eval_call_with_input(&program, b.name, &input_expr).unwrap();
            let tw_time = start.elapsed();
            let tw_str = format!("{tw_result}");
            assert_eq!(tw_str, b.expected.to_string(), "tree-walk mismatch for {}", b.name);

            // VM: warmup + measure
            let vm_args_warmup = input_to_vm_args(&input_expr, func).unwrap();
            let compiled = crate::vm::compile_function(func, &program).unwrap();
            let _ = crate::vm::execute_with_io(&compiled, vm_args_warmup, &program, &|_, _| {
                anyhow::bail!("no IO")
            });
            let vm_args = input_to_vm_args(&input_expr, func).unwrap();
            let start = Instant::now();
            let vm_result = crate::vm::execute_with_io(&compiled, vm_args, &program, &|_, _| {
                anyhow::bail!("no IO")
            }).unwrap();
            let vm_time = start.elapsed();
            let vm_str = format!("{vm_result}");
            assert_eq!(vm_str, b.expected.to_string(), "VM mismatch for {}", b.name);

            let speedup = tw_time.as_secs_f64() / vm_time.as_secs_f64();
            println!(
                "{:<25} {:>10.2}ms {:>10.2}ms {:>9.1}x",
                b.name,
                tw_time.as_secs_f64() * 1000.0,
                vm_time.as_secs_f64() * 1000.0,
                speedup,
            );
        }
        println!();
    }

    // ═════════════════════════════════════════════════════════════════════
    // make_shared_program_mut / read_back_program_mutations
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn make_shared_program_mut_creates_writable_clone() {
        let mut program = crate::ast::Program::default();
        // Add a function to verify the clone works
        let ops = crate::parser::parse("+fn hello ()->String\n  +return \"hi\"\n+end")
            .expect("parse failed");
        for op in &ops {
            crate::validator::apply_and_validate(&mut program, op).unwrap();
        }
        program.rebuild_function_index();

        let shared = make_shared_program_mut(&program);

        // Should be able to read the program
        {
            let p = shared.read().unwrap();
            assert!(p.get_function("hello").is_some());
        }

        // Should be able to write to the program
        {
            let mut p = shared.write().unwrap();
            p.functions.clear();
            p.rebuild_function_index();
        }

        // Verify mutation took effect
        {
            let p = shared.read().unwrap();
            assert!(p.get_function("hello").is_none());
        }
    }

    #[test]
    fn read_back_returns_mutated_state() {
        let mut program = crate::ast::Program::default();
        let ops = crate::parser::parse("+fn hello ()->String\n  +return \"hi\"\n+end")
            .expect("parse failed");
        for op in &ops {
            crate::validator::apply_and_validate(&mut program, op).unwrap();
        }
        program.rebuild_function_index();

        let shared = make_shared_program_mut(&program);

        // Mutate via the lock
        {
            let mut p = shared.write().unwrap();
            let new_ops = crate::parser::parse("+fn goodbye ()->String\n  +return \"bye\"\n+end")
                .expect("parse failed");
            for op in &new_ops {
                crate::validator::apply_and_validate(&mut p, op).unwrap();
            }
            p.rebuild_function_index();
        }

        // Read back should see the new function
        let readback = read_back_program_mutations(&shared);
        assert!(readback.is_some());
        let p = readback.unwrap();
        assert!(p.get_function("hello").is_some(), "original function should still exist");
        assert!(p.get_function("goodbye").is_some(), "new function should exist after mutation");
    }

    #[test]
    fn read_back_returns_original_when_no_mutation() {
        let program = crate::ast::Program::default();
        let shared = make_shared_program_mut(&program);

        // No mutations performed
        let readback = read_back_program_mutations(&shared);
        assert!(readback.is_some(), "read_back should succeed even without mutations");
        assert!(readback.unwrap().functions.is_empty(), "should return empty program");
    }

    #[test]
    fn shared_program_mut_thread_local_roundtrip() {
        let mut program = crate::ast::Program::default();
        let ops = crate::parser::parse("+fn test_fn ()->Int\n  +return 42\n+end")
            .expect("parse failed");
        for op in &ops {
            crate::validator::apply_and_validate(&mut program, op).unwrap();
        }
        program.rebuild_function_index();

        let shared = make_shared_program_mut(&program);
        set_shared_program_mut(Some(shared.clone()));

        // get_shared_program_mut should return the same Arc
        let retrieved = get_shared_program_mut();
        assert!(retrieved.is_some(), "should retrieve program_mut from thread-local");

        // Mutate via the retrieved handle
        {
            let lock = retrieved.unwrap();
            let mut p = lock.write().unwrap();
            p.functions.clear();
            p.rebuild_function_index();
        }

        // read_back from the original shared handle should see the mutation
        let readback = read_back_program_mutations(&shared);
        assert!(readback.is_some());
        assert!(readback.unwrap().functions.is_empty(), "mutations should propagate through shared Arc");

        // Clean up
        set_shared_program_mut(None);
    }

    #[test]
    fn move_symbols_in_mutation_builtin_list() {
        // Verify that the is_mutation_builtin check in eval.rs includes move_symbols.
        // This is a structural test — we check the eval path by calling through
        // the actual code path (via the +await dispatch).
        //
        // We build a program with a function that does +await move_symbols(...),
        // set only the read-only snapshot (no set_shared_program_mut), and verify
        // that the eval.rs fallback creates the mutable wrapper automatically.
        let source = "+fn helper ()->String\n  +return \"hi\"\n+end";
        let ops = crate::parser::parse(source).expect("parse failed");
        let mut program = crate::ast::Program::default();
        for op in &ops {
            crate::validator::apply_and_validate(&mut program, op).unwrap();
        }
        program.rebuild_function_index();

        // Set only read-only snapshot — mutation builtins need the fallback
        set_shared_program(Some(std::sync::Arc::new(program.clone())));
        set_shared_program_mut(None);

        // Build a function that calls +await move_symbols("helper", "Utils")
        let fn_source = "+fn do_move ()->String [io,async]\n  +await result:String = move_symbols(\"helper\", \"Utils\")\n  +return result\n+end";
        let fn_ops = crate::parser::parse(fn_source).expect("parse fn failed");
        for op in &fn_ops {
            crate::validator::apply_and_validate(&mut program, op).unwrap();
        }
        program.rebuild_function_index();

        // Create a mock handle — move_symbols doesn't go through IO dispatch,
        // it's handled directly in execute_await, so a mock handle is fine
        let handle = crate::coroutine::CoroutineHandle::new_mock(vec![]);
        let mut env = Env::new();
        env.set("__coroutine_handle", Value::CoroutineHandle(handle));

        // Set the program again with the new function
        set_shared_program(Some(std::sync::Arc::new(program.clone())));
        // The is_mutation_builtin check in eval.rs should create the mutable wrapper
        // for move_symbols (it wouldn't before this fix)
        let result = eval_function_body_pub(&program, &program.get_function("do_move").unwrap().body, &mut env);

        // Should succeed — if move_symbols wasn't in the is_mutation_builtin list,
        // it would fail with "program not available (no async context)"
        assert!(result.is_ok(), "move_symbols should work via eval.rs mutation builtin fallback: {:?}", result.err());
        let val = result.unwrap();
        let val_str = format!("{val}");
        assert!(val_str.contains("moved") || val_str.contains("Utils"),
            "should confirm move: {val_str}");

        // Clean up
        set_shared_program(None);
        set_shared_program_mut(None);
    }

    // ── String interning tests ────────────────────────────────────────

    #[test]
    fn test_env_interned_set_get() {
        // Basic: set a variable and get it back via the interned Env
        let mut env = Env::new();
        env.set("x", Value::Int(42));
        let val = env.get("x").unwrap();
        assert!(matches!(val, Value::Int(42)));
    }

    #[test]
    fn test_env_interned_scope_shadowing() {
        // Inner scope shadows outer scope; after pop, outer is visible again
        let mut env = Env::new();
        env.set("x", Value::Int(1));
        env.push_scope();
        env.set("x", Value::Int(2));
        assert!(matches!(env.get("x").unwrap(), Value::Int(2)));
        env.pop_scope();
        assert!(matches!(env.get("x").unwrap(), Value::Int(1)));
    }

    #[test]
    fn test_env_interned_undefined_variable() {
        // Looking up a variable that doesn't exist should return an error
        let mut env = Env::new();
        let result = env.get("nonexistent");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("undefined variable"), "expected undefined variable error, got: {msg}");
    }

    #[test]
    fn test_env_interned_set_existing() {
        // set_existing should update the variable in the outer scope
        let mut env = Env::new();
        env.set("counter", Value::Int(0));
        env.push_scope();
        // set_existing walks scopes to find "counter" in the outer scope
        env.set_existing("counter", Value::Int(10));
        // Inner scope doesn't have "counter", so get() finds outer scope's updated value
        assert!(matches!(env.get("counter").unwrap(), Value::Int(10)));
        env.pop_scope();
        // After pop, the outer scope should have the updated value
        assert!(matches!(env.get("counter").unwrap(), Value::Int(10)));
    }

    #[test]
    fn test_env_interned_get_raw() {
        // get_raw returns None for missing variables instead of an error
        let mut env = Env::new();
        assert!(env.get_raw("missing").is_none());
        env.set("present", Value::Bool(true));
        assert!(env.get_raw("present").is_some());
    }

    #[test]
    fn test_env_interned_snapshot_bindings() {
        // snapshot_bindings should return name-value pairs, excluding __ prefixed
        let mut env = Env::new();
        env.set("x", Value::Int(1));
        env.set("__internal", Value::Int(999));
        env.set("y", Value::string("hello"));
        let bindings = env.snapshot_bindings();
        assert_eq!(bindings.len(), 2, "should exclude __internal");
        // Bindings are sorted by name
        assert_eq!(bindings[0].0, "x");
        assert_eq!(bindings[1].0, "y");
    }

    #[test]
    fn test_env_interned_multiple_variables() {
        // Verify multiple variables with different types work correctly
        let mut env = Env::new();
        env.set("a", Value::Int(1));
        env.set("b", Value::Float(2.5));
        env.set("c", Value::Bool(true));
        env.set("d", Value::string("hello"));
        env.set("e", Value::None);

        assert!(matches!(env.get("a").unwrap(), Value::Int(1)));
        assert!(matches!(env.get("b").unwrap(), Value::Float(f) if (f - 2.5).abs() < f64::EPSILON));
        assert!(matches!(env.get("c").unwrap(), Value::Bool(true)));
        assert!(matches!(env.get("d").unwrap(), Value::String(s) if s.as_str() == "hello"));
        assert!(matches!(env.get("e").unwrap(), Value::None));
    }

    #[test]
    fn test_interned_eval_function_with_variables() {
        // End-to-end: evaluate a function that uses local variables
        let source = "\
+fn compute (x:Int, y:Int)->Int
  +let a:Int = x + 1
  +let b:Int = y * 2
  +let result:Int = a + b
  +return result

!test compute
  +with x=3 y=4 -> expect 12
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);

        let (fn_name, case) = &cases[0];
        let result = eval_test_case(&program, fn_name, case);
        assert!(result.is_ok(), "interned eval should pass: {:?}", result);
    }

    #[test]
    fn test_interned_eval_nested_scopes() {
        // Test that nested scopes (if/each/while blocks) work with interning
        let source = "\
+fn nested (x:Int)->Int
  +let result:Int = 0
  +if x > 0
    +let inner:Int = x + 10
    +set result = inner
  +end
  +return result

!test nested
  +with x=5 -> expect 15
  +with x=0 -> expect 0
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 2);

        for (fn_name, case) in &cases {
            let result = eval_test_case(&program, fn_name, case);
            assert!(result.is_ok(), "nested scope test should pass: {:?}", result);
        }
    }

    #[test]
    fn test_intern_name_consistency() {
        // Verify that the intern_name helper returns consistent ids
        let mut env = Env::new();
        let id1 = env.intern_name("test_var");
        let id2 = env.intern_name("test_var");
        let id3 = env.intern_name("other_var");
        assert_eq!(id1, id2, "same string should get same id");
        assert_ne!(id1, id3, "different strings should get different ids");
    }

    #[test]
    fn test_resolve_name_roundtrip() {
        // Verify that intern → resolve roundtrips correctly
        let env = Env::new();
        let id = env.intern_name("roundtrip_test");
        let resolved = env.resolve_name(id);
        assert_eq!(resolved, "roundtrip_test");
    }

    // ═════════════════════════════════════════════════════════════════════
    // Program interner + Env::new_with_interner tests
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn test_program_intern_all_names() {
        // Verify that rebuild_function_index populates the program interner
        let source = "\
+fn add (a:Int, b:Int)->Int
  +let sum:Int = a + b
  +return sum
";
        let mut program = build_program(source);
        // rebuild_function_index is called by build_program, so the interner
        // should already contain all names.
        assert!(program.interner.get("add").is_some(), "function name should be interned");
        assert!(program.interner.get("a").is_some(), "param 'a' should be interned");
        assert!(program.interner.get("b").is_some(), "param 'b' should be interned");
        assert!(program.interner.get("sum").is_some(), "local var 'sum' should be interned");
        // Well-known names should also be interned
        assert!(program.interner.get("__coroutine_handle").is_some());
        assert!(program.interner.get("true").is_some());
        assert!(program.interner.get("false").is_some());
    }

    #[test]
    fn test_env_new_with_interner_seeded() {
        // Env created with new_with_interner should have the same interned ids
        // as the program's interner
        let source = "\
+fn greet (name:String)->String
  +return name
";
        let program = build_program(source);
        let env = Env::new_with_shared_interner(&program.shared_interner);

        // The env's interner should know about the program's names
        let id_from_program = program.interner.get("name").unwrap();
        let id_from_env = env.intern_name("name");
        assert_eq!(id_from_program, id_from_env, "interned ids should match between program and env");
    }

    #[test]
    fn test_env_set_id_get_id() {
        // Verify that set_id and get_id work correctly as fast-path methods
        let mut env = Env::new();
        let id = env.intern_name("fast_var");
        env.set_id(id, Value::Int(99));
        let val = env.get_id(id);
        assert!(val.is_some(), "get_id should find the value");
        assert!(matches!(val.unwrap(), Value::Int(99)));
    }

    #[test]
    fn test_env_set_id_scope_isolation() {
        // Values set via set_id in inner scope should not be visible after pop
        let mut env = Env::new();
        let id = env.intern_name("scoped_var");
        env.push_scope();
        env.set_id(id, Value::Int(42));
        assert!(env.get_id(id).is_some());
        env.pop_scope();
        assert!(env.get_id(id).is_none(), "value should not be visible after scope pop");
    }

    #[test]
    fn test_env_get_id_not_found() {
        // get_id should return None for unknown ids
        let env = Env::new();
        assert!(env.get_id(99999).is_none());
    }

    #[test]
    fn test_program_interner_with_modules() {
        // Verify that module names, module function names, and shared vars are interned
        let source = "\
!module Math

+fn square (n:Int)->Int
  +return n * n
";
        let program = build_program(source);
        assert!(program.interner.get("Math").is_some(), "module name should be interned");
        assert!(program.interner.get("square").is_some(), "module function name should be interned");
        assert!(program.interner.get("n").is_some(), "param 'n' should be interned");
    }

    #[test]
    fn test_interner_eval_function_with_program_interner() {
        // End-to-end: evaluate a function using Env seeded from program's interner
        let source = "\
+fn double (x:Int)->Int
  +let result:Int = x * 2
  +return result

!test double
  +with x=5 -> expect 10
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case(&program, fn_name, case);
        assert!(result.is_ok(), "eval with program interner should work: {:?}", result);
    }

    #[test]
    fn test_interner_consistency_across_envs() {
        // Two Envs seeded from the same program interner should produce
        // the same interned ids, allowing values to be portable between them
        let source = "+fn identity (v:Int)->Int\n  +return v\n";
        let program = build_program(source);
        let env1 = Env::new_with_shared_interner(&program.shared_interner);
        let env2 = Env::new_with_shared_interner(&program.shared_interner);
        let id1 = env1.intern_name("v");
        let id2 = env2.intern_name("v");
        assert_eq!(id1, id2, "same interner seed should produce same ids");
    }

    // ═════════════════════════════════════════════════════════════════════
    // SharedInterner + SmallVec scope chain optimization tests
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn env_shared_interner_o1_clone() {
        // Creating an Env from a SharedInterner should be O(1) — the Arc is
        // shared, not cloned. Verify that lookup still works correctly.
        let mut base = StringInterner::new();
        base.intern("x");
        base.intern("y");
        base.intern("z");
        let shared = base.shared();

        let mut env = Env::new_with_shared_interner(&shared);
        env.set("x", Value::Int(1));
        env.set("y", Value::Int(2));

        assert_eq!(format!("{}", env.get("x").unwrap()), "1");
        assert_eq!(format!("{}", env.get("y").unwrap()), "2");
        // z is interned but not set as a variable — get should return Err
        assert!(env.get("z").is_err(), "z was interned but not set");
        assert!(env.get("nonexistent").is_err());
    }

    #[test]
    fn env_shared_interner_cow_on_new_name() {
        // When a truly new name is encountered, the SharedInterner should
        // copy-on-write: existing Envs sharing the same Arc are unaffected.
        let mut base = StringInterner::new();
        base.intern("known");
        let shared = base.shared();

        let mut env1 = Env::new_with_shared_interner(&shared);
        let mut env2 = Env::new_with_shared_interner(&shared);

        // Intern a new name in env1 (triggers copy-on-write)
        env1.set("brand_new", Value::Int(99));
        assert_eq!(format!("{}", env1.get("brand_new").unwrap()), "99");

        // env2 should NOT see "brand_new" because it has its own interner copy
        // (but env2 can still intern it independently)
        env2.set("brand_new", Value::Int(77));
        assert_eq!(format!("{}", env2.get("brand_new").unwrap()), "77");
    }

    #[test]
    fn env_smallvec_scope_chain_basic() {
        // The scope chain uses SmallVec<[_; 4]> — verify push/pop works
        // correctly through nested scopes without heap allocation for
        // the typical case (≤4 scopes deep).
        let mut env = Env::new();
        env.set("outer", Value::Int(1));

        env.push_scope();
        env.set("inner1", Value::Int(2));
        assert_eq!(format!("{}", env.get("outer").unwrap()), "1");
        assert_eq!(format!("{}", env.get("inner1").unwrap()), "2");

        env.push_scope();
        env.set("inner2", Value::Int(3));
        assert_eq!(format!("{}", env.get("outer").unwrap()), "1");
        assert_eq!(format!("{}", env.get("inner2").unwrap()), "3");

        env.push_scope();
        env.set("inner3", Value::Int(4));
        // Still within SmallVec inline capacity (4 scopes: root + 3 nested)
        assert_eq!(format!("{}", env.get("inner3").unwrap()), "4");

        env.pop_scope();
        assert!(env.get("inner3").is_err()); // inner3 is gone

        env.pop_scope();
        assert!(env.get("inner2").is_err()); // inner2 is gone

        env.pop_scope();
        assert!(env.get("inner1").is_err()); // inner1 is gone
        assert_eq!(format!("{}", env.get("outer").unwrap()), "1"); // outer still there
    }

    #[test]
    fn env_smallvec_spills_to_heap_gracefully() {
        // When scope depth exceeds SmallVec inline capacity (4), it should
        // spill to heap and continue working correctly.
        let mut env = Env::new();
        env.set("root", Value::Int(0));

        // Push 10 scopes (well beyond inline capacity of 4)
        for i in 1..=10 {
            env.push_scope();
            env.set(&format!("level_{i}"), Value::Int(i as i64));
        }

        // All variables should be accessible
        assert_eq!(format!("{}", env.get("root").unwrap()), "0");
        for i in 1..=10 {
            assert_eq!(
                format!("{}", env.get(&format!("level_{i}")).unwrap()),
                format!("{i}")
            );
        }

        // Pop all nested scopes
        for _ in 1..=10 {
            env.pop_scope();
        }
        assert_eq!(format!("{}", env.get("root").unwrap()), "0");
        assert!(env.get("level_1").is_err());
    }

    #[test]
    fn env_scope_shadowing_with_smallvec() {
        // Variable shadowing should work correctly with SmallVec scopes
        let mut env = Env::new();
        env.set("x", Value::Int(1));

        env.push_scope();
        env.set("x", Value::Int(2)); // shadows outer x
        assert_eq!(format!("{}", env.get("x").unwrap()), "2");

        env.pop_scope();
        assert_eq!(format!("{}", env.get("x").unwrap()), "1"); // outer x restored
    }

    #[test]
    fn program_shared_interner_rebuilt_on_mutation() {
        // After rebuilding the function index, the shared_interner should
        // contain all AST names and be usable for Env creation.
        let source = "\
+fn add (a:Int, b:Int)->Int
  +let sum:Int = a + b
  +return sum
";
        let program = build_program(source);

        // shared_interner should be populated
        assert!(program.shared_interner.get("add").is_some());
        assert!(program.shared_interner.get("a").is_some());
        assert!(program.shared_interner.get("b").is_some());
        assert!(program.shared_interner.get("sum").is_some());

        // Creating an Env from shared_interner should work
        let mut env = Env::new_with_shared_interner(&program.shared_interner);
        env.set("a", Value::Int(1));
        assert_eq!(format!("{}", env.get("a").unwrap()), "1");
    }

    #[test]
    fn shared_interner_ids_match_base_interner() {
        // IDs from SharedInterner should match the original StringInterner
        let source = "\
+fn greet (name:String)->String
  +return concat(\"hello \", name)
";
        let program = build_program(source);

        let base_id = program.interner.get("name").unwrap();
        let shared_id = program.shared_interner.get("name").unwrap();
        assert_eq!(base_id, shared_id, "IDs should be identical between base and shared interner");
    }

    // ═════════════════════════════════════════════════════════════════════
    // Performance optimization tests
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn test_concat_prealloc_strings() {
        // concat with multiple string arguments should produce correct result
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, r#"concat("hello", " ", "world")"#), r#""hello world""#);
    }

    #[test]
    fn test_concat_prealloc_mixed_types() {
        // concat with mixed types (string, int, bool) should format correctly
        let p = ast::Program::default();
        let result = eval_expr_str(&p, r#"concat("count: ", 42)"#);
        assert_eq!(result, r#""count: 42""#);
    }

    #[test]
    fn test_concat_empty_args() {
        // concat with no arguments should return empty string
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, r#"concat()"#), r#""""#);
    }

    #[test]
    fn test_concat_single_arg() {
        // concat with a single argument
        let p = ast::Program::default();
        assert_eq!(eval_expr_str(&p, r#"concat("solo")"#), r#""solo""#);
    }

    #[test]
    fn test_push_returns_extended_list() {
        // push should return a new list with the item appended
        let source = "\
+fn test_push ()->Int
  +let xs:List<Int> = list(1, 2, 3)
  +let ys:List<Int> = push(xs, 4)
  +return len(ys)

!test test_push
  +with -> expect 4
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case(&program, fn_name, case);
        assert!(result.is_ok(), "push should work: {:?}", result);
    }

    #[test]
    fn test_push_error_wrong_args() {
        // push with wrong number of args should error
        let p = ast::Program::default();
        let mut env = Env::new();
        let result = eval_builtin_or_user(&p, "push", vec![Value::list(vec![])], &mut env);
        assert!(result.is_err(), "push with 1 arg should error");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("expects 2 arguments"), "error should mention arg count: {msg}");
    }

    #[test]
    fn test_push_error_not_list() {
        // push on a non-list should error
        let p = ast::Program::default();
        let mut env = Env::new();
        let result = eval_builtin_or_user(
            &p,
            "push",
            vec![Value::Int(42), Value::Int(1)],
            &mut env,
        );
        assert!(result.is_err(), "push on non-list should error");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("push expects"), "error should mention push: {msg}");
    }

    #[test]
    fn test_push_preserves_original_list_semantics() {
        // In Adapsis, push returns a new list; verify the function semantics
        // work correctly in a loop accumulation pattern
        let source = "\
+fn accumulate (n:Int)->Int
  +let result:List<Int> = list()
  +let i:Int = 0
  +while i < n
    +set result = push(result, i)
    +set i = i + 1
  +end
  +return len(result)

!test accumulate
  +with n=4 -> expect 4
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case(&program, fn_name, case);
        assert!(result.is_ok(), "accumulate with push should work: {:?}", result);
    }

    #[test]
    fn test_module_function_lookup_with_index() {
        // Module-qualified function lookup should work after rebuild_function_index
        let source = "\
!module Math
+fn add (a:Int, b:Int)->Int
  +return a + b

!test Math.add
  +with a=3 b=4 -> expect 7
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case(&program, fn_name, case);
        assert!(result.is_ok(), "module function lookup should work: {:?}", result);
    }

    #[test]
    fn test_module_function_lookup_not_found() {
        // Looking up a non-existent module function should return None
        let mut program = ast::Program::default();
        program.rebuild_function_index();
        assert!(program.get_function("NonExistent.func").is_none());
    }

    // ═════════════════════════════════════════════════════════════════════
    // Name interning optimization tests
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn env_new_with_interner_seeds_names() {
        // When Env is created with a pre-populated interner, variable lookups
        // for pre-interned names should use cache hits (no new allocations).
        let mut interner = StringInterner::new();
        let id_x = interner.intern("x");
        let id_y = interner.intern("y");

        let mut env = Env::new_with_interner(&interner);
        env.set("x", Value::Int(42));
        env.set("y", Value::Int(99));

        // Look up by name — should hit the pre-interned cache
        assert!(matches!(env.get("x"), Ok(Value::Int(42))));
        assert!(matches!(env.get("y"), Ok(Value::Int(99))));

        // Look up by pre-interned id — fast path
        assert!(matches!(env.get_id(id_x), Some(Value::Int(42))));
        assert!(matches!(env.get_id(id_y), Some(Value::Int(99))));
    }

    #[test]
    fn env_new_with_interner_handles_unknown_names() {
        // Names not in the pre-seeded interner should still work (interned on demand)
        let interner = StringInterner::new(); // empty interner
        let mut env = Env::new_with_interner(&interner);
        env.set("dynamic_var", Value::string("hello"));

        assert!(matches!(env.get("dynamic_var"), Ok(Value::String(s)) if s.as_str() == "hello"));
    }

    #[test]
    fn env_undefined_variable_returns_error() {
        let mut env = Env::new();
        let result = env.get("nonexistent");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("undefined variable"),
            "error should mention 'undefined variable', got: {err_msg}"
        );
    }

    #[test]
    fn env_set_id_and_get_id_roundtrip() {
        let mut interner = StringInterner::new();
        let id = interner.intern("counter");
        let mut env = Env::new_with_interner(&interner);

        env.set_id(id, Value::Int(0));
        assert!(matches!(env.get_id(id), Some(Value::Int(0))));

        // Update via set_id
        env.set_id(id, Value::Int(1));
        assert!(matches!(env.get_id(id), Some(Value::Int(1))));
    }

    #[test]
    fn union_variant_hashset_used_in_eval() {
        // Verify that is_union_variant uses the HashSet-based lookup
        let source = "\
+type Color = Red | Green | Blue
+fn get_color () -> Color
  +return Red
";
        let program = build_program(source);
        // The HashSet should contain the variants after rebuild
        assert!(program.is_union_variant("Red"));
        assert!(program.is_union_variant("Green"));
        assert!(program.is_union_variant("Blue"));
        assert!(!program.is_union_variant("Yellow"));
    }

    #[test]
    fn user_function_call_uses_interned_env() {
        // When a user function is called via eval_builtin_or_user, the child Env
        // should be seeded with the program's interner for fast param lookups.
        let source = "\
+fn double (n:Int) -> Int
  +return n + n

!test double
  +with n=5 -> expect 10
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case(&program, fn_name, case);
        assert!(result.is_ok(), "user function call with interned env should work: {:?}", result);
    }

    // ═════════════════════════════════════════════════════════════════════
    // Arc-based Value optimisation tests
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn value_string_clone_is_cheap_arc_bump() {
        // Cloning a Value::String should share the same Arc allocation
        let v1 = Value::string("hello world");
        let v2 = v1.clone();
        // Both should point to the same underlying str (same Arc)
        match (&v1, &v2) {
            (Value::String(a), Value::String(b)) => {
                assert!(Arc::ptr_eq(a, b), "clone should share the Arc, not allocate a new string");
            }
            _ => panic!("expected String variants"),
        }
    }

    #[test]
    fn value_err_clone_preserves_value() {
        let v1 = Value::Err("some error".to_string());
        let v2 = v1.clone();
        match (&v1, &v2) {
            (Value::Err(a), Value::Err(b)) => {
                assert_eq!(a, b, "Err clone should preserve the error message");
            }
            _ => panic!("expected Err variants"),
        }
    }

    #[test]
    fn value_list_clone_shares_arc() {
        let items = vec![Value::Int(1), Value::Int(2), Value::Int(3)];
        let v1 = Value::list(items);
        let v2 = v1.clone();
        match (&v1, &v2) {
            (Value::List(a), Value::List(b)) => {
                assert!(Arc::ptr_eq(a, b), "List clone should share the Arc<Vec<Value>>");
            }
            _ => panic!("expected List variants"),
        }
    }

    #[test]
    fn value_struct_clone_shares_field_arc() {
        let mut fields = HashMap::new();
        fields.insert("x".to_string(), Value::Int(10));
        fields.insert("y".to_string(), Value::Int(20));
        let v1 = Value::strct("Point", fields);
        let v2 = v1.clone();
        match (&v1, &v2) {
            (Value::Struct(n1, f1), Value::Struct(n2, f2)) => {
                assert_eq!(n1, n2, "Struct name should be preserved");
                assert!(Arc::ptr_eq(f1, f2), "Struct fields Arc should be shared");
            }
            _ => panic!("expected Struct variants"),
        }
    }

    #[test]
    fn value_union_variant_clone_preserves_value() {
        let v1 = Value::Union {
            variant: intern::intern_display("Some"),
            payload: vec![Value::Int(42)],
        };
        let v2 = v1.clone();
        match (&v1, &v2) {
            (Value::Union { variant: a, .. }, Value::Union { variant: b, .. }) => {
                assert_eq!(a, b, "Union variant name should be preserved");
            }
            _ => panic!("expected Union variants"),
        }
    }

    #[test]
    fn value_string_display_correct() {
        let v = Value::string("hello");
        assert_eq!(format!("{v}"), r#""hello""#);
    }

    #[test]
    fn value_err_display_correct() {
        let v = Value::Err("fail".to_string());
        assert_eq!(format!("{v}"), "Err(fail)");
    }

    #[test]
    fn value_struct_field_lookup_with_arc_keys() {
        // Ensure struct field access works with Arc<str> keys
        let source = "\
+type Point = x:Int, y:Int

+fn get_x (p:Point) -> Int
  +return p.x

!test get_x
  +with x=10 y=20 -> expect 10
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case(&program, fn_name, case);
        assert!(result.is_ok(), "struct field access with Arc keys should work: {:?}", result);
    }

    #[test]
    fn value_list_operations_with_arc_wrapper() {
        // push, get, len should all work through Arc<Vec<Value>>
        let source = "\
+fn list_ops ()->Int
  +let items:List<Int> = list(1, 2, 3)
  +let items2:List<Int> = push(items, 4)
  +return len(items2)
+end

!test list_ops
  +with -> expect 4
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case(&program, fn_name, case);
        assert!(result.is_ok(), "list operations through Arc<Vec> should work: {:?}", result);
    }

    #[test]
    fn value_string_builtin_ops_through_arc() {
        // String builtins (concat, len, split, trim, etc.) should work with Arc<str>
        let source = "\
+fn string_ops (s:String) -> String
  +let trimmed:String = trim(s)
  +let upper:String = concat(trimmed, \"!\")
  +return upper

!test string_ops
  +with s=\"  hello  \" -> expect \"hello!\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case(&program, fn_name, case);
        assert!(result.is_ok(), "string builtins through Arc<str> should work: {:?}", result);
    }

    #[test]
    fn value_matches_equality_with_arc() {
        // Value::matches() should work correctly with Arc<str> internals
        let a = Value::string("hello");
        let b = Value::string("hello");
        let c = Value::string("world");
        assert!(a.matches(&b), "same content Arc<str> should match");
        assert!(!a.matches(&c), "different content should not match");

        // Err matching
        let ea = Value::Err("fail".to_string());
        let eb = Value::Err("fail".to_string());
        let ec = Value::Err("other".to_string());
        assert!(ea.matches(&eb));
        assert!(!ea.matches(&ec));

        // Struct matching
        let mut f1 = HashMap::new();
        f1.insert("x".to_string(), Value::Int(1));
        let mut f2 = HashMap::new();
        f2.insert("x".to_string(), Value::Int(1));
        let s1 = Value::strct("P", f1);
        let s2 = Value::strct("P", f2);
        assert!(s1.matches(&s2));

        // List matching
        let l1 = Value::list(vec![Value::Int(1), Value::Int(2)]);
        let l2 = Value::list(vec![Value::Int(1), Value::Int(2)]);
        let l3 = Value::list(vec![Value::Int(1), Value::Int(3)]);
        assert!(l1.matches(&l2));
        assert!(!l1.matches(&l3));
    }

    #[test]
    fn value_is_truthy_with_arc_string() {
        assert!(Value::string("x").is_truthy());
        assert!(!Value::string("").is_truthy());
    }

    #[test]
    fn value_match_pattern_with_arc_union() {
        // +match on union variants should work correctly with Arc<str> variant names
        let source = "\
+type Color = Red | Green | Blue

+fn color_name (c:Color)->String
  +match c
  +case Red
    +return \"red\"
  +case Green
    +return \"green\"
  +case Blue
    +return \"blue\"
  +end
+end

!test color_name
  +with c=Red -> expect \"red\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case(&program, fn_name, case);
        assert!(result.is_ok(), "union match with Arc variant names should work: {:?}", result);
    }

    #[test]
    fn value_each_loop_with_arc_list() {
        // +each over a List should work with Arc<Vec<Value>> wrapper
        let source = "\
+fn sum_list (items:List<Int>)->Int
  +let total:Int = 0
  +each items item:Int
    +set total = total + item
  +end
  +return total
+end

!test sum_list
  +with items=list(1, 2, 3) -> expect 6
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case(&program, fn_name, case);
        assert!(result.is_ok(), "+each with Arc<Vec> should work: {:?}", result);
    }

    // ═════════════════════════════════════════════════════════════════════
    // Function dispatch with interned fn_index / module_index
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn interned_fn_index_dispatch_top_level() {
        // End-to-end: calling a top-level user function should dispatch through
        // the interned fn_index, not string comparison.
        let source = "\
+fn triple (n:Int)->Int
  +return n * 3

!test triple
  +with n=7 -> expect 21
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case(&program, fn_name, case);
        assert!(result.is_ok(), "top-level function dispatch via interned index: {:?}", result);
    }

    #[test]
    fn interned_fn_index_dispatch_module_qualified() {
        // Module-qualified function call dispatches through interned module_index + fn_index.
        let source = "\
!module Calc

+fn square (n:Int)->Int
  +return n * n

!test Calc.square
  +with n=6 -> expect 36
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case(&program, fn_name, case);
        assert!(result.is_ok(), "module function dispatch via interned index: {:?}", result);
    }

    #[test]
    fn interned_fn_index_cross_function_call() {
        // Function A calls function B — both dispatched through interned indices.
        let source = "\
+fn helper (x:Int)->Int
  +return x + 1

+fn main_fn (x:Int)->Int
  +call y:Int = helper(x)
  +return y * 2

!test main_fn
  +with x=4 -> expect 10
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case(&program, fn_name, case);
        assert!(result.is_ok(), "cross-function dispatch via interned index: {:?}", result);
    }

    #[test]
    fn interned_union_variant_lookup_in_match() {
        // Union variant dispatch uses interned HashSet<InternedId> via is_union_variant.
        let source = "\
+type Shape = Circle(Int) | Square(Int)

+fn area (s:Shape)->Int
  +match s
  +case Circle(r)
    +return r * r * 3
  +case Square(side)
    +return side * side
  +end

!test area
  +with s=Circle(5) -> expect 75
  +with s=Square(4) -> expect 16
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        for (fn_name, case) in &cases {
            let result = eval_test_case(&program, fn_name, case);
            assert!(result.is_ok(), "union variant dispatch via interned set: {:?}", result);
        }
    }

    #[test]
    fn interned_fn_dispatch_unknown_function_error() {
        // Calling a non-existent function should produce a clear error, not panic.
        let program = build_program("+fn noop ()->Int\n  +return 0\n");
        let mut env = Env::new_with_shared_interner(&program.shared_interner);
        let result = eval_builtin_or_user(&program, "nonexistent_fn", vec![], &mut env);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("undefined function"), "should say undefined: {msg}");
    }

    // ═════════════════════════════════════════════════════════════════════
    // Interned struct fields / union variant tests
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn interned_struct_field_keys_are_u32() {
        // Value::Struct should use InternedId (u32) keys, not String.
        let mut fields = HashMap::new();
        fields.insert("x".to_string(), Value::Int(10));
        fields.insert("y".to_string(), Value::Int(20));
        let val = Value::strct("Point", fields);
        match &val {
            Value::Struct(name_id, field_map) => {
                // Keys should be u32 (InternedId)
                assert_eq!(field_map.len(), 2);
                let x_id = intern::intern_display("x");
                let y_id = intern::intern_display("y");
                assert!(matches!(field_map.get(&x_id), Some(Value::Int(10))), "expected x=10");
                assert!(matches!(field_map.get(&y_id), Some(Value::Int(20))), "expected y=20");
                // Name should resolve back to "Point"
                assert_eq!(intern::resolve_display(*name_id), "Point");
            }
            _ => panic!("expected Struct"),
        }
    }

    #[test]
    fn interned_struct_display_resolves_names() {
        // Value::Struct Display should render field names correctly via the display interner.
        let mut fields = HashMap::new();
        fields.insert("name".to_string(), Value::string("alice"));
        fields.insert("age".to_string(), Value::Int(30));
        let val = Value::strct("User", fields);
        let display = format!("{val}");
        assert!(display.contains("User{"), "display should show struct name: {display}");
        assert!(display.contains("name:"), "display should show field 'name': {display}");
        assert!(display.contains("age:"), "display should show field 'age': {display}");
        assert!(display.contains("alice"), "display should show field value: {display}");
        assert!(display.contains("30"), "display should show field value: {display}");
    }

    #[test]
    fn interned_union_variant_display() {
        // Union variant Display should resolve the interned variant name.
        let val = Value::Union {
            variant: intern::intern_display("Some"),
            payload: vec![Value::Int(42)],
        };
        let display = format!("{val}");
        assert_eq!(display, "Some(42)");
    }

    #[test]
    fn interned_union_no_payload_display() {
        let val = Value::Union {
            variant: intern::intern_display("None"),
            payload: vec![],
        };
        let display = format!("{val}");
        assert_eq!(display, "None");
    }

    #[test]
    fn interned_struct_get_field_by_string() {
        // Value::get_field should look up by string name via the display interner.
        let mut fields = HashMap::new();
        fields.insert("x".to_string(), Value::Int(42));
        let val = Value::strct("P", fields);
        assert!(matches!(val.get_field("x"), Some(Value::Int(42))), "should find field x");
        assert!(val.get_field("y").is_none(), "should not find field y");
        assert!(val.get_field("nonexistent").is_none(), "should not find nonexistent field");
    }

    #[test]
    fn interned_struct_field_access_in_eval() {
        // Full eval roundtrip: struct init → field access using interned keys
        let source = "\
+type Config = host:String, port:Int

+fn get_port (c:Config) -> Int
  +return c.port

!test get_port
  +with host=\"localhost\" port=8080 -> expect 8080
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case(&program, fn_name, case);
        assert!(result.is_ok(), "interned struct field access: {:?}", result);
        let result_str = result.unwrap();
        assert!(result_str.contains("8080"), "result should contain 8080: {result_str}");
    }

    #[test]
    fn interned_union_match_dispatch() {
        // Full eval roundtrip: union variant construction → match dispatch using interned IDs
        let source = "\
+type Result = Success(Int) | Failure(String)

+fn unwrap_result (r:Result) -> Int
  +match r
  +case Success(val)
    +return val
  +case Failure(msg)
    +return -1
  +end

!test unwrap_result
  +with r=Success(42) -> expect 42
  +with r=Failure(\"oops\") -> expect -1
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        for (fn_name, case) in &cases {
            let result = eval_test_case(&program, fn_name, case);
            assert!(result.is_ok(), "interned union match dispatch: {:?}", result);
        }
    }

    #[test]
    fn interned_struct_empty_name_matches_any() {
        // Struct with empty name (anonymous) should match any named struct
        let mut f1 = HashMap::new();
        f1.insert("x".to_string(), Value::Int(1));
        let named = Value::strct("Point", f1);

        let mut f2: HashMap<InternedId, Value> = HashMap::new();
        f2.insert(intern::intern_display("x"), Value::Int(1));
        let anon = Value::strct_interned(intern::intern_display(""), f2);

        assert!(named.matches(&anon), "named struct should match anonymous");
        assert!(anon.matches(&named), "anonymous struct should match named");
    }

    #[test]
    fn interned_struct_different_names_dont_match() {
        let mut f1 = HashMap::new();
        f1.insert("x".to_string(), Value::Int(1));
        let s1 = Value::strct("A", f1);

        let mut f2 = HashMap::new();
        f2.insert("x".to_string(), Value::Int(1));
        let s2 = Value::strct("B", f2);

        assert!(!s1.matches(&s2), "different named structs should not match");
    }

    #[test]
    fn interned_union_variant_equality() {
        // Two unions with the same interned variant should be equal
        let v1 = Value::Union {
            variant: intern::intern_display("Ok"),
            payload: vec![Value::Int(1)],
        };
        let v2 = Value::Union {
            variant: intern::intern_display("Ok"),
            payload: vec![Value::Int(1)],
        };
        assert!(v1.matches(&v2));

        // Different variants should not match
        let v3 = Value::Union {
            variant: intern::intern_display("Err"),
            payload: vec![Value::Int(1)],
        };
        assert!(!v1.matches(&v3));
    }

    #[test]
    fn interned_struct_strct_interned_roundtrip() {
        // strct_interned should produce identical values to strct
        let mut string_fields = HashMap::new();
        string_fields.insert("a".to_string(), Value::Int(1));
        string_fields.insert("b".to_string(), Value::Int(2));
        let via_strct = Value::strct("T", string_fields);

        let mut interned_fields: HashMap<InternedId, Value> = HashMap::new();
        interned_fields.insert(intern::intern_display("a"), Value::Int(1));
        interned_fields.insert(intern::intern_display("b"), Value::Int(2));
        let via_interned = Value::strct_interned(intern::intern_display("T"), interned_fields);

        assert!(via_strct.matches(&via_interned), "strct and strct_interned should produce matching values");
    }

    #[test]
    fn interned_display_interner_fallback_for_unknown_id() {
        // resolve_display should return a fallback for unknown IDs
        let result = intern::resolve_display(999_999);
        assert!(result.starts_with("<id:"), "unknown ID should get fallback: {result}");
    }

    // ── Async multi-parameter tests (regression for display interner bug) ───

    #[test]
    fn test_async_multi_param_function_binds_params_via_runtime() {
        // Regression test: async functions with multiple parameters previously
        // failed with "undefined variable" because the display interner was
        // not installed on the worker thread spawned by
        // eval_test_case_with_runtime. bind_input_to_params uses
        // intern::resolve_display / intern::intern_display to match struct
        // field names to parameter names, which requires the thread-local
        // display interner to be populated.
        let source = "\
+fn fetch_issues (owner:String, repo:String)->String [io,async]
  +await resp:String = http_get(concat(\"https://api.github.com/repos/\", owner, \"/\", repo, \"/issues\"))
  +return resp
";
        let program = build_program(source);

        let test_source = "\
!test fetch_issues
  +with owner=\"torvalds\" repo=\"linux\" -> expect \"issues_json\"
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];

        let mocks = vec![IoMock {
            operation: "http_get".to_string(),
            patterns: vec!["api.github.com".to_string()],
            response: "issues_json".to_string(),
        }];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
        assert!(
            result.is_ok(),
            "async multi-param test should pass (display interner must be set on worker thread): {:?}",
            result
        );
    }

    #[tokio::test]
    async fn test_async_multi_param_function_binds_params_via_spawn_blocking() {
        // Same regression test but through eval_test_case_async (the
        // spawn_blocking path). Ensures the display interner is installed
        // on the tokio blocking thread pool thread as well.
        let source = "\
+fn fetch_issues (owner:String, repo:String)->String [io,async]
  +await resp:String = http_get(concat(\"https://api.github.com/repos/\", owner, \"/\", repo, \"/issues\"))
  +return resp
";
        let program = build_program(source);

        let test_source = "\
!test fetch_issues
  +with owner=\"torvalds\" repo=\"linux\" -> expect \"issues_json\"
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];

        let mocks = vec![IoMock {
            operation: "http_get".to_string(),
            patterns: vec!["api.github.com".to_string()],
            response: "issues_json".to_string(),
        }];

        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let result = eval_test_case_async(&program, fn_name, case, &mocks, tx, &[]).await;
        assert!(
            result.is_ok(),
            "async multi-param test via eval_test_case_async should pass: {:?}",
            result
        );
    }

    #[test]
    fn test_async_multi_param_wrong_expected_fails() {
        // Error case: async multi-param function test with wrong expected
        // value should fail cleanly (not with "undefined variable").
        let source = "\
+fn fetch_issues (owner:String, repo:String)->String [io,async]
  +await resp:String = http_get(concat(\"https://api.github.com/repos/\", owner, \"/\", repo, \"/issues\"))
  +return resp
";
        let program = build_program(source);

        let test_source = "\
!test fetch_issues
  +with owner=\"torvalds\" repo=\"linux\" -> expect \"wrong_value\"
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];

        let mocks = vec![IoMock {
            operation: "http_get".to_string(),
            patterns: vec!["api.github.com".to_string()],
            response: "actual_issues".to_string(),
        }];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
        assert!(
            result.is_err(),
            "async test with wrong expected should fail"
        );
        let err = result.unwrap_err().to_string();
        // The error should be a value mismatch, NOT "undefined variable"
        assert!(
            !err.contains("undefined variable"),
            "error should NOT be 'undefined variable' (params should bind correctly): {err}"
        );
    }

    #[tokio::test]
    async fn test_async_session_multi_param_function() {
        // End-to-end test through the session flow: define an async function
        // with multiple parameters, add mocks, and run tests.
        let mut session = crate::session::Session::new();

        let define_source = "\
+fn fetch_issues (owner:String, repo:String)->String [io,async]
  +await resp:String = http_get(concat(\"https://api.github.com/repos/\", owner, \"/\", repo, \"/issues\"))
  +return resp
";
        let results = session.apply_async(define_source, None).await;
        assert!(results.is_ok(), "define should succeed: {:?}", results);

        let mock_source = "!mock http_get \"api.github.com\" -> \"session_issues\"";
        let results = session.apply_async(mock_source, None).await;
        assert!(results.is_ok(), "mock should succeed: {:?}", results);

        let test_source = "\
!test fetch_issues
  +with owner=\"torvalds\" repo=\"linux\" -> expect \"session_issues\"
";
        let results = session.apply_async(test_source, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(
            results[0].1,
            "async multi-param test via session should pass: {:?}",
            results[0]
        );
    }

    // ── Value accessor / constructor tests ──────────────────────────

    #[test]
    fn value_string_uses_arc_for_cheap_clone() {
        let v = Value::string("hello");
        let v2 = v.clone();
        // Both clones share the same Arc allocation
        if let (Value::String(a), Value::String(b)) = (&v, &v2) {
            assert!(Arc::ptr_eq(a, b), "clone should share Arc pointer");
        } else {
            panic!("expected String variants");
        }
    }

    #[test]
    fn value_list_uses_arc_for_cheap_clone() {
        let v = Value::list(vec![Value::Int(1), Value::Int(2)]);
        let v2 = v.clone();
        if let (Value::List(a), Value::List(b)) = (&v, &v2) {
            assert!(Arc::ptr_eq(a, b), "clone should share Arc pointer");
        } else {
            panic!("expected List variants");
        }
    }

    #[test]
    fn value_as_str_returns_inner_slice() {
        let v = Value::string("café");
        assert_eq!(v.as_str(), Some("café"));
        assert_eq!(Value::Int(42).as_str(), None);
        assert_eq!(Value::None.as_str(), None);
    }

    #[test]
    fn value_as_list_returns_inner_slice() {
        let v = Value::list(vec![Value::Int(1), Value::string("x")]);
        let slice = v.as_list().unwrap();
        assert_eq!(slice.len(), 2);
        assert!(matches!(&slice[0], Value::Int(1)));
        assert!(Value::Int(0).as_list().is_none());
    }

    #[test]
    fn value_as_list_mut_cow_semantics() {
        // When there's only one Arc reference, as_list_mut should not clone
        let mut v = Value::list(vec![Value::Int(10)]);
        {
            let inner = v.as_list_mut().unwrap();
            inner.push(Value::Int(20));
        }
        let slice = v.as_list().unwrap();
        assert_eq!(slice.len(), 2);
        assert!(matches!(&slice[1], Value::Int(20)));

        // When there are multiple Arc references, as_list_mut should CoW-clone
        let v2 = v.clone();
        {
            let inner = v.as_list_mut().unwrap();
            inner.push(Value::Int(30));
        }
        // v was mutated (now has 3 elements), v2 still has 2
        assert_eq!(v.as_list().unwrap().len(), 3);
        assert_eq!(v2.as_list().unwrap().len(), 2);
    }

    #[test]
    fn value_as_list_mut_returns_none_for_non_list() {
        let mut v = Value::Int(5);
        assert!(v.as_list_mut().is_none());
    }

    // ── Shared variable resolution in nested function calls ───────────

    #[test]
    fn shared_var_accessible_in_nested_call_during_test() {
        // Function A calls function B, both in the same module.
        // B accesses a shared variable. When testing A, B's shared
        // variable access must still work.
        let source = "\
!module Counter
+shared count:Int = 10
+fn get_count ()->Int
  +return count
+end

+fn doubled_count ()->Int
  +return get_count() * 2
+end
";
        let program = build_program(source);
        let mut env = Env::new();
        env.populate_shared_from_program(&program);

        // Test via eval_function_body_named (mimics test runner)
        let func = program.get_function("Counter.doubled_count").unwrap();
        FN_NAME_STACK.with(|s| s.borrow_mut().push("Counter.doubled_count".to_string()));
        let result = eval_function_body(&program, &func.body, &mut env).unwrap();
        FN_NAME_STACK.with(|s| s.borrow_mut().pop());

        // get_count() is called unqualified from doubled_count —
        // the fix qualifies it to "Counter.get_count" on FN_NAME_STACK
        // so `count` resolves as "Counter.count" in the shared cache.
        assert!(matches!(result, Value::Int(20)), "expected 20, got {result}");
    }

    #[test]
    fn shared_var_set_persists_across_eval_calls() {
        let source = r#"
!module Bot
+shared bot_token:String = ""
+fn set_token (raw:String)->String [mut]
  +let trimmed:String = trim(raw)
  +set bot_token = trimmed
  +return bot_token
+end

+fn auth_header ()->String
  +return concat("Bearer ", bot_token)
+end
"#;
        let program = build_program(source);
        let runtime = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState::default()));
        set_shared_runtime(Some(runtime.clone()));
        set_shared_program(Some(std::sync::Arc::new(program.clone())));

        let set_result = eval_call_with_input(&program, "Bot.set_token", &parser::Expr::String("  abc123  ".to_string())).unwrap();
        assert_eq!(set_result, "\"abc123\"");

        let header_result = eval_call_with_input(&program, "Bot.auth_header", &parser::Expr::StructLiteral(vec![])).unwrap();
        assert_eq!(header_result, "\"Bearer abc123\"");

        let state = runtime.read().unwrap();
        assert!(matches!(state.shared_vars.get("Bot.bot_token"), Some(Value::String(s)) if s.as_str() == "abc123"));
    }

    #[test]
    fn shared_var_set_and_read_work_in_test_context() {
        let source = r#"
!module Bot
+shared bot_token:String = ""
+fn configure_and_read (raw:String)->String [mut]
  +let trimmed:String = trim(raw)
  +set bot_token = trimmed
  +return concat("Bearer ", bot_token)
+end

!test Bot.configure_and_read
  +with "  xyz  " -> expect "Bearer xyz"
"#;
        let program = build_program(source);
        let case = extract_test_cases(source).remove(0).1;
        let result = eval_test_case(&program, "Bot.configure_and_read", &case).unwrap();
        assert!(result.contains("expected \"Bearer xyz\""), "unexpected test result: {result}");
    }

    #[test]
    fn undeclared_shared_var_still_errors() {
        let source = r#"
!module Bot
+fn auth_header ()->String
  +return concat("Bearer ", bot_token)
+end
"#;
        let program = build_program(source);
        set_shared_program(Some(std::sync::Arc::new(program.clone())));

        let result = eval_call_with_input(&program, "Bot.auth_header", &parser::Expr::StructLiteral(vec![])).unwrap();
        assert!(result.contains("undefined variable `bot_token`"), "expected undefined variable error, got: {result}");
    }

    #[test]
    fn shared_var_nested_call_different_module_no_cross_leak() {
        // Function in module A calls function in module B.
        // Module B's function should see module B's shared vars,
        // NOT module A's shared vars with the same name.
        let source = "\
!module Alpha
+shared val:Int = 100

+fn get_both ()->Int
  +return val + Beta.get_val()
+end

!module Beta
+shared val:Int = 5

+fn get_val ()->Int
  +return val
+end
";
        let program = build_program(source);
        let mut env = Env::new();
        env.populate_shared_from_program(&program);

        let func = program.get_function("Alpha.get_both").unwrap();
        FN_NAME_STACK.with(|s| s.borrow_mut().push("Alpha.get_both".to_string()));
        let result = eval_function_body(&program, &func.body, &mut env).unwrap();
        FN_NAME_STACK.with(|s| s.borrow_mut().pop());

        // Alpha.val=100 + Beta.val=5 = 105
        assert!(matches!(result, Value::Int(105)), "expected 105, got {result}");
    }

    #[test]
    fn shared_var_inaccessible_from_wrong_module() {
        // A function in module A should NOT be able to access module B's
        // shared variables via bare name.
        let source = "\
!module OnlyHere
+shared secret:Int = 42
+fn get_secret ()->Int
  +return secret
+end

!module Other
+fn try_access ()->Int
  +return secret
+end
";
        let program = build_program(source);
        let mut env = Env::new();
        env.populate_shared_from_program(&program);

        let func = program.get_function("Other.try_access").unwrap();
        FN_NAME_STACK.with(|s| s.borrow_mut().push("Other.try_access".to_string()));
        let result = eval_function_body(&program, &func.body, &mut env);
        FN_NAME_STACK.with(|s| s.borrow_mut().pop());

        // Should fail because "Other.secret" doesn't exist
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("undefined variable"), "Expected 'undefined variable' error, got: {err_msg}");
    }

    #[test]
    fn qualify_function_name_returns_qualified_for_module_fn() {
        let source = "\
!module MyMod
+fn helper ()->Int
  +return 1
+end
";
        let program = build_program(source);
        assert_eq!(program.qualify_function_name("helper"), "MyMod.helper");
    }

    #[test]
    fn qualify_function_name_returns_bare_for_top_level_fn() {
        let source = "\
+fn top_level ()->Int
  +return 1
+end
";
        let program = build_program(source);
        assert_eq!(program.qualify_function_name("top_level"), "top_level");
    }

    #[test]
    fn qualify_function_name_preserves_already_qualified() {
        let source = "\
!module Foo
+fn bar ()->Int
  +return 1
+end
";
        let program = build_program(source);
        assert_eq!(program.qualify_function_name("Foo.bar"), "Foo.bar");
    }

    #[test]
    fn qualify_function_name_unknown_returns_as_is() {
        let program = build_program("");
        assert_eq!(program.qualify_function_name("nonexistent"), "nonexistent");
    }

    #[test]
    fn shared_var_nested_call_via_eval_call_with_input() {
        // Test the !eval path: eval_call_with_input should also
        // support shared variables in nested function calls.
        let source = "\
!module Store
+shared price:Int = 50
+fn get_price ()->Int
  +return price
+end

+fn total (qty:Int)->Int
  +return get_price() * qty
+end

!eval Store.total qty=3
";
        let program = build_program(source);
        let input = parser::parse(source)
            .unwrap()
            .into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .unwrap();
        let result = eval_call_with_input(&program, &input.function_name, &input.input).unwrap();
        assert_eq!(result, "150");
    }

    #[test]
    fn shared_var_deeply_nested_calls() {
        // A calls B calls C, all accessing shared state
        let source = "\
!module Deep
+shared base:Int = 7
+fn c ()->Int
  +return base
+end

+fn b ()->Int
  +return c() + 1
+end

+fn a ()->Int
  +return b() + 2
+end
";
        let program = build_program(source);
        let mut env = Env::new();
        env.populate_shared_from_program(&program);

        let func = program.get_function("Deep.a").unwrap();
        FN_NAME_STACK.with(|s| s.borrow_mut().push("Deep.a".to_string()));
        let result = eval_function_body(&program, &func.body, &mut env).unwrap();
        FN_NAME_STACK.with(|s| s.borrow_mut().pop());

        // base=7, c()=7, b()=8, a()=10
        assert!(matches!(result, Value::Int(10)), "expected 10, got {result}");
    }

    // bytecode VM placeholder
}
