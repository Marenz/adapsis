use anyhow::{Result, anyhow, bail};
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
        if let Some(rt) = get_shared_runtime() {
            env.set_runtime(rt);
        }
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
        if let Some(rt) = get_shared_runtime() {
            env.set_runtime(rt);
        }
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
        if let Some(rt) = get_shared_runtime() {
            env.set_runtime(rt);
        }
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
                    let value = eval_expr_standalone(program, &sv.default).unwrap_or(Value::Int(0));
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
                state
                    .shared_vars
                    .entry(key.clone())
                    .or_insert_with(|| value.clone());
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
    if candidates.is_empty() {
        String::new()
    } else {
        format!(". Did you mean: {}?", candidates.join(", "))
    }
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

    let func = program.get_function(function_name).ok_or_else(|| {
        anyhow!(
            "function `{function_name}` not found{}",
            crate::eval::suggest_similar(program, function_name)
        )
    })?;

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
pub(crate) fn input_to_vm_args(
    input: &parser::Expr,
    func: &ast::FunctionDecl,
) -> Result<Vec<Value>> {
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
            bail!(
                "no arguments provided for {} parameter(s)",
                func.params.len()
            );
        }
        // Single value with single param → use directly
        _ if func.params.len() == 1 => Ok(vec![input_val]),
        _ => bail!(
            "cannot convert input to VM args for {} parameters",
            func.params.len()
        ),
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
        parser::Expr::Unary { expr: inner, .. } => expr_contains_io_builtin(inner),
        parser::Expr::FieldAccess { base, .. } => expr_contains_io_builtin(base),
        parser::Expr::StructLiteral(fields) => {
            fields.iter().any(|f| expr_contains_io_builtin(&f.value))
        }
        parser::Expr::Cast { expr: inner, .. } => expr_contains_io_builtin(inner),
        parser::Expr::Int(_)
        | parser::Expr::Float(_)
        | parser::Expr::Bool(_)
        | parser::Expr::String(_)
        | parser::Expr::Ident(_) => false,
    }
}

/// Check if an expression calls a user function that has [io] or [async] effects.
pub fn expr_calls_io_function(expr: &parser::Expr, program: &crate::ast::Program) -> bool {
    match expr {
        parser::Expr::Call { callee, args } => {
            let name = parser_callee_name(callee);
            if let Some(func) = program.get_function(&name) {
                if func
                    .effects
                    .iter()
                    .any(|e| matches!(e, crate::ast::Effect::Io | crate::ast::Effect::Async))
                {
                    return true;
                }
            }
            args.iter().any(|a| expr_calls_io_function(a, program))
        }
        parser::Expr::Binary { left, right, .. } => {
            expr_calls_io_function(left, program) || expr_calls_io_function(right, program)
        }
        parser::Expr::Unary { expr: inner, .. }
        | parser::Expr::Cast { expr: inner, .. }
        | parser::Expr::FieldAccess { base: inner, .. } => expr_calls_io_function(inner, program),
        parser::Expr::StructLiteral(fields) => fields
            .iter()
            .any(|f| expr_calls_io_function(&f.value, program)),
        _ => false,
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
            let all_match = func.params.iter().all(|p| {
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
                    let value =
                        eval_expr_standalone(program, &shared.default).unwrap_or(Value::Int(0));
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
    let func = program.get_function(function_name).ok_or_else(|| {
        anyhow!(
            "function `{function_name}` not found{}",
            crate::eval::suggest_similar(program, function_name)
        )
    })?;

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

    let msg = check_test_result(
        result,
        &func.return_type,
        &input,
        &expected,
        case.matcher.as_ref(),
    )?;

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
pub(crate) fn fork_runtime_for_test(
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
    let func = program.get_function(function_name).ok_or_else(|| {
        anyhow!(
            "function `{function_name}` not found{}",
            crate::eval::suggest_similar(program, function_name)
        )
    })?;

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
    // Pre-populate io_mocks and function_stubs so they're available during tests.
    let stubs = get_shared_meta()
        .and_then(|meta| meta.lock().ok().map(|m| m.function_stubs.clone()))
        .unwrap_or_default();
    let forked_meta: crate::session::SharedMeta = {
        let mut m = crate::session::SessionMeta::new();
        m.io_mocks = mocks.to_vec();
        m.function_stubs = stubs.clone();
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
                let handle = crate::coroutine::CoroutineHandle::new_mock_with_stubs(mocks, stubs);
                env.set("__coroutine_handle", Value::CoroutineHandle(handle));

                bind_input_to_params(&program, func, &input, &mut env);
                // Use eval_function_body_named so FN_NAME_STACK is populated
                // with the qualified function name — required for +shared
                // variable resolution in Env::get()/set_existing().
                let result = eval_function_body_named(&program, &fn_name, &func.body, &mut env);
                let msg =
                    check_test_result(result, &return_type, &input, &expected, matcher.as_ref())?;

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
    let func = program.get_function(function_name).ok_or_else(|| {
        anyhow!(
            "function `{function_name}` not found{}",
            crate::eval::suggest_similar(program, function_name)
        )
    })?;

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
        let stubs = get_shared_meta()
            .and_then(|meta| meta.lock().ok().map(|m| m.function_stubs.clone()))
            .unwrap_or_default();
        let handle = crate::coroutine::CoroutineHandle::new_mock_with_stubs(mocks, stubs);
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
            parser::TestMatcher::Contains(s) => raw_text.contains(s.as_str()),
            parser::TestMatcher::StartsWith(s) => raw_text.starts_with(s.as_str()),
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
        "routes" => match after.matcher.as_str() {
            "contains" => {
                let found = http_routes
                    .iter()
                    .any(|r| r.path.contains(&after.value) || r.handler_fn.contains(&after.value));
                if !found {
                    bail!(
                        "+after routes contains \"{}\": no matching route found (routes: {:?})",
                        after.value,
                        http_routes
                            .iter()
                            .map(|r| format!("{} {} -> {}", r.method, r.path, r.handler_fn))
                            .collect::<Vec<_>>()
                    );
                }
            }
            other => bail!("+after routes: unknown matcher `{other}` (expected `contains`)"),
        },
        "modules" => match after.matcher.as_str() {
            "contains" => {
                let found = program
                    .modules
                    .iter()
                    .any(|m| m.name.contains(&after.value));
                if !found {
                    bail!(
                        "+after modules contains \"{}\": no matching module found (modules: {:?})",
                        after.value,
                        program.modules.iter().map(|m| &m.name).collect::<Vec<_>>()
                    );
                }
            }
            other => bail!("+after modules: unknown matcher `{other}` (expected `contains`)"),
        },
        "mocks" => match after.matcher.as_str() {
            "contains" => {
                if let Some(mocks) = mocks {
                    let found = mocks.iter().any(|m| {
                        m.operation.contains(&after.value)
                            || m.patterns.iter().any(|p| p.contains(&after.value))
                    });
                    if !found {
                        bail!(
                            "+after mocks contains \"{}\": no matching mock found",
                            after.value
                        );
                    }
                } else {
                    bail!("+after mocks: mock state not available in this test context");
                }
            }
            other => bail!("+after mocks: unknown matcher `{other}` (expected `contains`)"),
        },
        "tasks" => {
            // Tasks are runtime-level and not directly inspectable from the program.
            // For now, this is a no-op with a warning.
            eprintln!("[test] +after tasks check skipped (tasks require live runtime context)");
        }
        other => {
            bail!("+after: unknown target `{other}` (expected routes, modules, mocks, or tasks)")
        }
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

pub(crate) fn eval_function_body(
    program: &ast::Program,
    body: &[ast::Statement],
    env: &mut Env,
) -> Result<Value> {
    for stmt in body {
        // Update task snapshot if this is a tracked coroutine task.
        if let Some(Value::CoroutineHandle(handle)) = env.get_raw("__coroutine_handle") {
            if handle.task_id.is_some() {
                let fn_name = FN_NAME_STACK.with(|s| {
                    s.borrow()
                        .last()
                        .cloned()
                        .unwrap_or_else(|| "<top>".to_string())
                });
                let depth = FN_NAME_STACK.with(|s| s.borrow().len());
                handle.update_snapshot(&fn_name, Some(stmt.id.clone()), depth, env.snapshot_bindings());
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
                                    if binding != "_" {
                                        env.set(binding, inner.as_ref().clone());
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
                                if arm.bindings.len() == 1 && arm.bindings[0] != "_" {
                                    env.set(&arm.bindings[0], as_union.clone());
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
                            bail!("+match: no arm matched Ok");
                        }
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
                                    if binding != "_" {
                                        env.set(binding, Value::string(msg.clone()));
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
                                if arm.bindings.len() == 1 && arm.bindings[0] != "_" {
                                    env.set(&arm.bindings[0], as_union.clone());
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
                            bail!("+match: no arm matched Err");
                        }
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
                                if !matches!(result, Value::None) {
                                    return Ok(result);
                                }
                                matched = true;
                                break;
                            }
                        }
                        if !matched {
                            bail!("+match: no arm matched None");
                        }
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
                        || matches!(
                            call.callee.as_str(),
                            "symbols_list"
                                | "source_get"
                                | "callers_get"
                                | "callees_get"
                                | "deps_get"
                                | "routes_list"
                                | "route_list"
                                | "test_run"
                        );
                    if needs_program_read {
                        set_shared_program(Some(std::sync::Arc::new(program.clone())));
                    }
                    // Ensure mutation builtins can write to the program AST via thread-local.
                    let is_mutation_builtin = matches!(
                        call.callee.as_str(),
                        "mutate"
                            | "fn_remove"
                            | "type_remove"
                            | "module_remove"
                            | "module_create"
                            | "fn_replace"
                            | "move_symbols"
                    );
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
            // Source statements — send IoRequest::SourceAdd in async context
            ast::StatementKind::Source(op) => {
                match op {
                    ast::SourceOp::Add {
                        source_type,
                        alias,
                        handler,
                    } => {
                        let (src_type_str, interval_ms) = match source_type {
                            ast::SourceType::Timer(expr) => {
                                let val = eval_ast_expr(program, expr, env)?;
                                let ms = match &val {
                                    Value::Int(n) => *n as u64,
                                    _ => bail!("timer interval must be Int, got {}", val),
                                };
                                ("timer".to_string(), Some(ms))
                            }
                            ast::SourceType::Channel => ("channel".to_string(), None),
                            ast::SourceType::Event(module, event) => {
                                (format!("event:{}.{}", module, event), None)
                            }
                        };
                        // Determine the current module name
                        let module_name = match env.get_raw("__module_name") {
                            Some(Value::String(s)) => s.as_ref().clone(),
                            _ => {
                                if let Some(dot) = handler.rfind('.') {
                                    handler[..dot].to_string()
                                } else {
                                    "unknown".to_string()
                                }
                            }
                        };
                        // Build fully-qualified handler name
                        let full_handler = if handler.contains('.') {
                            handler.clone()
                        } else {
                            format!("{}.{}", module_name, handler)
                        };
                        // Send through coroutine handle if in async context
                        if let Some(Value::CoroutineHandle(handle)) =
                            env.get_raw("__coroutine_handle")
                        {
                            let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
                            let io_tx = handle.io_sender();
                            let _ = io_tx.blocking_send(crate::coroutine::IoRequest::SourceAdd {
                                module_name,
                                source_type: src_type_str,
                                interval_ms,
                                alias: alias.clone(),
                                handler: full_handler,
                                reply: reply_tx,
                            });
                            match reply_rx.blocking_recv() {
                                Ok(Ok(_msg)) => { /* source registered */ }
                                Ok(Err(e)) => bail!("source add failed: {}", e),
                                Err(_) => bail!("source add: reply channel closed"),
                            }
                        }
                        // If no coroutine handle (sync context), silently skip
                    }
                    ast::SourceOp::Replace {
                        source_type,
                        alias,
                        handler,
                    } => {
                        // Evaluate timer expression if present, but no runtime dispatch yet
                        if let ast::SourceType::Timer(expr) = source_type {
                            let _val = eval_ast_expr(program, expr, env)?;
                        }
                        let _ = (alias, handler); // suppress unused warnings
                    }
                    ast::SourceOp::Remove { alias } => {
                        let _ = alias; // no-op for now
                    }
                    ast::SourceOp::List => {
                        // no-op for now
                    }
                }
            }
            // Event statements — evaluate emit expressions
            ast::StatementKind::Event(op) => {
                if let ast::EventOp::Emit { value, .. } = op {
                    let _val = eval_ast_expr(program, value, env)?;
                }
            }
        }
    }

    // If no explicit return, return None
    Ok(Value::None)
}

// Shared thread-local state (moved to shared_state.rs to break the
// circular dependency with coroutine.rs). Re-exported here so existing
// callers that use `eval::set_shared_*` / `eval::get_shared_*` /
// `eval::EvalContext` continue to compile without changes.
pub use crate::shared_state::*;

std::thread_local! {
    static CALL_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Stack of function names for the current call chain (used by task snapshots).
    static FN_NAME_STACK: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
    /// Thread-local string interner for variable name interning.
    /// Env uses this to convert string names to u32 ids for fast scope lookups.
    static STRING_INTERNER: std::cell::RefCell<StringInterner> = std::cell::RefCell::new(StringInterner::new());
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

fn eval_builtin_string(callee: &str, args: Vec<Value>) -> Result<Value> {
    match callee {
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
                "Add the missing effect to your function signature, e.g. [io,async] or [fail]."
                    .to_string()
            } else if lower.contains("named arguments are not supported") {
                "Adapsis function calls use positional arguments only. Instead of `func(name=value)`, \
                 use `func(value)`. Check the function signature with `?source Module.func` to see the \
                 correct parameter order.".to_string()
            } else if lower.contains("expected `key=value` pair") {
                "This error often occurs when `!eval` input is misinterpreted as key=value pairs. \
                 For function calls, use: `!eval Module.func value1 value2` or \
                 `!eval Module.func(value1, value2)`. Named arguments like `func(key=value)` are not supported.".to_string()
            } else if lower.contains("is not a valid command") && lower.contains("!done") {
                "Use `!done` (with the `!` prefix) to signal task completion. All Adapsis commands \
                 start with `!` — bare words like `DONE` are not recognized.".to_string()
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
        _ => unreachable!("eval_builtin_string called with non-string callee: {callee}"),
    }
}

fn eval_builtin_list(callee: &str, args: Vec<Value>) -> Result<Value> {
    match callee {
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
        _ => unreachable!("eval_builtin_list called with non-list callee: {callee}"),
    }
}

fn eval_builtin_math(callee: &str, args: Vec<Value>) -> Result<Value> {
    match callee {
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
        _ => unreachable!("eval_builtin_math called with non-math callee: {callee}"),
    }
}

fn eval_builtin_bitwise(callee: &str, args: Vec<Value>) -> Result<Value> {
    match callee {
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
        _ => unreachable!("eval_builtin_bitwise called with non-bitwise callee: {callee}"),
    }
}

pub fn eval_builtin_or_user(
    program: &ast::Program,
    callee: &str,
    args: Vec<Value>,
    env: &mut Env,
) -> Result<Value> {
    // Function stubs: intercept user function calls during tests.
    if program.get_function(callee).is_some() {
        if let Some(Value::CoroutineHandle(handle)) = env.get_raw("__coroutine_handle") {
            let qualified = program.qualify_function_name(callee);
            let stub_expr = handle
                .try_stub(&qualified, &args)?
                .or(if qualified != callee {
                    handle.try_stub(callee, &args)?
                } else {
                    None
                });
            if let Some(expr_str) = stub_expr {
                let parsed_expr = crate::parser::parse_single_expr(&expr_str)
                    .map_err(|e| anyhow::anyhow!("stub expression parse error: {e}"))?;
                return eval_parser_expr_with_program(&parsed_expr, program);
            }
        }
    }

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
        // String operations
        "len" | "length"
        | "error_suggest" | "failure_suggest"
        | "concat"
        | "to_string" | "str"
        | "char_at"
        | "substring" | "substr"
        | "starts_with"
        | "ends_with"
        | "contains"
        | "regex_match"
        | "regex_replace"
        | "index_of"
        | "split"
        | "trim"
        | "json_get"
        | "json_array_len"
        | "json_escape"
        | "base64_encode" => eval_builtin_string(callee, args),
        // List operations
        "list" | "push" | "get" | "join" => eval_builtin_list(callee, args),
        // Math operations
        "abs" | "sqrt" | "pow" | "floor"
        | "to_int" | "parse_int" | "int"
        | "digit_value"
        | "is_digit_char" => eval_builtin_math(callee, args),
        // Bitwise and numeric operations
        "bit_and" | "bit_or" | "bit_xor" | "bit_not"
        | "bit_shl" | "shl"
        | "bit_shr" | "shr"
        | "left_rotate" | "rotl"
        | "to_hex"
        | "char_code" | "ord"
        | "from_char_code" | "chr"
        | "u32_wrap"
        | "max" | "min" => eval_builtin_bitwise(callee, args),
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
                    if let Some(Value::CoroutineHandle(handle)) = env.get_raw("__coroutine_handle")
                    {
                        let handle = handle.clone();
                        return handle.execute_await(callee, &args);
                    }
                    bail!(
                        "`{callee}` is an async IO operation, use: +await result:String = {callee}({})",
                        args.iter()
                            .enumerate()
                            .map(|(i, _)| format!("arg{i}"))
                            .collect::<Vec<_>>()
                            .join(", ")
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
pub fn eval_parser_expr_with_program(
    expr: &parser::Expr,
    program: &ast::Program,
) -> Result<Value> {
    eval_parser_expr_impl(expr, program, None)
}

/// Like `eval_parser_expr_with_program`, but threads an environment through
/// so that `__coroutine_handle` is available to IO builtin calls.
/// Used by `eval_inline_expr_with_io` for `!eval shell_exec(...)` etc.
fn eval_parser_expr_with_env(
    expr: &parser::Expr,
    program: &ast::Program,
    env: &mut Env,
) -> Result<Value> {
    eval_parser_expr_impl(expr, program, Some(env))
}

/// Unified implementation for evaluating parser expressions with optional env.
///
/// When `env` is `None`: side-effect guards are applied (reject IO/async/mut functions),
/// and fresh envs are created from the program for function calls.
/// When `env` is `Some`: variables are resolved from the env, shared state and
/// coroutine handles are inherited, and no side-effect guard is applied.
fn eval_parser_expr_impl(
    expr: &parser::Expr,
    program: &ast::Program,
    mut env: Option<&mut Env>,
) -> Result<Value> {
    // Helper: check for side-effect functions (only in "no env" / test-expression mode)
    fn check_side_effects(func: &ast::FunctionDecl, name: &str, has_env: bool) -> Result<()> {
        if !has_env {
            let has_side_effects = func.effects.iter().any(|e| {
                matches!(
                    e,
                    ast::Effect::Io | ast::Effect::Async | ast::Effect::Mut | ast::Effect::Unsafe
                )
            });
            if has_side_effects {
                bail!(
                    "cannot call `{name}` in test expression: function has side effects {:?} — \
                     use !mock and an async test wrapper instead",
                    func.effects
                );
            }
        }
        Ok(())
    }

    // Helper: build an env for calling a user function, inheriting from ambient env if present.
    fn make_call_env(program: &ast::Program, env: Option<&Env>) -> Env {
        let mut call_env = Env::new_with_shared_interner(&program.shared_interner);
        if let Some(ambient) = env {
            call_env.inherit_shared_from(ambient);
            if let Some(handle) = ambient.get_raw("__coroutine_handle") {
                call_env.set("__coroutine_handle", handle.clone());
            }
        } else {
            call_env.populate_shared_from_program(program);
        }
        call_env
    }

    // Helper: evaluate sub-expressions, threading env through when present.
    // Uses a loop instead of .map() to allow sequential &mut borrows.
    fn eval_args(
        args: &[parser::Expr],
        program: &ast::Program,
        mut env: Option<&mut Env>,
    ) -> Result<Vec<Value>> {
        let mut result = Vec::with_capacity(args.len());
        for a in args {
            result.push(eval_parser_expr_impl(a, program, env.as_deref_mut())?);
        }
        Ok(result)
    }

    match expr {
        parser::Expr::Ident(name) => {
            // With env: check variables first
            if let Some(ref mut env) = env {
                if let Ok(val) = env.get(name) {
                    return Ok(val);
                }
            }
            // Check union variants
            if is_union_variant(program, name) {
                return Ok(Value::Union {
                    variant: intern::intern_display(name),
                    payload: vec![],
                });
            }
            // Check zero-arg user functions
            if let Some(func) = program.get_function(name) {
                if func.params.is_empty() {
                    check_side_effects(&func, name, env.is_some())?;
                    let mut call_env = make_call_env(program, env.as_deref());
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
            // Union variant constructor
            if is_union_variant(program, &name) {
                let payload = eval_args(args, program, env)?;
                return Ok(Value::Union {
                    variant: intern::intern_display(&name),
                    payload,
                });
            }
            // User function call
            if let Some(func) = program.get_function(&name) {
                check_side_effects(&func, &name, env.is_some())?;
                let eval_args_vec = eval_args(args, program, env.as_deref_mut())?;
                let mut call_env = make_call_env(program, env.as_deref());
                for (param, arg) in func.params.iter().zip(eval_args_vec) {
                    call_env.set(&param.name, arg);
                }
                let qualified = program.qualify_function_name(&name);
                FN_NAME_STACK.with(|s| s.borrow_mut().push(qualified));
                let result = eval_function_body(program, &func.body, &mut call_env);
                FN_NAME_STACK.with(|s| s.borrow_mut().pop());
                return result;
            }
            // Builtin function (skip Ok/Err/Some/None — handled by standalone)
            if crate::builtins::is_builtin(&name)
                && !matches!(name.as_str(), "Ok" | "Err" | "Some" | "None")
            {
                let eval_args_vec = eval_args(args, program, env.as_deref_mut())?;
                if let Some(env) = env {
                    return eval_builtin_or_user(program, &name, eval_args_vec, env);
                } else {
                    let mut fresh_env = Env::new_with_shared_interner(&program.shared_interner);
                    return eval_builtin_or_user(program, &name, eval_args_vec, &mut fresh_env);
                }
            }
            eval_parser_expr_standalone(expr)
        }
        parser::Expr::StructLiteral(fields) => {
            let empty_id = intern::intern_display("");
            let mut field_map: HashMap<InternedId, Value> = HashMap::new();
            for f in fields {
                let val = eval_parser_expr_impl(&f.value, program, env.as_deref_mut())?;
                field_map.insert(intern::intern_display(&f.name), val);
            }
            Ok(Value::strct_interned(empty_id, field_map))
        }
        parser::Expr::Unary { op, expr: inner } => {
            let val = eval_parser_expr_impl(inner, program, env)?;
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
            let l = eval_parser_expr_impl(left, program, env.as_deref_mut())?;
            let r = eval_parser_expr_impl(right, program, env)?;
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
    let func = program.get_function(function_name).ok_or_else(|| {
        anyhow!(
            "function `{function_name}` not found{}",
            crate::eval::suggest_similar(program, function_name)
        )
    })?;

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
mod tests;
