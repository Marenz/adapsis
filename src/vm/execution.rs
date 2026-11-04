//! Admission and execution are different outcomes. All graph compilation, root argument
//! checks and sync/IO eligibility happen before the first instruction. Only NotAdmitted
//! permits tree fallback. Once execution starts, no failure permits replay, even before IO.
use super::*;

#[derive(Debug)]
pub enum VmAttempt {
    NotAdmitted(anyhow::Error),
    Executed(Result<Value>),
}

pub fn attempt(
    function: &ast::FunctionDecl,
    args: Vec<Value>,
    program: &ast::Program,
    allow_io: bool,
    io_handler: &dyn Fn(&str, &[Value]) -> Result<Value>,
) -> VmAttempt {
    let prepared = (|| {
        let compiled = compile_function(function, program)?;
        let prepared = Prepared::new(&compiled, &args, program, allow_io, Some(function))?;
        Ok::<_, anyhow::Error>((compiled, prepared))
    })();
    match prepared {
        Err(error) => VmAttempt::NotAdmitted(error),
        Ok((compiled, prepared)) => {
            VmAttempt::Executed(prepared.run(&compiled, args, program, io_handler))
        }
    }
}

/// Direct execution APIs still perform the root shape check and complete preparation;
/// their Result API does not itself authorize fallback. Normal dispatch uses `attempt`.
#[allow(dead_code)] // Direct VM API; normal routing uses the typed `attempt` outcome.
pub fn execute_with_io(
    compiled: &CompiledFunction,
    args: Vec<Value>,
    program: &ast::Program,
    io_handler: &dyn Fn(&str, &[Value]) -> Result<Value>,
) -> Result<Value> {
    Prepared::new(compiled, &args, program, true, None)?.run(compiled, args, program, io_handler)
}

#[cfg(test)]
pub fn execute(
    compiled: &CompiledFunction,
    args: Vec<Value>,
    program: &ast::Program,
) -> Result<VmResult> {
    let mut prepared = Prepared::new(compiled, &args, program, true, None)?;
    run_loop(
        initial_state(compiled, args),
        program,
        &mut prepared.cache,
        &mut prepared.interner,
    )
}

struct Prepared {
    cache: CompileCache,
    interner: StringInterner,
}

impl Prepared {
    fn new(
        root: &CompiledFunction,
        args: &[Value],
        program: &ast::Program,
        allow_io: bool,
        root_function: Option<&ast::FunctionDecl>,
    ) -> Result<Self> {
        field_admission::check_arguments(root, args)?;
        let mut this = Self {
            cache: CompileCache::new(),
            interner: root.interner.clone(),
        };
        // Pointer identity is confined to this immutable snapshot. Alias spellings get
        // separate cache keys but share the same prepared callee, including recursive SCCs.
        let mut by_identity = HashMap::new();
        if let Some(function) = root_function {
            by_identity.insert(function as *const ast::FunctionDecl, root.clone());
        }
        let mut pending = vec![root.clone()];
        while let Some(compiled) = pending.pop() {
            for instruction in compiled.bytecode.iter() {
                if !allow_io && matches!(instruction, Op::AwaitIo(..)) {
                    bail!("vm: synchronous entry requires tree for reachable IO");
                }
                let Op::Call(id, _) = instruction else {
                    continue;
                };
                if this.cache.contains_key(id) {
                    continue;
                }
                let name = this
                    .interner
                    .resolve(*id)
                    .ok_or_else(|| anyhow::anyhow!("vm: invalid callee name id {id}"))?;
                let function = program.get_function(name).ok_or_else(|| {
                    anyhow::anyhow!("vm: function `{name}` not found during preparation")
                })?;
                let identity = function as *const ast::FunctionDecl;
                let callee = if let Some(existing) = by_identity.get(&identity) {
                    existing.clone()
                } else {
                    // Root admission already traversed this entire graph. Do not repeat
                    // those graph walks per frame; compile each unique callee once.
                    let callee = emit_function(function, program, &this.interner)?;
                    this.interner = callee.interner.clone();
                    by_identity.insert(identity, callee.clone());
                    pending.push(callee.clone());
                    callee
                };
                // Resolve by the original call spelling, never by the callee's bare
                // display name (which may belong to another module's function).
                this.cache.insert(*id, callee);
            }
        }
        Ok(this)
    }

    fn run(
        mut self,
        compiled: &CompiledFunction,
        args: Vec<Value>,
        program: &ast::Program,
        io_handler: &dyn Fn(&str, &[Value]) -> Result<Value>,
    ) -> Result<Value> {
        let mut state = initial_state(compiled, args);
        loop {
            match run_loop(state, program, &mut self.cache, &mut self.interner)? {
                VmResult::Done(value) => return Ok(value),
                VmResult::Await {
                    op_name,
                    args,
                    vm_state,
                } => {
                    let value = io_handler(&op_name, &args)?;
                    state = vm_state;
                    state.stack.push(value);
                }
            }
        }
    }
}

fn initial_state(compiled: &CompiledFunction, args: Vec<Value>) -> VmState {
    let mut state = VmState {
        function_name: compiled.name.clone(),
        ip: 0,
        bytecode: Arc::clone(&compiled.bytecode),
        locals: vec![Value::None; compiled.local_count.max(args.len())],
        stack: Vec::with_capacity(32),
        call_stack: Vec::new(),
    };
    for (slot, value) in state.locals.iter_mut().zip(args) {
        *slot = value;
    }
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preparation_reuses_aliases_and_recursive_callees_without_running_them() {
        let program = crate::vm::scalar_failure_tests::program("+module M\n+fn child (n:Int)->Int\n  +if n == 0\n    +return 1\n  +end\n  +return M.child(n - 1)\n+end\n+fn root ()->Int\n  +return child(3) + M.child(2)\n+end\n");
        let function = program.get_function("M.root").unwrap();
        let compiled = compile_function(function, &program).unwrap();
        let prepared = Prepared::new(&compiled, &[], &program, false, Some(function)).unwrap();
        assert_eq!(
            prepared.cache.len(),
            2,
            "one entry per spelling, not per frame"
        );
        let bare = prepared
            .cache
            .get(&prepared.interner.get("child").unwrap())
            .unwrap();
        let qualified = prepared
            .cache
            .get(&prepared.interner.get("M.child").unwrap())
            .unwrap();
        assert!(
            Arc::ptr_eq(&bare.bytecode, &qualified.bytecode),
            "same callee emitted exactly once"
        );
        assert!(matches!(
            prepared
                .run(&compiled, vec![], &program, &|_, _| bail!("no IO"))
                .unwrap(),
            Value::Int(2)
        ));
    }

    #[test]
    fn preparation_cost_sample_keeps_scalar_control_on_vm() {
        let program = crate::vm::scalar_failure_tests::program("+fn step (n:Int)->Int\n  +return n + 1\n+end\n+fn root (n:Int)->Int\n  +return step(n) + step(n)\n+end\n");
        let function = program.get_function("root").unwrap();
        let compiled = compile_function(function, &program).unwrap();
        let started = std::time::Instant::now();
        for _ in 0..1_000 {
            let prepared =
                Prepared::new(&compiled, &[Value::Int(3)], &program, false, Some(function))
                    .unwrap();
            assert_eq!(prepared.cache.len(), 1);
        }
        let preparation = started.elapsed();
        let started = std::time::Instant::now();
        for _ in 0..1_000 {
            let result = attempt(function, vec![Value::Int(3)], &program, false, &|_, _| {
                bail!("no IO")
            });
            assert!(matches!(result, VmAttempt::Executed(Ok(Value::Int(8)))));
        }
        eprintln!(
            "VM_PREPARATION_COST iterations=1000 prepare={preparation:?} admitted_full={:?}",
            started.elapsed()
        );
    }
}
