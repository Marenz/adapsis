//! The interpreter's stack (review finding F4): the call-depth guard holds on the
//! evaluation stack, and running out of stack anywhere is an error, not an abort.
use super::*;

const LIBRARY: &str = "\
+module StackLib
+fn deep (n:Int)->Int
  +if n > 0
    +return StackLib.deep(n - 1)
  +end
  +return 0
+end
+fn ping (n:Int)->Int
  +if n > 0
    +return StackLib.pong(n - 1)
  +end
  +return 0
+end
+fn pong (n:Int)->Int
  +if n > 0
    +return StackLib.ping(n - 1)
  +end
  +return 0
+end
+type Step = Go(Int) | Stop
+fn arms (n:Int)->Int
  +let items:List<Int> = list(n)
  +each items item:Int
    +let step:Step = Go(item)
    +match step
    +case Go(left)
      +if left > 0
        +while left > 0
          +return StackLib.arms(left - 1)
        +end
      +end
      +return 0
    +case Stop
      +return 0
    +end
  +end
  +return 0
+end
+fn deep_io (n:Int)->Int [io,async]
  +if n > 0
    +await below:Int = StackLib.deep_io(n - 1)
    +return below
  +end
  +return 0
+end
";

fn program() -> ast::Program {
    let mut program = ast::Program::default();
    for op in crate::parser::parse(LIBRARY).unwrap() {
        crate::validator::apply_and_validate(&mut program, &op).unwrap();
    }
    program.rebuild_function_index();
    program
}

fn call(program: &ast::Program, function: &str, depth: i64) -> Result<Value> {
    let handle = crate::coroutine::CoroutineHandle::new_mock(vec![]);
    let mut env = invocation::async_tree_env(program, handle);
    eval_builtin_or_user(program, function, vec![Value::Int(depth)], &mut env)
}

/// Stack bytes per recursion level of `function`, measured between two depths.
fn frame_cost(program: &ast::Program, function: &str) -> usize {
    let used = |depth: i64| {
        let top = stack::reset_and_mark();
        call(program, function, depth).unwrap();
        stack::deepest_below(top)
    };
    (used(120) - used(20)) / 100
}

#[test]
fn the_call_depth_guard_holds_on_the_evaluation_stack() {
    let program = program();
    on_deep_stack(|| {
        let mut costs = Vec::new();
        for function in ["StackLib.deep", "StackLib.ping", "StackLib.arms", "StackLib.deep_io"] {
            // Each nested call of these functions is one call level: `f(n)` is n + 1 levels.
            let limit = MAX_CALL_DEPTH as i64 - 1;
            // `eval_builtin_or_user` here is the entry, not a counted level: one more.
            assert_eq!(call(&program, function, limit).unwrap().to_string(), "0", "{function}");
            let error = call(&program, function, limit + 2).unwrap_err().to_string();
            assert!(error.contains("maximum call depth (256) exceeded"), "{function}: {error}");
            assert!(call(&program, function, 3).is_ok(), "{function}: the guard resets");
            let cost = frame_cost(&program, function);
            println!("STACK {function}: {cost} bytes per level, {} KiB for 256 levels", cost * 256 / 1024);
            costs.push(cost);
        }
        // 256 levels of the costliest shape use a small part of the evaluation stack.
        let worst = costs.into_iter().max().unwrap();
        assert!(worst * 256 < crate::run_limits::EVAL_STACK_BYTES / 4, "{worst} bytes per level");
        assert!(worst > 1024, "the measurement is not plausible: {worst}");
    })
    .unwrap();
}

/// On a small stack (a tokio worker or blocking thread has 2 MiB) the depth guard is not
/// reached; the stack check refuses the call instead of letting the process abort.
#[test]
fn running_out_of_stack_is_an_error_on_any_thread() {
    let program = program();
    for (stack_bytes, function) in [(2 * 1024 * 1024, "StackLib.deep"), (2 * 1024 * 1024, "StackLib.arms"), (1024 * 1024, "StackLib.ping"), (512 * 1024, "StackLib.deep_io")] {
        let program = program.clone();
        let outcome = std::thread::Builder::new()
            .stack_size(stack_bytes)
            .spawn(move || {
                let deep = call(&program, function, 250).map(|value| value.to_string()).map_err(|error| error.to_string());
                // The thread is still usable afterwards.
                assert_eq!(call(&program, function, 5).unwrap().to_string(), "0");
                deep
            })
            .unwrap()
            .join()
            .expect("the thread survived");
        println!("STACK {function} at depth 250 on {} KiB: {outcome:?}", stack_bytes / 1024);
        match outcome {
            Ok(value) => assert_eq!(value, "0"),
            Err(error) => assert!(error.contains("call stack exhausted in `"), "{function}: {error}"),
        }
    }
    // Deeper than any 2 MiB stack can hold, and beyond the depth guard: still an error.
    let program2 = program.clone();
    let error = std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(move || call(&program2, "StackLib.arms", 400).unwrap_err().to_string())
        .unwrap()
        .join()
        .unwrap();
    assert!(error.contains("call stack exhausted") || error.contains("maximum call depth"), "{error}");
}

#[test]
fn on_deep_stack_propagates_values_and_panics() {
    assert_eq!(on_deep_stack(|| 6 * 7).unwrap(), 42);
    let caught = std::panic::catch_unwind(|| on_deep_stack(|| -> i32 { panic!("fixture panic on the evaluation thread") }));
    let payload = caught.unwrap_err();
    assert_eq!(payload.downcast_ref::<&str>().copied(), Some("fixture panic on the evaluation thread"));
    // A tokio runtime context is carried over.
    let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let _guard = runtime.enter();
    assert!(on_deep_stack(|| tokio::runtime::Handle::try_current().is_ok()).unwrap());
}

/// #96 on the evaluation thread: the JIT's "inside compile" flag, its panic location and
/// the test seam are per thread, the cache is shared. A compile that panics on the
/// 64 MiB thread is contained there, journaled once, and leaves the shared cache usable
/// from the thread that started the evaluation.
#[test]
fn jit_compile_panic_on_the_evaluation_thread_is_contained_and_the_cache_stays_usable() {
    let mut program = ast::Program::default();
    for operation in parser::parse("+fn probe ()->Int\n  +return 11\n+end\n").unwrap() {
        crate::validator::apply_and_validate(&mut program, &operation).unwrap();
    }
    program.rebuild_function_index();
    let input = parser::parse_single_expr("{}").unwrap();
    let cache = new_jit_cache();
    let (attempts, journaled) = on_deep_stack(|| {
        let _seam = jit_seam::hook(|| {
            assert!(jit_contained_panic(Some(std::panic::Location::caller())));
            panic!("injected JIT compile panic on the evaluation thread (test)")
        });
        for _ in 0..3 {
            let actual = eval_compiled_or_interpreted_cached(&program, "probe", &input, Some(&cache), 1).unwrap();
            assert_eq!(actual, ("11".into(), false));
        }
        assert!(!jit_contained_panic(None), "the flag is cleared after the contained compile");
        (jit_seam::attempts(), jit_seam::journaled())
    })
    .unwrap();
    assert_eq!((attempts, journaled), (1, 1), "one compile, one journal entry, then the remembered refusal");
    assert!(!jit_contained_panic(None), "the caller's thread was never inside a compile");
    assert!(!cache.is_poisoned());
    // The refusal made on the evaluation thread holds on this one, the next revision is
    // compiled here, and the evaluation thread then reuses that module.
    let _seam = jit_seam::hook(|| {});
    assert_eq!(eval_compiled_or_interpreted_cached(&program, "probe", &input, Some(&cache), 1).unwrap(), ("11".into(), false));
    assert_eq!(jit_seam::attempts(), 0, "the refused revision was compiled again on another thread");
    assert_eq!(eval_compiled_or_interpreted_cached(&program, "probe", &input, Some(&cache), 2).unwrap(), ("11".into(), true));
    assert_eq!(jit_seam::attempts(), 1);
    let reused = on_deep_stack(|| {
        let _seam = jit_seam::hook(|| panic!("a cached module must not be compiled again"));
        let actual = eval_compiled_or_interpreted_cached(&program, "probe", &input, Some(&cache), 2).unwrap();
        (actual, jit_seam::attempts())
    })
    .unwrap();
    assert_eq!(reused, (("11".into(), true), 0));
}
