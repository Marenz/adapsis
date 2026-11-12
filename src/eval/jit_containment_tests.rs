//! #96: a panic inside the JIT compiler is contained, and results are the
//! same whether a function runs compiled or on the VM/tree path.
use super::*;

fn program(source: &str) -> ast::Program {
    let mut program = ast::Program::default();
    for operation in parser::parse(source).unwrap() {
        crate::validator::apply_and_validate(&mut program, &operation).unwrap();
    }
    program.rebuild_function_index();
    program
}

fn no_input() -> parser::Expr {
    parser::parse_single_expr("{}").unwrap()
}

/// A compile on this thread now panics the way Cranelift does. Before it
/// panics it asks what the process panic hook asks (the test binary has no
/// such hook): "is this panic contained?", passing a location.
fn panicking_compiler() -> jit_seam::Reset {
    jit_seam::hook(|| {
        assert!(jit_contained_panic(Some(std::panic::Location::caller())), "the hook must be told the panic is contained");
        panic!("injected JIT compile panic (test) with payload SENTINEL_PAYLOAD")
    })
}

fn journal_entries_for_jit() -> Vec<serde_json::Value> {
    let snapshot: serde_json::Value = serde_json::from_str(&crate::error_journal::query(0)).unwrap();
    snapshot["entries"].as_array().unwrap().iter().filter(|entry| entry["source"] == "jit.compile").cloned().collect()
}

#[test]
fn jit_compile_panic_falls_back_to_the_vm_and_is_remembered_for_the_revision() {
    let program = program("+fn probe ()->Int\n  +return 11\n+end\n");
    let cache = new_jit_cache();
    let seam = panicking_compiler();
    // The call survives the panic and returns the VM/tree result.
    assert_eq!(eval_compiled_or_interpreted_cached(&program, "probe", &no_input(), Some(&cache), 1).unwrap(), ("11".into(), false));
    assert!(!cache.is_poisoned(), "the cache lock must not be left poisoned");
    assert!(matches!(&*cache.lock().unwrap(), Some((1, slot)) if slot.is_refused()), "the refusal is remembered");
    assert_eq!((jit_seam::attempts(), jit_seam::journaled()), (1, 1));
    // Same revision again: no new compile, no new panic, no new journal entry.
    for _ in 0..5 {
        assert_eq!(eval_compiled_or_interpreted_cached(&program, "probe", &no_input(), Some(&cache), 1).unwrap(), ("11".into(), false));
    }
    assert_eq!((jit_seam::attempts(), jit_seam::journaled()), (1, 1), "the revision was compiled or journaled again");
    // The journal names the event and where it happened (file name and line,
    // no directories), and nothing of the panic's text. The journal is shared
    // by all tests of the process, so the entries are picked by their source;
    // "exactly one per panic" is the per-thread count asserted above.
    assert!(!jit_contained_panic(None), "outside a compile the panic hook journals panics as before");
    let entries = journal_entries_for_jit();
    assert!(!entries.is_empty(), "the contained panic was not journaled");
    for entry in &entries {
        assert_eq!(entry["category"], "panic");
        let message = entry["message"].as_str().unwrap_or_default();
        assert!(message.starts_with("the JIT compiler panicked at "), "{message}");
        assert!(message.ends_with("; this program revision runs without the JIT"), "{message}");
        assert!(!message.contains('/'), "a path was journaled: {message}");
    }
    assert!(
        entries.iter().any(|entry| entry["message"].as_str().is_some_and(|message| message.contains(" at jit_containment_tests.rs:"))),
        "the location the hook reported is in the entry: {entries:?}"
    );
    let journal = crate::error_journal::query(0);
    assert!(!journal.contains("SENTINEL_PAYLOAD") && !journal.contains("injected JIT compile panic"), "the panic payload was journaled");
    // The compiler works again: the refused revision stays refused (it is not
    // retried), the next revision is compiled and the cache is fully usable.
    drop(seam);
    assert_eq!(eval_compiled_or_interpreted_cached(&program, "probe", &no_input(), Some(&cache), 1).unwrap(), ("11".into(), false));
    assert_eq!(jit_seam::attempts(), 1);
    for _ in 0..3 {
        assert_eq!(eval_compiled_or_interpreted_cached(&program, "probe", &no_input(), Some(&cache), 2).unwrap(), ("11".into(), true));
    }
    assert_eq!(jit_seam::attempts(), 2, "revision 2 is compiled once and then reused");
    assert!(matches!(&*cache.lock().unwrap(), Some((2, slot)) if slot.is_compiled()));
}

#[test]
fn jit_compile_panic_never_leaves_code_of_an_older_revision_callable_119() {
    let old = program("+fn probe ()->Int\n  +return 11\n+end\n");
    let new = program("+fn probe ()->Int\n  +return 22\n+end\n");
    let cache = new_jit_cache();
    assert_eq!(eval_compiled_or_interpreted_cached(&old, "probe", &no_input(), Some(&cache), 1).unwrap(), ("11".into(), true));
    let _seam = panicking_compiler();
    let actual = eval_compiled_or_interpreted_cached(&new, "probe", &no_input(), Some(&cache), 2).unwrap();
    assert_eq!(actual, ("22".into(), false), "must run the current program on the VM, never cached revision 1");
    assert!(matches!(&*cache.lock().unwrap(), Some((2, slot)) if slot.is_refused()), "the old module is gone");
    // Asking for the old revision again does not resurrect its code either.
    assert_eq!(eval_compiled_or_interpreted_cached(&old, "probe", &no_input(), Some(&cache), 1).unwrap(), ("11".into(), false));
}

#[test]
fn jit_compile_panic_without_a_cache_falls_back_every_time() {
    let program = program("+fn probe ()->String\n  +return \"text\"\n+end\n");
    let _seam = panicking_compiler();
    for _ in 0..2 {
        assert_eq!(eval_compiled_or_interpreted(&program, "probe", &no_input()).unwrap(), ("\"text\"".into(), false));
    }
    assert_eq!((jit_seam::attempts(), jit_seam::journaled()), (2, 2), "without a cache nothing can be remembered");
}

#[test]
fn jit_ordinary_compile_refusal_is_unchanged_and_not_journaled() {
    // `nope` cannot be compiled, so the whole program is refused with an error.
    let program = program("+fn probe ()->Int\n  +return 11\n+end\n+fn nope ()->List<Int>\n  +return list()\n+end\n");
    let cache = new_jit_cache();
    let _seam = jit_seam::hook(|| {});
    for _ in 0..2 {
        assert_eq!(eval_compiled_or_interpreted_cached(&program, "probe", &no_input(), Some(&cache), 1).unwrap(), ("11".into(), false));
        assert!(cache.lock().unwrap().is_none(), "an ordinary refusal leaves the slot empty, as before");
    }
    assert_eq!(jit_seam::journaled(), 0);
}

#[test]
fn jit_poisoned_cache_lock_is_recovered_instead_of_disabling_the_jit() {
    let program = program("+fn probe ()->Int\n  +return 11\n+end\n");
    let cache = new_jit_cache();
    assert_eq!(eval_compiled_or_interpreted_cached(&program, "probe", &no_input(), Some(&cache), 1).unwrap(), ("11".into(), true));
    // Something panics while it holds the cache lock.
    let holder = cache.clone();
    let _ = std::thread::spawn(move || {
        let _guard = holder.lock().unwrap();
        panic!("poison the JIT cache lock (test)");
    })
    .join();
    assert!(cache.is_poisoned());
    // Before #96 every later call silently skipped the JIT for good.
    assert_eq!(eval_compiled_or_interpreted_cached(&program, "probe", &no_input(), Some(&cache), 1).unwrap(), ("11".into(), true));
    assert!(!cache.is_poisoned(), "the poison flag is cleared");
    assert!(matches!(&*cache.lock().unwrap(), Some((1, slot)) if slot.is_compiled()));
}

/// The value a function gives on the VM, and on the tree-walker.
fn reference(program: &ast::Program, function: &str, input: &parser::Expr) -> (String, String) {
    let declared = program.get_function(function).unwrap();
    let vm = match try_vm_execute(program, declared, input) {
        vm::VmAttempt::Executed(Ok(value)) => format!("{value}"),
        vm::VmAttempt::Executed(Err(error)) => panic!("{function}: VM error {error}"),
        vm::VmAttempt::NotAdmitted(error) => panic!("{function}: VM did not admit it: {error}"),
    };
    (vm, eval_call_with_input(program, function, input).unwrap())
}

#[test]
fn jit_results_equal_the_vm_for_calls_recursion_and_string_literals() {
    let program = program(crate::compiler::relocation_tests::CALLS_AND_STRINGS);
    let cache = new_jit_cache();
    for (function, input) in [
        ("double", "{n: 21}"),
        ("quad", "{n: 3}"),
        ("fact", "{n: 0}"),
        ("fact", "{n: 12}"),
        ("is_even", "{n: 40}"),
        ("is_even", "{n: 41}"),
        ("is_odd", "{n: 41}"),
        ("parity", "{n: 6}"),
        ("parity", "{n: 7}"),
        ("greeting", "{}"),
        ("relay", "{}"),
        ("phrase", "{}"),
        ("pick", "{n: 2}"),
        ("pick", "{n: 3}"),
    ] {
        let input = parser::parse_single_expr(input).unwrap();
        let (vm, tree) = reference(&program, function, &input);
        assert_eq!(vm, tree, "{function}: VM and tree disagree");
        let (value, compiled) = eval_compiled_or_interpreted_cached(&program, function, &input, Some(&cache), 1).unwrap();
        assert!(compiled, "{function} must run on the JIT for this comparison to mean anything");
        assert_eq!(value, vm, "{function}: the JIT result differs from the VM");
    }
}

// --- a cache hit is proven by what was compiled, not by the revision --------

#[test]
fn jit_same_revision_with_different_code_never_runs_the_old_code() {
    let cache = new_jit_cache();
    let _seam = jit_seam::hook(|| {});
    // A writer that changes the program without changing the revision (seven
    // write-back sites do exactly that).
    let old = program("+fn probe ()->Int\n  +return 11\n+end\n");
    let new = program("+fn probe ()->Int\n  +return 22\n+end\n");
    assert_eq!(eval_compiled_or_interpreted_cached(&old, "probe", &no_input(), Some(&cache), 5).unwrap(), ("11".into(), true));
    assert_eq!(eval_compiled_or_interpreted_cached(&new, "probe", &no_input(), Some(&cache), 5).unwrap(), ("22".into(), true));
    assert_eq!(jit_seam::attempts(), 2, "the new program was compiled, not served from the slot of the old one");
    // And back: the old program at the same revision is its own code again.
    assert_eq!(eval_compiled_or_interpreted_cached(&old, "probe", &no_input(), Some(&cache), 5).unwrap(), ("11".into(), true));
    assert_eq!(jit_seam::attempts(), 3);
}

#[test]
fn jit_revision_reused_after_a_rewind_never_runs_code_of_the_earlier_program() {
    let cache = new_jit_cache();
    let first = program("+fn probe ()->Int\n  +return 11\n+end\n");
    let second = program("+fn probe ()->Int\n  +return 22\n+end\n+fn helper ()->Int\n  +return 1\n+end\n");
    // Revision 2 is compiled, the session is rewound to 1 (nothing is
    // evaluated there), and a different edit becomes "revision 2" again.
    let again = program("+fn probe ()->Int\n  +return helper() + 32\n+end\n+fn helper ()->Int\n  +return 1\n+end\n");
    for (version, revision, expected) in [(&first, 1, "11"), (&second, 2, "22"), (&again, 2, "33"), (&again, 2, "33"), (&second, 2, "22")] {
        assert_eq!(
            eval_compiled_or_interpreted_cached(version, "probe", &no_input(), Some(&cache), revision).unwrap(),
            (expected.to_string(), true)
        );
    }
}

#[test]
fn jit_unchanged_program_stays_compiled_without_recompiling() {
    let cache = new_jit_cache();
    let _seam = jit_seam::hook(|| {});
    let live = program(
        "+fn probe ()->Int\n  +return helper() + 10\n+end\n+fn helper ()->Int\n  +return 1\n+end\n+module Extra\n+fn old ()->Int\n  +return 4\n+end\n",
    );
    for _ in 0..3 {
        // Every caller evaluates its own snapshot (a clone) of the program.
        let snapshot = live.clone();
        assert_eq!(eval_compiled_or_interpreted_cached(&snapshot, "probe", &no_input(), Some(&cache), 1).unwrap(), ("11".into(), true));
    }
    assert_eq!(jit_seam::attempts(), 1, "a clone of the same program is the same input");
    // A module function is part of the input (callee names resolve into
    // modules): merging a change to one recompiles, once. The CoW write-back
    // keeps the `Arc` of every function it does not touch, so nothing else
    // about the program looks different.
    let mut merged = live.clone();
    let with_module = {
        let mut next = live.clone();
        for operation in parser::parse("+module Extra\n+fn unrelated ()->Int\n  +return 5\n+end\n").unwrap() {
            crate::validator::apply_and_validate(&mut next, &operation).unwrap();
        }
        next.rebuild_function_index();
        next
    };
    merged.merge_changed_from(&live, &with_module);
    assert!(merged.get_function("Extra.unrelated").is_some(), "the merge added the module function");
    assert!(std::sync::Arc::ptr_eq(&merged.functions[0], &live.functions[0]), "the merge kept the untouched top-level function");
    for _ in 0..3 {
        assert_eq!(eval_compiled_or_interpreted_cached(&merged.clone(), "probe", &no_input(), Some(&cache), 1).unwrap(), ("11".into(), true));
    }
    assert_eq!(jit_seam::attempts(), 2, "a changed module function is a different input; it is compiled once and then warm");
    // A rebuilt program with equal text has new `Arc`s: recompiled once
    // (conservative), with the right result, and then warm again.
    let rebuilt = program("+fn probe ()->Int\n  +return helper() + 10\n+end\n+fn helper ()->Int\n  +return 1\n+end\n");
    for _ in 0..2 {
        assert_eq!(eval_compiled_or_interpreted_cached(&rebuilt, "probe", &no_input(), Some(&cache), 1).unwrap(), ("11".into(), true));
    }
    assert_eq!(jit_seam::attempts(), 3);
}

#[test]
fn jit_remembered_refusal_belongs_to_the_program_that_panicked() {
    let cache = new_jit_cache();
    let first = program("+fn probe ()->Int\n  +return 11\n+end\n");
    let other = program("+fn probe ()->Int\n  +return 22\n+end\n");
    let seam = panicking_compiler();
    for _ in 0..3 {
        assert_eq!(eval_compiled_or_interpreted_cached(&first, "probe", &no_input(), Some(&cache), 1).unwrap(), ("11".into(), false));
    }
    assert_eq!((jit_seam::attempts(), jit_seam::journaled()), (1, 1), "the same program is not compiled again");
    // Another program under the same revision number is not covered by that
    // refusal: it gets its own attempt (which panics here as well)...
    for _ in 0..3 {
        assert_eq!(eval_compiled_or_interpreted_cached(&other, "probe", &no_input(), Some(&cache), 1).unwrap(), ("22".into(), false));
    }
    assert_eq!((jit_seam::attempts(), jit_seam::journaled()), (2, 2));
    // ...and once the compiler works, a different program is compiled although
    // the slot holds a refusal for its revision number.
    drop(seam);
    let _counting = jit_seam::hook(|| {});
    assert_eq!(eval_compiled_or_interpreted_cached(&first, "probe", &no_input(), Some(&cache), 1).unwrap(), ("11".into(), true));
    assert_eq!(jit_seam::attempts(), 1);
}

/// A module with a nested module, built by hand (the nesting is what matters).
fn nested(mut outer: ast::Module, inner: ast::Module) -> ast::Module {
    outer.modules.push(inner);
    outer
}

#[test]
fn jit_compile_input_covers_all_functions_and_types_and_nothing_else() {
    use crate::compiler::input::CompileInput;
    use std::sync::Arc;
    let source = "+type Point = {x:Int, y:Int}\n+fn probe ()->Int\n  +return 1\n+end\n+module Shapes\n+type Size = {w:Int, h:Int}\n+fn area (s:Size)->Int\n  +return s.w * s.h\n+end\n+module Extra\n+fn one ()->Int\n  +return 1\n+end\n";
    let mut live = program(source);
    let inner = program("+module Inner\n+type Deep = {d:Int}\n+fn deep ()->Int\n  +return 3\n+end\n").modules.remove(0);
    live.modules[1] = nested(live.modules[1].clone(), inner);
    live.rebuild_function_index();
    let input = CompileInput::of(&live);
    assert!(input.matches(&live) && input.matches(&live.clone()));
    // The input holds the same function objects at every depth, the same
    // type declarations, and the same module structure.
    let (reduced, shapes, extra) = (input.program(), &input.program().modules[0], &input.program().modules[1]);
    assert!(Arc::ptr_eq(&reduced.functions[0], &live.functions[0]));
    assert!(Arc::ptr_eq(&shapes.functions[0], &live.modules[0].functions[0]));
    assert!(Arc::ptr_eq(&extra.modules[0].functions[0], &live.modules[1].modules[0].functions[0]));
    assert_eq!((&reduced.types, &shapes.types, &extra.modules[0].types), (&live.types, &live.modules[0].types, &live.modules[1].modules[0].types));
    assert_eq!((shapes.name.as_str(), extra.name.as_str(), extra.modules[0].name.as_str()), ("Shapes", "Extra", "Inner"));
    // Lookups give the same answers as in the live program.
    for name in ["probe", "area", "one", "Shapes.area", "Extra.one", "deep", "Extra.Inner.deep", "absent", "Shapes.absent"] {
        assert_eq!(reduced.get_function(name).map(|f| f as *const _), live.get_function(name).map(|f| f as *const _), "{name}");
    }

    let edited = |change: &dyn Fn(&mut ast::Program)| {
        let mut next = live.clone();
        change(&mut next);
        next
    };
    let copy = |function: &Arc<ast::FunctionDecl>| Arc::new((**function).clone());
    let other_struct = program("+type Point = {x:Int, y:Int, z:Int}\n").types[0].clone();
    // Changes of what the passes can reach: no longer the same input.
    let different: Vec<(&str, Box<dyn Fn(&mut ast::Program)>)> = vec![
        ("a top-level struct changed", Box::new(|next| next.types[0] = other_struct.clone())),
        ("a module's struct changed", Box::new(|next| next.modules[0].types[0] = other_struct.clone())),
        ("a module's struct was removed", Box::new(|next| next.modules[0].types.clear())),
        ("a nested module's struct changed", Box::new(|next| next.modules[1].modules[0].types[0] = other_struct.clone())),
        ("a type was added", Box::new(|next| next.types.push(other_struct.clone()))),
        ("a top-level function was removed", Box::new(|next| next.functions.clear())),
        ("a top-level function was added", Box::new(|next| next.functions.push(next.functions[0].clone()))),
        ("a top-level function was replaced by an equal copy", Box::new(|next| next.functions[0] = copy(&next.functions[0]))),
        ("a module function was replaced", Box::new(|next| next.modules[0].functions[0] = copy(&next.modules[0].functions[0]))),
        ("a module function was removed", Box::new(|next| next.modules[1].functions.clear())),
        ("a module function was added", Box::new(|next| next.modules[1].functions.push(next.functions[0].clone()))),
        ("a nested module function was replaced", Box::new(|next| next.modules[1].modules[0].functions[0] = copy(&next.modules[1].modules[0].functions[0]))),
        ("a nested module was removed", Box::new(|next| next.modules[1].modules.clear())),
        ("a module was renamed", Box::new(|next| next.modules[0].name = "Renamed".into())),
        ("the module order changed", Box::new(|next| next.modules.swap(0, 1))),
        ("a module was removed", Box::new(|next| drop(next.modules.remove(0)))),
    ];
    for (label, change) in &different {
        assert!(!input.matches(&edited(change.as_ref())), "{label}: still counted as the same input");
    }
    // Changes of what they cannot reach: still the same input.
    let same: Vec<(&str, Box<dyn Fn(&mut ast::Program)>)> = vec![
        ("require_modules", Box::new(|next| next.require_modules = !next.require_modules)),
        ("a module doc", Box::new(|next| next.modules[0].doc = Some("documented".into()))),
        ("module routes", Box::new(|next| {
            next.modules[0].routes.push(ast::HttpRoute { method: "GET".into(), path: "/x".into(), handler_fn: "Shapes.area".into() })
        })),
        ("pending routes", Box::new(|next| next.pending_routes.clear())),
    ];
    for (label, change) in &same {
        assert!(input.matches(&edited(change.as_ref())), "{label}: counted as a different input");
    }
}

/// The reviewer's first witness (#96 delta review): a module function that
/// shadows a builtin name and is `[fail]`. A bare `concat` in a top-level
/// function resolves to it for admission, so the whole program is refused.
/// The reduced input must be refused in the same way, and the dispatch must
/// not answer `compiled` for the unrelated `root`.
#[test]
fn jit_module_function_shadowing_a_builtin_keeps_its_admission_refusal() {
    use crate::compiler::input::CompileInput;
    let live = program(
        "+fn root ()->Int\n  +return 11\n+end\n+fn sibling ()->String\n  +return concat(\"a\", \"b\")\n+end\n\
         +module M\n+fn concat (a:String, b:String)->String [fail]\n  +return \"different\"\n+end\n",
    );
    let full = crate::compiler::compile(&live).err().map(|error| error.to_string());
    let reduced = crate::compiler::compile(CompileInput::of(&live).program()).err().map(|error| error.to_string());
    assert!(full.is_some(), "the live program is refused by the #97 admission");
    assert_eq!(full, reduced, "the input is refused for the same reason");
    let cache = new_jit_cache();
    for _ in 0..2 {
        assert_eq!(eval_compiled_or_interpreted_cached(&live, "root", &no_input(), Some(&cache), 1).unwrap(), ("11".into(), false));
    }
    assert!(cache.lock().unwrap().is_none(), "nothing was compiled");
}

/// The reviewer's second witness: changing only a module function changes
/// what admission decides, so it must change the identity.
#[test]
fn jit_changing_only_a_module_function_changes_the_identity() {
    use crate::compiler::input::CompileInput;
    let mut live = program(
        "+fn root ()->Int\n  +return 11\n+end\n+fn sibling ()->String\n  +return concat(\"a\", \"b\")\n+end\n\
         +module M\n+fn concat (a:String, b:String)->String\n  +return \"other\"\n+end\n",
    );
    let cache = new_jit_cache();
    let _seam = jit_seam::hook(|| {});
    let input = CompileInput::of(&live);
    assert!(crate::compiler::compile(&live).is_ok() && crate::compiler::compile(input.program()).is_ok());
    assert_eq!(eval_compiled_or_interpreted_cached(&live, "root", &no_input(), Some(&cache), 1).unwrap(), ("11".into(), true));
    for operation in parser::parse("+module M\n+fn concat (a:String, b:String)->String [fail]\n  +return \"other\"\n+end\n").unwrap() {
        crate::validator::apply_and_validate(&mut live, &operation).unwrap();
    }
    live.rebuild_function_index();
    assert!(!input.matches(&live), "the module function changed compile admission but not the identity");
    assert!(crate::compiler::compile(&live).is_err());
    // Same revision number, warm slot: the old module is not used any more.
    assert_eq!(eval_compiled_or_interpreted_cached(&live, "root", &no_input(), Some(&cache), 1).unwrap(), ("11".into(), false));
    assert!(cache.lock().unwrap().is_none(), "the stale module was dropped and the refused program left no slot");
    assert_eq!(jit_seam::attempts(), 2);
}

/// The dispatch compiles `CompileInput::program()`, not the live program.
/// If `compile` ever reads a part of the program that the input leaves out,
/// compiling from the input and from the full program stop agreeing; this
/// test compares the two over every program available to the tests.
#[test]
fn jit_compiling_from_the_input_equals_compiling_from_the_program() {
    use crate::compiler::input::CompileInput;
    let mut programs: Vec<(String, ast::Program)> = crate::compiler::relocation_tests::example_programs();
    programs.push(("calls and strings".into(), program(crate::compiler::relocation_tests::CALLS_AND_STRINGS)));
    programs.push((
        "structs and modules".into(),
        program("+type Point = {x:Int, y:Int}\n+fn sum (p:Point)->Int\n  +return p.x + p.y\n+end\n+module Shapes\n+type Size = {w:Int, h:Int}\n+fn area (s:Size)->Int\n  +return s.w * s.h\n+end\n"),
    ));
    // Top-level functions come first in a source; `+module` switches context.
    let library = "+module Lib\n+fn one ()->Int\n  +return 1\n+end\n+fn risky ()->Int [fail]\n  +return 2\n+end\n";
    for (label, top_level) in [
        ("qualified call into a module", "+fn root ()->Int\n  +return Lib.one()\n+end\n"),
        ("bare call resolving into a module", "+fn root ()->Int\n  +return one()\n+end\n"),
        ("bare call to a fallible module function", "+fn root ()->Int\n  +return risky()\n+end\n"),
        ("qualified call to a fallible module function", "+fn root ()->Int\n  +return Lib.risky() + 1\n+end\n"),
        ("module call in a condition", "+fn root ()->Int\n  +if Lib.one() > 0\n    +return 1\n  +end\n  +return 0\n+end\n"),
        ("top-level function wins over a module function", "+fn one ()->Int\n  +return 7\n+end\n+fn root ()->Int\n  +return one()\n+end\n"),
        ("no call into the module at all", "+fn root ()->Int\n  +return 5\n+end\n"),
    ] {
        programs.push((label.into(), program(&format!("{top_level}{library}"))));
    }
    // A module function shadowing a builtin name, pure and fallible.
    for (label, effects) in [("module function shadowing concat", ""), ("fallible module function shadowing concat", " [fail]")] {
        programs.push((
            label.into(),
            program(&format!(
                "+fn root ()->Int\n  +return 11\n+end\n+fn sibling ()->String\n  +return concat(\"a\", \"b\")\n+end\n\
                 +module M\n+fn concat (a:String, b:String)->String{effects}\n  +return \"other\"\n+end\n"
            )),
        ));
    }
    // A nested module, with a type the top level uses and a function of its own.
    let mut with_nested = program("+fn root (p:Deep)->Int\n  +return p.d\n+end\n+fn other ()->Int\n  +return deep()\n+end\n+module Outer\n+fn outer ()->Int\n  +return 1\n+end\n");
    let inner = program("+module Inner\n+type Deep = {d:Int}\n+fn deep ()->Int\n  +return 3\n+end\n").modules.remove(0);
    with_nested.modules[0] = nested(with_nested.modules[0].clone(), inner);
    with_nested.rebuild_function_index();
    programs.push(("nested module".into(), with_nested));
    let (mut compiled_both, mut refused_both) = (0, 0);
    for (label, live) in &programs {
        let _ = crate::compiler::relocation::take_emitted();
        let from_program = crate::compiler::compile(live);
        let code_from_program = crate::compiler::relocation::take_emitted();
        let from_input = crate::compiler::compile(CompileInput::of(live).program());
        let code_from_input = crate::compiler::relocation::take_emitted();
        match (from_program, from_input) {
            (Ok(full), Ok(reduced)) => {
                compiled_both += 1;
                assert_eq!(full.relocations, reduced.relocations, "{label}: different relocations");
                assert!(!code_from_program.is_empty() || live.functions.is_empty(), "{label}: no machine code was recorded");
                assert!(code_from_program == code_from_input, "{label}: the machine code differs");
            }
            (Err(full), Err(reduced)) => {
                refused_both += 1;
                assert_eq!(full.to_string(), reduced.to_string(), "{label}: refused for different reasons");
            }
            (full, reduced) => panic!("{label}: full program compiled={}, input compiled={}", full.is_ok(), reduced.is_ok()),
        }
    }
    assert!(compiled_both >= 5 && refused_both >= 5, "{compiled_both} compiled, {refused_both} refused");
}

/// Informational: what the identity check costs on a warm hit.
#[test]
#[ignore = "measurement"]
fn jit_cache_identity_cost_measurement() {
    use crate::compiler::input::CompileInput;
    use std::hint::black_box;
    let mut source = String::from("+fn probe ()->Int\n  +return 11\n+end\n");
    for index in 0..300 {
        source.push_str(&format!("+type T{index} = {{a:Int, b:Int}}\n+fn f{index} (n:Int)->Int\n  +return n + {index}\n+end\n"));
    }
    let live = program(&source);
    let input = CompileInput::of(&live);
    let rounds = 200_000u32;
    let start = std::time::Instant::now();
    for _ in 0..rounds {
        assert!(black_box(&input).matches(black_box(&live)));
    }
    let check = start.elapsed() / rounds;
    let cache = new_jit_cache();
    assert_eq!(eval_compiled_or_interpreted_cached(&live, "probe", &no_input(), Some(&cache), 1).unwrap(), ("11".into(), true));
    let calls = 20_000u32;
    let start = std::time::Instant::now();
    for _ in 0..calls {
        black_box(eval_compiled_or_interpreted_cached(&live, "probe", &no_input(), Some(&cache), 1).unwrap());
    }
    let call = start.elapsed() / calls;
    // The same number of functions without any type declaration: the part of
    // the check that is pointer comparison only.
    let mut functions_only = String::from("+fn probe ()->Int\n  +return 11\n+end\n");
    for index in 0..300 {
        functions_only.push_str(&format!("+fn f{index} (n:Int)->Int\n  +return n + {index}\n+end\n"));
    }
    let plain = program(&functions_only);
    let plain_input = CompileInput::of(&plain);
    let start = std::time::Instant::now();
    for _ in 0..rounds {
        assert!(black_box(&plain_input).matches(black_box(&plain)));
    }
    let pointers = start.elapsed() / rounds;
    let start = std::time::Instant::now();
    for _ in 0..2_000 {
        black_box(black_box(&live).clone());
    }
    let clone = start.elapsed() / 2_000;
    // A program shaped like the live node: everything in modules.
    let mut modular = String::from("+fn probe ()->Int\n  +return 11\n+end\n");
    for module in 0..10 {
        modular.push_str(&format!("+module M{module}\n"));
        for index in 0..40 {
            modular.push_str(&format!("+fn f{index} (n:Int)->Int\n  +return n + {index}\n+end\n"));
        }
    }
    let modular = program(&modular);
    let modular_input = CompileInput::of(&modular);
    let start = std::time::Instant::now();
    for _ in 0..rounds {
        assert!(black_box(&modular_input).matches(black_box(&modular)));
    }
    let modules = start.elapsed() / rounds;
    let start = std::time::Instant::now();
    for _ in 0..2_000 {
        black_box(CompileInput::of(black_box(&modular)));
    }
    let build = start.elapsed() / 2_000;
    eprintln!(
        "JIT_IDENTITY_COST 301 functions + 300 struct types: check={check:?}, whole warm call={call:?}; 301 functions, no types: check={pointers:?}; \
         for scale, cloning that program (done per API call)={clone:?}; 1 top-level + 400 functions in 10 modules, no types: \
         check={modules:?}, building the input (once per compile)={build:?}"
    );
}
