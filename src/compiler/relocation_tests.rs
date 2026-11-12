//! #96: nothing the JIT emits may depend on the distance between separately
//! allocated blocks. These tests look at the relocations of the real machine
//! code; they do not need the blocks to be far apart to fail.
use super::admission_measurement_tests::program;
use super::{compile, failure_admission, is_compilable_function};
use crate::{ast, parser, validator};
use cranelift::codegen::binemit::Reloc;
use std::collections::BTreeSet;

/// User-to-user calls (plain, nested, recursive, mutually recursive, in a
/// condition, returning a string) and string literals (returned, passed to a
/// runtime helper): every construct that used to produce a PC-relative
/// relocation. No function takes a String parameter: the JIT mishandles the
/// length of string parameters (a separate, older defect, see the #96 report).
pub(crate) const CALLS_AND_STRINGS: &str = "\
+fn double (n:Int)->Int
  +return n * 2
+end
+fn quad (n:Int)->Int
  +return double(double(n))
+end
+fn fact (n:Int)->Int
  +if n <= 1
    +return 1
  +end
  +return n * fact(n - 1)
+end
+fn is_even (n:Int)->Int
  +if n == 0
    +return 1
  +end
  +return is_odd(n - 1)
+end
+fn is_odd (n:Int)->Int
  +if n == 0
    +return 0
  +end
  +return is_even(n - 1)
+end
+fn greeting ()->String
  +return \"hello\"
+end
+fn relay ()->String
  +return greeting()
+end
+fn phrase ()->String
  +return concat(greeting(), \" world\")
+end
+fn pick (n:Int)->String
  +if is_even(n) == 1
    +return concat(\"even\", \"!\")
  +end
  +return relay()
+end
+fn parity (n:Int)->Int
  +if is_even(n) > 0
    +return 2
  +end
  +return 1
+end
";

fn relocations_of<'a>(compiled: &'a super::CompiledProgram, function: &str) -> &'a [Reloc] {
    &compiled.relocations.iter().find(|(name, _)| name == function).unwrap_or_else(|| panic!("{function} was not compiled")).1
}

#[test]
fn jit_user_calls_and_string_literals_need_only_absolute_relocations() {
    let compiled = match compile(&program(CALLS_AND_STRINGS)) {
        Ok(compiled) => compiled,
        Err(error) => panic!("the program must compile without distance-limited relocations: {error}"),
    };
    // Each of these functions references another block (a callee or a string
    // literal), so each has relocations; the check below is not vacuous.
    for (function, at_least) in
        [("quad", 2), ("fact", 1), ("is_even", 1), ("is_odd", 1), ("greeting", 1), ("relay", 1), ("phrase", 3), ("pick", 4), ("parity", 1)]
    {
        let kinds = relocations_of(&compiled, function);
        assert!(kinds.len() >= at_least, "{function}: expected at least {at_least} relocations, got {kinds:?}");
    }
    assert!(relocations_of(&compiled, "double").is_empty(), "a leaf without literals references nothing");
    for (function, kinds) in &compiled.relocations {
        assert!(
            kinds.iter().all(|kind| *kind == Reloc::Abs8),
            "{function}: {kinds:?} contains a relocation that depends on block distance"
        );
    }
}

/// A string literal in every position the code generator handles, and a user
/// call in every statement form. All literals and all user calls go through
/// one code path each (`compile_expr` for `Literal::String`, `compile_call`
/// and the condition emitter for calls); this pins that no position bypasses
/// them. Only the relocation kinds are checked here, not results.
const POSITIONS: &[(&str, &str, &str, usize)] = &[
    ("literal returned", "+fn f ()->String\n  +return \"text\"\n+end\n", "f", 1),
    (
        "literal as argument",
        "+fn blank (s:String)->Int\n  +if NOT s\n    +return 1\n  +end\n  +return 0\n+end\n+fn f ()->Int\n  +return blank(\"text\")\n+end\n",
        "f",
        2,
    ),
    ("literals as concat operands", "+fn f ()->String\n  +return concat(\"a\", \"b\")\n+end\n", "f", 3),
    ("nested concat", "+fn f ()->String\n  +return concat(concat(\"a\", \"b\"), \"c\")\n+end\n", "f", 5),
    ("literals as equality operands", "+fn f ()->Int\n  +return \"a\" == \"b\"\n+end\n", "f", 3),
    (
        "literals in if/elif/else arms",
        "+fn f (n:Int)->String\n  +if n > 0\n    +return \"positive\"\n  +elif n < 0\n    +return \"negative\"\n  +else\n    +return \"zero\"\n  +end\n+end\n",
        "f",
        3,
    ),
    (
        "literal in a while body",
        "+fn f (n:Int)->String\n  +while n > 0\n    +return \"loop\"\n  +end\n  +return \"done\"\n+end\n",
        "f",
        2,
    ),
    (
        "literals in match arms",
        "+fn f (n:Int)->String\n  +match n\n  +case 1\n    +return \"one\"\n  +case _\n    +return \"other\"\n  +end\n+end\n",
        "f",
        2,
    ),
    ("user call in +call", "+fn g ()->Int\n  +return 1\n+end\n+fn f ()->Int\n  +call n:Int = g()\n  +return n\n+end\n", "f", 1),
    ("user call returning a string", "+fn g ()->String\n  +return \"text\"\n+end\n+fn f ()->String\n  +return g()\n+end\n", "f", 1),
    ("user call in let", "+fn g ()->Int\n  +return 1\n+end\n+fn f ()->Int\n  +let x:Int = g()\n  +return x\n+end\n", "f", 1),
    ("user call as argument", "+fn g (n:Int)->Int\n  +return n\n+end\n+fn f ()->Int\n  +return g(g(g(1)))\n+end\n", "f", 3),
    ("user call in arithmetic", "+fn g ()->Int\n  +return 1\n+end\n+fn f ()->Int\n  +return g() + g() * 2\n+end\n", "f", 2),
    (
        "user call in a condition",
        "+fn g (n:Int)->Int\n  +return n\n+end\n+fn f (n:Int)->Int\n  +if g(n) > 0\n    +return 1\n  +end\n  +return 0\n+end\n",
        "f",
        1,
    ),
    (
        "user call in a logical condition",
        "+fn g ()->Bool\n  +return true\n+end\n+fn h ()->Bool\n  +return false\n+end\n+fn f ()->Int\n  +if g() AND NOT h()\n    +return 1\n  +end\n  +return 0\n+end\n",
        "f",
        2,
    ),
    (
        "user call in a while condition and body",
        "+fn g (n:Int)->Int\n  +return n\n+end\n+fn f (n:Int)->Int\n  +while g(n) > 0\n    +return g(n)\n  +end\n  +return 0\n+end\n",
        "f",
        2,
    ),
];

/// Positions that do not exist in compiled code today because the JIT refuses
/// the construct. If one of them starts to compile, it belongs in `POSITIONS`.
const NOT_COMPILED: &[(&str, &str)] = &[
    (
        "literal as a field of a struct literal argument",
        "+type Named = {label:String, n:Int}\n+fn g (p:Named)->String\n  +return p.label\n+end\n+fn f ()->String\n  +return g({label: \"text\", n: 1})\n+end\n",
    ),
    (
        "literal as a field of a struct binding",
        "+type Named = {label:String, n:Int}\n+fn f ()->Int\n  +let p:Named = {label: \"text\", n: 1}\n  +return p.n\n+end\n",
    ),
    ("literal compared in a condition", "+fn f (s:String)->Int\n  +if s == \"a\"\n    +return 1\n  +end\n  +return 0\n+end\n"),
    // Refused by `string_length_admission`: a string whose length the code
    // generator does not know, where the length is consumed. A string binding
    // that is never read that way is dead code and leaves no relocation, so
    // these positions have no compiled form.
    ("literal bound by let and returned", "+fn f ()->String\n  +let s:String = \"text\"\n  +return s\n+end\n"),
    ("literal assigned by set and returned", "+fn f ()->String\n  +let s:String = \"a\"\n  +set s = \"b\"\n  +return s\n+end\n"),
    ("literal compared with a parameter", "+fn f (s:String)->Int\n  +return s != \"a\"\n+end\n"),
    ("string result bound by +call and returned", "+fn g ()->String\n  +return \"text\"\n+end\n+fn f ()->String\n  +call s:String = g()\n  +return s\n+end\n"),
];

#[test]
fn jit_every_literal_and_call_position_needs_only_absolute_relocations() {
    for (label, source) in NOT_COMPILED {
        assert!(compile(&program(source)).is_err(), "{label}: compiles now; add it to POSITIONS");
    }
    let mut problems = Vec::new();
    for (label, source, function, at_least) in POSITIONS {
        let compiled = match compile(&program(source)) {
            Ok(compiled) => compiled,
            Err(error) => {
                problems.push(format!("{label}: must compile: {error}"));
                continue;
            }
        };
        let kinds = relocations_of(&compiled, function);
        if kinds.len() < *at_least {
            problems.push(format!("{label}: expected at least {at_least} relocations, got {kinds:?}"));
        }
        for (name, kinds) in &compiled.relocations {
            if !kinds.iter().all(|kind| *kind == Reloc::Abs8) {
                problems.push(format!("{label}: {name}: {kinds:?}"));
            }
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

#[test]
fn jit_results_with_absolute_relocations_are_correct() {
    let mut compiled = compile(&program(CALLS_AND_STRINGS)).unwrap();
    assert_eq!(compiled.call_i64("quad", &[5]).unwrap(), 20);
    assert_eq!(compiled.call_i64("fact", &[10]).unwrap(), 3_628_800);
    assert_eq!(compiled.call_i64("is_even", &[10]).unwrap(), 1);
    assert_eq!(compiled.call_i64("is_even", &[7]).unwrap(), 0);
    assert_eq!(compiled.call_i64("is_odd", &[7]).unwrap(), 1);
    assert_eq!(compiled.call_string("greeting", &[]).unwrap(), "hello");
    assert_eq!(compiled.call_string("relay", &[]).unwrap(), "hello");
    assert_eq!(compiled.call_string("phrase", &[]).unwrap(), "hello world");
    assert_eq!(compiled.call_string("pick", &[4]).unwrap(), "even!");
    assert_eq!(compiled.call_string("pick", &[5]).unwrap(), "hello");
    assert_eq!((compiled.call_i64("parity", &[4]).unwrap(), compiled.call_i64("parity", &[5]).unwrap()), (2, 1));
    // Deep recursion through absolute calls.
    assert_eq!(compiled.call_i64("is_even", &[20_000]).unwrap(), 1);
}

#[test]
fn jit_relocation_check_refuses_a_distance_limited_relocation() {
    // The guard itself: a function declared the default (colocated) way must be
    // refused with an ordinary error before anything is finalized.
    use cranelift::prelude::*;
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{Linkage, Module};
    let mut flags = settings::builder();
    flags.set("is_pic", "false").unwrap();
    let isa = cranelift_native::builder().unwrap().finish(settings::Flags::new(flags)).unwrap();
    let mut module = JITModule::new(JITBuilder::with_isa(isa, cranelift_module::default_libcall_names()));
    let mut signature = module.make_signature();
    signature.returns.push(AbiParam::new(types::I64));
    let callee = module.declare_function("callee", Linkage::Local, &signature).unwrap();
    let caller = module.declare_function("caller", Linkage::Local, &signature).unwrap();
    for (id, target) in [(callee, None), (caller, Some(callee))] {
        let mut ctx = module.make_context();
        ctx.func.signature = signature.clone();
        let mut builder_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
        let block = builder.create_block();
        builder.switch_to_block(block);
        builder.seal_block(block);
        let value = match target {
            None => builder.ins().iconst(types::I64, 7),
            Some(target) => {
                // The default declaration: colocated, i.e. `call rel32` on x86-64.
                let reference = module.declare_func_in_func(target, builder.func);
                let call = builder.ins().call(reference, &[]);
                builder.inst_results(call)[0]
            }
        };
        builder.ins().return_(&[value]);
        builder.finalize();
        module.define_function(id, &mut ctx).unwrap();
        let checked = super::relocation::check(&ctx, "probe");
        match target {
            None => assert!(checked.unwrap().is_empty()),
            Some(_) => {
                let error = checked.unwrap_err().to_string();
                assert!(error.contains("relocation") && error.contains("#96"), "{error}");
            }
        }
        module.clear_context(&mut ctx);
    }
    // Never finalized: the relocation that could panic is never applied.
}

/// Every example program, parsed like the admission inventory does.
pub(crate) fn example_programs() -> Vec<(String, ast::Program)> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    let mut paths: Vec<_> = std::fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "ax"))
        .collect();
    paths.sort();
    let mut programs = Vec::new();
    for path in paths {
        let mut next = ast::Program::default();
        let valid = parser::parse(&std::fs::read_to_string(&path).unwrap())
            .map(|operations| operations.iter().all(|operation| validator::apply_and_validate(&mut next, operation).is_ok()))
            .unwrap_or(false);
        if valid {
            next.rebuild_function_index();
            programs.push((path.file_name().unwrap().to_string_lossy().into_owned(), next));
        }
    }
    programs
}

/// `root` together with the top-level functions it needs, compiled alone.
/// The compiler names a missing callee, which is how the set is grown; a
/// function that cannot be compiled even with its callees gives `None`.
fn compile_with_callees(program: &ast::Program, root: &str) -> Option<super::CompiledProgram> {
    let mut wanted: BTreeSet<String> = BTreeSet::from([root.to_string()]);
    loop {
        let mut reduced = program.clone();
        reduced.functions.retain(|function| wanted.contains(&function.name));
        reduced.rebuild_function_index();
        match compile(&reduced) {
            Ok(compiled) => return Some(compiled),
            Err(error) => {
                let text = error.to_string();
                let missing = text.strip_prefix("undefined function `").and_then(|rest| rest.split('`').next())?;
                if !program.functions.iter().any(|function| function.name == missing) || !wanted.insert(missing.to_string()) {
                    return None;
                }
            }
        }
    }
}

#[test]
fn jit_examples_need_only_absolute_relocations() {
    let (mut eligible, mut functions, mut with_relocations, mut whole_programs) = (0, 0, 0, 0);
    for (file, program) in example_programs() {
        // What production compiles: the whole program, when all of it is compilable.
        if let Ok(compiled) = compile(&program) {
            whole_programs += 1;
            for (function, kinds) in &compiled.relocations {
                assert!(kinds.iter().all(|kind| *kind == Reloc::Abs8), "{file}: {function}: {kinds:?}");
            }
        }
        // Every function the JIT would accept as a root, with its callees.
        for function in &program.functions {
            if !is_compilable_function(function) || failure_admission::check_root(function, &program).is_err() {
                continue;
            }
            eligible += 1;
            // Not every admitted root compiles (unsupported builtins and
            // constructs are ordinary compile errors); those never run on the JIT.
            let Some(compiled) = compile_with_callees(&program, &function.name) else { continue };
            functions += 1;
            for (name, kinds) in &compiled.relocations {
                with_relocations += usize::from(!kinds.is_empty());
                assert!(kinds.iter().all(|kind| *kind == Reloc::Abs8), "{file}: {name} (for {}): {kinds:?}", function.name);
            }
        }
    }
    eprintln!(
        "JIT_RELOCATIONS examples: admitted_roots={eligible} compiled_roots={functions} functions_with_relocations={with_relocations} \
         whole_programs={whole_programs}"
    );
    assert!(functions >= 5, "the examples no longer exercise the JIT: {functions} roots compiled");
    assert!(with_relocations > 0, "no example function references another block; the check would be vacuous");
}

/// Informational, never a gate: the cost of calling through an absolute
/// address, against the colocated code emitted before #96, in one process.
/// Run with `--ignored --nocapture`. The colocated variant is exactly the
/// code that can panic in a process with distant allocations.
#[test]
#[ignore = "measurement"]
fn jit_call_cost_measurement() {
    use std::hint::black_box;
    use std::time::{Duration, Instant};
    let source = "\
+fn fib (n:Int)->Int
  +if n < 2
    +return n
  +end
  +return fib(n - 1) + fib(n - 2)
+end
+fn word ()->String
  +return \"hello\"
+end
+fn phrase ()->String
  +return concat(word(), \" world\")
+end
";
    let program = program(source);
    let mut absolute = compile(&program).unwrap();
    super::relocation::COLOCATED_FOR_TESTS.with(|flag| flag.set(true));
    let colocated = compile(&program);
    super::relocation::COLOCATED_FOR_TESTS.with(|flag| flag.set(false));
    let mut colocated = colocated.unwrap();
    assert!(colocated.relocations.iter().any(|(_, kinds)| kinds.contains(&Reloc::X86CallPCRel4)), "the comparison variant is not the old code");
    let time = |work: &mut dyn FnMut()| {
        let start = Instant::now();
        work();
        start.elapsed()
    };
    let mut best = [[Duration::MAX; 3]; 2];
    for _ in 0..15 {
        for (variant, compiled) in [&mut absolute, &mut colocated].into_iter().enumerate() {
            let fib = time(&mut || assert_eq!(compiled.call_i64("fib", black_box(&[32])).unwrap(), 2_178_309));
            let word = time(&mut || {
                for _ in 0..200_000 {
                    assert_eq!(compiled.call_string("word", black_box(&[])).unwrap().len(), 5);
                }
            });
            let phrase = time(&mut || {
                for _ in 0..200_000 {
                    assert_eq!(compiled.call_string("phrase", black_box(&[])).unwrap().len(), 11);
                }
            });
            for (slot, sample) in best[variant].iter_mut().zip([fib, word, phrase]) {
                *slot = (*slot).min(sample);
            }
        }
    }
    for (label, row) in ["absolute (now)", "colocated (before)"].iter().zip(best) {
        eprintln!("JIT_CALL_COST {label}: fib(32) [about 7 million calls]={:?} word()x200000={:?} phrase()x200000={:?}", row[0], row[1], row[2]);
    }
}
