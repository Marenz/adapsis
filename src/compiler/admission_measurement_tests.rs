//! Reproducible safe-control timing sample (informational, never a flaky speed gate).
use crate::{ast, compiler, eval, parser, validator};
use std::{hint::black_box, time::Instant};

#[test]
fn jit_admission_corpus_inventory() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for (label, folder, cumulative) in [
        ("examples", "examples", false),
        (
            "strict-library",
            "tests/fixtures/python-parity/library",
            true,
        ),
    ] {
        let mut paths: Vec<_> = std::fs::read_dir(root.join(folder))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "ax"))
            .collect();
        paths.sort();
        let mut accumulated = ast::Program::default();
        let mut programs = Vec::new();
        let mut invalid_files = 0;
        for path in paths {
            let mut next = if cumulative {
                accumulated.clone()
            } else {
                ast::Program::default()
            };
            let valid = parser::parse(&std::fs::read_to_string(path).unwrap())
                .map(|ops| {
                    ops.iter()
                        .all(|op| validator::apply_and_validate(&mut next, op).is_ok())
                })
                .unwrap_or(false);
            if !valid {
                invalid_files += 1;
                continue;
            }
            next.rebuild_function_index();
            if cumulative {
                accumulated = next;
            } else {
                programs.push(next);
            }
        }
        if cumulative {
            programs.push(accumulated);
        }
        let (mut total, mut local_eligible, mut graph_eligible, mut emitted_programs) =
            (0, 0, 0, 0);
        for program in &programs {
            emitted_programs +=
                usize::from(super::failure_admission::check_program(program).is_ok());
            let mut functions: Vec<_> = program.functions.iter().map(AsRef::as_ref).collect();
            let mut modules: Vec<_> = program.modules.iter().collect();
            while let Some(module) = modules.pop() {
                functions.extend(module.functions.iter().map(AsRef::as_ref));
                modules.extend(&module.modules);
            }
            for function in functions {
                total += 1;
                let local = compiler::is_compilable_function(function);
                local_eligible += usize::from(local);
                graph_eligible += usize::from(
                    local && super::failure_admission::check_root(function, program).is_ok(),
                );
            }
        }
        assert!(total > 0);
        eprintln!("JIT_ADMISSION {label}: functions={total} local_eligible={local_eligible} graph_eligible={graph_eligible} whole_program_safety_admitted={emitted_programs}/{} invalid_files={invalid_files}", programs.len());
    }
}

pub(super) fn program(source: &str) -> ast::Program {
    let mut program = ast::Program::default();
    for op in parser::parse(source).unwrap() {
        validator::apply_and_validate(&mut program, &op).unwrap();
    }
    program.rebuild_function_index();
    program
}

#[test]
fn jit_safe_numeric_control_cost_sample() {
    for (name, body, input, expected) in [
        ("arithmetic", "+return n * 3 + 7", 9, 34),
        (
            "branch",
            "+if n > 0\n    +return n + 1\n  +else\n    +return n - 1\n  +end",
            9,
            10,
        ),
    ] {
        let source = format!("+fn probe (n:Int)->Int\n  {body}\n+end\n");
        let program = program(&source);
        let start = Instant::now();
        let mut jit = compiler::compile(&program).unwrap();
        let compile = start.elapsed();
        let start = Instant::now();
        for _ in 0..10_000 {
            assert_eq!(
                jit.call_i64("probe", black_box(&[input])).unwrap(),
                expected
            );
        }
        let direct = start.elapsed();
        let cache = eval::new_jit_cache();
        let input = parser::parse_single_expr(&format!("{{n: {input}}}")).unwrap();
        let expected = expected.to_string();
        let warm =
            eval::eval_compiled_or_interpreted_cached(&program, "probe", &input, Some(&cache), 1)
                .unwrap();
        assert_eq!(warm, (expected.clone(), true));
        let start = Instant::now();
        for _ in 0..10_000 {
            let (value, used_jit) = eval::eval_compiled_or_interpreted_cached(
                &program,
                "probe",
                black_box(&input),
                Some(&cache),
                1,
            )
            .unwrap();
            assert!(used_jit, "safe control must remain on the JIT");
            assert_eq!(value, expected);
        }
        eprintln!(
            "JIT_SAFE_COST {name}: compile={compile:?} direct_10000={direct:?} cached_10000={:?}",
            start.elapsed()
        );
    }
}
