//! Measurement (#114): how many corpus functions compile for the VM, printed for the
//! record. The assertion guards against over-refusal: no refused function may be one whose
//! whole reachable graph uses only ordinary field names (those are the ones a struct-only
//! proof failure would wrongly send to the tree walker).
use crate::{ast, parser, validator, vm};
use std::path::Path;

fn load(path: &Path, program: &mut ast::Program) -> bool {
    let Ok(source) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(ops) = parser::parse(&source) else {
        return false;
    };
    let mut candidate = program.clone();
    for op in &ops {
        if validator::apply_and_validate(&mut candidate, op).is_err() {
            return false;
        }
    }
    candidate.rebuild_function_index();
    *program = candidate;
    true
}

fn functions(program: &ast::Program) -> Vec<(String, &ast::FunctionDecl)> {
    let mut out: Vec<_> = program
        .functions
        .iter()
        .map(|f| (f.name.clone(), f.as_ref()))
        .collect();
    fn walk<'a>(
        prefix: &str,
        modules: &'a [ast::Module],
        out: &mut Vec<(String, &'a ast::FunctionDecl)>,
    ) {
        for m in modules {
            let path = if prefix.is_empty() {
                m.name.clone()
            } else {
                format!("{prefix}.{}", m.name)
            };
            for f in &m.functions {
                out.push((format!("{path}.{}", f.name), f.as_ref()));
            }
            walk(&path, &m.modules, out);
        }
    }
    walk("", &program.modules, &mut out);
    out
}

fn measure(label: &str, programs: &[ast::Program]) -> (usize, usize) {
    let (mut ok, mut err) = (0, 0);
    let mut reasons = std::collections::BTreeMap::<String, usize>::new();
    let mut tree: Vec<String> = Vec::new();
    for program in programs {
        for (name, function) in functions(program) {
            match vm::compile_function(function, program) {
                Ok(_) => ok += 1,
                Err(e) => {
                    err += 1;
                    let text = e.to_string();
                    *reasons.entry(text.chars().take(70).collect()).or_default() += 1;
                    if text.contains("pseudo-field") {
                        // Reachable closure by callee spelling (get_function), own + callees.
                        let mut seen = std::collections::HashSet::new();
                        let mut todo = vec![function];
                        let mut pseudo = false;
                        while let Some(f) = todo.pop() {
                            if !seen.insert(f as *const _) {
                                continue;
                            }
                            let dump = format!("{:?}", f.body);
                            pseudo |= [
                                "len",
                                "is_empty",
                                "is_ok",
                                "is_err",
                                "unwrap",
                                "error",
                                "unwrap_err",
                            ]
                            .iter()
                            .any(|n| dump.contains(&format!("field: \"{n}\"")));
                            for part in dump.split("callee: \"").skip(1) {
                                let callee = part.split('"').next().unwrap();
                                if let Some(c) = program.get_function(callee) {
                                    todo.push(c);
                                }
                            }
                        }
                        tree.push(format!(
                            "{name}{}",
                            if pseudo {
                                ""
                            } else {
                                " [ordinary-fields-only]"
                            }
                        ));
                    }
                }
            }
        }
    }
    eprintln!(
        "BLAST {label}: functions={} vm_ok={ok} vm_refused={err}",
        ok + err
    );
    for (reason, n) in reasons {
        eprintln!("BLAST   {n} x {reason}");
    }
    eprintln!("BLAST   pseudo-field-refused: {tree:?}");
    assert!(
        !tree
            .iter()
            .any(|name| name.contains("[ordinary-fields-only]")),
        "{label}: over-refusal: {tree:?}"
    );
    (ok, err)
}

#[test]
fn blast_radius_over_examples_and_parity_fixtures() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for (label, dir, cumulative) in [
        ("examples/", "examples", false),
        (
            "parity examples",
            "tests/fixtures/python-parity/examples",
            false,
        ),
        (
            "parity library",
            "tests/fixtures/python-parity/library",
            true,
        ),
    ] {
        let mut files: Vec<_> = std::fs::read_dir(root.join(dir))
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "ax"))
            .collect();
        files.sort();
        assert!(!files.is_empty(), "{dir}");
        let mut programs = Vec::new();
        let mut shared = ast::Program::default();
        for file in &files {
            if cumulative {
                load(file, &mut shared);
            } else {
                let mut p = ast::Program::default();
                if load(file, &mut p) {
                    programs.push(p);
                }
            }
        }
        if cumulative {
            programs.push(shared);
        }
        measure(label, &programs);
    }
}
