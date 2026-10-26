//! Strict, offline corpus inventory and roundtrip evidence. An invalid fixture
//! is reported as invalid, never counted as represented or silently omitted.
use super::ast_equality as equality;
mod scope;
mod tests;

use crate::{ast, parser, validator};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, path::Path};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum Status {
    Verified,
    InputError,
    InvalidProgram,
    Unsupported,
    AstMismatch,
    NonIdempotent,
}

#[derive(Debug, Serialize)]
struct Entry {
    file: String,
    target: String,
    kind: String,
    status: Status,
    detail: String,
}

#[derive(Default, Debug, Serialize)]
struct Report {
    files: usize,
    entries: Vec<Entry>,
}

impl Report {
    fn record(&mut self, file: &str, target: &str, kind: &str, result: Check) {
        let (status, detail) = result.err().unwrap_or((Status::Verified, String::new()));
        self.entries.push(Entry {
            file: file.into(),
            target: target.into(),
            kind: kind.into(),
            status,
            detail,
        });
    }

    fn count(&self, status: Status) -> usize {
        self.entries.iter().filter(|e| e.status == status).count()
    }

    fn success(&self) -> bool {
        self.files > 0
            && !self.entries.is_empty()
            && self.entries.iter().all(|e| e.status == Status::Verified)
    }
}

type Check = Result<(), (Status, String)>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    files: Vec<String>,
}

struct Fixture {
    file: String,
    operations: Vec<parser::Operation>,
}

fn load(base: &Path, report: &mut Report) -> Vec<Fixture> {
    let result = std::fs::read_to_string(base.join("manifest.json"))
        .map_err(|e| e.to_string())
        .and_then(|text| serde_json::from_str::<Manifest>(&text).map_err(|e| e.to_string()));
    let manifest = match result {
        Ok(m) => m,
        Err(e) => {
            report.record("manifest.json", "", "input", Err((Status::InputError, e)));
            return vec![];
        }
    };
    let expected: BTreeSet<_> = manifest.files.iter().cloned().collect();
    if expected.is_empty() || expected.len() != manifest.files.len() {
        report.record(
            "manifest.json",
            "",
            "input",
            Err((
                Status::InputError,
                "empty or duplicate file inventory".into(),
            )),
        );
    }
    let mut actual = BTreeSet::new();
    for dir in ["examples", "library"] {
        let entries = match std::fs::read_dir(base.join(dir)) {
            Ok(entries) => entries,
            Err(e) => {
                report.record(dir, "", "input", Err((Status::InputError, e.to_string())));
                continue;
            }
        };
        for entry in entries {
            match entry {
                Ok(entry) => {
                    let file = format!("{dir}/{}", entry.file_name().to_string_lossy());
                    if !entry.file_type().is_ok_and(|t| t.is_file()) {
                        report.record(
                            &file,
                            "",
                            "input",
                            Err((Status::InputError, "fixture must be a regular file".into())),
                        );
                    }
                    actual.insert(file);
                }
                Err(e) => report.record(dir, "", "input", Err((Status::InputError, e.to_string()))),
            }
        }
    }
    for file in actual.symmetric_difference(&expected) {
        report.record(
            file,
            "",
            "input",
            Err((
                Status::InputError,
                "manifest/directory inventory mismatch".into(),
            )),
        );
    }
    let mut fixtures = Vec::new();
    for file in expected {
        if !valid_fixture_path(&file) {
            report.record(
                &file,
                "",
                "input",
                Err((Status::InputError, "invalid fixture path".into())),
            );
            continue;
        }
        report.files += 1;
        if !std::fs::symlink_metadata(base.join(&file)).is_ok_and(|m| m.file_type().is_file()) {
            report.record(
                &file,
                "",
                "input",
                Err((Status::InputError, "missing or non-regular fixture".into())),
            );
            continue;
        }
        let parsed = std::fs::read_to_string(base.join(&file))
            .map_err(|e| e.to_string())
            .and_then(|source| parser::parse(&source).map_err(|e| e.to_string()));
        match parsed {
            Ok(operations) if !operations.is_empty() => fixtures.push(Fixture { file, operations }),
            Ok(_) => report.record(
                &file,
                "",
                "input",
                Err((Status::InputError, "empty fixture".into())),
            ),
            Err(e) => report.record(&file, "", "input", Err((Status::InputError, e))),
        }
    }
    fixtures
}

fn valid_fixture_path(file: &str) -> bool {
    let Some((dir, name)) = file.split_once('/') else {
        return false;
    };
    matches!(dir, "examples" | "library")
        && !name.contains('/')
        && !name.contains('\\')
        && name.ends_with(".ax")
        && name != ".ax"
        && !name.starts_with('.')
}

fn build_into(program: &mut ast::Program, operations: &[parser::Operation]) -> Result<(), String> {
    let mut candidate = program.clone();
    for op in operations {
        validator::apply_and_validate(&mut candidate, op).map_err(|e| e.to_string())?;
    }
    candidate.rebuild_function_index();
    *program = candidate;
    Ok(())
}

fn run(base: &Path) -> Report {
    let mut report = Report::default();
    let fixtures = load(base, &mut report);
    let mut library = ast::Program::default();
    let mut failed = BTreeSet::new();
    for fixture in fixtures.iter().filter(|f| f.file.starts_with("library/")) {
        if let Err(e) = build_into(&mut library, &fixture.operations) {
            report.record(
                &fixture.file,
                "",
                "validation",
                Err((Status::InvalidProgram, e)),
            );
            failed.insert(fixture.file.clone());
        }
    }
    report.record(
        "library/*",
        "",
        "program",
        if failed.is_empty() {
            check_program(&library)
        } else {
            Err((
                Status::InvalidProgram,
                "library has validation failures".into(),
            ))
        },
    );
    for fixture in fixtures {
        if fixture.file.starts_with("library/") {
            inventory(
                &library,
                &fixture,
                failed.contains(&fixture.file),
                &mut report,
            );
        } else {
            let mut program = ast::Program::default();
            let invalid = if let Err(e) = build_into(&mut program, &fixture.operations) {
                report.record(
                    &fixture.file,
                    "",
                    "validation",
                    Err((Status::InvalidProgram, e)),
                );
                true
            } else {
                false
            };
            inventory(&program, &fixture, invalid, &mut report);
            report.record(
                &fixture.file,
                "",
                "program",
                if invalid {
                    Err((Status::InvalidProgram, "fixture failed validation".into()))
                } else {
                    check_program(&program)
                },
            );
        }
    }
    report
}

fn inventory(program: &ast::Program, fixture: &Fixture, invalid: bool, report: &mut Report) {
    inventory_ops(
        program,
        &fixture.file,
        "",
        &fixture.operations,
        invalid,
        report,
    );
    // Applying the whole source coalesces repeated declarations. Such a source
    // sequence needs its own representation; the final Program is not evidence
    // that an earlier overwritten declaration roundtripped.
    let mut seen = BTreeSet::new();
    let mut duplicates = BTreeSet::new();
    duplicate_targets("", &fixture.operations, &mut seen, &mut duplicates);
    for target in duplicates {
        for entry in report
            .entries
            .iter_mut()
            .filter(|e| e.file == fixture.file && e.target == target)
        {
            if entry.status != Status::InvalidProgram {
                entry.status = Status::Unsupported;
                entry.detail = "repeated declaration requires source-sequence roundtrip, not final-state comparison".into();
            }
        }
    }
}

fn duplicate_targets(
    scope: &str,
    ops: &[parser::Operation],
    seen: &mut BTreeSet<String>,
    duplicates: &mut BTreeSet<String>,
) {
    for op in ops {
        let name = match op {
            parser::Operation::Module(m) => Some(m.name.as_str()),
            parser::Operation::Function(f) => Some(f.name.as_str()),
            parser::Operation::Type(t) => Some(t.name.as_str()),
            _ => None,
        };
        if let Some(name) = name {
            let target = qualified(scope, name);
            if !seen.insert(target.clone()) {
                duplicates.insert(target.clone());
            }
            if let parser::Operation::Module(m) = op {
                duplicate_targets(&target, &m.body, seen, duplicates);
            }
        }
    }
}

fn inventory_ops(
    program: &ast::Program,
    file: &str,
    scope: &str,
    ops: &[parser::Operation],
    invalid: bool,
    report: &mut Report,
) {
    use parser::Operation as Op;
    for (index, op) in ops.iter().enumerate() {
        let fallback = || {
            Err((
                if invalid {
                    Status::InvalidProgram
                } else {
                    Status::Unsupported
                },
                if invalid {
                    "parent fixture failed validation".into()
                } else {
                    "no Python roundtrip surface for this operation".into()
                },
            ))
        };
        match op {
            Op::Module(module) => {
                let target = qualified(scope, &module.name);
                let check = if invalid {
                    fallback()
                } else {
                    check_module(program, &target)
                };
                report.record(file, &target, "module", check);
                if module.doc.is_some() {
                    report.record(
                        file,
                        &target,
                        "module-doc",
                        if invalid {
                            fallback()
                        } else {
                            check_module(program, &target)
                        },
                    );
                }
                inventory_ops(program, file, &target, &module.body, invalid, report);
            }
            Op::Function(function) => {
                let target = qualified(scope, &function.name);
                let check = if invalid {
                    fallback()
                } else {
                    check_function(program, &target)
                };
                report.record(file, &target, "function", check);
            }
            Op::Type(ty) => {
                let target = qualified(scope, &ty.name);
                report.record(
                    file,
                    &target,
                    "type",
                    if invalid {
                        fallback()
                    } else {
                        check_type(program, &target)
                    },
                );
            }
            Op::SharedVar(_)
            | Op::Startup(_)
            | Op::Shutdown(_)
            | Op::ModuleSource(_)
            | Op::Route { .. }
            | Op::Doc { .. } => {
                let result = if invalid || scope.is_empty() {
                    fallback()
                } else {
                    check_module(program, scope)
                };
                report.record(
                    file,
                    &format!("{scope}#{index}"),
                    &operation_kind(op),
                    result,
                );
            }
            // Tests and runtime commands must remain visible: validator is a
            // no-op for several of these, so Program equality cannot cover them.
            _ => report.record(
                file,
                &format!("{scope}#{index}"),
                &operation_kind(op),
                fallback(),
            ),
        }
    }
}

fn qualified(scope: &str, name: &str) -> String {
    if scope.is_empty() {
        name.into()
    } else {
        format!("{scope}.{name}")
    }
}

fn operation_kind(op: &parser::Operation) -> String {
    // Variant name only: never print source expressions or literal values.
    format!("{op:?}")
        .split(['(', '{', ' '])
        .next()
        .unwrap_or("operation")
        .to_owned()
}

fn rebuilt(
    text: &str,
    module: &str,
    catalog: &ast::Program,
) -> Result<ast::Program, (Status, String)> {
    let lowered = super::lower_module_with_program(text, module, catalog)
        .map_err(|e| (Status::Unsupported, e.to_string()))?;
    let mut program = ast::Program::default();
    build_into(&mut program, &lowered.operations).map_err(|e| (Status::Unsupported, e))?;
    Ok(program)
}

fn check_function(program: &ast::Program, target: &str) -> Check {
    let original = program.get_function(target).ok_or((
        Status::InvalidProgram,
        "function missing after validation".into(),
    ))?;
    let module = target.rsplit_once('.').map_or("", |(module, _)| module);
    scope::check_function(program, module, original).map_err(|e| (Status::InvalidProgram, e))?;
    let Some((module, _)) = target.rsplit_once('.') else {
        return Err((
            Status::Unsupported,
            "production Python function printer/lowerer has no top-level scope surface".into(),
        ));
    };
    let text =
        super::print_function(program, target).map_err(|e| (Status::Unsupported, e.to_string()))?;
    let fresh = rebuilt(&text, module, program)?;
    let candidate = fresh.get_function(target).ok_or((
        Status::AstMismatch,
        "function missing after re-lowering".into(),
    ))?;
    if !equality::function(original, candidate) {
        return Err((
            Status::AstMismatch,
            format!(
                "{} changed (only typed NodeIds ignored)",
                equality::function_difference(original, candidate)
            ),
        ));
    }
    let mut second_catalog = program.clone();
    *second_catalog.get_function_mut(target).unwrap() = candidate.clone();
    let second = super::print_function(&second_catalog, target)
        .map_err(|e| (Status::Unsupported, e.to_string()))?;
    same_text(&text, &second)?;
    let again = rebuilt(&second, module, &second_catalog)?;
    if !equality::function(original, again.get_function(target).unwrap()) {
        return Err((
            Status::AstMismatch,
            "second lowering changed function".into(),
        ));
    }
    Ok(())
}

fn find_module<'a>(program: &'a ast::Program, target: &str) -> Option<&'a ast::Module> {
    let mut modules = &program.modules;
    let mut result = None;
    for part in target.split('.') {
        let module = modules.iter().find(|m| m.name == part)?;
        result = Some(module);
        modules = &module.modules;
    }
    result
}

fn check_module(program: &ast::Program, target: &str) -> Check {
    let original = find_module(program, target).ok_or((
        Status::InvalidProgram,
        "module missing after validation".into(),
    ))?;
    for function in &original.functions {
        scope::check_function(program, target, function)
            .map_err(|e| (Status::InvalidProgram, e))?;
    }
    let text =
        super::print_module(program, target).map_err(|e| (Status::Unsupported, e.to_string()))?;
    let fresh = rebuilt(&text, target, program)?;
    let candidate = find_module(&fresh, target)
        .ok_or((Status::AstMismatch, "module missing after lowering".into()))?;
    if !equality::module(original, candidate) {
        return Err((
            Status::AstMismatch,
            format!(
                "{} changed (only typed NodeIds ignored)",
                equality::module_difference(original, candidate)
            ),
        ));
    }
    let mut second_catalog = program.clone();
    let slot = second_catalog
        .modules
        .iter_mut()
        .find(|m| m.name == target)
        .ok_or((
            Status::Unsupported,
            "nested module printing is not available".into(),
        ))?;
    *slot = candidate.clone();
    second_catalog.rebuild_function_index();
    let second = super::print_module(&second_catalog, target)
        .map_err(|e| (Status::Unsupported, e.to_string()))?;
    same_text(&text, &second)?;
    let again = rebuilt(&second, target, program)?;
    if !equality::module(original, find_module(&again, target).unwrap()) {
        return Err((Status::AstMismatch, "second lowering changed module".into()));
    }
    Ok(())
}

fn check_type(program: &ast::Program, target: &str) -> Check {
    let Some((module, name)) = target.rsplit_once('.') else {
        return Err((
            Status::Unsupported,
            "production Python lowerer has no top-level type scope surface".into(),
        ));
    };
    let original = find_module(program, module)
        .and_then(|m| m.types.iter().find(|t| t.name() == name))
        .ok_or((
            Status::InvalidProgram,
            "type missing after validation".into(),
        ))?;
    let text =
        super::print_type(program, target).map_err(|e| (Status::Unsupported, e.to_string()))?;
    let fresh = rebuilt(&text, module, program)?;
    let candidate = find_module(&fresh, module)
        .and_then(|m| m.types.iter().find(|t| t.name() == name))
        .ok_or((Status::AstMismatch, "type missing after lowering".into()))?;
    if !equality::ty(original, candidate) {
        return Err((Status::AstMismatch, "type changed".into()));
    }
    let mut second_catalog = program.clone();
    let slot = second_catalog
        .modules
        .iter_mut()
        .find(|m| m.name == module)
        .and_then(|m| m.types.iter_mut().find(|t| t.name() == name))
        .ok_or((
            Status::Unsupported,
            "nested type printing is not available".into(),
        ))?;
    *slot = candidate.clone();
    second_catalog.rebuild_function_index();
    let second = super::print_type(&second_catalog, target)
        .map_err(|e| (Status::Unsupported, e.to_string()))?;
    same_text(&text, &second)?;
    let again = rebuilt(&second, module, program)?;
    let candidate = find_module(&again, module)
        .and_then(|m| m.types.iter().find(|t| t.name() == name))
        .unwrap();
    if !equality::ty(original, candidate) {
        return Err((Status::AstMismatch, "second lowering changed type".into()));
    }
    Ok(())
}

fn same_text(first: &str, second: &str) -> Check {
    if first == second {
        Ok(())
    } else {
        Err((
            Status::NonIdempotent,
            "second print differs byte-for-byte".into(),
        ))
    }
}

fn check_program(program: &ast::Program) -> Check {
    if !program.functions.is_empty() || !program.types.is_empty() {
        return Err((
            Status::Unsupported,
            "whole-program printer/lowerer lacks top-level scope".into(),
        ));
    }
    if program.modules.is_empty() {
        return Err((Status::Unsupported, "no represented declarations".into()));
    }
    let mut rebuilt_program = ast::Program::default();
    let mut texts = Vec::new();
    for module in &program.modules {
        check_module(program, &module.name)?;
        let text = super::print_module(program, &module.name)
            .map_err(|e| (Status::Unsupported, e.to_string()))?;
        let lowered = super::lower_module_with_program(&text, &module.name, program)
            .map_err(|e| (Status::Unsupported, e.to_string()))?;
        build_into(&mut rebuilt_program, &lowered.operations)
            .map_err(|e| (Status::Unsupported, e))?;
        texts.push(text);
    }
    if !equality::program(program, &rebuilt_program) {
        return Err((Status::AstMismatch, "aggregate program changed".into()));
    }
    for (module, text) in rebuilt_program.modules.iter().zip(texts) {
        let second = super::print_module(&rebuilt_program, &module.name)
            .map_err(|e| (Status::Unsupported, e.to_string()))?;
        same_text(&text, &second)?;
    }
    Ok(())
}

#[test]
fn python_parity() {
    let report = run(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/python-parity"
    )));
    eprintln!(
        "PARITY_SUMMARY files={} entries={} verified={} invalid={} input_errors={} unsupported={} ast_mismatches={} non_idempotent={}",
        report.files,
        report.entries.len(),
        report.count(Status::Verified),
        report.count(Status::InvalidProgram),
        report.count(Status::InputError),
        report.count(Status::Unsupported),
        report.count(Status::AstMismatch),
        report.count(Status::NonIdempotent)
    );
    let mut groups = std::collections::BTreeMap::<String, usize>::new();
    for entry in &report.entries {
        *groups
            .entry(format!("{:?}/{}", entry.status, entry.kind))
            .or_default() += 1;
    }
    eprintln!("PARITY_GROUPS {}", serde_json::to_string(&groups).unwrap());
    diagnostic_summary(&report);
    println!("{}", serde_json::to_string(&report).unwrap());
    assert!(
        report.success(),
        "strict corpus parity remains incomplete; see structured report"
    );
}

fn diagnostic_summary(report: &Report) {
    let mut paths = std::collections::BTreeMap::<String, usize>::new();
    let mut roots = Vec::new();
    let mut blockers = std::collections::BTreeMap::<String, usize>::new();
    for entry in &report.entries {
        if entry.status == Status::AstMismatch {
            let path = entry
                .detail
                .chars()
                .filter(|c| !c.is_ascii_digit())
                .collect();
            *paths.entry(path).or_default() += 1;
        }
        if entry.status == Status::InvalidProgram
            && (entry.kind == "validation"
                || (entry.kind == "function" && entry.detail != "parent fixture failed validation"))
        {
            roots.push((&entry.file, &entry.target, &entry.detail));
        }
        if entry.status == Status::Unsupported && entry.kind == "function" {
            let category = [
                "top-level",
                "and/or",
                "type mismatch",
                "IO/async",
                "reserved/unsafe",
                "parameter identifier",
                "shadowed",
                "fall through",
            ]
            .into_iter()
            .find(|label| entry.detail.contains(label))
            .unwrap_or("other");
            *blockers.entry(category.into()).or_default() += 1;
        }
    }
    eprintln!(
        "PARITY_AST_PATHS {}",
        serde_json::to_string(&paths).unwrap()
    );
    eprintln!(
        "PARITY_INVALID_ROOTS {}",
        serde_json::to_string(&roots).unwrap()
    );
    eprintln!(
        "PARITY_FUNCTION_BLOCKERS {}",
        serde_json::to_string(&blockers).unwrap()
    );
}
