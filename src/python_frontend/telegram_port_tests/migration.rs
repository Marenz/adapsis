//! Operator-only migration preparation. Never called by the runtime or normal tests.
//! All diagnostics are categorical: neither parser errors nor live ASTs may be printed.
use super::*;
use std::{collections::BTreeMap, fs, path::Path};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::io::Write;
use sha2::{Digest, Sha256};
mod slots;
mod capture;
mod provenance;

type Checked<T> = Result<T, &'static str>;
const FORMAT: &str = include_str!("../../../tests/fixtures/telegram-python-port/origin/TgFormat.ax");
const FORMAT_FROZEN: &str = include_str!("../../../tests/fixtures/python-parity/library/TgFormat.ax");
const FORMAT_TESTS: &str = include_str!("../../../tests/fixtures/telegram-python-port/python/test_tg_format.py");

fn digest(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }

fn module(source: &str, name: &str) -> Checked<parser::ModuleDecl> {
    let ops = parser::parse(source).map_err(|_| "source parse failed")?;
    let mut modules = ops.into_iter().filter_map(|op| match op {
        parser::Operation::Module(m) => Some(m), _ => None,
    });
    let m = modules.next().ok_or("module absent")?;
    if m.name != name || modules.next().is_some() { return Err("module identity mismatch"); }
    Ok(m)
}

fn functions(m: &parser::ModuleDecl) -> BTreeMap<String, &parser::FunctionDecl> {
    m.body.iter().filter_map(|op| match op {
        parser::Operation::Function(f) => Some((f.name.clone(), f)), _ => None,
    }).collect()
}

fn exact<T: std::fmt::Debug>(a: &T, b: &T) -> bool { format!("{a:?}") == format!("{b:?}") }

/// Compare *every* parser AST before validation, including the malformed frozen voice
/// function. Only enumerated live literal slots are allowed to differ from the freeze.
fn verify_live(live: &parser::ModuleDecl, frozen: &parser::ModuleDecl, count: usize) -> Checked<()> {
    let mut expected = frozen.clone();
    if live.name == "TelegramBot" { slots::hydrate(&mut expected, live, false)?; }
    let a = functions(live);
    let b = functions(&expected);
    if a.len() != count || b.len() != count || a.keys().ne(b.keys()) {
        return Err("live function inventory drift");
    }
    for (name, f) in a {
        if !exact(f, b[&name]) { return Err("unaccounted live function AST drift"); }
    }
    let surface = |m: &parser::ModuleDecl| m.body.iter().filter(|op| !matches!(op,
        parser::Operation::Function(_) | parser::Operation::Test(_))).cloned().collect::<Vec<_>>();
    if !exact(&surface(live), &surface(&expected)) || live.doc != expected.doc {
        return Err("unaccounted live module surface drift");
    }
    Ok(())
}

fn build(modules: &[parser::ModuleDecl]) -> Checked<ast::Program> {
    let mut p = ast::Program::default();
    for m in modules {
        let mut m = m.clone();
        m.body.retain(|op| !matches!(op, parser::Operation::Test(_)));
        validator::apply_and_validate(&mut p, &parser::Operation::Module(m))
            .map_err(|_| "module validation failed")?;
    }
    p.rebuild_function_index();
    Ok(p)
}

fn same_program(a: &ast::Program, b: &ast::Program) -> Checked<()> {
    if !crate::python_frontend::ast_equality::program(a,b) || a.pending_routes != b.pending_routes {
        return Err("whole program AST or route mismatch");
    }
    Ok(())
}

/// Preserve grouped replacement sets. Combine carried corrected-origin cases with the
/// public paired cases, deduplicating exact cases only (never a target's earlier case).
fn tests(m: &parser::ModuleDecl, source: &str) -> Checked<Vec<parser::Operation>> {
    let lowered = crate::python_frontend::lower_tests(source, &m.name)
        .map_err(|_| "paired test lowering failed")?;
    let mut grouped: Vec<parser::TestMutation> = Vec::new();
    for op in m.body.iter().chain(&lowered.operations) {
        if let parser::Operation::Test(t) = op {
            let mut t = t.clone();
            if !t.function_name.contains('.') { t.function_name = format!("{}.{}",m.name,t.function_name); }
            if let Some(existing) = grouped.iter_mut().find(|x| x.function_name == t.function_name) {
                for case in t.cases {
                    if !existing.cases.iter().any(|old| exact(old,&case)) { existing.cases.push(case); }
                }
            } else { grouped.push(t); }
        }
    }
    Ok(grouped.into_iter().map(parser::Operation::Test).collect())
}

fn validate_cases(p: &ast::Program, tests: &[parser::Operation]) -> Checked<usize> {
    let mut count = 0;
    for op in tests {
        let parser::Operation::Test(t) = op else { return Err("unexpected test operation"); };
        let f = p.get_function(&t.function_name).ok_or("test target missing")?;
        if !f.effects.is_empty() { return Err("operator test attempted effectful case"); }
        for case in &t.cases {
            eval::eval_test_case(p, &t.function_name, case)
                .map_err(|_| "pure migrated test failed")?;
            count += 1;
        }
    }
    Ok(count)
}

fn private_file(dir: &Path, name: &str, bytes: &[u8]) -> Checked<()> {
    let mut f = fs::OpenOptions::new().write(true).create_new(true).mode(0o600)
        .open(dir.join(name)).map_err(|_| "private output creation failed")?;
    f.write_all(bytes).and_then(|_| f.sync_all()).map_err(|_| "private output write failed")
}

fn export(live_dir: &Path, output: &Path) -> Checked<serde_json::Value> {
    if output.parent() != Some(Path::new("/tmp/opencode")) || output.exists() {
        return Err("output must be a new direct child of /tmp/opencode");
    }
    let live_bot = fs::read_to_string(live_dir.join("TelegramBot.ax")).map_err(|_| "live bot read failed")?;
    let live_format = fs::read_to_string(live_dir.join("TgFormat.ax")).map_err(|_| "live format read failed")?;
    let bot = module(&live_bot,"TelegramBot")?;
    let format = module(&live_format,"TgFormat")?;
    verify_live(&bot,&module(FROZEN,"TelegramBot")?,42)?;
    verify_live(&format,&module(FORMAT_FROZEN,"TgFormat")?,18)?;
    let mut expected_bot = module(ORIGIN,"TelegramBot")?;
    let asr_changed = slots::hydrate(&mut expected_bot,&bot,true)?;
    let capture_path=std::env::var("ADAPSIS_MIGRATION_CAPTURE").ok();
    let capture_provenance=if let Some(path)=&capture_path {
        Some(provenance::verify(Path::new(path),&[digest(live_bot.as_bytes()),digest(live_format.as_bytes())])?)
    } else {None};
    if let Some(path)=&capture_path {capture::overlay(&mut expected_bot,Path::new(path))?;}
    let expected_format = module(FORMAT,"TgFormat")?;
    verify_authored_fixture()?;
    let expected = build(&[expected_format.clone(),expected_bot.clone()])?;
    if capture_path.is_some() {capture::startup_files(&expected_bot,&expected)?;}
    let (files, cases) = roundtrip(&expected, &expected_bot, &expected_format)?;
    fs::DirBuilder::new().mode(0o700).create(output).map_err(|_| "private directory creation failed")?;
    if fs::metadata(output).map_err(|_| "private directory stat failed")?.permissions().mode() & 0o777 != 0o700 {
        return Err("private directory mode incorrect");
    }
    let mut hashes = BTreeMap::new();
    for (name, bytes) in &files { private_file(output,name,bytes)?; hashes.insert(name.clone(),digest(bytes)); }
    let manifest = serde_json::json!({"schema_version":3,"superseded":false,
        "activation_allowed":false,"requires_manager_review_and_runtime_capture":true,
        "live_function_counts":[42,18],"output_function_counts":[48,18],
        "shared_slots_preserved":9,"runtime_shared_state_captured":capture_path.is_some(),
        "capture_provenance":capture_provenance,
        "live_sha256":[digest(live_bot.as_bytes()),digest(live_format.as_bytes())],
        "asr_endpoint_changed_from_fixture":asr_changed,"pure_cases_passed":cases,"artifacts":hashes,
        "callback_delivery_certified":false,"activation_performed":false});
    private_file(output,"manifest.json",serde_json::to_string_pretty(&manifest)
        .map_err(|_| "manifest encoding failed")?.as_bytes())?;
    Ok(manifest)
}

fn verify_authored_fixture() -> Checked<()> {
    let format = module(FORMAT,"TgFormat")?;
    let expected = build(&[format.clone(),module(ORIGIN,"TelegramBot")?])?;
    let mut python = build(&[format])?;
    let lowered=crate::python_frontend::lower_module_with_program(PYTHON,"TelegramBot",&python)
        .map_err(|_| "authored Python fixture lowering failed")?;
    for op in &lowered.operations {
        validator::apply_and_validate(&mut python,op).map_err(|_| "authored fixture validation failed")?;
    }
    same_program(&expected,&python)
}

fn roundtrip(expected: &ast::Program, bot: &parser::ModuleDecl, format: &parser::ModuleDecl)
    -> Checked<(BTreeMap<String,Vec<u8>>,usize)> {
    let mut rebuilt = ast::Program::default();
    let mut files = BTreeMap::new();
    let mut cases = 0;
    for (name,m,test_source) in [("TgFormat",format,FORMAT_TESTS),("TelegramBot",bot,TESTS)] {
        let py = crate::python_frontend::print_module(expected,name).map_err(|_| "Python print failed")?;
        let lowered = crate::python_frontend::lower_module_with_program(&py,name,&rebuilt)
            .map_err(|_| "production Python lowering failed")?;
        for op in &lowered.operations {
            validator::apply_and_validate(&mut rebuilt,op).map_err(|_| "lowered module validation failed")?;
        }
        crate::python_frontend::validate_runtime_arguments(&lowered.operations,&rebuilt)
            .map_err(|_| "runtime argument validation failed")?;
        let tests = tests(m,test_source)?;
        cases += validate_cases(&rebuilt,&tests)?;
        // Library restoration prefixes the owner itself; emitting qualified targets
        // would produce Module.Module.fn and silently lose persisted coverage (#110).
        let mut stored_tests=tests.clone();
        for op in &mut stored_tests {
            let parser::Operation::Test(t)=op else {return Err("non-test stored operation");};
            t.function_name=t.function_name.strip_prefix(&format!("{name}."))
                .ok_or("stored target outside module")?.to_owned();
            if t.function_name.contains('.') {return Err("nested stored test target unsupported");}
        }
        let mut canonical = lowered.canonical_adapsis;
        for op in &stored_tests {
            canonical.push('\n');
            canonical.push_str(&crate::python_frontend::emit::operations(std::slice::from_ref(op))
                .map_err(|_| "test emission failed")?);
        }
        let parsed = parser::parse(&canonical).map_err(|_| "canonical reparse failed")?;
        let parsed_tests: Vec<_> = parsed.iter().flat_map(|op| match op {
            parser::Operation::Module(m) => m.body.as_slice(),
            _ => std::slice::from_ref(op),
        }).filter(|op| matches!(op,parser::Operation::Test(_))).cloned().collect();
        if !exact(&stored_tests,&parsed_tests) { return Err("grouped stored tests changed on reparse"); }
        files.insert(format!("{name}.py"),py.into_bytes());
        files.insert(format!("{name}.ax"),canonical.into_bytes());
    }
    same_program(expected,&rebuilt)?;
    let reparsed = build(&[
        module(std::str::from_utf8(&files["TgFormat.ax"]).map_err(|_| "canonical encoding")?,"TgFormat")?,
        module(std::str::from_utf8(&files["TelegramBot.ax"]).map_err(|_| "canonical encoding")?,"TelegramBot")?,
    ])?;
    same_program(expected,&reparsed)?;
    for name in ["TgFormat","TelegramBot"] {
        let py = crate::python_frontend::print_module(&rebuilt,name).map_err(|_| "second Python print failed")?;
        if py.as_bytes() != files[&format!("{name}.py")] { return Err("Python print not idempotent"); }
    }
    Ok((files,cases))
}

#[test]
fn telegram_migration_rejects_unaccounted_function_and_surface_drift() {
    let frozen=module(FROZEN,"TelegramBot").unwrap();
    assert!(verify_live(&frozen,&frozen,42).is_ok());
    let mut changed=frozen.clone();
    let parser::Operation::Function(f)=changed.body.iter_mut().find(|op| matches!(op,parser::Operation::Function(f) if f.name=="extract_text")).unwrap() else {panic!()};
    f.return_type=parser::TypeExpr::Named("Int".into());
    assert!(verify_live(&changed,&frozen,42).is_err());
    let mut changed=frozen.clone();changed.doc=Some("unexpected drift".into());
    assert!(verify_live(&changed,&frozen,42).is_err());
    changed=frozen.clone();changed.body.retain(|op| !matches!(op,parser::Operation::Route{..}));
    assert!(verify_live(&changed,&frozen,42).is_err());
}

#[test]
fn telegram_migration_groups_every_case_and_refuses_unsafe_output() {
    let m=module(FORMAT,"TgFormat").unwrap();
    let grouped=tests(&m,FORMAT_TESTS).unwrap();
    let mut targets=std::collections::HashSet::new();
    for op in grouped { let parser::Operation::Test(t)=op else {panic!()};assert!(targets.insert(t.function_name));assert!(!t.cases.is_empty()); }
    assert!(export(Path::new("/does-not-exist"),Path::new("/home/forbidden")).is_err());
    assert!(export(Path::new("/does-not-exist"),Path::new("/tmp/opencode")).is_err());
}

#[test]
#[ignore = "operator-only: ADAPSIS_MODULES_DIR must name the private staged export"]
fn telegram_migration_private_library_roundtrip() {
    let result = library_roundtrip();
    assert!(result.is_ok(),"private library verification refused: {}",result.err().unwrap_or("unknown"));
}

fn library_roundtrip() -> Checked<()> {
    let dir=std::env::var("ADAPSIS_MODULES_DIR").map_err(|_| "explicit staged library required")?;
    let path=Path::new(&dir);
    if path.parent()!=Some(Path::new("/tmp/opencode")) ||
        fs::symlink_metadata(path).map_err(|_| "staging metadata")?.file_type().is_symlink() ||
        fs::metadata(path).map_err(|_| "staging metadata")?.permissions().mode() & 0o777 != 0o700 {
        return Err("not an owner-only staging directory");
    }
    let manifest:serde_json::Value=serde_json::from_slice(&fs::read(path.join("manifest.json"))
        .map_err(|_| "manifest absent")?).map_err(|_| "manifest invalid")?;
    if manifest["schema_version"]!=3 || manifest["superseded"]!=false {return Err("obsolete candidate manifest");}
    let mut modules=Vec::new();let mut count=0;
    for name in ["TelegramBot","TgFormat"] {
        let file=path.join(format!("{name}.ax"));
        let meta=fs::symlink_metadata(&file).map_err(|_| "artifact metadata")?;
        if !meta.is_file() || meta.permissions().mode() & 0o777 != 0o600 {return Err("artifact not private regular file");}
        let bytes=fs::read(&file).map_err(|_| "artifact read")?;
        if manifest["artifacts"][format!("{name}.ax")] != digest(&bytes) {return Err("artifact hash mismatch");}
        let m=module(std::str::from_utf8(&bytes).map_err(|_| "artifact encoding")?,name)?;
        for op in &m.body {if let parser::Operation::Test(t)=op {count+=t.cases.len();}}
        modules.push(m);
    }
    let mut expected=build(&modules)?;
    for m in &modules {
        for op in &m.body {
            if let parser::Operation::Test(t)=op {
                if t.function_name.contains('.') {return Err("library test target must be bare");}
                crate::session::store_test(&mut expected,&format!("{}.{}",m.name,t.function_name),&t.cases);
            }
        }
    }
    let mut loaded=ast::Program::default();
    let state=crate::library::load_module_library(&mut loaded);
    if state.loaded_modules.len()!=2 || !state.errors.lock().map_err(|_| "library state lock")?.is_empty() {
        return Err("private library load failed");
    }
    let restored=loaded.modules.iter().flat_map(|m|&m.functions).map(|f|f.tests.len()).sum::<usize>();
    if restored!=count || count!=manifest["pure_cases_passed"].as_u64().ok_or("case count absent")? as usize {
        return Err("stored case restoration lost cases");
    }
    let baseline=loaded.clone();
    // The real Os startup drains this registration queue before running lifecycle
    // blocks (main.rs). Model that consumption without registering any live route.
    // Otherwise load+reload artificially queues the same route twice. Assert the
    // reloaded queue equals the original exactly; never ignore route differences.
    loaded.pending_routes.clear();
    crate::library::reload_module(&mut loaded,"",Some(&state)).map_err(|_| "private library reload failed")?;
    if !crate::python_frontend::ast_equality::program(&baseline,&loaded) {return Err("library reload program AST drift");}
    if baseline.pending_routes!=loaded.pending_routes {return Err("library reload pending route drift");}
    if !crate::python_frontend::ast_equality::program(&expected,&loaded) {return Err("library restored AST differs from expected");}
    if expected.pending_routes!=loaded.pending_routes {return Err("library expected pending routes differ");}
    private_file(path,"library-verified.json",serde_json::to_string(&serde_json::json!({
        "schema_version":3,"load_reload_passed":true,"stored_cases":count,
        "manifest_sha256":digest(&fs::read(path.join("manifest.json")).map_err(|_| "manifest read")?),
        "artifacts":manifest["artifacts"],
    })).map_err(|_| "attestation encoding")?.as_bytes())?;
    println!("MIGRATION_LIBRARY_VERIFIED modules=2 cases={count}");
    Ok(())
}

#[test]
#[ignore = "operator-only: requires explicit private input/output paths; never activates"]
fn telegram_migration_private_export() {
    let input = std::env::var("ADAPSIS_MIGRATION_INPUT").expect("explicit migration input required");
    let output = std::env::var("ADAPSIS_MIGRATION_OUTPUT").expect("explicit migration output required");
    match export(Path::new(&input),Path::new(&output)) {
        Ok(manifest) => println!("MIGRATION_PREPARED {manifest}"),
        Err(category) => panic!("migration preparation refused: {category}"),
    }
}
