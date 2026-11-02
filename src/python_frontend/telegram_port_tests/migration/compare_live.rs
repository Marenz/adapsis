//! Read-only comparison after daemon persistence has rewritten canonical formatting.
//! No library loading, persistence, eval, networking or shared-state access.
use super::*;

fn grouped(m:&parser::ModuleDecl)->Checked<BTreeMap<String,Vec<parser::TestCase>>> {
    let mut out=BTreeMap::new();
    for op in &m.body {
        if let parser::Operation::Test(t)=op {
            if out.insert(t.function_name.clone(),t.cases.clone()).is_some() {
                return Err("duplicate stored target in comparison");
            }
        }
    }
    Ok(out)
}

fn compare(live:&Path,artifact:&Path)->Checked<()> {
    let mut current=Vec::new();let mut expected=Vec::new();let mut tests_match=true;
    for name in ["TelegramBot","TgFormat"] {
        let a=fs::read(live.join(format!("{name}.ax"))).map_err(|_| "live read failed")?;
        let b=fs::read(artifact.join(format!("{name}.ax"))).map_err(|_| "artifact read failed")?;
        let am=module(std::str::from_utf8(&a).map_err(|_| "live encoding")?,name)?;
        let bm=module(std::str::from_utf8(&b).map_err(|_| "artifact encoding")?,name)?;
        let at=grouped(&am)?;let bt=grouped(&bm)?;
        let equal=exact(&at,&bt);tests_match &= equal;
        println!("MIGRATION_READONLY module={name} live_sha256={} artifact_sha256={} function_count={} stored_cases={} grouped_case_ast_equal={equal} migration_helpers_absent={}",
            digest(&a),digest(&b),functions(&am).len(),at.values().map(Vec::len).sum::<usize>(),
            !functions(&am).keys().any(|n|n.starts_with("migration_")));
        current.push(am);expected.push(bm);
    }
    let current=build(&current)?;let expected=build(&expected)?;
    let mut modules_match=true;
    for (a,b) in current.modules.iter().zip(&expected.modules) {
        let equal=crate::python_frontend::ast_equality::module(a,b);
        modules_match &= equal;
        println!("MIGRATION_READONLY module={} full_module_ast_equal={equal}",a.name);
        if !equal {
            println!("MIGRATION_READONLY difference_path={}",crate::python_frontend::ast_equality::module_difference(a,b));
        }
    }
    let routes=current.pending_routes==expected.pending_routes;
    println!("MIGRATION_READONLY registration_batch_equal={routes}");
    if !tests_match || !modules_match || !routes {return Err("deployed/artifact AST comparison differs");}
    same_program(&current,&expected)
}

#[test]
#[ignore="operator read-only: explicit deployed library and private authored artifact paths"]
fn telegram_migration_compare_deployed_artifact() {
    let live=std::env::var("ADAPSIS_MIGRATION_COMPARE_LIVE").expect("explicit live comparison path");
    let artifact=std::env::var("ADAPSIS_MIGRATION_COMPARE_ARTIFACT").expect("explicit artifact comparison path");
    if let Err(category)=compare(Path::new(&live),Path::new(&artifact)) {
        panic!("read-only migration comparison: {category}");
    }
}
