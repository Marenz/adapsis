use super::{Diagnostic, LoweredSource, tests_print};
use crate::{ast as a, parser as p, validator};
use rustpython_parser::{ast as py, ast::Ranged};
use std::collections::{BTreeMap, HashSet};

#[derive(Debug)]
pub struct LoweredPythonModule {
    pub declarations: LoweredSource,
    pub tests: Vec<p::TestMutation>,
}

struct StoredGroup {
    function: String,
    target: String,
    cases: Vec<p::TestCase>,
}

struct TestDefinition {
    name: String,
    target: String,
    source: String,
    start: usize,
    end: usize,
}

#[derive(Default)]
struct ModuleRefs {
    types: Vec<String>,
    callees: Vec<String>,
}

/// Checked save printing plus its proven-qualification ledger: every bare local
/// call that printed qualified, with the number of rewritten call sites.
#[derive(Debug, Clone, Default)]
pub struct QualificationLedger {
    pub names: std::collections::BTreeSet<String>,
    pub call_sites: usize,
    /// CATALOG-LOCAL classification: the printed catalog may omit modules
    /// that also declare a name, so these undercount order-dependence. The live
    /// save (`library::source_io::persist_live`) recomputes against the whole
    /// Program; its receipt is authoritative.
    /// Proven names declared exactly once in the printed catalog.
    pub catalog_unique: std::collections::BTreeSet<String>,
    /// Proven names declared by several modules of the printed catalog.
    pub catalog_order_dependent: std::collections::BTreeSet<String>,
    /// (member, callee) sites whose callee is catalog-order-dependent.
    pub catalog_order_dependent_sites: Vec<(String, String)>,
}

pub fn print_python_module(program: &a::Program, module_name: &str) -> Result<String, Diagnostic> {
    print_python_module_with_ledger(program, module_name).map(|(text, _)| text)
}

pub fn print_python_module_with_ledger(
    program: &a::Program,
    module_name: &str,
) -> Result<(String, QualificationLedger), Diagnostic> {
    if !super::identifier(module_name) {
        return Err(diag(module_name, "module name must be a simple identifier"));
    }
    validate_save_catalog(program, module_name)?;
    let module = find_module(program, module_name)?;
    preflight_module_members(module)?;

    let refs = collect_module_refs(module);
    validate_callees(program, &refs)?;
    let owners_before = type_owner_map(program, module_name, &refs.types)?;

    let (text, proven) = render_python_module(program, module)?;
    // The only permitted AST difference: ledgered bare→qualified call sites.
    let (normalized, sites) = super::qualification::qualify_module(module, &proven);

    let without_current = without_module(program, module_name);
    let lowered = lower_python_module(&text, module_name, &without_current)?;
    let mut candidate_program = without_current;
    for operation in &lowered.declarations.operations {
        validator::apply_and_validate(&mut candidate_program, operation).map_err(|e| {
            diag(
                module_name,
                format!("candidate module validation failed: {e}"),
            )
        })?;
    }
    candidate_program.rebuild_function_index();
    let candidate = candidate_program
        .modules
        .iter()
        .find(|m| m.name == module_name)
        .ok_or_else(|| {
            diag(
                module_name,
                "candidate module missing after declaration roundtrip",
            )
        })?;
    preflight_module_members(candidate)?;
    let candidate_refs = collect_module_refs(candidate);
    validate_callees(&candidate_program, &candidate_refs)?;
    let owners_after = type_owner_map(&candidate_program, module_name, &candidate_refs.types)?;
    if owners_before != owners_after {
        return Err(diag(
            module_name,
            "type owner resolution changed through the Python roundtrip; refusing unsafe save",
        ));
    }

    let expected = stored_groups(module)?;
    compare_test_groups(&expected, &lowered.tests)?;

    for function in &module.functions {
        if function.tests.is_empty() {
            continue;
        }
        let qualified = format!("{}.{}", module.name, function.name);
        if let Some(target) = candidate_program.get_function_mut(&qualified) {
            target.tests = function.tests.clone();
        }
    }
    candidate_program.rebuild_function_index();
    let rebuilt = candidate_program
        .modules
        .iter()
        .find(|m| m.name == module_name)
        .ok_or_else(|| {
            diag(
                module_name,
                "candidate module missing after test attachment",
            )
        })?;
    if !super::ast_equality::module(&normalized, rebuilt) {
        return Err(diag(
            module_name,
            "Python roundtrip changed module structure or stored tests beyond proven call qualification; refusing save",
        ));
    }
    let (again, again_proven) = render_python_module(&candidate_program, rebuilt)?;
    if again != text || !again_proven.is_empty() {
        return Err(diag(
            module_name,
            "second Python print differs byte-for-byte; refusing save",
        ));
    }
    let (catalog_order_dependent, unique): (std::collections::BTreeSet<_>, _) = proven
        .iter()
        .cloned()
        .partition(|name| super::qualification::declaration_count(program, name) > 1);
    let catalog_order_dependent_sites = sites
        .iter()
        .filter(|(_, callee)| catalog_order_dependent.contains(callee))
        .cloned()
        .collect();
    Ok((
        text,
        QualificationLedger {
            names: proven,
            call_sites: sites.len(),
            catalog_unique: unique,
            catalog_order_dependent,
            catalog_order_dependent_sites,
        },
    ))
}

pub fn lower_python_module(
    source: &str,
    module_name: &str,
    catalog: &a::Program,
) -> Result<LoweredPythonModule, Diagnostic> {
    if !super::identifier(module_name) {
        return Err(diag(module_name, "module name must be a simple identifier"));
    }
    validate_load_catalog(catalog, module_name)?;
    let cx = super::Context {
        source,
        module: module_name,
    };
    let suite = cx.parse()?;
    let (module_source, definitions) = split_combined(&cx, &suite, source)?;

    let declarations = super::lower_module_with_program(&module_source, module_name, catalog)?;
    prevalidate_raw_declarations(&declarations.operations, module_name)?;
    let local = local_functions(&declarations.operations)?;
    let mut grouped = Vec::<(String, Vec<p::TestCase>)>::new();
    for definition in &definitions {
        lower_test_definition(definition, module_name, &local, &mut grouped)?;
    }

    let mut candidate_program = catalog.clone();
    for operation in &declarations.operations {
        validator::apply_and_validate(&mut candidate_program, operation).map_err(|e| {
            diag(
                module_name,
                format!("candidate module validation failed: {e}"),
            )
        })?;
    }
    candidate_program.rebuild_function_index();
    let candidate = candidate_program
        .modules
        .iter()
        .find(|m| m.name == module_name)
        .ok_or_else(|| diag(module_name, "candidate module missing after lowering"))?;
    check_raw_declaration_coverage(&declarations.operations, candidate, module_name)?;
    preflight_module_members(candidate)?;
    let refs = collect_module_refs(candidate);
    validate_callees(&candidate_program, &refs)?;
    type_owner_map(&candidate_program, module_name, &refs.types)?;

    let tests = grouped
        .into_iter()
        .map(|(target, cases)| p::TestMutation {
            function_name: target,
            cases,
        })
        .collect();
    Ok(LoweredPythonModule {
        declarations,
        tests,
    })
}

fn render_python_module(
    program: &a::Program,
    module: &a::Module,
) -> Result<(String, std::collections::BTreeSet<String>), Diagnostic> {
    let (mut text, proven) = super::printer::module_body_qualifying(program, module)
        .map_err(|message| diag(&module.name, message))?;
    let groups = stored_groups(module)?;
    if groups.is_empty() {
        return Ok((text, proven));
    }
    let blocks = render_test_groups(&module.name, module, &groups)?;
    if !text.ends_with('\n') {
        text.push('\n');
    }
    text.push('\n');
    text.push_str(&blocks.join("\n"));
    Ok((text, proven))
}

fn render_test_groups(
    module_name: &str,
    module: &a::Module,
    groups: &[StoredGroup],
) -> Result<Vec<String>, Diagnostic> {
    let mut used: HashSet<String> = module
        .functions
        .iter()
        .map(|function| function.name.clone())
        .collect();
    let mut counter = 0usize;
    let mut blocks = Vec::new();
    for group in groups {
        for (index, case) in group.cases.iter().enumerate() {
            loop {
                counter += 1;
                let candidate = format!("test_case_{counter}");
                if used.insert(candidate.clone()) {
                    break;
                }
            }
            let name = format!("test_case_{counter}");
            let body = tests_print::print_case(&group.function, case).map_err(|message| {
                diag(
                    module_name,
                    format!("stored test {} case {}: {message}", group.target, index + 1),
                )
            })?;
            let decorator =
                super::emit::quote(&group.target).map_err(|message| diag(module_name, message))?;
            blocks.push(format!(
                "@stored_test({decorator})\ndef {name}() -> None:\n{body}"
            ));
        }
    }
    Ok(blocks)
}

fn stored_groups(module: &a::Module) -> Result<Vec<StoredGroup>, Diagnostic> {
    let mut groups = Vec::new();
    for function in &module.functions {
        if function.tests.is_empty() {
            continue;
        }
        let target = format!("{}.{}", module.name, function.name);
        let mut cases = Vec::new();
        for (index, case) in function.tests.iter().enumerate() {
            cases.push(stored_case_to_parser(&module.name, &target, index, case)?);
        }
        groups.push(StoredGroup {
            function: function.name.clone(),
            target,
            cases,
        });
    }
    Ok(groups)
}

fn stored_case_to_parser(
    module_name: &str,
    target: &str,
    index: usize,
    case: &a::TestCase,
) -> Result<p::TestCase, Diagnostic> {
    // A present typed payload is authoritative: invalid typed data refuses the
    // save instead of falling back to the display strings. Absent payloads are
    // legacy records and keep the string importer.
    if let Some(typed) = &case.typed {
        return typed.case().map_err(|_| {
            diag(
                module_name,
                format!("stored test {target} case {} has an invalid typed payload", index + 1),
            )
        });
    }
    let input = p::parse_test_input(0, &case.input).map_err(|e| {
        diag(
            module_name,
            format!("stored test {target} case {} input: {e}", index + 1),
        )
    })?;
    let expected = p::parse_test_input(0, &case.expected).map_err(|e| {
        diag(
            module_name,
            format!("stored test {target} case {} expectation: {e}", index + 1),
        )
    })?;
    let matcher = match case.matcher.as_deref() {
        None => None,
        Some("AnyOk") => Some(p::TestMatcher::AnyOk),
        Some("AnyErr") => Some(p::TestMatcher::AnyErr),
        Some(value) => Some(parse_matcher(value).ok_or_else(|| {
            diag(
                module_name,
                format!(
                    "stored test {target} case {} has unsupported matcher {value:?}",
                    index + 1
                ),
            )
        })?),
    };
    let after_checks = case
        .after_checks
        .iter()
        .map(|check| p::AfterCheck {
            target: check.target.clone(),
            matcher: check.matcher.clone(),
            value: check.value.clone(),
        })
        .collect();
    Ok(p::TestCase {
        input,
        expected,
        matcher,
        after_checks,
    })
}

fn parse_matcher(value: &str) -> Option<p::TestMatcher> {
    if let Some(text) = value.strip_prefix("contains:") {
        return Some(p::TestMatcher::Contains(text.to_string()));
    }
    if let Some(text) = value.strip_prefix("starts_with:") {
        return Some(p::TestMatcher::StartsWith(text.to_string()));
    }
    if let Some(text) = value.strip_prefix("ErrContaining:") {
        return Some(p::TestMatcher::ErrContaining(text.to_string()));
    }
    None
}

fn compare_test_groups(
    expected: &[StoredGroup],
    actual: &[p::TestMutation],
) -> Result<(), Diagnostic> {
    if expected.len() != actual.len() {
        return Err(diag(
            "",
            format!(
                "stored test target count changed: {} before, {} after",
                expected.len(),
                actual.len()
            ),
        ));
    }
    for (group, mutation) in expected.iter().zip(actual) {
        if group.target != mutation.function_name {
            return Err(diag(
                "",
                format!(
                    "stored test target changed: {} became {}",
                    group.target, mutation.function_name
                ),
            ));
        }
        if format!("{:?}", group.cases) != format!("{:?}", mutation.cases) {
            return Err(diag(
                "",
                format!("stored test cases changed for {}", group.target),
            ));
        }
    }
    Ok(())
}

fn split_combined(
    cx: &super::Context<'_>,
    suite: &[py::Stmt],
    source: &str,
) -> Result<(String, Vec<TestDefinition>), Diagnostic> {
    let mut definitions = Vec::new();
    let mut ranges = Vec::new();
    let mut names = HashSet::new();
    for statement in suite {
        let Some(definition) = test_definition(cx, statement, source)? else {
            continue;
        };
        if !names.insert(definition.name.clone()) {
            return Err(cx.at(
                statement,
                format!("duplicate stored test definition {}", definition.name),
            ));
        }
        ranges.push((definition.start, definition.end));
        definitions.push(definition);
    }
    Ok((blank_ranges(source, &ranges), definitions))
}

fn test_definition(
    cx: &super::Context<'_>,
    statement: &py::Stmt,
    source: &str,
) -> Result<Option<TestDefinition>, Diagnostic> {
    let (name, args, returns, body, decorators, asynchronous, type_params, type_comment) =
        match statement {
            py::Stmt::FunctionDef(f) => (
                f.name.as_str(),
                &f.args,
                f.returns.as_deref(),
                &f.body,
                &f.decorator_list,
                false,
                &f.type_params,
                f.type_comment.as_ref(),
            ),
            py::Stmt::AsyncFunctionDef(f) => (
                f.name.as_str(),
                &f.args,
                f.returns.as_deref(),
                &f.body,
                &f.decorator_list,
                true,
                &f.type_params,
                f.type_comment.as_ref(),
            ),
            _ => return Ok(None),
        };

    let mut target = None;
    for decorator in decorators {
        if let Some(value) = stored_test_target(cx, decorator)? {
            if target.replace(value).is_some() {
                return Err(cx.at(decorator, "duplicate @stored_test decorator"));
            }
        }
    }
    let Some(target) = target else {
        return Ok(None);
    };
    if asynchronous {
        return Err(cx.at(statement, "stored test definitions must be synchronous"));
    }
    if decorators.len() != 1 {
        return Err(cx.at(
            statement,
            "stored test definitions cannot carry other decorators",
        ));
    }
    if !type_params.is_empty() || type_comment.is_some() {
        return Err(cx.at(
            statement,
            "stored test definitions cannot have generics or type comments",
        ));
    }
    if !args.posonlyargs.is_empty()
        || !args.args.is_empty()
        || args.vararg.is_some()
        || !args.kwonlyargs.is_empty()
        || args.kwarg.is_some()
    {
        return Err(cx.at(statement, "stored test definitions require zero parameters"));
    }
    if let Some(annotation) = returns {
        if !matches!(annotation, py::Expr::Constant(c) if c.value == py::Constant::None) {
            return Err(cx.at(
                annotation,
                "stored test return annotation must be None or omitted",
            ));
        }
    }
    if body.is_empty() {
        return Err(cx.at(statement, "stored test definition has an empty body"));
    }
    let body_start = line_start(source, usize::from(body[0].start()));
    let body_end = usize::from(statement.end());
    let mut body_text = source[body_start..body_end].to_string();
    if !body_text.ends_with('\n') {
        body_text.push('\n');
    }
    let source_text = format!("def {name}() -> None:\n{body_text}");
    let start = decorators
        .iter()
        .map(|d| usize::from(d.start()))
        .fold(usize::from(statement.start()), usize::min);
    let start = line_start(source, start);
    let end = line_end(source, usize::from(statement.end()));
    Ok(Some(TestDefinition {
        name: name.to_string(),
        target,
        source: source_text,
        start,
        end,
    }))
}

fn stored_test_target(
    cx: &super::Context<'_>,
    decorator: &py::Expr,
) -> Result<Option<String>, Diagnostic> {
    match decorator {
        py::Expr::Name(n) if n.id.as_str() == "stored_test" => Err(cx.at(
            decorator,
            "stored test decorator requires exactly one string target: @stored_test(\"Module.fn\")",
        )),
        py::Expr::Call(call) if matches!(call.func.as_ref(), py::Expr::Name(n) if n.id.as_str() == "stored_test") =>
        {
            if !call.keywords.is_empty() || call.args.len() != 1 {
                return Err(cx.at(
                    decorator,
                    "stored test decorator requires exactly one string target: @stored_test(\"Module.fn\")",
                ));
            }
            match &call.args[0] {
                py::Expr::Constant(constant) => match &constant.value {
                    py::Constant::Str(value) => Ok(Some(value.clone())),
                    _ => Err(cx.at(
                        &call.args[0],
                        "stored test decorator target must be a string literal",
                    )),
                },
                _ => Err(cx.at(
                    &call.args[0],
                    "stored test decorator target must be a string literal",
                )),
            }
        }
        _ => Ok(None),
    }
}

fn lower_test_definition(
    definition: &TestDefinition,
    module_name: &str,
    local: &HashSet<String>,
    grouped: &mut Vec<(String, Vec<p::TestCase>)>,
) -> Result<(), Diagnostic> {
    let lowered = super::lower_tests(&definition.source, module_name).map_err(|error| {
        diag(
            module_name,
            format!(
                "stored test {} ({}): {}",
                definition.name, definition.target, error.message
            ),
        )
    })?;
    let mut target = None;
    let mut cases = Vec::new();
    for operation in lowered.operations {
        match operation {
            p::Operation::Test(test) => {
                if target.replace(test.function_name).is_some() {
                    return Err(diag(
                        module_name,
                        format!(
                            "stored test {} lowered to multiple targets",
                            definition.name
                        ),
                    ));
                }
                cases.extend(test.cases);
            }
            _ => {
                return Err(diag(
                    module_name,
                    format!(
                        "stored test {} lowered to a non-test operation",
                        definition.name
                    ),
                ));
            }
        }
    }
    let actual = target.ok_or_else(|| {
        diag(
            module_name,
            format!("stored test {} lowered to no test", definition.name),
        )
    })?;
    if actual != definition.target {
        return Err(diag(
            module_name,
            format!(
                "stored test declared target {} did not match lowered target {actual}",
                definition.target
            ),
        ));
    }
    let prefix = format!("{module_name}.");
    let bare = definition.target.strip_prefix(&prefix).ok_or_else(|| {
        diag(
            module_name,
            format!(
                "stored test target {} does not belong to this module",
                definition.target
            ),
        )
    })?;
    if bare.contains('.') || !local.contains(bare) {
        return Err(diag(
            module_name,
            format!(
                "stored test target {} is not a function declared by this module",
                definition.target
            ),
        ));
    }
    if let Some((_, existing)) = grouped.iter_mut().find(|(name, _)| name == &actual) {
        existing.extend(cases);
    } else {
        grouped.push((actual, cases));
    }
    Ok(())
}

fn local_functions(operations: &[p::Operation]) -> Result<HashSet<String>, Diagnostic> {
    let [p::Operation::Module(module)] = operations else {
        return Err(diag(
            "",
            "module lowering did not produce exactly one module declaration",
        ));
    };
    let mut names = HashSet::new();
    for operation in &module.body {
        if let p::Operation::Function(function) = operation {
            names.insert(function.name.clone());
        }
    }
    Ok(names)
}

fn prevalidate_raw_declarations(
    operations: &[p::Operation],
    module_name: &str,
) -> Result<(), Diagnostic> {
    let [p::Operation::Module(module)] = operations else {
        return Err(diag(
            module_name,
            "module lowering did not produce exactly one module declaration",
        ));
    };
    if module.name != module_name {
        return Err(diag(
            module_name,
            format!("module lowering produced module {}", module.name),
        ));
    }
    let mut routes = HashSet::<(String, String)>::new();
    let mut sources = HashSet::<String>::new();
    for operation in &module.body {
        match operation {
            p::Operation::Route { method, path, .. } => {
                if !routes.insert((method.clone(), path.clone())) {
                    return Err(diag(
                        module_name,
                        format!(
                            "duplicate route registration {method} {path}; refusing before validator merging"
                        ),
                    ));
                }
            }
            p::Operation::ModuleSource(source) => {
                if !sources.insert(source.name.clone()) {
                    return Err(diag(
                        module_name,
                        format!("duplicate source alias {}; refusing", source.name),
                    ));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn check_raw_declaration_coverage(
    operations: &[p::Operation],
    candidate: &a::Module,
    module_name: &str,
) -> Result<(), Diagnostic> {
    let [p::Operation::Module(module)] = operations else {
        return Err(diag(
            module_name,
            "module lowering did not produce exactly one module declaration",
        ));
    };
    if module.name != module_name || module.doc != candidate.doc {
        return Err(diag(
            module_name,
            "module identity or documentation changed during validation",
        ));
    }
    let mut functions = Vec::new();
    let mut types = Vec::new();
    let mut shared = Vec::new();
    let mut routes = Vec::new();
    let mut sources = Vec::new();
    let mut startup = 0usize;
    let mut shutdown = 0usize;
    for operation in &module.body {
        match operation {
            p::Operation::Function(function) => functions.push(function.name.clone()),
            p::Operation::Type(declaration) => types.push(declaration.name.clone()),
            p::Operation::SharedVar(declaration) => shared.push(declaration.name.clone()),
            p::Operation::Route {
                method,
                path,
                handler_fn,
            } => {
                let handler = if handler_fn.contains('.') {
                    handler_fn.clone()
                } else {
                    format!("{module_name}.{handler_fn}")
                };
                routes.push((method.clone(), path.clone(), handler));
            }
            p::Operation::ModuleSource(source) => sources.push(source),
            p::Operation::Startup(_) => startup += 1,
            p::Operation::Shutdown(_) => shutdown += 1,
            _ => {
                return Err(diag(
                    module_name,
                    "unexpected raw declaration operation; refusing incomplete save",
                ));
            }
        }
    }
    if startup > 1 || shutdown > 1 {
        return Err(diag(
            module_name,
            "duplicate startup/shutdown declarations have no exact member mapping",
        ));
    }
    let candidate_functions: Vec<_> = candidate
        .functions
        .iter()
        .map(|function| function.name.clone())
        .collect();
    if functions != candidate_functions {
        return Err(diag(
            module_name,
            "raw function declarations were merged or dropped during validation",
        ));
    }
    let candidate_types: Vec<_> = candidate
        .types
        .iter()
        .map(|declaration| declaration.name().to_string())
        .collect();
    if types != candidate_types {
        return Err(diag(
            module_name,
            "raw type declarations were merged or dropped during validation",
        ));
    }
    let candidate_shared: Vec<_> = candidate
        .shared_vars
        .iter()
        .map(|declaration| declaration.name.clone())
        .collect();
    if shared != candidate_shared {
        return Err(diag(
            module_name,
            "raw shared declarations were merged or dropped during validation",
        ));
    }
    if routes.len() != candidate.routes.len() {
        return Err(diag(
            module_name,
            "raw route declarations were merged or dropped during validation",
        ));
    }
    for (raw, actual) in routes.iter().zip(&candidate.routes) {
        if raw.0 != actual.method || raw.1 != actual.path || raw.2 != actual.handler_fn {
            return Err(diag(
                module_name,
                "raw route declaration changed during validation",
            ));
        }
    }
    if sources.len() != candidate.sources.len() {
        return Err(diag(
            module_name,
            "raw source declarations were merged or dropped during validation",
        ));
    }
    for (raw, actual) in sources.iter().zip(&candidate.sources) {
        if raw.name != actual.name
            || raw.source_type != actual.source_type
            || raw.config != actual.config
            || raw.handler != actual.handler
        {
            return Err(diag(
                module_name,
                "raw source declaration changed during validation",
            ));
        }
    }
    if (startup == 1) != candidate.startup.is_some() {
        return Err(diag(
            module_name,
            "startup declaration changed during validation",
        ));
    }
    if (shutdown == 1) != candidate.shutdown.is_some() {
        return Err(diag(
            module_name,
            "shutdown declaration changed during validation",
        ));
    }
    Ok(())
}

fn preflight_module_members(module: &a::Module) -> Result<(), Diagnostic> {
    if !module.modules.is_empty() {
        return Err(diag(
            &module.name,
            "nested submodules have no Python module-file surface; refusing save",
        ));
    }
    if !module.event_decls.is_empty() {
        return Err(diag(
            &module.name,
            "module-level event declarations have no Python module-file surface; refusing save",
        ));
    }
    let functions: HashSet<&str> = module
        .functions
        .iter()
        .map(|function| function.name.as_str())
        .collect();
    let prefix = format!("{}.", module.name);
    let mut routed = HashSet::new();
    for route in &module.routes {
        let handler = route.handler_fn.strip_prefix(&prefix).ok_or_else(|| {
            diag(
                &module.name,
                format!(
                    "route handler {} does not belong to module {}",
                    route.handler_fn, module.name
                ),
            )
        })?;
        if handler.contains('.') || !functions.contains(handler) {
            return Err(diag(
                &module.name,
                format!("route handler {} has no local function", route.handler_fn),
            ));
        }
        if !routed.insert(handler.to_string()) {
            return Err(diag(
                &module.name,
                format!("multiple @route declarations target {handler}; refusing save"),
            ));
        }
    }
    let mut sourced = HashSet::new();
    for source in &module.sources {
        if !functions.contains(source.handler.as_str()) {
            return Err(diag(
                &module.name,
                format!(
                    "source handler {} has no local function (expected a bare name)",
                    source.handler
                ),
            ));
        }
        if !sourced.insert(source.handler.clone()) {
            return Err(diag(
                &module.name,
                format!(
                    "multiple @source declarations target {}; refusing save",
                    source.handler
                ),
            ));
        }
    }
    Ok(())
}

fn validate_save_catalog(program: &a::Program, module_name: &str) -> Result<(), Diagnostic> {
    if !program.functions.is_empty() || !program.types.is_empty() {
        return Err(diag(
            module_name,
            "save catalog must not contain top-level functions or types",
        ));
    }
    if !program.pending_routes.is_empty() {
        return Err(diag(
            module_name,
            "save catalog contains pending routes; supply a closed catalog",
        ));
    }
    let mut found = false;
    for module in &program.modules {
        if module.name != module_name {
            continue;
        }
        if found {
            return Err(diag(module_name, "save catalog contains duplicate modules"));
        }
        found = true;
    }
    if !found {
        return Err(diag(module_name, "module not found in save catalog"));
    }
    Ok(())
}

fn validate_load_catalog(catalog: &a::Program, module_name: &str) -> Result<(), Diagnostic> {
    if !catalog.functions.is_empty() || !catalog.types.is_empty() {
        return Err(diag(
            module_name,
            "load catalog must not contain top-level functions or types",
        ));
    }
    if !catalog.pending_routes.is_empty() {
        return Err(diag(
            module_name,
            "load catalog contains pending routes; supply a fresh dependency closure",
        ));
    }
    if catalog.modules.iter().any(|m| m.name == module_name) {
        return Err(diag(
            module_name,
            "load catalog contains a stale copy of the module being lowered",
        ));
    }
    Ok(())
}

fn find_module<'a>(program: &'a a::Program, name: &str) -> Result<&'a a::Module, Diagnostic> {
    let mut matches = program.modules.iter().filter(|module| module.name == name);
    let module = matches
        .next()
        .ok_or_else(|| diag(name, "module not found in save catalog"))?;
    if matches.next().is_some() {
        return Err(diag(name, "save catalog contains duplicate modules"));
    }
    Ok(module)
}

fn without_module(program: &a::Program, module_name: &str) -> a::Program {
    let mut without = program.clone();
    without.modules.retain(|module| module.name != module_name);
    without.rebuild_function_index();
    without
}

fn type_owner_map(
    program: &a::Program,
    module_name: &str,
    names: &[String],
) -> Result<BTreeMap<String, String>, Diagnostic> {
    let types = super::types::catalog_from_program(program);
    let mut owners = BTreeMap::new();
    for name in names {
        let owner = match super::types::resolve_name(&types, module_name, name)
            .map_err(|message| diag(module_name, message))?
        {
            Some(owner) => owner,
            // Runtime-defined types (builtins::RUNTIME_TYPES) have no declaring
            // module; a user declaration of the same name still wins above.
            None if crate::builtins::is_runtime_type(name) => format!("<runtime>.{name}"),
            None => {
                return Err(diag(
                    module_name,
                    format!("type {name} is not declared in the supplied catalog"),
                ))
            }
        };
        owners.insert(name.clone(), owner);
    }
    Ok(owners)
}

fn validate_callees(program: &a::Program, refs: &ModuleRefs) -> Result<(), Diagnostic> {
    for callee in &refs.callees {
        if program.get_function(callee).is_none() {
            return Err(diag(
                "",
                format!("qualified callable {callee} is absent from the supplied catalog"),
            ));
        }
    }
    Ok(())
}

fn collect_module_refs(module: &a::Module) -> ModuleRefs {
    let mut refs = ModuleRefs::default();
    for ty in &module.types {
        match ty {
            a::TypeDecl::Struct(decl) => {
                for field in &decl.fields {
                    collect_type(&field.ty, &mut refs.types);
                }
            }
            a::TypeDecl::TaggedUnion(decl) => {
                for variant in &decl.variants {
                    for ty in &variant.payload {
                        collect_type(ty, &mut refs.types);
                    }
                }
            }
        }
    }
    for shared in &module.shared_vars {
        collect_type(&shared.ty, &mut refs.types);
        collect_expr(&shared.default, &mut refs);
    }
    for lifecycle in [&module.startup, &module.shutdown].into_iter().flatten() {
        for statement in &lifecycle.body {
            collect_statement(statement, &mut refs);
        }
    }
    for function in &module.functions {
        for param in &function.params {
            collect_type(&param.ty, &mut refs.types);
        }
        collect_type(&function.return_type, &mut refs.types);
        for statement in &function.body {
            collect_statement(statement, &mut refs);
        }
    }
    refs.types.sort();
    refs.types.dedup();
    refs.callees.sort();
    refs.callees.dedup();
    refs
}

fn collect_type(ty: &a::Type, out: &mut Vec<String>) {
    match ty {
        a::Type::Int | a::Type::Float | a::Type::Bool | a::Type::String | a::Type::Byte => {}
        a::Type::List(inner)
        | a::Type::Set(inner)
        | a::Type::Option(inner)
        | a::Type::Result(inner) => collect_type(inner, out),
        a::Type::Map(key, value) => {
            collect_type(key, out);
            collect_type(value, out);
        }
        a::Type::Struct(name) | a::Type::TaggedUnion(name) => out.push(name.clone()),
    }
}

fn collect_expr(expr: &a::Expr, refs: &mut ModuleRefs) {
    match expr {
        a::Expr::Literal(_) | a::Expr::Identifier(_) => {}
        a::Expr::FieldAccess { base, .. } => collect_expr(base, refs),
        a::Expr::Call(call) => collect_call(call, refs),
        a::Expr::Binary { left, right, .. } => {
            collect_expr(left, refs);
            collect_expr(right, refs);
        }
        a::Expr::Unary { expr, .. } => collect_expr(expr, refs),
        a::Expr::StructInit { ty, fields } => {
            if !ty.is_empty() {
                refs.types.push(ty.clone());
            }
            for field in fields {
                collect_expr(&field.value, refs);
            }
        }
    }
}

fn collect_call(call: &a::CallExpr, refs: &mut ModuleRefs) {
    if call.callee.contains('.') {
        refs.callees.push(call.callee.clone());
    }
    for arg in &call.args {
        collect_expr(arg, refs);
    }
}

fn collect_statement(statement: &a::Statement, refs: &mut ModuleRefs) {
    use a::StatementKind::*;
    match &statement.kind {
        Let { ty, value, .. } => {
            collect_type(ty, &mut refs.types);
            collect_expr(value, refs);
        }
        Call { binding, call } => {
            if let Some(binding) = binding {
                collect_type(&binding.ty, &mut refs.types);
            }
            collect_call(call, refs);
        }
        Check { condition, .. } => collect_expr(condition, refs),
        Branch {
            condition,
            then_body,
            else_body,
        } => {
            collect_expr(condition, refs);
            for statement in then_body.iter().chain(else_body) {
                collect_statement(statement, refs);
            }
        }
        Return { value } => collect_expr(value, refs),
        Each {
            iterator,
            binding,
            body,
        } => {
            collect_expr(iterator, refs);
            collect_type(&binding.ty, &mut refs.types);
            for statement in body {
                collect_statement(statement, refs);
            }
        }
        Set { value, .. } => collect_expr(value, refs),
        Match { expr, arms } => {
            collect_expr(expr, refs);
            for arm in arms {
                for statement in &arm.body {
                    collect_statement(statement, refs);
                }
            }
        }
        Await { ty, call, .. } => {
            collect_type(ty, &mut refs.types);
            collect_call(call, refs);
        }
        Spawn { call, binding } => {
            if let Some(binding) = binding {
                collect_type(&binding.ty, &mut refs.types);
            }
            collect_call(call, refs);
        }
        While { condition, body } => {
            collect_expr(condition, refs);
            for statement in body {
                collect_statement(statement, refs);
            }
        }
        Yield { value } => collect_expr(value, refs),
        Source(operation) => match operation {
            a::SourceOp::Add { source_type, .. } | a::SourceOp::Replace { source_type, .. } => {
                collect_source_type(source_type, refs)
            }
            a::SourceOp::Remove { .. } | a::SourceOp::List => {}
        },
        Event(operation) => {
            if let a::EventOp::Emit { value, .. } = operation {
                collect_expr(value, refs);
            }
        }
    }
}

fn collect_source_type(source_type: &a::SourceType, refs: &mut ModuleRefs) {
    if let a::SourceType::Timer(expr) = source_type {
        collect_expr(expr, refs);
    }
}

fn blank_ranges(source: &str, ranges: &[(usize, usize)]) -> String {
    let mut out = String::with_capacity(source.len());
    for (index, character) in source.char_indices() {
        let blanked = ranges
            .iter()
            .any(|(start, end)| index >= *start && index < *end);
        if blanked && character != '\n' && character != '\r' {
            for _ in 0..character.len_utf8() {
                out.push(' ');
            }
        } else {
            out.push(character);
        }
    }
    out
}

fn line_start(source: &str, offset: usize) -> usize {
    let offset = offset.min(source.len());
    source[..offset]
        .rfind('\n')
        .map(|index| index + 1)
        .unwrap_or(0)
}

fn line_end(source: &str, offset: usize) -> usize {
    let offset = offset.min(source.len());
    source[offset..]
        .find('\n')
        .map(|index| offset + index)
        .unwrap_or(source.len())
}

fn diag(module: &str, message: impl Into<String>) -> Diagnostic {
    Diagnostic {
        module: module.to_string(),
        line: 1,
        column: 1,
        message: message.into(),
    }
}
