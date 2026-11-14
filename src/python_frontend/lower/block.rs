//! An ad-hoc statement block (#129, DESIGN L2), lowered as the body of ONE function that
//! the caller never declares. Statements go through the unchanged `surface`, `usage` and
//! `Lower` rules of a submitted function body; only the frame differs:
//!
//! * no parameters, and the block is absent from the signature catalog (no self-call);
//! * the first typed `return` fixes the result type, later returns must match it;
//! * falling off the end is allowed (the lifecycle-block precedent).
//!
//! The source is parsed as written, never wrapped in a synthetic `def`, so every
//! diagnostic position is the position in the submitted text.
use super::*;

/// The frontend's inert effect declarations for a block: the names a function would
/// write in `@effects(...)` plus `async`. They decide which call sites lower, not what
/// the caller is allowed to do.
const EFFECTS: [&str; 4] = ["io", "fail", "mut", "async"];

/// Program-free structural preflight of a block. The returned references must be
/// authorized before any runtime signature is supplied to [`lower`].
pub(in crate::python_frontend) fn preflight(
    source: &str,
    scope: &str,
) -> Result<Preflight, Diagnostic> {
    let cx = Context {
        source,
        module: scope,
    };
    if !super::super::identifier(scope) {
        return Err(cx.error(0, "module must be a simple ASCII identifier without dunders"));
    }
    let suite = cx.parse()?;
    statements_only(&cx, &suite)?;
    let mut references = Vec::new();
    let mut type_references = Vec::new();
    super::super::surface::statements(&cx, &suite, false, &mut references, &mut type_references)?;
    let usage = super::super::references::usage::collect_block(&cx, &suite)?;
    Ok(assemble_preflight(
        references,
        type_references,
        Vec::new(),
        Vec::new(),
        usage,
    ))
}

/// A block executes statements; it cannot declare anything. Say so before the generic
/// statement diagnostics, because the fix is a different tool.
fn statements_only(cx: &Context<'_>, suite: &[py::Stmt]) -> Result<(), Diagnostic> {
    for stmt in suite {
        match stmt {
            py::Stmt::FunctionDef(_) | py::Stmt::AsyncFunctionDef(_) | py::Stmt::ClassDef(_) => {
                return Err(cx.at(stmt, "a run block executes statements only; declare functions and types with the mutate tool, then call them as Module.function"));
            }
            py::Stmt::Import(_) | py::Stmt::ImportFrom(_) => {
                return Err(cx.at(stmt, "import is unsupported; call runtime builtins directly and user functions as Module.function"));
            }
            _ => {}
        }
    }
    if suite.iter().all(|stmt| matches!(stmt, py::Stmt::Pass(_))) {
        return Err(cx.error(0, "run block is empty; write at least one statement"));
    }
    Ok(())
}

/// Strict lowering against the supplied (authority-filtered) signatures. The result is
/// the ordinary single-module operation shape, so every later check accepts it as is.
pub(in crate::python_frontend) fn lower(
    source: &str,
    scope: &str,
    function: &str,
    signatures: HashMap<String, Signature>,
    types: &TypeCatalog,
    shared_vars: &HashMap<String, Ty>,
) -> Result<LoweredSource, Diagnostic> {
    let cx = Context {
        source,
        module: scope,
    };
    let suite = cx.parse()?;
    let references = preflight(source, scope)?;
    require_resolved_references(&cx, &references, &signatures, types)?;
    let effects: Vec<String> = EFFECTS.iter().map(|effect| effect.to_string()).collect();
    let mut lower = Lower {
        cx: &cx,
        signatures: &signatures,
        types,
        shared_vars,
        env: HashMap::new(),
        seen: HashSet::new(),
        result: Ty::Unknown,
        infer_result: true,
        effects: &effects,
    };
    let body = lower.block(&suite)?;
    let return_type = declared_result(&lower.result);
    let operations = vec![p::Operation::Module(p::ModuleDecl {
        name: scope.into(),
        body: vec![p::Operation::Function(p::FunctionDecl {
            name: function.into(),
            params: Vec::new(),
            return_type,
            effects,
            body,
            doc: None,
        })],
        doc: None,
    })];
    let source_map = suite.iter().map(|stmt| cx.span(stmt, 0)).collect();
    cx.finish(operations, source_map)
}

/// Called for every `return` of a block. An unresolved (`Unknown`) value never fixes or
/// contradicts the type. The Result-return rule of declared functions (a plain `T` under
/// `Result[T]`) is deliberately NOT applied: a block is executed raw, without invocation
/// normalization, so nothing would wrap the plain value.
pub(super) fn infer_result(
    lower: &mut Lower<'_, '_>,
    actual: &Ty,
    node: &impl py::Ranged,
) -> Result<(), Diagnostic> {
    if lower.result == Ty::Unknown {
        lower.result = actual.clone();
        return Ok(());
    }
    if !compatible(&lower.result, actual) {
        return Err(lower.cx.at(node, format!("return type mismatch: an earlier return of this block has type {:?}, this one has {actual:?}; a block returns one type (convert with to_string(x) if you want text)", lower.result)));
    }
    Ok(())
}

/// The declaration's return annotation is metadata for the unchanged downstream type
/// guards; execution never reads it. An unresolved result is declared as text.
fn declared_result(result: &Ty) -> p::TypeExpr {
    if *result != Ty::Unknown && super::super::result_returns::known(result) {
        result.parser()
    } else {
        Ty::Str.parser()
    }
}
