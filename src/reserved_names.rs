//! The `__` identifier prefix is reserved for runtime-installed bindings.
//!
//! The runtime installs capability-bearing values into the evaluator's
//! environment under `__`-prefixed names (`__coroutine_handle`, `__module_name`).
//! IO sites find them by name, so a program that could *read* one could copy the
//! handle into a spawn argument or a shared variable, and a program that could
//! *bind* one could reinstall a stashed handle later — in a delayed task, a timer,
//! a route handler or another speaker's turn — and act under the original turn's
//! identity and grants. User-written source therefore may not name them at all.
//!
//! The only `__` names user-visible ASTs may contain are the parser's own
//! placeholder captures for nested match patterns (`__pat_<n>`), which appear
//! exclusively as match-arm bindings of an arm with nested patterns.
use crate::ast;
use anyhow::{Result, bail};

pub(crate) const PREFIX: &str = "__";

pub(crate) fn is_reserved(name: &str) -> bool {
    name.starts_with(PREFIX)
}

fn parser_placeholder(name: &str) -> bool {
    name.strip_prefix("__pat_")
        .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

fn refuse(name: &str, role: &str) -> Result<()> {
    if is_reserved(name) {
        bail!(
            "`{name}` is reserved: identifiers starting with `{PREFIX}` are runtime-internal \
             and cannot be used as a {role}. Rename it (for example `{}`).",
            name.trim_start_matches('_')
        );
    }
    Ok(())
}

/// The scope an ad-hoc `run` block is lowered in when the caller names no module (#129).
///
/// The block's declaration is never applied, so this is a name for resolution and
/// diagnostics only. If a real module could carry it, such a block would resolve that
/// module's functions, types and `+shared` variables as its own; therefore no module with
/// this name can be declared. It is an ordinary identifier (the Python operation IR
/// accepts nothing else as a module name), chosen to be useless as a real module name.
pub(crate) const RUN_SCOPE: &str = "RunToolScratch";

/// Every module declaration (mutation, `module_create`, library load, replay) passes the
/// validator's `apply_module`, which calls this.
pub(crate) fn check_module(name: &str) -> Result<()> {
    if name == RUN_SCOPE {
        bail!(
            "module name `{RUN_SCOPE}` is reserved for the run tool's ad-hoc blocks and \
             cannot be declared; choose a different module name"
        );
    }
    Ok(())
}

pub(crate) fn check_param(name: &str) -> Result<()> {
    refuse(name, "parameter")
}

/// Checks one converted statement's own names and expressions. Nested bodies are
/// converted (and therefore checked) statement by statement by the caller.
pub(crate) fn check_statement(statement: &ast::Statement) -> Result<()> {
    use ast::StatementKind as S;
    match &statement.kind {
        S::Let { name, value, .. } => {
            refuse(name, "binding")?;
            expr(value)
        }
        S::Set { name, value } => {
            refuse(name, "assignment target")?;
            expr(value)
        }
        S::Call { binding, call } => {
            binding_name(binding.as_ref())?;
            call_expr(call)
        }
        S::Await { name, call, .. } => {
            refuse(name, "binding")?;
            call_expr(call)
        }
        S::Spawn { call, binding } => {
            binding_name(binding.as_ref())?;
            call_expr(call)
        }
        S::Each {
            iterator, binding, ..
        } => {
            refuse(&binding.name, "loop binding")?;
            expr(iterator)
        }
        S::Match {
            expr: subject,
            arms,
        } => {
            expr(subject)?;
            for arm in arms {
                for name in &arm.bindings {
                    if !(arm.patterns.is_some() && parser_placeholder(name)) {
                        refuse(name, "match capture")?;
                    }
                }
                for pattern in arm.patterns.iter().flatten() {
                    match_pattern(pattern)?;
                }
            }
            Ok(())
        }
        S::Check { condition, .. } | S::While { condition, .. } | S::Branch { condition, .. } => {
            expr(condition)
        }
        S::Return { value } | S::Yield { value } => expr(value),
        S::Source(
            ast::SourceOp::Add { source_type, .. } | ast::SourceOp::Replace { source_type, .. },
        ) => match source_type {
            ast::SourceType::Timer(interval) => expr(interval),
            _ => Ok(()),
        },
        S::Source(_) => Ok(()),
        S::Event(ast::EventOp::Emit { value, .. }) => expr(value),
        S::Event(_) => Ok(()),
    }
}

fn binding_name(binding: Option<&ast::Binding>) -> Result<()> {
    binding.map_or(Ok(()), |b| refuse(&b.name, "binding"))
}

fn match_pattern(pattern: &ast::MatchPattern) -> Result<()> {
    match pattern {
        ast::MatchPattern::Binding(name) => refuse(name, "match capture"),
        ast::MatchPattern::Variant { sub_patterns, .. } => {
            sub_patterns.iter().try_for_each(match_pattern)
        }
        ast::MatchPattern::Literal(_) => Ok(()),
    }
}

fn call_expr(call: &ast::CallExpr) -> Result<()> {
    call.args.iter().try_for_each(expr)
}

fn expr(value: &ast::Expr) -> Result<()> {
    match value {
        ast::Expr::Identifier(name) => refuse(name, "variable"),
        ast::Expr::FieldAccess { base, .. } => expr(base),
        ast::Expr::Call(call) => call_expr(call),
        ast::Expr::Binary { left, right, .. } => {
            expr(left)?;
            expr(right)
        }
        ast::Expr::Unary { expr: inner, .. } => expr(inner),
        ast::Expr::StructInit { fields, .. } => fields.iter().try_for_each(|f| expr(&f.value)),
        ast::Expr::Literal(_) => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use crate::{ast, parser, validator};

    fn apply(source: &str) -> anyhow::Result<()> {
        let mut program = ast::Program::default();
        for op in parser::parse(source)? {
            validator::apply_and_validate(&mut program, &op)?;
        }
        Ok(())
    }

    fn refused(source: &str, role: &str) {
        let error = apply(source)
            .expect_err("reserved name must be refused")
            .to_string();
        assert!(
            error.contains("`__coroutine_handle` is reserved"),
            "{error}"
        );
        assert!(error.contains(role), "{error}");
    }

    #[test]
    fn reserved_names_refuse_read_let_and_parameter() {
        refused(
            "+module M\n+fn leak ()->String [io,async]\n  +let h:String = to_string(__coroutine_handle)\n  +return h\n+end\n",
            "variable",
        );
        refused(
            "+module M\n+fn reinstall (h:String)->String\n  +let __coroutine_handle:String = h\n  +return h\n+end\n",
            "binding",
        );
        refused(
            "+module M\n+fn take (__coroutine_handle:String)->String\n  +return \"x\"\n+end\n",
            "parameter",
        );
    }

    #[test]
    fn reserved_names_refuse_set_each_await_spawn_and_capture() {
        for (body, role) in [
            (
                "  +set __coroutine_handle = x\n  +return x\n",
                "assignment target",
            ),
            (
                "  +each list(x) __coroutine_handle:String\n    +return x\n  +end\n  +return x\n",
                "loop binding",
            ),
            (
                "  +await __coroutine_handle:String = file_read(x)\n  +return x\n",
                "binding",
            ),
        ] {
            refused(
                &format!("+module M\n+fn f (x:String)->String [io,async]\n{body}+end\n"),
                role,
            );
        }
        refused(
            "+module M\n+type R = Ok(String) | Err(String)\n+fn f (r:R)->String\n  +match r\n  +case Ok(__coroutine_handle)\n    +return \"a\"\n  +case Err(e)\n    +return e\n  +end\n+end\n",
            "match capture",
        );
    }

    #[test]
    fn reserved_names_refuse_inline_eval_reads() {
        let program = ast::Program::default();
        let eval = |source: &str| {
            let (sender, _receiver) = tokio::sync::mpsc::channel(1);
            crate::eval::eval_inline_expr_with_io(
                &program,
                &parser::parse_single_expr(source).unwrap(),
                sender,
                None,
            )
        };
        let error = eval("to_string(__coroutine_handle)")
            .unwrap_err()
            .to_string();
        assert!(error.contains("reserved"), "{error}");
        assert!(
            matches!(eval("to_string(42)").unwrap(), crate::eval::Value::String(s) if s.as_str() == "42")
        );
    }

    #[test]
    fn reserved_names_refuse_top_level_statements() {
        let op = parser::parse("+let __coroutine_handle:Int = 1\n")
            .unwrap()
            .remove(0);
        let error = validator::convert_statement_op(&op)
            .unwrap_err()
            .to_string();
        assert!(error.contains("is reserved"), "{error}");
    }

    #[test]
    fn reserved_names_allow_parser_nested_pattern_placeholders_and_strings() {
        apply(
            "+module M\n+type Inner = A(Int) | B\n+type Outer = Wrap(Inner) | None\n+fn f (o:Outer)->Int\n  +match o\n  +case Wrap(A(n))\n    +return n\n  +case _\n    +return 0\n  +end\n+end\n",
        )
        .unwrap();
        // String contents are data, not identifiers (TgFormat's markdown regex).
        apply("+module M\n+fn g ()->String\n  +return \"__bold__\"\n+end\n").unwrap();
    }
}
