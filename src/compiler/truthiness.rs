//! Type-aware NOT. String pointers and Float values must not be integer-tested.
//! Unsupported representations refuse compilation so the ordinary evaluator can
//! fall back before execution; no tree truthiness or public ABI is changed.
use super::*;

pub(super) fn parameter_types(function: &ast::FunctionDecl) -> HashMap<String, Option<ast::Type>> {
    let mut types = HashMap::new();
    for param in &function.params {
        types
            .entry(param.name.clone())
            .and_modify(|ty| *ty = None)
            .or_insert(Some(param.ty.clone()));
    }
    invalidate_bindings(&function.body, &mut types);
    // String ABI slots share the legacy variable namespace. Refuse ambiguous
    // synthetic-slot names (including assignments) rather than reading a scalar
    // as the length. Named/struct parameter expansion is not certified here.
    let names: Vec<_> = types.keys().cloned().collect();
    for param in &function.params {
        if names.contains(&format!("{}_len", param.name))
            || names.contains(&format!("{}_ptr", param.name))
            || function
                .params
                .iter()
                .any(|p| matches!(p.ty, ast::Type::Struct(_) | ast::Type::TaggedUnion(_)))
        {
            types.insert(param.name.clone(), None);
        }
    }
    types
}

fn invalidate_bindings(body: &[ast::Statement], types: &mut HashMap<String, Option<ast::Type>>) {
    use ast::StatementKind as S;
    for statement in body {
        match &statement.kind {
            S::Let { name, .. } | S::Set { name, .. } | S::Await { name, .. } => {
                types.insert(name.clone(), None);
            }
            S::Call {
                binding: Some(b), ..
            }
            | S::Spawn {
                binding: Some(b), ..
            } => {
                types.insert(b.name.clone(), None);
            }
            S::Branch {
                then_body,
                else_body,
                ..
            } => {
                invalidate_bindings(then_body, types);
                invalidate_bindings(else_body, types);
            }
            S::While { body, .. } => invalidate_bindings(body, types),
            S::Each { binding, body, .. } => {
                types.insert(binding.name.clone(), None);
                invalidate_bindings(body, types);
            }
            S::Match { arms, .. } => {
                for arm in arms {
                    for name in &arm.bindings {
                        types.insert(name.clone(), None);
                    }
                    invalidate_bindings(&arm.body, types);
                }
            }
            _ => {}
        }
    }
}

pub(super) fn record_binding(ctx: &mut CompilationContext, name: &str, _ty: &ast::Type) {
    // Local annotations alone are not proof of the runtime value representation.
    ctx.value_types.insert(name.into(), None);
}

pub(super) fn compile_not(ctx: &mut CompilationContext, expression: &ast::Expr) -> Result<Value> {
    let truth = emit_truthy(ctx, expression, Policy::Not)?;
    Ok(ctx.builder.ins().icmp_imm(IntCC::Equal, truth, 0))
}

#[derive(Clone, Copy, PartialEq)]
enum Policy {
    Not,
    Condition,
    Argument,
    NotArgument,
}

pub(super) fn compile_condition(
    ctx: &mut CompilationContext,
    expression: &ast::Expr,
) -> Result<Value> {
    emit_truthy(ctx, expression, Policy::Condition)
}

fn emit_truthy(
    ctx: &mut CompilationContext,
    expression: &ast::Expr,
    policy: Policy,
) -> Result<Value> {
    let ty = operand_type(ctx, expression, policy)?;
    if ty == ast::Type::String {
        let length = string_length(ctx, expression)?;
        return Ok(ctx.builder.ins().icmp_imm(IntCC::NotEqual, length, 0));
    }
    if !matches!(ty, ast::Type::Bool | ast::Type::Int | ast::Type::Float) {
        bail!("JIT NOT has no proven representation for {ty:?}; use the interpreter");
    }
    // Evaluate exactly once even when its truth value is constant (Float). A
    // call or failing subexpression must not disappear because its type is Float.
    let value = emit_proven_operand(ctx, expression, &ty, policy)?;
    if ty == ast::Type::Float {
        if ctx.builder.func.dfg.value_type(value) != types::F64 {
            bail!("JIT NOT Float representation mismatch; use the interpreter");
        }
        // Value::is_truthy treats EVERY Float as true, including zero and NaN.
        return Ok(ctx.builder.ins().iconst(types::I8, 1));
    }
    let actual = ctx.builder.func.dfg.value_type(value);
    if !matches!(actual, types::I8 | types::I16 | types::I32 | types::I64) {
        bail!("JIT NOT integer representation mismatch; use the interpreter");
    }
    Ok(ctx.builder.ins().icmp_imm(IntCC::NotEqual, value, 0))
}

fn string_length(ctx: &mut CompilationContext, expression: &ast::Expr) -> Result<Value> {
    let length = match expression {
        ast::Expr::Identifier(name) => {
            // Immutable entry-block ABI value, not a late lookup in vars: a
            // literal's `_last_str_len` scratch can alias a parameter's slot.
            // parameter_types invalidates any source rebinding throughout the
            // function, including writes after a loop header is compiled.
            ctx.string_parameter_lengths
                .get(name)
                .copied()
                .ok_or_else(|| {
                    anyhow!(
                        "JIT NOT lacks a proven string length for `{name}`; use the interpreter"
                    )
                })?
        }
        ast::Expr::Literal(ast::Literal::String(text)) => {
            ctx.builder.ins().iconst(types::I64, text.len() as i64)
        }
        _ => bail!(
            "JIT NOT string expression has no proven length representation; use the interpreter"
        ),
    };
    Ok(length)
}

/// Every identifier in the admitted recursive shape uses its original SSA
/// identity. Never hand compound operands back to compile_expr's mutable names.
fn emit_proven_operand(
    ctx: &mut CompilationContext,
    expression: &ast::Expr,
    ty: &ast::Type,
    policy: Policy,
) -> Result<Value> {
    use ast::{Expr as E, Literal as L, UnaryOp as U};
    match expression {
        E::Literal(L::Int(n)) => Ok(ctx.builder.ins().iconst(types::I64, *n)),
        E::Literal(L::Float(n)) => Ok(ctx.builder.ins().f64const(*n)),
        E::Literal(L::Bool(b)) => Ok(ctx.builder.ins().iconst(types::I8, i64::from(*b))),
        E::Identifier(name) => ctx
            .scalar_parameter_values
            .get(name)
            .copied()
            .ok_or_else(|| {
                anyhow!("JIT NOT lacks a proven value identity for `{name}`; use the interpreter")
            }),
        E::Unary { op: U::Not, expr } => compile_not(ctx, expr),
        E::Unary { op: U::Neg, expr } => {
            let value = emit_proven_operand(ctx, expr, ty, policy)?;
            if *ty == ast::Type::Float {
                Ok(ctx.builder.ins().fneg(value))
            } else {
                Ok(ctx.builder.ins().ineg(value))
            }
        }
        // operand_type already proved zero arguments and an actual scalar
        // callee signature. Emit the call once even for always-truthy Float.
        E::Call(call) => emit_scalar_call(ctx, call, policy),
        E::Binary { left, op, right } if matches!(op, ast::BinaryOp::And | ast::BinaryOp::Or) => {
            compile_logical(ctx, left, op, right)
        }
        E::Binary { left, op, right } => emit_comparison(ctx, left, op, right, policy),
        _ => bail!("JIT NOT operand has no proven value emitter; use the interpreter"),
    }
}

fn operand_type(
    ctx: &CompilationContext,
    expression: &ast::Expr,
    policy: Policy,
) -> Result<ast::Type> {
    use ast::{Expr as E, Literal as L, Type as T};
    match expression {
        E::Literal(L::Int(_)) => Ok(T::Int),
        E::Literal(L::Float(_)) => Ok(T::Float),
        E::Literal(L::Bool(_)) => Ok(T::Bool),
        E::Literal(L::String(_)) => Ok(T::String),
        E::Identifier(name) => ctx
            .value_types
            .get(name)
            .and_then(Clone::clone)
            .ok_or_else(|| {
                anyhow!("JIT NOT cannot prove the binding type of `{name}`; use the interpreter")
            }),
        E::Unary {
            op: ast::UnaryOp::Not,
            expr,
        } => {
            let nested = if matches!(policy, Policy::Argument | Policy::NotArgument) {
                Policy::NotArgument
            } else {
                Policy::Not
            };
            operand_type(ctx, expr, nested)?;
            Ok(T::Bool)
        }
        E::Unary {
            op: ast::UnaryOp::Neg,
            expr,
        } => {
            let ty = operand_type(ctx, expr, policy)?;
            if matches!(ty, T::Int | T::Float) {
                Ok(ty)
            } else {
                bail!("JIT NOT negation requires a proven numeric operand; use the interpreter")
            }
        }
        E::Call(call) => scalar_call_type(ctx, call, policy),
        E::Binary { left, op, right } => {
            if matches!(policy, Policy::Not | Policy::NotArgument) {
                bail!("JIT NOT binary operands lack proven recursive value identity; use the interpreter");
            }
            let lhs = operand_type(ctx, left, policy)?;
            let rhs = operand_type(ctx, right, policy)?;
            if lhs == T::Float
                && rhs == T::Float
                && matches!(op, ast::BinaryOp::Equal | ast::BinaryOp::NotEqual)
            {
                bail!("JIT Float equality is unsupported: runtime epsilon comparison is not IEEE equality; use the interpreter");
            }
            if matches!(op, ast::BinaryOp::And | ast::BinaryOp::Or) {
                if !truthy_type(&lhs) || !truthy_type(&rhs) {
                    bail!("JIT logical operands have unproven representations");
                }
            } else if lhs != rhs || !comparison_allowed(op, &lhs) {
                bail!(
                    "JIT condition requires a homogeneous scalar comparison; use the interpreter"
                );
            }
            Ok(T::Bool)
        }
        _ => bail!("JIT NOT operand shape requires the interpreter"),
    }
}

fn scalar_type(ty: &ast::Type) -> bool {
    matches!(ty, ast::Type::Bool | ast::Type::Int | ast::Type::Float)
}

fn truthy_type(ty: &ast::Type) -> bool {
    scalar_type(ty) || *ty == ast::Type::String
}

fn scalar_call_type(
    ctx: &CompilationContext,
    call: &ast::CallExpr,
    policy: Policy,
) -> Result<ast::Type> {
    if ctx.program.is_union_variant(&call.callee) {
        bail!("JIT truthiness union constructor requires the interpreter");
    }
    if matches!(policy, Policy::Not | Policy::NotArgument) && !call.args.is_empty() {
        bail!("JIT NOT argument-bearing calls lack proven recursive value identity; use the interpreter");
    }
    let function = ctx
        .program
        .get_function(&call.callee)
        .ok_or_else(|| anyhow!("JIT truthiness cannot resolve user callee signature"))?;
    if function.params.len() != call.args.len() {
        bail!("JIT truthiness call arity mismatch");
    }
    if !scalar_type(&function.return_type) {
        bail!("JIT truthiness requires a scalar callee result");
    }
    if function
        .effects
        .iter()
        .any(|effect| matches!(effect, ast::Effect::Io | ast::Effect::Async))
    {
        bail!("JIT truthiness refuses IO/async calls; use the interpreter");
    }
    if (matches!(policy, Policy::Argument | Policy::NotArgument) || !call.args.is_empty())
        && function.effects.contains(&ast::Effect::Fail)
    {
        bail!("JIT truthiness refuses fallible argument calls until failure propagation is proven");
    }
    for (argument, parameter) in call.args.iter().zip(&function.params) {
        let actual = operand_type(ctx, argument, Policy::Argument)?;
        if !scalar_type(&actual) || actual != parameter.ty {
            bail!("JIT truthiness call arguments require exact scalar types; use the interpreter");
        }
    }
    Ok(function.return_type.clone())
}

fn emit_scalar_call(
    ctx: &mut CompilationContext,
    call: &ast::CallExpr,
    policy: Policy,
) -> Result<Value> {
    scalar_call_type(ctx, call, policy)?;
    let id = *ctx
        .all_functions
        .get(&call.callee)
        .ok_or_else(|| anyhow!("JIT truthiness callee is not compiled"))?;
    let mut values = Vec::with_capacity(call.args.len());
    for argument in &call.args {
        let ty = operand_type(ctx, argument, Policy::Argument)?;
        values.push(emit_proven_operand(ctx, argument, &ty, Policy::Argument)?);
    }
    let callee = ctx.module.declare_func_in_func(id, ctx.builder.func);
    let instruction = ctx.builder.ins().call(callee, &values);
    Ok(ctx.builder.inst_results(instruction)[0])
}

fn comparison_allowed(op: &ast::BinaryOp, ty: &ast::Type) -> bool {
    use ast::BinaryOp as B;
    match op {
        B::Equal | B::NotEqual => matches!(ty, ast::Type::Bool | ast::Type::Int),
        B::GreaterThan | B::GreaterThanOrEqual | B::LessThan | B::LessThanOrEqual => {
            matches!(ty, ast::Type::Int | ast::Type::Float)
        }
        _ => false,
    }
}

fn emit_comparison(
    ctx: &mut CompilationContext,
    left: &ast::Expr,
    op: &ast::BinaryOp,
    right: &ast::Expr,
    policy: Policy,
) -> Result<Value> {
    use ast::BinaryOp as B;
    let ty = operand_type(ctx, left, policy)?;
    let lhs = emit_proven_operand(ctx, left, &ty, policy)?;
    let rhs = emit_proven_operand(ctx, right, &ty, policy)?;
    if ty == ast::Type::Float {
        let condition = match op {
            B::GreaterThan => FloatCC::GreaterThan,
            B::GreaterThanOrEqual => FloatCC::GreaterThanOrEqual,
            B::LessThan => FloatCC::LessThan,
            B::LessThanOrEqual => FloatCC::LessThanOrEqual,
            _ => bail!("JIT unsupported scalar comparison"),
        };
        Ok(ctx.builder.ins().fcmp(condition, lhs, rhs))
    } else {
        let condition = match op {
            B::Equal => IntCC::Equal,
            B::NotEqual => IntCC::NotEqual,
            B::GreaterThan => IntCC::SignedGreaterThan,
            B::GreaterThanOrEqual => IntCC::SignedGreaterThanOrEqual,
            B::LessThan => IntCC::SignedLessThan,
            B::LessThanOrEqual => IntCC::SignedLessThanOrEqual,
            _ => bail!("JIT unsupported scalar comparison"),
        };
        Ok(ctx.builder.ins().icmp(condition, lhs, rhs))
    }
}

/// Both branches are validated before emitting either; evaluation remains lazy.
pub(super) fn compile_logical(
    ctx: &mut CompilationContext,
    left: &ast::Expr,
    op: &ast::BinaryOp,
    right: &ast::Expr,
) -> Result<Value> {
    let is_or = matches!(op, ast::BinaryOp::Or);
    for expression in [left, right] {
        if !truthy_type(&operand_type(ctx, expression, Policy::Condition)?) {
            bail!("JIT logical operand has unproven representation; use the interpreter");
        }
    }
    let lhs = compile_condition(ctx, left)?;
    let result = Variable::new(*ctx.var_counter as usize);
    *ctx.var_counter += 1;
    ctx.builder.declare_var(result, types::I8);
    let rhs_block = ctx.builder.create_block();
    let end_block = ctx.builder.create_block();
    if is_or {
        let short_block = ctx.builder.create_block();
        ctx.builder
            .ins()
            .brif(lhs, short_block, &[], rhs_block, &[]);
        ctx.builder.switch_to_block(short_block);
        ctx.builder.seal_block(short_block);
        let one = ctx.builder.ins().iconst(types::I8, 1);
        ctx.builder.def_var(result, one);
        ctx.builder.ins().jump(end_block, &[]);
    } else {
        let short_block = ctx.builder.create_block();
        let not_lhs = ctx.builder.ins().icmp_imm(IntCC::Equal, lhs, 0);
        ctx.builder
            .ins()
            .brif(not_lhs, short_block, &[], rhs_block, &[]);
        ctx.builder.switch_to_block(short_block);
        ctx.builder.seal_block(short_block);
        let zero = ctx.builder.ins().iconst(types::I8, 0);
        ctx.builder.def_var(result, zero);
        ctx.builder.ins().jump(end_block, &[]);
    }
    ctx.builder.switch_to_block(rhs_block);
    ctx.builder.seal_block(rhs_block);
    let rhs = compile_condition(ctx, right)?;
    ctx.builder.def_var(result, rhs);
    ctx.builder.ins().jump(end_block, &[]);
    ctx.builder.switch_to_block(end_block);
    ctx.builder.seal_block(end_block);
    Ok(ctx.builder.use_var(result))
}
