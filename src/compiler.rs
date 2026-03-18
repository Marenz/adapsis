//! Cranelift-based JIT compiler for Forge programs.
//!
//! Compiles Forge AST to native machine code via Cranelift IR.
//! For Phase 5, we support:
//! - Int → i64
//! - Float → f64
//! - Bool → i8 (0 or 1)
//! - Arithmetic, comparison, logic operators
//! - if/elif/else (Branch nodes)
//! - Function calls (direct)
//! - Check statements (conditional trap)
//!
//! Strings and structs are deferred to a later phase.

use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, FuncId, Linkage, Module};

use crate::ast;

/// A compiled Forge program, ready to execute.
pub struct CompiledProgram {
    module: JITModule,
    functions: HashMap<String, FuncId>,
}

impl CompiledProgram {
    /// Call a compiled function with i64 arguments, returning i64.
    /// This is the simplest calling convention for Phase 5.
    pub fn call_i64(&mut self, name: &str, args: &[i64]) -> Result<i64> {
        let func_id = self
            .functions
            .get(name)
            .ok_or_else(|| anyhow!("compiled function `{name}` not found"))?;
        let func_ptr = self.module.get_finalized_function(*func_id);

        // Safety: we trust the compiled code matches the signature
        unsafe {
            match args.len() {
                0 => {
                    let f: fn() -> i64 = std::mem::transmute(func_ptr);
                    Ok(f())
                }
                1 => {
                    let f: fn(i64) -> i64 = std::mem::transmute(func_ptr);
                    Ok(f(args[0]))
                }
                2 => {
                    let f: fn(i64, i64) -> i64 = std::mem::transmute(func_ptr);
                    Ok(f(args[0], args[1]))
                }
                3 => {
                    let f: fn(i64, i64, i64) -> i64 = std::mem::transmute(func_ptr);
                    Ok(f(args[0], args[1], args[2]))
                }
                4 => {
                    let f: fn(i64, i64, i64, i64) -> i64 = std::mem::transmute(func_ptr);
                    Ok(f(args[0], args[1], args[2], args[3]))
                }
                n => bail!("call_i64: unsupported argument count {n} (max 4)"),
            }
        }
    }
}

/// Compile a Forge program to native code via Cranelift JIT.
pub fn compile(program: &ast::Program) -> Result<CompiledProgram> {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    flag_builder.set("opt_level", "speed").unwrap();

    let isa_builder = cranelift_native::builder().map_err(|e| anyhow!("{e}"))?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|e| anyhow!("{e}"))?;

    let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(builder);

    let mut compiled_functions: HashMap<String, FuncId> = HashMap::new();

    // First pass: declare all functions (so they can call each other)
    let mut signatures: HashMap<String, Signature> = HashMap::new();
    for func in &program.functions {
        let sig = build_signature(&mut module, func)?;
        let func_id = module.declare_function(&func.name, Linkage::Local, &sig)?;
        compiled_functions.insert(func.name.clone(), func_id);
        signatures.insert(func.name.clone(), sig);
    }

    // Second pass: compile function bodies
    for func in &program.functions {
        let sig = signatures.get(&func.name).unwrap().clone();
        let func_id = *compiled_functions.get(&func.name).unwrap();

        let mut ctx = module.make_context();
        ctx.func.signature = sig;

        compile_function(&mut module, &mut ctx, func, &compiled_functions)?;

        module.define_function(func_id, &mut ctx)?;
        module.clear_context(&mut ctx);
    }

    // Finalize all functions
    module.finalize_definitions()?;

    Ok(CompiledProgram {
        module,
        functions: compiled_functions,
    })
}

/// Build a Cranelift signature from a Forge function declaration.
fn build_signature(module: &mut JITModule, func: &ast::FunctionDecl) -> Result<Signature> {
    let mut sig = module.make_signature();

    for param in &func.params {
        let cl_type = forge_type_to_cranelift(&param.ty)?;
        sig.params.push(AbiParam::new(cl_type));
    }

    let ret_type = forge_type_to_cranelift(&func.return_type)?;
    sig.returns.push(AbiParam::new(ret_type));

    Ok(sig)
}

/// Map Forge types to Cranelift types.
/// Phase 5 only supports numeric types.
fn forge_type_to_cranelift(ty: &ast::Type) -> Result<types::Type> {
    match ty {
        ast::Type::Int => Ok(types::I64),
        ast::Type::Float => Ok(types::F64),
        ast::Type::Bool => Ok(types::I8),
        ast::Type::Byte => Ok(types::I8),
        // Result<T> is compiled as T for now (error handling via traps)
        ast::Type::Result(inner) => forge_type_to_cranelift(inner),
        _ => bail!(
            "type {:?} not yet supported in compiler (Phase 5 supports Int, Float, Bool)",
            ty
        ),
    }
}

/// Compile a single function body to Cranelift IR.
fn compile_function(
    module: &mut JITModule,
    ctx: &mut codegen::Context,
    func: &ast::FunctionDecl,
    all_functions: &HashMap<String, FuncId>,
) -> Result<()> {
    let mut builder_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);

    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    // Map parameter names to SSA values
    let mut vars: HashMap<String, Variable> = HashMap::new();
    let mut var_counter: u32 = 0;

    for (i, param) in func.params.iter().enumerate() {
        let val = builder.block_params(entry_block)[i];
        let var = Variable::new(var_counter as usize);
        var_counter += 1;
        let cl_type = forge_type_to_cranelift(&param.ty)?;
        builder.declare_var(var, cl_type);
        builder.def_var(var, val);
        vars.insert(param.name.clone(), var);
    }

    let mut comp_ctx = CompilationContext {
        module,
        builder: &mut builder,
        vars: &mut vars,
        var_counter: &mut var_counter,
        all_functions,
        return_type: &func.return_type,
        terminated: false,
    };

    compile_body(&mut comp_ctx, &func.body)?;

    // If we get here without a return, emit a default return
    if !comp_ctx.terminated {
        let ret_type = forge_type_to_cranelift(&func.return_type)?;
        let default_val = if ret_type == types::F64 {
            comp_ctx.builder.ins().f64const(0.0)
        } else {
            comp_ctx.builder.ins().iconst(ret_type, 0)
        };
        comp_ctx.builder.ins().return_(&[default_val]);
    }

    builder.finalize();
    Ok(())
}

struct CompilationContext<'a, 'b> {
    module: &'a mut JITModule,
    builder: &'b mut FunctionBuilder<'a>,
    vars: &'b mut HashMap<String, Variable>,
    var_counter: &'b mut u32,
    all_functions: &'b HashMap<String, FuncId>,
    return_type: &'b ast::Type,
    terminated: bool,
}

fn alloc_var(ctx: &mut CompilationContext, name: &str, ty: types::Type) -> Variable {
    let var = Variable::new(*ctx.var_counter as usize);
    *ctx.var_counter += 1;
    ctx.builder.declare_var(var, ty);
    ctx.vars.insert(name.to_string(), var);
    var
}

fn compile_body(ctx: &mut CompilationContext, stmts: &[ast::Statement]) -> Result<()> {
    for stmt in stmts {
        if ctx.terminated {
            break;
        }
        compile_statement(ctx, stmt)?;
    }
    Ok(())
}

fn compile_statement(ctx: &mut CompilationContext, stmt: &ast::Statement) -> Result<()> {
    match &stmt.kind {
        ast::StatementKind::Let { name, ty, value } => {
            let cl_type = forge_type_to_cranelift(ty)?;
            let val = compile_expr(ctx, value, cl_type)?;
            let var = alloc_var(ctx, name, cl_type);
            ctx.builder.def_var(var, val);
            Ok(())
        }

        ast::StatementKind::Call { binding, call } => {
            let ret_type = if let Some(b) = binding {
                forge_type_to_cranelift(&b.ty)?
            } else {
                types::I64
            };
            let val = compile_call(ctx, call, ret_type)?;
            if let Some(b) = binding {
                let cl_type = forge_type_to_cranelift(&b.ty)?;
                let var = alloc_var(ctx, &b.name, cl_type);
                ctx.builder.def_var(var, val);
            }
            Ok(())
        }

        ast::StatementKind::Check {
            condition, on_fail, ..
        } => {
            // Compile condition, trap if false
            let cond = compile_expr(ctx, condition, types::I8)?;
            // Branch: if cond is true, continue; if false, trap
            let trap_block = ctx.builder.create_block();
            let ok_block = ctx.builder.create_block();

            ctx.builder.ins().brif(cond, ok_block, &[], trap_block, &[]);

            ctx.builder.switch_to_block(trap_block);
            ctx.builder.seal_block(trap_block);
            ctx.builder.ins().trap(TrapCode::user(0).unwrap());

            ctx.builder.switch_to_block(ok_block);
            ctx.builder.seal_block(ok_block);
            Ok(())
        }

        ast::StatementKind::Return { value } => {
            let ret_cl_type = forge_type_to_cranelift(ctx.return_type)?;
            let val = compile_expr(ctx, value, ret_cl_type)?;
            ctx.builder.ins().return_(&[val]);
            ctx.terminated = true;
            Ok(())
        }

        ast::StatementKind::Branch {
            condition,
            then_body,
            else_body,
        } => {
            let cond = compile_expr(ctx, condition, types::I8)?;
            let then_block = ctx.builder.create_block();
            let else_block = ctx.builder.create_block();
            let merge_block = ctx.builder.create_block();

            ctx.builder
                .ins()
                .brif(cond, then_block, &[], else_block, &[]);

            // Then branch
            ctx.builder.switch_to_block(then_block);
            ctx.builder.seal_block(then_block);
            let saved = ctx.terminated;
            ctx.terminated = false;
            compile_body(ctx, then_body)?;
            let then_terminated = ctx.terminated;
            if !then_terminated {
                ctx.builder.ins().jump(merge_block, &[]);
            }

            // Else branch
            ctx.builder.switch_to_block(else_block);
            ctx.builder.seal_block(else_block);
            ctx.terminated = false;
            compile_body(ctx, else_body)?;
            let else_terminated = ctx.terminated;
            if !else_terminated {
                ctx.builder.ins().jump(merge_block, &[]);
            }

            // Only terminated if both branches terminated
            ctx.terminated = then_terminated && else_terminated;

            ctx.builder.switch_to_block(merge_block);
            ctx.builder.seal_block(merge_block);
            Ok(())
        }

        ast::StatementKind::Each { .. } => {
            bail!("each loops not yet supported in compiler")
        }

        ast::StatementKind::Yield { .. } => {
            bail!("yield not yet supported in compiler")
        }
    }
}

fn compile_call(
    ctx: &mut CompilationContext,
    call: &ast::CallExpr,
    expected_type: types::Type,
) -> Result<Value> {
    let func_id = ctx
        .all_functions
        .get(&call.callee)
        .ok_or_else(|| anyhow!("undefined function `{}` in compiled code", call.callee))?;

    let local_callee = ctx.module.declare_func_in_func(*func_id, ctx.builder.func);

    let mut arg_vals = Vec::new();
    for arg in &call.args {
        // Infer type from the argument (simplified: assume i64 for now)
        let val = compile_expr(ctx, arg, types::I64)?;
        arg_vals.push(val);
    }

    let inst = ctx.builder.ins().call(local_callee, &arg_vals);
    Ok(ctx.builder.inst_results(inst)[0])
}

fn compile_expr(
    ctx: &mut CompilationContext,
    expr: &ast::Expr,
    expected_type: types::Type,
) -> Result<Value> {
    match expr {
        ast::Expr::Literal(lit) => match lit {
            ast::Literal::Int(n) => {
                if expected_type == types::F64 {
                    Ok(ctx.builder.ins().f64const(*n as f64))
                } else {
                    Ok(ctx.builder.ins().iconst(types::I64, *n))
                }
            }
            ast::Literal::Float(f) => Ok(ctx.builder.ins().f64const(*f)),
            ast::Literal::Bool(b) => Ok(ctx.builder.ins().iconst(types::I8, *b as i64)),
            ast::Literal::String(_) => {
                bail!("string literals not yet supported in compiler")
            }
        },

        ast::Expr::Identifier(name) => {
            let var = ctx
                .vars
                .get(name)
                .ok_or_else(|| anyhow!("undefined variable `{name}` in compiler"))?;
            Ok(ctx.builder.use_var(*var))
        }

        ast::Expr::Binary { left, op, right } => {
            // Determine if we're doing float or int arithmetic
            let is_float =
                expected_type == types::F64 || is_float_expr(left) || is_float_expr(right);
            let operand_type = if is_float { types::F64 } else { types::I64 };

            let lhs = compile_expr(ctx, left, operand_type)?;
            let rhs = compile_expr(ctx, right, operand_type)?;

            // Comparison ops always return I8 (bool)
            let is_comparison = matches!(
                op,
                ast::BinaryOp::Equal
                    | ast::BinaryOp::NotEqual
                    | ast::BinaryOp::GreaterThan
                    | ast::BinaryOp::LessThan
                    | ast::BinaryOp::GreaterThanOrEqual
                    | ast::BinaryOp::LessThanOrEqual
            );

            if is_float && !is_comparison {
                match op {
                    ast::BinaryOp::Add => Ok(ctx.builder.ins().fadd(lhs, rhs)),
                    ast::BinaryOp::Sub => Ok(ctx.builder.ins().fsub(lhs, rhs)),
                    ast::BinaryOp::Mul => Ok(ctx.builder.ins().fmul(lhs, rhs)),
                    ast::BinaryOp::Div => Ok(ctx.builder.ins().fdiv(lhs, rhs)),
                    _ => bail!("unsupported float op {:?}", op),
                }
            } else if is_float && is_comparison {
                let cc = match op {
                    ast::BinaryOp::Equal => FloatCC::Equal,
                    ast::BinaryOp::NotEqual => FloatCC::NotEqual,
                    ast::BinaryOp::GreaterThan => FloatCC::GreaterThan,
                    ast::BinaryOp::LessThan => FloatCC::LessThan,
                    ast::BinaryOp::GreaterThanOrEqual => FloatCC::GreaterThanOrEqual,
                    ast::BinaryOp::LessThanOrEqual => FloatCC::LessThanOrEqual,
                    _ => bail!("unsupported float comparison {:?}", op),
                };
                let cmp = ctx.builder.ins().fcmp(cc, lhs, rhs);
                Ok(cmp)
            } else if is_comparison {
                let cc = match op {
                    ast::BinaryOp::Equal => IntCC::Equal,
                    ast::BinaryOp::NotEqual => IntCC::NotEqual,
                    ast::BinaryOp::GreaterThan => IntCC::SignedGreaterThan,
                    ast::BinaryOp::LessThan => IntCC::SignedLessThan,
                    ast::BinaryOp::GreaterThanOrEqual => IntCC::SignedGreaterThanOrEqual,
                    ast::BinaryOp::LessThanOrEqual => IntCC::SignedLessThanOrEqual,
                    _ => bail!("unsupported int comparison {:?}", op),
                };
                let cmp = ctx.builder.ins().icmp(cc, lhs, rhs);
                Ok(cmp)
            } else {
                match op {
                    ast::BinaryOp::Add => Ok(ctx.builder.ins().iadd(lhs, rhs)),
                    ast::BinaryOp::Sub => Ok(ctx.builder.ins().isub(lhs, rhs)),
                    ast::BinaryOp::Mul => Ok(ctx.builder.ins().imul(lhs, rhs)),
                    ast::BinaryOp::Div => Ok(ctx.builder.ins().sdiv(lhs, rhs)),
                    ast::BinaryOp::Mod => Ok(ctx.builder.ins().srem(lhs, rhs)),
                    ast::BinaryOp::And => Ok(ctx.builder.ins().band(lhs, rhs)),
                    ast::BinaryOp::Or => Ok(ctx.builder.ins().bor(lhs, rhs)),
                    _ => bail!("unsupported int op {:?}", op),
                }
            }
        }

        ast::Expr::Unary { op, expr } => {
            let val = compile_expr(ctx, expr, expected_type)?;
            match op {
                ast::UnaryOp::Not => {
                    let one = ctx.builder.ins().iconst(types::I8, 1);
                    Ok(ctx.builder.ins().bxor(val, one))
                }
                ast::UnaryOp::Neg => {
                    if expected_type == types::F64 {
                        Ok(ctx.builder.ins().fneg(val))
                    } else {
                        let zero = ctx.builder.ins().iconst(types::I64, 0);
                        Ok(ctx.builder.ins().isub(zero, val))
                    }
                }
            }
        }

        ast::Expr::Call(call) => compile_call(ctx, call, expected_type),

        ast::Expr::FieldAccess { .. } => {
            bail!("field access not yet supported in compiler")
        }

        ast::Expr::StructInit { .. } => {
            bail!("struct construction not yet supported in compiler")
        }
    }
}

/// Heuristic: check if an expression likely produces a float value.
fn is_float_expr(expr: &ast::Expr) -> bool {
    match expr {
        ast::Expr::Literal(ast::Literal::Float(_)) => true,
        ast::Expr::Binary { left, right, .. } => is_float_expr(left) || is_float_expr(right),
        ast::Expr::Unary { expr, .. } => is_float_expr(expr),
        _ => false,
    }
}
