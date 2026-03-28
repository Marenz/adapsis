//! Cranelift-based JIT compiler for Adapsis programs.
//!
//! Compiles Adapsis AST to native machine code via Cranelift IR.
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

/// A compiled Adapsis program, ready to execute.
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

    /// Call a compiled function that returns a String (ptr, len pair).
    /// String args are passed as (ptr, len) pairs — so 1 string param = 2 i64 args.
    pub fn call_string(&mut self, name: &str, args: &[i64]) -> Result<String> {
        let func_id = self
            .functions
            .get(name)
            .ok_or_else(|| anyhow!("compiled function `{name}` not found"))?;
        let func_ptr = self.module.get_finalized_function(*func_id);

        // For string returns, the function returns (ptr: i64, len: i64)
        // We use a struct return convention
        #[repr(C)]
        struct StringReturn {
            ptr: i64,
            len: i64,
        }

        unsafe {
            let ret: StringReturn = match args.len() {
                0 => {
                    let f: fn() -> StringReturn = std::mem::transmute(func_ptr);
                    f()
                }
                1 => {
                    let f: fn(i64) -> StringReturn = std::mem::transmute(func_ptr);
                    f(args[0])
                }
                2 => {
                    let f: fn(i64, i64) -> StringReturn = std::mem::transmute(func_ptr);
                    f(args[0], args[1])
                }
                3 => {
                    let f: fn(i64, i64, i64) -> StringReturn = std::mem::transmute(func_ptr);
                    f(args[0], args[1], args[2])
                }
                4 => {
                    let f: fn(i64, i64, i64, i64) -> StringReturn = std::mem::transmute(func_ptr);
                    f(args[0], args[1], args[2], args[3])
                }
                n => bail!("call_string: unsupported argument count {n} (max 4)"),
            };

            let slice = std::slice::from_raw_parts(ret.ptr as *const u8, ret.len as usize);
            Ok(std::str::from_utf8(slice)
                .unwrap_or("<invalid utf8>")
                .to_string())
        }
    }
}

/// Check if a function can be compiled (only numeric types).
pub fn is_compilable_function(func: &ast::FunctionDecl) -> bool {
    // Check return type
    if !is_compilable_type(&func.return_type) {
        return false;
    }
    // Check param types
    for param in &func.params {
        if !is_compilable_type(&param.ty) {
            return false;
        }
    }
    // Check body for unsupported operations
    is_compilable_body(&func.body)
}

fn is_compilable_type(ty: &ast::Type) -> bool {
    match ty {
        ast::Type::Int
        | ast::Type::Float
        | ast::Type::Bool
        | ast::Type::Byte
        | ast::Type::String => true,
        ast::Type::Struct(_) => true,
        ast::Type::Result(inner) => is_compilable_type(inner),
        _ => false,
    }
}

fn is_compilable_body(stmts: &[ast::Statement]) -> bool {
    stmts.iter().all(|s| match &s.kind {
        ast::StatementKind::Let { ty, value, .. } => {
            is_compilable_type(ty) && is_compilable_expr(value)
        }
        ast::StatementKind::Set { value, .. } => is_compilable_expr(value),
        ast::StatementKind::Call { call, binding, .. } => {
            call.args.iter().all(is_compilable_expr)
                && binding.as_ref().is_none_or(|b| is_compilable_type(&b.ty))
        }
        ast::StatementKind::Check { condition, .. } => is_compilable_expr(condition),
        ast::StatementKind::Return { value } => is_compilable_expr(value),
        ast::StatementKind::Branch {
            condition,
            then_body,
            else_body,
        } => {
            is_compilable_expr(condition)
                && is_compilable_body(then_body)
                && is_compilable_body(else_body)
        }
        ast::StatementKind::While { condition, body } => {
            is_compilable_expr(condition) && is_compilable_body(body)
        }
        ast::StatementKind::Match { expr, arms } => {
            is_compilable_expr(expr) && arms.iter().all(|arm| is_compilable_body(&arm.body))
        }
        _ => false,
    })
}

fn is_compilable_expr(expr: &ast::Expr) -> bool {
    match expr {
        ast::Expr::Literal(_) => true,
        ast::Expr::Identifier(_) => true,
        ast::Expr::FieldAccess { base, .. } => is_compilable_expr(base),
        ast::Expr::StructInit { fields, .. } => fields.iter().all(|f| is_compilable_expr(&f.value)),
        ast::Expr::Binary { left, right, .. } => {
            is_compilable_expr(left) && is_compilable_expr(right)
        }
        ast::Expr::Unary { expr, .. } => is_compilable_expr(expr),
        ast::Expr::Call(call) => call.args.iter().all(is_compilable_expr),
    }
}

/// Check if an entire program can be fully compiled.
pub fn is_fully_compilable(program: &ast::Program) -> bool {
    program.functions.iter().all(|f| is_compilable_function(f))
}

// === String runtime functions (called from JIT code) ===
// Strings are (ptr: *const u8, len: i64) pairs.
// These functions are imported into the JIT module as symbols.

/// Allocate and return a new string from concatenating two strings.
/// Returns (ptr, len) packed as two i64 values via out-pointer.
extern "C" fn rt_string_concat(
    a_ptr: *const u8,
    a_len: i64,
    b_ptr: *const u8,
    b_len: i64,
    out_ptr: *mut i64, // writes [ptr, len] here
) {
    unsafe {
        let a = std::slice::from_raw_parts(a_ptr, a_len as usize);
        let b = std::slice::from_raw_parts(b_ptr, b_len as usize);
        let mut result = Vec::with_capacity(a.len() + b.len());
        result.extend_from_slice(a);
        result.extend_from_slice(b);
        let boxed = result.into_boxed_slice();
        let ptr = Box::into_raw(boxed);
        *out_ptr = (*ptr).as_ptr() as i64;
        *out_ptr.add(1) = (a_len + b_len) as i64;
    }
}

/// Compare two strings for equality. Returns 1 if equal, 0 if not.
extern "C" fn rt_string_eq(a_ptr: *const u8, a_len: i64, b_ptr: *const u8, b_len: i64) -> i64 {
    if a_len != b_len {
        return 0;
    }
    unsafe {
        let a = std::slice::from_raw_parts(a_ptr, a_len as usize);
        let b = std::slice::from_raw_parts(b_ptr, b_len as usize);
        if a == b {
            1
        } else {
            0
        }
    }
}

/// Return the length of a string.
extern "C" fn rt_string_len(_ptr: *const u8, len: i64) -> i64 {
    len
}

/// Compile an Adapsis program to native code via Cranelift JIT.
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

    // Register string runtime functions
    builder.symbol("rt_string_concat", rt_string_concat as *const u8);
    builder.symbol("rt_string_eq", rt_string_eq as *const u8);
    builder.symbol("rt_string_len", rt_string_len as *const u8);

    let mut module = JITModule::new(builder);

    let mut compiled_functions: HashMap<String, FuncId> = HashMap::new();

    // Declare string runtime functions
    let mut rt_concat_sig = module.make_signature();
    rt_concat_sig.params.push(AbiParam::new(types::I64)); // a_ptr
    rt_concat_sig.params.push(AbiParam::new(types::I64)); // a_len
    rt_concat_sig.params.push(AbiParam::new(types::I64)); // b_ptr
    rt_concat_sig.params.push(AbiParam::new(types::I64)); // b_len
    rt_concat_sig.params.push(AbiParam::new(types::I64)); // out_ptr
    let rt_concat_id =
        module.declare_function("rt_string_concat", Linkage::Import, &rt_concat_sig)?;

    let mut rt_eq_sig = module.make_signature();
    rt_eq_sig.params.push(AbiParam::new(types::I64)); // a_ptr
    rt_eq_sig.params.push(AbiParam::new(types::I64)); // a_len
    rt_eq_sig.params.push(AbiParam::new(types::I64)); // b_ptr
    rt_eq_sig.params.push(AbiParam::new(types::I64)); // b_len
    rt_eq_sig.returns.push(AbiParam::new(types::I64)); // 1 or 0
    let rt_eq_id = module.declare_function("rt_string_eq", Linkage::Import, &rt_eq_sig)?;

    let runtime_funcs = RuntimeFuncs {
        concat_id: rt_concat_id,
        eq_id: rt_eq_id,
    };

    // First pass: declare all functions (so they can call each other)
    let mut signatures: HashMap<String, Signature> = HashMap::new();
    for func in &program.functions {
        let sig = build_signature(&mut module, func, program)?;
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

        compile_function(
            &mut module,
            &mut ctx,
            func,
            &compiled_functions,
            program,
            &runtime_funcs,
        )?;

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

/// Build a Cranelift signature from an Adapsis function declaration.
fn build_signature(
    module: &mut JITModule,
    func: &ast::FunctionDecl,
    program: &ast::Program,
) -> Result<Signature> {
    let mut sig = module.make_signature();

    for param in &func.params {
        for cl_type in flatten_type(&param.ty, program) {
            sig.params.push(AbiParam::new(cl_type));
        }
    }

    for cl_type in flatten_type(&func.return_type, program) {
        sig.returns.push(AbiParam::new(cl_type));
    }

    Ok(sig)
}

/// Map Adapsis types to Cranelift types.
/// String is represented as two i64 values (ptr, len).
fn adapsis_type_to_cranelift(ty: &ast::Type) -> Result<types::Type> {
    match ty {
        ast::Type::Int => Ok(types::I64),
        ast::Type::Float => Ok(types::F64),
        ast::Type::Bool => Ok(types::I8),
        ast::Type::Byte => Ok(types::I8),
        ast::Type::String => Ok(types::I64), // ptr half — len is a second value
        ast::Type::Result(inner) => adapsis_type_to_cranelift(inner),
        _ => bail!("type {:?} not yet supported in compiler", ty),
    }
}

/// Check if a type is a string (needs two values: ptr + len).
fn is_string_type(ty: &ast::Type) -> bool {
    matches!(ty, ast::Type::String)
        || matches!(ty, ast::Type::Result(inner) if matches!(inner.as_ref(), ast::Type::String))
}

/// Compile a single function body to Cranelift IR.
fn compile_function(
    module: &mut JITModule,
    ctx: &mut codegen::Context,
    func: &ast::FunctionDecl,
    all_functions: &HashMap<String, FuncId>,
    program: &ast::Program,
    runtime_funcs: &RuntimeFuncs,
) -> Result<()> {
    let mut builder_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);

    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    // Map parameter names to SSA values
    // Structs and strings expand to multiple block params
    let mut vars: HashMap<String, Variable> = HashMap::new();
    let mut var_counter: u32 = 0;
    let mut block_param_idx: usize = 0;

    for param in &func.params {
        bind_param(
            &mut builder,
            entry_block,
            &mut vars,
            &mut var_counter,
            &mut block_param_idx,
            &param.name,
            &param.ty,
            program,
        );
    }

    let mut string_literals = Vec::new();
    let mut comp_ctx = CompilationContext {
        module,
        builder: &mut builder,
        vars: &mut vars,
        var_counter: &mut var_counter,
        all_functions,
        runtime_funcs,
        program,
        return_type: &func.return_type,
        terminated: false,
        string_literals: &mut string_literals,
    };

    compile_body(&mut comp_ctx, &func.body)?;

    // If we get here without a return, emit a default return
    if !comp_ctx.terminated {
        if is_string_type(&func.return_type) {
            let zero = comp_ctx.builder.ins().iconst(types::I64, 0);
            comp_ctx.builder.ins().return_(&[zero, zero]);
        } else {
            let ret_type = adapsis_type_to_cranelift(&func.return_type)?;
            let default_val = if ret_type == types::F64 {
                comp_ctx.builder.ins().f64const(0.0)
            } else {
                comp_ctx.builder.ins().iconst(ret_type, 0)
            };
            comp_ctx.builder.ins().return_(&[default_val]);
        }
    }

    builder.finalize();
    Ok(())
}

struct RuntimeFuncs {
    concat_id: FuncId,
    eq_id: FuncId,
}

/// Compute the flattened Cranelift types for an Adapsis type.
/// Structs expand to their fields, strings to (ptr, len).
fn flatten_type(ty: &ast::Type, program: &ast::Program) -> Vec<types::Type> {
    match ty {
        ast::Type::Int => vec![types::I64],
        ast::Type::Float => vec![types::F64],
        ast::Type::Bool => vec![types::I8],
        ast::Type::Byte => vec![types::I8],
        ast::Type::String => vec![types::I64, types::I64], // ptr, len
        ast::Type::Result(inner) => flatten_type(inner, program),
        ast::Type::Struct(name) => {
            // Look up struct fields and flatten each
            if let Some(fields) = get_struct_field_types(program, name) {
                fields
                    .iter()
                    .flat_map(|(_, ty)| flatten_type(ty, program))
                    .collect()
            } else {
                vec![types::I64] // unknown struct — treat as opaque pointer
            }
        }
        _ => vec![types::I64], // fallback
    }
}

/// Get field names and types for a struct from the program.
fn get_struct_field_types(program: &ast::Program, name: &str) -> Option<Vec<(String, ast::Type)>> {
    for td in &program.types {
        if let ast::TypeDecl::Struct(s) = td {
            if s.name == name {
                return Some(
                    s.fields
                        .iter()
                        .map(|f| (f.name.clone(), f.ty.clone()))
                        .collect(),
                );
            }
        }
    }
    for module in &program.modules {
        for td in &module.types {
            if let ast::TypeDecl::Struct(s) = td {
                if s.name == name {
                    return Some(
                        s.fields
                            .iter()
                            .map(|f| (f.name.clone(), f.ty.clone()))
                            .collect(),
                    );
                }
            }
        }
    }
    None
}

struct CompilationContext<'a, 'b> {
    module: &'a mut JITModule,
    builder: &'b mut FunctionBuilder<'a>,
    vars: &'b mut HashMap<String, Variable>,
    var_counter: &'b mut u32,
    all_functions: &'b HashMap<String, FuncId>,
    runtime_funcs: &'b RuntimeFuncs,
    #[allow(dead_code)]
    program: &'b ast::Program,
    return_type: &'b ast::Type,
    terminated: bool,
    string_literals: &'b mut Vec<(String, cranelift_module::DataId)>,
}

/// Bind a function parameter to variables, handling struct/string flattening.
fn bind_param(
    builder: &mut FunctionBuilder,
    block: cranelift::prelude::Block,
    vars: &mut HashMap<String, Variable>,
    var_counter: &mut u32,
    block_param_idx: &mut usize,
    name: &str,
    ty: &ast::Type,
    program: &ast::Program,
) {
    match ty {
        ast::Type::String => {
            let ptr_val = builder.block_params(block)[*block_param_idx];
            let len_val = builder.block_params(block)[*block_param_idx + 1];
            *block_param_idx += 2;

            let ptr_var = Variable::new(*var_counter as usize);
            *var_counter += 1;
            builder.declare_var(ptr_var, types::I64);
            builder.def_var(ptr_var, ptr_val);
            vars.insert(format!("{name}_ptr"), ptr_var);

            let len_var = Variable::new(*var_counter as usize);
            *var_counter += 1;
            builder.declare_var(len_var, types::I64);
            builder.def_var(len_var, len_val);
            vars.insert(format!("{name}_len"), len_var);

            vars.insert(name.to_string(), ptr_var);
        }
        ast::Type::Struct(type_name) => {
            if let Some(fields) = get_struct_field_types(program, type_name) {
                for (field_name, field_ty) in &fields {
                    let qualified = format!("{name}.{field_name}");
                    bind_param(
                        builder,
                        block,
                        vars,
                        var_counter,
                        block_param_idx,
                        &qualified,
                        field_ty,
                        program,
                    );
                }
            } else {
                // Unknown struct — treat as single i64
                let val = builder.block_params(block)[*block_param_idx];
                *block_param_idx += 1;
                let var = Variable::new(*var_counter as usize);
                *var_counter += 1;
                builder.declare_var(var, types::I64);
                builder.def_var(var, val);
                vars.insert(name.to_string(), var);
            }
        }
        _ => {
            let val = builder.block_params(block)[*block_param_idx];
            *block_param_idx += 1;
            let cl_type = flatten_type(ty, program)[0];
            let var = Variable::new(*var_counter as usize);
            *var_counter += 1;
            builder.declare_var(var, cl_type);
            builder.def_var(var, val);
            vars.insert(name.to_string(), var);
        }
    }
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
            let cl_type = adapsis_type_to_cranelift(ty)?;
            let val = compile_expr(ctx, value, cl_type)?;
            let var = alloc_var(ctx, name, cl_type);
            ctx.builder.def_var(var, val);
            Ok(())
        }

        ast::StatementKind::Call { binding, call } => {
            let ret_type = if let Some(b) = binding {
                adapsis_type_to_cranelift(&b.ty)?
            } else {
                types::I64
            };
            let val = compile_call(ctx, call, ret_type)?;
            if let Some(b) = binding {
                let cl_type = adapsis_type_to_cranelift(&b.ty)?;
                let var = alloc_var(ctx, &b.name, cl_type);
                ctx.builder.def_var(var, val);
            }
            Ok(())
        }

        ast::StatementKind::Check {
            condition,
            on_fail: _,
            ..
        } => {
            // Compile condition, trap if false
            let cond = compile_expr(ctx, condition, types::I8)?;
            // Branch: if cond is true, continue; if false, trap
            let trap_block = ctx.builder.create_block();
            let ok_block = ctx.builder.create_block();

            ctx.builder.ins().brif(cond, ok_block, &[], trap_block, &[]);

            ctx.builder.switch_to_block(trap_block);
            ctx.builder.seal_block(trap_block);
            ctx.builder.ins().trap(TrapCode::user(1).unwrap());

            ctx.builder.switch_to_block(ok_block);
            ctx.builder.seal_block(ok_block);
            Ok(())
        }

        ast::StatementKind::Return { value } => {
            if is_string_type(ctx.return_type) {
                let ptr = compile_expr(ctx, value, types::I64)?;
                let len = if let Some(len_var) = ctx.vars.get("_last_str_len") {
                    ctx.builder.use_var(*len_var)
                } else {
                    ctx.builder.ins().iconst(types::I64, 0)
                };
                ctx.builder.ins().return_(&[ptr, len]);
            } else {
                let ret_cl_type = adapsis_type_to_cranelift(ctx.return_type)?;
                let val = compile_expr(ctx, value, ret_cl_type)?;
                ctx.builder.ins().return_(&[val]);
            }
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
            let _saved = ctx.terminated;
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

        ast::StatementKind::Set { name, value } => {
            let var = ctx
                .vars
                .get(name)
                .ok_or_else(|| anyhow!("undefined variable `{name}` in compiler (use +let first)"))?
                .clone();
            let val = compile_expr(ctx, value, types::I64)?;
            ctx.builder.def_var(var, val);
            Ok(())
        }

        ast::StatementKind::While { condition, body } => {
            let header_block = ctx.builder.create_block();
            let body_block = ctx.builder.create_block();
            let exit_block = ctx.builder.create_block();

            // Jump from current block into the loop header
            ctx.builder.ins().jump(header_block, &[]);

            // --- Header: evaluate condition, branch to body or exit ---
            ctx.builder.switch_to_block(header_block);
            let cond = compile_expr(ctx, condition, types::I8)?;
            ctx.builder
                .ins()
                .brif(cond, body_block, &[], exit_block, &[]);

            // --- Body: execute statements, then jump back to header ---
            ctx.builder.switch_to_block(body_block);
            ctx.builder.seal_block(body_block);
            let saved_terminated = ctx.terminated;
            ctx.terminated = false;
            compile_body(ctx, body)?;
            if !ctx.terminated {
                ctx.builder.ins().jump(header_block, &[]);
            }

            // Now seal the header — it has two predecessors (entry jump + back-edge)
            ctx.builder.seal_block(header_block);

            // --- Exit block ---
            ctx.builder.switch_to_block(exit_block);
            ctx.builder.seal_block(exit_block);
            ctx.terminated = saved_terminated;
            Ok(())
        }

        ast::StatementKind::Each { .. } => {
            bail!("each loops not yet supported in compiler")
        }

        ast::StatementKind::Await { .. } => {
            bail!("await not yet supported in compiler")
        }
        ast::StatementKind::Spawn { .. } => {
            bail!("spawn not yet supported in compiler")
        }
        ast::StatementKind::Match { expr, arms } => {
            // Compile match as a chain of conditional branches.
            // The matched expression value is compared against each arm's variant.
            // For integer matching: variant name is parsed as an i64 literal.
            // For wildcard "_": always matches (default/else).
            let matched_val = compile_expr(ctx, expr, types::I64)?;
            let merge_block = ctx.builder.create_block();

            let mut all_terminated = true;

            for (i, arm) in arms.iter().enumerate() {
                let is_last = i == arms.len() - 1;

                if arm.variant == "_" {
                    // Wildcard — unconditional match
                    // If there are bindings, bind the whole value
                    if arm.bindings.len() == 1 && arm.bindings[0] != "_" {
                        let cl_type = types::I64;
                        let var = alloc_var(ctx, &arm.bindings[0], cl_type);
                        ctx.builder.def_var(var, matched_val);
                    }
                    let saved = ctx.terminated;
                    ctx.terminated = false;
                    compile_body(ctx, &arm.body)?;
                    let arm_terminated = ctx.terminated;
                    if !arm_terminated {
                        ctx.builder.ins().jump(merge_block, &[]);
                    }
                    all_terminated = all_terminated && arm_terminated;
                    ctx.terminated = saved;
                    break; // wildcard is always last effective arm
                }

                // Try to parse variant as an integer literal for comparison
                let arm_block = ctx.builder.create_block();
                let next_block = if is_last {
                    merge_block
                } else {
                    ctx.builder.create_block()
                };

                // Compare matched value with arm's variant (as integer tag or literal)
                let arm_val = if let Ok(n) = arm.variant.parse::<i64>() {
                    ctx.builder.ins().iconst(types::I64, n)
                } else if arm.variant == "true" {
                    ctx.builder.ins().iconst(types::I64, 1)
                } else if arm.variant == "false" {
                    ctx.builder.ins().iconst(types::I64, 0)
                } else {
                    // Named variant — use its index as the tag
                    ctx.builder.ins().iconst(types::I64, i as i64)
                };

                let cond = ctx.builder.ins().icmp(IntCC::Equal, matched_val, arm_val);
                ctx.builder
                    .ins()
                    .brif(cond, arm_block, &[], next_block, &[]);

                // Compile arm body
                ctx.builder.switch_to_block(arm_block);
                ctx.builder.seal_block(arm_block);
                let saved = ctx.terminated;
                ctx.terminated = false;
                compile_body(ctx, &arm.body)?;
                let arm_terminated = ctx.terminated;
                if !arm_terminated {
                    ctx.builder.ins().jump(merge_block, &[]);
                }
                all_terminated = all_terminated && arm_terminated;
                ctx.terminated = saved;

                // Switch to next arm's test block
                if !is_last {
                    ctx.builder.switch_to_block(next_block);
                    ctx.builder.seal_block(next_block);
                }
            }

            ctx.terminated = all_terminated;
            ctx.builder.switch_to_block(merge_block);
            ctx.builder.seal_block(merge_block);
            Ok(())
        }
        ast::StatementKind::Yield { .. } => {
            bail!("yield not yet supported in compiler")
        }
        ast::StatementKind::SourceAdd { .. }
        | ast::StatementKind::SourceRemove { .. }
        | ast::StatementKind::SourceReplace { .. }
        | ast::StatementKind::EventRegister { .. }
        | ast::StatementKind::EventEmit { .. } => {
            bail!("source/event statements not supported in compiler")
        }
    }
}

fn compile_call(
    ctx: &mut CompilationContext,
    call: &ast::CallExpr,
    _expected_type: types::Type,
) -> Result<Value> {
    // Handle builtin functions
    match call.callee.as_str() {
        "concat" => {
            if call.args.len() != 2 {
                bail!("concat() expects 2 arguments in compiler");
            }
            // Compile both string args → (ptr, len) each
            let a_ptr = compile_expr(ctx, &call.args[0], types::I64)?;
            let a_len = if let Some(v) = ctx.vars.get("_last_str_len") {
                ctx.builder.use_var(*v)
            } else {
                ctx.builder.ins().iconst(types::I64, 0)
            };

            let b_ptr = compile_expr(ctx, &call.args[1], types::I64)?;
            let b_len = if let Some(v) = ctx.vars.get("_last_str_len") {
                ctx.builder.use_var(*v)
            } else {
                ctx.builder.ins().iconst(types::I64, 0)
            };

            // Allocate stack space for the result (2 x i64: ptr, len)
            let slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                16,
                3, // align to 8 bytes
            ));
            let out_ptr = ctx.builder.ins().stack_addr(types::I64, slot, 0);

            // Call rt_string_concat(a_ptr, a_len, b_ptr, b_len, out_ptr)
            let concat_func = ctx
                .module
                .declare_func_in_func(ctx.runtime_funcs.concat_id, ctx.builder.func);
            ctx.builder
                .ins()
                .call(concat_func, &[a_ptr, a_len, b_ptr, b_len, out_ptr]);

            // Read result ptr and len from the stack slot
            let result_ptr = ctx
                .builder
                .ins()
                .load(types::I64, MemFlags::new(), out_ptr, 0);
            let result_len = ctx
                .builder
                .ins()
                .load(types::I64, MemFlags::new(), out_ptr, 8);

            // Store len for the next expression that needs it
            let len_var = Variable::new(*ctx.var_counter as usize);
            *ctx.var_counter += 1;
            ctx.builder.declare_var(len_var, types::I64);
            ctx.builder.def_var(len_var, result_len);
            ctx.vars.insert("_last_str_len".to_string(), len_var);

            return Ok(result_ptr);
        }
        _ => {}
    }

    // Regular function call
    let func_id = ctx
        .all_functions
        .get(&call.callee)
        .ok_or_else(|| anyhow!("undefined function `{}` in compiled code", call.callee))?;

    let local_callee = ctx.module.declare_func_in_func(*func_id, ctx.builder.func);

    let mut arg_vals = Vec::new();
    for arg in &call.args {
        let val = compile_expr(ctx, arg, types::I64)?;
        // If the arg was a string, also pass the len
        if is_string_ast_expr(arg) {
            arg_vals.push(val);
            if let Some(v) = ctx.vars.get("_last_str_len") {
                arg_vals.push(ctx.builder.use_var(*v));
            }
        } else {
            arg_vals.push(val);
        }
    }

    let inst = ctx.builder.ins().call(local_callee, &arg_vals);
    let results: Vec<Value> = ctx.builder.inst_results(inst).to_vec();

    // If the called function returns a string (2 values), store the len
    if results.len() == 2 {
        let len_var = Variable::new(*ctx.var_counter as usize);
        *ctx.var_counter += 1;
        ctx.builder.declare_var(len_var, types::I64);
        ctx.builder.def_var(len_var, results[1]);
        ctx.vars.insert("_last_str_len".to_string(), len_var);
    }

    Ok(results[0])
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
            ast::Literal::String(s) => {
                // Create a data section for the string bytes
                let data_id = ctx.module.declare_anonymous_data(false, false)?;
                let mut data_desc = DataDescription::new();
                data_desc.define(s.as_bytes().to_vec().into_boxed_slice());
                ctx.module.define_data(data_id, &data_desc)?;
                ctx.string_literals.push((s.clone(), data_id));

                // Get a pointer to the data
                let gv = ctx.module.declare_data_in_func(data_id, ctx.builder.func);
                let ptr = ctx.builder.ins().global_value(types::I64, gv);

                // Store the length in a temp variable so it can be retrieved
                let len = ctx.builder.ins().iconst(types::I64, s.len() as i64);
                let len_var = Variable::new(*ctx.var_counter as usize);
                *ctx.var_counter += 1;
                ctx.builder.declare_var(len_var, types::I64);
                ctx.builder.def_var(len_var, len);
                // Use a convention: _last_str_len holds the len of the last string expr
                ctx.vars.insert("_last_str_len".to_string(), len_var);

                Ok(ptr)
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
            // Check for string comparison
            let is_str = is_string_ast_expr(left) || is_string_ast_expr(right);
            if is_str && matches!(op, ast::BinaryOp::Equal | ast::BinaryOp::NotEqual) {
                // String comparison via runtime function
                let lhs_ptr = compile_expr(ctx, left, types::I64)?;
                let lhs_len = if let Some(v) = ctx.vars.get("_last_str_len") {
                    ctx.builder.use_var(*v)
                } else {
                    ctx.builder.ins().iconst(types::I64, 0)
                };

                let rhs_ptr = compile_expr(ctx, right, types::I64)?;
                let rhs_len = if let Some(v) = ctx.vars.get("_last_str_len") {
                    ctx.builder.use_var(*v)
                } else {
                    ctx.builder.ins().iconst(types::I64, 0)
                };

                let eq_func = ctx
                    .module
                    .declare_func_in_func(ctx.runtime_funcs.eq_id, ctx.builder.func);
                let inst = ctx
                    .builder
                    .ins()
                    .call(eq_func, &[lhs_ptr, lhs_len, rhs_ptr, rhs_len]);
                let result = ctx.builder.inst_results(inst)[0];

                if matches!(op, ast::BinaryOp::NotEqual) {
                    let one = ctx.builder.ins().iconst(types::I64, 1);
                    return Ok(ctx.builder.ins().bxor(result, one));
                }
                return Ok(result);
            }

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

        ast::Expr::FieldAccess { base, field } => {
            // Resolve to a flattened variable name like "req.name" or "req.name_ptr"
            let base_name = expr_to_var_name(base);

            // Try qualified name first: base.field
            let qualified = format!("{base_name}.{field}");
            if let Some(var) = ctx.vars.get(&qualified) {
                return Ok(ctx.builder.use_var(*var));
            }

            // For strings: base.field_ptr
            let ptr_name = format!("{qualified}_ptr");
            let ptr_var = ctx.vars.get(&ptr_name).copied();
            let len_name = format!("{qualified}_len");
            let len_var = ctx.vars.get(&len_name).copied();

            if let Some(pv) = ptr_var {
                if let Some(lv) = len_var {
                    let len_val = ctx.builder.use_var(lv);
                    let tmp = Variable::new(*ctx.var_counter as usize);
                    *ctx.var_counter += 1;
                    ctx.builder.declare_var(tmp, types::I64);
                    ctx.builder.def_var(tmp, len_val);
                    ctx.vars.insert("_last_str_len".to_string(), tmp);
                }
                return Ok(ctx.builder.use_var(pv));
            }

            // .len on strings
            if field == "len" {
                let len_name = format!("{base_name}_len");
                if let Some(var) = ctx.vars.get(&len_name) {
                    return Ok(ctx.builder.use_var(*var));
                }
            }

            bail!("cannot resolve field access `{base_name}.{field}` in compiler")
        }

        ast::Expr::StructInit { ty: _, fields } => {
            // Struct construction — compile each field value and store in variables
            // The "result" is just the first field's value; the caller reads all fields
            // by their variable names
            if fields.is_empty() {
                return Ok(ctx.builder.ins().iconst(types::I64, 0));
            }
            let mut first_val = None;
            for field in fields {
                let val = compile_expr(ctx, &field.value, types::I64)?;
                if first_val.is_none() {
                    first_val = Some(val);
                }
                // Store each field value so it can be accessed later
                let var = Variable::new(*ctx.var_counter as usize);
                *ctx.var_counter += 1;
                ctx.builder.declare_var(var, types::I64);
                ctx.builder.def_var(var, val);
                ctx.vars
                    .insert(format!("_struct_field_{}", field.name), var);
            }
            return Ok(first_val.unwrap_or_else(|| ctx.builder.ins().iconst(types::I64, 0)));
        }
    }
}

/// Convert an expression to a variable name for field access resolution.
fn expr_to_var_name(expr: &ast::Expr) -> String {
    match expr {
        ast::Expr::Identifier(name) => name.clone(),
        ast::Expr::FieldAccess { base, field } => {
            format!("{}.{field}", expr_to_var_name(base))
        }
        _ => format!("{expr:?}"),
    }
}

/// Heuristic: check if an expression is a string.
fn is_string_ast_expr(expr: &ast::Expr) -> bool {
    match expr {
        ast::Expr::Literal(ast::Literal::String(_)) => true,
        ast::Expr::Binary { left, right, .. } => {
            is_string_ast_expr(left) || is_string_ast_expr(right)
        }
        _ => false,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn make_param(name: &str, ty: ast::Type) -> ast::Param {
        ast::Param {
            id: String::new(),
            name: name.to_string(),
            ty,
        }
    }

    fn make_stmt(kind: ast::StatementKind) -> ast::Statement {
        ast::Statement {
            id: "s1".to_string(),
            kind,
        }
    }

    fn make_fn(
        name: &str,
        params: Vec<ast::Param>,
        ret: ast::Type,
        effects: Vec<ast::Effect>,
        body: Vec<ast::Statement>,
    ) -> ast::FunctionDecl {
        ast::FunctionDecl {
            id: String::new(),
            name: name.to_string(),
            params,
            return_type: ret,
            effects,
            body,
            tests: vec![],
        }
    }

    /// Build a simple "return a + b" body.
    fn add_body() -> Vec<ast::Statement> {
        vec![make_stmt(ast::StatementKind::Return {
            value: ast::Expr::Binary {
                left: Box::new(ast::Expr::Identifier("a".to_string())),
                op: ast::BinaryOp::Add,
                right: Box::new(ast::Expr::Identifier("b".to_string())),
            },
        })]
    }

    // ═════════════════════════════════════════════════════════════════════
    // 1. is_compilable_function
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn compilable_int_to_int() {
        let func = make_fn(
            "add",
            vec![
                make_param("a", ast::Type::Int),
                make_param("b", ast::Type::Int),
            ],
            ast::Type::Int,
            vec![],
            add_body(),
        );
        assert!(is_compilable_function(&func));
    }

    #[test]
    fn compilable_float_params() {
        let func = make_fn(
            "add_f",
            vec![
                make_param("a", ast::Type::Float),
                make_param("b", ast::Type::Float),
            ],
            ast::Type::Float,
            vec![],
            vec![make_stmt(ast::StatementKind::Return {
                value: ast::Expr::Binary {
                    left: Box::new(ast::Expr::Identifier("a".to_string())),
                    op: ast::BinaryOp::Add,
                    right: Box::new(ast::Expr::Identifier("b".to_string())),
                },
            })],
        );
        assert!(is_compilable_function(&func));
    }

    #[test]
    fn compilable_bool_return() {
        let func = make_fn(
            "is_pos",
            vec![make_param("x", ast::Type::Int)],
            ast::Type::Bool,
            vec![],
            vec![make_stmt(ast::StatementKind::Return {
                value: ast::Expr::Binary {
                    left: Box::new(ast::Expr::Identifier("x".to_string())),
                    op: ast::BinaryOp::GreaterThan,
                    right: Box::new(ast::Expr::Literal(ast::Literal::Int(0))),
                },
            })],
        );
        assert!(is_compilable_function(&func));
    }

    #[test]
    fn compilable_string_param() {
        // String params are compilable (compiled as ptr+len pair)
        let func = make_fn(
            "identity",
            vec![make_param("s", ast::Type::String)],
            ast::Type::String,
            vec![],
            vec![make_stmt(ast::StatementKind::Return {
                value: ast::Expr::Identifier("s".to_string()),
            })],
        );
        assert!(is_compilable_function(&func));
    }

    #[test]
    fn not_compilable_list_param() {
        let func = make_fn(
            "sum",
            vec![make_param(
                "items",
                ast::Type::List(Box::new(ast::Type::Int)),
            )],
            ast::Type::Int,
            vec![],
            vec![make_stmt(ast::StatementKind::Return {
                value: ast::Expr::Literal(ast::Literal::Int(0)),
            })],
        );
        assert!(!is_compilable_function(&func));
    }

    #[test]
    fn not_compilable_map_return() {
        let func = make_fn(
            "make_map",
            vec![],
            ast::Type::Map(Box::new(ast::Type::String), Box::new(ast::Type::Int)),
            vec![],
            vec![],
        );
        assert!(!is_compilable_function(&func));
    }

    #[test]
    fn not_compilable_option_param() {
        let func = make_fn(
            "unwrap",
            vec![make_param("x", ast::Type::Option(Box::new(ast::Type::Int)))],
            ast::Type::Int,
            vec![],
            vec![make_stmt(ast::StatementKind::Return {
                value: ast::Expr::Literal(ast::Literal::Int(0)),
            })],
        );
        assert!(!is_compilable_function(&func));
    }

    #[test]
    fn compilable_result_return() {
        // Result<Int> is compilable (unwraps to inner type)
        let func = make_fn(
            "validate",
            vec![make_param("x", ast::Type::Int)],
            ast::Type::Result(Box::new(ast::Type::Int)),
            vec![ast::Effect::Fail],
            vec![make_stmt(ast::StatementKind::Return {
                value: ast::Expr::Identifier("x".to_string()),
            })],
        );
        assert!(is_compilable_function(&func));
    }

    #[test]
    fn not_compilable_each_in_body() {
        // Each loops are not compilable
        let func = make_fn(
            "sum",
            vec![make_param("x", ast::Type::Int)],
            ast::Type::Int,
            vec![],
            vec![make_stmt(ast::StatementKind::Each {
                iterator: ast::Expr::Identifier("items".to_string()),
                binding: ast::Binding {
                    name: "item".to_string(),
                    ty: ast::Type::Int,
                },
                body: vec![],
            })],
        );
        assert!(!is_compilable_function(&func));
    }

    #[test]
    fn not_compilable_await_in_body() {
        let func = make_fn(
            "fetch",
            vec![],
            ast::Type::String,
            vec![ast::Effect::Io, ast::Effect::Async],
            vec![make_stmt(ast::StatementKind::Await {
                name: "data".to_string(),
                ty: ast::Type::String,
                call: ast::CallExpr {
                    callee: "http_get".to_string(),
                    args: vec![],
                },
            })],
        );
        assert!(!is_compilable_function(&func));
    }

    // ═════════════════════════════════════════════════════════════════════
    // 2. is_compilable_type
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn compilable_types() {
        assert!(is_compilable_type(&ast::Type::Int));
        assert!(is_compilable_type(&ast::Type::Float));
        assert!(is_compilable_type(&ast::Type::Bool));
        assert!(is_compilable_type(&ast::Type::Byte));
        assert!(is_compilable_type(&ast::Type::String));
        assert!(is_compilable_type(&ast::Type::Struct("Point".to_string())));
        assert!(is_compilable_type(&ast::Type::Result(Box::new(
            ast::Type::Int
        ))));
    }

    #[test]
    fn not_compilable_types() {
        assert!(!is_compilable_type(&ast::Type::List(Box::new(
            ast::Type::Int
        ))));
        assert!(!is_compilable_type(&ast::Type::Map(
            Box::new(ast::Type::String),
            Box::new(ast::Type::Int)
        )));
        assert!(!is_compilable_type(&ast::Type::Option(Box::new(
            ast::Type::Int
        ))));
        assert!(!is_compilable_type(&ast::Type::Set(Box::new(
            ast::Type::Int
        ))));
        assert!(!is_compilable_type(&ast::Type::TaggedUnion(
            "Color".to_string()
        )));
    }

    // ═════════════════════════════════════════════════════════════════════
    // 3. is_compilable_body / is_compilable_expr
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn compilable_expr_literal() {
        assert!(is_compilable_expr(&ast::Expr::Literal(ast::Literal::Int(
            42
        ))));
        assert!(is_compilable_expr(&ast::Expr::Literal(
            ast::Literal::Float(3.14)
        )));
        assert!(is_compilable_expr(&ast::Expr::Literal(ast::Literal::Bool(
            true
        ))));
        assert!(is_compilable_expr(&ast::Expr::Literal(
            ast::Literal::String("hello".to_string())
        )));
    }

    #[test]
    fn compilable_expr_identifier() {
        assert!(is_compilable_expr(&ast::Expr::Identifier("x".to_string())));
    }

    #[test]
    fn compilable_expr_binary() {
        let expr = ast::Expr::Binary {
            left: Box::new(ast::Expr::Identifier("a".to_string())),
            op: ast::BinaryOp::Mul,
            right: Box::new(ast::Expr::Literal(ast::Literal::Int(2))),
        };
        assert!(is_compilable_expr(&expr));
    }

    #[test]
    fn compilable_expr_nested_binary() {
        // (a + b) * (c - d)
        let expr = ast::Expr::Binary {
            left: Box::new(ast::Expr::Binary {
                left: Box::new(ast::Expr::Identifier("a".to_string())),
                op: ast::BinaryOp::Add,
                right: Box::new(ast::Expr::Identifier("b".to_string())),
            }),
            op: ast::BinaryOp::Mul,
            right: Box::new(ast::Expr::Binary {
                left: Box::new(ast::Expr::Identifier("c".to_string())),
                op: ast::BinaryOp::Sub,
                right: Box::new(ast::Expr::Identifier("d".to_string())),
            }),
        };
        assert!(is_compilable_expr(&expr));
    }

    #[test]
    fn compilable_expr_unary() {
        let expr = ast::Expr::Unary {
            op: ast::UnaryOp::Neg,
            expr: Box::new(ast::Expr::Identifier("x".to_string())),
        };
        assert!(is_compilable_expr(&expr));
    }

    #[test]
    fn compilable_expr_call() {
        let expr = ast::Expr::Call(ast::CallExpr {
            callee: "double".to_string(),
            args: vec![ast::Expr::Identifier("x".to_string())],
        });
        assert!(is_compilable_expr(&expr));
    }

    #[test]
    fn compilable_expr_field_access() {
        let expr = ast::Expr::FieldAccess {
            base: Box::new(ast::Expr::Identifier("point".to_string())),
            field: "x".to_string(),
        };
        assert!(is_compilable_expr(&expr));
    }

    #[test]
    fn compilable_body_let_return() {
        let body = vec![
            make_stmt(ast::StatementKind::Let {
                name: "result".to_string(),
                ty: ast::Type::Int,
                value: ast::Expr::Binary {
                    left: Box::new(ast::Expr::Identifier("x".to_string())),
                    op: ast::BinaryOp::Mul,
                    right: Box::new(ast::Expr::Literal(ast::Literal::Int(2))),
                },
            }),
            make_stmt(ast::StatementKind::Return {
                value: ast::Expr::Identifier("result".to_string()),
            }),
        ];
        assert!(is_compilable_body(&body));
    }

    #[test]
    fn compilable_body_with_branch() {
        let body = vec![make_stmt(ast::StatementKind::Branch {
            condition: ast::Expr::Binary {
                left: Box::new(ast::Expr::Identifier("x".to_string())),
                op: ast::BinaryOp::GreaterThan,
                right: Box::new(ast::Expr::Literal(ast::Literal::Int(0))),
            },
            then_body: vec![make_stmt(ast::StatementKind::Return {
                value: ast::Expr::Identifier("x".to_string()),
            })],
            else_body: vec![make_stmt(ast::StatementKind::Return {
                value: ast::Expr::Literal(ast::Literal::Int(0)),
            })],
        })];
        assert!(is_compilable_body(&body));
    }

    #[test]
    fn compilable_body_with_while() {
        let body = vec![make_stmt(ast::StatementKind::While {
            condition: ast::Expr::Binary {
                left: Box::new(ast::Expr::Identifier("i".to_string())),
                op: ast::BinaryOp::LessThan,
                right: Box::new(ast::Expr::Literal(ast::Literal::Int(10))),
            },
            body: vec![make_stmt(ast::StatementKind::Set {
                name: "i".to_string(),
                value: ast::Expr::Binary {
                    left: Box::new(ast::Expr::Identifier("i".to_string())),
                    op: ast::BinaryOp::Add,
                    right: Box::new(ast::Expr::Literal(ast::Literal::Int(1))),
                },
            })],
        })];
        assert!(is_compilable_body(&body));
    }

    #[test]
    fn not_compilable_body_with_spawn() {
        let body = vec![make_stmt(ast::StatementKind::Spawn {
            call: ast::CallExpr {
                callee: "bg".to_string(),
                args: vec![],
            },
            binding: None,
        })];
        assert!(!is_compilable_body(&body));
    }

    #[test]
    fn not_compilable_body_with_yield() {
        let body = vec![make_stmt(ast::StatementKind::Yield {
            value: ast::Expr::Literal(ast::Literal::Int(1)),
        })];
        assert!(!is_compilable_body(&body));
    }

    // ═════════════════════════════════════════════════════════════════════
    // 4. is_fully_compilable
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn fully_compilable_simple_program() {
        let mut program = ast::Program::default();
        program.functions.push(Arc::new(make_fn(
            "add",
            vec![
                make_param("a", ast::Type::Int),
                make_param("b", ast::Type::Int),
            ],
            ast::Type::Int,
            vec![],
            add_body(),
        )));
        assert!(is_fully_compilable(&program));
    }

    #[test]
    fn not_fully_compilable_with_list_function() {
        let mut program = ast::Program::default();
        program.functions.push(Arc::new(make_fn(
            "add",
            vec![
                make_param("a", ast::Type::Int),
                make_param("b", ast::Type::Int),
            ],
            ast::Type::Int,
            vec![],
            add_body(),
        )));
        program.functions.push(Arc::new(make_fn(
            "sum",
            vec![make_param(
                "items",
                ast::Type::List(Box::new(ast::Type::Int)),
            )],
            ast::Type::Int,
            vec![],
            vec![],
        )));
        assert!(!is_fully_compilable(&program));
    }

    #[test]
    fn fully_compilable_empty_program() {
        let program = ast::Program::default();
        assert!(is_fully_compilable(&program));
    }

    // ═════════════════════════════════════════════════════════════════════
    // 5. compile + call_i64 integration test
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn compile_and_call_add() {
        let mut program = ast::Program::default();
        program.functions.push(Arc::new(make_fn(
            "add",
            vec![
                make_param("a", ast::Type::Int),
                make_param("b", ast::Type::Int),
            ],
            ast::Type::Int,
            vec![],
            add_body(),
        )));

        let mut compiled = compile(&program).expect("compilation failed");
        let result = compiled.call_i64("add", &[3, 4]).expect("call failed");
        assert_eq!(result, 7);
    }

    #[test]
    fn compile_and_call_multiply() {
        let mut program = ast::Program::default();
        program.functions.push(Arc::new(make_fn(
            "mul",
            vec![
                make_param("a", ast::Type::Int),
                make_param("b", ast::Type::Int),
            ],
            ast::Type::Int,
            vec![],
            vec![make_stmt(ast::StatementKind::Return {
                value: ast::Expr::Binary {
                    left: Box::new(ast::Expr::Identifier("a".to_string())),
                    op: ast::BinaryOp::Mul,
                    right: Box::new(ast::Expr::Identifier("b".to_string())),
                },
            })],
        )));

        let mut compiled = compile(&program).expect("compilation failed");
        assert_eq!(compiled.call_i64("mul", &[5, 6]).unwrap(), 30);
    }

    #[test]
    fn compile_and_call_with_constant() {
        let mut program = ast::Program::default();
        program.functions.push(Arc::new(make_fn(
            "answer",
            vec![],
            ast::Type::Int,
            vec![],
            vec![make_stmt(ast::StatementKind::Return {
                value: ast::Expr::Literal(ast::Literal::Int(42)),
            })],
        )));

        let mut compiled = compile(&program).expect("compilation failed");
        assert_eq!(compiled.call_i64("answer", &[]).unwrap(), 42);
    }

    #[test]
    fn compile_call_missing_function() {
        let mut program = ast::Program::default();
        program.functions.push(Arc::new(make_fn(
            "add",
            vec![
                make_param("a", ast::Type::Int),
                make_param("b", ast::Type::Int),
            ],
            ast::Type::Int,
            vec![],
            add_body(),
        )));
        let mut compiled = compile(&program).unwrap();
        let result = compiled.call_i64("nonexistent", &[1, 2]);
        assert!(result.is_err());
    }

    // ═════════════════════════════════════════════════════════════════════
    // 6. adapsis_type_to_cranelift
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn type_mapping_int() {
        assert_eq!(
            adapsis_type_to_cranelift(&ast::Type::Int).unwrap(),
            types::I64
        );
    }

    #[test]
    fn type_mapping_float() {
        assert_eq!(
            adapsis_type_to_cranelift(&ast::Type::Float).unwrap(),
            types::F64
        );
    }

    #[test]
    fn type_mapping_bool() {
        assert_eq!(
            adapsis_type_to_cranelift(&ast::Type::Bool).unwrap(),
            types::I8
        );
    }

    #[test]
    fn type_mapping_byte() {
        assert_eq!(
            adapsis_type_to_cranelift(&ast::Type::Byte).unwrap(),
            types::I8
        );
    }

    #[test]
    fn type_mapping_string() {
        // String maps to I64 (ptr half)
        assert_eq!(
            adapsis_type_to_cranelift(&ast::Type::String).unwrap(),
            types::I64
        );
    }

    #[test]
    fn type_mapping_result_int() {
        // Result<Int> unwraps to the inner type
        assert_eq!(
            adapsis_type_to_cranelift(&ast::Type::Result(Box::new(ast::Type::Int))).unwrap(),
            types::I64
        );
    }

    #[test]
    fn type_mapping_list_unsupported() {
        assert!(adapsis_type_to_cranelift(&ast::Type::List(Box::new(ast::Type::Int))).is_err());
    }

    // ═════════════════════════════════════════════════════════════════════
    // 7. is_string_type
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn is_string_type_true() {
        assert!(is_string_type(&ast::Type::String));
        assert!(is_string_type(&ast::Type::Result(Box::new(
            ast::Type::String
        ))));
    }

    #[test]
    fn is_string_type_false() {
        assert!(!is_string_type(&ast::Type::Int));
        assert!(!is_string_type(&ast::Type::Float));
        assert!(!is_string_type(&ast::Type::Bool));
        assert!(!is_string_type(&ast::Type::Result(Box::new(
            ast::Type::Int
        ))));
    }
}
