//! OS spawn dispatch shared by the actual IO loop and synthetic pipeline tests.
use crate::{ast, coroutine, eval, ingress::InvocationOrigin, session};
use std::sync::Arc;

pub(crate) struct Context {
    pub io: Arc<coroutine::Runtime>,
    pub program: Arc<tokio::sync::RwLock<ast::Program>>,
    pub runtime: session::SharedRuntime,
    pub meta: session::SharedMeta,
}

pub(crate) fn dispatch(
    context: Context,
    function_name: String,
    args: Vec<eval::Value>,
    origin: Option<InvocationOrigin>,
    reply: tokio::sync::oneshot::Sender<anyhow::Result<coroutine::TaskId>>,
) {
    let id = context.io.next_task_id();
    let registry = context.io.task_registry.clone();
    let info = coroutine::TaskInfo {
        id,
        function_name: function_name.clone(),
        status: coroutine::WaitReason::Running,
        started_at: format!(
            "{}s",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        ),
    };
    let Ok(mut tasks) = registry.lock() else {
        let _ = reply.send(Err(anyhow::anyhow!("spawn registry unavailable")));
        return;
    };
    tasks.insert(id, info);
    drop(tasks);
    let _ = reply.send(Ok(id));
    let monitored = function_name.clone();
    let task = tokio::task::spawn_blocking(move || {
        let program = context.program.blocking_read().clone();
        let result = run(&context, &program, &function_name, args, origin, id);
        if let Ok(mut tasks) = context.io.task_registry.lock() {
            if let Some(info) = tasks.get_mut(&id) {
                info.status = match result {
                    Ok(value) => coroutine::WaitReason::Completed(format!("{value}")),
                    Err(error) => {
                        crate::logging::diagnostic("spawn-error", &error);
                        coroutine::WaitReason::Failed(error.to_string())
                    }
                };
            }
        }
    });
    tokio::spawn(crate::monitor_spawned_task(task, registry, id, monitored));
}

fn run(
    context: &Context,
    program: &ast::Program,
    function: &str,
    args: Vec<eval::Value>,
    origin: Option<InvocationOrigin>,
    id: i64,
) -> anyhow::Result<eval::Value> {
    let function_decl = program
        .get_function(function)
        .ok_or_else(|| anyhow::anyhow!("function not found"))?;
    let ctx = eval::EvalContext::new_minimal(
        context.runtime.clone(),
        context.meta.clone(),
        program,
        eval::make_shared_program_mut(program),
    );
    ctx.install();
    let handle = coroutine::CoroutineHandle::new_with_task(
        context.io.io_sender(),
        id,
        context.io.task_registry.clone(),
        context.io.snapshot_registry.clone(),
    )
    .with_origin(origin);
    let mut env = eval::Env::new_with_shared_interner(&program.shared_interner);
    env.populate_shared_from_program(program);
    env.set_shared_runtime(context.runtime.clone());
    eval::init_missing_shared_runtime_vars(program, &context.runtime);
    env.set("__coroutine_handle", eval::Value::CoroutineHandle(handle));
    for (param, value) in function_decl.params.iter().zip(args) {
        env.set(&param.name, value);
    }
    eval::eval_function_body_named(program, function, &function_decl.body, &mut env)
}
