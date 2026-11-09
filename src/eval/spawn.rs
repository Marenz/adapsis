//! Spawn statement dispatch preserves the handle's opaque ingress origin.
use super::*;

/// Kept out of line: the tree walker's recursion depth guard assumes a bounded
/// per-statement stack frame, so spawn-only locals must not inflate it.
#[inline(never)]
pub(super) fn execute(
    program: &ast::Program,
    call: &ast::CallExpr,
    binding: Option<&ast::Binding>,
    env: &mut Env,
) -> Result<()> {
    let handle = match env.get_raw("__coroutine_handle") {
        Some(Value::CoroutineHandle(handle)) => handle.clone(),
        _ => bail!("+spawn requires async context"),
    };
    let args = call
        .args
        .iter()
        .map(|arg| eval_ast_expr(program, arg, env))
        .collect::<Result<Vec<_>>>()?;
    let (reply, mut receiver) = tokio::sync::oneshot::channel();
    handle
        .io_sender()
        .try_send(crate::coroutine::IoRequest::Spawn {
            function_name: program.qualify_function_name(&call.callee),
            args,
            origin: handle.invocation_origin(),
            reply,
        })
        .map_err(|error| anyhow::anyhow!("+spawn failed to send: {error}"))?;
    if let Some(binding) = binding {
        let task_id = loop {
            match receiver.try_recv() {
                Ok(result) => break result?,
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                    std::thread::sleep(std::time::Duration::from_millis(1))
                }
                Err(error) => bail!("spawn reply failed: {error}"),
            }
        };
        env.set(&binding.name, Value::TaskHandle(task_id));
    }
    Ok(())
}
