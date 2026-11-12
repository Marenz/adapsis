//! Request construction shared by trusted, bound conversation reply dispatchers.
use crate::coroutine::{IoRequest, TaskId};
use crate::eval::Value;
use tokio::sync::oneshot;

pub(crate) fn request(
    function_name: String,
    args: Vec<Value>,
) -> (IoRequest, oneshot::Receiver<anyhow::Result<TaskId>>) {
    let (reply, receiver) = oneshot::channel();
    (
        IoRequest::Spawn {
            function_name,
            args,
            reply,
        },
        receiver,
    )
}
