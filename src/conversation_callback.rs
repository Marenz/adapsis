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
            // Platform reply callbacks have their own trusted handle, exactly
            // as before the shared builder. This is not a model permission or
            // inheritance of the calling model block's program-freeze state.
            program_frozen: false,
            reply,
        },
        receiver,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn trusted_callback_builder_preserves_inline_spawn_fields_and_reply_channel() {
        let (request, receiver) = request(
            "CallbackFixture.reply".into(),
            vec![Value::string("destination"), Value::string("message")],
        );
        let IoRequest::Spawn {
            function_name,
            args,
            program_frozen,
            reply,
        } = request
        else {
            panic!("callback builder must use Spawn")
        };
        assert_eq!(function_name, "CallbackFixture.reply");
        assert!(
            !program_frozen,
            "same trusted handle as the pre-merge inline Spawn"
        );
        let strings: Vec<_> = args
            .into_iter()
            .map(|arg| match arg {
                Value::String(text) => text.to_string(),
                _ => panic!("argument changed"),
            })
            .collect();
        assert_eq!(strings, ["destination", "message"]);
        reply.send(Ok(17)).unwrap();
        assert_eq!(
            receiver.await.unwrap().unwrap(),
            17,
            "the caller observes the actual Spawn reply channel"
        );
    }
}
