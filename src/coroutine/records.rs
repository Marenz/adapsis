//! Generic record IO adapter. No production provisioning or task policy.
use super::{CoroutineHandle, IoRequest, TurnIdentity, Value, WaitReason};
use crate::memory_graph::records::{
    MAX_BATCH_BYTES, MAX_KEY_BYTES, RecordBinding, RecordStore, StoreError,
};
use serde_json::json;
use tokio::sync::oneshot;
#[cfg(test)]
mod tests;

#[derive(Clone)]
pub(super) struct Access {
    store: RecordStore,
    binding: RecordBinding,
}

impl std::fmt::Debug for Access {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RecordAccess(<bound>)")
    }
}

enum Operation {
    Read(String),
    Commit(String, String),
    Receipt(String),
}

pub struct Request {
    access: Option<Access>,
    turn: Option<TurnIdentity>,
    operation: Operation,
}

impl std::fmt::Debug for Request {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RecordRequest(<redacted>)")
    }
}

impl Request {
    #[cfg(test)]
    pub(crate) fn is_read_for_test(&self) -> bool {
        matches!(self.operation, Operation::Read(_))
    }

    #[cfg(test)]
    pub(crate) fn is_commit_for_test(&self) -> bool {
        self.is_commit()
    }

    #[cfg(test)]
    pub(crate) fn commit_for_test(&self) -> Option<(String, String)> {
        match &self.operation {
            Operation::Commit(id, batch) => Some((id.clone(), batch.clone())),
            _ => None,
        }
    }
    fn is_commit(&self) -> bool {
        matches!(self.operation, Operation::Commit(..))
    }

    fn execute(self) -> Result<String, StoreError> {
        let committing = self.is_commit();
        let access = self.access.ok_or(StoreError::Denied)?;
        let binding = Some(&access.binding);
        let turn = self.turn.as_ref();
        let value = match self.operation {
            Operation::Read(request) => json!({
                "format":"records_v1", "records":access.store.read(binding, turn, &request)?
            }),
            Operation::Commit(id, request) => json!({
                "format":"records_v1", "receipt":access.store.commit(binding, turn, &id, &request)?
            }),
            Operation::Receipt(id) => json!({
                "format":"records_v1", "receipt":access.store.receipt(binding, turn, &id)?
            }),
        };
        serde_json::to_string(&value).map_err(|_| {
            if committing {
                StoreError::OutcomeUnknown
            } else {
                StoreError::Unavailable
            }
        })
    }
}

pub(super) async fn dispatch(
    request: Request,
    reply: oneshot::Sender<anyhow::Result<Result<String, StoreError>>>,
) {
    let unknown = request.is_commit();
    let outcome = tokio::task::spawn_blocking(move || request.execute())
        .await
        .unwrap_or_else(|_| {
            Err(if unknown {
                StoreError::OutcomeUnknown
            } else {
                StoreError::Unavailable
            })
        });
    let _ = reply.send(Ok(outcome));
}

impl CoroutineHandle {
    pub(super) fn execute_records(&self, op: &str, args: &[Value]) -> Option<Value> {
        if !matches!(op, "records_read" | "records_commit" | "records_receipt") {
            return None;
        }
        let failure = |error: StoreError| Value::Err(error.to_string());
        let Some(access) = &self.record_access else {
            return Some(failure(StoreError::Denied));
        };
        if let Err(error) =
            access
                .store
                .check_access(&access.binding, self.turn.as_ref(), op != "records_read")
        {
            return Some(failure(error));
        }
        let operation = match (op, args) {
            ("records_read", [Value::String(request)]) if request.len() <= MAX_BATCH_BYTES => {
                Operation::Read(request.as_ref().clone())
            }
            ("records_commit", [Value::String(id), Value::String(request)])
                if !id.is_empty()
                    && id.len() <= MAX_KEY_BYTES
                    && request.len() <= MAX_BATCH_BYTES =>
            {
                Operation::Commit(id.as_ref().clone(), request.as_ref().clone())
            }
            ("records_receipt", [Value::String(id)])
                if !id.is_empty() && id.len() <= MAX_KEY_BYTES =>
            {
                Operation::Receipt(id.as_ref().clone())
            }
            _ => return Some(failure(StoreError::InvalidRequest)),
        };
        let request = Request {
            access: self.record_access.clone(),
            turn: self.turn.clone(),
            operation,
        };
        let (reply, receiver) = oneshot::channel();
        let result = self.send_and_wait(
            WaitReason::Running,
            IoRequest::Records { request, reply },
            receiver,
        );
        Some(match result {
            Ok(Ok(body)) => Value::Ok(Box::new(Value::string(body))),
            Ok(Err(error)) => failure(error),
            Err(_) => failure(if op == "records_commit" {
                StoreError::OutcomeUnknown
            } else {
                StoreError::Unavailable
            }),
        })
    }

    #[cfg(test)]
    pub(crate) fn with_records_for_test(
        mut self,
        store: RecordStore,
        binding: RecordBinding,
    ) -> Self {
        self.record_access = Some(Access { store, binding });
        self
    }
}
