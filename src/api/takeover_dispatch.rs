//! Shared production dispatch into the existing takeover loop. No replacement loop.
use super::{llm_handlers::handle_llm_takeover, record_context::RecordTurn};
use crate::{ast, coroutine, ingress, memory_graph, permissions, session};
use std::sync::Arc;

pub(crate) struct Input {
    pub context: String,
    pub message: String,
    pub attachment: Option<crate::attachment::Attachment>,
    pub metadata: Option<String>,
    pub reply_fn: Option<String>,
    pub reply_arg: Option<String>,
    pub permission_model: Option<String>,
    pub origin: Option<ingress::InvocationOrigin>,
}

pub(crate) struct Host {
    pub meta: session::SharedMeta,
    pub program: Arc<tokio::sync::RwLock<ast::Program>>,
    pub runtime: session::SharedRuntime,
    pub url: String,
    pub model: String,
    pub key: Option<String>,
    pub sender: tokio::sync::mpsc::Sender<coroutine::IoRequest>,
    pub tasks: coroutine::TaskRegistry,
    pub snapshots: coroutine::TaskSnapshotRegistry,
    pub opencode_lock: Arc<tokio::sync::Mutex<()>>,
    pub opencode_dir: String,
    pub training: Option<Arc<tokio::sync::Mutex<tokio::fs::File>>>,
    pub log: Option<Arc<tokio::sync::Mutex<tokio::fs::File>>>,
    pub access: permissions::AccessLevel,
    pub permissions: Arc<permissions::PermissionConfig>,
    pub save: Option<tokio::sync::mpsc::Sender<()>>,
    pub native: bool,
    pub graph: Arc<memory_graph::MemoryGraph>,
    pub embedder: Arc<memory_graph::MemoryEmbedder>,
    pub record_store: Option<memory_graph::records::RecordStore>,
}

fn set_profile(host: &Host, input: &Input) {
    host.meta
        .lock()
        .unwrap()
        .conversations
        .get_or_create(&input.context)
        .permission_model = input.permission_model.clone();
}

pub(crate) fn dispatch(
    host: Host,
    input: Input,
    reply: tokio::sync::oneshot::Sender<anyhow::Result<String>>,
) {
    // Preserve ordinary dispatch's pre-loop profile update. Verified calls cannot
    // create/mutate a conversation until their one-time admission succeeds.
    if input.origin.is_none() {
        set_profile(&host, &input);
    }
    tokio::spawn(async move {
        let result = run(host, input).await;
        let _ = reply.send(result);
    });
}

async fn run(host: Host, input: Input) -> anyhow::Result<String> {
    let (claim, record_turn) = if let Some(origin) = &input.origin {
        if input.attachment.is_some() {
            anyhow::bail!("verified ingress binding mismatch");
        }
        let claim = origin
            .consume(
                &input.context,
                &input.message,
                input.metadata.as_deref(),
                input.reply_fn.as_deref(),
                input.reply_arg.as_deref(),
            )
            .await
            .map_err(|error| match error {
                ingress::IngressError::StateUnknown => {
                    anyhow::anyhow!("verified ingress admission outcome unknown")
                }
                _ => anyhow::anyhow!("verified ingress admission refused"),
            })?;
        // Pin THIS turn's own dispatch profile, not whatever the shared
        // conversation field held (a prior speaker's, or a concurrent dispatch's).
        let context = RecordTurn::claimed(
            &claim,
            host.record_store.clone(),
            &host.permissions,
            host.access,
            &host.model,
            input.permission_model.as_deref(),
        );
        set_profile(&host, &input);
        (Some(claim), Some(context))
    } else {
        (None, None)
    };
    // Revoke on every exit, including panics/cancellation of this future.
    let _revoke = record_turn.clone().map(RevokeOnDrop);
    let result = handle_llm_takeover(
        input.context,
        input.message,
        input.attachment,
        input.metadata,
        host.graph,
        host.embedder,
        input.reply_fn,
        input.reply_arg,
        host.meta,
        host.program,
        host.runtime,
        &host.url,
        &host.model,
        host.key,
        host.sender,
        host.tasks,
        host.snapshots,
        host.opencode_lock,
        host.opencode_dir,
        host.training,
        host.access,
        host.permissions,
        host.save,
        host.log,
        host.native,
        record_turn.clone(),
    )
    .await;
    drop(_revoke);
    if let Some(claim) = claim {
        let confirmed = result.is_ok() && record_turn.as_ref().is_some_and(|turn| turn.completed());
        claim
            .finish(confirmed)
            .await
            .map_err(|_| anyhow::anyhow!("verified ingress terminal outcome unknown"))?;
    }
    result
}

struct RevokeOnDrop(std::sync::Arc<RecordTurn>);

impl Drop for RevokeOnDrop {
    fn drop(&mut self) {
        self.0.revoke();
    }
}
