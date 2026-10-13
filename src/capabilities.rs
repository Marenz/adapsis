//! Host implementations of the scoped conversational tools.
mod music;
mod cad;
mod video;
mod web;

use std::sync::Arc;
use anyhow::Result;
use crate::coroutine::{IoRequest, scoped::Request};

fn select_cad_references(
    graph: &crate::memory_graph::MemoryGraph,
    principal: &str,
    context: &str,
    use_recent_images: bool,
) -> Result<Vec<crate::attachment::Attachment>> {
    if !use_recent_images {
        return Ok(Vec::new());
    }
    graph.recent_image_attachments(principal, context)
}

pub async fn dispatch(
    request: Request,
    graph: Arc<crate::memory_graph::MemoryGraph>,
    embedder: Arc<crate::memory_graph::MemoryEmbedder>,
    sender: tokio::sync::mpsc::Sender<IoRequest>,
) -> Result<String> {
    dispatch_with_correction_embedding(request, graph, sender, move |replacement| {
        embedder.embed_passages(&[replacement.to_string()])?
            .into_iter().next().ok_or_else(|| anyhow::anyhow!(
                "FastEmbed returned no embedding for the correction"))
    }).await
}

async fn dispatch_with_correction_embedding<F>(
    request: Request,
    graph: Arc<crate::memory_graph::MemoryGraph>,
    sender: tokio::sync::mpsc::Sender<IoRequest>,
    embed_correction: F,
) -> Result<String>
where
    F: FnOnce(&str) -> Result<Vec<f32>> + Send + 'static,
{
    match request {
        Request::History { principal, context, limit, before_ms } =>
            tokio::task::spawn_blocking(move || graph.conversation_history(&principal, &context, limit, before_ms)).await?,
        Request::Forget { principal, context, memory_id } => {
            let forgotten = tokio::task::spawn_blocking(move || graph.forget_in_context(&principal, &context, &memory_id)).await??;
            Ok(if forgotten { "Memory forgotten." } else { "Nothing forgotten: no matching memory in this conversation." }.into())
        }
        Request::Correct { principal, context, may_write, target_id, replacement,
                valid_from_ms, valid_until_ms } => {
            let target_for_message = target_id.clone();
            let corrected = tokio::task::spawn_blocking(move || {
                let embedding = embed_correction(&replacement)?;
                graph.correct_memory(&crate::memory_graph::CorrectionDraft {
                    target_id, replacement, context_id: context, principal_id: principal,
                    may_write, embedding, valid_from_ms, valid_until_ms,
                    created_at_ms: crate::memory_graph::now_ms(),
                })
            }).await??;
            Ok(match corrected {
                Some(id) => format!("corrected as {id}"),
                None => format!("nothing corrected: {target_for_message} is not a memory this conversation can change"),
            })
        }
        Request::Music { context, caption, duration, lyrics } => music::start(context, caption, duration, lyrics, sender),
        Request::Cad { principal, context, title, source, material, use_recent_images } => {
            let graph_for_refs = graph.clone();
            let principal_for_refs = principal.clone();
            let context_for_refs = context.clone();
            let references = tokio::task::spawn_blocking(move || {
                select_cad_references(&graph_for_refs, &principal_for_refs, &context_for_refs,
                    use_recent_images)
            }).await??;
            cad::start(context, title, source, material, references, sender)
        }
        Request::VideoStabilize { principal, context } =>
            video::start(principal, context, graph, sender),
        Request::Search { query } => web::search(&query).await,
        Request::Read { url } => web::read(&url).await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedding() -> Vec<f32> {
        let mut embedding = vec![0.0; 384];
        embedding[0] = 1.0;
        embedding
    }

    #[test]
    fn cad_references_default_to_none() -> Result<()> {
        let graph = crate::memory_graph::MemoryGraph::in_memory()?;
        assert!(select_cad_references(&graph, "speaker", "context", false)?.is_empty());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn scoped_dispatch_requires_write_authority_for_global_correction() -> Result<()> {
        use crate::coroutine::{CoroutineHandle, TurnIdentity};
        use crate::eval::Value;
        use crate::memory_graph::{CanonicalDraft, MemoryScope, SourceMessage};

        let graph = Arc::new(crate::memory_graph::MemoryGraph::in_memory()?);
        for (context, message) in [("context:a", "message:a"), ("context:b", "message:b")] {
            graph.ingest_message(&SourceMessage {
                id: message.into(), platform_message_id: Some(message.into()),
                context_id: context.into(), context_kind: "test".into(),
                speaker_id: "speaker".into(), speaker_name: "speaker".into(),
                role: "user".into(), content: "source".into(), created_at_ms: 1,
            }, "admin")?;
        }
        let target = graph.remember_canonical(&CanonicalDraft {
            content: "global old".into(), context_id: "context:a".into(),
            principal_id: "speaker".into(), embedding: embedding(),
            scope: MemoryScope::Global, created_at_ms: 2,
        })?;

        let (tx, mut rx) = tokio::sync::mpsc::channel(2);
        let worker_graph = graph.clone();
        let worker_sender = tx.clone();
        let worker = tokio::spawn(async move {
            for _ in 0..2 {
                let IoRequest::Scoped { request, reply } = rx.recv().await.unwrap() else {
                    panic!("unexpected IO request");
                };
                let result = dispatch_with_correction_embedding(
                    request,
                    worker_graph.clone(),
                    worker_sender.clone(),
                    |_| Ok(embedding()),
                ).await;
                reply.send(result).unwrap();
            }
        });

        let invoke = |may_write| {
            let tx = tx.clone();
            let target = target.clone();
            tokio::task::spawn_blocking(move || {
                CoroutineHandle::new(tx).with_turn(Some(TurnIdentity {
                    context: "context:b".into(), principal: "speaker".into(),
                    may_write, guest_tools: None,
                })).execute_await("memory_correct", &[
                    Value::string(target), Value::string("global new"),
                ])
            })
        };
        let denied = invoke(false).await??;
        assert_eq!(denied.to_string(), format!(
            "\"nothing corrected: {target} is not a memory this conversation can change\""));
        let allowed = invoke(true).await??;
        assert!(allowed.to_string().starts_with("\"corrected as memory:correction:"));
        worker.await?;

        let active = graph.canonical_for_context("speaker", "context:b", 40, 8_000)?;
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].content, "global new");
        Ok(())
    }
}
