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
    sender: tokio::sync::mpsc::Sender<IoRequest>,
) -> Result<String> {
    match request {
        Request::History { principal, context, limit, before_ms } =>
            tokio::task::spawn_blocking(move || graph.conversation_history(&principal, &context, limit, before_ms)).await?,
        Request::Forget { principal, context, memory_id } => {
            let forgotten = tokio::task::spawn_blocking(move || graph.forget_in_context(&principal, &context, &memory_id)).await??;
            Ok(if forgotten { "Memory forgotten." } else { "Nothing forgotten: no matching memory in this conversation." }.into())
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

    #[test]
    fn cad_references_default_to_none() -> Result<()> {
        let graph = crate::memory_graph::MemoryGraph::in_memory()?;
        assert!(select_cad_references(&graph, "speaker", "context", false)?.is_empty());
        Ok(())
    }
}
