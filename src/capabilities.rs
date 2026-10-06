//! Host implementations of the scoped conversational tools.
mod music;
mod web;

use std::sync::Arc;
use anyhow::Result;
use crate::coroutine::{IoRequest, scoped::Request};

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
        Request::Search { query } => web::search(&query).await,
        Request::Read { url } => web::read(&url).await,
    }
}
