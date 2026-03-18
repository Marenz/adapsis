//! HTTP + WebSocket server for the Forge browser interface.

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::{Html, IntoResponse};
use axum::routing::get;
use axum::Router;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tracing::info;

use crate::ast;
use crate::events::{self, EventBus, ForgeEvent};
use crate::llm::{LlmBackend, LlmClient};
use crate::orchestrator;

/// Shared application state.
pub struct AppState<B: LlmBackend> {
    pub event_bus: EventBus,
    pub program: Mutex<ast::Program>,
    pub llm: LlmClient<B>,
    pub max_iterations: usize,
}

pub async fn serve<B: LlmBackend + Send + Sync + 'static>(
    llm: LlmClient<B>,
    max_iterations: usize,
    port: u16,
) -> anyhow::Result<()> {
    let event_bus = EventBus::new();
    let state = Arc::new(AppState {
        event_bus,
        program: Mutex::new(ast::Program::default()),
        llm,
        max_iterations,
    });

    let app = Router::new()
        .route("/", get(index_handler))
        .route("/ws", get(ws_handler))
        .route(
            "/api/run",
            get({
                let state = Arc::clone(&state);
                move |axum::extract::Query(params): axum::extract::Query<RunParams>| {
                    run_handler(State(state), params)
                }
            }),
        )
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    info!("Forge browser interface at http://127.0.0.1:{port}");

    axum::serve(listener, app).await?;
    Ok(())
}

#[derive(serde::Deserialize)]
struct RunParams {
    task: String,
}

async fn run_handler<B: LlmBackend + Send + Sync + 'static>(
    State(state): State<Arc<AppState<B>>>,
    params: RunParams,
) -> impl IntoResponse {
    let event_bus = state.event_bus.clone();
    let task = params.task.clone();
    let max_iterations = state.max_iterations;

    // Clone what we need and spawn the task
    tokio::spawn(async move {
        // We need a fresh LlmClient for the spawned task — but we can't clone it
        // across the spawn boundary easily. Instead, use the event bus from the
        // orchestrator's perspective by running it inline.
        // For now, just send a placeholder — the real integration happens in the
        // `serve_and_run` function below.
        event_bus.send(ForgeEvent::IterationStart {
            iteration: 0,
            max_iterations,
        });
        event_bus.send(ForgeEvent::ProgramState {
            summary: format!("Task received: {task}"),
        });
    });

    "ok"
}

async fn ws_handler<B: LlmBackend + Send + Sync + 'static>(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState<B>>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(socket, state))
}

async fn handle_ws<B: LlmBackend + Send + Sync + 'static>(
    mut socket: WebSocket,
    state: Arc<AppState<B>>,
) {
    info!("WebSocket client connected");

    // Send current program state
    let program = state.program.lock().await;
    let snapshot = events::snapshot_program(&program);
    drop(program);

    if let Ok(json) = serde_json::to_string(&snapshot) {
        let _ = socket.send(Message::Text(json.into())).await;
    }

    // Subscribe to events and forward to client
    let mut rx = state.event_bus.subscribe();
    loop {
        match rx.recv().await {
            Ok(event) => {
                if let Ok(json) = serde_json::to_string(&event) {
                    if socket.send(Message::Text(json.into())).await.is_err() {
                        break;
                    }
                }
            }
            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                tracing::warn!("WebSocket client lagged, skipped {n} events");
            }
            Err(_) => break,
        }
    }

    info!("WebSocket client disconnected");
}

async fn index_handler() -> Html<&'static str> {
    Html(include_str!("../web/index.html"))
}

/// Run the server AND the orchestrator together.
/// The orchestrator sends events to the event bus, the server streams them to browsers.
pub async fn serve_and_run<B: LlmBackend + Send + Sync + Clone + 'static>(
    llm: LlmClient<B>,
    max_iterations: usize,
    port: u16,
    task: String,
) -> anyhow::Result<()> {
    let event_bus = EventBus::new();
    let state = Arc::new(AppState {
        event_bus: event_bus.clone(),
        program: Mutex::new(ast::Program::default()),
        llm: llm.clone(),
        max_iterations,
    });

    let app = Router::new()
        .route("/", get(index_handler))
        .route("/ws", get(ws_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    info!("Forge browser interface at http://127.0.0.1:{port}");
    info!("Open the URL in your browser, then the task will start automatically.");

    // Run server and orchestrator concurrently
    let server = axum::serve(listener, app);
    let orchestrator = async {
        // Wait a moment for clients to connect
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        let mut orch =
            orchestrator::Orchestrator::with_event_bus(llm, max_iterations, event_bus.clone());
        if let Err(e) = orch.run(&task).await {
            event_bus.send(ForgeEvent::MutationError {
                message: format!("orchestrator error: {e}"),
            });
        }
        event_bus.send(ForgeEvent::Done);
    };

    tokio::select! {
        result = server => { result?; }
        _ = orchestrator => {
            // Orchestrator finished but keep server running
            info!("Task complete. Server still running — press Ctrl+C to stop.");
            std::future::pending::<()>().await;
        }
    }

    Ok(())
}
