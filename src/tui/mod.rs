mod input;
mod state;
mod ui;

use std::io::{self, Stdout};
use std::time::Duration;

use anyhow::{Context, Result};
use crossterm::event::{DisableMouseCapture, EnableMouseCapture, Event, EventStream};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use tokio::sync::mpsc;

use self::input::Action;
use self::state::{AppState, ClientMessage, ServerEvent, StatusSnapshot};

type TuiTerminal = Terminal<CrosstermBackend<Stdout>>;

pub(crate) async fn run(api_url: &str) -> Result<()> {
    let api_url = api_url.trim_end_matches('/').to_string();
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .build()
        .context("build TUI HTTP client")?;
    let mut terminal = TerminalSession::new()?;
    let result = run_loop(&mut terminal.terminal, client, api_url).await;
    terminal.restore();
    result
}

async fn run_loop(
    terminal: &mut TuiTerminal,
    client: reqwest::Client,
    api_url: String,
) -> Result<()> {
    let (tx, mut rx) = mpsc::channel::<ClientMessage>(256);
    let status_task = tokio::spawn(poll_status(client.clone(), api_url.clone(), tx.clone()));
    let mut events = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_millis(50));
    let mut turn_task: Option<tokio::task::JoinHandle<()>> = None;
    let mut state = AppState::new(api_url.clone());

    loop {
        terminal.draw(|frame| ui::draw(frame, &mut state))?;
        tokio::select! {
            event = events.next() => {
                let Some(event) = event else { break; };
                match event.context("read terminal event")? {
                    Event::Key(key) => match input::handle_key(&mut state, key) {
                        Action::None => {}
                        Action::Quit => break,
                        Action::Submit(message) => {
                            state.begin_turn(message.clone());
                            let task_client = client.clone();
                            let task_api = api_url.clone();
                            let task_tx = tx.clone();
                            turn_task = Some(tokio::spawn(async move {
                                stream_turn(task_client, task_api, message, task_tx).await;
                            }));
                        }
                    },
                    Event::Mouse(mouse) => input::handle_mouse(&mut state, mouse),
                    Event::Paste(value) => state.composer.insert_str(&value),
                    Event::Resize(_, _) | Event::FocusGained | Event::FocusLost => {}
                }
            }
            message = rx.recv() => {
                if let Some(message) = message {
                    state.apply(message);
                }
            }
            _ = tick.tick() => {}
        }
    }

    status_task.abort();
    if let Some(task) = turn_task {
        task.abort();
    }
    Ok(())
}

async fn poll_status(client: reqwest::Client, api_url: String, tx: mpsc::Sender<ClientMessage>) {
    loop {
        let status = fetch_status(&client, &api_url).await;
        if tx.send(ClientMessage::Status(status)).await.is_err() {
            return;
        }
        tokio::time::sleep(Duration::from_secs(3)).await;
    }
}

async fn fetch_status(client: &reqwest::Client, api_url: &str) -> Result<StatusSnapshot, String> {
    let endpoint = format!("{api_url}/api/status");
    let response = client
        .get(&endpoint)
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .map_err(|error| actionable_connection_error(api_url, "/api/status", &error))?;
    if !response.status().is_success() {
        return Err(format!("{endpoint} returned HTTP {}", response.status()));
    }
    response
        .json()
        .await
        .map_err(|error| format!("invalid status response from {endpoint}: {error}"))
}

async fn stream_turn(
    client: reqwest::Client,
    api_url: String,
    message: String,
    tx: mpsc::Sender<ClientMessage>,
) {
    if let Err(error) = stream_turn_inner(&client, &api_url, &message, &tx).await {
        let _ = tx.send(ClientMessage::StreamFailure(error)).await;
        return;
    }
    let _ = tx.send(ClientMessage::StreamClosed).await;
}

async fn stream_turn_inner(
    client: &reqwest::Client,
    api_url: &str,
    message: &str,
    tx: &mpsc::Sender<ClientMessage>,
) -> Result<(), String> {
    let path = "/api/ask-stream";
    let endpoint = format!("{api_url}{path}");
    let response = client
        .post(&endpoint)
        .json(&serde_json::json!({"message": message}))
        .send()
        .await
        .map_err(|error| actionable_connection_error(api_url, path, &error))?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!(
            "{endpoint} returned HTTP {status}: {}",
            body.trim().chars().take(300).collect::<String>()
        ));
    }

    let mut parser = crate::client_sse::SseParser::new();
    let mut bytes = response.bytes_stream();
    while let Some(chunk) = bytes.next().await {
        let chunk = chunk.map_err(|error| format!("stream read failed: {error}"))?;
        for payload in parser.feed(&chunk)? {
            let event = ServerEvent::parse(&payload)?;
            tx.send(ClientMessage::Stream(event))
                .await
                .map_err(|_| "TUI closed while receiving the stream".to_string())?;
        }
    }
    if parser.has_pending_data() {
        return Err("stream ended with an incomplete SSE event".to_string());
    }
    Ok(())
}

fn actionable_connection_error(api_url: &str, path: &str, error: &reqwest::Error) -> String {
    format!(
        "cannot reach AdapsisOS at {api_url}{path}: {error}. Start it with `adapsis os --port N ...` or pass the running instance with `--api`"
    )
}

struct TerminalSession {
    terminal: TuiTerminal,
    restored: bool,
}

impl TerminalSession {
    fn new() -> Result<Self> {
        enable_raw_mode().context("enable terminal raw mode")?;
        let mut stdout = io::stdout();
        if let Err(error) = execute!(stdout, EnterAlternateScreen, EnableMouseCapture) {
            let _ = disable_raw_mode();
            return Err(error).context("enter alternate terminal screen");
        }
        let terminal = match Terminal::new(CrosstermBackend::new(stdout)) {
            Ok(terminal) => terminal,
            Err(error) => {
                let _ = disable_raw_mode();
                let mut stdout = io::stdout();
                let _ = execute!(stdout, LeaveAlternateScreen, DisableMouseCapture);
                return Err(error).context("initialize terminal renderer");
            }
        };
        Ok(Self {
            terminal,
            restored: false,
        })
    }

    fn restore(&mut self) {
        if self.restored {
            return;
        }
        let _ = disable_raw_mode();
        let _ = execute!(
            self.terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        );
        let _ = self.terminal.show_cursor();
        self.restored = true;
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        self.restore();
    }
}
