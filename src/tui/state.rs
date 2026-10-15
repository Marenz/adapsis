use std::collections::VecDeque;

use serde::Deserialize;

use super::input::Composer;

const MAX_TIMELINE_ENTRIES: usize = 500;
const MAX_TIMELINE_CHARS: usize = 500_000;

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
pub(crate) struct StatusSnapshot {
    #[serde(default)]
    pub revision: usize,
    #[serde(default)]
    pub functions: Vec<String>,
    #[serde(default)]
    pub types: Vec<String>,
    #[serde(default)]
    pub routes: Vec<serde_json::Value>,
    #[serde(default)]
    pub plan: Vec<PlanStep>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub(crate) struct PlanStep {
    pub status: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ServerEvent {
    Start(String),
    User,
    Iteration {
        detail: String,
        current: Option<u64>,
    },
    Thinking(String),
    Content(String),
    Text(String),
    Code(String),
    Result {
        message: String,
        success: bool,
    },
    Test {
        message: String,
        pass: bool,
    },
    Eval(String),
    Query {
        query: String,
        response: String,
    },
    Feedback(String),
    Error(String),
    Complete,
    Unknown {
        kind: String,
        payload: String,
    },
}

impl ServerEvent {
    pub(crate) fn parse(payload: &str) -> Result<Self, String> {
        let value: serde_json::Value = serde_json::from_str(payload)
            .map_err(|error| format!("invalid JSON event: {error}"))?;
        let kind = value
            .get("type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "stream event has no string `type` field".to_string())?;
        let text = |key: &str| {
            value
                .get(key)
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        Ok(match kind {
            "start" => Self::Start(text("message")),
            "user" => Self::User,
            "iteration" => {
                let numeric_fallback = value.get("n").and_then(|n| n.as_u64());
                let detail = match text("detail") {
                    detail if !detail.is_empty() => detail,
                    _ => numeric_fallback
                        .map(|n| format!("iteration {n}"))
                        .unwrap_or_default(),
                };
                Self::Iteration {
                    current: iteration_number(&detail).or(numeric_fallback),
                    detail,
                }
            }
            "thinking" => Self::Thinking(text("text")),
            "content" => Self::Content(text("text")),
            "text" => Self::Text(text("text")),
            "code" => Self::Code(text("code")),
            "result" => Self::Result {
                message: text("message"),
                success: value
                    .get("success")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
            },
            "test" => Self::Test {
                message: text("message"),
                pass: value.get("pass").and_then(|v| v.as_bool()).unwrap_or(false),
            },
            "eval" => Self::Eval(text("result")),
            "query" => Self::Query {
                query: text("query"),
                response: text("response"),
            },
            "feedback" => Self::Feedback(text("message")),
            "error" => Self::Error(text("message")),
            "done" | "end" => Self::Complete,
            _ => Self::Unknown {
                kind: kind.to_string(),
                payload: value.to_string(),
            },
        })
    }
}

fn iteration_number(detail: &str) -> Option<u64> {
    detail
        .split(|character: char| !character.is_ascii_digit())
        .find(|part| !part.is_empty())?
        .parse()
        .ok()
}

fn iteration_label(detail: &str) -> String {
    let detail = detail.trim();
    detail
        .strip_prefix("iteration")
        .map(|rest| format!("Iteration{rest}"))
        .unwrap_or_else(|| detail.to_string())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EntryKind {
    User,
    Thinking,
    Assistant,
    Code,
    Result,
    Test,
    Eval,
    Query,
    Feedback,
    Error,
    System,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TimelineEntry {
    pub id: u64,
    pub kind: EntryKind,
    pub text: String,
    pub success: Option<bool>,
    pub live: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ServerConnection {
    Connecting,
    Online,
    Offline(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TurnState {
    Idle,
    Streaming,
    Complete,
    Failed,
}

pub(crate) enum ClientMessage {
    Status(Result<StatusSnapshot, String>),
    Stream(ServerEvent),
    StreamFailure(String),
    StreamClosed,
}

pub(crate) struct AppState {
    pub api_url: String,
    pub composer: Composer,
    pub timeline: VecDeque<TimelineEntry>,
    pub connection: ServerConnection,
    pub turn: TurnState,
    pub status: Option<StatusSnapshot>,
    pub help_visible: bool,
    pub follow: bool,
    pub scroll_top: usize,
    pub layout_total_lines: usize,
    pub layout_viewport_lines: usize,
    next_id: u64,
    timeline_chars: usize,
    pending_echo: Option<String>,
    live_content_id: Option<u64>,
    turn_had_errors: bool,
}

impl AppState {
    pub(crate) fn new(api_url: String) -> Self {
        Self {
            api_url,
            composer: Composer::default(),
            timeline: VecDeque::new(),
            connection: ServerConnection::Connecting,
            turn: TurnState::Idle,
            status: None,
            help_visible: false,
            follow: true,
            scroll_top: 0,
            layout_total_lines: 0,
            layout_viewport_lines: 0,
            next_id: 1,
            timeline_chars: 0,
            pending_echo: None,
            live_content_id: None,
            turn_had_errors: false,
        }
    }

    pub(crate) fn begin_turn(&mut self, message: String) {
        self.push(EntryKind::User, message.clone(), None, false);
        self.pending_echo = Some(message);
        self.live_content_id = None;
        self.turn_had_errors = false;
        self.turn = TurnState::Streaming;
        self.follow_latest();
    }

    pub(crate) fn apply(&mut self, message: ClientMessage) {
        match message {
            ClientMessage::Status(status) => self.apply_status(status),
            ClientMessage::Stream(event) => self.apply_server_event(event),
            ClientMessage::StreamFailure(error) => self.fail_turn(error),
            ClientMessage::StreamClosed if self.turn == TurnState::Streaming => {
                self.fail_turn("stream closed before the server sent `end`".to_string());
            }
            ClientMessage::StreamClosed => {}
        }
    }

    fn apply_status(&mut self, status: Result<StatusSnapshot, String>) {
        match status {
            Ok(status) => {
                self.status = Some(status);
                self.connection = ServerConnection::Online;
            }
            Err(error) => {
                let is_new = !matches!(
                    &self.connection,
                    ServerConnection::Offline(previous) if previous == &error
                );
                self.connection = ServerConnection::Offline(error.clone());
                if is_new {
                    self.push(
                        EntryKind::Error,
                        format!("Status connection failed: {error}"),
                        Some(false),
                        false,
                    );
                }
            }
        }
    }

    fn apply_server_event(&mut self, event: ServerEvent) {
        match event {
            ServerEvent::Start(message) => self.accept_start(message),
            // `user` is an internal context/log event. The submitted message was
            // already added immediately and confirmed by `start`; rendering this
            // would duplicate it and expose prompt scaffolding.
            ServerEvent::User => {}
            ServerEvent::Iteration { detail, current }
                if current.is_none_or(|current| current > 1) && !detail.trim().is_empty() =>
            {
                self.push(EntryKind::System, iteration_label(&detail), None, false);
            }
            ServerEvent::Iteration { .. } => {}
            ServerEvent::Thinking(text) => self.append_or_push(EntryKind::Thinking, text, true),
            ServerEvent::Content(text) => self.append_content(text),
            ServerEvent::Text(text) => self.finalize_content(text),
            ServerEvent::Code(code) => {
                self.push(EntryKind::Code, code, None, false);
            }
            ServerEvent::Result { message, success } => {
                self.record_outcome(EntryKind::Result, message, success)
            }
            ServerEvent::Test { message, pass } => {
                self.record_outcome(EntryKind::Test, message, pass)
            }
            ServerEvent::Eval(result) => {
                self.push(EntryKind::Eval, result, None, false);
            }
            ServerEvent::Query { query, response } => {
                self.push(
                    EntryKind::Query,
                    format!("{query}\n{response}"),
                    None,
                    false,
                );
            }
            ServerEvent::Feedback(message) => {
                self.push(EntryKind::Feedback, message, None, false);
            }
            ServerEvent::Error(message) => {
                self.turn_had_errors = true;
                self.push(EntryKind::Error, message, Some(false), false);
            }
            ServerEvent::Complete => self.complete_turn(),
            ServerEvent::Unknown { kind, payload } => {
                self.push(
                    EntryKind::System,
                    format!("Unknown event `{kind}`: {payload}"),
                    None,
                    false,
                );
            }
        }
    }

    fn accept_start(&mut self, message: String) {
        if self.pending_echo.as_deref() == Some(message.as_str()) {
            self.pending_echo = None;
        } else if !message.is_empty() {
            self.push(EntryKind::User, message, None, false);
        }
        self.turn = TurnState::Streaming;
    }

    fn append_content(&mut self, text: String) {
        if text.is_empty() {
            return;
        }
        if let Some(id) = self.live_content_id {
            if self.append_to(id, &text) {
                return;
            }
        }
        let id = self.push(EntryKind::Assistant, text, None, true);
        self.live_content_id = Some(id);
    }

    fn finalize_content(&mut self, text: String) {
        if let Some(id) = self.live_content_id.take() {
            if let Some(entry) = self.timeline.iter_mut().find(|entry| entry.id == id) {
                self.timeline_chars = self.timeline_chars.saturating_sub(entry.text.len());
                entry.text = text;
                entry.live = false;
                self.timeline_chars += entry.text.len();
                self.enforce_bounds();
                return;
            }
        }
        if !text.is_empty() {
            self.push(EntryKind::Assistant, text, None, false);
        }
    }

    fn record_outcome(&mut self, kind: EntryKind, message: String, success: bool) {
        self.turn_had_errors |= !success;
        self.push(kind, message, Some(success), false);
    }

    fn complete_turn(&mut self) {
        self.pending_echo = None;
        self.live_content_id = None;
        self.turn = if self.turn_had_errors {
            TurnState::Failed
        } else {
            TurnState::Complete
        };
        self.push(
            if self.turn_had_errors {
                EntryKind::Error
            } else {
                EntryKind::System
            },
            if self.turn_had_errors {
                "Turn ended with reported errors"
            } else {
                "Stream complete"
            }
            .to_string(),
            self.turn_had_errors.then_some(false),
            false,
        );
    }

    fn fail_turn(&mut self, error: String) {
        self.turn_had_errors = true;
        self.turn = TurnState::Failed;
        self.live_content_id = None;
        self.push(EntryKind::Error, error, Some(false), false);
    }

    fn append_or_push(&mut self, kind: EntryKind, text: String, live: bool) {
        if text.is_empty() {
            return;
        }
        if let Some(entry) = self.timeline.back_mut().filter(|entry| entry.kind == kind) {
            entry.text.push_str(&text);
            entry.live = live;
            self.timeline_chars += text.len();
            self.enforce_bounds();
        } else {
            self.push(kind, text, None, live);
        }
    }

    fn append_to(&mut self, id: u64, text: &str) -> bool {
        let Some(entry) = self.timeline.iter_mut().find(|entry| entry.id == id) else {
            return false;
        };
        entry.text.push_str(text);
        self.timeline_chars += text.len();
        self.enforce_bounds();
        true
    }

    fn push(&mut self, kind: EntryKind, text: String, success: Option<bool>, live: bool) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.timeline_chars += text.len();
        self.timeline.push_back(TimelineEntry {
            id,
            kind,
            text,
            success,
            live,
        });
        self.enforce_bounds();
        id
    }

    fn enforce_bounds(&mut self) {
        while self.timeline.len() > MAX_TIMELINE_ENTRIES || self.timeline_chars > MAX_TIMELINE_CHARS
        {
            let Some(removed) = self.timeline.pop_front() else {
                break;
            };
            self.timeline_chars = self.timeline_chars.saturating_sub(removed.text.len());
            if !self.follow {
                self.scroll_top = self.scroll_top.saturating_sub(logical_lines(&removed.text));
            }
            if self.live_content_id == Some(removed.id) {
                self.live_content_id = None;
            }
        }
    }

    pub(crate) fn update_layout(&mut self, total_lines: usize, viewport_lines: usize) {
        self.layout_total_lines = total_lines;
        self.layout_viewport_lines = viewport_lines;
        let bottom = total_lines.saturating_sub(viewport_lines);
        if self.follow {
            self.scroll_top = bottom;
        } else {
            self.scroll_top = self.scroll_top.min(bottom);
        }
    }

    pub(crate) fn scroll_up(&mut self, amount: usize) {
        if self.follow {
            self.scroll_top = self
                .layout_total_lines
                .saturating_sub(self.layout_viewport_lines);
        }
        self.follow = false;
        self.scroll_top = self.scroll_top.saturating_sub(amount);
    }

    pub(crate) fn scroll_down(&mut self, amount: usize) {
        self.follow = false;
        let bottom = self
            .layout_total_lines
            .saturating_sub(self.layout_viewport_lines);
        self.scroll_top = self.scroll_top.saturating_add(amount).min(bottom);
    }

    pub(crate) fn page_size(&self) -> usize {
        self.layout_viewport_lines.saturating_sub(1).max(1)
    }

    pub(crate) fn follow_latest(&mut self) {
        self.follow = true;
        self.scroll_top = self
            .layout_total_lines
            .saturating_sub(self.layout_viewport_lines);
    }
}

fn logical_lines(text: &str) -> usize {
    text.lines().count().max(1) + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_server_event_contract() {
        assert_eq!(
            ServerEvent::parse(r#"{"type":"iteration","detail":"iteration 3/10"}"#).unwrap(),
            ServerEvent::Iteration {
                detail: "iteration 3/10".to_string(),
                current: Some(3),
            }
        );
        assert_eq!(
            ServerEvent::parse(r#"{"type":"user","text":"internal user context"}"#).unwrap(),
            ServerEvent::User
        );
        assert_eq!(
            ServerEvent::parse(r#"{"type":"result","message":"no","success":false}"#).unwrap(),
            ServerEvent::Result {
                message: "no".to_string(),
                success: false,
            }
        );
        assert_eq!(
            ServerEvent::parse(r#"{"type":"query","query":"?symbols","response":"A"}"#).unwrap(),
            ServerEvent::Query {
                query: "?symbols".to_string(),
                response: "A".to_string(),
            }
        );
        assert_eq!(
            ServerEvent::parse(r#"{"type":"end"}"#).unwrap(),
            ServerEvent::Complete
        );
        assert!(ServerEvent::parse("not json").is_err());
    }

    #[test]
    fn shows_iteration_detail_and_suppresses_duplicate_user_log_event() {
        let mut state = AppState::new("http://local".to_string());
        state.begin_turn("build it".to_string());
        state.apply(ClientMessage::Stream(
            ServerEvent::parse(r#"{"type":"start","message":"build it"}"#).unwrap(),
        ));
        state.apply(ClientMessage::Stream(
            ServerEvent::parse(
                r#"{"type":"user","text":"Working directory: /tmp\nUser: build it"}"#,
            )
            .unwrap(),
        ));
        state.apply(ClientMessage::Stream(
            ServerEvent::parse(r#"{"type":"iteration","detail":"iteration 3/10"}"#).unwrap(),
        ));

        assert_eq!(
            state
                .timeline
                .iter()
                .filter(|entry| entry.kind == EntryKind::User)
                .count(),
            1
        );
        assert!(
            !state
                .timeline
                .iter()
                .any(|entry| entry.text.contains("Working directory"))
        );
        assert!(
            state
                .timeline
                .iter()
                .any(|entry| { entry.kind == EntryKind::System && entry.text == "Iteration 3/10" })
        );
    }

    #[test]
    fn reduces_stream_into_typed_entries_and_replaces_live_content() {
        let mut state = AppState::new("http://local".to_string());
        state.begin_turn("build it".to_string());
        state.apply(ClientMessage::Stream(ServerEvent::Start(
            "build it".to_string(),
        )));
        state.apply(ClientMessage::Stream(ServerEvent::Thinking(
            "checking".to_string(),
        )));
        state.apply(ClientMessage::Stream(ServerEvent::Content(
            "draft <code>x".to_string(),
        )));
        state.apply(ClientMessage::Stream(ServerEvent::Text(
            "final answer".to_string(),
        )));
        state.apply(ClientMessage::Stream(ServerEvent::Result {
            message: "applied".to_string(),
            success: true,
        }));
        state.apply(ClientMessage::Stream(ServerEvent::Complete));

        assert_eq!(state.turn, TurnState::Complete);
        assert_eq!(
            state
                .timeline
                .iter()
                .filter(|e| e.kind == EntryKind::User)
                .count(),
            1
        );
        assert!(
            state
                .timeline
                .iter()
                .any(|e| { e.kind == EntryKind::Assistant && e.text == "final answer" && !e.live })
        );
        assert!(!state.timeline.iter().any(|e| e.text.contains("draft")));
    }

    #[test]
    fn failures_never_reduce_to_success() {
        let mut state = AppState::new("http://local".to_string());
        state.begin_turn("go".to_string());
        state.apply(ClientMessage::Stream(ServerEvent::Result {
            message: "failed".to_string(),
            success: false,
        }));
        state.apply(ClientMessage::Stream(ServerEvent::Complete));
        assert_eq!(state.turn, TurnState::Failed);
        assert_eq!(state.timeline.back().unwrap().kind, EntryKind::Error);

        let mut disconnected = AppState::new("http://local".to_string());
        disconnected.begin_turn("go".to_string());
        disconnected.apply(ClientMessage::StreamClosed);
        assert_eq!(disconnected.turn, TurnState::Failed);
        assert!(
            disconnected
                .timeline
                .back()
                .unwrap()
                .text
                .contains("before")
        );
    }

    #[test]
    fn incoming_events_do_not_move_review_position() {
        let mut state = AppState::new("http://local".to_string());
        for n in 0..20 {
            state.push(EntryKind::System, format!("line {n}"), None, false);
        }
        state.update_layout(40, 10);
        state.scroll_up(8);
        let reviewed_top = state.scroll_top;
        state.apply(ClientMessage::Stream(ServerEvent::Feedback(
            "new".to_string(),
        )));
        assert!(!state.follow);
        assert_eq!(state.scroll_top, reviewed_top);
        state.follow_latest();
        assert!(state.follow);
    }

    #[test]
    fn timeline_is_bounded() {
        let mut state = AppState::new("http://local".to_string());
        for n in 0..(MAX_TIMELINE_ENTRIES + 20) {
            state.push(EntryKind::System, n.to_string(), None, false);
        }
        assert_eq!(state.timeline.len(), MAX_TIMELINE_ENTRIES);
        assert_eq!(state.timeline.front().unwrap().text, "20");
    }

    #[test]
    fn status_failure_is_actionable_visible_and_deduplicated() {
        let mut state = AppState::new("http://local".to_string());
        let error = "cannot reach /api/status; pass --api".to_string();
        state.apply(ClientMessage::Status(Err(error.clone())));
        state.apply(ClientMessage::Status(Err(error)));
        assert!(matches!(state.connection, ServerConnection::Offline(_)));
        assert_eq!(state.timeline.len(), 1);
        assert!(state.timeline[0].text.contains("--api"));
    }
}
