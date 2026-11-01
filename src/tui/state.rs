use std::collections::{HashMap, VecDeque};

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
    Thinking {
        text: String,
        reasoning_tokens: Option<u64>,
    },
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
    NativeComplete {
        completed: bool,
        has_errors: bool,
    },
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
            "thinking" => Self::Thinking {
                text: text("text"),
                reasoning_tokens: reasoning_tokens(&value),
            },
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
            "done" if value["completed"].is_boolean() => Self::NativeComplete {
                completed: value["completed"] == true,
                has_errors: value["has_errors"] == true,
            },
            "done" | "end" => Self::Complete,
            _ => Self::Unknown {
                kind: kind.to_string(),
                payload: value.to_string(),
            },
        })
    }
}

fn reasoning_tokens(value: &serde_json::Value) -> Option<u64> {
    value
        .get("reasoning_tokens")
        .and_then(serde_json::Value::as_u64)
        .or_else(|| {
            value
                .get("usage")
                .and_then(|usage| usage.get("reasoning_tokens"))
                .and_then(serde_json::Value::as_u64)
        })
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
    pub reasoning_tokens: Option<u64>,
    pub interrupted: bool,
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
    pub show_thinking: bool,
    pub follow: bool,
    pub scroll_top: usize,
    pub layout_total_lines: usize,
    pub layout_viewport_lines: usize,
    pub(crate) entry_heights: HashMap<u64, usize>,
    next_id: u64,
    timeline_chars: usize,
    pending_echo: Option<String>,
    live_content_id: Option<u64>,
    thinking_entry_id: Option<u64>,
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
            show_thinking: false,
            follow: true,
            scroll_top: 0,
            layout_total_lines: 0,
            layout_viewport_lines: 0,
            entry_heights: HashMap::new(),
            next_id: 1,
            timeline_chars: 0,
            pending_echo: None,
            live_content_id: None,
            thinking_entry_id: None,
            turn_had_errors: false,
        }
    }

    pub(crate) fn begin_turn(&mut self, message: String) {
        self.finish_thinking();
        self.thinking_entry_id = None;
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
        match &event {
            ServerEvent::Thinking { .. }
            | ServerEvent::Start(_)
            | ServerEvent::User
            | ServerEvent::Iteration { .. }
            | ServerEvent::Unknown { .. } => {}
            ServerEvent::Error(_)
            | ServerEvent::NativeComplete {
                completed: false, ..
            } => self.interrupt_thinking(),
            _ => self.finish_thinking(),
        }
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
            ServerEvent::Thinking {
                text,
                reasoning_tokens,
            } => self.record_thinking(text, reasoning_tokens),
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
            ServerEvent::NativeComplete {
                completed,
                has_errors,
            } => {
                self.pending_echo = None;
                self.live_content_id = None;
                self.turn_had_errors |= has_errors;
                self.turn = if completed {
                    TurnState::Complete
                } else {
                    TurnState::Failed
                };
                let text = if !completed {
                    "Task incomplete — completion was not confirmed"
                } else if self.turn_had_errors {
                    "Task complete — reported errors occurred during the turn"
                } else {
                    "Task complete"
                };
                self.push(
                    if completed {
                        EntryKind::System
                    } else {
                        EntryKind::Error
                    },
                    text.into(),
                    Some(completed),
                    false,
                );
            }
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
                self.entry_heights.remove(&id);
                self.enforce_bounds();
                return;
            }
        }
        if !text.is_empty() {
            self.push(EntryKind::Assistant, text, None, false);
        }
    }

    fn record_thinking(&mut self, text: String, reasoning_tokens: Option<u64>) {
        if !text.is_empty() {
            let mut appended_id = None;
            if let Some(id) = self.thinking_entry_id {
                if let Some(entry) = self
                    .timeline
                    .iter_mut()
                    .find(|entry| entry.id == id && entry.live)
                {
                    entry.text.push_str(&text);
                    entry.interrupted = false;
                    appended_id = Some(id);
                }
            }
            if let Some(id) = appended_id {
                self.timeline_chars += text.len();
                self.entry_heights.remove(&id);
                self.enforce_bounds();
            } else {
                let id = self.push(EntryKind::Thinking, text, None, true);
                self.thinking_entry_id = Some(id);
            }
        }
        if let Some(tokens) = reasoning_tokens {
            if let Some(id) = self.thinking_entry_id {
                let mut updated = false;
                if let Some(entry) = self
                    .timeline
                    .iter_mut()
                    .find(|entry| entry.id == id && entry.live)
                {
                    entry.reasoning_tokens = Some(tokens);
                    updated = true;
                }
                if updated {
                    self.entry_heights.remove(&id);
                }
            }
        }
    }

    fn finish_thinking(&mut self) {
        let finished = self
            .timeline
            .iter_mut()
            .filter_map(|entry| {
                if entry.kind == EntryKind::Thinking && entry.live {
                    entry.live = false;
                    entry.interrupted = false;
                    Some(entry.id)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        for id in finished {
            self.entry_heights.remove(&id);
        }
    }

    fn interrupt_thinking(&mut self) {
        let interrupted = self
            .timeline
            .iter_mut()
            .filter_map(|entry| {
                if entry.kind == EntryKind::Thinking && entry.live {
                    entry.live = false;
                    entry.interrupted = true;
                    Some(entry.id)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        for id in interrupted {
            self.entry_heights.remove(&id);
        }
    }

    fn record_outcome(&mut self, kind: EntryKind, message: String, success: bool) {
        self.turn_had_errors |= !success;
        self.push(kind, message, Some(success), false);
    }

    fn complete_turn(&mut self) {
        self.finish_thinking();
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
        self.interrupt_thinking();
        self.turn_had_errors = true;
        self.turn = TurnState::Failed;
        self.live_content_id = None;
        self.push(EntryKind::Error, error, Some(false), false);
    }

    fn append_to(&mut self, id: u64, text: &str) -> bool {
        let Some(entry) = self.timeline.iter_mut().find(|entry| entry.id == id) else {
            return false;
        };
        entry.text.push_str(text);
        self.timeline_chars += text.len();
        self.entry_heights.remove(&id);
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
            reasoning_tokens: None,
            interrupted: false,
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
            let measured_height = self.entry_heights.remove(&removed.id);
            if !self.follow {
                if let Some(height) = measured_height {
                    let separator = usize::from(!self.timeline.is_empty());
                    self.scroll_top = self
                        .scroll_top
                        .saturating_sub(height.saturating_add(separator));
                }
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
            ServerEvent::parse(r#"{"type":"thinking","text":"checking"}"#).unwrap(),
            ServerEvent::Thinking {
                text: "checking".to_string(),
                reasoning_tokens: None,
            }
        );
        assert_eq!(
            ServerEvent::parse(
                r#"{"type":"thinking","text":"checking","reasoning_tokens":12,"completion_tokens":900}"#
            )
            .unwrap(),
            ServerEvent::Thinking {
                text: "checking".to_string(),
                reasoning_tokens: Some(12),
            }
        );
        assert_eq!(
            ServerEvent::parse(
                r#"{"type":"thinking","text":"checking","usage":{"reasoning_tokens":13,"total_tokens":900}}"#
            )
            .unwrap(),
            ServerEvent::Thinking {
                text: "checking".to_string(),
                reasoning_tokens: Some(13),
            }
        );
        assert_eq!(
            ServerEvent::parse(
                r#"{"type":"thinking","text":"checking","completion_tokens":900,"total_tokens":901}"#
            )
            .unwrap(),
            ServerEvent::Thinking {
                text: "checking".to_string(),
                reasoning_tokens: None,
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
        assert!(!state
            .timeline
            .iter()
            .any(|entry| entry.text.contains("Working directory")));
        assert!(state
            .timeline
            .iter()
            .any(|entry| { entry.kind == EntryKind::System && entry.text == "Iteration 3/10" }));
    }

    #[test]
    fn reduces_stream_into_typed_entries_and_replaces_live_content() {
        let mut state = AppState::new("http://local".to_string());
        state.begin_turn("build it".to_string());
        state.apply(ClientMessage::Stream(ServerEvent::Start(
            "build it".to_string(),
        )));
        state.apply(ClientMessage::Stream(ServerEvent::Thinking {
            text: "checking".to_string(),
            reasoning_tokens: None,
        }));
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
        assert!(state
            .timeline
            .iter()
            .any(|e| { e.kind == EntryKind::Assistant && e.text == "final answer" && !e.live }));
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
        assert!(disconnected
            .timeline
            .back()
            .unwrap()
            .text
            .contains("before"));
    }

    #[test]
    fn native_completion_is_distinct_from_stream_end_and_corrected_errors() {
        for (completed, has_errors, expected) in [
            (false, false, TurnState::Failed),
            (false, true, TurnState::Failed),
            (true, true, TurnState::Complete),
        ] {
            let mut state = AppState::new("http://local".into());
            state.begin_turn("go".into());
            let event =
                serde_json::json!({"type":"done","completed":completed,"has_errors":has_errors});
            state.apply(ClientMessage::Stream(
                ServerEvent::parse(&event.to_string()).unwrap(),
            ));
            assert_eq!(state.turn, expected);
            assert_eq!(state.timeline.back().unwrap().success, Some(completed));
            assert_eq!(
                state.timeline.back().unwrap().text.contains("incomplete"),
                !completed
            );
        }
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
    fn thinking_stays_preserved_while_hidden_and_finishes_on_content() {
        let mut state = AppState::new("http://local".to_string());
        assert!(!state.show_thinking);
        state.begin_turn("go".to_string());
        state.apply(ClientMessage::Stream(ServerEvent::Thinking {
            text: "reasoning-secret".to_string(),
            reasoning_tokens: None,
        }));
        let thinking = state
            .timeline
            .iter()
            .find(|entry| entry.kind == EntryKind::Thinking)
            .expect("thinking entry");
        assert!(thinking.live);
        assert!(!thinking.interrupted);
        assert_eq!(thinking.text, "reasoning-secret");

        state.apply(ClientMessage::Stream(ServerEvent::Content(
            "answer".to_string(),
        )));
        let thinking = state
            .timeline
            .iter()
            .find(|entry| entry.kind == EntryKind::Thinking)
            .expect("thinking entry");
        assert!(!thinking.live);
        assert!(!thinking.interrupted);
        assert_eq!(thinking.text, "reasoning-secret");
        assert_eq!(state.turn, TurnState::Streaming);
    }

    #[test]
    fn exact_reasoning_counts_update_without_turning_content_counts_into_reasoning() {
        let mut state = AppState::new("http://local".to_string());
        state.begin_turn("go".to_string());
        state.apply(ClientMessage::Stream(ServerEvent::Thinking {
            text: "reasoning".to_string(),
            reasoning_tokens: None,
        }));
        state.apply(ClientMessage::Stream(ServerEvent::Thinking {
            text: String::new(),
            reasoning_tokens: Some(21),
        }));
        let thinking = state
            .timeline
            .iter()
            .find(|entry| entry.kind == EntryKind::Thinking)
            .expect("thinking entry");
        assert_eq!(thinking.reasoning_tokens, Some(21));
        assert_eq!(thinking.text, "reasoning");
    }

    #[test]
    fn late_token_only_event_does_not_mutate_completed_reasoning() {
        let mut state = AppState::new("http://local".to_string());
        state.begin_turn("go".to_string());
        state.apply(ClientMessage::Stream(ServerEvent::Thinking {
            text: "reasoning".to_string(),
            reasoning_tokens: None,
        }));
        state.apply(ClientMessage::Stream(ServerEvent::Content(
            "answer".to_string(),
        )));
        state.apply(ClientMessage::Stream(ServerEvent::Thinking {
            text: String::new(),
            reasoning_tokens: Some(999),
        }));
        let thinking = state
            .timeline
            .iter()
            .find(|entry| entry.kind == EntryKind::Thinking)
            .expect("thinking entry");
        assert!(!thinking.live);
        assert_eq!(thinking.reasoning_tokens, None);
        assert_eq!(thinking.text, "reasoning");
    }

    #[test]
    fn failed_turn_marks_open_thinking_as_interrupted() {
        let mut state = AppState::new("http://local".to_string());
        state.begin_turn("go".to_string());
        state.apply(ClientMessage::Stream(ServerEvent::Thinking {
            text: "partial reasoning".to_string(),
            reasoning_tokens: None,
        }));
        state.apply(ClientMessage::StreamClosed);
        let thinking = state
            .timeline
            .iter()
            .find(|entry| entry.kind == EntryKind::Thinking)
            .expect("thinking entry");
        assert!(!thinking.live);
        assert!(thinking.interrupted);
        assert_eq!(state.turn, TurnState::Failed);
    }

    #[test]
    fn metadata_events_do_not_finish_open_reasoning() {
        let mut state = AppState::new("http://local".to_string());
        state.begin_turn("go".to_string());
        state.apply(ClientMessage::Stream(ServerEvent::Thinking {
            text: "partial reasoning".to_string(),
            reasoning_tokens: None,
        }));
        for event in [
            ServerEvent::Start(String::new()),
            ServerEvent::User,
            ServerEvent::Iteration {
                detail: "iteration 2/10".to_string(),
                current: Some(2),
            },
            ServerEvent::Unknown {
                kind: "usage".to_string(),
                payload: "{}".to_string(),
            },
        ] {
            state.apply(ClientMessage::Stream(event));
            let thinking = state
                .timeline
                .iter()
                .find(|entry| entry.kind == EntryKind::Thinking)
                .expect("thinking entry");
            assert!(thinking.live);
            assert!(!thinking.interrupted);
        }

        state.apply(ClientMessage::Stream(ServerEvent::Thinking {
            text: " continued".to_string(),
            reasoning_tokens: None,
        }));
        let thinking_entries = state
            .timeline
            .iter()
            .filter(|entry| entry.kind == EntryKind::Thinking)
            .collect::<Vec<_>>();
        assert_eq!(thinking_entries.len(), 1);
        assert_eq!(thinking_entries[0].text, "partial reasoning continued");
        assert!(thinking_entries[0].live);

        state.apply(ClientMessage::Stream(ServerEvent::Content(
            "answer".to_string(),
        )));
        let thinking = state
            .timeline
            .iter()
            .find(|entry| entry.kind == EntryKind::Thinking)
            .expect("thinking entry");
        assert!(!thinking.live);
        assert_eq!(thinking.text, "partial reasoning continued");
    }

    #[test]
    fn reasoning_counts_do_not_update_a_previous_turn() {
        let mut state = AppState::new("http://local".to_string());
        state.begin_turn("one".to_string());
        state.apply(ClientMessage::Stream(ServerEvent::Thinking {
            text: "old reasoning".to_string(),
            reasoning_tokens: None,
        }));
        state.apply(ClientMessage::Stream(ServerEvent::Complete));
        state.begin_turn("two".to_string());
        state.apply(ClientMessage::Stream(ServerEvent::Thinking {
            text: String::new(),
            reasoning_tokens: Some(99),
        }));
        let thinking = state
            .timeline
            .iter()
            .find(|entry| entry.kind == EntryKind::Thinking)
            .expect("thinking entry");
        assert_eq!(thinking.text, "old reasoning");
        assert_eq!(thinking.reasoning_tokens, None);
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
