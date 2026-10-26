use std::collections::HashMap;

use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui::Frame;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use super::state::{AppState, EntryKind, ServerConnection, TimelineEntry, TurnState};

const ACCENT: Color = Color::Rgb(76, 201, 240);
const MUTED: Color = Color::Rgb(125, 135, 150);
const PANEL: Color = Color::Rgb(42, 49, 61);

pub(crate) fn draw(frame: &mut Frame<'_>, state: &mut AppState) {
    let composer_height = composer_height(state, frame.area().width);
    let areas = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(4),
            Constraint::Length(composer_height),
            Constraint::Length(1),
        ])
        .split(frame.area());

    render_header(frame, state, areas[0]);
    render_timeline(frame, state, areas[1]);
    render_composer(frame, state, areas[2]);
    render_footer(frame, state, areas[3]);
    if state.help_visible {
        render_help(frame);
    }
}

fn render_header(frame: &mut Frame<'_>, state: &AppState, area: Rect) {
    let (connection, connection_style) = match &state.connection {
        ServerConnection::Connecting => {
            ("connecting".to_string(), Style::default().fg(Color::Yellow))
        }
        ServerConnection::Online => ("connected".to_string(), Style::default().fg(Color::Green)),
        ServerConnection::Offline(error) => (
            format!("offline: {}", one_line(error, 52)),
            Style::default().fg(Color::Red),
        ),
    };
    let status = state.status.as_ref();
    let revision = status.map_or(0, |status| status.revision);
    let functions = status.map_or(0, |status| status.functions.len());
    let plan = status.map_or((0, 0), |status| {
        (
            status
                .plan
                .iter()
                .filter(|step| step.status == "done")
                .count(),
            status.plan.len(),
        )
    });
    let line = Line::from(vec![
        Span::styled(
            " ADAPSIS ",
            Style::default().fg(Color::Black).bg(ACCENT).bold(),
        ),
        Span::raw("  "),
        Span::styled(connection, connection_style.bold()),
        Span::styled(
            format!(
                "  rev {revision}  fn {functions}  plan {}/{}",
                plan.0, plan.1
            ),
            Style::default().fg(MUTED),
        ),
        Span::styled(
            format!("  {}", one_line(&state.api_url, 36)),
            Style::default().fg(PANEL),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(line).block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(PANEL)),
        ),
        area,
    );
}

fn render_timeline(frame: &mut Frame<'_>, state: &mut AppState, area: Rect) {
    let inner_width = area.width.saturating_sub(2).max(1);
    let (text, measured_heights) = timeline_text(state, inner_width);
    state.entry_heights = measured_heights;
    let paragraph = Paragraph::new(text)
        .block(
            Block::default()
                .title(" timeline ")
                .title_style(Style::default().fg(ACCENT))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(PANEL)),
        )
        .wrap(Wrap { trim: false });
    // Exact wrapped line count from ratatui's own word wrapper, minus the two
    // border rows the paragraph reports. A hand-rolled estimate under-counts
    // word-wrapped text, which shortened `bottom` and left the newest line
    // (e.g. a just-submitted echo) below the viewport even in follow mode.
    let total_lines = paragraph.line_count(inner_width).saturating_sub(2);
    let viewport_lines = area.height.saturating_sub(2) as usize;
    state.update_layout(total_lines, viewport_lines);
    let scroll = state.scroll_top.min(u16::MAX as usize) as u16;
    frame.render_widget(paragraph.scroll((scroll, 0)), area);
}

fn timeline_text(state: &AppState, inner_width: u16) -> (Text<'static>, HashMap<u64, usize>) {
    if state.timeline.is_empty() {
        return (
            Text::from(Line::styled(
                "  Send a natural-language request to begin.",
                Style::default().fg(MUTED).italic(),
            )),
            HashMap::new(),
        );
    }
    let mut lines = Vec::new();
    let mut heights = HashMap::new();
    for entry in &state.timeline {
        let block = entry_lines(entry, state.show_thinking);
        if block.is_empty() {
            continue;
        }
        if !lines.is_empty() {
            lines.push(Line::raw(""));
        }
        let height = Paragraph::new(block.clone())
            .wrap(Wrap { trim: false })
            .line_count(inner_width);
        heights.insert(entry.id, height);
        lines.extend(block);
    }
    (Text::from(lines), heights)
}

fn entry_lines(entry: &TimelineEntry, show_thinking: bool) -> Vec<Line<'static>> {
    let (label, style) = entry_style(entry);
    if entry.kind == EntryKind::Thinking {
        return thinking_lines(entry, label, style, show_thinking);
    }
    let text_lines = display_lines(&entry.text, entry.kind == EntryKind::Code);
    let first = text_lines.first().cloned().unwrap_or_default();
    let live = if entry.live { " ..." } else { "" };
    let mut lines = vec![Line::from(vec![
        Span::styled(format!(" {label:<8}"), style.add_modifier(Modifier::BOLD)),
        Span::styled(first, style),
        Span::styled(live, Style::default().fg(MUTED)),
    ])];
    lines.extend(text_lines.into_iter().skip(1).map(|line| {
        if line.is_empty() {
            Line::raw("")
        } else {
            Line::from(vec![Span::raw("          "), Span::styled(line, style)])
        }
    }));
    lines
}

fn thinking_lines(
    entry: &TimelineEntry,
    label: &'static str,
    style: Style,
    show_thinking: bool,
) -> Vec<Line<'static>> {
    let state = if entry.live {
        "Thinking\u{2026}"
    } else if entry.interrupted {
        "Thinking (interrupted)"
    } else {
        "Thought"
    };
    let summary = match thinking_token_label(entry) {
        Some(count) => format!("{state} ({count})"),
        None => state.to_string(),
    };
    let mut lines = vec![Line::from(vec![
        Span::styled(format!(" {label:<8}"), style.add_modifier(Modifier::BOLD)),
        Span::styled(summary, style),
    ])];
    if show_thinking {
        lines.extend(display_lines(&entry.text, false).into_iter().map(|line| {
            if line.is_empty() {
                Line::raw("")
            } else {
                Line::from(vec![Span::raw("          "), Span::styled(line, style)])
            }
        }));
    }
    lines
}

fn thinking_token_label(entry: &TimelineEntry) -> Option<String> {
    if let Some(tokens) = entry.reasoning_tokens {
        return Some(format!("{tokens} tokens"));
    }
    let estimate = estimated_reasoning_tokens(&entry.text);
    (estimate > 0).then(|| format!("~{estimate} tokens, estimated"))
}

fn estimated_reasoning_tokens(text: &str) -> u64 {
    let characters = text
        .chars()
        .filter(|character| !character.is_whitespace())
        .count();
    characters as u64 / 4
}

fn display_lines(text: &str, preserve_blank_runs: bool) -> Vec<String> {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut lines = Vec::new();
    let mut in_fence = false;
    for line in normalized.split('\n') {
        let trimmed = line.trim_start();
        let fence = trimmed.starts_with("```") || trimmed.starts_with("~~~");
        if fence {
            in_fence = !in_fence;
            lines.push(line.to_string());
        } else if !in_fence && !preserve_blank_runs && line.trim().is_empty() {
            if lines
                .last()
                .is_some_and(|previous| previous.trim().is_empty())
            {
                continue;
            }
            lines.push(String::new());
        } else {
            lines.push(line.to_string());
        }
    }
    while lines.last().is_some_and(|line| line.trim().is_empty()) {
        lines.pop();
    }
    if !preserve_blank_runs {
        while lines.first().is_some_and(|line| line.trim().is_empty()) {
            lines.remove(0);
        }
    }
    lines
}

fn entry_style(entry: &TimelineEntry) -> (&'static str, Style) {
    let base = Style::default();
    match entry.kind {
        EntryKind::User => ("YOU", base.fg(Color::White)),
        EntryKind::Thinking => ("THINK", base.fg(MUTED).italic()),
        EntryKind::Assistant => ("ADAPSIS", base.fg(Color::Rgb(220, 226, 235))),
        EntryKind::Code => ("CODE", base.fg(Color::Cyan)),
        EntryKind::Result => outcome_style("RESULT", entry.success),
        EntryKind::Test => outcome_style("TEST", entry.success),
        EntryKind::Eval => ("EVAL", base.fg(Color::Magenta)),
        EntryKind::Query => ("QUERY", base.fg(Color::Blue)),
        EntryKind::Feedback => ("FEEDBACK", base.fg(Color::Yellow)),
        EntryKind::Error => ("ERROR", base.fg(Color::Red)),
        EntryKind::System => ("SYSTEM", base.fg(MUTED)),
    }
}

fn outcome_style(label: &'static str, success: Option<bool>) -> (&'static str, Style) {
    let color = match success {
        Some(true) => Color::Green,
        Some(false) => Color::Red,
        None => MUTED,
    };
    (label, Style::default().fg(color))
}

fn render_composer(frame: &mut Frame<'_>, state: &AppState, area: Rect) {
    let streaming = state.turn == TurnState::Streaming;
    let title = if streaming {
        " compose (turn streaming; draft retained) "
    } else {
        " compose "
    };
    let value = if state.composer.is_empty() {
        Text::from(Line::styled(
            "Describe what you want Adapsis to do...",
            Style::default().fg(MUTED).italic(),
        ))
    } else {
        Text::raw(state.composer.text())
    };
    frame.render_widget(
        Paragraph::new(value)
            .block(
                Block::default()
                    .title(title)
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(if streaming { MUTED } else { ACCENT })),
            )
            .wrap(Wrap { trim: false }),
        area,
    );
    if !state.help_visible {
        place_cursor(frame, state, area);
    }
}

fn place_cursor(frame: &mut Frame<'_>, state: &AppState, area: Rect) {
    let width = area.width.saturating_sub(2).max(1) as usize;
    let prefix: String = state
        .composer
        .text()
        .chars()
        .take(state.composer.cursor())
        .collect();
    let mut row = 0usize;
    let mut column = 0usize;
    for character in prefix.chars() {
        if character == '\n' {
            row += 1;
            column = 0;
            continue;
        }
        let character_width = UnicodeWidthChar::width(character).unwrap_or(0);
        if column > 0 && column + character_width > width {
            row += 1;
            column = 0;
        }
        column += character_width;
        row += column / width;
        column %= width;
    }
    let max_row = area.height.saturating_sub(2) as usize;
    frame.set_cursor_position((
        area.x + 1 + column.min(width.saturating_sub(1)) as u16,
        area.y + 1 + row.min(max_row.saturating_sub(1)) as u16,
    ));
}

fn render_footer(frame: &mut Frame<'_>, state: &AppState, area: Rect) {
    let follow = if state.follow { "FOLLOW" } else { "REVIEW" };
    let turn = match state.turn {
        TurnState::Idle => "idle",
        TurnState::Streaming => "streaming",
        TurnState::Complete => "complete",
        TurnState::Failed => "error",
    };
    let line = Line::from(vec![
        Span::styled(format!(" {follow} "), Style::default().fg(ACCENT).bold()),
        Span::styled(format!("{turn} "), Style::default().fg(MUTED)),
        Span::styled(
            format!("THINK:{} ", if state.show_thinking { "on" } else { "off" }),
            Style::default().fg(if state.show_thinking {
                Color::Yellow
            } else {
                MUTED
            }),
        ),
        Span::raw(
            "Enter send  Alt+Enter newline  Ctrl+T thinking  PgUp/PgDn scroll  Ctrl+F latest  F1 help  Ctrl+Q quit",
        ),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

fn render_help(frame: &mut Frame<'_>) {
    let area = centered_rect(68, 72, frame.area());
    let help = Text::from(vec![
        Line::styled("Adapsis TUI", Style::default().fg(ACCENT).bold()),
        Line::raw(""),
        Line::raw("Enter             Send the current natural-language request"),
        Line::raw("Alt/Shift+Enter   Insert a newline (Ctrl+J also works)"),
        Line::raw("PageUp/PageDown   Review older/newer timeline output"),
        Line::raw("Alt+Up/Alt+Down   Scroll one line"),
        Line::raw("Ctrl+F            Explicitly resume following latest output"),
        Line::raw("Ctrl+T            Show or hide model reasoning"),
        Line::raw("Esc               Close help or clear the composer"),
        Line::raw("Ctrl+Q / Ctrl+C   Quit (disconnects this client only)"),
        Line::raw(""),
        Line::styled(
            "The daemon remains authoritative. Closing the client does not claim to cancel server work.",
            Style::default().fg(Color::Yellow),
        ),
        Line::raw(""),
        Line::styled("F1 or Esc to close", Style::default().fg(MUTED)),
    ]);
    frame.render_widget(Clear, area);
    frame.render_widget(
        Paragraph::new(help)
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: false })
            .block(
                Block::default()
                    .title(" help ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(ACCENT)),
            ),
        area,
    );
}

fn composer_height(state: &AppState, terminal_width: u16) -> u16 {
    let width = terminal_width.saturating_sub(2).max(1) as usize;
    let rows: usize = state
        .composer
        .text()
        .split('\n')
        .map(|line| UnicodeWidthStr::width(line).max(1).div_ceil(width))
        .sum();
    (rows as u16 + 2).clamp(3, 8)
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let vertical = Layout::vertical([
        Constraint::Percentage((100 - percent_y) / 2),
        Constraint::Percentage(percent_y),
        Constraint::Percentage((100 - percent_y) / 2),
    ])
    .split(area);
    Layout::horizontal([
        Constraint::Percentage((100 - percent_x) / 2),
        Constraint::Percentage(percent_x),
        Constraint::Percentage((100 - percent_x) / 2),
    ])
    .split(vertical[1])[1]
}

fn one_line(value: &str, limit: usize) -> String {
    let flattened = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if flattened.chars().count() <= limit {
        flattened
    } else {
        format!(
            "{}...",
            flattened
                .chars()
                .take(limit.saturating_sub(3))
                .collect::<String>()
        )
    }
}

#[cfg(test)]
mod tests {
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    use super::*;
    use crate::tui::state::{ClientMessage, ServerEvent};

    #[test]
    fn renders_narrow_and_resized_layouts() {
        for (width, height) in [(28, 8), (100, 35)] {
            let backend = TestBackend::new(width, height);
            let mut terminal = Terminal::new(backend).unwrap();
            let mut state = AppState::new("http://127.0.0.1:3002".to_string());
            state.begin_turn("A multiline\nrequest".to_string());
            state.apply(ClientMessage::Stream(ServerEvent::Thinking {
                text: "Inspecting the current state".to_string(),
                reasoning_tokens: None,
            }));
            state.help_visible = true;
            terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        }
    }

    fn buffer_text(terminal: &Terminal<TestBackend>) -> String {
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect()
    }

    fn buffer_rows(terminal: &Terminal<TestBackend>) -> Vec<String> {
        let buffer = terminal.backend().buffer();
        let width = buffer.area.width as usize;
        buffer
            .content()
            .chunks(width)
            .map(|row| row.iter().map(|cell| cell.symbol()).collect())
            .collect()
    }

    fn row_of(rows: &[String], needle: &str) -> usize {
        rows.iter()
            .position(|row| row.contains(needle))
            .unwrap_or_else(|| panic!("missing {needle:?} in rows:\n{rows:#?}"))
    }

    fn blank_timeline_row(row: &str) -> bool {
        row.chars()
            .all(|character| character == '\u{2502}' || character.is_whitespace())
    }

    #[test]
    fn follow_keeps_the_latest_echo_visible_after_word_wrapping() {
        let backend = TestBackend::new(40, 14);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut state = AppState::new("http://127.0.0.1:3002".to_string());
        state.show_thinking = true;

        // Long word-wrapped turn before the echo. The old character-count
        // estimate under-counted its wrapped height, so follow-mode scroll (and
        // the just-submitted echo) ended up below the viewport.
        let long = (0..60)
            .map(|n| format!("w{n:07}"))
            .collect::<Vec<_>>()
            .join(" ");
        state.apply(ClientMessage::Stream(ServerEvent::Thinking {
            text: long,
            reasoning_tokens: None,
        }));
        state.apply(ClientMessage::Stream(ServerEvent::Complete));

        // Enter echoes the user message and resumes following the latest output.
        state.begin_turn("ECHO-MARKER-12345".to_string());
        assert!(state.follow);

        terminal.draw(|frame| draw(frame, &mut state)).unwrap();

        assert!(
            buffer_text(&terminal).contains("ECHO-MARKER-12345"),
            "the newest entry (the echo) must be visible in follow mode:\n{}",
            buffer_text(&terminal)
        );
    }

    #[test]
    fn hidden_thinking_never_leaks_and_the_row_reports_state_and_count() {
        let backend = TestBackend::new(64, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut state = AppState::new("http://127.0.0.1:3002".to_string());
        state.begin_turn("go".to_string());
        state.apply(ClientMessage::Stream(ServerEvent::Thinking {
            text: "SECRET-REASONING".to_string(),
            reasoning_tokens: None,
        }));

        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        let visible = buffer_text(&terminal);
        assert!(!visible.contains("SECRET-REASONING"), "{visible}");
        assert!(visible.contains("Thinking\u{2026}"), "{visible}");
        assert!(visible.contains("~4 tokens, estimated"), "{visible}");
        assert!(visible.contains("THINK:off"), "{visible}");

        state.show_thinking = true;
        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        let visible = buffer_text(&terminal);
        assert!(visible.contains("SECRET-REASONING"), "{visible}");
        assert!(visible.contains("THINK:on"), "{visible}");

        state.show_thinking = false;
        state.apply(ClientMessage::Stream(ServerEvent::Content(
            "answer".to_string(),
        )));
        state.apply(ClientMessage::Stream(ServerEvent::Text(
            "final answer".to_string(),
        )));
        state.apply(ClientMessage::Stream(ServerEvent::Complete));
        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        let visible = buffer_text(&terminal);
        assert!(!visible.contains("SECRET-REASONING"), "{visible}");
        assert!(visible.contains("Thought"), "{visible}");
        println!("{}", buffer_rows(&terminal).join("\n"));

        state.show_thinking = true;
        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        let visible = buffer_text(&terminal);
        assert!(visible.contains("SECRET-REASONING"), "{visible}");
        assert!(visible.contains("Thought"), "{visible}");
        println!("{}", buffer_rows(&terminal).join("\n"));
    }

    #[test]
    fn token_counts_are_exact_only_when_reasoning_tokens_are_supplied() {
        let backend = TestBackend::new(64, 14);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut state = AppState::new("http://127.0.0.1:3002".to_string());
        state.apply(ClientMessage::Stream(ServerEvent::Thinking {
            text: "short".to_string(),
            reasoning_tokens: Some(123),
        }));
        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        let visible = buffer_text(&terminal);
        assert!(visible.contains("123 tokens"), "{visible}");
        assert!(!visible.contains("estimated"), "{visible}");

        let mut estimated = AppState::new("http://127.0.0.1:3002".to_string());
        estimated.apply(ClientMessage::Stream(
            ServerEvent::parse(
                r#"{"type":"thinking","text":"abcdefgh","completion_tokens":999,"total_tokens":1000}"#,
            )
            .unwrap(),
        ));
        let mut terminal = Terminal::new(TestBackend::new(64, 14)).unwrap();
        terminal.draw(|frame| draw(frame, &mut estimated)).unwrap();
        let visible = buffer_text(&terminal);
        assert!(!visible.contains("999"), "{visible}");
        assert!(visible.contains("~2 tokens, estimated"), "{visible}");
    }

    #[test]
    fn paragraph_gaps_collapse_while_fenced_and_code_whitespace_is_preserved() {
        let mut state = AppState::new("http://127.0.0.1:3002".to_string());
        state.apply(ClientMessage::Stream(ServerEvent::Text(
            "PARA-ONE\n\n\nPARA-TWO".to_string(),
        )));
        let mut terminal = Terminal::new(TestBackend::new(60, 20)).unwrap();
        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        let rows = buffer_rows(&terminal);
        let first = row_of(&rows, "PARA-ONE");
        let second = row_of(&rows, "PARA-TWO");
        assert_eq!(second, first + 2, "{rows:#?}");
        assert!(blank_timeline_row(&rows[first + 1]), "{rows:#?}");

        let mut fenced = AppState::new("http://127.0.0.1:3002".to_string());
        fenced.apply(ClientMessage::Stream(ServerEvent::Text(
            "INTRO\n\n```\nALPHA\n\n\nBETA\n```\n\n\nOUTRO".to_string(),
        )));
        let mut terminal = Terminal::new(TestBackend::new(60, 20)).unwrap();
        terminal.draw(|frame| draw(frame, &mut fenced)).unwrap();
        let rows = buffer_rows(&terminal);
        let alpha = row_of(&rows, "ALPHA");
        let beta = row_of(&rows, "BETA");
        assert_eq!(beta, alpha + 3, "{rows:#?}");
        assert!(blank_timeline_row(&rows[alpha + 1]), "{rows:#?}");
        assert!(blank_timeline_row(&rows[alpha + 2]), "{rows:#?}");
        assert_eq!(row_of(&rows, "OUTRO"), beta + 3, "{rows:#?}");

        let mut code = AppState::new("http://127.0.0.1:3002".to_string());
        code.apply(ClientMessage::Stream(ServerEvent::Code(
            "CODE-A\n\n\nCODE-B".to_string(),
        )));
        let mut terminal = Terminal::new(TestBackend::new(60, 16)).unwrap();
        terminal.draw(|frame| draw(frame, &mut code)).unwrap();
        let rows = buffer_rows(&terminal);
        assert_eq!(
            row_of(&rows, "CODE-B"),
            row_of(&rows, "CODE-A") + 3,
            "{rows:#?}"
        );
    }

    #[test]
    fn measured_entry_heights_follow_collapsed_display_lines() {
        let mut terminal = Terminal::new(TestBackend::new(40, 14)).unwrap();
        let mut state = AppState::new("http://127.0.0.1:3002".to_string());
        state.apply(ClientMessage::Stream(ServerEvent::Text(
            "ALPHA\n\n\n\n\nBETA".to_string(),
        )));
        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        let id = state.timeline[0].id;
        assert_eq!(state.entry_heights.get(&id), Some(&3));
    }

    #[test]
    fn hidden_thinking_eviction_keeps_a_narrow_review_viewport_stable() {
        let mut terminal = Terminal::new(TestBackend::new(18, 12)).unwrap();
        let mut state = AppState::new("http://127.0.0.1:3002".to_string());
        for n in 0..260 {
            state.apply(ClientMessage::Stream(ServerEvent::Thinking {
                text: format!("TH-{n}"),
                reasoning_tokens: None,
            }));
            state.apply(ClientMessage::Stream(ServerEvent::Feedback(format!(
                "MARK-{n}"
            ))));
        }
        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        state.scroll_up(5);
        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        assert!(!state.follow);
        let before = buffer_rows(&terminal)[4..7].to_vec();
        assert!(
            before
                .iter()
                .any(|row| row.contains("THINK") || row.contains("FEEDBACK")),
            "{before:#?}"
        );

        for n in 0..10 {
            state.apply(ClientMessage::Stream(ServerEvent::Feedback(format!(
                "NEW-{n}"
            ))));
        }
        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        assert!(!state.follow);
        assert!(
            state.scroll_top
                <= state
                    .layout_total_lines
                    .saturating_sub(state.layout_viewport_lines)
        );
        let after = buffer_rows(&terminal)[4..7].to_vec();
        assert_eq!(after, before, "viewport content should stay anchored");
        println!("F1 entry-limit before:\n{}", before.join("\n"));
        println!("F1 entry-limit after:\n{}", after.join("\n"));
    }

    #[test]
    fn hidden_thinking_char_eviction_keeps_a_narrow_review_viewport_stable() {
        let mut terminal = Terminal::new(TestBackend::new(18, 12)).unwrap();
        let mut state = AppState::new("http://127.0.0.1:3002".to_string());
        for n in 0..4 {
            state.apply(ClientMessage::Stream(ServerEvent::Thinking {
                text: "H".repeat(120_000),
                reasoning_tokens: None,
            }));
            state.apply(ClientMessage::Stream(ServerEvent::Feedback(format!(
                "HUGE-END-{n}"
            ))));
        }
        for n in 0..30 {
            state.apply(ClientMessage::Stream(ServerEvent::Thinking {
                text: format!("TH-{n}"),
                reasoning_tokens: None,
            }));
            state.apply(ClientMessage::Stream(ServerEvent::Feedback(format!(
                "MARK-{n}"
            ))));
        }
        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        state.scroll_up(5);
        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        assert!(!state.follow);
        let before = buffer_rows(&terminal)[4..7].to_vec();
        assert!(
            before
                .iter()
                .any(|row| row.contains("THINK") || row.contains("FEEDBACK")),
            "{before:#?}"
        );

        state.apply(ClientMessage::Stream(ServerEvent::Thinking {
            text: "J".repeat(120_000),
            reasoning_tokens: None,
        }));
        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        assert!(!state.follow);
        assert!(
            state.scroll_top
                <= state
                    .layout_total_lines
                    .saturating_sub(state.layout_viewport_lines)
        );
        let after = buffer_rows(&terminal)[4..7].to_vec();
        assert_eq!(
            after, before,
            "char eviction should keep the viewport anchored"
        );
        println!("F1 char-limit before:\n{}", before.join("\n"));
        println!("F1 char-limit after:\n{}", after.join("\n"));
    }

    #[test]
    fn late_token_event_does_not_change_the_narrow_rendered_count() {
        let mut terminal = Terminal::new(TestBackend::new(18, 18)).unwrap();
        let mut state = AppState::new("http://127.0.0.1:3002".to_string());
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
        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        let visible = buffer_text(&terminal);
        assert!(!visible.contains("999"), "{visible}");
        assert!(visible.contains("~2 tokens"), "{visible}");
        assert!(visible.contains("estimated"), "{visible}");
        println!("F2 narrow rows:\n{}", buffer_rows(&terminal).join("\n"));
    }

    #[test]
    fn hidden_thinking_survives_resize_and_expands_at_the_followed_bottom() {
        let mut terminal = Terminal::new(TestBackend::new(48, 16)).unwrap();
        let mut state = AppState::new("http://127.0.0.1:3002".to_string());
        state.begin_turn("go".to_string());
        state.apply(ClientMessage::Stream(ServerEvent::Thinking {
            text: "SECRET-REASONING".to_string(),
            reasoning_tokens: None,
        }));
        state.apply(ClientMessage::Stream(ServerEvent::Text(
            "answer".to_string(),
        )));
        state.apply(ClientMessage::Stream(ServerEvent::Complete));
        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        assert!(state.follow);
        assert!(!buffer_text(&terminal).contains("SECRET-REASONING"));

        terminal.backend_mut().resize(28, 16);
        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        assert!(state.follow);
        assert!(!buffer_text(&terminal).contains("SECRET-REASONING"));
        assert!(buffer_text(&terminal).contains("Thought"));
        assert!(buffer_text(&terminal).contains("THINK:off"));
        println!("{}", buffer_rows(&terminal).join("\n"));

        state.show_thinking = true;
        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        assert!(state.follow);
        assert!(buffer_text(&terminal).contains("SECRET-REASONING"));
        assert!(buffer_text(&terminal).contains("THINK:on"));
        println!("{}", buffer_rows(&terminal).join("\n"));

        state.show_thinking = false;
        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        assert!(state.follow);
        assert!(!buffer_text(&terminal).contains("SECRET-REASONING"));
    }

    #[test]
    fn toggling_in_review_mode_keeps_the_position_within_bounds() {
        let mut terminal = Terminal::new(TestBackend::new(40, 12)).unwrap();
        let mut state = AppState::new("http://127.0.0.1:3002".to_string());
        for n in 0..30 {
            state.apply(ClientMessage::Stream(ServerEvent::Feedback(format!(
                "filler {n}"
            ))));
        }
        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        state.scroll_up(4);
        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        assert!(!state.follow);
        let reviewed_top = state.scroll_top;
        assert!(reviewed_top > 0);

        state.show_thinking = true;
        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        assert!(!state.follow);
        assert!(state.scroll_top <= reviewed_top);
        assert!(
            state.scroll_top
                <= state
                    .layout_total_lines
                    .saturating_sub(state.layout_viewport_lines)
        );
    }

    #[test]
    fn footer_keeps_the_thinking_state_visible_on_a_narrow_streaming_turn() {
        let mut terminal = Terminal::new(TestBackend::new(28, 16)).unwrap();
        let mut state = AppState::new("http://127.0.0.1:3002".to_string());
        state.begin_turn("go".to_string());
        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        let footer = buffer_rows(&terminal).pop().expect("footer row");
        assert!(footer.contains("THINK:off"), "{footer:?}");
        state.show_thinking = true;
        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        let footer = buffer_rows(&terminal).pop().expect("footer row");
        assert!(footer.contains("THINK:on"), "{footer:?}");
    }

    #[test]
    fn help_and_footer_expose_the_thinking_toggle() {
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
        let mut state = AppState::new("http://127.0.0.1:3002".to_string());
        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        let visible = buffer_text(&terminal);
        assert!(visible.contains("THINK:off"), "{visible}");
        assert!(visible.contains("Ctrl+T thinking"), "{visible}");

        state.help_visible = true;
        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        let visible = buffer_text(&terminal);
        assert!(visible.contains("Ctrl+T"), "{visible}");
        assert!(
            visible.contains("Show or hide model reasoning"),
            "{visible}"
        );
    }

    #[test]
    fn a_turn_without_reasoning_has_no_thinking_or_thought_row() {
        let mut terminal = Terminal::new(TestBackend::new(64, 16)).unwrap();
        let mut state = AppState::new("http://127.0.0.1:3002".to_string());
        state.begin_turn("go".to_string());
        state.apply(ClientMessage::Stream(ServerEvent::Text(
            "answer".to_string(),
        )));
        state.apply(ClientMessage::Stream(ServerEvent::Complete));
        terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        let visible = buffer_text(&terminal);
        assert!(!visible.contains("Thought"), "{visible}");
        assert!(!visible.contains("Thinking\u{2026}"), "{visible}");
    }
}
