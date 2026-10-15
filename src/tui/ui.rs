use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
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
    let text = timeline_text(state);
    let paragraph = Paragraph::new(text)
        .block(
            Block::default()
                .title(" timeline ")
                .title_style(Style::default().fg(ACCENT))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(PANEL)),
        )
        .wrap(Wrap { trim: false });
    let total_lines = timeline_line_count(state, inner_width as usize);
    let viewport_lines = area.height.saturating_sub(2) as usize;
    state.update_layout(total_lines, viewport_lines);
    let scroll = state.scroll_top.min(u16::MAX as usize) as u16;
    frame.render_widget(paragraph.scroll((scroll, 0)), area);
}

fn timeline_text(state: &AppState) -> Text<'static> {
    if state.timeline.is_empty() {
        return Text::from(Line::styled(
            "  Send a natural-language request to begin.",
            Style::default().fg(MUTED).italic(),
        ));
    }
    let mut lines = Vec::new();
    for entry in &state.timeline {
        lines.extend(entry_lines(entry));
        lines.push(Line::raw(""));
    }
    Text::from(lines)
}

fn timeline_line_count(state: &AppState, width: usize) -> usize {
    if state.timeline.is_empty() {
        return 1;
    }
    state
        .timeline
        .iter()
        .map(|entry| {
            let live_width = if entry.live { 4 } else { 0 };
            entry
                .text
                .split('\n')
                .map(|line| {
                    (10 + UnicodeWidthStr::width(line) + live_width)
                        .max(1)
                        .div_ceil(width.max(1))
                })
                .sum::<usize>()
                + 1
        })
        .sum()
}

fn entry_lines(entry: &TimelineEntry) -> Vec<Line<'static>> {
    let (label, style) = entry_style(entry);
    let mut text_lines = entry.text.lines();
    let first = text_lines.next().unwrap_or_default().to_string();
    let live = if entry.live { " ..." } else { "" };
    let mut lines = vec![Line::from(vec![
        Span::styled(format!(" {label:<8}"), style.add_modifier(Modifier::BOLD)),
        Span::styled(first, style),
        Span::styled(live, Style::default().fg(MUTED)),
    ])];
    lines.extend(text_lines.map(|line| {
        Line::from(vec![
            Span::raw("          "),
            Span::styled(line.to_string(), style),
        ])
    }));
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
        Span::styled(format!("{turn}  "), Style::default().fg(MUTED)),
        Span::raw(
            "Enter send  Alt+Enter newline  PgUp/PgDn scroll  Ctrl+F latest  F1 help  Ctrl+Q quit",
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
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    use super::*;
    use crate::tui::state::{ClientMessage, ServerEvent};

    #[test]
    fn renders_narrow_and_resized_layouts() {
        for (width, height) in [(28, 8), (100, 35)] {
            let backend = TestBackend::new(width, height);
            let mut terminal = Terminal::new(backend).unwrap();
            let mut state = AppState::new("http://127.0.0.1:3002".to_string());
            state.begin_turn("A multiline\nrequest".to_string());
            state.apply(ClientMessage::Stream(ServerEvent::Thinking(
                "Inspecting the current state".to_string(),
            )));
            state.help_visible = true;
            terminal.draw(|frame| draw(frame, &mut state)).unwrap();
        }
    }
}
