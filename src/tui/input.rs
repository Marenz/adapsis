use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent, MouseEventKind};

use super::state::{AppState, TurnState};

#[derive(Default)]
pub(crate) struct Composer {
    text: Vec<char>,
    cursor: usize,
}

impl Composer {
    pub(crate) fn text(&self) -> String {
        self.text.iter().collect()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    pub(crate) fn cursor(&self) -> usize {
        self.cursor
    }

    pub(crate) fn insert(&mut self, character: char) {
        self.text.insert(self.cursor, character);
        self.cursor += 1;
    }

    pub(crate) fn insert_str(&mut self, value: &str) {
        for character in value.chars() {
            self.insert(character);
        }
    }

    fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.text.remove(self.cursor);
        }
    }

    fn delete(&mut self) {
        if self.cursor < self.text.len() {
            self.text.remove(self.cursor);
        }
    }

    fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
    }

    fn take_trimmed(&mut self) -> Option<String> {
        let value = self.text().trim().to_string();
        if value.is_empty() {
            return None;
        }
        self.clear();
        Some(value)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Action {
    None,
    Submit(String),
    Quit,
}

pub(crate) fn handle_key(state: &mut AppState, key: KeyEvent) -> Action {
    if matches!(key.kind, KeyEventKind::Release) {
        return Action::None;
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('q' | 'c'))
    {
        return Action::Quit;
    }
    if state.help_visible {
        if matches!(key.code, KeyCode::Esc | KeyCode::F(1)) {
            state.help_visible = false;
        }
        return Action::None;
    }
    if key.code == KeyCode::F(1)
        || (key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('h'))
    {
        state.help_visible = true;
        return Action::None;
    }
    if handle_timeline_key(state, key) {
        return Action::None;
    }
    handle_composer_key(state, key)
}

fn handle_timeline_key(state: &mut AppState, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::PageUp => state.scroll_up(state.page_size()),
        KeyCode::PageDown => state.scroll_down(state.page_size()),
        KeyCode::Up if key.modifiers.contains(KeyModifiers::ALT) => state.scroll_up(1),
        KeyCode::Down if key.modifiers.contains(KeyModifiers::ALT) => state.scroll_down(1),
        KeyCode::Char('f') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            state.follow_latest()
        }
        _ => return false,
    }
    true
}

fn handle_composer_key(state: &mut AppState, key: KeyEvent) -> Action {
    match key.code {
        KeyCode::Enter
            if key
                .modifiers
                .intersects(KeyModifiers::ALT | KeyModifiers::SHIFT) =>
        {
            state.composer.insert('\n');
        }
        KeyCode::Enter => {
            if state.turn != TurnState::Streaming {
                if let Some(message) = state.composer.take_trimmed() {
                    return Action::Submit(message);
                }
            }
        }
        KeyCode::Char('j') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            state.composer.insert('\n');
        }
        KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            state.composer.insert(character);
        }
        KeyCode::Backspace => state.composer.backspace(),
        KeyCode::Delete => state.composer.delete(),
        KeyCode::Left => state.composer.cursor = state.composer.cursor.saturating_sub(1),
        KeyCode::Right => {
            state.composer.cursor = (state.composer.cursor + 1).min(state.composer.text.len())
        }
        KeyCode::Home => state.composer.cursor = 0,
        KeyCode::End => state.composer.cursor = state.composer.text.len(),
        KeyCode::Esc => state.composer.clear(),
        _ => {}
    }
    Action::None
}

pub(crate) fn handle_mouse(state: &mut AppState, mouse: MouseEvent) {
    match mouse.kind {
        MouseEventKind::ScrollUp => state.scroll_up(3),
        MouseEventKind::ScrollDown => state.scroll_down(3),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent::new(code, modifiers)
    }

    #[test]
    fn composer_supports_newline_edit_and_submit() {
        let mut state = AppState::new("local".to_string());
        handle_key(&mut state, key(KeyCode::Char('a'), KeyModifiers::NONE));
        handle_key(&mut state, key(KeyCode::Enter, KeyModifiers::ALT));
        handle_key(&mut state, key(KeyCode::Char('b'), KeyModifiers::NONE));
        assert_eq!(state.composer.text(), "a\nb");
        assert_eq!(
            handle_key(&mut state, key(KeyCode::Enter, KeyModifiers::NONE)),
            Action::Submit("a\nb".to_string())
        );
        assert!(state.composer.is_empty());
    }

    #[test]
    fn streaming_blocks_a_second_submit_without_discarding_text() {
        let mut state = AppState::new("local".to_string());
        state.turn = TurnState::Streaming;
        state.composer.insert_str("wait");
        assert_eq!(
            handle_key(&mut state, key(KeyCode::Enter, KeyModifiers::NONE)),
            Action::None
        );
        assert_eq!(state.composer.text(), "wait");
    }

    #[test]
    fn help_and_follow_are_explicit_interactions() {
        let mut state = AppState::new("local".to_string());
        state.update_layout(100, 20);
        handle_key(&mut state, key(KeyCode::PageUp, KeyModifiers::NONE));
        assert!(!state.follow);
        handle_key(&mut state, key(KeyCode::Char('f'), KeyModifiers::CONTROL));
        assert!(state.follow);
        handle_key(&mut state, key(KeyCode::F(1), KeyModifiers::NONE));
        assert!(state.help_visible);
        handle_key(&mut state, key(KeyCode::Esc, KeyModifiers::NONE));
        assert!(!state.help_visible);
        assert_eq!(
            handle_key(&mut state, key(KeyCode::Char('q'), KeyModifiers::CONTROL)),
            Action::Quit
        );
    }
}
