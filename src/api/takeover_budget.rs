//! Takeover round accounting and reply hygiene, shared by the legacy
//! `handle_llm_takeover` loop and the native takeover driver (#85).

pub(crate) const DEFAULT_TAKEOVER_ITERATION_BUDGET: usize = 10;
pub(crate) const MAX_TAKEOVER_ITERATION_BUDGET: usize = 50;

pub(crate) fn takeover_iteration_budget(text: &str) -> Option<usize> {
    let start = text.find("<iteration_budget>")? + "<iteration_budget>".len();
    let end = text[start..].find("</iteration_budget>")? + start;
    text[start..end].trim().parse().ok()
}

#[derive(Default)]
pub(crate) struct TakeoverIterationEstimate {
    observed_meaningful_response: bool,
}

impl TakeoverIterationEstimate {
    pub(crate) fn observe(&mut self, text: &str, code: &str) -> Option<Option<usize>> {
        if self.observed_meaningful_response || (text.trim().is_empty() && code.trim().is_empty()) {
            return None;
        }
        self.observed_meaningful_response = true;
        Some(takeover_iteration_budget(text))
    }
}

pub(crate) fn visible_takeover_prose(text: &str) -> String {
    let mut clean = text.to_string();
    while let Some(start) = clean.find("<think>") {
        if let Some(end) = clean[start..].find("</think>") {
            clean.replace_range(start..start + end + 8, "");
        } else {
            break;
        }
    }
    clean = clean.replace("</think>", "");
    while let Some(start) = clean.find("<iteration_budget>") {
        let Some(end) = clean[start..].find("</iteration_budget>") else {
            break;
        };
        clean.replace_range(start..start + end + "</iteration_budget>".len(), "");
    }
    if let Some(start) = clean.find("<code>") {
        clean.truncate(start);
    }
    clean.trim().trim_end_matches("!done").trim().to_string()
}

pub(crate) const EMPTY_RESPONSE_CORRECTION: &str = "Your previous response was empty. Resume from the latest tool result without repeating completed or irreversible actions. Execute the next necessary operation, or end with <code>!done</code> if the task is complete.";
pub(crate) const EMPTY_RESPONSE_ERROR: &str = "Entschuldigung — das Modell hat gerade leer geantwortet. Die Nachricht und der vollständige Verlauf sind gespeichert; bitte versuch es noch einmal.";
pub(crate) const TAKEOVER_CALL_LIMIT_REPLY: &str = "I reached the 50-generation safety limit before I could produce the final summary. The conversation and completed operations are saved.";

#[derive(Default)]
pub(crate) struct TakeoverCallBudget { used: usize }

impl TakeoverCallBudget {
    pub(crate) fn reserve(&mut self) -> bool {
        if self.used >= MAX_TAKEOVER_ITERATION_BUDGET {
            return false;
        }
        self.used += 1;
        true
    }

    pub(crate) fn has_remaining(&self) -> bool {
        self.used < MAX_TAKEOVER_ITERATION_BUDGET
    }

    pub(crate) fn used(&self) -> usize {
        self.used
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum EmptyResponseAction {
    Retry(&'static str),
    Terminate(&'static str),
}

#[derive(Default)]
pub(crate) struct EmptyResponseRetry {
    retried: bool,
}

impl EmptyResponseRetry {
    pub(crate) fn on_empty(&mut self, replacement_available: bool) -> EmptyResponseAction {
        if self.retried || !replacement_available {
            return EmptyResponseAction::Terminate(EMPTY_RESPONSE_ERROR);
        }
        self.retried = true;
        EmptyResponseAction::Retry(EMPTY_RESPONSE_CORRECTION)
    }

    pub(crate) fn on_non_empty(&mut self) {
        self.retried = false;
    }
}

pub(crate) fn set_empty_response_error(reply_text: &mut String, error: &str) {
    *reply_text = error.to_string();
}

pub(crate) fn prepare_empty_replacement(iteration: usize, iteration_budget: &mut usize,
    call_budget: &TakeoverCallBudget) -> bool {
    if iteration + 1 >= *iteration_budget
        && *iteration_budget < MAX_TAKEOVER_ITERATION_BUDGET
        && call_budget.has_remaining()
    {
        *iteration_budget += 1;
    }
    iteration + 1 < *iteration_budget && call_budget.has_remaining()
}
