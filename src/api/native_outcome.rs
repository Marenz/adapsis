//! Interactive turn completion is not task completion or remote delivery (#151).
use super::native_tools::NativeToolResult;

pub(super) const REFUSAL: &str = "Finish was not accepted: this turn has no visible answer or confirmed separately delivered outcome.";
pub(super) const CORRECTION: &str = "Finish was not accepted because this interactive turn has no visible reply. Provide an answer, clarification, or truthful status, then call finish alone. Continue from existing results without repeating completed actions. Tool success and queue/start notices are not delivered-outcome evidence.";
pub(super) const INCOMPLETE: &str = "This turn is incomplete: no visible answer or confirmed separately delivered outcome was provided. Any completed operations are retained.";

#[derive(Clone, Copy)]
pub(super) enum Rejection {
    CorrectOnce,
    Stop,
}

#[derive(Default)]
pub(super) struct Outcome {
    corrected: bool,
    pending: Option<Rejection>,
}

impl Outcome {
    pub(super) fn check_finish(&mut self, visible_reply: &str, result: &mut NativeToolResult) {
        let execution = &result.execution;
        // Existing test/save gates run first. A genuine runtime save warning is
        // itself a visible result; a generic tool success or callback String is
        // not. This baseline has no typed, turn-bound remote-delivery receipt.
        if !execution.done_accepted
            || execution.has_errors
            || !visible_reply.trim().is_empty()
            || execution
                .unsaved_notice
                .as_ref()
                .is_some_and(|text| !text.trim().is_empty())
        {
            return;
        }
        result.refuse_finish(REFUSAL);
        self.pending = Some(if self.corrected {
            Rejection::Stop
        } else {
            Rejection::CorrectOnce
        });
        self.corrected = true;
    }

    pub(super) fn take_rejection(&mut self) -> Option<Rejection> {
        self.pending.take()
    }
}
