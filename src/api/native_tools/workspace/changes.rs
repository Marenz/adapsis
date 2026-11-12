//! File changes and one foreground command. Same held work lock/binding as
//! read; no live Program snapshot, source writer or session write-back.
use super::*;
use crate::workspace::{Cancel, Exit, RunOptions, RunRequest};

impl Work {
    pub(super) fn change(
        &mut self,
        path: &str,
        edit: Option<(&str, &str, bool)>,
        content: Option<&str>,
    ) -> Result<String, Error> {
        let (site, context) = (self.site.clone(), self.context.clone());
        let (workspace, binding) = self.bound()?;
        let result = match edit {
            Some((old, new, all)) => {
                workspace::edit(workspace, &mut binding.ledger, path, old, new, all)?.render()
            }
            None => workspace::write(
                workspace,
                &mut binding.ledger,
                path,
                content.unwrap_or_default(),
            )?
            .render(),
        };
        if !matches!(
            site.bindings
                .save_ledger(&context, workspace, &binding.ledger),
            Ok(true)
        ) {
            self.bound = Bound::Unknown;
            return Err(Error::CommandFailed(format!(
                "{result}\nThe file ALREADY changed, but its ledger could not be recorded. Do not repeat the edit; read it again before another change."
            )));
        }
        Ok(result)
    }

    pub(super) fn bash(
        &mut self,
        mut request: RunRequest,
        cancel: Option<&Cancel>,
    ) -> Result<String, Error> {
        let (policy, root, cap) = (
            self.policy.clone(),
            self.site.bindings.root().to_path_buf(),
            self.caps.bash,
        );
        let (workspace, _) = self.bound()?;
        workspace::push_guard::check(&request.command, &workspace.base).map_err(|error| {
            crate::error_journal::record(
                crate::error_journal::Category::Refusal,
                "workspace.push",
                "workspace push mistake guard refused a command",
            );
            error
        })?;
        policy.check(&request.command).map_err(|reason| {
            crate::error_journal::record(
                crate::error_journal::Category::Refusal,
                "workspace.shell_policy",
                "workspace shell policy refused a command",
            );
            Error::CommandRefused(reason)
        })?;
        let sweep_note = workspace::foreground::ready(&root)?;
        let prefix = workspace::foreground::prefix(&root);
        request.max_output_bytes = Some(
            request
                .max_output_bytes
                .unwrap_or(workspace::limits::RUN_OUTPUT_DEFAULT_BYTES)
                .min(cap),
        );
        let mut options = RunOptions::new(&policy);
        options.cancel = cancel;
        options.unit_prefix = &prefix;
        let result = workspace::run_with(workspace, &request, &options)?;
        self.cleanup_confirmed = result.containment == workspace::Contained::Scope
            && result.exit != Exit::CleanupUncertain
            && result.cleanup_errors.is_empty();
        if result.exit == Exit::CleanupUncertain || !result.cleanup_errors.is_empty() {
            crate::error_journal::record(
                crate::error_journal::Category::Workspace,
                "bash",
                "foreground cleanup could not be confirmed; do not replay the command",
            );
        }
        let success = result.exit == Exit::Code(0) && result.cleanup_errors.is_empty();
        let header = format!("bash in {}\n", workspace::error::shown_name(&result.cwd));
        let mut text = format!(
            "{header}{}",
            result.render_within(cap.saturating_sub(header.len()).max(1024))
        );
        if result.containment == workspace::Contained::ProcessGroup {
            text.push_str("\n[note: process-group fallback, not a systemd scope; a process that detaches with setsid can escape cleanup]");
        }
        if !sweep_note.is_empty() {
            text.push_str(&format!("\n[note: {sweep_note}]"));
        }
        if success {
            Ok(text)
        } else {
            Err(Error::CommandFailed(text))
        }
    }
}
