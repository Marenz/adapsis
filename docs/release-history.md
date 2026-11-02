# Release deployment history

Archived checkpoint records from ROADMAP.md. Their relative wording and pending
claims describe the original checkpoint, not the current deployment.

## Deployed 2026-11-02 — #118 grouped Python tests, release `88b4275`

- Writer `ses_e67e01ef9ffettWZDQAXWveOxc`, manager
  `ses_e8d6d9d1bffe8G7KEI4yUw8DPi`. Clean release HEAD `3ac0f3b` merged exact
  reviewed checkpoint `463d839` with `--no-ff --no-commit`; no conflicts.
  Production/test source is byte-identical to `463d839`. Manager verified the
  integration, committed/pushed `88b4275`, and installed/restarted here; this
  supersedes the historical `e424fc9` deployment below.
- #118 groups all Python cases for an exact target within one submission;
  targets execute first-seen, cases stay source-ordered per target, and later
  submissions replace the stored set. Per-def source spans remain available.
  HTTP/SSE mock-provider recovery tests cover repair, two-case persistence,
  reload and auto-retest rejection when the first case breaks. #87/#88 and native
  opt-in remain unchanged. This is offline protocol evidence, not live task quality.
- Own-target `cargo build --release` passed (33.28s). Focused
  `cargo test --release` filters: native_python_repair 4, native_ 166,
  tests_print 8, telegram_port 73 passed; zero failures/ignored, CLI tests filtered.
  Logs `/tmp/opencode/release-463d839-<filter>.log`. Whitespace check clean.
  Full suite not duplicated: manager reports accepted checkpoint evidence
  1991 passed / strict parity alone failed / 2 ignored, CLI 5 passed; merged source
  is identical. Prior strict parity residuals remain (324/58/191).
- Artifact `target/release/adapsis` SHA-256
  `4ef591486f0d74850d78affde7b6b77f2e3e2c8bb3b7133fdbfbc42b41ab0414`.
  `/home` reported 15 GB free (100% rounded usage) after the build.
- Independent post-deployment read-only verification: service active/running,
  PID 1538200, NRestarts=0; `/proc/1538200/exe` SHA-256 matches the artifact above.
  One active poll task, zero active non-poll tasks; poll status is exactly
  `http_get(https://api.telegram.org)`. `llm_get_model()` returns
  `"chatgpt/gpt-6-astra"`, success=true. Binary rollback:
  `/home/marenz/.local/bin/adapsis.bak-e424fc9-20261102-120538`.
- Pre-install recheck found no creative permission/context drift and configured
  and runtime Astra agreed. No live repair-task or Telegram reply certification
  yet: repair evidence above is mock-provider/offline. Deployment actions belong
  to manager; this worker performed read-only verification and this release log
  update only. Owner's plan/roadmap feature HOLD remains unchanged.
