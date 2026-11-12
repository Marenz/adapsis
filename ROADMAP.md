# AdapsisOS Roadmap

## W2 repository edit/test turn (2026-11-12, candidate, not deployed)

- Baseline `ce40048`: W1 accepted/committed as `aed5a2f`, exact live Python/save-outcome
  integration accepted/committed as `5888268`, exact reviewed image hotfix `7201856`
  merged as `ce40048` (focused image/takeover 65 passed). No #136 uncommitted work.
- Adds owner-private edit/write/bash, foreground-drop cancellation and a direct-loopback
  HTTP cancel escape hatch, push-mistake guard and digest-scoped startup cleanup.
  No job tools, W3 turn budget, new cancellation builtin or Telegram /stop. The separate
  ACK147 receipt worker is not duplicated here. Its live source/docs `c7a4927`/`1f19c3f`
  still require integration AFTER this W2 freeze/commit, before any deployment.
- Real mock-provider turn edits a repository file, writes a test script and executes it
  successfully, preserving the live Program/revision. Other regressions cover freshness,
  command failure, protected push before effects, dirty/running/unverifiable close,
  actual HTTP cancellation/client disconnect, metadata/privacy and UTF-8 output bounds.
- Read-ledger measurement: 50-read real owner turn448.385ms/51durable writes including
  claim; offline read/hash+one-flush1.558ms (different overhead). Retain per-call durable
  ledger writes; no batch-flush policy change. Evidence `/tmp/opencode/w2-read50-measurement.log`.
- Combined focused gate: 618 passed, 0 failed, 1 manual-measurement test ignored;
  production release build passes with no new warnings. The coordinated ONE full gate
  and independent review are pending.
  No live instance, registry, credentials or deployment changed.
- Independent W2 review rejected ONLY two ordinary-Git push semantics: source-only
  refspecs can follow a configured mapping to protected main; plain force overrides a
  stale lease even when a lease flag is present. Minimal correction requires every
  refspec to be nonempty `source:refs/heads/<destination>` and rejects plain force in
  either flag order. Advertised command is now
  `git push origin HEAD:refs/heads/adapsis/<name>`; lease-only remains supported.
  Real local-bare regressions cover mapping, stale lease rejection and correct lease
  success. Corrected focused push/workspace/native gate: 525 passed, 0 failed,
  1 manual measurement ignored. Corrected full gate and push-only delta review pending.

## W1 integration with the exact live Python line (2026-11-12, accepted)

- Parents: workspace W1 `aed5a2f` and exact live-plus-docs checkpoint `b779d1b`.
  Manager committed/pushed the accepted integration as `5888268`. No #136 work-in-progress, W2 changes,
  registry file or deployment is included.
- Approved semantic resolutions: `!watch` uses plain model view/token-only
  calibration (no active-turn accounting or cold start), with workspace batches
  hidden; coherent Program/meta snapshot helpers construct the new WorkingSet
  save checkpoint/commit mark; source-save write-back consumes that mark once;
  the shared panic hook keeps payload-free journaling and suppresses ONLY the
  duplicate journal entry of a contained JIT compile panic; the incoming stored
  nesting scope (250) and exact submission-limit (100) tests supersede the old
  reduced-depth test adaptation. Both lines' remaining tests and guards stay.
- Added interaction assertions: watch leaves an active turn's probe unchanged
  and hides its workspace results; P3's concurrent shared-key/chat changes
  survive a save wait followed by a workspace call; an accepted unsaved finish
  preserves both its notice and a cutoff marker; unsaved and bound-workspace
  runtime-tail lines leave the stable prefix unchanged; the real process panic
  hook emits one contained-compile entry and still journals a later uncontained
  panic on the same thread without its payload. The #136 general whole-metadata
  write-back window remains open.
- Fixture-only integration: incoming route/enrolment tests declared the now-reserved
  RunToolScratch module. They now parse/validate an ordinary fixture name, then install
  it explicitly as pre-existing/cached scratch state and rebuild indexes. Both assert
  that ordinary scratch declarations remain refused and scratch is never persisted;
  the real route still performs the mutation and starts its source save.
- Gates: seven targeted interactions pass; combined focused gate 750 passed / 0 failed /
  4 ignored, plus 7 startup integration cases. Default-JSON/stored-nesting/submission/
  block gate: 31 passed. Logs `/tmp/opencode/w1-live-integration-{targeted,focused,json-nesting}.log`.
  ONE full run: 2779 passed / 1 known strict Python-parity failure / 9 ignored,
  integration5/9/4 passed; integration-only review ACCEPT. Staged hash at acceptance
  `0ea8868fe90d465f284026cf4799cd9b64bbd518a3e355a2c74be29386838d07`.

## Workspace tools W1: lifecycle and read-only tools in owner turns (#129, 2026-11-12)

- On `work/workspace-tools` after the merges of the run-tool line (`f12c426`) and the
  Python release line (`3d29a13`) and the cache-epoch fix (`17c566f`).
- Wired for the writer class on a node with a workspace registry: `workspace_open`,
  `workspace_use`, `workspace_status`, `workspace_close`, `read`, `search`, `glob`.
  Binding and read ledger persist in the workspace state directory. Results are plain
  text bounded by the generating model's per-message cap. A provider length stop is a
  correction round now, not a provider error.
- After the independent review (REJECT as first staged: workspace ownership, shared
  history) one correction on top: workspace tools only in owner-private conversations
  (administrator's DM; HTTP `main` from this node only), workspace batches hidden from
  every other turn's view, atomic claim under the workspace reservation with the work
  lock owned by the blocking call, fail-closed binding directory, names escaped, result
  survives a model switch, training log and events without content.
- Delta review closed the original ownership/cancellation/shared-history blockers.
  Its remaining training-log blocker is corrected to structural metadata only (no
  header/error/prose, no arbitrary JSON key or correlation id). Final-status length
  stops retain the cutoff marker without an extra generation; migration exclusions
  use complete batch positions so reused call ids cannot hide unrelated results; file
  names escape the complete Unicode Bidi_Control set. Final delta review is pending.
- Next: W2 (`edit`, `write`, `bash`, cancel, push guard), W3 (`job_*`, workspace turn
  budget), then live acceptance on this node (#115 task). No registry file exists on any
  node yet, so nothing changes for a running instance until the owner creates one.
- Not deployed. AGENTS.md "Workspace tools in owner turns" has the invariants.

## Restricted run + JIT fix: merged and clean (2026-11-12)

- `d819b0d` restricted `run` for guests and execute-only speakers (adversarial
  review ACCEPT) and `f09de27` the #96 JIT fix (independent review ACCEPT) are
  merged on `work/run-tool`. The merge had no conflicts.
- Gate on the merged tree: 20 consecutive whole-binary runs, 0 `compiled_blob`
  panics (baseline before the fix: 5 of 20); one full run 2380 passed, only the
  documented strict Python-parity failure, integration 5/5/4.
- Not deployed, not wired into the live node. The entry below is the earlier
  checkpoint and is superseded where it says the #96 failure is outstanding.

## Recovery checkpoint — restricted native `run` accepted review, full gate pending (#129/#138, 2026-11-10)

- Cumulative restricted-`run` candidate on `work/run-tool`, HEAD `44e66e8`, is
  **accepted by adversarial review**, including the final shared pre-search regex
  capture/group/scratch/weighted-work admission for both regex builtins. Reviewer
  verified the restored candidate unchanged. Source is frozen; this is a recovery
  checkpoint, **not completion, a green full suite, a commit or a deployment**.
- Reviewed candidate digest (source before this docs-only checkpoint):
  `80ca561da0ca1a208174cde608909f90fccce2c096dac7b300cd217f7c473cf5`.
  Algorithm: SHA-256 of `git diff HEAD --binary`, followed by each sorted untracked
  file as NUL + `UNTRACKED` + NUL + path + NUL + file bytes. Tracked diff alone:
  `9a0a9c970570f46dee5127ae40c528cf8737afc2da1760a08bcf012534cb38f6`.
  Adding this documentation changes the whole-worktree digest, not the accepted source.
- Final full release gate: **2345 passed / 2 failed / 6 ignored**, integration
  targets **5 + 5 + 4 passed**. Failures explicitly remain outstanding:
  expected strict `python_parity` (334 verified / 58 invalid / 181 unsupported),
  and **#96**: `eval::tests::test_eval_func_name_still_works` panicked in
  Cranelift `compiled_blob.rs:56` with `TryFromIntError`. Do not describe the suite
  as green. Separate read-only #96 root-cause investigation is in progress;
  no #96 fix is included in this candidate.
- Logs: `/tmp/opencode/shared-regex-focused.log` (14 focused tests passed),
  `/tmp/opencode/run-tool-full-shared-regex.log` (full gate); reviewer regression
  seeds and logs: `/tmp/opencode/run-review/`.
- Sessions: manager `ses_e8d6d9d1bffe8G7KEI4yUw8DPi`; writer
  `ses_e3ae0b257ffekvdCXKh6wtL44R`; adversarial reviewer
  `ses_e3a974b00ffejrrcEK7TtvpUc9` (final minimal shared-regex **ACCEPT**).
  Writer is pinned to OpenAI Sol; independently observed weekly quota was **93%**
  before the last correction, with a **<=75% remaining pause threshold**.
- Status: candidate **not committed and not deployed**. Hold all source edits,
  tests and commits pending the manager's separate #96/full-gate decision.

## Workspace tools #129 — S3 recovery (2026-11-10)

- **Current: final independent review ACCEPT.** Reviewer
  `ses_e3e9595a2ffeksds1nSOBYRxZ2` closed original B1–B5/F1–F3 and accepted
  the final memfd-lifetime delta; manager verified the three tested source
  hashes below. Final full release gate: **2456 passed / 1 known strict
  Python-parity failure / 6 ignored**, integration **5/5/4 passed**.
  S1–S3 are library foundations only: **not wired into native tools and not
  live-certified/deployed**. Manager owns the reviewed-source/docs commit.
- Nonblocking followups are tracked in **#139**: account for unreadable job
  records as potentially running, a global state-tree budget beyond the
  per-directory cap, and the conservative prospective edit-cap edge. These
  were not silently folded into the final lifetime correction.

### Historical S3 review checkpoints — superseded, not current commit status

The frozen/uncommitted labels and base hashes below describe those earlier
checkpoints only; the current review outcome is ACCEPT above. They do not
claim that a later manager integration remains uncommitted.

- **Final minimal memfd-lifetime delta (frozen):** the delta review closed
  B1–B5/F1–F3 but found an ABA race when cancel/deadline returned from handoff
  waiting before the adapter opened its source. `drop(environment)` now occurs
  only after supervision stops/reaps the adapter. Successful handoff may close
  the adapter's own reader earlier, but the parent conservatively retains its
  descriptor until reap; cancel/deadline alone never authorize fd reuse.
- Two deterministic options-scoped inert-adapter/stop-barrier regressions cover
  cancellation and deadline: pause before source open, attempt fd reuse with 64
  other-request sentinel memfds, verify original inode remains pinned and only
  original env is consumed, one invocation/no replay, then ordinary env control.
  Both tests fail when the old early drop is restored and pass with the fix.
  Production change is only the ownership/drop order; adapter override is
  `cfg(test)` and scoped to the run options, with no process-global mutation.
- Final affected release gates: run tests **23/23**, repair tests **6/6**.
  Exactly one final full suite: **2456 passed / 1 known strict Python-parity
  failure / 6 ignored**, integration **5/5/4 passed**, unit run 40.21 s.
  Logs: `/tmp/opencode/ws-memfd-negative.log`, `ws-memfd-run-focused.log`,
  `ws-memfd-repair-focused.log`, `ws-memfd-full.log` (same directory).
  Final `git diff --check` passes and test-unit listing is empty. Cached quota
  remains **94%**, Sol pin unchanged. No extra full gates or source edits after
  this checkpoint. Nonblocking unreadable-job concurrency accounting, global
  state-tree budget and conservative exact edit-cap followups are untouched.
- Historical pre-integration base HEAD: `c809071df5b7e1dcf139e0209b80ce1a30dadfe1`.
  Tested minimal-delta source SHA-256:
  - `run.rs`: `b2fd760806a987f19573e4e0dfc98ed257b30efd868be3db95cb1bba9b1d0524`
  - `contain.rs`: `16aba0806a7febfe91aadddb6d34ec7ae5d9d015bdefb811e23cc073bc8d15a7`
  - `repair_tests.rs`: `05a5bd5164c417161caac5314b6bfc7732e6d051f0d1df6347c9bd55e4894800`

- **Independent review repair checkpoint (supersedes the initial S3 freeze
  below):** the initial candidate was rejected for B1–B5. One bounded repair
  pass also addresses F1–F3, with no wiring or feature expansion. Frozen for
  the same reviewer's delta recheck: focused release **183 passed**; one full
  repair `cargo test --release --no-fail-fast` **2454 passed / 1 known strict
  Python-parity failure / 6 ignored**, integration **5/5/4 passed**. Logs:
  `/tmp/opencode/ws-repair-release-focused.log`, `/tmp/opencode/ws-repair-full.log`.
- B1: static trusted `/usr/bin/python3 -I` adapters structurally separate clean
  supervisors/clients from payload env dictionaries. Foreground CLOEXEC memfds
  are explicit handoffs, 0600, bounded to 512 KiB, and closed after adapter read;
  no env values are in argv. Establishment precedes payload `BASH_ENV`; startup
  effects/exit cannot trigger Auto replay. Jobs stage private JSON env through
  nofollow descriptors, then small-process `Popen(env=...)` (not a daemon fork).
  Python is an explicit dependency; it does not sandbox the owner's command.
- B2/B3/B4: shared per-workspace reservation serializes start/close/list/prune/
  forget and status observation. Fallible enumeration propagates directory and
  per-entry errors; malformed/unreadable/missing-existing-history `last-id`
  refuses allocation. Private state uses held directory descriptors, nofollow
  open/mkdir/read/atomic-write/unlink operations and recursive nofollow removal;
  static links refuse and swapped paths cannot redirect operations outside.
  Configured-root ancestors and bind mounts remain operator-trusted limitations.
- B5: unit signal/inspection errors propagate cleanup uncertainty, rather than
  silently substituting process-group proof. `Exit::CleanupUncertain` and
  `RunResult.cleanup_errors` forbid a successful/fully-stopped claim or replay.
  Real scopes still stop detached children; injected post-start kill and
  show/cgroup failures return truthful uncertainty.
- F1/F2/F3: edit inputs at most 2 MiB each, checked prospective output at most
  16 MiB before allocation, at most 10 ambiguity offsets retained; search has a
  64 MiB per-file byte ceiling even for a growing newline-free stream, with a
  partial-result note. Semantic v1 job-record invariants are checked before a
  terminal record is used or pruned. Record fields/version remain v1; malformed
  type-correct data is refused/preserved. Env staging changed from unpublished
  shell/NUL handling to private JSON, with no deployed staging migration claim.
- All eight review IDs have discriminating tests, including a barrier start/
  close race, outside sentinel link checks and a held-dir swap, endless generated
  reader, expansion refusal preserving file/ledger, BASH_ENV effects exactly
  once, and semantic-record preservation. Final test-unit listing is empty and
  `git diff --check` passes. No repair commits/deployment/stash. Final readable
  cached OpenAI weekly quota **94%**, pinned Sol unchanged; no further tests
  after this docs-only recovery update.

- S1/S2 accepted and committed by manager as `cce4cb2` / `c809071`.
  S3 writer continues in `/home/marenz/Projects/adapsis-ws` on
  `work/workspace-tools`; manager session `ses_e8d6d9d1bffe8G7KEI4yUw8DPi`.
- Owner routing/budget decision: Anthropic exhausted; resume approved S3 on
  OpenAI, no extra agents. Preserve about 70% weekly quota; pause and report
  at <=75% remaining or an unreadable meter. Cached weekly quota at resumption
  is 97%. No commits/deployment by this worker.
- Initial S3 candidate (then uncommitted): systemd scopes for foreground/cancel, services and private
  versioned disk records for restart-surviving jobs, head/tail output and
  per-repository environment. Frozen after focused release **173 passed** and
  one full `cargo test --release --no-fail-fast`: **2444 passed / 1 known
  strict Python-parity failure / 6 ignored**, plus integration **5/5/4 passed**.
  Logs: `/tmp/opencode/ws-s3-focused.log`, `/tmp/opencode/ws-s3-full.log`.
  No additional full gate after these docs-only updates.
- Job record `version: 1`: private `job.json`, command/cwd/unit identity,
  start/end timestamps, runtime/output caps, environment names only, state,
  exit status and end reason. `last-id` prevents retention from reusing ids;
  unit names encode component lengths and a 128-bit state-root digest to
  separate hyphenated repo/workspace pairs and isolated stores. A manager
  that cannot be reached means unverifiable, never proof a job stopped;
  close refuses those records as well as confirmed running jobs.
- Regression evidence includes killing a separate starter process, reopening
  its record and finishing the same surviving job; detached `setsid` children
  are stopped by scopes/services; manager environment is absent from jobs;
  both foreground containments and cancellation are covered. Only uniquely
  prefixed test units were used; final `adapsis-wstest-*` listing is empty.
  No live service/config changes, commits or pushes by this worker.
- Session `ses_e3a2152a2fferDqyzS23BuI8K9` pinned to
  `openai/gpt-6.1-sol` (medium), per owner. Final readable cached weekly quota
  **97%**; freeze/report completed above the <=75% pause threshold.
- Owner secret decision (2026-11-10, verbatim): "I think our secrets should
  generally be in a \"secret store\" and if there are any found through bash
  then so be it. they should be properly set anyway and of course the bot
  should actively try to avoid reading any in .env files and such, but it does
  that anyway nowadays, no?" N5/N6/L2 in DESIGN.md record ordinary
  owner-authority bash, no mandatory bwrap filesystem secret masking, and
  required behavioral guidance against deliberate credential discovery/read/
  dump in bash/read/search. That guidance is for the wiring slice; an actual
  secret store/provider and migration remain undecided/unimplemented, and
  active avoidance has not been verified. Named capabilities and existing
  logging/redaction gates remain required. No secret scans were performed.

## Deployed 2026-11-12 — here (Kronk), `8b985cb`, real source-save acknowledgements (#146)

- **Live:** commit `8b985cb`, binary sha256
  `8c8e8a4d55990272ecd06d15927dbd213432cf710dbcda18600b054b99623046`,
  replacing `b7bfa64` / `fa04e367…`. The five manifest-managed Python modules
  are unchanged; 275 of 275 stored tests passed at startup. sleek and edox are
  unchanged.
- **Owner-approved deployment, executed by the manager:** fresh checkpoint,
  inhibitor and install checks passed; NRestarts 0; five modules, one route, two
  permanent loops, no library errors; shared-variable probes unchanged.
  Telegram `SAVE146-OK` delivered (message 549037), with no provider error from
  the newly deployed stable `prompt_cache_key` (release-line `7dc69aa`).
- **Live save-outcome proof:** `/api/mutate` adding `Peer.upgrade_probe` returned
  `saved to Peer.py` and the file was already changed when the response arrived;
  removal returned saved again and the library returned byte-identical.
  Revision after verification: 248. The earlier acknowledgement pitfall is
  resolved by runtime save outcomes; saving remains automatic.
- **Acceptance evidence:** independent delta review accepted `8b985cb`; reviewer
  probes passed. Full suite 2386 passed / 1 known strict-parity failure
  (334 / 58 / 181, zero mismatches) / 7 ignored; upgrade rehearsal 216 of 216;
  gate self-test 9 of 9 deliberately wrong runs rejected.
- **Existing recovery material retained:**
  `~/.local/share/adapsis/upgrade-146/pre-upgrade-bundle/`, previous binary
  `~/.local/bin/adapsis.bak-fa04e367`; operator procedure in
  `~/.local/share/adapsis/rehearsal-146/REPORT.md`. Rehearsal proved the previous
  binary reads the new one's session and managed library, including a new module.
- **Still pending:** #136 (the pre-existing whole-metadata write-back concurrency
  window; this change avoids extending it across the save wait); #146's later
  module-library git-commit boundary; trusted `/api/mutate` rendering can omit a
  named successful-save line when the active model lacks Read permission.
- **Owner direction:** during this development phase prioritize fix-forward and
  minimal release ceremony, without new compatibility or rollback projects.
  Existing backups stay; review, tests and live verification remain required.

## Historical deployment 2026-11-12 16:12 CET — here (Kronk), `b7bfa64`, manifest-managed Python library (#128)

- **Live on here:** commit `b7bfa64` (`work/python-source-library`: Result runtime,
  typed execution/replay, manifest-managed `.py` library, run-time module
  enrolment, one nesting limit, merged with the release line `6136115`). Binary
  sha256 `fa04e3677b4adf70aafc65196c7ff6facaece2c155f3f9b36de272359c7ce3e8`,
  replacing `11487f02b5cc97998e6a91b8b2f0e9fa3bb7615914dcf51cedec84fe1b8b4f08`.
- **Library:** `~/.config/adapsis/modules/` now holds `library-sources.json` and
  five reviewed Python modules (Chronica, Peer, TelegramBot, TgFormat, Watchdog;
  function counts 24 / 9 / 48 / 18 / 12; 275 stored cases). The `.ax` files were
  moved out. Module deltas shipped with the switch: Chronica `pull_page` returns
  `Result[str]` and `writer` matches it; Peer `eval`/`mutate` renamed to
  `remote_eval`/`remote_mutate`; the `telegram_token` default is blank and the
  token was scrubbed from the session (2 occurrences).
- **Unit:** drop-in `adapsis.service.d/10-source-refusal.conf`
  (`RestartPreventExitStatus=78`, `OnFailure=adapsis-start-failed.service`) and the
  helper unit `adapsis-start-failed.service`. Refused starts exit 78 and are not
  restarted.
- **Switch, executed by the manager per the runbook, no deviation:** preflight exit
  0 through the journal (the saved file was 8 history entries behind; the
  requested save closed it), start inhibitor proven, final bundle taken after the
  stop, five stopped `.ax` verified against the reviewed hashes, install, start.
  Verified: active, NRestarts 0, 275 of 275 stored cases re-run, 5 managed sources,
  one webhook route, two permanent loops, no library errors, shared-variable check
  as predicted, a Telegram round trip, a Python edit made through the bot (only
  `TgFormat.py` changed) and its removal (reviewed bytes restored), a clean
  restart at revision 246.
- **Evidence before the switch:** full suite 2354 passed / 1 known strict-parity
  failure (334 / 58 / 181) / 7 ignored; independent review of `a488baa` rejected
  the switch for rollback and rehearsal defects; the correction `b7bfa64` and
  the third revision of the operator bundle were accepted; rehearsal gate 192 of 192 checks, gate
  self-test 9 of 9 wrong runs rejected.
- **Rollback:** restore `~/.local/share/adapsis/switch-128/pre-switch-bundle/` as
  one unit (session at revision 244, five `.ax`, old binary, unit state), after
  archiving the candidate-era session and library. The old binary cannot read a
  session that holds a Python mutation record. Old binary also at
  `~/.local/bin/adapsis.bak-11487f02`. Runbook:
  `~/.local/share/adapsis/rehearsal-128/REPORT.md` (step 8); notes:
  `~/.local/share/adapsis/switch-128/notes/RESULT.txt`.
- **Not changed:** sleek and edox (older binaries, legacy `.ax` libraries, no
  manifest); the Ladybug database; permissions and context files.
- **Open after the switch:**
  - The mutation acknowledgement says the source save is pending; the bot repeated
    that as "the file may not have changed yet" after a save that had succeeded
    within the same second. Not changed; the owner is deciding how saving should work.
  - Whole-module removal (#126), `!sandbox discard` after an enrolment, printer
    parentheses (#140), session growth and the whole-metadata write-back (#136),
    a read-only route for effective shared values, the non-atomic write of a
    function move in both directions.
  - The bundle and older backups still contain the bot token
    (`rehearsal-128/token-copies.paths`); rotating the token would settle all of them.

Everything below is history. Statements there that the library still persists
`.ax`, that the default format is unchanged, or that no deployment has happened
describe the state at their own date, not here after 2026-11-12.

## Historical acceptance checkpoint — typed Python execution/replay (#121/#122)

- Uncommitted candidate on manager-integrated baseline `8911fba`: typed operations
  share the existing executor/session application, while legacy string wrappers
  parse and delegate. Native pinned authorization, #87/#88, #83, #118 and Result
  activation semantics remain. `legacy_export()` is explicit and fallible, not
  required by normal Python execution, history reconstruction or typed re-tests.
- Exact original Python mutation text/module is retained with versioned requested
  IR and the actual affected-module after-state. Rejected/partial updates restore
  their actual outcome rather than replaying rejected code. Rewind is staged,
  never executes historical effects, and refuses uncertain legacy outcomes.
- Approved #122 residual: Python declaration edits after an earlier rewind refuse
  before submission effects; future history is not deleted or branched implicitly.
  Evaluation/tests remain usable. Standalone historical test-event reconstruction
  and unrecorded library baselines are not newly provided by full rewind.
- Library serializers/loaders are separate owners. Existing `.ax` persistence hooks
  remain until their reviewed Python backend integrates; this is not live maintained
  `.py` authority or in-flight exactly-once recovery. No deployment by this worker.
- Gates: typed fixtures 27 passed; native 175 passed; session 53 passed; scoped 23
  passed; Python 375 passed plus CLI 5; Result CLI 4. One full release gate:
  **2091 passed / 1 known strict-parity failure / 2 ignored**, Python CLI 5 and
  Result CLI 4 passed. Strict inventory unchanged: 324 verified / 58 invalid /
  191 unsupported, zero mismatches/non-idempotence. Log: `/tmp/opencode/typed-full.log`.
- Exporter-unavailable child-process tests use the actual HTTP/SSE and CLI core;
  completed-work save/load, two-case automatic re-test and checkpoint replay work
  while explicit legacy export fails. No alternate test-only execution path.

## Accepted frontend checkpoint — bounded Python Result (G7, `902038b`)

- Baseline `aa8f7f4` includes accepted C-core `113f80f` and the synthetic legacy
  String-wrapper regression. This frontend slice admits user Result awaits with
  Result bindings and plain exact-known-inner returns under Result[T] (G7).
  Nested Result and outer Option need explicit wrappers (Some shares Value::Ok).
  List[Option] remains admissible. Assignment/argument compatibility,
  explicit-constructor payload policy, async/IO/arity and native authority gates remain.
- Actual dispatch-signature checks refuse outer Result mismatch in either direction.
  Native Python evaluate/test additionally preflight the current reachable stored graph
  before IO, using the shared iterative AST walker and tree lookup, snapshot-local pointer
  identity, and categorical diagnostics without hidden names. This is not payload typing
  or transitive authorization. Inspection/resubmission and nested binding tests accompany native scalar
  consumers, typed Check/operational-error separation, and synthetic scoped IO tests.
  No live-memory IO or runtime/quarantine changes are included.
- Strict parity: **334 verified / 58 invalid / 181 unsupported**, 54 files/573 entries;
  zero AST mismatches/non-idempotence. Ten promotions from 324/58/191: seven functions
  (`complex_system.validate_name/validate_age`, `multi_param_chain.validate_request`,
  `result_chain.validate`, `simple_syntax.validate_user`, `trace_test.validate`,
  `validate.validate`) and three programs (`simple_syntax`, `trace_test`, `validate`).
  Two still-Unsupported programs change diagnostics (`multi_param_chain`, `result_chain`).
  Frozen fixture bytes and parity success criteria are unchanged.
- Freeze validation: focused Result matrix 13/13. One full
  `cargo test --release --no-fail-fast`: 2032 passed, the known strict parity gate
  alone failed, 2 ignored; Python CLI 5/5 and Result-contract CLI 4/4 passed.
  Log: `/tmp/opencode/full_gate_frontend_result.log`. Strict parity remains RED;
  frontend acceptance and deployment still require their separate reviews.
- Review correction evidence: both defects failed before correction (outer Option
  implicit return admitted, reverse signature drift accepted); the installed native
  caller also reached inert IO before the new entry preflight. Corrected focused gate
  (`result_returns native_ python_call_ vm_user_await python_parity`) has 206 passed,
  only the known strict parity failure; parity stays 334/58/181. Log:
  `/tmp/opencode/frontend_result_review_final.log`. No second full suite. Tests cover
  unchanged callers after Result→Int/Bool and Int→Result edits, direct/transitive native
  evaluate/test refusal with zero IO and unchanged Program, cycles/bare vs qualified
  same-name lookup, unrelated-graph isolation, and explicit Ok(Some)/Ok(None).
- #121 makes Python authoritative maintained source, with .ax a temporary importer;
  serializer/direct-IR/persistence work remains separately owned. This slice does not
  change LoweredSource or canonical APIs. Historical parity is evidence, not that goal.
- C deployment remains blocked: `Chronica.pull_page` forwards Result under a String
  signature and its writer consumes String. Four other audited String wrappers have
  unknown external consumers. Option A (accurate Result[String] signatures plus explicit
  callers) is selected, but private migration needs separate authorization after frontend
  acceptance and persisted Python source support. No further live reads or migration here.

## Historical acceptance integration — `b0ffd8c` + `113f80f`

- This isolated worktree combines accepted Result C-core `113f80f` with release
  memory/privacy (#87/#88), Python grouped tests (#118), scalar admission (#83),
  and structural IR guard (#121 Step A). Merge approval and commit belong to the
  manager; typed execution/replay B+C starts only after that committed baseline.
- The timer conflict retains C's actual-declaration invocation together with
  release content-gated logging and categorical errors. Both scalar and C-result
  test modules remain registered. No source-format or library activation is added.
- Current deployment reported by manager: `6ddcfc1`, Astra, Python-authored
  TelegramBot/TgFormat. Verified live scalar repair and completed-work restart
  are not evidence of this merged Result runtime, which is not deployed here.
- Owner has authorized Python-authoritative source work (#121) and Python-domain
  task work (#120). Historical hold/scope notes below describe their checkpoints,
  not current authorization. Native remains opt-in; #89/#76 acceptance stays open.
- Focused merge gates: `native_` 174 passed; `ir_surface` 11 passed;
  `--test result_contract_cli` 4 passed; `native_python_repair` 4 passed;
  `logging` 8 passed. No full-suite rerun or deployment in this baseline step.
  Logs: `/tmp/opencode/baseline-c-{native,ir,cli-result,repair,logging}.log`.
- Both-parent review retained release logging/memory/opt-in behavior and C's
  actual-declaration entry points. Historical release and Result evidence follows
  unchanged except for explicitly historical headings. The foundation archive
  below is byte-identical to the corresponding `b0ffd8c` tail.

## Historical release integration records

## Deployed 2026-11-02 — release `6ddcfc1`, narrow #83 scalar admission

- Manager deployed the saved candidate; independent read-only verification by
  `ses_e67e01ef9ffettWZDQAXWveOxc`: PID1784882 active/running, NRestarts0,
  `/proc` executable SHA-256
  `218791e69156b8e33be5b60abf99f093113cce654afd843a8a1dc06b79bde385`.
  Runtime model remains `chatgpt/gpt-6-astra`; runtime symbols retain the migrated
  TelegramBot48/TgFormat18 functions. Manager reports host-only polling status.
- Binary rollback: `~/.local/bin/adapsis.bak-88b4275-20261102-155153`.
  This was binary-only deployment; private Python-authored module files were not
  replaced. Prior module provenance/AST and exact text-reply evidence remains below.
- Integration: reviewed `a433def` plus exact `9781f3c..50884a9` replacement delta.
  The rejected Rust task prototype and module registration are absent. Python task
  examples are inactive pure modules, not a deployed task service. Generic records
  backend `1e0ed81` is NOT included. No storage/executor/authority wiring follows.
- Offline build passed; integration full suite: 2004 passed, strict python_parity
  alone failed, 5 ignored; CLI5 passed. Strict parity remains324/58/191 with no AST
  mismatches. Focused python_scalar7/native_python_repair4/native_172/task_domain2/
  module_tests10 all passed. Logs: `/tmp/opencode/release-a433def-50884a9-*.log`.
- Observer independently verified request543083/reply543087 and scoped native batches:
  Python buggy mutate -> two failed cases (24/23, 3/2) -> Python inspect -> final-expression-only
  repair -> EXACT SAME test source passes (23, 2) -> Python eval23 -> accepted finish/visible summary.
  Zero `.ax` fallback; saved Program has repaired three-statement function and two passed tests.
- This verifies live narrow scalar Python repair, not restart recovery of the new module,
  general #83 coverage, native parity or default cutover. #83 retains call-free scalar/data
  and safe-shared-default proofs. This writer performed no additional live checks or builds.

## Live Python-authored Telegram modules — 2026-11-02, exact text reply observed

- Manager `ses_e8d6d9d1bffe8G7KEI4yUw8DPi` completed reviewed v3 helper/drain,
  runtime capture, fresh recapture, whole-service stop, final export/library gates,
  pair installation and one start. Runtime binary at migration was `88b4275` (history below);
  migration tooling checkpoint `9ffd3fa` adds no production runtime feature.
  Durable `.py` files record authorship; the running library stores lowered canonical `.ax`.
- Durable owner-only provenance/rollback bundle:
  `~/.local/share/adapsis/telegram-python-migration-20261102/{backup,capture,python-authored}`.
  Preserve captured effective settings/cursor on rollback; session alone omits shared state.
- Installed Python-lowered canonical hashes: TelegramBot
  `201fd1d90f31b7caf62191de623ff9538cb45c5a18bf6a9826d410f89fb6f4cf`;
  TgFormat `0e4d1e25009b3cc4123122833b46f744f405ded1b53b6c0337d365753ec1297f`.
  Normal library persistence subsequently serialized these as
  `9c50c2fabcc18e9cf18704dcc401f591d41bbca9d0398e6916472fc2b118e408` and
  `e03f323ab47ef42932971bd6ae31b323d0c772233912d2f083c45844b6ae32bf` respectively.
- Writer `ses_e67e01ef9ffettWZDQAXWveOxc` independently proved exact full module
  ASTs (only typed NodeIds ignored), all grouped cases (127+94=221), and route
  registration batch equality against the final artifacts. No semantic drift;
  read-only test `telegram_migration_compare_deployed_artifact` passed (1/1).
  Runtime symbols confirm 48/18 functions, two outcome types, TgTextTurn return
  on process_update and no migration helpers. PID1636185 active, NRestarts0,
  one poll task, zero active non-poll tasks at observation.
- Manager observed post-install Telegram request542631 -> reply542632 at
  12:31:22 UTC, exactly `PYTHONLIVE7429`. Earlier post-install request542629
  requested `KRONK_PYTHON_LIVE_20261102_B`; reply542630 stripped underscores,
  an observed end-to-end residual whose cause (model or formatter) is unestablished.
  No pre-migration baseline is acceptance evidence; no live repair-task/attachment certification;
  #104 failure ACK and #99 splitter residuals remain. Plan/roadmap feature HOLD unchanged.

The complete historical `88b4275`/#118 deployment record is preserved verbatim in
[docs/release-history.md](docs/release-history.md); its status claims refer to that checkpoint.

## Deployed 2026-11-02 — here (Kronk), release `e424fc9`, Astra default

- Release writer: session `ses_e67e01ef9ffettWZDQAXWveOxc`, scoped exclusively to
  `/home/marenz/Projects/adapsis-release`; manager `ses_e8d6d9d1bffe8G7KEI4yUw8DPi`.
- Starting release HEAD `3969170` contains deployed binary checkpoint `1ed6424`
  plus deployment/gateway history notes. Exact incoming checkpoint: `108e4dc`.
  `git merge --no-ff --no-commit 108e4dc` merged without conflicts. Manager accepted
  integration, committed/pushed `e424fc9`, installed it and restarted here.
  Release writer independently verified the running binary and read-only API below.
- Production delta since `1ed6424`: G2 permits `input` as a binding only; #114
  rejects unproven VM field accesses before execution; G1 admits typed Python
  pseudo-field properties; #112 structurally redacts task status (including Debug,
  with raw Serialize removed); #116 strips request URLs from HTTP transport errors.
  Module-less Python surfaces and stored-test printing remain test-only. The
  strict parity harness now expects 324 verified / 58 invalid / 191 unsupported.
- #87 turn-bound native memory tools, #88 private logging, TUI and #89 opt-in
  native routing are retained. No native-default flip or capability-parity claim.
  Historical pending/deployment sections below describe their own checkpoints;
  this section records the current deployment and retains its measured candidate gates.
- Audit: auto-merges in AGENTS.md, ROADMAP.md, native tool test registration,
  API tests, coroutine runtime and native prompt. No manual conflict resolution
  or production integration fixes needed. Incoming runtime changes preserve
  guarded logging and memory dispatch; pseudo-field lowering keeps exact AST
  shape and relies on pre-execution tree fallback, never post-IO replay.
- Measured offline gates on the merged candidate (2026-11-02):
  - `cargo build --release`: passed, 34.53s, own `target/`.
  - `cargo test --release --no-fail-fast`: 1986 passed / 1 failed / 2 ignored;
    CLI integration 5 passed / 0 failed. ONLY failure: `python_parity`.
    `PARITY_SUMMARY files=54 entries=573 verified=324 invalid=58 input_errors=0 unsupported=191 ast_mismatches=0 non_idempotent=0`.
  - `cargo run --release -- memory-eval`: 6 cases, recall 1.000000; duplicate,
    precedence, contradiction, ACL, citation, stale and omission counts all 0;
    `passed=true`.
  - All `examples/*.ax`: 28 files / 0 failures (counted, loop returns nonzero on
    any failure). Python CLI basic: 2 passed / 0 failed; flow: 7 passed / 0 failed.
  - `cargo test --release <filter>`: native_ 162, takeover_native 13,
    task_status 9, http_error_text 5, field_admission 14 passed; no failures or
    ignored tests in these filtered runs (CLI tests filtered out).
  - `git diff --check` and `git diff --cached --check`: clean.
  - Logs: `/tmp/opencode/release-108e4dc-{full,memory,native_,takeover_native,task_status,http_error_text,field_admission}.log`.
- Artifact: `target/release/adapsis`, SHA-256
  `7a91463243d68a2c12bdd40d0906e1c99dcaca5fd95cc51ebae4c241cca341f3`.
  `/home` has 16 GB free (99% used) after gates.
- Residuals: strict Python parity remains red; test-only module-less support is
  not a new native capability. #114 trades VM coverage for safe tree fallback;
  field/Result/JIT parity is not complete. Task locals (#115) and attachment
  filename derivation remain outside the redaction fixes. No live provider or
  bot delivery certification follows from offline tests. Existing historical
  deployment and gateway notes below have not been re-verified by this worker.
- Deployment verification: service active/running, PID 1468478, NRestarts=0;
  `/proc/1468478/exe` SHA-256 matches the artifact above. Binary rollback:
  `/home/marenz/.local/bin/adapsis.bak-1ed6424-20261102-113203`.
- Here's runtime AND configured service model are now `chatgpt/gpt-6-astra`, with
  no explicit reasoning-effort variant. Manager changed only the unit's model
  argument and ran daemon-reload; runtime `llm_set_model` applied the same model
  without a second restart. Unit backup: `adapsis.service.bak-before-astra-20261102`.
  No permissions/context changes. Actual configured permissions path remains
  `/home/marenz/.config/adapsis/permissions.toml`. This overrides the earlier
  here-node alias/model notes below; no other node's model is changed.
- Read-only deployment-input preflight: one real creative principal is shape-
  equivalent to the sanitized reviewed placeholder, 10 allowed tools and the
  creative model table match exactly; actual creative context matches the ignored
  local reference byte-for-byte (4959 bytes, SHA-256
  `7790ad3a34f6fbed4abb127367e9ed8b8fcdb58593999100ba2488acc4528826`).
- Live verification after deployment (writer session above): `/api/status` HTTP200,
  revision 237, mutations 237, history entries 120, one route. Top-level functions
  and types count 0 (these arrays do not enumerate module members). `/api/tasks`
  HTTP200: one active poll task, zero active non-poll tasks, redacted status exactly
  `http_get(https://api.telegram.org)` (#112). `llm_get_model()` returns
  `"chatgpt/gpt-6-astra"`, success=true. No private program source printed.
- Controlled pure `/api/eval` smoke: `json_valid("{")` -> false and
  `json_kind("true", "")` -> `"bool"`, each HTTP200/success=true, using the required
  `function:""` field. Manager's earlier omitted-function request returned 422
  before execution; corrected request succeeded.
- Provider smoke before deployment: local gateway HTTP200 for
  `chatgpt/gpt-6-astra`, benign "Reply OK", max_tokens128, exact-OK content category.
  This checks basic generation only, not native tool protocol or task quality.
  **No live Telegram reply certification yet.** Offline residuals above remain.
  Deployment/config changes were manager actions; this writer only performed
  authorized read-only checks, pure eval/provider smoke and release-doc edits.

## Deployed 2026-11-01 19:34 CET — here (Kronk), release/2026-11-01 `1ed6424`

- Binary sha256 `1c3e5285925e0cc42bf98e1eb33e08d9c26c01cf17bdb087e36d9dc6db64f39e`
  installed at `~/.local/bin/adapsis`; running PID verified via `/proc/<pid>/exe`.
  Rollback: `~/.local/bin/adapsis.bak-6685599-20261101-193414` (previous prod `6685599`).
- Contents: work/native-tools `a4ab092` + work/issue-87-88 `e28fe29` (#87 native memory
  tools, #88 private logs, #69 TUI, #89 native stays opt-in). Later native-tools commits
  `fae91e7`, `2e1e018`, `d1ae977` are test/fixture-only and do not change the binary.
- Service config unchanged by owner request (same ExecStart incl. `--native-tools`,
  `--model deepseek-v4-flash`, permissions file, creative principal mapping and context file).
- Verified: active/running, NRestarts=0, session revision 237 loaded, 28 library modules,
  webhook route registered, `TelegramBot.startup` completed, `TelegramBot.poll_loop` polling.
  No live Telegram smoke message sent. sleek and edox not deployed.
- Follow-ups filed: #112 (/api/tasks exposes token prefix), #113 (route message under
  library-error category).
- 2026-11-01 ~20:10: llm-gateway alias `deepseek-v4-flash` repointed from
  `deepseek/deepseek-v4-flash` (DeepSeek balance ~$0.38) to
  `deepinfra/deepseek-ai/DeepSeek-V4-Flash` (owner-approved). Direct and alias test
  requests returned 200; gateway and adapsis services active. Backup:
  `~/.config/llm-gateway/config.json.bak-20261101-deepinfra`.
- 2026-11-02 ~02:30: DeepInfra returned 402 "positive balance required" on every
  request, so the bot's alias was failing. Rolled `deepseek-v4-flash` back to
  `deepseek/deepseek-v4-flash` (DeepSeek direct answered 200; balance low).
  Backup of the DeepInfra state: `config.json.bak-20261102-deepinfra402`.


Release 2026-11-01 combines work/native-tools a4ab092 and work/issue-87-88 e28fe29
(both branched from production 6685599). Sections below are kept verbatim from each
side: the first block is the #86 Python-parity line, the second the #87/#88/TUI line
(native tools remain OPT-IN via --native-tools).

## Historical recovery — C-core activation checkpoint (committed as `113f80f`)

- Owner authorizes the broader work and deployment toward a useful programming/task/agent
  bot that can eventually replace OpenCode. Current worker scope is the approved bounded
  interpreter-only C-core activation, after accepted Result A, JIT/cache, scalar VM safety
  and invocation plumbing. Result normalization, typed call modes, preserved user-await
  Results, shared trace execution and structural adapter errors are approved together.
  No commit, push, service changes, live-module access or live IO in this writer's scope.
- Model rule: owner now permits OpenAI and DeepSeek; manager/default remains
  `openai/gpt-6-astra`, default variant (no explicit high effort), no Anthropic.
  This supersedes the earlier OpenAI-only and historical model-routing notes below.
- Main checkout baseline `8d8cfaa`. Manager reports current live release `6ddcfc1`, healthy
  service, model `chatgpt/gpt-6-astra` with default effort; no live re-verification by this
  writer. Earlier `88b4275`/deployment record `2019299` and its SHA belong to historical
  release evidence, not this current activation. Historical DeepInfra 402 caused an alias rollback to direct
   DeepSeek; the current reported runtime selection is Astra.
- Subsequent manager/release evidence: Python-authored live TelegramBot/TgFormat installed
  via tooling `9ffd3fa`, exact AST + 221 cases verified; real Telegram marker message
  542631/reply 542632 exactly `PYTHONLIVE7429`, release record `5b788be`, binary `88b4275`
  unchanged. This is a real text-reply proof, not proof of the new C runtime. Broader
  programming smoke exposed #83 restart-helper collision (native Python test/eval denied;
  .ax repaired the module and stored two cases), not a Result regression. Manager now
  reports live Python repair, restart and photo paths proven; none certifies pending C-core.
- Ownership: manager `ses_e8d6d9d1bffe8G7KEI4yUw8DPi`; sole main-checkout writer
  `ses_e8b17a23affedgH4V52JdwxNDL`; release writer `ses_e67e01ef9ffettWZDQAXWveOxc`.
  Main writer must not touch the release checkout. Review session
  `ses_e82647931ffewdBFqa9bGCmPIz` remains separate from write ownership. #118 is fixed
  by `463d839`, deployed and closed; two cases survive save/load. The release worker's
  live Python TelegramBot investigation is separate: this writer must not touch live modules.
- Accepted main work: G1 `fce692c`; #114 `5fb1ca8`; #112 `ce9096a` + structural hardening
  `56ce420`; #116 `ab2000a`; Result slice A `75da3aa`; JIT safety `2e9b3fc`; scalar VM
  safety `bbfe932`; invocation plumbing `8d8cfaa`. #97/#119/#86 source-fix status is
  distinct from deployment; this uncommitted C-core must not activate automatically.
  Strict parity remains RED: 324 verified / 58 invalid / 191 unsupported (573 entries).
- Superseding owner direction (#121): Python is the authoritative maintained source;
  .ax becomes a temporary importer. Separate Astra/DeepSeek worktrees handle serializer
  and persistence; no overlap with this C writer. Historical strict parity remains useful
  evidence, but preserving .ax spelling is no longer the ultimate acceptance goal. C does
  not change frontend guards, source formats or parity success criteria.
- Approved next sequence: Result contract A, then B (VM Check/unwind and fail-closed JIT
  admission), C (invocation/await normalization), then G7 frontend admission. Parked:
  G3 anonymous literals, G4 json_get typing,
  G5 shared Some, G6 vars, QualifiedEquivalent classification and the strict success
  definition. Preserve frozen fixtures and existing acceptance criteria pending decisions.
- Task/workspace implementation in the Python domain (#120) and model/turn permission
  changes are separately authorized/owned, not this writer's scope. No crossover into
  task/project/plan features here. Idle watchdog is explicitly disabled; do not arm it.
- Adoption is behavioral: #76 manager-workflow gates and #89's latest owner correction
  require native conversation capability parity before a default flip; retain legacy and
  the guarded OpenCode integration. Language parity alone is not cutover evidence.

### Approved Result contract — #82/#97/#86; slice A only

- This is Adapsis semantics, not Python inference. At every genuine invocation boundary,
  declared Result<T> wraps successful plain T once; explicit Ok/Err is preserved. Ordinary
  calls and user awaits preserve Result. No block/branch/loop-body normalization.
- A typed language Check/propagation failure unwinds scalar callers and becomes Err once
  at a declared-Result invocation. Operational/internal anyhow failures remain execution
  errors. Failure classification uses a Rust type, never rendered text. Context-wrapped
  language failures retain their original label; source chains of operational errors survive.
- `+call x:T` unwraps one Ok and propagates explicit Err. `+call x:Result<T>` remains an
  explicit compatible catching form. An intentional handler may continue; the failing
  callee and unwound scalar callers must not execute later effects. Nested Results require
  explicit outer Ok/Err at each Result layer; no recursive flattening or implicit nesting.
- Slice A adds `eval/result_contract.rs`: `LanguageFailure::{Check,Propagation}`,
  `complete_invocation`, `propagate_call`, `capture_call` and legacy `wrap_success`.
  Only `wrap_success` is connected: it replaces three identical existing success-wrapping
  blocks (tree eval entry, test comparator, Result-typed +call), preserving their behavior.
  The new typed contract is tested but NOT activated at current Check/call/await boundaries.
  Legacy catch-all error conversion and nested success wrapping are unchanged in A.
- Scope/compatibility: no VM, JIT, frontend, fixture, persistence or activation change.
  Before behavior-changing B/C activates, audit ordinary consumers expecting plain T from
  Result functions, user await into plain T, +call Result catching operational errors, scalar
  +call accepting explicit Err, and tests that mask wrappers. Preserve #114/await admission
  and the no-IO-replay guard. Release integration at 108e4dc excludes this work.

| Current discrepancy (not repaired by A) | Planned discriminating B/C gate |
|---|---|
| VM Check is PushErr+Return; scalar caller may continue | B: `check_unwinds_scalar_callers_before_effects`; direct VM/tree/normal async, zero requests before failure or exactly one preceding request, never the later request |
| JIT Check traps and Result ABI erases wrapper | B: `fallible_graph_refused_before_jit_execution`; safe numeric control stays eligible, hazardous direct compile and cached paths refuse before execution |
| Tree named entry/test comparator wrap; ordinary/parser/VM calls do not | C: `result_boundary_matrix`; structural plain/Ok/Err values through named, inline, session/API, CLI, stubs and spawned/route invocations |
| Tree +call unwraps Ok but keeps explicit Err; VM lacks that unwrap | C: `propagate_or_capture_call_matrix`; explicit Err unwinds, Result binding catches language failure only, one-layer unwrap preserves nested Result data |
| User await strips Ok; builtin await preserves it | C: `user_builtin_await_result_parity`; same-source/existing declarations, post-await failure, ordered synthetic IO and zero replay |
| Broad anyhow catches conflate language and operational failure | B/C: `failure_category_matrix`; identical text with different types, preserved source chain, operational error cannot be caught as Result |
| Native flags/callbacks can confuse application result with task success | C plus #104/#76: assert structural Result and execution status separately; no parsing formatted Err or claiming delivery/merge success from evaluator Ok |

Slice A's helper tests prove wrapping, explicit typed capture/propagation, nested-wrapper
requirements, payload identity and operational-error preservation only. The IO/entry/JIT
rows above are future integration gates, not claimed evidence from these pure tests.
Useful completion trials after C: validation→transform→write (invalid input means zero
writes); fetch→decode→reply (one request per step, deliberate recovery only); worker
validation→tool execution (failed Check means no later shell/tool action and truthful report).

Accepted slice A (`75da3aa`): 15 contract tests pass; `eval::` 324 passed /
1 ignored, `native_` 149 passed, `python_` 350 passed plus the existing strict-parity failure.
One full release gate: 1962 passed / strict parity 1 failed / 2 ignored; CLI 5/5. Strict
counts unchanged at 324/58/191. Log: `/tmp/opencode/full_gate_result_a.log`. No production
invocation migration or release activation follows from this helper-only checkpoint.

### Accepted JIT-only safety checkpoint — #97/#119 (`2e9b3fc`)

- #119 reproduced before repair: revision 1 `probe` returned 11 on JIT; revision 2
  changed it to 22 and introduced an unsupported List-returning sibling. Rebuild failed,
  yet normal cached dispatch returned `(11, used_jit=true)` instead of `(22, false)`.
  Evidence: `/tmp/opencode/jit_119_before.log`; no trap execution was needed.
- New program-aware JIT failure admission refuses reachable Check, `[fail]`, Result
  contracts/bindings/named fields and Result constructors. Direct compilation checks
  every emitted top-level function; cached entry checks the current root graph as well.
  Shared await-admission AST visitor is exposed crate-wide (visibility only); VM
  admission/execution behavior is unchanged. The decision is never cached across revisions.
- JIT dispatch extracted to `eval/jit_dispatch.rs`: cached revision must match exactly;
  clear stale cache before rebuild; failed compilation cannot invoke the old module.
  Matched warm same-revision and later successful revision controls remain on JIT.
- Former Check-trap tests now assert direct compile refusal, retaining their tree/VM
  operand/short-circuit and known VM-error-as-data cases. JSON-builtin refusal still has
  an independent undefined-callee control; admission is not proof of JSON JIT support.
  Result helper tests add triple nesting with a missing deepest wrapper, a middle Err,
  and an extra terminal wrapper pin (no general payload-type validation).
- Static compilation blast sample: examples 99 functions, local JIT-eligibility hint
  41 -> 39, new current-root graph eligibility 39; strict library 161 functions, local
  83 -> 81, graph 81. These are admission/hint counts, NOT emitted-code success counts:
  compiler backend limitations remain, and library functions are inside modules while
  the compiler emits top-level functions only. Whole-program safety admission allows
  19/26 valid example programs; two invalid files are excluded, not counted as covered.
- Safe numeric timing sample, 10,000 invocations per mode, before -> after:
  arithmetic direct 216 -> 222 microseconds, warm cached 2.10 -> 4.17 ms;
  numeric branch direct 216 -> 212 microseconds, warm cached 2.30 -> 4.08 ms.
  Both controls assert `used_jit=true`; timing is informational, noisy single samples,
  not a performance threshold. Logs: `/tmp/opencode/jit_safety_{before,after}.log`.
- Still open: VM NotAdmitted/Executed distinction, scalar Check unwind, temporary VM
  quarantine for Result/catching graphs, combined B/C normalization and explicit catch
  activation. No caller behavior migration, frontend relaxation, or callback ACK change
  ships in this checkpoint. Run source compatibility audit before behavior-changing
  VM/B/C activation; no production self-modification or replay-after-IO relaxation.
- Review freeze: focused admission/cache/Result/await tests 44 passed; native 149 and
  Telegram port 73 passed. One full release gate: 1979 passed / strict parity 1 failed /
  2 ignored, CLI 5/5; strict counts unchanged at 324/58/191. Log:
  `/tmp/opencode/full_gate_jit_safety.log`. No exhaustive mutation campaign was run.

### Accepted VM scalar-safety checkpoint — #97 (`bbfe932`)

- Baseline reproduced on a genuinely VM-admitted graph: ordinary scalar failing helper
  between two builtin awaits emitted `[before, after]`; required result is `[before]` and
  a typed failure. Log `/tmp/opencode/vm_scalar_before.log`. No user-await/field/Result
  admission hides the reproduced failure. Check now emits RaiseCheck and aborts the scalar
  VM stack; tree Check produces the same LanguageFailure::Check label.
- `VmAttempt::NotAdmitted` is pre-execution only; `Executed(Err)` never retries tree, with
  or without earlier IO. Sync preserves legacy Err(...) formatting; async returns the
  execution error. Operational errors stay operational. The no-IO-replay guard remains.
- Eager preparation compiles/reuses each unique reachable callee before any instruction,
  shares one growing interner, resolves original call spellings and terminates cycles.
  Nested unsupported code (e.g. shared reads) and non-Await builtin IO select tree before
  the first effect; missing prepared bytecode is an internal execution failure.
- Result contracts/captures/constructors temporarily select tree, preserving legacy
  catching (including operational capture) until combined B/C. No full catch activation,
  Result normalization, frontend relaxation or callback ACK reinterpretation. Tests retain
  literal Result opcode evidence through a cfg(test)-only pure `compile_result_opcode_fixture`
  plus production refusal/fallback assertions; those fixtures cannot execute Check or IO.
- Focused regressions count actual synthetic IoRequests: Check failure before IO emits
  zero; after earlier IO emits only before; success continues once; operational execution
  failure also suppresses later requests. Tail/recursive/nested-argument failure, a tree
  stub replay discriminator, alias identity, nested shared-code fallback and explicit
  Result capture recovery are covered. Frozen fixture bytes are unchanged.
- VM emission eligibility: examples 52 -> 49 of 99; strict library 107 -> 99 of 161.
  Preparation can additionally refuse nested unsupported callees. Informational scalar
  sample (1,000 iterations, one helper called twice): preparation 4.59 ms, full admitted
  compile/execute 8.40 ms; noisy current measurements, not a before/after speed claim.
  Logs `/tmp/opencode/vm_scalar_blast.log` and `/tmp/opencode/vm_scalar_cost.log`.
- Review freeze: 16 new tests; focused VM suite 91 passed, eval 331 passed/1 ignored,
  native 149, Telegram port 73, await/field admission 14 each. One full release gate:
  1995 passed / strict parity 1 failed / 2 ignored; CLI 5/5, strict still 324/58/191.
  Full log `/tmp/opencode/full_gate_vm_scalar.log`. No broad mutation campaign.

### Accepted C invocation/observer plumbing — behavior-preserving (`8d8cfaa`)

- `eval/invocation.rs` receives the actual FunctionDecl and a declared or selected-stub
  body, returns the existing raw outcome, and optionally calls a trusted Rust observer.
  No production caller installs that observer; nested callees do not inherit it, and it
  receives no coroutine handle or nested private body. Stub selection/authority, program-only
  expression context, named frame setup and depth guards remain unchanged.
- Moved real named/parser/nested/async/test/spawned/source/route invocation sites to the
  wrapper; raw startup/lifecycle and synthetic statement blocks remain outside it. CLI
  run-async historically used an unframed body entry; its explicit legacy-unframed wrapper
  retains that behavior (including the root shared-scope limitation) in this checkpoint.
- No normalization/capture/propagation/await activation: current raw-vs-named-result
  discrepancy, external error rendering and legacy catch-all behavior are retained.
  Legacy trace is untouched and explicitly characterized: a branch returning 7 is skipped
  and trace reports a later 9; a Check inside a branch is likewise skipped. Correcting this
  requires the later shared-executor/trace C checkpoint, not an implicit plumbing change.
- Metadata-only release audit (no live reads by main writer): migrated TelegramBot48 /
  TgFormat18, no Result-returning functions, no +call captures or user-Result awaits;
  two Result-parameter helpers and seven Result-typed builtin awaits. Preserve app unions
  and builtin Err behavior. The frozen old Result callback signature is not a live claim.
- C activation must explicitly migrate operational catch consumers: self-hosted REPL
  division currently reaches a broad +call Result catch; use an authored domain Check/Err
  where recovery is intended, never string classification or frozen-fixture edits. Test
  observers may assert execution failure but may not create recoverable language values.
  Other live-module metadata audit, generic ACK #104, and frontend guard relaxation remain
  separate gates. No task/workspace/plan features are authorized for this writer.
- Review freeze: 10 boundary/observer characterizations pass; eval 341 passed/1 ignored,
  VM 91, native 149 and Telegram port 73 passed. One full release gate: 2005 passed /
  strict parity 1 failed / 2 ignored, CLI 5/5; strict unchanged324/58/191. Log:
  `/tmp/opencode/full_gate_c_plumbing.log`. No expanded mutation campaign.

### Historical C-core activation freeze — interpreter-only (now committed as `113f80f`)

- The actual-FunctionDecl wrapper now calls complete_invocation once for declared bodies
  and selected stubs. Ordinary calls/user await preserve Result; +call T propagates explicit
  Err and unwraps one Ok; +call Result captures typed language failure only. Arguments are
  evaluated before the target's catch scope. Lifecycle/raw blocks have no inferred contract;
  CLI's legacy unframed setup remains, but its declared return contract is normalized.
- Shared statement_execution/tracing modules replace the partial trace interpreter.
  Branch/loop/match bodies really execute; root-only observation does not inherit into
  private callees and never re-evaluates expressions for logging. Trace appends normalized
  invocation output and gains no IO handle. Test/trace observers can assert/report an
  operational error without exposing it as a recoverable language value.
- Deliberate compatibility changes: broad operational +call Result capture is removed.
  Division-error recovery now aborts with zero recovery requests; an explicit domain Check
  counterpart recovers once. Named and VM sync adapters preserve Rust Err structurally,
  so API/native flags now report execution failure; application Value::Err and error-looking
  Strings remain success data. CLI eval success=false and run-async Rust Err exit nonzero;
  app Err values still exit successfully. No generic task ACK #104 reinterpretation.
- Frozen legacy Telegram wrapper is misannotated String but returns builtin Result. User
  await now preserves it, so BOTH success/failure reach the String consumer error after
  one request. Only its characterization changed, not frozen bytes. Corrected app-union
  port and migrated 48/18 metadata are distinct and retain their existing IO contracts.
- Structural/IO matrix covers declaration, parser/named, nested/stub, async/tracked-task,
  trace, test, API/session/native, route and CLI entries. Validation→write emits zero writes
  on domain failure, one on success. Before/fail/after and argument-failure cases suppress
  later effects; intended typed recovery emits once. Operational failure after earlier IO
  reports success=false with one request and unchanged Program snapshot.
- Result graphs stay quarantined off production VM/JIT. No Python G7/user-Result-await
  guard relaxation, fixture exemptions, task feature changes or deployment. Other live
  signatures, routes and operational-catch consumers require a separately authorized
  metadata/source audit before activation. Existing successful live smoke is not C proof.
- Review freeze: 14 new unit/boundary tests and four new CLI tests. Focused eval350 /
  1 ignored, API241, VM91, native/port gates included in the full run. One full release
  --no-fail-fast: 2019 passed / strict parity 1 failed / 2 ignored; existing Python CLI5/5
  and Result-contract CLI4/4. Strict unchanged324/58/191 (zero AST mismatches/non-idempotence).
  Log `/tmp/opencode/full_gate_c_core.log`. No expanded mutation campaign or deployment.

## Historical recovery — Python parity #86 (2026-10-28 onward, red)

### Accepted slice — G1 (Python pseudo-field properties), commit `fce692c`, baseline `5fb1ca8`

- PRODUCTION frontend change (native mutate/test/evaluate/inspect), owner-approved choices:
  names `len`, `is_empty`, `is_ok`, `is_err`, `unwrap` only; PROPERTY form lowering to the
  same `FieldAccess` AST the `.ax` source has; type-gated (str/list for `len`/`is_empty`,
  Result for `is_ok`/`is_err`/`unwrap`); a real struct field of the same name wins (the
  pseudo-field table is only consulted for a non-struct base); everything else keeps a
  refusal, with dedicated reasons for Map/Set and for `.len` on a Result. Documented as
  NOT runnable CPython (README, native prompt, `pseudo_fields.rs`).
- Code: new `python_frontend/pseudo_fields.rs` (type table + refusal text) and a
  7-line branch in `expressions.rs::field_access`. The printer already emitted
  `base.field`; the preflight `surface::attribute_access` already accepted these names
  (it never type-gated), so no other production file changed. Execution: the VM's
  `GetField` is untouched; #114's `field_admission` routes these functions to the tree.
- Strict PARITY before -> after (54 files / 573 entries): verified 298 -> 324 (+26),
  unsupported 217 -> 191, invalid 58, mismatches 0, non-idempotent 0. Entry diff: 21
  `function` and 5 `program` Unsupported -> Verified, plus 12 entries that stay
  Unsupported with a later reason; no Verified entry lost. Programs gained: expr_parser,
  http_server, imperative, self_read, tokenizer. Identical to the earlier probe. Residual
  top-level function blockers: G3 anonymous literal 4, Int/Str type-mismatch + comparison
  (G4 json_get typing) 3, G7 implicit Ok wrap 7, G5 shared `Some` 2.
- Evidence: 8 frontend tests (`pseudo_field_tests.rs`: AST equality with `.ax` for every
  form + tree results, print/reprint identity, chains/conditions/loops, same-name struct
  field precedence incl. VM compile, 17-case refusal matrix, native-entry agreement, VM
  refusal + async-after-IO tree result), 3 native tests (`python_pseudo_field_tests.rs`:
  mutate+test+evaluate+inspect, six refusals leaving the program unchanged, async
  `.len` after an awaited read with one synthetic FileRead), 1 harness test. 19 mutations
  (every table row, wrong result types, three leaks, both refusal texts, struct-field
  precedence x2, field renaming, #114 admission disabled) all fail tests; each restored.

### Accepted slice — G2 (`input` as a binding name), baseline `762de63`

- PRODUCTION frontend change (native mutate/test/inspect, `adapsis test --language python`),
  owner-approved. `mod.rs::forbidden_binding(name)` = `forbidden` minus `input`, used at
  exactly five binding sites: parameters (`lower.rs::ordinary_args`), locals/loop targets
  (`Lower::bind`), match captures, value names and field-access bases (`surface.rs`).
  Every other role keeps full `forbidden`: callee (`direct_name`), function, type, member,
  shared-variable, annotation and attribute names. `input` is Python's stdin builtin, not an
  execution escape, and not an Adapsis builtin; the non-executing frontend can only shadow
  it. The A3 test exemplar for "unprintable member" moved from `input` to `eval`.
- Strict PARITY before -> after (54 files / 573 entries): verified 291 -> 298 (+7),
  unsupported 224 -> 217, invalid 58, mismatches 0, non-idempotent 0. Entry diff: 30 entries
  changed, no Verified entry lost. 7 `function` Unsupported -> Verified (adapsis_in_adapsis
  parse_atom/parse_let/parse_return, expr_parser parse/calc/parse_atom, Peer.eval_payload);
  23 stay Unsupported with a NEW next reason: 18 functions and 5 programs now stop on
  pseudo-field access (G1, deferred behind #114). Identical to the earlier read-only probe.
- Evidence: 7 frontend tests (`input_binding_tests.rs`: bindings at every site through the
  free and the native entry, callable refusal bound/unbound, an existing `.ax` `+fn input`
  never callable qualified or bare, function/type/member/shared/annotation/attribute
  refusals, other escape names unchanged, `input=` test keyword, `.ax` print roundtrip),
  2 native tests (`python_input_name_tests.rs`: mutate+test+evaluate+inspect with an `input`
  parameter and local; five refusals leave the program unchanged), 1 harness test. 13
  mutations (each of the 5 sites, exemption widened to `eval`, and exemption leaking into
  each of callee/function/type/member/shared/annotation/attribute) each fail the intended
  tests; files restored and verified.

### Accepted slice — A3+B2 (whole programs + module-less stored tests), baseline `58b6999`

- `#[cfg(test)]` / harness only; no runtime, validator, native-tool, fixture, `lower.rs`,
  `expressions.rs`, `types.rs` or `printer.rs` change; `success()` untouched.
- A3 `check_program`: top-level types + functions are printed as ONE module-less submission
  (`parity/top_level.rs::print_all`: types in program order, then functions), lowered by
  `lower_toplevel` against the program WITHOUT them, and rebuilt in the order top-level types
  -> modules (unchanged `print_module`/`lower_module_with_program` path) -> top-level
  functions. Then `equality::program` over the whole Program, the module reprints and the
  top-level reprint must be byte-identical. Module-only programs are unchanged. Per-member
  exactness is already enforced inside `print_toplevel_type/function`, so a failing member
  surfaces its own reason as the program's reason.
- B2 `parity/test_blocks.rs`: owner-"" blocks are exact BY CONSTRUCTION. The runtime runs
  `+test name` against `name` verbatim (`Program::get_function`: top-level first, else the
  first module), and the new test-only `tests_lower::lower_tests_toplevel` keeps a bare
  target bare and a dotted target dotted (`qualify` now uses `types::owned_name`, identical
  to the old `format!` for every module). The harness requires the target to exist
  (InvalidProgram otherwise) and every lowered declaration to carry exactly the runtime
  string (AstMismatch otherwise). Module-scoped blocks keep the session rule (#110).
  `tests_lower.rs` is the one non-harness file touched (behavior-neutral refactor +
  the cfg(test) entry). The Python test-data literal policy is unchanged.
- #111: `tests_print::literal` refuses any NEGATIVE numeric constant node (`Int(<0)`,
  sign-negative `Float`) with an explicit reason. The parser yields such a node only for
  keyword shorthand (`+with a=-1`); `{a:-1}`, positional, list items and expectations are
  `Unary Neg`, which Python's `-1` also lowers to. Without the refusal two corpus blocks
  (`simple_syntax` #4, #5) are AstMismatch; with it they are Unsupported (#111).
- Strict PARITY before -> after (same 54 files / 573 entries): verified 251 -> 291 (+40),
  unsupported 264 -> 224, invalid 58, mismatches 0, non-idempotent 0. Entry-by-entry diff
  (67 entries changed, none left Verified): 30 top-level `Test` Unsupported -> Verified,
  10 `program` Unsupported -> Verified, 13 `Test` and 14 `program` stay Unsupported with a
  concrete reason (test data outside the literal policy: `Some`, records, calls; `input`,
  pseudo-field, anonymous-literal, `vars` blockers; unprintable module bare calls).
- Evidence: 9 new unit/harness tests (6 in `parity/tests.rs`, 3 in
  `tests_print_tests.rs`) plus 2 changed existing `parity/tests.rs` tests. 11 mutations (negative Int, negative Float, target existence,
  target-string equality, module entry for owner "", `qualify`, top-level function rebuild,
  top-level type rebuild, catalog keeping top-level functions, type ordering) each fail
  the intended tests; files restored and verified. ONE surviving mutation, documented:
  dropping the top-level REPRINT comparison in `check_program` is undetected because no
  natural fixture produces a non-deterministic printer (the equivalent module check is
  likewise covered only through `same_text` itself); `print_all` determinism across a
  rebuild is asserted directly instead.

### Accepted slice — A2 (module-less functions, owner ""), baseline `5057d8a`

- `#[cfg(test)]` surface only; no authority model, never reachable from native tools or any
  runtime path. `success()` untouched; frozen corpus untouched.
- `top_level.rs::lower` now also accepts annotated `def`/`async def` (types and functions mix,
  emitted in source order). Signatures: qualified `Module.f` keys for module functions
  (unchanged `collect_signatures`) plus BARE keys for top-level functions only
  (`collect_toplevel_signatures`, plus the submitted functions after a first pass, so
  recursion, mutual recursion and forward references resolve). A bare callee that is not a
  top-level function is refused ("unknown local callable"); module functions stay reachable
  qualified only, even though `Program::get_function` would fall back to the first module.
  An unresolvable qualified callee is refused at the top level (module behavior unchanged).
- Only non-test-surface edit: `expressions.rs::call` now decides `Target::User` from
  "bare name with a signature" (`local_user`) instead of `resolved.contains('.')`, and builds
  `resolved` with `types::owned_name` (bare at owner ""; identical to the old
  `Module.name` for every module, whose name is non-empty). Printer: a bare user call is
  printable only at owner "" and only for a real `program.functions` entry
  (`toplevel_callee`, `cfg!(test)`), and `param_type_hints` has the matching bare branch;
  module printing keeps refusing every unqualified non-builtin call. `lower.rs` gained one
  `#[cfg(test)]` fn (`toplevel_functions`); `printer.rs::collect_signatures` was refactored
  through the shared `function_signature` helper (behavior-neutral).
- Declaration collision rules (explicit messages): builtin/alias/IO builtin, frontend helper,
  Python-reserved/forbidden names (existing `Definition::from_stmt`), `startup`/`shutdown`,
  `@route`/`@source`, duplicate in the submission, existing top-level function, module name,
  ANY union variant (existing or submitted) and ANY type name. Two unions sharing a variant
  stay accepted; using it is reported at the use site ("ambiguous union variant").
  Local bindings named like a top-level function are refused by the existing bind rule.
- Harness `parity/top_level.rs::check_function`: print -> lower into the program WITHOUT the
  function -> `equality::function` -> second print byte-identical -> second lowering equal.
  `print_toplevel_function` picks `program.functions` only.
- Strict PARITY before -> after (same 54 files / 573 entries): verified 207 -> 251 (+44),
  unsupported 308 -> 264, invalid 58, mismatches 0, non-idempotent 0. 44 function entries
  moved Unsupported -> Verified; the other 43 top-level function entries stay Unsupported
  with concrete reasons instead of "no top-level scope surface": `input` parameter (G2) 24,
  pseudo-field access (G1) 10, anonymous struct literal (G3) 4, json_get typing (G4) 3, shared
  `Some` ambiguity (G5) 2. The probe predicted ~43 (44 measured; the cause of the
  one-entry difference was not investigated).
  Programs (A3) and top-level stored tests (B2) are untouched.
- Evidence: 13 new unit tests (`top_level_function_tests.rs`: unwrapped ops, recursion on
  tree and VM, mutual recursion + forward reference, call of an earlier submission, struct
  literal argument hint through a bare callee, types+functions order, qualified/unknown
  callees, a 15-case name-collision matrix plus duplicates/decorators/binding, shared
  variants, effect/await rules for bare user calls, module regression: module-local bare
  calls still qualified/refused and a module function still cannot print a bare call).
  15 mutations (printer owner guard, top-level-function guard, hints, `Target::User`, resolved
  qualification, dotted-callee guard, bare signatures, every collision rule, lifecycle,
  decorators, duplicate, type catalog) each fail exactly the targeted test(s); every file
  restored and verified.

### Accepted slice — A1 (module-less type surface, owner ""), baseline `d1ae977`

- `#[cfg(test)]` only; no runtime, validator, native-tool, fixture, `lower.rs`, `types.rs` or
  `expressions.rs` change, parity.rs `success()` untouched. Only non-test source touched:
  `Context::error` labels an empty module owner `<top-level>` (unreachable for modules,
  whose names must be non-empty identifiers).
- New `top_level.rs`: `lower_toplevel(source, &Program)` returns UNWRAPPED
  `Operation::Type`s with `Context.module == ""`, which is the real catalog owner
  (`owned_name("", n) == n`, `catalog_from_program` inserts `program.types` with owner "").
  It refuses shared variables, functions (A2), docstrings, imports and statements with named
  reasons. `print_toplevel_type` (printer.rs) picks `program.types` only (a module type with
  the same bare name is a different owner) and proves the text re-lowers to an identical type.
  Harness `parity/top_level.rs::check_type`: print -> lower into the program WITHOUT the type
  (the validator's own duplicate rule stays in force) -> `equality::ty` -> second print
  byte-identical -> second lowering equal.
- Declaration-time TYPE collision rules (new, explicit messages): type vs module name; type
  vs any union variant (existing top-level or module types, or submitted) and the symmetric
  variant vs any type name. Existing member rules still apply (builtin/syntax/keyword/dunder
  type names, duplicates). Two unions sharing a variant (`Some`) stay accepted: the ambiguity
  is a use-site fact (A2). Replacement is not simulated: the validator rejects duplicate
  top-level types.
- Strict PARITY before -> after (same 54 files / 573 entries): verified 184 -> 207 (+23),
  unsupported 331 -> 308 (-23), invalid 58, mismatches 0, non-idempotent 0. Only `type`
  entries moved: Unsupported/type 24 -> 1, Verified/type 0 -> 23 (entry-by-entry diff:
  exactly 24 entries changed, all `type`). Remaining: `adapsis_in_adapsis.Env` (member
  `vars` is an unsafe member name, G6). Matches the sizing probe (+23) exactly.
- The harness's pre-existing repeated-target rule already reports a module and a type
  sharing a spelling in one file as Unsupported ("repeated declaration"); the lowering rule
  is covered separately in unit tests.
- Evidence: 7 unit tests (unwrapped ops; owner "" vs module owner; every type shape incl.
  recursive, Option/None_, builtin-named variants roundtrips + identical reprint; 26-case
  collision/rejection matrix incl. builtin, syntax, keyword, dunder, duplicate, module,
  variant-vs-type both directions, shared/function/docstring/import/empty; legitimate
  overlaps; unprintable member) + 1 harness test. Mutation checks (drop module rule, drop
  variant rule, wrong owner) each fail tests; files restored byte-for-byte. Targeted
  `python_` 308 passed + the expected strict-corpus failure; ONE full gate 1874 passed /
  strict parity 1 failed / 2 ignored; CLI 5/5. Log /tmp/opencode/full_gate_a1.log.
- `#110` is now referenced in the bare-test-target reason text (detail strings of 3 entries
  changed, statuses and group counts unchanged).
- Frozen for review; no commit. Next (approved): A2 (top-level functions), then A3+B2
  (sign quirk #111: refuse negative numeric in keyword shorthand).

### Accepted checkpoint `d1ae977` — B1 (module stored-test printer, strict 184/58/331)

### Accepted checkpoint `2e1e018` — D3 (TgFormat stored tests: 39 carried, 17 ax-only, 1 redacted)

### Accepted checkpoint `fae91e7` — D2 (TelegramBot original stored tests)

- All 61 TelegramBot stored cases dispositioned (41 carried, 2 ax-only, 1 covered elsewhere,
  14 redacted, 3 obsolete); manifest-driven gate with set equality.

### Accepted checkpoint `a4ab092` — D1 (doc, shared paths, startup, route)

- Exact module doc, all 9 shared in frozen order, startup (qualification + effect spelling,
  no semantic repair), webhook route; real-HTTP route test; #108 pinned.

### Accepted checkpoint `9fa4da2` — polling slice C (poll_loop), 42/42 functions

- poll_loop bounded to 1000 cycles with clear-before-spawn successor (#107); latch kept on
  outer errors (#109); Bool latch not an identity.

### Accepted checkpoint `2a5d0cd` — polling slice B (poll_once, baseline `da46d2e`)

- poll_once (typed_poll_once_v1): explicit builtin Result, json_valid guard, no-result and
  no-valid-updates backoff; 41/42.

### Accepted checkpoint `da46d2e` — json_valid primitive (baseline `614051a`)

- Pure total json_valid(String)->Bool == the shared serde parse step of json_get/
  json_kind/json_array_len succeeds; not strict RFC 8259; >127 nested containers and
  1e400 false, u64 max valid (validity != representability); JIT refuses implicitly.

### Accepted checkpoint `614051a` — polling slice A (baseline `8c38ace`)

- Fixtures/tests/docs only. Four originals: extract_update_id (unchanged AST/doc,
  original 42/0 cases), init (unchanged), handle_update (spawn-target qualification
  only) and dispatch_updates (typed_dispatch_v1) -> 40/42, 2 pending (poll_once,
  poll_loop). NEW pure batch_update_id_valid separately inventoried (46 total with
  the text entry, validators and upload classifier). Shared intake_mode/poll_offset
  added; token_path/allowed_path/poll_running pending.
- dispatch_updates repairs (R3/R4): non-array/absent result -> "no result" via
  json_kind; invalid/missing/non-int/negative/>=i64::MAX update_id or non-object
  element skipped without advance or spawn, batch continues, ", skipped k" only when
  k>0. Offset rule and advance-before-spawn preserved (R5): fire-and-forget, at-most-once,
  no outcome (#104), same-chat ordering inside a batch not guaranteed.
- Limits kept: malformed top-level JSON and ids above i64 stay terminal (json_valid is
  a separate approved checkpoint); handle_update returns "{}" unconditionally and cannot
  authenticate webhook senders (#108); Python spells async last so the corrected
  origin uses [io,mut,async] (effect set equal).
- Four named normal-async tests on both origins intercept +spawn as IoRequest (qualified
  name, exact args/order, capacity 256 > 100-update page, process_update never runs) and
  read typed shared poll_offset back; 84 pure cases (18 new). Targeted 50/50; ONE full
  gate 1829 passed / strict parity 1 failed / 2 ignored; CLI 5/5. Log
  /tmp/opencode/full_gate_a.log.
- Authored manifest/origin/Python blobs (slice A, historical):
  `134870e6175c256d160d5ff1aca88ed3521492ba`,
  `73c69dae365ef7116903d6187eed34e7c6d7f0e4`,
  `f62ae623548c40bdcaaf0f5611f005b3c777aaab`.
  Independently accepted and committed as 614051a; no deployment.

### Accepted checkpoint `8c38ace` — outbound closure

- Three originals send_reply_with_attachment/send_audio_file/extract_reply (36/42) plus
  the NEW pure telegram_upload_attempt; caption '&' refused before IO (#106); nonempty
  text must be Sent before one upload; #104 callback ACK and #99 unchanged.

### Accepted checkpoint `fc2952b` — original-name intake

- Two originals process_update+describe_update_kind→33/42;9pending. Three NEW
  pure validators separately inventoried with existing NEWtextentry (37total).
  No dispatcher/new shared declarations needed. Already-reviewed helper/entry
  implementations unchanged; original description helper exactAST/doc preserved.
- Explicit stricter consumed-field preflight: required objects/intidentity/date,
  bounds incl date<=9223372036854775, optionalmessage_id int/missing compatibility,
  strings/media containers/selected IDs/numericmetadata; emptyphoto fallthrough,
  invalid lower-priority consumed fields defer0IO. Floatlatitude and harmless
  unusedaudio.duration allowed. MalformedJSON terminal0IO; authorization beforeIO.
- Preserve valid input text/caption/image/video/voice/fallback precedence, exact
  structured principal/context/profile/metadata/callback construction. Return
  TgTextTurn/[fail] propagates outcomes; getFile guidance delivery means only
  guidanceattempt, not mediaprocessing success. Binaryerror terminal/noretry.
  Original awaited-call arguments proved modulo qualification and folded identical
  group/supergroup dispatch; no false whole-source equivalence claim.
- Five named tests directly invoke originalprocess_update on bothorigins, no driver:
  raw6modes×3chattypes×2roles×5outcomes; precedence/metadata, malformed/unauth0IO,
  fallback kinds, terminal failures/guidance. Exact ordered requests, memorybytes/
  MIME/name, no unexpected replay.63Telegram purecases incl6NEWvalidator cases,
  38TgFormat; targeted43/43. Onefullgate1822passed/strictparity1failed/2ignored;
  CLI5/5; log `tool_18affd088001GxvfTWvM8k1QIH`.
- Authored manifest/origin/Python blobs:
  `215be83f8cef91b3aab10d17c198a6853885ce16`,
  `16f0835665972f9a3dcc6ba37aa46f19db92364d`,
  `4dfecc6054b135a10c37923d69542d2c76d202b4`.
  Accepted and checkpointed atfc2952b; no liveIO/deployment. At this checkpoint
  outbound attachments/webhook/poll/lifecycle/#99/#104 remained separate.

### Accepted checkpoint `016711a` — two voice helpers

- Fixtures/tests/docs only;31/42originals+1NEWtextentry,11originalspending.
  transcribe_voice is an intentional #91 authored control-flow repair, not inferred
  equivalence to malformed duplicate-Err/out-of-scope-e source. Explicit getFile
  Result binding/outer error placeholder, [fail], actual upload Result passed intact
  to transcription_upload_response. Both file_path and text require nonempty
  JSON Strings; wrongkind/missing returns existing placeholders, legitimate
  null/{} /[] Strings preserved. JSON parse and binary outer errors are terminal.
- Helper inverse-guard proof preserves all remaining original AST/signature/doc.
  Strict global corpus unchanged/still invalid; only the port original-equivalence
  subset handles malformed transcribe_voice through a separate authored-repair
  gate. It remains in original inventory and full corrected/Python AST/idempotence.
  Named coverage enforced: trusted Result helper17cases/tree+VM/bothorigins;
  normal async34cases/origin checking exact1/2/3requests and no fourth/retry.
  Lookup failures stop1, download failure stops2, upload outcomes finish3; no
  live ASR/network/files, caller/process_update/context/takeover integration.
- Targeted38/38; single fullgate1817passed/strictparity1failed/2ignored;CLI5/5.
  Log `tool_18ae802c4001WxBSoBgqxiBBzD`. #99/#104 remain open. Current authored
  manifest/origin/Python blobs:
  `73812273798591049536755bfb9be391c9b77432`,
  `2597eb267b35aac54c11f074f8aecbfda7db02ff`,
  `7a1a205585e04bb4fcdd2a571722bc379381a00a`.
  Accepted and checkpointed at016711a; no deployment.

### Accepted checkpoint `3c10387` — media acquisition/handlers

- Fixtures/tests/docs only: adds telegram_file_path, telegram_file_error_message
  and three attachment handlers. **29/42 originals plus1NEW text entry**, 13
  originals pending; TgFormat remains18/18. NEW test-only PortMediaDriver is
  excluded from inventory/source parity; original process_update remains pending.
- Actual Attachment annotation/roundtrip admission passed without frontend changes.
  File lookup explicitly awaits/matches builtin Result while retaining original
  String/error-String contract (inverse AST proof); malformed JSON is terminal.
  Attachment-handler takeover prefixes are preserved; nested TgTextTurn delivery
  outcomes propagate without extra retry. Semantic repair ledger now11 functions.
- Both-origin normal async fixtures intercept getFile/binary/takeover/reply requests,
  checking private/group/supergroup × admin/non-admin × photo/video, metadata,
  principal/profile/callback and tiny memory-backed Attachment bytes/name/MIME.
  No-reply, HTML success, plain fallback success, rejection and uncertainty covered;
  acquisition/parse/binary/takeover errors stop later requests. No live IO.
- Targeted port **35/35**; pure cases51Telegram+38TgFormat per origin/tier;
  exact combined AST/idempotence and source-fidelity checks pass. One full gate:
  **1814 passed, strict corpus alone failed, 2 ignored; CLI5/5**, log
  `tool_18ad87056001s1tyPm13lmCcNP`. Frozen corpus/runtime/old wrapper unchanged
  against3f81a9e. No #91 voice, #99 splitting or #104 callback ACK resolution.
- Current authored manifest/origin/Python blobs (not historical proofs):
  `832b2641358270dbad9e58963348481d0cd1cadc`,
  `f04b14733900e2a9fabc70125e945840a14139a7`,
  `ad6ab3c68f1f5bf83f8587091ce7e7bae10561cd`.
  Accepted and checkpointed at3c10387; no deployment.

### Accepted checkpoint `3f81a9e` — #105 NEW-entry adoption

- Fixtures/tests/docs only. The introduced process_text_update entry now checks
  exact json_kind labels before conversion/auth/IO: int chat/from/date with prior
  value bounds, string chat.type/text, and every present media kind deferred.
  Missing/wrong-kind IDs now return Ignored before to_int; primitive parse/u64
  errors remain terminal/no IO. Integer date0 is accepted. No original function,
  handler, runtime, catalog, primitive or permission rule changed.
- Removed only the NEW-entry List.contains identity guard and text rendering/
  prefix heuristics. Strings null/{} /[] /prefix-leading now reach the correct
  structured principal/profile routing; non-String collision partners defer0IO.
  Chat-type wrong kinds ignore0IO. Huge represented-u64 errors and serde-rounded
  Float IDs ignore/error0IO, while signed-i64 edge IDs pass valid bounds.
- **Original #105 presence gate GREEN with unchanged assertions:** all11media
  keys ×2origins present-empty defer with zero IO; absent controls remain admitted.
  The diagnostic still proves json_get alone erases absence/present-empty, so no
  json_get semantic change is hidden. No exclusions, regexes or raw-JSON scans.
- Targeted port **32/32**; exact combined AST/idempotence and existing routing,
  profile, outcome/state-isolation and unchanged pure gates pass. One full gate
  **1811 passed, strict corpus alone failed, 2 ignored; CLI5/5**. Log
  `tool_18a63a9f0001dV4TpnBuPZYcrt`. Corpus54/573 remains **116/58/399**,
  other buckets0. No original process_update parity or deployed/remote ACK claim.
- Count remains **24/42 originals plus1NEW entry**, 18originalspending, TgFormat
  18/18functions with other module surfaces pending. #91/#99/#104 remain separate.
  Authored manifest/origin/Python blobs recorded now:
  `bb40ae70239db2601b802c4de96e0209b2b2ea1d`,
  `6feb6273874b8974a080bb8686293ba42422b9c9`,
  `c8c6b1356686b8ea11b1445744403910e5793d54`.
  Adoption is explicitly ledgered against3e58e30; primitive/frozen corpus/wrapper
  code is unchanged. Accepted and checkpointed at3f81a9e.

### Accepted checkpoint `3e58e30` — #105 json_kind primitive

- Pure `json_kind(json,path)->String`, exact two String arguments, no aliases/IO:
  closed missing/null/bool/int/float/string/array/object labels. Extracted
  `eval/json.rs` shares one unchanged resolver with json_get, preserving its
  scalar conversions, missing/null/aggregate outputs and error text. Oversized
  eval/mod.rs and builtins.rs shrink; JSON registry constants live in a leaf.
- Exact legacy path grammar retained: empty dot components ignored/root, usize
  components (including +0/00) index arrays even on objects; other/overflowing
  numeric components are literal keys. No new bracket/escape syntax or malformed
  String-path error. Wrong container/out-of-bounds is missing. Argument type/arity
  and malformed JSON are actionable evaluator errors.
- `int` means represented signed-i64. Represented u64 above i64 returns Err;
  finite serde floating representation is float, including 1.0/-0.0 and oversized
  integer lexemes rounded into floats. Parser overflow errors; no lexical-precision
  guarantee, dependency or serde-feature change. json_get behavior is unchanged.
- New accurate frontend `(str,str)->str` signature, registry metadata, legacy and
  native prompt guidance and Python example docs. Tests cover type/presence
  collision pairs, root/path/index/key quirks, min/max i64/u64 overflow/rounded
  floats/exponent overflow, invalid JSON/arguments, unchanged get outputs, no IO,
  Python exact roundtrip and native mutate/test/inspect/evaluate with hidden-state
  canary and authorization denial. No port heuristic adoption or fixture edits.
- Native protocol limitation explicitly pinned: evaluator Err is rendered in
  `eval_results` as `= Err(json_kind...)` while `has_errors=false`; this is NOT a
  correctly flagged error status or a successful kind String. Program snapshots
  remain unchanged. The legacy mutation_results vector also carries eval reports,
  so emptiness of that vector is not a mutation test. Generic protocol untouched.
- Targeted json_kind **7/7**, json_get regressions **5/5**, frontend catalog **7/7**,
  registry **42/42**. Single full gate **1808 passed, 2 failed, 2 ignored; CLI5/5**.
  Failures are unchanged corpus and deliberately retained #105 port presence gate.
  Log `tool_18a4a8d3c001x6vHUIJ7ODvyoG`. Corpus54/573 remains **116/58/399**,
  other buckets zero, no causal113→116 attribution. No deployment/port-completion
  claim; adoption is a separate subsequent scope after primitive review/checkpoint.
- New untracked review files: eval/json.rs +json/tests.rs, builtins/json.rs,
  python_frontend/json_kind_tests.rs, api/native_tools/json_kind_tests.rs.
  Frozen/port fixtures and their tests are byte-unchanged from a731d12. Manager
  owns checkpoint after independent review; no writer commit/push/liveIO/deploy.

### Accepted checkpoint `a731d12` — partial text-routing RED evidence

- Five more original functions: is_admin/is_allowed and the three text handlers.
  **24/42 originals** now have fixtures; **18 originals remain pending**.
  `process_text_update` is a separately ledgered NEW partial entry, not counted
  among 42 and not a replacement/stub for still-pending process_update. No runtime,
  frontend, permission, callback dispatcher, original-corpus or wrapper changes.
- Handlers return TgTextTurn: NoImmediateReply is not delivery success; nested
  DeliveryAttempt preserves the exact TgSendAttempt. Exact AST prefix comparison
  preserves original takeover context/profile/callback/metadata construction
  (only format_sender qualification differs). Corrected-source/Python combined
  AST and idempotence pass. Named union payload references retain their canonical
  Struct spelling but resolve to the declared union; actual nested values pass.
- Coverage is explicit: unchanged 49-case direct-VM pure gate, separate normal
  shared-state contracts (with direct VM refusal controls and restored fake state),
  original IO contracts and NEW-entry tests. Both-origin intercepted real
  IoRequest::LlmTakeover tests verify private/group/supergroup, admin/allow-list
  policy, sender != chat, structured from.id principal, forged name/embedded-prefix
  controls, profiles/callback arguments, immediate outcome propagation and empty
  reply behavior. No model, remote or callback execution/ACK is certified (#104).
- NEW-entry dynamic guards use List.contains/Value::matches to prevent Bool true,
  integral Float or numeric String becoming an administrator Int. Nonzero chat,
  positive sender and explicit nonnegative integer date precede authorization.
  Huge/nonnumeric/null/missing/object/array inputs fail/ignore before IO without
  panic; date 0 is admitted. Stub positive controls prove guards precede auth
  helpers. This depends on runtime identity semantics, not sound JSON static typing.
- Text null/object/array values collapse into Strings in json_get. The explicitly
  partial entry therefore conservatively defers text `null` or starting `{`/`[`,
  including legitimate same-looking Strings. No equivalent change is made to
  original extract_text/handlers. This heuristic is not a complete raw-text policy.
- **[#105](../../issues/105) gate intentionally RED:** absent media and present
  malformed `""` both yield identical json_get String(""). For **11 media keys ×
  both origins**, present-empty currently emits **1 takeover versus required 0**.
  The absent control is admitted. No exemption, regex workaround or narrower
  success claim hides the failure; new entry must not be deployed.
- Targeted port **29 passed / 1 failed**. One full gate **1801 passed, 2 failed,
  2 ignored; CLI 5/5**. Failures: strict corpus and #105 presence gate. Log
  `tool_18a249871001Lpi5gnJ40TGwiq`. Corpus unchanged 54/573, **116/58/399**,
  other buckets zero. New review files: `telegram_port_tests/routing.rs` and
  `routing/tests.rs`. Existing source/wrapper/pure expected values remain intact.
- json_kind is **design accepted only**, not implemented: pure closed type labels,
  same resolver/path semantics as json_get, absent vs null, actual signed-i64
  integer vs Float, explicit malformed/path/out-of-range error behavior. A separate
  reviewed primitive should replace NEW-entry identity/text heuristics and turn
  the RED presence gate green; no generalized coercion or parallel resolver.
- Authored manifest/origin/Python blobs recorded now:
  `d265ace303facc296d39a27f10a1c902620bdbe9`,
  `0b9503e9ed39bea2444bda76504f973dbb68842e`,
  `bc0f301bed9b2ed23d29c2f60e1017f3afc6c649`.
  #91/#99/#104 and full port/deployment remain separate. Await evidence review;
  manager owns checkpointing. No live IO, credentials read, writer commit/push/deploy.

### Accepted checkpoint `d9b8a5f` — two-function reply closure

- Fixtures/tests/docs only: add send_message and send_reply, bringing TelegramBot
  to **19/42 original functions** (15 pure, 4 IO), with **23 functions pending**.
  Both preserve parameters but deliberately change return to TgSendAttempt and
  effects to io/fail/async, explicitly await their qualified callee and propagate
  the outcome unchanged. Changes are separately ledgered semantic repairs; no
  additional retry or runtime/frontend/permission change is introduced.
- send_message retains the exact existing TgFormat formatting and 3800 split
  limit; #99 is not fixed or certified. send_reply preserves destination parsing.
  Combined-module production signatures, exact whole-program AST equality and
  print/re-lower/idempotence pass; Python runtime fixtures lower both TelegramBot
  and TgFormat from Python, rather than substituting the `.ax` dependency.
- Both-origin normal eval_async_function fixtures assert exact HTML/body/headers,
  negative and leading-zero destinations, Sent/Rejected/Uncertain propagation,
  confirmed-rejection fallback order, no caller retries, malformed terminal errors
  and zero requests for empty text/invalid destination. Only short inputs and
  fake token/intercepted requests are used. Existing frozen wrapper fixture and
  frozen corpus are unchanged; no exactly-once remote delivery claim follows.
- **[#104](../../issues/104) remains an integration blocker.** Read-only source
  inspection: main.rs spawned-task paths near 1373/1783 turn every evaluator
  Ok(val) into Completed(format!(val)); await_spawned_callback_with_timeout50–76
  accepts all Completed values; optional callback dispatch129–131 returns
  `notified`. Tagged Rejected/Uncertain would be misclassified as success there.
  This slice certifies callable outcome propagation only, **not callback delivery
  acknowledgement**. Dispatcher/routing edits are out of scope; later design must
  be transport-independent/legacy-compatible, not parse formatted strings or
  hardcode Telegram tags in the generic runtime.
- Targeted port **22/22**. One full gate **1794 passed, strict corpus alone failed,
  2 ignored; CLI 5/5**. Log `tool_189f1b88b001aPvQjLSfFWbeRq`. Corpus remains
  **116/58/399** over 54 files/573 entries, other buckets zero; no 113→116
  attribution. New review file:
  `telegram_port_tests/send_chain/runtime/reply_closure.rs`.
- Authored current manifest/origin/Python blobs respectively:
  `9825305b5c72096fec8ea65bf9fab5f85cab795c`,
  `352e9a45e95e969da5479baef7340e5af3215bac`,
  `a4cf09e55a087f3d98c93f8073b7c5b976ffb631`.
  Routing/media/lifecycle, remaining shared declarations and original stored-test
  surfaces remain pending; #91 repair is separate. Manager owns checkpoint after
  independent review. No live IO, writer commit, push or deployment.

### Accepted checkpoint `b3cf08e` — corrected send-chain policy

- Fixtures/tests/docs only. TelegramBot now has **17/42 original functions**:
  15 pure helpers plus repaired send_message_part/send_message_parts. New
  `TgSendAttempt` is an application union with Sent/Rejected/Uncertain variants,
  each carrying String; it is not Result and does not change runtime unwrapping.
  The original sanitized empty telegram_token shared declaration is copied
  exactly; tests inject only `fixture-token` in isolated state and intercept IO.
- The two repaired functions deliberately change return type (String→union),
  effects (`io,fail,async`), control flow and docs, preserving input parameters.
  `semantic_repairs` records these separately from qualification-only changes.
  Corrected-origin/Python whole-program AST equality, print/re-lower/second-print
  identity, union schema and runtime-argument validation all pass with the existing
  frontend. No runtime/permission/typing permissiveness or user-Result-await
  relaxation was introduced. Earlier frozen-wrapper tests are untouched.
- Builtin http_post Result is explicitly matched. Only actual parsed Bool true
  yields Sent; Bool false yields Rejected and authorizes **one** plain fallback.
  Any non-Sent after that optional fallback terminates remaining chunks; failure
  cannot be masked by later success. Transport/missing/nonboolean status yields
  Uncertain. Malformed JSON raises a terminal parse error (not a uniform tag).
  Empty/invalid index yields Uncertain with zero requests; forged legacy `last`
  data cannot manufacture Sent. Outer await rebinding and empty wildcard-arm
  execution are proven by actual normal-route tests, not just syntax inspection.
- Tests invoke production `eval_async_function` on **both origins**, assert raw
  Union results and exact URL/fake-token/body/header/order/count. Cases include
  whitespace, Bool versus strings/numbers/null/objects/arrays/missing status,
  malformed JSON on first/plain attempt, transport error containing `ok:true`,
  second rejection/uncertainty, multiple chunks and early termination. These are
  local synthetic attempts, **not exactly-once remote-delivery evidence**. All
  chunks are explicit and short; #99 remains unfixed and no service-limit-safe
  splitting is claimed. json_get's wrong static String catalog remains explicit;
  the rendering discriminator is supported by runtime type-path tests only.
- Targeted port **20/20**, including 5 new send-contract tests. Pure paired cases
  are now **49 TelegramBot / 38 TgFormat**, passing both origins and tree/direct
  VM; the two effectful functions have separately required normal IO coverage.
  Single full gate **1792 passed, strict corpus alone failed, 2 ignored; CLI 5/5**.
  Log `tool_183bdc377001I5xt3mp0AYgNsT`. Frozen corpus remains **116/58/399** over
  54 files/573 entries, other buckets zero, no 113→116 causal attribution.
- Remaining: **25 TelegramBot functions**, eight original shared declarations,
  module doc/startup/route/original stored tests and higher send/reply/intake/media
  integration; TgFormat original stored tests remain pending. #91 repair and full
  port/deployment acceptance remain open. New untracked review inputs are
  `telegram_port_tests/send_chain.rs` and `send_chain/runtime.rs`.
- Current authored blobs (recorded now): manifest
  `b2c45fada813ac94c8fb68c2dfc96f4f99cbf9d1`; Telegram origin
  `f04115ffb50dd85be2f4b4562ca4a66c14b37d80`; Python
  `a77449e0fbae33da057f765ba30c26d3522ad538`; paired tests
  `fae6cf3542ac0092b6e813d4725f44040e711690`. Frozen originals/provenance untouched.
  This bounded application-policy slice awaits independent review; manager owns
  checkpointing. No live IO, credentials read, push or deployment.

### Accepted checkpoint `862ccc2` — pure port closure expansion

- Fixtures/tests/docs only. **14/42 TelegramBot functions** and **18/18 TgFormat
  functions** are now in the separate corrected-origin/Python pairs. Telegram's
  added pure helpers are parse_id_list, extract_sender_name, speaker_principal,
  source_metadata. TgFormat's module doc is included unchanged. No frontend or
  runtime semantic edits, no send-path repair, no stubs and no frozen-corpus edits.
- New test-only `telegram_port_tests/qualification.rs` applies only the approved
  local-call qualification to cloned frozen function ASTs, then checks exact
  equality ignoring typed NodeIds. Per-function site counts are ledgered and
  verified: **37 TgFormat sites, 5 TelegramBot sites**. Corrected-origin↔Python
  AST equality, print/re-lower and second-print byte identity pass. Qualification
  intentionally changes source spelling/resolution; this is not frozen-source
  parity or a production global name-normalization change.
- Every implemented function has a paired assertion. **46 TelegramBot cases**
  and **38 TgFormat cases** pass on both `.ax`/Python origins and tree/direct VM,
  including code fences/HTML escaping, Unicode, split boundaries, ID-list loops,
  media metadata and principal-from-from.id despite spoofed names/text prefixes.
  VM result checks now use the existing Value comparator instead of potentially
  ambiguous display-string equality. No existing accepted expectation was changed.
- Two new characterizations preserve inherited quirks, not desirable contracts:
  missing message_id becomes quoted-empty content because to_string uses Value
  Display; missing date errors exactly `to_int: cannot parse '""' as integer`.
  **[#99](../../issues/99)**: 5000 ASCII content characters remain one part at
  limit 3800, plain (5000 raw characters) or pre-wrapped (5011 raw HTML characters).
  Content alone exceeds the documented Telegram 4096 cap. Splitting is **not
  service-limit safe**; no live Telegram rejection was measured or sourcefix made.
- Targeted port **15/15**. Single final full gate **1787 passed, strict corpus
  alone failed, 2 ignored; CLI 5/5**. Log `tool_18380c2f10017KYq3nwwlNt5B6`.
  Frozen corpus remains 54 files/573 entries, **116/58/399**, other buckets zero;
  no causal/per-entry 113→116 attribution. Full parity and deployment remain open.
- Remaining: **28 TelegramBot functions**, its module doc/shared/startup/route
  surfaces, original stored tests for both modules, full intake/takeover/media/
  lifecycle/delivery integration. #91 voice repair still unimplemented. TgFormat
  function coverage is not complete original-module surface coverage.
- Send-error repair remains **design only**: explicit builtin Result match and
  trusted app outcome tagging; success only actual parsed Bool true, one plain
  fallback only actual Bool false. Uncertain transport/malformed/nonboolean status
  or rejected plain fallback stops remaining chunks. This is a deliberate future
  corrected-origin policy change. `to_string(json_get(...))` is not proof of sound
  static typing; actual Bool/string/number/null/missing/malformed type-path tests
  are required. No global user-Result-await loosening or exactly-once remote
  delivery claim. Current send wrappers remain unchanged and still error on Err.
- Authored now (not historical baseline proof): manifest blob
  `b345d790bf9507a8de31972c4b3f460c2eea0fa3`;
  Telegram origin/Python/tests blobs respectively
  `1934b33dc355019d12d2470a396e70bde6e6f507`,
  `99426da7d97dc7a5cbf589c506c1a77cfbabfcb0`,
  `df9c957e03d4370e54e17821204c33be0622445f`;
  TgFormat origin/Python/tests blobs respectively
  `29894ad367edb5650f6b37f737c87e370200868e`,
  `6125a46151eed1471f5b79ca373dd718149040f2`,
  `149b15b438270d085e9c2b7fa667dddd59100b64`.
  Frozen original blobs remain pinned/verified in the manifest. This slice awaits
  independent review; manager owns the next checkpoint. No live IO/push/deploy.

### Accepted checkpoint `e5337b0` — #82 dispatch admission, baseline `a2551b7`

- Manager checkpoint `a2551b7` contains the previously accepted compiler, partial
  port evidence and narrow #98 repair. This new slice changes production only in
  `vm/await_admission.rs` plus its declaration and first-line check in
  `vm::compile_function`. The check runs before Compiler construction, argument
  slot allocation or root bytecode emission; no VM instruction has executed.
- An iterative worklist exhaustively visits executable StatementKind/Expr slots:
  call arguments (including nested calls), let/set/return/yield values, call/
  spawn/await targets, checks, branch conditions/both bodies, while/each
  conditions/iterators/bodies, match subjects/all arms, record values, field bases,
  unary/binary operands, source timer expressions and emitted event payloads.
  Source handler names are future-registration metadata, not immediate calls.
  A reachable Await resolving to a user function produces a categorical compile
  error selecting the existing tree path before execution. Unknown targets and
  builtin-only graphs retain their existing behavior.
- Function identity is a pointer used solely as a visited-set key within the
  borrowed immutable snapshot: no dereference, persistent cache or cross-revision
  reuse. The supplied root is scanned directly even if owned outside/unregistered
  in Program; callees resolve through the original spelling and `get_function`.
  Same-named functions in different modules are distinct identities. SCCs terminate;
  caller-module-first rewriting is forbidden. False-positive graph edges for
  builtin/variant collisions may conservatively choose tree, never redirect calls.
- `eval_async_function`/CoroutineHandle/`io_performed`/continuation and replay
  guards are untouched. Whole-root admission prevents an earlier builtin effect
  being executed before discovering a transitive user await. Tests cover exact
  request order/count before/after IO and in nested arguments, child failure with
  no later effect/replay, original mocks/stubs, guest denial before mocks/requests,
  and bound turn identity. These do not certify every historical permission issue.
- Normal frozen-wrapper dispatch now matches the existing explicit tree diagnostic
  for both bare and qualified targets: one request then bare String on success;
  one request then `index_of expects (String, String)` for transport Err. Only
  normal-route Rust test expectations changed; source fixture/provenance bytes and
  direct-tree diagnostics are preserved. Wrapper unboxing/error handling, #97,
  and the Python Result-returning-user-await restriction are **not** changed.
- Performance tradeoff: any syntactically reachable user Await, even in an untaken
  branch, makes the invocation use tree (including pure async user calls). A
  synthetic 128-function recursive SCC with repeated edges visits **1025 work
  items**; **1000 scans took 23971 µs** in the final author targeted release test
  (~24 µs/scan, with test counters). This is only synthetic scan cost, **not** a
  production latency benchmark. Lazy compiles may repeat scans; no global cache.
- Author targeted: user-await 13/13, port 13/13, existing VM 41/41, native call-site
  8/8, scoped 20/20 and unchanged no-user-await post-IO division/no-replay 1/1.
  Single final full gate: **1785 passed, strict corpus alone failed, 2 ignored;
  CLI 5/5**. Log `tool_1834efc94001YdvI1XigmzXSvF`. Corpus remains 54/573 with
  **116/58/399**, other buckets zero. This slice awaits independent review.
- Review inputs: new `vm/await_admission.rs`, `vm/await_admission/tests.rs`,
  `vm/await_admission/tests/dispatch.rs`; two integration lines in `vm.rs` and
  the normal-route wrapper assertions. Current wrapper Rust test blob (recorded
  now after this change): `6ada414dd8737af07be8ccb60b9b951f415a0a43`; the older
  authored-hash table below is the prior checkpoint, not a claim of current bytes.
  No writer commit, push, deployment, live IO or frontend policy relaxation.
- Correction to an earlier binder report: `eval::input_to_vm_args` is
  **pub(crate)**, not private, but accepts named fields rather than lower_tests'
  multi-argument `_index` carriers. The richer canonical tree binder is private.
  The existing tested port adapter is unchanged in this dispatch slice.

### Accepted checkpoint `a2551b7` — narrow #98 repair

- `vm/equality.rs` routes only differing Value variants, excluding Int↔Float
  numeric pairs, through the tree's shared `Value::matches` and its negation.
  `matches` is minimally `pub(crate)`. VM same-family/mixed-numeric `binary_cmp`
  and all ordered opcodes are unchanged. No empty-string special case, json_get
  coercion, frontend typing change or JIT change was introduced.
- New both-order heterogeneous Eq/Neq matrix covers Int/String (empty and numeric
  text), Float/String, Bool/String, None/String, Ok/Err/Union versus String and
  Bool/Int. Same-type scalar controls and ordered mixed-type errors remain.
  Existing residuals are deliberately characterized separately: VM Int↔Float
  equality coercion differs from tree; NaN/identical-infinity Neq remains unequal
  between tiers; same-family Result/Union/None equality remains unsupported in VM.
  These #90/#86 residuals prevent a claim of full equality parity.
- **Unmodified port assertions now pass:** TelegramBot **28/28 on both origins
  and both tree/direct VM**, TgFormat 7/7 likewise. The first-slice RED gate was
  not exempted or rewritten. Frozen originals, port `.ax`/Python/test files and
  manifest remain byte-identical. Their README/observed-status metadata preserve
  the earlier RED-evidence checkpoint; this recovery header records the current
  pending runtime correction. Frontend json_get's String catalog mismatch remains
  unresolved; #82 user-await/wrapper behavior is unchanged.
- Targeted equality 3/3, port 13/13, existing VM 41/41. Final author
  `cargo test --release --no-fail-fast`: **1772 passed, strict corpus alone failed,
  2 ignored; CLI 5/5**. Full log `tool_18326f8420017vJBiGyyjCQ0pJ`.
  Corpus remains **116/58/399** across 54 files/573 entries; other buckets zero.
  No attribution for 113→116. `git diff --check` clean.
- New untracked `src/vm/equality.rs` must accompany the cumulative untracked
  compiler/port evidence files at review. No commit/push/deployment. Narrow #98
  repair received independent bounded ACCEPT (VM 47/47 and port 13/13); the preceding port slice was accepted
  only as RED evidence, not as a complete functional port. #82 dispatch design
  is next after this review, based on the preserved wrapper diagnostics.

### Authored port evidence hashes — recorded now, 2026-10-28

These are Git blob hashes computed from the **current authored working-tree
files**, after the bounded #98 review. They are not a claim that these new files
existed in baseline `05d85cc` or any earlier historical checkpoint. The source
provenance hashes in the port manifest separately identify the frozen originals.
Paths below are relative to `tests/fixtures/telegram-python-port/` unless noted:

| Authored file | Git blob |
|---|---|
| `manifest.json` | `9346a60cdf9ffa3407e8216b1aed501b80b839d6` |
| `origin/TelegramBot.ax` | `617280c450eb209ab3b38d8405ef4c7427bee2f3` |
| `origin/TgFormat.ax` | `a95ac4d862f69822d7617ef7300f01a0e78bf33e` |
| `python/TelegramBot.py` | `4281f8934c8957e5da14455602e3996c4f704c47` |
| `python/TgFormat.py` | `00e5b1e58312c235885d278e8a9cba17a1baf1b7` |
| `python/test_telegram_bot.py` | `0fd159ab44d1444b064a679af0d9a4e4eb2cf9c4` |
| `python/test_tg_format.py` | `61aa42fc491640f34cd2c05e914002f6b5b91068` |
| `src/python_frontend/telegram_port_tests.rs` (repo-relative) | `8b43c5142f8169bb6b763216f4f3820e68b01e28` |
| `src/python_frontend/telegram_port_tests/io_contracts.rs` (repo-relative) | `2db7a4837269ce27460184078bedb9098669ddee` |
| `src/python_frontend/telegram_port_tests/wrapper_contract.rs` (repo-relative) | `87313e911ca403c1a0b3d5594837d8b9b4beabcb` |

### Accepted earlier checkpoint — first partial port, RED evidence gate

- Tests/fixtures/docs plus minimal test-module wiring only; the independently
  accepted compiler slice below was preserved. Separate files under
  `tests/fixtures/telegram-python-port/` contain **10/42 TelegramBot functions**
  and **3/18 TgFormat functions**, `.ax`/Python pairs, literal paired tests,
  README and manifest. Exact source Git blob hashes and baseline are pinned;
  implemented/pending sets account for every original function, with a
  per-function transformation ledger. Frozen corpus files/manifest are untouched.
- Selected bodies/docs exactly match their frozen function ASTs. Corrected partial
  origin→Python AST equality and print/re-lower/second-print identity pass. Source
  declaration selection/order and new test-data provenance are disclosed. No
  stubs, #91 voice repair or full module/shared/lifecycle/route port is hidden.
- **[#98](../../issues/98): keep direct-VM port gate RED.** All 28 TelegramBot
  tree cases pass on each origin; direct VM passes 25 and fails cases 1/3/4:
  numeric chat/sender IDs compared to empty String. Actual `json_get` scalar
  variants are String/Int/Float/Bool (null/object/array have String encodings),
  unlike the String catalog declaration. Both source origins fail identically;
  no coercion, ignore list or source rewrite hides this runtime discrepancy.
  TgFormat's 7 cases pass both origins and both tested tiers.
- Six contract tests cover production CoroutineHandle requests/replies and actual
  JSON variants using in-memory replies, with no real file/sleep/network/provider
  work. This is valid-call runtime contract evidence, **not complete frontend IO
  signature/arity support**. Missing IO table entries remain a #86 gap.
- **[#82](../../issues/82): wrapper diagnostic corrected and pinned.** Normal
  `eval_async_function(send_message_parts)` emits zero requests and fails unknown
  user-await operation for both bare and qualified targets. It is not a
  qualification-only bug. Qualification-only test-copy changes are inverse-AST
  verified. Explicit direct-tree diagnostic reaches the boundary: successful
  `send_message_part` returns Ok(String), user await unwraps it, caller succeeds
  after one request; transport Err remains wrapped and caller fails at `index_of`
  after one request, before fallback send. Normal-dispatch success is not claimed.
- Final author gate: **1768 passed, 2 failed, 2 ignored; CLI 5/5**. Failures are
  the unchanged strict corpus and explicit direct-VM port gate. Targeted port
  tests **12 passed / 1 failed**. Full log: `tool_1831c9387001ynVtLFOgkJ2K6P`.
  Strict corpus remains 54/573, **116 verified / 58 invalid / 399 unsupported**,
  other status buckets zero, still no causal/per-entry 113→116 attribution.
- New untracked review inputs: `src/python_frontend/telegram_port_tests.rs`, its
  `io_contracts.rs` and `wrapper_contract.rs` children, and the whole separate
  fixture directory. Include earlier untracked `compiler/truthiness.rs` and
  `python_frontend/condition_tier_tests.rs` when reviewing the cumulative cut.
  No commit/push/deployment/live IO. First port evidence awaits independent review;
  full corrected-origin port, #86/#90/#96/#97/#98/#82 remain open.

### Accepted compiler checkpoint — immutable normalization (uncommitted)

- Built on the accepted NOT checkpoint below. All JIT Boolean consumers now use
  one representation-aware normalization: AND/OR, NOT, Check, Branch and While.
  Input evaluation is preserved even for always-truthy Float; branch results are
  canonical I8. Identity-safe homogeneous scalar comparisons are admitted for
  conditions; NOT retains its narrower binary/argument-call refusal contract.
- Review correction: **Float Eq/Neq now refuse condition compilation**, including
  nested conditions, rather than emitting IEEE equality for the runtime's epsilon
  comparison. Near-equal finite values, signed zero, infinities from scalar calls
  and NaN AST cases have explicit refusal coverage; finite epsilon fallback is
  separately verified through the normal evaluator with `used_jit=false`.
  Float ordered comparisons and Bool/Int Eq/Neq remain direct-JIT controls.
  **OPEN #90 residual:** VM Neq uses `abs(a-b)>=EPS` while tree uses
  `!(abs(a-b)<EPS)`; NaN/identical infinities therefore disagree. JIT refusal
  does not fix that VM fallback discrepancy. Global float equality is unchanged.
- Manager-approved exception preserves the two existing direct-JIT
  `runtime_semantics_tests.rs` division-program gates: condition calls may take
  exact Bool/Int/Float arguments, emitted recursively from proven values in
  authored order. No String/unknown conversions or IO/async callees. Explicit
  `[fail]` callees in the new argument-call path, including nested arguments,
  refuse compilation until [#97](../../issues/97) is resolved. Zero-argument
  fallible calls retain their existing **tier-specific trap characterization**;
  a scalar signature alone does not prove the runtime returned wrapper/value,
  and caller admission never certifies all callee internals.
- Local/rebound/unknown conditions refuse compilation before execution, including
  mutable-counter loops. The normal evaluator falls back to VM/tree; tests assert
  the result and `used_jit=false`, not fake direct-JIT evidence. No mutable-binding
  machinery or general compiler namespace refactor was added. Coarse admission
  still accepts more shapes than final compilation.
- Author final `cargo test --release --no-fail-fast`: **1756 passed, strict parity
  alone failed, 2 ignored; CLI 5/5**. Corpus remains **116/58/399** (54 files,
  573 entries; other status buckets zero), no 113→116 attribution. Full log
  `tool_182f4ee0c0012k4GVf4G1AEFwc`. Targeted: condition tests 6/6, compiler 41/41,
  VM 41/41, runtime semantics 7/7, logical 17/17, native no-replay 1/1. Final
  `git diff --check` clean. Independent Astra ACCEPT followed, including release
  build, condition 6/6 and direct refusal/ordered controls; no whole #90/#86 or
  deployment acceptance follows from that bounded review.
- New untracked files to include in review/durable cut: `src/compiler/truthiness.rs`
  and `src/python_frontend/condition_tier_tests.rs`. #96 remains open; direct
  nested-call tests use current-executable children with deadlines, not fallback.
- **[#97](../../issues/97) author-observed bug, not intended semantics:**
  `fail_bool()->Bool [fail]` fails a Check; caller `probe()->Int [io,async]`
  then attempts `file_exists`. Normal `eval::eval_async_function` runs the VM:
  caller performs **one subsequent effect**, whereas direct tree evaluation
  aborts with **zero**. VM Check currently emits Err+Return from its callee frame;
  the caller continues. `trusted_ax_failed_check_caller_effect_characterization`
  pins this discrepancy separately from normalization correctness. It must not
  be cited as acceptable failure propagation or all-tier parity. A separate
  trusted `.ax` fixture verifies that an actual VM division error after one IO
  request does not replay that request. Neither is evidence of JIT/Python IO
  admission. No #82/#94 guards were loosened. #90 and #86 both remain open.

### Current accepted-but-uncommitted checkpoint

- Durable baseline: `05d85cc`. The pending frontend Boolean-admission/warning
  cleanup plus narrow representation-safe JIT NOT slice has **independent Astra
  ACCEPT**, including the subsequent String and scalar scratch-collision repairs.
  Runtime code is frozen pending the next approved design; no commit, push or
  deployment has been authorized here. Acceptance is **not** whole #90/#86 or
  deployment acceptance. Full sanitized Python TelegramBot port is still absent.
- Author final `cargo test --release --no-fail-fast`: **1750 passed, 1 strict
  parity failure, 2 ignored; CLI 5/5**. Targeted logical suite **17/17**.
  Final author log: `tool_182cd0c46001A7J44EJkArxih6` under the local OpenCode
  tool-output directory. Reviewer independently passed release build and 17
  logical tests, and direct CLI String/scalar collision cases returned **1/0**.
- Corpus: **54 files / 573 entries; 116 verified, 58 invalid, 399 unsupported**;
  input errors / AST mismatches / non-idempotent entries all zero. There is **no
  verified per-entry or causal attribution for 113→116**. Zero AST mismatches
  among admitted entries is not full feature coverage.
- Current sessions/models (all explicitly confirmed `openai/gpt-6-astra`):
  manager `ses_e8d6d9d1bffe8G7KEI4yUw8DPi`;
  author `ses_e8b17a23affedgH4V52JdwxNDL`;
  independent reviewer `ses_e82647931ffewdBFqa9bGCmPIz`.
  Earlier provider/reviewer history is historical, not current routing.
- **Include new untracked `src/compiler/truthiness.rs`** in any later authorized
  durable cut; ordinary tracked-file diff summaries omit it. It holds NOT's
  proven-operand emission and mechanically extracted, still-unfixed AND/OR.
- **#90 remains open:** AND/OR and Check/Branch/While normalization is the next
  design-only bounded step. The NOT slice deliberately refuses binary operands,
  argument-bearing calls and unproven/rebound values before execution, retaining
  tree/VM fallback; zero-argument scalar callees are not globally certified.
- **#96 remains open:** parallel-suite JIT relocation panic, documented below.
  Child isolation of the regression avoids the observed interference; it does
  not fix production relocation or establish its root cause.

### Authorized next scopes — compiler slice, then TelegramBot port

- Next #90 implementation is shared normalization for AND/OR and
  Check/Branch/While using **representation-proven immutable** operands. Mutable
  condition dependencies must refuse JIT compilation before execution and use
  existing VM/tree fallback; no tracked mutable-binding write-path extension is
  authorized. Inventory found no direct-JIT loop execution test: compiler.rs
  `compilable_body_with_while` tests only coarse admission, and VM loop tests are
  bytecode evidence. This distinction must remain explicit. Existing direct-JIT
  short-circuit user-call tests in `runtime_semantics_tests.rs` must be retained.
- Immediately after that bounded slice is frozen/handed to independent review,
  the sole writer's priority becomes the TelegramBot port, ahead of optional JIT
  work. [#86](../../issues/86) authorizes a **separate reviewed corrected sanitized
  `.ax` origin plus complete paired Python port**, with provenance and an explicit
  transformation ledger. The frozen parity corpus stays untouched; exact AST and
  idempotence for the corrected origin form a separate port gate, never a claim
  that the original fixture or whole parity gate passed. No runtime async or
  permission guard loosening, new builtin, live Telegram/network/credentials or
  deployment is authorized.
- Concrete port transformations/blockers (line numbers reference the frozen
  `tests/fixtures/python-parity/library/TelegramBot.ax`, not a live library):
  1. Bare local calls throughout (`is_admin:64`, `process_update:234`, startup
     `spawn poll_loop:21`, route `handle_update:25`) have no exact Python spelling
     under `python_frontend/printer.rs`'s bare-call refusal. Qualify calls in the
     **new corrected origin**, recording the deliberate dispatch/AST change.
  2. Repair [#91](../../issues/91)'s malformed `transcribe_voice:410–425` nested
     match/duplicate Err/out-of-scope `e` in that new origin. Original fixture
     remains invalid; repair is not evidence its original AST was representable.
  3. `send_message:442–445` and `send_message_parts:605–613` return async calls
     directly. Rewrite to annotated await-bind-then-return in the new origin;
     `python_frontend/call_sites.rs`'s async gate remains intact.
  4. [#94](../../issues/94): `send_reply_with_attachment:457–466` directly matches
     `http_upload` after awaiting `send_message`; `transcribe_voice:419` does so
     after binary download. Use direct-builtin await-bind-then-match in the new
     origin and test no replay after prior IO. Address #82 only if a concrete
     port call awaits a Result-returning **user** function, not as a blanket
     prerequisite for fallible IO builtins.
  5. Verify actual IO arity/wrappers for startup `read_file`/`file_exists:13`,
     `poll_once`/`http_get:84`, `process_update`/binary Attachment downloads:234,
     takeover metadata/attachment overloads in `handle_*:323–375`, and
     `send_message_part`/`http_post:596`. Pure `builtin_signatures.rs` inventory
     does not establish these. Use `runtime_arguments.rs`, `call_sites.rs` and
     production coroutine dispatch with trusted in-memory IoRequest responses.
     Current IO typing is **not complete**: `expressions.rs` special-cases
     `read_file -> String` and `http_get -> Result<String>`, while many required
     IO results remain Unknown. `runtime_arguments.rs` checks exact user-call
     arity but treats bare IO as builtin/effect checks, not a complete signature
     table. Do not add broad Unknown acceptance or call this strict IO coverage.
     Actual coroutine `http_post` returns Result<String>; frozen
     `send_message_part:598` binds it as String, then its user-function result is
     consumed by `send_message_parts:609–610`/`telegram_response_ok`. The first
     port scope must test the actual direct-builtin wrapper **and intervening
     user-await boundary** before diagnosing or changing that binding. It is not
     an ignored result; do not silently unbox it or count a correction as original
     AST parity. `llm_takeover` metadata overloads likewise require exact request
     field assertions, not only an await-mode acceptance test.
  6. Build missing complete Python source and paired tests: start with a port
     skeleton, pure `extract_*`, `source_metadata:202`, `speaker_principal:191`,
     `format_sender:140`, `telegram_response_ok:587` and TgFormat dependencies.
     Then raw-update admin/guest/group routing through `process_update:234`,
     takeover/callback binding, photo/video/voice flows, split/retry order
     `send_message_parts:605`, and bounded startup/poll/shared/spawn coverage
     `13,68–108`. Existing `bundled_modules_tests.rs` and `api/takeover_native`
     are harness entrypoints, not substitutes for executing the Python port.
- These port tasks can progress on tree/VM with trusted mocks; optional mutable
  JIT support and open #96 relocation are **not** port prerequisites. Port code
  must not be interleaved into the current compiler slice.

### Historical recovery ledger — superseded snapshots, not current status

The entries below retain intermediate measurements, rejected claims and later
corrections. Words such as “current”, “awaiting review” or “not implemented” in
this ledger refer to that earlier checkpoint; the header above is authoritative.

- Full feature equality is **not achieved**. Last pre-foundational-fix gate:
  54 fixture files / 573 inventory entries; 114 verified, 58 invalid-program
  entries (4 roots plus descendants), 329 unsupported, 72 exact AST mismatches.
  Release tests: 1662 passed, parity gate failed, 2 ignored; CLI 5/5 passed.
  These are historical measured counts, not current-fix verification or file coverage.
- Foundational steps 1–2 now reject non-exact bare-call inspection, share full
  structural comparison after typed NodeId clearing, and restore known operand
  checks. Latest `cargo test --release python_`: **130 passed, 2 failed** — the
  corpus gate and the separate new owner-colliding-type native regression.
  Corpus remains 54 files / 573 entries: 90 verified, 58 invalid, 425 unsupported;
  zero AST mismatches now means earlier refusal, not fixed representation gaps.
  Full release suite has not been rerun for this foundational delta. Target slot
  yielded to TUI after the focused run; no main builds until manager handoff.
- After the next manager handoff, bounded owner-aware catalog work fixed the
  duplicate-type regression. Current release gate: **1672 passed, corpus parity
  alone failed, 2 ignored; CLI 5/5 passed**. Corpus counts remain 90 verified /
  58 invalid / 425 unsupported. Internal type identities now retain declaration
  and signature owners without changing persisted AST names. Both-order native
  struct/recursive-union inspect/resubmit and hidden-competitor tests pass.
  This slice awaits independent review before the native-reference pass.
- Review found a hidden-schema risk in filtered unique fallback after owner
  erasure. The bounded pre-executor binding guard now checks full-state metadata
  (owner/kind/order only), including global union reclassification, and refuses
  colliding external fallback before schema validation. Native canaries and safe
  local/top-level/no-collision controls pass. Latest release gate: **1678 passed,
  strict corpus parity alone failed, 2 ignored; CLI 5/5 passed**. Corpus counts
  remain unchanged. Independent bounded re-review accepted this correction;
  whole #86 remains red and no release/deployment approval follows.
- Slice 2 was subsequently authorized and implemented (server-derived target/
  same-submission union-constructor roles, sequential lexical field scopes,
  neutral collision refusal, owner-qualified type closure): final combined gate
  **1705 passed, strict parity alone failed, 2 ignored; CLI 5/5**. Frozen for
  independent review. External bare Variants remain explicitly unsupported and
  visible. #94 after-earlier-IO direct-match limitation is documented; slice 3
  (shared defaults) is not implemented.
- #95 fixed and slice 3 implemented as the next authorized pair. #95: native
  Python mutation refuses a declared function whose spelling collides with a
  variant in any module (state becomes unreachable), while the runtime recheck is
  now scoped to the reachable call graph so legacy collisions keep unrelated
  functions usable. Slice 3: module shared declarations are structurally
  preflighted, shared-only submissions are valid, and defaults are lowered by a
  data-only walker and type-checked against the complete local catalog. Final
  combined gate: **1712 passed, strict parity alone failed, 2 ignored; CLI 5/5**.
  Frozen for independent review. Record defaults remain outside the native
  literal policy by existing design (frontend accepts, native tool refuses).
- Reviewer REJECT follow-up fixed: #95 reachable-graph walkers now recurse into
  call arguments (nested-argument consumers no longer escape the check); the
  declaration guard uses the POST-SUBMISSION catalog and checks BOTH directions
  (function-vs-variant and variant-vs-function), so single-mutate and reverse-order
  collisions are refused. Corrected a factual misreport: typed RECORD shared
  defaults are natively admitted end-to-end; only VARIANT constructor defaults are
  refused by the native literal gate while the frontend accepts them — a known
  frontend/native policy inconsistency, deliberately not widened. Qualified-call
  over-refusal is recorded as a known safe-direction limitation.
- Pending #90 frontend follow-through admits statically Bool call operands and
  comparison expressions in `and`/`or`; the prior totality restriction was removed.
  Python-source tests exercise recursion/division skipping and Bool calls in NOT
  and conditions in tree/VM. The request-count fixture is **legacy `.ax` only**:
  implicit async calls in Boolean operands are not supported Python evidence.
  No async guard was relaxed. The previous 113→116 count observation has no
  verified per-entry attribution to `05d85cc`; do not attribute all three to calls.
- Pending narrow JIT NOT correction: Bool/Int zero-test; Float always truthy
  (including ±0 and NaN), with operand evaluation preserved. Strings use length,
  never pointer-nullness. Proven forms are literals and untouched, unambiguous
  String ABI parameters; local/rebound strings and string-returning calls refuse
  compilation before execution because their length tracking is not certified.
  Direct compiled **Int-return wrappers** cover empty/nonempty/Unicode String,
  Float zero/negative/nonzero/NaN and Bool/Int controls. A separately executed,
  core-disabled child proves a Float-returning call's Check trap is not skipped.
  NOT helpers live in `compiler/truthiness.rs`; the existing AND/OR IR was moved
  there mechanically to keep compiler.rs under 2000 lines.
  Review found a real scratch collision: String parameter `_last_str` had its
  `_last_str_len` lookup overwritten by a subsequent literal. The correction
  captures the immutable entry-block ABI length Value before parameter binding,
  independently of the mutable name map; source rebinding still invalidates the
  proof for the whole function. Both collision directions have exact compiled
  Int-wrapper assertions. The older pure AND/OR test also now uses Int wrappers.
  A second review reproduced the scalar equivalent (`_last_str_len:Int` shadowed
  by literal scratch). The cohesive correction captures immutable scalar entry
  Values too and emits admitted operands recursively without generic mutable-name
  lookup: literals, untouched parameters, nested NOT and numeric negation.
  Binary operands and argument-bearing calls now refuse JIT compilation, including
  when nested; tree/VM can execute them through the existing pre-execution fallback.
  Zero-argument scalar user calls resolve the actual signature and emit once;
  their internals are not certified generally namespace-safe by this fix. No
  generic compiler namespace refactor or new name blacklist was introduced.
  Both scalar collision directions plus nested NOT/negation and diagnostic-bearing
  safe refusal controls pass; original String/Float/error-evaluation tests remain.
  Freeze checks after correction: logical suite **17/17**; full release **1750 passed, 1 strict
  parity failure, 2 ignored**; CLI **5/5**. Current corpus: 54 files, 573 entries,
  **116 verified / 58 invalid / 399 unsupported**, zero input errors, AST
  mismatches or non-idempotent entries. Aggregate counts do not establish the
  identity/cause of the three-entry increase. Test compilation still emits
  existing warnings; the earlier zero-warning statement concerned non-test build.
- **OPEN JIT relocation/concurrency evidence:** the first full parallel run of
  the new scratch-collision nested-call regression panicked at
  `cranelift-jit-0.119.1/src/compiled_blob.rs:56:80`, `Result::unwrap()` on
  `TryFromIntError(())`, while its targeted run passed. That full run had 1747
  passes, this failure and strict parity failure. The fixture now runs both exact
  compiled results in the current test executable's isolated child, with a
  60-second kill deadline; panic/nonzero/timeout fails the parent, and a completion
  marker proves both directions ran. Isolation avoids this observed concurrent
  allocation/relocation failure in the final run; it **does not fix or close**
  the underlying JIT relocation issue. No production relocation code changed.
- **OPEN #90 residual — next bounded priority after NOT acceptance:** JIT
  AND/OR still branch on raw values, use XOR-one for the AND left condition, and
  store an unnormalized RHS in an I8 result variable
  (`compiler/truthiness.rs::compile_logical`). `compiler.rs::compile_statement`
  Check/Branch/While likewise feed `compile_expr(..., I8)` straight to `brif`;
  expected type does not convert a literal/identifier's actual representation.
  Proposed minimal direct-JIT Int-return-wrapper repros: `if 2 AND true`,
  `if true AND 2`, `if ""`, `if 0.0`, `check required 0.0`, and a bounded
  `while ""` that immediately returns 1 if entered. Compare against tree/VM:
  Int 2 is truthy, empty String false, **Float 0.0 true**. Run potentially invalid
  IR/trap cases in isolated children. Shared normalization is not implemented in
  this freeze, and all-tier truthiness/parity is **not closed**. String-return ABI
  propagation, arbitrary local bindings and parser-expression evaluation also
  require separate evidence; do not infer those from the narrow NOT tests.
- #90 tree/VM semantics aligned for the covered operations. Truthiness is now a
  single public `Value::is_truthy`; the VM's `pop_bool` delegates to it so
  `AND`/`OR`/`NOT`/`BranchIf`/`BranchIfNot` share one definition (String/None/Err
  rows are `.ax`-only since the Python frontend emits Bool). VM `AND`/`OR` compile
  to short-circuit branches via a new `ToBool` opcode plus the existing
  Jump/BranchIf/BranchIfNot patching; the JIT emits branch + result variable for
  both ops (compiler.rs `compile_expr`, the And/Or arm; it previously used
  `band`/`bor` with both operands eagerly compiled). Differential tests assert
  tree/VM agreement on skipped errors, recursion and mocked IO counts; direct JIT
  coverage is narrower. The old claim that these establish general three-tier
  parity was incorrect. Earlier eager-RHS characterization tests were inverted,
  but the OPEN residual above still prevents closure.
- Builtin signature completeness (#86): new `python_frontend/builtin_signatures.rs`
  gives every pure BUILTINS name and alias a checked arity/result signature
  (grounded in the runtime's own `expects N argument` guards and return values).
  Audit tests assert catalog-to-registry agreement, full alias reachability, and
  no uncatalogued reachable pure builtin. Gaps found and fixed: `home_dir` was
  missing entirely; `map`/`set`/`concat`/`Ok`/`Err`/`Some` are special-cased
  outside the catalog by design. `str` is an Adapsis alias for `to_string` but
  stays rejected in Python source as a coercion (documented, not silently
  rebound). IO builtins are outside this catalog: they are await-validated by
  `call_sites.rs`. Reviewer follow-up (a) was initially reported as done but was
  comment-only: `check_runtime_variant`'s fallthrough call arm accepted ANY
  non-list call, so `check_runtime` was permissive despite the strict contract in
  its comment. Now that arm matches only the literal `list` builtin, so an empty
  or non-matching `variants` set makes every non-list call fall through to
  `literal()`. Unreachable in production (parse-time `authored::literal`
  pre-gates those positions) but the ordered hardening is real, with unit tests.
- Next authorized pair implemented. (1) Native literal gate widened ONLY for
  shared-variable defaults whose declared annotation is a union and whose tree
  shape is data-only variant construction (literal payloads, `list(...)`; a
  zero-payload variant uses the bare constructor spelling). Spawn bindings have no
  defaults and are not part of this widening. The slice3 walker still owns
  arity/type checking, nested calls stay refused, and the #95 function/variant
  declaration guards are untouched, so a variant default cannot smuggle a
  same-named function. Records were already natively admitted. The validator now
  union-reclassifies `+shared` annotations like function signatures, which also
  corrects `.ax` written through the same path (previously a union-typed shared
  var persisted as `Struct(name)`). Inherited caveat: the shared arm resolves the
  annotation by bare name, so a pre-existing bare union-name ambiguity is treated
  the same way as elsewhere in the frontend rather than being resolved here. (2) #92:
  `i64::MIN / -1` wraps to `i64::MIN` and `i64::MIN % -1` is `0` in tree, VM and
  JIT via shared `checked_div`/`checked_rem` (JIT routes through runtime helpers);
  zero divisors keep their previous error/trap behaviour. Pending combined gate
  and re-freeze.
- Earlier design note (superseded for slices 1-2 by the authorizations above): (1) exhaustive
  direct-call admission visitor and builtin-only IO-match exception; (2) scoped
  variant/function/value-reference classification, preserving authority-first
  lookup and bare-user-call refusal; (3) structural shared-declaration preflight
  plus typed recursively data-only defaults. Each slice needs actual native
  refusal/no-mutation regressions and independent review before broadening.
  Builtin signatures, runtime #90/#92 and the full TelegramBot port remain
  separately sequenced; current source/tool descriptions must stay truthful.
- Slice 1 was subsequently authorized and implemented (explicit root call modes,
  exhaustive executable traversal, native/direct-validator regressions). It is
  **not accepted yet**: full release run found 1690 passing / 2 failing / 2 ignored
  (strict parity plus the new native synchronous-Result-match positive test).
  Bounded differential diagnosis proves identical `.ax`/Python ASTs: tree returns
  5 for Ok(5), 0 for Err; VM and normal native routing return None because VM
  MatchVariant only handles Value::Union, not dedicated Ok/Err values. No await/IO
  is involved; this is distinct from #82. The positive assertion stays red;
  runtime correction awaits separate authorization. Slices 2–3 remain unstarted.
- #93 was separately authorized: a cohesive VM match helper now recognizes
  dedicated Ok/Err values and supplies the existing payload/bool stack protocol.
  Targeted Result-match gates pass, including unchanged native expected 5,
  original `.ax`/Python AST equality, nested match statements, Union controls
  and no IO replay. Final combined #93 + slice1 release gate: **1697 passed,
  strict parity alone failed, 2 ignored; CLI 5/5 passed**. Both deltas are frozen
  for independent review; neither acceptance nor deployment is implied. Option,
  nested payload patterns, non-exhaustive matches and JIT remain separate.
- Sole main writer: `ses_e8b17a23affedgH4V52JdwxNDL` (Astra). Manager/release:
  `ses_e8d6d9d1bffe8G7KEI4yUw8DPi`; P0 reviewer:
  `ses_e888436abffenlBGkG1Ek8bZHF`. Independent #87/#88/#89 and TUI worktrees
  remain separate. Shared main target builds are manager-coordinated.
- Approved current slice: exact typed-NodeId-only production comparison,
  rejection of unsafe call qualification/Unknown admission, native regressions,
  truthful prompt and module-lowering extraction. Owner-aware catalogs and other
  parity/authority gaps need separate bounded plans. #90 runtime Boolean behavior
  and #91 original TelegramBot source correction are independent gates.
- Required acceptance: a runnable checked-in full sanitized TelegramBot Python
  port with paired tests, exact AST/idempotence and mocked raw-update behavior
  across admin/guest/group routing, text/media/replies and lifecycle/shared/spawn.
  Diagnose the invalid original visibly; use a separately reviewed #91 correction,
  never silently repair the fixture. No live Telegram/network or permission changes.
- Complementary owner requirement: native conversation capability parity (#89),
  including a permission-checked `!opencode` equivalent, has its own implementation
  owner. The #86 TelegramBot port must test protocol/takeover interactions but
  does not absorb agent/plan/roadmap/model/mock/stub/remove tool implementation.
  Native-default deployment remains deferred until both acceptance gates pass.
- No commit, push, integration deployment or service restart authorized.


## Historical release candidate — #87/#88 + TUI, native still opt-in (#89 parity gate)

2026-10-26 superseding owner decision: **conversation capability parity and
independent review are required before native becomes the default.** Retain
legacy and the guarded external `!opencode` integration (process/model gates,
external runner boundary). Native schemas do not replace OpenCode's development
environment. No deployment or production self-modification was authorized here.

- Parent committed the accepted #87/#88 manifest as `5445a76`, then cherry-picked
  the independently reviewed three-file TUI change as `1d700fe` on this worktree.
  Both commits are local; the candidate is **NOT DEPLOYED**. Main #86 remains a
  separate RED workstream and is not included.
- A new, **uncommitted correction** above `1d700fe` reverses only the default
  flip: no flags means legacy; `--native-tools` actively opts eligible turns in;
  `--legacy-tools` explicitly forces legacy and takes precedence in either flag
  order. The production resolver is exercised by the CLI test. Ineligible turns
  remain whole-turn legacy; native has no per-feature fallback. Native memory,
  privacy logging, TUI, and existing guarded legacy `!opencode` are unchanged.
- Correction manifest: `src/main.rs` (flag help, production boolean resolver,
  CLI test), `src/api/mod.rs` (field documentation), `AGENTS.md`, `ROADMAP.md`.
  No source drift outside that list; no permission/context edits, installs,
  restarts, service operations or writer commits. Parent will create a new
  correction commit, never amend the accepted commits.
- Prior source acceptance remains recorded below: Astra
  `ses_e888436abffenlBGkG1Ek8bZHF` accepted #87/#88; parent reports the TUI delta
  independently accepted twice. This opt-in correction is queued for a bounded
  read-only re-review; source acceptance does not itself authorize deployment.

### Corrected combined verification and artifact

All commands used `CARGO_TARGET_DIR=/tmp/opencode/target-wt87`, never the main
checkout target. `cargo build --release` passed; `cargo test --release cli_tests`
**1 passed**; full `cargo test --release` **1628 unit passed, 2 ignored**, plus
**5 integration passed**; `cargo test --release tui` **32 passed**;
`cargo test --release native_` **102 passed**; `cargo test --release
takeover_native` **13 passed**. `memory-eval`: six cases, recall 1.0, all violation
counts zero, `passed=true`. All `.ax` examples passed; Python basic **2/2**, flow
**7/7**. `git diff --check` clean.

- Corrected build/CLI/full/focused/memory-eval transcript:
  `/home/marenz/.local/share/opencode/tool-output/tool_17806241b001R93vnr32NozDUr`.
- Earlier combined verification at committed `1d700fe` (before the superseding
  opt-in correction) also passed; transcript:
  `/home/marenz/.local/share/opencode/tool-output/tool_177fe35b0001P2Gc0LnPx4xGtN`.
- Final corrected binary: `/tmp/opencode/target-wt87/release/adapsis`.
  SHA-256: `d025fddeb021ea430c46e97313926b59fc20da008c56ecce02f77bd141fe82f7`.
  This is built from `1d700fe` plus the uncommitted opt-in correction. No Git/date
  stamp exists; commit alone does not alter the tested artifact. Rehash if the
  parent rebuilds. ROADMAP is not compiled into it.

### Actual creative deployment-input audit — still requires operator resolution

Read-only audit compared deployment TOML and directly corresponding context
files only. No personal names/IDs/contents were emitted, and no chats or memory
were accessed. The sanitized public placeholder is not treated as a real user.

- One actual principal maps to the creative profile. The live tool allowlist and
  reviewed model-profile fields match the public reviewed fragment exactly.
- That principal's deployed context exists. A main-checkout **ignored local
  reference also exists and is byte-identical**; the worktree has no copy. The
  real local reference is not Git-tracked, so equality alone does **not** prove
  its review provenance or approve the actual principal mapping.
- Content/path hashes and the alias-only comparison were reported privately to
  the manager. Reproducible read-only audit script:
  `/tmp/opencode/wt87-creative-preflight.py`. It creates no approval baseline.
- **Smallest resolution:** operator/reviewer confirms that the existing actual
  principal-to-creative mapping and that exact ignored local context (identified
  by the reported hashes) are the intended reviewed deployment inputs. Then
  revalidate equality and the service's actual configured permissions path
  before parent-authorized deployment. No permission rewrite or fabricated
  context file is needed merely to make the sanitized example checker pass.
  Until that review confirmation, the creative deployment gate remains
  **INCOMPLETE / release-blocking**, not a claim that the live config is wrong.

## Historical pre-commit handoff — #87/#88/#89 (default-flip scope superseded)

The checkpoint below records the earlier accepted implementation. Parent has
since committed it as `5445a76`; its default-flip decision and artifact/status
are superseded by the current opt-in correction and combined evidence above.

Recovery checkpoint 2026-10-26. Sole writer worktree:
`/home/marenz/Projects/adapsis-wt87`, branch `work/issue-87-88`, based on
`668559986187e388d522e904637657f5d48f6e32` (#85). **Final independent Astra review
ACCEPTED; offline gates green. Changes are NOT COMMITTED or DEPLOYED.** No
installation, restart or service modification was performed by this writer.
Release authority remains manager `ses_e8d6d9d1bffe8G7KEI4yUw8DPi`.

- **#87:** native `memory_inspect`, `memory_remember`, `memory_correct` and
  `memory_forget` dispatch the existing turn-bound coroutine builtins. No caller
  context/principal/authority fields; only advertised with `TurnIdentity` and
  available to eligible write-capable non-guest turns. HTTP without identity
  omits memory schemas and claims. Inventory guidance requires authorized
  inspection first, distinguishes bounded surfaced records from exhaustive
  storage, and forbids another-private-chat trawl. Existing ACLs stay intact:
  non-guest forget can authorize cross-context contributions through membership
  in all governing groups; it is not artificially restricted to this context.
- **#88:** default incidental stdout/stderr/activity-log diagnostics carry
  categories and byte counts. Native provider end states cannot print error
  payloads; provider/serde/regex errors and command arguments are redacted at
  logging sinks. Event tags never authorize their arbitrary detail strings.
  `ADAPSIS_LOG_CONTENT=1` restores debug content. Rich caller/UI results remain;
  provider-error apologies omit raw diagnostics. Tests capture actual production
  paths in child processes with immutable policies, not shared flag overrides.
  Explicit language print/CLI output and operational identifiers are outside
  incidental-content redaction. `--training-log` is deliberately **default-on**
  at `training.jsonl` and content-bearing; its behavior is unchanged.
- **#89 (superseded, bounded scope):** native protocol is the default;
  `--native-tools` remains accepted as a no-op, `--legacy-tools` opts out, and
  legacy implementation is retained. Guest/execute-only turns remain whole-turn
  legacy. A write-capable native turn has **no per-feature fallback** for
  agents/plans/roadmap/opencode/mocks; unavailable features remain unavailable.
- Runtime `Memory`/`MemFix` cleanup was completed and verified absent from the
  saved session by the parent; it is not an outstanding local task.
- **Main checkout #86 is a separate RED workstream**, per manager/reviewer
  status. It is not in this release or its acceptance. Parent intends to commit
  and deploy this independently reviewed branch before integrating #86.

### Acceptance and exact verification

- Final read-only Astra reviewer: `ses_e888436abffenlBGkG1Ek8bZHF`.
  Final verdict **ACCEPTED**, no remaining release-blocking findings within the
  reviewed #87/#88/#89 boundaries. Reviewer checked the new production-premise
  canaries and accepted author build/test evidence; live/deploy gates remain
  parent-owned. Writer session: `ses_e8aac711bffeXf5bq9NAdQrc8U`.
- All builds/tests used `CARGO_TARGET_DIR=/tmp/opencode/target-wt87`, release-only.
  Final gates: `cargo build --release`; `cargo test --release` **1609 passed,
  2 ignored, 0 failed**, plus **5 integration passed**; `cargo test --release
  native_` **102 passed**; `cargo test --release takeover_native` **13 passed**.
  `cargo run --release -- memory-eval`: six cases, recall 1.0, all violation
  counts zero, `passed=true`. All `examples/*.ax` passed (no FAILED output).
  Python `basic.py`/`test_basic.py`: **2 passed**; `flow.py`/`test_flow.py`:
  **7 passed**. `git diff --check` clean.
- Complete final build/test/subset/memory-eval transcript:
  `/home/marenz/.local/share/opencode/tool-output/tool_177db57c5001YVyg6259eWOxHS`.
  Earlier full-suite transcript:
  `/home/marenz/.local/share/opencode/tool-output/tool_177d7a0b8001bH8hXHPPntx7oQ`.
  Example/Python command output is retained in the writer session; the exact
  results above are its recovery summary, not a newly rerun test transcript.
- Tested binary: `/tmp/opencode/target-wt87/release/adapsis`.
  SHA-256: `9154f11f7360bf6a0de3c88c389e4d14a7cf4ae8ed1d9424d158e9fe7a4d2ec1`.
  Package version remains `0.1.0`. No Git SHA/date/dirty stamp mechanism was
  found in crate source/build configuration. `build.rs` only emits OpenSSL
  link directives; commit alone does not stamp or change this existing binary.
  Cargo may rerun that build script after package-file changes (it has no
  explicit rerun-if inputs), but neither ROADMAP nor AGENTS text is compiled
  into this binary. No build/test rerun was needed for this docs-only handoff.
  If the parent rebuilds, re-hash that resulting artifact rather than assuming
  reproducible bytes across a changed environment.

### Read-only creative deployment preflight — INCOMPLETE (sanitized inputs)

- Ran `python3 tools/check_guest_creative_config.py config/guest-creative.toml
  /home/marenz/.config/adapsis/permissions.toml`. Exit **1**:
  `live config is missing principals.telegram:user:100000101`.
  **This is the sanitized repository's GuestUser placeholder, not an identified
  real Telegram user.** The literal example-to-live mismatch does not establish
  that the real deployed permissions are wrong. A reviewed mapping to intended
  live deployment inputs is needed before this check can certify them.
- Required comparison `cmp contexts/telegram_user_100000101.md
  /home/marenz/.config/adapsis/contexts/telegram_user_100000101.md` could not run:
  the **reviewed worktree context file is absent** (ignored/local-only input).
  No replacement was manufactured and no live input was overwritten.
- These checks do not certify the real deployment inputs. Parent must resolve
  the intended reviewed inputs before a creative-profile deployment. This inspection
  used deployment configuration only, not any other user's chats or memory.

### Intended commit manifest (parent only; nothing committed here)

The following list includes all source/docs changes and all new test files;
exclude unrelated main-checkout #86 work:

```text
AGENTS.md
ROADMAP.md
src/api/execute.rs
src/api/execution_logging_tests.rs        (new, untracked)
src/api/llm_handlers.rs
src/api/logging_gate_tests.rs             (new, untracked)
src/api/mod.rs
src/api/native_tools.rs
src/api/native_tools/tests.rs
src/api/native_turn.rs
src/api/takeover_logging_tests.rs         (new, untracked)
src/api/takeover_native.rs
src/api/takeover_native_tests.rs
src/api/tests.rs
src/capabilities/cad.rs
src/capabilities/music.rs
src/capabilities/video.rs
src/context_prompt.rs
src/coroutine/mod.rs
src/library.rs
src/llm.rs
src/llm/logging_tests.rs                  (new, untracked)
src/logging.rs                           (new)
src/main.rs
src/memory_graph/inspect.rs
src/parser.rs
src/prompt/native.rs
```

## Historical checkpoint — native tools for takeover conversations (#85)

The following #85 checkpoint is historical; #85 is now the worktree baseline
`6685599`. Current release status and protocol default are recorded above.

Owner request 2026-10-25: "switch everything" to native/Python. Issue:
https://github.com/Marenz/adapsis/issues/85. Phase: **implemented offline, all
gates green; awaiting independent review, then manager commit/deploy + Telegram
smoke.** Not committed, installed, deployed or restarted.

- With `--native-tools`, write-capable non-guest `llm_takeover` turns (Telegram
  admin DMs, groups while the admin speaks) use the shared native driver
  `NativeLoop::drive` (`src/api/native_turn.rs`); `src/api/takeover_native.rs`
  only decides eligibility and renders the reply. Guests, execute-only/read-only
  speakers and processes without the flag keep the legacy protocol unchanged.
- Explicitly unavailable natively (stated in the prompt): plan, roadmap,
  `!agent`, `!opencode`, model switching, mocks/stubs, removal. The speaker block
  reports agents/opencode as unavailable for native turns.
- Fixed live bug: native requests sent `temperature: 0.7`, which
  `anthropic/claude-opus-5` rejects with 400; now omitted (regression in the
  HTTP/stream lifecycle and takeover fixtures).
- Gates: `cargo test --release` 1589 passed + 5 CLI, 2 ignored (6 new
  `takeover_native` tests); release build; `memory-eval` passed; 28 `.ax`
  examples; Python CLI examples 2/2 and 7/7.
- Remaining: independent review, deploy here, live Telegram smoke on the dev-bot
  chain (opus-5 primary). Guest native schemas are a separate later slice.

## Completed — working Python frontend (#81)

**User-approved stop gate (2026-10-25):** "your stop gate is the working python impl".
Native HTTP tools #78 are committed as `8e10876` on `work/native-tools`; the
checkout was clean before this design-stage recovery update. Build a working
Python-subset frontend through the guarded inspect → mutate → test → evaluate →
finish workflow, with the existing AST/runtime authoritative and `.ax` retained.
A standalone syntax demo or CPython `eval`/`exec` is not the stop gate.

- Issue: https://github.com/Marenz/adapsis/issues/81.
- Current phase: **#81 stage complete; stop gate met.** Code is committed locally
  as `61b3911` on `work/native-tools` (on top of native tools `8e10876`); the
  checkout was clean after that commit. All final gates passed and both
  independent reviews accepted. The release binary was built and tested locally
  only: **not installed, deployed, pushed or restarted as a service**, and no
  live-provider trials were run. Gateway fixes are separately committed as
  `0f7d7ea` in `~/Projects/llm-gateway` and likewise not installed.
- Historical progression: initial offline gates passed (1569 Rust unit tests plus
  4 CLI subprocess tests, 2 ignored; Python-focused 38+4, native 70, parser helper
  5) before review corrections; final corrected counts are recorded below.
- Sole Adapsis integration/docs writer: `ses_e8d51b0b2ffe2YU2n1nfJe0WbS`.
  Manager/release authority: `ses_e8d6d9d1bffe8G7KEI4yUw8DPi`.
- Isolated frontend writer: `ses_e8d0359d6ffe9vFSOf5LDkQyNI`, worktree
  `/home/marenz/Projects/adapsis-python-frontend`, owning only new
  `src/python_frontend/**` (lowering, canonical serialization and Python printer).
  Its seven finalized files were imported; no duplicate target/cache was created.
- Python reviewers (read-only): Astra critical integration/authority/CLI/
  persistence `ses_e8d27edc9ffeFZJshutWc7PbXM`; DeepSeek Pro language semantics,
  roundtrip and parser `ses_e8d279e0bffe6hI60QWbBO7jmg`. All corrections remain
  under main-checkout sole-writer ownership. Both reviewers accepted all final
  corrections, including the last printer/IO-registry/Result-await changes,
  conditional on the final complete offline gates.
- Parallel gateway compatibility writer: `ses_e8d51200cffe40osdeEbEWe0JC`, in
  `~/Projects/llm-gateway`, branch `work/native-tool-compat`, baseline `98d7240`.
  Gateway #2 and Python #81 are distinct workstreams; no shared source ownership.
  Gateway fixes are now committed locally as `0f7d7ea` on that branch; 70 offline
  tests, check and release build passed, with independent DeepSeek acceptance.
  Its checkout is clean. It has not been installed/restarted or live-certified;
  gateway #2 remains open and no gateway build is running.
- Preserve #78's pinned authority, strict caller-authored-reference validation,
  literal test/input policy, immutable turn completion baseline, CoW write-back,
  complete tool history and explicit completion reporting. Unsupported Python
  semantics must fail with source locations, never silently approximate behavior.
- No deployment, service changes or live model-effectiveness claims. Existing
  main target was reused; builds were coordinated with the gateway writer, with
  no new target trees. Last disk state: `/home` 31 GiB free, `/tmp` 15 GiB free.

### Python stage status
- [x] Parser/lowering and subset contract approved
- [x] Initial frontend implementation and main native integration
- [x] Initial typed/control-flow/collection examples and guarded end-to-end tests
- [x] Trusted mocked read_file acceptance (test-only IO, no live worker)
- [x] Initial AST/source/session/library roundtrips and unsupported-AST diagnostics
- [x] Review corrections: Python argument binding, async call fidelity and helper shadowing
- [x] Single-function inspect→resubmit with permission-filtered runtime signatures
- [x] Independent reviews and complete offline release gates
- [x] Code committed locally (`61b3911`); local binary tested; not deployed

### Final Python acceptance evidence
- `cargo test --release python_`: **52 unit + 5 CLI subprocess tests passed**.
- `cargo test --release native_`: **84 passed**.
- `cargo test --release`: **1583 unit tests passed, 2 ignored**, plus **5 CLI
  integration tests passed**. These gates were rerun after the final mechanical
  admission-helper extraction; parse → authorize → finalize → reauthorize →
  runtime-check ordering and error messages stayed unchanged.
- `cargo build --release` passed. Package version remains 0.1.0. The current
  executable's `test --help` documents language/module/paired-test flags.
- `cargo run --release -- memory-eval` passed: 6 cases, recall 1.0, zero duplicate,
  precedence, contradiction, ACL, citation, stale and omission violations.
- All **28 existing `.ax` example files passed**; `node tools/test_native_events.mjs`
  passed. Actual provider-free Python commands passed:
  ```sh
  ./target/release/adapsis test examples/python/basic.py --language python --module PyBasic --tests examples/python/test_basic.py
  # TEST SUMMARY: 2 passed, 0 failed
  ./target/release/adapsis test examples/python/flow.py --language python --module PyFlow --tests examples/python/test_flow.py
  # TEST SUMMARY: 7 passed, 0 failed
  ```
- Both independent source reviews accepted all corrections; Astra separately
  confirmed the final admission-helper extraction. No remaining review blocker.
  Runtime follow-ups #82 (user Result-await semantics) and #83 (narrower helper
  collision usability) remain explicitly deferred, with safe frontend refusal.
- `Cargo.lock` adds 28 parser-related packages without upgrading existing locked
  dependencies. No CPython, origin metadata, deployment, service restart, live
  provider trial, or claim of improved model effectiveness is part of this gate.
- Three release warnings remain: the existing structural-only native validator
  API and reserved frontend source-map fields are unused by production callers.
  Diagnostics themselves use actual source spans. Existing main target only;
  final free space remains about 31 GiB on `/home`, 15 GiB on `/tmp`.

### Review corrections (accepted; retained as design record)
- Keep Python positional-vs-keyword call semantics: exact parameter count/names
  are checked at dispatch and in the isolated CLI test path. `_0`, `_1`, …
  parameter/keyword names are reserved to avoid ambiguity with legacy positional
  carriers. Newly submitted direct calls use same-source plus current-program
  signatures; never re-analyze pre-existing trusted bodies transitively (#79).
- Direct user-call async effects must match await usage; synchronous functions
  cannot be awaited and async functions cannot be silently called without await.
  Runtime-provided asynchronous IO builtins remain supported.
- Recheck helper collisions at Python test/evaluate, not only mutation: later
  `.ax` edits must not intercept a previously installed Python list/concat/len
  helper. Without origin metadata, Python execution can conservatively reject
  runtime builtin-name collisions in the current program, including after reload.
  This includes `BUILTINS` and `IO_BUILTINS` and all aliases. Python inspection
  runs the same neutral helper safety check separately from its filtered
  signature catalog. Narrower helper-specific usability is tracked in #83.
- Approved two-phase mutation admission: complete parser-only supported-syntax,
  declaration/effect/annotation and direct-reference preflight with no existence
  checks; authorize module Write and direct-reference Execute using the pinned
  turn permissions; then per-call final lowering with only authorized runtime
  signatures plus same-source declarations. The resulting canonical operations
  still pass all existing native checks and executor gates. This enables an
  inspected single function with nested calls/arithmetic/loop conditions to be
  resubmitted without requiring its entire module. No hidden signature may
  influence diagnostics before authorization. Batches remain sequential, not
  atomic; later calls see earlier successfully applied mutations.
- Result-returning **user-function awaits** are rejected in Python lowering,
  direct runtime-call checks and Python inspection until the inherited runtime
  unwrapping policy is reconciled (#82). No interpreter semantics were changed.
  Direct IO builtin Result awaits remain supported: an in-memory IoRequest
  responder verifies the actual Ok wrapper through native evaluation, without
  any HTTP/network request. Trusted mocked read_file remains the runnable IO test.
- The approved quote-aware parser correction is isolated in
  `src/parser/test_values.rs`: quoted/escaped parentheses/braces and test `->`
  separators remain literal data. i64::MIN remains explicitly unsupported.
- Last disk/build state: existing main target only; 31 GiB free on `/home`,
  15 GiB free on `/tmp`. No deployment, service changes or live-provider trials.

### Approved Python v1 contract
- Exact `rustpython-parser = "=0.4.0"`, default location/bigint features. The
  crate was published 2024-08-06; the repository remains maintained. Its MSRV
  1.72.1 works with current Rust 1.93.1. Ruff's current published parser needs a
  newer toolchain and is not selected. Parse using `ast::Suite::parse`; convert
  UTF-8 byte spans to one-based source line/column diagnostics using the source
  text (including character-aware columns).
- Lower into typed `parser::Operation` declarations/tests, then a trusted
  canonical `.ax` emitter and structural roundtrip validation before the existing
  guarded native executor. This is node serialization, never text substitution
  or Python execution. Keep the same authorization and immutable completion gate.
- Native mutate/test add `language: "adapsis" | "python"` (default Adapsis);
  Python requires explicit `module`, Adapsis rejects the new module field.
  Evaluate adds the same optional language for literal input conversion, keeping
  the qualified `function` name. Inspect source adds `format: "adapsis" | "python"`
  (default Adapsis); other inspection kinds reject format. Finish is unchanged.
- Python modules contain an optional docstring plus annotated `def`/`async def`.
  Parameters, return values and new locals require annotations; reassignments
  lower to Set. Types: int, float, bool, str, list[T], Result[T]. Regular functions
  cannot return None (the runtime has no Unit type). Integer literals range from
  -9223372036854775807 through 9223372036854775807; i64::MIN is rejected because
  canonical `.ax` cannot parse its positive magnitude before unary minus.
  Numeric storage/overflow remains explicitly the existing typed
  runtime model, not a claim of arbitrary-precision CPython semantics.
- Support if/elif/else, while without else, and for over statically typed lists.
  A for target is fresh and confined to its body: reject prior bindings, reuse
  and reads outside the loop instead of silently changing Python's target
  leakage. New block locals cannot escape their block; existing variables can
  be reassigned. No break/continue or loop else.
- Collections are read-only values: list literals, iteration and len(list). Mutable
  aliases, append, item/slice assignment, += and subscripts are deferred. Numeric
  +, -, * and single comparisons are supported; string + maps to concat only
  when both operands are statically strings. Reject /, //, %, **, chained
  comparisons and operand-returning truthiness. Boolean and/or require Bool and
  total call-free operands unless exact short-circuit lowering is provided.
- Bare local calls lower to explicit metadata-module qualification; cross-module
  calls use Module.fn directly. Builtin-name definitions, callable variables,
  dynamic attributes, dunders, eval/exec/open/getattr and imports are rejected.
  No classes, lambdas, comprehensions, f-strings, ternaries, exceptions or nested
  defs. `@effects("io", "fail", "mut")` is a closed inert declaration; async def
  supplies async. Await is supported only in an annotated direct-call binding.
- Python tests are zero-argument `def test_*` wrappers containing one equality
  assertion of a direct function call against a literal, optionally with a
  docstring. They lower to TestMutation, never runtime function definitions or
  arbitrary test execution. Inputs/expectations accept scalar/read-only list
  literals; native Result matchers/global after assertions remain deferred.
- Python evaluate parses only literal arguments (including True/False and list
  syntax), not general expressions. The literal policy may admit canonical
  `list(literals...)` only with collision checks proving no user function/union
  variant can intercept construction; arbitrary nested calls remain forbidden.
  Python mutation lowering also checks synthetic helper collisions (including
  concat for string addition and builtin calls), even against otherwise
  authorized user functions; Python syntax must retain the promised operation.
- Mocked IO acceptance uses trusted harness-seeded session IO mocks and the
  existing mock-only test runtime, with a String-returning read_file example.
  Python cannot install mocks and no new mock schema/global state is introduced.
- AST is the only persisted truth: no frontend-origin metadata, sidecars or `.ax`
  header markers. Explicit Python inspection prints supported AST after reload;
  unsupported legacy forms return precise target/construct errors. Canonical
  Python print→lower→print is idempotent with normalized semantic AST equivalence;
  formatting/comments are not retained. Submitted functions replace existing
  functions as `.ax` does; unsubmitted functions keep CoW identity. No inferred
  deletes or whole-file diff system.
- Release cases: typed scalar, local calls, if, while, list/for, list arguments,
  explicit cross-module calls, bools, reserved-name/authority rejection,
  canonical source roundtrip and session/library reload. HTTP and SSE fixtures
  must demonstrate Python mutate → premature finish refusal → Python test →
  Python-literal evaluate → finish. Include location negatives, caller-authored
  nested references, list-constructor collisions and trusted mocked IO. Run
  Python/native/full Rust, example, memory and UI gates plus independent reviews.
- Direct executable demonstration uses the existing isolated CLI test runtime:
  `adapsis test examples/python/basic.py --language python --module PyBasic
  --tests examples/python/test_basic.py`. Production lowerers supply canonical
  operations to the existing test path; Python module/tests flags are required
  together, with no zero-test success. Legacy `.ax` defaults remain unchanged.
  No CPython, provider, live library state or new mock-authoring API is involved.

## Archived foundation and pre-native history

The completed #78 foundation and earlier roadmap sections are preserved verbatim
in [the historical archive](docs/roadmap-native-foundation-history.md).
