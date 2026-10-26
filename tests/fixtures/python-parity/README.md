# Python parity corpus

This is an offline inventory, not a passing-target allowlist. `manifest.json`
names every `.ax` fixture in `examples/` and `library/`. Missing, extra,
unreadable, empty, or non-regular files are hard diagnostics. Tests read only
these checked-in fixtures, never the operator's live module library.

Run:

```sh
cargo test --release parity::tests
cargo test --release python_frontend::parity::python_parity -- --exact --nocapture
```

The full gate is currently **red**. Its JSON report and `PARITY_*` summaries
separate verified roundtrips, input errors, invalid-program diagnostics,
unsupported representation, exact AST differences, and non-idempotent prints.
Counts are inventory **entries**, including aggregate module/program checks and
descendants of invalid fixtures; they are not counts of independent failures or
fully covered files. `PARITY_INVALID_ROOTS` identifies root diagnostics.

Every original parser operation is inventoried. Test/eval/query/trace commands
are retained as explicit unsupported surfaces even when the validator ignores
them. Top-level functions/types are also explicit blockers: the production
Python printer/lowerer currently requires a module context. A module failure
is retained even when each member can print. Repeated declarations cannot be
certified from the final coalesced program state.

For supported module functions/types/modules, the gate reparses Python using
the production lowerer, reconstructs the canonical AST, compares exact
structure while clearing only actual `NodeId` fields, then prints and lowers
again. Callee spelling, effects/order, user fields named `id`, documentation,
shared defaults, lifecycle bodies, routes and stored test metadata are not
normalized away. AST mismatch reports name structural paths, not literal data.

## Invalid originals and redaction

The original sanitized files are preserved, including invalid examples and
library code. `Diag.run` reads undeclared module-local shared variables.
`TelegramBot.transcribe_voice` has a misindented second inner `Err` arm and a
following reference to `e` outside that arm. Dedicated audit tests retain the
AST/undefined-name diagnostics. Their current original on-disk library source
was checked separately; this does not establish the daemon's loaded AST or
actual delivery behavior. Invalid records never count as verified coverage.

Library fixture literal payloads were redacted at authoring time. A one-time
operator check found identical lexical structure in all 26 current originals
after masking string/numeric payloads and comments. This is not behavioral
equivalence: changed literal values can alter URLs, expected results and lengths.
No live code or fixture function is executed by this harness. Runtime Boolean
portability (#90) is a separate, still-blocking issue; AST fidelity alone does
not certify equivalent tree-walker, VM and JIT behavior.
