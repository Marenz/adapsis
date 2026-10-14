# Memory Quality Fixtures

`v1/` is a deterministic, network-free corpus for `adapsis memory-eval`. It never
opens the configured Ladybug database, reads a home directory, initializes an
embedding provider, or calls a model. The runner creates an isolated in-memory
Ladybug graph and exercises production message ingestion, canonical and extracted
memory creation, ranked recall, context-scoped inspection, ACLs, citations, and
rendering.

Run the repository gate from any directory with:

```bash
cargo run --release --manifest-path /path/to/adapsis/Cargo.toml -- memory-eval
```

Machine-readable JSON goes to stdout and a one-line human summary goes to stderr.
`--report /tmp/memory-quality.json` additionally writes the same JSON to an
explicit path. A threshold violation returns nonzero after emitting the report.

## Version 1 Schema

- `corpus.json` contains invented messages, sparse 384-dimensional test vectors,
  canonical and extracted memory drafts, narrow synthetic attachment references,
  explicit denial edges, a generated noise series, and expected query results.
- `extraction-snapshot.json` is a frozen, sanitized annotation of provider output.
  Its candidate-to-gold labels measure extraction precision/recall independently;
  candidates are not used to seed retrieval.
- `baseline.json` contains exact deterministic quality/cost results, one measured
  reference latency distribution, and explicit pass/fail thresholds. Latency is
  gated only by the generous ceilings, never by equality with the reference run.

Ratios are rounded to six decimal places. Retrieval precision and recall aggregate
logical memory keys across cases. Unauthorized recall counts returned keys listed
as forbidden for a case and has a hard threshold of zero. An extracted result has
an unsupported citation when it has no citation or cites a source outside the
fixture's explicit support set. Stale rate is stale retrieval occurrences divided
by current-or-stale occurrences; neutral load records are excluded.

The exact-id ACL cases deliberately exercise #55's current-context clause twice:
one principal legitimately belongs to both source and destination contexts, and the
fixture administrator belongs to every context. Both must still receive an empty
result when inspecting context A for a context-B memory. Their foreign keys remain
explicitly forbidden so removing the context-or-global governance clause trips the
hard unauthorized-recall gate.

`prompt_chars` is the sum of `canonical_prompt_chars` and
`ranked_recall_prompt_chars`. The former counts the exact production canonical
injection for every retrieval case under the default 40-memory/8000-character
bounds; the latter counts the exact ranked-recall block where that path is used.
The corpus carries an explicit `canonical-prompt-accounting` coverage tag, and a
test requires the German paraphrase case's canonical contribution to be nonzero
and included in both its case total and the aggregate report.
`output_chars` counts production `memory_inspect` rendering. Characters are Unicode
scalar values. The documented token estimate is
`ceil((prompt_chars + output_chars) / 4)`; it is a stable budgeting estimate, not a
provider tokenizer.

## Version 2 Prompt-Selection Gate

`v1/` is preserved byte-for-byte as historical #56 evidence. Production prompt
deduplication intentionally changes its returned sets, so the default `memory-eval`
gate now reads `v2/` rather than rewriting the v1 baseline.

The v2 corpus measures canonical and ranked channels independently, with each case
declaring its own representative count and rendered-row character limits. Its report
includes selected representatives plus collapsed, count, character and bounded-scan
omission counts. Hard-zero gates cover semantic duplicates, correction-precedence
violations, contradiction false merges, ACL leaks, unsupported citations, stale
records and omission-accounting mismatches. Retrieval recall may not regress.

The threshold sweep uses reviewed scalar cosine scores produced by the actual
`intfloat/multilingual-e5-small` production path (FastEmbed 5.17.3, model revision
`614241f622f53c4eeff9890bdc4f31cfecc418b3`, `passage:` prefix, mean pooling) on
public synthetic EN/PL/DE/ES phrases. No vector or model/cache binary is checked in.
The selected cosine `0.84` is the highest tested threshold with zero false merges
and zero missed multilingual equivalents after compatibility guards. Higher values
lose the EN/PL equivalent; high-scoring negative, changed-time and changed-name
controls are retained separately by the guard.

The ranked row-character sweep tests 512, 1024, 2048, 4000 and 6000 characters.
The production default of 4000 is the smallest candidate preserving every expected
ranked representative in the padded calibration case. Header/statistics text remains
outside the row limit but is included in the reported full prompt cost.

The v1 baseline has 12 cases, including the two discriminating current-context
authorization checks above. Its stale/current timezone pair is seeded through the
production `correct_memory` transaction: the stale assertion remains available to
authorized exact audit inspection but is excluded from ordinary inspection and
canonical injection, so the stale-rate threshold is zero. The adversarial semantic
query still returns one authorized but irrelevant group memory. Extraction snapshot
precision/recall are both `0.75`; aggregate retrieval precision is `0.923077`.
Tighten these only when production behavior actually improves.

## Adding A Sanitized Regression

1. Reproduce the behavior conceptually; never copy exact private text.
2. Replace people, contexts, message IDs, attachment references, and model run IDs
   with invented fixture names that cannot be mapped back to the original.
3. Rewrite the content while preserving only the linguistic property under test,
   such as German-to-English paraphrase or correction order.
4. Replace real timestamps with small synthetic or relative values. Do not retain
   dates, platform message IDs, paths, URLs, credentials, attachment hashes, or
   filenames.
5. Represent attachments only through `reference_kind`, an invented MIME type, and
   empty storage metadata created by the narrow fixture seeder.
6. Use a sparse synthetic vector already in `embeddings`, or add an invented
   one-hot vector. Never paste provider vectors derived from private content.
7. Add explicit expected, forbidden, and citation-support keys. Confirm the new
   case fails before the production fix and passes afterward.
8. Run the gate with `--report /tmp/...`, inspect the machine report, and update
   `baseline.json` manually in a reviewable patch only when intended. The runner
   has no baseline-write option and tests never rewrite it.
