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

The v1 baseline has 12 cases, including the two discriminating current-context
authorization checks above. It intentionally records limitations that predate
contradiction work in #57: one stale assertion is still returned (`1/14`, rate
`0.071429`), and the adversarial semantic query returns one authorized but
irrelevant group memory, giving retrieval precision `0.928571`. Extraction
snapshot precision/recall are both `0.75`. Tighten these only when production
behavior actually improves.

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
