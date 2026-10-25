# Native tools + Python frontend vs legacy protocol — measured comparison

Date: 2026-10-25
Branch: `work/native-tools` @ `7ec0a72`
Binary: `./target/release/adapsis` (release build of 7ec0a72)

## What was measured

Three conversation conditions, identical task wording apart from a one-line
language directive:

| condition | flags | how the model acts |
|---|---|---|
| **legacy** | *(no `--native-tools`)* | writes Adapsis in `<code>` blocks, ends with `!done` |
| **native-Adapsis** | `--native-tools` | native tool protocol with `language="adapsis"` |
| **native-Python** | `--native-tools` | native tool protocol with `language="python"` |

8 small tasks (typed arithmetic, if/elif classification, while-loop sum, list
iteration, string concat, cross-function call, bool xor, a >2-statement function
that requires tests). Each task fixes an exact module name + function signature.
Scoring used **hidden** `/api/eval` cases the model never saw: a run is a
success only if the turn reported completion **and** every hidden case returned
the exact expected value.

Models (only routes verified working through the gateway at the time):
`deepseek-v4-flash` and `deepseek-v4-pro`. 3 repeats per cell → **144 runs**,
each on a fresh isolated instance (loopback port 3110+, per-run temp
session/modules/contexts/memory, `--access-level adapsis-only`, no
permissions file, `--host 127.0.0.1`). Production instance on :3002 was never
touched.

## Results (condition × model, n=24 per cell)

| condition / model | success | completed | provider calls | in tok | out tok | lat (ms) |
|---|---|---|---|---|---|---|
| legacy / deepseek-v4-flash | **96%** | 100% | 5.2 | 166,477 | 879 | 10,558 |
| legacy / deepseek-v4-pro | **100%** | 100% | 3.0 | 99,220 | 4,086 | 39,744 |
| native-Adapsis / deepseek-v4-flash | **17%** | 17% | 19.5 | 166,462 | 10,677 | 74,283 |
| native-Adapsis / deepseek-v4-pro | **8%** | 8% | 19.1 | 288,171 | 22,481 | 186,502 |
| native-Python / deepseek-v4-flash | **100%** | 100% | 4.2 | 15,483 | 452 | 4,935 |
| native-Python / deepseek-v4-pro | **100%** | 100% | 4.1 | 15,824 | 716 | 9,856 |

### Per-task success (6 runs per cell = 2 models × 3 repeats)

| task | legacy | native-Adapsis | native-Python |
|---|---|---|---|
| arith_add | 6/6 | 0/6 | 6/6 |
| bool_xor | 5/6 | 1/6 | 6/6 |
| classify_sign | 6/6 | 1/6 | 6/6 |
| concat | 6/6 | 0/6 | 6/6 |
| cross_call | 6/6 | 1/6 | 6/6 |
| multi_steps (needs tests) | 6/6 | 3/6 | 6/6 |
| sum_list | 6/6 | 0/6 | 6/6 |
| sum_to | 6/6 | 0/6 | 6/6 |

### Failure classes (non-successful runs)

| condition / model | failure class | count |
|---|---|---|
| legacy / flash | wrong-result (function never defined; `!done` accepted) | 1 |
| native-Adapsis / flash | no-completion-validation (invalid tool payloads, never `finish`) | 20 |
| native-Adapsis / pro | no-completion-validation | 18 |
| native-Adapsis / pro | protocol-error | 1 |
| native-Adapsis / pro | timeout (>300s) | 3 |

## Key findings

1. **native-Python is the clear winner for these models**: 100% success on both
   models, ~4 provider calls, ~15k input tokens, ~5–10s. It is ~10× cheaper
   (input tokens) than legacy and ~10× cheaper than the failed native-Adapsis
   path.
2. **native-Adapsis is not usable by DeepSeek through this protocol.** Both
   models mostly fail (17% / 8%) by emitting invalid tool payloads
   (`mutate forbids top-level query`, `parse error: could not parse type`,
   `native inputs … must be literals`) and never reaching a successful
   `finish` within the 20-iteration budget. Pro is additionally very slow: its
   reasoning loop burns ~15–30s and ~15k tokens per iteration, so three pro
   runs exceeded the 300s per-run timeout after 16–19 calls (~360–400k input
   tokens) still unfinished.
3. **legacy remains highly reliable** (96% / 100%) but is expensive: the full
   Adapsis language spec is re-sent every request (166k in-tokens for flash).
   The single legacy flash failure is a *false completion*: the model emitted
   `!done` without defining the function, and `!done` was accepted because
   there were no untested functions — completion was reported but no work
   existed.
4. The >2-statement task (`multi_steps`, which requires passing tests before
   `eval`/`finish`) was the *most* successful native-Adapsis task (3/6),
   suggesting the test gate itself is not what defeats the native-Adapsis
   model; the raw Adapsis tool payload emission is.

## Caveats (must be read with the numbers)

1. **The native prompt is Python-first by design** (`src/prompt/native.rs`
   spends its first section teaching the Python workflow). The native-Adapsis
   condition is therefore *under-instructed* relative to the legacy Adapsis
   prompt, which carries the full language spec. This is a confound: the
   native-Adapsis failures measure "DeepSeek emitting Adapsis inside the native
   tool envelope with a Python-first prompt", not a pure Adapsis-syntax
   comparison.
2. **Legacy input-token totals include the full Adapsis language spec** in every
   request; native input-token totals do not. Token figures are not comparable
   across the legacy vs native boundary as a proxy for "efficiency of the
   model's own output" — they reflect the prompt design.
3. **DeepSeek only, n=3 per cell**, and the task set lies entirely within the
   Python subset (the intersection of both languages). No Anthropic/ChatGPT
   routes were measured (unavailable at the time). Small-n; do not over-read
   per-task cells.
4. **Mock-free live provider.** This is not the offline fixture gate; the
   gateway's tool-call passthrough (llm-gateway #2) was not certified upstream,
   but a live smoke confirmed the gateway forwards `tool_calls` end-to-end.
5. **300s per-run timeout.** Three native-Adapsis/pro runs hit it while still
   working (16–19 calls in); they are recorded as `timeout` with their partial
   token/call counts.
6. **Completion ≠ correctness (legacy).** "completed" for legacy is "`!done`
   accepted / no iteration-limit error"; a model can satisfy it without doing
   the work (the one flash failure). Native `completed:true` requires a
   successful `finish` against the turn baseline.
7. Hidden evals use Adapsis expression syntax (named args) for `/api/eval`
   regardless of the model's language; string results are compared with their
   surrounding quotes.

## Method / definitions

- **success** = turn reported completion AND all hidden `/api/eval` cases match
  exact expected values.
- **provider calls** = `[llm:main] latency_ms=… usage: …` lines in the instance
  stderr (one LLM generation per iteration, both protocols).
- **tokens / latency** parsed from those same stderr lines
  (`input=… output=… provider_total=…`); latency is the gateway-side per-call
  elapsed time, wall time is the client-side `/api/ask` round trip.
- **failure classes**: `no-completion-validation` (never `finish`, invalid tool
  payloads observed), `protocol-error`, `timeout`, `wrong-result` (completed but
  hidden checks fail), `untested-eval-refusal` (eval refused for an untested
  >2-statement function — none occurred in the final set).

## Reproducibility

```sh
./target/release/adapsis os --port <p> -s /tmp/opencode/compare/runs/<id>/session.json \
  --url http://127.0.0.1:4000 --model deepseek-v4-flash \
  --access-level adapsis-only --host 127.0.0.1 \
  --log-file .../adapsisos.log --training-log .../training.jsonl [--native-tools]
# env: ADAPSIS_MODULES_DIR, ADAPSIS_CONTEXTS_DIR, ADAPSIS_MEMORY_DB -> per-run temp dirs
```

Harness: `tools/native_compare.py --smoke` (3 runs) or the full
`--concurrency 3 --repeats 3` run. Task set + hidden cases:
`tools/native_compare_tasks.json`.

## Artifacts

- `tools/native_compare.py` — harness (sole writer of new `tools/` files).
- `tools/native_compare_tasks.json` — 8 tasks + hidden eval cases.
- `/tmp/opencode/compare/results/runs.jsonl` — 144 per-run records.
- `/tmp/opencode/compare/results/summary.txt` — aggregate table.
- `/tmp/opencode/compare/smoke/` — archived smoke run (3 records).
- `/tmp/opencode/compare/runs/<run_id>/` — per-run instance logs (token/latency
  accounting source).
