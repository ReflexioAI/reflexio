# Extraction golden-set eval runner (AI-judged)

Scaffolding to score the **extractor** (profiles + playbooks) against curated
gold cases, and to catch regressions when the extraction prompts change.

It is **AI-judged** and **float-scored** — unlike the reflection /
consolidation *decision*-evals (which classify a decision into a discrete
kind), extraction quality is graded by an LLM judge on two continuous
dimensions. The judge, rubric, golden cases, and fixtures already exist; this
package adds the **runner** that wires them together (load → extract / supply →
judge → aggregate).

> **This is bounded scaffolding plus a tiny illustrative fixture, not a
> curated eval dataset.** The cases in `../golden_set/extraction/*.yaml` exist
> only to exercise the harness end-to-end.

## What it reuses (not re-implemented here)

| Piece | Lives in | Role |
| --- | --- | --- |
| `LLMJudge.score(*, expected, actual)` | `tests/eval/judge.py` | the shared judge — returns `JudgeScore{signal_f1, answer_correctness, grounded_rate, rationale}` |
| `extraction_rubric.yaml` | `tests/eval/judge_prompts/` | the extraction rubric (`{expected}` / `{actual}`) |
| golden cases | `tests/eval/golden_set/extraction/*.yaml` | `id`, `sessions`, `expected_profiles`, `expected_playbooks`, `must_NOT_include_profiles`, `notes_for_judge` |
| `extraction_case` / `extraction_judge` fixtures | `tests/eval/conftest.py` | parametrize over cases; provide a stub judge (or real LLM under `REFLEXIO_EVAL_REAL_JUDGE=1`) |

This package adds only `runner.py` (`run_eval` / `score_case` / `EvalResults` /
`CaseOutcome`) — it contains no scoring prompt of its own.

## Metrics

The judge scores each case on two dimensions in `[0, 1]`:

- **`signal_f1`** — did the extraction capture the expected *signals*, including
  nuance the case flags (TTL granularity, supersession, rationale detail)?
- **`grounded_rate`** — are the emitted items' `source_span`s genuinely verbatim
  in the session transcript (no hallucinated spans)?

(`answer_correctness` is a search-only dimension, pinned to 0 here and ignored.)

`EvalResults` aggregates per-case scores into:
- `signal_f1_mean`, `grounded_rate_mean` (arithmetic means; 0.0 when empty),
- `pass_rate(threshold=0.7)` — fraction of cases with `signal_f1 >= threshold`
  **and** `grounded_rate >= threshold`,
- `n`, and a `summary()` block.

There is **no kind label or confusion matrix** — extraction is graded, not
classified.

## Running

The default `extraction_judge` is stubbed, so the harness runs without
credentials:

```bash
uv run pytest tests/eval/extraction -o 'addopts=' -q
```

Set `REFLEXIO_EVAL_REAL_JUDGE=1` (with an API key) to score against a real LLM
judge. The `@skip_low_priority` `test_real_judge_smoke` (gated by
`RUN_LOW_PRIORITY=1`) exercises the live-judge path.

## Supplying extractions

`run_eval` takes either a parallel `extractions` list (precomputed
`(profiles, playbooks)` per case — the tests use the case's own gold items as a
perfect-extraction baseline) or an `extraction_provider` callable. To evaluate
the **live extractor** end-to-end, supply a provider that runs
`PlaybookExtractor.run()` / `ProfileExtractor.run()` over a case's `sessions`
with a real `request_context` and returns the produced `(profiles, playbooks)`.
Wiring that live provider is a follow-up; the harness ships with
precomputed-extraction tests plus the real-judge smoke. Produced items may be
`UserProfile` / `UserPlaybook` entities or plain dicts (a `_to_dict` shim
normalizes both for the judge).
