# Consolidator decision-eval harness (AI-judged)

Scaffolding to measure whether the **consolidator** makes the right decision
for a new playbook fragment against the existing rows, and to catch
regressions when the `playbook_consolidation` prompt changes.

It is **AI-judged**: no human gold labels are required at scoring time — an
LLM-judge labels each case and the headline metric is *agreement with the
judge*. A cheap deterministic *kind accuracy* is tracked alongside.

> **This is bounded scaffolding plus a tiny illustrative fixture, not a
> curated eval dataset.** Real cases are curated later. The fixture in
> `fixtures/illustrative_cases.json` exists only to exercise the harness
> end-to-end.

## Layout

| File | Purpose |
| --- | --- |
| `case.py` | `ConsolidationEvalCase` schema (`existing` rows + one `candidate`) + `kind_for_decision` |
| `judge.py` | `judge_consolidation_decision` AI-judge (panel of N, default 1, majority vote) returning `ConsolidationVerdict{correct, self_contradiction, reason}` |
| `runner.py` | `run_eval` runner + `EvalResults` metrics (kind accuracy, judge agreement, over/under-merge, self-contradiction) |
| `fixtures/` | `illustrative_cases.json` + loader (`load_illustrative_cases`) |

Mirrors the conventions of `tests/eval/reflection/` (Pydantic verdict,
`LiteLLMClient`-style client, model never hardcoded — it comes from the
rubric's `judge_model`).

## Kind mapping (explicit, not heuristic)

Unlike the reflection harness — whose decision carries no mode label and must
be reconstructed from field presence — a consolidation decision carries an
explicit discriminator `kind` (`unify` / `reject_new` / `differentiate` /
`independent`). So `kind_for_decision(decision)` simply returns
`decision.kind`; there is no heuristic to get wrong.

## What the judge adds: the self-contradiction dimension

The verdict carries a second flag beyond `correct`:

- **`self_contradiction`** — for a produced `unify`, did the merge fold together
  rules that contradict on the *same* situation, or collapse distinct do/avoid
  rules? This is **only the LLM's call** — the mechanical same-polarity guard
  was deliberately retired (orientation lives in rule wording, judged by the
  model). The rubric encodes the Option-B contract: a skill MAY hold
  mixed-polarity rules for *different* sub-aspects, but MUST NOT unify rules
  that contradict on the *same* situation (those go to `differentiate` /
  `reject_new`), and compose must preserve distinct rules.

For non-`unify` decisions `self_contradiction` is not applicable (recorded as
`None`, excluded from the rate denominator).

## Metrics

Deterministic (no LLM):
- `kind_accuracy` — fraction of cases whose produced `kind` equals `gold_kind`.
- `over_merge_rate` — of cases that should stay separate (`gold_kind` ∈
  {`differentiate`, `independent`}), the fraction that produced a merge/drop
  (`unify` / `reject_new`).
- `under_merge_rate` — of cases that should merge (`gold_kind` ∈ {`unify`,
  `reject_new`}), the fraction kept separate (`differentiate` / `independent`).

AI-judged:
- `judge_accuracy` — fraction the judge marked `correct` (None if none judged).
- `self_contradiction_rate` — among produced-`unify` judged cases, the fraction
  the judge flagged as self-contradicting (None when there is no judged
  `unify`; `0.0` when there are judged unifies but none flagged).

Panel ties break **conservatively, in opposite directions**: `correct` ties →
`False` (reject), `self_contradiction` ties → `True` (flag).

## Fixture coverage

| Case id | gold_kind |
| --- | --- |
| `unify_same_trigger_duplicate` | `unify` (classic dedup) |
| `unify_compose_related_subaspects` | `unify` (do-rule + avoid-rule, different sub-aspects → compose) |
| `differentiate_same_situation_contradiction` | `differentiate` (opposite advice, same trigger — must NOT unify) |
| `reject_new_redundant` | `reject_new` |
| `independent_unrelated` | `independent` |

## Running

Tests mock the LLM judge and the produced decision — no real API calls:

```bash
uv run pytest tests/eval/consolidation -o 'addopts=' -q
```

Anything that would hit a real API is decorated `@skip_low_priority` (the
real-*judge* smoke test, `test_real_judge_smoke`).

## Running against the live consolidator

`run_eval` accepts either a parallel `decisions` list (precomputed, as the
default tests use) or a `decision_provider` callable that maps a case to a
produced decision. To evaluate the **live** consolidator end-to-end, use the
provider in `providers.py`:

```python
from tests.eval.consolidation.providers import make_consolidation_decision_provider

provider = make_consolidation_decision_provider(llm_client=real_client, request_context=ctx)
results = run_eval(cases=load_illustrative_cases(), decision_provider=provider, llm_client=judge)
```

It builds `UserPlaybook` entities from each case's `existing`/`candidate`, calls
the consolidator's `_consolidation_decisions(...)` seam (real
`playbook_consolidation` prompt + LLM, no search/apply), and returns the
produced `ConsolidationDecision`. This path makes real LLM calls, so it lives
behind the `@skip_low_priority` smoke `test_live_consolidation_provider_real`
(run with `RUN_LOW_PRIORITY=1` + an API key). A non-skipped mocked-seam test
covers the provider's construction in default CI.

## Comparing two prompt versions (deviation guard)

To gate a candidate `playbook_consolidation` version against a baseline (catch
regressions when iterating the prompt across versions), use the shared CLI which
runs this harness under both pinned versions and fails on regression:

```bash
uv run python -m tests.eval.prompt_deviation_guard \
    --component consolidation --candidate-version vX.Y.Z --judge-model claude-haiku-4-5
```

See `tests/eval/prompt_deviation_guard.py` (gate logic unit-tested in
`tests/eval/test_prompt_deviation_guard.py`).
