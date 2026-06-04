# Reflection decision-eval harness (AI-judged)

Scaffolding to measure whether the **reflection** step makes the right
decision for a cited memory item, and to catch regressions when the
`memory_reflection` prompt changes.

It is **AI-judged**: no human gold labels are required at scoring time —
an LLM-judge labels each case and the headline metric is *agreement with
the judge*. A cheap deterministic *label accuracy* is tracked alongside.

> **This is bounded scaffolding plus a tiny illustrative fixture, not a
> curated eval dataset.** Real cases are curated later. The fixture in
> `fixtures/illustrative_cases.json` exists only to exercise the harness
> end-to-end.

## Layout

| File | Purpose |
| --- | --- |
| `case.py` | `ReflectionEvalCase` schema + `label_for_decision` field-presence → label mapping |
| `judge.py` | `judge_reflection_decision` AI-judge (panel of N, default 1, majority vote) |
| `runner.py` | `run_eval` runner + `EvalResults` metrics (accuracy, confusion, false-tighten, over-specialization) |
| `fixtures/` | `illustrative_cases.json` + loader (`load_illustrative_cases`) |

Mirrors the conventions of the existing golden-set harness in
`tests/eval/` (Pydantic verdict, `LiteLLMClient`-style client, model
never hardcoded — it comes from the rubric's `judge_model`).

## Label mapping (field-presence → coarse label)

The live `ReflectionDecision` carries **no mode label**; the outcome is
encoded by which replacement fields are set. For scoring we collapse
that into a coarse label with this precedence:

1. `new_profile_time_to_live` set → `ttl`
2. `new_trigger` changed → `tighten` / `widen` (longer trigger = narrower
   = tighten; shorter = broader = widen; ambiguous → `scope`)
3. `new_content` substantively changed → `rewrite`
4. nothing set → `no_change`

There is no mechanical `flip` label: an orientation-reversing rewrite is
not distinguishable from any other rewrite without a polarity heuristic
(retired — orientation is wording, judged by the LLM). Such a case is
labeled `rewrite`; whether the rule was *correctly* reversed is the
AI judge's call.

## Fixture coverage

| Case id | Label |
| --- | --- |
| `no_change_stable_preference` | `no_change` |
| `tighten_overbroad_trigger` | `tighten` |
| `widen_too_narrow_trigger` | `widen` |
| `flip_polarity_reversal` | `rewrite` (orientation reversal; judge verifies the reversal) |

## Running

Tests mock the LLM judge and the produced decision — no real API calls:

```bash
uv run pytest tests/eval/reflection -o 'addopts=' -q
```

Anything that would hit a real API is decorated `@skip_low_priority`.

## Running against the live reflection step

`run_eval` takes either a parallel `decisions` list (precomputed, as the default
tests use) or a `decision_provider` callable. To evaluate the **live** reflection
step end-to-end, use the provider in `providers.py`:

```python
from tests.eval.reflection.providers import make_reflection_decision_provider

provider = make_reflection_decision_provider(llm_client=real_client, request_context=ctx)
results = run_eval(cases=load_illustrative_cases(), decision_provider=provider, llm_client=judge)
```

It builds the cited `UserPlaybook` / `UserProfile` from each case's `cited_item`,
runs the real `ReflectionExtractor` (real `memory_reflection` prompt + LLM; no
storage), and returns the produced `ReflectionDecision`. This makes real LLM
calls, so it lives behind the `@skip_low_priority` smoke
`test_live_reflection_provider_real` (run with `RUN_LOW_PRIORITY=1` + an API
key); a non-skipped mocked-seam test covers the provider's construction in
default CI.

## Comparing two prompt versions (deviation guard)

To gate a candidate `memory_reflection` version against a baseline (catch
regressions when iterating the prompt across versions), use the shared CLI which
runs this harness under both pinned versions and fails on regression:

```bash
uv run python -m tests.eval.prompt_deviation_guard \
    --component reflection --candidate-version vX.Y.Z --judge-model claude-haiku-4-5
```

See `tests/eval/prompt_deviation_guard.py` (gate logic unit-tested in
`tests/eval/test_prompt_deviation_guard.py`).
