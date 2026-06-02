# Multi-round memory-scenario harness (AI-judged)

A lightweight, in-process harness that runs **controlled multi-round memory
scenarios** to measure whether the **extractor + consolidator + reflector**
evolve a playbook correctly *as a chain* over several rounds — the coverage the
SWE-bench A/B pair can't give (a single A→B pair never exercises the consolidator
or reflector). No tmux, no Docker, no agent.

It reuses the existing pieces wholesale: the three **live providers**
(`tests/eval/{extraction,consolidation,reflection}/providers.py`) and the three
**component judges**. It adds only the round loop, a thin apply shim, and the
scenario fixtures.

## How it works

```
book := seed_book
for round in scenario.rounds:
    learn   → extraction_provider(interactions) → playbooks
              consolidation_provider(existing=book, candidate=p) → decision
              judge the decision; apply it to `book`   (book grows/merges)
    reflect → reflection_provider(window, cited=book[id]) → decision
              judge the decision; apply it to `book`   (cited rule revised)
end-state → optional judge of the final `book` vs gold_end_state
```

`book` **accumulates across rounds** (held in memory; applied via the shim), so a
later `reflect` round sees the rules earlier `learn` rounds extracted and
consolidated. Each round's decision is the **real component's output**, judged by
that component's own judge.

## Not a uniform scorecard

Different components measure different things, so each round is judged by its
**own** component judge with its native verdict (consolidation: kind +
self-contradiction; reflection: label/intent; extraction: signal). The only
cross-component aggregate is a thin boolean roll-up:

`ScenarioResult.scenario_passed` = every round judged-correct **and** the
end-state not judged-wrong.

This is the deliberate "standardize = shared code, not a shared metric" choice.

## The apply shim (`book.py`)

`apply_consolidation` / `apply_reflection` are **test-only glue** that
mechanically reflect a real decision into the in-memory `book` so the next round
can read accumulated state — they do **not** re-implement service apply
semantics. The shim mirrors the consolidator's id contract: `unify`
`archive_existing_ids` are **list positions**; `differentiate` `existing_id` is a
`BookRule.id`.

## Fixtures (`fixtures/scenarios.json`)

| id | exercises | gold |
| --- | --- | --- |
| `compose_grows_skill` | extractor + consolidator | a learn round grows a multi-rule skill (`unify`) |
| `no_self_contradiction` | consolidator (the linchpin) | opposing advice on the same trigger → `differentiate`, not merged |
| `reflect_corrects` | reflector | a cited rule that failed → `reflect` tightens it |

## Running

The mocked CI tests run with no LLM and no credentials:

```bash
uv run pytest tests/eval/scenarios -o 'addopts=' -q
```

The `@skip_low_priority` `test_scenario_real` runs one scenario end-to-end
against real **haiku** providers + judges (run with `RUN_LOW_PRIORITY=1` + an API
key). It's the "as-designed" multi-round signal; CI is mocked.
