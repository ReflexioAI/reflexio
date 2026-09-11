# /tests/eval/scenarios
Description: Controlled multi-round extraction/consolidation evaluation.

## Main Entry Points


- **`case.py`** — scenario model
- **`runner.py`** — round orchestration
- **`book.py`** — test-only state application
- **`fixtures/scenarios.json`** — illustrative scenarios

## Purpose


Inspect how learning evolves over multiple rounds and detect contradictory consolidation.

## Architecture Pattern


Each round extracts, consolidates against the accumulated in-memory book, judges the decision, and applies it through the test shim.

The fixtures cover:

- composing a new rule into an existing deploy skill; and
- differentiating opposing advice instead of creating a contradiction.

Run the deterministic mocked tests with:

```bash
uv run pytest tests/eval/scenarios -o 'addopts=' -q
```

The `@skip_low_priority` smoke uses live Haiku providers and judges when
`RUN_LOW_PRIORITY=1` and provider credentials are available.
