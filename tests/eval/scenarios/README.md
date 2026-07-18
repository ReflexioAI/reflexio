# Multi-round learning-scenario harness (AI-judged)

This lightweight in-process harness measures how the extractor and playbook
consolidator evolve a playbook over controlled learning rounds.

Each round feeds interactions to the extraction provider, routes produced
playbooks through the consolidation provider against the accumulated in-memory
book, judges the decision, and applies it through the test-only shim in
`book.py`. An optional end-state judge compares the final book with the
scenario's expected outcome.

The fixtures cover:

- composing a new rule into an existing deploy skill; and
- differentiating opposing advice instead of creating a contradiction.

Run the deterministic mocked tests with:

```bash
uv run pytest tests/eval/scenarios -o 'addopts=' -q
```

The `@skip_low_priority` smoke uses live Haiku providers and judges when
`RUN_LOW_PRIORITY=1` and provider credentials are available.
