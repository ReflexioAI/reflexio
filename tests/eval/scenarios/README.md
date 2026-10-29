# Multi-round memory evaluation

Each round extracts against literal source interactions, runs the audited
Create/Replace/Remove method, independently judges the complete decision batch,
and projects every output into the next round's in-memory book. Projection uses
the production decision validator and projector. Existing-only cleanup is valid;
unmentioned rows retain their identities. Evidence coordinates remain stable
across rounds, so retained rules still cite their original observations.

`case.py` defines scenarios; `runner.py` orchestrates rounds; `book.py` applies
validated batches. The two synthetic scenarios exercise composition and
contradiction avoidance. They establish harness behavior, not production quality
or SQL transaction correctness.

```bash
uv run pytest tests/eval/scenarios -o 'addopts=' -q
```

Default tests use scripted providers. The low-priority live smoke requires a
manual paid-provider opt-in and credentials.
