# Playbook ask_human invocation eval

This eval scores the resumable playbook extractor's decision to call
`ask_human`. It is a binary precision/recall eval over natural agent-user
trajectories: the positive label means missing shared/org context is required
to know the durable positive action, target, procedure, policy, or standard.

The golden set lives in `tests/eval/golden_set/playbook_ask_human/cases.yaml`.
It contains 36 cases:

- 12 positive `ask_human` cases.
- 12 negative cases where a playbook is extractable without asking.
- 12 negative cases where no durable playbook should be extracted.

The cases cover customer support, education, finance, healthcare,
general-purpose assistance, digital-employee/internal-ops, sales, and
legal/compliance workflows. Transcripts should stay natural; they should not
mention `ask_human` or Agent Builder directly.

Run the local checks:

```bash
uv run pytest tests/eval/playbook_ask_human -o 'addopts=' -q
uv run ruff check tests/eval/playbook_ask_human
uv run pyright tests/eval/playbook_ask_human
```

