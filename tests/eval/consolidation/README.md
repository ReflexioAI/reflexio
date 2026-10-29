# Consolidation evaluation

This harness evaluates complete Create/Replace/Remove decision batches against
source-bearing cases. Its five synthetic fixtures exercise the harness; they
are not curated evidence of production quality.

`case.py` models original sessions, candidate and incumbent evidence coordinates,
and the expected batch summary (`create`, `replace`, `remove`, or `mixed`).
`providers.py` runs the actual audited reconciliation method using the supplied
client and its configured model. It includes both draft and independent audit,
returns the complete batch, and rejects missing source references. Retrieval,
transactional persistence, and governance have separate service-level tests.

`runner.py` reports deterministic action-summary accuracy, independent judge
agreement, and contradictions across the resulting output memory. Action names
alone do not establish semantic correctness: Replace can combine or split rows.
The judge receives the complete batch and original source messages. Correctness
ties reject; contradiction ties flag the case for review. Over/under-merge rates
were removed because their former four-action classification no longer applies.

Run keyless deterministic tests:

```bash
uv run pytest tests/eval/consolidation -o 'addopts=' -q
```

Live provider/judge tests are paid manual opt-ins behind `@skip_low_priority`;
never enable them merely to run the default regression suite.

A live evaluation can construct a provider with:

```python
provider = make_consolidation_decision_provider(llm_client=real_client)
results = run_eval(cases=load_illustrative_cases(), decision_provider=provider, llm_client=judge)
```

For manual paid prompt comparisons, supply directories containing all three
current reconciliation prompt assets. The guard freezes each exact bundle and
records its SHA256 identity; it does not select a PromptManager version.

```bash
uv run python -m tests.eval.prompt_deviation_guard \
  --component consolidation --candidate-prompts /path/to/candidate \
  --judge-model claude-haiku-4-5
```

`--baseline-prompts` defaults to the shipped bundle. Exit codes are 0 for pass,
1 for metric regression, and 2 for evaluation/infrastructure failure.
