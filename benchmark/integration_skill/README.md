# Integration skill behavioral benchmark

Measure whether a coding agent can repair an existing managed-cloud integration using the portable skill. This is a local application-integration benchmark, not a latency benchmark for Reflexio or evidence that an offline tuning run completed.

The default experiment has two applications (Python SDK and Python 3.11-compatible HTTP), two pinned skill versions, and three repetitions per version: 12 sequential Codex runs. Both applications already search, filter context, call the model, and publish responses; the deliberate defect is missing retrieved-learning references. The complete skill directory changes between versions; the application, local API guide, dependency environment, prompt, model, and reasoning setting remain the same. The HTTP fixture is compatible with Python 3.11 but runs under the common Python 3.12 benchmark interpreter.

Versions: original `7ce76a843eef201f53029a027cba21a914a3ed29`, updated `91956d6c59865746a1591062dfa0ff2202285b90` (PR #594). The updated commit must be available locally, e.g. `git fetch origin team/integration-tuning-attribution`. Run from a checkout whose SDK matches the original commit; the candidate changes only skill documents. No installation or model calls occur in ordinary test collection.

## Setup and run

Requires Linux, Git, uv, Python 3.12, a working authenticated Codex CLI supporting permission profiles, and SDK/pytest dependencies installed before the experiment. Build the lightweight wheel using the same command as repository CI:

```bash
(cd client_dist && uv build --wheel --out-dir /tmp/reflexio-skill-benchmark-wheels)
uv venv /tmp/reflexio-skill-benchmark-env --python 3.12
uv pip install --python /tmp/reflexio-skill-benchmark-env/bin/python \
  /tmp/reflexio-skill-benchmark-wheels/reflexio_client-0.2.16-py3-none-any.whl pytest
PYTHONPATH="$PWD" nice -n 10 /tmp/reflexio-skill-benchmark-env/bin/python \
  -m benchmark.integration_skill.runner \
  --python /tmp/reflexio-skill-benchmark-env/bin/python \
  --output /tmp/reflexio-integration-skill-results
```

The output directory must be new. Defaults: `gpt-6.1-sol`, medium reasoning, three repetitions, 900 seconds per run. `--model`, `--repetitions`, and `--timeout` permit explicit experiments; results from different settings must not be pooled. Runs alternate original/updated order, reversing it for the second repetition. There is no automatic retry or feedback to the agent after grading.

Each trial receives a fresh application and a separate copy of the installed environment. The runner temporarily copies CLI authentication into a private directory, removes it after execution, and loads only its own configuration: no inherited MCP servers, hooks, web search, or application credentials. Command execution can edit the application but cannot edit the SDK, skill, support doubles, dependency manifest, original tests, or instructions. A real sandbox probe verifies protected-file writes and socket creation fail before each run. The independent grader executes separately with read-only application access and network disabled. Agent model calls still use the Codex service; this is not a free/no-model-cost benchmark.

## Grading and interpretation

The external grader invokes the actual edited `handle_turn`. It captures model input and outgoing publication through a fake HTTP transport; the SDK case uses real SDK request validation and serialization. It checks populated, empty, unsuccessful, exception, consecutive-turn, and concurrent-request scenarios. Concurrent turns deliberately share a user/session while retaining distinct learning IDs; a barrier forces their model calls to overlap. A user playbook and agent playbook deliberately share the same numeric ID to test kind preservation.

Every populated turn must keep the expected context and publish exactly its retained references on the agent interaction. Discarded candidates are excluded. IDs must be strings, identity fields must agree, user responses must remain correct, and each turn must search/model/publish exactly once. A run passes only if all behavioral checks pass, protected inputs are unchanged, and the agent completes successfully. Infrastructure errors and timeouts remain visible; they do not count as passes.

`manifest.json` records the pinned skills and prompt. `results.json` records per-run checks, process status, elapsed time, usage when emitted, observed pytest commands, and changed files. Each trial preserves its transcript, final agent report, grading output, and diff. `report.md` provides the comparison. Missing token usage is explicitly counted; token totals cover only reported usage. Inspect transcripts and final reports for unsupported live/tuning claims and unnecessary changes before publishing conclusions; keyword matching is not a correctness grader. Twelve trials are an initial comparison, not statistical proof. Equal pass counts mean no observed improvement on these fixtures.

## Harness checks

```bash
PYTHONPATH="$PWD" /tmp/reflexio-skill-benchmark-env/bin/python -m pytest \
  benchmark/integration_skill/tests -q -c /dev/null -p no:cacheprovider
uv run ruff check benchmark/integration_skill
uv run ruff format --check benchmark/integration_skill
uv run pyright --project benchmark/integration_skill/pyrightconfig.json \
  --pythonpath /tmp/reflexio-skill-benchmark-env/bin/python --threads 4
```

Tests exercise actual broken and repaired applications, injected regressions, event parsing, timeout termination, and result aggregation without agent/model calls. Live cloud attribution and Braintrust-connected integrations are outside this initial benchmark. No production configuration, server, schema, or customer data changes.
