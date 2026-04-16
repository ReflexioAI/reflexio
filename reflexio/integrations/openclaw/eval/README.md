# Reflexio-OpenClaw Integration Eval

> Part of the [OpenClaw Integration](../README.md). See also the [Reflexio Code Map](../../../README.md) for project-wide context.

End-to-end evaluation suite for the Reflexio-OpenClaw integration. It starts a
throwaway local Reflexio server (SQLite, temp directory), exercises each scenario,
and reports pass/fail with timing.

## What This Evaluates

| Category | What is checked |
|---|---|
| **capture** | Conversations published via `publish_interaction` produce extracted user playbooks |
| **retrieve** | `search` returns playbooks whose trigger matches the query; irrelevant ones are absent |
| **multi_user** | Two user IDs see their own playbooks; agent playbooks are visible to both |
| **aggregation** | Similar corrections from multiple users consolidate into deduplicated agent playbooks |
| **resilience** | CLI commands fail gracefully (no traceback crash) when the server is unreachable |

## How to Run

```bash
# Full suite
uv run python -m reflexio.integrations.openclaw.eval.runner

# Single scenario
uv run python -m reflexio.integrations.openclaw.eval.runner --scenario correction_capture

# Multiple scenarios
uv run python -m reflexio.integrations.openclaw.eval.runner \
    --scenario playbook_retrieval \
    --scenario search_relevance
```

The runner exits with code `0` if every scenario passes, `1` otherwise.

> **Note**: The runner starts its own isolated server — no existing Reflexio
> instance is required.  The server needs an LLM API key in the environment
> (e.g. `OPENAI_API_KEY`) for the `capture` scenarios that trigger playbook
> extraction.  `retrieve`, `aggregation`, and `resilience` scenarios seed data
> directly and do not call the LLM.

## Scenario Categories

### `capture`
Publishes a realistic conversation containing a user correction and asserts that
the server extracted at least one user playbook matching the corrected behaviour.

Scenarios: `correction_capture`, `tool_failure_extraction`

### `retrieve`
Seeds playbooks directly (no LLM) and verifies that semantic search returns the
relevant one and suppresses irrelevant ones.

Scenarios: `playbook_retrieval`, `search_relevance`

### `multi_user`
Publishes corrections from two different user IDs, runs aggregation, then checks
that a search from each user's perspective includes agent-level playbooks.

Scenarios: `multi_user_isolation`

### `aggregation`
Seeds three semantically similar user playbooks and runs aggregation, then checks
that the resulting agent playbooks are de-duplicated (≤ expected count).

Scenarios: `aggregation_dedup`, `cron_aggregation`

### `resilience`
Stops the server, runs a CLI command that would normally contact it, and verifies
the command exits cleanly without an unhandled traceback.

Scenarios: `graceful_degradation`

## Adding New Scenarios

1. Open `dataset.json` and append a new object to the `"scenarios"` array.
2. Give it a unique `"id"`, a `"category"`, and a `"steps"` list.
3. Each step is `{"action": "<name>", "params": {...}}`.

Available actions and their required params:

| Action | Key params |
|---|---|
| `publish_interaction` | `user_id`, `agent_version`, `interactions` (list of `{role, content}`) |
| `wait_extraction` | `timeout_s` (default 60) |
| `verify_playbook_exists` | `user_id`, `trigger_contains`, `content_contains` |
| `seed_user_playbook` | `user_id`, `agent_version`, `content`, `trigger`, `rationale`, `blocking_issue` |
| `search` | `query`, `user_id` |
| `verify_result_contains` | `field` (content/trigger), `contains` |
| `verify_result_not_contains` | `field`, `contains` |
| `run_aggregation` | `agent_version`, `wait` (bool) |
| `verify_has_agent_playbooks` | _(none)_ |
| `verify_agent_playbook_count` | `agent_version`, `max_expected`, `content_contains` |
| `run_cli` | `command` (list) |
| `verify_exit_code` | `expected` |
| `run_cli_expect_failure` | `command`, `should_not_crash` |
| `stop_server` / `start_server` / `verify_server_running` | _(none)_ |

Steps execute sequentially; the first failure stops the scenario.  State (last
search results, last exit code) resets between scenarios.

## See Also

- [OpenClaw Integration README](../README.md) -- parent integration overview
- [Code Map (root README)](../../../README.md) -- high-level overview of all Reflexio components
