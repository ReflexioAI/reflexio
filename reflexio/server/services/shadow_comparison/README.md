# shadow_comparison
Description: Per-turn regular-vs-shadow comparison service that judges shadow-bearing assistant turns and stores dashboard-facing verdicts outside session-level evaluation.

## Main Entry Points

| File | Purpose |
|------|---------|
| `worker.py` | Process-local bounded daemon queue (`ShadowComparisonWorker`) used by publish paths; re-resolves `get_reflexio(org_id)` inside workers so delayed jobs use current org config/storage. |
| `dispatcher.py` | Filters publish-request interactions to assistant turns with `shadow_content`, builds request-local transcript context, invokes the judge, and saves verdicts when storage supports shadow-comparison tables. |
| `judge.py` | `ShadowComparisonJudge` renders the `shadow_comparison` prompt through `PromptManager`, calls `LiteLLMClient.generate_chat_response()` with structured output, and stamps prompt version/position metadata. |
| `outcome.py` | Pure helpers for position randomization and deriving Reflexio-relative win/tie/loss from stored verdicts. |

## Purpose

1. **Compare safely at turn level** - Avoids session-level trajectory contamination by judging each shadow-bearing assistant turn independently.
2. **Keep publish latency bounded** - Enqueues jobs to a small local worker pool and drops with telemetry when the queue is full.
3. **Separate verdict storage** - Writes `ShadowComparisonVerdict` rows only when the active storage backend advertises support; failures do not abort publishing.

## Architecture Pattern

`GenerationService` publish flow enqueues a `ShadowComparisonJob` with `org_id`, interactions, `session_id`, and `agent_version`; `worker.py` drains the queue, rehydrates the current `Reflexio` instance, and calls `dispatcher.py`. The dispatcher sorts request-local interactions, formats prior-turn context, then calls `judge.py` for each eligible assistant turn. Verdict rows are independent of `AgentSuccessEvaluationResult.regular_vs_shadow`, which is historical/nullable only.

## Requirements / Problems to Avoid

- **Do not resurrect session-level shadow comparison** — multi-turn user messages react to the regular trajectory, not the shadow trajectory.
- **Do not pass live storage/request-context/LLM objects into queued jobs** — `worker.py` intentionally re-resolves them from `org_id` to survive cache/config changes.
- **Use `LiteLLMClient` and `PromptManager` only** — no direct OpenAI/Claude clients and no hardcoded judge prompts.
- **Treat judge/save failures as best-effort** — one bad turn or unsupported storage backend must not fail the publish request.
