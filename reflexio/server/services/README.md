# server/services
Description: Core business-logic layer — LLM orchestration, extraction, evaluation, optimization, search preparation, storage access, and long-running operation state.

> This is a directory-local index. For the full request flow, workflow tables (versioning, generation modes, cluster change detection), and the `OperationStateManager` use cases, see the parent [server README](../README.md#services).

**Service Boundary**: services own the LLM/extraction/evaluation/storage logic; API endpoints only authenticate, build `RequestContext`, and delegate into `Reflexio` or a focused service helper.

## Orchestration & Base Infrastructure

| File | Purpose |
|------|---------|
| `generation_service.py` | `GenerationService` — saves interactions, runs profile + playbook generation in parallel (ThreadPoolExecutor), schedules deferred evaluation when `session_id` is present. |
| `base_generation_service.py` | `BaseGenerationService` — abstract base; the **Service Pattern** (load configs → create actors → run in parallel → save results). Per-extractor timeout `EXTRACTOR_TIMEOUT_SECONDS = 300`. |
| `operation_state_utils.py` | `OperationStateManager` — all `_operation_state` access (progress, concurrency locks, extractor/aggregator bookmarks, cluster fingerprints, cancellation). |
| `extractor_config_utils.py`, `extractor_interaction_utils.py` | Filter extractors by source / `allow_manual_trigger` / names; per-extractor stride + window + bookmark handling. |
| `deduplication_utils.py`, `service_utils.py`, `embedding_text.py` | LLM dedup helpers (used by `ProfileDeduplicator` + `PlaybookConsolidator`), message construction / JSON extraction / response logging, embedding text builders. |

## Generation Services

| Directory | Entry class | Key files |
|-----------|-------------|-----------|
| `profile/` | `ProfileGenerationService` | `profile_extractor.py`, `profile_deduplicator.py`, `profile_updater.py` |
| `playbook/` | `PlaybookGenerationService` | `playbook_extractor.py`, `playbook_consolidator.py`, `playbook_aggregator.py` (cluster-fingerprint change detection) — has its own [README](playbook/README.md) |
| `agent_success_evaluation/` | `AgentSuccessEvaluationService` | `agent_success_evaluator.py` (session-level), `delayed_group_evaluator.py` (`GroupEvaluationScheduler`, 10-min defer), `group_evaluation_runner.py`, `regen_jobs.py` |
| `reflection/` | `ReflectionService` | `reflection_extractor.py` — post-horizon reflection; runs **before** extraction so extractors read post-reflection state |

## Async Extraction

| Directory | Purpose |
|-----------|---------|
| `extraction/` | Resumable extraction agent: `resumable_agent.py`, `resume_scheduler.py`, `resume_worker.py`, `pending_tool_call_dispatch.py` (`ask_human`), `tools.py`, `plan.py`, `agent_run_records.py`, `invariants.py`. Long-horizon / tool-mediated extraction continues outside the request path. |

## Evaluation, Search & Integrations

| Path | Purpose |
|------|---------|
| `shadow_comparison/` | `ShadowComparisonJudge` — per-turn regular-vs-shadow verdicts written to a separate table (session-level shadow was retracted due to trajectory contamination). |
| `evaluation_overview/` | Dashboard rollups: `service.py`, `hero_state.py`, `distribution.py`, `group_aggregation.py`, `rule_attribution.py`, `shadow_aggregation.py`, `eval_sampler.py`. |
| `playbook_optimizer/` | Scenario-based playbook optimization: `optimizer.py`, `scheduler.py`, `rollout.py`, `judge.py`, `scenario_resolver.py`, `gepa_adapter.py`, `assistant_webhook.py`. |
| `braintrust/` | Braintrust export/sync: `service.py`, `client.py`, `_cron.py`, `_encryption.py`. |
| `pre_retrieval/` | `QueryReformulator` (`_query_reformulator.py`) + `_document_expander.py` — query rewrite & doc expansion for recall. |
| `unified_search_service.py` | `run_unified_search()` — two-phase parallel search across profiles / agent playbooks / user playbooks. |
| `retrieval/` | `relevance_floor.py` — result relevance thresholding. |

## Persistence & Config

| Path | Purpose |
|------|---------|
| `storage/` | `storage_base/` (`BaseStorage` split by domain) + `sqlite_storage/` + `retention*.py`. Access via `request_context.storage` only. |
| `configurator/` | `DefaultConfigurator` — loads YAML config and creates the storage backend. |

## Key Rules

- **NEVER instantiate services bypassing the Service Pattern** — extend `BaseGenerationService`; load YAML configs, create actors, run in parallel, save to storage.
- **NEVER import storage implementations directly** — use `request_context.storage` (`BaseStorage`).
- **ALWAYS use `LiteLLMClient`** for completions/embeddings and `request_context.prompt_manager.render_prompt(...)` for prompts — no hardcoded prompts, no direct OpenAI/Claude clients.
- **All `_operation_state` writes go through `OperationStateManager`** — don't touch the table directly (it backs locks, bookmarks, progress, and cancellation).
- **`tool_can_use` lives at root `Config`** — shared by playbook extraction and success evaluation, not per-service.
