# server/services
Description: Core business-logic layer — LLM orchestration, extraction, evaluation, optimization, search preparation, storage access, and long-running operation state.

> This is a directory-local index. For the full request flow, workflow tables (versioning, generation modes, cluster change detection), and the `OperationStateManager` use cases, see the parent [server README](../README.md#services).

**Service Boundary**: services own the LLM/extraction/evaluation/storage logic; API endpoints only authenticate, build `RequestContext`, and delegate into `Reflexio` or a focused service helper.

## LLM Pipeline Module Contract

LLM/pipeline modules use a shared vocabulary across OSS and enterprise:
`service.py` for request-path entry points, `runner.py` for manual/background
workflow entry points, `scheduler.py` for periodic/deferred execution,
`config.py` for module-owned config, `models.py` for module-local data shapes,
and `components/` for internal extractors, judges, proposers, aggregators,
evaluators, gates, and resolvers.

Files are optional. Do not create empty files only to satisfy the vocabulary.
Complete cutover migrations update consumers, tests, docs, and monkeypatch
strings before deleting old import paths in the same PR.

## Orchestration & Base Infrastructure

| File | Purpose |
|------|---------|
| `generation_service.py` | `GenerationService` — saves interactions, runs/defer-runs profile + playbook generation, schedules deferred evaluation when `session_id` is present, exposes `run_deferred_learning()` for the non-durable publish worker, and the durable `compute_deferred_learning()` → `persist_deferred_learning()` → `emit_deferred_learning_side_effects()` split (compute outside any scope; only persist + fence inside `commit_scope()`). |
| `publish_learning_worker.py` | Deferred post-persist publish learning worker — queues async publish learning after durable interaction writes and requeues under publish limiter pressure. |
| `base_generation_service.py` + `base_generation/` | `BaseGenerationService` stable import surface plus mixins for batch progress, config filtering, extraction lifecycle, should-run prechecks, status transitions, and usage billing. Per-extractor timeout `EXTRACTOR_TIMEOUT_SECONDS = 300`. |
| `operation_state_utils.py` | `OperationStateManager` — all `_operation_state` access (progress, concurrency locks, extractor/aggregator bookmarks, cluster fingerprints, cancellation). |
| `extractor_config_utils.py`, `extractor_interaction_utils.py` | Filter extractors by source / `allow_manual_trigger` / names; per-extractor stride + window + bookmark handling. |
| `deduplication_utils.py`, `service_utils.py`, `embedding_text.py` | LLM dedup helpers (used by `ProfileConsolidator` + `PlaybookConsolidator`), message construction / JSON extraction / response logging, embedding text builders. |

## Generation Services

| Directory | Entry class | Key files |
|-----------|-------------|-----------|
| `profile/` | `ProfileGenerationService` | `service.py`, `components/extractor.py`, `components/consolidator.py` |
| `playbook/` | `PlaybookGenerationService` | `components/extractor.py`, `components/consolidator.py`, `components/aggregator.py` (cluster-fingerprint change detection) — has its own [README](playbook/README.md) |
| `agent_success_evaluation/` | `AgentSuccessEvaluationService` | `service.py` (session-level service), `runner.py` (`run_group_evaluation`), `scheduler.py` (`GroupEvaluationScheduler`, 10-min defer), `sampling.py` (per-family session gates), `regen_jobs.py`, `components/evaluator.py` |

## Durable and Async Extraction

| Directory | Purpose |
|-----------|---------|
| `durable_learning/` | Durable `learning_jobs` scheduler + worker. Gated by `REFLEXIO_DURABLE_LEARNING_QUEUE`; discovers orgs with pending work, claims jobs with leases, then per job runs `compute_deferred_learning()` **outside** any writer transaction, applies `persist_deferred_learning()` + fenced `complete_learning_job()` inside a short `storage.commit_scope()`, and fires `emit_deferred_learning_side_effects()` post-commit — fenced completion guarantees exactly-once side effects. |
| `extraction/` | Shared async extraction runtime: `resumable_agent.py`, `resume_scheduler.py`, `resume_worker.py`, `pending_tool_call_dispatch.py` (`ask_human`), `prior_answer_search.py`, `agent_run_records.py`, and `outcome.py`. Long-horizon / tool-mediated extraction continues outside the request path. See [README](extraction/README.md). |

## Evaluation, Search & Integrations

| Path | Purpose |
|------|---------|
| `shadow_comparison/` | `ShadowComparisonJudge` (`judge.py`) plus pure outcome helpers (`outcome.py`) - per-turn regular-vs-shadow verdicts written to a separate table. Compact by design; see [README](shadow_comparison/README.md). |
| `evaluation_overview/` | Dashboard/read-side rollups: `service.py` entry point, `components/` aggregation helpers, and root `eval_sampler.py` shared with regenerate jobs. See [README](evaluation_overview/README.md). |
| `playbook_optimizer/` | Scenario-based playbook optimization: mature flat package with `optimizer.py`, `scheduler.py`, `rollout.py`, `judge.py`, `models.py`, `scenario_resolver.py`, `gepa_adapter.py`, and `assistant_webhook.py`. See [README](playbook_optimizer/README.md). |
| `braintrust/` | Braintrust export/sync: `service.py`, `client.py`, `_cron.py`, `_encryption.py`. |
| `lineage/` | Current-record resolution and tombstone GC: `resolve.py`, `gc_scheduler.py`. |
| `governance/` | Subject-reference and retention/barrier policy contracts: `config.py`, `service.py`, `subject_refs.py`. Storage and lineage code use these helpers to preserve governance constraints during writes, supersedes, and cleanup. |
| `pre_retrieval/` | `QueryReformulator` (`_query_reformulator.py`) + `DocumentExpander` (`_document_expander.py`) - query rewrite and doc expansion for recall. Compact by design; see [README](pre_retrieval/README.md). |
| `tagging/` | `TaggingService` (`service.py`) + deferred `tagging_scheduler.py` - post-generation profile/playbook tagging. Compact by design; see [README](tagging/README.md). |
| `unified_search_service.py` | `run_unified_search()` — two-phase parallel search across profiles / agent playbooks / user playbooks. |
| `retrieval/` | `relevance_floor.py` — result relevance thresholding. `temporal.py` — temporal post-processing driven by reformulation signals: query time windows → per-arm SQL filters, near-duplicate freshness collapse for current-value questions, timestamp ordering for latest-value questions. (Superseded/expired rows are already excluded by storage search SQL.) |

## Persistence & Config

| Path | Purpose |
|------|---------|
| `storage/` | `storage_base/` and `sqlite_storage/` keep legacy domain facades while focused subpackages own `profiles/`, `playbook/`, `agent_run/`, `governance/`, durable `learning_jobs`, and SQLite `base/` helpers; plus `governance_validation.py` and `retention*.py`. Access via `request_context.storage` only. |
| `configurator/` | `DefaultConfigurator` — loads YAML config and creates the storage backend. |

## Key Rules

- **NEVER instantiate services bypassing the Service Pattern** — extend `BaseGenerationService`; load YAML configs, create actors, run in parallel, save to storage.
- **NEVER import storage implementations directly** — use `request_context.storage` (`BaseStorage`).
- **ALWAYS use `LiteLLMClient`** for completions/embeddings and `request_context.prompt_manager.render_prompt(...)` for prompts — no hardcoded prompts, no direct OpenAI/Claude clients.
- **All `_operation_state` writes go through `OperationStateManager`** — don't touch the table directly (it backs locks, bookmarks, progress, and cancellation).
- **All durable queue claims go through `LearningJobStoreABC`** — do not update `learning_jobs` directly; completion is fenced by `claim_token` and must roll back the surrounding `commit_scope` when superseded.
- **`tool_can_use` lives at root `Config`** — shared by playbook extraction and success evaluation, not per-service.
- **Preserve governance subject refs/barriers** — route validation through `services/governance/` and `storage/governance_validation.py`; do not bypass retention or subject-write checks in storage implementations.
