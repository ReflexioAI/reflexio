# /reflexio/server/services
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

## Main Entry Points


### Orchestration & Base Infrastructure


| File | Purpose |
|------|---------|
| `generation_service.py` | `GenerationService` — admits interactions through the durable stream engine, schedules deferred evaluation, and coordinates manual generation under the shared user lease. |
| `search_metering_worker.py` | Bounded process-local search-metering queue. Four daemon workers emit `search_request` and `learning_applied` after the response path, with fail-open drops, a five-second shutdown drain, and trace-linked `search.metering` transactions; persistence requires a registered usage recorder. |
| `base_generation_service.py` + `base_generation/` | `BaseGenerationService` stable import surface plus mixins for batch progress, config filtering, extraction lifecycle, should-run prechecks, status transitions, and usage billing. Per-extractor timeout `EXTRACTOR_TIMEOUT_SECONDS = 300`. |
| `operation_state_utils.py` | `OperationStateManager` — all `_operation_state` access (progress, concurrency locks, extractor/aggregator bookmarks, cluster fingerprints, cancellation). |
| `extractor_config_utils.py`, `extractor_interaction_utils.py` | Filter extractors by source / `allow_manual_trigger` / names; per-extractor stride + window + bookmark handling. |
| `deduplication_utils.py`, `service_utils.py`, `embedding_text.py` | LLM dedup helpers (used by `ProfileConsolidator` + `PlaybookConsolidator`), message construction / JSON extraction / response logging, embedding text builders. |

### Generation Services


| Directory | Entry class | Key files |
|-----------|-------------|-----------|
| `profile/` | `ProfileGenerationService` | `service.py`, `components/extractor.py`, `components/consolidator.py` |
| `playbook/` | `PlaybookGenerationService` | `aggregation_trigger.py` durably signals work; `aggregation_scheduler.py` claims fenced bounded units; `components/aggregator.py` performs same-version centroid matching and residual clustering — see [README](playbook/README.md) |
| `agent_success_evaluation/` | `AgentSuccessEvaluationService` | `service.py` (session-level service), `runner.py` (`run_group_evaluation`), `scheduler.py` (`GroupEvaluationScheduler`, 10-min defer), `regen_jobs.py`, `components/evaluator.py` |

### Durable and Async Extraction


| Directory | Purpose |
|-----------|---------|
| `durable_learning/` | `admission.py` commits incoming streams; `scheduler.py` / `worker.py` run bounded user turns; `window_executor.py` / `window_codec.py` compute frozen windows; `profile_package_executor.py` explicitly qualifies source-package profile publication without changing worker selection; `user_lease.py` shares ownership with manual/resumed generation; `waiting.py` bounds HTTP waits; `local.py` recovers standalone-library work. |
| `extraction/` | Shared async extraction runtime: `resumable_agent.py`, `resume_scheduler.py`, `resume_worker.py`, `pending_tool_call_dispatch.py` (`ask_human`), `prior_answer_search.py`, `agent_run_records.py`, and `outcome.py`. Long-horizon / tool-mediated extraction continues outside the request path. See [README](extraction/README.md). |
| `input_compression.py`, `input_compression_llm.py` | Shared source-backed compression with extraction/session policies, exact original excerpts, bounded hierarchical selection, provider-free admission planning and guarded provider dispatch. Callable qualification adapters; importing them does not switch worker protocols. |
| `playbook/components/context_compression.py`, `extraction/reconciliation_journal.py` | Consolidator integration: authorized excerpt rendering, complete consumer request sizing, and immutable selections with atomic attribution to existing durable request reservations. Uses the shared compressor; owns no executor or publication path. |

### Evaluation, Search & Integrations


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
| `search_observer.py` | Optional best-effort observer of final successful unified searches; carries correlation and learning IDs without response content. Uses the shared bounded callback executor with drop-oldest overflow and explicit tenant scope rebinding; observer latency/failure does not block retrieval. |
| `search_exposure.py` | Optional synchronous recorder contract for final user-playbook result sets. Enterprise capability registration installs the recorder, which authenticated unified/direct search invokes before metering/response. A recorder failure does NOT fail the search: it is logged, reported as `search_exposure.recorder_failed`, and returned as `RECORDER_FAILED`. An uncorrelated batch is recorded and flagged `integrity_state=incomplete`, not refused. Shared `create_app()` has no default recorder; other constructions persist exposures only when they register one. Direct search omits empty batches. Batch-size and scoped-owner validation happens before recorder lookup and before unified-response acceptance; invalid results are not contained as recorder failures. Unscoped searches may include multiple owners. |
| `retrieval/` | `relevance_floor.py` — result relevance thresholding. `temporal.py` — temporal post-processing driven by reformulation signals: query time windows → per-arm SQL filters, near-duplicate freshness collapse for current-value questions, timestamp ordering for latest-value questions. `user_context_guard.py` — high-precision detection of explicit personalization opt-outs, including Simplified and Traditional Chinese, before user-context retrieval. (Superseded/expired rows are already excluded by storage search SQL.) |

### Persistence & Config


| Path | Purpose |
|------|---------|
| `storage/` | `storage_base/` and `sqlite_storage/` keep legacy domain facades while focused subpackages own `profiles/`, `playbook/`, `agent_run/`, `governance`, durable extraction streams, and SQLite `base/` helpers. SQLite hybrid search preserves Porter FTS for ASCII and adds bounded Unicode substring candidates only when a query contains at least one non-ASCII alphanumeric character, including mixed-script queries; emoji-only and punctuation-only queries are ineligible. `storage_base/playbook/_aggregation.py` defines fenced aggregation state; SQLite implements it in the matching playbook package. Access via `request_context.storage` only. |
| `configurator/` | `DefaultConfigurator` — loads YAML config and creates the storage backend. |

## Purpose


Translate admitted interactions into profiles/playbooks, evaluate sessions, and retrieve learning through shared storage and configuration contracts.

For current/latest search queries, temporal ordering consumes the full admitted
candidate pool after relevance filtering, including configured recency/floor
pools and session backfill. Apply the request's final `top_k` after that ordering;
the default temporal working-pool minimum is not an intermediate result ceiling.

## Architecture Pattern


Generation uses configured actors and fenced persistence; deferred workers rebuild request context rather than retaining request objects. Search composes reformulation, parallel entity retrieval, filtering, optional exposure recording, and asynchronous metering. Keep service-owned implementations behind their public entry points.

## Requirements / Problems to Avoid


- **Generation pipelines extend `BaseGenerationService`** — load configs, create actors, run in parallel, and save through storage. Focused search/read-side services retain their own entry points.
- **NEVER import storage implementations directly** — use `request_context.storage` (`BaseStorage`).
- **ALWAYS use `LiteLLMClient`** for completions/embeddings and `request_context.prompt_manager.render_prompt(...)` for prompts — no hardcoded prompts, no direct OpenAI/Claude clients.
- **All `_operation_state` writes go through `OperationStateManager`** — don't touch the table directly (it backs locks, bookmarks, progress, and cancellation).
- **SQLite agent-run writes share `commit_scope()`** — run state, pending calls,
  dependencies and finalization receipts join an enclosing transaction. Standalone
  mutations use the same owner for atomic commit/rollback; nested and no-op calls
  cannot publish earlier writes. Keep model calls outside these transactions.
- **Automatic extraction uses `storage_base/_extraction_stream.py`** — admission locks the user work row first; window output writes, cursor advancement and receipts share one fenced transaction.
- **Aggregation state is per agent version** — scheduled intake, centroid matching, cluster membership, invalidation, and agent-playbook generation must never cross versions; keep LLM and clustering work outside `commit_scope()`.
- **`tool_can_use` lives at root `Config`** — shared by playbook extraction and success evaluation, not per-service.
- **Preserve governance subject refs/barriers** — route validation through `services/governance/` and `storage/governance_validation.py`; do not bypass retention or subject-write checks in storage implementations.


## Source compression planning

`input_compression.plan_compression_input(...)` prepares original sources and
exact section geometry without dispatching a model or reading/writing cached
selections. Its transient `CompressionPlan` exposes section digests, complete
request token estimates (including the response schema and 20% margin), and a
contract witness. The existing `compress_input(...)` interface prepares and
executes through this same implementation; prompts, node cache keys and the
`source-compression-v6` selection policy remain unchanged.

Callers may compare/persist the witness before admitting a durable parent, then
call `plan.execute(complete=..., load=..., save=...)`. Rebuild the plan from
original authorized sources on resume and require the stored witness to match.
Only the digest witness is a storage value: do not serialize the transient plan,
which retains private source text and pure output-rendering callbacks. Keep those
callbacks fixed and name their behavior in `output_contract`. The witness binds
scope, source provenance, policy, model, schema, all budgets and ordered sections.

This low-level planner estimates the compressor's complete requests; native model
capacity/output reservation checks and final consumer prompt fit remain the
caller's responsibility. It does not grant source authorization, reserve durable
provider attempts, prove semantic retention, or enable a worker protocol. Every
actual dispatch, including repairs and retries, still needs the caller's shared
durable quota guard. Passthrough plans have zero selection sections/calls.

For native model admission, `input_compression_llm.plan_with_client(...)` builds
this core plan once, then checks every initial leaf through the actual LiteLLM
request builder and existing complete-request capacity guard. This includes
provider schema instructions, direct-answer mode and the configured output
reservation. A custom endpoint requires its context declaration. These checks
also run for cache-only compression work; passthrough has no model leaf to check.
The planner rejects an oversized native leaf without repacking or dispatching.

The returned `ClientCompressionPlan` exposes `.compression` for core geometry,
its stronger `.witness`, and `.execute(before_call=..., load=..., save=...)`.
Persist/compare that native witness in the durable caller: it additionally binds
actual generation policy, the pinned transport ladder, capacity declarations,
initial augmented message digests and measured request sizes. Existing v6 node
keys remain compatible; they do not replace this stronger parent binding.
Execution checks model/policy and initial-message drift before reading cached
selections. Reductions and corrections still undergo native checks on every
actual dispatch. Preparation never reads checkpoints, calls the lease/quota hook,
or adds usage. Execution forwards observed usage once to the parent capture;
elapsed time includes native preparation and execution, excluding the caller's
wait between them. `CompressionPlan.check_requests(checker)` is the low-level
seam for these fresh initial messages; a checker cannot change saved geometry.
These library interfaces do not admit new durable work or enable compression in
production workers.

The admitted durable consolidator caller uses these interfaces through
`playbook/components/context_compression.py`. It keeps small fitting input
presentations unchanged and checks every actual consumer request, including
tool replies and revisions. Oversized evidence is selected under the same
24-request owner as drafting and auditing, reserving at least two downstream
calls. Existing durable windows store the plan witness and node receipts in
`reconciliation_context`; parent attempt reservation and consolidation attribution
commit atomically. Cached selections are revalidated against the original
authorized sources on resume. Legacy admitted conversations keep their original
protocol. Opaque CLI providers retain their existing audited adapter without
native capacity claims. Coordinator-free manual consolidation retains its existing
execution path. Unchanged packets retain the original citation sources, including
expert-interaction field boundaries. Selected compressed fields remain valid
citation evidence even when the joined content field is omitted.
This code path needs separate evidence-preservation and total-duration
qualification before promotion; the shared library's qualification alone does
not qualify the consolidator.
