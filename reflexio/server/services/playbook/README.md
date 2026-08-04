# Playbook Service
Description: Evidence-grounded playbook extraction, candidate review, aggregation, and consolidation pipeline

> Part of the [Reflexio Server](../../README.md). See also the [Prompt Bank](../../prompt/prompt_bank/README.md) for prompt template details.

## Main Entry Points

- **Service Orchestrator**: `service.py` - Manages playbook extraction lifecycle (regular, rerun, manual modes)
- **Playbook Extractor**: `components/extractor.py` - Extracts user playbooks from interactions via LLM
- **Candidate Reviewer**: `components/reviewer.py` - Accepts, narrowly revises, or rejects validated normal-extraction candidates before consolidation
- **Persisted Review Service**: `review_service.py` - Re-reviews a bounded created-at selection and optionally commits each completed decision newest-first
- **Playbook Aggregator**: `components/aggregator.py` - Matches same-version cluster centroids, clusters residuals, and generates agent playbooks
- **Aggregation Scheduler**: `aggregation_scheduler.py` - Claims fenced per-version work and runs one bounded aggregation unit
- **Playbook Consolidator**: `components/consolidator.py` - Reconciles reviewed candidates against existing storage with evidence-aware accounting and overlap guards

## Supporting Files

| File | Purpose |
|------|---------|
| `playbook_service_constants.py` | Prompt IDs for all playbook operations |
| `playbook_service_utils.py` | Request dataclasses, Pydantic output schemas, message construction utilities |
| `playbook_evidence.py` | Strict evidence validation, call-local reference checks, and persisted provenance helpers |
| `review_service.py` | Time-window selection, persisted evidence reconstruction, reporting, and newest-first per-playbook apply |
| `aggregation_trigger.py` | Converts post-generation activity into an idempotent durable scheduling signal |
| `aggregation_scheduler.py` | Polling, fleet claim/lease handling, retries, and structured aggregation progress telemetry |
| `aggregation_prompt_processing.py` | Optional aggregation-boundary interfaces and helpers for prompt preprocessing, contextual prompt guidance, and output post-processing |

## Architecture

### Data Flow

```
Interactions
  -> PlaybookExtractor (per-extractor, extraction-only, parallel)
    -> PlaybookCandidateReviewer (normal strict-evidence candidates only)
      -> PlaybookConsolidator (consolidates reviewed vs existing DB playbooks)
        -> UserPlaybook (validated evidence + persisted provenance) -> Storage
        -> Durable aggregation signal (coalesces new work into the hourly window)
          -> PlaybookAggregationScheduler (bounded work; hourly idle minimum)
            -> PlaybookAggregator (incremental clustering)
              -> AgentPlaybook (aggregated insights) -> Storage
```

### Playbook Extraction (`components/extractor.py`)

Extends `BaseGenerationService` extractor pattern. Each extractor:
1. Checks stride_size threshold before running
2. Constructs messages from interactions (via `service_utils.py`)
3. Formats request boundaries and visible turns with call-local references for strict normal extraction
4. Runs the LLM with the versioned extraction prompts
5. Validates required fields, allowed turn references, atomic scope, and exact duplicates
6. Resolves validated local references to exact source text and persisted request/interaction provenance

Malformed structured output receives one bounded repair attempt. An invalid
candidate is dropped independently so a valid sibling in the same response can
continue; an unresolved malformed response fails the extraction run.

**Tool Analysis**: Reads `tool_can_use` from root `Config` for tool usage analysis and resumable extraction decisions.

### Candidate Review (`components/reviewer.py`)

Strict normal candidates enter a fresh same-model review call before
consolidation. The reviewer receives the request-bounded chronology, validated
referenced turns, artifact-availability context, and relevant existing playbooks.
Every candidate must be accounted for exactly once as `accept`, `revise`, or
`reject`. Revisions may narrow unsupported wording but cannot add evidence or
create a lesson that extraction missed. Reviewer output receives one bounded
repair attempt and otherwise fails closed. Expert and legacy extraction paths
do not use this reviewer.

### Persisted Review (`review_service.py`)

`POST /api/review_user_playbooks` selects the newest current user playbooks in
an inclusive creation-time window, capped by `top_k`. A row whose original
generation window or cited evidence can no longer be reconstructed yields a
`skip` decision and the run continues.

Manual review reconstructs context from the full interaction window persisted on
the row's finalized playbook-extraction run, plus any extra cited interactions
retained through consolidation. It never substitutes the current extractor
window or silently falls back to the smaller cited-evidence subset. Automatic
post-generation review continues to use the generation call's configured window.
Only the playbook row's cited interaction IDs become candidate evidence units;
the rest of the generation window is ancillary chronology. Evidence spans are
rebuilt from those exact stored interactions instead of requiring every cited
span to remain duplicated in the row's bounded combined `source_span`.

Report mode runs inline and returns `accept`, `edit`, `reject`, and `skip`
decisions without writes. Apply mode runs in the **background** (one LLM call
plus one transaction per playbook would otherwise risk a proxy timeout), so its
response carries only the `run_id`. It reviews one playbook at a time, prepares
any successor embedding, then commits that decision before reviewing the next
row. An edit goes through the shared `apply_playbook_edit` primitive — the
replacement is inserted as CURRENT and the incumbent atomically superseded, with
a `revise` lineage event under the run's `run_id`; rejects archive the incumbent
and accepts perform no write. A later failure stops the run without rolling back
earlier decisions.

### Playbook Aggregation (`components/aggregator.py`)

Normal generation durably schedules bounded incremental aggregation through
`aggregation_trigger.py`; `aggregation_scheduler.py` claims due work across
processes. A successful drained run waits at least
`REFLEXIO_AGGREGATION_MIN_INTERVAL_SECONDS` (default 3600) before the next
scheduled cycle; new durable signals preserve that due time so changes coalesce,
while a remaining backlog continues in bounded follow-up units. The manual
`/api/run_playbook_aggregation` route remains a fenced administrative full rerun.

Incremental work is isolated by `agent_version`:

1. Admit only the newest configured window of CURRENT, nonempty rows that have
   no durable disposition; rows that fall below its monotonic cutoff never re-enter.
2. Match embedded residuals against active centroids for that version only.
3. Group matches per cluster and make one LLM call with the current agent
   playbook plus at most the 100 newest members of that run's newly matched
   delta; the complete delta still receives durable membership.
4. Cluster only unmatched residuals: agglomerative below 50 rows, HDBSCAN at 50 or more.
5. Embed each generated agent playbook and use that embedding—not a mean of user
   embeddings—as the cluster centroid for future matches.
6. Commit replacement, lineage, membership, centroid swap, and supersession atomically.

Each refresh or rebuild cluster owns that atomic scope independently. If its
expected agent changes after generation, only that cluster rolls back and its
selected members return to residual work; unrelated generated clusters still commit.

`REFLEXIO_MAX_CLUSTERING_PLAYBOOKS` (default 20,000) is the maximum recent
unclustered discovery window for scheduled work. Older unclustered rows are
intentionally ignored rather than backfilled later. Invalidation repair is a
separate path whose generation input is capped at the 100 most recent retained
sources per affected cluster. New-cluster generation similarly uses at most 100
centroid-representative sources while retaining the complete discovered
membership. The setting is also the fail-before-mutation
total-input cap for an administrative full rerun; it no longer prevents an
organization with a larger corpus from making incremental progress. Retryable
LLM outcomes stay residual. A semantic-null update attaches the delta but keeps
the current agent playbook and centroid; semantic-null output for a new cluster
becomes a terminal no-op. Missing embeddings remain pending.

Creating a user playbook only arms intake discovery; it does not add a no-op
invalidation event. If the current agent playbook for an active cluster is
edited, rejected, archived, or deleted outside aggregation, that cluster is
retired atomically and its members return to residual discovery. This prevents
a stale centroid or rejected canonical rule from becoming the base of a later
incremental refresh.

The portable state contract is
`storage/storage_base/playbook/_aggregation.py`; SQLite implements it in
`storage/sqlite_storage/playbook/_aggregation.py`. Claims are lease-fenced, and
all multi-write effects must remain inside `storage.commit_scope()`.

**Optional prompt processing**: deployments can register an
`AggregationPromptProcessor` via the `AGGREGATION_PROMPT_PROCESSOR` ServiceKey
(`register_service`). The aggregator applies the processor only at the
aggregation prompt boundary, carries an opaque per-cluster processing context,
injects extra prompt guidance only when preprocessing changed prompt input, and
post-processes generated outputs before storage or model-response logging.

**Change Log**: The legacy `playbook_aggregation_change_logs` table is retired (Track B, 2026-06-24) — the aggregator no longer writes it. The change-log view is reconstructed on demand from `lineage_event` via `reconstruct_playbook_aggregation_change_log` (`lib/_agent_playbook.py`): each run emits `op=aggregate` events (the "added" side) and `status_change→superseded` events from the supersede calls (the "removed" side), grouped by the run's `request_id`. Per-row `updated` pairing is not reconstructed (`updated_agent_playbooks=[]`, a tolerated parity delta).

**Requirements / Problems to Avoid**:

- Never cluster, centroid-match, or attach rows across agent versions.
- Always keep an active cluster's centroid equal to its current agent playbook embedding.
- Rebuild lifecycle-invalidated clusters from the 100 most recent remaining
  CURRENT members (including a single member), never from the invalidated agent
  text, then restore every retained membership before superseding the prior agent.
- Never rediscover the full corpus on the scheduled path; use durable dispositions and bounded anti-join intake.
- Never hold `commit_scope()` while embedding, clustering, or calling the LLM.
- Do not clear pending state on partial failure, limiter deferral, or a lost lease.
- Drain lifecycle invalidations in fixed 100-event pages before model work, and
  retry failed repairs on the cluster clock rather than member clocks.

### Playbook Consolidation (`components/consolidator.py`)

Consolidates newly extracted playbooks against existing playbooks in the database via LLM semantic matching. For each NEW vs EXISTING pair the LLM returns one of four decision kinds, and the consolidator applies the chosen kind:

- `unify` — merge multiple rows into one, archiving members and emitting one merged row.
- `reject_new` — drop the new candidate because one or more existing rows already cover it.
- `differentiate` — archive the existing row and emit two refined rows (one per side) with sharpened triggers.
- `independent` — both rows are kept; the new candidate is inserted alongside the existing row.

The LLM must account for every new candidate exactly once and receives validated
evidence groups rather than raw database identifiers. Malformed, blank,
partition-invalid, or overlapping-survivor decisions receive one bounded repair
attempt. Reject-all is a valid result. If repair exhausts, the batch fails closed
rather than inserting unreviewed candidates. Deterministic guards preserve
source unions and reviewer revisions and prevent overlapping duplicate survivors.

Inline consolidation always runs during generation (the legacy `deduplicator` feature flag is retired). The enterprise repo additionally ships a **scheduled second-pass job** (`reflexio_ext/server/services/playbook_reconsolidation/`) that re-runs consolidation daily over already-persisted rows — user playbooks per `(user_id, agent_version)` and agent playbooks per `agent_version` — by rendering each duplicate group as NEW candidates against an empty EXISTING side via `_consolidation_decisions`, then tombstoning merged sources with `merge_records` lineage.

## Prompt IDs

| Constant | Prompt ID | Used By |
|----------|-----------|---------|
| `PLAYBOOK_SHOULD_GENERATE_PROMPT_ID` | `playbook_should_generate` | PlaybookExtractor |
| `PLAYBOOK_EXTRACTION_CONTEXT_PROMPT_ID` | `playbook_extraction_context` | PlaybookExtractor |
| `PLAYBOOK_EXTRACTION_PROMPT_ID` | `playbook_extraction_main` | PlaybookExtractor |
| `PLAYBOOK_CANDIDATE_REVIEW_PROMPT_ID` | `playbook_candidate_review` | PlaybookCandidateReviewer |
| `PLAYBOOK_AGGREGATION_PROMPT_ID` | `playbook_aggregation` | PlaybookAggregator |

## Key Output Schemas (in `playbook_service_utils.py`)

| Class | Purpose |
|-------|---------|
| `StructuredPlaybookContent` | Output from playbook extraction prompt |
| `StructuredReferencedExtractedPlaybookList` | Strict normal candidates with rationale and turn references |
| `StructuredExtractedPlaybookList` | Inactive expert strict contract with copied evidence spans |
| `PlaybookGenerationRequest` | Request dataclass for playbook extraction |
| `PlaybookAggregatorRequest` | Request dataclass for playbook aggregation |

## See Also

- [Server README](../../README.md) -- FastAPI backend component overview
- [Prompt Bank README](../../prompt/prompt_bank/README.md) -- versioned prompt template system used by playbook prompts
