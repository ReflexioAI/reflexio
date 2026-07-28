# Playbook Service
Description: Evidence-grounded playbook extraction, candidate review, aggregation, and consolidation pipeline

> Part of the [Reflexio Server](../../README.md). See also the [Prompt Bank](../../prompt/prompt_bank/README.md) for prompt template details.

## Main Entry Points

- **Service Orchestrator**: `service.py` - Manages playbook extraction lifecycle (regular, rerun, manual modes)
- **Playbook Extractor**: `components/extractor.py` - Extracts user playbooks from interactions via LLM
- **Candidate Reviewer**: `components/reviewer.py` - Accepts, narrowly revises, or rejects validated normal-extraction candidates before consolidation
- **Playbook Aggregator**: `components/aggregator.py` - Clusters similar user playbooks and generates aggregated insights
- **Playbook Consolidator**: `components/consolidator.py` - Reconciles reviewed candidates against existing storage with evidence-aware accounting and overlap guards

## Supporting Files

| File | Purpose |
|------|---------|
| `playbook_service_constants.py` | Prompt IDs for all playbook operations |
| `playbook_service_utils.py` | Request dataclasses, Pydantic output schemas, message construction utilities |
| `playbook_evidence.py` | Strict evidence validation, call-local reference checks, and persisted provenance helpers |
| `aggregation_prompt_processing.py` | Optional aggregation-boundary interfaces and helpers for prompt preprocessing, contextual prompt guidance, and output post-processing |

## Architecture

### Data Flow

```
Interactions
  -> PlaybookExtractor (per-extractor, extraction-only, parallel)
    -> PlaybookCandidateReviewer (normal strict-evidence candidates only)
      -> PlaybookConsolidator (consolidates reviewed vs existing DB playbooks)
        -> UserPlaybook (validated evidence + persisted provenance) -> Storage
        -> PlaybookAggregator (manual trigger)
          -> AgentPlaybook (aggregated insights) -> Storage
```

### Playbook Extraction (`components/extractor.py`)

Extends `BaseGenerationService` extractor pattern. Each extractor:
1. Checks stride_size threshold before running
2. Constructs messages from interactions (via `service_utils.py`)
3. Formats request boundaries and interactions with call-local references for strict normal extraction
4. Runs the LLM with the versioned extraction prompts
5. Validates required fields, verbatim evidence spans, atomic scope, and exact duplicates
6. Resolves validated local references to persisted request and interaction provenance

Malformed structured output receives one bounded repair attempt. An invalid
candidate is dropped independently so a valid sibling in the same response can
continue; an unresolved malformed response fails the extraction run.

**Tool Analysis**: Reads `tool_can_use` from root `Config` for tool usage analysis and resumable extraction decisions.

### Candidate Review (`components/reviewer.py`)

Strict normal candidates enter a fresh same-model review call before
consolidation. The reviewer receives the request-bounded chronology, validated
evidence units, artifact-availability context, and relevant existing playbooks.
Every candidate must be accounted for exactly once as `accept`, `revise`, or
`reject`. Revisions may narrow unsupported wording but cannot add evidence or
create a lesson that extraction missed. Reviewer output receives one bounded
repair attempt and otherwise fails closed. Expert and legacy extraction paths
do not use this reviewer.

### Playbook Aggregation (`components/aggregator.py`)

Triggered manually via `/api/run_playbook_aggregation`. Clusters user playbooks and generates consolidated insights.

**Key Methods**:
- `get_clusters(user_playbooks, config)` - HDBSCAN/Agglomerative clustering on embeddings
- `aggregate()` - Full aggregation pipeline with LLM-based consolidation

**Optional prompt processing**: deployments can register an
`AggregationPromptProcessor` via the `AGGREGATION_PROMPT_PROCESSOR` ServiceKey
(`register_service`). The aggregator applies the processor only at the
aggregation prompt boundary, carries an opaque per-cluster processing context,
injects extra prompt guidance only when preprocessing changed prompt input, and
post-processes generated outputs before storage or model-response logging.

**Change Log**: The legacy `playbook_aggregation_change_logs` table is retired (Track B, 2026-06-24) — the aggregator no longer writes it. The change-log view is reconstructed on demand from `lineage_event` via `reconstruct_playbook_aggregation_change_log` (`lib/_agent_playbook.py`): each run emits `op=aggregate` events (the "added" side) and `status_change→superseded` events from the supersede calls (the "removed" side), grouped by the run's `request_id`. Per-row `updated` pairing is not reconstructed (`updated_agent_playbooks=[]`, a tolerated parity delta).

**Clustering**: Embeds user playbooks -> HDBSCAN clustering -> falls back to Agglomerative if too few clusters

### Playbook Consolidation (`components/consolidator.py`)

Consolidates newly extracted playbooks against existing playbooks in the database via LLM semantic matching. For each NEW vs EXISTING pair the LLM returns one of five decision kinds, and the consolidator applies the chosen kind:

- `duplicate` — merge multiple rows into one, archiving members and emitting one merged row.
- `prefer_new` — archive the existing row and insert the new candidate unchanged.
- `prefer_existing` — drop the new candidate; the existing row wins.
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
| `StructuredExtractedPlaybookList` | Strict extraction candidates with rationale and evidence spans |
| `PlaybookGenerationRequest` | Request dataclass for playbook extraction |
| `PlaybookAggregatorRequest` | Request dataclass for playbook aggregation |

## See Also

- [Server README](../../README.md) -- FastAPI backend component overview
- [Prompt Bank README](../../prompt/prompt_bank/README.md) -- versioned prompt template system used by playbook prompts
