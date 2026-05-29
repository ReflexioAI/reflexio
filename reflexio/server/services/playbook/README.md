# Playbook Service
Description: Playbook extraction, aggregation, and consolidation pipeline

> Part of the [Reflexio Server](../../README.md). See also the [Prompt Bank](../../prompt/prompt_bank/README.md) for prompt template details.

## Main Entry Points

- **Service Orchestrator**: `playbook_generation_service.py` - Manages playbook extraction lifecycle (regular, rerun, manual modes)
- **Playbook Extractor**: `playbook_extractor.py` - Extracts user playbooks from interactions via LLM
- **Playbook Aggregator**: `playbook_aggregator.py` - Clusters similar user playbooks and generates aggregated insights
- **Playbook Consolidator**: `playbook_consolidator.py` - Reconciles newly extracted playbooks against existing storage via LLM. Decides per pair to merge as duplicates, prefer the new entry, prefer the existing entry, differentiate (split with refined triggers), or keep both as independent

## Supporting Files

| File | Purpose |
|------|---------|
| `playbook_service_constants.py` | Prompt IDs for all playbook operations |
| `playbook_service_utils.py` | Request dataclasses, Pydantic output schemas, message construction utilities |

## Architecture

### Data Flow

```
Interactions
  -> PlaybookExtractor (per-extractor, extraction-only, parallel)
    -> PlaybookConsolidator (consolidates new vs existing DB playbooks)
      -> UserPlaybook (with optional blocking_issue) -> Storage
        -> PlaybookAggregator (manual trigger)
          -> AgentPlaybook (aggregated insights) -> Storage
```

### Playbook Extraction (`playbook_extractor.py`)

Extends `BaseGenerationService` extractor pattern. Each extractor:
1. Checks stride_size threshold before running
2. Constructs messages from interactions (via `service_utils.py`)
3. Runs LLM with `playbook_extraction_main` prompt
4. Parses `StructuredPlaybookContent` output (trigger, instruction, pitfall, blocking_issue)
5. Saves `UserPlaybook` to storage

**Tool Analysis**: Reads `tool_can_use` from root `Config` for tool usage analysis and blocking issue detection.

### Playbook Aggregation (`playbook_aggregator.py`)

Triggered manually via `/api/run_playbook_aggregation`. Clusters user playbooks and generates consolidated insights.

**Key Methods**:
- `get_clusters(user_playbooks, config)` - HDBSCAN/Agglomerative clustering on embeddings
- `aggregate()` - Full aggregation pipeline with LLM-based consolidation
- `_build_change_log()` - Builds `PlaybookAggregationChangeLog` with before/after snapshots (added/removed/updated playbooks)

**Change Log**: After each aggregation, saves a `PlaybookAggregationChangeLog` to storage. In full_archive mode, all old playbooks are "removed" and new ones "added". In incremental mode, maps old->new via fingerprints to detect updates. Saving is best-effort (failures logged, don't block aggregation).

**Clustering**: Embeds user playbooks -> HDBSCAN clustering -> falls back to Agglomerative if too few clusters

### Playbook Consolidation (`playbook_consolidator.py`)

Consolidates newly extracted playbooks against existing playbooks in the database via LLM semantic matching. For each NEW vs EXISTING pair the LLM returns one of five decision kinds, and the consolidator applies the chosen kind:

- `duplicate` — merge multiple rows into one, archiving members and emitting one merged row.
- `prefer_new` — archive the existing row and insert the new candidate unchanged.
- `prefer_existing` — drop the new candidate; the existing row wins.
- `differentiate` — archive the existing row and emit two refined rows (one per side) with sharpened triggers.
- `independent` — both rows are kept; the new candidate is inserted alongside the existing row.

A safety fallback inserts any new candidate that no decision consumed, so extracted data is never silently dropped on a malformed LLM response.

## Prompt IDs

| Constant | Prompt ID | Used By |
|----------|-----------|---------|
| `PLAYBOOK_EXTRACTION_SHOULD_GENERATE_PROMPT_ID` | `playbook_should_generate` | PlaybookExtractor |
| `PLAYBOOK_EXTRACTION_CONTEXT_PROMPT_ID` | `playbook_extraction_context` | PlaybookExtractor |
| `PLAYBOOK_EXTRACTION_PROMPT_ID` | `playbook_extraction_main` | PlaybookExtractor |
| `PLAYBOOK_GENERATION_PROMPT_ID` | `playbook_generation` | PlaybookAggregator |

## Key Output Schemas (in `playbook_service_utils.py`)

| Class | Purpose |
|-------|---------|
| `StructuredPlaybookContent` | Output from playbook extraction prompt |
| `PlaybookGenerationRequest` | Request dataclass for playbook extraction |
| `PlaybookAggregatorRequest` | Request dataclass for playbook aggregation |

## See Also

- [Server README](../../README.md) -- FastAPI backend component overview
- [Prompt Bank README](../../prompt/prompt_bank/README.md) -- versioned prompt template system used by playbook prompts
