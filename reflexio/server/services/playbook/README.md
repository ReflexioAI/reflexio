# /reflexio/server/services/playbook
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
- **Optional Input Observer**: `input_capture.py` - Lazy, detached pre-decision snapshots for an explicitly installed research provider; absent providers add no serialization or model calls, and observer failures leave reconciliation running
- **Rule Evidence Preview**: `rule_evidence.run_rule_evidence` - Standalone four-dimensional diagnosis and one exact-clause repair, using a caller-supplied LLM invocation; returns a proposal or abstention and has no storage/publication authority

## Supporting Files


| File | Purpose |
|------|---------|
| `playbook_service_constants.py` | Prompt IDs for all playbook operations |
| `playbook_service_utils.py` | Request dataclasses, Pydantic output schemas, message construction utilities |
| `playbook_evidence.py` | Strict evidence validation, call-local reference checks, and persisted provenance helpers |
| `rule_evidence/` | Owned/version-matched source schemas, packaged judge/tuner prompts, compact diagnostics and indexed exact-clause repair; see [usage](rule_evidence/README.md) |
| `components/overflow_consolidation.py` | Bounded metadata proposals, exact historical quotations, and semantic verification before overflow consolidation applies original rows |
| `components/overflow_evidence.py` | Canonical historical row fingerprints checked under final commit locks |
| `source_quoted_overflow.py` | Multi-chunk recovery: original User-message copying, bounded publication, immutable source-policy receipt and cached-plan validation |
| `review_window.py` | Shared fail-closed reconstruction of persisted review chronology for automatic and manual review |
| `review_service.py` | Time-window selection, persisted evidence reconstruction, reporting, and newest-first per-playbook apply |
| `aggregation_trigger.py` | Converts post-generation activity into an idempotent durable scheduling signal |
| `aggregation_scheduler.py` | Polling, fleet claim/lease handling, retries, and structured aggregation progress telemetry |
| `aggregation_prompt_processing.py` | Optional aggregation-boundary interfaces and helpers for prompt preprocessing, contextual prompt guidance, and output post-processing |

## Purpose


Turn evidenced corrections into user playbooks, review/consolidate them, and aggregate same-version user learning into agent playbooks while retaining provenance and lifecycle invariants.

## Architecture Pattern


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
do not use this reviewer. New user-playbook generation requires a non-empty
`user_id`. A legacy durable run whose stored binding is null may resume only
when every persisted source interaction and request proves the same owner;
missing or mixed-owner evidence fails closed before review or persistence.

### Persisted Review (`review_service.py`)


`POST /api/review_user_playbooks` selects the newest current user playbooks in
an inclusive creation-time window, capped by `top_k`. A row whose original
generation window or cited evidence can no longer be reconstructed yields a
`skip` decision and the run continues.

Manual review reconstructs context from the full interaction window persisted on
the row's finalized playbook-extraction run, plus any extra cited interactions
retained through consolidation. Automatic review uses the exact
`source_interaction_ids` on the extraction agent run, including when finalization
is retried later. Neither path re-runs the sliding last-K query or silently falls
back to a smaller evidence subset, so interactions published after extraction
cannot change the review chronology.
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
processes. A successful drained run waits a code-owned 3600 seconds before the next
scheduled cycle; new durable signals preserve that due time so changes coalesce,
while a remaining backlog continues in bounded follow-up units. The manual
`/api/run_playbook_aggregation` route remains a fenced administrative full rerun.

The scheduler processes organizations sequentially. An exception escaping an
organization's execution defers that organization for five minutes using
monotonic time; later organizations continue after the failing call returns.
Failed repair attempts also respect the five-minute repair interval. Logs
identify the organization and failing stage, and provider enumeration failures
remain separate tick-level errors. Durable retries and lease fencing are unchanged.

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
`../storage/storage_base/playbook/_aggregation.py`; SQLite implements it in
`../storage/sqlite_storage/playbook/_aggregation.py`. Claims are lease-fenced, and
all multi-write effects must remain inside `storage.commit_scope()`.

**Optional prompt processing**: deployments can register an
`AggregationPromptProcessor` via the `AGGREGATION_PROMPT_PROCESSOR` ServiceKey
(`register_service`). The aggregator applies the processor only at the
aggregation prompt boundary, carries an opaque per-cluster processing context,
injects extra prompt guidance only when preprocessing changed prompt input, and
post-processes generated outputs before storage or model-response logging.

**Change Log**: The legacy `playbook_aggregation_change_logs` table is retired (Track B, 2026-06-24) — the aggregator no longer writes it. The change-log view is reconstructed on demand from `lineage_event` via `reconstruct_playbook_aggregation_change_log` (`../../../lib/_agent_playbook.py`): each run emits `op=aggregate` events (the "added" side) and `status_change→superseded` events from the supersede calls (the "removed" side), grouped by the run's `request_id`. Per-row `updated` pairing is not reconstructed (`updated_agent_playbooks=[]`, a tolerated parity delta).

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

When extraction overflow admission is enabled, the grounding reviewer uses each
candidate's original quotations and leaves historical redundancy to consolidation.
Consolidation proposes actions from per-candidate retrieval metadata, scans the
complete selected historical spans into bounded exact quotations, then verifies
the combined decisions before applying the original storage models. Equal source
bytes share source geometry; row-specific semantic receipts, fingerprints and
original source IDs remain distinct. A proposal that verification does not
clear (unsafe or undetermined verdict, undetermined coverage, no verified
quotation) completes with the no-merge plan: every NEW is inserted unchanged and
no historical row is touched. Final row locks protect scope, status, content and
evidence changes.
A historical row's evidence comes from one helper, `historical_evidence`. Its
`source_span` is used when non-blank. Legacy rows created before extraction
stored the verbatim quote have none; their stored rule text (`Trigger`,
`Content`, `Rationale`, `Blocking issue`, in that order, blank fields omitted)
is scanned and quoted instead, under the same exact-offset rules. Scan and
verification payloads mark such a row `evidence_kind: "stored_rule"` with a fixed
note, and its partition plan is keyed by a `stored_rule:`-prefixed digest that
never collides with a span digest. The guarantee is weaker: for stored-rule
evidence, verification proves no stored constraint is dropped, not fidelity to
the original conversation. Rows with a span keep their exact scan and
verification payloads, step keys and partition digests. A row with neither a
span nor any rule text completes with the no-merge plan.
Missing historical scan receipts now use lossless bounded original snippets:
the model selects displayed IDs, and the server copies exact text and absolute
character ranges into the existing quotation receipt format. Every primary and
neighbor character remains visible, author role stays unknown, and earlier
quotations cannot be removed. Physical quoted coverage avoids saving repeated
neighbor evidence. Conservative snippet costs guide selection within the unchanged
2048-token carrier limit; insufficient coverage or capacity blocks publication.
Valid v1 quotation receipts remain reusable without provider calls; missing work
and new receipts use a separate v2 key bound to the complete presentation and schema.
Overflow review and final verification preserve the local scope of code examples
and test values, including conditional permission checks and admission versus
execution. Incomplete upstream excerpts cannot establish omitted prerequisites
or universal requirements. Their trusted policy is separate from source data;
normal non-overflow reviewer prompts remain unchanged.

Inline consolidation always runs during generation (the legacy `deduplicator` feature flag is retired). The enterprise repo additionally ships a **scheduled second-pass job** (`reflexio_ext/server/services/playbook_reconsolidation/`) that re-runs consolidation daily over already-persisted rows — user playbooks per `(user_id, agent_version)` and agent playbooks per `agent_version` — by rendering each duplicate group as NEW candidates against an empty EXISTING side via `_consolidation_decisions`, then tombstoning merged sources with `merge_records` lineage.

## Key Endpoints / Commands / Contracts


- **Review**: `POST /api/review_user_playbooks`; report is inline, apply is background and returns a run ID.
- **Aggregate**: `POST /api/run_playbook_aggregation`; administrative reruns retain lease fencing and the input cap.
- **Lifecycle**: see the [complete route map](../../README.md#api-endpoints) for create/update/status/upgrade/downgrade/delete and change-log reads.

### Prompt IDs


| Constant | Prompt ID | Used By |
|----------|-----------|---------|
| `PLAYBOOK_SHOULD_GENERATE_PROMPT_ID` | `playbook_should_generate` | PlaybookExtractor |
| `PLAYBOOK_EXTRACTION_CONTEXT_PROMPT_ID` | `playbook_extraction_context` | PlaybookExtractor |
| `PLAYBOOK_EXTRACTION_PROMPT_ID` | `playbook_extraction_main` | PlaybookExtractor |
| `PLAYBOOK_CANDIDATE_REVIEW_PROMPT_ID` | `playbook_candidate_review` | PlaybookCandidateReviewer |
| `PLAYBOOK_AGGREGATION_PROMPT_ID` | `playbook_aggregation` | PlaybookAggregator |

Reviewer v1.14.0 and v1.15.0 are inactive evidence-first experiments. Their internal
`EvidenceFirstReviewOutput` carries bounded exact supporting excerpts, validated
against the candidate's final retained source units. Policy is separated from
untrusted input data. Excerpts are available to private evaluation artifacts,
but are stripped before applying decisions and never replace saved rationale.
Exact quoting proves provenance, not that the surviving rule is justified. The
active v1.3 response contract is unchanged.

### Key Output Schemas (in `playbook_service_utils.py`)


| Class | Purpose |
|-------|---------|
| `StructuredPlaybookContent` | Output from playbook extraction prompt |
| `StructuredReferencedExtractedPlaybookList` | Strict normal candidates with rationale and turn references |
| `StructuredExtractedPlaybookList` | Inactive expert strict contract with copied evidence spans |
| `PlaybookGenerationRequest` | Request dataclass for playbook extraction |
| `PlaybookAggregatorRequest` | Request dataclass for playbook aggregation |

## Requirements / Problems to Avoid


- **Fail closed on unreconstructable or invalid evidence**; manual review skips an unreconstructable row instead of inventing chronology.
- **Preserve per-version aggregation, bounded discovery, and transaction rules** listed in the aggregation section above.
- **Keep normal candidate review separate from expert/legacy paths**; a revision cannot invent evidence or a missing lesson.

## See Also


- [Server README](../../README.md) -- FastAPI backend component overview
- [Prompt Bank README](../../prompt/prompt_bank/README.md) -- versioned prompt template system used by playbook prompts


## Durable explicit aggregation

`aggregation_operations.py` executes persisted full-archive requests before the
scheduler's automatic work. The POST/GET API and SDK return
`PlaybookAggregationOperation` receipts from the shared schema module. Requests
are idempotent within the organization/project and serialized with automatic and
synchronous manual aggregation through the existing organization lease.

The aggregator commits generated outputs, lineage, bookkeeping and the successful
operation receipt in the same fenced storage transaction. Zero-output success is
valid; transient generation failure retries (up to five attempts) rather than
being mistaken for zero output. Restart recovery discovers persisted queued,
running and retrying operations; completed operations are never resubmitted.
Optional optimization/tagging is outside the successful receipt contract.

Receipt support is advertised separately as
`supports_durable_playbook_aggregation_operations` (false on the portable base,
true on SQLite when sqlite-vec loaded). Fenced full reruns write vector-backed
cluster state, so a missing extension blocks new operation admission. Existing
receipts remain readable when execution support becomes unavailable.
Incremental aggregation support alone does not authorize these
APIs. Disabling automatic incremental work does not hide existing receipts or
prevent explicit full reruns. Heartbeats, claim notifications, retry reporting,
and finalization retain the scheduler's organization/project scope. Incomplete
aggregation results never become successful receipts. Erasing source user
playbooks cancels active receipts on current-row deletion or content purge, so
a stale output transaction cannot commit derived output after that erasure.
Retiring or moving a current input invalidates the captured claim through the
state-version fence; the operation can recover with a fresh claim. Removing an
already excluded historical tombstone does not cancel unrelated work. These
fences use database state rather than application-clock comparisons.
Deleting an already purged empty source skeleton also leaves fresh operations
unchanged, since those sources are excluded from their input snapshots.
The still-owned worker records its retry and releases its lease after input
state changes, preserving the newer scheduling state instead of waiting for
lease expiry. A replaced worker cannot change the new worker's receipt or lease.
Synchronous full reruns use the same
lease when either incremental or durable-operation support is enabled. Explicit
runs with missing source embeddings retry rather than declaring success for a
partial corpus, including when the corpus is below the count threshold.
Readiness uses the bounded pre-compute snapshot before that threshold decision;
the same snapshot is reused for generation. Empty or whitespace-only source
bodies are excluded before the snapshot limit, matching incremental intake, so
erased lineage skeletons cannot block later work or displace surviving inputs.
Mock-LLM mode continues to support keyless trigger clustering.

SQLite replaces its erasure-fencing triggers in one immediate transaction, so
concurrent connections cannot erase inputs between dropping and recreating a
trigger. Failed replacement rolls back the previous trigger definitions.
For local OSS apps whose organization is resolved from each request, durable
admission and active receipt polling register that organization's local worker;
polling can resume persisted work after process registration is lost. Terminal
receipt reads do not start workers. Platform and self-host deployments retain
their enterprise scheduler ownership.

Historical source selection uses the same one-correction capacity guard as original
selection. An oversized selection can reduce only its new snippets once; retained
quotations and the 2,048-token carrier ceiling remain fixed. A second over-capacity
result stops with a typed hold before further fallback dispatch or a new receipt.

Overflow review also rejects inferred defaults and workflow details absent from
the source: inspecting or searching a file cannot establish execution, and a
selective exclusion cannot establish a default that excludes every other case.
Conditional rules must retain their supported applicability and exceptions.

Multi-chunk overflow playbooks use source-quoted recovery. The protected pass
selects original instruction/correction references; the server copies the complete
bounded original `content` message only when its transcript role is User. A
reference selecting “deploy” cannot remove “Do not” elsewhere in that message.
Assistant, tool and historical unknown-author text cannot supply new instructions.
Transcript role User does not establish human authorship or deployment permission.

This path bypasses prose extraction, candidate review and generative consolidation.
Content, source span and trigger are exact copies; the trigger gives the existing
search path original text to embed. Rationale remains empty and notes contain
neutral source provenance. Embedding preparation explicitly disables generated
document expansion while retaining embeddings. The serialized publication
(including all copies and metadata) is capped at 8,192 tokens. Oversized complete messages
are omitted, not clipped; later smaller messages remain eligible. Repeated
selections of one source copy it once. There is no semantic deduplication or
historical retirement, so duplicate lessons and conservative omissions are possible.

A versioned `source_quoted_publication` receipt binds source, scope, selection and
row meaning before the existing fenced write plan. It separately seals vectors
at storage float32 precision. Validated vectors are checkpointed with their digest and reused after an
interruption before outcome caching, without another embedding call.
Initial row IDs must be zero, and active/waiting child continuations cannot be bypassed. Cached multi-chunk protocol
plans without that receipt, changed text/vectors, and retirement/merge proposals are
refused. Existing drafts and attempt journals are not rewritten or reset. Copied
rows receive their creation timestamp from the trusted clock at fenced persistence,
so cached timestamps cannot alter chronology. Copied
rules are excluded from automatic optimizer admission and GEPA publication;
later normal consolidation excludes them from retrieval and prefetched history,
and persistence rejects cached plans that would overwrite or retire them.
Ordinary human-reviewed agent aggregation remains a separate derived workflow.
An unavailable live embedding before sealing remains retryable; malformed cached
plan structures are held for investigation.
Operational statistics and model attribution are restored from trusted projection
values before commit, so cached metadata cannot skip the derived scheduler handoff.

One-chunk protocol work retains accept-unchanged/discard-revision review, and
ordinary non-overflow review still supports revisions. Fully scanned windows may
complete with zero new rows; originals and existing learnings remain intact.

Automatic consolidation requests an internal, default-off source-quoted exclusion
before storage candidate limits (SQLite and enterprise PostgreSQL retrieval).
Ordinary user search still returns these original instructions. Source-projection
create lineage names every original interaction ID; cached plans missing that
lineage fail closed without rewriting saved events or resetting the attempt budget.
Structurally valid checkpoints with invalid source references or quotations also
report a content-free configuration failure before publication. They retain their
journal and use the existing configuration retry backoff; this is not a quota
refund or a guarantee of a permanent hold.


The exact JSON `notes.policy` marker is reserved for the server's validated
source projection. Ordinary model-written notes that claim it are preserved
verbatim inside a prose envelope at extraction, ordinary candidate resolution
(including saved parent outputs), and ordinary cached-plan publication. This
normalization precedes candidate classification and never runs on validated
source-projection plans or persisted-row reads. No public notes-write endpoint
is exposed. Reserved markers in saved ordinary parent output fail with the typed
configuration error before proposal re-keying; their existing quota/receipts are
retained. Cached final ordinary write plans can normalize notes without model
work, vector changes, or regenerating proposals. Existing stored accidental marker collisions cannot be distinguished
retroactively from genuine source rows by their notes alone; this prospective
reservation does not backfill or rewrite historical rows.

### Extraction citation contract

`playbook_evidence.PlaybookExtractionContract` selects the output schema and
validator and supplies the citation instruction consumed by ordinary/expert message
constructors. Fresh, overflow-chunk and resumed calls pass that selected instruction;
an adopted stored schema determines its own instruction. Generic same-model repair
inherits the same request contract and retains its existing attempt bound.

Ordinary unprepared inputs use visible `T#` IDs. Prepared inputs use exact
`i<id>:<field>:<start>:<end>` evidence IDs; these cannot be replaced by a coarse turn
alias when a turn contains several fields or excerpts. Strict quoted-evidence
contracts instead require the provided `turn_ref` plus an exact original quotation.
Legacy schema names/shapes and expert prompt activation remain unchanged. Preferences
cite only User-originated evidence; corrections may also cite prior agent behavior.
No role, ownership, source-span or carrier-capacity validation is relaxed.

Ordinary prompt patches 4.6.1/1.5.1 use representation-neutral citation wording;
shared schema descriptions no longer assume turn-only references. Historical prompt
content stays unchanged; only superseded activation flags are disabled. Explicit
historical/custom prompt pins retain their own
wording; default-template updates do not silently rewrite those pins. Request tests
inspect both the complete serialized schema and messages on initial/repair calls,
not just presence of a prepared-source reminder. Scripted responses establish
contract mechanics, not live-model accuracy or compressor superiority.
