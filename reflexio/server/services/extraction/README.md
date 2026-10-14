# /reflexio/server/services/extraction
Description: Shared async extraction runtime for profile and playbook pipelines.

## Purpose


This package is intentionally domain-neutral. Profile and playbook modules own
their prompts, schemas, extractor semantics, and storage decisions; this package
owns the reusable runtime needed when extraction pauses, waits for more
information, and resumes outside the request path.

## Main Entry Points


| File | Responsibility |
|------|----------------|
| `resumable_agent.py` | Runs extraction agents, records agent usage, injects resolved pending-tool context, and exposes pending tool-call helpers. |
| `pending_tool_call_dispatch.py` | Implements the `ask_human` and pending-info tool dispatch flow. |
| `prior_answer_search.py` | Finds and formats previous human answers for async extraction context. |
| `agent_run_records.py` | Builds durable extraction-agent run records and source interaction identity. |
| `resume_scheduler.py` | Discovers and schedules due paused/finalization work in a background singleton. |
| `recovery.py` | Content-free inspection and strict preview manifests for explicitly selected saved-output recovery. Never opens the scheduler gate. |
| `resume_worker.py` | Resumes paused runs, rebuilds request context, and records retry state. Finalization uses an immutable run-keyed receipt so learning writes and retry billing reuse the same persisted IDs. |
| `outcome.py` | Provides the generic extraction outcome wrapper used by callers. |
| `input_preparation.py` | Ownership-checked field collection and bounded, lossless legacy chunks with original field/character spans. The new collector does not impose legacy chunk geometry. |
| `compression.py` | Qualification adapter for the shared compressor; restores original range references and checks window authority before and after preparation. Does not select a production worker protocol. |
| `protected_record.py` | Accumulates source-verified quotations for cross-chunk context; capacity failure never silently removes an earlier correction. |
| `protected_pass.py` | Builds that context with checkpointed source-validated model passes; keeps scoped approvals and reported versus observed outcomes distinct. Bounded repair feedback identifies invalid quotation entries without disclosing source text; inner-JSON escaping and exact spelling remain source invariants. |
| `overflow_coordinator.py` | Owns scoped provider admission, monotonic durable attempt reservations, immutable step receipts and historical observed usage. Never publishes child learnings. |
| `overflow_presentation.py` | Supplies prepared history only to extraction prompts, retaining original models and global source labels. |
| `overflow_pipeline.py` | Stages all original chunks, reconciles candidates, verifies coverage and reuses source-bound saved output. |
| `overflow_answers.py` | Pins complete human answers, resumes saved children under the window lease, and validates answer/dependency witnesses before writes. |
| `chunk_gate.py` | Reuses configured domain gate prompts for each chunk; a negative result requires complete valid negative coverage. |

## Architecture Pattern


Extraction records durable agent state and pending human calls; the scheduler discovers resumable work and the worker reconstructs context, resumes the agent, and commits finalization with an immutable receipt. Domain services retain prompts and learning semantics.

## Requirements / Problems to Avoid


- Keep profile-specific and playbook-specific extraction behavior in their own
  modules; call this package only for shared async runtime concerns.
- Commit resumable learning writes, lineage changes, and the agent-run
  finalization receipt in one storage transaction. Retries must return the
  receipt's persisted IDs instead of repeating finalization.
- Add a new file here when the behavior is shared by more than one extraction
  caller or is part of the resumable runtime itself.
- Split into subpackages only when a responsibility grows large enough that a
  flat package stops being easier to scan.
- Do not recreate removed legacy modules such as `tools.py`, `plan.py`, or
  `invariants.py`; use the current focused files above.
- New durable extraction runs require a non-empty `user_id`. Nullable stored
  bindings remain readable for backward compatibility; playbook resume derives
  an in-memory owner only from complete, unanimous persisted source evidence.

## Key Endpoints / Commands / Contracts


### Overflow recovery

New preparation version 3 includes the original `Request.source` alongside
transcript role, including an empty source as unknown. Protected quotations and
carried proof retain verified producer labels: a `User` role alone does not prove
human authorship or approval. Frozen version 1/2 plans keep their prior framing
and receipts.

Any `overflow` key in a parent snapshot excludes it from legacy retry,
finalization, discovery and selected recovery. A null or malformed marker also
blocks native window execution, even when the feature is disabled or mock/image
handling would otherwise bypass admission; it cannot reset the attempt journal.
The full saved attempt journal is validated both on load and under the locked
provider reservation. Damaged usage or provenance cannot dispatch a paid call;
reserved attempts with unknown usage remain consumed and are never refunded.
The saved checkpoint must retain its stage and bounded completed-step map;
missing progress is refused rather than treated as a fresh extraction.
Completed steps retain a structured receipt even for false or empty outputs;
null receipts are refused before another provider reservation.
After a configuration failure, validated saved output regains its completed
agent state inside the learning-write transaction before publication. Cached
write plans and custom extractors share this recovery fence; human-resume and
already-finalized states are preserved.
Corrections to consumed answers
after protocol completion remain durable and unconsumed while the protocol
parent retains its finalized state; answer readiness never assigns legacy
resume ownership. Automatic recomputation
of these completed protocol runs is currently unsupported. The original output
and attempt journal remain intact.

Overflow runs preserve original timestamps and session identities in their
prepared sources. A typed provider context-capacity rejection can freeze one
smaller source plan before the next worker turn, retaining all consumed quota,
usage and completed receipts. Only the rejected unfinished chunk is split;
unaffected source geometry remains stable. A second rejection or an unsafe
tool continuation holds the window rather than resetting work. Legacy cached
write plans keep their existing publication path, and adoption preserves known
original usage separately from new attempts. Unknown historical usage is never
manufactured. Admission remains disabled by default.

New `MOCK_LLM_RESPONSE` simulations retain the ordinary mock extraction path,
without admitting a provider-budget protocol. Switching an existing durable
agent run to mock mode refuses computation before publication; resume that work
in real mode to preserve its saved output, source coverage and consumed quota.

New protected passes ask the model to select IDs for bounded original snippets;
the server copies their exact text. Every selected ID becomes a separate
quotation bound to one original field. A contained range can narrow one shown
snippet; an offsetless field ID resolves only when it names one shown snippet.
Canonical source offsets are retained. Unknown or ambiguous IDs and record-capacity
overflow cannot become saved receipts or publish output. Existing v1 quotation
receipts remain reusable under their original source and quota checks.
Complete-prompt admission targets 65,536 budgeted tokens after the 20% counting
margin, while reserving output and respecting each model's actual capacity.
One capacity correction may select fewer new snippets; a failed correction
remains a typed capacity hold and cannot spend fallback calls. Shared quotations
already physically covered are reused without duplicating their record cost.
Reconstructed playbooks retain the persisted parent's creation time, and each
consolidation receipt binds only its participating historical witnesses. Resumed
turns therefore reuse unchanged decisions as time passes or other history is read;
the complete historical witness set still fences publication.

Protected context exposes only its literal quotation at original source offsets.
Final reconciliation and cached recovery reuse proof carried by registered child
outputs or that quoted context; scanning the full source does not make its
unshown remainder valid prompt evidence. Profile publication keeps canonical
quotation, notes and reader-perspective fields through consolidation.

New image-bearing windows keep the established native multimodal path. Existing
textual manifests cannot switch protocols or erase quota. Configuration failures
remain visible and retry with existing backoff after correction; capacity/quota
failures stay held. Full human answers that exceed the complete-prompt target
fail before dispatch without clipping, publication or an automatic retry loop.

### Resume discovery


When an `org_id_provider` is installed, the scheduler calls it on every tick and
treats its actionable org list as authoritative for cross-ref discovery. Without
a provider, local storage discovery remains the fallback. Each provider result
selects a current org for reading scheduler configuration, so an org retained
from an earlier tick cannot block later discovery if it becomes stale. If the
provider itself raises, the scheduler still attempts the last bootstrap org for
that tick. Before draining each discovered org context, it expires pending tool
calls on that context's storage ref; separate refs have separate queues.

### Selected recovery and progress

`ExtractionResumeWorker.recover_selected` uses an exact-ID storage claim, then
revalidates the preview under the user lease before normal receipt-protected
finalization. Claims and status updates are owner-fenced. `progress_stage` and
`last_progress_at` change at lifecycle boundaries; an atomic receipt records
`results_committed`. Legacy rows keep unknown progress. `recovery_history` holds
operation/status metadata only. Do not treat a row timestamp as a heartbeat or
call the private finalizers as an operator replay API.

Selected recovery requires original per-source digests captured before generation.
Missing historical witnesses stay blocked; never backfill them from current rows.
Sources are compared before claim and under the worker lease, using batches of
100 selected IDs. Reports contain neither source content nor digests. The check
catches source replacement observed at revalidation; arbitrary external writers
are not made participants in the worker lease by this feature.
Under-lease safety refusals preserve output in terminal `failed` status. Ordinary
retries of interrupted selected work retain these checks until an atomic receipt
exists; committed receipts keep the normal billing-delivery replay path.
All finalization retries with a receipt replay billing from stored result IDs,
without source reconstruction or result generation, even after source deletion.
Existing owner proof remains required for legacy playbook records.

### Overflow preparation (implementation in progress)

Coordinator admission, gates and domain chunk factories are wired behind the
default-off overflow flag; the complete feature remains under qualification.
`llm/input_budget.py` admits complete built provider requests through a
run-scoped guard, including schema, tools, output reserve and fallback attempts.
The shared SQL storage checkpoint operation fences the window/user lease and
uses a manifest revision to prevent stale checkpoint writes. Provider attempt
reservations and immutable observed usage survive retries; unknown usage never
refunds an attempt. Native pending-human staging is wired with race and rollback verification;
full publish/answer/persistence qualification and bounded historical consolidation
remain before enabling the end-to-end feature. The worker recognizes typed suspension
without completing a window or increasing its failure count. Protected-pass,
gate and presentation helpers have deterministic provider verification; this
does not establish real-model evidence-selection quality.

Budget/configuration failures hold incomplete coverage for explicit recovery,
preserve the original inputs and quota, and report a controlled `failed` learning
status. They cannot publish partial output or continually spend provider calls.
The healthy sibling cursor and other users remain runnable. An all-false gate
finalizes its otherwise empty parent inside the same fenced coverage commit.

Under a scoped coordinator/presentation, the resumable agent stages child
outputs without completing the parent. It checkpoints synchronous tool messages
and the remaining child step count before the next model turn; retries reuse
that continuation. A pending human/tool dependency suspends the child, and its
protocol wake-up/resume routing still needs integration. Custom extraction/gate
templates that omit the complete prepared source fail before provider dispatch.

Native overflow requests without a configured output cap explicitly request the
model's advertised output maximum before admission. Configured caps are preserved;
custom endpoints require their own positive request/client cap and context
capacity declaration. Other calls retain their original provider defaults.

## Shared input compression (qualification protocol)

`reflexio.server.services.input_compression` implements bounded, source-backed
hierarchical compression with fixed `extraction` and `offline_tuning` policies.
`extraction.compression.compress_extraction_window` prepares every original text
field through the existing ownership-checked input preparation, including tool
payloads. It returns an evidence package, never a learning or write plan.
Its result's `source_digest` is the canonical complete-window witness, including
empty interactions; the core compressor's internal DTO digest only binds caches.
When all original conversation-content fields together occupy at most half the
actual output budget, the window adapter preserves them unchanged. Tool evidence
uses the remaining space; every intermediate and final measurement includes
these preserved fields. Their witness also binds cached selections. This avoids
making short instructions, approvals and assistant qualifications compete with
thousands of tool results. It does not establish human authorship from a role,
or guarantee preservation when conversation content itself is too large.

Inputs already fitting the caller's output renderer bypass model work. Larger
inputs are split at Unicode character boundaries, independently compressed in
bounded sections and combined in chronological pairs. A fitting union avoids
another model call. Intermediate products must fit the next request; the root
retains the full final output budget. Source reading and calls remain capped.

Model prompts hoist shared source origins and use short evidence IDs. Exact
duplicate text may reference its first displayed occurrence, without merging
source identities or chronology. The server copies selected original spans and
restores full references; input-origin metadata remains server-owned.
The guarded LLM seam requests only evidence-reference groups,
never generated factual descriptions, summaries, unresolved-status claims or
omission claims. The host supplies fixed navigation text and measured omission
metadata and labels. Preserved conversation fields have their own host-created group.
Every cached selection is revalidated against the current visible source and
the caller's actual output renderer. Cache identities bind scope, source,
policy/schema, effective model route/generation settings and rendering contract.

`input_compression_llm.compress_with_client` checks actual provider requests,
including repairs, preserves the parent token capture and scoped durable guard,
and distinguishes observed provider usage from proxy token estimates. The new
complete-input target is 32,768 tokens with the existing 20% margin and output
reservation. This does not silently change the old overflow guard's 65,536-token
default or upgrade an admitted extraction journal.

These are separately callable qualification adapters, not a new default worker
protocol. Existing extraction publication, source-quoted recovery branches,
cursor/billing effects and human continuations are not switched by importing
them. A successful source scan and exact citations do not prove semantic recall.
Wiring new admissions requires downstream extraction qualification and a new
versioned journal contract; old admitted runs must retain their original policy.

Whole-point packing retains all citations of an accepted point together and
reports omitted points rather than repeatedly asking the provider to count its
copied quotations. This is a bounded partial result, not proof of semantic
completeness: a model can still place counterevidence in a separate dropped
point. Exact-source and downstream quality qualification remain required.

After the existing bounded semantic repair, the guarded seam can discard entire
groups containing unseen references from a complete schema-valid response.
Every retained group is revalidated. No individual reference is guessed or
removed from a retained group, and parse, refusal, model-route, provider-budget and
integrity failures are not converted to accepted results. Host omission notices
and `provider_reference_recoveries` make this partial selection visible. That
counter describes new provider work in this operation; cache hits may carry
earlier omission notices without a new recovery.

`packet.omission_counts` carries two bounded host-owned counters through unions,
model reductions and cache replay: `capacity_points` and
`unseen_reference_groups`. They count discarded groups/points across processing
stages, not distinct missing source passages. They remain visible when a later
selection has no new human-readable omission notice. Actual output measurements
include this metadata; the offline adapter also retains it in the analyst's
non-citable compaction summary.

Before accepting capacity loss, native selection validation reports how many
whole points would be dropped and asks for tighter original passages through
the existing bounded semantic-repair mechanism. This can remove unrelated
neighboring logs without the server severing a point's evidence. If the complete
repaired selection still exceeds capacity, the host may retain fitting whole
points, recording the partial result and omission totals. All repair attempts
consume the same guarded quota; source-integrity errors still abort.
Complete selections are admitted first at their true size. Space for a new
omission notice is reserved only when the complete selection cannot fit.
