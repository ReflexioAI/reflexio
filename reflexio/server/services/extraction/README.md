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
| `pending_tool_admission.py` | Owns org/storage admission at fresh/resume construction and direct tool effects; rejects unadopted rollout vetoes. |
| `pending_tool_call_dispatch.py` | Implements the `ask_human` and pending-info tool dispatch flow. |
| `prior_answer_search.py` | Finds and formats previous human answers; captures immutable complete prior context before package planning, without admission or journal writes. |
| `agent_run_records.py` | Builds durable extraction-agent run records and source interaction identity. |
| `resume_scheduler.py` | Discovers and schedules due paused/finalization work in a background singleton. |
| `recovery.py` | Content-free inspection and strict preview manifests for explicitly selected saved-output recovery. Never opens the scheduler gate. |
| `resume_worker.py` | Resumes paused runs, rebuilds request context, and records retry state. Finalization uses an immutable run-keyed receipt so learning writes and retry billing reuse the same persisted IDs. |
| `outcome.py` | Provides the generic extraction outcome wrapper used by callers. |
| `overflow_protocol.py` | Pure validation of the inactive outer-v2 source-package descriptor, frozen stage budgets and typed source-selection receipts. |
| `input_preparation.py` | Ownership-checked field collection and bounded, lossless legacy chunks with original field/character spans. The new collector does not impose legacy chunk geometry. |
| `compression.py` | Provider-free window planning and guarded execution for the shared compressor; restores original range references and checks window authority before and after execution. Does not select a production worker protocol. |
| `source_package.py` | Pure consumer rendering and validated, immutable presentation of selected original ranges, attribution and visible quotation boundaries. |
| `package_compression.py` | Compresses against an already-admitted package journal, binding exact per-node inputs/native contracts to durable selections and actual charged attempts. Does not admit, consume or publish a window. |
| `profile_package_preparation.py` | Freezes profile prompt context, templates, actual native ladders/schema/tools/output policy and complete input pricing before package compression. No admission or worker selection. |
| `profile_package_consumer.py` | Consumes an already-admitted package through one real gate and the durable profile child; never publishes its parent. |
| `profile_package_consolidator.py` | Journals frozen OLD inputs and real native duplicate-check decisions, preserving typed host fallback and paid attempts. |
| `profile_package_state.py` | Canonical consumed-field witnesses shared by profile runtime and the locked SQL publication fence. |
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
When that fence (or a mid-pass digest change) finds a witnessed playbook changed,
it raises `ComputationEvidenceChangedError`. Retrying the cached plan cannot pass,
so the worker logs `Extraction outcome discarded for recompute` and, in one
lease-fenced transaction (`discard_extraction_computation`), clears the window's
cached outcome and the checkpoint's `historical_playbooks`, then schedules the
ordinary configuration retry. The next attempt recomputes against current
playbooks. Content-addressed completed steps and historical chunks are kept, and
spent provider attempts are never refunded, so repeated recomputes end in the
attempt-budget hold.

Protected context exposes only its literal quotation at original source offsets.
An immutable window admitted before `extractor_name` retirement can create its
first parent after retirement. The shared `matches_frozen_window_binding` proof
authenticates the stored parent snapshot with its stored hash, then accepts an
exact window policy or a window with only that extra retired label. Coordinator
readmission and child validation use the same proof, including project and all
other binding fields. Historical snapshots, lookup identities, child receipts
and spent quota remain unchanged; arbitrary model normalization and unknown
window fields are not compatibility proofs.

Profile chunk checkpoints distinguish the complete rendered prompt hash from an
instruction witness: the same source/schema and context-free rendering plus the
unrendered template identities. If only existing-profile context changes between
attempts, the ordinary fenced journal reselects the affected children. Old
content-addressed receipts and consumed provider attempts remain intact; they
cannot satisfy a new prompt or increase its remaining quota. Genuine instruction
or schema changes, active/waiting continuations and legacy prompt mismatches
without this witness remain held. An unchanged legacy prompt can acquire the
witness. Fully computed saved outputs retain their existing current-contract
revalidation path. Only hashes are added, not another copy of profile text.

Final reconciliation and cached recovery reuse proof carried by registered child
outputs or that quoted context; scanning the full source does not make its
unshown remainder valid prompt evidence. Profile publication keeps canonical
quotation, notes and reader-perspective fields through consolidation.

New image-bearing windows keep the established native multimodal path. Existing
textual manifests cannot switch protocols or erase quota. Configuration failures
remain visible and retry with existing backoff after correction; capacity/quota
failures stay held. Full human answers that exceed the complete-prompt target
fail before dispatch without clipping, publication or an automatic retry loop.
A response cut off by its output limit (`finish_reason="length"`) is counted
against its exact request fingerprint in the checkpoint. After
`MAX_TRUNCATED_REQUEST_DISPATCHES` (4) such dispatches, the same request is
refused before dispatch (`OutputTruncatedRequestError`) and the window is held as
`input_budget_exceeded`, instead of resending it until the 128-attempt budget is
spent. Truncation by a reasoning model is often transient, so a request that
finishes within those dispatches proceeds normally. A re-armed window is held again unless the request changed (model, output
cap or input).

### Resume discovery


When an `org_id_provider` is installed, the scheduler calls it on every tick and
treats its actionable org list as authoritative for cross-ref discovery. Without
a provider, local storage discovery remains the fallback. Each provider result
selects a current org for reading scheduler configuration, so an org retained
from an earlier tick cannot block later discovery if it becomes stale. If the
provider itself raises, the scheduler still attempts the last bootstrap org for
that tick. Before draining each discovered org context, it expires pending tool
calls on that context's storage ref; separate refs have separate queues.

A provider's empty list is read as proof that the fleet is idle, which is what
lets the wait double towards `IDLE_BACKOFF_CAP_SECONDS`. A provider that catches
its own discovery failures must therefore never answer one with `[]`: it reports
the failure as it already does and raises `ResumeDiscoveryIncomplete`, carrying
whatever orgs it did find. Those orgs are drained (or the last bootstrap org,
when there are none), the tick is never counted as idle, and the scheduler does
not report the failure again -- any other exception is treated as an unreported
provider failure and is reported there.

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


### Source-package journal prerequisite (inactive)

Outer journal version 2 requires `input_protocol.name=source-package`, descriptor
version 1 and preparation version 4. The unchanged current executor rejects that
preparation version before legacy input preparation or cached publication. The
existing enable flag does not admit this protocol. Outer-v1 fields, prompts,
partitions, model receipts and quota behavior retain their established path.
An already admitted outer-v1 run or marker-free legacy output cannot be converted
to outer2 through adoption.

`OverflowCoordinator.initial_package_manifest` copies an explicit frozen native
plan descriptor. It binds the original source witness, DTO cache witness, native
plan/contract digests, ordered sections and exact parent scope. The future caller
must rebuild and compare the actual model/schema/pricing/output/passthrough and
consumer contracts and resolve selections against original sources. Ordered
section witnesses need not be unique; every position counts toward geometry,
and the full native plan witness binds actual source identities. Structural
journal validation alone does not prove those semantics or source coverage.
No source text, API credentials or endpoint credentials belong in the descriptor.

The profile follow-on allowance is `F=Lg+16Le+2Ld` using actual resolved native
ladder lengths (the gate term is zero when skipped). Source-quoted playbooks use
`F=2Li`. At most 48 sections are admitted and `max(0,2S-1)+F` must fit 128.
The compressor cap is `128-F`; each consumer stage has its own frozen cap. These
are cumulative physical-dispatch limits, not completion guarantees: failures,
repairs, fallbacks and crashes may exhaust a stage before it produces a result.
Cached outputs cost no new attempt. Unused consumer capacity does not enlarge the
compressor cap after restart.

`attempt_stage(stage, minimum_remaining_attempts=0)` attributes every reservation
inside the existing shared SQL CAS. The minimum names additional required work
in that same stage after the proposed dispatch. Counts are recomputed from saved
attempts, including calls with unknown usage after a crash. Missing attribution,
changed scope, stale revisions and exhausted stage/global caps reject before a
transport call. Observations cannot change stage or previously observed usage.
The yielded list contains actual newly reserved IDs for binding a node receipt.
Consumer stages compare the actual native guard's ordered literal model ladder
with the frozen stage ladder before reserving. Compression remains guarded by
the native planner. This comparison does not validate the full route, schema or
generation contract: future runtime composition must still prove those witnesses.
The CAS also compares the caller's frozen descriptor with the locked saved copy.

`complete_selection_step` saves a distinct source-selection receipt with original,
plan, node, input-reference and request-contract digests and actual compressor
attempt IDs. Its output is the exact server-normalized native cache choice:
reference groups, fixed host labels, local omission counters and host diagnostics.
Those omissions survive cache reuse verbatim; receipts do not invent authored
model provenance. `input_refs_digest` must describe the actual visible inputs for
that node, and `request_contract_digest` must describe its actual native request
contract. The original load/save signatures still expose only a hashed node key
and normalized choice. `plan.execute(inspect_node=...)` additionally reports the
actual visible-original-source digest before each node's cache access, and the
native plan exposes `request_contract_digest` for its actual frozen generation
descriptor and resolved ladder. Rehashing the node key and naming it either
witness is invalid. The node key already binds the source
witness, messages and schema, but does not substitute for these separately named
proofs. Source-selection receipts are bounded to 64 KiB, with at most 128 completed
steps. Typed consumer model-output receipts retain their native output/trace
contract; no new small limit clips complete human/tool results. Host-only passthrough/union
coverage and package publication are later integration work, not fake paid
selection receipts. Provider input rejection cannot invoke legacy repartition.

An old worker may expose a configuration failure and update its ordinary retry
backoff/status for outer2, but snapshot, revision, paid attempts, receipts and
committed output must survive unchanged. A new coordinator can reload that failed
parent without refunding quota. The frozen actual outer-v1 reader fixture and
HTTP/storage tests protect this mixed-version boundary. None of this enables a
new runtime, changes billing, deploys code, or establishes semantic qualification.

### Provider-free window planning and source presentation

`plan_extraction_window` returns an `ExtractionCompressionPlan` before any lease,
cache or provider callbacks run. It combines the full original-window witness
(including empty interactions) with the native plan witness. Rebuild and compare
that witness under the caller's durable scope before execution; the narrower
`source_cache_digest` alone is not an authorization or publication receipt.
Execution retains before/after authority checks and per-dispatch checks. Text
plans refuse images rather than silently dropping their evidence.

The default output contract still counts packet JSON. A caller may supply a
pure `measure_output` with a separately versioned `output_contract` to count the
complete judge and extractor input instead. For that contract the result's
`output_tokens` means the measured complete consumer input, not packet-only
size or provider output usage. Instructions, existing context, tools and output
schemas must all be included; the input limit must also leave each consumer's
actual output reservation within its model capacity. `measure_request_input_tokens`
uses the same complete-input counter and 20% margin as native admission, but
returns oversized counts so packing can remove whole groups. Actual dispatch
still needs the rejecting admission guard on every continuation and repair.
If the fixed rendering cannot fit even with no selected evidence, planning
refuses before native section preparation or provider work; the shared core
enforces the same rule for other consumer renderers.
An optional inexpensive `estimate_output` only sizes advisory source menus;
exact measurement remains authoritative. Both callbacks must be pure, frozen
and named by the rendering contract. Planning time and execution time are counted
without including the caller's wait between them.

`source_package_presentation` verifies each selected range and all original
metadata, ordering and point references against `ExtractionSources`. It freezes
prompt bytes and the selected visible original spans. Its renderer includes
producer/role attribution and labels host navigation as non-citable. A quotation
checker must use only these visible spans; it may join adjacent original ranges
but cannot recover text across a gap. This presentation does not prove complete
processing, own tenant authorization, or install a runtime context.

The profile prompt builder accepts explicit `source_text`, and
`build_profile_gate_prompt` renders the existing criteria from explicit inputs.
They allow faithful prompt costing without a fabricated legacy chunk or
coordinator. Durable callers must also freeze real template identities and add
all native tool/schema/answer augmentation. These planning seams do not admit
new source-package workers or change existing saved journals.

### Durable package compression handoff (inactive)

`package_compression.compress_source_package` requires a matching already-admitted
outer2 descriptor. It uses the real native planner, coordinator, stage-attempt
guard and SQL checkpoint CAS. Per-node selection receipts retain the exact
visible-source and generation-contract witnesses, canonical cache choice and
attempt IDs reserved while computing that node. Cache replay compares those
bindings and revalidates selected originals and rendered capacity. A process
restart reuses committed nodes without refunding or paying for them again.

Full original-source witnesses are checked at operation boundaries. Each actual
dispatch retains the existing scoped CAS lease/identity checks; the bridge does
not add a full blob read or duplicate lease query per call. Standalone OSS's
explicit empty project ID is a distinct cache scope, never a wildcard; omitted
projects and empty organization/user identities remain invalid.
Both boundaries also lock and revalidate the current durable parent, descriptor,
revision, project and active window through the same shared check used by CAS.
Cache-only and passthrough runs cannot rely on a cancelled or changed in-memory
parent. This validation does not advance the journal or refund consumed attempts.

The returned result and validated source presentation do not mark window
coverage, run the judge/extractor, finalize a parent, write learnings or bill a
completed extraction. Consumer contract composition and worker admission remain
separate integration work. No default worker selection or enable flag changes.

### Complete profile consumer preparation (inactive)

`profile_package_original_seed` builds the existing profile prompt from original
history for the existing bounded prior-answer query. Capture that query once
with `capture_prior_knowledge_context`; it uses the same org-scoped retrieval
and preserves complete answers, including conditions after 1,200 characters.
The immutable capture stores only the query digest, suffix and exact resolved
answer witnesses. It writes no journal and grants no authority: admission and
publication must still validate answer lifetime and source/lease ownership.
`prepare_profile_package` rejects a capture from a different seed or retrieval
policy rather than repeating lookup after compression.
When pending-input tools are active, the capture is required even when its
lookup found no matches; an omitted lookup must not stand in for an empty result.
Its required `source` argument is the actual service/run trigger, including
None for a rerun; it must not be inferred from the newest original request.

Preparation freezes template identities, selected existing-profile context,
actual pending-tool specs/configuration, original output schemas, native full
ladders and generation-contract hashes. Potentially credential-bearing route
descriptors and context-limit keys are hashed, not copied into the snapshot.
Pending-tool signing secrets are also excluded; only their configuration digest
is retained. Runtime callers must obtain actual secrets from live approved
configuration rather than reconstructing credentials from this private snapshot.
The real request builder runs inside a temporary dispatch-refusing extraction
guard, preserving native output-reservation/model-policy order and restoring
the enclosing guard on both success and failure. No provider or quota callback
runs during this preparation.

The compressor measures the maximum COMPLETE initial gate/agent input across
all native rungs, including schema instructions, real tools, full prior answers
and the existing 20% counting margin. Its input ceiling is the minimum of 32,768
and each rung's context minus its native output reservation. The packet's raw
size and provider usage are separate quantities. Fixed prompt overflow fails
before compression work; consumer output defaults are preserved.

The prepared request checker verifies route/generation/schema/tool policy,
ordered ladder, built messages, native bounded repairs and 32K input before a
caller reserves quota. Plain parser corrections and validator repairs have
different native formats; their checks remain separate. A model echo remains
untrusted, never source evidence. The caller must still authenticate the raw
seed/continuation, bind the contract to its durable parent, install the check
before every dispatch, and handle dynamic dedup prompts and continuations.
Initial-fit pricing does not reserve space for every possible repair or tool
continuation. A fitting first request can therefore fail on a larger repair;
the completed attempt remains consumed and no successful result is returned.
Consumer-contract preflight also runs before cache access or passthrough, before
each new compression dispatch and at operation completion. Template, schema and
consumer-ladder drift cannot wait until final packet measurement to be rejected.
No parent admission, consumer dispatch, processed coverage, publication,
billing or runtime enablement is introduced by preparation alone.

`profile_package_consumer.consume_profile_package` is an explicit entry point
for an already-admitted profile package. It verifies the frozen preparation,
original inputs and pinned human answers before compression, records the actual
processed node keys and final package identity, and feeds that exact presentation
to one gate and the existing durable eight-turn child loop. Every native consumer
dispatch now requires the complete prepared request contract before quota
reservation, including repairs, fallback rungs and complete tool continuations.
The same quota CAS binds actual paid attempts to the logical step; bindings are
immutable. Cached gate and child receipts must reference real attempts at their
exact stage and step, including completed turns before a waiting restart.
Unknown lost-response attempts remain consumed without inventing a result.
Unmatched signing configuration is refused before any paid work.

Cache-only replay rebuilds the package from selection receipts and refuses
missing work; it does not spend a new model call. Passthrough and host unions
have processing witnesses without invented model provenance or paid attempts.
These witnesses establish processing, not semantic completeness or publication
authority. Child completion leaves the parent running with no committed output.
This entry point does not install worker admission, deduplicate candidates,
authorize final coverage, publish profiles, advance cursors, bill completion or
activate runtime selection. Those remaining integration boundaries must be
qualified separately before selecting this path in a worker.


### Explicit profile parent qualification

`durable_learning/profile_package_executor.py::ProfilePackageWindowExecutor`
connects the prepared package to the existing atomic window publication and
outbox. Ordinary workers still select `WindowExecutor`; this entry point adds
no flag, deployment or automatic runtime activation.

It freezes the actual template versions, selected profile context, complete prior
answers and native consumer contract before admitting an outer2 parent. Resume
rebuilds that preparation from authoritative originals and frozen context rather
than repeating prior lookup or selecting newly activated prompts. Legacy parents
retain their own protocol and are refused by this entry point.

`profile_package_consolidator.py` reuses the ordinary native duplicate checker,
freezing its retrieved OLD rows, ordering, template and actual paid decision.
A first-parsed or transport fallback remains explicitly a host fallback; malformed
cached receipts cannot enter that fallback. Converted candidate IDs, TTL clocks,
resolved write plan and precomputed vectors are frozen once. Their binding includes
the actual replayed child output/provenance, source and consumer contract. Shared
checkpoint CAS forbids changing those host decisions after capture.

`profile_package_state.py` defines the consumed-field witness for existing profile
context and duplicate-check rows. The shared SQL fence reads bounded named-column
batches and holds row locks through publication. Deleted, expired, superseded or
changed OLD rows block both fresh and cached publication, including no-action
dedup decisions. Original inputs, answers, lease and project retain their existing
fences. Profiles, finalization receipt, cursor and effects commit together.

Positive extraction counts original fields once for its source-input billing basis;
physical provider usage comes from the durable attempt journal. A negative gate
produces no learning or source-input charge, while its actual model usage remains
in the existing strict token outbox. Processing receipts prove processing, not
semantic recall. Live quality qualification and default worker selection remain
separate gates.

Playbook request construction uses the shared `PlaybookExtractionContract` citation
instruction with its selected schema and validator. `resume_worker` passes the
stored schema selection through to message construction; overflow calls use the
actual adopted schema where present. Exact-span and quoted-evidence modes remain
distinct, while the existing structured repair inherits the same contract. See
[`playbook/README.md`](../playbook/README.md#extraction-citation-contract).


## Retiring a pending-tool rollout policy

The former `resumable_extraction_agent.enabled` and `enabled_org_ids` sitevar
fields no longer select execution. Their framework is removed; the existing
org `pending_tool_call_config.enabled` owns pending-tool admission. Storage is
still required. Disabling new tools does not disable the permanent extraction
loop or change the existing resume scheduler/finalization policy.

Stock installations need no change: the shipped legacy policy was globally
`enabled: true`, and a missing file is neutral. A retained globally true JSON
policy is verified as neutral without rewriting it; its allowlist had no effect.
Non-neutral, unreadable or ambiguous legacy files are refused before tools can
be enabled, including during direct continuations and direct tool handlers. JSON retains priority over
TXT. An org already configured off stays off without needing the old file.

For an operator-managed legacy cohort policy, adopt it explicitly before upgrade
or before enabling pending tools:

1. Preserve the original policy and org configurations privately. Enumerate all
   orgs and any custom defaults used to create future orgs; this process does
   not discover or rewrite customer configuration automatically.
2. With the old valid policy, an org was allowed when global `enabled` was true
   or its ID was in `enabled_org_ids`. For every denied org, set the existing
   `pending_tool_call_config.enabled` to false through the normal config update
   path. For allowed orgs, preserve their current opt-in value. If a custom
   future-org template opts in despite a global/cohort denial, make that
   template opt out too. Enterprise project overlays cannot override this org setting; other hosts must
   adopt any custom policy overrides too.
3. Read back the effective org settings and verify workers have refreshed their
   config. Preserve existing per-tool settings and stored extraction evidence.
4. Remove the retired rollout policy only after adopting every denial. Removing
   it alone is not proof of adoption. No automatic value copy or live migration
   runs when importing or invoking extraction.

A non-neutral file cannot be used to authorize an allowlisted org after
retirement; it requires the explicit adoption above. Missing configs and absent
storage remain closed. Do not replace the deleted resolver with a helper that
returns true, or reuse auth/tenant checks as a rollout gate.
