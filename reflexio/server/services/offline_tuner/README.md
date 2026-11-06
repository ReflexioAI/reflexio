# /reflexio/server/services/offline_tuner
Description: Engine of the offline user-playbook tuner: deterministic evidence selection, judge+tuner data contracts, citation verification, qualification fixtures and sweep progress records.

## Main Entry Points


| Path | Responsibility |
|------|----------------|
| `config.py` | `OfflineTunerConfig`, generation-model resolution and the problem-line log level |
| `evidence/open_world_selection.py`, `evidence/open_world_policy.py` | Deterministic, model-free selection of tuning evidence under the fixed production policy |
| `evidence/open_world_models.py`, `evidence/open_world_snapshot_assembly.py`, `evidence/manifests.py` | Pure evidence records, canonical manifests and snapshot assembly from rows a storage backend has read |
| `evidence/compaction.py`, `evidence/shared_compaction_checkpoint.py`, `evidence/native_capacity.py` | Bounded source-backed session compaction, its versioned checkpoint codec and verified native context limits |
| `evidence/regeneration_fence.py`, `open_world/fence.py` | Regeneration fence digests: refuse a second attempt on identical evidence |
| `open_world/models.py`, `open_world/identity.py` | Frozen judge+tuner data contracts, canonical serialization and stable call identities |
| `open_world/analyst.py` | Role-separated analyst adapters over `LiteLLMClient` and `PromptManager` |
| `open_world/citations.py` | Span-scoped citation verification tolerant of rendering noise only |
| `open_world/qualification.py`, `open_world/rule_evidence_authority.py` | Versioned synthetic qualification fixtures and the immutable identities of the evidence-driven method |
| `open_world/reducer.py`, `open_world/runner.py` | Deterministic memo reduction and the shared job lease |
| `open_world/shared_compaction.py` | Shared compression dispatch and cache journal inside one fenced job |
| `evidence/open_world_store.py` | Atomic freeze and reload of one canonical evidence bundle through the storage port; not re-exported from `evidence/__init__` (import cycle) |
| `open_world/evidence_documents.py`, `open_world/compaction.py`, `open_world/hierarchical_compaction.py`, `open_world/shared_compaction_preparer.py` | Frozen-evidence rehydration, native consumer sizing and shared long-input preparation |
| `open_world/rule_evidence.py`, `open_world/rule_evidence_runtime.py`, `open_world/diagnostics.py` | Single-method diagnostics, exact-clause proposals, durable calls on `RuleEvidenceStorage` and terminal-failure diagnostics |
| `open_world/publication.py` | `OpenWorldProvisionalPublisher`: provisional publication of one accepted attempt |
| `open_world_invocations.py`, `open_world_job_metadata.py` | Refusal vocabulary, durable invocation records and advisory job-metadata keys |
| `open_world_port.py`, `open_world_records.py` | `OpenWorldStoragePort` (the consumer Protocols a storage backend implements to run the active open-world path) and the storage records and refusal errors it returns and raises |
| `exposure_receipts.py` | Signed search receipts and their verification; `receipt_project_id()` is the project resolver issuer and local verifier share |
| `search_projection.py`, `models.py` | Immutable candidate search projection and `CandidateRevision` |
| `exposure_ledger.py` | `ExposureLedgerRecorder`: turns a final search batch into user-playbook exposure events on the org's storage |
| `retention.py`, `sweep_progress.py` | Bounded retention sweeps over a supplied storage, and per-tick progress records |

Package data: `open_world/qualification_cases/` and `open_world/qualification_responses/`
hold the versioned qualification suite and its recorded responses. Code loads
them with `importlib.resources.files(__package__)`, so they ship inside the wheel.

## Purpose


1. **Select evidence** - choose one owned playbook's reconstructed, historically version-matched sessions deterministically, before any model call.
2. **Define the judge+tuner contracts** - the request, memo, decision and invocation records every attempt persists, with stable identities.
3. **Verify model output** - citations must resolve to frozen source spans; qualification fixtures gate which analysts may run.
4. **Record progress** - one durable progress record per sweep tick and the outcome vocabulary operators read.

## Architecture Pattern


```
evidence rows (read by the host's storage backend)
   -> evidence/open_world_selection + open_world_snapshot_assembly  (pure, deterministic)
   -> open_world/identity + models                                  (canonical request + identity)
   -> open_world/analyst                                            (LiteLLMClient + PromptManager)
   -> open_world/citations + reducer                                (verify, reduce to one decision)
   -> sweep_progress                                                (per-tick outcome record)
```

No module here imports a storage backend or reads the deployment mode. The
package defines no storage implementation, scheduler or HTTP route: the host
application supplies persistence, the sweep loop and publication. Modules that
write receive a storage object or protocol from the caller; `exposure_ledger.py`
alone resolves the org's storage through the cached `Reflexio` instance.

## Requirements / Problems to Avoid


- **Do not import a storage backend here** - persistence belongs to the host behind its own storage seam; take storage as an argument or protocol.
- **Prompt versions are pinned identities** - `open_world/analyst.py` renders `offline_tuner_open_world_discovery`, `offline_tuner_open_world_held_out` and `offline_tuner_open_world_candidate` from the shared prompt bank. Qualification binds each template digest, and persisted identities render historical versions, so add a new version rather than editing one, and regenerate `qualification_responses/` with it.
- **Do not loosen citation verification into edit distance** - `open_world/citations.py` folds rendering differences only; a one-character substitution of user text must stay rejected.
- **Qualification data is versioned** - change `qualification_cases/` or `qualification_responses/` only together with the suite version and digests that `open_world/qualification.py` checks.
- **Completion labels are optional metadata** - a single unlabeled, unknown-outcome, success-only or failure-only session can supply evidence; do not reintroduce a success/failure contrast requirement.
