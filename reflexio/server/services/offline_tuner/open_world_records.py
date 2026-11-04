"""Storage records and refusal errors for the open-world tuner.

Pure types: the rows a storage backend returns for the durable open-world
path, and the exceptions it raises. No backend and no Protocol lives here --
the consumer contracts are in ``open_world_port``. It sits at the
``offline_tuner`` package level, beside ``open_world_invocations``, so a storage
backend can import it without running the ``evidence`` package's ``__init__``.

Every refusal has exactly ONE exception class, defined here, and every backend
raises that class. ``open_world_invocations`` maps a refusal to its terminal
reason by the class's ``refusal`` attribute; a second class carrying the same
refusal from another module would be caught by nobody's ``except``.
"""

from dataclasses import dataclass
from typing import Literal

from reflexio.models.api_schema.domain import (
    OptimizationJobClaim,
    PlaybookOptimizationArtifact,
    PlaybookOptimizationJob,
)
from reflexio.server.services.offline_tuner.open_world_invocations import (
    OpenWorldRefusal,
    OpenWorldRefusalError,
)
from reflexio.server.services.storage.error import (
    OptimizationArtifactIntegrityError,
    StorageError,
)

OPEN_WORLD_EVIDENCE_ARTIFACT_KIND = "open_world_evidence_bundle"
OPEN_WORLD_EVIDENCE_OPTIMIZER_KIND = "offline_tuner_open_world"
OPEN_WORLD_DISCOVERY_MEMO_ARTIFACT_KIND = "open_world_discovery_memo"
OPEN_WORLD_CANDIDATE_ARTIFACT_KIND = "open_world_candidate"
OPEN_WORLD_ATTEMPT_DECISION_ARTIFACT_KIND = "open_world_attempt_decision"


class OpenWorldEvidenceTargetStaleError(StorageError, OpenWorldRefusalError):
    """The selected user playbook is no longer the current exact version."""

    refusal = OpenWorldRefusal.TARGET_STALE


class OpenWorldEvidenceGovernanceError(StorageError, OpenWorldRefusalError):
    """The complete evidence subject vector is no longer writable."""

    refusal = OpenWorldRefusal.GOVERNANCE_CHANGED


class OpenWorldEvidenceLeaseConflictError(StorageError):
    """Another worker owns the live lease for an incomplete freeze."""


class OpenWorldInvocationLeaseConflictError(StorageError, OpenWorldRefusalError):
    """The optimizer job lease is no longer current for this fence."""

    refusal = OpenWorldRefusal.LEASE_LOST


class OpenWorldInvocationConflictError(StorageError, OpenWorldRefusalError):
    """A durable invocation or artifact identity is bound to different bytes."""

    refusal = OpenWorldRefusal.INVOCATION_CONFLICT


class OpenWorldInvocationIdentityBoundError(OpenWorldInvocationConflictError):
    """This exact analyst-input identity already belongs to a different job.

    A SUBCLASS, so every existing ``except OpenWorldInvocationConflictError``
    keeps catching it and no caller's behaviour changes by accident. What the
    narrower type buys is the one caller that must tell this apart: the durable
    row identity is derived without ``job_id`` and the evidence bundle is
    quantized per UTC day, so a second optimization job for the same playbook on
    the same day recomputes an identity a previous job already owns.

    It is NOT recoverable in-process, and the distinction earns its own type for
    DIAGNOSIS rather than for recovery. Every other conflict this class's parent
    reports is a genuine disagreement about bytes; this one is a slot that a
    same-day sibling job is holding, which clears at the next UTC day boundary
    when the bundle digest changes. The runner turns it into
    :class:`OpenWorldInvocationSlotExhaustedError` so the job reports
    ``invocation_slot_pinned`` instead of ``infrastructure_failure``.

    Do not "fix" this by deriving a per-attempt row identity. That was tried:
    the storage authorization checks recompute ``analyst_input_identity`` from
    a preimage carrying no attempt number, so the advanced row prepared, the
    analyst was called and billed, and the answer was then refused at persist.
    """

    refusal = OpenWorldRefusal.IDENTITY_BOUND


class OpenWorldInvocationSlotExhaustedError(OpenWorldInvocationConflictError):
    """The question cannot be asked: its durable row is bound to another job.

    The terminal statement of the condition
    :class:`OpenWorldInvocationIdentityBoundError` reports. No provider call was
    made, and none can be made for this question today.

    A subclass of the conflict class ON PURPOSE. Should a caller ever forget to
    handle it, it degrades to the pre-existing conflict behaviour rather than
    escaping the runner unhandled -- the same reasoning that put
    ``LiteLLMClientError`` in ``run()``'s handler list after it was found to
    match nothing and abandon the job mid-stage.
    """

    refusal = OpenWorldRefusal.SLOT_PINNED


class OpenWorldInvocationManifestError(StorageError, OpenWorldRefusalError):
    """The invocation does not identify a session in the frozen manifest."""

    refusal = OpenWorldRefusal.MANIFEST_CONFLICT


class OpenWorldQualificationAuthorityStaleError(StorageError, OpenWorldRefusalError):
    """Durable qualification authority is invalid or no longer current."""

    refusal = OpenWorldRefusal.AUTHORITY_STALE


class OpenWorldInvocationIntegrityError(
    OptimizationArtifactIntegrityError, OpenWorldRefusalError
):
    """An invocation write refused for a reason that is not a conflict or a lease.

    Malformed or non-canonical input, a digest that does not match its bytes,
    an unwritable governance subject vector, or a refusal message the
    classifier does not know. A subclass of the storage integrity error, so every
    existing ``except OptimizationArtifactIntegrityError`` still matches.
    """

    refusal = OpenWorldRefusal.INTEGRITY


#: The exception each refusal an invocation write can report is raised as. The
#: refusals not here (governance, staleness, bundle schema, rehydration) come
#: from other layers, never from a backend's invocation error mapping.
INVOCATION_ERROR_BY_REFUSAL: dict[OpenWorldRefusal, type[StorageError]] = {
    OpenWorldRefusal.LEASE_LOST: OpenWorldInvocationLeaseConflictError,
    OpenWorldRefusal.MANIFEST_CONFLICT: OpenWorldInvocationManifestError,
    OpenWorldRefusal.IDENTITY_BOUND: OpenWorldInvocationIdentityBoundError,
    OpenWorldRefusal.INVOCATION_CONFLICT: OpenWorldInvocationConflictError,
    OpenWorldRefusal.AUTHORITY_STALE: OpenWorldQualificationAuthorityStaleError,
    OpenWorldRefusal.INTEGRITY: OpenWorldInvocationIntegrityError,
}


class OpenWorldOutcomeContractUnavailableError(StorageError):
    """A persisted outcome contract cannot be rebuilt into its descriptor.

    Every condition that raises this is a broken server-side invariant -- a
    missing contract row the FK is supposed to make impossible, a contract with
    no surviving session outcome to recover ``source_id`` from, a source that
    does not hash to the stored ``source_digest``, or an ``allowed_values``
    that is not the canonical three. It is a ``StorageError`` subclass on
    purpose: ``OpenWorldPhase3Runner.run`` already maps ``StorageError`` to
    ``infrastructure_failure``, which is the accurate and retryable reason. A
    bare ``ValueError`` would land on the ``analyst_unqualified`` handler and
    report a false reason to the user.
    """


class OpenWorldPublicationRetryIdentityMismatchError(StorageError):
    """The committed result does not belong to the presented durable claim."""


class OpenWorldPublicationIncumbentChangedError(StorageError):
    """The live incumbent no longer matches the frozen publication input."""


@dataclass(frozen=True)
class OpenWorldEvidenceStorageRecord:
    """Rows returned from one atomic freeze or validated load."""

    job: PlaybookOptimizationJob
    artifact: PlaybookOptimizationArtifact
    claim: OptimizationJobClaim | None = None


@dataclass(frozen=True)
class OpenWorldInvocationRow:
    """One durable row of ``offline_tuner_open_world_invocations``."""

    job_id: int
    analyst_input_identity: str
    state: Literal["prepared", "completed"]
    component_identity_digest: str
    input_digest: str
    component_role: Literal["discovery", "candidate", "held_out"]
    manifest_role: Literal["held_out", "safety_only"] | None
    user_id: str | None
    session_id: str | None
    qualification_result_digest: str
    trajectory_digest: str | None
    candidate_artifact_digest: str | None
    candidate_analyst_input_identity: str | None
    output_json: str | None
    output_digest: str | None
    observed_provider: str | None
    observed_model_revision: str | None
    called_at: int | None
    latency_ms: int | None
    usage_json: str | None


@dataclass(frozen=True)
class OpenWorldAttemptContextRow:
    """One durable row of ``offline_tuner_open_world_attempt_context``."""

    job_id: int
    bundle_digest: str
    outcome_contract_digest: str
    context_json: str
    context_digest: str
    created_at: int
    qualification_authority_epoch: int | None = None
    qualification_authority_digest: str | None = None


@dataclass(frozen=True)
class OpenWorldQualificationAuthorityRow:
    """The singleton carrying current open-world qualification authority."""

    epoch: int
    state: Literal["active", "unavailable"]
    authority_digest: str
    discovery_component_identity_digest: str | None
    discovery_qualification_suite_digest: str | None
    discovery_qualification_result_digest: str | None
    held_out_component_identity_digest: str | None
    held_out_qualification_suite_digest: str | None
    held_out_qualification_result_digest: str | None
    candidate_generator_identity_digest: str | None
    candidate_generator_authorization_digest: str | None
    unavailable_reason: str | None
    updated_at: int
    method_authority_json: str | None = None


@dataclass(frozen=True)
class OpenWorldSessionMemoRow:
    """One durable row of ``offline_tuner_open_world_session_memos``."""

    job_id: int
    manifest_role: Literal["held_out", "safety_only"]
    user_id: str
    session_id: str
    trajectory_digest: str
    candidate_digest: str
    candidate_analyst_input_identity: str
    bundle_digest: str
    qualification_result_digest: str
    component_identity_digest: str
    content_json: str
    content_digest: str
    created_at: int


@dataclass(frozen=True)
class OnlineSessionMemoRow:
    """One durable row of ``user_playbook_online_session_memos``.

    Distinct from :class:`OpenWorldSessionMemoRow`, which is the OFFLINE
    held-out/safety memo keyed by ``(job_id, manifest_role, user, session)``.
    This one is keyed by the deployment lifecycle and carries
    ``trajectory_digest`` in its identity, so a session that is RE-finalized on
    a longer trajectory earns a fresh memo rather than inheriting the earlier
    revision's verdict.

    It lives with the storage records because the loader that produces it must
    not import a tuner module beyond this one.
    """

    deployment_lifecycle_id: int
    successor_fingerprint: str
    finalized_session_id: str
    trajectory_digest: str
    analyst_input_identity: str
    finalized_user_id: str
    control: str
    outcome_id: str
    outcome_revision: int
    outcome_contract_digest: str
    finalized_trajectory_digest: str
    content_json: str
    content_digest: str
    created_at: int


__all__ = [
    "OPEN_WORLD_ATTEMPT_DECISION_ARTIFACT_KIND",
    "OPEN_WORLD_CANDIDATE_ARTIFACT_KIND",
    "OPEN_WORLD_DISCOVERY_MEMO_ARTIFACT_KIND",
    "OPEN_WORLD_EVIDENCE_ARTIFACT_KIND",
    "OPEN_WORLD_EVIDENCE_OPTIMIZER_KIND",
    "INVOCATION_ERROR_BY_REFUSAL",
    "OnlineSessionMemoRow",
    "OpenWorldInvocationIntegrityError",
    "OpenWorldEvidenceGovernanceError",
    "OpenWorldEvidenceLeaseConflictError",
    "OpenWorldEvidenceStorageRecord",
    "OpenWorldAttemptContextRow",
    "OpenWorldEvidenceTargetStaleError",
    "OpenWorldInvocationConflictError",
    "OpenWorldInvocationLeaseConflictError",
    "OpenWorldInvocationManifestError",
    "OpenWorldInvocationRow",
    "OpenWorldOutcomeContractUnavailableError",
    "OpenWorldQualificationAuthorityRow",
    "OpenWorldPublicationIncumbentChangedError",
    "OpenWorldPublicationRetryIdentityMismatchError",
    "OpenWorldQualificationAuthorityStaleError",
    "OpenWorldSessionMemoRow",
    "StorageError",
]
