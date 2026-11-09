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
from typing import Literal, NamedTuple

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

#: The terminal outcomes an open-world ``failed`` stage advance accepts: every
#: attempt failure reason plus ``abandoned``, which the freeze stamps on an
#: expired job of another identity before retaking its target. Any other
#: outcome leaves the job untouched and the advance returns ``False``. The
#: PostgreSQL stage-advance function restates this set as SQL literals; the
#: shared port contract holds both backends to it.
OPEN_WORLD_FAILED_TERMINAL_OUTCOMES: frozenset[str] = frozenset(
    {
        "infrastructure_failure",
        "analyst_unqualified",
        "stale_incumbent",
        "governance_invalidated",
        "regeneration_fenced",
        "invocation_slot_pinned",
        "bundle_schema_unsupported",
        "abandoned",
        "evidence_manifest_conflict",
    }
)

#: The terminal outcomes an open-world ``abstained`` stage advance accepts,
#: under the same contract as :data:`OPEN_WORLD_FAILED_TERMINAL_OUTCOMES`.
OPEN_WORLD_ABSTAINED_TERMINAL_OUTCOMES: frozenset[str] = frozenset(
    {
        "no_grounded_hypothesis",
        "evidence_view_incomplete",
        "analyst_unqualified",
        "heldout_evidence_failed",
        "stale_incumbent",
        "governance_invalidated",
    }
)


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


# ===========================================================================
# Sweep progress: the queue's admission rules and the rows it returns
# ===========================================================================
#
# Every storage backend applies these rules in its sweep-queue read, and the
# sweep (``sweep_progress``) relies on them. They live here, with the storage
# records, so a backend can import them without importing the sweep, which
# resolves storage itself. A backend whose rules are SQL literals (Postgres)
# cannot import them; a cross-backend contract test holds it to these values.

# ---------------------------------------------------------------------------
# THE ONE RETRY-ELIGIBLE OUTCOME
# ---------------------------------------------------------------------------
#
# Every other tick outcome spends the playbook's day. This one does not, and it
# is the only one that does not.
#
# WHY IT HAD TO BE ITS OWN STATE. Both obvious answers for a rate-limited tick
# are wrong, and they are wrong in opposite directions:
#
#   * Write an ordinary day-bucket record and the playbook is skipped until
#     00:00 UTC over a provider condition that typically clears in one window.
#     The day is spent on an attempt that was never answered.
#   * Write nothing and the queue read -- whose whole predicate is "no progress
#     record today" -- returns the same playbook on the very next tick, every
#     tick, and the sweep takes ONE playbook per tick, so the entire org's
#     queue stalls behind it.
#
# So the record is written (the queue moves on) and carries a state that lets
# the row STOP blocking once a cooldown has elapsed: attempted, not completed,
# retry eligible.
#
# WHY DISPATCHING ON THE LABEL IS ACCEPTABLE HERE. `20260911020000` gave
# `outcome` no vocabulary CHECK on the express ground that "a column nothing
# dispatches on" must not acquire one -- and this makes exactly one read
# dispatch on exactly one value. That is a narrower coupling than a CHECK: a
# new outcome label still needs no migration, and an unrecognised label simply
# spends the day, which is the safe direction. What it DOES create is a
# dual-expression identity between this constant and the literal inside the
# storage SQL, which must be asserted by a test rather than kept in step by
# hand.
RETRY_ELIGIBLE_OUTCOME = "rate_limited"

#: How long a rate-limited record blocks its playbook before the queue may
#: return it.
#:
#: Not a provider-derived number -- MiniMax publishes no `Retry-After` and no
#: 429 contract, so there is nothing to derive it from. It is a QUEUE-FAIRNESS
#: number: at the 120s tick interval, 900s is roughly seven ticks in which
#: other playbooks get their turn, which is what stops one rate-limited
#: playbook monopolising a serialized queue. Shorter and the stall returns;
#: much longer and a limit that clears inside the hour loses the day anyway.
RATE_LIMIT_RETRY_COOLDOWN_SECONDS = 900

#: Ceiling on how many ticks ONE playbook may consume in one UTC day.
#:
#: This is the money bound, and without it the cooldown above is an unbounded
#: spend authorisation: a provider rate-limited all day would hand the same
#: playbook a fresh attempt every 15 minutes, ~96 times, each one re-paying for
#: whichever stages completed before the limit was hit. Three means the first
#: attempt plus at most two retries.
MAX_SAME_DAY_SWEEP_ATTEMPTS = 3

# ---------------------------------------------------------------------------
# THE EVIDENCE-GATED RE-ATTEMPT
# ---------------------------------------------------------------------------
#
# The second exception to "a record spends the day", enforced by the storage
# backend's sweep-queue read. A playbook swept minutes
# after its first exposure records `ineligible_*` for want of judged sessions;
# without this it waits until 00:00 UTC however much traffic arrives. The row
# stops blocking once the cooldown below has elapsed AND a session that exposed
# the playbook has a newer outcome or a newer request. The new-request arm is
# what keeps it from deadlocking: the outcome bridge judges on behalf of the
# tick's playbook, so a spent playbook's sessions need never get a new outcome.
#
# The cap is MAX_SAME_DAY_SWEEP_ATTEMPTS, SHARED with the rate-limited
# exception rather than a second budget: `attempt_count` counts every record of
# the day whatever its outcome. Like RETRY_ELIGIBLE_OUTCOME, both values are
# restated as literals in a storage backend's SQL, which must agree with them.

#: Outcome-label prefix whose rows may be re-attempted the same day. The sweep
#: writes ``ineligible_<reason>`` from this (``SWEEP_OUTCOME_INELIGIBLE_PREFIX``
#: aliases it), and the queue read matches it with ``starts_with``.
EVIDENCE_REATTEMPT_PREFIX = "ineligible_"

#: How long an ``ineligible_*`` record blocks its playbook before new evidence
#: may re-admit it. Shorter than the rate-limit cooldown because nothing
#: external has to clear -- the evidence gate already requires new activity --
#: so this only stops one busy session re-admitting its playbook every tick.
EVIDENCE_REATTEMPT_COOLDOWN_SECONDS = 300

#: How far BEFORE an ``ineligible_*`` attempt a request still counts as new
#: evidence. The attempt's bridge pass only judges sessions that are READY --
#: a reaction settled ``REACTION_SETTLE_SECONDS`` (30), or the session idle for
#: the judge's floor ``GROUP_EVALUATION_DELAY_SECONDS`` (600) -- so a session
#: whose last request landed less than 600s before the attempt was still
#: PENDING at it: invisible to that attempt, judgeable later. Without this the
#: rule only counts activity AFTER the attempt, and a sweep that lands seconds
#: after the last turn of sessions that then end loses the playbook its day.
#: 600 is the longest pending case; it also covers the 30s settle and the 300s
#: re-judge interval. Restated as a literal in a storage backend's SQL, which
#: must agree with it.
EVIDENCE_REATTEMPT_LOOKBACK_SECONDS = 600

#: The ordering key a playbook that has NEVER had a tick carries.
#:
#: The SQL orders ``NULLS FIRST`` because a never-attempted playbook is the
#: oldest thing there is. A sentinel BELOW every real epoch is what makes plain
#: tuple comparison in Python reproduce that ordering exactly -- the merge in
#: the daemon is a ``min`` over entries from several orgs, and ``None`` would
#: make that raise ``TypeError`` while ``0`` would collide with a (legal, if
#: never-observed) stored ``attempted_at`` of zero. Anything a caller sorts by
#: therefore sorts the same way the database did.
NEVER_ATTEMPTED_EPOCH = -1


class SweepQueueEntry(NamedTuple):
    """One queue position: the playbook, and the key it is ordered by.

    The KEY is carried, not just the id, and that is the whole point of this
    type. ``run_open_world_sweep(org)`` is per-org, so a caller holding only
    ids can do no better than visit orgs in some fixed sequence -- which serves
    whichever org it reaches first every night and starves the rest forever,
    the exact failure fleet-wide oldest-attempt-first ordering exists to
    prevent. Storage is per-org tenant schema, so there is no single global
    query to ask instead; the ordering key has to leave this function for the
    merge to happen anywhere at all.
    """

    playbook_id: int
    last_attempt_epoch: int


class SweepProgressRecord(NamedTuple):
    """The progress row as stored after one sweep tick was recorded.

    ``outcome`` is the trimmed label storage kept, ``job_id`` the surviving job
    (a re-record without one keeps the earlier one), and ``attempted_at`` the
    epoch second of the ONE clock read that also chose ``progress_day``.
    ``progress_day`` is that instant's UTC day as an ISO ``YYYY-MM-DD`` string,
    which is what the sweep logs.
    """

    user_playbook_id: int
    progress_day: str
    outcome: str
    job_id: int | None
    attempted_at: int


__all__ = [
    "OPEN_WORLD_ABSTAINED_TERMINAL_OUTCOMES",
    "OPEN_WORLD_ATTEMPT_DECISION_ARTIFACT_KIND",
    "OPEN_WORLD_CANDIDATE_ARTIFACT_KIND",
    "OPEN_WORLD_FAILED_TERMINAL_OUTCOMES",
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
    "EVIDENCE_REATTEMPT_COOLDOWN_SECONDS",
    "EVIDENCE_REATTEMPT_LOOKBACK_SECONDS",
    "EVIDENCE_REATTEMPT_PREFIX",
    "MAX_SAME_DAY_SWEEP_ATTEMPTS",
    "NEVER_ATTEMPTED_EPOCH",
    "RATE_LIMIT_RETRY_COOLDOWN_SECONDS",
    "RETRY_ELIGIBLE_OUTCOME",
    "SweepProgressRecord",
    "SweepQueueEntry",
]
