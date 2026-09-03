from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Final, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from reflexio.defaults import DEFAULT_AGENT_VERSION

from ..common import (
    NEVER_EXPIRES_TIMESTAMP,
    BlockingIssue,
    BlockingIssueKind,
    CapturesUnknownFields,
    ToolUsed,
    sanitise_for_log,
)
from ..playbook_diagnosis import PlaybookDiagnosis
from ..validators import (
    EmbeddingVector,
    NonEmptyStr,
    PersistedSessionOutcomeSource,
    SessionOutcomeSource,
    TimeRangeValidatorMixin,
    _validate_image_url,
)
from .enums import (
    OperationStatus,
    PlaybookStatus,
    ProfileTimeToLive,
    RegularVsShadow,
    ReviewUserPlaybookReasonCode,
    SessionOutcomeFailureReason,
    SessionOutcomeKind,
    Status,
    UserActionType,
)

__all__ = [
    "NEVER_EXPIRES_TIMESTAMP",
    "BlockingIssue",
    "BlockingIssueKind",
    "ToolUsed",
    "CitationKind",
    "Citation",
    "RetrievedLearningKind",
    "RetrievedLearning",
    "LearningImpact",
    "Interaction",
    "Request",
    "UserProfile",
    "UserPlaybook",
    "ProfileChangeLog",
    "AgentPlaybook",
    "AgentSuccessEvaluationResult",
    "RetrievedLearningEvaluationResult",
    "DeleteUserProfileRequest",
    "DeleteUserProfileResponse",
    "DeleteUserInteractionRequest",
    "DeleteUserInteractionResponse",
    "DeleteRequestRequest",
    "DeleteRequestResponse",
    "DeleteSessionRequest",
    "DeleteSessionResponse",
    "SessionOutcomeRecord",
    "SetSessionOutcomeRequest",
    "SetSessionOutcomeResponse",
    "GetSessionOutcomesRequest",
    "GetSessionOutcomesResponse",
    "DeleteAgentPlaybookRequest",
    "DeleteAgentPlaybookResponse",
    "DeleteUserPlaybookRequest",
    "DeleteUserPlaybookResponse",
    "BulkDeleteResponse",
    "DeleteRequestsByIdsRequest",
    "DeleteProfilesByIdsRequest",
    "DeleteAgentPlaybooksByIdsRequest",
    "DeleteUserPlaybooksByIdsRequest",
    "ClearUserDataRequest",
    "ClearUserDataResponse",
    "InteractionData",
    "PublishUserInteractionRequest",
    "PublishUserInteractionResponse",
    "WhoamiResponse",
    "MyConfigResponse",
    "AddUserPlaybookRequest",
    "AddUserPlaybookResponse",
    "AddAgentPlaybookRequest",
    "AddAgentPlaybookResponse",
    "AddUserProfileRequest",
    "AddUserProfileResponse",
    "ProfileChangeLogResponse",
    "PublicStructuredData",
    "PublicUserPlaybook",
    "PublicAgentPlaybook",
    "user_playbook_to_public",
    "agent_playbook_to_public",
    "PublicGetUserPlaybooksResponse",
    "PublicGetAgentPlaybooksResponse",
    "PublicSearchUserPlaybookResponse",
    "PublicSearchAgentPlaybookResponse",
    "PublicUnifiedSearchResponse",
    "AgentPlaybookSnapshot",
    "AgentPlaybookUpdateEntry",
    "PlaybookAggregationChangeLog",
    "PlaybookAggregationChangeLogResponse",
    "OptimizerKind",
    "OpenWorldDeploymentLifecycleState",
    "OptimizationJobStage",
    "OptimizationTerminalOutcome",
    "OptimizationArtifactKind",
    "OptimizationJobClaim",
    "PlaybookOptimizationJob",
    "PlaybookOptimizationArtifact",
    "PlaybookOptimizationCandidate",
    "PlaybookOptimizationEvaluation",
    "PlaybookOptimizationEvent",
    "OpenWorldQualificationClass",
    "OPEN_WORLD_QUALIFICATION_CLASSES",
    "OPEN_WORLD_QUALIFICATION_RECORD_SCHEMA_VERSION",
    "OpenWorldQualificationClassCount",
    "OpenWorldQualificationRecord",
    "AgentPlaybookSourceWindow",
    "agent_playbook_to_snapshot",
    "RunPlaybookAggregationRequest",
    "RunPlaybookAggregationResponse",
    "RerunProfileGenerationRequest",
    "RerunProfileGenerationResponse",
    "ManualProfileGenerationRequest",
    "ManualProfileGenerationResponse",
    "ManualPlaybookGenerationRequest",
    "ManualPlaybookGenerationResponse",
    "RerunPlaybookGenerationRequest",
    "RerunPlaybookGenerationResponse",
    "ReviewUserPlaybookEdit",
    "ReviewUserPlaybookResult",
    "ReviewUserPlaybooksRequest",
    "ReviewUserPlaybooksResponse",
    "UpgradeProfilesRequest",
    "UpgradeProfilesResponse",
    "DowngradeProfilesRequest",
    "DowngradeProfilesResponse",
    "UpgradeUserPlaybooksRequest",
    "UpgradeUserPlaybooksResponse",
    "DowngradeUserPlaybooksRequest",
    "DowngradeUserPlaybooksResponse",
    "OperationStatusInfo",
    "GetOperationStatusRequest",
    "GetOperationStatusResponse",
    "CancelOperationRequest",
    "CancelOperationResponse",
    "AdminInvalidateCacheRequest",
    "AdminInvalidateCacheResponse",
    "LineageEvent",
    "LineageContext",
    "RecordRef",
    "LearningStatusResponse",
]


def canonicalize_artifact_json(content_json: str) -> str:
    """Validate and serialize durable artifact content using the proof contract."""
    try:
        value = json.loads(
            content_json,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {constant}")
            ),
        )
        # Imported lazily because publication's contracts reference OptimizerKind
        # from this module while defining the shared RFC 8785 encoder.
        from reflexio.server.services.playbook.publication import canonical_json_bytes

        return canonical_json_bytes(value).decode("utf-8")
    except (TypeError, ValueError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("artifact content_json must be valid JSON") from exc


# ===============================
# Data Models
# ===============================

type CitationKind = Literal["playbook", "profile", "user_playbook", "agent_playbook"]


class Citation(CapturesUnknownFields):
    """A playbook or profile item the agent cited as influential.

    Carried inline on an Assistant ``InteractionData`` row to mark
    which previously-injected playbook rule or user-profile row
    materially shaped that response. The server stores these for retrieval
    attribution and evaluation.

    Attributes:
        kind (CitationKind): Which kind of cited item this references.
            ``"playbook"`` is the legacy compatibility value.
            ``"user_playbook"`` is the direct tuner target.
            ``"agent_playbook"`` references an org-level playbook row.
            ``"profile"`` references a user profile row.
        real_id (str): Stable storage id — ``user_playbook_id`` for
            user playbooks, ``agent_playbook_id`` for agent playbooks,
            and ``profile_id`` for profiles.
        tag (str): Injection-time rank tag (e.g. ``"r1-301"``,
            ``"p1-0f37"``). Per-injection, not stable across sessions;
            kept as a debug aid.
        title (str): Short human-readable label for logs and UI.
    """

    kind: CitationKind
    real_id: str
    tag: str = ""
    title: str = ""


# Canonical kinds accepted and persisted by retrieved-learning evaluation.
# Unlike ``Citation.kind`` there is no legacy ``"playbook"`` alias.
type RetrievedLearningKind = Literal["profile", "user_playbook", "agent_playbook"]

type LearningImpact = Literal["positive", "negative", "neutral"]


class RetrievedLearning(CapturesUnknownFields):
    """A learning the caller retrieved and injected into the agent context.

    Deliberately minimal — just the identity pair. It does NOT reuse
    ``Citation``: citations carry injection-time debug fields (``tag``,
    ``title``) that callers should not need to supply (or see) when declaring
    what was retrieved.

    Attributes:
        kind (RetrievedLearningKind): Which kind of learning this references.
        learning_id (str): Stable storage id — ``profile_id`` for profiles,
            ``user_playbook_id`` for user playbooks, ``agent_playbook_id``
            for agent playbooks (numeric ids as decimal strings).
    """

    kind: RetrievedLearningKind
    learning_id: str = Field(min_length=1, max_length=1_000)


# information about the user interaction sent by the client
class Interaction(BaseModel):
    interaction_id: int = 0  # 0 = placeholder for DB auto-increment
    user_id: str
    request_id: str
    created_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))
    role: str = "User"
    content: str = ""
    token_count: int | None = Field(default=None, ge=0)
    user_action: UserActionType = UserActionType.NONE
    user_action_description: str = ""
    interacted_image_url: str = ""
    image_encoding: str = ""  # base64 encoded image
    shadow_content: str = ""
    expert_content: str = ""
    tools_used: list[ToolUsed] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    # Every learning retrieved and injected for this turn — including ones
    # that did not end up influencing the response (contrast: ``citations``
    # is the agent's claim of influence).
    retrieved_learnings: list[RetrievedLearning] = Field(default_factory=list)
    embedding: EmbeddingVector = []

    @field_validator("interacted_image_url", mode="after")
    @classmethod
    def validate_image_url(cls, v: str) -> str:
        return _validate_image_url(v)


class Request(BaseModel):
    """A user-issued request that begins or continues a session.

    A Request is the unit of work the agent reacts to. Multiple Requests
    share a ``session_id`` to form a multi-turn session.

    Attributes:
        request_id (str): Unique identifier for this request.
        user_id (str): Owner of the request.
        created_at (int): Unix epoch seconds at request creation. Defaults
            to the current UTC time.
        source (str): Producer/workflow label. Persisted reads preserve legacy
            values verbatim; new publish inputs use the strict source contract.
        agent_version (str): The agent version that handled this request.
        session_id (str): Non-empty session this request belongs to.
        evaluation_only (bool): Whether this request is stored for
            session-level evaluation only and must be excluded from
            profile/playbook learning windows.
        retrieval_experiment_id (str | None): Retrieval experiment attached
            to this request by the publishing agent.
        retrieval_experiment_arm (str | None): Deterministic user assignment
            for the attached retrieval experiment.
    """

    request_id: str
    user_id: str
    created_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))
    source: PersistedSessionOutcomeSource = ""
    agent_version: str = ""
    session_id: NonEmptyStr
    evaluation_only: bool = False
    retrieval_experiment_id: NonEmptyStr | None = None
    retrieval_experiment_arm: Literal["treatment", "holdout"] | None = None

    @model_validator(mode="after")
    def validate_retrieval_experiment_pair(self) -> Self:
        if (self.retrieval_experiment_id is None) != (
            self.retrieval_experiment_arm is None
        ):
            raise ValueError(
                "retrieval_experiment_id and retrieval_experiment_arm must be provided together"
            )
        return self


# information about the user profile generated from the user interaction
# output of the profile generation service send back to the client
class UserProfile(BaseModel):
    profile_id: str
    user_id: str
    content: str
    last_modified_timestamp: int
    generated_from_request_id: str
    profile_time_to_live: ProfileTimeToLive = ProfileTimeToLive.INFINITY
    # this is the expiration date calculated based on last modified timestamp and profile time to live instead of generated timestamp
    expiration_timestamp: int = NEVER_EXPIRES_TIMESTAMP
    custom_features: dict | None = None
    source: str | None = None
    status: Status | None = None  # indicates the status of the profile
    extractor_names: list[str] | None = (
        None  # Retained provenance data column (merged on dedup); new profiles write None.
    )
    expanded_terms: str | None = None
    tags: list[str] | None = None  # None = not yet tagged; [] = tagged, no match
    source_interaction_ids: list[int] = Field(default_factory=list)
    embedding: EmbeddingVector = []
    source_span: str | None = None
    notes: str | None = None
    reader_angle: str | None = None
    merged_into: str | None = None
    superseded_by: str | None = None


# user playbook for agents
class UserPlaybook(BaseModel):
    user_playbook_id: int = 0
    user_id: str | None = None  # optional for backward compatibility
    agent_version: str
    request_id: str
    playbook_name: str = ""
    created_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))
    content: str = ""
    trigger: str | None = None
    rationale: str | None = None
    blocking_issue: BlockingIssue | None = None
    status: Status | None = (
        None  # Status.PENDING (from rerun), None (current), Status.ARCHIVED (old)
    )
    source: str | None = None  # source of the interaction that generated this playbook
    source_interaction_ids: list[int] = Field(default_factory=list)
    expanded_terms: str | None = None
    tags: list[str] | None = None  # None = not yet tagged; [] = tagged, no match
    embedding: EmbeddingVector = []
    source_span: str | None = None
    notes: str | None = None
    reader_angle: str | None = None
    merged_into: int | None = None
    superseded_by: int | None = None
    governance_subject_ref: str | None = Field(default=None, exclude=True)
    retired_at: int | None = Field(default=None, exclude=True)


class ProfileChangeLog(BaseModel):
    id: int
    user_id: str
    request_id: str
    created_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))
    added_profiles: list[UserProfile]
    removed_profiles: list[UserProfile]


class AgentPlaybook(BaseModel):
    agent_playbook_id: int = 0
    playbook_name: str = ""
    agent_version: str
    created_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))
    content: str
    trigger: str | None = None
    rationale: str | None = None
    blocking_issue: BlockingIssue | None = None
    playbook_status: PlaybookStatus = PlaybookStatus.PENDING
    playbook_metadata: str = ""
    expanded_terms: str | None = None
    tags: list[str] | None = None  # None = not yet tagged; [] = tagged, no match
    embedding: EmbeddingVector = []
    status: Status | None = (
        None  # used for tracking intermediate states during playbook aggregation. Status.ARCHIVED for playbooks during aggregation process, None for current playbooks
    )
    merged_into: int | None = None
    superseded_by: int | None = None


# 'offline_tuner_legacy' and 'optimizer_legacy_unknown' were never written by
# running code. They were assigned to pre-existing rows by a one-time heuristic
# backfill (supabase/data/tenant/20260723000000:54-61) and are retained
# permanently as HISTORICAL labels: dropping them would make those rows violate
# the tenant CHECK and abort the contract migration.
OptimizerKind = Literal[
    "gepa",
    "offline_tuner_open_world",
    "offline_tuner_legacy",
    "optimizer_legacy_unknown",
]

OpenWorldDeploymentLifecycleState = Literal[
    "provisional", "confirmed", "restored", "displaced", "erased"
]

OptimizationJobStage = Literal[
    "evidence_frozen",
    "discovery_analyzed",
    "candidate_generated",
    "held_out_analyzed",
    "publishing",
    "applied",
    "abstained",
    "failed",
]

# Phase 7 removed the four members whose NAMES contain 'replay'. Seven more were
# reachable only through that same replay arm and are now equally dead: no
# writer, no reader, no tenant routine that can set them. They are RETAINED
# rather than removed, the way 'offline_tuner_legacy' is retained above -- and
# for the same two reasons. Removing them narrows the tenant CHECK a second
# time, and a CHECK narrowing is a one-way door this branch has already spent
# once; and historical rows written before the replay retirement still carry
# them, so a narrowed CHECK would abort the validating migration on the first
# organization holding one.
#
# Read a member of this set as "an outcome an earlier era could record", never
# as a state the current tuner can reach. Pinned by
# tests/models/test_terminal_outcome_reachability.py so it cannot silently grow:
# a new member added here is a new outcome somebody must show is WRITABLE.
#
# 'deployment_unsupported' is the trap. The same spelling is also
# OfflineTunerUnavailableReason -- a config-enablement rejection code in
# reflexio_ext capability_status.py, with ~40 live references. Those are a
# different vocabulary on a different type, so "grep says it is used" does not
# make this member reachable.
RETAINED_UNREACHABLE_TERMINAL_OUTCOMES: frozenset[str] = frozenset(
    {
        "insufficient_negative_evidence",
        "insufficient_positive_evidence",
        "insufficient_coverage",
        "deployment_unsupported",
        "candidate_regressed",
        "candidate_did_not_improve",
        "publication_failed",
    }
)

OptimizationTerminalOutcome = Literal[
    "applied",
    "insufficient_negative_evidence",
    "insufficient_positive_evidence",
    "insufficient_coverage",
    "deployment_unsupported",
    "candidate_regressed",
    "candidate_did_not_improve",
    "incumbent_changed",
    "generation_failed",
    "publication_failed",
    "governance_erased",
    "no_grounded_hypothesis",
    "analyst_unqualified",
    "heldout_evidence_failed",
    "stale_incumbent",
    "governance_invalidated",
    "infrastructure_failure",
]

OptimizationArtifactKind = Literal[
    "expected_population_manifest",
    "generation_selection",
    "candidate",
    "candidate_search_projection",
    "open_world_evidence_bundle",
    "open_world_discovery_memo",
    "open_world_candidate",
    "open_world_attempt_decision",
]

Sha256Digest = str


class OptimizationJobClaim(BaseModel):
    """One renewable optimizer lease identified by a monotonic fence."""

    job_id: int
    owner: str
    fence: int = Field(ge=1)
    expires_at: int


class PlaybookOptimizationJob(BaseModel):
    """One end-to-end optimizer run for a single playbook target.

    Lifecycle: ``pending`` → ``running`` → ``completed`` | ``failed`` |
    ``skipped``. ``best_candidate_id`` and ``successor_target_id`` are set
    when the run produces a winner or commits a successor playbook.
    """

    job_id: int = 0
    optimizer_kind: OptimizerKind = "optimizer_legacy_unknown"
    target_kind: Literal["agent_playbook", "user_playbook"]
    target_id: int
    status: Literal["pending", "running", "completed", "skipped", "failed"] = "pending"
    best_candidate_id: int | None = None
    successor_target_id: int | None = None
    decision_reason: str = ""
    metadata_json: str = "{}"
    discovery_key: str | None = None
    attempt_key: str | None = None
    lease_owner: str | None = None
    lease_fence: int = Field(default=0, ge=0)
    lease_expires_at: int | None = None
    stage: OptimizationJobStage | None = None
    terminal_outcome: OptimizationTerminalOutcome | None = None
    expected_population_manifest_digest: Sha256Digest | None = None
    generation_selection_manifest_digest: Sha256Digest | None = None
    replay_manifest_digest: Sha256Digest | None = None
    candidate_content_digest: Sha256Digest | None = None
    search_projection_digest: Sha256Digest | None = None
    publication_scope_digest: Sha256Digest | None = None
    created_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))
    updated_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))

    @field_validator(
        "expected_population_manifest_digest",
        "generation_selection_manifest_digest",
        "replay_manifest_digest",
        "candidate_content_digest",
        "search_projection_digest",
        "publication_scope_digest",
    )
    @classmethod
    def validate_sha256_digest(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError("optimizer proof digests must be lowercase SHA-256 hex")
        return value


class PlaybookOptimizationArtifact(BaseModel):
    """One typed, content-bearing singleton artifact owned by an optimizer job."""

    artifact_id: int = 0
    job_id: int
    artifact_kind: OptimizationArtifactKind
    content_json: str
    content_digest: Sha256Digest
    created_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))
    updated_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))

    @field_validator("content_json")
    @classmethod
    def canonicalize_content_json(cls, value: str) -> str:
        return canonicalize_artifact_json(value)

    @field_validator("content_digest")
    @classmethod
    def validate_content_digest(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("artifact digest must be lowercase SHA-256 hex")
        return value

    @model_validator(mode="after")
    def validate_content_digest_matches_content(self) -> Self:
        expected = sha256(self.content_json.encode()).hexdigest()
        if self.content_digest != expected:
            raise ValueError("artifact digest must match canonical content_json")
        return self


class PlaybookOptimizationCandidate(BaseModel):
    """A playbook content variant proposed by GEPA during a job.

    Multiple proposals with identical content collapse to one row (deduped
    by ``content`` inside the GEPA adapter). ``aggregate_score`` and
    ``is_winner`` are populated only for the run's chosen winner.
    """

    candidate_id: int = 0
    job_id: int
    candidate_index: int = 0
    content: str
    parent_candidate_ids: list[int] = Field(default_factory=list)
    aggregate_score: float | None = None
    is_winner: bool = False
    metadata_json: str = "{}"
    created_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))


class PlaybookOptimizationEvaluation(BaseModel):
    """Pairwise judgement of one candidate vs. the incumbent on one window.

    Both rollouts are stored as JSON for offline reproducibility.
    ``verdict='aborted'`` means the assistant backend failed and the row
    carries no useful signal — the optimizer treats any aborted evaluation
    as fatal for the run.
    """

    evaluation_id: int = 0
    job_id: int
    candidate_id: int
    target_kind: Literal["agent_playbook", "user_playbook"]
    target_id: int
    scenario_user_playbook_id: int | None = None
    source_interaction_ids: list[int] = Field(default_factory=list)
    score: float = 0.0
    verdict: Literal["candidate", "incumbent", "tie", "aborted"] = "tie"
    likert: int = Field(default=0, ge=0, le=5)
    rationale: str = ""
    asi_json: str = "{}"
    incumbent_rollout_json: str = "[]"
    candidate_rollout_json: str = "[]"
    created_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))


class PlaybookOptimizationEvent(BaseModel):
    """One GEPA callback (``on_*``) event captured for offline inspection.

    The optimizer's ``_GEPAStorageCallback`` forwards every dispatched
    callback into a row of this type. ``event_type`` is the callback name
    minus the ``on_`` prefix; ``payload_json`` is a depth-bounded
    serialization of the callback's argument.
    """

    event_id: int = 0
    job_id: int
    event_type: str
    payload_json: str = "{}"
    created_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))


OpenWorldQualificationClass = Literal[
    "citation_fidelity",
    "abstention",
    "support",
    "refutation",
    "insufficiency",
    "unsupported_causal_claim_rejection",
    "prompt_injection_resistance",
]

OPEN_WORLD_QUALIFICATION_CLASSES: Final[tuple[OpenWorldQualificationClass, ...]] = (
    "citation_fidelity",
    "abstention",
    "support",
    "refutation",
    "insufficiency",
    "unsupported_causal_claim_rejection",
    "prompt_injection_resistance",
)

OPEN_WORLD_QUALIFICATION_RECORD_SCHEMA_VERSION: Final[str] = (
    "offline-tuner-open-world-qualification-result-v1"
)


def _validate_lowercase_sha256(label: str, value: str) -> str:
    """Return ``value`` when it is a lowercase SHA-256 hex digest.

    Args:
        label (str): Field name used in the raised error message.
        value (str): Candidate digest.

    Returns:
        str: The validated digest.

    Raises:
        ValueError: If ``value`` is not 64 lowercase hex characters.
    """
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be lowercase SHA-256 hex")
    return value


class OpenWorldQualificationClassCount(BaseModel):
    """Diagnostic required/passed counts for one safety-critical class.

    Counts are never combined into a score: pass-all qualification is decided
    by the reducer, and these values exist only to explain one result.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    qualification_class: OpenWorldQualificationClass
    required: int = Field(ge=0)
    passed_required: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_passed_within_required(self) -> Self:
        if self.passed_required > self.required:
            raise ValueError("qualification passed_required may not exceed required")
        return self


class OpenWorldQualificationRecord(BaseModel):
    """One immutable pass-all qualification result for an analyst identity.

    The record carries no customer data or model output: only the pinned
    component identity, the suite it was measured against, the canonical
    result digest, per-class diagnostic counts for every one of the seven
    safety-critical classes in canonical order, and the sorted, unique
    digests of the observations that produced it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["offline-tuner-open-world-qualification-result-v1"] = (
        "offline-tuner-open-world-qualification-result-v1"
    )
    component_identity_digest: Sha256Digest
    suite_digest: Sha256Digest
    result_digest: Sha256Digest
    class_counts: tuple[OpenWorldQualificationClassCount, ...]
    passed: bool
    observation_digests: tuple[Sha256Digest, ...] = ()
    created_at: int = Field(
        default_factory=lambda: int(datetime.now(UTC).timestamp()), ge=0
    )

    @field_validator("component_identity_digest", "suite_digest", "result_digest")
    @classmethod
    def validate_identity_digests(cls, value: str) -> str:
        return _validate_lowercase_sha256("qualification digest", value)

    @field_validator("class_counts")
    @classmethod
    def validate_class_counts(
        cls, value: tuple[OpenWorldQualificationClassCount, ...]
    ) -> tuple[OpenWorldQualificationClassCount, ...]:
        observed = tuple(count.qualification_class for count in value)
        if observed != OPEN_WORLD_QUALIFICATION_CLASSES:
            raise ValueError(
                "qualification class_counts must list every safety-critical "
                "class exactly once in canonical order"
            )
        return value

    @field_validator("observation_digests")
    @classmethod
    def validate_observation_digests(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for digest in value:
            _validate_lowercase_sha256("qualification observation digest", digest)
        if list(value) != sorted(set(value)):
            raise ValueError(
                "qualification observation digests must be sorted and unique"
            )
        return value

    @model_validator(mode="after")
    def validate_passed_requires_all_class_counts(self) -> Self:
        if self.passed and any(
            count.passed_required != count.required for count in self.class_counts
        ):
            raise ValueError(
                "qualification passed=true requires every class to pass required"
            )
        return self

    def semantic_key(self) -> tuple[Any, ...]:
        """Return the conflict-detection identity, excluding ``created_at``."""
        return (
            self.schema_version,
            self.component_identity_digest,
            self.suite_digest,
            self.result_digest,
            tuple(
                (count.qualification_class, count.required, count.passed_required)
                for count in self.class_counts
            ),
            self.passed,
            self.observation_digests,
        )


class AgentPlaybookSourceWindow(BaseModel):
    """Replayable source window snapshotted when an agent playbook is generated."""

    user_playbook_id: int
    source_interaction_ids: list[int] = Field(default_factory=list)


class AgentSuccessEvaluationResult(BaseModel):
    result_id: int = 0
    user_id: str = ""
    agent_version: str
    session_id: str
    is_success: bool
    failure_type: str | None = None
    failure_reason: str | None = None
    evaluation_name: str | None = None
    created_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))
    regular_vs_shadow: RegularVsShadow | None = None
    number_of_correction_per_session: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of user turns in the session that corrected or redirected an "
            "earlier agent response or action."
        ),
    )
    user_turns_to_resolution: int | None = None
    is_escalated: bool = False
    tags: list[str] | None = None
    embedding: EmbeddingVector = []


class RetrievedLearningEvaluationResult(BaseModel):
    """Latest per-learning relevance/impact verdict for one interaction.

    New rows are unique per ``(user_id, session_id, interaction_id, kind,
    learning_id)``. Nullable interaction fields preserve read compatibility
    with legacy session-level rows while the table continues to hold only the
    most recent successfully persisted evaluation set for a session.

    Attributes:
        result_id (int): DB auto-increment identifier (0 = placeholder).
        user_id (str): Session owner.
        session_id (str): Evaluated session.
        agent_version (str): Version supplied to group evaluation;
            informational, not part of the uniqueness key.
        interaction_id (int | None): Interaction that received the learning.
            ``None`` only for legacy session-level rows.
        interaction_created_at (int | None): Timestamp of the target
            interaction. ``None`` only for legacy session-level rows.
        kind (RetrievedLearningKind): The learning kind.
        learning_id (str): Stable storage id, matching
            ``RetrievedLearning.learning_id``.
        is_relevant (bool | None): Whether the learning applies to its target
            interaction. ``None`` only when the relevance judge/chunk failed.
        relevance_reason (str): Judge reasoning; empty when ``is_relevant``
            is ``None``.
        impact (LearningImpact | None): Whether the learning improved,
            harmed, or did not materially change the response. ``None`` only
            when the impact judge/chunk failed.
        impact_reason (str): Judge reasoning; empty when ``impact`` is
            ``None``.
        created_at (int): Earliest request timestamp in the evaluated
            session.
    """

    result_id: int = 0
    user_id: str
    session_id: str
    agent_version: str = ""
    interaction_id: int | None = None
    interaction_created_at: int | None = None
    kind: RetrievedLearningKind
    learning_id: str
    is_relevant: bool | None = None
    relevance_reason: str = ""
    impact: LearningImpact | None = None
    impact_reason: str = ""
    diagnosis: PlaybookDiagnosis | None = None
    evaluated_playbook_digest: str | None = None
    diagnosis_evidence_complete: bool = False
    created_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))


class LineageEvent(BaseModel):
    """Append-only, content-free provenance record. NEVER carries content/PII.

    Attributes:
        event_id (int): PK assigned by storage (0 = not yet persisted).
        org_id (str): Owning org (tenant) — required for RLS / isolation.
        entity_type (str): One of "profile" | "user_playbook" | "agent_playbook".
        entity_id (str): The affected record's id, stringified (profile_id is str).
        op (str): create|revise|merge|aggregate|archive|soft_delete|hard_delete|purge|status_change.
        prov_relation (str): W3C PROV relation (see spec §14).
        source_ids (list[str]): Records merged/superseded into entity_id.
        actor (str): Who/what triggered it (consolidator|offline_optimizer|...).
        request_id (str): Triggering request — part of the idempotency key.
        reason (str): Free-text rationale (no PII).
        created_at (int): Unix epoch seconds (0 = unset; storage stamps it).
        from_status (str | None): Status before a transition.
        to_status (str | None): Status after a transition.
        status_namespace (str | None): Namespace for status values.
        model_name (str | None): Observed model for a content-shaping operation.
        provider (str | None): Observed provider for that operation.
    """

    event_id: int = 0
    org_id: str
    entity_type: str
    entity_id: str
    op: str
    prov_relation: str = ""
    source_ids: list[str] = []
    actor: str = ""
    request_id: str = ""
    reason: str = ""
    created_at: int = 0
    from_status: str | None = None
    to_status: str | None = None
    status_namespace: str | None = None
    model_name: str | None = None
    provider: str | None = None


class LineageContext(BaseModel):
    """Caller-supplied intent the storage layer can't infer.

    Required for merge/supersede/aggregate; optional for create/revise/archive.
    """

    op_kind: str
    actor: str = ""
    source_ids: list[str] = []
    reason: str = ""
    request_id: str | None = None
    model_name: str | None = None
    provider: str | None = None


class RecordRef(BaseModel):
    """Result of resolve_current — the live survivor's id and whether its body was purged.

    Attributes:
        id: Primary key of the live survivor record.
        is_purged: True when the survivor's content body has been blanked by
            ``purge_content`` (GDPR/erasure).  Any consumer that dereferences the
            resolved record's content MUST treat ``is_purged=True`` as "erased —
            skip or treat as absent."  Reading blank content as if it were valid
            is a silent data-quality bug.
    """

    id: str
    is_purged: bool = False


# ===============================
# Request Models
# ===============================


# delete user profile request
class DeleteUserProfileRequest(BaseModel):
    user_id: NonEmptyStr
    profile_id: str = ""
    search_query: str = ""


# delete user profile response
class DeleteUserProfileResponse(BaseModel):
    success: bool
    message: str = ""


# delete user interaction request
class DeleteUserInteractionRequest(BaseModel):
    user_id: NonEmptyStr
    interaction_id: int = Field(gt=0)


# delete user interaction response
class DeleteUserInteractionResponse(BaseModel):
    success: bool
    message: str = ""


# delete request request
class DeleteRequestRequest(BaseModel):
    request_id: NonEmptyStr


# delete request response
class DeleteRequestResponse(BaseModel):
    success: bool
    message: str = ""


# delete session request
class DeleteSessionRequest(BaseModel):
    session_id: NonEmptyStr


# delete session response
class DeleteSessionResponse(BaseModel):
    success: bool
    message: str = ""
    deleted_requests_count: int = 0


class SessionOutcomeRecord(BaseModel):
    """Persisted outcome row, including its immutable historical source."""

    outcome_id: NonEmptyStr | None = None
    outcome_revision: int | None = Field(default=None, ge=1)
    user_id: str
    session_id: NonEmptyStr
    outcome: SessionOutcomeKind
    occurred_at: int = Field(ge=0)
    source: PersistedSessionOutcomeSource
    label: str | None = Field(default=None, max_length=128)
    value: float | None = Field(default=None, allow_inf_nan=False)
    metadata: dict[str, Any] | None = None
    outcome_contract_digest: Sha256Digest | None = None
    finalized_trajectory_digest: Sha256Digest | None = None
    created_at: int = Field(ge=0)

    @field_validator("outcome_contract_digest", "finalized_trajectory_digest")
    @classmethod
    def validate_sha256_digest(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError("outcome identity digests must be lowercase SHA-256 hex")
        return value

    @model_validator(mode="after")
    def validate_identity_shape(self) -> Self:
        identity = (
            self.outcome_id,
            self.outcome_revision,
            self.outcome_contract_digest,
            self.finalized_trajectory_digest,
        )
        if any(value is None for value in identity) and not all(
            value is None for value in identity
        ):
            raise ValueError(
                "outcome identity fields must be all populated or all null"
            )
        return self


class SetSessionOutcomeRequest(CapturesUnknownFields):
    session_id: NonEmptyStr
    outcome: SessionOutcomeKind
    occurred_at: int = Field(ge=0)
    label: str | None = Field(default=None, max_length=128)
    value: float | None = Field(default=None, allow_inf_nan=False)
    metadata: dict[str, Any] | None = None

    @field_validator("label")
    @classmethod
    def _strip_non_empty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped

    @field_validator("metadata")
    @classmethod
    def _bound_metadata(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        try:
            encoded = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must contain only valid JSON values") from exc
        if len(encoded) > 16 * 1024:
            raise ValueError("metadata must encode to at most 16384 bytes")
        return value


class SetSessionOutcomeResponse(BaseModel):
    success: bool
    recorded: bool = False
    reason: SessionOutcomeFailureReason | None = None
    message: str = ""
    user_id: str | None = None
    source: PersistedSessionOutcomeSource | None = None
    outcome_id: NonEmptyStr | None = None
    outcome_revision: int | None = Field(default=None, ge=1)
    outcome_contract_digest: Sha256Digest | None = None
    finalized_trajectory_digest: Sha256Digest | None = None

    @field_validator("outcome_contract_digest", "finalized_trajectory_digest")
    @classmethod
    def validate_sha256_digest(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError("outcome identity digests must be lowercase SHA-256 hex")
        return value

    @model_validator(mode="after")
    def validate_identity_shape(self) -> Self:
        identity = (
            self.outcome_id,
            self.outcome_revision,
            self.outcome_contract_digest,
            self.finalized_trajectory_digest,
        )
        if any(value is None for value in identity) and not all(
            value is None for value in identity
        ):
            raise ValueError(
                "outcome identity fields must be all populated or all null"
            )
        return self


class GetSessionOutcomesRequest(CapturesUnknownFields):
    session_ids: list[NonEmptyStr] | None = Field(default=None, max_length=100)
    user_id: str | None = None
    source: SessionOutcomeSource | None = None
    outcome: SessionOutcomeKind | None = None
    label: str | None = None
    start_time: int | None = Field(default=None, ge=0)
    end_time: int | None = Field(default=None, ge=0)
    top_k: int = Field(default=100, ge=1, le=1_000)
    offset: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_range_and_ids(self) -> Self:
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.start_time > self.end_time
        ):
            raise ValueError("start_time must be less than or equal to end_time")
        if self.session_ids:
            self.session_ids = list(dict.fromkeys(self.session_ids))
        return self


class GetSessionOutcomesResponse(BaseModel):
    success: bool
    session_outcomes: list[SessionOutcomeRecord] = Field(default_factory=list)
    message: str = ""


# delete agent playbook request
class DeleteAgentPlaybookRequest(BaseModel):
    agent_playbook_id: int = Field(gt=0)


# delete agent playbook response
class DeleteAgentPlaybookResponse(BaseModel):
    success: bool
    message: str = ""


# delete user playbook request
class DeleteUserPlaybookRequest(BaseModel):
    user_playbook_id: int = Field(gt=0)


# delete user playbook response
class DeleteUserPlaybookResponse(BaseModel):
    success: bool
    message: str = ""


class BulkDeleteResponse(BaseModel):
    success: bool
    deleted_count: int = 0
    message: str = ""


class DeleteRequestsByIdsRequest(BaseModel):
    request_ids: list[str] = Field(min_length=1, max_length=10_000)


class DeleteProfilesByIdsRequest(BaseModel):
    profile_ids: list[str] = Field(min_length=1, max_length=10_000)


class DeleteAgentPlaybooksByIdsRequest(BaseModel):
    agent_playbook_ids: list[int] = Field(min_length=1, max_length=10_000)


class DeleteUserPlaybooksByIdsRequest(BaseModel):
    user_playbook_ids: list[int] = Field(min_length=1, max_length=10_000)


# Clear all data scoped to a single user_id (session outcomes, interactions,
# requests, user playbooks, profiles). Used by paired-protocol harnesses (e.g. SWE-bench) to
# isolate per-task data on a shared storage backend without nuking sibling
# tasks' rows. Intentionally does NOT touch agent_playbooks — they are the
# cross-project rollup of skills and have no user_id column.
class ClearUserDataRequest(BaseModel):
    user_id: NonEmptyStr


class ClearUserDataResponse(BaseModel):
    success: bool
    deleted_counts: dict[str, int] = Field(default_factory=dict)
    message: str | None = None


# user provided interaction data from the request
class InteractionData(CapturesUnknownFields):
    created_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))
    role: str = Field(default="User", max_length=1_000)
    content: str = Field(default="", max_length=1_000_000)
    shadow_content: str = Field(default="", max_length=1_000_000)
    expert_content: str = Field(default="", max_length=1_000_000)
    user_action: UserActionType = UserActionType.NONE
    user_action_description: str = Field(default="", max_length=10_000)
    interacted_image_url: str = Field(default="", max_length=2_048)
    image_encoding: str = Field(
        default="", max_length=15_000_000
    )  # base64 encoded image
    tools_used: list[ToolUsed] = Field(default_factory=list, max_length=1_000)
    citations: list[Citation] = Field(default_factory=list, max_length=1_000)
    # Learnings (profiles / user playbooks / agent playbooks) the caller
    # retrieved and injected into the agent context for this turn. Distinct
    # from ``citations`` (agent-claimed influence): this is everything that
    # was injected, whether or not it helped.
    retrieved_learnings: list[RetrievedLearning] = Field(
        default_factory=list, max_length=1_000
    )

    @field_validator("interacted_image_url", mode="after")
    @classmethod
    def validate_image_url(cls, v: str) -> str:
        return _validate_image_url(v)

    def reportable_unknown_fields(self) -> list[str]:
        """Unknown key names worth telling the caller about, nested included.

        Excludes ``_BENIGN_UNKNOWN_KEYS`` -- by LEAF name, so a duplicated
        ``user_id`` is suppressed identically whether it sits at the top level
        or on ``tools_used[0]``. Nested models report as a path, e.g.
        ``tools_used[0].stat``, because a caller matching only top-level names
        would not recognise the format otherwise.

        Sorted, because only the first few names survive the render cap: an
        unsorted list puts every top-level name ahead of every nested one, so a
        payload with five top-level typos could never show a nested path at all.

        Returns:
            list[str]: Sorted field paths, empty when nothing unrecognised was
                sent.
        """
        names = [
            name
            for name in self.unknown_field_names()
            if name not in _BENIGN_UNKNOWN_KEYS
        ]
        for attribute in ("tools_used", "citations", "retrieved_learnings"):
            for index, item in enumerate(getattr(self, attribute)):
                names.extend(
                    f"{attribute}[{index}].{nested}"
                    for nested in item.unknown_field_names()
                    if nested not in _BENIGN_UNKNOWN_KEYS
                )
        return sorted(names)

    def carries_content(self) -> bool:
        """
        Whether this interaction carries anything worth storing.

        The single source of truth for "is this interaction empty", shared by
        ``PublishUserInteractionRequest``'s boundary validator and the
        ``precondition_checks`` defense-in-depth guard so the two cannot drift.

        Every content-bearing field counts, not just ``content``: a
        tool-call-only agent turn, a shadow/expert-only row, and an image-only
        turn all carry real information. Note ``user_action`` is compared
        against ``UserActionType.NONE`` rather than tested for truthiness --
        ``UserActionType`` is a ``StrEnum`` whose NONE member is the *truthy*
        string ``"none"``, and a truthiness test there is what silently
        disabled this check for the entire life of the guard.

        Returns:
            bool: True if any content-bearing field is populated.
        """
        if self.user_action != UserActionType.NONE:
            return True
        # Text fields are stripped: "   " is not content. Mirrors the
        # ``session_id`` guard in ``precondition_checks``, which already
        # rejects whitespace-only values.
        return any(
            value.strip() if isinstance(value, str) else value
            for value in (getattr(self, name) for name in CONTENT_BEARING_FIELD_NAMES)
        )

    def shape_error(self) -> str | None:
        """A contradiction in this interaction the caller must fix, or None.

        These are genuine caller mistakes with no sensible recovery, so they
        are fatal to the request. Emptiness is deliberately NOT one of them --
        see ``PublishUserInteractionRequest.validate_interaction_shapes``.

        Both rules used to live only in ``precondition_checks``, which on the
        default ``wait_for_response=False`` path runs inside a background task
        whose result is discarded, so neither was ever reportable. Keeping them
        here lets the request model raise a 422 on both paths while
        ``precondition_checks`` delegates, so the two cannot diverge.

        Returns:
            str | None: A caller-facing reason, or None when acceptable.
        """
        if self.user_action != UserActionType.NONE and not self.user_action_description:
            return "user_action requires a user_action_description"
        if self.interacted_image_url and self.image_encoding:
            return "interacted_image_url and image_encoding cannot both be set"
        return None


# Every field ``carries_content`` consults, in one place. The prose in the
# error message is DERIVED from this tuple rather than hand-written beside it,
# so adding a field to the predicate cannot leave the caller-facing list stale.
CONTENT_BEARING_FIELD_NAMES: Final = (
    "content",
    "shadow_content",
    "expert_content",
    "interacted_image_url",
    "image_encoding",
    "tools_used",
    "citations",
    "retrieved_learnings",
)

# Request-level identifiers callers routinely duplicate onto each interaction.
# Still stripped, but never warned about: both plugins send `user_id` on every
# turn, so warning would emit one entry per interaction on every correct publish
# and train operators to ignore the channel before it carries real signal.
_BENIGN_UNKNOWN_KEYS: Final = frozenset({"user_id", "session_id"})

# How many original indices to name when reporting skipped empty rows.
_MAX_SKIPPED_INDICES: Final = 10

# Unknown key names are caller-controlled: unbounded in length and count, and
# free to contain newlines. They are echoed into both the HTTP response and a
# log line, so they need bounding on THREE axes -- per-name length, names per
# interaction, and total entries in the warning list -- plus control-character
# stripping. Unbounded, 1000 interactions x N long bogus keys produced a
# ~350 KB response body and one enormous log record.
_MAX_REPORTED_NAMES: Final = 5
_MAX_WARNING_ENTRIES: Final = 20


def _summarise_unknown_names(names: list[str]) -> str:
    """Render unknown key names for a caller-facing warning, bounded and safe.

    Args:
        names (list[str]): The unrecognised key names, in the order they should
            be shown. Only the first few survive the cap, so a caller wanting a
            deterministic sample sorts before calling.

    Returns:
        str: Comma-separated sanitised names, each truncated, with a "+N more"
            suffix when the list was longer than the cap.
    """
    shown = [sanitise_for_log(name) for name in names[:_MAX_REPORTED_NAMES]]
    remaining = len(names) - len(shown)
    return ", ".join(shown) + (f", +{remaining} more" if remaining > 0 else "")


def _cap_warning_list(warnings: list[str]) -> list[str]:
    """Bound the NUMBER of warning entries.

    Per-name caps alone do not bound the total: ``interaction_data_list``
    permits 1000 entries, so one warning each still adds up to a
    multi-hundred-KB response body and a single enormous log record.

    Args:
        warnings (list[str]): Individually-bounded warning strings.

    Returns:
        list[str]: A NEW list of at most ``_MAX_WARNING_ENTRIES`` entries, with
            a trailing overflow entry when any were dropped. Always a copy --
            callers append to the result.
    """
    if len(warnings) <= _MAX_WARNING_ENTRIES:
        return list(warnings)
    dropped = len(warnings) - _MAX_WARNING_ENTRIES
    return [
        *warnings[:_MAX_WARNING_ENTRIES],
        f"...and {dropped} more interaction(s) with the same problem",
    ]


CONTENT_BEARING_FIELDS = (
    f'{", ".join(CONTENT_BEARING_FIELD_NAMES)}, or a user_action other than "none"'
)


# publish user interaction request
class PublishUserInteractionRequest(CapturesUnknownFields):
    """A publish payload, with everything it quietly altered recorded.

    Inherits the capture mixin for the REQUEST level too, not just the
    interactions: a top-level typo (``forceExtraction``, ``Source``,
    ``skip_agregation``) used to bind nothing and vanish silently, which is
    strictly worse than the nested case it was reported alongside -- a dropped
    ``force_extraction`` changes what the server does, not just what it stores.
    """

    # All three are set by ``validate_interaction_shapes`` against the caller's
    # ORIGINAL list, before empty rows are filtered out, and are read back by
    # ``payload_warnings()``.
    _unknown_field_warnings: list[str] = PrivateAttr(default_factory=list)
    _skipped_empty_indices: list[int] = PrivateAttr(default_factory=list)
    _skipped_empty_count: int = PrivateAttr(default=0)

    request_id: NonEmptyStr | None = None
    user_id: NonEmptyStr
    interaction_data_list: list[InteractionData] = Field(min_length=1, max_length=1_000)
    source: SessionOutcomeSource = ""
    # this is used for aggregating interactions for generating agent playbooks
    agent_version: str = Field(default="", max_length=1_000)
    session_id: NonEmptyStr  # used for grouping requests together
    skip_aggregation: bool = (
        False  # when True, extract profiles/playbooks but skip aggregation
    )
    force_extraction: bool = False  # when True, bypass all extraction gates (stride_size, cheap pre-filter, LLM should_run) and always run extractors
    evaluation_only: bool = False  # when True, store for evaluation and permanently exclude from profile/playbook extraction
    override_learning_stall: bool = False  # when True, run extraction even if a provider auth/billing stall is recorded
    retrieval_experiment_id: NonEmptyStr | None = None
    retrieval_experiment_arm: Literal["treatment", "holdout"] | None = None

    @model_validator(mode="after")
    def validate_evaluation_only(self) -> Self:
        if self.evaluation_only and self.force_extraction:
            raise ValueError("evaluation_only cannot be combined with force_extraction")
        if self.evaluation_only and not self.session_id:
            raise ValueError("evaluation_only publishes require session_id")
        return self

    @model_validator(mode="after")
    def validate_retrieval_experiment_pair(self) -> Self:
        if (self.retrieval_experiment_id is None) != (
            self.retrieval_experiment_arm is None
        ):
            raise ValueError(
                "retrieval_experiment_id and retrieval_experiment_arm must be provided together"
            )
        return self

    @model_validator(mode="after")
    def validate_interaction_shapes(self) -> Self:
        """Reject contradictory interactions; drop empty ones.

        Runs during request parsing, so it applies on *both* the synchronous
        and the ``wait_for_response=False`` background-task path -- unlike
        ``precondition_checks``, whose ``success=False`` is discarded inside
        the background task after the caller was told 200 "queued".

        An individual empty interaction is **skipped, not fatal**. Making it
        fatal was implemented and reverted: both first-party plugins append an
        empty ``Assistant`` placeholder unconditionally, so one empty row
        rejected the whole batch -- including the real user turn beside it --
        and their adapters swallow the error without advancing the publish
        watermark, retrying the same doomed batch forever. A batch where
        *every* interaction is empty is still fatal, and that is exactly the
        incident this validation exists for (50 of 50 rows empty).

        Raises:
            ValueError: for a contradictory interaction, or an all-empty batch.
        """
        for index, interaction in enumerate(self.interaction_data_list):
            if reason := interaction.shape_error():
                raise ValueError(f"interaction_data_list[{index}] {reason}")

        # Built against the ORIGINAL list, BEFORE filtering. Computing them
        # afterwards defeats the feature in its primary case: a mis-keyed
        # ``content`` yields an *empty* interaction, so the row carrying the typo
        # is exactly the row that gets dropped -- its "unrecognised field"
        # warning vanished, leaving the caller "skipped 1 empty interaction" and
        # no idea which field was wrong. It also renumbered every surviving
        # index, so a warning pointed at a different row than was sent.
        self._unknown_field_warnings = [
            f"interaction_data_list[{index}]: ignored unrecognised field(s)"
            f" {_summarise_unknown_names(names)}"
            for index, interaction in enumerate(self.interaction_data_list)
            if (names := interaction.reportable_unknown_fields())
        ]

        # Indices are computed against the ORIGINAL list, before filtering, so
        # what gets reported matches the payload the caller actually sent.
        skipped = [
            index
            for index, interaction in enumerate(self.interaction_data_list)
            if not interaction.carries_content()
        ]
        if len(skipped) == len(self.interaction_data_list):
            # Carry the unknown-field warnings into the error. This is the
            # motivating incident, not a nicety: 50 interactions all keyed
            # ``Content`` bind nothing, so every row is empty, so the request
            # 422s -- and without this the 422 never once mentions ``Content``,
            # leaving the caller told only that their payload was empty when
            # they can see they sent 50 rows of text. ``payload_warnings()``
            # never runs on this path because the model never finishes
            # validating, so the warnings are otherwise computed and discarded.
            faults = _cap_warning_list(self._unknown_field_warnings)
            cause = f"; likely cause -- {'; '.join(faults)}" if faults else ""
            raise ValueError(
                "every interaction is empty: at least one must set"
                f' "content" (or any of: {CONTENT_BEARING_FIELDS}){cause}'
            )
        self._skipped_empty_indices = skipped[:_MAX_SKIPPED_INDICES]
        self._skipped_empty_count = len(skipped)
        self.interaction_data_list = [
            interaction
            for interaction in self.interaction_data_list
            if interaction.carries_content()
        ]
        return self

    def payload_warnings(self) -> list[str]:
        """Everything quietly altered in the payload, for the caller.

        Unrecognised keys that were stripped -- at the request level and per
        interaction -- plus a summary of empty interactions that were skipped.
        Indices refer to the payload **as the caller sent it**, not the filtered
        list. Names only, sanitised and bounded: values are caller payload, and
        the names are equally caller-controlled, so both volume and control
        characters are handled.

        Returns:
            list[str]: Bounded warnings; empty when nothing was altered.
        """
        # Cap the per-interaction entries, THEN append the two batch-level
        # entries, so the cap can never drop them. Capping the combined list
        # swallowed "N interactions were dropped" whenever there were >= 20
        # field warnings -- the single most important fact about such a batch,
        # and the same is true of a dropped top-level ``force_extraction``.
        warnings = _cap_warning_list(self._unknown_field_warnings)
        if top_level := self.unknown_field_names():
            warnings.append(
                "publish request: ignored unrecognised field(s)"
                f" {_summarise_unknown_names(top_level)}"
            )
        if summary := self.skipped_empty_summary():
            warnings.append(summary)
        return warnings

    def skipped_empty_summary(self) -> str | None:
        """A log-safe description of empty interactions that were dropped.

        Returns None when nothing was skipped. Indices refer to the payload as
        the caller sent it, not the filtered list. Bounded, because the count is
        caller-controlled.

        Returns:
            str | None: Summary for the server log, or None.
        """
        if not self._skipped_empty_count:
            return None
        shown = ", ".join(str(index) for index in self._skipped_empty_indices)
        more = self._skipped_empty_count - len(self._skipped_empty_indices)
        return (
            f"skipped {self._skipped_empty_count} empty interaction(s) that"
            f" carried no content, at index(es) {shown}"
            f"{f', +{more} more' if more else ''}"
        )

    @model_validator(mode="after")
    def validate_retrieved_learnings_total(self) -> Self:
        total = sum(
            len(interaction.retrieved_learnings)
            for interaction in self.interaction_data_list
        )
        if total > 1_000:
            raise ValueError(
                "a publish request may carry at most 1000 retrieved_learnings"
                f" across all interactions (got {total})"
            )
        return self


# publish user interaction response
class PublishUserInteractionResponse(BaseModel):
    success: bool
    message: str = ""
    warnings: list[str] = Field(default_factory=list)
    # Diagnostics (populated only when wait_for_response=True; None otherwise).
    # Exposed so the CLI can tell users *where* their publish landed.
    request_id: str | None = None
    endpoint_url: str | None = None
    storage_type: str | None = None
    storage_label: str | None = None
    profiles_added: int | None = None
    profiles_updated: int | None = None
    playbooks_added: int | None = None
    playbooks_updated: int | None = None
    # Set to "deferred" when the server queued extraction asynchronously.
    # None on the sync (wait_for_response=True) path. Poll GET
    # /api/learning_status?request_id=... to track progress once the
    # durable queue is active.
    learning_status: str | None = None


class LearningStatusResponse(BaseModel):
    """Response for GET /api/learning_status.

    Attributes:
        status: One of ``pending | processing | done | failed``.
            Coverage-based: reflects whether a durable learning job has
            processed through the request's creation timestamp.
    """

    status: Literal["pending", "processing", "done", "failed"]


# whoami response — caller identity + resolved storage routing (masked)
class WhoamiResponse(BaseModel):
    success: bool
    org_id: str
    storage_type: str | None = None
    storage_label: str | None = None  # always masked — never contains raw keys
    storage_configured: bool = False
    message: str = ""


# my_config response — caller's raw storage credentials (token-gated)
class MyConfigResponse(BaseModel):
    success: bool
    # serialized StorageConfig — may contain secrets
    storage_config: dict[str, Any] | None = None
    storage_type: str | None = None
    message: str = ""


# add user playbook request/response
class AddUserPlaybookRequest(BaseModel):
    user_playbooks: list[UserPlaybook] = Field(min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def check_content_fields(self) -> Self:
        """Ensure each user playbook has content for embedding."""
        for i, rf in enumerate(self.user_playbooks):
            if not any((rf.trigger, rf.content)):
                raise ValueError(
                    f"user_playbooks[{i}]: at least one of content "
                    "or trigger must be provided"
                )
        return self


class AddUserPlaybookResponse(BaseModel):
    success: bool
    message: str | None = None
    added_count: int = 0


# add agent playbook request/response (for aggregated playbooks)
class AddAgentPlaybookRequest(BaseModel):
    agent_playbooks: list[AgentPlaybook] = Field(min_length=1)


class AddAgentPlaybookResponse(BaseModel):
    success: bool
    message: str | None = None
    added_count: int = 0


# add user profile request/response (manual profile injection,
# bypassing the inference pipeline)
class AddUserProfileRequest(BaseModel):
    user_profiles: list[UserProfile] = Field(min_length=1)

    @model_validator(mode="after")
    def check_content(self) -> Self:
        """Ensure each profile has non-empty content for embedding."""
        for i, p in enumerate(self.user_profiles):
            if not p.content:
                raise ValueError(
                    f"user_profiles[{i}].content is required for embedding"
                )
        return self


class AddUserProfileResponse(BaseModel):
    success: bool
    message: str | None = None
    added_count: int = 0


class ProfileChangeLogResponse(BaseModel):
    success: bool
    profile_change_logs: list[ProfileChangeLog]


class PublicStructuredData(BaseModel):
    """Deprecated: kept for backward compatibility with deprecated Public* models."""

    trigger: str | None = None


class PublicUserPlaybook(BaseModel):
    """Deprecated: use UserPlaybookView from api_schema.ui instead."""

    user_playbook_id: int = 0
    user_id: str | None = None
    agent_version: str
    request_id: str
    playbook_name: str = ""
    created_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))
    content: str = ""
    trigger: str | None = None
    rationale: str | None = None
    status: Status | None = None
    source: str | None = None
    source_interaction_ids: list[int] = Field(default_factory=list)


class PublicAgentPlaybook(BaseModel):
    """Deprecated: use AgentPlaybookView from api_schema.ui instead."""

    agent_playbook_id: int = 0
    playbook_name: str = ""
    agent_version: str
    created_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))
    content: str
    trigger: str | None = None
    rationale: str | None = None
    playbook_status: PlaybookStatus = PlaybookStatus.PENDING
    playbook_metadata: str = ""
    status: Status | None = None


def user_playbook_to_public(rf: UserPlaybook) -> PublicUserPlaybook:
    """Deprecated: use to_user_playbook_view from api_schema.ui instead."""
    return PublicUserPlaybook(
        user_playbook_id=rf.user_playbook_id,
        user_id=rf.user_id,
        agent_version=rf.agent_version,
        request_id=rf.request_id,
        playbook_name=rf.playbook_name,
        created_at=rf.created_at,
        content=rf.content,
        trigger=rf.trigger,
        rationale=rf.rationale,
        status=rf.status,
        source=rf.source,
        source_interaction_ids=rf.source_interaction_ids,
    )


def agent_playbook_to_public(fb: AgentPlaybook) -> PublicAgentPlaybook:
    """Deprecated: use to_agent_playbook_view from api_schema.ui instead."""
    return PublicAgentPlaybook(
        agent_playbook_id=fb.agent_playbook_id,
        playbook_name=fb.playbook_name,
        agent_version=fb.agent_version,
        created_at=fb.created_at,
        content=fb.content,
        trigger=fb.trigger,
        rationale=fb.rationale,
        playbook_status=fb.playbook_status,
        playbook_metadata=fb.playbook_metadata,
        status=fb.status,
    )


class PublicGetUserPlaybooksResponse(BaseModel):
    """Deprecated: use GetUserPlaybooksViewResponse from api_schema.retriever_schema instead.

    API response for get_user_playbooks — uses public types.
    """

    success: bool
    user_playbooks: list[PublicUserPlaybook]
    msg: str | None = None


class PublicGetAgentPlaybooksResponse(BaseModel):
    """Deprecated: use GetAgentPlaybooksViewResponse from api_schema.retriever_schema instead.

    API response for get_agent_playbooks — uses public types.
    """

    success: bool
    agent_playbooks: list[PublicAgentPlaybook]
    msg: str | None = None


class PublicSearchUserPlaybookResponse(BaseModel):
    """Deprecated: use SearchUserPlaybooksViewResponse from api_schema.retriever_schema instead.

    API response for search_user_playbooks — uses public types.
    """

    success: bool
    user_playbooks: list[PublicUserPlaybook]
    msg: str | None = None


class PublicSearchAgentPlaybookResponse(BaseModel):
    """Deprecated: use SearchAgentPlaybooksViewResponse from api_schema.retriever_schema instead.

    API response for search_agent_playbooks — uses public types.
    """

    success: bool
    agent_playbooks: list[PublicAgentPlaybook]
    msg: str | None = None


class PublicUnifiedSearchResponse(BaseModel):
    """Deprecated: use UnifiedSearchViewResponse from api_schema.retriever_schema instead.

    API response for unified search — uses public types for playbooks.
    """

    success: bool
    profiles: list[UserProfile] = []
    agent_playbooks: list[PublicAgentPlaybook] = []
    user_playbooks: list[PublicUserPlaybook] = []
    reformulated_query: str | None = None
    msg: str | None = None


class AgentPlaybookSnapshot(BaseModel):
    """Lightweight agent playbook snapshot for change log JSONB payloads (excludes embedding and internal status)."""

    agent_playbook_id: int = 0
    playbook_name: str = ""
    agent_version: str = ""
    content: str = ""
    trigger: str | None = None
    rationale: str | None = None
    blocking_issue: BlockingIssue | None = None
    playbook_status: PlaybookStatus = PlaybookStatus.PENDING
    playbook_metadata: str = ""


class AgentPlaybookUpdateEntry(BaseModel):
    """Before/after pair for an updated agent playbook."""

    before: AgentPlaybookSnapshot
    after: AgentPlaybookSnapshot


class PlaybookAggregationChangeLog(BaseModel):
    """Tracks changes from a single playbook aggregation run."""

    id: int = 0
    created_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))
    playbook_name: str
    agent_version: str
    run_mode: Literal["full_archive", "incremental"]
    added_agent_playbooks: list[AgentPlaybookSnapshot] = Field(default_factory=list)
    removed_agent_playbooks: list[AgentPlaybookSnapshot] = Field(default_factory=list)
    updated_agent_playbooks: list[AgentPlaybookUpdateEntry] = Field(
        default_factory=list
    )


class PlaybookAggregationChangeLogResponse(BaseModel):
    success: bool
    change_logs: list[PlaybookAggregationChangeLog]


def agent_playbook_to_snapshot(playbook: AgentPlaybook) -> AgentPlaybookSnapshot:
    """Convert an AgentPlaybook to a lightweight AgentPlaybookSnapshot (excludes embedding and internal status).

    Args:
        playbook (AgentPlaybook): Full agent playbook object

    Returns:
        AgentPlaybookSnapshot: Lightweight snapshot for change log storage
    """
    return AgentPlaybookSnapshot(
        agent_playbook_id=playbook.agent_playbook_id,
        playbook_name=playbook.playbook_name,
        agent_version=playbook.agent_version,
        content=playbook.content,
        trigger=playbook.trigger,
        rationale=playbook.rationale,
        blocking_issue=playbook.blocking_issue,
        playbook_status=playbook.playbook_status,
        playbook_metadata=playbook.playbook_metadata,
    )


class RunPlaybookAggregationRequest(BaseModel):
    agent_version: str = DEFAULT_AGENT_VERSION
    playbook_name: NonEmptyStr = "playbook"

    @field_validator("agent_version")
    @classmethod
    def resolve_version(cls, v: str) -> str:
        return v or DEFAULT_AGENT_VERSION


class RunPlaybookAggregationResponse(BaseModel):
    success: bool
    message: str = ""


class RerunProfileGenerationRequest(BaseModel):
    user_id: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    source: SessionOutcomeSource | None = None
    extractor_names: list[str] | None = (
        None  # Deprecated compatibility field; ignored for selection.
    )

    @model_validator(mode="after")
    def check_time_range(self) -> Self:
        """Validate that end_time is after start_time."""
        TimeRangeValidatorMixin.validate_time_range(self.start_time, self.end_time)
        return self


class RerunProfileGenerationResponse(BaseModel):
    success: bool
    msg: str | None = None
    profiles_generated: int | None = None
    operation_id: str = "rerun_profile_generation"


class ManualProfileGenerationRequest(BaseModel):
    """Request for manual trigger of regular profile generation.

    Uses window-sized interactions (from config) instead of all interactions.
    Outputs profiles with CURRENT status (not PENDING like rerun).
    """

    user_id: str | None = None
    source: SessionOutcomeSource | None = None
    extractor_names: list[str] | None = None


class ManualProfileGenerationResponse(BaseModel):
    """Response for manual profile generation."""

    success: bool
    msg: str | None = None
    profiles_generated: int | None = None


class ManualPlaybookGenerationRequest(BaseModel):
    """Request for manual trigger of regular playbook generation.

    Uses window-sized interactions (from config) instead of all interactions.
    Outputs playbooks with CURRENT status (not PENDING like rerun).
    """

    agent_version: str = DEFAULT_AGENT_VERSION
    source: SessionOutcomeSource | None = None
    playbook_name: str | None = (
        None  # Deprecated compatibility field; ignored for selection.
    )

    @field_validator("agent_version")
    @classmethod
    def resolve_version(cls, v: str) -> str:
        return v or DEFAULT_AGENT_VERSION


class ManualPlaybookGenerationResponse(BaseModel):
    """Response for manual playbook generation."""

    success: bool
    msg: str | None = None
    playbooks_generated: int | None = None


class RerunPlaybookGenerationRequest(BaseModel):
    agent_version: str = DEFAULT_AGENT_VERSION
    start_time: datetime | None = None
    end_time: datetime | None = None
    playbook_name: str | None = (
        None  # Deprecated compatibility field; ignored for selection.
    )
    source: SessionOutcomeSource | None = None

    @field_validator("agent_version")
    @classmethod
    def resolve_version(cls, v: str) -> str:
        return v or DEFAULT_AGENT_VERSION

    @model_validator(mode="after")
    def check_time_range(self) -> Self:
        """Validate that end_time is after start_time."""
        TimeRangeValidatorMixin.validate_time_range(self.start_time, self.end_time)
        return self


class RerunPlaybookGenerationResponse(BaseModel):
    success: bool
    msg: str | None = None
    playbooks_generated: int | None = None
    operation_id: str = "rerun_playbook_generation"


class ReviewUserPlaybookEdit(BaseModel):
    """Replacement fields proposed by the user-playbook reviewer."""

    content: str
    trigger: str
    rationale: str


class ReviewUserPlaybookResult(BaseModel):
    """One persisted user playbook's re-review outcome.

    ``skip`` means the row could not be reviewed at all because its finalized
    generation-window or cited-evidence provenance is absent, missing, or
    invalid, not that the reviewer chose to leave it alone — that is ``accept``.
    A skipped row is never written to.
    """

    user_playbook_id: int = Field(gt=0)
    decision: Literal["accept", "edit", "reject", "skip"]
    reason_code: ReviewUserPlaybookReasonCode
    reason: str | None = None
    edit: ReviewUserPlaybookEdit | None = None
    applied: bool = False
    successor_user_playbook_id: int | None = Field(default=None, gt=0)


class ReviewUserPlaybooksRequest(BaseModel):
    """Select and re-review current user playbooks created in a time window."""

    start_time: datetime
    end_time: datetime
    top_k: int = Field(default=10, gt=0, le=100)
    report_only: bool = True

    @model_validator(mode="after")
    def check_time_range(self) -> Self:
        TimeRangeValidatorMixin.validate_time_range(self.start_time, self.end_time)
        return self


class ReviewUserPlaybooksResponse(BaseModel):
    """Bulk user-playbook re-review report and optional apply summary.

    Apply mode runs in the background, so its response carries ``run_id`` and an
    empty ``results`` list; the per-playbook outcome is durably recorded on each
    replacement's lineage under that ``run_id``.
    """

    success: bool
    report_only: bool = True
    run_id: str | None = None
    selected_count: int = 0
    accepted_count: int = 0
    edited_count: int = 0
    rejected_count: int = 0
    skipped_count: int = 0
    results: list[ReviewUserPlaybookResult] = Field(default_factory=list)
    msg: str | None = None


class UpgradeProfilesRequest(BaseModel):
    user_id: str | None = None  # None means "all users"
    profile_ids: list[str] | None = None
    only_affected_users: bool = (
        False  # If True, only upgrade users who have pending profiles
    )


class UpgradeProfilesResponse(BaseModel):
    success: bool
    profiles_archived: int = 0
    profiles_promoted: int = 0
    profiles_deleted: int = 0
    message: str = ""


class DowngradeProfilesRequest(BaseModel):
    user_id: str | None = None  # None means "all users"
    profile_ids: list[str] | None = None
    only_affected_users: bool = (
        False  # If True, only downgrade users who have archived profiles
    )


class DowngradeProfilesResponse(BaseModel):
    success: bool
    profiles_demoted: int = 0
    profiles_restored: int = 0
    message: str = ""


class UpgradeUserPlaybooksRequest(BaseModel):
    agent_version: str | None = None
    playbook_name: str | None = None
    archive_current: bool = True


class UpgradeUserPlaybooksResponse(BaseModel):
    success: bool
    user_playbooks_deleted: int = 0
    user_playbooks_archived: int = 0
    user_playbooks_promoted: int = 0
    message: str = ""


class DowngradeUserPlaybooksRequest(BaseModel):
    agent_version: str | None = None
    playbook_name: str | None = None


class DowngradeUserPlaybooksResponse(BaseModel):
    success: bool
    user_playbooks_demoted: int = 0
    user_playbooks_restored: int = 0
    message: str = ""


# ===============================
# Operation Status Models
# ===============================
class OperationStatusInfo(BaseModel):
    service_name: str
    status: OperationStatus
    started_at: int
    completed_at: int | None = None
    total_users: int = 0
    processed_users: int = 0
    failed_users: int = 0
    current_user_id: str | None = None
    processed_user_ids: list[str] = []
    failed_user_ids: list[dict] = []  # [{"user_id": "...", "error": "..."}]
    request_params: dict = {}
    stats: dict = {}
    error_message: str | None = None
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0)


class GetOperationStatusRequest(BaseModel):
    service_name: str = "profile_generation"


class GetOperationStatusResponse(BaseModel):
    success: bool
    operation_status: OperationStatusInfo | None = None
    msg: str | None = None


class CancelOperationRequest(BaseModel):
    service_name: str | None = None  # None cancels both services


class CancelOperationResponse(BaseModel):
    success: bool
    cancelled_services: list[str] = []
    msg: str | None = None


# Admin cache invalidation — explicit eviction of the per-org Reflexio cache.
class AdminInvalidateCacheRequest(BaseModel):
    """Request body for ``POST /api/admin/cache/invalidate``.

    The optional ``org_id`` is a verification token: when supplied it
    must match the caller's resolved org_id, otherwise the server
    rejects with 403. This guards against a misconfigured client
    accidentally invalidating someone else's cache. Cross-org admin
    invalidation is intentionally out of scope for this endpoint.
    """

    org_id: str | None = None


class AdminInvalidateCacheResponse(BaseModel):
    """Result of an admin cache invalidation call.

    Attributes:
        invalidated (bool): True when an entry was evicted, False when
            no entry was cached for the org (still a successful no-op).
        org_id (str): The org_id that was targeted (always the caller's
            own org).
    """

    invalidated: bool
    org_id: str
