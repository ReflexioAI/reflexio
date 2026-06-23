from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from reflexio.defaults import DEFAULT_AGENT_VERSION

from ..common import (
    NEVER_EXPIRES_TIMESTAMP,
    BlockingIssue,
    BlockingIssueKind,
    ToolUsed,
)
from ..validators import (
    EmbeddingVector,
    NonEmptyStr,
    TimeRangeValidatorMixin,
    _validate_image_url,
)
from .enums import (
    OperationStatus,
    PlaybookStatus,
    ProfileTimeToLive,
    RegularVsShadow,
    Status,
    UserActionType,
)

__all__ = [
    "NEVER_EXPIRES_TIMESTAMP",
    "BlockingIssue",
    "BlockingIssueKind",
    "ToolUsed",
    "Citation",
    "Interaction",
    "Request",
    "UserProfile",
    "UserPlaybook",
    "ProfileChangeLog",
    "AgentPlaybook",
    "AgentSuccessEvaluationResult",
    "DeleteUserProfileRequest",
    "DeleteUserProfileResponse",
    "DeleteUserInteractionRequest",
    "DeleteUserInteractionResponse",
    "DeleteRequestRequest",
    "DeleteRequestResponse",
    "DeleteSessionRequest",
    "DeleteSessionResponse",
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
    "PlaybookOptimizationJob",
    "PlaybookOptimizationCandidate",
    "PlaybookOptimizationEvaluation",
    "PlaybookOptimizationEvent",
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
    "ShareLink",
    "AdminInvalidateCacheRequest",
    "AdminInvalidateCacheResponse",
    "PlaybookRetrievalLog",
    "LineageEvent",
    "LineageContext",
    "RecordRef",
]

# ===============================
# Data Models
# ===============================


class Citation(BaseModel):
    """A playbook or profile item the agent cited as influential.

    Carried inline on an Assistant ``InteractionData`` row to mark
    which previously-injected playbook rule or user-profile row
    materially shaped that response. The server uses these to drive
    reflection (does the cited rule still look right after seeing how
    it was applied?).

    Attributes:
        kind (Literal["playbook", "profile"]): Which kind of cited
            item this references.
        real_id (str): Stable storage id — ``user_playbook_id`` for
            playbooks, ``profile_id`` for profiles.
        tag (str): Injection-time rank tag (e.g. ``"r1-301"``,
            ``"p1-0f37"``). Per-injection, not stable across sessions;
            kept as a debug aid.
        title (str): Short human-readable label for logs and UI.
    """

    kind: Literal["playbook", "profile"]
    real_id: str
    tag: str = ""
    title: str = ""


# information about the user interaction sent by the client
class Interaction(BaseModel):
    interaction_id: int = 0  # 0 = placeholder for DB auto-increment
    user_id: str
    request_id: str
    created_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))
    role: str = "User"
    content: str = ""
    user_action: UserActionType = UserActionType.NONE
    user_action_description: str = ""
    interacted_image_url: str = ""
    image_encoding: str = ""  # base64 encoded image
    shadow_content: str = ""
    expert_content: str = ""
    tools_used: list[ToolUsed] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
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
        source (str): Free-form origin tag (integration name, etc.).
        agent_version (str): The agent version that handled this request.
        session_id (str): Non-empty session this request belongs to.
        evaluation_only (bool): Whether this request is stored for
            session-level evaluation only and must be excluded from
            profile/playbook learning windows.
    """

    request_id: str
    user_id: str
    created_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))
    source: str = ""
    agent_version: str = ""
    session_id: NonEmptyStr
    evaluation_only: bool = False


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


class PlaybookOptimizationJob(BaseModel):
    """One end-to-end optimizer run for a single playbook target.

    Lifecycle: ``pending`` → ``running`` → ``completed`` | ``failed`` |
    ``skipped``. ``best_candidate_id`` and ``successor_target_id`` are set
    when the run produces a winner or commits a successor playbook.
    """

    job_id: int = 0
    target_kind: Literal["agent_playbook", "user_playbook"]
    target_id: int
    status: Literal["pending", "running", "completed", "skipped", "failed"] = "pending"
    best_candidate_id: int | None = None
    successor_target_id: int | None = None
    decision_reason: str = ""
    metadata_json: str = "{}"
    created_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))
    updated_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))


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
    number_of_correction_per_session: int = 0
    user_turns_to_resolution: int | None = None
    is_escalated: bool = False
    embedding: EmbeddingVector = []


class PlaybookRetrievalLog(BaseModel):
    """A log entry recording which playbooks were shown to a user during a request.

    Used by the offline playbook tuner to correlate retrieval decisions with
    downstream outcomes. ``retrieval_log_id`` is assigned by the storage layer;
    ``shown_playbook_ids`` carries agent-playbook ids only (scores deferred to v2).

    Attributes:
        retrieval_log_id (int): Primary key assigned by storage (0 = not yet persisted).
        request_id (str): The request during which playbooks were retrieved.
        session_id (str): The session that owns the request.
        user_id (str): The user the request belongs to.
        shown_playbook_ids (list[int]): Ordered list of agent_playbook_id values shown.
        agent_version (str | None): Agent version string at retrieval time, if available.
        created_at (int): Unix epoch seconds at log creation time (0 = unset).
    """

    retrieval_log_id: int = 0
    request_id: str
    session_id: str
    user_id: str
    shown_playbook_ids: list[int] = []  # ids only (v1); scores deferred to v2 (M1)
    agent_version: str | None = None
    created_at: int = 0


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
        actor (str): Who/what triggered it (consolidator|reflection|offline_optimizer|...).
        request_id (str): Triggering request — part of the idempotency key.
        reason (str): Free-text rationale (no PII).
        created_at (int): Unix epoch seconds (0 = unset; storage stamps it).
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


class LineageContext(BaseModel):
    """Caller-supplied intent the storage layer can't infer.

    Required for merge/supersede/aggregate; optional for create/revise/archive.
    """

    op_kind: str
    actor: str = ""
    source_ids: list[str] = []
    reason: str = ""
    request_id: str | None = None


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


class ShareLink(BaseModel):
    """A shareable public link that maps a token to a resource within an org.

    Args:
        id (int): Primary key assigned by the storage layer.
        org_id (str): The organization that owns the share link.
        token (str): The share token (unique). Format: shr_<org_id_b64>.<random>.
        resource_type (str): One of "profile", "request", "session", "user_playbook", "agent_playbook".
        resource_id (str): The ID of the resource being shared.
        created_at (int | None): Unix timestamp of creation.
        expires_at (int | None): Optional Unix timestamp of expiration. None means never expires.
        created_by_email (str | None): Optional email of the user who created the link.
    """

    id: int
    org_id: str
    token: str
    resource_type: str
    resource_id: str
    created_at: int | None = None
    expires_at: int | None = None
    created_by_email: str | None = None


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
    request_ids: list[str] = Field(min_length=1)


class DeleteProfilesByIdsRequest(BaseModel):
    profile_ids: list[str] = Field(min_length=1)


class DeleteAgentPlaybooksByIdsRequest(BaseModel):
    agent_playbook_ids: list[int] = Field(min_length=1)


class DeleteUserPlaybooksByIdsRequest(BaseModel):
    user_playbook_ids: list[int] = Field(min_length=1)


# Clear all data scoped to a single user_id (interactions, requests, user
# playbooks, profiles). Used by paired-protocol harnesses (e.g. SWE-bench) to
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
class InteractionData(BaseModel):
    created_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))
    role: str = "User"
    content: str = ""
    shadow_content: str = ""
    expert_content: str = ""
    user_action: UserActionType = UserActionType.NONE
    user_action_description: str = ""
    interacted_image_url: str = ""
    image_encoding: str = ""  # base64 encoded image
    tools_used: list[ToolUsed] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)

    @field_validator("interacted_image_url", mode="after")
    @classmethod
    def validate_image_url(cls, v: str) -> str:
        return _validate_image_url(v)


# publish user interaction request
class PublishUserInteractionRequest(BaseModel):
    request_id: NonEmptyStr | None = None
    user_id: NonEmptyStr
    interaction_data_list: list[InteractionData] = Field(min_length=1)
    source: str = ""
    agent_version: str = (
        ""  # this is used for aggregating interactions for generating agent playbooks
    )
    session_id: NonEmptyStr  # used for grouping requests together
    skip_aggregation: bool = (
        False  # when True, extract profiles/playbooks but skip aggregation
    )
    force_extraction: bool = False  # when True, bypass all extraction gates (stride_size, cheap pre-filter, LLM should_run) and always run extractors
    evaluation_only: bool = False  # when True, store for evaluation and permanently exclude from profile/playbook extraction
    override_learning_stall: bool = False  # when True, run extraction even if a provider auth/billing stall is recorded

    @model_validator(mode="after")
    def validate_evaluation_only(self) -> Self:
        if self.evaluation_only and self.force_extraction:
            raise ValueError("evaluation_only cannot be combined with force_extraction")
        if self.evaluation_only and not self.session_id:
            raise ValueError("evaluation_only publishes require session_id")
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
    user_playbooks: list[UserPlaybook] = Field(min_length=1)

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
    source: str | None = None
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
    source: str | None = None
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
    source: str | None = None
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
    source: str | None = None

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
