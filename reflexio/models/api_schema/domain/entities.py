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
]

# ===============================
# Data Models
# ===============================


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
    embedding: EmbeddingVector = []

    @field_validator("interacted_image_url", mode="after")
    @classmethod
    def validate_image_url(cls, v: str) -> str:
        return _validate_image_url(v)


class Request(BaseModel):
    request_id: str
    user_id: str
    created_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))
    source: str = ""
    agent_version: str = ""
    session_id: str | None = None


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
    extractor_names: list[str] | None = None
    expanded_terms: str | None = None
    embedding: EmbeddingVector = []


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
    embedding: EmbeddingVector = []


class ProfileChangeLog(BaseModel):
    id: int
    user_id: str
    request_id: str
    created_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))
    added_profiles: list[UserProfile]
    removed_profiles: list[UserProfile]
    mentioned_profiles: list[UserProfile]


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
    embedding: EmbeddingVector = []
    status: Status | None = (
        None  # used for tracking intermediate states during playbook aggregation. Status.ARCHIVED for playbooks during aggregation process, None for current playbooks
    )


class AgentSuccessEvaluationResult(BaseModel):
    result_id: int = 0
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

    @field_validator("interacted_image_url", mode="after")
    @classmethod
    def validate_image_url(cls, v: str) -> str:
        return _validate_image_url(v)


# publish user interaction request
class PublishUserInteractionRequest(BaseModel):
    user_id: NonEmptyStr
    interaction_data_list: list[InteractionData] = Field(min_length=1)
    source: str = ""
    agent_version: str = (
        ""  # this is used for aggregating interactions for generating agent playbooks
    )
    session_id: str | None = None  # used for grouping requests together
    skip_aggregation: bool = (
        False  # when True, extract profiles/playbooks but skip aggregation
    )
    force_extraction: bool = (
        False  # when True, bypass batch_interval checks and always run extractors
    )


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
    blocking_issue: BlockingIssue | None = None


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
    blocking_issue: BlockingIssue | None = None
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
    blocking_issue: BlockingIssue | None = None
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
        blocking_issue=rf.blocking_issue,
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
        blocking_issue=fb.blocking_issue,
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
    playbook_name: NonEmptyStr

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
    extractor_names: list[str] | None = None

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
    playbook_name: str | None = None  # Optional filter by playbook name

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
    playbook_name: str | None = None
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
