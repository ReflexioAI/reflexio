from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator

from ..common import NEVER_EXPIRES_TIMESTAMP, BlockingIssue, ToolUsed
from ..validators import _validate_image_url
from .enums import (
    PlaybookStatus,
    ProfileTimeToLive,
    RegularVsShadow,
    Status,
    UserActionType,
)

__all__ = [
    "InteractionView",
    "ProfileView",
    "UserPlaybookView",
    "AgentPlaybookView",
    "EvaluationResultView",
    "ProfileChangeLogView",
]

# ===============================
# View Models (user-facing, without embeddings)
# ===============================


class InteractionView(BaseModel):
    """User-facing Interaction — excludes embedding and image_encoding."""

    interaction_id: int = 0
    user_id: str
    request_id: str
    created_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))
    role: str = "User"
    content: str = ""
    user_action: UserActionType = UserActionType.NONE
    user_action_description: str = ""
    interacted_image_url: str = ""
    shadow_content: str = ""
    expert_content: str = ""
    tools_used: list[ToolUsed] = Field(default_factory=list)

    @field_validator("interacted_image_url", mode="after")
    @classmethod
    def validate_image_url(cls, v: str) -> str:
        return _validate_image_url(v)


class ProfileView(BaseModel):
    """User-facing UserProfile — excludes embedding."""

    profile_id: str
    user_id: str
    content: str
    last_modified_timestamp: int
    generated_from_request_id: str
    profile_time_to_live: ProfileTimeToLive = ProfileTimeToLive.INFINITY
    expiration_timestamp: int = NEVER_EXPIRES_TIMESTAMP
    custom_features: dict | None = None
    source: str | None = None
    status: Status | None = None
    extractor_names: list[str] | None = None


class UserPlaybookView(BaseModel):
    """User-facing UserPlaybook — excludes embedding."""

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


class AgentPlaybookView(BaseModel):
    """User-facing AgentPlaybook — excludes embedding."""

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


class EvaluationResultView(BaseModel):
    """User-facing AgentSuccessEvaluationResult — excludes embedding."""

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


class ProfileChangeLogView(BaseModel):
    """User-facing ProfileChangeLog — uses ProfileView lists."""

    id: int
    user_id: str
    request_id: str
    created_at: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))
    added_profiles: list[ProfileView]
    removed_profiles: list[ProfileView]
    mentioned_profiles: list[ProfileView]
