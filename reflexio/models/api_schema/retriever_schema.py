from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import BaseModel, Field, model_validator

from ..config_schema import SearchMode
from .common import BlockingIssue
from .service_schemas import (
    AgentPlaybook,
    AgentSuccessEvaluationResult,
    Interaction,
    PlaybookStatus,
    Request,
    Status,
    UserPlaybook,
    UserProfile,
)
from .ui.entities import (
    AgentPlaybookView,
    EvaluationResultView,
    InteractionView,
    ProfileChangeLogView,
    ProfileView,
    UserPlaybookView,
)
from .validators import (
    NonEmptyStr,
    TimeRangeValidatorMixin,
)


class SearchInteractionRequest(BaseModel):
    user_id: NonEmptyStr
    request_id: str | None = None
    query: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    top_k: int | None = Field(default=None, gt=0)
    most_recent_k: int | None = Field(default=None, gt=0)
    search_mode: SearchMode = SearchMode.HYBRID

    @model_validator(mode="after")
    def check_time_range(self) -> Self:
        """Validate that end_time is after start_time."""
        TimeRangeValidatorMixin.validate_time_range(self.start_time, self.end_time)
        return self


class SearchUserProfileRequest(BaseModel):
    user_id: NonEmptyStr
    generated_from_request_id: str | None = None
    query: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    top_k: int | None = Field(default=10, gt=0)
    source: str | None = None
    custom_feature: str | None = None
    extractor_name: str | None = None
    threshold: float | None = Field(default=0.4, ge=0.0, le=1.0)
    enable_reformulation: bool | None = False
    search_mode: SearchMode = SearchMode.HYBRID

    @model_validator(mode="after")
    def check_time_range(self) -> Self:
        """Validate that end_time is after start_time."""
        TimeRangeValidatorMixin.validate_time_range(self.start_time, self.end_time)
        return self


class SearchInteractionResponse(BaseModel):
    success: bool
    interactions: list[Interaction]
    msg: str | None = None


class SearchUserProfileResponse(BaseModel):
    success: bool
    user_profiles: list[UserProfile]
    msg: str | None = None


class GetInteractionsRequest(BaseModel):
    user_id: NonEmptyStr
    start_time: datetime | None = None
    end_time: datetime | None = None
    top_k: int | None = Field(default=30, gt=0)

    @model_validator(mode="after")
    def check_time_range(self) -> Self:
        """Validate that end_time is after start_time."""
        TimeRangeValidatorMixin.validate_time_range(self.start_time, self.end_time)
        return self


class GetInteractionsResponse(BaseModel):
    success: bool
    interactions: list[Interaction]
    msg: str | None = None


class GetUserProfilesRequest(BaseModel):
    user_id: NonEmptyStr
    start_time: datetime | None = None
    end_time: datetime | None = None
    top_k: int | None = Field(default=30, gt=0)
    status_filter: list[Status | None] | None = None

    @model_validator(mode="after")
    def check_time_range(self) -> Self:
        """Validate that end_time is after start_time."""
        TimeRangeValidatorMixin.validate_time_range(self.start_time, self.end_time)
        return self


class GetUserProfilesResponse(BaseModel):
    success: bool
    user_profiles: list[UserProfile]
    msg: str | None = None


class GetProfileStatisticsResponse(BaseModel):
    success: bool
    current_count: int = 0
    pending_count: int = 0
    archived_count: int = 0
    expiring_soon_count: int = 0
    msg: str | None = None


class SetConfigResponse(BaseModel):
    success: bool
    msg: str | None = None


class GetUserPlaybooksRequest(BaseModel):
    limit: int | None = Field(default=100, gt=0)
    user_id: str | None = None
    playbook_name: str | None = None
    agent_version: str | None = None
    status_filter: list[Status | None] | None = None


class GetUserPlaybooksResponse(BaseModel):
    success: bool
    user_playbooks: list[UserPlaybook]
    msg: str | None = None


class GetAgentPlaybooksRequest(BaseModel):
    limit: int | None = Field(default=100, gt=0)
    playbook_name: str | None = None
    agent_version: str | None = None
    status_filter: list[Status | None] | None = None
    playbook_status_filter: PlaybookStatus | None = None


class GetAgentPlaybooksResponse(BaseModel):
    success: bool
    agent_playbooks: list[AgentPlaybook]
    msg: str | None = None


class SearchUserPlaybookRequest(BaseModel):
    """Request for searching user playbooks with semantic/text search and filtering.

    Args:
        query (str, optional): Query for semantic/text search
        user_id (str, optional): Filter by user (via request_id linkage to requests table)
        agent_version (str, optional): Filter by agent version
        playbook_name (str, optional): Filter by playbook name
        start_time (datetime, optional): Start time for created_at filter
        end_time (datetime, optional): End time for created_at filter
        status_filter (list[Optional[Status]], optional): Filter by status (None for CURRENT, PENDING, ARCHIVED)
        top_k (int, optional): Maximum number of results to return. Defaults to 10
        threshold (float, optional): Similarity threshold for vector search. Defaults to 0.4
    """

    query: str | None = None
    user_id: str | None = None
    agent_version: str | None = None
    playbook_name: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    status_filter: list[Status | None] | None = None
    top_k: int | None = Field(default=10, gt=0)
    threshold: float | None = Field(default=0.4, ge=0.0, le=1.0)
    enable_reformulation: bool | None = False
    search_mode: SearchMode = SearchMode.HYBRID

    @model_validator(mode="after")
    def check_time_range(self) -> Self:
        """Validate that end_time is after start_time."""
        TimeRangeValidatorMixin.validate_time_range(self.start_time, self.end_time)
        return self


class SearchUserPlaybookResponse(BaseModel):
    """Response for searching user playbooks.

    Args:
        success (bool): Whether the search was successful
        user_playbooks (list[UserPlaybook]): List of matching user playbooks
        msg (str, optional): Additional message
    """

    success: bool
    user_playbooks: list[UserPlaybook]
    msg: str | None = None


class SearchAgentPlaybookRequest(BaseModel):
    """Request for searching aggregated agent playbooks with semantic/text search and filtering.

    Args:
        query (str, optional): Query for semantic/text search
        agent_version (str, optional): Filter by agent version
        playbook_name (str, optional): Filter by playbook name
        start_time (datetime, optional): Start time for created_at filter
        end_time (datetime, optional): End time for created_at filter
        status_filter (list[Optional[Status]], optional): Filter by status (None for CURRENT, PENDING, ARCHIVED)
        playbook_status_filter (PlaybookStatus, optional): Filter by playbook status (PENDING, APPROVED, REJECTED)
        top_k (int, optional): Maximum number of results to return. Defaults to 10
        threshold (float, optional): Similarity threshold for vector search. Defaults to 0.4
    """

    query: str | None = None
    agent_version: str | None = None
    playbook_name: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    status_filter: list[Status | None] | None = None
    playbook_status_filter: PlaybookStatus | None = None
    top_k: int | None = Field(default=10, gt=0)
    threshold: float | None = Field(default=0.4, ge=0.0, le=1.0)
    enable_reformulation: bool | None = False
    search_mode: SearchMode = SearchMode.HYBRID

    @model_validator(mode="after")
    def check_time_range(self) -> Self:
        """Validate that end_time is after start_time."""
        TimeRangeValidatorMixin.validate_time_range(self.start_time, self.end_time)
        return self


class SearchAgentPlaybookResponse(BaseModel):
    """Response for searching aggregated agent playbooks.

    Args:
        success (bool): Whether the search was successful
        agent_playbooks (list[AgentPlaybook]): List of matching agent playbooks
        msg (str, optional): Additional message
    """

    success: bool
    agent_playbooks: list[AgentPlaybook]
    msg: str | None = None


class GetAgentSuccessEvaluationResultsRequest(BaseModel):
    limit: int | None = Field(default=100, gt=0)
    agent_version: str | None = None


class GetAgentSuccessEvaluationResultsResponse(BaseModel):
    success: bool
    agent_success_evaluation_results: list[AgentSuccessEvaluationResult]
    msg: str | None = None


class GetRequestsRequest(BaseModel):
    user_id: str | None = None
    request_id: str | None = None
    session_id: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    top_k: int | None = Field(default=30, gt=0)
    offset: int | None = Field(default=0, ge=0)

    @model_validator(mode="after")
    def check_time_range(self) -> Self:
        """Validate that end_time is after start_time."""
        TimeRangeValidatorMixin.validate_time_range(self.start_time, self.end_time)
        return self


class RequestData(BaseModel):
    request: Request
    interactions: list[Interaction]


class Session(BaseModel):
    session_id: str
    requests: list[RequestData]


class GetRequestsResponse(BaseModel):
    success: bool
    sessions: list[Session]
    has_more: bool = False
    msg: str | None = None


class UpdatePlaybookStatusRequest(BaseModel):
    agent_playbook_id: int = Field(gt=0)
    playbook_status: PlaybookStatus


class UpdatePlaybookStatusResponse(BaseModel):
    success: bool
    msg: str | None = None


class UpdateAgentPlaybookRequest(BaseModel):
    """Generic update for an agent playbook. All fields except ID are optional."""

    agent_playbook_id: int = Field(gt=0)
    playbook_name: str | None = None
    content: str | None = None
    trigger: str | None = None
    rationale: str | None = None
    blocking_issue: BlockingIssue | None = None
    playbook_status: PlaybookStatus | None = None


class UpdateAgentPlaybookResponse(BaseModel):
    success: bool
    msg: str | None = None


class UpdateUserPlaybookRequest(BaseModel):
    """Generic update for a user playbook. All fields except ID are optional."""

    user_playbook_id: int = Field(gt=0)
    playbook_name: str | None = None
    content: str | None = None
    trigger: str | None = None
    rationale: str | None = None
    blocking_issue: BlockingIssue | None = None


class UpdateUserPlaybookResponse(BaseModel):
    success: bool
    msg: str | None = None


class UpdateUserProfileRequest(BaseModel):
    """Partial update for an existing user profile.

    Only non-None fields are applied. ``user_id`` and ``profile_id`` are
    required; all other fields are optional, matching the UI edit flow
    where the user typically changes ``content`` and/or ``custom_features``.
    """

    user_id: str
    profile_id: str
    content: str | None = None
    custom_features: dict[str, object] | None = None


class UpdateUserProfileResponse(BaseModel):
    success: bool
    msg: str | None = None


class TimeSeriesDataPoint(BaseModel):
    """A single data point in a time series."""

    timestamp: int = Field(gt=0)  # Unix timestamp
    value: int = Field(ge=0)  # Count or metric value


class PeriodStats(BaseModel):
    """Statistics for a specific time period."""

    total_profiles: int = Field(ge=0)
    total_interactions: int = Field(ge=0)
    total_playbooks: int = Field(ge=0)
    success_rate: float = Field(ge=0.0, le=100.0)  # Percentage (0-100)


class DashboardStats(BaseModel):
    """Comprehensive dashboard statistics including current and previous periods."""

    current_period: PeriodStats
    previous_period: PeriodStats
    interactions_time_series: list[TimeSeriesDataPoint]
    profiles_time_series: list[TimeSeriesDataPoint]
    playbooks_time_series: list[TimeSeriesDataPoint]
    evaluations_time_series: list[TimeSeriesDataPoint]  # Success rate over time


class GetDashboardStatsRequest(BaseModel):
    """Request for dashboard statistics.

    Args:
        days_back (int): Number of days to include in time series data. Defaults to 30.
    """

    days_back: int | None = Field(default=30, gt=0)


class GetDashboardStatsResponse(BaseModel):
    """Response containing dashboard statistics."""

    success: bool
    stats: DashboardStats | None = None
    msg: str | None = None


# ===============================
# Query Reformulation Models
# ===============================


class ConversationTurn(BaseModel):
    """A single turn in a conversation history.

    Args:
        role (str): The role of the speaker (e.g., "user", "agent")
        content (str): The message content
    """

    role: NonEmptyStr
    content: NonEmptyStr


class ReformulationResult(BaseModel):
    """Output of the query reformulation pipeline.

    Args:
        standalone_query (str): Clean, normalized natural language query with
            conversation context resolved, abbreviations expanded, grammar fixed.
    """

    standalone_query: str


# ===============================
# Unified Search Models
# ===============================


class UnifiedSearchRequest(BaseModel):
    """Request for unified search across all entity types.

    Args:
        query (str): Search query text
        top_k (int, optional): Maximum results per entity type. Defaults to 5
        threshold (float, optional): Similarity threshold for vector search. Defaults to 0.3
        agent_version (str, optional): Filter by agent version (agent_playbooks, user_playbooks)
        playbook_name (str, optional): Filter by playbook name (agent_playbooks, user_playbooks)
        user_id (str, optional): Filter by user ID (profiles, user_playbooks)
        conversation_history (list[ConversationTurn], optional): Prior conversation turns for context-aware query rewriting
    """

    query: NonEmptyStr
    top_k: int | None = Field(default=5, gt=0)
    threshold: float | None = Field(default=0.3, ge=0.0, le=1.0)
    agent_version: str | None = None
    playbook_name: str | None = None
    user_id: str | None = None
    conversation_history: list[ConversationTurn] | None = None
    enable_reformulation: bool | None = False
    search_mode: SearchMode = SearchMode.HYBRID


class UnifiedSearchResponse(BaseModel):
    """Response containing search results from all entity types.

    Args:
        success (bool): Whether the search was successful
        profiles (list[UserProfile]): Matching user profiles
        agent_playbooks (list[AgentPlaybook]): Matching aggregated agent playbooks
        user_playbooks (list[UserPlaybook]): Matching user playbooks
        reformulated_query (str, optional): The query used after reformulation (None if reformulation disabled)
        msg (str, optional): Additional message
    """

    success: bool
    profiles: list[UserProfile] = []
    agent_playbooks: list[AgentPlaybook] = []
    user_playbooks: list[UserPlaybook] = []
    reformulated_query: str | None = None
    msg: str | None = None


# ===============================
# View Response Types (user-facing, without embeddings)
# ===============================


class GetInteractionsViewResponse(BaseModel):
    """API response for retrieving interactions — uses View types."""

    success: bool
    interactions: list[InteractionView]
    msg: str | None = None


class GetProfilesViewResponse(BaseModel):
    """API response for retrieving profiles — uses View types."""

    success: bool
    user_profiles: list[ProfileView]
    msg: str | None = None


class SearchInteractionsViewResponse(BaseModel):
    """API response for searching interactions — uses View types."""

    success: bool
    interactions: list[InteractionView]
    msg: str | None = None


class SearchProfilesViewResponse(BaseModel):
    """API response for searching profiles — uses View types."""

    success: bool
    user_profiles: list[ProfileView]
    msg: str | None = None


class GetEvaluationResultsViewResponse(BaseModel):
    """API response for retrieving evaluation results — uses View types."""

    success: bool
    agent_success_evaluation_results: list[EvaluationResultView]
    msg: str | None = None


class ProfileChangeLogViewResponse(BaseModel):
    """API response for profile change logs — uses View types."""

    success: bool
    profile_change_logs: list[ProfileChangeLogView]


class RequestDataView(BaseModel):
    """A single request with its interactions, using View types."""

    request: Request
    interactions: list[InteractionView]


class SessionView(BaseModel):
    """A session containing requests, using View types."""

    session_id: str
    requests: list[RequestDataView]


class GetRequestsViewResponse(BaseModel):
    """API response for retrieving requests — uses View types."""

    success: bool
    sessions: list[SessionView]
    has_more: bool = False
    msg: str | None = None


class UnifiedSearchViewResponse(BaseModel):
    """API response for unified search — uses View types."""

    success: bool
    profiles: list[ProfileView] = []
    agent_playbooks: list[AgentPlaybookView] = []
    user_playbooks: list[UserPlaybookView] = []
    reformulated_query: str | None = None
    msg: str | None = None


class GetUserPlaybooksViewResponse(BaseModel):
    """API response for retrieving user playbooks — uses View types."""

    success: bool
    user_playbooks: list[UserPlaybookView]
    msg: str | None = None


class GetAgentPlaybooksViewResponse(BaseModel):
    """API response for retrieving agent playbooks — uses View types."""

    success: bool
    agent_playbooks: list[AgentPlaybookView]
    msg: str | None = None


class SearchUserPlaybooksViewResponse(BaseModel):
    """API response for searching user playbooks — uses View types."""

    success: bool
    user_playbooks: list[UserPlaybookView]
    msg: str | None = None


class SearchAgentPlaybooksViewResponse(BaseModel):
    """API response for searching agent playbooks — uses View types."""

    success: bool
    agent_playbooks: list[AgentPlaybookView]
    msg: str | None = None
