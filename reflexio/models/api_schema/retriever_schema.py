from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from ..config_schema import SearchMode
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
    # Caller correlation IDs for billing attribution on the Application line.
    # Optional; when populated they flow into the usage event request_id /
    # session_id columns via _meter_applied_learnings in server/api.py.
    request_id: str | None = None
    session_id: str | None = None
    query: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    top_k: int | None = Field(default=10, gt=0)
    source: str | None = None
    custom_feature: str | None = None
    extractor_name: str | None = (
        None  # Deprecated compatibility field; accepted but ignored.
    )
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


class RerankUserProfilesRequest(BaseModel):
    """Cross-encoder rerank for a list of profile ids.

    Use after ``search_user_profiles`` (or any other source of candidate ids)
    when initial results are noisy. The server fetches each candidate's full
    content, scores ``(query, content)`` pairs with a CPU cross-encoder, and
    returns the top_k profiles sorted by descending score.

    Args:
        user_id (str): The user whose profiles to rerank.
        query (str): The reranking query.
        profile_ids (list[str]): Candidate profile ids; ids that don't belong
            to ``user_id`` (or don't exist) are silently dropped.
        top_k (int): Maximum number of profiles to return. Defaults to 10.
    """

    user_id: NonEmptyStr
    query: NonEmptyStr
    profile_ids: list[str]
    top_k: int = Field(default=10, gt=0)


class RerankUserProfilesResponse(BaseModel):
    """Response from :class:`RerankUserProfilesRequest`.

    Args:
        success (bool): Whether the rerank call succeeded.
        user_profiles (list[UserProfile]): Profiles sorted by descending
            cross-encoder score, capped at ``top_k``.
        msg (str, optional): Diagnostic message (e.g. how many ids were
            silently dropped because they didn't resolve).
    """

    success: bool
    user_profiles: list[UserProfile]
    msg: str | None = None


class StorageStatsRequest(BaseModel):
    """Request lightweight metadata about a user's stored profiles + playbooks.

    Useful before deciding ``top_k`` for retrieval — sized counts and
    timestamp ranges let the agent pick a sensible cap rather than a fixed
    constant.

    Args:
        user_id (str): The user to inspect.
    """

    user_id: NonEmptyStr


class StorageStatsResponse(BaseModel):
    """Response from :class:`StorageStatsRequest`.

    Args:
        profile_count (int): Total number of profiles for the user across
            all statuses.
        playbook_count (int): Total number of user playbooks for the user
            across all statuses.
        oldest_profile_modified (datetime, optional): UTC timestamp of the
            oldest profile's ``last_modified_timestamp``; None when the user
            has no profiles.
        newest_profile_modified (datetime, optional): UTC timestamp of the
            newest profile's ``last_modified_timestamp``; None when the user
            has no profiles.
        success (bool): Whether the lookup succeeded.
        msg (str, optional): Diagnostic message.
    """

    profile_count: int = Field(default=0, ge=0)
    playbook_count: int = Field(default=0, ge=0)
    oldest_profile_modified: datetime | None = None
    newest_profile_modified: datetime | None = None
    success: bool
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
    # Caller correlation IDs for billing attribution on the Application line.
    # Optional; consumed by _meter_applied_learnings in server/api.py.
    request_id: str | None = None
    session_id: str | None = None


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
    # Caller correlation IDs for billing attribution on the Application line.
    # Optional; consumed by _meter_applied_learnings in server/api.py.
    request_id: str | None = None
    session_id: str | None = None

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
        playbook_status_filter (PlaybookStatus | list[PlaybookStatus], optional):
            Filter by playbook approval status. Accepts either a single
            ``PlaybookStatus`` (matched with ``=``) or a list (matched with
            ``IN (...)``) so callers can request multiple approval states in
            a single storage query without per-status fan-out. Defaults to
            None (no status predicate).
        top_k (int, optional): Maximum number of results to return. Defaults to 10
        threshold (float, optional): Similarity threshold for vector search. Defaults to 0.4
    """

    query: str | None = None
    agent_version: str | None = None
    playbook_name: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    status_filter: list[Status | None] | None = None
    playbook_status_filter: PlaybookStatus | list[PlaybookStatus] | None = None
    top_k: int | None = Field(default=10, gt=0)
    threshold: float | None = Field(default=0.4, ge=0.0, le=1.0)
    enable_reformulation: bool | None = False
    search_mode: SearchMode = SearchMode.HYBRID
    # Caller correlation IDs for billing attribution on the Application line.
    # Optional; consumed by _meter_applied_learnings in server/api.py.
    request_id: str | None = None
    session_id: str | None = None

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
    playbook_status: PlaybookStatus | None = None


class UpdateAgentPlaybookResponse(BaseModel):
    success: bool
    msg: str | None = None


class UpdateUserPlaybookRequest(BaseModel):
    """Generic update for a user playbook. All fields except ID are optional.

    Two new optional fields: ``status`` lets a client archive
    candidates surfaced by the lib's ``get_memory_review`` (or
    ``GET /api/get_memory_review``); ``playbook_metadata`` lets a
    client stamp a ``{"superseded_by": <id>}`` reference on a
    playbook that has been replaced by a newer one. Both are
    backwards-compatible — older clients that omit them get the
    same behaviour as before (no status change, no metadata change).
    """

    user_playbook_id: int = Field(gt=0)
    playbook_name: str | None = None
    content: str | None = None
    trigger: str | None = None
    rationale: str | None = None
    status: Status | None = None
    playbook_metadata: str | None = None


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


class PlaybookApplicationStat(BaseModel):
    """Per-rule application stats derived from interaction citations.

    Aggregates the JSON ``citations`` column on interactions to surface how
    often each individual playbook or profile has been cited by the agent in
    a given time window. Used by the claude-smart dashboard to show users a
    "track record" for each rule so the impact of a learning is visible
    rather than abstract.

    Args:
        real_id (str): Stable id of the cited item — ``user_playbook_id``,
            ``agent_playbook_id``, or ``profile_id`` (always serialized as a
            string).
        kind (str): ``"playbook"`` or ``"profile"`` — the citation kind, as
            recorded on ``Interaction.citations``.
        title (str): Human-readable label for the rule. Empty string when
            the underlying row has been deleted but old citations remain.
        applied_count (int): Number of interactions in the window whose
            citations referenced this ``(kind, real_id)``.
        last_applied_at (int | None): Unix epoch seconds of the most recent
            interaction citing this rule. ``None`` when no citation matches.
            Matches the int-epoch convention used elsewhere in the dashboard
            (e.g. ``Interaction.created_at``).
        last_interaction_id (int | None): ``interaction_id`` of the most
            recent citing interaction; useful for deep-linking from the
            dashboard.
    """

    real_id: str
    kind: Literal["playbook", "profile"]
    title: str = ""
    applied_count: int = Field(ge=0)
    last_applied_at: int | None = None
    last_interaction_id: int | None = None


class GetPlaybookApplicationStatsRequest(BaseModel):
    """Request for per-rule application stats.

    Args:
        days_back (int): Look-back window in days. Defaults to 30; must be
            positive.
    """

    days_back: int = Field(default=30, gt=0)


class GetPlaybookApplicationStatsResponse(BaseModel):
    """Response containing per-rule application stats.

    Args:
        success (bool): Whether the call succeeded.
        stats (list[PlaybookApplicationStat]): One row per cited rule, sorted
            by ``applied_count`` descending.
        msg (str | None): Optional error message when ``success`` is False.
    """

    success: bool
    stats: list[PlaybookApplicationStat] = Field(default_factory=list)
    msg: str | None = None


class InjectionStat(BaseModel):
    """Per-entity injection rollup derived from ``usage_events``.

    One row per ``(entity_type, entity_id)`` over a look-back window.
    Unlike :class:`PlaybookApplicationStat` which counts the *applied* side
    (citations on assistant turns), ``InjectionStat`` counts the *surfaced*
    side (per-entity rows in ``usage_events`` for ``learning_injection``).
    The two views answer different questions:
    ``PlaybookApplicationStat`` tells you "what influenced the agent's
    response?"; ``InjectionStat`` tells you "what was rendered into the
    context window, and at what cost?".

    Title is NOT included; callers that need the playbook / profile name
    join with the corresponding tables using ``entity_id``.

    Args:
        entity_type (str): ``"user_playbook"``, ``"agent_playbook"`` or
            ``"profile"``.
        entity_id (str): Storage id of the surfaced entity.
        surfaced_count (int): Times the entity was injected in the window.
        distinct_session_count (int): Number of distinct sessions that
            triggered an injection of this entity in the window.
        total_prompt_tokens (int): Sum of per-entity ``prompt_tokens``
            (token count under ``cl100k_base`` via
            :func:`reflexio.server.billing_signals.count_input_tokens`).
        first_injected_at (int | None): Unix epoch seconds.
        last_injected_at (int | None): Unix epoch seconds.
        last_session_id (str): Most-recent session id; empty when unknown.
    """

    entity_type: str
    entity_id: str
    surfaced_count: int = Field(ge=0)
    distinct_session_count: int = Field(ge=0)
    total_prompt_tokens: int = Field(ge=0)
    first_injected_at: int | None = None
    last_injected_at: int | None = None
    last_session_id: str = ""


class GetInjectionStatsRequest(BaseModel):
    """Request for per-entity injection stats.

    Args:
        days_back (int): Look-back window in days. Defaults to 30; must be
            positive.
    """

    days_back: int = Field(default=30, gt=0)


class GetInjectionStatsResponse(BaseModel):
    """Response containing per-entity injection stats.

    Args:
        success (bool): Whether the call succeeded.
        stats (list[InjectionStat]): One row per ``(entity_type,
            entity_id)``, sorted by ``surfaced_count`` descending.
        msg (str | None): Optional error message when ``success`` is False.
    """

    success: bool
    stats: list[InjectionStat] = Field(default_factory=list)
    msg: str | None = None


class MemoryReviewCandidate(BaseModel):
    """A user playbook flagged for memory review.

    Surfaces playbooks that are stale, duplicated, low-utility, or
    superseded. One row per ``entity_id``; ``signals`` carries the
    detected reason(s) (a single row can have multiple signals).
    Channel-agnostic — the same response shape works for any
    reflexio user, regardless of which channel adapter (claude-smart,
    Codex, etc.) drove the writes.

    Note: in v1 only ``user_playbooks`` are reviewed. The
    ``entity_type`` field is fixed to ``"user_playbook"`` for now; a
    follow-up will add profile review (and widen the type to
    ``Literal["user_playbook", "profile"]``) once the storage layer
    supports it.

    Args:
        entity_type (str): Always ``"user_playbook"`` in v1.
        entity_id (str): Storage id of the playbook.
        title (str): Human-readable label (playbook_name or first
            80 chars of playbook content).
        signals (list[str]): One or more of ``"stale"`` and
            ``"high_cost_low_cite"``. ``"duplicate"`` and
            ``"supersedeable"`` are reserved for a follow-up and are
            not emitted in v1.
        score (int): Higher = stronger signal. Ordering is by score
            descending; the score encodes signal priority
            (``stale`` = 50-99, ``high_cost_low_cite`` = 30-49) so
            sorting by score groups by primary signal and orders
            within each group by strength.
        injection_count (int): Times injected in the look-back window.
        citation_count (int): Times cited on assistant turns in the
            look-back window. From the existing
            :meth:`get_playbook_application_stats` rollup.
        last_injected_at (int | None): Unix epoch seconds.
        last_cited_at (int | None): Unix epoch seconds.
        last_modified_at (int | None): Unix epoch seconds.
    """

    entity_type: Literal["user_playbook"]
    entity_id: str
    title: str = ""
    signals: list[
        Literal["stale", "duplicate", "high_cost_low_cite", "supersedeable"]
    ]
    score: int = Field(ge=0)
    injection_count: int = Field(ge=0)
    citation_count: int = Field(ge=0)
    last_injected_at: int | None = None
    last_cited_at: int | None = None
    last_modified_at: int | None = None


class GetMemoryReviewRequest(BaseModel):
    """Request for a memory review candidate list.

    Args:
        days_back (int): Look-back window in days. Defaults to 60; must be
            positive.
        user_id (str | None): User whose playbooks should be reviewed.
            Required unless ``include_all_users`` is true.
        include_all_users (bool): Explicit opt-in for org-wide review.
        signal_filter (list[str] | None): Optional whitelist of
            ``signals`` to include. When omitted, all signals are
            returned. Useful for the dashboard's "show me only stale"
            view.
    """

    days_back: int = Field(default=60, gt=0)
    user_id: NonEmptyStr | None = None
    include_all_users: bool = False
    signal_filter: list[
        Literal["stale", "duplicate", "high_cost_low_cite", "supersedeable"]
    ] | None = None

    @model_validator(mode="after")
    def check_scope(self) -> Self:
        """Require explicit scope: one user, or deliberate org-wide review."""
        if not self.include_all_users and self.user_id is None:
            raise ValueError("user_id is required unless include_all_users is true")
        return self


class GetMemoryReviewResponse(BaseModel):
    """Response containing a memory review candidate list.

    Args:
        success (bool): Whether the call succeeded.
        candidates (list[MemoryReviewCandidate]): One row per flagged
            entity, sorted by ``score`` descending. Empty when
            storage is not configured.
        msg (str | None): Optional error message when ``success`` is
            False.
    """

    success: bool
    candidates: list[MemoryReviewCandidate] = Field(default_factory=list)
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


UnifiedSearchEntityType = Literal["profiles", "user_playbooks", "agent_playbooks"]


class UnifiedSearchRequest(BaseModel):
    """Request for unified search across all entity types.

    Args:
        query (str): Search query text
        top_k (int, optional): Maximum results per entity type. Defaults to 5
        threshold (float, optional): Similarity threshold for vector search. Defaults to 0.3
        agent_version (str, optional): Filter by agent version (agent_playbooks, user_playbooks)
        playbook_name (str, optional): Filter by playbook name (agent_playbooks, user_playbooks)
        user_id (str, optional): Filter by user ID (profiles, user_playbooks)
        entity_types (list[str], optional): Entity types to search. When omitted,
            searches profiles, user_playbooks, and agent_playbooks.
        agent_playbook_status_filter (list[PlaybookStatus], optional): Approval
            statuses to include for agent_playbooks. When omitted, defaults to
            ``[APPROVED, PENDING]`` so that REJECTED playbooks are suppressed
            from results — a rejection in the dashboard immediately hides the
            playbook. Pass an explicit list to opt into REJECTED items.
        conversation_history (list[ConversationTurn], optional): Prior conversation turns for context-aware query rewriting
    """

    query: NonEmptyStr
    top_k: int | None = Field(default=5, gt=0)
    threshold: float | None = Field(default=0.3, ge=0.0, le=1.0)
    agent_version: str | None = None
    playbook_name: str | None = None
    user_id: str | None = None
    entity_types: list[UnifiedSearchEntityType] | None = None
    agent_playbook_status_filter: list[PlaybookStatus] | None = None
    conversation_history: list[ConversationTurn] | None = None
    enable_reformulation: bool | None = False
    enable_agent_answer: bool | None = False
    search_mode: SearchMode = SearchMode.HYBRID
    # Caller correlation IDs for billing attribution on the Application line.
    # Optional; consumed by _meter_applied_learnings in server/api.py.
    request_id: str | None = None
    session_id: str | None = None


class UnifiedSearchResponse(BaseModel):
    """Response containing search results from all entity types.

    Args:
        success (bool): Whether the search was successful
        profiles (list[UserProfile]): Matching user profiles
        agent_playbooks (list[AgentPlaybook]): Matching aggregated agent playbooks
        user_playbooks (list[UserPlaybook]): Matching user playbooks
        reformulated_query (str, optional): The query used after reformulation (None if reformulation disabled)
        msg (str, optional): Additional message
        agent_answer (str, optional): LLM-synthesised answer populated by the agentic backend;
            None for classic backend.
    """

    success: bool
    profiles: list[UserProfile] = []
    agent_playbooks: list[AgentPlaybook] = []
    user_playbooks: list[UserPlaybook] = []
    reformulated_query: str | None = None
    msg: str | None = None
    agent_answer: str | None = None
    agent_trace: str | None = None
    rehydrated_text: str | None = None


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
    agent_trace: str | None = None
    rehydrated_text: str | None = None


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
