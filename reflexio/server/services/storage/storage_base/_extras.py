from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

from reflexio.models.api_schema.braintrust_schema import (
    BraintrustConnection,
    ImportedScore,
)
from reflexio.models.api_schema.domain import (
    Interaction,
    PlaybookAggregationChangeLog,
    ProfileChangeLog,
)
from reflexio.models.api_schema.retriever_schema import (
    InjectionStat,
    MemoryReviewCandidate,
    PlaybookApplicationStat,
)


class ExtrasMixin:
    """Mixin for dashboard, profile change logs, playbook aggregation change logs, and misc methods."""

    # ==============================
    # Dashboard methods
    # ==============================

    @abstractmethod
    def get_dashboard_stats(self, days_back: int = 30) -> dict:
        """Get comprehensive dashboard statistics including counts and time-series data.

        Args:
            days_back (int): Number of days to include in time series data

        Returns:
            dict: Dictionary containing:
                - current_period: Stats for the current period (days_back)
                - previous_period: Stats for the previous period (for trend calculation)
                - interactions_time_series: List of time series data points (raw, ungrouped)
                - profiles_time_series: List of time series data points (raw, ungrouped)
                - playbooks_time_series: List of time series data points (raw, ungrouped)
                - evaluations_time_series: List of time series data points (raw, ungrouped)
        """
        raise NotImplementedError

    def get_playbook_application_stats(
        self, days_back: int = 30
    ) -> list[PlaybookApplicationStat]:
        """Return per-rule citation counts derived from interaction citations.

        Aggregates the JSON ``citations`` column on ``interactions`` over the
        last ``days_back`` days and groups by ``(kind, real_id)``. Joins with
        the playbook / profile tables to populate human-readable titles.

        Concrete default returns ``[]`` so backends that do not yet implement
        this method degrade gracefully (the dashboard simply shows no stats)
        rather than raising 500s. Storage backends should override with a
        real implementation — see ``sqlite_storage._extras`` for the
        reference implementation.

        Args:
            days_back (int): Look-back window in days. Must be positive.

        Returns:
            list[PlaybookApplicationStat]: One row per cited ``(kind,
                real_id)``, sorted by ``applied_count`` descending. Empty
                when the backend has no implementation.
        """
        del days_back
        return []

    @abstractmethod
    def record_usage_event(
        self,
        *,
        org_id: str,
        event_name: str,
        event_category: str,
        user_id: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
        pipeline: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        caller_type: str | None = None,
        count_value: int = 1,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        billing_input_tokens: int | None = None,
        platform_llm: bool | None = None,
        platform_storage: bool | None = None,
        duration_ms: int | None = None,
        error_kind: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Insert one row into the ``usage_events`` table.

        Persistence sink for the process-global ``record_usage_event`` hook
        in :mod:`reflexio.server.usage_metrics`. Wired by the CLI
        entrypoint (``reflexio.server.__main__``) via
        :class:`reflexio.server.services.usage_event_sink.SqliteUsageEventSink`.

        Backends MUST implement this method. The hook is process-global, so
        failure here would silently drop observability data; ``SqliteUsageEventSink``
        catches and logs all exceptions to avoid breaking the caller's
        hot path.

        Args:
            org_id: Org id (matches ``self.org_id`` for org-scoped
                backends; passed explicitly for clarity and multi-tenant
                flexibility).
            event_name (str): ``"learning_injection"``,
                ``"learning_applied"``, ``"extraction_tokens"``,
                ``"learnings_generated"``, etc.
            event_category (str): ``"application"``, ``"learning"``, etc.
            user_id (str | None): Caller's user id.
            request_id (str | None): Correlation id.
            session_id (str | None): Conversation id.
            pipeline (str | None): Logical pipeline (e.g., ``"unified_search"``).
            entity_type (str | None): ``"playbook"`` / ``"profile"`` for
                per-entity events.
            entity_id (str | None): Storage id of the surfaced entity.
            caller_type (str | None): Caller classification
                (e.g., ``"production_agent"``).
            count_value (int): Event multiplicity; default 1.
            prompt_tokens (int | None): Tokens for the rendered content.
            completion_tokens (int | None): Tokens for completions.
            billing_input_tokens (int | None): Input-anchored billed tokens.
            platform_llm (bool | None): Whether the platform supplies the LLM.
            platform_storage (bool | None): Whether the platform supplies
                storage.
            duration_ms (int | None): Wall-clock duration in milliseconds.
            error_kind (str | None): Error classification.
            metadata (Mapping[str, Any] | None): Free-form per-event metadata.
        """
        raise NotImplementedError

    def get_injection_stats(
        self, days_back: int = 30
    ) -> list[InjectionStat]:
        """Per-entity injection rollup aggregated over the look-back window.

        Aggregates the ``usage_events`` table for rows of
        ``event_name = "learning_injection"`` within ``days_back`` and
        groups by ``(entity_type, entity_id)``. Titles are NOT included
        in the rollup; callers can join with the playbook / profile
        tables when needed.

        Concrete default returns ``[]`` so backends that do not yet
        implement this method degrade gracefully. Storage backends
        should override with a real implementation — see
        ``sqlite_storage._extras`` for the reference implementation.

        Args:
            days_back (int): Look-back window in days. Must be positive.

        Returns:
            list[InjectionStat]: One row per ``(entity_type, entity_id)``,
                sorted by ``surfaced_count`` descending and then by
                ``last_injected_at`` descending. Empty when the backend
                has no implementation.
        """
        del days_back
        return []

    def get_memory_review_candidates(
        self, days_back: int = 60
    ) -> list["MemoryReviewCandidate"]:
        """Surface entities flagged for memory review.

        Channel-agnostic: the result list is the same shape regardless
        of which channel adapter (claude-smart, Codex, custom) drove
        the writes. Combines four signals:

        - ``stale``: not used in ``days_back`` and not modified recently.
        - ``duplicate``: clusters of near-duplicate content (best-effort,
          O(n²); a periodic batch job is the long-term answer for
          installations with thousands of playbooks).
        - ``high_cost_low_cite``: injected often, cited rarely.
        - ``supersedeable``: appears in a recent
          ``playbook_aggregation_change_logs`` entry as a removed rule.

        Concrete default returns ``[]`` so backends that do not yet
        implement this method degrade gracefully. See
        ``sqlite_storage._extras`` for the reference implementation.

        Args:
            days_back (int): Look-back window in days. Must be positive.

        Returns:
            list[MemoryReviewCandidate]: Sorted by ``(signals, -score)``.
                Empty when the backend has no implementation.
        """
        del days_back
        return []

    @abstractmethod
    def get_profile_statistics(self) -> dict:
        """Get profile count statistics by status.

        Returns:
            dict with keys: current_count, pending_count, archived_count, expiring_soon_count
        """
        raise NotImplementedError

    # ==============================
    # Profile Change Log methods
    # ==============================

    @abstractmethod
    def add_profile_change_log(self, profile_change_log: ProfileChangeLog) -> None:
        """Add a profile change log entry."""
        raise NotImplementedError

    @abstractmethod
    def get_profile_change_logs(self, limit: int = 100) -> list[ProfileChangeLog]:
        """Get profile change logs for an organization."""
        raise NotImplementedError

    @abstractmethod
    def delete_profile_change_log_for_user(self, user_id: str) -> None:
        """Delete all profile change logs for a user."""
        raise NotImplementedError

    @abstractmethod
    def delete_all_profile_change_logs(self) -> None:
        """Delete all profile change logs."""
        raise NotImplementedError

    # ==============================
    # Playbook Aggregation Change Log methods
    # ==============================

    @abstractmethod
    def add_playbook_aggregation_change_log(
        self, change_log: PlaybookAggregationChangeLog
    ) -> None:
        """Add a playbook aggregation change log entry."""
        raise NotImplementedError

    @abstractmethod
    def get_playbook_aggregation_change_logs(
        self,
        playbook_name: str,
        agent_version: str,
        limit: int = 100,
    ) -> list[PlaybookAggregationChangeLog]:
        """Get playbook aggregation change logs filtered by playbook_name and agent_version."""
        raise NotImplementedError

    @abstractmethod
    def delete_all_playbook_aggregation_change_logs(self) -> None:
        """Delete all playbook aggregation change logs."""
        raise NotImplementedError

    # ==============================
    # Misc methods
    # ==============================

    @abstractmethod
    def get_interactions_by_request_ids(
        self, request_ids: list[str]
    ) -> list[Interaction]:
        """Fetch interactions by their request IDs.

        Args:
            request_ids (list[str]): List of request IDs to fetch interactions for

        Returns:
            list[Interaction]: List of matching interaction objects
        """
        raise NotImplementedError

    @abstractmethod
    def get_interactions_by_ids(self, interaction_ids: list[int]) -> list[Interaction]:
        """Fetch interactions by interaction ids, ordered by created_at."""
        raise NotImplementedError

    # ==============================
    # Evaluation-overview support (default no-ops; backends override)
    # ==============================

    def count_sessions_with_shadow_content(
        self,
        from_ts: int,  # noqa: ARG002
        to_ts: int,  # noqa: ARG002
    ) -> int:
        """Return the number of sessions with non-empty shadow content in the window.

        Default implementation returns 0; concrete backends should override
        once shadow-mode publishing lands.
        """
        return 0

    def get_interactions_by_session(
        self,
        session_id: str,  # noqa: ARG002
    ) -> list[Interaction]:
        """Return the interactions belonging to a single session (default []).

        Default implementation returns []; concrete backends should override.
        """
        return []

    # ==============================
    # Braintrust connector (default no-ops; backends override)
    # ==============================

    def save_braintrust_connection(self, connection: BraintrustConnection) -> None:
        """Persist a Braintrust connection (default no-op).

        Concrete backends should upsert by `org_id`. The default no-op
        keeps tests and dev mode workable until per-backend implementations
        land.

        Args:
            connection (BraintrustConnection): Encrypted connection record.
        """

    def get_braintrust_connection(
        self,
        org_id: str,  # noqa: ARG002 — default no-op; concrete backends use it
    ) -> BraintrustConnection | None:
        """Fetch the persisted Braintrust connection for an org.

        Args:
            org_id (str): The Reflexio org.

        Returns:
            BraintrustConnection | None: The stored record, or None if the
                org has not connected (or no backend override yet).
        """
        return None

    def delete_braintrust_connection(
        self,
        org_id: str,  # noqa: ARG002 — default no-op; concrete backends use it
    ) -> None:
        """Delete the org's Braintrust connection (default no-op).

        Args:
            org_id (str): The Reflexio org to disconnect.
        """

    def save_imported_scores(self, scores: list[ImportedScore]) -> None:
        """Persist a batch of imported scorer outputs (default no-op).

        Concrete backends should upsert by `(source, source_run_id,
        scorer_name)` so re-syncs are idempotent.

        Args:
            scores (list[ImportedScore]): Scores to persist.
        """

    def get_imported_scores(
        self,
        org_id: str,  # noqa: ARG002
        from_ts: int,  # noqa: ARG002
        to_ts: int,  # noqa: ARG002
    ) -> list[ImportedScore]:
        """Return imported scores for the org in `[from_ts, to_ts]` (default []).

        Default implementation returns []; concrete backends override.
        Used by EvaluationOverviewService to surface Braintrust tiles.
        """
        return []
