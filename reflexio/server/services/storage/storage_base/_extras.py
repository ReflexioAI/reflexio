from abc import abstractmethod

from reflexio.models.api_schema.braintrust_schema import (
    BraintrustConnection,
    ImportedScore,
)
from reflexio.models.api_schema.domain import (
    Interaction,
)
from reflexio.models.api_schema.internal_schema import SessionCitation
from reflexio.models.api_schema.retriever_schema import PlaybookApplicationStat


class ExtrasMixin:
    """Mixin for dashboard, profile change logs, and misc methods."""

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
    def get_profile_statistics(self) -> dict:
        """Get profile count statistics by status.

        Returns:
            dict with keys: current_count, pending_count, archived_count, expiring_soon_count
        """
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

    def get_citations_by_session_ids(
        self,
        session_ids: list[str],
    ) -> list[SessionCitation]:
        """Return rule/profile citations for the requested sessions.

        Default implementation uses ``get_interactions_by_session`` so legacy
        backends keep working. SQL backends should override with a bulk
        request/interactions join.
        """
        out: list[SessionCitation] = []
        for session_id in set(session_ids):
            for interaction in self.get_interactions_by_session(session_id):
                for cite in getattr(interaction, "citations", []) or []:
                    if isinstance(cite, dict):
                        kind = cite.get("kind")
                        real_id = cite.get("real_id")
                        title = cite.get("title") or ""
                    else:
                        kind = getattr(cite, "kind", None)
                        real_id = getattr(cite, "real_id", None)
                        title = getattr(cite, "title", "") or ""
                    if kind and real_id:
                        out.append(
                            SessionCitation(
                                user_id="",
                                session_id=session_id,
                                kind=str(kind),
                                real_id=str(real_id),
                                title=str(title),
                            )
                        )
        return out

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
