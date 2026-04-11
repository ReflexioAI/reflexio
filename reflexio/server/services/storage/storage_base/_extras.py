from abc import abstractmethod

from reflexio.models.api_schema.domain import (
    Interaction,
    PlaybookAggregationChangeLog,
    ProfileChangeLog,
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
