import logging
from datetime import UTC, datetime

from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    AgentSuccessEvaluationResult,
    Interaction,
    PlaybookAggregationChangeLog,
    ProfileChangeLog,
    Status,
    UserPlaybook,
    UserProfile,
)

logger = logging.getLogger(__name__)


class ExtrasMixin:
    # ------------------------------------------------------------------
    # Interaction helpers
    # ------------------------------------------------------------------

    def get_interactions_by_request_ids(
        self, request_ids: list[str]
    ) -> list[Interaction]:
        if not request_ids:
            return []
        request_id_set = set(request_ids)
        interactions = self._list_entities_recursive(
            self._interactions_dir(), Interaction
        )
        return [i for i in interactions if i.request_id in request_id_set]

    # ------------------------------------------------------------------
    # Dashboard methods
    # ------------------------------------------------------------------

    def get_dashboard_stats(self, days_back: int = 30) -> dict:  # noqa: C901
        """Get comprehensive dashboard statistics including counts and time-series data.

        Args:
            days_back (int): Number of days to include in time series data

        Returns:
            dict: Dictionary containing current_period, previous_period, and raw time_series data
        """
        current_time = int(datetime.now(UTC).timestamp())
        seconds_in_period = days_back * 24 * 60 * 60
        current_period_start = current_time - seconds_in_period
        previous_period_start = current_period_start - seconds_in_period

        current_stats = {
            "total_profiles": 0,
            "total_interactions": 0,
            "total_playbooks": 0,
            "success_rate": 0.0,
        }
        previous_stats = {
            "total_profiles": 0,
            "total_interactions": 0,
            "total_playbooks": 0,
            "success_rate": 0.0,
        }

        interactions_ts: list[dict] = []
        profiles_ts: list[dict] = []
        playbooks_ts: list[dict] = []
        evaluations_ts: list[dict] = []

        # Process interactions
        all_interactions = self._list_entities_recursive(
            self._interactions_dir(), Interaction
        )
        for interaction in all_interactions:
            timestamp = interaction.created_at
            if timestamp >= current_period_start:
                current_stats["total_interactions"] += 1
                interactions_ts.append({"timestamp": timestamp, "value": 1})
            elif timestamp >= previous_period_start:
                previous_stats["total_interactions"] += 1

        # Process profiles
        all_profiles = self._list_entities_recursive(self._profiles_dir(), UserProfile)
        for profile in all_profiles:
            timestamp = profile.last_modified_timestamp
            if timestamp >= current_period_start:
                current_stats["total_profiles"] += 1
                profiles_ts.append({"timestamp": timestamp, "value": 1})
            elif timestamp >= previous_period_start:
                previous_stats["total_profiles"] += 1

        # Process user playbooks
        user_playbook_count_current = 0
        user_playbook_count_previous = 0
        all_user_playbooks = self._list_entities(
            self._user_playbooks_dir(), UserPlaybook
        )
        for playbook in all_user_playbooks:
            timestamp = playbook.created_at
            if timestamp >= current_period_start:
                user_playbook_count_current += 1
                playbooks_ts.append({"timestamp": timestamp, "value": 1})
            elif timestamp >= previous_period_start:
                user_playbook_count_previous += 1

        # Process agent playbooks
        agent_playbook_count_current = 0
        agent_playbook_count_previous = 0
        all_agent_playbooks = self._list_entities(
            self._agent_playbooks_dir(), AgentPlaybook
        )
        for playbook in all_agent_playbooks:
            timestamp = playbook.created_at
            if timestamp >= current_period_start:
                agent_playbook_count_current += 1
            elif timestamp >= previous_period_start:
                agent_playbook_count_previous += 1

        current_stats["total_playbooks"] = (
            user_playbook_count_current + agent_playbook_count_current
        )
        previous_stats["total_playbooks"] = (
            user_playbook_count_previous + agent_playbook_count_previous
        )

        # Process evaluations
        success_count_current = 0
        total_eval_current = 0
        success_count_previous = 0
        total_eval_previous = 0

        all_evals = self._list_entities(
            self._evaluations_dir(), AgentSuccessEvaluationResult
        )
        for result in all_evals:
            timestamp = result.created_at
            if timestamp >= current_period_start:
                total_eval_current += 1
                if result.is_success:
                    success_count_current += 1
                success_value = 100 if result.is_success else 0
                evaluations_ts.append({"timestamp": timestamp, "value": success_value})
            elif timestamp >= previous_period_start:
                total_eval_previous += 1
                if result.is_success:
                    success_count_previous += 1

        current_stats["success_rate"] = (
            (success_count_current / total_eval_current * 100)
            if total_eval_current > 0
            else 0.0
        )
        previous_stats["success_rate"] = (
            (success_count_previous / total_eval_previous * 100)
            if total_eval_previous > 0
            else 0.0
        )

        return {
            "current_period": current_stats,
            "previous_period": previous_stats,
            "interactions_time_series": sorted(
                interactions_ts, key=lambda x: x["timestamp"]
            ),
            "profiles_time_series": sorted(profiles_ts, key=lambda x: x["timestamp"]),
            "playbooks_time_series": sorted(playbooks_ts, key=lambda x: x["timestamp"]),
            "evaluations_time_series": sorted(
                evaluations_ts, key=lambda x: x["timestamp"]
            ),
        }

    # ------------------------------------------------------------------
    # Statistics methods
    # ------------------------------------------------------------------

    def get_profile_statistics(self) -> dict:
        """Get profile count statistics by status.

        Returns:
            dict with keys: current_count, pending_count, archived_count, expiring_soon_count
        """
        current_timestamp = int(datetime.now(UTC).timestamp())
        expiring_soon_timestamp = current_timestamp + (7 * 24 * 60 * 60)

        stats = {
            "current_count": 0,
            "pending_count": 0,
            "archived_count": 0,
            "expiring_soon_count": 0,
        }

        all_profiles = self._list_entities_recursive(self._profiles_dir(), UserProfile)
        for profile in all_profiles:
            if profile.status is None:
                stats["current_count"] += 1
            elif profile.status == Status.PENDING:
                stats["pending_count"] += 1
            elif profile.status == Status.ARCHIVED:
                stats["archived_count"] += 1

            if (
                profile.status is None
                and profile.expiration_timestamp > current_timestamp
                and profile.expiration_timestamp <= expiring_soon_timestamp
            ):
                stats["expiring_soon_count"] += 1

        return stats

    # ------------------------------------------------------------------
    # Profile Change Log methods    # ------------------------------------------------------------------

    def add_profile_change_log(self, profile_change_log: ProfileChangeLog) -> None:
        with self._lock:
            next_id = self._next_id(self._profile_change_logs_dir())
            path = self._profile_change_logs_dir() / f"{next_id}.json"
            self._write_entity(path, profile_change_log)

    def get_profile_change_logs(self, limit: int = 100) -> list[ProfileChangeLog]:
        with self._lock:
            logs = self._list_entities(
                self._profile_change_logs_dir(), ProfileChangeLog
            )
        return logs[:limit]

    def delete_profile_change_log_for_user(self, user_id: str) -> None:
        with self._lock:
            log_dir = self._profile_change_logs_dir()
            if not log_dir.exists():
                return
            for p in list(log_dir.glob("*.json")):
                log = self._read_entity(p, ProfileChangeLog)
                if log.user_id == user_id:
                    p.unlink()

    def delete_all_profile_change_logs(self) -> None:
        with self._lock:
            self._clear_dir(self._profile_change_logs_dir())

    # ------------------------------------------------------------------
    # Playbook Aggregation Change Log methods
    # ------------------------------------------------------------------

    def add_playbook_aggregation_change_log(
        self, change_log: PlaybookAggregationChangeLog
    ) -> None:
        with self._lock:
            next_id = self._next_id(self._playbook_agg_change_logs_dir())
            path = self._playbook_agg_change_logs_dir() / f"{next_id}.json"
            self._write_entity(path, change_log)

    def get_playbook_aggregation_change_logs(
        self,
        playbook_name: str,
        agent_version: str,
        limit: int = 100,
    ) -> list[PlaybookAggregationChangeLog]:
        with self._lock:
            all_logs = self._list_entities(
                self._playbook_agg_change_logs_dir(), PlaybookAggregationChangeLog
            )
        logs = [
            log
            for log in all_logs
            if log.playbook_name == playbook_name and log.agent_version == agent_version
        ]
        logs.sort(key=lambda x: x.created_at, reverse=True)
        return logs[:limit]

    def delete_all_playbook_aggregation_change_logs(self) -> None:
        with self._lock:
            self._clear_dir(self._playbook_agg_change_logs_dir())
