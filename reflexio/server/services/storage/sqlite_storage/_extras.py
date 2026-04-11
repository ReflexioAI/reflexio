"""Analytics and change log methods for SQLite storage."""

import sqlite3
from typing import Any

from reflexio.models.api_schema.service_schemas import (
    Interaction,
    PlaybookAggregationChangeLog,
    ProfileChangeLog,
)

from ._base import (
    SQLiteStorageBase,
    _epoch_now,
    _epoch_to_iso,
    _iso_to_epoch,
    _json_dumps,
    _row_to_interaction,
    _row_to_playbook_aggregation_change_log,
    _row_to_profile_change_log,
)


class ExtrasMixin:
    """Mixin providing analytics, change log, and misc operations."""

    # Type hints for instance attributes/methods provided by SQLiteStorageBase via MRO
    _lock: Any
    conn: sqlite3.Connection
    _execute: Any
    _fetchall: Any

    # ------------------------------------------------------------------
    # Interaction helpers
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def get_interactions_by_request_ids(
        self, request_ids: list[str]
    ) -> list[Interaction]:
        if not request_ids:
            return []
        ph = ",".join("?" for _ in request_ids)
        rows = self._fetchall(
            f"SELECT * FROM interactions WHERE request_id IN ({ph}) ORDER BY created_at ASC",  # noqa: S608
            request_ids,
        )
        return [_row_to_interaction(r) for r in rows]

    _fetchone: Any
    _fetchall: Any

    # ------------------------------------------------------------------
    # Dashboard / Analytics methods
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def get_dashboard_stats(self, days_back: int = 30) -> dict:
        current_time = _epoch_now()
        seconds_in_period = days_back * 24 * 60 * 60
        current_start = current_time - seconds_in_period
        previous_start = current_start - seconds_in_period

        current_start_iso = _epoch_to_iso(current_start)
        current_time_iso = _epoch_to_iso(current_time)
        previous_start_iso = _epoch_to_iso(previous_start)

        def count_in(table: str, time_col: str, start: Any, end: Any) -> int:
            row = self._fetchone(
                f"SELECT COUNT(*) as cnt FROM {table} WHERE {time_col} >= ? AND {time_col} <= ?",
                (start, end),
            )
            return row["cnt"] if row else 0

        def count_in_lt(table: str, time_col: str, start: Any, end: Any) -> int:
            row = self._fetchone(
                f"SELECT COUNT(*) as cnt FROM {table} WHERE {time_col} >= ? AND {time_col} < ?",
                (start, end),
            )
            return row["cnt"] if row else 0

        current_stats: dict[str, int | float] = {
            "total_interactions": count_in(
                "interactions", "created_at", current_start_iso, current_time_iso
            ),
            "total_profiles": count_in(
                "profiles", "last_modified_timestamp", current_start, current_time
            ),
            "total_playbooks": (
                count_in(
                    "user_playbooks", "created_at", current_start_iso, current_time_iso
                )
                + count_in(
                    "agent_playbooks", "created_at", current_start_iso, current_time_iso
                )
            ),
        }

        previous_stats: dict[str, int | float] = {
            "total_interactions": count_in_lt(
                "interactions", "created_at", previous_start_iso, current_start_iso
            ),
            "total_profiles": count_in_lt(
                "profiles", "last_modified_timestamp", previous_start, current_start
            ),
            "total_playbooks": (
                count_in_lt(
                    "user_playbooks",
                    "created_at",
                    previous_start_iso,
                    current_start_iso,
                )
                + count_in_lt(
                    "agent_playbooks",
                    "created_at",
                    previous_start_iso,
                    current_start_iso,
                )
            ),
        }

        # Success rates
        def calc_success_rate(rows: list[sqlite3.Row]) -> float:
            if not rows:
                return 0.0
            total = len(rows)
            success = sum(1 for r in rows if r["is_success"])
            return success / total * 100

        eval_current = self._fetchall(
            "SELECT is_success FROM agent_success_evaluation_result WHERE created_at >= ? AND created_at <= ?",
            (current_start_iso, current_time_iso),
        )
        eval_previous = self._fetchall(
            "SELECT is_success FROM agent_success_evaluation_result WHERE created_at >= ? AND created_at < ?",
            (previous_start_iso, current_start_iso),
        )
        current_stats["success_rate"] = calc_success_rate(eval_current)
        previous_stats["success_rate"] = calc_success_rate(eval_previous)

        # Time series
        interactions_ts = self._fetchall(
            "SELECT created_at FROM interactions WHERE created_at >= ? AND created_at <= ? ORDER BY created_at",
            (current_start_iso, current_time_iso),
        )
        profiles_ts = self._fetchall(
            "SELECT last_modified_timestamp FROM profiles WHERE last_modified_timestamp >= ? AND last_modified_timestamp <= ? ORDER BY last_modified_timestamp",
            (current_start, current_time),
        )
        playbooks_ts = self._fetchall(
            "SELECT created_at FROM user_playbooks WHERE created_at >= ? AND created_at <= ? ORDER BY created_at",
            (current_start_iso, current_time_iso),
        )
        evals_ts = self._fetchall(
            "SELECT created_at, is_success FROM agent_success_evaluation_result WHERE created_at >= ? AND created_at <= ? ORDER BY created_at",
            (current_start_iso, current_time_iso),
        )

        return {
            "current_period": current_stats,
            "previous_period": previous_stats,
            "interactions_time_series": [
                {"timestamp": _iso_to_epoch(r["created_at"]), "value": 1}
                for r in interactions_ts
            ],
            "profiles_time_series": [
                {"timestamp": r["last_modified_timestamp"], "value": 1}
                for r in profiles_ts
            ],
            "playbooks_time_series": [
                {"timestamp": _iso_to_epoch(r["created_at"]), "value": 1}
                for r in playbooks_ts
            ],
            "evaluations_time_series": [
                {
                    "timestamp": _iso_to_epoch(r["created_at"]),
                    "value": 100 if r["is_success"] else 0,
                }
                for r in evals_ts
            ],
        }

    # ------------------------------------------------------------------
    # Statistics methods
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def get_profile_statistics(self) -> dict:
        current_ts = _epoch_now()
        expiring_soon_ts = current_ts + (7 * 24 * 60 * 60)

        rows = self._fetchall(
            "SELECT status, expiration_timestamp FROM profiles WHERE expiration_timestamp >= ?",
            (current_ts,),
        )
        stats = {
            "current_count": 0,
            "pending_count": 0,
            "archived_count": 0,
            "expiring_soon_count": 0,
        }
        for r in rows:
            s = r["status"]
            exp = r["expiration_timestamp"]
            if s is None:
                stats["current_count"] += 1
                if exp is not None and exp <= expiring_soon_ts:
                    stats["expiring_soon_count"] += 1
            elif s == "pending":
                stats["pending_count"] += 1
            elif s == "archived":
                stats["archived_count"] += 1
        return stats

    # ------------------------------------------------------------------
    # Profile Change Log methods
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def add_profile_change_log(self, profile_change_log: ProfileChangeLog) -> None:
        self._execute(
            """INSERT INTO profile_change_logs
               (user_id, request_id, created_at, added_profiles, removed_profiles, mentioned_profiles)
               VALUES (?,?,?,?,?,?)""",
            (
                profile_change_log.user_id,
                profile_change_log.request_id,
                profile_change_log.created_at,
                _json_dumps(
                    [p.model_dump() for p in profile_change_log.added_profiles]
                ),
                _json_dumps(
                    [p.model_dump() for p in profile_change_log.removed_profiles]
                ),
                _json_dumps(
                    [p.model_dump() for p in profile_change_log.mentioned_profiles]
                ),
            ),
        )

    @SQLiteStorageBase.handle_exceptions
    def get_profile_change_logs(self, limit: int = 100) -> list[ProfileChangeLog]:
        rows = self._fetchall(
            "SELECT * FROM profile_change_logs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [_row_to_profile_change_log(r) for r in rows]

    @SQLiteStorageBase.handle_exceptions
    def delete_profile_change_log_for_user(self, user_id: str) -> None:
        self._execute("DELETE FROM profile_change_logs WHERE user_id = ?", (user_id,))

    @SQLiteStorageBase.handle_exceptions
    def delete_all_profile_change_logs(self) -> None:
        self._execute("DELETE FROM profile_change_logs")

    # ------------------------------------------------------------------
    # Playbook Aggregation Change Log methods
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def add_playbook_aggregation_change_log(
        self, change_log: PlaybookAggregationChangeLog
    ) -> None:
        self._execute(
            """INSERT INTO playbook_aggregation_change_logs
               (created_at, playbook_name, agent_version, run_mode,
                added_playbooks, removed_playbooks, updated_playbooks)
               VALUES (?,?,?,?,?,?,?)""",
            (
                change_log.created_at,
                change_log.playbook_name,
                change_log.agent_version,
                change_log.run_mode,
                _json_dumps(
                    [fb.model_dump() for fb in change_log.added_agent_playbooks]
                ),
                _json_dumps(
                    [fb.model_dump() for fb in change_log.removed_agent_playbooks]
                ),
                _json_dumps(
                    [
                        {"before": e.before.model_dump(), "after": e.after.model_dump()}
                        for e in change_log.updated_agent_playbooks
                    ]
                ),
            ),
        )

    @SQLiteStorageBase.handle_exceptions
    def get_playbook_aggregation_change_logs(
        self,
        playbook_name: str,
        agent_version: str,
        limit: int = 100,
    ) -> list[PlaybookAggregationChangeLog]:
        rows = self._fetchall(
            """SELECT * FROM playbook_aggregation_change_logs
               WHERE playbook_name = ? AND agent_version = ?
               ORDER BY created_at DESC LIMIT ?""",
            (playbook_name, agent_version, limit),
        )
        return [_row_to_playbook_aggregation_change_log(r) for r in rows]

    @SQLiteStorageBase.handle_exceptions
    def delete_all_playbook_aggregation_change_logs(self) -> None:
        self._execute("DELETE FROM playbook_aggregation_change_logs")
