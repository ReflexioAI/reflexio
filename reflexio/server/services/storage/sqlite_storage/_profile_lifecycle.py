"""Profile payload reads and erasure under the existing SQLite writer owner."""

from typing import Any

from reflexio.server.services.profile.lifecycle import whole_epoch
from reflexio.server.services.profile.lifecycle_deadlines import PAYLOAD_FIELDS


class SQLiteProfileLifecycleMixin:
    _fetchall: Any
    in_commit_scope: bool
    delete_profiles_by_ids: Any

    def profile_lifecycle_due_users(
        self, *, now: int, after_user_id: str = ""
    ) -> list[str]:
        whole_epoch(now)
        rows = self._fetchall(
            "SELECT DISTINCT user_id FROM profiles WHERE user_id>? "
            "AND expiration_timestamp<? "
            "AND json_extract(custom_features, '$.profile_lifecycle_v1.explicit_storage_deadline') IS NOT NULL "
            "ORDER BY user_id LIMIT 256",
            (after_user_id, now),
        )
        return [str(row["user_id"]) for row in rows]

    def profile_lifecycle_rows(
        self,
        user_id: str,
        project_id: str,
        *,
        after_profile_id: str = "",
        profile_id: str | None = None,
        lock: bool = False,
    ) -> list[dict[str, Any]]:
        if not user_id or project_id:
            raise ValueError("SQLite lifecycle requires a user and no project")
        if lock and not self.in_commit_scope:
            raise ValueError("Locked lifecycle read requires its publication scope")
        predicate = "profile_id=?" if profile_id is not None else "profile_id>?"
        # Identifiers are fixed schema fields; values remain bound parameters.
        rows = self._fetchall(
            f"SELECT {','.join(PAYLOAD_FIELDS)} FROM profiles "
            f"WHERE user_id=? AND {predicate} ORDER BY profile_id LIMIT ?",  # noqa: S608
            (
                user_id,
                profile_id if profile_id is not None else after_profile_id,
                1 if profile_id is not None else 256,
            ),
        )
        return [dict(row) for row in rows]

    def delete_profile_lifecycle_payload(
        self, user_id: str, project_id: str, profile_id: str, *, reason: str
    ) -> int:
        if not self.in_commit_scope:
            raise ValueError("Lifecycle erasure requires its publication scope")
        if not self.profile_lifecycle_rows(
            user_id, project_id, profile_id=profile_id, lock=True
        ):
            return 0
        return self.delete_profiles_by_ids([profile_id], reason=reason)
