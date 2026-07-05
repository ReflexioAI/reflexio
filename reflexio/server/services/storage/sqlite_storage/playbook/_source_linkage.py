"""Agent playbook source-linkage methods for SQLite storage."""

import sqlite3
from collections.abc import Sequence
from typing import Any

from reflexio.models.api_schema.service_schemas import AgentPlaybookSourceWindow

from .._base import (
    SQLiteStorageBase,
    _json_dumps,
    _json_loads,
)


class PlaybookSourceLinkageMixin:
    """Mixin providing agent playbook source-linkage CRUD for SQLite storage."""

    # Type hints for instance attributes/methods provided by SQLiteStorageBase via MRO
    _lock: Any
    conn: sqlite3.Connection
    _fetchall: Any
    _subject_ref_for_user_id: Any
    _assert_subject_writable_locked: Any
    _own_transaction: Any

    @SQLiteStorageBase.handle_exceptions
    def set_source_user_playbook_ids_for_agent_playbook(
        self, agent_playbook_id: int, user_playbook_ids: list[int]
    ) -> None:
        self.set_source_windows_for_agent_playbook(
            agent_playbook_id,
            [
                AgentPlaybookSourceWindow(
                    user_playbook_id=upid, source_interaction_ids=[]
                )
                for upid in user_playbook_ids
            ],
        )

    @SQLiteStorageBase.handle_exceptions
    def get_source_user_playbook_ids_for_agent_playbook(
        self, agent_playbook_id: int
    ) -> list[int]:
        return [
            window.user_playbook_id
            for window in self.get_source_windows_for_agent_playbook(agent_playbook_id)
        ]

    @SQLiteStorageBase.handle_exceptions
    def get_source_user_playbook_ids_for_agent_playbooks(
        self, agent_playbook_ids: Sequence[int]
    ) -> dict[int, list[int]]:
        if not agent_playbook_ids:
            return {}
        unique_ids = list(dict.fromkeys(int(apid) for apid in agent_playbook_ids))
        ph = ",".join("?" for _ in unique_ids)
        rows = self._fetchall(
            f"""SELECT agent_playbook_id, user_playbook_id
                FROM agent_playbook_source_user_playbooks
                WHERE agent_playbook_id IN ({ph})
                ORDER BY agent_playbook_id ASC, user_playbook_id ASC""",
            unique_ids,
        )
        by_agent_id: dict[int, list[int]] = {apid: [] for apid in unique_ids}
        seen_by_agent_id: dict[int, set[int]] = {apid: set() for apid in unique_ids}
        for row in rows:
            agent_playbook_id = int(row["agent_playbook_id"])
            user_playbook_id = int(row["user_playbook_id"])
            seen = seen_by_agent_id.setdefault(agent_playbook_id, set())
            if user_playbook_id not in seen:
                by_agent_id.setdefault(agent_playbook_id, []).append(user_playbook_id)
                seen.add(user_playbook_id)
        return by_agent_id

    @SQLiteStorageBase.handle_exceptions
    def set_source_windows_for_agent_playbook(
        self,
        agent_playbook_id: int,
        source_windows: list[AgentPlaybookSourceWindow],
    ) -> None:
        by_id: dict[int, list[int]] = {}
        for window in source_windows:
            ids = by_id.setdefault(window.user_playbook_id, [])
            seen = set(ids)
            for source_id in window.source_interaction_ids:
                if source_id not in seen:
                    ids.append(source_id)
                    seen.add(source_id)
        with self._lock:
            own_txn = self._own_transaction()
            try:
                if own_txn:
                    self.conn.execute("BEGIN IMMEDIATE")
                for user_playbook_id in by_id:
                    row = self.conn.execute(
                        """SELECT user_id, governance_subject_ref FROM user_playbooks
                           WHERE user_playbook_id = ?""",
                        (user_playbook_id,),
                    ).fetchone()
                    if row is None:
                        raise ValueError(
                            f"User playbook {user_playbook_id} not found for source window"
                        )
                    subject_ref = row["governance_subject_ref"]
                    if not isinstance(subject_ref, str) or not subject_ref:
                        subject_ref = self._subject_ref_for_user_id(str(row["user_id"]))
                    self._assert_subject_writable_locked(subject_ref)
                self.conn.execute(
                    "DELETE FROM agent_playbook_source_user_playbooks WHERE agent_playbook_id = ?",
                    (agent_playbook_id,),
                )
                self.conn.executemany(
                    """INSERT OR IGNORE INTO agent_playbook_source_user_playbooks
                       (agent_playbook_id, user_playbook_id, source_interaction_ids)
                       VALUES (?, ?, ?)""",
                    [
                        (
                            agent_playbook_id,
                            upid,
                            _json_dumps(source_interaction_ids) or "[]",
                        )
                        for upid, source_interaction_ids in by_id.items()
                    ],
                )
                if own_txn:
                    self.conn.commit()
            except Exception:
                if own_txn:
                    self.conn.rollback()
                raise

    @SQLiteStorageBase.handle_exceptions
    def get_source_windows_for_agent_playbook(
        self, agent_playbook_id: int
    ) -> list[AgentPlaybookSourceWindow]:
        rows = self._fetchall(
            """SELECT user_playbook_id, source_interaction_ids
               FROM agent_playbook_source_user_playbooks
               WHERE agent_playbook_id = ?
               ORDER BY user_playbook_id ASC""",
            (agent_playbook_id,),
        )
        return [
            AgentPlaybookSourceWindow(
                user_playbook_id=int(row["user_playbook_id"]),
                source_interaction_ids=_json_loads(row["source_interaction_ids"]) or [],
            )
            for row in rows
        ]
