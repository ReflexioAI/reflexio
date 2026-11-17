"""Durable latest accepted-access epochs on the existing SQLite writer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from reflexio.server import search_runtime
from reflexio.server.services.profile.attention import ATTENTION_BATCH_SIZE, bounded_ids
from reflexio.server.services.profile.lifecycle import whole_epoch

ATTENTION_DDL = """
CREATE TABLE IF NOT EXISTS profile_access_epochs (
    org_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    last_served_at INTEGER NOT NULL CHECK(last_served_at >= 0),
    PRIMARY KEY (org_id, user_id, profile_id)
);
"""


class ProfileAttentionStoreMixin:
    conn: Any
    _lock: Any
    org_id: str
    commit_scope: Any
    _subject_ref_for_user_id: Any
    _assert_subject_writable_locked: Any

    def get_profile_access_epochs(
        self, *, user_id: str, profile_ids: Sequence[str]
    ) -> dict[str, int]:
        ids = bounded_ids(user_id, profile_ids)
        found: dict[str, int] = {}
        with self._lock:
            for offset in range(0, len(ids), ATTENTION_BATCH_SIZE):
                search_runtime.checkpoint()
                batch = ids[offset : offset + ATTENTION_BATCH_SIZE]
                placeholders = ",".join("?" for _ in batch)
                rows = self.conn.execute(
                    "SELECT profile_id,last_served_at FROM profile_access_epochs "
                    f"WHERE org_id=? AND user_id=? AND profile_id IN ({placeholders})",  # noqa: S608
                    [self.org_id, user_id, *batch],
                ).fetchall()
                found.update({str(row[0]): int(row[1]) for row in rows})
        return found

    def record_served_profile_access(
        self, *, user_id: str, profile_ids: Sequence[str], accessed_at: int
    ) -> None:
        ids = bounded_ids(user_id, profile_ids)
        whole_epoch(accessed_at)
        if not ids:
            return
        with self.commit_scope():
            self._assert_subject_writable_locked(self._subject_ref_for_user_id(user_id))
            # The content-free old epoch survives individual payload erasure.
            # Only an existing nonempty row may receive a NEW access epoch;
            # whole-owner erasure clears this ledger under its write barrier.
            self.conn.executemany(
                """INSERT INTO profile_access_epochs(org_id,user_id,profile_id,last_served_at)
                   SELECT ?,user_id,profile_id,? FROM profiles
                   WHERE user_id=? AND profile_id=? AND content<>''
                   ON CONFLICT(org_id,user_id,profile_id) DO UPDATE
                   SET last_served_at=max(last_served_at,excluded.last_served_at)""",
                [(self.org_id, accessed_at, user_id, identity) for identity in ids],
            )
