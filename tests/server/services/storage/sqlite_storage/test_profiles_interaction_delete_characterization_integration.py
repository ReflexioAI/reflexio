"""Characterization tests: SQLite interaction-delete row removal (Tier-1 Task 1).

Phase-A characterization net for the InteractionStore bucket of the ``ProfileMixin``
decomposition (``_profiles.py`` -> ProfileStore / InteractionStore / Search). These pin
the CURRENT row-removal contract of the four interaction-delete methods so a later
VERBATIM code move into the InteractionStore bucket is provably safe:
  - ``delete_user_interaction``
  - ``delete_all_interactions_for_user``
  - ``delete_all_interactions``
  - ``delete_oldest_interactions``

Scope note (deliberate): these tests assert ONLY that the interaction ROW is gone after
each delete. The atomic FTS + vec cleanup for all four methods (the interactions_fts /
interactions_vec sidecar rows are removed inside the same lock/commit as the row DELETE)
is already fully characterized by
``tests/server/services/storage/test_lineage_b3d_interactions_integration.py`` — this
file intentionally does not duplicate those FTS/vec assertions, it complements them with
a bucket-local row-removal net.
"""

from __future__ import annotations

import time

import pytest

from reflexio.models.api_schema.service_schemas import (
    DeleteUserInteractionRequest,
    Interaction,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def _store(tmp_path, org_id: str = "int-del-char-org") -> SQLiteStorage:
    s = SQLiteStorage(org_id=org_id, db_path=str(tmp_path / f"{org_id}.db"))
    s.migrate()
    return s


def _make_interaction(
    interaction_id: int,
    user_id: str = "u1",
    content: str = "content",
    request_id: str = "r1",
    created_at: int | None = None,
) -> Interaction:
    return Interaction(
        interaction_id=interaction_id,
        user_id=user_id,
        request_id=request_id,
        content=content,
        created_at=created_at if created_at is not None else int(time.time()),
    )


def _row_count(s: SQLiteStorage, interaction_id: int) -> int:
    row = s.conn.execute(
        "SELECT COUNT(*) AS cnt FROM interactions WHERE interaction_id = ?",
        (interaction_id,),
    ).fetchone()
    return row["cnt"] if row else 0


class TestInteractionDeleteRowRemoval:
    def test_delete_user_interaction_removes_row(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_interaction("u1", _make_interaction(1))
        s.delete_user_interaction(
            DeleteUserInteractionRequest(user_id="u1", interaction_id=1)
        )
        assert _row_count(s, 1) == 0

    def test_delete_all_interactions_for_user_removes_rows(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_interaction("u1", _make_interaction(1))
        s.add_user_interaction("u1", _make_interaction(2))
        s.add_user_interaction("u2", _make_interaction(3, user_id="u2"))

        s.delete_all_interactions_for_user("u1")

        assert _row_count(s, 1) == 0
        assert _row_count(s, 2) == 0
        # Another user's row is untouched.
        assert _row_count(s, 3) == 1

    def test_delete_all_interactions_removes_all_rows(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_interaction("u1", _make_interaction(1))
        s.add_user_interaction("u2", _make_interaction(2, user_id="u2"))

        s.delete_all_interactions()

        assert _row_count(s, 1) == 0
        assert _row_count(s, 2) == 0

    def test_delete_oldest_interactions_removes_oldest_row(self, tmp_path) -> None:
        s = _store(tmp_path)
        now = int(time.time())
        s.add_user_interaction("u1", _make_interaction(1, created_at=now - 100))
        s.add_user_interaction("u1", _make_interaction(2, created_at=now))

        deleted = s.delete_oldest_interactions(1)

        assert deleted == 1
        assert _row_count(s, 1) == 0  # oldest removed
        assert _row_count(s, 2) == 1  # newest retained
