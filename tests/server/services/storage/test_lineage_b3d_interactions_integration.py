"""Integration tests for B3d: interaction delete fts+row+vec atomicity (SQLite only).

Both delete_user_interaction and delete_all_interactions_for_user previously called
the self-committing _fts_delete helper BEFORE the interactions-row DELETE, leaving a
crash window where the FTS entry was durably gone but the interaction row still existed.

The fix rewrites both methods to do the FTS DELETE and the row DELETE inside a single
`with self._lock:` block with one `conn.commit()`, mirroring the already-correct
delete_all_interactions and delete_oldest_interactions siblings.

A follow-on fix (B3d vec) also adds interactions_vec cleanup to all four delete methods
inside the same atomic block, closing the orphaned-vector leak.

These tests assert that after each method: the interaction row is gone AND its
interactions_fts entry is also gone. When sqlite-vec is available, also assert that
the interactions_vec entry is gone.
"""

from __future__ import annotations

import pytest

from reflexio.models.api_schema.service_schemas import (
    DeleteUserInteractionRequest,
    Interaction,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store(tmp_path) -> SQLiteStorage:
    s = SQLiteStorage(org_id="b3d-org", db_path=str(tmp_path / "b3d.db"))
    s.migrate()
    return s


def _make_interaction(
    user_id: str = "u1",
    interaction_id: int = 1,
    content: str = "content",
    request_id: str = "req1",
) -> Interaction:
    import time

    return Interaction(
        interaction_id=interaction_id,
        user_id=user_id,
        request_id=request_id,
        content=content,
        created_at=int(time.time()),
    )


def _fts_count(s: SQLiteStorage, interaction_id: int) -> int:
    """Return the number of interactions_fts rows for a given interaction rowid."""
    row = s.conn.execute(
        "SELECT COUNT(*) AS cnt FROM interactions_fts WHERE rowid = ?",
        (interaction_id,),
    ).fetchone()
    return row["cnt"] if row else 0


def _interaction_row_count(s: SQLiteStorage, interaction_id: int) -> int:
    """Return 1 if the interactions row exists, 0 otherwise."""
    row = s.conn.execute(
        "SELECT COUNT(*) AS cnt FROM interactions WHERE interaction_id = ?",
        (interaction_id,),
    ).fetchone()
    return row["cnt"] if row else 0


def _vec_count(s: SQLiteStorage, interaction_id: int) -> int:
    """Return number of interactions_vec rows for a given rowid (0 when sqlite-vec absent)."""
    if not s._has_sqlite_vec:
        return 0
    row = s.conn.execute(
        "SELECT COUNT(*) AS cnt FROM interactions_vec WHERE rowid = ?",
        (interaction_id,),
    ).fetchone()
    return row["cnt"] if row else 0


# ---------------------------------------------------------------------------
# delete_user_interaction: row + FTS + vec all cleaned atomically
# ---------------------------------------------------------------------------


class TestDeleteUserInteractionAtomicity:
    def test_interaction_row_gone_after_delete(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_interaction("u1", _make_interaction("u1", 1, "click", "req1"))

        s.delete_user_interaction(
            DeleteUserInteractionRequest(user_id="u1", interaction_id=1)
        )

        assert _interaction_row_count(s, 1) == 0

    def test_fts_entry_gone_after_delete(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_interaction("u1", _make_interaction("u1", 2, "click", "req2"))
        assert _fts_count(s, 2) == 1

        s.delete_user_interaction(
            DeleteUserInteractionRequest(user_id="u1", interaction_id=2)
        )

        assert _fts_count(s, 2) == 0

    def test_vec_entry_gone_after_delete(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_interaction("u1", _make_interaction("u1", 3, "click", "req3"))
        if not s._has_sqlite_vec:
            pytest.skip("sqlite-vec not available")
        assert _vec_count(s, 3) == 1

        s.delete_user_interaction(
            DeleteUserInteractionRequest(user_id="u1", interaction_id=3)
        )

        assert _vec_count(s, 3) == 0

    def test_other_user_interaction_untouched(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_interaction("u1", _make_interaction("u1", 10, "a", "r10"))
        s.add_user_interaction("u2", _make_interaction("u2", 11, "b", "r11"))

        s.delete_user_interaction(
            DeleteUserInteractionRequest(user_id="u1", interaction_id=10)
        )

        assert _interaction_row_count(s, 10) == 0
        assert _interaction_row_count(s, 11) == 1
        assert _fts_count(s, 10) == 0
        assert _fts_count(s, 11) == 1
        if s._has_sqlite_vec:
            assert _vec_count(s, 10) == 0
            assert _vec_count(s, 11) == 1

    def test_delete_nonexistent_interaction_is_noop(self, tmp_path) -> None:
        s = _store(tmp_path)
        # Should not raise
        s.delete_user_interaction(
            DeleteUserInteractionRequest(user_id="u1", interaction_id=999)
        )
        assert _interaction_row_count(s, 999) == 0


# ---------------------------------------------------------------------------
# delete_all_interactions_for_user: rows + FTS + vec all cleaned atomically
# ---------------------------------------------------------------------------


class TestDeleteAllInteractionsForUserAtomicity:
    def test_interaction_rows_gone_after_delete(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_interaction("u1", _make_interaction("u1", 20, "a", "r20"))
        s.add_user_interaction("u1", _make_interaction("u1", 21, "b", "r21"))

        s.delete_all_interactions_for_user("u1")

        assert _interaction_row_count(s, 20) == 0
        assert _interaction_row_count(s, 21) == 0

    def test_fts_entries_gone_after_delete(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_interaction("u1", _make_interaction("u1", 30, "a", "r30"))
        s.add_user_interaction("u1", _make_interaction("u1", 31, "b", "r31"))
        assert _fts_count(s, 30) == 1
        assert _fts_count(s, 31) == 1

        s.delete_all_interactions_for_user("u1")

        assert _fts_count(s, 30) == 0
        assert _fts_count(s, 31) == 0

    def test_vec_entries_gone_after_delete(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_interaction("u1", _make_interaction("u1", 32, "a", "r32"))
        s.add_user_interaction("u1", _make_interaction("u1", 33, "b", "r33"))
        if not s._has_sqlite_vec:
            pytest.skip("sqlite-vec not available")
        assert _vec_count(s, 32) == 1
        assert _vec_count(s, 33) == 1

        s.delete_all_interactions_for_user("u1")

        assert _vec_count(s, 32) == 0
        assert _vec_count(s, 33) == 0

    def test_other_user_interactions_untouched(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_interaction("u1", _make_interaction("u1", 40, "a", "r40"))
        s.add_user_interaction("u2", _make_interaction("u2", 41, "b", "r41"))

        s.delete_all_interactions_for_user("u1")

        assert _interaction_row_count(s, 40) == 0
        assert _fts_count(s, 40) == 0
        assert _interaction_row_count(s, 41) == 1
        assert _fts_count(s, 41) == 1
        if s._has_sqlite_vec:
            assert _vec_count(s, 40) == 0
            assert _vec_count(s, 41) == 1

    def test_delete_all_interactions_for_user_no_interactions_is_noop(
        self, tmp_path
    ) -> None:
        s = _store(tmp_path)
        # Should not raise for a user with no interactions
        s.delete_all_interactions_for_user("u_empty")


# ---------------------------------------------------------------------------
# delete_all_interactions: rows + FTS + vec all cleaned atomically
# ---------------------------------------------------------------------------


class TestDeleteAllInteractionsAtomicity:
    def test_rows_fts_vec_all_gone_after_delete(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_interaction("u1", _make_interaction("u1", 50, "a", "r50"))
        s.add_user_interaction("u2", _make_interaction("u2", 51, "b", "r51"))

        s.delete_all_interactions()

        assert _interaction_row_count(s, 50) == 0
        assert _interaction_row_count(s, 51) == 0
        assert _fts_count(s, 50) == 0
        assert _fts_count(s, 51) == 0
        if s._has_sqlite_vec:
            assert _vec_count(s, 50) == 0
            assert _vec_count(s, 51) == 0

    def test_delete_all_interactions_no_rows_is_noop(self, tmp_path) -> None:
        s = _store(tmp_path)
        # Should not raise when table is empty
        s.delete_all_interactions()


# ---------------------------------------------------------------------------
# delete_oldest_interactions: rows + FTS + vec all cleaned atomically
# ---------------------------------------------------------------------------


class TestDeleteOldestInteractionsAtomicity:
    def test_oldest_rows_fts_vec_gone_after_delete(self, tmp_path) -> None:
        import time

        s = _store(tmp_path)
        # Insert two interactions with different created_at so ordering is deterministic
        old = Interaction(
            interaction_id=60,
            user_id="u1",
            request_id="r60",
            content="old",
            created_at=int(time.time()) - 100,
        )
        new = Interaction(
            interaction_id=61,
            user_id="u1",
            request_id="r61",
            content="new",
            created_at=int(time.time()),
        )
        s.add_user_interaction("u1", old)
        s.add_user_interaction("u1", new)

        deleted = s.delete_oldest_interactions(1)

        assert deleted == 1
        assert _interaction_row_count(s, 60) == 0
        assert _fts_count(s, 60) == 0
        if s._has_sqlite_vec:
            assert _vec_count(s, 60) == 0
        # Newer interaction untouched
        assert _interaction_row_count(s, 61) == 1
        assert _fts_count(s, 61) == 1
        if s._has_sqlite_vec:
            assert _vec_count(s, 61) == 1

    def test_delete_oldest_interactions_count_zero_is_noop(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_interaction("u1", _make_interaction("u1", 70, "x", "r70"))
        deleted = s.delete_oldest_interactions(0)
        assert deleted == 0
        assert _interaction_row_count(s, 70) == 1

    def test_delete_oldest_interactions_empty_table_is_noop(self, tmp_path) -> None:
        s = _store(tmp_path)
        deleted = s.delete_oldest_interactions(5)
        assert deleted == 0


# ---------------------------------------------------------------------------
# delete_user_interaction: cross-user sidecar protection (Fix A)
# ---------------------------------------------------------------------------


class TestDeleteUserInteractionCrossUser:
    """Fix A: u1 deleting u2's interaction_id leaves u2's row+fts+vec intact."""

    def test_cross_user_row_untouched(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_interaction("u2", _make_interaction("u2", 80, "b", "r80"))
        # u1 tries to delete u2's interaction_id
        s.delete_user_interaction(
            DeleteUserInteractionRequest(user_id="u1", interaction_id=80)
        )
        assert _interaction_row_count(s, 80) == 1

    def test_cross_user_fts_untouched(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_interaction("u2", _make_interaction("u2", 81, "b", "r81"))
        assert _fts_count(s, 81) == 1
        s.delete_user_interaction(
            DeleteUserInteractionRequest(user_id="u1", interaction_id=81)
        )
        assert _fts_count(s, 81) == 1

    def test_cross_user_vec_untouched(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_interaction("u2", _make_interaction("u2", 82, "b", "r82"))
        if not s._has_sqlite_vec:
            pytest.skip("sqlite-vec not available")
        assert _vec_count(s, 82) == 1
        s.delete_user_interaction(
            DeleteUserInteractionRequest(user_id="u1", interaction_id=82)
        )
        assert _vec_count(s, 82) == 1
