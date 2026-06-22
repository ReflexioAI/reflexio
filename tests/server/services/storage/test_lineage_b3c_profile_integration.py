"""Integration tests for B3c: profile delete search-cleanup after commit + supersede guard.

Finding 1 (SQLite ONLY): Four profile-delete methods previously called self-committing
_fts_delete_profile/_vec_delete BEFORE the row DELETE+emit+commit. The fix moves search
cleanup to AFTER the commit, exactly like delete_agent_playbooks_by_ids in _playbook.py.

Finding 2 (both backends): supersede_profiles_by_ids accepted empty request_id, which
corrupts get_lineage_events(request_id=R) grouping. Guard added: raise ValueError.

These tests verify:
- The four fixed methods still delete the row AND clean up FTS + vec after commit.
- A hard_delete lineage event was emitted for each deleted profile.
- supersede_profiles_by_ids raises (StorageError wrapping ValueError) on empty request_id.
"""

from __future__ import annotations

import pytest

from reflexio.models.api_schema.domain.enums import ProfileTimeToLive, Status
from reflexio.models.api_schema.service_schemas import (
    DeleteUserProfileRequest,
    UserProfile,
)
from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store(tmp_path) -> SQLiteStorage:
    s = SQLiteStorage(org_id="b3c-org", db_path=str(tmp_path / "b3c.db"))
    s.migrate()
    return s


def _make_profile(
    user_id: str = "u1",
    profile_id: str = "p1",
    content: str = "content",
    status: Status | None = None,
) -> UserProfile:
    from datetime import UTC, datetime

    return UserProfile(
        user_id=user_id,
        profile_id=profile_id,
        content=content,
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id=f"req_{profile_id}",
        profile_time_to_live=ProfileTimeToLive.INFINITY,
        status=status,
    )


def _fts_count(s: SQLiteStorage, profile_id: str) -> int:
    """Return the number of FTS rows for a profile_id."""
    row = s.conn.execute(
        "SELECT COUNT(*) AS cnt FROM profiles_fts WHERE profile_id = ?", (profile_id,)
    ).fetchone()
    return row["cnt"] if row else 0


# ---------------------------------------------------------------------------
# Finding 1: search cleanup happens after commit, not before DELETE
# ---------------------------------------------------------------------------


class TestDeleteUserProfileSearchCleanup:
    """delete_user_profile: profile row gone + FTS gone + hard_delete event emitted."""

    def test_profile_row_and_fts_gone_after_delete(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u1", [_make_profile("u1", "dp1")])
        assert _fts_count(s, "dp1") == 1

        s.delete_user_profile(DeleteUserProfileRequest(user_id="u1", profile_id="dp1"))

        assert s.get_profile_by_id("dp1") is None
        # FTS row must be cleaned up even though cleanup now happens after commit.
        assert _fts_count(s, "dp1") == 0

    def test_hard_delete_event_emitted(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u1", [_make_profile("u1", "dp2")])
        s.delete_user_profile(DeleteUserProfileRequest(user_id="u1", profile_id="dp2"))
        events = s.get_lineage_events(entity_id="dp2")
        assert any(e.op == "hard_delete" for e in events)

    def test_vec_row_gone_after_delete(self, tmp_path) -> None:
        s = _store(tmp_path)
        if not s._has_sqlite_vec:
            pytest.skip("sqlite-vec not loaded")
        s.add_user_profile("u1", [_make_profile("u1", "dp3")])
        rowid_row = s.conn.execute(
            "SELECT rowid FROM profiles WHERE profile_id = ?", ("dp3",)
        ).fetchone()
        assert rowid_row is not None
        rowid = rowid_row["rowid"]
        s._vec_upsert("profiles_vec", rowid, [0.1] * s.embedding_dimensions)
        assert rowid in {
            r["rowid"]
            for r in s.conn.execute("SELECT rowid FROM profiles_vec").fetchall()
        }
        s.delete_user_profile(DeleteUserProfileRequest(user_id="u1", profile_id="dp3"))
        assert rowid not in {
            r["rowid"]
            for r in s.conn.execute("SELECT rowid FROM profiles_vec").fetchall()
        }


class TestDeleteAllProfilesForUserSearchCleanup:
    """delete_all_profiles_for_user: all rows gone + FTS gone + events emitted."""

    def test_profile_rows_and_fts_gone(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile(
            "u2",
            [_make_profile("u2", "dfu1"), _make_profile("u2", "dfu2")],
        )
        assert _fts_count(s, "dfu1") == 1
        assert _fts_count(s, "dfu2") == 1

        s.delete_all_profiles_for_user("u2")

        assert s.get_user_profile("u2") == []
        assert _fts_count(s, "dfu1") == 0
        assert _fts_count(s, "dfu2") == 0

    def test_hard_delete_events_emitted_per_profile(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile(
            "u2", [_make_profile("u2", "dfu3"), _make_profile("u2", "dfu4")]
        )
        s.delete_all_profiles_for_user("u2")
        for pid in ["dfu3", "dfu4"]:
            events = s.get_lineage_events(entity_id=pid)
            assert any(e.op == "hard_delete" for e in events), (
                f"no hard_delete for {pid}"
            )

    def test_vec_rows_gone_after_delete(self, tmp_path) -> None:
        s = _store(tmp_path)
        if not s._has_sqlite_vec:
            pytest.skip("sqlite-vec not loaded")
        s.add_user_profile("u2", [_make_profile("u2", "dfu5")])
        rowid_row = s.conn.execute(
            "SELECT rowid FROM profiles WHERE profile_id = ?", ("dfu5",)
        ).fetchone()
        assert rowid_row is not None
        rowid = rowid_row["rowid"]
        s._vec_upsert("profiles_vec", rowid, [0.1] * s.embedding_dimensions)

        s.delete_all_profiles_for_user("u2")

        assert rowid not in {
            r["rowid"]
            for r in s.conn.execute("SELECT rowid FROM profiles_vec").fetchall()
        }


class TestDeleteAllProfilesByStatusSearchCleanup:
    """delete_all_profiles_by_status: rows gone + FTS gone + events emitted."""

    def test_profile_rows_and_fts_gone(self, tmp_path) -> None:
        s = _store(tmp_path)
        p1 = _make_profile("u1", "dbs1", status=Status.ARCHIVED)
        p2 = _make_profile("u1", "dbs2", status=Status.ARCHIVED)
        s.add_user_profile("u1", [p1, p2])
        assert _fts_count(s, "dbs1") == 1
        assert _fts_count(s, "dbs2") == 1

        deleted = s.delete_all_profiles_by_status(Status.ARCHIVED)

        assert deleted == 2
        assert _fts_count(s, "dbs1") == 0
        assert _fts_count(s, "dbs2") == 0

    def test_hard_delete_events_emitted(self, tmp_path) -> None:
        s = _store(tmp_path)
        p1 = _make_profile("u1", "dbs3", status=Status.ARCHIVED)
        s.add_user_profile("u1", [p1])
        s.delete_all_profiles_by_status(Status.ARCHIVED)
        events = s.get_lineage_events(entity_id="dbs3")
        assert any(e.op == "hard_delete" for e in events)

    def test_vec_rows_gone_after_delete(self, tmp_path) -> None:
        s = _store(tmp_path)
        if not s._has_sqlite_vec:
            pytest.skip("sqlite-vec not loaded")
        p1 = _make_profile("u1", "dbs4", status=Status.ARCHIVED)
        s.add_user_profile("u1", [p1])
        rowid_row = s.conn.execute(
            "SELECT rowid FROM profiles WHERE profile_id = ?", ("dbs4",)
        ).fetchone()
        assert rowid_row is not None
        rowid = rowid_row["rowid"]
        s._vec_upsert("profiles_vec", rowid, [0.1] * s.embedding_dimensions)

        s.delete_all_profiles_by_status(Status.ARCHIVED)

        assert rowid not in {
            r["rowid"]
            for r in s.conn.execute("SELECT rowid FROM profiles_vec").fetchall()
        }


class TestDeleteProfilesByIdsSearchCleanup:
    """delete_profiles_by_ids: rows gone + FTS gone + events emitted."""

    def test_profile_rows_and_fts_gone(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile(
            "u1",
            [_make_profile("u1", "dpi1"), _make_profile("u1", "dpi2")],
        )
        assert _fts_count(s, "dpi1") == 1
        assert _fts_count(s, "dpi2") == 1

        deleted = s.delete_profiles_by_ids(["dpi1", "dpi2"])

        assert deleted == 2
        assert _fts_count(s, "dpi1") == 0
        assert _fts_count(s, "dpi2") == 0

    def test_hard_delete_events_emitted_actor_system(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u1", [_make_profile("u1", "dpi3")])
        s.delete_profiles_by_ids(["dpi3"])
        events = [
            e for e in s.get_lineage_events(entity_id="dpi3") if e.op == "hard_delete"
        ]
        assert len(events) == 1
        assert events[0].actor == "system"

    def test_vec_rows_gone_after_delete(self, tmp_path) -> None:
        s = _store(tmp_path)
        if not s._has_sqlite_vec:
            pytest.skip("sqlite-vec not loaded")
        s.add_user_profile("u1", [_make_profile("u1", "dpi4")])
        rowid_row = s.conn.execute(
            "SELECT rowid FROM profiles WHERE profile_id = ?", ("dpi4",)
        ).fetchone()
        assert rowid_row is not None
        rowid = rowid_row["rowid"]
        s._vec_upsert("profiles_vec", rowid, [0.1] * s.embedding_dimensions)

        s.delete_profiles_by_ids(["dpi4"])

        assert rowid not in {
            r["rowid"]
            for r in s.conn.execute("SELECT rowid FROM profiles_vec").fetchall()
        }

    def test_emit_false_suppresses_event_but_still_cleans_fts(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u1", [_make_profile("u1", "dpi5")])
        assert _fts_count(s, "dpi5") == 1

        s.delete_profiles_by_ids(["dpi5"], emit_hard_delete=False)

        assert _fts_count(s, "dpi5") == 0
        events = s.get_lineage_events(entity_id="dpi5")
        assert not any(e.op == "hard_delete" for e in events)


# ---------------------------------------------------------------------------
# Finding 2: supersede_profiles_by_ids raises on empty request_id
# ---------------------------------------------------------------------------


class TestSupersedeProfilesEmptyRequestIdGuard:
    """supersede_profiles_by_ids must raise StorageError (wrapping ValueError) for empty request_id."""

    def test_raises_on_empty_string(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u1", [_make_profile("u1", "sup1")])
        # handle_exceptions wraps ValueError → StorageError
        with pytest.raises(StorageError):
            s.supersede_profiles_by_ids("u1", ["sup1"], request_id="")

    def test_raises_on_empty_string_no_ids(self, tmp_path) -> None:
        """Empty request_id check fires even for non-empty profile_ids list before any DB work."""
        s = _store(tmp_path)
        # Note: empty profile_ids returns 0 before the guard; this verifies the guard
        # fires on non-empty profile_ids with empty request_id.
        s.add_user_profile("u1", [_make_profile("u1", "sup2")])
        with pytest.raises(StorageError):
            s.supersede_profiles_by_ids("u1", ["sup2"], request_id="")

    def test_no_raise_on_nonempty_request_id(self, tmp_path) -> None:
        """Non-empty request_id must not raise."""
        s = _store(tmp_path)
        s.add_user_profile("u1", [_make_profile("u1", "sup3")])
        result = s.supersede_profiles_by_ids("u1", ["sup3"], request_id="req-abc")
        assert result == ["sup3"]

    def test_empty_ids_returns_empty_without_guard_check(self, tmp_path) -> None:
        """Empty profile_ids returns [] (short-circuit before the request_id check)."""
        s = _store(tmp_path)
        result = s.supersede_profiles_by_ids("u1", [], request_id="")
        assert result == []
