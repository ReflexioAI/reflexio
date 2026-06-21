"""Integration tests for B3f: atomic profile deletes (close rowid-reuse race, #196).

B3c moved search cleanup to AFTER commit (outside the lock). This left a rowid-reuse
race for profiles: profiles.profile_id is a TEXT PK backed by SQLite's IMPLICIT
(reusable) rowid. In the window between commit and cleanup re-acquiring the lock, a
concurrent INSERT can reuse the freed rowid. The cleanup then deletes the NEW profile's
profiles_vec row.

B3f fixes all 4 profile-delete methods and delete_all_profiles to perform
fts + vec + row + lineage in ONE `with self._lock:` / single `conn.commit()`.
No cleanup runs outside the lock. This mirrors the B3d interaction pattern.

Also verifies delete_all_profiles now cleans profiles_vec (it leaked before B3f).
"""

from __future__ import annotations

import pytest

from reflexio.models.api_schema.domain.enums import ProfileTimeToLive, Status
from reflexio.models.api_schema.service_schemas import (
    DeleteUserProfileRequest,
    UserProfile,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store(tmp_path) -> SQLiteStorage:
    s = SQLiteStorage(org_id="b3f-org", db_path=str(tmp_path / "b3f.db"))
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
    row = s.conn.execute(
        "SELECT COUNT(*) AS cnt FROM profiles_fts WHERE profile_id = ?", (profile_id,)
    ).fetchone()
    return row["cnt"] if row else 0


def _vec_rowids(s: SQLiteStorage) -> set[int]:
    return {
        r["rowid"] for r in s.conn.execute("SELECT rowid FROM profiles_vec").fetchall()
    }


# ---------------------------------------------------------------------------
# delete_user_profile: row + fts + vec removed atomically
# ---------------------------------------------------------------------------


class TestDeleteUserProfileAtomic:
    """Row, FTS, and vec removed in one atomic block (no post-commit cleanup)."""

    def test_row_and_fts_removed(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u1", [_make_profile("u1", "a1")])
        assert _fts_count(s, "a1") == 1

        s.delete_user_profile(DeleteUserProfileRequest(user_id="u1", profile_id="a1"))

        assert s.get_profile_by_id("a1") is None
        assert _fts_count(s, "a1") == 0

    def test_vec_removed_atomically(self, tmp_path) -> None:
        s = _store(tmp_path)
        if not s._has_sqlite_vec:
            pytest.skip("sqlite-vec not loaded")
        s.add_user_profile("u1", [_make_profile("u1", "a2")])
        rowid_row = s.conn.execute(
            "SELECT rowid FROM profiles WHERE profile_id = ?", ("a2",)
        ).fetchone()
        assert rowid_row is not None
        rowid = rowid_row["rowid"]
        s._vec_upsert("profiles_vec", rowid, [0.1] * s.embedding_dimensions)
        assert rowid in _vec_rowids(s)

        s.delete_user_profile(DeleteUserProfileRequest(user_id="u1", profile_id="a2"))

        # vec removed inside the lock before commit — no post-commit cleanup race
        assert rowid not in _vec_rowids(s)

    def test_hard_delete_event_emitted(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u1", [_make_profile("u1", "a3")])
        s.delete_user_profile(DeleteUserProfileRequest(user_id="u1", profile_id="a3"))
        events = s.get_lineage_events(entity_id="a3")
        assert any(e.op == "hard_delete" for e in events)


# ---------------------------------------------------------------------------
# delete_all_profiles_for_user: all rows + fts + vec removed atomically
# ---------------------------------------------------------------------------


class TestDeleteAllProfilesForUserAtomic:
    """All profile rows, FTS, and vec removed atomically for a user."""

    def test_rows_and_fts_removed(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u2", [_make_profile("u2", "b1"), _make_profile("u2", "b2")])
        assert _fts_count(s, "b1") == 1
        assert _fts_count(s, "b2") == 1

        s.delete_all_profiles_for_user("u2")

        assert s.get_user_profile("u2") == []
        assert _fts_count(s, "b1") == 0
        assert _fts_count(s, "b2") == 0

    def test_vec_removed_atomically(self, tmp_path) -> None:
        s = _store(tmp_path)
        if not s._has_sqlite_vec:
            pytest.skip("sqlite-vec not loaded")
        s.add_user_profile("u2", [_make_profile("u2", "b3")])
        rowid_row = s.conn.execute(
            "SELECT rowid FROM profiles WHERE profile_id = ?", ("b3",)
        ).fetchone()
        assert rowid_row is not None
        rowid = rowid_row["rowid"]
        s._vec_upsert("profiles_vec", rowid, [0.1] * s.embedding_dimensions)
        assert rowid in _vec_rowids(s)

        s.delete_all_profiles_for_user("u2")

        assert rowid not in _vec_rowids(s)

    def test_hard_delete_events_emitted(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u2", [_make_profile("u2", "b4"), _make_profile("u2", "b5")])
        s.delete_all_profiles_for_user("u2")
        for pid in ["b4", "b5"]:
            events = s.get_lineage_events(entity_id=pid)
            assert any(e.op == "hard_delete" for e in events), (
                f"no hard_delete for {pid}"
            )


# ---------------------------------------------------------------------------
# delete_all_profiles_by_status: rows + fts + vec removed atomically
# ---------------------------------------------------------------------------


class TestDeleteAllProfilesByStatusAtomic:
    """Profiles with the given status are removed along with fts and vec atomically."""

    def test_rows_and_fts_removed(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile(
            "u1",
            [
                _make_profile("u1", "c1", status=Status.ARCHIVED),
                _make_profile("u1", "c2", status=Status.ARCHIVED),
            ],
        )
        assert _fts_count(s, "c1") == 1
        assert _fts_count(s, "c2") == 1

        deleted = s.delete_all_profiles_by_status(Status.ARCHIVED)

        assert deleted == 2
        assert _fts_count(s, "c1") == 0
        assert _fts_count(s, "c2") == 0

    def test_vec_removed_atomically(self, tmp_path) -> None:
        s = _store(tmp_path)
        if not s._has_sqlite_vec:
            pytest.skip("sqlite-vec not loaded")
        s.add_user_profile("u1", [_make_profile("u1", "c3", status=Status.ARCHIVED)])
        rowid_row = s.conn.execute(
            "SELECT rowid FROM profiles WHERE profile_id = ?", ("c3",)
        ).fetchone()
        assert rowid_row is not None
        rowid = rowid_row["rowid"]
        s._vec_upsert("profiles_vec", rowid, [0.1] * s.embedding_dimensions)
        assert rowid in _vec_rowids(s)

        s.delete_all_profiles_by_status(Status.ARCHIVED)

        assert rowid not in _vec_rowids(s)

    def test_hard_delete_events_emitted(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u1", [_make_profile("u1", "c4", status=Status.ARCHIVED)])
        s.delete_all_profiles_by_status(Status.ARCHIVED)
        events = s.get_lineage_events(entity_id="c4")
        assert any(e.op == "hard_delete" for e in events)


# ---------------------------------------------------------------------------
# delete_profiles_by_ids: rows + fts + vec removed atomically
# ---------------------------------------------------------------------------


class TestDeleteProfilesByIdsAtomic:
    """Profiles by ID are removed along with fts and vec atomically."""

    def test_rows_and_fts_removed(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u1", [_make_profile("u1", "d1"), _make_profile("u1", "d2")])
        assert _fts_count(s, "d1") == 1
        assert _fts_count(s, "d2") == 1

        deleted = s.delete_profiles_by_ids(["d1", "d2"])

        assert deleted == 2
        assert _fts_count(s, "d1") == 0
        assert _fts_count(s, "d2") == 0

    def test_vec_removed_atomically(self, tmp_path) -> None:
        s = _store(tmp_path)
        if not s._has_sqlite_vec:
            pytest.skip("sqlite-vec not loaded")
        s.add_user_profile("u1", [_make_profile("u1", "d3")])
        rowid_row = s.conn.execute(
            "SELECT rowid FROM profiles WHERE profile_id = ?", ("d3",)
        ).fetchone()
        assert rowid_row is not None
        rowid = rowid_row["rowid"]
        s._vec_upsert("profiles_vec", rowid, [0.1] * s.embedding_dimensions)
        assert rowid in _vec_rowids(s)

        s.delete_profiles_by_ids(["d3"])

        assert rowid not in _vec_rowids(s)

    def test_hard_delete_event_emitted_actor_system(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u1", [_make_profile("u1", "d4")])
        s.delete_profiles_by_ids(["d4"])
        events = [
            e for e in s.get_lineage_events(entity_id="d4") if e.op == "hard_delete"
        ]
        assert len(events) == 1
        assert events[0].actor == "system"

    def test_emit_false_suppresses_event_and_cleans_fts(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u1", [_make_profile("u1", "d5")])
        assert _fts_count(s, "d5") == 1

        s.delete_profiles_by_ids(["d5"], emit_hard_delete=False)

        assert _fts_count(s, "d5") == 0
        events = s.get_lineage_events(entity_id="d5")
        assert not any(e.op == "hard_delete" for e in events)


# ---------------------------------------------------------------------------
# delete_all_profiles: vec is now also wiped (previously leaked, #196)
# ---------------------------------------------------------------------------


class TestDeleteAllProfilesVecWipe:
    """delete_all_profiles wipes profiles_vec (was a leak before B3f)."""

    def test_vec_wiped_on_delete_all(self, tmp_path) -> None:
        s = _store(tmp_path)
        if not s._has_sqlite_vec:
            pytest.skip("sqlite-vec not loaded")
        s.add_user_profile("u1", [_make_profile("u1", "e1"), _make_profile("u1", "e2")])
        # Manually plant vec rows to ensure they are present before wipe.
        for pid in ["e1", "e2"]:
            rowid_row = s.conn.execute(
                "SELECT rowid FROM profiles WHERE profile_id = ?", (pid,)
            ).fetchone()
            assert rowid_row is not None
            s._vec_upsert(
                "profiles_vec", rowid_row["rowid"], [0.1] * s.embedding_dimensions
            )
        assert len(_vec_rowids(s)) >= 2

        s.delete_all_profiles()

        assert len(_vec_rowids(s)) == 0

    def test_fts_and_rows_still_wiped(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u1", [_make_profile("u1", "e3")])
        s.delete_all_profiles()
        assert s.count_all_profiles() == 0
        assert _fts_count(s, "e3") == 0


# ---------------------------------------------------------------------------
# delete_user_profile: cross-user sidecar protection (Fix A)
# ---------------------------------------------------------------------------


class TestDeleteUserProfileCrossUser:
    """Fix A: u1 deleting u2's profile_id leaves u2's row+fts+vec intact."""

    def test_cross_user_row_untouched(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u2", [_make_profile("u2", "x1")])
        s.delete_user_profile(DeleteUserProfileRequest(user_id="u1", profile_id="x1"))
        assert s.get_profile_by_id("x1") is not None

    def test_cross_user_fts_untouched(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u2", [_make_profile("u2", "x2")])
        assert _fts_count(s, "x2") == 1
        s.delete_user_profile(DeleteUserProfileRequest(user_id="u1", profile_id="x2"))
        assert _fts_count(s, "x2") == 1

    def test_cross_user_vec_untouched(self, tmp_path) -> None:
        s = _store(tmp_path)
        if not s._has_sqlite_vec:
            pytest.skip("sqlite-vec not loaded")
        s.add_user_profile("u2", [_make_profile("u2", "x3")])
        rowid_row = s.conn.execute(
            "SELECT rowid FROM profiles WHERE profile_id = ?", ("x3",)
        ).fetchone()
        assert rowid_row is not None
        rowid = rowid_row["rowid"]
        # Verify vec row exists
        assert rowid in _vec_rowids(s)
        s.delete_user_profile(DeleteUserProfileRequest(user_id="u1", profile_id="x3"))
        # u2's vec row must still be there
        assert rowid in _vec_rowids(s)
