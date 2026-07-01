"""Characterization tests: SQLite profile/interaction atomicity (Tier-1 Task 1).

Phase-A invariant guards established BEFORE the ``ProfileMixin`` decomposition
(``_profiles.py`` -> ProfileStore / InteractionStore / Search). They pin the CURRENT
commit / lineage-event / no-op behavior of the top-risk, atomicity-sensitive profile
MUTATION + STATUS methods and the interaction INSERT path so a "tidying" reorder during
the mixin split is caught by a failing test. Modeled on the gold standards
``test_lineage_b3f_profile_atomic_integration.py`` (profile deletes) and
``test_playbook_atomicity_characterization_integration.py`` (crash-window pattern).

The invariant each test pins (see ``.claude/rules/reflexio-patterns.md`` — "SQLite
storage: lineage events & atomicity"): a lineage event is recorded ONLY for a mutation
that actually committed. The mutation and its guarded lineage event share one
``conn.commit()``; the self-committing ``_fts_*`` / ``_vec_*`` index maintenance runs
AFTER that commit. A refactor that reorders ``mutation -> rowcount-guarded emit ->
commit`` (emitting before/independently of the commit, or dropping the ``rowcount``
guard) must fail one of these tests.

Methods characterized here (SQLite side), each with happy-path effect + lineage,
a no-op / phantom guard, and a crash-window rollback:
  - ``update_user_profile_by_id`` — in-place edit, one ``revise`` event (actor=api).
  - ``supersede_profiles_by_ids`` — soft-delete to SUPERSEDED, per-id rowcount guard,
    one ``status_change`` event each under a SHARED request_id (actor=dedup).
  - ``archive_profile_by_id`` — NULL -> ARCHIVED, one ``status_change`` event.
  - ``update_all_profiles_status`` — batch status flip, one ``status_change`` event per
    updated row under a shared batch request_id; returns the updated rowcount.
  - ``_insert_interaction`` (via ``add_user_interaction``) — INSERT wrapped in
    BEGIN IMMEDIATE / commit-or-rollback; FTS + vec written only AFTER commit.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

import reflexio.server.services.storage.sqlite_storage._profiles as profiles_mod
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    Status,
    UserProfile,
)
from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _store(tmp_path, org_id: str = "prof-atomic-org") -> SQLiteStorage:
    s = SQLiteStorage(org_id=org_id, db_path=str(tmp_path / f"{org_id}.db"))
    s.migrate()
    return s


def _make_profile(
    user_id: str,
    profile_id: str,
    content: str = "content",
    status: Status | None = None,
) -> UserProfile:
    return UserProfile(
        user_id=user_id,
        profile_id=profile_id,
        content=content,
        last_modified_timestamp=int(time.time()),
        generated_from_request_id=f"req_{profile_id}",
        status=status,
    )


def _events(s: SQLiteStorage, profile_id: str, op: str) -> list:
    return [e for e in s.get_lineage_events(entity_id=str(profile_id)) if e.op == op]


def _interaction_row_count(s: SQLiteStorage, interaction_id: int) -> int:
    row = s.conn.execute(
        "SELECT COUNT(*) AS cnt FROM interactions WHERE interaction_id = ?",
        (interaction_id,),
    ).fetchone()
    return row["cnt"] if row else 0


def _fts_count(s: SQLiteStorage, interaction_id: int) -> int:
    row = s.conn.execute(
        "SELECT COUNT(*) AS cnt FROM interactions_fts WHERE rowid = ?",
        (interaction_id,),
    ).fetchone()
    return row["cnt"] if row else 0


def _vec_count(s: SQLiteStorage, interaction_id: int) -> int:
    if not s._has_sqlite_vec:
        return 0
    row = s.conn.execute(
        "SELECT COUNT(*) AS cnt FROM interactions_vec WHERE rowid = ?",
        (interaction_id,),
    ).fetchone()
    return row["cnt"] if row else 0


# ---------------------------------------------------------------------------
# update_user_profile_by_id — revise event + no-op + crash-window
# ---------------------------------------------------------------------------


class TestUpdateUserProfileById:
    def test_happy_path_updates_content_and_emits_one_revise(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u1", [_make_profile("u1", "p1", content="original")])

        s.update_user_profile_by_id(
            "u1", "p1", _make_profile("u1", "p1", content="edited")
        )

        prof = s.get_profile_by_id("p1")
        assert prof is not None
        assert prof.content == "edited"
        evs = _events(s, "p1", "revise")
        assert len(evs) == 1
        assert evs[0].actor == "api"

    def test_missing_profile_is_noop_no_event(self, tmp_path) -> None:
        s = _store(tmp_path)
        # No profile "ghost" exists — the method returns early (no row) and emits nothing.
        s.update_user_profile_by_id(
            "u1", "ghost", _make_profile("u1", "ghost", content="x")
        )
        assert s.get_profile_by_id("ghost") is None
        assert _events(s, "ghost", "revise") == []

    def test_crash_window_emit_failure_leaves_no_partial_state(self, tmp_path) -> None:
        """A failing revise emit rolls back the UPDATE and writes no phantom event."""
        s = _store(tmp_path)
        s.add_user_profile("u1", [_make_profile("u1", "p1", content="original")])

        with (
            patch.object(
                profiles_mod,
                "_append_event_stmt",
                side_effect=RuntimeError("simulated emit failure"),
            ),
            pytest.raises(StorageError, match="simulated emit failure"),
        ):
            s.update_user_profile_by_id(
                "u1", "p1", _make_profile("u1", "p1", content="edited")
            )

        # The method never committed; discard the would-be UPDATE and assert durability.
        s.conn.rollback()
        prof = s.get_profile_by_id("p1")
        assert prof is not None
        assert prof.content == "original"
        assert _events(s, "p1", "revise") == []


# ---------------------------------------------------------------------------
# supersede_profiles_by_ids — status_change event + no-op + crash-window
# ---------------------------------------------------------------------------


def _superseded_events(s: SQLiteStorage, profile_id: str) -> list:
    return [
        e
        for e in s.get_lineage_events(entity_id=str(profile_id))
        if e.op == "status_change" and e.to_status == "superseded"
    ]


class TestSupersedeProfilesByIds:
    def test_happy_path_supersedes_and_emits_one_event_each(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u1", [_make_profile("u1", "p1"), _make_profile("u1", "p2")])

        committed = s.supersede_profiles_by_ids("u1", ["p1", "p2"], "shared-run")

        assert committed == ["p1", "p2"]
        for pid in ("p1", "p2"):
            # Excluded from default reads; readable as tombstone with SUPERSEDED status.
            assert s.get_profile_by_id(pid) is None
            tomb = s.get_profile_by_id(pid, include_tombstones=True)
            assert tomb is not None
            assert tomb.status == Status.SUPERSEDED
            evs = _superseded_events(s, pid)
            assert len(evs) == 1
            assert evs[0].request_id == "shared-run"
            assert evs[0].actor == "dedup"

    def test_already_superseded_is_noop_no_new_event(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u1", [_make_profile("u1", "p1")])

        first = s.supersede_profiles_by_ids("u1", ["p1"], "run-1")
        assert first == ["p1"]
        second = s.supersede_profiles_by_ids("u1", ["p1"], "run-2")
        assert second == []

        # Only the first call's event exists — the no-op must not emit a phantom.
        evs = _superseded_events(s, "p1")
        assert len(evs) == 1
        assert evs[0].request_id == "run-1"

    def test_nonexistent_id_returns_empty_no_event(self, tmp_path) -> None:
        s = _store(tmp_path)
        assert s.supersede_profiles_by_ids("u1", ["ghost"], "run-miss") == []
        assert _superseded_events(s, "ghost") == []

    def test_empty_request_id_raises_before_write(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u1", [_make_profile("u1", "p1")])

        with pytest.raises((ValueError, StorageError), match="non-empty"):
            s.supersede_profiles_by_ids("u1", ["p1"], "")

        # Row remains CURRENT, no event.
        assert s.get_profile_by_id("p1") is not None
        assert _superseded_events(s, "p1") == []

    def test_crash_window_emit_failure_leaves_no_partial_state(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u1", [_make_profile("u1", "p1")])

        with (
            patch.object(
                profiles_mod,
                "_append_event_stmt",
                side_effect=RuntimeError("simulated emit failure"),
            ),
            pytest.raises(StorageError, match="simulated emit failure"),
        ):
            s.supersede_profiles_by_ids("u1", ["p1"], "run-crash")

        s.conn.rollback()
        prof = s.get_profile_by_id("p1")
        assert prof is not None
        assert prof.status is None or prof.status != Status.SUPERSEDED
        assert _superseded_events(s, "p1") == []


# ---------------------------------------------------------------------------
# archive_profile_by_id — status_change event + no-op + crash-window
# ---------------------------------------------------------------------------


def _archive_events(s: SQLiteStorage, profile_id: str) -> list:
    return [
        e
        for e in s.get_lineage_events(entity_id=str(profile_id))
        if e.op == "status_change" and e.to_status == "archived"
    ]


class TestArchiveProfileById:
    def test_happy_path_archives_and_emits_one_event(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u1", [_make_profile("u1", "p1")])

        assert s.archive_profile_by_id("u1", "p1") is True

        tomb = s.get_profile_by_id("p1", include_tombstones=True)
        assert tomb is not None
        assert tomb.status == Status.ARCHIVED
        assert len(_archive_events(s, "p1")) == 1

    def test_missing_profile_returns_false_no_event(self, tmp_path) -> None:
        s = _store(tmp_path)
        assert s.archive_profile_by_id("u1", "ghost") is False
        assert _archive_events(s, "ghost") == []

    def test_crash_window_emit_failure_leaves_no_partial_state(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u1", [_make_profile("u1", "p1")])

        with (
            patch.object(
                profiles_mod,
                "_append_event_stmt",
                side_effect=RuntimeError("simulated emit failure"),
            ),
            pytest.raises(StorageError, match="simulated emit failure"),
        ):
            s.archive_profile_by_id("u1", "p1")

        s.conn.rollback()
        prof = s.get_profile_by_id("p1")
        assert prof is not None
        assert prof.status is None
        assert _archive_events(s, "p1") == []


# ---------------------------------------------------------------------------
# update_all_profiles_status — batch status_change + no-op + crash-window
# ---------------------------------------------------------------------------


def _status_change_events(s: SQLiteStorage, profile_id: str) -> list:
    return [
        e
        for e in s.get_lineage_events(entity_id=str(profile_id))
        if e.op == "status_change"
    ]


class TestUpdateAllProfilesStatus:
    def test_happy_path_flips_status_and_emits_event_each(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u1", [_make_profile("u1", "p1")])
        s.add_user_profile("u2", [_make_profile("u2", "p2")])

        updated = s.update_all_profiles_status(None, Status.ARCHIVED)

        assert updated == 2
        for pid in ("p1", "p2"):
            tomb = s.get_profile_by_id(pid, include_tombstones=True)
            assert tomb is not None
            assert tomb.status == Status.ARCHIVED
            assert len(_status_change_events(s, pid)) == 1

    def test_no_matching_rows_is_noop_returns_zero(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u1", [_make_profile("u1", "p1")])  # status NULL

        # Nothing is ARCHIVED, so flipping ARCHIVED->PENDING matches no row.
        updated = s.update_all_profiles_status(Status.ARCHIVED, Status.PENDING)

        assert updated == 0
        assert _status_change_events(s, "p1") == []

    def test_crash_window_emit_failure_leaves_no_partial_state(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_profile("u1", [_make_profile("u1", "p1")])

        with (
            patch.object(
                profiles_mod,
                "_append_event_stmt",
                side_effect=RuntimeError("simulated emit failure"),
            ),
            pytest.raises(StorageError, match="simulated emit failure"),
        ):
            s.update_all_profiles_status(None, Status.ARCHIVED)

        s.conn.rollback()
        prof = s.get_profile_by_id("p1")
        assert prof is not None
        assert prof.status is None
        assert _status_change_events(s, "p1") == []


# ---------------------------------------------------------------------------
# _insert_interaction (via add_user_interaction) — INSERT + FTS/vec after commit
# ---------------------------------------------------------------------------


class TestInsertInteractionAtomicity:
    def test_happy_path_writes_row_fts_and_vec(self, tmp_path) -> None:
        s = _store(tmp_path)
        s.add_user_interaction(
            "u1",
            Interaction(
                interaction_id=1,
                user_id="u1",
                request_id="r1",
                content="hello world",
                created_at=int(time.time()),
            ),
        )
        assert _interaction_row_count(s, 1) == 1
        # FTS + vec are written AFTER the row commit (index maintenance).
        assert _fts_count(s, 1) == 1
        if s._has_sqlite_vec:
            assert _vec_count(s, 1) == 1

    def test_auto_assigns_interaction_id_when_omitted(self, tmp_path) -> None:
        s = _store(tmp_path)
        interaction = Interaction(
            user_id="u1",
            request_id="r1",
            content="auto id",
            created_at=int(time.time()),
        )
        s.add_user_interaction("u1", interaction)
        # The INSERT (no explicit id) assigns lastrowid back onto the model.
        assert interaction.interaction_id
        assert _interaction_row_count(s, interaction.interaction_id) == 1

    def test_transaction_abort_leaves_no_row_fts_or_vec(self, tmp_path) -> None:
        """A failure inside the BEGIN IMMEDIATE block rolls back — nothing persists.

        ``_insert_interaction`` wraps the INSERT in BEGIN IMMEDIATE / commit and, on any
        exception, calls ``conn.rollback()`` and re-raises. We force the failure via the
        subject-writability guard (raised after BEGIN IMMEDIATE) and assert the aborted
        transaction leaves no interaction row; the FTS/vec upserts run only after a
        successful commit and so are never reached on the failure path.
        """
        s = _store(tmp_path)

        with (
            patch.object(
                SQLiteStorage,
                "_assert_subject_writable_locked",
                side_effect=RuntimeError("simulated abort"),
            ),
            pytest.raises(StorageError, match="simulated abort"),
        ):
            s.add_user_interaction(
                "u1",
                Interaction(
                    interaction_id=999,
                    user_id="u1",
                    request_id="r9",
                    content="doomed",
                    created_at=int(time.time()),
                ),
            )

        assert _interaction_row_count(s, 999) == 0
        assert _fts_count(s, 999) == 0
        if s._has_sqlite_vec:
            assert _vec_count(s, 999) == 0
