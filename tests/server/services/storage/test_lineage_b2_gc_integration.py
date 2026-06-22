"""Integration tests: gc_expired_tombstones + legal-hold stub (Lineage Phase B2).

Tests cover:
  (a) Aged tombstone (MERGED) with retired_at past cutoff is GC'd: row deleted + hard_delete event.
  (b) Fresh tombstone (recent retired_at) and CURRENT row are NOT deleted.
  (c) ARCHIVED aged tombstone IS GC'd (broader eligible set than _TOMBSTONE).
  (d) PB-8b regression: tombstone with OLD created_at but RECENT retired_at is NOT GC'd.
      Proves the GC ages on retired_at, not created_at.
  (e) retired_at = NULL tombstone (pre-T1 row) is NOT GC'd.
  (f) Legal-hold: monkeypatched hold skips one row — no delete, no hard_delete event.
  (g) Idempotent: second GC call on an already-empty set returns 0, adds no events.
  (h) Atomic rollback: mid-operation failure leaves no partial state.
"""

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

import reflexio.server.services.storage.sqlite_storage._lineage as _lineage_mod
from reflexio.models.api_schema.domain.entities import UserPlaybook
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive, Status
from reflexio.models.api_schema.service_schemas import AgentPlaybook, UserProfile
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _store(tmp_path, org_id: str = "org-gc") -> SQLiteStorage:
    s = SQLiteStorage(org_id=org_id, db_path=str(tmp_path / "t.db"))
    s.migrate()
    return s


def _now_epoch() -> int:
    return int(datetime.now(UTC).timestamp())


def _make_profile(
    user_id: str = "u1",
    profile_id: str = "p1",
    content: str = "c",
    ts: int | None = None,
) -> UserProfile:
    return UserProfile(
        user_id=user_id,
        profile_id=profile_id,
        content=content,
        last_modified_timestamp=ts if ts is not None else _now_epoch(),
        generated_from_request_id=f"req_{profile_id}",
        profile_time_to_live=ProfileTimeToLive.INFINITY,
    )


def _make_user_playbook(user_id: str = "u1", content: str = "c") -> UserPlaybook:
    return UserPlaybook(
        user_id=user_id, agent_version="v1", request_id="r1", content=content
    )


def _make_agent_playbook(
    playbook_name: str = "pb", content: str = "c"
) -> AgentPlaybook:
    return AgentPlaybook(
        playbook_name=playbook_name, agent_version="v1", content=content
    )


def _set_playbook_status(
    s: SQLiteStorage, table: str, pk: str, pk_val: int, status: str
) -> None:
    """Directly set the status of a playbook row for test seeding."""
    s.conn.execute(
        f"UPDATE {table} SET status = ? WHERE {pk} = ?",  # noqa: S608
        (status, pk_val),
    )
    s.conn.commit()


def _set_profile_status(s: SQLiteStorage, profile_id: str, status: str) -> None:
    s.conn.execute(
        "UPDATE profiles SET status = ? WHERE profile_id = ?",
        (status, profile_id),
    )
    s.conn.commit()


def _set_retired_at(
    s: SQLiteStorage, table: str, pk: str, pk_val: int | str, retired_at: int | None
) -> None:
    """Directly set retired_at on a row for test seeding."""
    s.conn.execute(
        f"UPDATE {table} SET retired_at = ? WHERE {pk} = ?",  # noqa: S608
        (retired_at, pk_val),
    )
    s.conn.commit()


def _hard_delete_events(s: SQLiteStorage, entity_id: str) -> list:
    return [
        e for e in s.get_lineage_events(entity_id=entity_id) if e.op == "hard_delete"
    ]


# ---------------------------------------------------------------------------
# (a) Aged tombstone (MERGED) with retired_at past cutoff is GC'd
# ---------------------------------------------------------------------------


def test_gc_deletes_aged_merged_tombstone(tmp_path):
    s = _store(tmp_path)
    pb = _make_user_playbook()
    s.save_user_playbooks([pb])
    pid = pb.user_playbook_id
    _set_playbook_status(
        s, "user_playbooks", "user_playbook_id", pid, Status.MERGED.value
    )

    # Set retired_at to well before the cutoff
    old_retired_at = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp())
    _set_retired_at(s, "user_playbooks", "user_playbook_id", pid, old_retired_at)

    cutoff = int(datetime(2021, 1, 1, tzinfo=UTC).timestamp())
    deleted = s.gc_expired_tombstones(
        entity_type="user_playbook", older_than_epoch=cutoff
    )

    assert deleted == 1
    # Row is gone
    row = s.conn.execute(
        "SELECT * FROM user_playbooks WHERE user_playbook_id = ?", (pid,)
    ).fetchone()
    assert row is None
    # hard_delete event emitted
    events = _hard_delete_events(s, str(pid))
    assert len(events) == 1
    assert events[0].actor == "system"
    assert events[0].reason == "ttl-gc"


# ---------------------------------------------------------------------------
# (b) Fresh tombstone (recent retired_at) and CURRENT row are NOT deleted
# ---------------------------------------------------------------------------


def test_gc_skips_fresh_tombstone_and_current(tmp_path):
    s = _store(tmp_path)
    # Fresh merged tombstone — retired_at is right now (after any historical cutoff)
    pb_fresh = _make_user_playbook(content="fresh")
    s.save_user_playbooks([pb_fresh])
    pid_fresh = pb_fresh.user_playbook_id
    _set_playbook_status(
        s, "user_playbooks", "user_playbook_id", pid_fresh, Status.MERGED.value
    )
    # retired_at is current — after the historical cutoff
    _set_retired_at(
        s,
        "user_playbooks",
        "user_playbook_id",
        pid_fresh,
        int(datetime(2025, 1, 1, tzinfo=UTC).timestamp()),
    )

    # CURRENT row (no status, retired_at NULL)
    pb_current = _make_user_playbook(content="current")
    s.save_user_playbooks([pb_current])
    pid_current = pb_current.user_playbook_id
    # status is NULL (CURRENT) — must NOT be deleted even if old

    cutoff = int(datetime(2021, 1, 1, tzinfo=UTC).timestamp())
    deleted = s.gc_expired_tombstones(
        entity_type="user_playbook", older_than_epoch=cutoff
    )

    assert deleted == 0
    # Both rows still exist
    assert (
        s.conn.execute(
            "SELECT 1 FROM user_playbooks WHERE user_playbook_id = ?", (pid_fresh,)
        ).fetchone()
        is not None
    )
    assert (
        s.conn.execute(
            "SELECT 1 FROM user_playbooks WHERE user_playbook_id = ?", (pid_current,)
        ).fetchone()
        is not None
    )


# ---------------------------------------------------------------------------
# (c) ARCHIVED aged tombstone IS GC'd
# ---------------------------------------------------------------------------


def test_gc_deletes_aged_archived_tombstone(tmp_path):
    s = _store(tmp_path)
    pb = _make_agent_playbook(playbook_name="archbook")
    saved = s.save_agent_playbooks([pb])
    apid = saved[0].agent_playbook_id

    _set_playbook_status(
        s, "agent_playbooks", "agent_playbook_id", apid, Status.ARCHIVED.value
    )
    _set_retired_at(
        s,
        "agent_playbooks",
        "agent_playbook_id",
        apid,
        int(datetime(2019, 6, 1, tzinfo=UTC).timestamp()),
    )

    cutoff = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp())
    deleted = s.gc_expired_tombstones(
        entity_type="agent_playbook", older_than_epoch=cutoff
    )

    assert deleted == 1
    row = s.conn.execute(
        "SELECT * FROM agent_playbooks WHERE agent_playbook_id = ?", (apid,)
    ).fetchone()
    assert row is None
    events = _hard_delete_events(s, str(apid))
    assert len(events) == 1


# ---------------------------------------------------------------------------
# (d) PB-8b regression: OLD created_at but RECENT retired_at → NOT GC'd
# ---------------------------------------------------------------------------


def test_gc_pb8b_old_created_at_recent_retired_at_not_gc_d(tmp_path):
    """Tombstone with OLD created_at but RECENT retired_at must NOT be GC'd.

    This is the PB-8b regression: proves the GC ages on retired_at (retirement
    instant), not created_at or last_modified_timestamp.  Any row that was retired
    recently must be preserved regardless of how old the underlying content is.
    """
    s = _store(tmp_path)

    # --- User playbook: OLD created_at, RECENT retired_at ---
    pb = _make_user_playbook(content="old-content")
    s.save_user_playbooks([pb])
    pid = pb.user_playbook_id
    _set_playbook_status(
        s, "user_playbooks", "user_playbook_id", pid, Status.MERGED.value
    )

    # Force created_at to 2018 (well before cutoff)
    old_iso = "2018-01-01T00:00:00+00:00"
    s.conn.execute(
        "UPDATE user_playbooks SET created_at = ? WHERE user_playbook_id = ?",
        (old_iso, pid),
    )
    # But retired_at is recent (2025 — well after cutoff)
    recent_retired_at = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp())
    _set_retired_at(s, "user_playbooks", "user_playbook_id", pid, recent_retired_at)
    s.conn.commit()

    cutoff = int(datetime(2021, 1, 1, tzinfo=UTC).timestamp())
    deleted = s.gc_expired_tombstones(
        entity_type="user_playbook", older_than_epoch=cutoff
    )

    assert deleted == 0, (
        "Tombstone with recent retired_at must NOT be GC'd even when created_at is ancient"
    )
    assert (
        s.conn.execute(
            "SELECT 1 FROM user_playbooks WHERE user_playbook_id = ?", (pid,)
        ).fetchone()
        is not None
    )

    # --- Profile: OLD last_modified_timestamp, RECENT retired_at ---
    old_ts = int(datetime(2018, 6, 1, tzinfo=UTC).timestamp())
    p = _make_profile(profile_id="pb8b-profile", ts=old_ts)
    s.add_user_profile("u1", [p])
    _set_profile_status(s, "pb8b-profile", Status.SUPERSEDED.value)
    # Ensure last_modified_timestamp is old
    s.conn.execute(
        "UPDATE profiles SET last_modified_timestamp = ? WHERE profile_id = ?",
        (old_ts, "pb8b-profile"),
    )
    # But retired_at is recent
    _set_retired_at(s, "profiles", "profile_id", "pb8b-profile", recent_retired_at)
    s.conn.commit()

    deleted_p = s.gc_expired_tombstones(entity_type="profile", older_than_epoch=cutoff)

    assert deleted_p == 0, (
        "Profile with recent retired_at must NOT be GC'd even when last_modified_timestamp is ancient"
    )
    assert (
        s.conn.execute(
            "SELECT 1 FROM profiles WHERE profile_id = ?",
            ("pb8b-profile",),
        ).fetchone()
        is not None
    ), "PB-8b profile must still exist when retired_at is recent"


# ---------------------------------------------------------------------------
# (e) retired_at = NULL tombstone (pre-T1 row) is NOT GC'd
# ---------------------------------------------------------------------------


def test_gc_null_retired_at_not_gc_d(tmp_path):
    """A tombstone with retired_at = NULL (pre-T1 row) must never be GC'd.

    NULL means no retirement clock; the GC must treat it as 'not yet eligible'.
    """
    s = _store(tmp_path)

    pb = _make_user_playbook(content="pre-t1")
    s.save_user_playbooks([pb])
    pid = pb.user_playbook_id
    _set_playbook_status(
        s, "user_playbooks", "user_playbook_id", pid, Status.MERGED.value
    )

    # Explicitly set retired_at = NULL (simulating a pre-T1 tombstone)
    _set_retired_at(s, "user_playbooks", "user_playbook_id", pid, None)

    # Use a far-future cutoff — would GC anything eligible
    cutoff = int(datetime(2030, 1, 1, tzinfo=UTC).timestamp())
    deleted = s.gc_expired_tombstones(
        entity_type="user_playbook", older_than_epoch=cutoff
    )

    assert deleted == 0, "Pre-T1 tombstone (retired_at=NULL) must NOT be GC'd"
    assert (
        s.conn.execute(
            "SELECT 1 FROM user_playbooks WHERE user_playbook_id = ?", (pid,)
        ).fetchone()
        is not None
    )


# ---------------------------------------------------------------------------
# (f) Legal-hold: held row skipped, no event, others still deleted
# ---------------------------------------------------------------------------


def test_gc_legal_hold_skips_held_row(tmp_path, monkeypatch):
    s = _store(tmp_path)

    pb_held = _make_user_playbook(content="held")
    pb_free = _make_user_playbook(content="free")
    s.save_user_playbooks([pb_held, pb_free])
    held_id = pb_held.user_playbook_id
    free_id = pb_free.user_playbook_id

    old_retired_at = int(datetime(2019, 1, 1, tzinfo=UTC).timestamp())
    for pid in (held_id, free_id):
        _set_playbook_status(
            s, "user_playbooks", "user_playbook_id", pid, Status.MERGED.value
        )
        _set_retired_at(s, "user_playbooks", "user_playbook_id", pid, old_retired_at)

    # Monkeypatch: only held_id is on legal hold
    def mock_hold(org_id: str, entity_type: str, entity_id: str) -> bool:
        return entity_id == str(held_id)

    monkeypatch.setattr(s, "_is_on_legal_hold", mock_hold)

    cutoff = int(datetime(2021, 1, 1, tzinfo=UTC).timestamp())
    deleted = s.gc_expired_tombstones(
        entity_type="user_playbook", older_than_epoch=cutoff
    )

    assert deleted == 1
    # held row still exists
    assert (
        s.conn.execute(
            "SELECT 1 FROM user_playbooks WHERE user_playbook_id = ?", (held_id,)
        ).fetchone()
        is not None
    )
    # free row deleted
    assert (
        s.conn.execute(
            "SELECT 1 FROM user_playbooks WHERE user_playbook_id = ?", (free_id,)
        ).fetchone()
        is None
    )
    # No hard_delete event for held row
    assert len(_hard_delete_events(s, str(held_id))) == 0
    # hard_delete event for free row
    assert len(_hard_delete_events(s, str(free_id))) == 1


# ---------------------------------------------------------------------------
# (g) Idempotent: second GC call returns 0 and adds no new events
# ---------------------------------------------------------------------------


def test_gc_idempotent(tmp_path):
    s = _store(tmp_path)
    pb = _make_user_playbook()
    s.save_user_playbooks([pb])
    pid = pb.user_playbook_id
    _set_playbook_status(
        s, "user_playbooks", "user_playbook_id", pid, Status.MERGED.value
    )
    _set_retired_at(
        s,
        "user_playbooks",
        "user_playbook_id",
        pid,
        int(datetime(2018, 1, 1, tzinfo=UTC).timestamp()),
    )

    cutoff = int(datetime(2021, 1, 1, tzinfo=UTC).timestamp())
    first = s.gc_expired_tombstones(
        entity_type="user_playbook", older_than_epoch=cutoff
    )
    assert first == 1

    events_after_first = s.get_lineage_events(entity_id=str(pid))

    second = s.gc_expired_tombstones(
        entity_type="user_playbook", older_than_epoch=cutoff
    )
    assert second == 0

    events_after_second = s.get_lineage_events(entity_id=str(pid))
    assert len(events_after_second) == len(events_after_first)


# ---------------------------------------------------------------------------
# Edge: unknown entity_type raises ValueError
# ---------------------------------------------------------------------------


def test_gc_unknown_entity_type_raises(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(ValueError, match="unknown entity_type"):
        s.gc_expired_tombstones(entity_type="bogus_type", older_than_epoch=_now_epoch())


# ---------------------------------------------------------------------------
# Edge: empty table returns 0 without error
# ---------------------------------------------------------------------------


def test_gc_empty_table_returns_zero(tmp_path):
    s = _store(tmp_path)
    cutoff = int(datetime(2030, 1, 1, tzinfo=UTC).timestamp())
    result = s.gc_expired_tombstones(entity_type="profile", older_than_epoch=cutoff)
    assert result == 0


# ---------------------------------------------------------------------------
# Boundary exclusivity: cutoff is EXCLUSIVE (strictly-less-than)
# ---------------------------------------------------------------------------


def test_gc_boundary_at_cutoff_not_deleted(tmp_path):
    """A tombstone whose retired_at == older_than_epoch must NOT be deleted.

    The contract is strictly-less-than.
    """
    s = _store(tmp_path)
    cutoff_epoch = int(datetime(2022, 6, 15, 12, 0, 0, tzinfo=UTC).timestamp())

    # Row at EXACTLY the cutoff second — must survive GC.
    pb_at = _make_user_playbook(content="at-boundary")
    s.save_user_playbooks([pb_at])
    pid_at = pb_at.user_playbook_id
    _set_playbook_status(
        s, "user_playbooks", "user_playbook_id", pid_at, Status.MERGED.value
    )
    _set_retired_at(s, "user_playbooks", "user_playbook_id", pid_at, cutoff_epoch)

    # Row one second BEFORE the cutoff — must be deleted.
    pb_before = _make_user_playbook(content="before-boundary")
    s.save_user_playbooks([pb_before])
    pid_before = pb_before.user_playbook_id
    _set_playbook_status(
        s, "user_playbooks", "user_playbook_id", pid_before, Status.MERGED.value
    )
    _set_retired_at(
        s, "user_playbooks", "user_playbook_id", pid_before, cutoff_epoch - 1
    )

    deleted = s.gc_expired_tombstones(
        entity_type="user_playbook", older_than_epoch=cutoff_epoch
    )

    assert deleted == 1
    # Row at boundary survives.
    assert (
        s.conn.execute(
            "SELECT 1 FROM user_playbooks WHERE user_playbook_id = ?", (pid_at,)
        ).fetchone()
        is not None
    ), "Row at exact cutoff epoch must NOT be deleted (exclusive boundary)"
    # Row before boundary is gone.
    assert (
        s.conn.execute(
            "SELECT 1 FROM user_playbooks WHERE user_playbook_id = ?", (pid_before,)
        ).fetchone()
        is None
    ), "Row one second before cutoff must be deleted"


def test_gc_boundary_profile_at_cutoff_not_deleted(tmp_path):
    """Profile boundary: retired_at at cutoff survives, before doesn't."""
    s = _store(tmp_path)
    cutoff_epoch = int(datetime(2022, 6, 15, 12, 0, 0, tzinfo=UTC).timestamp())

    p_at = _make_profile(profile_id="at-boundary", ts=_now_epoch())
    p_before = _make_profile(profile_id="before-boundary", ts=_now_epoch())
    s.add_user_profile("u1", [p_at])
    s.add_user_profile("u1", [p_before])
    _set_profile_status(s, "at-boundary", Status.MERGED.value)
    _set_profile_status(s, "before-boundary", Status.MERGED.value)
    _set_retired_at(s, "profiles", "profile_id", "at-boundary", cutoff_epoch)
    _set_retired_at(s, "profiles", "profile_id", "before-boundary", cutoff_epoch - 1)

    deleted = s.gc_expired_tombstones(
        entity_type="profile", older_than_epoch=cutoff_epoch
    )

    assert deleted == 1
    assert (
        s.conn.execute(
            "SELECT 1 FROM profiles WHERE profile_id = ?", ("at-boundary",)
        ).fetchone()
        is not None
    ), "Profile at exact cutoff epoch must NOT be deleted (exclusive boundary)"
    assert (
        s.conn.execute(
            "SELECT 1 FROM profiles WHERE profile_id = ?", ("before-boundary",)
        ).fetchone()
        is None
    ), "Profile one second before cutoff must be deleted"


# ---------------------------------------------------------------------------
# (g) limit guard: non-positive limit returns 0 without touching the DB
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_limit", [0, -1, -100])
def test_gc_non_positive_limit_returns_zero(tmp_path, bad_limit: int):
    s = _store(tmp_path)
    pb = _make_user_playbook()
    s.save_user_playbooks([pb])
    pid = pb.user_playbook_id
    _set_playbook_status(
        s, "user_playbooks", "user_playbook_id", pid, Status.MERGED.value
    )
    _set_retired_at(
        s,
        "user_playbooks",
        "user_playbook_id",
        pid,
        int(datetime(2019, 1, 1, tzinfo=UTC).timestamp()),
    )

    cutoff = int(datetime(2021, 1, 1, tzinfo=UTC).timestamp())
    result = s.gc_expired_tombstones(
        entity_type="user_playbook", older_than_epoch=cutoff, limit=bad_limit
    )

    assert result == 0
    # Row must NOT have been deleted
    assert (
        s.conn.execute(
            "SELECT 1 FROM user_playbooks WHERE user_playbook_id = ?", (pid,)
        ).fetchone()
        is not None
    ), "Row must survive when limit <= 0"


# ---------------------------------------------------------------------------
# (h) Atomic rollback: mid-operation failure leaves no partial state
# ---------------------------------------------------------------------------


def test_gc_rollback_on_mid_operation_failure(tmp_path):
    """A failure after emitting the hard_delete event but before DELETE must
    roll back — no orphan event, no partial deletion.

    We patch the module-level ``_append_event_stmt`` so it writes the first
    event then raises, simulating a failure partway through the write block.
    The ``except`` branch in ``gc_expired_tombstones`` must rollback so neither
    the partial event nor the row deletion persists.
    """
    s = _store(tmp_path)

    # Seed two tombstone rows so the loop calls _append_event_stmt twice.
    pb1 = _make_user_playbook(content="one")
    pb2 = _make_user_playbook(content="two")
    s.save_user_playbooks([pb1, pb2])
    old_retired_at = int(datetime(2019, 1, 1, tzinfo=UTC).timestamp())
    for pb in (pb1, pb2):
        _set_playbook_status(
            s,
            "user_playbooks",
            "user_playbook_id",
            pb.user_playbook_id,
            Status.MERGED.value,
        )
        _set_retired_at(
            s, "user_playbooks", "user_playbook_id", pb.user_playbook_id, old_retired_at
        )

    real_append = _lineage_mod._append_event_stmt
    call_count = 0

    def failing_append(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise RuntimeError("simulated mid-loop failure")
        return real_append(*args, **kwargs)

    cutoff = int(datetime(2021, 1, 1, tzinfo=UTC).timestamp())
    with (
        patch.object(_lineage_mod, "_append_event_stmt", side_effect=failing_append),
        pytest.raises(RuntimeError, match="simulated mid-loop failure"),
    ):
        s.gc_expired_tombstones(entity_type="user_playbook", older_than_epoch=cutoff)

    # Both rows must still exist — rollback prevented any partial deletion
    for pb in (pb1, pb2):
        assert (
            s.conn.execute(
                "SELECT 1 FROM user_playbooks WHERE user_playbook_id = ?",
                (pb.user_playbook_id,),
            ).fetchone()
            is not None
        ), f"Row {pb.user_playbook_id} must survive after rolled-back GC failure"

    # No hard_delete events should have persisted (rolled back with the transaction)
    for pb in (pb1, pb2):
        events = _hard_delete_events(s, str(pb.user_playbook_id))
        assert len(events) == 0, (
            f"Expected 0 hard_delete events after rollback for {pb.user_playbook_id}, "
            f"got {len(events)}"
        )
