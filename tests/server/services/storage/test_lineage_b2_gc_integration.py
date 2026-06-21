"""Integration tests: gc_expired_tombstones + legal-hold stub (Lineage Phase B2).

Tests cover:
  (a) Aged tombstone (MERGED) past cutoff is GC'd: row deleted + hard_delete event.
  (b) Fresh tombstone and CURRENT row are NOT deleted.
  (c) ARCHIVED aged tombstone IS GC'd (broader eligible set than _TOMBSTONE).
  (d) PB-8 straddle: TEXT ISO created_at (playbooks) and INTEGER last_modified_timestamp
      (profiles) both apply the cutoff correctly — only genuinely-older rows purged.
  (e) Legal-hold: monkeypatched hold skips one row — no delete, no hard_delete event.
  (f) Idempotent: second GC call on an already-empty set returns 0, adds no events.
"""

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.domain.entities import UserPlaybook
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive, Status
from reflexio.models.api_schema.service_schemas import AgentPlaybook, UserProfile
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.sqlite_storage._base import _epoch_to_iso

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


def _set_playbook_created_at(
    s: SQLiteStorage, table: str, pk: str, pk_val: int, iso_ts: str
) -> None:
    """Force the created_at TEXT column on a playbook row for age comparisons."""
    s.conn.execute(
        f"UPDATE {table} SET created_at = ? WHERE {pk} = ?",  # noqa: S608
        (iso_ts, pk_val),
    )
    s.conn.commit()


def _set_profile_last_modified(s: SQLiteStorage, profile_id: str, ts: int) -> None:
    s.conn.execute(
        "UPDATE profiles SET last_modified_timestamp = ? WHERE profile_id = ?",
        (ts, profile_id),
    )
    s.conn.commit()


def _hard_delete_events(s: SQLiteStorage, entity_id: str) -> list:
    return [
        e for e in s.get_lineage_events(entity_id=entity_id) if e.op == "hard_delete"
    ]


# ---------------------------------------------------------------------------
# (a) Aged tombstone (MERGED) past cutoff is GC'd
# ---------------------------------------------------------------------------


def test_gc_deletes_aged_merged_tombstone(tmp_path):
    s = _store(tmp_path)
    pb = _make_user_playbook()
    s.save_user_playbooks([pb])
    pid = pb.user_playbook_id
    _set_playbook_status(
        s, "user_playbooks", "user_playbook_id", pid, Status.MERGED.value
    )

    # Age the row to well before the cutoff — use production format (+00:00, no fractional seconds)
    old_iso = _epoch_to_iso(int(datetime(2020, 1, 1, tzinfo=UTC).timestamp()))
    _set_playbook_created_at(s, "user_playbooks", "user_playbook_id", pid, old_iso)

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
# (b) Fresh tombstone and CURRENT row are NOT deleted
# ---------------------------------------------------------------------------


def test_gc_skips_fresh_tombstone_and_current(tmp_path):
    s = _store(tmp_path)
    # Fresh merged tombstone — created_at is right now
    pb_fresh = _make_user_playbook(content="fresh")
    s.save_user_playbooks([pb_fresh])
    pid_fresh = pb_fresh.user_playbook_id
    _set_playbook_status(
        s, "user_playbooks", "user_playbook_id", pid_fresh, Status.MERGED.value
    )
    # Note: created_at is already current, so it's after any historical cutoff

    # CURRENT row (no status)
    pb_current = _make_user_playbook(content="current")
    s.save_user_playbooks([pb_current])
    pid_current = pb_current.user_playbook_id
    _set_playbook_created_at(
        s,
        "user_playbooks",
        "user_playbook_id",
        pid_current,
        _epoch_to_iso(int(datetime(2019, 1, 1, tzinfo=UTC).timestamp())),
    )
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
    _set_playbook_created_at(
        s,
        "agent_playbooks",
        "agent_playbook_id",
        apid,
        _epoch_to_iso(int(datetime(2019, 6, 1, tzinfo=UTC).timestamp())),
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
# (d) PB-8 straddle test: TEXT ISO vs INTEGER epoch type-correct comparison
# ---------------------------------------------------------------------------


def test_gc_straddle_playbook_text_iso_and_profile_int_epoch(tmp_path):
    s = _store(tmp_path)

    # --- User playbooks: TEXT ISO created_at ---
    pb_old = _make_user_playbook(content="old")
    pb_new = _make_user_playbook(content="new")
    s.save_user_playbooks([pb_old, pb_new])
    old_id = pb_old.user_playbook_id
    new_id = pb_new.user_playbook_id

    _set_playbook_status(
        s, "user_playbooks", "user_playbook_id", old_id, Status.MERGED.value
    )
    _set_playbook_status(
        s, "user_playbooks", "user_playbook_id", new_id, Status.MERGED.value
    )
    _set_playbook_created_at(
        s,
        "user_playbooks",
        "user_playbook_id",
        old_id,
        _epoch_to_iso(int(datetime(2018, 1, 1, tzinfo=UTC).timestamp())),
    )
    _set_playbook_created_at(
        s,
        "user_playbooks",
        "user_playbook_id",
        new_id,
        _epoch_to_iso(int(datetime(2025, 1, 1, tzinfo=UTC).timestamp())),
    )

    cutoff_pb = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp())
    deleted_pb = s.gc_expired_tombstones(
        entity_type="user_playbook", older_than_epoch=cutoff_pb
    )
    assert deleted_pb == 1
    assert (
        s.conn.execute(
            "SELECT 1 FROM user_playbooks WHERE user_playbook_id = ?", (old_id,)
        ).fetchone()
        is None
    )
    assert (
        s.conn.execute(
            "SELECT 1 FROM user_playbooks WHERE user_playbook_id = ?", (new_id,)
        ).fetchone()
        is not None
    )

    # --- Profiles: INTEGER last_modified_timestamp ---
    old_ts = int(datetime(2018, 6, 1, tzinfo=UTC).timestamp())
    new_ts = int(datetime(2025, 6, 1, tzinfo=UTC).timestamp())

    p_old = _make_profile(profile_id="prof-old", ts=old_ts)
    p_new = _make_profile(profile_id="prof-new", ts=new_ts)
    s.add_user_profile("u1", [p_old])
    s.add_user_profile("u1", [p_new])
    _set_profile_status(s, "prof-old", Status.SUPERSEDED.value)
    _set_profile_status(s, "prof-new", Status.SUPERSEDED.value)
    # Ensure last_modified_timestamp is set to our desired values
    _set_profile_last_modified(s, "prof-old", old_ts)
    _set_profile_last_modified(s, "prof-new", new_ts)

    cutoff_pr = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp())
    deleted_pr = s.gc_expired_tombstones(
        entity_type="profile", older_than_epoch=cutoff_pr
    )
    assert deleted_pr == 1
    assert (
        s.conn.execute(
            "SELECT 1 FROM profiles WHERE profile_id = ?", ("prof-old",)
        ).fetchone()
        is None
    )
    assert (
        s.conn.execute(
            "SELECT 1 FROM profiles WHERE profile_id = ?", ("prof-new",)
        ).fetchone()
        is not None
    )


# ---------------------------------------------------------------------------
# (e) Legal-hold: held row skipped, no event, others still deleted
# ---------------------------------------------------------------------------


def test_gc_legal_hold_skips_held_row(tmp_path, monkeypatch):
    s = _store(tmp_path)

    pb_held = _make_user_playbook(content="held")
    pb_free = _make_user_playbook(content="free")
    s.save_user_playbooks([pb_held, pb_free])
    held_id = pb_held.user_playbook_id
    free_id = pb_free.user_playbook_id

    for pid in (held_id, free_id):
        _set_playbook_status(
            s, "user_playbooks", "user_playbook_id", pid, Status.MERGED.value
        )
        _set_playbook_created_at(
            s,
            "user_playbooks",
            "user_playbook_id",
            pid,
            _epoch_to_iso(int(datetime(2019, 1, 1, tzinfo=UTC).timestamp())),
        )

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
# (f) Idempotent: second GC call returns 0 and adds no new events
# ---------------------------------------------------------------------------


def test_gc_idempotent(tmp_path):
    s = _store(tmp_path)
    pb = _make_user_playbook()
    s.save_user_playbooks([pb])
    pid = pb.user_playbook_id
    _set_playbook_status(
        s, "user_playbooks", "user_playbook_id", pid, Status.MERGED.value
    )
    _set_playbook_created_at(
        s,
        "user_playbooks",
        "user_playbook_id",
        pid,
        _epoch_to_iso(int(datetime(2018, 1, 1, tzinfo=UTC).timestamp())),
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


def test_gc_boundary_playbook_at_cutoff_not_deleted(tmp_path):
    """A playbook whose created_at == older_than_epoch must NOT be deleted.

    The contract is strictly-less-than.  The cutoff string is formatted by
    _epoch_to_iso, matching the production write path exactly, so the
    lexicographic ``<`` comparison is byte-for-byte consistent.
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
    _set_playbook_created_at(
        s, "user_playbooks", "user_playbook_id", pid_at, _epoch_to_iso(cutoff_epoch)
    )

    # Row one second BEFORE the cutoff — must be deleted.
    pb_before = _make_user_playbook(content="before-boundary")
    s.save_user_playbooks([pb_before])
    pid_before = pb_before.user_playbook_id
    _set_playbook_status(
        s, "user_playbooks", "user_playbook_id", pid_before, Status.MERGED.value
    )
    _set_playbook_created_at(
        s,
        "user_playbooks",
        "user_playbook_id",
        pid_before,
        _epoch_to_iso(cutoff_epoch - 1),
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
    """Profile (INTEGER age column) boundary: row at cutoff survives, row before doesn't."""
    s = _store(tmp_path)
    cutoff_epoch = int(datetime(2022, 6, 15, 12, 0, 0, tzinfo=UTC).timestamp())

    p_at = _make_profile(profile_id="at-boundary", ts=cutoff_epoch)
    p_before = _make_profile(profile_id="before-boundary", ts=cutoff_epoch - 1)
    s.add_user_profile("u1", [p_at])
    s.add_user_profile("u1", [p_before])
    _set_profile_status(s, "at-boundary", Status.MERGED.value)
    _set_profile_status(s, "before-boundary", Status.MERGED.value)
    _set_profile_last_modified(s, "at-boundary", cutoff_epoch)
    _set_profile_last_modified(s, "before-boundary", cutoff_epoch - 1)

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
    _set_playbook_created_at(
        s,
        "user_playbooks",
        "user_playbook_id",
        pid,
        _epoch_to_iso(int(datetime(2019, 1, 1, tzinfo=UTC).timestamp())),
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
    from unittest.mock import patch

    import reflexio.server.services.storage.sqlite_storage._lineage as _lineage_mod

    s = _store(tmp_path)

    # Seed two tombstone rows so the loop calls _append_event_stmt twice.
    pb1 = _make_user_playbook(content="one")
    pb2 = _make_user_playbook(content="two")
    s.save_user_playbooks([pb1, pb2])
    for pb in (pb1, pb2):
        _set_playbook_status(
            s,
            "user_playbooks",
            "user_playbook_id",
            pb.user_playbook_id,
            Status.MERGED.value,
        )
        _set_playbook_created_at(
            s,
            "user_playbooks",
            "user_playbook_id",
            pb.user_playbook_id,
            _epoch_to_iso(int(datetime(2019, 1, 1, tzinfo=UTC).timestamp())),
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
