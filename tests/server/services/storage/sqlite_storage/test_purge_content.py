"""Task 1: has_inbound_lineage_refs query for purge-vs-hard-delete decision."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain.entities import LineageContext, UserProfile
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def _ctx(rid: str = "r1") -> LineageContext:
    return LineageContext(op_kind="revise", actor="test", reason="t", request_id=rid)


def _profile(pid: str, uid: str, content: str) -> UserProfile:
    return UserProfile(
        profile_id=pid,
        user_id=uid,
        content=content,
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id="req-1",
    )


@pytest.fixture
def storage(tmp_path):
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        yield SQLiteStorage(org_id="0", db_path=str(tmp_path / "t.db"))


def test_has_inbound_lineage_refs_true_when_pointed_to(storage):
    # Two profiles; B supersedes A → A.superseded_by=B, so B has an inbound ref.
    storage.add_user_profile("alice", [_profile("A", "alice", "old")])
    storage.add_user_profile("alice", [_profile("B", "alice", "new")])
    storage.supersede_record(
        entity_type="profile", incumbent_id="A", successor_id="B", context=_ctx()
    )
    assert (
        storage.has_inbound_lineage_refs(entity_type="profile", entity_id="B") is True
    )
    assert (
        storage.has_inbound_lineage_refs(entity_type="profile", entity_id="A") is False
    )


def test_purge_blanks_body_keeps_skeleton(storage):
    storage.add_user_profile("alice", [_profile("A", "alice", "alice@x.com secret")])
    storage.add_user_profile("alice", [_profile("B", "alice", "new")])
    storage.supersede_record(
        entity_type="profile", incumbent_id="A", successor_id="B", context=_ctx()
    )
    assert storage.purge_content(entity_type="profile", entity_id="A") is True
    row = storage.conn.execute(
        "SELECT content, user_id, status, superseded_by FROM profiles WHERE profile_id='A'"
    ).fetchone()
    assert row["content"] == ""  # body blanked
    assert row["user_id"] == ""  # PII blanked to '' (NOT NULL)
    assert row["status"] == "superseded"  # skeleton kept
    assert row["superseded_by"] == "B"  # pointer kept


def test_purge_emits_one_pii_free_event_idempotent(storage):
    storage.add_user_profile("alice", [_profile("A", "alice", "x")])
    storage.add_user_profile("alice", [_profile("B", "alice", "y")])
    storage.supersede_record(
        entity_type="profile", incumbent_id="A", successor_id="B", context=_ctx()
    )
    storage.purge_content(entity_type="profile", entity_id="A")
    storage.purge_content(entity_type="profile", entity_id="A")  # re-run
    events = storage.get_lineage_events(
        entity_type="profile", entity_id="A", org_id="0"
    )
    purges = [e for e in events if e.op == "purge"]
    assert len(purges) == 1  # deterministic request_id → no duplicate
    assert "alice" not in (purges[0].actor or "")  # event carries no user identifier
    assert purges[0].request_id == "purge_A"


def test_purge_returns_false_for_missing_entity(storage):
    assert (
        storage.purge_content(entity_type="profile", entity_id="nonexistent") is False
    )


class _BoomOnCommit:
    """Thin proxy around sqlite3.Connection that raises on the first commit call.

    sqlite3.Connection.commit is a C-level slot and cannot be monkeypatched
    directly, so we wrap the real connection in a proxy and swap s.conn.
    """

    def __init__(self, real_conn: object) -> None:
        self._real = real_conn
        self._boom = True  # raise on next commit

    def commit(self) -> None:
        if self._boom:
            raise RuntimeError("crash")
        self._real.commit()  # type: ignore[attr-defined]

    def rollback(self) -> None:
        self._real.rollback()  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)


def test_purge_atomic_no_phantom_event(tmp_path):
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        s = SQLiteStorage(org_id="0", db_path=str(tmp_path / "t.db"))
    s.add_user_profile("alice", [_profile("A", "alice", "x")])
    s.add_user_profile("alice", [_profile("B", "alice", "y")])
    s.supersede_record(
        entity_type="profile", incumbent_id="A", successor_id="B", context=_ctx()
    )
    # Swap in a proxy that raises on the first commit (post-UPDATE, pre-durability).
    real_conn = s.conn
    proxy = _BoomOnCommit(real_conn)
    s.conn = proxy  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="crash"):
        s.purge_content(entity_type="profile", entity_id="A")
    # Restore real connection and roll back the aborted transaction.
    s.conn = real_conn
    real_conn.rollback()
    # Neither the blank nor the event survived.
    row = real_conn.execute(
        "SELECT content FROM profiles WHERE profile_id='A'"
    ).fetchone()
    assert row["content"] != ""  # body intact
    purges = [
        e
        for e in s.get_lineage_events(entity_type="profile", entity_id="A", org_id="0")
        if e.op == "purge"
    ]
    assert purges == []


def test_has_inbound_lineage_refs_true_when_merged_into(storage):
    # Two profiles; A is merged into B → A.merged_into=B, so B has an inbound ref.
    storage.add_user_profile("bob", [_profile("C", "bob", "old")])
    storage.add_user_profile("bob", [_profile("D", "bob", "new")])
    storage.merge_records(
        entity_type="profile",
        survivor_id="D",
        source_ids=["C"],
        context=_ctx(rid="r2"),
    )
    assert (
        storage.has_inbound_lineage_refs(entity_type="profile", entity_id="D") is True
    )
    assert (
        storage.has_inbound_lineage_refs(entity_type="profile", entity_id="C") is False
    )


def test_clear_user_data_purges_referenced_keeps_chain(tmp_path):
    """Chain A→B→C(live) + standalone D; clear_user_data purges tombstones/referenced,
    hard-deletes unreferenced standalone rows, and chain still resolves after erasure.

    Ordering invariant: A and B are tombstones (superseded_by/merged_into set) so they
    are purge-eligible. C is the live survivor but has inbound lineage refs (B points to
    it), so it is also purged rather than hard-deleted. D is unreferenced → hard-deleted.
    """
    from reflexio.server.services.lineage.resolve import resolve_current

    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        s = SQLiteStorage(org_id="0", db_path=str(tmp_path / "t.db"))

    # Build chain A→B→C with supersede then merge; D is standalone.
    s.add_user_profile("alice", [_profile("A", "alice", "a")])
    s.add_user_profile("alice", [_profile("B", "alice", "b")])
    s.add_user_profile("alice", [_profile("C", "alice", "c")])
    s.add_user_profile("alice", [_profile("D", "alice", "d")])
    s.supersede_record(
        entity_type="profile", incumbent_id="A", successor_id="B", context=_ctx("r1")
    )
    s.merge_records(
        entity_type="profile", source_ids=["B"], survivor_id="C", context=_ctx("r2")
    )

    counts = s.clear_user_data("alice")

    # D must be hard-deleted (row gone).
    assert (
        s.conn.execute("SELECT 1 FROM profiles WHERE profile_id='D'").fetchone() is None
    )
    # C must be content-purged (skeleton kept, body blanked).
    c_row = s.conn.execute(
        "SELECT content FROM profiles WHERE profile_id='C'"
    ).fetchone()
    assert c_row is not None and c_row["content"] == ""
    # Chain A→B→C still resolves via lineage pointers.
    ref = resolve_current(s, "profile", "A")
    assert ref is not None and ref.id == "C" and ref.is_purged is True
    # Count checks: 3 purged (A, B, C), 1 hard-deleted (D).
    assert counts["purged_profiles"] == 3
    assert counts["profiles"] == 1


def test_clear_user_data_cross_user_chain_purges_other_users_survivor(tmp_path):
    """Cross-user: alice's tombstone A points to bob's live C.
    Erasing bob must PURGE C (not hard-delete it), so alice's A still resolves.
    Proves has_inbound_lineage_refs is not scoped to the user being erased.
    """
    from reflexio.server.services.lineage.resolve import resolve_current

    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        s = SQLiteStorage(org_id="0", db_path=str(tmp_path / "t.db"))

    s.add_user_profile("alice", [_profile("A", "alice", "a")])
    s.add_user_profile("bob", [_profile("C", "bob", "c")])  # survivor owned by bob
    s.supersede_record(
        entity_type="profile", incumbent_id="A", successor_id="C", context=_ctx()
    )

    s.clear_user_data("bob")  # erase bob; alice's tombstone A still points at C

    # C must be PURGED (skeleton kept), not hard-deleted.
    c_row = s.conn.execute(
        "SELECT content FROM profiles WHERE profile_id='C'"
    ).fetchone()
    assert c_row is not None, (
        "C must not be hard-deleted — alice's chain still references it"
    )
    # Alice's A still resolves to C via the pointer.
    ref = resolve_current(s, "profile", "A")
    assert ref is not None and ref.id == "C"
