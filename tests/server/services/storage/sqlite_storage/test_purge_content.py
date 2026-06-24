"""Task 1: has_inbound_lineage_refs query for purge-vs-hard-delete decision."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain.entities import (
    AgentPlaybook,
    AgentPlaybookSourceWindow,
    LineageContext,
    UserPlaybook,
    UserProfile,
)
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


def test_purge_with_empty_content_still_blanks_other_pii(storage):
    """Row with content='' but other PII populated must be fully purged.

    Guards against the former ``AND content != ''`` guard that would skip the
    UPDATE entirely when content was already blank, leaving user_id / embedding /
    tags / etc. intact.
    """
    # Insert a profile and then zero out content manually, leaving user_id set.
    storage.add_user_profile("alice", [_profile("E", "alice", "initial body")])
    # Blank only content so the row has content='' but user_id='alice' still live.
    storage.conn.execute("UPDATE profiles SET content='' WHERE profile_id='E'")
    storage.conn.commit()

    # Sanity: user_id is still populated before the purge.
    before = storage.conn.execute(
        "SELECT content, user_id FROM profiles WHERE profile_id='E'"
    ).fetchone()
    assert before["content"] == ""
    assert before["user_id"] == "alice"

    # Purge must return True (row exists) and blank user_id even though content was ''.
    assert storage.purge_content(entity_type="profile", entity_id="E") is True

    after = storage.conn.execute(
        "SELECT content, user_id FROM profiles WHERE profile_id='E'"
    ).fetchone()
    assert after["content"] == "", "content already blank — stays blank"
    assert after["user_id"] == "", (
        "user_id must be blanked even when content was already ''"
    )

    # Exactly one purge event recorded.
    events = storage.get_lineage_events(
        entity_type="profile", entity_id="E", org_id="0"
    )
    purges = [e for e in events if e.op == "purge"]
    assert len(purges) == 1


def test_purge_idempotent_on_already_purged_row(storage):
    """Re-running purge on an already-purged row yields exactly one op=purge event.

    The deterministic request_id ``"purge_{entity_id}"`` + INSERT OR IGNORE on the
    unique key ensures no duplicate event is recorded.
    """
    storage.add_user_profile("bob", [_profile("F", "bob", "some content")])
    storage.purge_content(entity_type="profile", entity_id="F")  # first purge
    storage.purge_content(entity_type="profile", entity_id="F")  # re-run

    events = storage.get_lineage_events(
        entity_type="profile", entity_id="F", org_id="0"
    )
    purges = [e for e in events if e.op == "purge"]
    assert len(purges) == 1, f"Expected 1 purge event, got {len(purges)}"
    assert purges[0].request_id == "purge_F"


def test_resolve_current_returns_is_purged_for_purged_survivor(storage):
    """Guard: resolve_current returns is_purged=True when the live survivor was purged.

    Contract lock-in for Task 6 (purge_content companion).  No production consumer
    currently dereferences the resolved record's content, so no skip-logic is needed
    today.  This test ensures the signal any future consumer relies on is stable and
    cannot regress silently.

    Audit finding: as of 2026-06-23 there are ZERO call sites outside of
    resolve_current itself and test code that call resolve_current and then read
    the returned RecordRef's content — the function is only consumed by tests and
    the clear_user_data path (which only uses the resolved id, not the content).
    No skip-guard was added to any consumer because none read content.
    """
    from reflexio.server.services.lineage.resolve import resolve_current

    # A→B (live survivor): purge B's content, then resolve from A.
    storage.add_user_profile("alice", [_profile("A", "alice", "old body")])
    storage.add_user_profile("alice", [_profile("B", "alice", "live body")])
    storage.supersede_record(
        entity_type="profile", incumbent_id="A", successor_id="B", context=_ctx()
    )

    # Before purge: is_purged must be False.
    ref_before = resolve_current(storage, "profile", "A")
    assert ref_before is not None
    assert ref_before.id == "B"
    assert ref_before.is_purged is False

    # Purge the live survivor's content.
    storage.purge_content(entity_type="profile", entity_id="B")

    # After purge: resolving from A must yield is_purged=True on the same id.
    ref_after = resolve_current(storage, "profile", "A")
    assert ref_after is not None
    assert ref_after.id == "B"
    assert ref_after.is_purged is True


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


def test_clear_user_data_tombstone_only_user(tmp_path):
    """Erasure reaches users whose ONLY rows are tombstones.

    Regression guard for the GDPR bug where ``clear_user_data`` enumerated
    profiles/user_playbooks with ``status_filter=[None, ARCHIVED, PENDING]``,
    which excluded SUPERSEDED and MERGED rows.  A user whose sole profile was
    superseded by another user's profile would survive erasure entirely.

    Scenario: alice's profile A is superseded by bob's profile B.
    After supersede, A has ``status='superseded'`` and A.superseded_by=B.
    Erasing alice must still reach A (her only row) and purge it.
    """
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        s = SQLiteStorage(org_id="0", db_path=str(tmp_path / "t.db"))

    # alice has one profile that ends up superseded by bob's profile.
    s.add_user_profile("alice", [_profile("A", "alice", "alice@secret.com")])
    s.add_user_profile("bob", [_profile("B", "bob", "bob content")])
    s.supersede_record(
        entity_type="profile", incumbent_id="A", successor_id="B", context=_ctx()
    )

    # Confirm A is now a tombstone with status='superseded'.
    row = s.conn.execute(
        "SELECT status, superseded_by FROM profiles WHERE profile_id='A'"
    ).fetchone()
    assert row["status"] == "superseded"
    assert row["superseded_by"] == "B"

    # Erase alice — her only profile is a tombstone; erasure must reach it.
    counts = s.clear_user_data("alice")

    # A is a tombstone (superseded_by=B set) → _is_lineage_tombstone returns True
    # → _partition_purge_vs_delete puts A in the purge set (not hard-delete).
    # The row must still exist, content blanked, and count as purged.
    a_row = s.conn.execute(
        "SELECT content, user_id FROM profiles WHERE profile_id='A'"
    ).fetchone()
    assert a_row is not None, (
        "Tombstone A must be content-purged (skeleton kept), not hard-deleted"
    )
    assert a_row["content"] == "", "Tombstone profile A must have its content blanked"
    assert counts["purged_profiles"] == 1, (
        f"Expected 1 purged profile, got {counts['purged_profiles']}"
    )
    assert counts["profiles"] == 0, (
        f"Expected 0 hard-deleted profiles, got {counts['profiles']}"
    )


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
    assert c_row["content"] == "", (
        "C's content must be blanked (purged), not merely kept"
    )
    # Alice's A still resolves to C via the pointer, and C is marked purged.
    ref = resolve_current(s, "profile", "A")
    assert ref is not None and ref.id == "C"
    assert ref.is_purged is True, (
        "Resolved survivor C must be marked is_purged after content purge"
    )


# ---------------------------------------------------------------------------
# F1 — user_playbook purge path
# ---------------------------------------------------------------------------


def _user_playbook(user_id: str, content: str = "secret body") -> UserPlaybook:
    return UserPlaybook(
        user_id=user_id,
        agent_version="v1",
        request_id="req-upb",
        content=content,
    )


def test_purge_user_playbook_blanks_body_user_id_null(storage):
    """purge_content on a user_playbook tombstone blanks body and sets user_id=NULL.

    user_playbooks.user_id is NULLABLE, so the purge SQL sets it to NULL
    (unlike profiles where user_id is NOT NULL and is blanked to '').
    The skeleton (status, pointers) must be kept intact and exactly one
    op=purge lineage event must be recorded.
    """
    # Create incumbent and successor user_playbooks.
    up_inc = _user_playbook("alice", "alice@secret.com")
    up_suc = _user_playbook("alice", "replacement body")
    storage.save_user_playbooks([up_inc])
    storage.save_user_playbooks([up_suc])
    inc_id = up_inc.user_playbook_id
    suc_id = up_suc.user_playbook_id

    # Supersede the incumbent — now it is a tombstone.
    storage.supersede_record(
        entity_type="user_playbook",
        incumbent_id=str(inc_id),
        successor_id=str(suc_id),
        context=_ctx("r-upb"),
    )

    # Confirm the tombstone state.
    row_before = storage.conn.execute(
        "SELECT user_id, superseded_by, status FROM user_playbooks WHERE user_playbook_id=?",
        (inc_id,),
    ).fetchone()
    assert row_before["user_id"] == "alice"
    assert row_before["superseded_by"] == suc_id

    # Purge the tombstone.
    result = storage.purge_content(entity_type="user_playbook", entity_id=str(inc_id))
    assert result is True

    # Content and user_id must be blanked; user_id is NULLABLE → goes to NULL.
    row = storage.conn.execute(
        "SELECT content, user_id, status, superseded_by FROM user_playbooks WHERE user_playbook_id=?",
        (inc_id,),
    ).fetchone()
    assert row["content"] == ""
    assert row["user_id"] is None  # NULLABLE column → NULL, not ''
    assert row["status"] == "superseded"  # skeleton kept
    assert row["superseded_by"] == suc_id  # pointer kept

    # Exactly one op=purge lineage event.
    events = storage.get_lineage_events(
        entity_type="user_playbook", entity_id=str(inc_id), org_id="0"
    )
    purges = [e for e in events if e.op == "purge"]
    assert len(purges) == 1
    assert purges[0].request_id == f"purge_{inc_id}"


def test_purge_agent_playbook_raises_value_error(storage):
    """purge_content with entity_type='agent_playbook' must raise ValueError."""
    with pytest.raises(ValueError, match="agent_playbook"):
        storage.purge_content(entity_type="agent_playbook", entity_id="x")


# ---------------------------------------------------------------------------
# F2 — chunk-boundary: clear_user_data handles more rows than chunk size
# ---------------------------------------------------------------------------


def test_clear_user_data_chunk_boundary(tmp_path, monkeypatch):
    """clear_user_data deletes all rows even when the count exceeds the chunk size.

    Monkeypatches the ``chunked`` helper used by ``_delete_in_chunks`` to force
    a chunk size of 2, then seeds 5 profiles so the delete-set crosses the chunk
    boundary. All rows must be deleted/purged without error.
    """
    import reflexio.server.services.storage.sqlite_storage._base as _base_mod
    from reflexio.server.services.storage.retention_mixin import chunked as real_chunked

    def chunked_size2(values, chunk_size=500):  # type: ignore[misc]
        return real_chunked(values, chunk_size=2)

    monkeypatch.setattr(_base_mod, "chunked", chunked_size2)

    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        s = SQLiteStorage(org_id="0", db_path=str(tmp_path / "t.db"))

    # Seed 5 standalone profiles for "alice" — all go to the hard-delete set.
    for i in range(5):
        s.add_user_profile("alice", [_profile(f"P{i}", "alice", f"body {i}")])

    counts = s.clear_user_data("alice")

    # All 5 profiles hard-deleted (no lineage refs, no tombstone).
    assert counts["profiles"] == 5, f"expected 5 hard-deleted, got {counts['profiles']}"
    assert counts["purged_profiles"] == 0

    # Confirm the table is empty for alice.
    remaining = s.conn.execute(
        "SELECT count(*) AS n FROM profiles WHERE user_id='alice'"
    ).fetchone()["n"]
    assert remaining == 0


# ---------------------------------------------------------------------------
# F3 — source-window orphan: clear_user_data cleans join rows for hard-deleted playbooks
# ---------------------------------------------------------------------------


def test_clear_user_data_no_orphan_source_window_rows(tmp_path):
    """After clear_user_data, no orphan rows remain in agent_playbook_source_user_playbooks
    for the erased user's hard-deleted playbooks.

    Source-window cleanup applies only to the HARD-DELETE set; a purged playbook
    keeps its skeleton and join rows. Here the user_playbook has no lineage refs,
    so it is hard-deleted and its join rows must be gone.
    """
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        s = SQLiteStorage(org_id="0", db_path=str(tmp_path / "t.db"))

    # Create a user_playbook for alice (standalone, no lineage → will be hard-deleted).
    upb = _user_playbook("alice", "alice's playbook body")
    s.save_user_playbooks([upb])
    upb_id = upb.user_playbook_id

    # Create an agent_playbook and link it to the user_playbook via a source window.
    [ap] = s.save_agent_playbooks(
        [AgentPlaybook(agent_version="v1", content="agent body")]
    )
    ap_id = ap.agent_playbook_id
    s.set_source_windows_for_agent_playbook(
        ap_id,
        [
            AgentPlaybookSourceWindow(
                user_playbook_id=upb_id, source_interaction_ids=[1]
            )
        ],
    )

    # Confirm the join row exists before erasure.
    join_before = s.conn.execute(
        "SELECT count(*) AS n FROM agent_playbook_source_user_playbooks WHERE user_playbook_id=?",
        (upb_id,),
    ).fetchone()["n"]
    assert join_before == 1

    # Erase alice — user_playbook is standalone → hard-deleted → join row must be removed.
    counts = s.clear_user_data("alice")
    assert counts["user_playbooks"] == 1  # hard-deleted, not purged
    assert counts["purged_user_playbooks"] == 0

    # No orphan join rows for the deleted user_playbook_id.
    join_after = s.conn.execute(
        "SELECT count(*) AS n FROM agent_playbook_source_user_playbooks WHERE user_playbook_id=?",
        (upb_id,),
    ).fetchone()["n"]
    assert join_after == 0, (
        f"Orphan source-window row survived for hard-deleted user_playbook_id={upb_id}"
    )
