"""Integration tests: B3 parity gate — reconstruct_profile_change_log vs legacy table.

This is the DROP GATE for ProfileChangeLog.  Before the legacy
``profile_change_logs`` table can be dropped, these tests must stay green.

Three cases are documented (matching the B3 spec):

1. NORMAL SUPERSEDE — reconstruction equals legacy output (added/removed by
   profile content and profile_id).
2. LEGACY-MISSING — a legacy row has no corresponding lineage event.  The
   reconstruction simply returns no row; the discrepancy is *tolerated*
   (best-effort drop), not a failure.
3. PURGED TOMBSTONE — the incumbent tombstone was GC'd after the legacy row
   was written.  Reconstruction yields an empty removed_profiles list rather
   than crashing; the legacy row retains the original content.  Both outcomes
   are documented and accepted.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from reflexio.lib._profiles import reconstruct_profile_change_log
from reflexio.models.api_schema.domain.entities import (
    LineageContext,
    ProfileChangeLog,
    UserProfile,
)
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store(tmp_path) -> SQLiteStorage:
    s = SQLiteStorage(org_id="parity-org", db_path=str(tmp_path / "parity.db"))
    s.migrate()
    return s


def _make_profile(
    user_id: str = "u1",
    profile_id: str = "p1",
    content: str = "hello world",
) -> UserProfile:
    return UserProfile(
        user_id=user_id,
        profile_id=profile_id,
        content=content,
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id=f"req_{profile_id}",
        profile_time_to_live=ProfileTimeToLive.INFINITY,
    )


def _dual_write_supersede(
    s: SQLiteStorage,
    *,
    user_id: str,
    old_id: str,
    new_id: str,
    request_id: str,
    old_content: str = "old content",
    new_content: str = "new content",
) -> tuple[UserProfile, UserProfile]:
    """Dual-write: legacy add_profile_change_log AND supersede_record.

    Mirrors production ProfileGenerationService which calls both paths
    during the transition window before the legacy table is dropped.
    Returns (old_profile, new_profile).
    """
    old = _make_profile(user_id=user_id, profile_id=old_id, content=old_content)
    new = _make_profile(user_id=user_id, profile_id=new_id, content=new_content)
    s.add_user_profile(user_id, [old])
    s.add_user_profile(user_id, [new])

    ctx = LineageContext(op_kind="revise", actor="reflection", request_id=request_id)
    s.supersede_record(
        entity_type="profile",
        incumbent_id=old_id,
        successor_id=new_id,
        context=ctx,
    )

    # Legacy side: write to profile_change_logs table
    legacy_log = ProfileChangeLog(
        id=0,
        user_id=user_id,
        request_id=request_id,
        created_at=int(datetime.now(UTC).timestamp()),
        added_profiles=[new],
        removed_profiles=[old],
        mentioned_profiles=[],
    )
    s.add_profile_change_log(legacy_log)
    return old, new


# ---------------------------------------------------------------------------
# Case 1 — NORMAL SUPERSEDE: reconstruction equals legacy
# ---------------------------------------------------------------------------


def test_reconstruction_parity_gate_normal_supersede(tmp_path):
    """CORE INVARIANT: reconstruction == legacy for a normal supersede.

    For every legacy ProfileChangeLog row produced by a supersede,
    reconstruct_profile_change_log must yield a matching row with
    identical added_profiles and removed_profiles (by content and
    profile_id), and mentioned_profiles=[].
    """
    s = _store(tmp_path)
    old_p, new_p = _dual_write_supersede(
        s,
        user_id="u1",
        old_id="p-old",
        new_id="p-new",
        request_id="req-parity",
        old_content="old facts",
        new_content="new facts",
    )

    # Legacy
    legacy_rows = s.get_profile_change_logs()
    assert len(legacy_rows) == 1, "expected exactly one legacy row"
    legacy = legacy_rows[0]

    # Reconstruction
    recon = reconstruct_profile_change_log(s)
    assert recon.success
    assert len(recon.profile_change_logs) == 1, "expected exactly one reconstructed row"
    row = recon.profile_change_logs[0]

    # --- request_id ---
    assert row.request_id == legacy.request_id

    # --- added_profiles: match by profile_id and content ---
    legacy_added = {p.profile_id: p.content for p in legacy.added_profiles}
    recon_added = {p.profile_id: p.content for p in row.added_profiles}
    assert recon_added == legacy_added, (
        f"added_profiles mismatch: recon={recon_added} legacy={legacy_added}"
    )

    # --- removed_profiles: match by profile_id and content ---
    legacy_removed = {p.profile_id: p.content for p in legacy.removed_profiles}
    recon_removed = {p.profile_id: p.content for p in row.removed_profiles}
    assert recon_removed == legacy_removed, (
        f"removed_profiles mismatch: recon={recon_removed} legacy={legacy_removed}"
    )

    # --- mentioned_profiles always empty on both sides ---
    assert legacy.mentioned_profiles == []
    assert row.mentioned_profiles == []

    # --- source_ids on the lineage event includes the incumbent ---
    events = s.get_lineage_events(entity_id="p-new")
    revise_events = [e for e in events if e.op == "revise"]
    assert revise_events, "revise event must exist"
    assert "p-old" in revise_events[0].source_ids


def test_reconstruction_parity_gate_multi_supersede(tmp_path):
    """Three supersedes: each reconstructed row matches its legacy counterpart."""
    s = _store(tmp_path)
    pairs = [
        ("u1", f"old-{i}", f"new-{i}", f"req-multi-{i}", f"old-c-{i}", f"new-c-{i}")
        for i in range(3)
    ]
    for user_id, old_id, new_id, req_id, old_c, new_c in pairs:
        _dual_write_supersede(
            s,
            user_id=user_id,
            old_id=old_id,
            new_id=new_id,
            request_id=req_id,
            old_content=old_c,
            new_content=new_c,
        )

    legacy_by_req = {row.request_id: row for row in s.get_profile_change_logs()}
    recon_by_req = {
        row.request_id: row
        for row in reconstruct_profile_change_log(s).profile_change_logs
    }

    # Every legacy request_id must have a matching reconstruction.
    for req_id, legacy in legacy_by_req.items():
        assert req_id in recon_by_req, f"reconstruction missing req_id={req_id}"
        recon_row = recon_by_req[req_id]

        legacy_added = {p.profile_id: p.content for p in legacy.added_profiles}
        recon_added = {p.profile_id: p.content for p in recon_row.added_profiles}
        assert recon_added == legacy_added

        legacy_removed = {p.profile_id: p.content for p in legacy.removed_profiles}
        recon_removed = {p.profile_id: p.content for p in recon_row.removed_profiles}
        assert recon_removed == legacy_removed


# ---------------------------------------------------------------------------
# Case 2 — LEGACY-MISSING: legacy row with no lineage event (tolerated)
# ---------------------------------------------------------------------------


def test_reconstruction_parity_gate_legacy_missing(tmp_path):
    """A legacy row with no lineage event reconstructs as nothing — TOLERATED.

    This happens if a legacy row was written by old code that did not
    yet emit lineage events.  The parity checker classifies this as
    LEGACY-MISSING and tolerates it (not a failure).
    """
    s = _store(tmp_path)

    old_p = _make_profile(user_id="u1", profile_id="p-legacy-only-old")
    new_p = _make_profile(user_id="u1", profile_id="p-legacy-only-new")
    s.add_user_profile("u1", [old_p, new_p])

    # Write ONLY the legacy row — no lineage event.
    legacy_log = ProfileChangeLog(
        id=0,
        user_id="u1",
        request_id="req-legacy-only",
        created_at=int(datetime.now(UTC).timestamp()),
        added_profiles=[new_p],
        removed_profiles=[old_p],
        mentioned_profiles=[],
    )
    s.add_profile_change_log(legacy_log)

    # Legacy has one row; reconstruction has none — that's the LEGACY-MISSING case.
    legacy_rows = s.get_profile_change_logs()
    recon = reconstruct_profile_change_log(s)

    assert len(legacy_rows) == 1
    assert recon.profile_change_logs == [], (
        "LEGACY-MISSING: no lineage event means nothing to reconstruct"
    )


# ---------------------------------------------------------------------------
# Case 3 — PURGED TOMBSTONE: blank body in reconstruction (documented)
# ---------------------------------------------------------------------------


def test_reconstruction_parity_gate_purged_tombstone(tmp_path):
    """Purged tombstone: reconstruction yields empty removed_profiles (tolerated).

    When GC hard-deletes the tombstone, the legacy row retains the
    original content.  Reconstruction returns removed_profiles=[] rather
    than crashing.  Both outcomes are documented and accepted.
    """
    s = _store(tmp_path)
    old_p, new_p = _dual_write_supersede(
        s,
        user_id="u1",
        old_id="p-gc-old",
        new_id="p-gc-new",
        request_id="req-gc",
        old_content="content before gc",
        new_content="content after gc",
    )

    # Simulate GC hard-deleting the tombstone.
    s.conn.execute("DELETE FROM profiles WHERE profile_id = ?", ("p-gc-old",))
    s.conn.commit()

    # Legacy row still has the content.
    legacy_rows = s.get_profile_change_logs()
    assert len(legacy_rows) == 1
    assert legacy_rows[0].removed_profiles[0].content == "content before gc"

    # Reconstruction yields empty removed_profiles — graceful blank, not a crash.
    recon = reconstruct_profile_change_log(s)
    assert recon.success
    assert len(recon.profile_change_logs) == 1
    row = recon.profile_change_logs[0]
    assert row.removed_profiles == [], (
        "PURGED TOMBSTONE: reconstruction yields empty removed list, not a crash"
    )
    # added_profiles still has the survivor.
    assert len(row.added_profiles) == 1
    assert row.added_profiles[0].profile_id == "p-gc-new"
