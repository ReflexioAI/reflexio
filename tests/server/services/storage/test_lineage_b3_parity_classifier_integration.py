"""Integration tests: B3 parity classifier.

Verifies that ``classify_change_log_parity`` labels each class correctly on
seeded data:
  - MATCH: legacy and reconstruction agree on content.
  - RECON-MISSING: legacy row exists AND a reconstructible signal exists but
    reconstruction dropped the run (real gap → exit non-zero).
  - LEGACY-MISSING: legacy row exists but NO reconstructible signal (tolerated —
    predates soft-delete or purged).
  - CONTENT_MISMATCH: both sides exist but content differs, or recon-only.
"""

from __future__ import annotations

# Import run_parity_check from the script (it calls storage, so still lives there).
import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from reflexio.lib._lineage_parity import (
    ParityClass,
    classify_change_log_parity,
)
from reflexio.models.api_schema.domain.entities import (
    ProfileChangeLog,
    UserProfile,
)
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "lineage_b3_parity_check.py"
_spec = importlib.util.spec_from_file_location("lineage_b3_parity_check", _SCRIPT)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("lineage_b3_parity_check", _mod)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

run_parity_check = _mod.run_parity_check

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store(tmp_path) -> SQLiteStorage:
    s = SQLiteStorage(org_id=f"cls-org-{tmp_path.name}", db_path=str(tmp_path / "c.db"))
    s.migrate()
    return s


def _make_profile(
    user_id: str = "u1",
    profile_id: str = "p1",
    content: str = "c",
    request_id: str = "",
) -> UserProfile:
    return UserProfile(
        user_id=user_id,
        profile_id=profile_id,
        content=content,
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id=request_id or f"gen_{profile_id}",
        profile_time_to_live=ProfileTimeToLive.INFINITY,
    )


def _legacy_row(
    user_id: str,
    request_id: str,
    added: list[UserProfile],
    removed: list[UserProfile],
) -> ProfileChangeLog:
    return ProfileChangeLog(
        id=0,
        user_id=user_id,
        request_id=request_id,
        created_at=int(datetime.now(UTC).timestamp()),
        added_profiles=added,
        removed_profiles=removed,
        mentioned_profiles=[],
    )


# ---------------------------------------------------------------------------
# Unit-level: classify_change_log_parity labels correctly on hand-crafted inputs
# ---------------------------------------------------------------------------


def test_classify_match() -> None:
    """MATCH when both sides have identical added/removed content."""
    p_old = _make_profile("u1", "p-old", "old c")
    p_new = _make_profile("u1", "p-new", "new c")
    legacy = [_legacy_row("u1", "req-1", [p_new], [p_old])]
    recon = [_legacy_row("u1", "req-1", [p_new], [p_old])]
    results = classify_change_log_parity(
        legacy, recon, reconstructible_request_ids=set(), read_cap_hit=False
    )
    assert len(results) == 1
    assert results[0].classification == ParityClass.MATCH


def test_classify_legacy_only_with_reconstructible_signal_is_recon_missing() -> None:
    """RECON-MISSING when legacy row exists AND a reconstructible signal is present."""
    p_old = _make_profile("u1", "p-old", "old c")
    p_new = _make_profile("u1", "p-new", "new c")
    legacy = [_legacy_row("u1", "req-gap", [p_new], [p_old])]
    recon: list[ProfileChangeLog] = []
    results = classify_change_log_parity(
        legacy,
        recon,
        reconstructible_request_ids={"req-gap"},
        read_cap_hit=False,
    )
    assert len(results) == 1
    assert results[0].classification == ParityClass.RECON_MISSING


def test_classify_legacy_only_without_reconstructible_signal_is_legacy_missing() -> (
    None
):
    """LEGACY-MISSING when legacy row exists but no reconstructible signal (tolerated)."""
    p_old = _make_profile("u1", "p-old", "old c")
    p_new = _make_profile("u1", "p-new", "new c")
    legacy = [_legacy_row("u1", "req-old", [p_new], [p_old])]
    recon: list[ProfileChangeLog] = []
    results = classify_change_log_parity(
        legacy,
        recon,
        reconstructible_request_ids=set(),
        read_cap_hit=False,
    )
    assert len(results) == 1
    assert results[0].classification == ParityClass.LEGACY_MISSING


def test_classify_content_mismatch_both_exist() -> None:
    """CONTENT_MISMATCH when both sides exist but content disagrees."""
    p_old = _make_profile("u1", "p-old", "old c")
    p_new_legacy = _make_profile("u1", "p-new", "correct content")
    p_new_recon = _make_profile("u1", "p-new", "WRONG content")
    legacy = [_legacy_row("u1", "req-mismatch", [p_new_legacy], [p_old])]
    recon = [_legacy_row("u1", "req-mismatch", [p_new_recon], [p_old])]
    results = classify_change_log_parity(
        legacy,
        recon,
        reconstructible_request_ids=set(),
        read_cap_hit=False,
    )
    assert len(results) == 1
    assert results[0].classification == ParityClass.CONTENT_MISMATCH


def test_classify_recon_only_is_content_mismatch() -> None:
    """CONTENT_MISMATCH when recon has a row but legacy list is empty."""
    p_old = _make_profile("u1", "p-old", "old c")
    p_new = _make_profile("u1", "p-new", "new c")
    legacy: list[ProfileChangeLog] = []
    recon = [_legacy_row("u1", "req-orphan", [p_new], [p_old])]
    results = classify_change_log_parity(
        legacy,
        recon,
        reconstructible_request_ids=set(),
        read_cap_hit=False,
    )
    assert len(results) == 1
    assert results[0].classification == ParityClass.CONTENT_MISMATCH


def test_classify_mixed() -> None:
    """Mixed: one MATCH, one LEGACY_MISSING (no signal), one CONTENT_MISMATCH (recon-only)."""
    p = _make_profile

    # MATCH
    l_match = _legacy_row(
        "u1", "req-match", [p("u1", "pa", "ca")], [p("u1", "pb", "cb")]
    )
    r_match = _legacy_row(
        "u1", "req-match", [p("u1", "pa", "ca")], [p("u1", "pb", "cb")]
    )

    # LEGACY_MISSING (legacy only, no reconstructible signal)
    l_gap = _legacy_row("u1", "req-gap", [p("u1", "pc", "cc")], [])

    # CONTENT_MISMATCH (recon only)
    r_orphan = _legacy_row("u1", "req-orphan", [p("u1", "pd", "cd")], [])

    results = classify_change_log_parity(
        [l_match, l_gap],
        [r_match, r_orphan],
        reconstructible_request_ids=set(),
        read_cap_hit=False,
    )
    by_req = {r.request_id: r.classification for r in results}

    assert by_req["req-match"] == ParityClass.MATCH
    assert by_req["req-gap"] == ParityClass.LEGACY_MISSING
    assert by_req["req-orphan"] == ParityClass.CONTENT_MISMATCH


# ---------------------------------------------------------------------------
# Integration-level: run_parity_check against live SQLite storage
# ---------------------------------------------------------------------------


def test_run_parity_check_all_match(tmp_path) -> None:
    """run_parity_check returns all MATCH when both paths are seeded together.

    Uses the real dedup path:
      - new profiles carry generated_from_request_id == request_id (add signal)
      - old profiles are soft-deleted via supersede_profiles_by_ids (removal signal:
        status_change+superseded event)
    """
    s = _store(tmp_path)

    for i in range(3):
        req_id = f"req-full-{i}"
        old_p = _make_profile("u1", f"p-old-{i}", f"old-{i}", request_id="seed")
        new_p = _make_profile("u1", f"p-new-{i}", f"new-{i}", request_id=req_id)
        s.add_user_profile("u1", [old_p])
        s.add_user_profile("u1", [new_p])
        # Dedup removal: emits status_change(to_status="superseded") under req_id
        s.supersede_profiles_by_ids("u1", [f"p-old-{i}"], req_id)
        s.add_profile_change_log(_legacy_row("u1", req_id, [new_p], [old_p]))

    results = run_parity_check(s)
    match_results = [r for r in results if r.request_id.startswith("req-full-")]
    assert match_results, "expected at least one req-full-* result (non-vacuous guard)"
    assert all(r.classification == ParityClass.MATCH for r in match_results), (
        f"expected all MATCH but got: {[(r.request_id, r.classification) for r in match_results]}"
    )


def test_run_parity_check_detects_recon_missing(tmp_path) -> None:
    """run_parity_check flags RECON-MISSING when legacy-only row exists with a reconstructible signal.

    To produce RECON_MISSING we need:
      1. A status_change+superseded lineage event for "req-gap-only" (puts it
         in the reconstructible set).
      2. The superseded profile is physically deleted (reconstruction can't
         fetch it → removed=[]).
      3. No profile has generated_from_request_id="req-gap-only" → added=[].
      4. Reconstruction skips the row (empty added+removed).
      5. Legacy table has the row → RECON_MISSING.
    """
    s = _store(tmp_path)

    old_p = _make_profile("u1", "p-gap-old", "old", request_id="seed-old")
    s.add_user_profile("u1", [old_p])
    # Dedup soft-delete: emits status_change+superseded with request_id="req-gap-only".
    s.supersede_profiles_by_ids("u1", ["p-gap-old"], "req-gap-only")
    # Simulate GC hard-deleting the tombstone so reconstruction can't find the profile.
    s.conn.execute("DELETE FROM profiles WHERE profile_id = ?", ("p-gap-old",))
    s.conn.commit()

    # Legacy row exists for "req-gap-only" — reconstruction produces nothing (empty add+remove).
    s.add_profile_change_log(
        _legacy_row("u1", "req-gap-only", [], [old_p])  # legacy retains the content
    )

    results = run_parity_check(s)
    gaps = [r for r in results if r.classification == ParityClass.RECON_MISSING]
    assert gaps, "expected at least one RECON-MISSING"
    assert any(r.request_id == "req-gap-only" for r in gaps)


def test_run_parity_check_recon_only_is_content_mismatch(tmp_path) -> None:
    """run_parity_check classifies a recon-only run as CONTENT_MISMATCH (not RECON-MISSING).

    Seeds a real dedup run (status_change+superseded event + profile with
    generated_from_request_id) WITHOUT writing a legacy row — this represents
    a run that used the new path but didn't dual-write to the legacy table.
    The reconstruction produces a row; the legacy table has none → CONTENT_MISMATCH
    (recon-only). This is a gap (exit 1) but distinct from a RECON-MISSING gap.
    """
    s = _store(tmp_path)

    req_id = "req-orphan-only"
    old_p = _make_profile("u1", "p-orphan-old", "old", request_id="orphan-seed")
    new_p = _make_profile("u1", "p-orphan-new", "new", request_id=req_id)
    s.add_user_profile("u1", [old_p])
    s.add_user_profile("u1", [new_p])
    # Dedup removal — no legacy row written.
    s.supersede_profiles_by_ids("u1", ["p-orphan-old"], req_id)

    results = run_parity_check(s)
    # Recon-only → CONTENT_MISMATCH (reconstruction produced a run absent from legacy).
    content_mismatch = [
        r for r in results if r.classification == ParityClass.CONTENT_MISMATCH
    ]
    assert any(r.request_id == req_id for r in content_mismatch), (
        f"Expected CONTENT_MISMATCH for {req_id!r}; got {[(r.request_id, r.classification) for r in results]}"
    )
    # No RECON-MISSING → only CONTENT_MISMATCH (recon-only) which IS a gap and
    # would cause exit 1, but not a RECON_MISSING gap.
    gaps = [r for r in results if r.classification == ParityClass.RECON_MISSING]
    assert not gaps, "recon-only run should not trigger RECON-MISSING"
