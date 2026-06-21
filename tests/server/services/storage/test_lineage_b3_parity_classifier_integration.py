"""Integration tests: B3 parity classifier (lineage_b3_parity_check.py).

Verifies that ``classify_parity`` labels each class correctly on seeded data:
  - MATCH: legacy and reconstruction agree on content
  - RECON-MISSING: legacy row exists, no lineage event (real gap → exit non-zero)
  - LEGACY-MISSING: lineage event exists, no legacy row (tolerated)
"""

from __future__ import annotations

# Script under test lives outside the package; import via sys.path trick.
import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

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

classify_parity = _mod.classify_parity
ParityClass = _mod.ParityClass
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
# Unit-level: classify_parity labels correctly on hand-crafted inputs
# ---------------------------------------------------------------------------


def test_classify_match():
    """MATCH when both sides have identical added/removed content."""
    p_old = _make_profile("u1", "p-old", "old c")
    p_new = _make_profile("u1", "p-new", "new c")
    legacy = [_legacy_row("u1", "req-1", [p_new], [p_old])]
    recon = [_legacy_row("u1", "req-1", [p_new], [p_old])]
    results = classify_parity(legacy, recon)
    assert len(results) == 1
    assert results[0].classification == ParityClass.MATCH


def test_classify_recon_missing_no_lineage_event():
    """RECON-MISSING when legacy has a row but recon list is empty."""
    p_old = _make_profile("u1", "p-old", "old c")
    p_new = _make_profile("u1", "p-new", "new c")
    legacy = [_legacy_row("u1", "req-gap", [p_new], [p_old])]
    recon: list[ProfileChangeLog] = []
    results = classify_parity(legacy, recon)
    assert len(results) == 1
    assert results[0].classification == ParityClass.RECON_MISSING


def test_classify_recon_missing_content_mismatch():
    """RECON-MISSING when both sides exist but content disagrees."""
    p_old = _make_profile("u1", "p-old", "old c")
    p_new_legacy = _make_profile("u1", "p-new", "correct content")
    p_new_recon = _make_profile("u1", "p-new", "WRONG content")
    legacy = [_legacy_row("u1", "req-mismatch", [p_new_legacy], [p_old])]
    recon = [_legacy_row("u1", "req-mismatch", [p_new_recon], [p_old])]
    results = classify_parity(legacy, recon)
    assert len(results) == 1
    assert results[0].classification == ParityClass.RECON_MISSING


def test_classify_legacy_missing():
    """LEGACY-MISSING when recon has a row but legacy list is empty."""
    p_old = _make_profile("u1", "p-old", "old c")
    p_new = _make_profile("u1", "p-new", "new c")
    legacy: list[ProfileChangeLog] = []
    recon = [_legacy_row("u1", "req-orphan", [p_new], [p_old])]
    results = classify_parity(legacy, recon)
    assert len(results) == 1
    assert results[0].classification == ParityClass.LEGACY_MISSING


def test_classify_mixed():
    """Mixed: one MATCH, one RECON-MISSING, one LEGACY-MISSING."""
    p = _make_profile

    # MATCH
    l_match = _legacy_row(
        "u1", "req-match", [p("u1", "pa", "ca")], [p("u1", "pb", "cb")]
    )
    r_match = _legacy_row(
        "u1", "req-match", [p("u1", "pa", "ca")], [p("u1", "pb", "cb")]
    )

    # RECON-MISSING (legacy only)
    l_gap = _legacy_row("u1", "req-gap", [p("u1", "pc", "cc")], [])

    # LEGACY-MISSING (recon only)
    r_orphan = _legacy_row("u1", "req-orphan", [p("u1", "pd", "cd")], [])

    results = classify_parity([l_match, l_gap], [r_match, r_orphan])
    by_req = {r.request_id: r.classification for r in results}

    assert by_req["req-match"] == ParityClass.MATCH
    assert by_req["req-gap"] == ParityClass.RECON_MISSING
    assert by_req["req-orphan"] == ParityClass.LEGACY_MISSING


# ---------------------------------------------------------------------------
# Integration-level: run_parity_check against live SQLite storage
# ---------------------------------------------------------------------------


def test_run_parity_check_all_match(tmp_path):
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
    assert all(r.classification == ParityClass.MATCH for r in match_results), (
        f"expected all MATCH but got: {[(r.request_id, r.classification) for r in match_results]}"
    )


def test_run_parity_check_detects_recon_missing(tmp_path):
    """run_parity_check flags RECON-MISSING when legacy-only row exists."""
    s = _store(tmp_path)

    old_p = _make_profile("u1", "p-gap-old", "old")
    new_p = _make_profile("u1", "p-gap-new", "new")
    s.add_user_profile("u1", [old_p, new_p])
    # Only legacy row — no lineage event.
    s.add_profile_change_log(_legacy_row("u1", "req-gap-only", [new_p], [old_p]))

    results = run_parity_check(s)
    gaps = [r for r in results if r.classification == ParityClass.RECON_MISSING]
    assert gaps, "expected at least one RECON-MISSING"
    assert any(r.request_id == "req-gap-only" for r in gaps)


def test_run_parity_check_tolerates_legacy_missing(tmp_path):
    """run_parity_check does not fail for LEGACY-MISSING rows.

    Seeds a real dedup run (status_change+superseded event + profile with
    generated_from_request_id) WITHOUT writing a legacy row — this represents
    a run that used the new path but didn't dual-write to the legacy table.
    The reconstruction produces a row; the legacy table has none → LEGACY-MISSING.
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
    legacy_missing = [
        r for r in results if r.classification == ParityClass.LEGACY_MISSING
    ]
    assert legacy_missing, "expected at least one LEGACY-MISSING"
    assert any(r.request_id == req_id for r in legacy_missing)
    # No RECON-MISSING → exit code would be 0.
    gaps = [r for r in results if r.classification == ParityClass.RECON_MISSING]
    assert not gaps, "LEGACY-MISSING should not trigger a real gap"
