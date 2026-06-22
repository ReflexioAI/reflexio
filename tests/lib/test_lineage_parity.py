"""Unit tests for reflexio.lib._lineage_parity.

Covers:
- MATCH: both sides agree on added/removed.
- CONTENT_MISMATCH (in-both): rows exist on both sides but differ.
- CONTENT_MISMATCH (recon-only): reconstruction produced a run absent from legacy.
- RECON_MISSING: legacy-only and request_id IS in reconstructible_request_ids.
- LEGACY_MISSING: legacy-only and request_id NOT in reconstructible_request_ids.
- Tolerated updated delta: added/removed equal but updated_profiles differ → MATCH.
- Duplicate request_id on either side → INCONCLUSIVE (not raised).
- read_cap_hit=True → single INCONCLUSIVE result.
- Import identity: script uses the lib function, not a local redefinition.
"""

from __future__ import annotations

from reflexio.lib._lineage_parity import (
    ParityClass,
    ParityResult,
    classify_change_log_parity,
)
from reflexio.models.api_schema.domain.entities import ProfileChangeLog, UserProfile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _profile(profile_id: str, content: str, user_id: str = "u1") -> UserProfile:
    """Build a minimal UserProfile with only the fields used by the predicate."""
    return UserProfile(
        profile_id=profile_id,
        user_id=user_id,
        content=content,
        last_modified_timestamp=0,
        generated_from_request_id="",
    )


def _log(
    request_id: str,
    added: list[UserProfile] | None = None,
    removed: list[UserProfile] | None = None,
    updated: list[UserProfile] | None = None,
) -> ProfileChangeLog:
    """Build a minimal ProfileChangeLog row."""
    return ProfileChangeLog(
        id=0,
        user_id="u1",
        request_id=request_id,
        added_profiles=added or [],
        removed_profiles=removed or [],
        mentioned_profiles=updated or [],
    )


def _classify(
    legacy: list[ProfileChangeLog],
    recon: list[ProfileChangeLog],
    *,
    reconstructible: set[str] | None = None,
    read_cap_hit: bool = False,
) -> list[ParityResult]:
    """Convenience wrapper with a safe default for reconstructible_request_ids."""
    return classify_change_log_parity(
        legacy,
        recon,
        reconstructible_request_ids=reconstructible or set(),
        read_cap_hit=read_cap_hit,
    )


# ---------------------------------------------------------------------------
# MATCH
# ---------------------------------------------------------------------------


def test_match_identical_added_removed() -> None:
    p1 = _profile("p1", "hello")
    legacy = [_log("req1", added=[p1])]
    recon = [_log("req1", added=[p1])]

    results = _classify(legacy, recon)

    assert len(results) == 1
    assert results[0].classification == ParityClass.MATCH
    assert results[0].request_id == "req1"


def test_match_empty_rows() -> None:
    """Both sides have the same empty added/removed → MATCH."""
    legacy = [_log("req1")]
    recon = [_log("req1")]

    results = _classify(legacy, recon)

    assert len(results) == 1
    assert results[0].classification == ParityClass.MATCH


def test_match_tolerated_updated_delta() -> None:
    """added/removed equal but mentioned_profiles differ → still MATCH.

    ProfileChangeLog.mentioned_profiles is used as the updated field in
    legacy; _rows_match deliberately ignores it.
    """
    p1 = _profile("p1", "content-a")
    p2 = _profile("p2", "content-b")

    legacy = [_log("req1", added=[p1], removed=[], updated=[p2])]
    recon = [_log("req1", added=[p1], removed=[], updated=[])]

    results = _classify(legacy, recon)

    assert len(results) == 1
    assert results[0].classification == ParityClass.MATCH


# ---------------------------------------------------------------------------
# CONTENT_MISMATCH (both sides have the request_id but content differs)
# ---------------------------------------------------------------------------


def test_content_mismatch_different_added() -> None:
    legacy = [_log("req1", added=[_profile("p1", "old")])]
    recon = [_log("req1", added=[_profile("p1", "new")])]

    results = _classify(legacy, recon)

    assert len(results) == 1
    r = results[0]
    assert r.classification == ParityClass.CONTENT_MISMATCH
    assert r.request_id == "req1"
    assert "differ" in r.detail


def test_content_mismatch_different_removed() -> None:
    p1 = _profile("p1", "same-added")
    legacy = [_log("req1", added=[p1], removed=[_profile("r1", "old-removed")])]
    recon = [_log("req1", added=[p1], removed=[_profile("r1", "new-removed")])]

    results = _classify(legacy, recon)

    assert len(results) == 1
    assert results[0].classification == ParityClass.CONTENT_MISMATCH


# ---------------------------------------------------------------------------
# CONTENT_MISMATCH (recon-only)
# ---------------------------------------------------------------------------


def test_content_mismatch_recon_only() -> None:
    """A run that exists only in reconstruction → CONTENT_MISMATCH."""
    recon = [_log("req-recon-only", added=[_profile("p1", "extra")])]

    results = _classify([], recon)

    assert len(results) == 1
    r = results[0]
    assert r.classification == ParityClass.CONTENT_MISMATCH
    assert r.request_id == "req-recon-only"
    assert "absent from legacy" in r.detail


# ---------------------------------------------------------------------------
# RECON_MISSING (legacy-only, id IN reconstructible)
# ---------------------------------------------------------------------------


def test_recon_missing_when_reconstructible_signal_exists() -> None:
    legacy = [_log("req-gap", added=[_profile("p1", "content")])]

    results = _classify(legacy, [], reconstructible={"req-gap"})

    assert len(results) == 1
    r = results[0]
    assert r.classification == ParityClass.RECON_MISSING
    assert r.request_id == "req-gap"
    assert "reconstructible" in r.detail


# ---------------------------------------------------------------------------
# LEGACY_MISSING (legacy-only, id NOT in reconstructible — tolerated)
# ---------------------------------------------------------------------------


def test_legacy_missing_when_no_reconstructible_signal() -> None:
    legacy = [_log("req-old", added=[_profile("p1", "content")])]

    # reconstructible is empty — no signal for this id
    results = _classify(legacy, [], reconstructible=set())

    assert len(results) == 1
    r = results[0]
    assert r.classification == ParityClass.LEGACY_MISSING
    assert r.request_id == "req-old"
    assert "tolerated" in r.detail


# ---------------------------------------------------------------------------
# Duplicate request_id → INCONCLUSIVE (must NOT raise)
# ---------------------------------------------------------------------------


def test_duplicate_legacy_request_id_inconclusive_no_raise() -> None:
    p1 = _profile("p1", "c1")
    p2 = _profile("p2", "c2")
    legacy = [_log("req-dup", added=[p1]), _log("req-dup", added=[p2])]

    results = _classify(legacy, [])

    dup_results = [r for r in results if r.request_id == "req-dup"]
    assert len(dup_results) == 1
    assert dup_results[0].classification == ParityClass.INCONCLUSIVE
    assert "duplicate" in dup_results[0].detail


def test_duplicate_recon_request_id_inconclusive_no_raise() -> None:
    p1 = _profile("p1", "c1")
    p2 = _profile("p2", "c2")
    recon = [_log("req-dup", added=[p1]), _log("req-dup", added=[p2])]

    results = _classify([], recon)

    dup_results = [r for r in results if r.request_id == "req-dup"]
    assert len(dup_results) == 1
    assert dup_results[0].classification == ParityClass.INCONCLUSIVE


def test_duplicate_excluded_from_other_classification() -> None:
    """Duplicate ids must not also appear as MATCH/RECON_MISSING/etc."""
    p1 = _profile("p1", "c1")
    legacy = [
        _log("req-dup", added=[p1]),
        _log("req-dup", added=[p1]),  # duplicate
        _log("req-ok", added=[p1]),
    ]
    recon = [_log("req-ok", added=[p1])]

    results = _classify(legacy, recon, reconstructible={"req-dup"})

    classes = {r.request_id: r.classification for r in results}
    assert classes["req-dup"] == ParityClass.INCONCLUSIVE
    assert classes["req-ok"] == ParityClass.MATCH
    # req-dup must not also appear as RECON_MISSING
    assert sum(1 for r in results if r.request_id == "req-dup") == 1


# ---------------------------------------------------------------------------
# read_cap_hit → single INCONCLUSIVE
# ---------------------------------------------------------------------------


def test_read_cap_hit_returns_single_inconclusive() -> None:
    legacy = [_log("req1", added=[_profile("p1", "c")])]
    recon = [_log("req2", added=[_profile("p2", "c")])]

    results = _classify(legacy, recon, read_cap_hit=True)

    assert len(results) == 1
    r = results[0]
    assert r.classification == ParityClass.INCONCLUSIVE
    assert r.request_id == "*"
    assert "cap" in r.detail


def test_read_cap_hit_ignores_data() -> None:
    """Even with matching data, read_cap_hit=True → INCONCLUSIVE only."""
    p1 = _profile("p1", "c")
    legacy = [_log("req1", added=[p1])]
    recon = [_log("req1", added=[p1])]

    results = _classify(legacy, recon, read_cap_hit=True)

    assert len(results) == 1
    assert results[0].classification == ParityClass.INCONCLUSIVE


# ---------------------------------------------------------------------------
# Multiple request_ids sorted deterministically
# ---------------------------------------------------------------------------


def test_multiple_request_ids_sorted() -> None:
    p1 = _profile("p1", "c")
    legacy = [
        _log("req-b", added=[p1]),
        _log("req-a", added=[p1]),
    ]
    recon = [
        _log("req-b", added=[p1]),
        _log("req-a", added=[p1]),
    ]

    results = _classify(legacy, recon)

    assert [r.request_id for r in results] == ["req-a", "req-b"]
    assert all(r.classification == ParityClass.MATCH for r in results)


# ---------------------------------------------------------------------------
# Import-identity test: script delegates to the lib function
# ---------------------------------------------------------------------------


def test_script_uses_lib_classify_change_log_parity() -> None:
    """Verify the parity script re-exports classify_change_log_parity from the lib.

    Importing the script module must expose the same function object as the lib,
    confirming there is no local redefinition.
    """

    # The script does sys.path.insert at import time; that's fine for this check.
    # We use importlib to load it as a module without executing __main__.
    import importlib.util
    from pathlib import Path

    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "lineage_b3_parity_check.py"
    )
    spec = importlib.util.spec_from_file_location(
        "lineage_b3_parity_check", script_path
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    # The script must import classify_change_log_parity from the lib — same object.
    assert hasattr(module, "classify_change_log_parity"), (
        "Script does not expose classify_change_log_parity"
    )
    assert module.classify_change_log_parity is classify_change_log_parity, (
        "Script's classify_change_log_parity is not the lib's function — "
        "a local redefinition was found"
    )
