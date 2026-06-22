"""Integration tests for the lineage dual-read divergence shim.

Covers:
  (a) divergence emitted on a real RECON-MISSING mismatch.
  (b) no divergence on tolerated deltas (MATCH, LEGACY_MISSING).
  (c) coverage usage event recorded with correct metadata.
  (d) non-disruption — dual_read_diff returns None and does not alter the legacy read.
  (e) error path — exception in reconstruct is swallowed; lineage.reconstruct.error emitted.

All tests use real SQLite storage in isolated temp dirs (mocked LLM via the
project-level conftest). The feature flag is enabled via the standard
``_get_feature_flags_config`` patch mechanism (see
``tests/server/site_var/test_feature_flags.py``).
"""

from __future__ import annotations

import types
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain.entities import ProfileChangeLog, UserProfile
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive
from reflexio.server.lineage_parity_shim import dual_read_diff
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.usage_metrics import (
    UsageEvent,
    configure_usage_event_recorder,
)

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Feature-flag config that enables the dual-read diff for all orgs.
# ---------------------------------------------------------------------------

_FLAG_ENABLED_CONFIG: dict[str, Any] = {
    "lineage_dual_read_diff": {"enabled": True, "enabled_org_ids": []},
}

_FLAG_DISABLED_CONFIG: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_storage(tmp_path, worker_id: str) -> SQLiteStorage:
    """Create and migrate a fresh SQLiteStorage for one test."""
    s = SQLiteStorage(
        org_id=f"shim-test-{worker_id}-{tmp_path.name}",
        db_path=str(tmp_path / "test.db"),
    )
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


def _make_change_log(
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


def _make_reflexio_stub(storage: SQLiteStorage) -> object:
    """Minimal stub that duck-types ``reflexio.request_context.storage``."""
    request_context = types.SimpleNamespace(storage=storage)
    return types.SimpleNamespace(request_context=request_context)


class _CapturingRecorder:
    """Simple list-accumulating usage event recorder."""

    def __init__(self) -> None:
        self.events: list[UsageEvent] = []

    def __call__(self, event: UsageEvent) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def storage(tmp_path, worker_id):
    """Real SQLite storage, isolated per test."""
    return _make_storage(tmp_path, worker_id)


@pytest.fixture
def reflexio_stub(storage):
    """Minimal object that exposes .request_context.storage."""
    return _make_reflexio_stub(storage)


@pytest.fixture
def capturing_recorder():
    """Install a capturing usage recorder; reset to None after the test."""
    recorder = _CapturingRecorder()
    configure_usage_event_recorder(recorder)
    yield recorder
    configure_usage_event_recorder(None)


# ---------------------------------------------------------------------------
# (a) Divergence emitted on a real RECON-MISSING mismatch
# ---------------------------------------------------------------------------


def test_divergence_emitted_for_recon_missing(
    storage: SQLiteStorage,
    reflexio_stub: object,
    capturing_recorder: _CapturingRecorder,
    worker_id: str,
) -> None:
    """(a) A RECON-MISSING run produces a lineage.reconstruct.divergence anomaly.

    Seeding strategy (mirrors test_run_parity_check_detects_recon_missing):
      1. Add an old profile, then supersede it with request_id="req-missing".
         This emits a status_change+superseded lineage event → puts the id in
         the reconstructible set.
      2. Hard-delete the tombstone row so reconstruction can't fetch it →
         added=[], removed=[] → reconstruction drops the run entirely.
      3. Write a legacy row for "req-missing" → RECON-MISSING.
    """
    org_id = storage.org_id

    old_p = _make_profile("u1", "p-gap", "gap-content", request_id="seed")
    storage.add_user_profile("u1", [old_p])
    storage.supersede_profiles_by_ids("u1", ["p-gap"], "req-missing")
    # Hard-delete the tombstone so reconstruction yields nothing.
    storage.conn.execute("DELETE FROM profiles WHERE profile_id = ?", ("p-gap",))
    storage.conn.commit()

    # Legacy table retains the row.
    storage.add_profile_change_log(_make_change_log("u1", "req-missing", [], [old_p]))

    anomalies: list[tuple[str, dict[str, Any]]] = []

    def _capture(message: str, **kwargs: Any) -> None:
        anomalies.append((message, kwargs))

    with (
        patch(
            "reflexio.server.site_var.feature_flags._get_feature_flags_config",
            return_value=_FLAG_ENABLED_CONFIG,
        ),
        patch(
            "reflexio.server.lineage_parity_shim.capture_anomaly",
            side_effect=_capture,
        ),
    ):
        dual_read_diff(reflexio_stub, org_id)

    divergence_calls = [
        (msg, kw) for msg, kw in anomalies if msg == "lineage.reconstruct.divergence"
    ]
    assert divergence_calls, (
        "expected at least one lineage.reconstruct.divergence anomaly"
    )
    assert any(kw.get("kind") == "RECON-MISSING" for _, kw in divergence_calls), (
        f"expected kind=RECON-MISSING in divergence anomalies, got: {divergence_calls}"
    )


# ---------------------------------------------------------------------------
# (b) No divergence on tolerated deltas
# ---------------------------------------------------------------------------


def test_no_divergence_on_matching_run(
    storage: SQLiteStorage,
    reflexio_stub: object,
    capturing_recorder: _CapturingRecorder,
    worker_id: str,
) -> None:
    """(b) A clean dedup run that reconstructs to MATCH emits zero divergences.

    Also seeds a LEGACY_MISSING run (legacy row exists but has no reconstructible
    signal — the legacy row's content never received a generated_from_request_id
    stamp and no supersede event was recorded for that request_id) — also
    tolerated: no divergence anomaly.

    Seeding note: ``_make_profile`` sets ``generated_from_request_id`` to the
    given ``request_id``.  The "seed" profile for the old-profile slot must use
    the SAME request_id that also has a legacy row, otherwise reconstruction
    sees a phantom run for "seed-*" with no matching legacy row → CONTENT_MISMATCH.
    Here we use a single dedup run where both the old profile and the new profile
    have their generated_from_request_id set to the same run's request_id (which
    is fine because the supersede event is what puts the run in the reconstructible
    set, and reconstruction uses generated_from_request_id for add and the
    superseded event for remove).
    """
    org_id = storage.org_id

    # --- MATCH run ---
    # old_match uses generated_from_request_id="req-match-seed" — we also add a
    # legacy row for "req-match-seed" so it doesn't become a phantom CONTENT_MISMATCH.
    # The actual dedup run is "req-match".
    old_match = _make_profile(
        "u1", "p-match-old", "old-match", request_id="req-match-seed"
    )
    new_match = _make_profile("u1", "p-match-new", "new-match", request_id="req-match")
    storage.add_user_profile("u1", [old_match])
    storage.add_user_profile("u1", [new_match])
    storage.supersede_profiles_by_ids("u1", ["p-match-old"], "req-match")
    # Legacy row for the dedup run.
    storage.add_profile_change_log(
        _make_change_log("u1", "req-match", [new_match], [old_match])
    )
    # Legacy row for the seed profile's request_id (no-op: add-only, matches recon).
    storage.add_profile_change_log(
        _make_change_log("u1", "req-match-seed", [old_match], [])
    )

    # --- LEGACY_MISSING run (no reconstructible signal; tolerated) ---
    # Write a legacy row for a request_id that has zero reconstructible signals
    # (no profile with that generated_from_request_id, no supersede event for it).
    legacy_only_p = _make_profile(
        "u1", "p-legacy-only", "legacy-content", request_id="req-legacy-only-seed"
    )
    # Persist the profile so the legacy change log row is valid, then write the
    # legacy row with a DIFFERENT request_id that has no reconstructible signal.
    storage.add_user_profile("u1", [legacy_only_p])
    storage.add_profile_change_log(
        _make_change_log("u1", "req-no-signal", [legacy_only_p], [])
    )
    # Also add a legacy row for "req-legacy-only-seed" to cover the profile stamp.
    storage.add_profile_change_log(
        _make_change_log("u1", "req-legacy-only-seed", [legacy_only_p], [])
    )
    # Neither req-no-signal nor req-legacy-only-seed has a supersede event,
    # but req-legacy-only-seed DOES have a profile stamp so reconstruction will
    # produce it → we wrote the legacy row above for it.
    # req-no-signal has no stamp and no supersede event → LEGACY_MISSING (tolerated).

    anomalies: list[tuple[str, dict[str, Any]]] = []

    def _capture(message: str, **kwargs: Any) -> None:
        anomalies.append((message, kwargs))

    with (
        patch(
            "reflexio.server.site_var.feature_flags._get_feature_flags_config",
            return_value=_FLAG_ENABLED_CONFIG,
        ),
        patch(
            "reflexio.server.lineage_parity_shim.capture_anomaly",
            side_effect=_capture,
        ),
    ):
        dual_read_diff(reflexio_stub, org_id)

    divergence_calls = [
        (msg, kw) for msg, kw in anomalies if msg == "lineage.reconstruct.divergence"
    ]
    assert not divergence_calls, (
        f"expected zero divergence anomalies for MATCH+LEGACY_MISSING data; got: {divergence_calls}"
    )


# ---------------------------------------------------------------------------
# (c) Coverage usage event recorded
# ---------------------------------------------------------------------------


def test_coverage_usage_event_recorded(
    storage: SQLiteStorage,
    reflexio_stub: object,
    capturing_recorder: _CapturingRecorder,
    worker_id: str,
) -> None:
    """(c) Exactly one lineage.reconstruct.coverage event is emitted.

    Seeds three runs, all of which produce MATCH:
      - "req-add-only": new profile (add-only), no removed_profiles in legacy row.
      - "seed-rem": seed profile for the remove-bearing run; reconstruction
        produces an add-only row for it; we add a matching legacy row.
      - "req-remove-bearing": the actual dedup run; legacy row has removed_profiles.

    Asserts: event_category="lineage", add_only_runs=2, remove_bearing_runs=1,
    outcome="match" (zero divergences).
    """
    org_id = storage.org_id

    # --- add-only MATCH run ---
    new_add = _make_profile("u1", "p-add-new", "add-new", request_id="req-add-only")
    storage.add_user_profile("u1", [new_add])
    # No superseded profile → removed=[] in reconstruction.
    storage.add_profile_change_log(
        _make_change_log("u1", "req-add-only", [new_add], [])
    )

    # --- remove-bearing MATCH run ---
    old_rem = _make_profile("u1", "p-rem-old", "old-rem", request_id="seed-rem")
    new_rem = _make_profile(
        "u1", "p-rem-new", "new-rem", request_id="req-remove-bearing"
    )
    storage.add_user_profile("u1", [old_rem])
    storage.add_user_profile("u1", [new_rem])
    storage.supersede_profiles_by_ids("u1", ["p-rem-old"], "req-remove-bearing")
    storage.add_profile_change_log(
        _make_change_log("u1", "req-remove-bearing", [new_rem], [old_rem])
    )
    # Add legacy row for "seed-rem" so reconstruction (which produces an add-only
    # row for old_rem's generated_from_request_id) finds a matching legacy entry
    # → MATCH instead of CONTENT_MISMATCH.
    storage.add_profile_change_log(_make_change_log("u1", "seed-rem", [old_rem], []))

    with patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value=_FLAG_ENABLED_CONFIG,
    ):
        dual_read_diff(reflexio_stub, org_id)

    coverage_events = [
        e
        for e in capturing_recorder.events
        if e.event_name == "lineage.reconstruct.coverage"
    ]
    assert len(coverage_events) == 1, (
        f"expected exactly one coverage event; got {len(coverage_events)}"
    )
    evt = coverage_events[0]
    assert evt.event_category == "lineage"

    meta = evt.metadata
    assert isinstance(meta.get("add_only_runs"), int), (
        f"add_only_runs must be int; got {type(meta.get('add_only_runs'))}"
    )
    assert isinstance(meta.get("remove_bearing_runs"), int), (
        f"remove_bearing_runs must be int; got {type(meta.get('remove_bearing_runs'))}"
    )
    # req-add-only and seed-rem both produce add-only MATCH rows (legacy removed=[]).
    assert meta["add_only_runs"] == 2, (
        f"expected add_only_runs=2; got {meta['add_only_runs']}"
    )
    assert meta["remove_bearing_runs"] == 1, (
        f"expected remove_bearing_runs=1; got {meta['remove_bearing_runs']}"
    )
    assert evt.outcome == "match", (
        f"expected outcome='match' (zero divergences); got {evt.outcome!r}"
    )


# ---------------------------------------------------------------------------
# (d) Non-disruption
# ---------------------------------------------------------------------------


def test_non_disruption_returns_none_and_does_not_mutate_legacy(
    storage: SQLiteStorage,
    reflexio_stub: object,
    capturing_recorder: _CapturingRecorder,
    worker_id: str,
) -> None:
    """(d) dual_read_diff returns None; legacy read is identical before and after.

    Also asserts that calling with the flag OFF produces the same legacy result
    (flag gate has zero side-effects).
    """
    org_id = storage.org_id

    old_p = _make_profile("u1", "p-nd-old", "nd-old", request_id="seed-nd")
    new_p = _make_profile("u1", "p-nd-new", "nd-new", request_id="req-nd")
    storage.add_user_profile("u1", [old_p])
    storage.add_user_profile("u1", [new_p])
    storage.supersede_profiles_by_ids("u1", ["p-nd-old"], "req-nd")
    storage.add_profile_change_log(_make_change_log("u1", "req-nd", [new_p], [old_p]))

    legacy_before = storage.get_profile_change_logs()

    # With flag ON.
    with patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value=_FLAG_ENABLED_CONFIG,
    ):
        result_on = dual_read_diff(reflexio_stub, org_id)

    legacy_after_on = storage.get_profile_change_logs()

    # With flag OFF.
    capturing_recorder.events.clear()
    with patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value=_FLAG_DISABLED_CONFIG,
    ):
        result_off = dual_read_diff(reflexio_stub, org_id)

    # A disabled flag must emit ZERO usage events (no side-effects at all).
    assert not capturing_recorder.events, (
        f"expected no usage events when flag is OFF; got: {capturing_recorder.events}"
    )

    legacy_after_off = storage.get_profile_change_logs()

    assert result_on is None, (
        f"expected None from dual_read_diff (flag ON); got {result_on!r}"
    )
    assert result_off is None, (
        f"expected None from dual_read_diff (flag OFF); got {result_off!r}"
    )

    # Legacy read must be unchanged by the shim.
    def _key(row: ProfileChangeLog) -> str:
        return row.request_id

    assert sorted(legacy_before, key=_key) == sorted(legacy_after_on, key=_key), (
        "legacy rows changed after dual_read_diff (flag ON)"
    )
    assert sorted(legacy_before, key=_key) == sorted(legacy_after_off, key=_key), (
        "legacy rows changed after dual_read_diff (flag OFF)"
    )


# ---------------------------------------------------------------------------
# (e) Error path
# ---------------------------------------------------------------------------


def test_error_path_swallowed_and_emits_anomaly(
    storage: SQLiteStorage,
    reflexio_stub: object,
    capturing_recorder: _CapturingRecorder,
    worker_id: str,
) -> None:
    """(e) Exception inside reconstruct is swallowed; lineage.reconstruct.error emitted.

    Monkeypatches reconstruct_profile_change_log in the shim's module namespace
    to raise a ValueError. Asserts:
      - dual_read_diff does not raise,
      - a lineage.reconstruct.error anomaly is captured with error="ValueError",
      - no lineage.reconstruct.divergence anomaly is emitted.
    """
    org_id = storage.org_id

    anomalies: list[tuple[str, dict[str, Any]]] = []

    def _capture(message: str, **kwargs: Any) -> None:
        anomalies.append((message, kwargs))

    def _exploding_reconstruct(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("simulated reconstruction failure")

    with (
        patch(
            "reflexio.server.site_var.feature_flags._get_feature_flags_config",
            return_value=_FLAG_ENABLED_CONFIG,
        ),
        patch(
            "reflexio.server.lineage_parity_shim.reconstruct_profile_change_log",
            side_effect=_exploding_reconstruct,
        ),
        patch(
            "reflexio.server.lineage_parity_shim.capture_anomaly",
            side_effect=_capture,
        ),
    ):
        # Must not raise.
        result = dual_read_diff(reflexio_stub, org_id)

    assert result is None, f"expected None; got {result!r}"

    error_calls = [
        (msg, kw) for msg, kw in anomalies if msg == "lineage.reconstruct.error"
    ]
    assert error_calls, "expected at least one lineage.reconstruct.error anomaly"
    assert any(kw.get("error") == "ValueError" for _, kw in error_calls), (
        f"expected error='ValueError' in anomaly kwargs; got: {error_calls}"
    )

    divergence_calls = [
        (msg, kw) for msg, kw in anomalies if msg == "lineage.reconstruct.divergence"
    ]
    assert not divergence_calls, (
        f"no divergence anomaly expected on error path; got: {divergence_calls}"
    )
