"""Integration tests for the lineage dual-read shim's observability hardening.

Covers two slices of the read-permission hardening work:

  B2 — the shim's exception path emits ``capture_anomaly(..., level="error")``
       (not ``"warning"``) so a genuine "the shim is broken" failure reaches the
       Discord ``environment:production AND level:error`` rule.

  B4 — the false-clean guard: a run that could NOT actually reconstruct anything
       (empty reconstructed change log + empty reconstructible signal set) while
       the legacy change log is non-empty AND has remove-bearing rows must NOT be
       reported as ``outcome="match"``.  It is reported as ``"degraded"`` and a
       distinct ``lineage.reconstruct.degraded`` (level="error") anomaly fires.
       Control: a legitimately add-only org with genuinely no removals still
       reaches a true ``"match"`` (no false ``degraded``).

All tests use real SQLite storage in isolated temp dirs (mocked LLM via the
project-level conftest).  Patterns mirror
``tests/server/test_lineage_dual_read_shim_integration.py``.
"""

from __future__ import annotations

import types
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain.entities import (
    ProfileChangeLog,
    ProfileChangeLogResponse,
    UserProfile,
)
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive
from reflexio.server import usage_metrics as usage_metrics_module
from reflexio.server.lineage_parity_shim import dual_read_diff
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.usage_metrics import (
    UsageEvent,
    configure_usage_event_recorder,
)

pytestmark = pytest.mark.integration

_FLAG_ENABLED_CONFIG: dict[str, Any] = {
    "lineage_dual_read_diff": {"enabled": True, "enabled_org_ids": []},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_storage(tmp_path, worker_id: str) -> SQLiteStorage:
    s = SQLiteStorage(
        org_id=f"false-clean-{worker_id}-{tmp_path.name}",
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
    request_context = types.SimpleNamespace(storage=storage)
    return types.SimpleNamespace(request_context=request_context)


class _CapturingRecorder:
    def __init__(self) -> None:
        self.events: list[UsageEvent] = []

    def __call__(self, event: UsageEvent) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def storage(tmp_path, worker_id):
    return _make_storage(tmp_path, worker_id)


@pytest.fixture
def reflexio_stub(storage):
    return _make_reflexio_stub(storage)


@pytest.fixture
def capturing_recorder():
    # Save/restore the process-global recorder rather than resetting to None, so
    # this fixture can't clobber a recorder configured by another test/fixture
    # (order-dependent failures otherwise).
    previous_recorder = usage_metrics_module._recorder
    recorder = _CapturingRecorder()
    configure_usage_event_recorder(recorder)
    yield recorder
    configure_usage_event_recorder(previous_recorder)


# ---------------------------------------------------------------------------
# B2 — exception path emits level="error"
# ---------------------------------------------------------------------------


def test_error_path_emits_error_level(
    storage: SQLiteStorage,
    reflexio_stub: object,
    capturing_recorder: _CapturingRecorder,
    worker_id: str,
) -> None:
    """B2: a failed reconstruction emits lineage.reconstruct.error at level='error'.

    Forces the storage read inside the shim to raise, then asserts the swallowed
    anomaly is emitted with ``level="error"`` (so it reaches the Discord
    production rule), not the old ``level="warning"``.
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
        result = dual_read_diff(reflexio_stub, org_id)

    assert result is None

    error_calls = [
        (msg, kw) for msg, kw in anomalies if msg == "lineage.reconstruct.error"
    ]
    assert error_calls, "expected a lineage.reconstruct.error anomaly"
    assert all(kw.get("level") == "error" for _, kw in error_calls), (
        f"expected level='error' on every error anomaly; got: {error_calls}"
    )


# ---------------------------------------------------------------------------
# B4 — false-clean guard: degenerate reconstruction is NOT reported as match
# ---------------------------------------------------------------------------


def test_degenerate_reconstruction_reports_degraded_not_match(
    storage: SQLiteStorage,
    reflexio_stub: object,
    capturing_recorder: _CapturingRecorder,
    worker_id: str,
) -> None:
    """B4: empty reconstruction + remove-bearing legacy → outcome='degraded'.

    Simulates the production failure mode: reconstruction reads succeed but
    return an EMPTY signal set (e.g. ``get_lineage_events`` returns [] on an
    anon-keyed ref) while the legacy change log retains remove-bearing rows.
    Without the guard, every legacy row classifies as LEGACY_MISSING (tolerated)
    → zero divergences → a FALSE ``outcome="match"``.

    Asserts the coverage event reports ``outcome="degraded"`` (NOT ``"match"``)
    and a distinct ``lineage.reconstruct.degraded`` (level="error") anomaly fires.
    """
    org_id = storage.org_id

    # Seed a real, remove-bearing legacy change log row.
    old_p = _make_profile("u1", "p-removed", "removed-content", request_id="req-rm")
    new_p = _make_profile("u1", "p-kept", "kept-content", request_id="req-rm")
    storage.add_user_profile("u1", [old_p, new_p])
    storage.add_profile_change_log(_make_change_log("u1", "req-rm", [new_p], [old_p]))

    anomalies: list[tuple[str, dict[str, Any]]] = []

    def _capture(message: str, **kwargs: Any) -> None:
        anomalies.append((message, kwargs))

    # Reconstruction reads succeed but yield NOTHING (the degenerate case): both
    # the reconstructed change log and the reconstructible signal set are empty.
    def _empty_reconstruct(*_args: Any, **_kwargs: Any) -> ProfileChangeLogResponse:
        return ProfileChangeLogResponse(success=True, profile_change_logs=[])

    with (
        patch(
            "reflexio.server.site_var.feature_flags._get_feature_flags_config",
            return_value=_FLAG_ENABLED_CONFIG,
        ),
        patch(
            "reflexio.server.lineage_parity_shim.reconstruct_profile_change_log",
            side_effect=_empty_reconstruct,
        ),
        patch(
            "reflexio.server.lineage_parity_shim.profile_reconstructible_request_ids",
            return_value=set(),
        ),
        patch(
            "reflexio.server.lineage_parity_shim.capture_anomaly",
            side_effect=_capture,
        ),
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
    assert evt.outcome == "degraded", (
        f"expected outcome='degraded' for degenerate reconstruction; got {evt.outcome!r}"
    )
    assert evt.outcome != "match", "a degenerate run must NOT report a false 'match'"
    assert evt.metadata.get("degraded") is True

    degraded_calls = [
        (msg, kw) for msg, kw in anomalies if msg == "lineage.reconstruct.degraded"
    ]
    assert degraded_calls, "expected a lineage.reconstruct.degraded anomaly"
    assert all(kw.get("level") == "error" for _, kw in degraded_calls), (
        f"expected level='error' on degraded anomaly; got: {degraded_calls}"
    )


def test_add_only_org_reaches_true_match_no_false_degraded(
    storage: SQLiteStorage,
    reflexio_stub: object,
    capturing_recorder: _CapturingRecorder,
    worker_id: str,
) -> None:
    """B4 control: a genuinely add-only org with no removals still reports 'match'.

    A legitimately add-only org has no remove-bearing legacy rows, so the
    false-clean guard must NOT fire — even though reconstruction produces no
    removals.  Asserts ``outcome="match"`` and that NO degraded anomaly is emitted.
    """
    org_id = storage.org_id

    # Add-only run: a new profile, legacy row with no removals, reconstructs MATCH.
    new_add = _make_profile("u1", "p-add", "add-content", request_id="req-add")
    storage.add_user_profile("u1", [new_add])
    storage.add_profile_change_log(_make_change_log("u1", "req-add", [new_add], []))

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

    coverage_events = [
        e
        for e in capturing_recorder.events
        if e.event_name == "lineage.reconstruct.coverage"
    ]
    assert len(coverage_events) == 1, (
        f"expected exactly one coverage event; got {len(coverage_events)}"
    )
    evt = coverage_events[0]
    assert evt.outcome == "match", (
        f"expected a true 'match' for an add-only org; got {evt.outcome!r}"
    )
    assert evt.metadata.get("degraded") is False
    assert evt.metadata["add_only_runs"] >= 1
    assert evt.metadata["remove_bearing_runs"] == 0

    degraded_calls = [
        (msg, kw) for msg, kw in anomalies if msg == "lineage.reconstruct.degraded"
    ]
    assert not degraded_calls, (
        f"add-only org must NOT trigger a degraded anomaly; got: {degraded_calls}"
    )
