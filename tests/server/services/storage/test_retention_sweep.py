"""``sweep_retention_caps`` probes, warns, deletes, and never escapes.

These tests are the enterprise ``TestCleanupStorageTables`` class retargeted at
the new module: the behaviour being asserted (lock, per-target isolation, delete
arithmetic) did not change when the caller moved, so the assertions did not
either. What is new is the headroom half -- ``retention.cap.approaching`` and
``retention.cap.enforced`` -- which had no equivalent on the request path.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from reflexio.server.services.storage import retention_sweep
from reflexio.server.services.storage.retention_sweep import sweep_retention_caps

_ORG = "org-1"


@pytest.fixture
def storage() -> MagicMock:
    return MagicMock()


@pytest.fixture
def granted_lock():
    """``OperationStateManager`` patched to always grant the lease."""
    with patch.object(retention_sweep, "OperationStateManager") as cls:
        cls.return_value.acquire_simple_lock.return_value = True
        yield cls


@pytest.fixture
def anomalies():
    """Capture ``capture_anomaly`` calls as ``(name, tags)`` pairs."""
    seen: list[tuple[str, dict]] = []
    with patch.object(
        retention_sweep,
        "capture_anomaly",
        lambda message, **tags: seen.append((message, tags)),
    ):
        yield seen


def _limits(**kwargs: int):
    return patch.object(
        retention_sweep, "get_row_retention_limits", return_value=dict(kwargs)
    )


def test_a_disabled_limit_issues_no_probe(storage, granted_lock, anomalies):
    """Review Focus 2: ``REFLEXIO_ROW_LIMIT_X=0`` disables the cap entirely."""
    with _limits(interactions=0):
        assert sweep_retention_caps(_ORG, storage) == 0

    storage.count_retention_target_rows.assert_not_called()
    assert anomalies == []


def test_a_count_below_the_limit_deletes_nothing(storage, granted_lock, anomalies):
    storage.count_retention_target_rows.return_value = 100

    with _limits(interactions=500):
        assert sweep_retention_caps(_ORG, storage) == 0

    storage.delete_oldest_retention_target_rows.assert_not_called()
    assert anomalies == []


def test_a_count_equal_to_the_limit_deletes(storage, granted_lock, anomalies):
    """Review Focus 3: the boundary is ``count < limit`` returns, so == deletes."""
    storage.count_retention_target_rows.return_value = 500
    storage.delete_oldest_retention_target_rows.return_value = 100

    with _limits(interactions=500):
        assert sweep_retention_caps(_ORG, storage) == 100

    storage.delete_oldest_retention_target_rows.assert_called_once_with(
        "interactions", 100
    )
    assert [name for name, _ in anomalies] == ["retention.cap.enforced"]


def test_the_warn_boundary_is_inclusive(storage, granted_lock, anomalies):
    """Review Focus 5: rows == 0.90 * limit warns rather than staying silent."""
    storage.count_retention_target_rows.return_value = 450

    with _limits(interactions=500):
        assert sweep_retention_caps(_ORG, storage) == 0

    storage.delete_oldest_retention_target_rows.assert_not_called()
    assert [name for name, _ in anomalies] == ["retention.cap.approaching"]
    _, tags = anomalies[0]
    assert tags["rows"] == 450
    assert tags["limit"] == 500
    assert tags["target"] == "interactions"


def test_just_below_the_warn_boundary_is_silent(storage, granted_lock, anomalies):
    storage.count_retention_target_rows.return_value = 449

    with _limits(interactions=500):
        sweep_retention_caps(_ORG, storage)

    assert anomalies == []


def test_an_enforced_delete_reports_what_it_removed(storage, granted_lock, anomalies):
    storage.count_retention_target_rows.return_value = 600
    storage.delete_oldest_retention_target_rows.return_value = 120

    with _limits(interactions=500):
        assert sweep_retention_caps(_ORG, storage) == 120

    assert [name for name, _ in anomalies] == ["retention.cap.enforced"]
    _, tags = anomalies[0]
    assert tags["rows_before"] == 600
    assert tags["deleted"] == 120
    assert tags["limit"] == 500


def test_a_refused_lease_probes_nothing(storage, anomalies):
    with patch.object(retention_sweep, "OperationStateManager") as cls:
        cls.return_value.acquire_simple_lock.return_value = False
        with _limits(interactions=1):
            assert sweep_retention_caps(_ORG, storage) == 0

    storage.count_retention_target_rows.assert_not_called()


def test_a_failing_target_does_not_stop_the_rest(storage, granted_lock, anomalies):
    """Review Focus 1: a backend missing one hook must not cost the other 16."""
    storage.count_retention_target_rows.side_effect = lambda target: (
        1 / 0 if target == "broken" else 600
    )
    storage.delete_oldest_retention_target_rows.return_value = 120

    with _limits(broken=500, interactions=500):
        assert sweep_retention_caps(_ORG, storage) == 120

    storage.delete_oldest_retention_target_rows.assert_called_once_with(
        "interactions", 120
    )


def test_the_lease_is_released_when_a_target_raises(storage, granted_lock):
    """Review Focus 4: a held lease would suppress the next tick for 600s."""
    storage.count_retention_target_rows.side_effect = RuntimeError("boom")

    with _limits(interactions=500):
        sweep_retention_caps(_ORG, storage)

    granted_lock.return_value.release_simple_lock.assert_called_once()


def test_a_failure_outside_target_isolation_emits_its_own_anomaly(storage, anomalies):
    """The sweep absorbs its own errors, so it owes its own failure signal."""
    with patch.object(retention_sweep, "OperationStateManager") as cls:
        cls.return_value.acquire_simple_lock.side_effect = RuntimeError("no lease")
        with _limits(interactions=500):
            assert sweep_retention_caps(_ORG, storage) == 0

    assert [name for name, _ in anomalies] == ["retention.sweep.failed"]
    _, tags = anomalies[0]
    assert tags["error_type"] == "RuntimeError"


def test_a_slow_pass_reports_itself(storage, granted_lock, anomalies, monkeypatch):
    monkeypatch.setattr(retention_sweep, "SLOW_SWEEP_SECONDS", 0.0)
    storage.count_retention_target_rows.return_value = 0

    with _limits(interactions=500):
        sweep_retention_caps(_ORG, storage)

    assert [name for name, _ in anomalies] == ["retention.sweep.slow"]
