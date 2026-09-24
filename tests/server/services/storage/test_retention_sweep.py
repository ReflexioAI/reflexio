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
        assert sweep_retention_caps(_ORG, storage).deleted == 0

    storage.count_retention_target_rows.assert_not_called()
    assert anomalies == []


def test_a_count_below_the_limit_deletes_nothing(storage, granted_lock, anomalies):
    storage.count_retention_target_rows.return_value = 100

    with _limits(interactions=500):
        assert sweep_retention_caps(_ORG, storage).deleted == 0

    storage.delete_oldest_retention_target_rows.assert_not_called()
    assert anomalies == []


def test_a_count_equal_to_the_limit_deletes(storage, granted_lock, anomalies):
    """Review Focus 3: the boundary is ``count < limit`` returns, so == deletes."""
    storage.count_retention_target_rows.return_value = 500
    storage.delete_oldest_retention_target_rows.return_value = 100

    with _limits(interactions=500):
        assert sweep_retention_caps(_ORG, storage).deleted == 100

    storage.delete_oldest_retention_target_rows.assert_called_once_with(
        "interactions", 100
    )
    assert [name for name, _ in anomalies] == ["retention.cap.enforced"]


def test_the_warn_boundary_is_inclusive(storage, granted_lock, anomalies):
    """Review Focus 5: rows == 0.90 * limit warns rather than staying silent."""
    storage.count_retention_target_rows.return_value = 450

    with _limits(interactions=500):
        assert sweep_retention_caps(_ORG, storage).deleted == 0

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
        assert sweep_retention_caps(_ORG, storage).deleted == 120

    assert [name for name, _ in anomalies] == ["retention.cap.enforced"]
    _, tags = anomalies[0]
    assert tags["rows_before"] == 600
    assert tags["deleted"] == 120
    assert tags["limit"] == 500


def test_a_refused_lease_probes_nothing(storage, anomalies):
    with patch.object(retention_sweep, "OperationStateManager") as cls:
        cls.return_value.acquire_simple_lock.return_value = False
        with _limits(interactions=1):
            assert sweep_retention_caps(_ORG, storage).deleted == 0

    storage.count_retention_target_rows.assert_not_called()


def test_a_failing_target_does_not_stop_the_rest(storage, granted_lock, anomalies):
    """One target raising must not cost the other 16."""
    storage.count_retention_target_rows.side_effect = lambda target: (
        1 / 0 if target == "broken" else 600
    )
    storage.delete_oldest_retention_target_rows.return_value = 120

    with _limits(broken=500, interactions=500):
        assert sweep_retention_caps(_ORG, storage).deleted == 120

    storage.delete_oldest_retention_target_rows.assert_called_once_with(
        "interactions", 120
    )


def test_a_backend_missing_the_hook_does_not_stop_the_rest(granted_lock, anomalies):
    """Review Focus 1, with the exception the condition actually raises.

    The sibling above uses a ``MagicMock``, on which EVERY attribute exists --
    so it raises ``ZeroDivisionError`` and never reaches the ``AttributeError``
    that a backend genuinely lacking ``RetentionMixin`` would produce. That is
    the gap the type-ignore comment in ``retention_sweep.py`` points at, and a
    test named for a condition it cannot reach is a check that cannot fail.
    """

    class _HookLessStorage:
        """A backend that never mixed in ``RetentionMixin``."""

        def __init__(self) -> None:
            self.deleted: list[tuple[str, int]] = []

        def count_retention_target_rows(self, target_name: str) -> int:
            if target_name == "hookless":
                raise AttributeError("count_retention_target_rows")
            return 600

        def delete_oldest_retention_target_rows(
            self, target_name: str, count: int
        ) -> int:
            self.deleted.append((target_name, count))
            return count

    storage = _HookLessStorage()

    with _limits(hookless=500, interactions=500, profiles=500):
        result = sweep_retention_caps(_ORG, storage)  # type: ignore[arg-type]

    assert storage.deleted == [("interactions", 120), ("profiles", 120)], (
        f"a missing hook stopped the siblings: {storage.deleted}"
    )
    assert result.deleted == 240
    assert [name for name, _ in anomalies] == [
        "retention.cap.enforced",
        "retention.cap.enforced",
    ]


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
            assert sweep_retention_caps(_ORG, storage).deleted == 0

    assert [name for name, _ in anomalies] == ["retention.sweep.failed"]
    _, tags = anomalies[0]
    assert tags["error_type"] == "RuntimeError"


def test_a_slow_pass_reports_itself(storage, granted_lock, anomalies, monkeypatch):
    monkeypatch.setattr(retention_sweep, "SLOW_SWEEP_SECONDS", 0.0)
    storage.count_retention_target_rows.return_value = 0

    with _limits(interactions=500):
        sweep_retention_caps(_ORG, storage)

    assert [name for name, _ in anomalies] == ["retention.sweep.slow"]


# ---------------------------------------------------------------------------
# The sweep must REFUSE an unbound delete, not merely happen to sit where one
# cannot occur
# ---------------------------------------------------------------------------
#
# `project_id` was read only to decorate anomaly tags, so nothing in this module
# checked it before deleting -- the invariant rested entirely on where the call
# site sits. And `error_reporting._normalize_tags` DROPS None values, so an
# unbound enforcement in an enterprise deployment was indistinguishable in Sentry
# from a correct OSS one: both simply carried no `project_id` tag.


class _RecordingProvider:
    """A registered work-scope provider that reports no bound project."""

    def current(self):
        return None

    def bind(self, scope):  # pragma: no cover - never called here
        raise AssertionError("not used")


@pytest.fixture
def unbound_provider():
    from reflexio.server.extensions import register_service
    from reflexio.server.work_scope import WORK_SCOPE_PROVIDER

    register_service(WORK_SCOPE_PROVIDER, _RecordingProvider(), override=True)
    yield
    register_service(WORK_SCOPE_PROVIDER, None, override=True)


def test_an_unbound_pass_refuses_to_probe_when_a_provider_is_registered(
    storage, granted_lock, anomalies, unbound_provider
):
    """A registered provider plus no bound project means the scope was lost.

    In that deployment an unbound pass reads zero rows under the row-level
    policies and reports success, so it is invisible to any count-based check.
    Refusing is the only way it becomes an event rather than a silence.
    """
    storage.count_retention_target_rows.return_value = 10_000_000

    with _limits(interactions=500):
        assert sweep_retention_caps(_ORG, storage).deleted == 0

    storage.count_retention_target_rows.assert_not_called()
    storage.delete_oldest_retention_target_rows.assert_not_called()
    assert [name for name, _ in anomalies] == ["retention.sweep.unbound"]


def test_oss_without_a_provider_is_not_treated_as_unbound(
    storage, granted_lock, anomalies
):
    """The converse: no provider registered is OSS, where no project exists.

    Without this the refusal above would disable retention entirely on SQLite.
    """
    storage.count_retention_target_rows.return_value = 600
    storage.delete_oldest_retention_target_rows.return_value = 120

    with _limits(interactions=500):
        assert sweep_retention_caps(_ORG, storage).deleted == 120

    assert [name for name, _ in anomalies] == ["retention.cap.enforced"]


def test_an_unbound_project_is_tagged_distinguishably(
    storage, granted_lock, anomalies, unbound_provider
):
    """`_normalize_tags` drops None, so an absent tag cannot mean two things."""
    with _limits(interactions=500):
        sweep_retention_caps(_ORG, storage)

    _, tags = anomalies[0]
    assert tags["project_id"] == "<unbound>", (
        f"an unbound pass must not be tag-identical to a correct OSS pass: {tags}"
    )


def test_nothing_escapes_the_sweep_into_the_scheduler(storage, anomalies):
    """The module docstring promises this, and the call site has no backstop.

    `gc_scheduler._sweep_project_data` calls this with no try/except precisely
    because the sweep absorbs its own errors. An escape aborts the remaining
    projects AND the enterprise per-org governance sweep, and on the serial
    fan-out path every remaining org in the tick.
    """
    boom = patch.object(
        retention_sweep,
        "get_row_retention_limits",
        side_effect=RuntimeError("limits blew up"),
    )
    with boom:
        assert sweep_retention_caps(_ORG, storage).deleted == 0

    assert [name for name, _ in anomalies] == ["retention.sweep.failed"]


def test_every_target_failing_fails_the_pass(granted_lock, anomalies):
    """A backend-wide transient hits all targets and must not read as clean.

    One target raising stays isolated (see the sibling above); all of them
    raising is the shape `PGRST002` takes, and absorbing it target by target
    would hand the scheduler a tick that looks successful.
    """

    class _DeadBackend:
        def count_retention_target_rows(self, target_name: str) -> int:
            raise RuntimeError("PGRST002 schema cache cold")

        def delete_oldest_retention_target_rows(
            self, *_a: object
        ) -> int:  # pragma: no cover
            raise AssertionError("must not delete")

    with _limits(interactions=500, profiles=500):
        result = sweep_retention_caps(_ORG, _DeadBackend())  # type: ignore[arg-type]

    assert result.failed, "all targets failed but the pass reported success"
    assert [name for name, _ in anomalies] == ["retention.sweep.all_targets_failed"]


def test_one_target_failing_still_does_not_fail_the_pass(storage, granted_lock):
    """The deliberate non-escalation, pinned so the fix above did not widen it."""
    storage.count_retention_target_rows.side_effect = lambda target: (
        1 / 0 if target == "broken" else 100
    )

    with _limits(broken=500, interactions=500):
        result = sweep_retention_caps(_ORG, storage)

    assert not result.failed


def test_the_library_entry_point_throttles_per_org(storage, granted_lock, monkeypatch):
    """The embedded path has no scheduler, so it sweeps itself -- but not every publish.

    Regression guard for the P1 on #535: moving the sole sweep invocation onto
    `LineageGCScheduler` removed row caps entirely for `Reflexio.publish_interaction`,
    because `maybe_start_lineage_gc` is only ever called from the FastAPI lifespan.
    """
    monkeypatch.setattr(retention_sweep, "_library_last_sweep", {})
    storage.count_retention_target_rows.return_value = 0

    with _limits(interactions=500):
        first = retention_sweep.maybe_sweep_retention_caps_for_library(_ORG, storage)
        second = retention_sweep.maybe_sweep_retention_caps_for_library(_ORG, storage)

    assert first is not None, "the first embedded publish did not sweep at all"
    assert second is None, "the throttle did not suppress the immediate re-sweep"
    assert storage.count_retention_target_rows.call_count == 1


def test_the_library_throttle_is_per_org(storage, granted_lock, monkeypatch):
    """One org publishing must not suppress another org's caps."""
    monkeypatch.setattr(retention_sweep, "_library_last_sweep", {})
    storage.count_retention_target_rows.return_value = 0

    with _limits(interactions=500):
        assert (
            retention_sweep.maybe_sweep_retention_caps_for_library("org-a", storage)
            is not None
        )
        assert (
            retention_sweep.maybe_sweep_retention_caps_for_library("org-b", storage)
            is not None
        )
