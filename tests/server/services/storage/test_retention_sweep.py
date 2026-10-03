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
from reflexio.server.services.storage.retention_mixin import (
    RetentionMixin,
    RetentionProbe,
)
from reflexio.server.services.storage.retention_sweep import sweep_retention_caps

_ORG = "org-1"


@pytest.fixture
def storage() -> MagicMock:
    storage = MagicMock()
    # No cheap estimate: every target is counted exactly, as before the
    # estimate existed. Tests of the estimate gate set their own value.
    storage.estimate_retention_target_rows.return_value = None
    return storage


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
    # A slow pass names the table that made it slow.
    assert anomalies[0][1]["slowest_target"] == "interactions"


# ---------------------------------------------------------------------------
# The cheap estimate settles tables far below their cap without count(*)
# ---------------------------------------------------------------------------


def test_an_estimate_well_under_the_cap_skips_the_exact_count(
    storage, granted_lock, anomalies
):
    storage.estimate_retention_target_rows.return_value = 100

    with _limits(interactions=500):
        assert sweep_retention_caps(_ORG, storage).deleted == 0

    storage.count_retention_target_rows.assert_not_called()
    storage.delete_oldest_retention_target_rows.assert_not_called()


@pytest.mark.parametrize("estimate", [400, 499, 5000])
def test_an_estimate_near_or_over_the_cap_is_confirmed_exactly(
    storage, granted_lock, anomalies, estimate
):
    """At >= 80% of the cap the exact count decides -- the estimate never deletes."""
    storage.estimate_retention_target_rows.return_value = estimate
    storage.count_retention_target_rows.return_value = 100

    with _limits(interactions=500):
        assert sweep_retention_caps(_ORG, storage).deleted == 0

    storage.count_retention_target_rows.assert_called_once_with("interactions")
    storage.delete_oldest_retention_target_rows.assert_not_called()


# ---------------------------------------------------------------------------
# The batched probe: one call sizes every target, decisions are unchanged
# ---------------------------------------------------------------------------
#
# The tests above drive a `MagicMock`, which is not a `RetentionMixin`, so they
# exercise the per-target fallback (`probe_retention_targets_individually`) --
# the same estimate-then-count they always did. These drive a real mixin
# subclass whose `probe_retention_targets` answers in one call, as the enterprise
# backends do.


class _BatchedStorage(RetentionMixin):
    """A ``RetentionMixin`` whose probe is scripted, recording every call."""

    def __init__(self, probes: dict[str, RetentionProbe]) -> None:
        self.probes = probes
        self.probe_calls: list[dict[str, float]] = []
        self.counted: list[str] = []
        self.count_value = 0
        self.counts: dict[str, int] = {}
        self.deleted: list[tuple[str, int]] = []

    def probe_retention_targets(self, exact_count_from):  # type: ignore[override]
        self.probe_calls.append(dict(exact_count_from))
        return {
            name: self.probes[name] for name in exact_count_from if name in self.probes
        }

    def count_retention_target_rows(self, target_name: str) -> int:
        self.counted.append(target_name)
        return self.counts.get(target_name, self.count_value)

    def delete_oldest_retention_target_rows(self, target_name: str, count: int) -> int:
        self.deleted.append((target_name, count))
        return count

    # Abstract hooks the sweep never reaches through this double.
    def _retention_table_exists(self, table_name: str) -> bool:  # pragma: no cover
        raise AssertionError("not used")

    def _retention_count_rows(self, target):  # pragma: no cover
        raise AssertionError("not used")

    def _retention_select_oldest_keys(self, *a, **k):  # pragma: no cover
        raise AssertionError("not used")

    def _retention_delete_dependencies(self, *a, **k):  # pragma: no cover
        raise AssertionError("not used")

    def _retention_delete_target_rows(self, *a, **k):  # pragma: no cover
        raise AssertionError("not used")


def _exact(rows: int) -> RetentionProbe:
    return RetentionProbe(rows, exact=True)


def _estimate(rows: int) -> RetentionProbe:
    return RetentionProbe(rows, exact=False)


def test_one_probe_call_sizes_every_target(granted_lock, anomalies):
    storage = _BatchedStorage(
        {"interactions": _estimate(10), "profiles": _exact(0), "requests": _exact(5)}
    )

    with _limits(interactions=500, profiles=500, requests=500):
        assert sweep_retention_caps(_ORG, storage).deleted == 0  # type: ignore[arg-type]

    assert storage.probe_calls == [
        {"interactions": 400.0, "profiles": 400.0, "requests": 400.0}
    ], "the thresholds must be EXACT_COUNT_FROM_FRACTION of each cap"
    assert storage.counted == [], "an exact probe must not be counted again"
    assert anomalies == []


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (449, []),
        (450, ["retention.cap.approaching"]),
        (499, ["retention.cap.approaching"]),
        (500, ["retention.cap.enforced"]),
    ],
)
def test_an_exact_probe_decides_like_the_count_did(
    granted_lock, anomalies, rows, expected
):
    storage = _BatchedStorage({"interactions": _exact(rows)})
    storage.count_value = rows  # the table did not move since the probe

    with _limits(interactions=500):
        sweep_retention_caps(_ORG, storage)  # type: ignore[arg-type]

    assert [name for name, _ in anomalies] == expected
    assert storage.deleted == ([("interactions", 100)] if rows >= 500 else [])


@pytest.mark.parametrize("estimate", [400, 500, 5000])
def test_an_estimate_at_or_over_the_threshold_never_deletes_on_its_own(
    granted_lock, anomalies, estimate
):
    """A probe that returns an estimate where it should have counted is
    confirmed by an exact count; the estimate itself never warns or deletes."""
    storage = _BatchedStorage({"interactions": _estimate(estimate)})
    storage.count_value = 100

    with _limits(interactions=500):
        assert sweep_retention_caps(_ORG, storage).deleted == 0  # type: ignore[arg-type]

    assert storage.counted == ["interactions"]
    assert storage.deleted == []
    assert anomalies == []


def test_the_confirming_count_is_what_the_delete_rests_on(granted_lock, anomalies):
    storage = _BatchedStorage({"interactions": _estimate(450)})
    storage.count_value = 600

    with _limits(interactions=500):
        assert sweep_retention_caps(_ORG, storage).deleted == 120  # type: ignore[arg-type]

    _, tags = anomalies[0]
    assert tags["rows_before"] == 600


@pytest.mark.parametrize(
    ("snapshot", "now", "expected", "deleted"),
    [
        # A writer crossed the cap after the probe: the delete must still happen.
        (499, 500, ["retention.cap.enforced"], [("interactions", 100)]),
        # Rows were removed after the probe: no delete on the stale 500.
        (500, 470, ["retention.cap.approaching"], []),
        (450, 449, [], []),
    ],
)
def test_a_snapshot_that_would_warn_or_delete_is_recounted_first(
    granted_lock, anomalies, snapshot, now, expected, deleted
):
    """The probe is taken before every earlier target's turn, so a snapshot
    only ever settles "nothing to do"; anything that would warn or delete is
    decided on a count taken at its own turn, as the per-target loop did."""
    storage = _BatchedStorage({"interactions": _exact(snapshot)})
    storage.count_value = now

    with _limits(interactions=500):
        sweep_retention_caps(_ORG, storage)  # type: ignore[arg-type]

    assert storage.counted == ["interactions"]
    assert [name for name, _ in anomalies] == expected
    assert storage.deleted == deleted


def test_a_snapshot_below_the_warn_threshold_is_not_recounted(granted_lock, anomalies):
    storage = _BatchedStorage({"interactions": _exact(449)})

    with _limits(interactions=500):
        sweep_retention_caps(_ORG, storage)  # type: ignore[arg-type]

    assert storage.counted == []


def test_an_estimate_under_the_threshold_settles_the_target(granted_lock, anomalies):
    storage = _BatchedStorage({"interactions": _estimate(399)})

    with _limits(interactions=500):
        sweep_retention_caps(_ORG, storage)  # type: ignore[arg-type]

    assert storage.counted == []
    assert storage.deleted == []


def test_a_delete_reprobes_the_targets_after_it(granted_lock, anomalies):
    """A cascade can shrink a later target (``user_playbooks`` deletes from
    ``agent_playbook_source_user_playbooks``), so what was probed before the
    delete is not what the per-target loop would have seen. The targets after
    a delete are probed one at a time -- never the whole remainder again per
    delete, which would make an N-target sweep quadratic on SQLite."""
    storage = _BatchedStorage(
        {
            "profiles": _exact(0),
            "interactions": _exact(600),
            "requests": _exact(0),
            "skills": _exact(0),
        }
    )
    storage.counts = {"interactions": 600}  # still at cap when its turn comes
    storage.count_value = 7

    with _limits(profiles=500, interactions=500, requests=500, skills=500):
        sweep_retention_caps(_ORG, storage)  # type: ignore[arg-type]

    assert [list(call) for call in storage.probe_calls] == [
        ["profiles", "interactions", "requests", "skills"],
    ]
    # `interactions` is recounted before its delete; the two after it are
    # re-read rather than taken from the snapshot probed before the delete.
    assert storage.counted == ["interactions", "requests", "skills"], (
        "the targets after a delete must be re-read, not taken from the "
        "snapshot probed before it"
    )


def test_a_probe_error_fails_only_its_own_target(granted_lock, anomalies):
    storage = _BatchedStorage(
        {
            "broken": RetentionProbe(error=ZeroDivisionError("boom")),
            "interactions": _exact(600),
        }
    )
    storage.count_value = 600

    with _limits(broken=500, interactions=500):
        result = sweep_retention_caps(_ORG, storage)  # type: ignore[arg-type]

    assert result.deleted == 120
    assert not result.failed
    assert storage.deleted == [("interactions", 120)]


def test_every_probe_failing_fails_the_pass(granted_lock, anomalies):
    error = RetentionProbe(error=RuntimeError("PGRST002"))
    storage = _BatchedStorage({"interactions": error, "profiles": error})

    with _limits(interactions=500, profiles=500):
        result = sweep_retention_caps(_ORG, storage)  # type: ignore[arg-type]

    assert result.failed
    assert [name for name, _ in anomalies] == ["retention.sweep.all_targets_failed"]


def test_a_target_the_probe_omitted_is_probed_on_its_own(granted_lock, anomalies):
    storage = _BatchedStorage({"profiles": _exact(0)})
    storage.count_value = 600

    with _limits(profiles=500, interactions=500):
        assert sweep_retention_caps(_ORG, storage).deleted == 120  # type: ignore[arg-type]

    assert storage.counted == ["interactions"]


def test_a_probe_that_raises_outright_fails_the_pass_and_frees_the_lease(
    granted_lock, anomalies
):
    storage = _BatchedStorage({})
    storage.probe_retention_targets = MagicMock(side_effect=RuntimeError("down"))  # type: ignore[method-assign]

    with _limits(interactions=500):
        result = sweep_retention_caps(_ORG, storage)  # type: ignore[arg-type]

    assert result.failed
    assert [name for name, _ in anomalies] == ["retention.sweep.failed"]
    granted_lock.return_value.release_simple_lock.assert_called_once()


def test_a_slow_pass_attributes_probe_time(granted_lock, anomalies, monkeypatch):
    monkeypatch.setattr(retention_sweep, "SLOW_SWEEP_SECONDS", 0.0)
    storage = _BatchedStorage(
        {
            "interactions": RetentionProbe(0, exact=True, seconds=0.0),
            "requests": RetentionProbe(0, exact=True, seconds=7.0),
        }
    )

    with _limits(interactions=500, requests=500):
        sweep_retention_caps(_ORG, storage)  # type: ignore[arg-type]

    [(name, tags)] = anomalies
    assert name == "retention.sweep.slow"
    assert tags["slowest_target"] == "requests"
    assert "probe_seconds" in tags


# ---------------------------------------------------------------------------
# The sweep must REFUSE an unbound delete, not merely happen to sit where one
# cannot occur
# ---------------------------------------------------------------------------
#
# `project_id` was read only to decorate anomaly tags, so nothing in this module
# checked it before deleting -- the invariant rested entirely on where the call
# site sits. And `error_reporting._normalize_tags` DROPS None values, so an
# unbound enforcement in an enterprise deployment was indistinguishable in reports
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


# -- Retention policy: alert-only customer caps, and the age pass -------------

from reflexio.server.services.storage.retention import (  # noqa: E402
    FAIL_SAFE_RETENTION_POLICY,
    RetentionPolicy,
)
from reflexio.server.services.storage.retention_mixin import (  # noqa: E402
    AgeRetentionResult,
)


def _policy(policy: RetentionPolicy):
    return patch.object(retention_sweep, "_resolve_policy", return_value=policy)


def test_an_alert_only_customer_cap_pages_and_deletes_nothing(
    storage, granted_lock, anomalies
):
    storage.count_retention_target_rows.return_value = 600
    storage.delete_oldest_retention_target_rows.return_value = 100

    with (
        _limits(interactions=500, share_links=500),
        _policy(RetentionPolicy(customer_cap_action="alert")),
    ):
        assert sweep_retention_caps(_ORG, storage).deleted == 100

    # Customer data only alerts; an internal table keeps its oldest-first trim.
    storage.delete_oldest_retention_target_rows.assert_called_once_with(
        "share_links", 120
    )
    exceeded = [tags for name, tags in anomalies if name == "retention.cap.exceeded"]
    assert len(exceeded) == 1
    assert (exceeded[0]["target"], exceeded[0]["level"]) == ("interactions", "error")


def test_the_customer_cap_limit_replaces_the_env_limit(
    storage, granted_lock, anomalies
):
    storage.count_retention_target_rows.return_value = 600

    with (
        _limits(interactions=500),
        _policy(RetentionPolicy(customer_cap_action="alert", customer_cap_limit=2_000)),
    ):
        sweep_retention_caps(_ORG, storage)

    assert anomalies == []


def test_a_failing_policy_provider_fails_safe(storage, granted_lock, anomalies):
    """A provider outage must not become silent oldest-first deletion."""
    storage.count_retention_target_rows.return_value = 600

    def broken(org_id: str) -> RetentionPolicy:
        raise RuntimeError("billing db down")

    with (
        _limits(interactions=500),
        patch.object(
            retention_sweep,
            "get_service",
            lambda key: (
                broken if key is retention_sweep.RETENTION_POLICY_PROVIDER else None
            ),
        ),
    ):
        sweep_retention_caps(_ORG, storage)

    storage.delete_oldest_retention_target_rows.assert_not_called()
    assert [name for name, _ in anomalies] == [
        "retention.policy.failed",
        "retention.cap.exceeded",
    ]
    assert FAIL_SAFE_RETENTION_POLICY.age_days is None


@pytest.fixture
def mixin_storage() -> MagicMock:
    storage = MagicMock(spec=RetentionMixin)
    storage.expire_retention_target_rows.return_value = AgeRetentionResult()
    return storage


def test_age_enforcement_without_an_archiver_deletes_nothing(
    mixin_storage, granted_lock, anomalies
):
    with _limits(), _policy(RetentionPolicy(age_days=30, enforce_age=True)):
        assert sweep_retention_caps(_ORG, mixin_storage).deleted == 0

    mixin_storage.expire_retention_target_rows.assert_not_called()
    assert [name for name, _ in anomalies] == ["retention.age.no_archiver"]


def test_a_dry_run_never_hands_the_storage_an_archiver(
    mixin_storage, granted_lock, anomalies
):
    archiver = MagicMock(return_value=True)

    with _limits(), _policy(RetentionPolicy(age_days=30, archiver=archiver)):
        sweep_retention_caps(_ORG, mixin_storage)

    calls = mixin_storage.expire_retention_target_rows.call_args_list
    assert calls and all(call.kwargs["archiver"] is None for call in calls)


def test_age_enforcement_visits_interactions_before_requests(
    mixin_storage, granted_lock, anomalies
):
    archiver = MagicMock(return_value=True)
    mixin_storage.expire_retention_target_rows.return_value = AgeRetentionResult(
        eligible=2, deleted=2
    )

    with (
        _limits(),
        _policy(RetentionPolicy(age_days=30, enforce_age=True, archiver=archiver)),
    ):
        result = sweep_retention_caps(_ORG, mixin_storage)

    targets = [
        call.args[0]
        for call in mixin_storage.expire_retention_target_rows.call_args_list
    ]
    assert targets.index("interactions") < targets.index("requests")
    assert result.deleted == 2 * len(targets)
    assert all(
        call.kwargs["archiver"] is archiver
        for call in mixin_storage.expire_retention_target_rows.call_args_list
    )


def test_a_blocked_age_pass_raises_an_error_anomaly(
    mixin_storage, granted_lock, anomalies
):
    mixin_storage.expire_retention_target_rows.return_value = AgeRetentionResult(
        blocked="archive_failed"
    )
    archiver = MagicMock(return_value=False)

    with (
        _limits(),
        _policy(RetentionPolicy(age_days=30, enforce_age=True, archiver=archiver)),
    ):
        sweep_retention_caps(_ORG, mixin_storage)

    blocked = [tags for name, tags in anomalies if name == "retention.age.blocked"]
    assert blocked and all(tags["reason"] == "archive_failed" for tags in blocked)
