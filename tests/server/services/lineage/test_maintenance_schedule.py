"""One captured process schedule governs the entire shared maintenance worker."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reflexio.server.services.lineage import gc_scheduler as gc


@pytest.fixture(autouse=True)
def isolated_hooks(monkeypatch):
    for name in (
        "_per_org_sweep_hooks",
        "_global_sweep_hooks",
        "_always_global_sweep_hooks",
    ):
        monkeypatch.setattr(gc, name, [])


def test_process_cadence_survives_environment_changes(monkeypatch):
    monkeypatch.setenv("REFLEXIO_MAINTENANCE_INTERVAL_SECONDS", "37")
    config = SimpleNamespace(lineage_gc=SimpleNamespace(enabled=False))
    context = SimpleNamespace(
        org_id="org",
        storage=None,
        configurator=SimpleNamespace(get_config=lambda: config),
    )
    scheduler = gc.LineageGCScheduler(
        request_context_factory=MagicMock(return_value=context), bootstrap_org_id="org"
    )
    monkeypatch.setattr(scheduler, "_discover_org_ids", lambda _: [])
    monkeypatch.setattr(scheduler, "_gc_tick", MagicMock())
    monkeypatch.setattr(scheduler, "_org_fanout_workers", lambda _: 1)
    monkeypatch.setenv("REFLEXIO_MAINTENANCE_INTERVAL_SECONDS", "91")
    assert scheduler._run_once() == 37


def test_bootstrap_failure_uses_process_cadence(monkeypatch):
    monkeypatch.setenv("REFLEXIO_MAINTENANCE_INTERVAL_SECONDS", "37")
    always = []
    gc.register_always_global_sweep(lambda now: always.append(now) or 0)

    def broken(_):
        raise RuntimeError("configuration unavailable")

    scheduler = gc.LineageGCScheduler(
        request_context_factory=broken, bootstrap_org_id="org"
    )
    assert [scheduler._run_once() for _ in range(4)] == [37] * 4
    assert len(always) == 4


@pytest.mark.parametrize(
    "value", ["0", "-1", "1.5", "nan", "private://secret", str(10**100)]
)
def test_invalid_cadence_fails_before_bootstrap_effects(monkeypatch, value):
    monkeypatch.setenv("REFLEXIO_MAINTENANCE_INTERVAL_SECONDS", value)
    factory = MagicMock()
    with pytest.raises(
        ValueError, match="REFLEXIO_MAINTENANCE_INTERVAL_SECONDS"
    ) as error:
        gc.maybe_start_lineage_gc(factory, bootstrap_org_id="org")
    assert value not in str(error.value)
    factory.assert_not_called()


@pytest.mark.parametrize("value", [None, "", "  ", "86400"])
def test_default_cadence_and_timer_boundary(monkeypatch, value):
    import threading

    if value is None:
        monkeypatch.delenv("REFLEXIO_MAINTENANCE_INTERVAL_SECONDS", raising=False)
    else:
        monkeypatch.setenv("REFLEXIO_MAINTENANCE_INTERVAL_SECONDS", value)
    assert gc.MaintenanceSchedule.from_env().interval_seconds == 86400
    assert gc.MaintenanceSchedule(
        interval_seconds=int(threading.TIMEOUT_MAX)
    ).interval_seconds == int(threading.TIMEOUT_MAX)
    with pytest.raises(ValueError):
        gc.MaintenanceSchedule(interval_seconds=int(threading.TIMEOUT_MAX) + 1)


def test_worker_captures_gate_and_wait_provider_before_reinstall(monkeypatch):
    old_gate, new_gate = MagicMock(), MagicMock()
    old = gc.MaintenanceSchedule(
        interval_seconds=37,
        leader_gate=old_gate,
        healthy_interval_provider=lambda cadence: cadence - 2,
    )
    monkeypatch.setattr(gc, "_maintenance_schedule_hook", old)
    scheduler = gc.LineageGCScheduler(
        request_context_factory=MagicMock(), bootstrap_org_id="org"
    )
    gc.set_maintenance_schedule(
        gc.MaintenanceSchedule(interval_seconds=91, leader_gate=new_gate)
    )
    assert scheduler._leader_gate is old_gate
    assert scheduler._schedule is old
    assert scheduler._next_interval(old.interval_seconds) == 35
    newer = gc.LineageGCScheduler(
        request_context_factory=MagicMock(), bootstrap_org_id="org"
    )
    assert newer._leader_gate is new_gate
    assert newer._next_interval(newer._schedule.interval_seconds) == 91


@pytest.mark.parametrize("cadence,fast", [(37, 37), (900, 300)])
def test_bounded_failure_retries_use_captured_cadence(monkeypatch, cadence, fast):
    monkeypatch.setenv("REFLEXIO_MAINTENANCE_INTERVAL_SECONDS", str(cadence))

    def broken(_):
        raise RuntimeError("unavailable")

    scheduler = gc.LineageGCScheduler(
        request_context_factory=broken, bootstrap_org_id="org"
    )
    assert [scheduler._run_once() for _ in range(4)] == [fast, fast, fast, cadence]


def test_no_org_bootstrap_retry_is_independent(monkeypatch):
    from reflexio.server.auth import DEFAULT_ORG_ID

    monkeypatch.setenv("REFLEXIO_MAINTENANCE_INTERVAL_SECONDS", "37")
    scheduler = gc.LineageGCScheduler(
        request_context_factory=MagicMock(),
        bootstrap_org_id=DEFAULT_ORG_ID,
        org_id_provider=lambda: [],
    )
    assert scheduler._run_once() == 5


def test_unreadable_bootstrap_still_admits_registered_work(monkeypatch):
    monkeypatch.setenv("REFLEXIO_MAINTENANCE_INTERVAL_SECONDS", "37")
    called = []
    gc.register_always_global_sweep(lambda now: called.append(now) or 0)

    def broken(_):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(gc.LineageGCScheduler, "start", lambda _: None)
    scheduler = gc.maybe_start_lineage_gc(broken, bootstrap_org_id="org")
    assert scheduler is not None
    assert scheduler._run_once() == 37
    assert len(called) == 1
