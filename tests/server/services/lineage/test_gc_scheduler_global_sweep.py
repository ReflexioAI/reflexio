import types

import pytest

from reflexio.server.auth import DEFAULT_ORG_ID
from reflexio.server.services.lineage import gc_scheduler
from reflexio.server.services.lineage.gc_scheduler import (
    LineageGCScheduler,
    clear_global_sweeps,
    register_global_sweep,
)


def _scheduler() -> LineageGCScheduler:
    return LineageGCScheduler(
        request_context_factory=lambda _org_id: types.SimpleNamespace(),  # type: ignore[arg-type]
        bootstrap_org_id="org-boot",
    )


def _cfg(enabled: bool):
    return types.SimpleNamespace(
        expiry_reclamation=types.SimpleNamespace(enabled=enabled)
    )


@pytest.fixture(autouse=True)
def _isolate_hooks():
    clear_global_sweeps()
    clear_always = getattr(gc_scheduler, "clear_always_global_sweeps", None)
    if clear_always is not None:
        clear_always()
    yield
    clear_global_sweeps()
    if clear_always is not None:
        clear_always()


def test_global_sweep_runs_once_when_enabled():
    calls: list[int] = []
    register_global_sweep(lambda now: calls.append(now) or 3)
    _scheduler()._run_global_sweeps(_cfg(True))
    assert len(calls) == 1


def test_global_sweep_skipped_when_disabled():
    calls: list[int] = []
    register_global_sweep(lambda now: calls.append(now) or 0)
    _scheduler()._run_global_sweeps(_cfg(False))
    assert calls == []


def test_global_sweep_skipped_when_no_expiry_field():
    """_run_global_sweeps must gate on expiry_reclamation; cfg without the field skips sweeps."""
    calls: list[int] = []
    register_global_sweep(lambda now: calls.append(now) or 1)
    # SimpleNamespace() with NO expiry_reclamation attribute at all
    _scheduler()._run_global_sweeps(types.SimpleNamespace())
    assert calls == []


def test_global_sweep_failure_isolated(monkeypatch):
    ran: list[int] = []
    anomaly_calls: list[tuple] = []
    monkeypatch.setattr(
        gc_scheduler, "capture_anomaly", lambda *a, **k: anomaly_calls.append((a, k))
    )

    def boom(now: int) -> int:
        raise RuntimeError("sweep failed")

    register_global_sweep(boom)
    register_global_sweep(lambda now: ran.append(now) or 1)
    _scheduler()._run_global_sweeps(_cfg(True))
    assert len(ran) == 1  # second sweep still ran despite the first raising
    assert len(anomaly_calls) == 1
    assert anomaly_calls[0][0][0] == "lineage.global_sweep.failed"


def test_run_once_invokes_global_sweeps(monkeypatch):
    """Deleting the _run_global_sweeps() call from _run_once() must break this test."""
    calls: list[int] = []
    register_global_sweep(lambda now: calls.append(now) or 1)

    cfg = types.SimpleNamespace(
        lineage_gc=types.SimpleNamespace(poll_interval_seconds=10),
        expiry_reclamation=types.SimpleNamespace(enabled=True),
    )
    ctx = types.SimpleNamespace(
        configurator=types.SimpleNamespace(get_config=lambda: cfg),
    )
    scheduler = LineageGCScheduler(
        request_context_factory=lambda _org_id: ctx,  # type: ignore[arg-type]
        bootstrap_org_id="org-boot",
    )
    monkeypatch.setattr(scheduler, "_discover_org_ids", lambda _ctx: [])
    # Stub matches _gc_tick's real keyword-only max_workers param (added by
    # the bounded org fan-out) so _run_once's real calling convention still
    # resolves; the assertion below is unaffected — this stub is a no-op
    # regardless of the value it's called with.
    monkeypatch.setattr(scheduler, "_gc_tick", lambda _org_ids, **_kwargs: None)

    scheduler._run_once()

    assert len(calls) == 1


def test_run_once_invokes_always_global_sweep_when_expiry_disabled(monkeypatch):
    calls: list[int] = []
    register = getattr(gc_scheduler, "register_always_global_sweep", None)
    assert callable(register)
    register(lambda now: calls.append(now) or 1)

    cfg = types.SimpleNamespace(
        lineage_gc=types.SimpleNamespace(poll_interval_seconds=10),
        expiry_reclamation=types.SimpleNamespace(enabled=False),
    )
    ctx = types.SimpleNamespace(
        configurator=types.SimpleNamespace(get_config=lambda: cfg),
    )
    scheduler = LineageGCScheduler(
        request_context_factory=lambda _org_id: ctx,  # type: ignore[arg-type]
        bootstrap_org_id="org-boot",
    )
    monkeypatch.setattr(scheduler, "_discover_org_ids", lambda _ctx: [])
    monkeypatch.setattr(scheduler, "_gc_tick", lambda _org_ids, **_kwargs: None)

    scheduler._run_once()

    assert len(calls) == 1


def test_run_once_invokes_always_global_sweep_during_empty_fleet_retry():
    calls: list[int] = []
    gc_scheduler.register_always_global_sweep(lambda now: calls.append(now) or 1)
    factory_calls: list[str] = []
    scheduler = LineageGCScheduler(
        request_context_factory=lambda org_id: factory_calls.append(org_id),  # type: ignore[arg-type]
        bootstrap_org_id=DEFAULT_ORG_ID,
        org_id_provider=lambda: [DEFAULT_ORG_ID],
    )

    assert scheduler._run_once() == 5
    assert len(calls) == 1
    assert factory_calls == []


@pytest.mark.parametrize("bootstrap_failure", ["context", "config"])
def test_run_once_invokes_always_global_sweep_during_bootstrap_failure(
    bootstrap_failure: str,
):
    calls: list[int] = []
    gc_scheduler.register_always_global_sweep(lambda now: calls.append(now) or 1)

    def get_config():
        raise RuntimeError("config bootstrap failed")

    def request_context_factory(_org_id: str):
        if bootstrap_failure == "context":
            raise RuntimeError("context bootstrap failed")
        return types.SimpleNamespace(
            configurator=types.SimpleNamespace(get_config=get_config)
        )

    scheduler = LineageGCScheduler(
        request_context_factory=request_context_factory,  # type: ignore[arg-type]
        bootstrap_org_id="org-boot",
    )

    assert scheduler._run_once() == 86400
    assert len(calls) == 1
