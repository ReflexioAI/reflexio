import types

import pytest

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
    yield
    clear_global_sweeps()


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


def test_global_sweep_failure_isolated(monkeypatch):
    ran: list[int] = []
    monkeypatch.setattr(gc_scheduler, "capture_anomaly", lambda *_a, **_k: None)

    def boom(now: int) -> int:
        raise RuntimeError("sweep failed")

    register_global_sweep(boom)
    register_global_sweep(lambda now: ran.append(now) or 1)
    _scheduler()._run_global_sweeps(_cfg(True))
    assert len(ran) == 1  # second sweep still ran despite the first raising
