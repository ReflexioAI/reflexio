"""A swallowed lineage GC failure must shorten the next tick's wait.

``report_background_failure`` downgrades a transient failure to WARNING and
escalates only once the same scope has failed several consecutive times over
several minutes. A handler that swallows a failure without
``_record_tick_failure()`` leaves the scheduler on its daily cadence, so a
persistent outage there would need days of consecutive failures to page.
"""

import ast
import inspect
import threading
from types import SimpleNamespace

import pytest

from reflexio.models.config_schema import LineageGCConfig
from reflexio.server.services.lineage import gc_scheduler
from reflexio.server.services.lineage.gc_scheduler import LineageGCScheduler


def _calls(node: ast.AST) -> set[str]:
    names = set()
    for call in ast.walk(node):
        if isinstance(call, ast.Call):
            func = call.func
            names.add(
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", "")
            )
    return names


def test_every_reporting_handler_records_the_tick_failure():
    tree = ast.parse(inspect.getsource(gc_scheduler))
    handlers = [
        handler
        for handler in ast.walk(tree)
        if isinstance(handler, ast.ExceptHandler)
        and "report_background_failure" in _calls(handler)
    ]

    assert len(handlers) >= 10  # the scan must still see the scheduler's handlers
    missing = [h.lineno for h in handlers if "_record_tick_failure" not in _calls(h)]
    assert missing == [], f"handlers at lines {missing} swallow without a fast retry"


def test_a_failing_org_gets_the_fast_retry_not_the_daily_cadence():
    def broken_context(_org_id: str):
        raise ConnectionError("server closed the connection unexpectedly")

    scheduler = LineageGCScheduler(
        request_context_factory=broken_context,  # type: ignore[arg-type]
        bootstrap_org_id="org-boot",
    )
    scheduler._tick_had_failure = False

    scheduler._sweep_org("org-1")

    assert scheduler._next_interval(86400) <= gc_scheduler._FAILED_TICK_RETRY_SECONDS


# ---------------------------------------------------------------------------
# Per-org timeouts vs. the fast retry, driven through `_run_once` with the
# real bounded fan-out (production, 2026-10-12: one slow org bought the whole
# fleet three extra sweeps a day).
# ---------------------------------------------------------------------------

_DAY = 86400


def _ctx(org_id: str):
    config = SimpleNamespace(lineage_gc=LineageGCConfig(poll_interval_seconds=_DAY))
    return SimpleNamespace(
        org_id=org_id,
        storage=None,
        configurator=SimpleNamespace(get_config=lambda: config),
    )


@pytest.fixture
def fanout_scheduler(monkeypatch):
    """A scheduler sweeping ``org_ids`` with a 2-wide pool and a 0.2s budget."""
    anomalies: list[str] = []
    monkeypatch.setattr(
        gc_scheduler, "capture_anomaly", lambda name, **_kw: anomalies.append(name)
    )
    monkeypatch.setattr(gc_scheduler, "_ORG_SWEEP_TIMEOUT_SECONDS", 0.2)

    def _build(org_ids: list[str], factory=_ctx) -> LineageGCScheduler:
        scheduler = LineageGCScheduler(
            request_context_factory=factory,  # type: ignore[arg-type]
            bootstrap_org_id="org-boot",
            org_id_provider=lambda: list(org_ids),
        )
        scheduler._org_fanout_workers = lambda _ctx: 2  # type: ignore[method-assign]
        return scheduler

    return _build, anomalies


def test_a_timed_out_org_does_not_fail_the_tick_but_a_repeat_escalates(
    fanout_scheduler,
):
    build, anomalies = fanout_scheduler
    release = threading.Event()
    scheduler = build(["48", "49"])
    scheduler._sweep_org = lambda org_id: (  # type: ignore[method-assign]
        release.wait(5) if org_id == "48" else None
    )
    try:
        assert scheduler._run_once() == _DAY
        assert "lineage.gc.org_sweep_timeout_repeat" not in anomalies

        assert scheduler._run_once() == _DAY
        assert anomalies.count("lineage.gc.org_sweep_timeout_repeat") == 1
    finally:
        release.set()


def test_a_real_org_failure_still_takes_the_fast_retry(fanout_scheduler):
    def factory(org_id: str):
        if org_id == "bad":
            raise ConnectionError("server closed the connection unexpectedly")
        return _ctx(org_id)

    build, _ = fanout_scheduler
    scheduler = build(["good", "bad"], factory=factory)

    assert scheduler._run_once() == gc_scheduler._FAILED_TICK_RETRY_SECONDS


def test_a_straggler_failing_late_does_not_fail_the_next_tick(fanout_scheduler):
    build, _ = fanout_scheduler
    scheduler = build(["48"])
    release, straggler_done = threading.Event(), threading.Event()
    calls = iter(range(2))

    def sweep(_org_id: str) -> None:
        if next(calls) == 0:
            # Tick N: outlive the budget, then fail once tick N+1 is running.
            release.wait(5)
            scheduler._record_tick_failure()
            straggler_done.set()
        else:
            # Tick N+1: let the straggler fail now, inside this tick.
            release.set()
            assert straggler_done.wait(5)

    scheduler._sweep_org = sweep  # type: ignore[method-assign]
    try:
        assert scheduler._run_once() == _DAY  # tick N: timed out, not failed
        assert scheduler._run_once() == _DAY  # tick N+1: straggler ignored
    finally:
        release.set()
