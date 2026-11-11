"""GC pause observability and the once-per-process startup heap freeze."""

import asyncio
import gc
import logging
import multiprocessing.process
import threading
from collections.abc import Iterator, Mapping

import pytest

from reflexio.server import gc_pause
from reflexio.server.api import create_app
from reflexio.server.event_loop_lag import monitor_event_loop_lag
from reflexio.server.extensions import AppContext, Capability, CapabilityRegistry
from reflexio.server.operational_metrics import configure_operational_metrics

_LOGGER = "reflexio.server.gc_pause"


class _SteppingClock:
    """Each read advances by ``step`` seconds, so a collection lasts ``step``."""

    def __init__(self, step: float) -> None:
        self.step = step
        self.now = 0.0

    def __call__(self) -> float:
        self.now += self.step
        return self.now


class _Sink:
    def __init__(self) -> None:
        self.records: list[tuple[str, float, str, str | None]] = []

    def record(
        self,
        name: str,
        value: float,
        *,
        kind: str,
        unit: str | None,
        attributes: Mapping[str, str],
    ) -> None:
        self.records.append((name, value, kind, unit))


@pytest.fixture(autouse=True)
def _clean_hook() -> Iterator[None]:
    gc_pause.uninstall_gc_pause_monitor()
    yield
    gc_pause.uninstall_gc_pause_monitor()


def _pause_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if "event=gc_pause" in r.getMessage()
    ]


def test_slow_collection_logs_generation_duration_and_counts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gc_pause.install_gc_pause_monitor(clock=_SteppingClock(0.25))
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        gc.collect(2)

    lines = _pause_lines(caplog)
    assert any(
        "generation=2 duration_ms=250 " in line
        and "collected=" in line
        and "uncollectable=" in line
        for line in lines
    ), lines
    assert gc_pause.take_worst_gc_pause_ms() == pytest.approx(250)
    # Taking the window maximum resets it.
    assert gc_pause.take_worst_gc_pause_ms() == 0


def test_fast_collection_is_silent_but_still_measured(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gc_pause.install_gc_pause_monitor(clock=_SteppingClock(0.05))
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        gc.collect(2)

    assert not _pause_lines(caplog)
    assert gc_pause.take_worst_gc_pause_ms() == pytest.approx(50)


def test_install_is_idempotent_and_uninstall_removes_the_hook() -> None:
    gc_pause.install_gc_pause_monitor()
    gc_pause.install_gc_pause_monitor()
    assert sum(isinstance(cb, gc_pause._GcPauseTracker) for cb in gc.callbacks) == 1
    gc_pause.uninstall_gc_pause_monitor()
    assert not any(isinstance(cb, gc_pause._GcPauseTracker) for cb in gc.callbacks)
    assert gc_pause.take_worst_gc_pause_ms() is None


@pytest.mark.asyncio
async def test_lag_window_emits_the_worst_gc_pause_beside_the_lag() -> None:
    sink = _Sink()
    configure_operational_metrics(sink)
    gc_pause.install_gc_pause_monitor(clock=_SteppingClock(0.3))
    gc.collect(2)
    monitor = asyncio.create_task(
        monitor_event_loop_lag(sample_interval=0.01, report_every=0.05)
    )
    try:
        await asyncio.sleep(0.2)
    finally:
        monitor.cancel()
        with pytest.raises(asyncio.CancelledError):
            await monitor
        configure_operational_metrics(None)

    pauses = [r for r in sink.records if r[0] == "gc.pause.max"]
    assert pauses, "a closed window emits the gc pause gauge"
    assert all(kind == "gauge" and unit == "millisecond" for _, _, kind, unit in pauses)
    assert max(value for _, value, _, _ in pauses) >= 300


class _ObservedStartup(Capability):
    name = "observed_startup"

    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def on_startup(self, ctx: AppContext) -> None:
        self._events.append("capability_startup")


@pytest.mark.asyncio
async def test_lifespan_freezes_once_before_threads_forks_and_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(gc_pause, "_frozen", False)
    monkeypatch.setattr(gc, "freeze", lambda: events.append("freeze"))

    real_thread_start = threading.Thread.start
    real_process_start = multiprocessing.process.BaseProcess.start

    def thread_start(self: threading.Thread) -> None:
        events.append("thread_start")
        real_thread_start(self)

    def process_start(self: multiprocessing.process.BaseProcess) -> None:
        events.append("process_start")
        real_process_start(self)

    monkeypatch.setattr(threading.Thread, "start", thread_start)
    monkeypatch.setattr(multiprocessing.process.BaseProcess, "start", process_start)

    for _ in range(2):
        app = create_app(
            mount_data_plane=False,
            capabilities=CapabilityRegistry([_ObservedStartup(events)]),
        )
        async with app.router.lifespan_context(app):
            assert any(isinstance(cb, gc_pause._GcPauseTracker) for cb in gc.callbacks)
        assert not any(isinstance(cb, gc_pause._GcPauseTracker) for cb in gc.callbacks)

    assert events.count("freeze") == 1
    assert events[0] == "freeze", events
    assert "capability_startup" in events
