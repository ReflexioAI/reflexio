"""Unit tests for the shared :class:`ThreadedScheduler` base.

Exercise the thread mechanics in isolation (no real scheduler, no I/O): start /
stop / join / idempotent-start / idempotent-stop, the ``_run_once`` interval
seam, and the ``_should_start`` / ``_on_started`` / ``_on_stopped`` hooks.
"""

from __future__ import annotations

import threading
import time

from reflexio.server.scheduling import ThreadedScheduler


class _CountingScheduler(ThreadedScheduler):
    """Ticks as fast as it can, counting ticks; records lifecycle hook calls."""

    def __init__(self, *, should_start: bool = True) -> None:
        super().__init__(thread_name="test-counting-scheduler")
        self._should_start_flag = should_start
        self.ticks = 0
        self.started_calls = 0
        self.stopped_calls = 0
        self._first_tick = threading.Event()

    def _should_start(self) -> bool:
        return self._should_start_flag

    def _on_started(self) -> None:
        self.started_calls += 1

    def _on_stopped(self) -> None:
        self.stopped_calls += 1

    def _run_once(self) -> float:
        self.ticks += 1
        self._first_tick.set()
        return 0.001


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_start_spawns_running_thread_and_ticks() -> None:
    sched = _CountingScheduler()
    assert sched.is_running() is False

    sched.start()
    try:
        assert sched._first_tick.wait(timeout=2.0), "loop never ticked"
        assert sched.is_running() is True
        assert sched.started_calls == 1
        assert _wait_until(lambda: sched.ticks >= 2), "loop did not keep ticking"
    finally:
        sched.stop(timeout_seconds=2.0)

    assert sched.is_running() is False
    assert sched._thread is None
    assert sched.stopped_calls == 1


def test_stop_joins_and_halts_ticking() -> None:
    sched = _CountingScheduler()
    sched.start()
    assert sched._first_tick.wait(timeout=2.0)

    sched.stop(timeout_seconds=2.0)
    assert sched.is_running() is False
    ticks_after_stop = sched.ticks
    time.sleep(0.05)
    assert sched.ticks == ticks_after_stop, "loop kept ticking after stop"


def test_start_is_idempotent_while_running() -> None:
    sched = _CountingScheduler()
    sched.start()
    try:
        assert sched._first_tick.wait(timeout=2.0)
        first_thread = sched._thread
        sched.start()  # second start must be a no-op
        assert sched._thread is first_thread
        assert sched.started_calls == 1
    finally:
        sched.stop(timeout_seconds=2.0)


def test_stop_is_idempotent_when_not_running() -> None:
    sched = _CountingScheduler()
    # Stopping a never-started scheduler is safe and still fires the hook.
    sched.stop(timeout_seconds=1.0)
    assert sched.is_running() is False
    assert sched._thread is None
    assert sched.stopped_calls == 1


def test_should_start_false_vetoes_startup() -> None:
    sched = _CountingScheduler(should_start=False)
    sched.start()
    assert sched.is_running() is False
    assert sched._thread is None
    assert sched.started_calls == 0


def test_run_once_return_value_drives_wait_interval() -> None:
    """The value returned by ``_run_once`` is passed to ``stop_event.wait``."""
    waits: list[float] = []

    class _IntervalScheduler(ThreadedScheduler):
        def __init__(self) -> None:
            super().__init__(thread_name="test-interval-scheduler")

        def _run_once(self) -> float:
            return 42.0

    sched = _IntervalScheduler()
    original_wait = sched._stop_event.wait

    def capturing_wait(timeout: float | None = None) -> bool:
        waits.append(timeout)  # type: ignore[arg-type]
        sched._stop_event.set()  # exit after one iteration
        return original_wait(0)

    sched._stop_event.wait = capturing_wait  # type: ignore[method-assign]
    sched._run_loop()

    assert waits == [42.0]
