"""Measure garbage-collector pauses and freeze the startup heap.

A CPython collection holds the GIL for its whole duration, so every thread and
the event loop stop together while it runs. A full (generation 2) pass walks
every tracked container object; on a large heap that can take seconds and is
indistinguishable from any other stall in per-request timings. This module
makes those pauses visible:

- A WARNING ``event=gc_pause`` line for any collection at or above
  :data:`WARN_AFTER_SECONDS`, with the generation, duration and the
  collected/uncollectable counts CPython reports.
- A ``gc.pause.max`` gauge (milliseconds): the longest collection in each
  ``event_loop.lag`` reporting window, emitted beside that gauge so the two
  can be read against each other (see :mod:`reflexio.server.event_loop_lag`).

:func:`freeze_startup_heap` moves every object that exists once startup is
complete into the GC's permanent generation (``gc.freeze``). Those objects
are live for the process lifetime, so later full passes stop re-walking them,
and processes forked afterwards (the per-call LLM workers) do not write to
their pages when the child collects, which preserves copy-on-write sharing.
Collection thresholds are deliberately left unchanged.
"""

import gc
import logging
import time
from collections.abc import Callable
from typing import Any

__all__ = [
    "WARN_AFTER_SECONDS",
    "freeze_startup_heap",
    "install_gc_pause_monitor",
    "take_worst_gc_pause_ms",
    "uninstall_gc_pause_monitor",
]

logger = logging.getLogger(__name__)

WARN_AFTER_SECONDS = 0.1


class _GcPauseTracker:
    """``gc.callbacks`` hook timing each collection.

    CPython never runs two collections at once (the collector refuses to
    re-enter, and the GIL is held for the whole pass), and it invokes the
    ``stop`` callback in the same thread as the matching ``start``, so one
    start timestamp is sufficient.
    """

    def __init__(self, *, warn_after: float, clock: Callable[[], float]) -> None:
        self._warn_after = warn_after
        self._clock = clock
        self._started: float | None = None
        self.worst = 0.0

    def __call__(self, phase: str, info: dict[str, Any]) -> None:
        if phase == "start":
            self._started = self._clock()
            return
        if phase != "stop" or self._started is None:
            return
        duration = max(0.0, self._clock() - self._started)
        self._started = None
        self.worst = max(self.worst, duration)
        if duration >= self._warn_after:
            logger.warning(
                "event=gc_pause generation=%s duration_ms=%d collected=%s "
                "uncollectable=%s",
                info.get("generation"),
                int(duration * 1000),
                info.get("collected"),
                info.get("uncollectable"),
            )


_tracker: _GcPauseTracker | None = None
_frozen = False


def install_gc_pause_monitor(
    *,
    warn_after: float = WARN_AFTER_SECONDS,
    clock: Callable[[], float] = time.perf_counter,
) -> None:
    """Register the pause hook in ``gc.callbacks``; a second call is a no-op.

    Args:
        warn_after (float): Pause in seconds at which one collection logs a WARNING.
        clock (Callable[[], float]): Monotonic clock, injectable for tests.
    """
    global _tracker  # noqa: PLW0603 - one process-wide hook
    if _tracker is not None:
        return
    _tracker = _GcPauseTracker(warn_after=warn_after, clock=clock)
    gc.callbacks.append(_tracker)


def uninstall_gc_pause_monitor() -> None:
    """Remove the pause hook installed by :func:`install_gc_pause_monitor`."""
    global _tracker  # noqa: PLW0603 - one process-wide hook
    if _tracker is None:
        return
    if _tracker in gc.callbacks:
        gc.callbacks.remove(_tracker)
    _tracker = None


def take_worst_gc_pause_ms() -> float | None:
    """Return the longest pause since the previous call, then reset it.

    Returns:
        float | None: Milliseconds, or ``None`` when no hook is installed.
    """
    tracker = _tracker
    if tracker is None:
        return None
    worst, tracker.worst = tracker.worst, 0.0
    return worst * 1000


def freeze_startup_heap() -> bool:
    """Call ``gc.freeze()`` once per process.

    Call it when startup is complete and before any thread or child process
    the server owns has started, so every startup object is frozen and every
    forked child inherits a frozen heap.

    Returns:
        bool: True when this call froze the heap, False when it already was.
    """
    global _frozen  # noqa: PLW0603 - once per process
    if _frozen:
        return False
    # Collect first: a frozen object is never collected again, so cyclic
    # garbage left over from imports and warm-up would otherwise be retained
    # for the process lifetime.
    gc.collect()
    gc.freeze()
    _frozen = True
    logger.info("event=gc_freeze frozen_objects=%d", gc.get_freeze_count())
    return True
