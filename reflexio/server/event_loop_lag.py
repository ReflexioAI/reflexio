"""Measure how late the event loop runs its scheduled work.

Every coroutine in the process -- ``/health/live`` included -- shares one
event loop, so anything that holds it (CPU work on the loop, or a GIL held by
another thread) delays all of them at once. That is invisible in per-request
timings, which only show an unexplained gap. This task sleeps for a fixed
interval and records how much later than requested it woke up:

- ``event_loop.lag`` gauge (milliseconds): the worst lag in each reporting
  window, through :func:`record_health`, so a sink sees one point per window
  rather than one per sample.
- A WARNING ``event=event_loop_lag`` line for any single sample at or above
  :data:`WARN_AFTER_SECONDS`. The ALB liveness probe times out after 5 seconds,
  so a 1-second stall is an early signal, not yet an outage.
"""

import asyncio
import logging
import time
from collections.abc import Callable

from reflexio.server.operational_metrics import record_health

__all__ = [
    "REPORT_EVERY_SECONDS",
    "SAMPLE_INTERVAL_SECONDS",
    "WARN_AFTER_SECONDS",
    "monitor_event_loop_lag",
]

logger = logging.getLogger(__name__)

SAMPLE_INTERVAL_SECONDS = 0.5
REPORT_EVERY_SECONDS = 60.0
WARN_AFTER_SECONDS = 1.0


async def monitor_event_loop_lag(
    *,
    sample_interval: float = SAMPLE_INTERVAL_SECONDS,
    report_every: float = REPORT_EVERY_SECONDS,
    warn_after: float = WARN_AFTER_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Sample loop scheduling delay until cancelled.

    Args:
        sample_interval (float): Seconds requested per sleep.
        report_every (float): Seconds per gauge window.
        warn_after (float): Lag in seconds at which one sample logs a WARNING.
        clock (Callable[[], float]): Monotonic clock, injectable for tests.
    """
    window_started = clock()
    worst = 0.0
    while True:
        requested = clock()
        await asyncio.sleep(sample_interval)
        woke = clock()
        lag = max(0.0, woke - requested - sample_interval)
        worst = max(worst, lag)
        if lag >= warn_after:
            logger.warning(
                "event=event_loop_lag lag_ms=%d sample_interval_ms=%d",
                int(lag * 1000),
                int(sample_interval * 1000),
            )
        if woke - window_started >= report_every:
            record_health(
                "event_loop.lag", worst * 1000, kind="gauge", unit="millisecond"
            )
            window_started = woke
            worst = 0.0
