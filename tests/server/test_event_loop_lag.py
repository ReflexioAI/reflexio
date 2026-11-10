"""The event-loop lag monitor reports a blocked loop and runs for the app's lifetime."""

import asyncio
import logging
import time
from collections.abc import Mapping

import pytest

from reflexio.server.api import create_app
from reflexio.server.event_loop_lag import monitor_event_loop_lag
from reflexio.server.operational_metrics import configure_operational_metrics


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


@pytest.mark.asyncio
async def test_blocked_loop_warns_and_records_the_worst_lag(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sink = _Sink()
    configure_operational_metrics(sink)
    # The stall lands early in a 1 s window and the window closes on a quiet
    # sample, so the gauge carries the window's worst sample, not its last.
    monitor = asyncio.create_task(
        monitor_event_loop_lag(sample_interval=0.01, report_every=1.0, warn_after=0.1)
    )
    try:
        with caplog.at_level(logging.WARNING, logger="reflexio.server.event_loop_lag"):
            await asyncio.sleep(0.05)
            time.sleep(0.3)  # hold the loop, as on-loop CPU work does
            await asyncio.sleep(1.0)
    finally:
        monitor.cancel()
        with pytest.raises(asyncio.CancelledError):
            await monitor
        configure_operational_metrics(None)

    warnings = [
        r.getMessage()
        for r in caplog.records
        if "event=event_loop_lag" in r.getMessage()
    ]
    lags_ms = [int(w.split("lag_ms=")[1].split()[0]) for w in warnings]
    assert lags_ms and max(lags_ms) >= 250
    gauges = [r for r in sink.records if r[0] == "event_loop.lag"]
    assert gauges, "one window closed, so one gauge point is due"
    assert all(kind == "gauge" and unit == "millisecond" for _, _, kind, unit in gauges)
    assert gauges[0][1] >= 250


@pytest.mark.asyncio
async def test_idle_loop_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    monitor = asyncio.create_task(
        monitor_event_loop_lag(sample_interval=0.01, warn_after=0.5)
    )
    with caplog.at_level(logging.WARNING, logger="reflexio.server.event_loop_lag"):
        await asyncio.sleep(0.1)
    monitor.cancel()
    with pytest.raises(asyncio.CancelledError):
        await monitor
    assert not [r for r in caplog.records if "event=event_loop_lag" in r.getMessage()]


def _monitors() -> list[asyncio.Task]:
    return [t for t in asyncio.all_tasks() if t.get_name() == "event_loop_lag_monitor"]


@pytest.mark.asyncio
async def test_app_lifespan_runs_the_monitor() -> None:
    app = create_app(mount_data_plane=False)
    async with app.router.lifespan_context(app):
        running = _monitors()
        assert len(running) == 1
        assert not running[0].done()
    assert running[0].cancelled()
    assert not _monitors()
