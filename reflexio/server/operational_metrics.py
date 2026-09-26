"""Optional health measurements, independent of tracing and billing."""

from collections.abc import Mapping
from contextlib import suppress
from typing import Literal, Protocol

MetricKind = Literal["counter", "gauge", "distribution"]


class OperationalMetrics(Protocol):
    """Vendor-neutral sink installed by a deployment's capability lifecycle."""

    def record(
        self,
        name: str,
        value: float,
        *,
        kind: MetricKind,
        unit: str | None,
        attributes: Mapping[str, str],
    ) -> None: ...


_sink: OperationalMetrics | None = None


def configure_operational_metrics(sink: OperationalMetrics | None) -> None:
    """Install or clear the optional process-global sink."""
    global _sink
    _sink = sink


def record_health(
    name: str,
    value: float = 1,
    *,
    kind: MetricKind = "counter",
    unit: str | None = None,
    **attributes: str,
) -> None:
    """Best-effort measurement; telemetry must never change product behavior."""
    sink = _sink
    if sink is None:
        return
    # Do not log here: logging can itself use the failing telemetry sink.
    with suppress(Exception):
        sink.record(name, value, kind=kind, unit=unit, attributes=attributes)
