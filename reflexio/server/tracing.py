"""Optional tracing hook for request-path profiling.

This module intentionally has no observability-vendor dependency. Deployments
that want tracing can register a tracer; deployments that do not register one
pay only a cheap context-manager/no-op cost.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class TraceSpan(Protocol):
    """Minimal span handle exposed to shared Reflexio code."""

    def set_data(self, key: str, value: Any) -> None:
        """Attach low-cardinality metadata to the active span."""
        ...


class Tracer(Protocol):
    """Process-global tracer interface configured by enterprise startup."""

    def span(self, name: str, **data: Any) -> AbstractContextManager[TraceSpan]:
        """Create a span context manager."""
        ...


class _NoopTraceSpan:
    def set_data(self, key: str, value: Any) -> None:
        del key, value


_NOOP_SPAN = _NoopTraceSpan()
_tracer: Tracer | None = None


def configure_tracer(tracer: Tracer | None) -> None:
    """Set the process-global tracer.

    Args:
        tracer: Tracer implementation, or None to disable tracing.
    """
    global _tracer
    _tracer = tracer


@contextmanager
def profile_step(name: str, **data: Any) -> Iterator[TraceSpan]:
    """Profile a named step if tracing is configured.

    Tracing must never make product requests fail. Errors from creating,
    entering, or exiting the tracing span are logged and swallowed. Exceptions
    raised by the profiled product code are still propagated unchanged.
    """
    tracer = _tracer
    if tracer is None:
        yield _NOOP_SPAN
        return

    try:
        span_cm = tracer.span(name, **data)
        span = span_cm.__enter__()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tracer failed to start span %s: %s", name, exc)
        yield _NOOP_SPAN
        return

    try:
        yield span
    except BaseException as exc:
        try:
            span_cm.__exit__(type(exc), exc, exc.__traceback__)
        except Exception as tracer_exc:  # noqa: BLE001
            logger.warning("Tracer failed to finish span %s: %s", name, tracer_exc)
        raise
    else:
        try:
            span_cm.__exit__(None, None, None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tracer failed to finish span %s: %s", name, exc)


def set_span_data(span: TraceSpan, values: Mapping[str, Any]) -> None:
    """Best-effort helper for attaching multiple span fields."""
    for key, value in values.items():
        try:
            span.set_data(key, value)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tracer failed to set span data %s: %s", key, exc)
