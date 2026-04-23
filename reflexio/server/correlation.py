"""Request correlation ID for tracing concurrent operations.

Stores a short hex ID in a ContextVar so every log line emitted during a
single HTTP request (including ThreadPoolExecutor worker threads) can be
correlated back to the originating request.

Usage in worker threads::

    ctx = contextvars.copy_context()
    executor.submit(ctx.run, fn, *args)
"""

from __future__ import annotations

import contextvars
import logging
import uuid

# ContextVar holds the correlation ID for the current request.
# Default is empty string (non-request contexts like startup, CLI).
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)


def generate_correlation_id() -> str:
    """Generate a short 8-character hex correlation ID."""
    return uuid.uuid4().hex[:8]


class CorrelationIdFilter(logging.Filter):
    """Logging filter that injects correlation fields into every log record.

    Attach this filter to handlers so formatters can use:

    - ``%(correlation_id)s`` — the raw ID (empty string outside request context).
    - ``%(correlation_tag)s`` — a ready-to-embed ``"[<cid>] "`` string that
      collapses to ``""`` when there is no correlation ID. Use this in format
      strings to avoid rendering an empty ``[]`` at startup / CLI time.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        cid = correlation_id_var.get("")
        record.correlation_id = cid  # type: ignore[attr-defined]
        record.correlation_tag = f"[{cid}] " if cid else ""  # type: ignore[attr-defined]
        return True
