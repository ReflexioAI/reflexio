"""Optional, vendor-neutral error-reporting hooks for shared server code."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import Literal, Protocol

logger = logging.getLogger(__name__)

ErrorLevel = Literal["debug", "info", "warning", "error", "fatal"]


class ErrorReporter(Protocol):
    """Deployment-provided destination for diagnostic error signals."""

    def error_tags(self, tags: Mapping[str, str]) -> AbstractContextManager[None]:
        """Apply tags to errors reported while the context is active."""
        ...

    def set_error_tags(self, tags: Mapping[str, str]) -> None:
        """Apply tags to subsequent errors in the current execution context."""
        ...

    def capture_anomaly(
        self,
        message: str,
        *,
        level: ErrorLevel,
        tags: Mapping[str, str],
    ) -> None:
        """Report a non-fatal anomaly."""
        ...


_error_reporter: ErrorReporter | None = None


def configure_error_reporter(reporter: ErrorReporter | None) -> None:
    """Install a process-global reporter, or clear it with ``None``."""
    global _error_reporter
    _error_reporter = reporter


def _normalize_tags(tags: Mapping[str, object]) -> dict[str, str]:
    return {key: str(value) for key, value in tags.items() if value is not None}


@contextmanager
def error_tags(**tags: object) -> Iterator[None]:
    """Best-effort context tags for errors emitted inside the block."""
    reporter = _error_reporter
    if reporter is None:
        yield
        return

    try:
        reporter_context = reporter.error_tags(_normalize_tags(tags))
        reporter_context.__enter__()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error reporter failed to open tag context: %s", exc)
        yield
        return

    try:
        yield
    except BaseException as exc:
        try:
            reporter_context.__exit__(type(exc), exc, exc.__traceback__)
        except Exception as reporter_exc:  # noqa: BLE001
            logger.warning(
                "Error reporter failed to close tag context: %s", reporter_exc
            )
        raise
    else:
        try:
            reporter_context.__exit__(None, None, None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error reporter failed to close tag context: %s", exc)


def set_error_tags(**tags: object) -> None:
    """Best-effort tags for subsequent errors in the current context."""
    reporter = _error_reporter
    if reporter is None:
        return
    try:
        reporter.set_error_tags(_normalize_tags(tags))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error reporter failed to set tags: %s", exc)


def capture_anomaly(
    message: str,
    *,
    level: ErrorLevel = "warning",
    **tags: object,
) -> None:
    """Best-effort report of a non-fatal anomaly."""
    reporter = _error_reporter
    if reporter is None:
        return
    try:
        reporter.capture_anomaly(
            message,
            level=level,
            tags=_normalize_tags(tags),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error reporter failed to capture anomaly %r: %s", message, exc)
