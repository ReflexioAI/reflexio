"""Tests for the in-code uvicorn log-format tidy helper.

Uvicorn's default formatter right-pads short level names (``INFO`` →
``INFO:    ``) so they align with ``CRITICAL``. That alignment is
distracting in our multiplexed ``[backend ]`` stream, so we override
the formatter inside ``reflexio.server.__init__`` after uvicorn has
configured its loggers.
"""

from __future__ import annotations

import io
import logging

from reflexio.server import _tidy_uvicorn_log_format


def _make_record(name: str = "uvicorn", level: int = logging.INFO) -> logging.LogRecord:
    return logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=None,
        exc_info=None,
    )


def _capture(logger_name: str, record: logging.LogRecord) -> str:
    """Emit a record through the first handler on `logger_name` and
    capture its formatted output."""
    handler = logging.getLogger(logger_name).handlers[0]
    buf = io.StringIO()
    # Temporarily redirect the handler's stream so we don't spam stderr.
    original_stream = getattr(handler, "stream", None)
    try:
        handler.stream = buf  # type: ignore[attr-defined]
        handler.emit(record)
    finally:
        if original_stream is not None:
            handler.stream = original_stream  # type: ignore[attr-defined]
    return buf.getvalue().rstrip()


class TestTidyUvicornLogFormat:
    def test_formatter_drops_padding(self) -> None:
        # Simulate uvicorn: attach a padded formatter to the uvicorn logger.
        logger = logging.getLogger("uvicorn")
        logger.handlers.clear()
        handler = logging.StreamHandler(io.StringIO())
        handler.setFormatter(logging.Formatter("%(levelname)-9s %(message)s"))
        logger.addHandler(handler)

        _tidy_uvicorn_log_format()

        assert _capture("uvicorn", _make_record()) == "INFO: hello"

    def test_idempotent(self) -> None:
        logger = logging.getLogger("uvicorn.error")
        logger.handlers.clear()
        handler = logging.StreamHandler(io.StringIO())
        handler.setFormatter(logging.Formatter("%(levelname)-9s %(message)s"))
        logger.addHandler(handler)

        _tidy_uvicorn_log_format()
        formatter_after_first = handler.formatter
        _tidy_uvicorn_log_format()

        # No handler proliferation, and the second call produces an
        # equivalent plain formatter (type + format string match).
        assert len(logger.handlers) == 1
        assert handler.formatter is not None
        assert formatter_after_first is not None
        assert handler.formatter._fmt == formatter_after_first._fmt  # noqa: SLF001

    def test_handles_logger_without_handlers(self) -> None:
        # Uvicorn.access may not be configured in bare pytest runs.
        logger = logging.getLogger("uvicorn.access")
        logger.handlers.clear()

        # Must not raise.
        _tidy_uvicorn_log_format()

        assert logger.handlers == []
