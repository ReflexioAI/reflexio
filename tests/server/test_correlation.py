"""Tests for the correlation-id logging filter.

The filter injects both the raw ``correlation_id`` and a pre-formatted
``correlation_tag`` so format strings can avoid rendering an empty
``[]`` bracket pair outside request context (startup / CLI output).
"""

from __future__ import annotations

import logging

from reflexio.server.correlation import (
    CorrelationIdFilter,
    correlation_id_var,
)


def _make_record() -> logging.LogRecord:
    return logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=None,
        exc_info=None,
    )


class TestCorrelationIdFilter:
    def test_sets_empty_tag_outside_request_context(self) -> None:
        """Empty correlation ID must render as empty string, not ``[]``."""
        record = _make_record()
        # Ensure no CID leaks in from another test.
        token = correlation_id_var.set("")
        try:
            CorrelationIdFilter().filter(record)
        finally:
            correlation_id_var.reset(token)
        assert record.correlation_id == ""
        assert record.correlation_tag == ""

    def test_formats_tag_when_cid_present(self) -> None:
        record = _make_record()
        token = correlation_id_var.set("abc12345")
        try:
            CorrelationIdFilter().filter(record)
        finally:
            correlation_id_var.reset(token)
        assert record.correlation_id == "abc12345"
        assert record.correlation_tag == "[abc12345] "

    def test_format_string_renders_cleanly_without_cid(self) -> None:
        """Integration: format a record through the real format string."""
        record = _make_record()
        record.name = "reflexio_ext.server.db"
        token = correlation_id_var.set("")
        try:
            CorrelationIdFilter().filter(record)
        finally:
            correlation_id_var.reset(token)

        fmt = logging.Formatter(
            "%(correlation_tag)s%(name)s - %(levelname)s - %(message)s"
        )
        rendered = fmt.format(record)
        assert rendered == "reflexio_ext.server.db - INFO - hello"

    def test_format_string_renders_cid_when_present(self) -> None:
        record = _make_record()
        record.name = "reflexio.server.api"
        token = correlation_id_var.set("deadbeef")
        try:
            CorrelationIdFilter().filter(record)
        finally:
            correlation_id_var.reset(token)

        fmt = logging.Formatter(
            "%(correlation_tag)s%(name)s - %(levelname)s - %(message)s"
        )
        rendered = fmt.format(record)
        assert rendered == "[deadbeef] reflexio.server.api - INFO - hello"
