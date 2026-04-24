"""Tests for TZ-aware log formatters in reflexio.server.__init__."""

from __future__ import annotations

import logging
import re

from reflexio.server import _LLMIOFormatter, _TZAwareFormatter

_TZ_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} [+-]\d{2}:\d{2}(?: [A-Z]{1,5})?"
)


def _make_record(msg: str = "payload") -> logging.LogRecord:
    return logging.LogRecord(
        name="reflexio.server.services.tools",
        level=logging.DEBUG,
        pathname="",
        lineno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )


class TestTZAwareFormatter:
    def test_format_time_contains_offset(self) -> None:
        formatter = _TZAwareFormatter()
        record = _make_record()
        rendered = formatter.formatTime(record)
        assert _TZ_PATTERN.match(rendered), f"timestamp missing TZ offset: {rendered!r}"

    def test_format_substitutes_asctime_with_offset(self) -> None:
        """Verify the %(asctime)s path surfaces the TZ-aware timestamp."""
        formatter = _TZAwareFormatter("%(asctime)s %(levelname)s %(message)s")
        record = _make_record("hello")
        out = formatter.format(record)
        assert _TZ_PATTERN.search(out), f"asctime missing TZ offset: {out!r}"
        assert "hello" in out


class TestLLMIOFormatter:
    def test_rendered_header_includes_tz_offset(self) -> None:
        """The _LLMIOFormatter's header line must carry a TZ offset so
        llm_io.log readers in any zone can localise the timestamp."""
        formatter = _LLMIOFormatter()
        record = _make_record("full message payload")
        out = formatter.format(record)
        assert _TZ_PATTERN.search(out), f"header missing TZ offset: {out!r}"
