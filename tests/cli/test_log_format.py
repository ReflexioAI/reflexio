"""Tests for ``reflexio.cli.log_format`` — service prefixes and level highlighting."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from reflexio.cli.log_format import (
    _LEVEL_COLORS,
    format_service_line,
    highlight_log_level,
)


class TestHighlightLogLevel:
    """Cover the severity-highlighting branch of dev-server output."""

    @pytest.fixture(autouse=True)
    def _force_tty(self):
        """Pretend stdout is a TTY so ANSI codes are emitted."""
        with patch("reflexio.cli.log_format.sys.stdout.isatty", return_value=True):
            yield

    @pytest.mark.parametrize(
        "line,level",
        [
            ("ERROR:    [Errno 48] Address already in use", "ERROR"),
            ("[ERROR] something blew up", "ERROR"),
            ("ERROR - request failed", "ERROR"),
            ("CRITICAL: database is down", "CRITICAL"),
            ("WARNING:  deprecated option", "WARNING"),
            ("WARN - legacy client connected", "WARN"),
        ],
    )
    def test_recognised_level_wraps_line(self, line: str, level: str) -> None:
        out = highlight_log_level(line)
        assert out.startswith(f"\033[{_LEVEL_COLORS[level]}m")
        assert out.endswith("\033[0m")
        assert line in out

    @pytest.mark.parametrize(
        "line",
        [
            "INFO:     Application startup complete.",
            "DEBUG: pinging worker",
            "plain log without level",
            "the word ERROR appears later in the line",
            "[INFO] startup complete",
            "",
        ],
    )
    def test_unrecognised_line_unchanged(self, line: str) -> None:
        assert highlight_log_level(line) == line


class TestHighlightLogLevelNonTty:
    """Non-TTY output must stay plain so pipes / log files stay parseable."""

    def test_no_color_when_not_tty(self) -> None:
        with patch("reflexio.cli.log_format.sys.stdout.isatty", return_value=False):
            assert highlight_log_level("ERROR: boom") == "ERROR: boom"


class TestFormatServiceLine:
    """The service prefix wraps around the (possibly highlighted) body."""

    def test_tty_error_line_has_prefix_and_body_colored(self) -> None:
        with patch("reflexio.cli.log_format.sys.stdout.isatty", return_value=True):
            out = format_service_line("backend", "ERROR: port in use")
        assert "[backend ]" in out  # prefix still present & padded
        # Body is red-wrapped
        assert f"\033[{_LEVEL_COLORS['ERROR']}mERROR: port in use\033[0m" in out

    def test_tty_info_line_has_prefix_only(self) -> None:
        with patch("reflexio.cli.log_format.sys.stdout.isatty", return_value=True):
            out = format_service_line("backend", "INFO:     started")
        # Prefix has an escape, body does not
        assert out.count("\033[0m") == 1  # closes the prefix only

    def test_non_tty_plain_output(self) -> None:
        with patch("reflexio.cli.log_format.sys.stdout.isatty", return_value=False):
            out = format_service_line("backend", "ERROR: port in use")
        assert out == "[backend ] ERROR: port in use"
