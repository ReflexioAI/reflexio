"""Tests for ``reflexio.cli.log_format`` — service prefixes and level highlighting."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from reflexio.cli.log_format import (
    _LEVEL_COLORS,
    DuplicateFilter,
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


class TestDuplicateFilterSuppressionIsVisible:
    """A per-entity warning loop must not silently under-report its scope.

    ``DuplicateFilter`` keys on the message TEMPLATE, so N per-org warnings
    emitted in a loop share one key and collapse to roughly one per window.
    Dropping the rest silently reads as full coverage — a real migration audit
    affecting 19 orgs printed as 7 for exactly this reason.
    """

    def _record(self, msg: str, args: tuple = ()) -> logging.LogRecord:
        return logging.LogRecord(
            name="t",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg=msg,
            args=args,
            exc_info=None,
        )

    def test_suppressed_count_is_reported_on_the_next_emitted_record(self):
        filt = DuplicateFilter(window_seconds=300)
        first = self._record("org %s is wedged", ("org_1",))
        assert filt.filter(first) is True
        assert "suppressed" not in first.getMessage()

        for org in ("org_2", "org_3", "org_4"):
            assert filt.filter(self._record("org %s is wedged", (org,))) is False

        filt._recent.clear()  # simulate the window elapsing
        later = self._record("org %s is wedged", ("org_5",))
        assert filt.filter(later) is True
        assert "+3 similar suppressed" in later.getMessage()

    def test_emitted_message_survives_handler_reformatting(self):
        """The suffix must not be destroyed by the handler's own %-substitution."""
        filt = DuplicateFilter(window_seconds=300)
        assert filt.filter(self._record("org %s is wedged", ("org_1",))) is True
        assert filt.filter(self._record("org %s is wedged", ("org_2",))) is False
        filt._recent.clear()
        rec = self._record("org %s is wedged", ("org_3",))
        filt.filter(rec)
        # getMessage() is what handlers call; calling it twice must be stable
        # and must not raise on a template whose args were cleared.
        assert rec.getMessage() == rec.getMessage()
        assert "org_3" in rec.getMessage()
        assert "+1 similar suppressed" in rec.getMessage()

    def test_distinct_templates_do_not_share_a_suppression_count(self):
        filt = DuplicateFilter(window_seconds=300)
        assert filt.filter(self._record("alpha %s", ("a",))) is True
        assert filt.filter(self._record("alpha %s", ("b",))) is False
        beta = self._record("beta %s", ("c",))
        assert filt.filter(beta) is True
        assert "suppressed" not in beta.getMessage()

    def test_unsuppressed_records_are_left_untouched(self):
        filt = DuplicateFilter(window_seconds=300)
        rec = self._record("solo %s", ("x",))
        assert filt.filter(rec) is True
        assert rec.getMessage() == "solo x"
        assert rec.args == ("x",)


def _info(msg: str) -> logging.LogRecord:
    return logging.LogRecord("t", logging.INFO, __file__, 1, msg, (), None)


class TestDuplicateFilterEviction:
    """Stale keys are evicted once the cache grows past its bound."""

    def test_stale_entries_are_evicted_and_their_keys_admit_again(self, monkeypatch):
        clock = [1000.0]
        monkeypatch.setattr("reflexio.cli.log_format.time.monotonic", lambda: clock[0])
        filt = DuplicateFilter(window_seconds=5)
        for i in range(201):
            assert filt.filter(_info(f"stale {i}")) is True
        assert len(filt._recent) == 201

        clock[0] += 60  # every cached key is now outside the window
        assert filt.filter(_info("fresh")) is True
        assert set(filt._recent) == {("t", "fresh")}
        # An evicted key is admitted again rather than counted as a duplicate.
        assert filt.filter(_info("stale 0")) is True

    def test_live_entries_survive_eviction(self, monkeypatch):
        clock = [0.0]
        monkeypatch.setattr("reflexio.cli.log_format.time.monotonic", lambda: clock[0])
        filt = DuplicateFilter(window_seconds=5)
        for i in range(200):
            filt.filter(_info(f"old {i}"))
        clock[0] = 100.0
        filt.filter(_info("recent"))  # 201 entries; eviction not yet triggered
        clock[0] = 101.0
        assert filt.filter(_info("another")) is True  # triggers eviction
        assert ("t", "recent") in filt._recent
        assert ("t", "old 0") not in filt._recent
        # The surviving key is still inside its window, so it is suppressed.
        assert filt.filter(_info("recent")) is False


class TestPrintStartupBanner:
    """The startup banner reports every requested service, including failures."""

    def _banner(self, capsys, ports, **kwargs) -> str:
        from reflexio.cli.log_format import print_startup_banner

        with patch("reflexio.cli.log_format.sys.stdout.isatty", return_value=False):
            print_startup_banner(ports, **kwargs)
        return capsys.readouterr().out

    def test_ready_services_and_default_supabase(self, capsys) -> None:
        out = self._banner(capsys, {"backend": 8081, "docs": 8082})
        assert "Reflexio Dev Server" in out
        assert "Backend    http://localhost:8081" in out
        assert "Docs       http://localhost:8082" in out
        assert "Frontend" not in out
        assert "Supabase   http://localhost:54321" in out
        assert "\x1b[" not in out  # plain text when not a TTY
        assert "Failures" not in out
        assert "Dev log" in out
        assert "LLM I/O" in out

    def test_supabase_not_ready_and_failures_are_visible(self, capsys) -> None:
        out = self._banner(
            capsys,
            {"backend": 8081},
            supabase_port=55555,
            supabase_status="starting",
            failures=["frontend did not become ready", "docs exited with code 1"],
        )
        supabase_line = next(line for line in out.splitlines() if "Supabase" in line)
        assert "http://localhost:55555" in supabase_line
        assert "not ready" in supabase_line
        assert "Failures" in out
        assert "  - frontend did not become ready" in out
        assert "  - docs exited with code 1" in out

    def test_supabase_line_omitted_when_disabled(self, capsys) -> None:
        out = self._banner(capsys, {"frontend": 3000}, supabase_port=None)
        assert "Supabase" not in out
        assert "Frontend   http://localhost:3000" in out

    def test_paths_collapse_home_only_inside_home(
        self, capsys, tmp_path, monkeypatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        out = self._banner(
            capsys,
            {},
            supabase_port=None,
            config_paths={
                "env": str(home / ".reflexio" / ".env"),
                "config": "/etc/reflexio/config.json",
            },
            log_file=str(home / "logs" / "dev.log"),
        )
        assert "env        ~/.reflexio/.env" in out
        assert "config     /etc/reflexio/config.json" in out
        assert "Dev log    ~/logs/dev.log" in out
        assert str(home) not in out

    def test_tty_banner_colors_status(self, capsys) -> None:
        from reflexio.cli.log_format import print_startup_banner

        with patch("reflexio.cli.log_format.sys.stdout.isatty", return_value=True):
            print_startup_banner({"backend": 8081}, supabase_status="down")
        out = capsys.readouterr().out
        assert "\x1b[32mready\x1b[0m" in out
        assert "\x1b[1;31mnot ready\x1b[0m" in out
