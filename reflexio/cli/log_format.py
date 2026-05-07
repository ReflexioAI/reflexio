"""Log formatting utilities for the dev server.

Provides colored service prefixes for subprocess output, a duplicate log filter,
and a startup banner for the multi-service dev server.
"""

from __future__ import annotations

import itertools
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

# ANSI color codes for service prefixes
SERVICE_COLORS: dict[str, str] = {
    "backend": "34",  # blue
    "frontend": "32",  # green
    "docs": "35",  # magenta
    "supabase": "36",  # cyan
}

# ANSI codes for log-level severity highlighting in service output.
# Keys are matched against the level token captured by `_LEVEL_RE`.
_LEVEL_COLORS: dict[str, str] = {
    "ERROR": "31",  # red
    "CRITICAL": "1;31",  # bold red
    "WARNING": "33",  # yellow
    "WARN": "33",  # yellow (Next.js / some loggers)
}

# Match a log-level token at the start of a line, optionally bracketed,
# followed by a typical separator (":", whitespace, or " - "). Covers
# uvicorn ("ERROR:    msg"), stdlib logging ("[ERROR] msg"), and the
# "ERROR - msg" style used by Next.js / some custom loggers.
_LEVEL_RE = re.compile(r"^(?:\[)?(ERROR|CRITICAL|WARNING|WARN)(?:\])?(?::|\s+-\s+|\s+)")


# Canonical log file paths.
# Default: ~/.reflexio/logs/. The REFLEXIO_LOG_DIR env var overrides only the
# base directory (the ~ part) — the .reflexio/logs suffix is preserved so the
# on-disk layout stays consistent regardless of where the base points.
def _resolve_log_dir() -> Path:
    """Return the directory log files are written to.

    The ``REFLEXIO_LOG_DIR`` env var overrides the base directory only — the
    ``.reflexio/logs`` suffix is preserved so the on-disk layout matches the
    default home-relative layout. Resolved at module import time so the
    public ``DEV_LOG_FILE`` / ``LLM_IO_LOG_FILE`` constants are stable across
    a server's lifetime — change requires a restart.

    Returns:
        Path: Resolved log directory. Not created here; the rotating file
            handlers create it on first write.
    """
    base = os.environ.get("REFLEXIO_LOG_DIR")
    if base:
        base_path = Path(base).expanduser()
        if not base_path.is_absolute():
            base_path = Path.home() / base_path
        base_path = base_path.resolve()
    else:
        base_path = Path.home()
    return base_path / ".reflexio" / "logs"


LOG_DIR: Path = _resolve_log_dir()
DEV_LOG_FILE: str = str(LOG_DIR / "dev_server.log")
LLM_IO_LOG_FILE: str = str(LOG_DIR / "llm_io.log")

# Thread-safe sequential entry counter for LLM prompt/response entries
_llm_entry_counter = itertools.count(1)
_llm_entry_lock = threading.Lock()


def next_llm_entry_id() -> int:
    """Get the next sequential LLM log entry ID (thread-safe)."""
    with _llm_entry_lock:
        return next(_llm_entry_counter)


# Fixed-width for service prefix alignment
_PREFIX_WIDTH = 10


def colorize(text: str, ansi_code: str, *, bold: bool = False) -> str:
    """Wrap text in ANSI escape sequences for terminal color.

    Returns raw text when stdout is not a TTY (piped output, log files),
    keeping output clean for AI agents and file parsing.

    Args:
        text: The text to colorize.
        ansi_code: ANSI color code (e.g., "34" for blue).
        bold: If True, also apply bold formatting.

    Returns:
        str: Colorized text if TTY, raw text otherwise.
    """
    if not sys.stdout.isatty():
        return text
    prefix = f"\033[1;{ansi_code}m" if bold else f"\033[{ansi_code}m"
    return f"{prefix}{text}\033[0m"


def highlight_log_level(line: str) -> str:
    """Wrap a line in a severity color if it starts with a log-level token.

    Supports common formats: ``ERROR: msg``, ``[ERROR] msg``, ``ERROR - msg``.
    Returns the line unchanged when stdout is not a TTY or no level matches,
    keeping output clean for pipes, log files, and AI agents.

    Args:
        line: The log line content (without the service prefix).

    Returns:
        str: Color-wrapped line for ERROR/CRITICAL/WARNING; otherwise unchanged.
    """
    if not sys.stdout.isatty():
        return line
    match = _LEVEL_RE.match(line)
    if not match:
        return line
    code = _LEVEL_COLORS[match.group(1)]
    return f"\033[{code}m{line}\033[0m"


def format_service_line(service_name: str, line: str) -> str:
    """Format a log line with a colored, fixed-width service prefix.

    The line body is additionally wrapped in a severity color when it
    starts with a recognised log-level token (ERROR/CRITICAL/WARNING),
    so failures stand out against the scrolling dev-server output.

    Args:
        service_name: Name of the service (e.g., "backend", "frontend").
        line: The log line content.

    Returns:
        str: Formatted line like "[backend ] message".
    """
    color = SERVICE_COLORS.get(service_name, "37")  # default white
    padded = service_name.ljust(_PREFIX_WIDTH - 2)  # -2 for brackets
    prefix = colorize(f"[{padded}]", color)
    return f"{prefix} {highlight_log_level(line)}"


class DuplicateFilter(logging.Filter):
    """Suppress duplicate log messages within a time window.

    Keys on (logger_name, msg_template) — the message template string,
    not the formatted output. This is stable across different args since
    the template (e.g., "Supabase Storage for org %s uses URL %s") doesn't
    change between calls.

    Args:
        window_seconds: Time window in seconds to suppress duplicates.
    """

    def __init__(self, window_seconds: int = 5) -> None:
        super().__init__()
        self._recent: dict[tuple[str, str], float] = {}
        self._window = window_seconds
        self._lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False to suppress duplicate messages within the time window."""
        # record.msg can be any object (callers sometimes pass a list/dict
        # directly to logger.warning). Stringify so the key is always
        # hashable — otherwise this filter raises TypeError and crashes
        # whatever was being logged. Surfaced via the Nomic embedder
        # pre-warm path that calls logger.warning(load_return_list).
        key = (record.name, str(record.msg))
        now = time.monotonic()

        with self._lock:
            # Evict stale entries periodically (every 100 checks)
            if len(self._recent) > 200:
                cutoff = now - self._window
                self._recent = {k: v for k, v in self._recent.items() if v >= cutoff}

            last_seen = self._recent.get(key)
            if last_seen is not None and now - last_seen < self._window:
                return False
            self._recent[key] = now
            return True


def print_startup_banner(
    ports: dict[str, int],
    *,
    supabase_port: int | None = 54321,
    log_file: str = DEV_LOG_FILE,
    config_paths: dict[str, str] | None = None,
) -> None:
    """Print a consolidated startup summary banner with service URLs.

    Args:
        ports: Mapping of service name to port number.
        supabase_port: Supabase port, or None if not running.
        log_file: Path to the log file.
        config_paths: Optional mapping of config-label → path string (e.g.
            ``{"env": "~/.reflexio/.env", "config": "~/.reflexio/configs/config_default.json"}``).
            Renders as a "Config" section above the "Logs" line so operators
            can see at a glance which files the server actually loaded.
    """
    lines = []
    width = 44

    lines.append(f"\n{'=' * width}")
    lines.append(colorize("  Reflexio Dev Server", "1", bold=True))
    lines.append(f"{'-' * width}")

    for name in ("backend", "frontend", "docs"):
        if name in ports:
            url = f"http://localhost:{ports[name]}"
            color = SERVICE_COLORS.get(name, "37")
            label = colorize(f"  {name.capitalize():<11}", color)
            status = colorize("ready", "32")
            lines.append(f"{label}{url:<26}{status}")

    if supabase_port is not None:
        url = f"http://localhost:{supabase_port}"
        label = colorize("  Supabase   ", "36")
        status = colorize("ready", "32")
        lines.append(f"{label}{url:<26}{status}")

    home = str(Path.home())

    def _collapse_home(path: str) -> str:
        # Collapse HOME to ~ for readability; absolute paths stay absolute
        # so log scrapers and copy-paste still work when outside HOME.
        return "~" + path[len(home) :] if path.startswith(home) else path

    if config_paths:
        lines.append(f"{'-' * width}")
        for label, path in config_paths.items():
            lines.append(f"  {label:<11}{_collapse_home(str(path))}")

    lines.append(f"{'-' * width}")
    # Logs section — surface both the general dev log and the LLM I/O log.
    # LLM_IO_LOG_FILE is the one operators hit first when debugging prompt /
    # tool-call issues; it's opaque without this pointer.
    lines.append(f"  Dev log    {_collapse_home(log_file)}")
    lines.append(f"  LLM I/O    {_collapse_home(LLM_IO_LOG_FILE)}")
    lines.append(f"{'=' * width}\n")

    # Print all at once to avoid interleaving
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()
