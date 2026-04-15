"""Log formatting utilities for the dev server.

Provides colored service prefixes for subprocess output, a duplicate log filter,
and a startup banner for the multi-service dev server.
"""

from __future__ import annotations

import itertools
import logging
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

# Canonical log file paths — stored in ~/.reflexio/logs/ (not the project directory)
_LOG_DIR = str(Path.home() / ".reflexio" / "logs")
DEV_LOG_FILE = str(Path(_LOG_DIR) / "dev_server.log")
LLM_IO_LOG_FILE = str(Path(_LOG_DIR) / "llm_io.log")

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


def format_service_line(service_name: str, line: str) -> str:
    """Format a log line with a colored, fixed-width service prefix.

    Args:
        service_name: Name of the service (e.g., "backend", "frontend").
        line: The log line content.

    Returns:
        str: Formatted line like "[backend ] message".
    """
    color = SERVICE_COLORS.get(service_name, "37")  # default white
    padded = service_name.ljust(_PREFIX_WIDTH - 2)  # -2 for brackets
    prefix = colorize(f"[{padded}]", color)
    return f"{prefix} {line}"


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
        key = (record.name, record.msg)
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
) -> None:
    """Print a consolidated startup summary banner with service URLs.

    Args:
        ports: Mapping of service name to port number.
        supabase_port: Supabase port, or None if not running.
        log_file: Path to the log file.
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

    lines.append(f"{'-' * width}")
    lines.append(f"  Logs       {log_file}")
    lines.append(f"{'=' * width}\n")

    # Print all at once to avoid interleaving
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()
