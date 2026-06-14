import logging
import logging.handlers
import os
import sys
import time

import colorlog

# Eagerly import openai modules to prevent deadlocks when LiteLLM
# lazily imports them from multiple threads simultaneously.
# See: https://github.com/BerriAI/litellm/issues/4075
import openai  # noqa: F401
import openai.resources  # noqa: F401

from reflexio.cli.env_loader import load_reflexio_env
from reflexio.cli.paths import reflexio_home

# Load environment variables using shared discovery logic
load_reflexio_env()

# Default user data directory: ~/.reflexio/data/ (or REFLEXIO_LOG_DIR/.reflexio/data/).
_DEFAULT_DATA_DIR = str(reflexio_home() / "data")

# OpenAI related
OPENAI_API_KEY = os.environ.get(
    "OPENAI_API_KEY",
    "",
).strip()

# Local storage directory — houses the SQLite DB file.

LOCAL_STORAGE_PATH = (
    os.environ.get("LOCAL_STORAGE_PATH", "").strip() or _DEFAULT_DATA_DIR
)

# Interaction cleanup configuration

INTERACTION_CLEANUP_THRESHOLD = int(
    os.environ.get("INTERACTION_CLEANUP_THRESHOLD", "250000")
)

# Logging

# Custom log level for full LLM prompts — written to file only (below INFO=20)
LLM_PROMPT_LEVEL = 15
logging.addLevelName(LLM_PROMPT_LEVEL, "LLM_PROMPT")

# Custom log level for model response summaries (between INFO=20 and WARNING=30)
logging.addLevelName(25, "MODEL_RESPONSE")


class _ExcludeLLMPrompt(logging.Filter):
    """Exclude log records at the LLM_PROMPT level."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno != LLM_PROMPT_LEVEL


class _LLMPromptOnly(logging.Filter):
    """Accept only log records at the LLM_PROMPT level."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == LLM_PROMPT_LEVEL


class _TZAwareFormatter(logging.Formatter):
    """Formatter that appends the local UTC offset to every timestamp.

    Renders ``2026-04-24 10:20:51.238 -07:00 PDT`` (TZ abbreviation is
    optional and only appended on systems with tzdata available) so
    readers in any timezone can compute the instant unambiguously.
    Offset comes from the local system zoneinfo via
    ``time.strftime('%z')`` and is rewritten to ISO 8601 extended form
    (``-0700`` → ``-07:00``); falls back to ``+00:00`` on systems
    without a configured timezone.
    """

    default_time_format = "%Y-%m-%d %H:%M:%S"
    default_msec_format = "%s.%03d"

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:  # noqa: ARG002, N802
        ct = time.localtime(record.created)
        base = time.strftime(self.default_time_format, ct)
        msecs = int(record.msecs)
        # ISO 8601 extended form: "-0700" -> "-07:00" — the colon separator
        # reads more clearly as a UTC offset to humans skimming logs.
        raw_offset = time.strftime("%z", ct) or "+0000"
        offset = (
            f"{raw_offset[:3]}:{raw_offset[3:]}" if len(raw_offset) >= 5 else raw_offset
        )
        # Append the local TZ abbreviation (PDT / UTC / etc.) when available.
        # Some minimal containers without tzdata return "" here; the offset
        # alone stays machine-parseable regardless.
        tz_name = time.strftime("%Z", ct)
        if tz_name:
            return f"{base}.{msecs:03d} {offset} {tz_name}"
        return f"{base}.{msecs:03d} {offset}"


class _LLMIOFormatter(_TZAwareFormatter):
    """Format LLM prompts/responses with delimiters and entry IDs."""

    _HEADER = "═" * 64
    _FOOTER = "─" * 64

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record)
        message = record.getMessage()
        short_logger = record.name.rsplit(".", 1)[-1]
        # Use structured extra attributes when available; fall back to parsing
        entry_id = getattr(record, "entry_id", None)
        label = getattr(record, "label", None)
        entry_tag = f"[#{entry_id}]" if entry_id is not None else ""
        if label is None:
            label = message[:60]
        header_line = (
            f"{entry_tag} [{timestamp}] {label}"
            if entry_tag
            else f"[{timestamp}] {label}"
        )
        return (
            f"\n{self._HEADER}\n"
            f"{header_line}\n"
            f"Service: {short_logger}\n"
            f"{self._HEADER}\n"
            f"{message}\n"
            f"{self._FOOTER}\n"
        )


def _truthy_env(name: str) -> bool:
    """Return whether an environment variable is explicitly truthy."""
    raw = os.environ.get(name, "").strip().lower()
    return raw in ("true", "yes", "1", "on")


def _is_production_environment() -> bool:
    """Return whether this process is running in a production deployment."""
    return os.environ.get("ENVIRONMENT", "").strip().lower() in ("prod", "production")


def _debug_log_to_console_enabled() -> bool:
    """Return whether verbose console logging should be enabled.

    ``DEBUG_LOG_TO_CONSOLE`` is a local/dev switch. Deployments with
    ``ENVIRONMENT=production`` must stay quiet by default even if a copied local
    env file accidentally sets it;
    use ``REFLEXIO_ALLOW_PRODUCTION_DEBUG_LOGS=true`` for a deliberate incident
    override.
    """
    if not _truthy_env("DEBUG_LOG_TO_CONSOLE"):
        return False
    return not _is_production_environment() or _truthy_env(
        "REFLEXIO_ALLOW_PRODUCTION_DEBUG_LOGS"
    )


DEBUG_LOG_TO_CONSOLE = _debug_log_to_console_enabled()
root_logger = logging.getLogger()

if DEBUG_LOG_TO_CONSOLE:
    # Correlation ID filter — injects %(correlation_id)s into every record
    from reflexio.server.correlation import CorrelationIdFilter

    _cid_filter = CorrelationIdFilter()

    # Enable verbose logging to console with colored output
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # Excludes LLM_PROMPT (level 15)
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(correlation_tag)s%(name)s - %(levelname)s - %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "reset",
                "LLM_PROMPT": "thin",
                "MODEL_RESPONSE": "cyan",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
        console_handler.setFormatter(formatter)

        # Attach duplicate filter to console only
        from reflexio.cli.log_format import DuplicateFilter

        console_handler.addFilter(DuplicateFilter(window_seconds=5))
        console_handler.addFilter(_cid_filter)
        root_logger.addHandler(console_handler)

    # File handlers
    from reflexio.cli.log_format import DEV_LOG_FILE, LLM_IO_LOG_FILE, LOG_DIR

    # LOG_DIR honors REFLEXIO_LOG_DIR; mkdir here so RotatingFileHandler
    # doesn't crash when the resolved directory doesn't yet exist.
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # General log file — everything except LLM_PROMPT (those go to llm_io.log)
    file_handler = logging.handlers.RotatingFileHandler(
        DEV_LOG_FILE, maxBytes=10_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        _TZAwareFormatter(
            "%(asctime)s %(correlation_tag)s%(name)s %(levelname)s %(message)s"
        )
    )
    file_handler.addFilter(_ExcludeLLMPrompt())
    file_handler.addFilter(_cid_filter)
    root_logger.addHandler(file_handler)

    # LLM I/O log file — only LLM_PROMPT level, with structured delimiters
    llm_io_handler = logging.handlers.RotatingFileHandler(
        LLM_IO_LOG_FILE, maxBytes=10_000_000, backupCount=3, encoding="utf-8"
    )
    llm_io_handler.setLevel(logging.DEBUG)
    llm_io_handler.setFormatter(_LLMIOFormatter())
    llm_io_handler.addFilter(_LLMPromptOnly())
    root_logger.addHandler(llm_io_handler)

    root_logger.setLevel(logging.DEBUG)  # Allow all levels; handlers filter

    # Suppress noisy third-party loggers by name
    for _noisy in ("litellm", "LiteLLM", "httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    # Suppress known-noisy first-party loggers
    logging.getLogger("reflexio.server.site_var.site_var_manager").setLevel(
        logging.ERROR
    )
else:
    # Default to WARNING level when DEBUG_LOG_TO_CONSOLE is not set or is false
    root_logger.setLevel(logging.WARNING)
