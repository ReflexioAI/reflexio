import logging
import logging.handlers
import os
import sys
from pathlib import Path

import colorlog

# Eagerly import openai modules to prevent deadlocks when LiteLLM
# lazily imports them from multiple threads simultaneously.
# See: https://github.com/BerriAI/litellm/issues/4075
import openai  # noqa: F401
import openai.resources  # noqa: F401

from reflexio.cli.env_loader import load_reflexio_env

# Load environment variables using shared discovery logic
load_reflexio_env()

# Default user data directory: ~/.reflexio/data/
_DEFAULT_DATA_DIR = str(Path.home() / ".reflexio" / "data")

# OpenAI related
OPENAI_API_KEY = os.environ.get(
    "OPENAI_API_KEY",
    "",
).strip()

# Local file related

LOCAL_STORAGE_PATH = os.environ.get(
    "LOCAL_STORAGE_PATH", _DEFAULT_DATA_DIR
).strip() or _DEFAULT_DATA_DIR

# Local SQLite database file related

SQLITE_FILE_DIRECTORY = os.environ.get(
    "SQLITE_FILE_DIRECTORY", _DEFAULT_DATA_DIR
).strip() or _DEFAULT_DATA_DIR

# Interaction cleanup configuration

INTERACTION_CLEANUP_THRESHOLD = int(
    os.environ.get("INTERACTION_CLEANUP_THRESHOLD", "250000")
)
INTERACTION_CLEANUP_DELETE_COUNT = int(
    os.environ.get("INTERACTION_CLEANUP_DELETE_COUNT", "50000")
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


DEBUG_LOG_TO_CONSOLE = os.environ.get("DEBUG_LOG_TO_CONSOLE", "").strip().lower()
root_logger = logging.getLogger()

if DEBUG_LOG_TO_CONSOLE and DEBUG_LOG_TO_CONSOLE not in ("false", "0", "no"):
    # Enable verbose logging to console with colored output
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # Excludes LLM_PROMPT (level 15)
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(name)s - %(levelname)s - %(message)s",
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
        root_logger.addHandler(console_handler)

    # File handlers
    from reflexio.cli.log_format import DEV_LOG_FILE, LLM_IO_LOG_FILE

    _log_dir = Path.home() / ".reflexio" / "logs"
    _log_dir.mkdir(parents=True, exist_ok=True)

    # General log file — everything except LLM_PROMPT (those go to llm_io.log)
    file_handler = logging.handlers.RotatingFileHandler(
        DEV_LOG_FILE, maxBytes=10_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    file_handler.addFilter(_ExcludeLLMPrompt())
    root_logger.addHandler(file_handler)

    # LLM I/O log file — only LLM_PROMPT level, with structured delimiters
    _HEADER = "═" * 64
    _FOOTER = "─" * 64

    class _LLMIOFormatter(logging.Formatter):
        """Format LLM prompts/responses with delimiters and entry IDs."""

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
                f"\n{_HEADER}\n"
                f"{header_line}\n"
                f"Service: {short_logger}\n"
                f"{_HEADER}\n"
                f"{message}\n"
                f"{_FOOTER}\n"
            )

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
