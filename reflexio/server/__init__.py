import logging
import logging.handlers
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass

import colorlog

# Eagerly import openai modules to prevent deadlocks when LiteLLM
# lazily imports them from multiple threads simultaneously.
# See: https://github.com/BerriAI/litellm/issues/4075
import openai  # noqa: F401
import openai.resources  # noqa: F401

from reflexio.cli.env_loader import load_reflexio_env
from reflexio.cli.paths import reflexio_home
from reflexio.server.env_utils import env_truthy

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
    return env_truthy(raw)


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


_LOG_LEVEL_NAMES: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

#: Third-party loggers pinned to WARNING in both profiles. Verbose logging is for
#: OUR code; litellm at DEBUG buries it.
_NOISY_THIRD_PARTY = ("litellm", "LiteLLM", "httpx", "httpcore", "openai", "urllib3")

#: First-party loggers quiet by default because they are chatty and rarely the
#: thing being debugged.
_NOISY_FIRST_PARTY = (("reflexio.server.site_var.site_var_manager", logging.ERROR),)

#: The loggers a Sentry Logs floor is allowed to lower. Deliberately NOT the root:
#: `sentry_logs_level=info` previously surfaced INFO from first-party code only,
#: and widening it to the root would ship third-party INFO to a paid pipeline.
_FIRST_PARTY_ROOTS = ("reflexio", "reflexio_ext")


@dataclass(frozen=True)
class LoggingReport:
    """What :func:`configure_logging` actually did.

    Returned so a caller or a test can assert on the outcome instead of
    re-deriving it from the ``logging`` module — which is how a test ends up
    rebuilding the configuration it meant to observe, and then passing whatever
    the runtime did.

    Attributes:
        profile: ``"production"`` or ``"verbose"``.
        root_level: The level left on the root logger.
        handlers: ``"Type:LEVEL"`` for each root handler, in attachment order.
        pinned: ``(logger_name, level)`` for every logger this call set a level
            on, sorted by name.
    """

    profile: str
    root_level: int
    handlers: tuple[str, ...]
    pinned: tuple[tuple[str, int], ...]


def resolve_log_level(raw: str | None, default: int) -> int:
    """Map a level name to its numeric level, falling back on anything unusable.

    Args:
        raw: A level name, any case. ``None`` or blank means "not set".
        default: Returned when ``raw`` is unset or not a level name.

    Returns:
        int: The numeric logging level.
    """
    if raw is None or not raw.strip():
        return default
    return _LOG_LEVEL_NAMES.get(raw.strip().lower(), default)


def resolve_logging_env() -> dict[str, object]:
    """Read every environment variable the logging configuration depends on.

    The ONLY place this module reads the environment for logging, so
    :func:`configure_logging` stays a function of its arguments. That separation
    is the point: the configuration used to be a module body nobody could call,
    which is why working out what a deployed process could log meant running a
    child interpreter against it.

    Profile resolution, preserving the previous semantics exactly:

    * ``REFLEXIO_LOG_PROFILE`` is the one name to read; ``DEBUG_LOG_TO_CONSOLE``
      remains a truthy alias for ``verbose`` when the profile is unset.
    * A verbose REQUEST is still refused under ``ENVIRONMENT=production`` unless
      ``REFLEXIO_ALLOW_PRODUCTION_DEBUG_LOGS`` is set. That gate exists because a
      copied local env file must not make a production deployment verbose, and it
      now applies to the new name as well as the old one.

    Returns:
        dict: Keyword arguments for :func:`configure_logging`.
    """
    profile = os.environ.get("REFLEXIO_LOG_PROFILE", "").strip().lower()
    if profile in ("production", "verbose"):
        wants_verbose = profile == "verbose"
    else:
        wants_verbose = _truthy_env("DEBUG_LOG_TO_CONSOLE")

    verbose = wants_verbose and (
        not _is_production_environment()
        or _truthy_env("REFLEXIO_ALLOW_PRODUCTION_DEBUG_LOGS")
    )

    return {
        "verbose": verbose,
        "level": resolve_log_level(
            os.environ.get("REFLEXIO_LOG_LEVEL"), logging.WARNING
        ),
        "info_loggers": tuple(
            name.strip()
            for name in os.environ.get("REFLEXIO_INFO_LOGGERS", "").split(",")
            if name.strip()
        ),
    }


def _describe_handler(handler: logging.Handler) -> str:
    return f"{type(handler).__name__}:{logging.getLevelName(handler.level)}"


def _build_verbose_handlers(root: logging.Logger) -> None:
    """Attach the developer console and file handlers, at most once."""
    from reflexio.server.correlation import CorrelationIdFilter

    cid_filter = CorrelationIdFilter()

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # Excludes LLM_PROMPT (level 15)
        console_handler.setFormatter(
            colorlog.ColoredFormatter(
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
        )
        from reflexio.cli.log_format import DuplicateFilter

        console_handler.addFilter(DuplicateFilter(window_seconds=5))
        console_handler.addFilter(cid_filter)
        root.addHandler(console_handler)

    from reflexio.cli.log_format import DEV_LOG_FILE, LLM_IO_LOG_FILE, LOG_DIR

    # The file handlers are guarded too, which the module body they came from did
    # NOT do -- only its console handler was. As a module body that ran once the
    # omission was invisible; as a callable function a second call stacked two
    # more RotatingFileHandlers onto the root, so every record was written to the
    # same file twice.
    if any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers):
        return

    # LOG_DIR honors REFLEXIO_LOG_DIR; mkdir here so RotatingFileHandler doesn't
    # crash when the resolved directory doesn't yet exist.
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # General log file -- everything except LLM_PROMPT (those go to llm_io.log)
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
    file_handler.addFilter(cid_filter)
    root.addHandler(file_handler)

    # LLM I/O log file -- only LLM_PROMPT level, with structured delimiters
    llm_io_handler = logging.handlers.RotatingFileHandler(
        LLM_IO_LOG_FILE, maxBytes=10_000_000, backupCount=3, encoding="utf-8"
    )
    llm_io_handler.setLevel(logging.DEBUG)
    llm_io_handler.setFormatter(_LLMIOFormatter())
    llm_io_handler.addFilter(_LLMPromptOnly())
    root.addHandler(llm_io_handler)


def _build_production_handler(root: logging.Logger) -> None:
    """Attach the one stdout handler a container's log driver reads, at most once."""
    if any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    # INFO, below the root's own level on purpose: a logger explicitly raised to
    # INFO (REFLEXIO_INFO_LOGGERS, or a Sentry Logs floor) must still reach the
    # container log driver. The root level is what keeps everything else cheap.
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    root.addHandler(handler)


def configure_logging(
    *,
    verbose: bool,
    level: int = logging.WARNING,
    sentry_floor: int | None = None,
    info_loggers: Sequence[str] = (),
) -> LoggingReport:
    """Configure application logging and report what was done.

    Idempotent: calling it twice attaches no second handler and leaves the same
    levels, so a test may drive it directly.

    ``level`` and ``sentry_floor`` can only ever LOWER a threshold, never raise
    one. That asymmetry is the whole rule, and it is not symmetric by accident:

    * Lowering is REQUIRED. ``SENTRY_LOGS_LEVEL=info`` forwards nothing if INFO
      records never fire in the first place.
    * Raising is pure harm, and used to ship. ``SENTRY_LOGS_LEVEL=error`` -- a
      plausible "just send me errors" -- raised the first-party loggers to ERROR
      and deleted every WARNING line from stdout, which for a self-host customer
      is the only observability channel there is.
    * Raising is also UNNECESSARY, because Sentry filters its Logs pipeline on its
      own ``sentry_logs_level`` independently of the logger level: with a logger
      left at WARNING and that floor at ERROR, the WARNING still reaches stdout
      while only the error is forwarded.

    ``level`` applies to the ROOT logger, because it is the application's overall
    verbosity. ``sentry_floor`` applies to the first-party loggers only, which is
    the scope it already had -- widening it to the root would push third-party
    INFO into a paid pipeline.

    A note on the default. ``level`` defaults to WARNING rather than INFO because
    app-wide INFO on a busy server drove ~3x log volume and grew memory to the
    container ceiling over a few hours (production incident 2026-07-04). INFO is
    therefore opt-in: globally through ``REFLEXIO_LOG_LEVEL`` when an operator
    deliberately asks, or per subsystem through ``info_loggers``, which stays the
    surgical option and the one to prefer.

    Args:
        verbose: Take the developer profile -- colored console plus rotating
            file handlers, root at DEBUG.
        level: Root logger level for the production profile. Lower-only.
        sentry_floor: A Sentry Logs threshold to surface records for, or None.
            Lower-only, and applied to first-party loggers only.
        info_loggers: Logger-name prefixes to raise to INFO individually.

    Returns:
        LoggingReport: The resulting profile, root level, handlers and pinned
            loggers.
    """
    root = logging.getLogger()
    pinned: dict[str, int] = {}

    if verbose:
        _build_verbose_handlers(root)
        root.setLevel(logging.DEBUG)  # Allow all levels; handlers filter
    else:
        _build_production_handler(root)
        root.setLevel(min(level, logging.WARNING))

    for name in _NOISY_THIRD_PARTY:
        logging.getLogger(name).setLevel(logging.WARNING)
        pinned[name] = logging.WARNING
    for name, quiet_level in _NOISY_FIRST_PARTY:
        logging.getLogger(name).setLevel(quiet_level)
        pinned[name] = quiet_level

    for name in info_loggers:
        logging.getLogger(name).setLevel(logging.INFO)
        pinned[name] = logging.INFO

    if sentry_floor is not None:
        for name in _FIRST_PARTY_ROOTS:
            first_party = logging.getLogger(name)
            if sentry_floor < first_party.getEffectiveLevel():
                first_party.setLevel(sentry_floor)
                pinned[name] = sentry_floor

    return LoggingReport(
        profile="verbose" if verbose else "production",
        root_level=root.level,
        handlers=tuple(_describe_handler(h) for h in root.handlers),
        pinned=tuple(sorted(pinned.items())),
    )


#: Resolved once at import, and kept as a module attribute because callers read
#: it to name the branch they are in (`reflexio_ext/server/startup.py`, and the
#: reasoning in `publish_timing.py` and `offline_tuner/config.py`).
DEBUG_LOG_TO_CONSOLE = bool(resolve_logging_env()["verbose"])

#: Kept for backwards compatibility: modules imported this name.
root_logger = logging.getLogger()

#: The import still configures logging, so nothing downstream changes. What is
#: new is that the work now has a NAME, takes arguments, and returns what it did.
LOGGING_REPORT = configure_logging(**resolve_logging_env())  # type: ignore[arg-type]
