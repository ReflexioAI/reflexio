"""Uvicorn log configuration for the Reflexio backend.

This module exposes :data:`UVICORN_LOG_CONFIG`, a ``logging.config.dictConfig``
dict that is handed to uvicorn at launch time (see
:mod:`reflexio.server.__main__` and
:func:`reflexio.cli.run_services.build_backend_service`).

Why it lives here and not as an in-memory override:
    Uvicorn's default formatter pads short level names so ``INFO`` aligns
    with ``CRITICAL``. That alignment helps on a single-process console
    but adds noise to our multiplexed ``[backend ]`` dev-server stream.
    Rather than mutate uvicorn's already-configured loggers from inside
    the app module (which is invisible and fragile), we tell uvicorn the
    format upfront via its native ``log_config`` parameter. The dict
    stays discoverable — users who want a different format can edit
    this file or pass their own ``--log-config`` (the CLI flag wins).

Downstream, :func:`reflexio.cli.log_format.highlight_log_level` colorises
``ERROR:`` / ``WARNING:`` / ``CRITICAL:`` prefixes when stdout is a TTY,
so this module deliberately stays color-neutral.
"""

from __future__ import annotations

from typing import Any

# Plain level/message format — no ``%(levelprefix)s`` padding.
LEVEL_FORMAT = "%(levelname)s: %(message)s"

# Access-log fields mirror uvicorn's built-in AccessFormatter message shape,
# minus the padded level prefix.
ACCESS_FORMAT = (
    '%(levelname)s: %(client_addr)s - "%(request_line)s" %(status_code)s'
)


UVICORN_LOG_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": LEVEL_FORMAT},
        "access": {"format": ACCESS_FORMAT},
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "class": "logging.StreamHandler",
            "formatter": "access",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {
            "handlers": ["access"],
            "level": "INFO",
            "propagate": False,
        },
    },
}
