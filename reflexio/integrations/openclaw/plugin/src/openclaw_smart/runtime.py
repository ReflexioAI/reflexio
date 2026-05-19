"""Host/runtime state shared by openclaw-smart entrypoints.

openclaw-smart only runs in the openClaw context — there is no Claude Code
or Codex variant — but we keep ``set_host``/``host`` for symmetry with the
claude-smart API and to leave room for additional hosts in the future. The
reflexio ``agent_version`` is hardcoded so all openClaw projects roll up
into the same shared-learning bucket.
"""

from __future__ import annotations

import os

HOST_ENV = "OPENCLAW_SMART_HOST"

HOST_OPENCLAW = "openclaw"
VALID_HOSTS = frozenset({HOST_OPENCLAW})

_AGENT_VERSION = "openclaw"
_current_host: str | None = None


def set_host(value: str | None) -> str:
    """Set the current host, returning the normalized value."""
    global _current_host
    host_value = value if value in VALID_HOSTS else HOST_OPENCLAW
    _current_host = host_value
    os.environ[HOST_ENV] = host_value
    return host_value


def host() -> str:
    """Return the current host, defaulting to openClaw."""
    if _current_host is not None:
        return _current_host
    value = os.environ.get(HOST_ENV)
    return value if value in VALID_HOSTS else HOST_OPENCLAW


def is_openclaw() -> bool:
    """True when the current hook invocation came from openClaw."""
    return host() == HOST_OPENCLAW


def agent_version() -> str:
    """Reflexio agent version used for cross-project openClaw learning."""
    return _AGENT_VERSION
