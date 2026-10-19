"""Stable learning identity for the OpenClaw-only integration.

There is no runtime host selector: hooks always originate from OpenClaw.
"""

_AGENT_VERSION = "openclaw"


def agent_version() -> str:
    """Reflexio agent version used for cross-project OpenClaw learning."""
    return _AGENT_VERSION
