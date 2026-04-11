"""Default values for Reflexio identifiers. Single source of truth."""

import os

DEFAULT_AGENT_VERSION = "agent-v0"
DEFAULT_SERVER_URL = "https://www.reflexio.ai"
DEFAULT_USER_ID = "user-default"


def resolve_agent_version(agent_version: str = "") -> str:
    """Resolve agent_version from explicit value, env var, or default.

    Args:
        agent_version: Explicit value (empty string treated as unset)

    Returns:
        str: Resolved non-empty agent version
    """
    return (
        agent_version
        or os.environ.get("REFLEXIO_AGENT_VERSION", "")
        or DEFAULT_AGENT_VERSION
    )
