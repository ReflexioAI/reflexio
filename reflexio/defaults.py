"""Default values for Reflexio identifiers. Single source of truth."""

import os

from reflexio.removed_env import reject_removed_env

DEFAULT_AGENT_VERSION = "agent-v0"
DEFAULT_SERVER_URL = "https://www.reflexio.ai"
DEFAULT_USER_ID = "user-default"


def resolve_agent_version(agent_version: str | None = None) -> str:
    """Use caller attribution, or the historical sentinel for an omitted label.

    A host process may serve many agents/organizations. Its environment cannot
    choose the identity of a request that did not supply one. CLI filters use
    their existing explicit-target check instead of adopting this sentinel.
    """
    reject_removed_env(os.environ, {"REFLEXIO_AGENT_VERSION": ""})
    return agent_version or DEFAULT_AGENT_VERSION
