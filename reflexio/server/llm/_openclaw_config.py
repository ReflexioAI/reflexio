"""Resolve the operator's executable binding for server-side OpenClaw."""

import os
from pathlib import Path

from reflexio.removed_env import reject_removed_env
from reflexio.server.env_utils import env_str
from reflexio.server.llm._litellm_types import ProviderRequestGuardError

OPENCLAW_EXECUTABLE_ENV = "REFLEXIO_OPENCLAW_EXECUTABLE"


def openclaw_executable() -> str | None:
    """Return the authorized executable, or None when the route is disabled."""
    reject_removed_env(os.environ, canonical=OPENCLAW_EXECUTABLE_ENV)
    executable = env_str(OPENCLAW_EXECUTABLE_ENV)
    if not executable:
        return None
    path = Path(executable)
    if not (path.is_absolute() and path.is_file() and os.access(path, os.X_OK)):
        raise ProviderRequestGuardError(
            f"{OPENCLAW_EXECUTABLE_ENV} must name an absolute executable file"
        )
    return executable
