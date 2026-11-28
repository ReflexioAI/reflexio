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
    if not (
        Path(executable).is_absolute()
        and os.path.isfile(executable)  # noqa: PTH113 - validate the unnormalised path exactly
        and os.access(executable, os.X_OK)
    ):
        raise ProviderRequestGuardError(
            f"{OPENCLAW_EXECUTABLE_ENV} must name an absolute executable file"
        )
    return executable
