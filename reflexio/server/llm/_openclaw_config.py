"""Resolve the operator's executable binding for server-side OpenClaw."""

import os
from pathlib import Path

from reflexio.removed_env import reject_removed_env
from reflexio.server.llm._provider_errors import ProviderRequestGuardError

OPENCLAW_EXECUTABLE_ENV = "REFLEXIO_OPENCLAW_EXECUTABLE"


def configured_openclaw_executable() -> str | None:
    """Read operator intent without consulting executable availability."""
    reject_removed_env(os.environ, canonical=OPENCLAW_EXECUTABLE_ENV)
    executable = os.environ.get(OPENCLAW_EXECUTABLE_ENV, "")
    return executable if executable.strip() else None


def openclaw_executable() -> str | None:
    """Check filesystem availability at server initialization and dispatch."""
    executable = configured_openclaw_executable()
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
