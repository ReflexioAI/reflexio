"""Shared CLI state management for Reflexio CLI.

Provides a CliState dataclass that travels through the Typer context,
and a helper to lazily create a ReflexioClient from the state + env
resolution chain.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import typer

from reflexio.defaults import DEFAULT_AGENT_VERSION, DEFAULT_SERVER_URL, DEFAULT_USER_ID

if TYPE_CHECKING:
    from reflexio import ReflexioClient


@dataclass
class CliState:
    """Global CLI state passed via typer.Context.obj.

    Attributes:
        json_mode: Whether to output JSON envelopes
        server_url: Backend API server URL override (from --server-url flag)
        api_key: API key override (from --api-key flag)
    """

    json_mode: bool = False
    server_url: str | None = None
    api_key: str | None = None
    _client: ReflexioClient | None = field(default=None, repr=False)


def resolve_url(server_url: str | None = None) -> str:
    """Resolve server URL: explicit value > REFLEXIO_URL env var > default.

    Args:
        server_url: Explicit URL from CLI flag or caller.

    Returns:
        str: Resolved server URL.
    """
    return server_url or os.environ.get("REFLEXIO_URL") or DEFAULT_SERVER_URL


def resolve_api_key(api_key: str | None = None) -> str:
    """Resolve API key: explicit value > REFLEXIO_API_KEY env var > empty.

    Args:
        api_key: Explicit API key from CLI flag or caller.

    Returns:
        str: Resolved API key (may be empty).
    """
    return api_key or os.environ.get("REFLEXIO_API_KEY") or ""


def get_client(ctx: typer.Context) -> ReflexioClient:
    """Get or create a ReflexioClient from the CLI context.

    Resolution order (highest to lowest priority):
        1. CLI flags (--server-url, --api-key via CliState)
        2. Environment variables (REFLEXIO_URL, REFLEXIO_API_KEY)
        3. Defaults (https://www.reflexio.ai/, no api_key)

    Args:
        ctx: Typer context with CliState in ctx.obj

    Returns:
        ReflexioClient: Configured client instance
    """
    state: CliState = ctx.obj
    if state._client is not None:
        return state._client

    from reflexio import ReflexioClient

    state._client = ReflexioClient(
        url_endpoint=resolve_url(state.server_url),
        api_key=resolve_api_key(state.api_key),
    )
    return state._client


def resolve_user_id(user_id: str | None = None) -> str:
    """Resolve user_id from flag, env var, or default.

    Use this for commands that need to **assign** a user_id to stored
    data (e.g., ``publish``), where silently falling back to
    ``DEFAULT_USER_ID`` is sensible behavior. For commands that
    **filter** by user_id (e.g., ``search``, where a silent default
    would hide the user's real intent), use :func:`require_user_id`.

    Args:
        user_id: Explicit user_id from CLI flag

    Returns:
        str: Resolved user ID
    """
    if user_id:
        return user_id
    if env_val := os.environ.get("REFLEXIO_USER_ID"):
        return env_val
    return DEFAULT_USER_ID


def require_user_id(user_id: str | None = None, *, command_hint: str = "") -> str:
    """Resolve user_id from flag or env var — no silent default.

    Like :func:`resolve_user_id` but raises ``CliError`` instead of
    falling back to ``DEFAULT_USER_ID`` when nothing is specified.
    Use this in commands that **filter** by user_id, where a silent
    default would hide data under the "user-default" identifier
    instead of doing what the user meant.

    Args:
        user_id: Explicit user_id from CLI flag
        command_hint: Optional short phrase describing the command
            (e.g., "profile search") to include in the error message.

    Returns:
        str: Resolved user ID.

    Raises:
        CliError: If no user_id can be resolved from flag or env var.
    """
    from reflexio.cli.errors import EXIT_VALIDATION, CliError

    if user_id:
        return user_id
    if env_val := os.environ.get("REFLEXIO_USER_ID"):
        return env_val
    hint = f" for {command_hint}" if command_hint else ""
    raise CliError(
        error_type="validation",
        message=(
            f"--user-id is required{hint}. Set it via --user-id flag "
            "or REFLEXIO_USER_ID env var."
        ),
        exit_code=EXIT_VALIDATION,
    )


def resolve_agent_version(agent_version: str | None = None) -> str:
    """Resolve agent_version from flag, env var, or default.

    Use this for commands that need to **assign** an agent_version to
    stored data (e.g., ``publish``). For commands that **filter** by
    agent_version (e.g., ``regenerate``), use :func:`require_agent_version`.

    Args:
        agent_version: Explicit agent_version from CLI flag

    Returns:
        str: Resolved agent version
    """
    if agent_version:
        return agent_version
    if env_val := os.environ.get("REFLEXIO_AGENT_VERSION"):
        return env_val
    return DEFAULT_AGENT_VERSION


def require_agent_version(
    agent_version: str | None = None, *, command_hint: str = ""
) -> str:
    """Resolve agent_version from flag or env var — no silent default.

    Like :func:`resolve_agent_version` but raises ``CliError`` instead
    of falling back to ``DEFAULT_AGENT_VERSION``. Use this in commands
    that **filter or re-run against** a specific agent_version, where
    silently defaulting to ``agent-v0`` would produce misleading
    behavior.

    Args:
        agent_version: Explicit agent_version from CLI flag
        command_hint: Optional short phrase describing the command to
            include in the error message.

    Returns:
        str: Resolved agent version.

    Raises:
        CliError: If no agent_version can be resolved from flag or env var.
    """
    from reflexio.cli.errors import EXIT_VALIDATION, CliError

    if agent_version:
        return agent_version
    if env_val := os.environ.get("REFLEXIO_AGENT_VERSION"):
        return env_val
    hint = f" for {command_hint}" if command_hint else ""
    raise CliError(
        error_type="validation",
        message=(
            f"--agent-version is required{hint}. Set it via --agent-version "
            "flag or REFLEXIO_AGENT_VERSION env var."
        ),
        exit_code=EXIT_VALIDATION,
    )
