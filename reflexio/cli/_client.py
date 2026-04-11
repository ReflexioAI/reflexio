"""ReflexioClient factory with layered config resolution.

All connection settings come from CLI flags, environment variables
(typically loaded from ``~/.reflexio/.env`` at CLI startup), or
hard-coded defaults. No separate config file is used.
"""

from __future__ import annotations

import argparse
import os

from reflexio import ReflexioClient
from reflexio.defaults import DEFAULT_AGENT_VERSION, DEFAULT_SERVER_URL, DEFAULT_USER_ID


def _resolve(
    args: argparse.Namespace | None,
    arg_name: str,
    env_var: str,
    default: str,
) -> str:
    """Resolve a config value through the priority chain: CLI flag -> env var -> default.

    Args:
        args (argparse.Namespace | None): Parsed CLI arguments
        arg_name (str): Attribute name on the args namespace
        env_var (str): Environment variable name
        default (str): Fallback default value

    Returns:
        str: The resolved value
    """
    if args and hasattr(args, arg_name) and getattr(args, arg_name):
        return getattr(args, arg_name)
    if env_val := os.environ.get(env_var):
        return env_val
    return default


def create_client(args: argparse.Namespace | None = None) -> ReflexioClient:
    """Create a ReflexioClient with config resolved from CLI flags, env vars, and defaults.

    Resolution order (highest to lowest priority):
        1. CLI flags (args.url, args.api_key)
        2. Environment variables (REFLEXIO_URL, REFLEXIO_API_KEY)
        3. Defaults (https://www.reflexio.ai/, no api_key)

    Args:
        args (argparse.Namespace | None): Parsed CLI arguments, or None for env/defaults only

    Returns:
        ReflexioClient: Configured client instance
    """
    url = _resolve(args, "url", "REFLEXIO_URL", DEFAULT_SERVER_URL)
    api_key = _resolve(args, "api_key", "REFLEXIO_API_KEY", "")
    return ReflexioClient(url_endpoint=url, api_key=api_key)


def resolve_user_id(args: argparse.Namespace | None = None) -> str:
    """Resolve user_id from CLI flags, env var, or default.

    Args:
        args (argparse.Namespace | None): Parsed CLI arguments

    Returns:
        str: Resolved user ID
    """
    return _resolve(args, "user_id", "REFLEXIO_USER_ID", DEFAULT_USER_ID)


def resolve_agent_version(args: argparse.Namespace | None = None) -> str:
    """Resolve agent_version from CLI flags, env var, or default.

    Args:
        args (argparse.Namespace | None): Parsed CLI arguments

    Returns:
        str: Resolved agent version
    """
    return _resolve(
        args,
        "agent_version",
        "REFLEXIO_AGENT_VERSION",
        DEFAULT_AGENT_VERSION,
    )
