"""Small helpers for parsing environment-backed settings."""

from __future__ import annotations

import os
from collections.abc import Mapping


def env_get(env: Mapping[str, str], name: str) -> str:
    """Return a stripped environment value or an empty string."""
    return str(env.get(name, "")).strip()


def env_str(
    name: str,
    default: str = "",
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Return an environment value, treating unset OR blank as the default.

    Unlike ``os.getenv(name, default)``, an empty / whitespace-only value
    resolves to ``default`` rather than passing the empty string through. This
    closes the "set to blank behaves differently from unset" gap: a ``KEY=``
    line (which python-dotenv exports as ``""``) and an absent key both yield
    the default, so callers never silently receive ``""`` where they expected a
    real fallback (e.g. an ``int()``/``float()`` parse that would crash on ``""``).

    Args:
        name: Environment variable name.
        default: Value returned when the variable is unset or blank.
        env: Mapping to read from; defaults to ``os.environ``.

    Returns:
        The stripped environment value, or ``default`` when unset/blank.
    """
    raw = (env if env is not None else os.environ).get(name)
    return raw.strip() if raw and raw.strip() else default


def env_required(env: Mapping[str, str], name: str) -> str:
    """Return a required environment value or raise a startup error."""
    value = env_get(env, name)
    if not value:
        raise RuntimeError(f"{name} is required for enterprise startup")
    return value


def env_required_literal(
    env: Mapping[str, str],
    name: str,
    *,
    allowed: tuple[str, ...],
) -> str:
    """Return a required lowercased value constrained to allowed literals."""
    value = env_required(env, name).lower()
    if value not in allowed:
        allowed_text = " or ".join(repr(item) for item in allowed)
        raise RuntimeError(f"{name} must be {allowed_text} (got {value!r})")
    return value


def env_truthy(value: str) -> bool:
    """Return whether a string environment value is truthy."""
    return value.strip().lower() in {"1", "true", "yes", "on"}
