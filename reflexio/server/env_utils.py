"""Small helpers for parsing environment-backed settings."""

from __future__ import annotations

from collections.abc import Mapping


def env_get(env: Mapping[str, str], name: str) -> str:
    """Return a stripped environment value or an empty string."""
    return str(env.get(name, "")).strip()


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
