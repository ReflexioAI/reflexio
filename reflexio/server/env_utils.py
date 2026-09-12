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


_TRUE = "true"
_FALSE = "false"


class EnvBoolError(ValueError):
    """A boolean environment variable carried a value that is not true/false."""


def env_bool(
    name: str,
    *,
    default: bool,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Parse a boolean environment variable, refusing anything ambiguous.

    Accepts ``true`` / ``false`` only, case-insensitively and after stripping.
    Unset or blank resolves to ``default``, matching :func:`env_str`'s
    "set to blank behaves like unset" rule. **Every other value raises.**

    Raising is the point, and it is what this codebase previously lacked. The
    predecessor (``env_truthy``) was ``value.lower() in {...}``, so anything
    unrecognised returned ``False`` -- meaning ``REFLEXIO_REQUIRE_DATA_DB=ture``
    silently DISABLED a guard rather than failing. A typo that switches a safety
    knob off without a word is the same silent-green class as a CI lane that
    never runs.

    Erroring on unrecognised input is the universal convention: Go's
    ``strconv.ParseBool`` states "Any other value returns an error", and Pydantic
    raises ``bool_parsing``. Neither defaults.

    The accepted SET is deliberately narrower than either, which both admit
    ``1``/``0``. ``.env.template`` documents ~28 variables as
    ``options: true | false``, so accepting more would leave that documentation a
    half-truth -- and the direction of travel is narrowing, not widening: YAML
    1.2 removed ``yes``/``no``/``on``/``off`` as booleans outright after ``NO``
    (Norway) silently parsed as false.

    Args:
        name (str): Environment variable name, used in the error message.
        default (bool): Value returned when the variable is unset or blank.
        env (Mapping[str, str] | None): Mapping to read; defaults to os.environ.

    Returns:
        bool: The parsed value, or ``default`` when unset or blank.

    Raises:
        EnvBoolError: The variable is set to anything other than true/false.
    """
    raw = (env if env is not None else os.environ).get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value == _TRUE:
        return True
    if value == _FALSE:
        return False
    raise EnvBoolError(
        f"{name} must be {_TRUE!r} or {_FALSE!r} (got {raw.strip()!r}). "
        f"Values such as '1', '0', 'yes' and 'on' are deliberately NOT accepted: "
        f"they were parsed inconsistently across this codebase, and an "
        f"unrecognised value used to read as false without any error."
    )


def env_truthy(value: str) -> bool:
    """Return whether a string environment value is truthy.

    Deprecated in favour of :func:`env_bool`, which takes the variable NAME and
    can therefore name it when refusing a bad value. Retained because it is a
    public export of this package; it keeps the historical permissive set so an
    external caller's behaviour does not change silently.
    """
    return value.strip().lower() in {"1", "true", "yes", "on"}
