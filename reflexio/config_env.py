"""Lightweight canonical environment resolution shared by CLI and server.

No dotenv loading, server initialization, or cached process state: callers pass
an already selected environment mapping, and canonical nonblank values win.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from reflexio.removed_env import reject_removed_env

DEFAULT_STORAGE_POOL_SIZE = 10
DEFAULT_STORAGE_POOL_ACQUIRE_TIMEOUT = 30.0


class StoredConfigurationError(ValueError):
    """The saved configuration must be corrected before it can load."""


class ConfigurationMigrationError(StoredConfigurationError):
    """An explicit migration is required before this saved policy can load."""


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a positive integer") from None
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_seconds(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be finite and positive") from None
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def resolve_storage_pool_size(env: Mapping[str, str]) -> int:
    """Default capacity for shared Supabase SQL pools, not native Postgres."""
    return _positive_int(env, "REFLEXIO_STORAGE_POOL_SIZE", DEFAULT_STORAGE_POOL_SIZE)


@dataclass(frozen=True)
class PostgresPoolDefaults:
    """Native bootstrap fields; an unset reader inherits the writer policy."""

    pool_size: int
    pool_acquire_timeout: float
    read_pool_size: int | None
    read_pool_acquire_timeout: float | None


def resolve_postgres_pool_defaults(env: Mapping[str, str]) -> PostgresPoolDefaults:
    """Resolve native pool defaults without reading or caching process state.

    Blank means unset. Invalid supplied values fail rather than silently
    changing capacity. REFLEXIO_STORAGE_POOL_SIZE does not affect this backend;
    explicit per-target config overrides these bootstrap/worker defaults.
    """
    return PostgresPoolDefaults(
        pool_size=_positive_int(
            env, "REFLEXIO_POSTGRES_POOL_SIZE", DEFAULT_STORAGE_POOL_SIZE
        ),
        pool_acquire_timeout=_positive_seconds(
            env,
            "REFLEXIO_POSTGRES_POOL_ACQUIRE_TIMEOUT",
            DEFAULT_STORAGE_POOL_ACQUIRE_TIMEOUT,
        ),
        read_pool_size=(
            _positive_int(
                env, "REFLEXIO_POSTGRES_READ_POOL_SIZE", DEFAULT_STORAGE_POOL_SIZE
            )
            if env.get("REFLEXIO_POSTGRES_READ_POOL_SIZE", "").strip()
            else None
        ),
        read_pool_acquire_timeout=(
            _positive_seconds(
                env,
                "REFLEXIO_POSTGRES_READ_POOL_ACQUIRE_TIMEOUT",
                DEFAULT_STORAGE_POOL_ACQUIRE_TIMEOUT,
            )
            if env.get("REFLEXIO_POSTGRES_READ_POOL_ACQUIRE_TIMEOUT", "").strip()
            else None
        ),
    )


def resolve_data_db_url(env: Mapping[str, str]) -> str:
    """Resolve the canonical data target; never fall back to another DSN."""
    reject_removed_env(env, canonical="DATA_DB_URL")
    return env.get("DATA_DB_URL", "").strip()
