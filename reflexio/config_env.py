"""Lightweight canonical environment resolution shared by CLI and server.

No dotenv loading, server initialization, or cached process state: callers pass
an already selected environment mapping, and canonical nonblank values win.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

DEFAULT_STORAGE_POOL_SIZE = 10
DEFAULT_STORAGE_POOL_ACQUIRE_TIMEOUT = 30.0


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
    """Resolve one data database target, accepting backend-specific old names.

    Blank values are unset. With no selected backend, retain the historical
    Supabase-then-Postgres fallback used by import-time compatibility constants.
    A selected backend never falls through to the other backend's old name.
    """
    canonical = env.get("DATA_DB_URL", "").strip()
    if canonical:
        return canonical
    backend = env.get("REFLEXIO_STORAGE", "").strip().lower()
    if backend == "postgres":
        return env.get("POSTGRES_DB_URL", "").strip()
    if backend == "supabase":
        return env.get("DATA_SUPABASE_DB_URL", "").strip()
    return (
        env.get("DATA_SUPABASE_DB_URL", "").strip()
        or env.get("POSTGRES_DB_URL", "").strip()
    )
