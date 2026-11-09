"""Lightweight canonical environment resolution shared by CLI and server.

No dotenv loading, server initialization, or cached process state: callers pass
an already selected environment mapping, and canonical nonblank values win.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite

from reflexio.removed_env import reject_removed_env

DEFAULT_STORAGE_POOL_SIZE = 10


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


def resolve_storage_pool_size(env: Mapping[str, str]) -> int:
    """Default capacity for shared Supabase SQL pools, not native Postgres."""
    return _positive_int(env, "REFLEXIO_STORAGE_POOL_SIZE", DEFAULT_STORAGE_POOL_SIZE)


@dataclass(frozen=True)
class PostgresPoolDefaults:
    """Historical native capacity; an unset reader inherits the writer size."""

    pool_size: int
    read_pool_size: int | None


# Frozen generation9 capacity data, not a current execution default. Keep this
# independent of model defaults so a future tuning change cannot rewrite the
# predecessor's effective capacity during a skipped-version upgrade.
_LEGACY_NATIVE_POOL_DEFAULTS = PostgresPoolDefaults(
    pool_size=10,
    read_pool_size=None,
)


def resolve_postgres_pool_defaults(env: Mapping[str, str]) -> PostgresPoolDefaults:
    """Interpret the predecessor's native policy for transition or recovery.

    Blank means unset. Invalid supplied values fail rather than silently
    changing capacity. Defaults are frozen generation9 data; current execution
    owns its canonical storage_config instead. No process state is read/cached.
    """
    # These historical inputs no longer select execution. Still refuse invalid
    # supplied values instead of silently converting a broken predecessor.
    for name in (
        "REFLEXIO_POSTGRES_POOL_ACQUIRE_TIMEOUT",
        "REFLEXIO_POSTGRES_READ_POOL_ACQUIRE_TIMEOUT",
    ):
        raw = env.get(name, "").strip()
        if raw:
            try:
                value = float(raw)
            except ValueError:
                raise ValueError(f"{name} must be finite and positive") from None
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
    return PostgresPoolDefaults(
        pool_size=_positive_int(
            env, "REFLEXIO_POSTGRES_POOL_SIZE", _LEGACY_NATIVE_POOL_DEFAULTS.pool_size
        ),
        read_pool_size=(
            _positive_int(
                env,
                "REFLEXIO_POSTGRES_READ_POOL_SIZE",
                _LEGACY_NATIVE_POOL_DEFAULTS.pool_size,
            )
            if env.get("REFLEXIO_POSTGRES_READ_POOL_SIZE", "").strip()
            else None
        ),
    )


def resolve_data_db_url(env: Mapping[str, str]) -> str:
    """Resolve the canonical data target; never fall back to another DSN."""
    reject_removed_env(env, canonical="DATA_DB_URL")
    return env.get("DATA_DB_URL", "").strip()


POSTGRES_POOL_ENV_FIELDS = {
    "REFLEXIO_POSTGRES_POOL_SIZE": "pool_size",
    "REFLEXIO_POSTGRES_POOL_ACQUIRE_TIMEOUT": "pool_acquire_timeout",
    "REFLEXIO_POSTGRES_READ_POOL_SIZE": "read_pool_size",
    "REFLEXIO_POSTGRES_READ_POOL_ACQUIRE_TIMEOUT": "read_pool_acquire_timeout",
}


def reject_postgres_pool_conflicts(
    env: Mapping[str, str], selected: PostgresPoolDefaults
) -> None:
    """Old assignments are assertions, never a current source of pool policy.

    Compare only supplied nonblank assignments, including effective reader
    inheritance. The historical parser validates their shapes without using its
    defaults to fill any part of the selected canonical policy.
    """
    supplied = asdict(resolve_postgres_pool_defaults(env))
    effective = asdict(selected)
    if effective["read_pool_size"] is None:
        effective["read_pool_size"] = selected.pool_size
    for name, field in POSTGRES_POOL_ENV_FIELDS.items():
        if field not in effective:
            continue  # Validated retired clock, with no current policy to compare.
        if env.get(name, "").strip() and supplied[field] != effective[field]:
            raise ConfigurationMigrationError(
                "Obsolete native pool assignments conflict with saved storage_config. "
                "Select the intended canonical pool policy and remove obsolete "
                "assignments before restarting. No pool policy was imported."
            )
