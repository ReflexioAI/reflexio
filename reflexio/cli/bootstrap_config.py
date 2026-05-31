"""CLI bootstrap config: resolve and persist storage settings without a running server.

Provides the priority chain: CLI flag > env var (.env) > config file > default.
See docs_for_coding_agent/cli-config-state-management.md for the full design.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import typer

from .paths import reflexio_home

logger = logging.getLogger(__name__)

_VALID_STORAGE_BACKENDS = frozenset({"sqlite", "supabase", "postgres"})
_DEFAULT_ORG_ID = "self-host-org"
_DEFAULT_STORAGE = "sqlite"

# Maps StorageConfig subclass to backend string.  Lazy-imported in functions
# that need it so module-level import stays lightweight (no Pydantic at import).
_TYPE_TO_BACKEND: dict[type, str] = {}  # populated on first use


def _data_db_url_from_env() -> str:
    """Resolve the canonical data DB URL, accepting the legacy alias."""
    return os.environ.get("DATA_DB_URL", "") or os.environ.get(
        "DATA_SUPABASE_DB_URL", ""
    )


def _ensure_type_map() -> dict[type, str]:
    """Lazy-build the StorageConfig type → backend string map."""
    if not _TYPE_TO_BACKEND:
        from reflexio.models.config_schema import (
            StorageConfigPostgres,
            StorageConfigSQLite,
            StorageConfigSupabase,
        )

        _TYPE_TO_BACKEND.update(
            {
                StorageConfigSQLite: "sqlite",
                StorageConfigSupabase: "supabase",
                StorageConfigPostgres: "postgres",
            }
        )
    return _TYPE_TO_BACKEND


def _config_dir(base_dir: str | None = None) -> Path:
    """Return the config directory path."""
    if base_dir:
        return Path(base_dir) / "configs"
    return reflexio_home() / "configs"


def default_config_path(base_dir: str | None = None) -> Path:
    """Return the local config file path for the default self-host org.

    Shared helper so CLI display commands (e.g. ``config show``,
    ``config local``) resolve the same path without duplicating the
    ``<dir>/config_<org>.json`` convention.

    Args:
        base_dir: Override base directory (for testing). If None, uses ~/.reflexio/.

    Returns:
        Path: Absolute path to the default-org config file (may not exist).
    """
    return _config_dir(base_dir) / f"config_{_DEFAULT_ORG_ID}.json"


def load_storage_from_config(
    org_id: str = _DEFAULT_ORG_ID,
    *,
    base_dir: str | None = None,
) -> str | None:
    """Read storage type from the local config file.

    Args:
        org_id: Organization ID for the config file name.
        base_dir: Override base directory (for testing). If None, uses ~/.reflexio/.

    Returns:
        Storage backend string ("sqlite", "supabase", "postgres") or None if
        no config file exists or storage_config is unset.
    """
    config_path = _config_dir(base_dir) / f"config_{org_id}.json"
    if not config_path.exists():
        return None

    try:
        from reflexio.server.services.configurator.local_file_config_storage import (
            LocalFileConfigStorage,
        )

        storage = LocalFileConfigStorage(org_id, base_dir=base_dir)
        config = storage.load_config()
    except Exception:
        logger.debug("Failed to load config from %s", config_path, exc_info=True)
        return None

    sc = config.storage_config
    if sc is None:
        return None

    type_map = _ensure_type_map()
    return type_map.get(type(sc))


def save_storage_to_config(
    storage_type: str,
    org_id: str = _DEFAULT_ORG_ID,
    *,
    base_dir: str | None = None,
) -> None:
    """Update storage_config in the local config file.

    Loads the existing config, replaces only ``storage_config``, and saves.
    All other fields (extractors, api_keys, etc.) are preserved.

    Args:
        storage_type: Backend name ("sqlite", "supabase", "postgres").
        org_id: Organization ID for the config file name.
        base_dir: Override base directory (for testing).
    """
    from reflexio.models.config_schema import (
        StorageConfigPostgres,
        StorageConfigSQLite,
        StorageConfigSupabase,
    )
    from reflexio.server.services.configurator.local_file_config_storage import (
        LocalFileConfigStorage,
    )

    storage_obj = LocalFileConfigStorage(org_id, base_dir=base_dir)
    config = storage_obj.load_config()

    match storage_type:
        case "sqlite":
            config.storage_config = StorageConfigSQLite()
        case "supabase":
            url = os.environ.get("DATA_SUPABASE_URL", "")
            key = os.environ.get("DATA_SUPABASE_KEY", "")
            db_url = _data_db_url_from_env()
            if url and key and db_url:
                config.storage_config = StorageConfigSupabase(
                    url=url, key=key, db_url=db_url
                )
            else:
                logger.warning(
                    "Supabase storage requested but credentials are missing "
                    "(DATA_SUPABASE_URL, DATA_SUPABASE_KEY, DATA_DB_URL). "
                    "Keeping existing storage config."
                )
        case "postgres":
            db_url = os.environ.get("DATA_DB_URL", "").strip()
            schema = os.environ.get("REFLEXIO_POSTGRES_SCHEMA", "").strip()
            pool_size_raw = os.environ.get("REFLEXIO_POSTGRES_POOL_SIZE", "").strip()
            pool_size = int(pool_size_raw) if pool_size_raw.isdigit() else 10
            if pool_size < 1:
                logger.warning(
                    "Invalid REFLEXIO_POSTGRES_POOL_SIZE=%r (must be >= 1); using default",
                    pool_size_raw,
                )
                pool_size = 10
            timeout_raw = os.environ.get(
                "REFLEXIO_POSTGRES_POOL_ACQUIRE_TIMEOUT", ""
            ).strip()
            postgres_kwargs: dict[str, Any] = {
                "db_url": db_url,
                "schema": schema or None,
                "pool_size": pool_size,
            }
            if timeout_raw:
                try:
                    timeout = float(timeout_raw)
                except ValueError:
                    logger.warning(
                        "Invalid REFLEXIO_POSTGRES_POOL_ACQUIRE_TIMEOUT=%r; using default",
                        timeout_raw,
                    )
                else:
                    if timeout > 0:
                        postgres_kwargs["pool_acquire_timeout"] = timeout
                    else:
                        logger.warning(
                            "Invalid REFLEXIO_POSTGRES_POOL_ACQUIRE_TIMEOUT=%r "
                            "(must be > 0); using default",
                            timeout_raw,
                        )
            if db_url:
                config.storage_config = StorageConfigPostgres(**postgres_kwargs)
            else:
                logger.warning(
                    "Postgres storage requested but DATA_DB_URL "
                    "is missing. Keeping existing storage config."
                )
        case _:
            raise ValueError(f"Unknown storage type: {storage_type}")

    storage_obj.save_config(config)


def resolve_storage(cli_flag: str | None) -> str:
    """Resolve storage backend using priority: CLI flag > env var > config file > default.

    Do NOT use Typer's ``envvar=`` binding for ``--storage``. This function
    handles the full resolution chain so callers can distinguish explicit CLI
    flags (``cli_flag is not None``) from implicit fallback (``cli_flag is None``)
    for write-back decisions.

    Args:
        cli_flag: Value from ``--storage`` flag, or ``None`` if not passed.

    Returns:
        Resolved storage backend string.

    Raises:
        typer.BadParameter: If the resolved value is not a known backend.
    """
    # 1. CLI flag (explicit user intent)
    if cli_flag is not None:
        result = cli_flag.lower()
        if result not in _VALID_STORAGE_BACKENDS:
            raise typer.BadParameter(
                f"Invalid storage backend '{cli_flag}'. "
                f"Must be one of: {', '.join(sorted(_VALID_STORAGE_BACKENDS))}"
            )
        return result

    # 2. Environment variable (from .env or shell)
    env_val = os.environ.get("REFLEXIO_STORAGE")
    if env_val:
        env_norm = env_val.lower()
        if env_norm in _VALID_STORAGE_BACKENDS:
            return env_norm
        # Unknown value (e.g., legacy "disk" or a typo). Don't silently fall
        # through — surface a warning so operators notice that their env var
        # was ignored. Common case: REFLEXIO_STORAGE=disk left over from a
        # release that supported the now-removed disk backend.
        legacy_hint = (
            " The 'disk' backend was removed; migrate to 'sqlite'."
            if env_norm == "disk"
            else ""
        )
        logger.warning(
            "Ignoring unsupported REFLEXIO_STORAGE=%r; falling back to config/default. "
            "Supported: %s.%s",
            env_val,
            ", ".join(sorted(_VALID_STORAGE_BACKENDS)),
            legacy_hint,
        )

    # 3. Config file
    from_config = load_storage_from_config()
    if from_config and from_config in _VALID_STORAGE_BACKENDS:
        return from_config

    # 4. Hardcoded default
    return _DEFAULT_STORAGE
