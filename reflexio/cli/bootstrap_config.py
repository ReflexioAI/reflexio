"""CLI bootstrap config: resolve and persist storage settings without a running server.

Provides the priority chain: CLI flag > env var (.env) > config file > default.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer

from reflexio.config_env import (
    POSTGRES_POOL_ENV_FIELDS,
    PostgresPoolDefaults,
    reject_postgres_pool_conflicts,
    resolve_data_db_url,
    resolve_postgres_pool_defaults,
)
from reflexio.config_migration import StoredConfigurationError

from .paths import reflexio_home

logger = logging.getLogger(__name__)

_VALID_STORAGE_BACKENDS = frozenset({"sqlite", "supabase", "postgres"})
_DEFAULT_ORG_ID = "self-host-org"
_DEFAULT_STORAGE = "sqlite"


def default_org_id() -> str:
    """Resolve the default org id, honoring ``REFLEXIO_DEFAULT_ORG_ID``.

    Mirrors :func:`reflexio.server.auth.default_get_org_id` so the local CLI
    helpers (``setup``, ``config show`` / ``config local``) resolve the same
    ``config_<org>.json`` the running no-auth server reads. Uses ``os.environ``
    directly (not ``env_utils``) to keep this module's import lightweight — see
    the ``_TYPE_TO_BACKEND`` note above.
    """
    return os.environ.get("REFLEXIO_DEFAULT_ORG_ID", "").strip() or _DEFAULT_ORG_ID


# Maps StorageConfig subclass to backend string.  Lazy-imported in functions
# that need it so module-level import stays lightweight (no Pydantic at import).
_TYPE_TO_BACKEND: dict[type, str] = {}  # populated on first use


def _data_db_url_from_env() -> str:
    """Resolve the canonical data DB URL, accepting the legacy alias."""
    return resolve_data_db_url(os.environ)


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
    return _config_dir(base_dir) / f"config_{default_org_id()}.json"


def load_storage_from_config(
    org_id: str | None = None,
    *,
    base_dir: str | None = None,
) -> str | None:
    """Read storage type from the local config file.

    Args:
        org_id: Organization ID for the config file name. Defaults to the
            ``REFLEXIO_DEFAULT_ORG_ID``-aware default org.
        base_dir: Override base directory (for testing). If None, uses ~/.reflexio/.

    Returns:
        Storage backend string ("sqlite", "supabase", "postgres") or None if
        no config file exists or storage_config is unset.
    """
    org_id = org_id or default_org_id()
    config_path = _config_dir(base_dir) / f"config_{org_id}.json"
    if not config_path.exists():
        return None

    try:
        from reflexio.server.services.configurator.local_file_config_storage import (
            LocalFileConfigStorage,
        )

        storage = LocalFileConfigStorage(org_id, base_dir=base_dir)
        config = storage.load_config()
    except StoredConfigurationError:
        raise
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
    org_id: str | None = None,
    *,
    base_dir: str | None = None,
) -> None:
    """Update storage_config in the local config file.

    The caller owns stopping incompatible writers first. Select and prepare
    only ``storage_config`` under its file publication lock. Old explicit native
    selections retain their historical environment policy once; current files
    own their canonical budgets. Selecting SQLite again preserves its database
    path. Unrelated raw fields, including extension fields, are preserved.

    Args:
        storage_type: Backend name ("sqlite", "supabase", "postgres").
        org_id: Organization ID for the config file name. Defaults to the
            ``REFLEXIO_DEFAULT_ORG_ID``-aware default org.
        base_dir: Override base directory (for testing).
    """
    from reflexio.models.config_schema import (
        StorageConfigPostgres,
        StorageConfigSQLite,
        StorageConfigSupabase,
        validate_stored_config,
    )
    from reflexio.server.services.configurator.local_file_config_storage import (
        LocalFileConfigStorage,
    )

    org_id = org_id or default_org_id()
    replacement = None
    legacy_pool_fields: dict[str, Any] = {}
    legacy_environment = {
        name: os.environ.get(name, "") for name in POSTGRES_POOL_ENV_FIELDS
    }
    match storage_type:
        case "sqlite":
            replacement = StorageConfigSQLite()
        case "supabase":
            db_url = _data_db_url_from_env()
            if db_url:
                replacement = StorageConfigSupabase(db_url=db_url)
            else:
                logger.warning(
                    "Supabase storage requested but DATA_DB_URL is missing. "
                    "Keeping existing storage config."
                )
        case "postgres":
            db_url = os.environ.get("DATA_DB_URL", "").strip()
            schema = os.environ.get("REFLEXIO_POSTGRES_SCHEMA", "").strip()
            postgres_kwargs: dict[str, Any] = {
                "db_url": db_url,
                "schema": schema or None,
                "read_db_url": os.environ.get(
                    "REFLEXIO_POSTGRES_READ_DB_URL", ""
                ).strip()
                or None,
            }
            # Validate the selected old source before creating/preparing a file.
            legacy_pool_fields = asdict(
                resolve_postgres_pool_defaults(legacy_environment)
            )
            if db_url:
                replacement = StorageConfigPostgres(**postgres_kwargs)
            else:
                logger.warning(
                    "Postgres storage requested but DATA_DB_URL "
                    "is missing. Keeping existing storage config."
                )
        case _:
            raise ValueError(f"Unknown storage type: {storage_type}")

    if replacement is None:
        return

    from reflexio.config_file import NATIVE_POOL_AUTHORITY_VERSION, LocalConfigFile

    storage_obj = LocalFileConfigStorage(org_id, base_dir=base_dir)

    def update_storage(document: dict[str, Any], generation: int) -> None:
        existing = validate_stored_config(document).storage_config
        if storage_type == "sqlite" and isinstance(existing, StorageConfigSQLite):
            # A backend selection does not request a different SQLite database.
            return
        selected = replacement
        if isinstance(selected, StorageConfigPostgres):
            fields = selected.model_dump(mode="json")
            if generation < NATIVE_POOL_AUTHORITY_VERSION:
                # An old explicit native backend selection reconstructed these
                # budgets from its selected environment, even from SQLite.
                fields.update(legacy_pool_fields)
            else:
                if isinstance(existing, StorageConfigPostgres):
                    fields.update(
                        existing.model_dump(
                            mode="json", include=set(POSTGRES_POOL_ENV_FIELDS.values())
                        )
                    )
                current = StorageConfigPostgres.model_validate(fields)
                reject_postgres_pool_conflicts(
                    legacy_environment,
                    PostgresPoolDefaults(
                        pool_size=current.pool_size,
                        pool_acquire_timeout=current.pool_acquire_timeout,
                        read_pool_size=current.read_pool_size,
                        read_pool_acquire_timeout=current.read_pool_acquire_timeout,
                    ),
                )
                if any(value.strip() for value in legacy_environment.values()):
                    logger.warning(
                        "Obsolete native pool assignments no longer select capacity; "
                        "remove them after verifying canonical storage_config."
                    )
            selected = StorageConfigPostgres.model_validate(fields)
        document["storage_config"] = selected.model_dump(mode="json")

    initial = (
        storage_obj.get_default_config().model_dump(mode="json")
        if not Path(storage_obj.config_file).exists()
        else None
    )
    LocalConfigFile(storage_obj.config_file).prepare(
        edit=update_storage, initial_document=initial
    )


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
