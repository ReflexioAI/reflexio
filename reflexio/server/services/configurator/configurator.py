from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from reflexio.models.config_schema import (
    StorageConfig,
    StorageConfigSQLite,
)
from reflexio.server.services.configurator.base_configurator import BaseConfigurator
from reflexio.server.services.configurator.config_storage import ConfigStorage
from reflexio.server.services.configurator.local_file_config_storage import (
    LocalFileConfigStorage,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import BaseStorage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Storage factory functions — one per StorageConfig type
# ---------------------------------------------------------------------------


def _create_sqlite_storage(
    configurator: BaseConfigurator, config: StorageConfigSQLite
) -> BaseStorage:
    logger.info("Using SQLite storage for org %s", configurator.org_id)
    full_config = configurator.get_config()
    api_key_config = full_config.api_key_config if full_config else None
    llm_config = full_config.llm_config if full_config else None
    enable_document_expansion = (
        full_config.enable_document_expansion if full_config else False
    )
    db_path = resolve_oss_sqlite_db_path(configurator, config)
    return SQLiteStorage(
        org_id=configurator.org_id,
        db_path=db_path,
        api_key_config=api_key_config,
        llm_config=llm_config,
        enable_document_expansion=enable_document_expansion,
    )


def resolve_oss_sqlite_db_path(
    configurator: BaseConfigurator, storage_config: StorageConfig
) -> str | None:
    """Resolve the SQLite DB path used by the OSS storage factory.

    Args:
        configurator: Active configurator whose ``base_dir`` can isolate
            test/local storage.
        storage_config: Active storage configuration.

    Returns:
        The final SQLite DB path, or None when the storage config is not SQLite.
    """
    if not isinstance(storage_config, StorageConfigSQLite):
        return None
    if storage_config.db_path is not None:
        return storage_config.db_path
    if configurator.base_dir:
        return str(Path(configurator.base_dir) / "reflexio.db")

    from reflexio.server import LOCAL_STORAGE_PATH

    return str(Path(LOCAL_STORAGE_PATH) / "reflexio.db")


class DefaultConfigurator(BaseConfigurator):
    """OS configurator with LocalJson config storage and SQLite data storage."""

    _STORAGE_FACTORIES: dict[type[StorageConfig], Callable[..., BaseStorage]] = {
        StorageConfigSQLite: _create_sqlite_storage,
    }

    _STORAGE_READINESS_CHECKS: dict[type[StorageConfig], Callable[[Any], bool]] = {
        StorageConfigSQLite: lambda _: True,  # db_path defaults via env var if None
    }

    def _select_config_storage(
        self, org_id: str, base_dir: str | None
    ) -> ConfigStorage:
        if base_dir:
            return LocalFileConfigStorage(org_id=org_id, base_dir=base_dir)
        return LocalFileConfigStorage(org_id=org_id)


# ---------------------------------------------------------------------------
# Configurator class registry — allows enterprise to swap in its own class
# ---------------------------------------------------------------------------

_configurator_class: type[BaseConfigurator] = DefaultConfigurator


def set_configurator_class(cls: type[BaseConfigurator]) -> None:
    """Register a configurator class to be used by RequestContext."""
    global _configurator_class  # noqa: PLW0603
    _configurator_class = cls


def get_configurator_class() -> type[BaseConfigurator]:
    """Return the currently registered configurator class."""
    return _configurator_class
