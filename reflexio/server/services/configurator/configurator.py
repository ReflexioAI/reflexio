from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from reflexio.models.config_schema import (
    StorageConfig,
    StorageConfigDisk,
    StorageConfigSQLite,
)
from reflexio.server.services.configurator.base_configurator import BaseConfigurator
from reflexio.server.services.configurator.config_storage import ConfigStorage
from reflexio.server.services.configurator.local_file_config_storage import (
    LocalFileConfigStorage,
)
from reflexio.server.services.storage.disk_storage import DiskStorage
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
    # When base_dir is explicitly provided (e.g. tests with temp dirs)
    # and no db_path is configured, use base_dir for the SQLite DB
    # so the storage is isolated from the shared default database.
    db_path = config.db_path
    if db_path is None and configurator.base_dir:
        db_path = str(Path(configurator.base_dir) / "reflexio.db")
    return SQLiteStorage(
        org_id=configurator.org_id,
        db_path=db_path,
        api_key_config=api_key_config,
        llm_config=llm_config,
        enable_document_expansion=enable_document_expansion,
    )


def _create_disk_storage(
    configurator: BaseConfigurator, config: StorageConfigDisk
) -> BaseStorage:
    logger.info("Using disk storage (markdown + QMD) for org %s", configurator.org_id)
    return DiskStorage(
        org_id=configurator.org_id,
        base_dir=configurator.base_dir,
        config=config,
    )


class DefaultConfigurator(BaseConfigurator):
    """OS configurator with LocalJson config storage and Local+SQLite data storage."""

    _STORAGE_FACTORIES: dict[type[StorageConfig], Callable[..., BaseStorage]] = {
        StorageConfigSQLite: _create_sqlite_storage,
        StorageConfigDisk: _create_disk_storage,
    }

    _STORAGE_READINESS_CHECKS: dict[type[StorageConfig], Callable[[Any], bool]] = {
        StorageConfigSQLite: lambda _: True,  # db_path defaults via env var if None
        StorageConfigDisk: lambda c: bool(c.dir_path),
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
