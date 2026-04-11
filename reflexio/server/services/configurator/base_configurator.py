from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import BaseModel

from reflexio.models.config_schema import Config, StorageConfig, StorageConfigTest
from reflexio.server.services.configurator.config_storage import ConfigStorage
from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.storage_base import BaseStorage

logger = logging.getLogger(__name__)


class BaseConfigurator(ABC):
    """Abstract base for organization configurators.

    Subclasses must:
      - Set ``_STORAGE_FACTORIES`` and ``_STORAGE_READINESS_CHECKS`` class variables.
      - Implement ``_select_config_storage`` to choose the config persistence backend.

    All shared configuration and storage-dispatch logic lives here so that OS
    and enterprise implementations only override what differs.
    """

    _STORAGE_FACTORIES: ClassVar[dict[type[StorageConfig], Callable[..., BaseStorage]]]
    _STORAGE_READINESS_CHECKS: ClassVar[
        dict[type[StorageConfig], Callable[[Any], bool]]
    ]

    def __init__(
        self,
        org_id: str,
        base_dir: str | None = None,
        config: Config | None = None,
    ) -> None:
        self.org_id = org_id
        self.base_dir = base_dir

        self.config_storage = self._select_config_storage(org_id, base_dir)
        if not config:
            self.config = self.config_storage.load_config()
        else:
            self.config = config

        if not self.config:
            raise ValueError(f"Failed to load configuration for organization {org_id}")

    @abstractmethod
    def _select_config_storage(
        self, org_id: str, base_dir: str | None
    ) -> ConfigStorage:
        """Select and return the appropriate ConfigStorage backend.

        This is the primary extension point: OS uses LocalJson only, while
        enterprise uses a multi-tier priority system.
        """

    # ==========================
    # Configuration
    # ==========================

    def get_config(self) -> Config:
        return self.config

    def get_agent_context(self) -> str:
        context = self.get_config().agent_context_prompt
        if not context:
            return ""
        return context.strip()

    def set_config(self, config: Config) -> None:
        self.config = config
        self.config_storage.save_config(config=config)

    def set_config_by_name(
        self,
        config_name: str,
        config_value: str | int | float | bool | list | dict | BaseModel,
    ) -> None:
        if config_name not in self.config.model_fields:
            raise ValueError(f"Invalid config name: {config_name}")

        setattr(self.config, config_name, config_value)
        self.set_config(config=self.config)

    # ==========================
    # Storage
    # ==========================

    def get_current_storage_configuration(self) -> StorageConfig:
        """Return the currently configured storage config."""
        return self.get_config().storage_config

    def create_storage(self, storage_config: StorageConfig) -> BaseStorage | None:
        """Create a storage from the ``_STORAGE_FACTORIES`` registry.

        Returns None when *storage_config* is None (enterprise cloud mode
        where storage is not yet configured).
        """
        if storage_config is None:
            return None
        factory = self._STORAGE_FACTORIES.get(type(storage_config))
        if factory is None:
            raise ValueError(
                f"No storage factory registered for {type(storage_config).__name__}"
            )
        return factory(self, storage_config)

    def is_storage_configured(self) -> bool:
        """Check whether a valid, non-failed storage option is configured."""
        if not self.is_storage_config_ready_to_test(
            storage_config=self.get_current_storage_configuration(),
        ):
            return False
        return self.get_config().storage_config_test != StorageConfigTest.FAILED

    def is_storage_config_ready_to_test(self, storage_config: StorageConfig) -> bool:
        """Check whether *storage_config* is fully filled in and ready for a test connection."""
        check = self._STORAGE_READINESS_CHECKS.get(type(storage_config))
        return check(storage_config) if check else False

    def test_and_init_storage_config(
        self, storage_config: StorageConfig
    ) -> tuple[bool, str]:
        """Test whether *storage_config* is valid and initialise the storage.

        Returns:
            tuple[bool, str]: (success, message)
        """
        if not self.is_storage_config_ready_to_test(storage_config=storage_config):
            return False, "Storage configuration is not ready to test"

        try:
            storage = self.create_storage(storage_config=storage_config)
            if storage is not None:
                storage.migrate()
                return True, "Storage initialized successfully"
            return False, "Failed to create storage"
        except StorageError as e:
            logger.error("Storage initialization failed: %s", e.message)
            return False, e.message
        except Exception as e:
            logger.error("Storage initialization failed: %s", e)
            return False, str(e)
