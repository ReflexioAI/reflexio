from typing import Any

from reflexio.lib._base import ReflexioBase
from reflexio.models.api_schema.retriever_schema import SetConfigResponse
from reflexio.models.config_schema import Config


class ConfigMixin(ReflexioBase):
    def set_config(self, config: Config | dict) -> SetConfigResponse:
        """Set configuration for the organization.

        Args:
            config (Union[Config, dict]): The configuration to set

        Returns:
            dict: Response containing success status and message
        """
        try:
            configurator = self.request_context.configurator
            if isinstance(config, dict):
                config = configurator.normalize_config_payload(config)
                config = Config(**config)

            # Validate storage connection before setting config.
            # If no storage_config provided, preserve the existing one (callers
            # like get_config() don't expose storage_config for security).
            storage_config = config.storage_config
            if storage_config is None:
                storage_config = configurator.get_current_storage_configuration()
                config.storage_config = storage_config

            # Check if storage config is ready to test
            if not configurator.is_storage_config_ready_to_test(
                storage_config=storage_config
            ):
                return SetConfigResponse(
                    success=False, msg="Storage configuration is incomplete"
                )

            # Test and initialize storage connection
            (
                success,
                error_msg,
            ) = configurator.test_and_init_storage_config(storage_config=storage_config)

            if not success:
                return SetConfigResponse(
                    success=False,
                    msg=f"Failed to validate storage connection: {error_msg}",
                )

            # Only set config if validation passed
            configurator.set_config(config)

            return SetConfigResponse(success=True, msg="Configuration set successfully")
        except Exception as e:
            return SetConfigResponse(
                success=False, msg=f"Failed to set configuration: {str(e)}"
            )

    def get_config(self) -> Config:
        """Get configuration for the organization.

        Returns:
            Config: The current configuration
        """
        return self.request_context.configurator.get_config()

    def current_config_version(self) -> tuple[str, Any] | None:
        """Cheap probe of the persisted config version.

        Delegates to the underlying ``ConfigStorage.get_version()``.
        Used by the per-org Reflexio cache to detect out-of-band
        configuration changes (file edits, replica writes, direct DB
        updates) without doing a full reload on every request.

        Returns:
            tuple[str, Any] | None: ``("file", mtime)`` for file-backed
            storage, ``("db", version)`` for DB-backed storage, or
            ``None`` when probing is unsupported / fails (the cache
            treats ``None`` as "still fresh").
        """
        return self.request_context.configurator.config_storage.get_version()
