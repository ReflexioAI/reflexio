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
            if isinstance(config, dict):
                config = Config(**config)

            # Validate storage connection before setting config.
            # If no storage_config provided, preserve the existing one (callers
            # like get_config() don't expose storage_config for security).
            storage_config = config.storage_config
            if storage_config is None:
                storage_config = self.request_context.configurator.get_current_storage_configuration()
                config.storage_config = storage_config

            # Check if storage config is ready to test
            if not self.request_context.configurator.is_storage_config_ready_to_test(
                storage_config=storage_config
            ):
                return SetConfigResponse(
                    success=False, msg="Storage configuration is incomplete"
                )

            # Test and initialize storage connection
            (
                success,
                error_msg,
            ) = self.request_context.configurator.test_and_init_storage_config(
                storage_config=storage_config
            )

            if not success:
                return SetConfigResponse(
                    success=False,
                    msg=f"Failed to validate storage connection: {error_msg}",
                )

            # Only set config if validation passed
            self.request_context.configurator.set_config(config)

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
