from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.configurator.base_configurator import BaseConfigurator
from reflexio.server.services.configurator.configurator import get_configurator_class


class RequestContext:
    def __init__(
        self,
        org_id: str,
        storage_base_dir: str | None = None,
        configurator: BaseConfigurator | None = None,
    ):
        self.org_id = str(org_id)
        self.storage_base_dir = storage_base_dir
        cls = get_configurator_class()
        self.configurator = configurator or cls(org_id, base_dir=storage_base_dir)
        self.prompt_manager = PromptManager()
        self.storage = self.configurator.create_storage(
            storage_config=self.configurator.get_current_storage_configuration(),
        )

    def is_storage_configured(self) -> bool:
        """Check if storage is configured and available.

        Returns:
            bool: True if storage is configured, False otherwise
        """
        return self.storage is not None
