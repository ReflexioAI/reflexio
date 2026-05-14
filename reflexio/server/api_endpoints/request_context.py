from fastapi import Depends

from reflexio.server._auth import default_get_org_id
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


def get_request_context(
    org_id: str = Depends(default_get_org_id),
) -> RequestContext:
    """FastAPI dependency that builds a RequestContext for the calling org.

    The ``org_id`` parameter is resolved through :func:`default_get_org_id`,
    which enterprise deployments override via ``app.dependency_overrides`` in
    :func:`create_app` to plug in real auth (e.g. Bearer-token org resolution).
    Tests that need a fixed context override this function directly.

    Args:
        org_id (str): Organisation identifier resolved by the auth layer.

    Returns:
        RequestContext: Fully initialised context with storage attached.
    """
    return RequestContext(org_id=org_id)
