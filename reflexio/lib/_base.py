from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from reflexio.server.services.pre_retrieval import QueryReformulator


from reflexio.models.config_schema import SearchMode
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.model_defaults import (
    ModelRole,
    resolve_model_name,
)
from reflexio.server.services.configurator.base_configurator import BaseConfigurator
from reflexio.server.services.storage.storage_base import BaseStorage
from reflexio.server.site_var.site_var_manager import SiteVarManager

logger = logging.getLogger(__name__)

# Error message for when storage is not configured
STORAGE_NOT_CONFIGURED_MSG = (
    "Storage not configured. Please configure storage in settings first."
)


def _require_storage[T: BaseModel](
    response_type: type[T], *, msg_field: str = "message"
) -> Callable[..., Callable[..., T]]:
    """Decorator that guards a Reflexio method with storage-configured check and error handling.

    Args:
        response_type: The Pydantic response model to return on failure
        msg_field: Name of the message field on the response ('message' or 'msg')
    """

    def decorator(method: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(method)
        def wrapper(self: Reflexio, *args: Any, **kwargs: Any) -> T:
            if not self._is_storage_configured():
                return response_type.model_validate(
                    {"success": False, msg_field: STORAGE_NOT_CONFIGURED_MSG}
                )
            try:
                return method(self, *args, **kwargs)
            except Exception as e:
                return response_type.model_validate(
                    {"success": False, msg_field: str(e)}
                )

        return wrapper

    return decorator


class ReflexioBase:
    def __init__(
        self,
        org_id: str,
        storage_base_dir: str | None = None,
        configurator: BaseConfigurator | None = None,
    ) -> None:
        """Initialize Reflexio with organization ID and storage directory.

        Args:
            org_id (str): Organization ID
            storage_base_dir (str, optional): Base directory for storing data
        """
        self.org_id = org_id
        self.storage_base_dir = storage_base_dir
        self.request_context = RequestContext(
            org_id=org_id, storage_base_dir=storage_base_dir, configurator=configurator
        )

        # Create single LLM client for all services
        model_setting = SiteVarManager().get_site_var("llm_model_setting")
        site_var = model_setting if isinstance(model_setting, dict) else {}

        # Get API key config and LLM config from configuration if available
        config = self.request_context.configurator.get_config()
        api_key_config = config.api_key_config if config else None
        config_llm_config = config.llm_config if config else None

        generation_model_name = resolve_model_name(
            ModelRole.GENERATION,
            site_var_value=site_var.get("default_generation_model_name"),
            config_override=config_llm_config.generation_model_name
            if config_llm_config
            else None,
            api_key_config=api_key_config,
        )

        llm_config = LiteLLMConfig(
            model=generation_model_name,
            api_key_config=api_key_config,
        )
        self.llm_client = LiteLLMClient(llm_config)

    def _is_storage_configured(self) -> bool:
        """Check if storage is configured and available.

        Returns:
            bool: True if storage is configured, False otherwise
        """
        return self.request_context.is_storage_configured()

    def _get_storage(self) -> BaseStorage:
        """Return storage, raising if not configured."""
        storage = self.request_context.storage
        if storage is None:
            raise RuntimeError(STORAGE_NOT_CONFIGURED_MSG)
        return storage

    def _get_query_reformulator(self) -> QueryReformulator:
        """Lazily create and cache a QueryReformulator instance.

        Returns:
            QueryReformulator: Cached reformulator instance
        """
        if not hasattr(self, "_query_reformulator"):
            from reflexio.server.services.pre_retrieval import QueryReformulator

            # Resolve pre_retrieval_model_name: config override → site var → auto-detect
            model_setting = SiteVarManager().get_site_var("llm_model_setting")
            site_var = model_setting if isinstance(model_setting, dict) else {}
            config = self.request_context.configurator.get_config()
            config_llm_config = config.llm_config if config else None
            api_key_config = config.api_key_config if config else None

            pre_retrieval_model_name = resolve_model_name(
                ModelRole.PRE_RETRIEVAL,
                site_var_value=site_var.get("pre_retrieval_model_name"),
                config_override=config_llm_config.pre_retrieval_model_name
                if config_llm_config
                else None,
                api_key_config=api_key_config,
            )

            self._query_reformulator = QueryReformulator(
                llm_client=self.llm_client,
                prompt_manager=self.request_context.prompt_manager,
                model_name=pre_retrieval_model_name,
            )
        return self._query_reformulator

    def _maybe_get_query_embedding(
        self, query: str | None, search_mode: SearchMode
    ) -> list[float] | None:
        """Generate a query embedding if the search mode needs one and the storage supports it.

        Args:
            query (str, optional): The search query text
            search_mode (SearchMode): Requested search mode

        Returns:
            list[float] or None: Query embedding vector, or None if not needed/available
        """
        if not query or search_mode == SearchMode.FTS:
            return None
        storage = self._get_storage()
        if not hasattr(storage, "_get_embedding"):
            return None
        try:
            return storage._get_embedding(query, purpose="query")  # type: ignore[reportAttributeAccessIssue]
        except Exception as e:
            logger.warning("Failed to generate query embedding due to %s — falling back to FTS", e)
            return None

    def _reformulate_query(
        self, query: str | None, enabled: bool = False
    ) -> str | None:
        """Reformulate a search query using the query reformulator if enabled.

        Returns the reformulated query, or None if reformulation is disabled,
        the query is empty, or reformulation fails.

        Args:
            query (str, optional): The original search query
            enabled (bool): Whether query reformulation is enabled for this request

        Returns:
            str or None: Reformulated query, or None to use original query
        """
        if not query or not enabled:
            return None

        reformulator = self._get_query_reformulator()
        result = reformulator.rewrite(query)
        # Only return if different from original
        if result.standalone_query != query:
            return result.standalone_query
        return None


if TYPE_CHECKING:
    from reflexio.lib.reflexio_lib import Reflexio
