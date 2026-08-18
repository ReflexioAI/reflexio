"""
LiteLLM-based unified LLM client.

This module provides a unified interface to multiple LLM providers (OpenAI, Claude, Azure OpenAI)
using LiteLLM. It maintains the same interface as the existing LLMClient for easy replacement.
"""

import logging
import os

import litellm

from reflexio.models.config_schema import APIKeyConfig
from reflexio.server.llm._litellm_embedding import (
    _TRUNCATION_WARNED_MODELS as _TRUNCATION_WARNED_MODELS,
)

# Identity-preserving re-exports (SINK-2): every moved name is re-bound here by
# import, never redefined, so ``from ...litellm_client import <name>`` keeps
# resolving the SAME object/class the moved code uses and tests touch. The facade
# now composes the three concern mixins; the client-core __init__/creds/config
# accessors + create_litellm_client stay here.
from reflexio.server.llm._litellm_embedding import (
    EmbeddingMixin,
)
from reflexio.server.llm._litellm_embedding import (
    _get_embedding_encoding as _get_embedding_encoding,
)
from reflexio.server.llm._litellm_embedding import (
    _get_embedding_limit as _get_embedding_limit,
)
from reflexio.server.llm._litellm_embedding import (
    _truncate_for_embedding as _truncate_for_embedding,
)
from reflexio.server.llm._litellm_json_extraction import (
    _extract_json_from_string as _extract_json_from_string,
)
from reflexio.server.llm._litellm_json_extraction import (
    _sanitize_json_string as _sanitize_json_string,
)
from reflexio.server.llm._litellm_structured_output import (
    StructuredOutputMixin,
)
from reflexio.server.llm._litellm_subprocess import (
    _CompletionChoiceSnapshot as _CompletionChoiceSnapshot,
)
from reflexio.server.llm._litellm_subprocess import (
    _CompletionErrorSnapshot as _CompletionErrorSnapshot,
)
from reflexio.server.llm._litellm_subprocess import (
    _CompletionMessageSnapshot as _CompletionMessageSnapshot,
)
from reflexio.server.llm._litellm_subprocess import (
    _CompletionResponseSnapshot as _CompletionResponseSnapshot,
)
from reflexio.server.llm._litellm_subprocess import (
    _CompletionUsageSnapshot as _CompletionUsageSnapshot,
)
from reflexio.server.llm._litellm_subprocess import (
    _litellm_completion_worker as _litellm_completion_worker,
)
from reflexio.server.llm._litellm_subprocess import (
    _PromptTokenDetailsSnapshot as _PromptTokenDetailsSnapshot,
)
from reflexio.server.llm._litellm_text_generation import (
    ProviderRequestGuardError,
    StructuredOutputValidator,
    TextGenerationMixin,
)
from reflexio.server.llm._litellm_types import (
    LiteLLMClientError,
    LiteLLMConfig,
    StructuredOutputRepairError,
    ToolCallingChatResponse,
)
from reflexio.server.llm._litellm_types import (
    LLMHardTimeoutError as LLMHardTimeoutError,
)
from reflexio.server.llm._litellm_types import (
    StructuredOutputParseError as StructuredOutputParseError,
)
from reflexio.server.llm.providers.claude_code_provider import (
    register_if_enabled as _register_claude_code,
)
from reflexio.server.llm.providers.openclaw_provider import (
    register_if_enabled as _register_openclaw,
)

# Suppress LiteLLM's verbose logging
litellm.suppress_debug_info = True

# Opt-in registration of local CLI providers. All no-ops unless the
# matching env var is set. Safe to call at import.
_register_claude_code()
_register_openclaw()

# Public importer surface (the #1 invariant of the Tier-2.5 decomposition). These
# five names — plus the test-imported internals re-exported below the split — must
# stay importable from ``reflexio.server.llm.litellm_client`` for all ~102 importers.
__all__ = [
    "LiteLLMClient",
    "LiteLLMConfig",
    "LiteLLMClientError",
    "StructuredOutputRepairError",
    "ProviderRequestGuardError",
    "StructuredOutputValidator",
    "ToolCallingChatResponse",
    "create_litellm_client",
]


class LiteLLMClient(TextGenerationMixin, EmbeddingMixin, StructuredOutputMixin):
    """
    Unified LLM client using LiteLLM for multi-provider support.

    Supports OpenAI, Claude, and Azure OpenAI models through a consistent interface.
    Provides structured output support, multi-modal (image) input, and embeddings.
    """

    # Providers that use a simple "prefix/" -> api_key mapping
    _SIMPLE_PROVIDER_PREFIXES: dict[str, str] = {
        "gemini/": "gemini",
        "openrouter/": "openrouter",
        "minimax/": "minimax",
        "deepseek/": "deepseek",
        "zai/": "zai",
        "moonshot/": "moonshot",
        "xai/": "xai",
    }

    def __init__(self, config: LiteLLMConfig):
        """
        Initialize the LiteLLM client.

        Args:
            config: LiteLLM configuration containing model and provider settings.

        Raises:
            LiteLLMClientError: If initialization fails.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info("LiteLLM client initialized with model: %s", config.model)

        # Pre-resolve API key configuration for the main model
        self._api_key, self._api_base, self._api_version = self._resolve_api_key()

        # Lazily-resolved default embedding model. Populated on first call to
        # _resolve_default_embedding_model so a client built with no embedding
        # use case never pays the auto-detection cost.
        self._default_embedding_model: str | None = None

        # Enable Braintrust observability when API key is configured
        if os.environ.get("BRAINTRUST_API_KEY") and "braintrust" not in (
            litellm.callbacks or []
        ):
            litellm.callbacks = litellm.callbacks or []
            litellm.callbacks.append("braintrust")
            self.logger.info("Braintrust observability enabled")

    def _resolve_api_key(
        self, model: str | None = None, for_embedding: bool = False
    ) -> tuple[str | None, str | None, str | None]:
        """
        Resolve API key, base URL, and version from api_key_config based on model name.

        Args:
            model: Optional model name to resolve keys for. Defaults to self.config.model.
            for_embedding: If True, skip custom endpoint override (embeddings use their own provider).

        Returns:
            tuple[Optional[str], Optional[str], Optional[str]]: (api_key, api_base, api_version)
        """
        if not self.config.api_key_config:
            return None, None, None

        # Custom endpoint takes priority for non-embedding calls
        if not for_embedding:
            ce = self.config.api_key_config.custom_endpoint
            if ce and ce.api_key and ce.api_base:
                return ce.api_key, str(ce.api_base), None

        model_to_check = model or self.config.model
        model_lower = model_to_check.lower()

        return self._resolve_by_prefix(model_lower)

    def _resolve_by_prefix(
        self, model_lower: str
    ) -> tuple[str | None, str | None, str | None]:
        """Resolve API credentials by matching the model prefix to a provider.

        Args:
            model_lower: Lowercased model name string.

        Returns:
            tuple[Optional[str], Optional[str], Optional[str]]: (api_key, api_base, api_version)
        """
        akc = self.config.api_key_config
        if not akc:
            return None, None, None

        # claude-code/* routes through the Claude Code CLI (custom provider);
        # it has no API key config — auth comes from the CLI itself.
        if model_lower.startswith("claude-code/"):
            return None, None, None

        for prefix, attr in self._SIMPLE_PROVIDER_PREFIXES.items():
            if model_lower.startswith(prefix):
                provider_cfg = getattr(akc, attr, None)
                if provider_cfg:
                    return provider_cfg.api_key, None, None
                return None, None, None

        # DashScope (Qwen) — has an optional api_base
        if model_lower.startswith("dashscope/"):
            if akc.dashscope:
                return akc.dashscope.api_key, akc.dashscope.api_base, None
            return None, None, None

        # Azure OpenAI
        if model_lower.startswith("azure/"):
            if akc.openai and akc.openai.azure_config:
                azure = akc.openai.azure_config
                return azure.api_key, str(azure.endpoint), azure.api_version
            return None, None, None

        # Anthropic/Claude models
        if "claude" in model_lower or "anthropic" in model_lower:
            if akc.anthropic:
                return akc.anthropic.api_key, None, None
            return None, None, None

        # OpenAI models (default fallback)
        if akc.openai and akc.openai.api_key:
            return akc.openai.api_key, None, None

        return None, None, None

    def update_config(self, **kwargs) -> None:
        """
        Update client configuration.

        Args:
            **kwargs: Configuration parameters to update (model, temperature, etc.).
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.debug("Updated config: %s = %s", key, value)
                # Invalidate the embedding-default cache when the provider
                # surface changes — resolve_model_name(EMBEDDING) reads
                # api_key_config, so a swap must force a re-detect.
                if key == "api_key_config":
                    self._default_embedding_model = None
            else:
                self.logger.warning("Unknown config parameter: %s", key)

    def get_model(self) -> str:
        """
        Get the current model being used.

        Returns:
            Model name string.
        """
        return self.config.model

    def get_config(self) -> LiteLLMConfig:
        """
        Get the current configuration.

        Returns:
            Current LiteLLM configuration.
        """
        return self.config


def create_litellm_client(
    model: str,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    timeout: int = 60,
    max_retries: int = 3,
    api_key_config: APIKeyConfig | None = None,
    **kwargs,
) -> LiteLLMClient:
    """
    Create a LiteLLM client with simplified parameters.

    Args:
        model: Model name to use (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022').
        temperature: Temperature for response generation.
        max_tokens: Maximum tokens to generate.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts.
        api_key_config: Optional API key configuration from Config (overrides env vars).
        **kwargs: Additional configuration parameters.

    Returns:
        Configured LiteLLM client.
    """
    config = LiteLLMConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        api_key_config=api_key_config,
        **kwargs,
    )
    return LiteLLMClient(config)
