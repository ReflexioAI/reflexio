"""
LiteLLM-based unified LLM client.

This module provides a unified interface to multiple LLM providers (OpenAI, Claude, Azure OpenAI)
using LiteLLM. It maintains the same interface as the existing LLMClient for easy replacement.
"""

import hashlib
import json
import logging
import os
from typing import Any

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
    # Re-exported for the enterprise repair-ladder tests, which assert their own
    # budget against this one. It stays underscore-named because it is internal
    # to `_litellm_text_generation`; the `as` form is how this file already
    # surfaces `_litellm_completion_worker`, and it keeps the name out of
    # `__all__`.
    _LADDER_WALL_CLOCK_BUDGET_SECONDS as _LADDER_WALL_CLOCK_BUDGET_SECONDS,
)
from reflexio.server.llm._litellm_text_generation import (
    STRUCTURED_OUTPUT_CORRECTION_PREFIX,
    ProviderRequestGuardError,
    StructuredOutputValidator,
    TextGenerationMixin,
    is_structured_output_correction_turn,
    structured_output_correction_turn,
    structured_output_repair_idempotency_key,
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
from reflexio.server.llm.llm_utils import is_pydantic_model
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
# names — plus the test-imported internals re-exported below the split — must
# stay importable from ``reflexio.server.llm.litellm_client`` for all ~102
# importers.
#
# `structured_output_correction_turn` and `STRUCTURED_OUTPUT_CORRECTION_PREFIX`
# joined them because the enterprise open-world analyst reaches for the repair
# turn directly, and its boundary guard forbids importing
# `_litellm_text_generation` (a private module) from enterprise code. Surfacing
# them here is the sanctioned route: the allowlist in that guard is documented
# as only ever SHRINKING.
__all__ = [
    "LiteLLMClient",
    "LiteLLMConfig",
    "LiteLLMClientError",
    "StructuredOutputRepairError",
    "ProviderRequestGuardError",
    "StructuredOutputValidator",
    "ToolCallingChatResponse",
    "create_litellm_client",
    "STRUCTURED_OUTPUT_CORRECTION_PREFIX",
    "is_structured_output_correction_turn",
    "structured_output_correction_turn",
    "structured_output_repair_idempotency_key",
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

    def get_completion_request_contract(self, **kwargs: Any) -> dict[str, Any]:
        """Resolve primary completion identity without provider dispatch.

        Accepts the same per-call generation overrides (including model_role,
        max_tokens and temperature); messages are intentionally an empty probe.
        Generation overrides preserve explicit None values; fallback_models=None
        uses configured fallbacks, as in public chat generation. Fixed keys and
        absent values are stable for internal admission comparison/hashing.
        The generation-policy digest binds all remaining built provider fields,
        including tools, schema controls and native extensions, without returning
        arbitrary custom data. Messages, credentials/headers, observations and
        dispatch timeouts/retries are excluded and must be bound separately when
        relevant; prompt/schema witnesses remain caller-owned, including schema
        instructions embedded in messages. Pydantic response-format classes use
        their JSON Schema; other non-JSON values and nonfinite numbers are rejected;
        there is no stringification fallback for unsupported descriptors.
        API keys, messages and observability metadata are excluded, but route
        URLs can contain credentials: do not log or expose this descriptor.
        """
        if kwargs.get("fallback_models") is None:
            kwargs.pop("fallback_models", None)
        params = self._build_completion_params(messages=[], **kwargs)[0]
        return self._completion_request_contract_from_params(params)

    @staticmethod
    def _completion_request_contract_from_params(
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """One normalization for the public probe and internal transport guards."""
        contract = {
            key: params.get(key)
            for key in (
                "model",
                "api_base",
                "base_url",
                "api_version",
                "custom_llm_provider",
                "temperature",
                "top_p",
                "seed",
                "max_tokens",
                "max_completion_tokens",
            )
        }
        # Bind every remaining built generation parameter, including future
        # provider extensions. Credentials, observations and dispatch controls
        # are caller/transport concerns; message/schema witnesses stay separate.
        excluded = {
            "messages",
            "api_key",
            "azure_ad_token",
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
            "vertex_credentials",
            "headers",
            "extra_headers",
            "metadata",
            "callbacks",
            "success_callback",
            "failure_callback",
            "logger_fn",
            "litellm_logging_obj",
            "litellm_call_id",
            "timeout",
            "num_retries",
            "fallbacks",
            "structured_output_validator",
            "provider_request_guard",
        }
        policy = {key: value for key, value in params.items() if key not in excluded}
        if is_pydantic_model(policy.get("response_format")):
            policy["response_format"] = policy["response_format"].model_json_schema()
        contract["generation_policy_digest"] = hashlib.sha256(
            json.dumps(
                policy,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        return contract

    def get_completion_model_ladder(self, **kwargs: Any) -> tuple[str, ...]:
        """Resolve the ordered transport ladder without provider dispatch.

        Forward the same effective per-call overrides used for generation,
        including model, model_role and fallback_models. As in public chat
        generation, None uses configured fallbacks and [] disables them. This describes
        distinct network rungs, not retries or structured repair attempts.
        """
        if kwargs.get("fallback_models") is None:
            kwargs.pop("fallback_models", None)
        return tuple(self._resolve_ladder(**kwargs))

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
