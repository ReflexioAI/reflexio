"""Auto-detect available LLM providers and resolve default models by API key.

Resolution order (highest priority first):
    1. LLMConfig override (org-level configuration)
    2. llm_model_setting.json site var (non-empty string values)
    3. Auto-detect from available API keys in environment / APIKeyConfig
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reflexio.models.config_schema import APIKeyConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

_ENV_TO_PROVIDER: dict[str, str] = {
    "OPENAI_API_KEY": "openai",
    "ANTHROPIC_API_KEY": "anthropic",
    "GEMINI_API_KEY": "gemini",
    "DEEPSEEK_API_KEY": "deepseek",
    "OPENROUTER_API_KEY": "openrouter",
    "MINIMAX_API_KEY": "minimax",
    "DASHSCOPE_API_KEY": "dashscope",
    "XAI_API_KEY": "xai",
    "MOONSHOT_API_KEY": "moonshot",
    "ZAI_API_KEY": "zai",
}

# When multiple keys are set, prefer providers in this order.
_PROVIDER_PRIORITY: list[str] = [
    "anthropic",
    "gemini",
    "openrouter",
    "deepseek",
    "minimax",
    "dashscope",
    "xai",
    "moonshot",
    "zai",
    "openai",
]

# Maps APIKeyConfig field names to provider keys (field name == provider key
# for all current providers, but kept explicit for clarity).
_API_KEY_CONFIG_FIELDS: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "deepseek": "deepseek",
    "openrouter": "openrouter",
    "minimax": "minimax",
    "dashscope": "dashscope",
    "xai": "xai",
    "moonshot": "moonshot",
    "zai": "zai",
}


def detect_available_providers(
    api_key_config: APIKeyConfig | None = None,
) -> list[str]:
    """Detect available LLM providers from APIKeyConfig and/or environment variables.

    Args:
        api_key_config: Optional org-level API key configuration. Fields set here
            take precedence over environment variables.

    Returns:
        list[str]: Available provider keys in priority order.
    """
    available: set[str] = set()

    # Check APIKeyConfig fields
    if api_key_config:
        for field, provider in _API_KEY_CONFIG_FIELDS.items():
            if getattr(api_key_config, field, None) is not None:
                available.add(provider)

    # Check environment variables
    for env_var, provider in _ENV_TO_PROVIDER.items():
        if os.environ.get(env_var):
            available.add(provider)

    return [p for p in _PROVIDER_PRIORITY if p in available]


# ---------------------------------------------------------------------------
# Per-provider default models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderDefaults:
    """Default model names for a given provider.

    Args:
        generation: Model for content generation tasks.
        evaluation: Model for evaluation/scoring tasks.
        should_run: Model for lightweight "should run extraction" checks.
        pre_retrieval: Model for pre-retrieval query reformulation.
        embedding: Model for embedding generation, or None if provider has no embedding API.
    """

    generation: str
    evaluation: str
    should_run: str
    pre_retrieval: str
    embedding: str | None


_PROVIDER_DEFAULTS: dict[str, ProviderDefaults] = {
    "openai": ProviderDefaults(
        generation="gpt-5-mini",
        evaluation="gpt-5-mini",
        should_run="gpt-5-nano",
        pre_retrieval="gpt-5-nano",
        embedding="text-embedding-3-small",
    ),
    "anthropic": ProviderDefaults(
        generation="claude-sonnet-4-6",
        evaluation="claude-sonnet-4-6",
        should_run="claude-haiku-4-5-20251001",
        pre_retrieval="claude-haiku-4-5-20251001",
        embedding=None,
    ),
    "gemini": ProviderDefaults(
        generation="gemini/gemini-3-flash-preview",
        evaluation="gemini/gemini-3-flash-preview",
        should_run="gemini/gemini-3-flash-preview",
        pre_retrieval="gemini/gemini-3-flash-preview",
        embedding="gemini/text-embedding-004",
    ),
    "deepseek": ProviderDefaults(
        generation="deepseek/deepseek-chat",
        evaluation="deepseek/deepseek-chat",
        should_run="deepseek/deepseek-chat",
        pre_retrieval="deepseek/deepseek-chat",
        embedding=None,
    ),
    "openrouter": ProviderDefaults(
        generation="openrouter/google/gemini-3-flash-preview",
        evaluation="openrouter/google/gemini-3-flash-preview",
        should_run="openrouter/google/gemini-3-flash-preview",
        pre_retrieval="openrouter/google/gemini-3-flash-preview",
        embedding=None,
    ),
    "minimax": ProviderDefaults(
        generation="minimax/MiniMax-M2.7",
        evaluation="minimax/MiniMax-M2.7",
        should_run="minimax/MiniMax-M2.7",
        pre_retrieval="minimax/MiniMax-M2.7",
        embedding=None,
    ),
    "dashscope": ProviderDefaults(
        generation="dashscope/qwen-plus",
        evaluation="dashscope/qwen-plus",
        should_run="dashscope/qwen-turbo",
        pre_retrieval="dashscope/qwen-turbo",
        embedding=None,
    ),
    "xai": ProviderDefaults(
        generation="xai/grok-3-mini",
        evaluation="xai/grok-3-mini",
        should_run="xai/grok-3-mini",
        pre_retrieval="xai/grok-3-mini",
        embedding=None,
    ),
    "moonshot": ProviderDefaults(
        generation="moonshot/moonshot-v1-8k",
        evaluation="moonshot/moonshot-v1-8k",
        should_run="moonshot/moonshot-v1-8k",
        pre_retrieval="moonshot/moonshot-v1-8k",
        embedding=None,
    ),
    "zai": ProviderDefaults(
        generation="zai/glm-4-flash",
        evaluation="zai/glm-4-flash",
        should_run="zai/glm-4-flash",
        pre_retrieval="zai/glm-4-flash",
        embedding=None,
    ),
}


EMBEDDING_CAPABLE_PROVIDERS: frozenset[str] = frozenset(
    p for p, d in _PROVIDER_DEFAULTS.items() if d.embedding is not None
)


# ---------------------------------------------------------------------------
# Model role enum and resolution
# ---------------------------------------------------------------------------


class ModelRole(StrEnum):
    """Roles that require an LLM model name."""

    GENERATION = "generation"
    EVALUATION = "evaluation"
    SHOULD_RUN = "should_run"
    PRE_RETRIEVAL = "pre_retrieval"
    EMBEDDING = "embedding"


def _auto_detect_model(
    role: ModelRole,
    providers: list[str],
) -> str:
    """Pick the default model for *role* from the first available provider.

    For the EMBEDDING role, if the primary provider has no embedding support,
    search the remaining providers for one that does.

    Args:
        role: The model role to resolve.
        providers: Available providers in priority order.

    Returns:
        str: The resolved model name.

    Raises:
        RuntimeError: If no suitable provider is found.
    """
    if not providers:
        raise RuntimeError(
            "No LLM API keys found. Set at least one of: "
            + ", ".join(sorted(_ENV_TO_PROVIDER))
            + " in your .env file."
        )

    if role == ModelRole.EMBEDDING:
        # Search for first provider with embedding support
        for provider in providers:
            defaults = _PROVIDER_DEFAULTS[provider]
            if defaults.embedding:
                return defaults.embedding
        raise RuntimeError(
            "No embedding-capable LLM provider found. "
            "Set OPENAI_API_KEY or GEMINI_API_KEY for embedding support."
        )

    primary = providers[0]
    defaults = _PROVIDER_DEFAULTS[primary]
    return getattr(defaults, role.value)


def resolve_model_name(
    role: ModelRole,
    *,
    site_var_value: str | None = None,
    config_override: str | None = None,
    api_key_config: APIKeyConfig | None = None,
) -> str:
    """Resolve a model name using the 3-tier chain.

    Resolution order (highest priority first):
        1. config_override (from LLMConfig, org-level)
        2. site_var_value (from llm_model_setting.json, non-empty strings only)
        3. Auto-detect from available API keys

    Args:
        role: The model role to resolve.
        site_var_value: Value from llm_model_setting.json. Empty strings are treated as unset.
        config_override: Value from org-level LLMConfig.
        api_key_config: Optional org-level API key configuration for provider detection.

    Returns:
        str: The resolved model name.

    Raises:
        RuntimeError: If no API keys are available and no override is set.
    """
    if config_override:
        return config_override
    if site_var_value:
        return site_var_value
    providers = detect_available_providers(api_key_config)
    return _auto_detect_model(role, providers)


def validate_llm_availability(
    api_key_config: APIKeyConfig | None = None,
) -> None:
    """Validate that at least one LLM provider and one embedding provider are available.

    Should be called once during startup. Logs the auto-selected provider at INFO level.

    Args:
        api_key_config: Optional org-level API key configuration.

    Raises:
        RuntimeError: If no API keys are found, or if no embedding-capable provider is available.
    """
    providers = detect_available_providers(api_key_config)
    if not providers:
        raise RuntimeError(
            "No LLM API keys found. Set at least one of: "
            + ", ".join(sorted(_ENV_TO_PROVIDER))
            + " in your .env file."
        )

    logger.info("Auto-detected LLM providers (priority order): %s", providers)
    logger.info("Primary provider for generation: %s", providers[0])

    # Validate embedding availability
    embedding_provider = next(
        (p for p in providers if _PROVIDER_DEFAULTS[p].embedding), None
    )
    if not embedding_provider:
        raise RuntimeError(
            "No embedding-capable LLM provider found. "
            "Set OPENAI_API_KEY or GEMINI_API_KEY for embedding support."
        )
    logger.info("Embedding provider: %s", embedding_provider)
