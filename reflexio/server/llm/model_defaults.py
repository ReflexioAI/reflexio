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

# Env var opting into the Claude Code CLI provider (registered in
# reflexio.server.llm.providers.claude_code_provider). When set to "1"
# *and* the `claude` binary is on PATH, the provider is auto-detected
# with highest priority — reflexio will route extraction/evaluation
# calls through the local CLI instead of requiring an API key.
_CLAUDE_CODE_ENABLE_ENV = "CLAUDE_SMART_USE_LOCAL_CLI"
_CLAUDE_CODE_PROVIDER = "claude-code"

# Companion env var opting into the local ONNX embedder (registered in
# reflexio.server.llm.providers.local_embedding_provider). Requires the
# `chromadb` package to be importable — we detect that at runtime rather
# than making it a hard dep of reflexio itself.
_LOCAL_EMBEDDING_ENABLE_ENV = "CLAUDE_SMART_USE_LOCAL_EMBEDDING"
_LOCAL_EMBEDDING_PROVIDER = "local"

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

# When multiple keys are set, prefer providers in this order. The
# claude-code CLI provider sits at the top — when it's available, users
# are explicitly opting into local-auth extraction and should not be
# surprised by an OpenAI/Anthropic API bill from a leftover env var.
_PROVIDER_PRIORITY: list[str] = [
    _CLAUDE_CODE_PROVIDER,
    _LOCAL_EMBEDDING_PROVIDER,
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

    # Claude Code CLI and the local ONNX embedder are opt-in via their
    # own env vars + runtime requirements (`claude` on PATH for the CLI,
    # `chromadb` installed for the embedder). Their availability helpers
    # own the detection logic so there's one source of truth.
    from reflexio.server.llm.providers.claude_code_provider import (
        is_claude_code_available,
    )
    from reflexio.server.llm.providers.local_embedding_provider import (
        is_local_embedder_available,
    )

    if is_claude_code_available():
        available.add(_CLAUDE_CODE_PROVIDER)
    if is_local_embedder_available():
        available.add(_LOCAL_EMBEDDING_PROVIDER)

    return [p for p in _PROVIDER_PRIORITY if p in available]


# ---------------------------------------------------------------------------
# Per-provider default models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderDefaults:
    """Default model names for a given provider.

    Any field may be ``None`` for a role the provider does not support.
    For example, the ``local`` ONNX embedder has no generation model;
    the ``claude-code`` CLI has no embedding endpoint. ``_auto_detect_model``
    falls through to the next provider in priority order when the
    requested role is missing.

    Args:
        generation: Model for content generation, or None.
        evaluation: Model for evaluation/scoring, or None.
        should_run: Model for lightweight "should run extraction" checks, or None.
        pre_retrieval: Model for pre-retrieval query reformulation, or None.
        embedding: Model for embedding generation, or None.
        angle_reader: Fast-tier model for parallel extraction/search angle agents, or None.
        critic: Smart-tier model for extraction critics, or None.
        synthesizer: Smart-tier model for search synthesizers, or None.
        reconciler: Smart-tier model for cross-entity reconciler, or None.
    """

    generation: str | None
    evaluation: str | None
    should_run: str | None
    pre_retrieval: str | None
    embedding: str | None
    angle_reader: str | None = None
    critic: str | None = None
    synthesizer: str | None = None
    reconciler: str | None = None


_PROVIDER_DEFAULTS: dict[str, ProviderDefaults] = {
    # claude-code routes through the local Claude Code CLI via LiteLLM's
    # custom provider mechanism (see providers/claude_code_provider.py).
    # The model-name suffix after "claude-code/" is opaque — the CLI
    # picks whichever model the user has auth for.
    _CLAUDE_CODE_PROVIDER: ProviderDefaults(
        generation="claude-code/default",
        evaluation="claude-code/default",
        should_run="claude-code/default",
        pre_retrieval="claude-code/default",
        embedding=None,
        angle_reader="claude-code/default",
        critic="claude-code/default",
        synthesizer="claude-code/default",
        reconciler="claude-code/default",
    ),
    # local is an embedding-only provider that routes through an
    # in-process ONNX model (chromadb's all-MiniLM-L6-v2). Generation
    # roles stay None — use claude-code for those.
    _LOCAL_EMBEDDING_PROVIDER: ProviderDefaults(
        generation=None,
        evaluation=None,
        should_run=None,
        pre_retrieval=None,
        embedding="local/minilm-l6-v2",
    ),
    "openai": ProviderDefaults(
        generation="gpt-5-mini",
        evaluation="gpt-5-mini",
        should_run="gpt-5-nano",
        pre_retrieval="gpt-5-nano",
        embedding="text-embedding-3-small",
        angle_reader="gpt-5-nano",
        critic="gpt-5-mini",
        synthesizer="gpt-5-mini",
        reconciler="gpt-5-mini",
    ),
    "anthropic": ProviderDefaults(
        generation="claude-sonnet-4-6",
        evaluation="claude-sonnet-4-6",
        should_run="claude-haiku-4-5-20251001",
        pre_retrieval="claude-haiku-4-5-20251001",
        embedding=None,
        angle_reader="claude-haiku-4-5-20251001",
        critic="claude-sonnet-4-6",
        synthesizer="claude-sonnet-4-6",
        reconciler="claude-sonnet-4-6",
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
    # Tool-calling agentic pipeline roles — fast tier for parallel specialists,
    # smart tier for judgment/synthesis steps.
    ANGLE_READER = "angle_reader"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    RECONCILER = "reconciler"


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
            "No LLM provider available. Set at least one of: "
            + ", ".join(sorted(_ENV_TO_PROVIDER))
            + f" in your .env file, or set {_CLAUDE_CODE_ENABLE_ENV}=1 "
            "with the `claude` CLI on PATH to use the local Claude Code provider."
        )

    if role == ModelRole.EMBEDDING:
        # Search for first provider with embedding support
        for provider in providers:
            defaults = _PROVIDER_DEFAULTS[provider]
            if defaults.embedding:
                return defaults.embedding
        raise RuntimeError(
            "No embedding-capable LLM provider found. "
            f"Set OPENAI_API_KEY or GEMINI_API_KEY, or set "
            f"{_LOCAL_EMBEDDING_ENABLE_ENV}=1 with `chromadb` installed "
            "to use the local ONNX embedder."
        )

    # Non-embedding roles: fall through to the first provider whose slot
    # for this role is non-None. Lets embedding-only providers (e.g.
    # "local") sit in the priority list without breaking generation.
    for provider in providers:
        defaults = _PROVIDER_DEFAULTS[provider]
        model_name = getattr(defaults, role.value)
        if model_name:
            return model_name
    raise RuntimeError(f"No provider in {providers} supports role={role.value}.")


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
            "No LLM provider available. Set at least one of: "
            + ", ".join(sorted(_ENV_TO_PROVIDER))
            + f" in your .env file, or set {_CLAUDE_CODE_ENABLE_ENV}=1 "
            "with the `claude` CLI on PATH to use the local Claude Code provider."
        )

    logger.info("Auto-detected LLM providers (priority order): %s", providers)
    generation_provider = next(
        (p for p in providers if _PROVIDER_DEFAULTS[p].generation), None
    )
    if generation_provider:
        logger.info("Primary provider for generation: %s", generation_provider)
    else:
        logger.info("No generation-capable provider available")

    # Validate embedding availability
    embedding_provider = next(
        (p for p in providers if _PROVIDER_DEFAULTS[p].embedding), None
    )
    if not embedding_provider:
        raise RuntimeError(
            "No embedding-capable LLM provider found. "
            f"Set OPENAI_API_KEY or GEMINI_API_KEY, or set "
            f"{_LOCAL_EMBEDDING_ENABLE_ENV}=1 with `chromadb` installed "
            "to use the local ONNX embedder."
        )
    logger.info("Embedding provider: %s", embedding_provider)
