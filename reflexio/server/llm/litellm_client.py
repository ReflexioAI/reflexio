"""
LiteLLM-based unified LLM client.

This module provides a unified interface to multiple LLM providers (OpenAI, Claude, Azure OpenAI)
using LiteLLM. It maintains the same interface as the existing LLMClient for easy replacement.
"""

import base64
import json
import logging
import multiprocessing
import os
import pickle
import queue
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import litellm
import tiktoken
from pydantic import BaseModel

from reflexio.models.config_schema import APIKeyConfig
from reflexio.server.llm.image_utils import (
    SUPPORTED_IMAGE_MIME_TYPES,
    ImageEncodingError,
)
from reflexio.server.llm.image_utils import (
    encode_image_to_base64 as _encode_image_to_base64,
)
from reflexio.server.llm.llm_utils import (
    is_pydantic_model,
    strict_response_format_for_model,
)
from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name
from reflexio.server.llm.providers.claude_code_provider import (
    register_if_enabled as _register_claude_code,
)
from reflexio.server.llm.providers.openclaw_provider import (
    register_if_enabled as _register_openclaw,
)
from reflexio.server.llm.providers.embedding_service_provider import (
    EmbeddingUnavailableError,
    embedding_provider_mode,
    get_service_embeddings,
    should_use_embedding_service,
)
from reflexio.server.llm.providers.local_embedding_provider import (
    LocalEmbedder,
)
from reflexio.server.llm.providers.local_embedding_provider import (
    is_chromadb_importable as _is_chromadb_importable,
)
from reflexio.server.llm.providers.local_embedding_provider import (
    register_if_chromadb_available as _register_local_embedder,
)
from reflexio.server.llm.providers.nomic_embedding_provider import (
    NomicEmbedder,
)
from reflexio.server.llm.providers.nomic_embedding_provider import (
    is_nomic_model as _is_nomic_model,
)
from reflexio.server.llm.providers.nomic_embedding_provider import (
    register_if_enabled as _register_nomic_embedder,
)

# Suppress LiteLLM's verbose logging
litellm.suppress_debug_info = True

# Opt-in registration of local CLI providers. All no-ops unless the
# matching env var is set. Safe to call at import.
_register_claude_code()
_register_openclaw()
_register_local_embedder()
_register_nomic_embedder()

_LOGGER = logging.getLogger(__name__)

# OpenAI's documented max input length for text-embedding-3-* and ada-002 is
# 8191 tokens. Used as the fallback limit only when a model's name looks
# OpenAI-family but litellm's registry has no entry for it.
_OPENAI_EMBEDDING_FALLBACK_MAX_TOKENS = 8191

# Models whose truncation warning has already been emitted this process. Keeps
# batch backfills of millions of long docs from flooding logs — the first hit
# per model goes to WARNING, everything after to DEBUG.
_TRUNCATION_WARNED_MODELS: set[str] = set()

# Model-name prefixes that route through OpenAI's embedding API (and therefore
# share the 8191-token cap). Anything that does not start with one of these is
# treated as "unknown provider" when litellm has no registry entry.
_OPENAI_EMBEDDING_FAMILY_PREFIXES = ("text-embedding-", "openai/", "azure/")

# Python-to-JSON keyword replacements used by _sanitize_json_string.
_PYTHON_TO_JSON_REPLACEMENTS = {"True": "true", "False": "false", "None": "null"}


@lru_cache(maxsize=32)
def _get_embedding_limit(model: str) -> int | None:
    """
    Resolve the maximum input token count for an embedding model.

    Consults ``litellm.get_model_info`` first so provider-specific caps are
    respected (OpenAI ~8191, Cohere 512, Voyage 32000, etc.). When litellm has
    no entry for the model, falls back to the OpenAI 8191 cap only when the
    model name looks OpenAI-family; otherwise returns ``None`` to disable
    truncation for unknown providers (safer than over-truncating their input).

    Args:
        model (str): Embedding model name (e.g. 'text-embedding-3-small',
            'cohere/embed-english-v3.0').

    Returns:
        int | None: Maximum input tokens, or ``None`` when the limit is unknown
            and no safe fallback applies.
    """
    try:
        info = litellm.get_model_info(model)
    except Exception:
        info = None
    if info and info.get("mode") == "embedding":
        max_tokens = info.get("max_input_tokens")
        if isinstance(max_tokens, int) and max_tokens > 0:
            return max_tokens
    if model.startswith(_OPENAI_EMBEDDING_FAMILY_PREFIXES):
        return _OPENAI_EMBEDDING_FALLBACK_MAX_TOKENS
    return None


@lru_cache(maxsize=16)
def _get_embedding_encoding(model: str) -> tiktoken.Encoding:
    """
    Return the tiktoken encoding for an embedding model, falling back to cl100k_base.

    For non-OpenAI providers tiktoken does not know the real tokenizer, so the
    cl100k_base fallback is an approximate proxy for token counting. That is
    acceptable here because we truncate toward the provider's cap with the
    proxy, which tends to over-truncate by a small fraction rather than under-
    truncate and cause upstream 400s.

    Args:
        model (str): Embedding model name (e.g. 'text-embedding-3-small').

    Returns:
        tiktoken.Encoding: Encoder to use for token counting and truncation.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _reject_cloud_mode(embedding_model: str, mode: str) -> None:
    """
    Raise when a local-only embedding model is configured for cloud mode.

    Args:
        embedding_model (str): The resolved embedding model name.
        mode (str): The resolved embedding provider mode.

    Raises:
        EmbeddingUnavailableError: If ``mode`` is ``"cloud"``.
    """
    if mode == "cloud":
        raise EmbeddingUnavailableError(
            f"Local embedding model {embedding_model!r} cannot use cloud mode"
        )


def _truncate_for_embedding(
    text: str, model: str, max_tokens: int | None = None
) -> str:
    """
    Truncate a string so its token count fits within an embedding model's input limit.

    The token budget is auto-resolved from ``_get_embedding_limit`` by default.
    When the model has no known limit (unknown provider not in litellm's
    registry and not OpenAI-family), returns the text unchanged — over-
    truncating an unknown provider's input is worse than passing it through
    and letting the provider's own error surface.

    Args:
        text (str): Raw input text.
        model (str): Embedding model name, used to pick the tokenizer and the
            per-provider token cap.
        max_tokens (int | None): Override for the resolved budget. Primarily
            used by tests to exercise the truncation path on short strings;
            leave as ``None`` in production callers.

    Returns:
        str: Original text if it already fits (or the model has no known
            limit), otherwise a token-bounded prefix.
    """
    if not text:
        return text
    if max_tokens is None:
        max_tokens = _get_embedding_limit(model)
    if max_tokens is None:
        return text
    encoding = _get_embedding_encoding(model)
    tokens = encoding.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    if model in _TRUNCATION_WARNED_MODELS:
        _LOGGER.debug(
            "Truncating embedding input from %d to %d tokens for model %s",
            len(tokens),
            max_tokens,
            model,
        )
    else:
        _TRUNCATION_WARNED_MODELS.add(model)
        _LOGGER.warning(
            "Truncating embedding input from %d to %d tokens for model %s "
            "(further occurrences will be logged at DEBUG)",
            len(tokens),
            max_tokens,
            model,
        )
    return encoding.decode(tokens[:max_tokens])


@dataclass
class LiteLLMConfig:
    """
    Configuration for LiteLLM client.

    Args:
        model: Model name to use (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022').
        temperature: Temperature for response generation (0.0 to 2.0).
        max_tokens: Maximum tokens to generate.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts on the primary model. Passed
            directly to litellm's num_retries. Default 3.
        retry_delay: Currently unused — LiteLLM owns retry backoff. Kept for
            backward compatibility; remove in a follow-up sweep.
        top_p: Top-p sampling parameter.
        api_key_config: Optional API key configuration from Config (overrides env vars).
        fallback_models: Models LiteLLM tries in order after the primary
            exhausts num_retries. Passed directly to litellm's fallbacks param.
            Default is an empty list (no fallback) so local reflexio and the
            claude-smart integration are never silently routed to an unintended
            provider. Production opts in via the env var
            REFLEXIO_LLM_FALLBACK_MODELS (comma-separated, e.g. "gpt-5.4-mini").
            Self-references are deduped at request time.
    """

    model: str
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0
    top_p: float = 1.0
    api_key_config: APIKeyConfig | None = None
    fallback_models: list[str] = field(
        default_factory=lambda: [
            m.strip()
            for m in os.environ.get("REFLEXIO_LLM_FALLBACK_MODELS", "").split(",")
            if m.strip()
        ]
    )


# Reasoning models that routinely exceed the default 120s provider timeout on
# large extraction contexts. Values are floors, not overrides: the effective
# timeout is max(configured, floor), and an explicit per-call timeout kwarg
# always wins.
_MODEL_TIMEOUT_FLOOR_SECONDS: dict[str, int] = {
    "minimax/MiniMax-M3": 240,
}


@dataclass
class ToolCallingChatResponse:
    """Response from a chat call that was routed in tool-calling mode.

    Returned instead of ``str | BaseModel`` whenever the caller passes
    ``tools=...`` to ``generate_chat_response``. Callers inspect
    ``tool_calls`` to drive a tool loop; ``content`` is set on the
    terminal (non-tool) turn.

    Args:
        content: Text content from the model, or None when the model emitted tool calls.
        tool_calls: List of tool call objects from the model, or None on the terminal turn.
        finish_reason: The stop reason reported by the provider (e.g. "tool_calls", "stop").
        usage: Raw usage object from the LLM response (provider-dependent shape), or None.
        cost_usd: Estimated cost in USD for this call via litellm price table, or None when
            the provider is not in the table (local ONNX, claude-code CLI, etc.).
    """

    content: str | None
    tool_calls: list[Any] | None
    finish_reason: str | None
    usage: Any | None = None
    cost_usd: float | None = None


class LiteLLMClientError(Exception):
    """Custom exception for LiteLLM client errors."""


class StructuredOutputParseError(Exception):
    """Raised when a structured-output LLM call returns content that cannot be parsed.

    Caught by the retry loop in ``_make_request`` so a malformed response
    burns a retry attempt rather than silently returning unparsed content.
    """


class LLMHardTimeoutError(TimeoutError):
    """Raised when an LLM call exceeds the client-side wall-clock timeout."""


@dataclass
class _CompletionMessageSnapshot:
    content: str | None = None
    tool_calls: Any | None = None


@dataclass
class _CompletionChoiceSnapshot:
    message: _CompletionMessageSnapshot
    finish_reason: str | None = None


@dataclass
class _PromptTokenDetailsSnapshot:
    cached_tokens: int = 0


@dataclass
class _CompletionUsageSnapshot:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    prompt_tokens_details: _PromptTokenDetailsSnapshot | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


@dataclass
class _CompletionResponseSnapshot:
    choices: list[_CompletionChoiceSnapshot]
    usage: _CompletionUsageSnapshot | None = None
    model: str | None = None
    _hidden_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class _CompletionErrorSnapshot:
    type_name: str
    message: str
    model: str | None = None
    llm_provider: str | None = None


def _snapshot_completion_error(
    exc: BaseException, params: dict[str, Any]
) -> _CompletionErrorSnapshot:
    model = getattr(exc, "model", None) or params.get("model")
    llm_provider = getattr(exc, "llm_provider", None)
    return _CompletionErrorSnapshot(
        type_name=type(exc).__name__,
        message=str(exc),
        model=str(model) if model else None,
        llm_provider=str(llm_provider) if llm_provider else None,
    )


def _ensure_picklable(value: Any) -> Any:
    try:
        pickle.dumps(value)
    except Exception:
        return repr(value)
    return value


def _snapshot_completion_response(response: Any) -> _CompletionResponseSnapshot:
    choices: list[_CompletionChoiceSnapshot] = []
    for choice in getattr(response, "choices", []) or []:
        message = getattr(choice, "message", None)
        choices.append(
            _CompletionChoiceSnapshot(
                message=_CompletionMessageSnapshot(
                    content=getattr(message, "content", None),
                    tool_calls=_ensure_picklable(getattr(message, "tool_calls", None)),
                ),
                finish_reason=getattr(choice, "finish_reason", None),
            )
        )

    usage = getattr(response, "usage", None)
    usage_snapshot = None
    if usage is not None:
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        prompt_details_snapshot = None
        if prompt_details is not None:
            prompt_details_snapshot = _PromptTokenDetailsSnapshot(
                cached_tokens=int(getattr(prompt_details, "cached_tokens", 0) or 0)
            )
        usage_snapshot = _CompletionUsageSnapshot(
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            prompt_tokens_details=prompt_details_snapshot,
            cache_creation_input_tokens=getattr(
                usage, "cache_creation_input_tokens", None
            ),
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", None),
        )

    hidden_params = getattr(response, "_hidden_params", {}) or {}
    if not isinstance(hidden_params, dict):
        hidden_params = {}

    return _CompletionResponseSnapshot(
        choices=choices,
        usage=usage_snapshot,
        model=getattr(response, "model", None),
        _hidden_params={str(k): _ensure_picklable(v) for k, v in hidden_params.items()},
    )


def _picklable_completion_result(response: Any) -> Any:
    try:
        pickle.dumps(response)
    except Exception:
        return _snapshot_completion_response(response)
    return response


def _litellm_completion_worker(
    params: dict[str, Any], result_queue: multiprocessing.Queue
) -> None:
    try:
        result_queue.put(
            ("ok", _picklable_completion_result(litellm.completion(**params)))
        )
    except BaseException as exc:
        result_queue.put(("error", _snapshot_completion_error(exc, params)))


class LiteLLMClient:
    """
    Unified LLM client using LiteLLM for multi-provider support.

    Supports OpenAI, Claude, and Azure OpenAI models through a consistent interface.
    Provides structured output support, multi-modal (image) input, and embeddings.
    """

    SUPPORTED_IMAGE_FORMATS: set[str] = set(SUPPORTED_IMAGE_MIME_TYPES.keys())

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

    # Models that only support temperature=1.0 (custom values cause errors or degraded performance)
    TEMPERATURE_RESTRICTED_MODELS = {
        "gpt-5",
        "gpt-5.4-mini",
        "gpt-5-nano",
        "gpt-5-codex",
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
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

    def generate_response(
        self,
        prompt: str,
        system_message: str | None = None,
        images: list[str | bytes | dict] | None = None,
        image_media_type: str | None = None,
        **kwargs: Any,
    ) -> str | BaseModel | ToolCallingChatResponse:
        """
        Generate a response using the configured LLM.

        Args:
            prompt: The user prompt/message.
            system_message: Optional system message to set context.
            images: Optional list of images (file paths, bytes, or pre-formatted content blocks).
            image_media_type: Media type for images if passing bytes (e.g., 'image/png').
            **kwargs: Additional parameters including:
                - response_format: Pydantic BaseModel class for structured output
                - parse_structured_output: Whether to parse structured output (default True)
                - temperature: Override config temperature
                - max_tokens: Override config max_tokens

        Returns:
            Generated response content. Returns string for text responses,
            or BaseModel instance for Pydantic model responses.

        Raises:
            LiteLLMClientError: If the API call fails after all retries,
                or if response_format is not a Pydantic BaseModel class.
        """
        # Validate response_format if provided
        response_format = kwargs.get("response_format")
        if response_format is not None and not is_pydantic_model(response_format):
            raise LiteLLMClientError(
                "response_format must be a Pydantic BaseModel class, "
                f"got {type(response_format).__name__}"
            )

        # Build user message content
        user_content = self._build_user_content(prompt, images, image_media_type)

        # Build messages list
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_content})

        return self._make_request(messages, **kwargs)

    def generate_chat_response(
        self,
        messages: list[dict[str, Any]],
        system_message: str | None = None,
        *,
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        model_role: ModelRole | None = None,
        max_retries: int | None = None,
        fallback_models: list[str] | None = None,
        **kwargs: Any,
    ) -> str | BaseModel | ToolCallingChatResponse:
        """
        Generate a response from a list of chat messages.

        Args:
            messages: List of messages in chat format [{"role": "...", "content": "..."}].
            system_message: Optional system message to prepend.
            tools: Optional list of tool definitions for tool-calling mode.
                When provided, the return type is ``ToolCallingChatResponse``.
            tool_choice: Optional tool choice control ("auto", "none", "required",
                or a dict specifying a particular tool). Forwarded to the provider.
            model_role: Optional ``ModelRole`` to override the model selected for
                this request. The role is resolved via ``resolve_model_name`` using
                the client's ``api_key_config``.
            max_retries (int | None): Optional per-call override for the number of
                retry attempts. When ``None`` (the default), the value falls back to
                ``LiteLLMConfig.max_retries``.
            fallback_models (list[str] | None): Optional per-call override for the
                fallback model chain. When ``None`` (the default), the value falls
                back to ``LiteLLMConfig.fallback_models``.
            **kwargs: Additional parameters including:
                - response_format: Pydantic BaseModel class for structured output
                - parse_structured_output: Whether to parse structured output (default True)
                - temperature: Override config temperature
                - max_tokens: Override config max_tokens

        Returns:
            Generated response content. Returns string for text responses,
            ``BaseModel`` instance for Pydantic model responses, or
            ``ToolCallingChatResponse`` when ``tools`` is provided.

        Raises:
            LiteLLMClientError: If the API call fails after all retries,
                or if response_format is not a Pydantic BaseModel class.
        """
        # Validate response_format if provided
        response_format = kwargs.get("response_format")
        if response_format is not None and not is_pydantic_model(response_format):
            raise LiteLLMClientError(
                "response_format must be a Pydantic BaseModel class, "
                f"got {type(response_format).__name__}"
            )

        # Prepend system message if provided
        final_messages = list(messages)
        if system_message:
            # Check if first message is already a system message
            if final_messages and final_messages[0].get("role") == "system":
                # Merge with existing system message
                final_messages[0]["content"] = (
                    f"{system_message}\n\n{final_messages[0]['content']}"
                )
            else:
                final_messages.insert(0, {"role": "system", "content": system_message})

        # Forward tool-calling and model-role kwargs into _make_request
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if model_role is not None:
            kwargs["model_role"] = model_role
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        if fallback_models is not None:
            kwargs["fallback_models"] = fallback_models

        return self._make_request(final_messages, **kwargs)

    def _resolve_default_embedding_model(self) -> str:
        """
        Resolve the embedding model to use when callers do not specify one.

        Routes through the same auto-detection chain as the rest of reflexio
        (``resolve_model_name`` for ``ModelRole.EMBEDDING``) so a session that
        has the local ONNX embedder enabled — or any non-OpenAI provider —
        does not silently fall back to ``text-embedding-3-small`` and produce
        OpenAI 401s. Higher-precedence org config and site-var overrides are
        the caller's responsibility to resolve and pass via ``model=``; this
        helper handles only the auto-detect tier.

        Returns:
            str: The auto-detected embedding model name (cached after first call).

        Raises:
            RuntimeError: Propagated from ``resolve_model_name`` when no
                embedding-capable provider is available.
        """
        if self._default_embedding_model is None:
            self._default_embedding_model = resolve_model_name(
                ModelRole.EMBEDDING,
                api_key_config=self.config.api_key_config,
            )
        return self._default_embedding_model

    def get_embedding(
        self, text: str, model: str | None = None, dimensions: int | None = None
    ) -> list[float]:
        """
        Get embedding vector for the given text.

        Args:
            text: The text to get embedding for.
            model: Optional embedding model. When omitted, the model is
                auto-detected via ``resolve_model_name(ModelRole.EMBEDDING)``
                so callers inherit the local-embedder gate and any non-OpenAI
                provider configured for this client.
            dimensions: Optional number of dimensions for the embedding vector.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            LiteLLMClientError: If embedding generation fails.
        """
        embedding_model = model or self._resolve_default_embedding_model()
        mode = embedding_provider_mode(embedding_model)
        if mode == "off":
            raise EmbeddingUnavailableError("Embedding provider is disabled")
        if should_use_embedding_service(embedding_model):
            return get_service_embeddings(
                [text], model=embedding_model, dimensions=dimensions
            )[0]

        # local/nomic-embed-* must stay on the Nomic provider (137M params,
        # 768d Matryoshka-truncated to 512). Falling through to MiniLM would
        # mix embedding models inside existing vector stores.
        if _is_nomic_model(embedding_model):
            _reject_cloud_mode(embedding_model, mode)
            try:
                return NomicEmbedder.get().embed([text])[0]
            except Exception as e:
                raise LiteLLMClientError(
                    f"Nomic embedding generation failed: {str(e)}"
                ) from e

        # local/* models route through the in-process ONNX embedder — no
        # network call, no litellm API, no tiktoken truncation (the embedder
        # applies its own token cap). The dispatch is gated solely on
        # ``chromadb`` being importable; the env-var opt-in (claude-smart's
        # ``CLAUDE_SMART_USE_LOCAL_EMBEDDING``) is enforced earlier in the
        # auto-detection layer (see ``model_defaults._auto_detect_model``).
        if embedding_model.startswith("local/"):
            _reject_cloud_mode(embedding_model, mode)
            if not _is_chromadb_importable():
                raise LiteLLMClientError(
                    f"Embedding model {embedding_model!r} requires chromadb. "
                    "Run `pip install chromadb`."
                )
            try:
                return LocalEmbedder.get().embed([text])[0]
            except Exception as e:
                raise LiteLLMClientError(
                    f"Local embedding generation failed: {str(e)}"
                ) from e

        text = _truncate_for_embedding(text, embedding_model)

        try:
            params = {"model": embedding_model, "input": [text]}
            if dimensions:
                params["dimensions"] = dimensions

            # Resolve and add API key configuration if provided (overrides env vars)
            api_key, api_base, api_version = self._resolve_api_key(
                embedding_model, for_embedding=True
            )
            if api_key:
                params["api_key"] = api_key
            if api_base:
                params["api_base"] = api_base
            if api_version:
                params["api_version"] = api_version

            response = litellm.embedding(
                **params,
                timeout=self.config.timeout,
                num_retries=self.config.max_retries,
            )
            return response.data[0]["embedding"]
        except Exception as e:
            raise LiteLLMClientError(f"Embedding generation failed: {str(e)}") from e

    def get_embeddings(
        self,
        texts: list[str],
        model: str | None = None,
        dimensions: int | None = None,
    ) -> list[list[float]]:
        """
        Get embedding vectors for multiple texts in a single API call.

        Args:
            texts: List of texts to get embeddings for.
            model: Optional embedding model. When omitted, the model is
                auto-detected via ``resolve_model_name(ModelRole.EMBEDDING)``
                so callers inherit the local-embedder gate and any non-OpenAI
                provider configured for this client.
            dimensions: Optional number of dimensions for the embedding vectors.

        Returns:
            List of embedding vectors, one per input text, in the same order as input.

        Raises:
            LiteLLMClientError: If embedding generation fails.
        """
        if not texts:
            return []

        embedding_model = model or self._resolve_default_embedding_model()
        mode = embedding_provider_mode(embedding_model)
        if mode == "off":
            raise EmbeddingUnavailableError("Embedding provider is disabled")
        if should_use_embedding_service(embedding_model):
            return get_service_embeddings(
                list(texts), model=embedding_model, dimensions=dimensions
            )

        # See matching short-circuits in get_embedding above.
        if _is_nomic_model(embedding_model):
            _reject_cloud_mode(embedding_model, mode)
            try:
                return NomicEmbedder.get().embed(list(texts))
            except Exception as e:
                raise LiteLLMClientError(
                    f"Nomic batch embedding generation failed: {str(e)}"
                ) from e

        if embedding_model.startswith("local/"):
            _reject_cloud_mode(embedding_model, mode)
            if not _is_chromadb_importable():
                raise LiteLLMClientError(
                    f"Embedding model {embedding_model!r} requires chromadb. "
                    "Run `pip install chromadb`."
                )
            try:
                return LocalEmbedder.get().embed(list(texts))
            except Exception as e:
                raise LiteLLMClientError(
                    f"Local batch embedding generation failed: {str(e)}"
                ) from e

        texts = [_truncate_for_embedding(t, embedding_model) for t in texts]

        try:
            params = {"model": embedding_model, "input": texts}
            if dimensions:
                params["dimensions"] = dimensions

            # Resolve and add API key configuration if provided (overrides env vars)
            api_key, api_base, api_version = self._resolve_api_key(
                embedding_model, for_embedding=True
            )
            if api_key:
                params["api_key"] = api_key
            if api_base:
                params["api_base"] = api_base
            if api_version:
                params["api_version"] = api_version

            response = litellm.embedding(
                **params,
                timeout=self.config.timeout,
                num_retries=self.config.max_retries,
            )
            # Response data may not be in order, sort by index to ensure correct ordering
            sorted_data = sorted(response.data, key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]
        except Exception as e:
            raise LiteLLMClientError(
                f"Batch embedding generation failed: {str(e)}"
            ) from e

    def _build_completion_params(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> tuple[dict[str, Any], Any, bool, int, list[str]]:
        """Build completion request parameters from messages and kwargs.

        Args:
            messages: List of messages to send
            **kwargs: Additional parameters (response_format, max_retries, model, etc.)

        Returns:
            Tuple of (params dict, response_format, parse_structured_output,
            max_retries, fallback_models). ``fallback_models`` already has any
            entry equal to the primary model removed.
        """
        response_format = kwargs.pop("response_format", None)
        strict_response_format = kwargs.pop("strict_response_format", True)
        parse_structured_output = kwargs.pop("parse_structured_output", True)
        max_retries_arg = kwargs.pop("max_retries", self.config.max_retries)
        try:
            max_retries = max(1, int(max_retries_arg))
        except (TypeError, ValueError):
            max_retries = max(1, int(self.config.max_retries))

        # Per-call fallback_models wins over config when explicitly provided.
        # Use sentinel-style check so an explicit empty list disables fallback
        # for the call even when the config has fallbacks set.
        if "fallback_models" in kwargs:
            fallback_models_raw = kwargs.pop("fallback_models") or []
        else:
            fallback_models_raw = list(self.config.fallback_models)

        # Pop tool-calling kwargs before the final params.update(kwargs) so they
        # don't leak into the params dict twice.
        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)
        model_role: ModelRole | None = kwargs.pop("model_role", None)

        actual_model = kwargs.pop("model", self.config.model)

        # model_role takes priority over the default model but falls through
        # to the custom_endpoint override below (highest priority).
        if model_role is not None:
            actual_model = resolve_model_name(
                role=model_role,
                site_var_value=None,
                config_override=None,
                api_key_config=self.config.api_key_config,
            )

        ce = (
            self.config.api_key_config.custom_endpoint
            if self.config.api_key_config
            else None
        )
        if ce and ce.api_key and ce.api_base:
            actual_model = ce.model

        params: dict[str, Any] = {
            "model": actual_model,
            "messages": messages,
            "timeout": kwargs.pop(
                "timeout", self._effective_timeout_for_model(actual_model)
            ),
        }

        # Drop any fallback entry that points back at the primary — sending the
        # same broken endpoint twice never helps.
        fallback_models = [m for m in fallback_models_raw if m != actual_model]

        temperature = kwargs.pop("temperature", self.config.temperature)
        if self._is_temperature_restricted_model(actual_model):
            params["temperature"] = 1.0
        else:
            params["temperature"] = temperature

        # Determinism knob: `seed` is always injected (defaulting to 42) on
        # providers that honor it, since seed alone is cheap and harmless.
        # The companion temperature=0 override is opt-in via an explicit
        # REFLEXIO_LLM_SEED env var so that caller-configured temperature
        # flows through by default — silently clobbering a user's configured
        # temperature was surprising. Current-gen reasoning models (gpt-5-*)
        # ignore both knobs; the seed is best-effort.
        default_seed = 42
        seed_explicit = "REFLEXIO_LLM_SEED" in os.environ
        seed_raw = os.environ.get("REFLEXIO_LLM_SEED", str(default_seed))
        try:
            params["seed"] = int(seed_raw)
        except ValueError:
            self.logger.warning(
                "REFLEXIO_LLM_SEED=%r is not an int; falling back to default seed=%d",
                seed_raw,
                default_seed,
            )
            params["seed"] = default_seed
        # Keep seed best-effort without mutating LiteLLM's process-wide
        # drop_params setting. Providers that do not support seed can ignore it.
        params["drop_params"] = True
        if seed_explicit and not self._is_temperature_restricted_model(actual_model):
            params["temperature"] = 0.0

        max_tokens = kwargs.pop("max_tokens", self.config.max_tokens)
        if max_tokens:
            params["max_tokens"] = max_tokens
        if self.config.top_p != 1.0:
            params["top_p"] = self.config.top_p
        if response_format:
            params["response_format"] = self._provider_response_format(
                response_format=response_format,
                model=actual_model,
                strict_response_format=strict_response_format,
            )
        if tools is not None:
            params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice

        if actual_model != self.config.model:
            api_key, api_base, api_version = self._resolve_api_key(actual_model)
        else:
            api_key, api_base, api_version = (
                self._api_key,
                self._api_base,
                self._api_version,
            )
        if api_key:
            params["api_key"] = api_key
        if api_base:
            params["api_base"] = api_base
        if api_version:
            params["api_version"] = api_version

        params.update(kwargs)

        # Braintrust metadata for observability (no-op if callback not registered)
        if os.environ.get("BRAINTRUST_API_KEY"):
            params["metadata"] = {
                **params.get("metadata", {}),
                "project_name": os.environ.get("BRAINTRUST_PROJECT_NAME", "reflexio"),
            }
        params["messages"] = self._apply_prompt_caching(
            params["messages"], params["model"]
        )

        return (
            params,
            response_format,
            parse_structured_output,
            max_retries,
            fallback_models,
        )

    @staticmethod
    @lru_cache(maxsize=256)
    def _supports_response_schema(model: str) -> bool:
        try:
            return bool(litellm.supports_response_schema(model=model))
        except Exception:
            return False

    def _provider_response_format(
        self,
        *,
        response_format: Any,
        model: str,
        strict_response_format: bool,
    ) -> Any:
        """Return the provider-facing response_format while preserving parser schema.

        Callers pass a Pydantic model so local parsing stays type-safe. When
        LiteLLM says the target model supports JSON Schema response formats, we
        send an explicit strict schema to constrain generation. Unsupported
        providers keep the existing Pydantic response_format behavior.
        """

        if (
            strict_response_format
            and is_pydantic_model(response_format)
            and self._supports_response_schema(model)
        ):
            return strict_response_format_for_model(response_format)
        return response_format

    def _compute_cost_usd(self, response: Any, model: str | None) -> float | None:
        """Compute call cost in USD via the litellm price table.

        Falls back to None when the provider is not mapped (local ONNX,
        claude-code CLI, etc.) rather than failing the request.

        Args:
            response: Raw LLM response object.
            model: Fully-qualified model name used for the call.

        Returns:
            float | None: Cost in USD, or None when unavailable.
        """
        try:
            import litellm

            cost = litellm.completion_cost(completion_response=response, model=model)
            return float(cost) if cost else None
        except Exception:
            return None

    def _completion_with_hard_timeout(self, params: dict[str, Any]) -> Any:
        """Run ``litellm.completion`` with a client-side wall-clock bound.

        Some providers can exceed LiteLLM's ``timeout`` kwarg. Run the blocking
        call in a child process so the caller can fail, release locks, and
        terminate the in-flight provider request instead of waiting indefinitely.
        """
        provider_timeout = params.get("timeout", self.config.timeout)
        try:
            timeout_seconds = float(provider_timeout)
        except (TypeError, ValueError):
            timeout_seconds = float(self.config.timeout)
        grace_seconds = self._hard_timeout_grace_seconds()
        hard_timeout = max(0.001, timeout_seconds) + max(0.0, grace_seconds)

        if not self._should_process_isolate_completion(timeout_seconds, grace_seconds):
            return litellm.completion(**params)

        process_context = multiprocessing.get_context()
        result_queue = process_context.Queue(maxsize=1)
        process = process_context.Process(
            target=_litellm_completion_worker,
            args=(params, result_queue),
            daemon=True,
        )
        process.start()
        try:
            process.join(timeout=hard_timeout)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.0)
                raise LLMHardTimeoutError(
                    f"LLM request exceeded hard timeout of {hard_timeout:.3f}s "
                    f"(provider timeout={provider_timeout!r})"
                )

            try:
                status, payload = result_queue.get(timeout=1.0)
            except queue.Empty as exc:
                raise LiteLLMClientError(
                    "LLM request process exited without returning a result "
                    f"(exitcode={process.exitcode})"
                ) from exc

            if status == "ok":
                return payload
            # The worker always reports errors as a picklable snapshot.
            context_parts = [f"model={payload.model}"]
            if payload.llm_provider:
                context_parts.append(f"provider={payload.llm_provider}")
            raise LiteLLMClientError(
                "litellm.completion failed in isolated worker: "
                f"{payload.type_name}: {payload.message} "
                f"({', '.join(context_parts)})"
            )
        finally:
            result_queue.close()
            result_queue.join_thread()

    def _effective_timeout_for_model(self, model: str) -> int:
        """Return the configured timeout, raised to the model's floor if one exists.

        Args:
            model: Resolved model name (e.g. 'minimax/MiniMax-M3').

        Returns:
            int: max(config.timeout, per-model floor). Callers that pass an
            explicit timeout kwarg bypass this entirely.
        """
        return max(self.config.timeout, _MODEL_TIMEOUT_FLOOR_SECONDS.get(model, 0))

    def _hard_timeout_grace_seconds(self) -> float:
        raw = os.environ.get("REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS", "5") or "5"
        try:
            return max(0.0, float(raw))
        except ValueError:
            self.logger.warning(
                "Invalid REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS=%r; using 5",
                raw,
            )
            return 5.0

    def _should_process_isolate_completion(
        self, timeout_seconds: float, grace_seconds: float
    ) -> bool:
        """Use process isolation for real LiteLLM calls while preserving test doubles.

        Unit tests often monkeypatch ``litellm.completion`` with local closures
        that capture params in parent memory. Those closures cannot be observed
        through a subprocess, so only real LiteLLM functions and explicit short
        timeout tests go through the process path.
        """
        completion_module = getattr(litellm.completion, "__module__", "")
        if completion_module.startswith("litellm"):
            return True
        return timeout_seconds + grace_seconds < 1.0

    def _log_token_usage(self, params: dict[str, Any], response: Any) -> None:
        """Log token usage with cache statistics and cost from an LLM response.

        Args:
            params: Request parameters (for model name)
            response: LLM response object
        """
        usage = getattr(response, "usage", None)
        if not usage:
            return

        cache_info = ""
        details = getattr(usage, "prompt_tokens_details", None)
        if details:
            cached = getattr(details, "cached_tokens", 0)
            if cached:
                cache_info = f", cached: {cached}"
        cache_creation = getattr(usage, "cache_creation_input_tokens", None)
        cache_read = getattr(usage, "cache_read_input_tokens", None)
        if cache_creation or cache_read:
            cache_info = (
                f", cache_write: {cache_creation or 0}, cache_read: {cache_read or 0}"
            )

        cost = self._compute_cost_usd(response, params.get("model"))
        cost_suffix = f", cost: ${cost:.6f}" if cost is not None else ""

        self.logger.info(
            "Token usage - model: %s, input: %s, output: %s, total: %s%s%s",
            params.get("model"),
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
            cache_info,
            cost_suffix,
        )

    def _emit_fallback_observability(
        self, response: Any, params: dict[str, Any]
    ) -> None:
        """Surface fallback-routing info to logs and Sentry when applicable.

        LiteLLM rewrites ``response.model`` to the model that actually served
        the call, so we detect a fallback by comparing it against the model
        we asked for. The check is best-effort: any exception inside this
        helper is swallowed so observability never breaks the request.

        Args:
            response: The litellm completion response object.
            params: The params dict that was passed to ``litellm.completion`` —
                used to read the originally requested primary model name.
        """
        try:
            primary_model = params.get("model")
            hidden = getattr(response, "_hidden_params", {}) or {}
            served_model = (
                hidden.get("model_id")
                or hidden.get("model")
                or getattr(response, "model", None)
            )

            if not served_model or served_model == primary_model:
                return

            self.logger.info(
                "event=llm_fallback_used primary_model=%s served_model=%s",
                primary_model,
                served_model,
            )

            # Local import keeps sentry out of module-init paths the tests
            # exercise without a Sentry SDK installed. sentry_sdk is an
            # enterprise-only dependency; OSS callers run without it and the
            # ImportError is intentionally absorbed by the outer except.
            import sentry_sdk  # type: ignore[import-not-found]

            sentry_sdk.set_tag("llm.fallback_used", "true")
            sentry_sdk.set_tag("llm.primary_model", str(primary_model))
            sentry_sdk.set_tag("llm.fallback_model", str(served_model))
        except Exception:  # noqa: BLE001 — observability must not break the call
            return

    def _make_request(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> str | BaseModel | ToolCallingChatResponse:
        """
        Make a request to the LLM, delegating retries and fallback to litellm.

        Retry and fallback semantics are handed to ``litellm.completion`` via
        the native ``num_retries`` and ``fallbacks`` kwargs. Per the documented
        flow at https://docs.litellm.ai/docs/router_architecture, the primary
        model is tried ``num_retries+1`` times, then each fallback gets a single
        attempt. The one piece we still own at the client level is a single
        retry for ``StructuredOutputParseError``: LiteLLM cannot detect a
        post-hoc Pydantic re-validation failure because it sees a successful
        HTTP response.

        Args:
            messages: List of messages to send.
            **kwargs: Additional parameters (response_format, max_retries,
                fallback_models, tools, etc.).

        Returns:
            Response content as string, BaseModel instance, or
            ToolCallingChatResponse when the request was in tool-calling mode.

        Raises:
            LiteLLMClientError: If the request fails after all retries and
                fallbacks have been exhausted by litellm.
        """
        params, response_format, parse_structured_output, max_retries, fallbacks = (
            self._build_completion_params(messages, **kwargs)
        )

        # Hand retries + fallbacks to litellm. ``num_retries`` is the documented
        # alias for max_retries on litellm.completion.
        params["num_retries"] = max_retries
        if fallbacks:
            params["fallbacks"] = fallbacks

        request_start = time.perf_counter()
        self.logger.info(
            "event=llm_request_start model=%s timeout=%s has_response_format=%s num_retries=%d fallbacks=%s",
            params.get("model"),
            params.get("timeout"),
            response_format is not None,
            max_retries,
            fallbacks,
        )

        def _call_and_parse() -> str | BaseModel | ToolCallingChatResponse:
            response = self._completion_with_hard_timeout(params)
            self._emit_fallback_observability(response, params)
            message = response.choices[0].message  # type: ignore[reportAttributeAccessIssue]
            content = message.content
            self._log_token_usage(params, response)
            self.logger.info(
                "event=llm_request_end model=%s timeout=%s has_response_format=%s elapsed_seconds=%.3f success=%s",
                params.get("model"),
                params.get("timeout"),
                response_format is not None,
                time.perf_counter() - request_start,
                True,
            )

            # Tool-calling path: return a structured response instead of
            # going through _maybe_parse_structured_output.
            if "tools" in params:
                raw_usage = getattr(response, "usage", None)
                call_cost = self._compute_cost_usd(response, params.get("model"))
                return ToolCallingChatResponse(
                    content=content,
                    tool_calls=getattr(message, "tool_calls", None),
                    finish_reason=response.choices[0].finish_reason,  # type: ignore[reportAttributeAccessIssue]
                    usage=raw_usage,
                    cost_usd=call_cost,
                )

            return self._maybe_parse_structured_output(
                content,  # type: ignore[reportArgumentType]
                response_format,
                parse_structured_output,
            )

        try:
            try:
                return _call_and_parse()
            except StructuredOutputParseError:
                # LiteLLM's num_retries covers API errors, but a Pydantic
                # re-validation failure happens AFTER litellm sees a
                # successful 200 — so we owe one explicit second attempt at
                # the model. PR #121 documented this as a MiniMax-M3
                # mitigation.
                self.logger.warning(
                    "event=llm_parse_retry model=%s — primary returned malformed structured output, retrying once",
                    params.get("model"),
                )
                return _call_and_parse()
            except LLMHardTimeoutError:
                # The hard timeout kills the litellm subprocess, so litellm's
                # num_retries never gets a chance — we owe one explicit retry
                # at this level to cover transient provider hangs.
                self.logger.warning(
                    "event=llm_hard_timeout_retry model=%s — request hit hard timeout, retrying once",
                    params.get("model"),
                )
                return _call_and_parse()
        except Exception as e:
            self.logger.error(
                "event=llm_request_end model=%s elapsed_seconds=%.3f success=False error_type=%s error=%s",
                params.get("model"),
                time.perf_counter() - request_start,
                type(e).__name__,
                e,
            )
            raise LiteLLMClientError(f"API call failed: {e}") from e

    def _apply_prompt_caching(
        self, messages: list[dict[str, Any]], model: str
    ) -> list[dict[str, Any]]:
        """
        Apply prompt caching markers for supported providers.

        For Anthropic models, transforms the system message content into content-block
        format with cache_control markers to enable prefix caching.
        For other providers, returns messages unchanged.

        Args:
            messages: List of chat messages.
            model: Model name to determine provider.

        Returns:
            list[dict]: Messages with cache control applied where appropriate.
        """
        model_lower = model.lower()
        # The claude-code/* custom provider routes through the Claude Code CLI,
        # which does not accept Anthropic API cache_control content blocks.
        if model_lower.startswith("claude-code/"):
            return messages
        is_anthropic = "claude" in model_lower or "anthropic" in model_lower

        if not is_anthropic:
            return messages

        result = []
        for msg in messages:
            if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                # Transform system message to content-block format with cache_control
                result.append(
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": msg["content"],
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                )
            else:
                result.append(msg)

        return result

    def _build_user_content(
        self,
        prompt: str,
        images: list[str | bytes | dict] | None = None,
        image_media_type: str | None = None,
    ) -> str | list[dict[str, Any]]:
        """
        Build user content with optional images.

        Args:
            prompt: Text prompt.
            images: Optional list of images.
            image_media_type: Media type for byte images.

        Returns:
            String for text-only, or list of content blocks for multi-modal.
        """
        if not images:
            return prompt

        content_blocks = [{"type": "text", "text": prompt}]

        for image in images:
            if isinstance(image, dict):
                # Already formatted content block
                content_blocks.append(image)
            elif isinstance(image, bytes):
                # Raw bytes
                media_type = image_media_type or "image/png"
                base64_data = base64.b64encode(image).decode("utf-8")
                content_blocks.append(
                    self._create_image_content_block(base64_data, media_type)
                )
            elif isinstance(image, str):
                # File path or URL
                if image.startswith(("http://", "https://")):
                    # URL - use directly
                    content_blocks.append(
                        {"type": "image_url", "image_url": {"url": image}}  # type: ignore[reportArgumentType]
                    )
                else:
                    # File path
                    base64_data, media_type = self.encode_image_to_base64(image)
                    content_blocks.append(
                        self._create_image_content_block(base64_data, media_type)
                    )

        return content_blocks

    def _create_image_content_block(
        self, base64_data: str, media_type: str
    ) -> dict[str, Any]:
        """
        Create an image content block for the API.

        Args:
            base64_data: Base64-encoded image data.
            media_type: MIME type of the image.

        Returns:
            Image content block dictionary.
        """
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{base64_data}"},
        }

    def encode_image_to_base64(self, image_path: str) -> tuple[str, str]:
        """
        Encode an image file to base64.

        Delegates to :func:`reflexio.server.llm.image_utils.encode_image_to_base64`
        and wraps errors as :class:`LiteLLMClientError`.

        Args:
            image_path (str): Path to the image file.

        Returns:
            tuple[str, str]: ``(base64_data, media_type)`` pair.

        Raises:
            LiteLLMClientError: If the image cannot be read or format is unsupported.
        """
        try:
            return _encode_image_to_base64(image_path)
        except ImageEncodingError as exc:
            raise LiteLLMClientError(str(exc)) from exc

    def _is_temperature_restricted_model(self, model: str) -> bool:
        """
        Check if a model has temperature restrictions (e.g., GPT-5 and Gemini 3 models only support temperature=1.0).

        Args:
            model: Model name to check.

        Returns:
            True if the model has temperature restrictions.
        """
        model_lower = model.lower()
        # Strip provider routing prefixes (e.g., "openrouter/openai/gpt-5-nano" -> "gpt-5-nano")
        model_name = model_lower.rsplit("/", 1)[-1]
        # Check if model starts with any of the restricted model prefixes
        return any(
            model_name.startswith(restricted) or model_name == restricted
            for restricted in self.TEMPERATURE_RESTRICTED_MODELS
        )

    def _maybe_parse_structured_output(
        self,
        content: str,
        response_format: Any,
        parse_structured_output: bool,
    ) -> str | BaseModel:
        """
        Parse structured output if applicable.

        Args:
            content: Raw response content.
            response_format: Expected response format (must be a Pydantic BaseModel class).
            parse_structured_output: Whether to parse the output.

        Returns:
            String for text responses, or BaseModel instance for structured responses.
        """
        if not response_format or not parse_structured_output:
            return content

        if content is None:
            return content

        # If content is already a Pydantic model (some providers return parsed)
        if isinstance(content, BaseModel):
            return content

        # Try to parse JSON and convert to Pydantic model
        # Extract JSON from markdown code blocks if present
        json_str = self._extract_json_from_string(content)
        try:
            parsed = json.loads(json_str)

            # response_format must be a Pydantic model (validated at entry points)
            return response_format.model_validate(parsed)
        except Exception:
            # LLMs sometimes produce Python-style output (single quotes, True/False,
            # trailing commas). Try to sanitize before giving up.
            try:
                sanitized = self._sanitize_json_string(json_str)
                parsed = json.loads(sanitized)
                return response_format.model_validate(parsed)
            except Exception:
                # Last resort: json-repair can recover complete responses with
                # small syntax glitches, such as missing commas. Do not repair
                # likely truncation: the retry loop should request a fresh
                # complete response instead of accepting invented tail content.
                try:
                    from json_repair import repair_json

                    if self._looks_truncated_json(json_str):
                        raise StructuredOutputParseError(
                            "Structured output appears truncated"
                        )

                    repaired = repair_json(json_str, return_objects=True)
                    return response_format.model_validate(repaired)
                except Exception as e:
                    model = self.config.model
                    snippet = (
                        content[:200]
                        if isinstance(content, str)
                        else repr(content)[:200]
                    )
                    raise StructuredOutputParseError(
                        f"Structured output parse failed for model={model!r}: {e}. "
                        f"Content snippet: {snippet!r}"
                    ) from e

    def _extract_json_from_string(self, content: str) -> str:
        """
        Extract JSON from a string, handling markdown code blocks.

        Args:
            content: String potentially containing JSON.

        Returns:
            Extracted JSON string.
        """
        content = content.strip()

        # Prefer a balanced JSON container first. Structured JSON may contain
        # markdown fences inside string values; grabbing the first code block
        # would extract the inner snippet instead of the response object.
        json_container = self._extract_first_json_container(content)
        if json_container is not None:
            return json_container

        # Try to extract from markdown code blocks
        json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        matches = re.findall(json_block_pattern, content)
        if matches:
            return matches[0].strip()

        return content

    def _extract_first_json_container(self, content: str) -> str | None:
        """Return the first balanced JSON-like object/array in ``content``."""
        for start_idx, ch in enumerate(content):
            if ch not in "{[":
                continue
            end_idx = self._find_json_container_end(content, start_idx)
            if end_idx is None:
                continue
            candidate = content[start_idx : end_idx + 1]
            if self._is_parseable_json_candidate(candidate):
                return candidate
        return None

    @staticmethod
    def _find_json_container_end(content: str, start_idx: int) -> int | None:
        """Find the matching end of a JSON container, respecting strings."""
        pairs = {"{": "}", "[": "]"}
        stack = [pairs[content[start_idx]]]
        in_str = False
        escape = False

        for idx in range(start_idx + 1, len(content)):
            ch = content[idx]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch in pairs:
                stack.append(pairs[ch])
            elif ch in ("}", "]"):
                if not stack or stack.pop() != ch:
                    return None
                if not stack:
                    return idx
        return None

    def _is_parseable_json_candidate(self, candidate: str) -> bool:
        """Return True if a balanced candidate can parse after normal sanitizing."""
        try:
            json.loads(candidate)
            return True
        except Exception:
            try:
                json.loads(self._sanitize_json_string(candidate))
                return True
            except Exception:
                return False

    def _looks_truncated_json(self, json_str: str) -> bool:
        """
        Return True when a JSON-like string appears to end before it is complete.

        This intentionally only treats content with a JSON container opener as
        truncation. Plain text that is not JSON should proceed to the normal
        parse failure path.

        Args:
            json_str: Extracted JSON-like response text.

        Returns:
            True if the response has unclosed containers or strings.
        """
        stripped = json_str.strip()
        start_indices = [
            idx for idx in (stripped.find("{"), stripped.find("[")) if idx != -1
        ]
        if not stripped or not start_indices:
            return False
        stripped = stripped[min(start_indices) :]

        stack: list[str] = []
        in_str = False
        escape = False
        pairs = {"{": "}", "[": "]"}

        for ch in stripped:
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch in pairs:
                stack.append(pairs[ch])
            elif ch in ("}", "]") and (not stack or stack.pop() != ch):
                return False

        return in_str or bool(stack)

    def _sanitize_json_string(self, json_str: str) -> str:
        """
        Sanitize a JSON-like string that uses Python-style syntax into valid JSON.

        Handles common LLM issues: single quotes, Python True/False/None,
        and trailing commas before closing braces/brackets.

        Args:
            json_str: A JSON-like string that may contain Python-style syntax.

        Returns:
            A sanitized string closer to valid JSON.
        """
        s = json_str

        # Walk character-by-character to:
        #   1. Replace single-quoted strings with double-quoted strings
        #   2. Replace Python True/False/None with JSON true/false/null ONLY outside strings
        #   3. Handle escaped apostrophes inside single-quoted strings (e.g. 'didn\'t')
        #   4. Escape literal double quotes that end up inside double-quoted strings
        result = []
        in_double = False
        in_single = False
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == "\\" and (in_double or in_single):
                # Escaped character inside a string
                if i + 1 < len(s):
                    next_ch = s[i + 1]
                    if in_single and next_ch == "'":
                        # \' inside single-quoted string → literal apostrophe
                        # In JSON double-quoted strings, apostrophe needs no escape
                        result.append("'")
                        i += 2
                        continue
                    result.append(ch)
                    result.append(next_ch)
                    i += 2
                    continue
                result.append(ch)
            elif ch == '"' and not in_single:
                in_double = not in_double
                result.append(ch)
            elif ch == "'" and not in_double:
                in_single = not in_single
                result.append('"')  # swap single → double
            else:
                # Escape unescaped double quotes inside single-quoted strings
                # (they become part of a double-quoted JSON string)
                if in_single and ch == '"':
                    result.append('\\"')
                else:
                    result.append(ch)
            i += 1
        s = "".join(result)

        # Replace Python booleans/None with JSON equivalents only outside quoted strings.
        # We walk the already-double-quoted result so we only need to track double quotes.
        output = []
        in_str = False
        j = 0
        while j < len(s):
            if s[j] == "\\" and in_str:
                output.append(s[j : j + 2])
                j += 2
                continue
            if s[j] == '"':
                in_str = not in_str
                output.append(s[j])
                j += 1
                continue
            if not in_str:
                matched = False
                for py_val, json_val in _PYTHON_TO_JSON_REPLACEMENTS.items():
                    if s[j : j + len(py_val)] == py_val:
                        # Check word boundaries
                        before = s[j - 1] if j > 0 else " "
                        after = s[j + len(py_val)] if j + len(py_val) < len(s) else " "
                        if (
                            not before.isalnum()
                            and before != "_"
                            and not after.isalnum()
                            and after != "_"
                        ):
                            output.append(json_val)
                            j += len(py_val)
                            matched = True
                            break
                if not matched:
                    output.append(s[j])
                    j += 1
            else:
                output.append(s[j])
                j += 1
        s = "".join(output)

        # Remove trailing commas before } or ]
        return re.sub(r",\s*([}\]])", r"\1", s)

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
