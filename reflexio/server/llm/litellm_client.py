"""
LiteLLM-based unified LLM client.

This module provides a unified interface to multiple LLM providers (OpenAI, Claude, Azure OpenAI)
using LiteLLM. It maintains the same interface as the existing LLMClient for easy replacement.
"""

import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass
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
from reflexio.server.llm.llm_utils import is_pydantic_model

# Suppress LiteLLM's verbose logging
litellm.suppress_debug_info = True

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
        model: Model name to use (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022', 'azure/gpt-4')
        temperature: Temperature for response generation (0.0 to 2.0)
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds (exponential backoff)
        top_p: Top-p sampling parameter
        api_key_config: Optional API key configuration from Config (overrides env vars)
    """

    model: str
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout: int = 120
    max_retries: int = 1
    retry_delay: float = 1.0
    top_p: float = 1.0
    api_key_config: APIKeyConfig | None = None


class LiteLLMClientError(Exception):
    """Custom exception for LiteLLM client errors."""


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

    # Non-retryable error patterns
    NON_RETRYABLE_ERRORS = [
        "invalid_api_key",
        "unauthorized",
        "permission_denied",
        "quota_exceeded",
        "billing",
        "invalid_request",
        "authentication",
        "forbidden",
        "rate_limit",  # Rate limits are handled by LiteLLM internally
    ]

    # Models that only support temperature=1.0 (custom values cause errors or degraded performance)
    TEMPERATURE_RESTRICTED_MODELS = {
        "gpt-5",
        "gpt-5-mini",
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
        **kwargs,
    ) -> str | BaseModel:
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
        **kwargs,
    ) -> str | BaseModel:
        """
        Generate a response from a list of chat messages.

        Args:
            messages: List of messages in chat format [{"role": "...", "content": "..."}].
            system_message: Optional system message to prepend.
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

        return self._make_request(final_messages, **kwargs)

    def get_embedding(
        self, text: str, model: str | None = None, dimensions: int | None = None
    ) -> list[float]:
        """
        Get embedding vector for the given text.

        Args:
            text: The text to get embedding for.
            model: Optional embedding model (defaults to 'text-embedding-3-small').
            dimensions: Optional number of dimensions for the embedding vector.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            LiteLLMClientError: If embedding generation fails.
        """
        embedding_model = model or "text-embedding-3-small"
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

            response = litellm.embedding(**params, timeout=self.config.timeout)
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
            model: Optional embedding model (defaults to 'text-embedding-3-small').
            dimensions: Optional number of dimensions for the embedding vectors.

        Returns:
            List of embedding vectors, one per input text, in the same order as input.

        Raises:
            LiteLLMClientError: If embedding generation fails.
        """
        if not texts:
            return []

        embedding_model = model or "text-embedding-3-small"
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

            response = litellm.embedding(**params, timeout=self.config.timeout)
            # Response data may not be in order, sort by index to ensure correct ordering
            sorted_data = sorted(response.data, key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]
        except Exception as e:
            raise LiteLLMClientError(
                f"Batch embedding generation failed: {str(e)}"
            ) from e

    def _build_completion_params(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> tuple[dict[str, Any], Any, bool, int]:
        """Build completion request parameters from messages and kwargs.

        Args:
            messages: List of messages to send
            **kwargs: Additional parameters (response_format, max_retries, model, etc.)

        Returns:
            Tuple of (params dict, response_format, parse_structured_output, max_retries)
        """
        response_format = kwargs.pop("response_format", None)
        parse_structured_output = kwargs.pop("parse_structured_output", True)
        max_retries_arg = kwargs.pop("max_retries", self.config.max_retries)
        try:
            max_retries = max(1, int(max_retries_arg))
        except TypeError, ValueError:
            max_retries = max(1, int(self.config.max_retries))

        actual_model = kwargs.pop("model", self.config.model)
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
            "timeout": kwargs.pop("timeout", self.config.timeout),
            "num_retries": 0,
        }

        temperature = kwargs.pop("temperature", self.config.temperature)
        if self._is_temperature_restricted_model(actual_model):
            params["temperature"] = 1.0
        else:
            params["temperature"] = temperature

        max_tokens = kwargs.pop("max_tokens", self.config.max_tokens)
        if max_tokens:
            params["max_tokens"] = max_tokens
        if self.config.top_p != 1.0:
            params["top_p"] = self.config.top_p
        if response_format:
            params["response_format"] = response_format

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

        return params, response_format, parse_structured_output, max_retries

    def _log_token_usage(self, params: dict[str, Any], response: Any) -> None:
        """Log token usage with cache statistics from an LLM response.

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

        self.logger.info(
            "Token usage - model: %s, input: %s, output: %s, total: %s%s",
            params.get("model"),
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
            cache_info,
        )

    def _handle_retry_or_raise(
        self,
        error: Exception,
        params: dict[str, Any],
        attempt: int,
        max_retries: int,
        response_format: Any,
        elapsed_seconds: float,
    ) -> None:
        """Handle retry logic or raise on non-retryable/final errors.

        Args:
            error: The exception that occurred
            params: Request parameters (for logging)
            attempt: Current attempt index (0-based)
            max_retries: Maximum number of retries
            response_format: Response format (for logging)
            elapsed_seconds: Time elapsed for this attempt

        Raises:
            LiteLLMClientError: If the error is non-retryable or this was the last attempt
        """
        error_str = str(error).lower()

        self.logger.error(
            "event=llm_request_end model=%s timeout=%s has_response_format=%s attempt=%d/%d elapsed_seconds=%.3f success=%s error_type=%s error=%s",
            params.get("model"),
            params.get("timeout"),
            response_format is not None,
            attempt + 1,
            max_retries,
            elapsed_seconds,
            False,
            type(error).__name__,
            str(error),
        )

        if self._is_non_retryable_error(error_str):
            self.logger.error("Non-retryable error: %s", error)
            raise LiteLLMClientError(f"API call failed: {str(error)}") from error

        if attempt < max_retries - 1:
            delay = self.config.retry_delay * (2**attempt)
            self.logger.warning(
                "Request failed (attempt %s/%s): %s. Retrying in %ss...",
                attempt + 1,
                max_retries,
                error,
                delay,
            )
            time.sleep(delay)
        else:
            self.logger.error(
                "LLM request failed (model=%s, has_response_format=%s): %s",
                params.get("model"),
                response_format is not None,
                error,
            )

    def _make_request(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> str | BaseModel:
        """
        Make a request to the LLM with retry logic.

        Args:
            messages: List of messages to send.
            **kwargs: Additional parameters.

        Returns:
            Response content as string or BaseModel instance.

        Raises:
            LiteLLMClientError: If the request fails after all retries.
        """
        params, response_format, parse_structured_output, max_retries = (
            self._build_completion_params(messages, **kwargs)
        )

        last_error: Exception | None = None
        for attempt in range(max_retries):
            request_start = time.perf_counter()
            self.logger.info(
                "event=llm_request_start model=%s timeout=%s has_response_format=%s attempt=%d/%d",
                params.get("model"),
                params.get("timeout"),
                response_format is not None,
                attempt + 1,
                max_retries,
            )
            try:
                response = litellm.completion(**params)
                content = response.choices[0].message.content  # type: ignore[reportAttributeAccessIssue]
                elapsed_seconds = time.perf_counter() - request_start

                self._log_token_usage(params, response)

                self.logger.info(
                    "event=llm_request_end model=%s timeout=%s has_response_format=%s attempt=%d/%d elapsed_seconds=%.3f success=%s",
                    params.get("model"),
                    params.get("timeout"),
                    response_format is not None,
                    attempt + 1,
                    max_retries,
                    elapsed_seconds,
                    True,
                )

                return self._maybe_parse_structured_output(
                    content,  # type: ignore[reportArgumentType]
                    response_format,
                    parse_structured_output,  # type: ignore[reportArgumentType]
                )

            except Exception as e:
                last_error = e
                elapsed_seconds = time.perf_counter() - request_start
                self._handle_retry_or_raise(
                    e, params, attempt, max_retries, response_format, elapsed_seconds
                )

        raise LiteLLMClientError(
            f"API call failed after {max_retries} retries: {str(last_error)}"
        )

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
            except Exception as e:
                self.logger.warning("Failed to parse structured output: %s", e)
                return content

    def _extract_json_from_string(self, content: str) -> str:
        """
        Extract JSON from a string, handling markdown code blocks.

        Args:
            content: String potentially containing JSON.

        Returns:
            Extracted JSON string.
        """
        content = content.strip()

        # Try to extract from markdown code blocks
        json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        matches = re.findall(json_block_pattern, content)
        if matches:
            return matches[0].strip()

        # Try to find JSON object or array
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start_idx = content.find(start_char)
            end_idx = content.rfind(end_char)
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                return content[start_idx : end_idx + 1]

        return content

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

    def _is_non_retryable_error(self, error_str: str) -> bool:
        """
        Check if an error is non-retryable.

        Args:
            error_str: Error message string.

        Returns:
            True if error should not be retried.
        """
        return any(pattern in error_str for pattern in self.NON_RETRYABLE_ERRORS)

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
