"""
LiteLLM-based unified LLM client.

This module provides a unified interface to multiple LLM providers (OpenAI, Claude, Azure OpenAI)
using LiteLLM. It maintains the same interface as the existing LLMClient for easy replacement.
"""

import base64
import logging
import multiprocessing
import os
import queue
import time
from typing import Any

import litellm
from pydantic import BaseModel

from reflexio.models.config_schema import APIKeyConfig
from reflexio.server.llm._litellm_embedding import (
    _TRUNCATION_WARNED_MODELS as _TRUNCATION_WARNED_MODELS,
)

# Identity-preserving re-exports (SINK-2): every moved name is re-bound here by
# import, never redefined, so ``from ...litellm_client import <name>`` keeps
# resolving the SAME object/class the moved code uses and tests touch.
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
    _litellm_completion_worker,
)
from reflexio.server.llm._litellm_subprocess import (
    _PromptTokenDetailsSnapshot as _PromptTokenDetailsSnapshot,
)
from reflexio.server.llm._litellm_types import (
    LiteLLMClientError,
    LiteLLMConfig,
    LLMHardTimeoutError,
    StructuredOutputParseError,
    ToolCallingChatResponse,
)
from reflexio.server.llm.image_utils import (
    SUPPORTED_IMAGE_MIME_TYPES,
    ImageEncodingError,
)
from reflexio.server.llm.image_utils import (
    encode_image_to_base64 as _encode_image_to_base64,
)
from reflexio.server.llm.llm_utils import (
    is_pydantic_model,
)
from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name
from reflexio.server.llm.providers.claude_code_provider import (
    register_if_enabled as _register_claude_code,
)
from reflexio.server.llm.providers.local_embedding_provider import (
    register_if_chromadb_available as _register_local_embedder,
)
from reflexio.server.llm.providers.nomic_embedding_provider import (
    register_if_enabled as _register_nomic_embedder,
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
_register_local_embedder()
_register_nomic_embedder()

# Public importer surface (the #1 invariant of the Tier-2.5 decomposition). These
# five names — plus the test-imported internals re-exported below the split — must
# stay importable from ``reflexio.server.llm.litellm_client`` for all ~102 importers.
__all__ = [
    "LiteLLMClient",
    "LiteLLMConfig",
    "LiteLLMClientError",
    "ToolCallingChatResponse",
    "create_litellm_client",
]

_LOGGER = logging.getLogger(__name__)


# Per-model provider-timeout floors. Values are floors, not overrides: the
# effective timeout is max(configured, floor), and an explicit per-call timeout
# kwarg always wins.
#
# MiniMax-M3 was pinned to 240s when it was the sole model. That let a *hung*
# primary block ~240s before falling back, dominating the wasted time behind
# Sentry PYTHON-FASTAPI-62. It is now floored at the 120s default so a hang is
# abandoned sooner and the fallback (e.g. gpt-5-mini) is reached faster. This is
# the key post-deploy tuning knob: raise it if legitimately-slow calls start
# timing out, lower it to cut more waste.
_MODEL_TIMEOUT_FLOOR_SECONDS: dict[str, int] = {
    "minimax/MiniMax-M3": 120,
}


class LiteLLMClient(EmbeddingMixin, StructuredOutputMixin):
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

    def _coerce_timeout_seconds(self, params: dict[str, Any]) -> float:
        """Coerce ``params['timeout']`` to a float, falling back to the config
        default when it is missing or non-numeric."""
        try:
            return float(params.get("timeout", self.config.timeout))
        except (TypeError, ValueError):
            return float(self.config.timeout)

    def _completion_with_hard_timeout(
        self, params: dict[str, Any], hard_timeout: float
    ) -> Any:
        """Run ``litellm.completion`` with a client-side wall-clock bound.

        Some providers can exceed LiteLLM's ``timeout`` kwarg. Run the blocking
        call in a child process so the caller can fail, release locks, and
        terminate the in-flight provider request instead of waiting indefinitely.

        ``hard_timeout`` is the wall-clock kill bound for the whole subprocess.
        Because LiteLLM walks ``[primary, *fallbacks]`` inside this one call
        (copying ``timeout`` unchanged into each rung), the caller sizes
        ``hard_timeout`` to cover the entire fallback ladder, not a single
        attempt — otherwise the subprocess would be killed before LiteLLM ever
        reaches a fallback (the root cause of Sentry PYTHON-FASTAPI-62).
        """
        provider_timeout = params.get("timeout", self.config.timeout)
        # timeout_seconds + grace_seconds below only classify test doubles in
        # _should_process_isolate_completion (real litellm vs a monkeypatched
        # closure) — they do NOT size the kill bound, which is the caller's
        # ladder-wide ``hard_timeout``.
        timeout_seconds = self._coerce_timeout_seconds(params)
        grace_seconds = self._hard_timeout_grace_seconds()
        hard_timeout = max(0.001, hard_timeout)

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
        Make a request to the LLM, delegating cross-model fallback to litellm.

        Fallback is handed to ``litellm.completion`` via the native ``fallbacks``
        kwarg, but ``num_retries`` is forced to 0: same-model retry of a *hung*
        primary is what made the fallback unreachable and produced the 490s in
        Sentry PYTHON-FASTAPI-62 (see the body comment). So the primary is tried
        once, then each fallback once. The subprocess hard timeout is sized to
        cover that whole ladder. The one retry we still own at the client level
        is a single ``StructuredOutputParseError`` retry: LiteLLM cannot detect a
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
        params, response_format, parse_structured_output, _max_retries, fallbacks = (
            self._build_completion_params(messages, **kwargs)
        )

        # Hand the fallback ladder to litellm, but DISABLE same-model retries.
        # litellm walks [primary, *fallbacks] inside one litellm.completion call,
        # copying ``timeout`` unchanged into each rung. With num_retries>=1 it
        # retries a *hung* primary num_retries+1 times (each up to a full
        # provider timeout) before ever reaching a fallback — making the fallback
        # unreachable within any sane wall-clock bound (root cause of Sentry
        # PYTHON-FASTAPI-62). num_retries=0 makes the fallback LIST the resilience
        # mechanism: each model is tried once, in order.
        params["num_retries"] = 0
        if fallbacks:
            params["fallbacks"] = fallbacks

        # Size the hard (wall-clock) timeout to cover the WHOLE ladder. litellm
        # copies this single ``params["timeout"]`` into EVERY rung (primary + each
        # fallback), so every rung shares the primary's per-attempt budget and the
        # subprocess must be allowed to run ``(1 + len(fallbacks))`` of them plus
        # one grace buffer before being killed — otherwise it is killed before
        # litellm can reach a fallback.
        #
        # ASYMMETRIC-FLOOR FOOTGUN: because every rung shares one timeout, a
        # fallback whose _MODEL_TIMEOUT_FLOOR_SECONDS floor is HIGHER than the
        # primary's would run — and be killed — at the primary's shorter timeout,
        # reintroducing the "fallback killed early" failure this fix removes. The
        # floor table is single-valued today (MiniMax-M3 == the 120 default), so
        # this is latent; revisit the sizing (e.g. max floor across rungs, passed
        # as ``params["timeout"]``) before adding an asymmetric floor entry.
        per_attempt_timeout = self._coerce_timeout_seconds(params)
        hard_timeout = (
            1 + len(fallbacks)
        ) * per_attempt_timeout + self._hard_timeout_grace_seconds()

        request_start = time.perf_counter()
        self.logger.info(
            "event=llm_request_start model=%s timeout=%s has_response_format=%s num_retries=0 fallbacks=%s hard_timeout=%.3f",
            params.get("model"),
            params.get("timeout"),
            response_format is not None,
            fallbacks,
            hard_timeout,
        )

        def _call_and_parse() -> str | BaseModel | ToolCallingChatResponse:
            response = self._completion_with_hard_timeout(params, hard_timeout)
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
                tool_calls = getattr(message, "tool_calls", None)
                # Structured-output + tools: when the model ends the turn with a
                # plain (non-tool) response and a response_format was requested,
                # the content IS the final structured answer. Parse it here so a
                # tool-loop caller can finish on it. A malformed parse raises
                # StructuredOutputParseError, which the outer wrapper retries once.
                parsed_output: BaseModel | None = None
                if response_format is not None and not tool_calls:
                    parsed = self._maybe_parse_structured_output(
                        content,  # type: ignore[reportArgumentType]
                        response_format,
                        parse_structured_output,
                    )
                    if isinstance(parsed, BaseModel):
                        parsed_output = parsed
                return ToolCallingChatResponse(
                    content=content,
                    tool_calls=tool_calls,
                    finish_reason=response.choices[0].finish_reason,  # type: ignore[reportAttributeAccessIssue]
                    usage=raw_usage,
                    cost_usd=call_cost,
                    parsed_output=parsed_output,
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
                # litellm's fallbacks cover API/timeout errors, but a Pydantic
                # re-validation failure happens AFTER litellm sees a successful
                # 200 — litellm can't detect it, so we owe one explicit second
                # attempt at the model. PR #121 documented this as a MiniMax-M3
                # mitigation. (A hard timeout is NOT retried here: same-model
                # retry of a hang is what produced the 490s in PYTHON-FASTAPI-62;
                # the fallback ladder inside _call_and_parse handles it instead.)
                #
                # This second pass re-walks the full ladder, so the worst-case
                # wall clock is ~2x the ladder bound. That ceiling is only reached
                # if a model returns a malformed-but-successful 200 AND runs near
                # the timeout on BOTH passes — a hang (the common case) raises
                # LLMHardTimeoutError, which is not caught here and exits after a
                # single ladder.
                self.logger.warning(
                    "event=llm_parse_retry model=%s — primary returned malformed structured output, retrying once",
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
