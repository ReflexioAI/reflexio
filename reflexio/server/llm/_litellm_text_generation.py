"""Text-generation concern for ``LiteLLMClient`` — the largest bucket (Tier-2.5).

``TextGenerationMixin`` holds the chat/response entry points
(``generate_response``, ``generate_chat_response``), completion-param building,
the subprocess hard-timeout path, observability, the ``_make_request``
orchestrator, prompt caching, multimodal image handling, and
``_compute_cost_usd`` (kept as a retained method here — verbatim billing
exception->None semantics; called only by two text-gen methods, so no separate
cost module).

LLM-mock: every litellm call is via the shared ``litellm`` module attr
(``litellm.completion``) so the global ``patch("litellm.completion")`` mock and
the fork-inherited subprocess worker still intercept — NEVER ``from litellm
import completion``.

SINK-1 (patch-where-used): ``resolve_model_name`` is imported HERE for the
model-role path in ``_build_completion_params``; it is also imported into
``_litellm_embedding`` for the embedding path — the two are independent bindings.

Per-mixin TYPE_CHECKING stubs (Tier-1b idiom) self-type the foreign members these
methods read: the client-core attributes/creds resolver (``config``, ``logger``,
``_api_key``/``_api_base``/``_api_version``, ``_resolve_api_key`` on the facade)
and the two cross-mixin edges resolved via MRO at runtime
(``_provider_response_format`` + ``_maybe_parse_structured_output`` on
``StructuredOutputMixin`` — which is why structured-output moves first).

Bodies moved VERBATIM from the former monolithic ``litellm_client.py``.
"""

import base64
import logging
import multiprocessing
import os
import queue
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import litellm
from pydantic import BaseModel

from reflexio.server.llm._litellm_subprocess import _litellm_completion_worker
from reflexio.server.llm._litellm_types import (
    CompletionResult,
    LiteLLMClientError,
    LLMHardTimeoutError,
    ModelProvenance,
    StructuredOutputParseError,
    StructuredOutputRepairError,
    ToolCallingChatResponse,
)
from reflexio.server.llm._provider_concurrency import (
    ProviderCapSaturatedError,
    provider_slot,
)
from reflexio.server.llm.image_utils import (
    SUPPORTED_IMAGE_MIME_TYPES,
    ImageEncodingError,
)
from reflexio.server.llm.image_utils import (
    encode_image_to_base64 as _encode_image_to_base64,
)
from reflexio.server.llm.llm_utils import is_pydantic_model
from reflexio.server.llm.model_defaults import (
    ModelRole,
    default_max_tokens_for_model,
    resolve_model_name,
)

if TYPE_CHECKING:
    from reflexio.server.llm._litellm_types import LiteLLMConfig


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

_ZAI_CODING_API_BASE = "https://api.z.ai/api/coding/paas/v4"


# Upstream-provider errors that are EXPECTED and transient. By the time one of
# these reaches the request handler the fallback ladder is already exhausted,
# but the caller — not the client — owns fatality, and many callers degrade
# gracefully (FTS fallback for document expansion, skip-dedup for profile
# consolidation). Log these at WARNING so a flaky provider (e.g. minimax
# timeouts / 529 overload) can't flood ERROR-level alerts for handled failures;
# genuinely-unexpected errors (bugs, auth, malformed structured output) stay
# ERROR. Classified by exception TYPE NAME to avoid importing the heavy
# ``litellm``/``openai`` exception hierarchies at module import.
_TRANSIENT_LLM_ERROR_NAMES: frozenset[str] = frozenset(
    {
        "Timeout",
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailableError",
    }
)

_MAX_REPAIR_ERRORS = 8
_MAX_REPAIR_ERROR_CHARS = 4000
_MAX_REPAIR_ECHO_CHARS = 4000

# Worst-case cumulative wall-clock budget for a full ladder walk. Each rung now
# owns a per-single-attempt hard timeout, so the walk's worst case is the SUM of
# the per-rung hard timeouts (a slow rung no longer shares one ladder-wide
# budget). If that projected sum exceeds the upstream request budget the walk
# logs a one-line warning so an over-long ladder is visible before it eats a
# request slot. Advisory only — never fatal.
_LADDER_WALL_CLOCK_BUDGET_SECONDS = 600.0

StructuredOutputValidator = Callable[[BaseModel], Sequence[str]]


def _nonempty_string(value: Any) -> str | None:
    """Return trustworthy string metadata without coercing mocks or objects."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _response_hidden_params(response: Any) -> dict[str, Any]:
    hidden = getattr(response, "_hidden_params", None)
    return hidden if isinstance(hidden, dict) else {}


@dataclass
class _StructuredAttempt:
    value: str | BaseModel | ToolCallingChatResponse
    raw_content: str | None
    parsed_output: BaseModel | None
    finish_reason: str | None
    model: str
    provenance: ModelProvenance | None = None


def _is_expected_transient_llm_error(exc: BaseException) -> bool:
    """True for expected transient upstream failures (timeout / connection /
    rate-limit / overload), including our own ``LLMHardTimeoutError`` (a
    ``TimeoutError`` subclass raised when a provider hang is killed)."""
    if isinstance(exc, TimeoutError):  # incl. LLMHardTimeoutError
        return True
    return type(exc).__name__ in _TRANSIENT_LLM_ERROR_NAMES


def _rung_reason(error: Exception | None) -> str:
    """Classify why the ladder advanced past a rung, for the fallback signal.

    Distinguishes a broken-but-reachable primary (``parse_exhausted``) from an
    outage (``transport_error``) or a saturated fail-closed provider cap
    (``cap_saturated``) so alerting can page differently on each.

    Args:
        error: The failure that caused the previous rung to be abandoned.

    Returns:
        str: ``"parse_exhausted"``, ``"cap_saturated"``, or ``"transport_error"``.
    """
    if isinstance(error, StructuredOutputParseError | StructuredOutputRepairError):
        return "parse_exhausted"
    if isinstance(error, ProviderCapSaturatedError):
        return "cap_saturated"
    return "transport_error"


class ProviderRequestGuardError(RuntimeError):
    """Raised when a caller rejects built params at the provider boundary."""


ProviderRequestGuard = Callable[
    [dict[str, Any], float, tuple[str, ...], bool],
    None,
]


class TextGenerationMixin:
    """Chat/response generation, completion-param build, hard-timeout, cost, multimodal.

    Mixed into ``LiteLLMClient`` (first in the MRO); the ``self`` members these
    methods read are owned by the client-core ``__init__`` on the facade and by
    ``StructuredOutputMixin``. The stubs below (Tier-1b idiom) give pyright the
    foreign-member types without introducing shared class-level mutable state —
    NEVER assign the annotation-only members here.
    """

    SUPPORTED_IMAGE_FORMATS: set[str] = set(SUPPORTED_IMAGE_MIME_TYPES.keys())

    # Models that only support temperature=1.0 (custom values cause errors or degraded performance)
    TEMPERATURE_RESTRICTED_MODELS = {
        "gpt-5",
        "gpt-5.4-mini",
        "gpt-5-nano",
        "gpt-5-codex",
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
    }

    # Base-owned attributes these methods read (init'd in the facade ``__init__``).
    config: "LiteLLMConfig"
    logger: logging.Logger
    _api_key: str | None
    _api_base: str | None
    _api_version: str | None

    if TYPE_CHECKING:
        # Client-core credential resolver (stays on the facade per the split).
        def _resolve_api_key(
            self, model: str | None = ..., for_embedding: bool = ...
        ) -> tuple[str | None, str | None, str | None]: ...

        # Concrete on ``StructuredOutputMixin`` (resolved via MRO); declared
        # type-only so pyright can resolve these ``self.`` calls.
        def _provider_response_format(
            self, *, response_format: Any, model: str, strict_response_format: bool
        ) -> Any: ...

        @classmethod
        def _structured_output_strategy(
            cls, *, model: str, strict_response_format: bool
        ) -> str: ...

        def _prompt_schema_directive(
            self, *, response_format: type[BaseModel], tools_available: bool
        ) -> str: ...

        def _maybe_parse_structured_output(
            self,
            content: Any,
            response_format: Any,
            parse_structured_output: bool,
        ) -> "str | BaseModel": ...

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
        return self.generate_response_with_provenance(
            prompt,
            system_message,
            images,
            image_media_type,
            **kwargs,
        ).value

    def generate_response_with_provenance(
        self,
        prompt: str,
        system_message: str | None = None,
        images: list[str | bytes | dict] | None = None,
        image_media_type: str | None = None,
        **kwargs: Any,
    ) -> CompletionResult[str | BaseModel | ToolCallingChatResponse]:
        """Generate a response paired with observed model provenance."""
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
        structured_output_validator: StructuredOutputValidator | None = None,
        provider_request_guard: ProviderRequestGuard | None = None,
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
            structured_output_validator: Optional semantic validator for parsed
                structured output. Passing one opts the call into the corrective
                repair ladder for parse, blank, and semantic failures.
            provider_request_guard: Optional internal guard invoked on the exact
                built params immediately before each provider attempt.
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
        return self.generate_chat_response_with_provenance(
            messages,
            system_message,
            tools=tools,
            tool_choice=tool_choice,
            model_role=model_role,
            max_retries=max_retries,
            fallback_models=fallback_models,
            structured_output_validator=structured_output_validator,
            provider_request_guard=provider_request_guard,
            **kwargs,
        ).value

    def generate_chat_response_with_provenance(
        self,
        messages: list[dict[str, Any]],
        system_message: str | None = None,
        *,
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        model_role: ModelRole | None = None,
        max_retries: int | None = None,
        fallback_models: list[str] | None = None,
        structured_output_validator: StructuredOutputValidator | None = None,
        provider_request_guard: ProviderRequestGuard | None = None,
        **kwargs: Any,
    ) -> CompletionResult[str | BaseModel | ToolCallingChatResponse]:
        """Generate a chat response paired with observed model provenance."""
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
                # Merge with existing system message. Replace the slot with a NEW
                # dict rather than mutating in place — ``list(messages)`` is a
                # shallow copy that shares the caller's dict objects, so an
                # in-place edit would corrupt the caller's list (and re-prepend on
                # reuse/retry).
                final_messages[0] = {
                    **final_messages[0],
                    "content": f"{system_message}\n\n{final_messages[0]['content']}",
                }
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
        if structured_output_validator is not None:
            kwargs["structured_output_validator"] = structured_output_validator
        if provider_request_guard is not None:
            kwargs["provider_request_guard"] = provider_request_guard

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

        # An explicit ``model=None`` means "use the config default" — callers
        # like the eval judges forward an optional model straight through, and
        # a literal None would crash on ``.lower()`` during key resolution.
        # ``_resolve_primary_model`` applies the same model_role /
        # custom_endpoint precedence the fallback ladder resolution uses, so the
        # two never drift.
        actual_model = self._resolve_primary_model(
            kwargs.pop("model", None), model_role
        )

        params: dict[str, Any] = {
            "model": actual_model,
            "messages": messages,
            "timeout": kwargs.pop(
                "timeout", self._effective_timeout_for_model(actual_model)
            ),
        }

        # Drop any fallback entry that points back at the primary — sending the
        # same broken endpoint twice never helps. Also drop in-process ``local/*``
        # embedding models: they have no litellm completion route (they are served
        # in-process by ``_litellm_embedding.py``), so litellm would raise
        # ``BadRequestError: LLM Provider NOT provided`` deep inside its fallback
        # ladder if one ever landed in this list (Sentry PYTHON-FASTAPI-CV).
        fallback_models = [
            m
            for m in fallback_models_raw
            if m != actual_model and not m.startswith("local/")
        ]

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
        if max_tokens is None:
            # Provider-level guard: some providers (MiniMax-M3) stall into the
            # request timeout when max_tokens is omitted. See model_defaults.
            max_tokens = default_max_tokens_for_model(actual_model)
        if max_tokens:
            params["max_tokens"] = max_tokens
        if self.config.top_p != 1.0:
            params["top_p"] = self.config.top_p
        allowed_openai_params = list(kwargs.pop("allowed_openai_params", None) or [])
        self._apply_structured_output_transport(
            params=params,
            messages=messages,
            response_format=response_format,
            model=actual_model,
            strict_response_format=strict_response_format,
            tools_available=tools is not None,
            allowed_openai_params=allowed_openai_params,
        )
        if allowed_openai_params:
            params["allowed_openai_params"] = allowed_openai_params
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
        elif actual_model.lower().startswith("zai/"):
            params["api_base"] = _ZAI_CODING_API_BASE
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
    def _inject_system_instruction(
        messages: list[dict[str, Any]], instruction: str
    ) -> list[dict[str, Any]]:
        """Return a copied message list with ``instruction`` in the system turn."""
        final_messages = [dict(message) for message in messages]
        if final_messages and final_messages[0].get("role") == "system":
            existing = final_messages[0].get("content")
            final_messages[0]["content"] = (
                f"{existing}\n\n{instruction}" if existing else instruction
            )
        else:
            final_messages.insert(0, {"role": "system", "content": instruction})
        return final_messages

    def _resolve_primary_model(
        self, model: str | None, model_role: ModelRole | None
    ) -> str:
        """Resolve the primary model name honoring model_role + custom_endpoint.

        Applies the same precedence used by ``_build_completion_params`` so the
        ladder walker and the per-rung param builder never disagree on which
        model is the primary: an explicit ``model`` (or the config default),
        overridden by ``model_role`` resolution, overridden in turn by a
        configured custom endpoint (the highest-priority hard pin).

        Args:
            model: Explicit per-call model, or ``None`` for the config default.
            model_role: Optional role whose resolution overrides ``model``.

        Returns:
            str: The resolved primary model name.
        """
        actual_model = model or self.config.model
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
        return actual_model

    def _resolve_ladder(self, **kwargs: Any) -> list[str]:
        """Return ``[primary, *fallbacks]`` deduped, self-refs and local/* dropped.

        The reflexio-owned walk consumes this once and then rebuilds transport
        params per rung; ``local/*`` in-process embedding models have no litellm
        completion route (Sentry PYTHON-FASTAPI-CV) and self-referential entries
        would just retry the same broken endpoint, so both are filtered here.

        A configured custom endpoint is a single-model hard pin (every rung's
        ``_resolve_primary_model`` call re-pins to ``ce.model`` regardless of
        what the rung was), so fallback is meaningless there: short-circuit to
        a single-rung ladder rather than dispatching the same call N times
        under N different "fallback" labels and logging a false
        ``served_model`` on success.

        Args:
            **kwargs: The original per-call kwargs (``model``, ``model_role``,
                and optionally ``fallback_models``).

        Returns:
            list[str]: The ordered, deduped rung list beginning with the primary.
        """
        primary = self._resolve_primary_model(
            kwargs.get("model"), kwargs.get("model_role")
        )
        ce = (
            self.config.api_key_config.custom_endpoint
            if self.config.api_key_config
            else None
        )
        if ce and ce.api_key and ce.api_base:
            return [primary]
        if "fallback_models" in kwargs:
            fallback_raw = kwargs.get("fallback_models") or []
        else:
            fallback_raw = list(self.config.fallback_models)
        ladder: list[str] = [primary]
        for m in fallback_raw:
            if m and m != primary and not m.startswith("local/") and m not in ladder:
                ladder.append(m)
        return ladder

    def _apply_structured_output_transport(
        self,
        *,
        params: dict[str, Any],
        messages: list[dict[str, Any]],
        response_format: Any,
        model: str,
        strict_response_format: bool,
        tools_available: bool,
        allowed_openai_params: list[str],
    ) -> None:
        """Apply the provider-facing format while retaining local Pydantic parsing."""
        if not response_format:
            return
        strategy = (
            self._structured_output_strategy(
                model=model,
                strict_response_format=strict_response_format,
            )
            if is_pydantic_model(response_format)
            else "pydantic_passthrough"
        )
        if strategy != "prompt_json_object":
            params["response_format"] = self._provider_response_format(
                response_format=response_format,
                model=model,
                strict_response_format=strict_response_format,
            )
            return

        directive = self._prompt_schema_directive(
            response_format=response_format,
            tools_available=tools_available,
        )
        params["messages"] = self._inject_system_instruction(messages, directive)
        if not tools_available:
            params["response_format"] = {"type": "json_object"}
            if "response_format" not in allowed_openai_params:
                allowed_openai_params.append("response_format")

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
        This call dispatches exactly ONE rung (``fallbacks`` is never passed to
        LiteLLM — the reflexio-owned walk advances between rungs itself), so the
        caller sizes ``hard_timeout`` to a single attempt on one rung, not the
        whole ladder. A hung primary is killed at its own bound and the walk
        then starts the next rung fresh — preserving the Sentry PYTHON-FASTAPI-62
        property (a hung primary must not block the fallback) per rung.
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
            # Drain the result queue BEFORE joining the child. A large completion
            # payload overflows the OS pipe buffer, so the child's queue-feeder
            # thread blocks on ``put`` until a reader drains it — and the child
            # cannot exit while that thread is blocked. Joining first would then
            # deadlock the parent against a child that finished but is wedged on a
            # full pipe, tripping a *false* hard timeout (a large-but-successful
            # result reported as a wall-clock kill). Reading here unblocks the
            # feeder so the child can exit. The read is bounded by the same
            # ``hard_timeout`` budget the join used to enforce.
            deadline = time.monotonic() + hard_timeout
            result: tuple[str, Any] | None = None
            while result is None:
                remaining = deadline - time.monotonic()
                try:
                    result = result_queue.get(timeout=max(0.0, min(0.1, remaining)))
                    break
                except queue.Empty as exc:
                    if remaining <= 0:
                        # True wall-clock timeout: the child ran past the hard
                        # bound without producing a result. Kill it (if still
                        # alive) and surface the timeout; a child that exited
                        # without a result is a distinct failure.
                        if process.is_alive():
                            process.terminate()
                            process.join(timeout=1.0)
                            if process.is_alive():
                                process.kill()
                                process.join(timeout=1.0)
                            raise LLMHardTimeoutError(
                                f"LLM request exceeded hard timeout of {hard_timeout:.3f}s "
                                f"(provider timeout={provider_timeout!r})"
                            ) from exc
                        raise LiteLLMClientError(
                            "LLM request process exited without returning a result "
                            f"(exitcode={process.exitcode})"
                        ) from exc
                    if not process.is_alive():
                        # The child exited before the deadline. Give the queue one
                        # last read in case the feeder flushed the payload just
                        # before exit; otherwise it died without a result.
                        try:
                            result = result_queue.get(timeout=1.0)
                        except queue.Empty as exc2:
                            raise LiteLLMClientError(
                                "LLM request process exited without returning a result "
                                f"(exitcode={process.exitcode})"
                            ) from exc2

            # The loop only exits with a drained result (every no-result path
            # above raises); the assert makes that invariant explicit for pyright.
            assert result is not None  # noqa: S101
            status, payload = result
            # The payload is drained, so the feeder is unblocked and the child can
            # exit. Reap it (terminating if it lingers) to avoid a zombie.
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

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

    def _build_model_provenance(self, response: Any) -> ModelProvenance:
        """Build attribution only from metadata observed on the response.

        Model name is taken only from fields that represent the served completion,
        not request-side LiteLLM routing metadata. ``_hidden_params["model"]`` is
        intentionally ignored: LiteLLM often echoes the requested model there, which
        would record a configured route as if it had been observed.

        The claude-code LiteLLM route can execute different local host CLIs. Its
        public ``ModelResponse.model`` is the *requested* route string (same as
        other providers). Observed model is only the bridge stamp
        ``reflexio_served_model`` — never ``response.model``, which would launder
        the requested route as observed when the CLI does not report a served model.
        """
        hidden = _response_hidden_params(response)
        route_provider = _nonempty_string(hidden.get("reflexio_provider"))
        served_provider = _nonempty_string(hidden.get("reflexio_served_provider"))
        stamped_served = _nonempty_string(hidden.get("reflexio_served_model"))

        if route_provider == "claude-code":
            provider = served_provider
            model_name = stamped_served
        else:
            # Prefer an explicit bridge stamp, then the provider response body field.
            # Do not fall back to request-side hidden model / model_id keys.
            model_name = stamped_served or _nonempty_string(
                getattr(response, "model", None)
            )
            provider = (
                served_provider
                or _nonempty_string(hidden.get("custom_llm_provider"))
                or _nonempty_string(hidden.get("llm_provider"))
                or _nonempty_string(hidden.get("provider"))
            )

        return ModelProvenance(
            model_name=model_name,
            provider=provider,
        )

    def _emit_fallback_signal(
        self, primary_model: str, served_model: str, *, reason: str
    ) -> None:
        """Record that a fallback rung served the request (authoritative, loop-driven).

        The reflexio-owned walk knows exactly which rung served the call, so this
        replaces the old response-model-diffing heuristic (which never fired once
        fallbacks left litellm). Preserves the pre-existing wire format —
        ``event=llm_fallback_used`` plus the ``llm.fallback_used`` /
        ``llm.primary_model`` / ``llm.fallback_model`` Sentry tags — so dashboards
        and alerts keep working; adds ``llm.fallback_reason`` so alerting can page
        on a broken-but-reachable primary (``parse_exhausted``) differently from an
        outage (``transport_error`` / ``cap_saturated``).

        Args:
            primary_model: The originally requested primary model (ladder head).
            served_model: The fallback rung that actually served the request.
            reason: Why the ladder advanced past the primary (see ``_rung_reason``).
        """
        self.logger.info(
            "event=llm_fallback_used primary_model=%s served_model=%s reason=%s",
            primary_model,
            served_model,
            reason,
        )
        try:
            # Local import keeps sentry out of module-init paths the tests
            # exercise without a Sentry SDK installed. sentry_sdk is an
            # enterprise-only dependency; OSS callers run without it and the
            # ImportError is intentionally absorbed by the except.
            import sentry_sdk  # type: ignore[import-not-found]

            sentry_sdk.set_tag("llm.fallback_used", "true")
            sentry_sdk.set_tag("llm.primary_model", str(primary_model))
            sentry_sdk.set_tag("llm.fallback_model", str(served_model))
            sentry_sdk.set_tag("llm.fallback_reason", reason)
        except Exception:  # noqa: BLE001 — observability must not break the call
            return

    def _make_request(  # noqa: C901
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> CompletionResult[str | BaseModel | ToolCallingChatResponse]:
        """
        Make a request to the LLM via a reflexio-owned per-rung fallback walk.

        Fallback is NOT delegated to ``litellm.completion`` — the ``fallbacks``
        kwarg is never passed. Instead the client walks ``[primary, *fallbacks]``
        one rung at a time (``_resolve_ladder``), rebuilding transport params per
        rung so each provider gets its own structured-output strategy, api_base,
        and a per-single-attempt hard timeout. ``num_retries`` is forced to 0 on
        every rung: same-model retry of a *hung* primary is what made the fallback
        unreachable and produced the 490s in Sentry PYTHON-FASTAPI-62. Each rung is
        entered at most once per logical request; within a rung the primary keeps
        exactly one same-model parse-retry (plain path) or one same-model
        corrective repair turn (validator path). A rung reached via the repair
        path receives the ORIGINAL prompt, never the prior rung's repair
        conversation.

        Args:
            messages: List of messages to send.
            **kwargs: Additional parameters (response_format, max_retries,
                fallback_models, tools, etc.).

        Returns:
            Response content as string, BaseModel instance, or
            ToolCallingChatResponse when the request was in tool-calling mode.

        Raises:
            LiteLLMClientError: If every rung of the ladder fails.
            StructuredOutputRepairError: If the final rung's validator/repair
                budget is exhausted (typed so callers can keep the latest parse).
        """
        structured_output_validator: StructuredOutputValidator | None = kwargs.pop(
            "structured_output_validator", None
        )
        provider_request_guard: ProviderRequestGuard | None = kwargs.pop(
            "provider_request_guard", None
        )
        original_kwargs = dict(kwargs)

        def _finish(
            attempt: _StructuredAttempt,
        ) -> CompletionResult[str | BaseModel | ToolCallingChatResponse]:
            assert attempt.provenance is not None  # noqa: S101
            return CompletionResult(attempt.value, attempt.provenance)

        if structured_output_validator is not None and (
            original_kwargs.get("response_format") is None
            or not original_kwargs.get("parse_structured_output", True)
        ):
            raise ValueError(
                "structured_output_validator requires response_format and "
                "parse_structured_output=True"
            )

        def _prepare_turn(
            turn_messages: list[dict[str, Any]], turn_kwargs: dict[str, Any]
        ) -> tuple[dict[str, Any], Any, bool, float]:
            """Build single-rung completion params (never any ``fallbacks``).

            ``num_retries`` is forced to 0 and the hard (wall-clock) timeout is
            sized to a SINGLE attempt on this rung plus one grace buffer — the
            walk, not litellm, owns advancing to the next rung.
            """
            (
                params,
                response_format,
                parse_structured_output,
                _max_retries,
                _fallbacks,
            ) = self._build_completion_params(turn_messages, **dict(turn_kwargs))
            params["num_retries"] = 0
            params.pop("fallbacks", None)  # owned walk: never delegate to litellm
            per_attempt = self._coerce_timeout_seconds(params)
            hard_timeout = per_attempt + self._hard_timeout_grace_seconds()
            return params, response_format, parse_structured_output, hard_timeout

        def _is_refusal(response: Any, message: Any) -> bool:
            refusal = getattr(message, "refusal", None)
            if isinstance(refusal, str) and refusal.strip():
                return True
            choice = response.choices[0]  # type: ignore[reportAttributeAccessIssue]
            stop_reason = (
                getattr(message, "stop_reason", None)
                or getattr(choice, "stop_reason", None)
                or getattr(response, "stop_reason", None)
            )
            return stop_reason == "refusal"

        def _call_and_parse(
            turn_params: dict[str, Any],
            turn_response_format: Any,
            turn_parse_structured_output: bool,
            turn_hard_timeout: float,
            *,
            detect_refusal: bool,
        ) -> _StructuredAttempt:
            request_start = time.perf_counter()
            self.logger.info(
                "event=llm_request_start model=%s timeout=%s has_response_format=%s num_retries=0 hard_timeout=%.3f",
                turn_params.get("model"),
                turn_params.get("timeout"),
                turn_response_format is not None,
                turn_hard_timeout,
            )
            try:
                with provider_slot(turn_params["model"]):
                    if provider_request_guard is not None:
                        provider_request_guard(
                            turn_params,
                            turn_hard_timeout,
                            tuple(ladder),
                            turn_parse_structured_output,
                        )
                    response = self._completion_with_hard_timeout(
                        turn_params, turn_hard_timeout
                    )
                provenance = self._build_model_provenance(response)
                message = response.choices[0].message  # type: ignore[reportAttributeAccessIssue]
                content = message.content
                finish_reason = response.choices[0].finish_reason  # type: ignore[reportAttributeAccessIssue]
                self._log_token_usage(turn_params, response)
                self.logger.info(
                    "event=llm_request_end model=%s timeout=%s has_response_format=%s elapsed_seconds=%.3f success=%s",
                    turn_params.get("model"),
                    turn_params.get("timeout"),
                    turn_response_format is not None,
                    time.perf_counter() - request_start,
                    True,
                )

                if detect_refusal and _is_refusal(response, message):
                    raise StructuredOutputRepairError(
                        "Structured output repair stopped on provider refusal",
                        failure_kind="refusal",
                        model=str(turn_params.get("model")),
                        raw_content=content if isinstance(content, str) else None,
                    )

                if "tools" in turn_params:
                    raw_usage = getattr(response, "usage", None)
                    call_cost = self._compute_cost_usd(
                        response, turn_params.get("model")
                    )
                    tool_calls = getattr(message, "tool_calls", None)
                    parsed_output: BaseModel | None = None
                    if turn_response_format is not None and not tool_calls:
                        try:
                            parsed = self._maybe_parse_structured_output(
                                content,  # type: ignore[reportArgumentType]
                                turn_response_format,
                                turn_parse_structured_output,
                            )
                        except StructuredOutputParseError as exc:
                            exc.finish_reason = finish_reason
                            if exc.raw_content is None and isinstance(content, str):
                                exc.raw_content = content
                            exc.provenance = provenance
                            raise
                        if isinstance(parsed, BaseModel):
                            parsed_output = parsed
                    value = ToolCallingChatResponse(
                        content=content,
                        tool_calls=tool_calls,
                        finish_reason=finish_reason,
                        usage=raw_usage,
                        cost_usd=call_cost,
                        parsed_output=parsed_output,
                    )
                    return _StructuredAttempt(
                        value=value,
                        raw_content=content if isinstance(content, str) else None,
                        parsed_output=parsed_output,
                        finish_reason=finish_reason,
                        model=str(turn_params.get("model")),
                        provenance=provenance,
                    )

                try:
                    value = self._maybe_parse_structured_output(
                        content,  # type: ignore[reportArgumentType]
                        turn_response_format,
                        turn_parse_structured_output,
                    )
                except StructuredOutputParseError as exc:
                    exc.finish_reason = finish_reason
                    if exc.raw_content is None and isinstance(content, str):
                        exc.raw_content = content
                    exc.provenance = provenance
                    raise
                return _StructuredAttempt(
                    value=value,
                    raw_content=content if isinstance(content, str) else None,
                    parsed_output=value if isinstance(value, BaseModel) else None,
                    finish_reason=finish_reason,
                    model=str(turn_params.get("model")),
                    provenance=provenance,
                )
            except ProviderRequestGuardError:
                raise
            except (
                StructuredOutputParseError,
                StructuredOutputRepairError,
                ProviderCapSaturatedError,
            ):
                # Advance-worthy rung failures owned by the walk: parse/repair
                # exhaustion and a saturated fail-closed provider cap must reach
                # the walker UNWRAPPED so it can classify the reason and advance.
                raise
            except Exception as e:
                log = (
                    self.logger.warning
                    if _is_expected_transient_llm_error(e)
                    else self.logger.error
                )
                log(
                    "event=llm_request_end model=%s elapsed_seconds=%.3f success=False error_type=%s error=%s",
                    turn_params.get("model"),
                    time.perf_counter() - request_start,
                    type(e).__name__,
                    e,
                )
                raise LiteLLMClientError(f"API call failed: {e}") from e

        def _bounded_errors(errors: Sequence[str]) -> tuple[str, ...]:
            bounded: list[str] = []
            remaining = _MAX_REPAIR_ERROR_CHARS
            for error in errors[:_MAX_REPAIR_ERRORS]:
                text = str(error)
                if len(text) > remaining:
                    text = text[: max(0, remaining)] + "...(truncated)"
                bounded.append(text)
                remaining -= len(text)
                if remaining <= 0:
                    break
            if len(errors) > len(bounded):
                bounded.append(f"...({len(errors) - len(bounded)} more errors)")
            return tuple(bounded)

        def _echo_content(
            raw_content: str | None,
            finish_reason: str | None,
            turn_params: dict[str, Any],
        ) -> str:
            if finish_reason == "length":
                token_limit = turn_params.get("max_tokens")
                if token_limit:
                    return f"(output truncated at {token_limit} tokens)"
                return "(output truncated at the model output limit)"
            if raw_content is None or not raw_content.strip():
                return "(empty response)"
            if len(raw_content) <= _MAX_REPAIR_ECHO_CHARS:
                return raw_content
            half = _MAX_REPAIR_ECHO_CHARS // 2
            omitted = len(raw_content) - (half * 2)
            return (
                raw_content[:half]
                + f"\n...(omitted {omitted} characters from previous response)...\n"
                + raw_content[-half:]
            )

        def _repair_messages(
            base_messages: list[dict[str, Any]],
            *,
            raw_content: str | None,
            finish_reason: str | None,
            turn_params: dict[str, Any],
            schema_name: str,
            errors: Sequence[str],
        ) -> list[dict[str, Any]]:
            error_lines = "\n".join(f"- {error}" for error in _bounded_errors(errors))
            return [
                *base_messages,
                {
                    "role": "assistant",
                    "content": _echo_content(raw_content, finish_reason, turn_params),
                },
                {
                    "role": "user",
                    "content": (
                        "Your previous response failed validation for schema "
                        f"{schema_name}:\n{error_lines}\n\n"
                        "Reply with a corrected response that satisfies the schema. "
                        "Respond only with the corrected structured output."
                    ),
                },
            ]

        def _validate_attempt(
            attempt: _StructuredAttempt,
        ) -> tuple[bool, tuple[str, ...], str]:
            if (
                isinstance(attempt.value, ToolCallingChatResponse)
                and attempt.value.tool_calls
            ):
                return True, (), "semantic"
            parsed = attempt.parsed_output
            if parsed is None and isinstance(attempt.value, BaseModel):
                parsed = attempt.value
            if parsed is None:
                return False, ("Response was empty or did not parse.",), "parse"
            if structured_output_validator is None:
                raise RuntimeError("structured_output_validator is not configured")
            errors = tuple(structured_output_validator(parsed))
            if errors:
                return False, errors, "semantic"
            return True, (), "semantic"

        def _repair_error(
            *,
            failure_kind: str,
            attempt: _StructuredAttempt | None,
            errors: Sequence[str],
            model: str,
            first_parsed_provenance: ModelProvenance | None = None,
        ) -> StructuredOutputRepairError:
            return StructuredOutputRepairError(
                "Structured output repair exhausted",
                failure_kind="parse" if failure_kind == "parse" else "semantic",
                model=model,
                raw_content=attempt.raw_content if attempt else None,
                parsed_output=attempt.parsed_output if attempt else None,
                validation_errors=tuple(errors),
                first_parsed_provenance=first_parsed_provenance,
            )

        def _run_rung_plain(
            rung_messages: list[dict[str, Any]], rung_kwargs: dict[str, Any]
        ) -> CompletionResult[str | BaseModel | ToolCallingChatResponse]:
            """Serve one rung with no validator: initial call + one same-model parse-retry.

            Raises ``StructuredOutputParseError`` when both attempts return a
            malformed body (the walk catches it and advances), or
            ``LiteLLMClientError`` / ``ProviderCapSaturatedError`` on transport
            failure (likewise advance-worthy).
            """
            params, rf, parse_so, hard_timeout = _prepare_turn(
                rung_messages, rung_kwargs
            )
            try:
                return _finish(
                    _call_and_parse(
                        params, rf, parse_so, hard_timeout, detect_refusal=False
                    )
                )
            except StructuredOutputParseError:
                self.logger.warning(
                    "event=llm_parse_retry model=%s — malformed structured output, "
                    "retrying once on the same model",
                    params.get("model"),
                )
                return _finish(
                    _call_and_parse(
                        params, rf, parse_so, hard_timeout, detect_refusal=False
                    )
                )

        def _run_rung_validated(
            rung_messages: list[dict[str, Any]], rung_kwargs: dict[str, Any]
        ) -> CompletionResult[str | BaseModel | ToolCallingChatResponse]:
            """Serve one rung with the validator: initial call + one same-model repair turn.

            The corrective turn is built from ``rung_messages`` (this rung's
            ORIGINAL prompt, never a prior rung's repair conversation). Raises
            ``StructuredOutputRepairError`` when this rung's own repair budget is
            exhausted (the walk catches it and advances).
            """
            params, rf, parse_so, hard_timeout = _prepare_turn(
                rung_messages, rung_kwargs
            )
            schema_name = getattr(rf, "__name__", "structured output")
            latest_parsed_output: BaseModel | None
            first_parsed_provenance: ModelProvenance | None = None
            try:
                first_attempt = _call_and_parse(
                    params, rf, parse_so, hard_timeout, detect_refusal=True
                )
            except StructuredOutputParseError as exc:
                failure_kind = "parse"
                errors: tuple[str, ...] = exc.validation_errors or (str(exc),)
                if exc.validation_errors:
                    self.logger.warning(
                        "event=llm_structured_schema_validation_failed model=%s schema=%s errors=%s",
                        params.get("model"),
                        schema_name,
                        ",".join(exc.validation_errors),
                    )
                raw_content = exc.raw_content
                finish_reason = exc.finish_reason
                latest_parsed_output = None
            else:
                valid, errors, failure_kind = _validate_attempt(first_attempt)
                if valid:
                    return _finish(first_attempt)
                raw_content = first_attempt.raw_content
                finish_reason = first_attempt.finish_reason
                latest_parsed_output = first_attempt.parsed_output
                if first_attempt.parsed_output is not None:
                    first_parsed_provenance = first_attempt.provenance

            repair_base = _repair_messages(
                rung_messages,
                raw_content=raw_content,
                finish_reason=finish_reason,
                turn_params=params,
                schema_name=schema_name,
                errors=errors,
            )
            repair_params, repair_rf, repair_parse, repair_timeout = _prepare_turn(
                repair_base, rung_kwargs
            )
            self.logger.warning(
                "event=llm_structured_repair_attempted model=%s repair_target_model=%s schema=%s failure_kind=%s",
                params.get("model"),
                repair_params.get("model"),
                schema_name,
                failure_kind,
            )
            try:
                repair_attempt = _call_and_parse(
                    repair_params,
                    repair_rf,
                    repair_parse,
                    repair_timeout,
                    detect_refusal=True,
                )
            except StructuredOutputParseError as exc:
                repair_attempt = _StructuredAttempt(
                    value="",
                    raw_content=exc.raw_content,
                    parsed_output=None,
                    finish_reason=exc.finish_reason,
                    model=str(repair_params.get("model")),
                    provenance=getattr(exc, "provenance", None),
                )
                errors = exc.validation_errors or (str(exc),)
                if exc.validation_errors:
                    self.logger.warning(
                        "event=llm_structured_schema_validation_failed model=%s schema=%s errors=%s",
                        repair_params.get("model"),
                        schema_name,
                        ",".join(exc.validation_errors),
                    )
                failure_kind = "parse"
            except (LiteLLMClientError, ProviderCapSaturatedError) as exc:
                # Cap saturation is not a LiteLLMClientError subclass; still stamp
                # first-parsed so the ladder outer handler can salvage attribution.
                exc.first_parsed_provenance = first_parsed_provenance
                raise
            else:
                valid, errors, failure_kind = _validate_attempt(repair_attempt)
                if valid:
                    self.logger.info(
                        "event=llm_structured_repair_succeeded model=%s repair_target_model=%s schema=%s",
                        params.get("model"),
                        repair_params.get("model"),
                        schema_name,
                    )
                    return _finish(repair_attempt)
                if (
                    repair_attempt.parsed_output is not None
                    and first_parsed_provenance is None
                ):
                    first_parsed_provenance = repair_attempt.provenance

            # Within-rung roll-forward: keep the most recent output that parsed
            # (e.g. a semantic-fail before a final parse-fail) on the typed error.
            repair_attempt.parsed_output = (
                repair_attempt.parsed_output or latest_parsed_output
            )
            self.logger.warning(
                "event=llm_structured_repair_exhausted model=%s schema=%s failure_kind=%s",
                repair_attempt.model,
                schema_name,
                failure_kind,
            )
            raise _repair_error(
                failure_kind=failure_kind,
                attempt=repair_attempt,
                errors=errors,
                model=repair_attempt.model,
                first_parsed_provenance=first_parsed_provenance,
            )

        # Reflexio-owned per-rung walk. Each rung is entered at most once; the
        # walk (never litellm) owns cross-rung advancement.
        ladder = self._resolve_ladder(**original_kwargs)
        grace = self._hard_timeout_grace_seconds()
        projected = sum(
            self._effective_timeout_for_model(rung) + grace for rung in ladder
        )
        if projected > _LADDER_WALL_CLOCK_BUDGET_SECONDS:
            self.logger.warning(
                "event=llm_ladder_budget_exceeded projected_seconds=%.1f budget_seconds=%.1f "
                "ladder=%s — cumulative per-rung hard timeouts may exceed the upstream "
                "request budget",
                projected,
                _LADDER_WALL_CLOCK_BUDGET_SECONDS,
                ladder,
            )

        last_error: Exception | None = None
        # First accepted parse across the whole walk — not per-rung. Consolidator
        # salvage keeps the first parsed *content* via a shared validator closure;
        # this field must match that content's served model, not the last rung's.
        ladder_first_parsed_provenance: ModelProvenance | None = None
        for index, rung in enumerate(ladder):
            rung_kwargs = {**original_kwargs, "model": rung, "fallback_models": []}
            # ``model_role`` is already resolved into ``ladder``; leaving it in
            # per-rung kwargs would make every rung re-resolve to the same role
            # model and defeat the walk.
            rung_kwargs.pop("model_role", None)
            is_last = index == len(ladder) - 1
            try:
                if structured_output_validator is None:
                    value = _run_rung_plain(messages, rung_kwargs)
                else:
                    value = _run_rung_validated(messages, rung_kwargs)
            except (
                LiteLLMClientError,
                ProviderCapSaturatedError,
                StructuredOutputParseError,
            ) as exc:
                first_parsed = (
                    exc.provenance
                    if isinstance(exc, StructuredOutputParseError)
                    else exc.first_parsed_provenance
                )
                if first_parsed is not None and ladder_first_parsed_provenance is None:
                    ladder_first_parsed_provenance = first_parsed
                last_error = exc
                if not is_last:
                    continue
                # Final rung failed. Preserve the typed repair error (callers keep
                # the latest parse) and already-wrapped client errors as-is; wrap a
                # raw plain-path parse exhaustion (litellm saw a 200, so no turn
                # logged a request-end failure) and a cap-saturation. Always stamp
                # ladder-wide first-parsed so consolidator salvage keeps matching
                # model attribution.
                if isinstance(exc, StructuredOutputRepairError | LiteLLMClientError):
                    if ladder_first_parsed_provenance is not None:
                        exc.first_parsed_provenance = ladder_first_parsed_provenance
                    raise
                if isinstance(exc, StructuredOutputParseError):
                    self.logger.error(
                        "event=llm_request_end model=%s success=False error_type=%s error=%s",
                        rung,
                        type(exc).__name__,
                        exc,
                    )
                wrapped = LiteLLMClientError(
                    f"API call failed: {exc}",
                    first_parsed_provenance=ladder_first_parsed_provenance,
                )
                raise wrapped from exc
            else:
                if index > 0:
                    self._emit_fallback_signal(
                        ladder[0], rung, reason=_rung_reason(last_error)
                    )
                return value

        # A non-empty ladder always returns or raises above; guard the empty case.
        raise LiteLLMClientError(  # pragma: no cover
            f"All fallback rungs failed; last: {last_error}",
            first_parsed_provenance=ladder_first_parsed_provenance,
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
