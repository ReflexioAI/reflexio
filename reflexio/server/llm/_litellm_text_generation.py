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
    LiteLLMClientError,
    LLMHardTimeoutError,
    StructuredOutputParseError,
    StructuredOutputRepairError,
    ToolCallingChatResponse,
)
from reflexio.server.llm._provider_concurrency import provider_slot
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

StructuredOutputValidator = Callable[[BaseModel], Sequence[str]]


@dataclass
class _StructuredAttempt:
    value: str | BaseModel | ToolCallingChatResponse
    raw_content: str | None
    parsed_output: BaseModel | None
    finish_reason: str | None
    model: str


def _is_expected_transient_llm_error(exc: BaseException) -> bool:
    """True for expected transient upstream failures (timeout / connection /
    rate-limit / overload), including our own ``LLMHardTimeoutError`` (a
    ``TimeoutError`` subclass raised when a provider hang is killed)."""
    if isinstance(exc, TimeoutError):  # incl. LLMHardTimeoutError
        return True
    return type(exc).__name__ in _TRANSIENT_LLM_ERROR_NAMES


def _same_observed_model(primary_model: Any, served_model: Any) -> bool:
    """Return True when the served model is the primary model modulo provider-prefix drift.

    Some providers echo back the bare model name (e.g. ``MiniMax-M3``) even when
    LiteLLM was called through the provider-prefixed name (``minimax/MiniMax-M3``).
    Treat those as the same model so fallback observability does not misreport the
    primary as a fallback. The match is anchored on a ``/`` boundary in either
    direction, so ``gpt-4`` never matches ``gpt-40``.

    Args:
        primary_model: The originally requested model name (may be ``None``).
        served_model: The model LiteLLM reports as having served the call (may be ``None``).

    Returns:
        bool: True if the names are equal or differ only by a provider prefix.
    """
    if not primary_model or not served_model:
        return False
    primary = str(primary_model)
    served = str(served_model)
    return (
        primary == served
        or primary.endswith(f"/{served}")
        or served.endswith(f"/{primary}")
    )


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

        def _maybe_parse_structured_output(
            self,
            content: str,
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
        structured_output_validator: StructuredOutputValidator | None = None,
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
        actual_model = kwargs.pop("model", None) or self.config.model

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

            if not served_model or _same_observed_model(primary_model, served_model):
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

    def _make_request(  # noqa: C901
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
        structured_output_validator: StructuredOutputValidator | None = kwargs.pop(
            "structured_output_validator", None
        )
        original_kwargs = dict(kwargs)

        def _prepare_turn(
            turn_messages: list[dict[str, Any]], turn_kwargs: dict[str, Any]
        ) -> tuple[dict[str, Any], Any, bool, list[str], float]:
            (
                params,
                response_format,
                parse_structured_output,
                _max_retries,
                fallbacks,
            ) = self._build_completion_params(turn_messages, **dict(turn_kwargs))

            # Hand the fallback ladder to litellm, but DISABLE same-model retries.
            # litellm walks [primary, *fallbacks] inside one litellm.completion call,
            # copying ``timeout`` unchanged into each rung. With num_retries>=1 it
            # retries a *hung* primary num_retries+1 times before ever reaching a
            # fallback (root cause of Sentry PYTHON-FASTAPI-62).
            params["num_retries"] = 0
            if fallbacks:
                params["fallbacks"] = fallbacks

            # Size the hard (wall-clock) timeout to cover the WHOLE ladder. litellm
            # copies one ``timeout`` into every rung, so the subprocess must cover
            # primary + fallbacks plus one grace buffer before being killed.
            per_attempt_timeout = self._coerce_timeout_seconds(params)
            hard_timeout = (
                1 + len(fallbacks)
            ) * per_attempt_timeout + self._hard_timeout_grace_seconds()
            return (
                params,
                response_format,
                parse_structured_output,
                fallbacks,
                hard_timeout,
            )

        params, response_format, parse_structured_output, fallbacks, hard_timeout = (
            _prepare_turn(messages, original_kwargs)
        )
        if structured_output_validator is not None and (
            response_format is None or not parse_structured_output
        ):
            raise ValueError(
                "structured_output_validator requires response_format and "
                "parse_structured_output=True"
            )

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
            turn_fallbacks: list[str],
            turn_hard_timeout: float,
            *,
            detect_refusal: bool,
        ) -> _StructuredAttempt:
            request_start = time.perf_counter()
            self.logger.info(
                "event=llm_request_start model=%s timeout=%s has_response_format=%s num_retries=0 fallbacks=%s hard_timeout=%.3f",
                turn_params.get("model"),
                turn_params.get("timeout"),
                turn_response_format is not None,
                turn_fallbacks,
                turn_hard_timeout,
            )
            try:
                with provider_slot(turn_params["model"]):
                    response = self._completion_with_hard_timeout(
                        turn_params, turn_hard_timeout
                    )
                self._emit_fallback_observability(response, turn_params)
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
                    raise
                return _StructuredAttempt(
                    value=value,
                    raw_content=content if isinstance(content, str) else None,
                    parsed_output=value if isinstance(value, BaseModel) else None,
                    finish_reason=finish_reason,
                    model=str(turn_params.get("model")),
                )
            except (StructuredOutputParseError, StructuredOutputRepairError):
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
        ) -> StructuredOutputRepairError:
            return StructuredOutputRepairError(
                "Structured output repair exhausted",
                failure_kind="parse" if failure_kind == "parse" else "semantic",
                model=model,
                raw_content=attempt.raw_content if attempt else None,
                parsed_output=attempt.parsed_output if attempt else None,
                validation_errors=tuple(errors),
            )

        def _repair_target_kwargs() -> list[dict[str, Any]]:
            # One same-model corrective turn, then (when a fallback ladder is
            # configured) one final escalation turn to the first eligible
            # fallback model with the rest of the ladder behind it.
            targets = [dict(original_kwargs)]
            if fallbacks:
                escalation_kwargs = dict(original_kwargs)
                escalation_kwargs["model"] = fallbacks[0]
                escalation_kwargs["fallback_models"] = fallbacks[1:]
                targets.append(escalation_kwargs)
            return targets

        def _attempt_repair(
            first_error: StructuredOutputParseError | None,
            first_attempt: _StructuredAttempt | None,
        ) -> str | BaseModel | ToolCallingChatResponse:
            schema_name = getattr(response_format, "__name__", "structured output")
            if first_error is not None:
                failure_kind = "parse"
                errors: tuple[str, ...] = (str(first_error),)
                raw_content = first_error.raw_content
                finish_reason = first_error.finish_reason
                latest_parsed_output = None
            else:
                assert first_attempt is not None  # noqa: S101
                valid, errors, failure_kind = _validate_attempt(first_attempt)
                if valid:
                    return first_attempt.value
                raw_content = first_attempt.raw_content
                finish_reason = first_attempt.finish_reason
                latest_parsed_output = first_attempt.parsed_output

            # Each turn echoes the PREVIOUS attempt's output and errors, so
            # raw_content / finish_reason / errors roll forward per iteration.
            prior_params = params
            repair_attempt: _StructuredAttempt | None = None
            for target_kwargs in _repair_target_kwargs():
                repair_base = _repair_messages(
                    messages,
                    raw_content=raw_content,
                    finish_reason=finish_reason,
                    turn_params=prior_params,
                    schema_name=schema_name,
                    errors=errors,
                )
                (
                    repair_params,
                    repair_response_format,
                    repair_parse,
                    repair_fallbacks,
                    repair_timeout,
                ) = _prepare_turn(repair_base, target_kwargs)
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
                        repair_response_format,
                        repair_parse,
                        repair_fallbacks,
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
                    )
                    errors = (str(exc),)
                    failure_kind = "parse"
                else:
                    valid, errors, failure_kind = _validate_attempt(repair_attempt)
                    if valid:
                        self.logger.info(
                            "event=llm_structured_repair_succeeded model=%s repair_target_model=%s schema=%s",
                            params.get("model"),
                            repair_params.get("model"),
                            schema_name,
                        )
                        return repair_attempt.value
                raw_content = repair_attempt.raw_content
                finish_reason = repair_attempt.finish_reason
                latest_parsed_output = (
                    repair_attempt.parsed_output or latest_parsed_output
                )
                prior_params = repair_params

            assert repair_attempt is not None  # noqa: S101 — loop runs at least once
            self.logger.warning(
                "event=llm_structured_repair_exhausted model=%s schema=%s failure_kind=%s",
                repair_attempt.model,
                schema_name,
                failure_kind,
            )
            repair_attempt.parsed_output = (
                repair_attempt.parsed_output or latest_parsed_output
            )
            raise _repair_error(
                failure_kind=failure_kind,
                attempt=repair_attempt,
                errors=errors,
                model=repair_attempt.model,
            )

        if structured_output_validator is None:
            request_start = time.perf_counter()
            try:
                try:
                    return _call_and_parse(
                        params,
                        response_format,
                        parse_structured_output,
                        fallbacks,
                        hard_timeout,
                        detect_refusal=False,
                    ).value
                except StructuredOutputParseError:
                    self.logger.warning(
                        "event=llm_parse_retry model=%s — primary returned malformed structured output, retrying once",
                        params.get("model"),
                    )
                    return _call_and_parse(
                        params,
                        response_format,
                        parse_structured_output,
                        fallbacks,
                        hard_timeout,
                        detect_refusal=False,
                    ).value
            except LiteLLMClientError:
                raise
            except StructuredOutputParseError as exc:
                # A parse failure happens after litellm saw a 200, so no turn
                # logged a request-end failure — emit it here or log-based
                # failure metrics miss this class entirely.
                self.logger.error(
                    "event=llm_request_end model=%s elapsed_seconds=%.3f success=False error_type=%s error=%s",
                    params.get("model"),
                    time.perf_counter() - request_start,
                    type(exc).__name__,
                    exc,
                )
                raise LiteLLMClientError(f"API call failed: {exc}") from exc

        try:
            first_attempt = _call_and_parse(
                params,
                response_format,
                parse_structured_output,
                fallbacks,
                hard_timeout,
                detect_refusal=True,
            )
        except StructuredOutputParseError as exc:
            return _attempt_repair(exc, None)
        return _attempt_repair(None, first_attempt)

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
