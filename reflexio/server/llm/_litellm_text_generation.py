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
import hashlib
import logging
import multiprocessing
import os
import queue
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import litellm
from pydantic import BaseModel

from reflexio.server.error_reporting import set_error_tags
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
# a production timeout incident. It is now floored at the 120s default so a hang is
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

# A response of the wrong SHAPE spends one error slot per required field before
# it names a single unexpected key. Live evidence: an open-world discovery call
# echoed its own input payload back, and the nine required ``DiscoveryMemo``
# fields reported ``missing`` before one ``manifest_role: extra_forbidden``. At
# eight the corrective turn dropped exactly the line that says "you returned a
# different object" -- the one error that explains the other nine. The character
# budget below, not this count, is what actually bounds the volume.
_MAX_REPAIR_ERRORS = 12
_MAX_REPAIR_ERROR_CHARS = 4000
_MAX_REPAIR_ECHO_CHARS = 4000

# The corrective turn a validator-less ("plain") rung appends before its one
# same-model retry.
#
# Why it carries no model output
# ------------------------------
# The validated rung's repair turn echoes the malformed response back
# (``_echo_content``). The plain rung deliberately does NOT: its callers include
# the open-world analyst, whose prompt frames evidence as untrusted input, and
# re-injecting the model's own unframed output into a qualified prompt is a
# content-safety and attestation hazard both. Field paths and pydantic error
# tags are enough to steer a re-emit, and they are content-free by construction
# (``_safe_validation_errors``).
#
# The assistant placeholder exists only to keep the turn alternating for
# providers that require it; it is a fixed string, never the response.
STRUCTURED_OUTPUT_CORRECTION_PLACEHOLDER = "(previous response withheld)"
STRUCTURED_OUTPUT_CORRECTION_PREFIX = "Your previous response did not satisfy schema "
_MAX_CORRECTION_CHARS = 8000

# Worst-case cumulative wall-clock budget for a full ladder walk. Each rung now
# owns a per-single-attempt hard timeout, so the walk's worst case is the SUM of
# the per-rung hard timeouts (a slow rung no longer shares one ladder-wide
# budget). If that projected sum exceeds the upstream request budget the walk
# logs a one-line warning so an over-long ladder is visible before it eats a
# request slot. Advisory only — never fatal.
_LADDER_WALL_CLOCK_BUDGET_SECONDS = 600.0

StructuredOutputValidator = Callable[[BaseModel], Sequence[str]]


def _bounded_repair_errors(errors: Sequence[str]) -> tuple[str, ...]:
    """Clamp a validation-error list to the repair-prompt budget.

    Args:
        errors (Sequence[str]): Content-free error summaries, longest-lived
            shape being ``"field.path: error_type"``

    Returns:
        tuple[str, ...]: The bounded list, with a trailing count line when
            anything was dropped
    """
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


def _detach_request_state(value: Any) -> Any:
    """Return ``value`` with every mutable container copied, recursively.

    litellm's provider transports mutate the request state they are handed IN
    PLACE and hand the same object onward. The reproduced instance is
    ``OpenAIGPTConfig.validate_environment``, which does::

        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"

    on whatever ``extra_headers`` dict it received. Every route dispatched
    through litellm's own HTTP handler inherits that method -- ``minimax/*``
    among them. (The plain ``openai`` SDK routes do not, which is why the
    exposure looks provider-specific from any single call site. It is not.)

    That matters because ``_make_request`` treats the caller's kwargs as the
    STANDING description of the request: it rebuilds params for every rung of
    the ladder, and ``_corrective_kwargs`` re-reads ``extra_headers`` to derive
    the corrective retry's idempotency key. Handing the transport a reference
    to those objects means attempt 1's provider writes the provider's own
    headers into the caller's dict, and attempt 2 then carries them as though
    the caller had asked for them. A caller-supplied
    ``provider_request_guard`` that pins the exact request -- correctly --
    refuses that second crossing, which silently kills the one corrective
    retry the client exists to make.

    Copying is done at the point the params are BUILT rather than at the one
    call site that was observed to break, because the exposure belongs to any
    mutable value forwarded into ``params``, not to ``extra_headers``
    specifically. Only containers are copied: classes (``response_format``),
    callables, and pydantic models pass through by identity, both because the
    transport does not mutate them and because ``response_format`` must stay
    the caller's exact type for ``type(result.value) is response_format``
    checks downstream.

    Args:
        value (Any): Any request value about to be handed to the transport

    Returns:
        Any: An equal value sharing no mutable container with ``value``
    """
    if isinstance(value, dict):
        return {key: _detach_request_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_request_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_request_state(item) for item in value)
    return value


def structured_output_correction_turn(
    *, schema_name: str, errors: Sequence[str]
) -> list[dict[str, str]]:
    """Build the corrective turn a plain rung appends before its one retry.

    The turn carries the failing field paths and pydantic error tags and
    NOTHING else -- no fragment of the model's own response. See
    ``STRUCTURED_OUTPUT_CORRECTION_PLACEHOLDER`` for why.

    Args:
        schema_name (str): Name of the response schema that was violated
        errors (Sequence[str]): Content-free validation-error summaries

    Returns:
        list[dict[str, str]]: An assistant placeholder followed by the
            correction instruction, ready to append to the rung's messages
    """
    error_lines = "\n".join(f"- {error}" for error in _bounded_repair_errors(errors))
    correction = (
        f"{STRUCTURED_OUTPUT_CORRECTION_PREFIX}{schema_name}. Each line below is a "
        f"field path in that schema and the validation error it produced:\n"
        f"{error_lines}\n\n"
        "Reply again with only a JSON object that satisfies the schema. A "
        "'missing' line means that field was absent; an 'extra_forbidden' line "
        "means you returned a key the schema does not define, which usually "
        "means the object you returned was not the requested one."
    )
    return [
        {"role": "assistant", "content": STRUCTURED_OUTPUT_CORRECTION_PLACEHOLDER},
        {"role": "user", "content": correction[:_MAX_CORRECTION_CHARS]},
    ]


def is_structured_output_correction_turn(
    base_messages: Sequence[Any], messages: Sequence[Any]
) -> bool:
    """Return whether ``messages`` is ``base_messages`` plus one corrective turn.

    Exists so a caller-supplied ``provider_request_guard`` can tell this
    client's own bounded correction from a genuinely different second question
    without re-implementing (and drifting from) the construction above.

    Args:
        base_messages (Sequence[Any]): The messages sent on the first attempt
        messages (Sequence[Any]): The messages about to be sent

    Returns:
        bool: True only for an exact prefix match followed by the fixed
            assistant placeholder and a bounded correction instruction
    """
    if len(messages) != len(base_messages) + 2:
        return False
    if list(messages[: len(base_messages)]) != list(base_messages):
        return False
    placeholder, correction = messages[-2], messages[-1]
    if placeholder != {
        "role": "assistant",
        "content": STRUCTURED_OUTPUT_CORRECTION_PLACEHOLDER,
    }:
        return False
    if not isinstance(correction, dict) or set(correction) != {"role", "content"}:
        return False
    content = correction["content"]
    return (
        correction["role"] == "user"
        and isinstance(content, str)
        and content.startswith(STRUCTURED_OUTPUT_CORRECTION_PREFIX)
        and len(content) <= _MAX_CORRECTION_CHARS
    )


def structured_output_repair_idempotency_key(base_key: str) -> str:
    """Derive a distinct provider idempotency key for a corrective retry.

    A corrective retry is a DIFFERENT request, so it must not reuse the key of
    the attempt it is correcting: a provider honouring ``Idempotency-Key`` is
    entitled to replay the first answer for a repeated key, which would turn
    the correction into a silent no-op -- the whole defect this path exists to
    fix. This repository has been bitten by exactly that shape before, on a
    payment retry that reused its key and replayed a cached decline.

    Args:
        base_key (str): The idempotency key the first attempt sent

    Returns:
        str: A 64-character hex key, matching the width provider clients
            require, derived deterministically so a replayed request stays
            idempotent with itself
    """
    return hashlib.sha256(
        b"reflexio/llm/structured-output-correction-key/v1\x00" + base_key.encode()
    ).hexdigest()


# Bounds for the log rendering of pydantic validation errors. ``loc`` is built
# from ``str(part)`` in ``_safe_validation_errors`` with no clamp of its own, so
# an ``extra_forbidden`` entry carries a MODEL-CHOSEN key straight into the
# string. Whatever a log line renders bridges to external telemetry, so both the
# COUNT and the SHAPE of each entry are bounded here.
MAX_LOGGED_VALIDATION_ERRORS = 5
MAX_LOGGED_ERROR_LENGTH = 128

# ``_safe_validation_errors`` emits ``"<dotted.loc>: <error_type>"`` and, for a
# JSON syntax error, ``"<root>: json_invalid at line L column C"``. This admits
# exactly that alphabet -- identifiers, list indices, the ``<root>`` marker, the
# separator and spaces. Anything else (a newline, a quote, a control character,
# a non-ASCII key) fails the match and the whole entry is redacted. Commas are
# deliberately excluded so the joined line stays unambiguous.
_LOG_SAFE_VALIDATION_ERROR = re.compile(
    rf"[A-Za-z0-9_.:<>\[\] ]{{1,{MAX_LOGGED_ERROR_LENGTH}}}"
)
_REDACTED_VALIDATION_ERROR = "<redacted>"


def _log_safe_validation_errors(errors: Sequence[str]) -> str:
    """Render validation errors for a log sink, bounded in count and shape.

    The enterprise open-world diagnostics recorder clamps exactly this shape
    (``_safe_loc_component`` / ``MAX_LOC_COMPONENTS``) before persisting it;
    this is the same bound applied at the OSS logging boundary, which the
    telemetry bridge reads. The RETRY PROMPT still receives the unclamped
    errors: they go back to the provider that produced them, which is a
    different trust boundary from a log sink, and clamping there would blunt
    the correction the plain rung's corrective retry depends on.

    Args:
        errors (Sequence[str]): Entries from ``_safe_validation_errors``

    Returns:
        str: A comma-joined, bounded rendering; entries outside the schema
            alphabet are replaced by a fixed marker, and a trailing count
            stands in for anything past the cap
    """
    entries = list(errors)
    rendered = [
        entry
        if _LOG_SAFE_VALIDATION_ERROR.fullmatch(entry)
        else _REDACTED_VALIDATION_ERROR
        for entry in entries[:MAX_LOGGED_VALIDATION_ERRORS]
    ]
    if len(entries) > len(rendered):
        rendered.append(f"...({len(entries) - len(rendered)} more)")
    return ",".join(rendered)


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
    ``TimeoutError`` subclass raised when a provider hang is killed).

    Subprocess-isolated calls cross a pickle boundary, so their concrete
    provider exception cannot safely be re-raised in the parent. The wrapper
    retains ``upstream_error_type`` explicitly; checking it here prevents an
    expected provider outage from being promoted to an application ERROR just
    because isolation made the outer type ``LiteLLMClientError``.
    """
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TimeoutError):  # incl. LLMHardTimeoutError
            return True
        if type(current).__name__ in _TRANSIENT_LLM_ERROR_NAMES:
            return True
        if getattr(current, "upstream_error_type", None) in _TRANSIENT_LLM_ERROR_NAMES:
            return True
        current = current.__cause__ or current.__context__
    return False


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

    # Models that reject function tools on /v1/chat/completions while reasoning is
    # on. The API answers 400 with:
    #   "Function tools with reasoning_effort are not supported for gpt-5.6-luna
    #    in /v1/chat/completions. To use function tools, use /v1/responses or set
    #    reasoning_effort to 'none'."
    # Sending reasoning_effort="none" alongside the tools keeps these models usable
    # on the completions path.
    TOOLS_REQUIRE_REASONING_DISABLED_MODELS = {
        "gpt-5.6",
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
        # ladder if one ever landed in this list.
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
            if self._tools_require_reasoning_disabled(actual_model):
                # setdefault, and placed before params.update(kwargs), so an
                # explicit caller-supplied reasoning_effort still wins.
                params.setdefault("reasoning_effort", "none")
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
        completion route and self-referential entries
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
        then starts the next rung fresh — preserving the timeout-regression
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
                f"({', '.join(context_parts)})",
                upstream_error_type=payload.type_name,
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
        ``llm.primary_model`` / ``llm.fallback_model`` error-reporting tags — so
        dashboards and alerts keep working; adds ``llm.fallback_reason`` so alerting can page
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
        set_error_tags(
            **{
                "llm.fallback_used": "true",
                "llm.primary_model": primary_model,
                "llm.fallback_model": served_model,
                "llm.fallback_reason": reason,
            }
        )

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
        unreachable and produced 490-second requests. Each rung is
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

            Every mutable value is DETACHED from the caller's objects before
            it is built into ``params``: litellm's transports mutate request
            state in place (see ``_detach_request_state``), and this walk
            re-reads the caller's kwargs on the next rung and on the
            corrective retry. ``dict(turn_kwargs)`` alone was not enough — it
            copies the top level and leaves ``extra_headers`` shared.
            """
            (
                params,
                response_format,
                parse_structured_output,
                _max_retries,
                _fallbacks,
            ) = self._build_completion_params(
                _detach_request_state(turn_messages),
                **_detach_request_state(dict(turn_kwargs)),
            )
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
            error_lines = "\n".join(
                f"- {error}" for error in _bounded_repair_errors(errors)
            )
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

        def _corrective_kwargs(rung_kwargs: dict[str, Any]) -> dict[str, Any]:
            """Re-key a corrective retry so a provider cannot replay attempt 1.

            Only the idempotency header moves. Everything else -- model,
            sampling, timeout, response format -- is left exactly as the first
            attempt sent it, because the request being corrected is still the
            same question.

            ``headers`` is the CALLER's dict and is spread verbatim, so it must
            hold only what the caller asked for. That is guaranteed by
            ``_prepare_turn`` detaching request state before the transport ever
            sees it (``_detach_request_state``); without that, attempt 1's
            provider-injected ``Authorization``/``Content-Type`` would be
            spread into the corrective retry as caller intent.
            """
            headers = rung_kwargs.get("extra_headers")
            if not isinstance(headers, dict):
                return rung_kwargs
            base_key = headers.get("Idempotency-Key")
            if not isinstance(base_key, str) or not base_key:
                return rung_kwargs
            return {
                **rung_kwargs,
                "extra_headers": {
                    **headers,
                    "Idempotency-Key": structured_output_repair_idempotency_key(
                        base_key
                    ),
                },
            }

        def _run_rung_plain(
            rung_messages: list[dict[str, Any]], rung_kwargs: dict[str, Any]
        ) -> CompletionResult[str | BaseModel | ToolCallingChatResponse]:
            """Serve one rung with no validator: initial call + one same-model retry.

            The retry is CORRECTIVE whenever the parse failure produced
            content-free validation errors: the rung's original messages plus a
            bounded turn naming the field paths and pydantic error tags that
            failed. It was previously byte-identical, which at ``temperature=0``
            is two identical attempts at a schema the model has already misread
            -- the second could only fail the same way. When the failure names
            nothing (the deliberate "appears truncated" diagnostic, whose
            ``validation_errors`` is empty), the retry stays the byte-identical
            re-issue it always was.

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
            except StructuredOutputParseError as exc:
                if not exc.validation_errors:
                    self.logger.warning(
                        "event=llm_parse_retry model=%s — malformed structured output, "
                        "retrying once on the same model",
                        params.get("model"),
                    )
                    # REBUILD, never reuse ``params``. ``_prepare_turn`` detaches
                    # request state from the CALLER's objects, which protects the
                    # caller and every later rung -- but within this rung the one
                    # ``params`` dict has already crossed to the transport, and
                    # litellm's in-process route (``litellm.completion(**params)``)
                    # lets ``validate_environment`` write ``Authorization`` /
                    # ``Content-Type`` into ``params["extra_headers"]``. Passing
                    # the same object again sends those provider-injected headers
                    # as CALLER intent, which a ``provider_request_guard`` pinning
                    # the exact request correctly refuses -- killing the re-issue
                    # on the one branch (empty ``validation_errors``: the
                    # "appears truncated" diagnostic) that still needs it.
                    #
                    # The rebuilt turn is deliberately BYTE-IDENTICAL to attempt
                    # 1, idempotency key included. It is not re-keyed: nothing
                    # about the request has changed, and the qualified caller
                    # this path exists for pins ``extra_headers`` to the exact
                    # key it issued while budgeting a bounded number of
                    # IDENTICAL re-issues. Deriving a fresh key here would make
                    # the re-issue fail that pin -- trading a replay risk this
                    # branch does not carry (a truncated body is a transport
                    # /decoding outcome, not a decision the provider caches)
                    # for a guaranteed refusal.
                    (
                        reissue_params,
                        reissue_rf,
                        reissue_parse_so,
                        reissue_timeout,
                    ) = _prepare_turn(rung_messages, rung_kwargs)
                    return _finish(
                        _call_and_parse(
                            reissue_params,
                            reissue_rf,
                            reissue_parse_so,
                            reissue_timeout,
                            detect_refusal=False,
                        )
                    )
                schema_name = getattr(rf, "__name__", "structured output")
                # The COUNT, not the errors. An ``extra_forbidden`` entry puts a
                # MODEL-CHOSEN key in the pydantic ``loc``, and this line reaches
                # external telemetry via the logging bridge. The errors
                # themselves go into the retry prompt verbatim, back to the
                # provider that produced them. Where a log line does need them
                # (``_run_rung_validated``), they go through
                # ``_log_safe_validation_errors`` first -- ``loc`` is built from
                # ``str(part)`` with no charset, length or count clamp of its
                # own, so nothing else bounds what reaches the sink.
                self.logger.warning(
                    "event=llm_parse_retry_corrective model=%s schema=%s "
                    "validation_error_count=%d — malformed structured output, "
                    "retrying once on the same model with the validation errors "
                    "fed back",
                    params.get("model"),
                    schema_name,
                    len(exc.validation_errors),
                )
                (
                    retry_params,
                    retry_rf,
                    retry_parse_so,
                    retry_timeout,
                ) = _prepare_turn(
                    [
                        *rung_messages,
                        *structured_output_correction_turn(
                            schema_name=schema_name, errors=exc.validation_errors
                        ),
                    ],
                    _corrective_kwargs(rung_kwargs),
                )
                return _finish(
                    _call_and_parse(
                        retry_params,
                        retry_rf,
                        retry_parse_so,
                        retry_timeout,
                        detect_refusal=False,
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
                        _log_safe_validation_errors(exc.validation_errors),
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
            # Re-keyed through ``_corrective_kwargs``, exactly as the plain
            # rung's corrective retry is. The repair is a STRICTLY LARGER
            # change to the request than that one -- ``_repair_messages`` echoes the model's own raw
            # output back alongside the correction -- so reusing attempt 1's
            # ``Idempotency-Key`` would entitle a provider honouring the header
            # to replay the very answer being repaired, turning the repair into
            # the silent no-op ``structured_output_repair_idempotency_key``
            # exists to prevent.
            repair_params, repair_rf, repair_parse, repair_timeout = _prepare_turn(
                repair_base, _corrective_kwargs(rung_kwargs)
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
                        _log_safe_validation_errors(exc.validation_errors),
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

    def _tools_require_reasoning_disabled(self, model: str) -> bool:
        """Check whether a model needs reasoning off to accept function tools.

        Args:
            model: Model name to check.

        Returns:
            True if the model rejects function tools unless reasoning_effort is
            "none" on the chat/completions path.
        """
        model_lower = model.lower()
        # Strip provider routing prefixes (e.g., "openrouter/openai/gpt-5.6-luna").
        model_name = model_lower.rsplit("/", 1)[-1]
        return any(
            model_name.startswith(restricted) or model_name == restricted
            for restricted in self.TOOLS_REQUIRE_REASONING_DISABLED_MODELS
        )

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
