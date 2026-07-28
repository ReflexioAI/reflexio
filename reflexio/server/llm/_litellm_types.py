"""Shared types + exceptions for the ``litellm_client`` facade (Tier-2.5 leaf).

Stateless leaf module: the config dataclass, the tool-calling response dataclass,
and the three client exceptions. Both the facade and the concern mixins import
these, so keeping them in a dependency-free leaf prevents a mixin<->facade cycle.

Bodies are moved VERBATIM from the former monolithic ``litellm_client.py`` — no
behavior change. ``LiteLLMConfig``/``ToolCallingChatResponse``/``LiteLLMClientError``
are re-exported by import binding from the facade (SINK-2, identity-preserving)
so the ~102 importers are unchanged.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from reflexio.models.config_schema import APIKeyConfig


@dataclass(frozen=True)
class ModelProvenance:
    """Observed model and provider attribution for one completion."""

    model_name: str | None = None
    provider: str | None = None


@dataclass(frozen=True)
class CompletionResult[T]:
    """Completion value paired with its non-serializing provenance."""

    value: T
    provenance: ModelProvenance


@dataclass
class LiteLLMConfig:
    """
    Configuration for LiteLLM client.

    Args:
        model: Model name to use (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022').
        temperature: Temperature for response generation (0.0 to 2.0).
        max_tokens: Maximum tokens to generate.
        timeout: Request timeout in seconds.
        max_retries: Maximum same-model retry attempts. Used by the embedding
            path (litellm's num_retries) and clamped in _build_completion_params.
            NOT used on the chat-completion path — that forces num_retries=0 so a
            hung primary can't be retried before the fallback (PYTHON-FASTAPI-62).
            Default 3.
        retry_delay: Currently unused — LiteLLM owns retry backoff. Kept for
            backward compatibility; remove in a follow-up sweep.
        top_p: Top-p sampling parameter.
        api_key_config: Optional API key configuration from Config (overrides env vars).
        fallback_models: Models reflexio walks in order, one rung at a time,
            after the primary's single attempt fails. NOT passed to litellm's
            fallbacks param — the client owns the walk (`[primary, *fallbacks]`)
            and rebuilds request params (structured-output strategy, api_base,
            per-rung timeout) for each rung, so entries may mix providers and
            transports freely; a native-JSON-schema primary can fall back to a
            prompt-backed provider. `num_retries` is forced to 0 on every rung
            so a hung primary can't be retried before the fallback advances
            (PYTHON-FASTAPI-62). For opted-in structured-output repair, the
            first eligible network entry is also used as the final repair
            escalation model after same-model repair fails.
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
        parsed_output: When ``response_format`` is passed alongside ``tools`` and the model
            ends the turn with a plain (non-tool) response, the content parsed into the
            ``response_format`` schema. None when the turn emitted tool calls, when no
            ``response_format`` was requested, or when the content was not parseable.
    """

    content: str | None
    tool_calls: list[Any] | None
    finish_reason: str | None
    usage: Any | None = None
    cost_usd: float | None = None
    parsed_output: BaseModel | None = None


class LiteLLMClientError(Exception):
    """Custom exception for LiteLLM client errors.

    ``first_parsed_provenance`` is populated when a later structured-output
    repair transport failure leaves a parsed response available to a caller.
    """

    def __init__(
        self,
        message: str,
        *,
        first_parsed_provenance: ModelProvenance | None = None,
    ) -> None:
        super().__init__(message)
        self.first_parsed_provenance = first_parsed_provenance


class StructuredOutputRepairError(LiteLLMClientError):
    """Raised when an opted-in structured-output repair ladder is exhausted.

    Field pairing caveat: ``raw_content``/``validation_errors`` describe the
    LAST attempt, while ``parsed_output`` falls back to the most recent attempt
    that parsed at all. ``first_parsed_provenance`` is the first parse across the
    whole multi-rung walk (not merely the final rung), so salvage callers can
    pair it with the first accepted parsed content from a shared validator
    closure.
    """

    def __init__(
        self,
        message: str,
        *,
        failure_kind: Literal["parse", "semantic", "refusal"],
        model: str,
        raw_content: str | None = None,
        parsed_output: BaseModel | None = None,
        validation_errors: tuple[str, ...] = (),
        first_parsed_provenance: ModelProvenance | None = None,
    ) -> None:
        super().__init__(message, first_parsed_provenance=first_parsed_provenance)
        self.failure_kind = failure_kind
        self.model = model
        self.raw_content = raw_content
        self.parsed_output = parsed_output
        self.validation_errors = validation_errors


class StructuredOutputParseError(Exception):
    """Raised when a structured-output LLM call returns content that cannot be parsed.

    Caught by the retry loop in ``_make_request`` so a malformed response
    burns a retry attempt rather than silently returning unparsed content.
    """

    def __init__(
        self,
        message: str,
        *,
        raw_content: str | None = None,
        finish_reason: str | None = None,
        provenance: ModelProvenance | None = None,
        validation_errors: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.raw_content = raw_content
        self.finish_reason = finish_reason
        self.provenance = provenance
        self.validation_errors = validation_errors


class LLMHardTimeoutError(TimeoutError):
    """Raised when an LLM call exceeds the client-side wall-clock timeout."""
