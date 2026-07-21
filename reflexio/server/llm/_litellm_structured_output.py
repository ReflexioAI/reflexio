"""Structured-output concern for ``LiteLLMClient`` — schema selection + parsing (Tier-2.5).

``StructuredOutputMixin`` holds the response-format selection helpers
(``_supports_response_schema``, ``_provider_for_model``,
``_accepts_json_schema_response_format``, ``_provider_response_format``) and the
post-hoc parse orchestrator (``_maybe_parse_structured_output``), plus the
``_JSON_SCHEMA_PROVIDER_ALLOWLIST`` class constant they use.

SINK-1 (patch-where-used): ``assert_provider_safe_schema`` is imported HERE, so
tests patch it at ``_litellm_structured_output.assert_provider_safe_schema``
(the real guard raises under pytest; patching the old facade namespace would
no-op). ``litellm.supports_response_schema``/``get_llm_provider`` are called via
the shared ``litellm`` module attr.

Bodies moved VERBATIM; only the ``self``-typing (per-mixin TYPE_CHECKING stub of
``config``, Tier-1b idiom) is added. ``_provider_response_format`` and
``_maybe_parse_structured_output`` are the cross-mixin edges text-gen depends on,
so this module moves BEFORE ``_litellm_text_generation``.
"""

import json
from functools import lru_cache
from typing import TYPE_CHECKING, Any, get_args, get_origin

import litellm
from pydantic import BaseModel

from reflexio.server.llm._litellm_json_extraction import (
    _extract_json_from_string,
    _looks_truncated_json,
    _sanitize_json_string,
)
from reflexio.server.llm._litellm_types import StructuredOutputParseError
from reflexio.server.llm.llm_utils import (
    assert_provider_safe_schema,
    is_pydantic_model,
    prompt_schema_instruction,
    strict_response_format_for_model,
)

if TYPE_CHECKING:
    from reflexio.server.llm._litellm_types import LiteLLMConfig


def _single_list_field_name(response_format: type[BaseModel]) -> str | None:
    """Return the wrapper field for schemas shaped as ``{"items": [...]}``."""
    fields = getattr(response_format, "model_fields", {})
    if len(fields) != 1:
        return None

    field_name, field = next(iter(fields.items()))
    if _is_list_annotation(field.annotation):
        return field_name
    return None


def _is_list_annotation(annotation: Any) -> bool:
    """Return whether an annotation accepts a list value."""
    if annotation is list or get_origin(annotation) is list:
        return True
    return any(
        _is_list_annotation(arg)
        for arg in get_args(annotation)
        if arg is not type(None)
    )


def _validate_structured_payload(
    response_format: type[BaseModel],
    parsed: Any,
) -> BaseModel:
    if isinstance(parsed, list) and (
        field_name := _single_list_field_name(response_format)
    ):
        return response_format.model_validate({field_name: parsed})
    return response_format.model_validate(parsed)


class StructuredOutputMixin:
    """Response-format schema selection + structured-output parsing.

    Mixed into ``LiteLLMClient``; ``self.config`` is owned by the client-core
    ``__init__`` on the facade. The annotation-only stub below (Tier-1b idiom)
    gives pyright the foreign-member type without shared class-level state.
    """

    # OpenAI-compatible providers that accept a ``json_schema`` response_format
    # but that ``litellm.supports_response_schema`` reports as unsupported. For
    # these, the gate below would fall back to handing LiteLLM the raw Pydantic
    # model; LiteLLM then builds the ``json_schema`` itself and emits ``oneOf``
    # for discriminated unions, which strict structured-output endpoints reject
    # (Sentry PYTHON-FASTAPI-9J). Listing the provider here forces our own
    # normalized strict schema (``oneOf`` folded into ``anyOf``) to be sent.
    _JSON_SCHEMA_PROVIDER_ALLOWLIST: frozenset[str] = frozenset({"minimax"})
    _PROMPT_SCHEMA_PROVIDER_ALLOWLIST: frozenset[str] = frozenset({"zai"})

    # Base-owned attribute read for the parse-failure error message (init'd in
    # the facade ``__init__``). Annotation-only; NEVER assign here.
    config: "LiteLLMConfig"

    @staticmethod
    @lru_cache(maxsize=256)
    def _supports_response_schema(model: str) -> bool:
        try:
            return bool(litellm.supports_response_schema(model=model))
        except Exception:
            return False

    @staticmethod
    @lru_cache(maxsize=256)
    def _provider_for_model(model: str) -> str | None:
        try:
            return litellm.get_llm_provider(model)[1]
        except Exception:
            return None

    @classmethod
    def _accepts_json_schema_response_format(cls, model: str) -> bool:
        """Whether to send ``model`` an explicit strict ``json_schema`` schema.

        True when LiteLLM reports native response-schema support, or when the
        provider is a known OpenAI-compatible endpoint that LiteLLM
        under-reports (see ``_JSON_SCHEMA_PROVIDER_ALLOWLIST``). In the latter
        case LiteLLM would otherwise forward a ``json_schema`` it built itself,
        emitting ``oneOf`` for discriminated unions that the endpoint rejects.
        """
        if cls._supports_response_schema(model):
            return True
        return cls._provider_for_model(model) in cls._JSON_SCHEMA_PROVIDER_ALLOWLIST

    @classmethod
    def _structured_output_strategy(
        cls, *, model: str, strict_response_format: bool
    ) -> str:
        """Return the provider transport used for a Pydantic response schema."""
        if not strict_response_format:
            return "pydantic_passthrough"
        if cls._provider_for_model(model) in cls._PROMPT_SCHEMA_PROVIDER_ALLOWLIST:
            return "prompt_json_object"
        if cls._accepts_json_schema_response_format(model):
            return "native_json_schema"
        return "pydantic_passthrough"

    def _prompt_schema_directive(
        self, *, response_format: type[BaseModel], tools_available: bool
    ) -> str:
        """Build and guard the schema instruction used by prompt-only providers."""
        schema = response_format.model_json_schema()
        assert_provider_safe_schema(schema, name=response_format.__name__)
        return prompt_schema_instruction(schema, tools_available=tools_available)

    def _provider_response_format(
        self,
        *,
        response_format: Any,
        model: str,
        strict_response_format: bool,
    ) -> Any:
        """Return the provider-facing response_format while preserving parser schema.

        Callers pass a Pydantic model so local parsing stays type-safe. When the
        target model accepts a JSON Schema response format — either LiteLLM
        reports native support, or the provider is an OpenAI-compatible endpoint
        LiteLLM under-reports (see ``_accepts_json_schema_response_format``) — we
        send an explicit strict schema to constrain generation. Truly
        unsupported providers keep the existing Pydantic response_format
        behavior.
        """

        if not is_pydantic_model(response_format):
            return response_format

        # Build the native schema once and reuse it for both the boundary guard and
        # (when applicable) the strict normalizer, avoiding a second schema build.
        # Boundary guard: models inheriting StrictStructuredOutput are safe by
        # construction; this catches a model that forgot the base (raises under
        # tests, warns in prod) regardless of which path is taken below.
        schema = response_format.model_json_schema()
        assert_provider_safe_schema(schema, name=response_format.__name__)

        if (
            self._structured_output_strategy(
                model=model, strict_response_format=strict_response_format
            )
            == "native_json_schema"
        ):
            return strict_response_format_for_model(response_format, schema=schema)
        return response_format

    def _maybe_parse_structured_output(
        self,
        content: Any,
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
            raise StructuredOutputParseError(
                "Structured output response content was empty",
                raw_content=None,
            )

        # If content is already a Pydantic model (some providers return parsed)
        if isinstance(content, BaseModel):
            return content

        # Try to parse JSON and convert to Pydantic model
        # Extract JSON from markdown code blocks if present
        json_str = _extract_json_from_string(content)
        try:
            parsed = json.loads(json_str)

            # response_format must be a Pydantic model (validated at entry points)
            return _validate_structured_payload(response_format, parsed)
        except Exception:
            # LLMs sometimes produce Python-style output (single quotes, True/False,
            # trailing commas). Try to sanitize before giving up.
            try:
                sanitized = _sanitize_json_string(json_str)
                parsed = json.loads(sanitized)
                return _validate_structured_payload(response_format, parsed)
            except Exception:
                # Last resort: json-repair can recover complete responses with
                # small syntax glitches, such as missing commas. Do not repair
                # likely truncation: the retry loop should request a fresh
                # complete response instead of accepting invented tail content.
                try:
                    from json_repair import repair_json

                    if _looks_truncated_json(json_str):
                        raise StructuredOutputParseError(
                            "Structured output appears truncated",
                            raw_content=content,
                        )

                    repaired = repair_json(json_str, return_objects=True)
                    return _validate_structured_payload(response_format, repaired)
                except Exception as e:
                    model = self.config.model
                    # Do NOT embed the raw model output in the exception message:
                    # this exception is logged at ERROR and rides to Sentry/CloudWatch
                    # via the logging bridge, so a content snippet would leak Customer
                    # Content there. Log only the length; the raw text stays on
                    # `raw_content` for in-process repair (not serialized to logs).
                    content_len = len(content) if isinstance(content, str) else -1
                    raise StructuredOutputParseError(
                        f"Structured output parse failed for model={model!r}: {e}. "
                        f"Content length: {content_len} chars (content omitted from logs).",
                        raw_content=content if isinstance(content, str) else None,
                    ) from e
