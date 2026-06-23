"""Provider-safe base class for LLM structured-output Pydantic models.

Every Pydantic model sent to an LLM as a structured-output schema (a
``response_format`` or a tool's parameter schema) should inherit
``StrictStructuredOutput``. It carries a ``__get_pydantic_json_schema__`` hook
(via ``ProviderSafeUnionMixin``) that folds a discriminated union's ``oneOf`` into
``anyOf`` and drops ``discriminator`` in the *emitted* JSON schema, so strict
structured-output endpoints (OpenAI, minimax) accept it **by construction** — with
no provider-detection gate and no post-processing. Only the wire schema is
rewritten; the core (validation) schema keeps the discriminator, so keyed dispatch
and precise per-variant errors are preserved.

This module lives under ``models/`` (depending only on pydantic) so both
``models/api_schema/`` schemas and ``server/services/*`` models can inherit the
base without a layering inversion (a ``models/`` file importing from
``server/llm/`` would invert the dependency direction).
"""

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema


def _fold_oneof_to_anyof(node: Any) -> None:
    """Rewrite ``oneOf`` to ``anyOf`` and drop ``discriminator`` in place (recursive).

    Pydantic emits ``oneOf`` + ``discriminator`` for a discriminated union, which
    strict structured-output endpoints (OpenAI, minimax) reject. Folding into
    ``anyOf`` keeps the identical variant set so generation stays constrained;
    Pydantic still enforces the discriminator after parse, so semantics are
    preserved.

    This is a BLIND walk: it treats any dict key named ``oneOf``/``discriminator``
    as a schema keyword, so it must NOT be used where a model field could be
    literally named ``oneOf``/``discriminator`` (it would strip the property).
    That is fine for ``ProviderSafeUnionMixin``'s use on real output models;
    ``make_strict_json_schema`` instead folds inline within a structure-aware
    traversal precisely to avoid this. Precondition: ``node`` is a finite acyclic
    tree, as produced by ``model_json_schema()`` (which expresses recursion via
    ``$ref``/``$defs`` strings, never in-memory cycles) — there is no cycle guard.

    Args:
        node (Any): A JSON-schema fragment (dict, list, or scalar); mutated in place.
    """
    if isinstance(node, dict):
        one_of = node.pop("oneOf", None)
        node.pop("discriminator", None)
        if isinstance(one_of, list):
            node["anyOf"] = node.get("anyOf", []) + one_of
        for value in node.values():
            _fold_oneof_to_anyof(value)
    elif isinstance(node, list):
        for item in node:
            _fold_oneof_to_anyof(item)


# Statically the mixin must appear to derive from ``BaseModel`` so the ``super()``
# call below type-checks; at runtime it is a bare mixin (``object`` base), and
# ``super()`` resolves to ``BaseModel.__get_pydantic_json_schema__`` via the MRO
# of the concrete model (e.g. ``class X(ProviderSafeUnionMixin, BaseModel)``).
_ProviderSafeUnionBase = BaseModel if TYPE_CHECKING else object


class ProviderSafeUnionMixin(_ProviderSafeUnionBase):
    """Make a model emit a provider-safe JSON schema by construction.

    A model containing a Pydantic discriminated union serializes to JSON Schema
    with ``oneOf`` + ``discriminator``, which strict structured-output endpoints
    reject (the Sentry ``PYTHON-FASTAPI-9J`` incident). Mixing this in folds
    ``oneOf`` into ``anyOf`` (and drops ``discriminator``) at the model boundary,
    so every caller of ``model_json_schema()`` — litellm, instructor, our own
    path — gets a provider-safe schema **unconditionally**, without depending on
    a provider-detection gate (e.g. ``litellm.supports_response_schema``, which
    under-reports some providers). Only the JSON (wire) schema is rewritten; the
    core validation schema keeps the discriminator, so keyed dispatch and precise
    per-variant errors are preserved.

    Prefer inheriting ``StrictStructuredOutput`` (below) on output models rather
    than mixing this in directly.
    """

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # Chain via super() (not handler() directly) so a cooperative base that
        # also customizes __get_pydantic_json_schema__ is not silently shadowed;
        # BaseModel's default impl just invokes handler(core_schema).
        schema = super().__get_pydantic_json_schema__(core_schema, handler)
        _fold_oneof_to_anyof(schema)
        return schema


class StrictStructuredOutput(ProviderSafeUnionMixin, BaseModel):
    """Base for every Pydantic model sent to an LLM as a structured-output schema.

    Inherits the provider-safe schema hook from ``ProviderSafeUnionMixin`` so the
    emitted JSON schema folds any discriminated union (``oneOf``→``anyOf``, drop
    ``discriminator``) by construction — accepted by strict structured-output
    providers with no provider gate, including any *future* discriminated union.

    Deliberately adds NO ``model_config``: each model keeps its own ``extra=`` and
    ``json_schema_extra`` exactly as before. This base unifies the *provider-safety
    guarantee*, not the per-model config (centralizing config is a separate change —
    ``extra=`` is not uniform across models, so a blanket dedup would silently shift
    validation/serialization behavior).
    """
