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

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

# Keys whose *values* are name→subschema maps: their entries are user-chosen names
# (field names, $def names), not JSON-Schema keywords. Traversals must recurse into
# the subschemas but never treat the names themselves as keywords — otherwise a
# model with a field literally named ``oneOf`` would be mishandled.
_SCHEMA_NAME_MAP_KEYS = ("properties", "$defs", "definitions", "patternProperties")


def _fold_oneof_to_anyof(node: Any) -> None:
    """Rewrite ``oneOf`` to ``anyOf`` and drop ``discriminator`` in place (recursive).

    Pydantic emits ``oneOf`` + ``discriminator`` for a discriminated union, which
    strict structured-output endpoints (OpenAI, minimax) reject. Folding into
    ``anyOf`` keeps the identical variant set so generation stays constrained;
    Pydantic still enforces the discriminator after parse, so semantics are
    preserved.

    Structure-aware: under a name→subschema map (``properties``/``$defs``/…) it
    recurses into the subschemas but does NOT treat a property/``$def`` *name* as a
    keyword — so a model field literally named ``oneOf``/``discriminator`` is
    preserved (only keyword-position occurrences are folded). Precondition:
    ``node`` is a finite acyclic tree, as produced by ``model_json_schema()``
    (which expresses recursion via ``$ref``/``$defs`` strings, never in-memory
    cycles) — there is no cycle guard.

    Args:
        node (Any): A JSON-schema fragment (dict, list, or scalar); mutated in place.
    """
    if isinstance(node, dict):
        one_of = node.pop("oneOf", None)
        node.pop("discriminator", None)
        if isinstance(one_of, list):
            node["anyOf"] = node.get("anyOf", []) + one_of
        for key, value in node.items():
            if key in _SCHEMA_NAME_MAP_KEYS and isinstance(value, dict):
                for sub in value.values():
                    _fold_oneof_to_anyof(sub)
            else:
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
    reject (a production structured-output incident). Mixing this in folds
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


# ===============================
# Provider-output shape tolerance
# ===============================
#
# Structured-output providers occasionally rename a field or spell an enum value
# with a different separator/casing than the schema declares. The helpers below
# are the ONE place that canonicalizes those harmless variants, so every schema
# that tolerates provider drift does it identically instead of hand-rolling its
# own casefold/replace/alias chain (which is how two schemas silently start
# accepting different spellings).
#
# They are deliberately mechanical: they rename and recase, never invent or drop
# semantic content. Anything that needs judgement belongs in a model validator.


def normalize_provider_token(value: str, *, separator: str = "_") -> str:
    """Canonicalize a provider-emitted key or enum token.

    Trims, casefolds, and rewrites both the opposite separator and spaces to
    ``separator`` — so ``"Evidence Kind"``, ``"evidence-kind"``, and
    ``"evidence_kind"`` all collapse to one form.

    Args:
        value (str): Raw key or enum token from parsed LLM output.
        separator (str): The canonical separator for this vocabulary — ``"_"``
            for Python field names, ``"-"`` for hyphenated enum values.

    Returns:
        str: The canonical token.
    """
    opposite = "-" if separator == "_" else "_"
    return value.strip().casefold().replace(opposite, separator).replace(" ", separator)


def normalize_provider_keys(
    data: Mapping[Any, Any], aliases: Mapping[str, str]
) -> dict[str, Any]:
    """Rename provider key variants onto canonical field names.

    First key wins: when a payload carries both a canonical name and an alias for
    it, the earlier entry is kept rather than being silently overwritten by the
    later one.

    Args:
        data (Mapping[Any, Any]): One raw object from parsed LLM output.
        aliases (Mapping[str, str]): Canonical-token → field-name map. Keys must
            already be in ``normalize_provider_token`` form.

    Returns:
        dict[str, Any]: The same values under canonical field names.
    """
    normalized: dict[str, Any] = {}
    for key, item in data.items():
        canonical = normalize_provider_token(str(key))
        normalized.setdefault(aliases.get(canonical, canonical), item)
    return normalized


def normalize_provider_value(
    value: Any, aliases: Mapping[str, str], *, separator: str = "_"
) -> Any:
    """Canonicalize a provider-emitted enum-ish string value.

    Args:
        value (Any): Raw field value straight from the parsed LLM output.
        aliases (Mapping[str, str]): Canonical-token → canonical-value map.
        separator (str): The canonical separator for this vocabulary.

    Returns:
        Any: The canonical value, or ``value`` unchanged when it is not a string
        (so Pydantic reports the real type error instead of a coercion).
    """
    if not isinstance(value, str):
        return value
    canonical = normalize_provider_token(value, separator=separator)
    return aliases.get(canonical, canonical)


def find_schema_keyword(node: Any, keyword: str) -> bool:
    """Report whether ``keyword`` appears as a JSON-Schema *keyword* anywhere in ``node``.

    Structure-aware: an occurrence of ``keyword`` as a property/``$defs`` *name*
    (e.g. a model field literally named ``oneOf``) is NOT a match — only an
    occurrence in JSON-Schema keyword position counts. Use this (not a blind
    ``in``-walk) to check a schema for provider-unsafe keywords.

    Args:
        node (Any): A JSON-schema fragment (dict, list, or scalar).
        keyword (str): The JSON-Schema keyword to search for (e.g. ``"oneOf"``).

    Returns:
        bool: True if ``keyword`` appears in keyword position, else False.
    """
    if isinstance(node, dict):
        if keyword in node:
            return True
        for key, value in node.items():
            if key in _SCHEMA_NAME_MAP_KEYS and isinstance(value, dict):
                if any(find_schema_keyword(sub, keyword) for sub in value.values()):
                    return True
            elif find_schema_keyword(value, keyword):
                return True
        return False
    if isinstance(node, list):
        return any(find_schema_keyword(item, keyword) for item in node)
    return False
