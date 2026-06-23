import inspect
import logging
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema


def positive_int_env(name: str, default: int, logger: logging.Logger) -> int:
    """Resolve a strictly-positive int from environment variable ``name``.

    Falls back to ``default`` when the variable is unset/blank, not a valid
    integer (logging a warning in that case), or not strictly positive.

    Args:
        name (str): Environment variable to read.
        default (int): Value returned when the variable is missing or invalid.
        logger (logging.Logger): Logger used to warn on a non-integer value.

    Returns:
        int: The parsed positive integer, or ``default``.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; falling back to default %d", name, raw, default)
        return default
    return value if value > 0 else default


_STRICT_SCHEMA_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "maxItems",
        "maxLength",
        "maxProperties",
        "maximum",
        "minItems",
        "minLength",
        "minProperties",
        "minimum",
        "multipleOf",
        "pattern",
        "patternProperties",
        "propertyNames",
        "uniqueItems",
    }
)


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


def is_pydantic_model(response_format: Any) -> bool:
    """
    Check if response_format is a Pydantic BaseModel class.

    Args:
        response_format: Response format to check.

    Returns:
        True if response_format is a Pydantic BaseModel class, False otherwise.
    """
    return inspect.isclass(response_format) and issubclass(response_format, BaseModel)


def make_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON schema shaped for strict structured-output decoding.

    Strict structured-output providers generally require object schemas to
    forbid extra keys and list all properties as required. Pydantic optional
    fields are still preserved as nullable via their existing ``anyOf`` /
    ``type: ["...", "null"]`` schema, so the model can emit ``null`` instead
    of omitting the field. Provider-unsupported validation keywords are
    removed from the request schema; Pydantic still enforces them after parse.
    """

    strict_schema = deepcopy(schema)

    def visit(node: Any) -> None:
        if not isinstance(node, dict):
            return

        node.pop("default", None)
        for keyword in _STRICT_SCHEMA_UNSUPPORTED_KEYWORDS:
            node.pop(keyword, None)

        # Strict structured output (OpenAI) permits ``anyOf`` but rejects
        # ``oneOf`` and ``discriminator``. Folded INLINE here (deliberately NOT
        # via the shared ``_fold_oneof_to_anyof`` helper): this traversal is
        # schema-structure-aware and only treats ``oneOf``/``discriminator`` as
        # keywords at schema nodes, whereas the helper is a blind walk that would
        # also strip a *property literally named* ``oneOf`` (see its docstring).
        # Keep the two separate; they have different traversal contracts.
        one_of = node.pop("oneOf", None)
        node.pop("discriminator", None)
        if isinstance(one_of, list):
            node["anyOf"] = node.get("anyOf", []) + one_of

        properties = node.get("properties")
        if isinstance(properties, dict):
            node["additionalProperties"] = False
            node["required"] = list(properties.keys())
            for child in properties.values():
                visit(child)
        else:
            additional = node.get("additionalProperties")
            if isinstance(additional, dict):
                visit(additional)

        defs = node.get("$defs")
        if isinstance(defs, dict):
            for child in defs.values():
                visit(child)

        for key in ("items", "contains", "not"):
            visit(node.get(key))

        for key in ("anyOf", "allOf", "prefixItems"):
            children = node.get(key)
            if isinstance(children, list):
                for child in children:
                    visit(child)

    visit(strict_schema)
    return strict_schema


def strict_response_format_for_model(model: type[BaseModel]) -> dict[str, Any]:
    """Build a LiteLLM/OpenAI-compatible strict ``json_schema`` response format."""

    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "schema": make_strict_json_schema(model.model_json_schema()),
            "strict": True,
        },
    }
