import inspect
import logging
import os
from copy import deepcopy
from typing import Any

from pydantic import BaseModel


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
        # ``oneOf`` and ``discriminator``. Pydantic emits ``oneOf`` for
        # discriminated unions; fold it into ``anyOf`` so generation is
        # constrained to the same variants. Pydantic still enforces the
        # discriminator after parse, so semantics are preserved.
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
