import inspect
import logging
import os
import sys
from copy import deepcopy
from typing import Any

from pydantic import BaseModel

from reflexio.models.structured_output import find_schema_keyword

logger = logging.getLogger(__name__)

# JSON-Schema keywords that strict structured-output endpoints (OpenAI, minimax)
# reject; see PYTHON-FASTAPI-9J.
PROVIDER_UNSAFE_KEYWORDS = ("oneOf", "discriminator")


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
        # ``oneOf`` and ``discriminator``. Folded inline here, fused into this
        # single structure-aware pass (the shared ``_fold_oneof_to_anyof`` helper
        # does the same fold for the model-boundary hook; kept inline here to
        # avoid a second full-tree walk). ``visit`` only ever recurses into
        # property/$def *values*, never the name maps, so a field literally named
        # ``oneOf`` is preserved — same contract as the helper.
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


def assert_provider_safe_schema(schema: dict[str, Any], *, name: str = "") -> None:
    """Enforce that an emitted structured-output schema is provider-safe.

    Strict structured-output endpoints (OpenAI, minimax) reject ``oneOf`` /
    ``discriminator`` (Sentry PYTHON-FASTAPI-9J). Models that inherit
    ``StrictStructuredOutput`` are safe by construction; this is the runtime net
    at the call boundary for anything that bypasses that guarantee — a model that
    forgot the base, or a tool-argument / dynamically-built schema not covered by
    the registry contract test.

    Enforcement: under pytest (``"pytest" in sys.modules``) it RAISES so a
    regression fails CI loudly — including at import/collection time, which a
    per-test signal like ``PYTEST_CURRENT_TEST`` would miss. In prod it logs a
    warning (observability) and returns; it does NOT mutate what is sent. So a
    forgot-the-base model is meant to be caught **pre-merge** (by this raise plus
    the registry contract test), not auto-repaired at runtime: on the strict /
    allowlisted path ``make_strict_json_schema`` independently folds the schema,
    but on the raw passthrough path the warning is the only signal and an unfolded
    ``oneOf`` would still reach the provider. Keep every output model on
    ``StrictStructuredOutput``.

    Args:
        schema (dict[str, Any]): The emitted JSON schema to check.
        name (str): Identifier for the schema's source, used in the message.
    """
    offenders = [
        kw for kw in PROVIDER_UNSAFE_KEYWORDS if find_schema_keyword(schema, kw)
    ]
    if not offenders:
        return
    msg = (
        f"Structured-output schema {name or '<unnamed>'!r} contains provider-unsafe "
        f"keyword(s) {offenders}; strict providers reject these. Inherit "
        "StrictStructuredOutput so the schema folds oneOf->anyOf by construction "
        "(Sentry PYTHON-FASTAPI-9J)."
    )
    if "pytest" in sys.modules:
        raise ValueError(msg)
    logger.warning(msg)


def strict_response_format_for_model(
    model: type[BaseModel], schema: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build a LiteLLM/OpenAI-compatible strict ``json_schema`` response format.

    Args:
        model: The Pydantic model (supplies the schema ``name``).
        schema: Optional pre-built ``model.model_json_schema()`` to reuse, avoiding
            a second schema build when the caller already has one.
    """

    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "schema": make_strict_json_schema(
                schema if schema is not None else model.model_json_schema()
            ),
            "strict": True,
        },
    }
