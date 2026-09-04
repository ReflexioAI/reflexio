"""Render validation errors safely into an HTTP response body.

Two independent hazards make a raw ``errors()`` list unsafe to hand to FastAPI,
and both were live before this module existed:

**It does not always serialize.** Each entry carries a ``ctx`` whose ``error``
key is the *live exception object* a validator raised. A ``model_validator``
that raises ``ValueError`` therefore produces
``{'ctx': {'error': ValueError(...)}}``, and ``json.dumps`` on that raises
``TypeError: Object of type ValueError is not JSON serializable``. The handler
building the 400 then dies inside the exception handler, so the caller sees a
bare **500 Internal Server Error** instead of the reason. That is exactly what
``PUT /api/projects/{id}/config`` did for any override whose merged result
violated ``stride_size <= window_size``.

**It echoes the whole input.** Each entry's ``input`` key is the object that
failed validation -- for a ``Config`` that is the *entire configuration
document*, including ``storage_config`` (which carries a ``db_url`` with a
password), ``api_key_config``, ``llm_config``, and
``pending_tool_call_config.hmac_secrets``. Had the payload serialized, all of
it would have been returned to the caller. The serialization failure was
accidentally the only thing preventing a credential leak, so "make it
serializable" is precisely the wrong fix.

Keep ``type``, ``loc`` and ``msg``: they identify which field failed and why,
which is the entire purpose of returning validation errors, and none of the
three ever contains submitted values.

Takes the error *list* rather than the exception because the two callers raise
different types -- Pydantic's ``ValidationError`` and FastAPI's
``RequestValidationError`` -- that share only the ``errors()`` accessor.
"""

from collections.abc import Iterable, Mapping
from typing import Any

__all__ = ["safe_validation_errors"]

# `input` echoes the submitted document (secrets included); `ctx` holds the live
# exception object that breaks JSON serialization; `url` is a docs link that
# adds nothing to a machine-readable response.
_UNSAFE_KEYS = frozenset({"input", "ctx", "url"})


def safe_validation_errors(
    errors: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Strip submitted values and unserializable context from validation errors.

    Args:
        errors (Iterable[Mapping[str, Any]]): The raw list from
            ``ValidationError.errors()`` or ``RequestValidationError.errors()``.

    Returns:
        list[dict[str, Any]]: One dict per error carrying ``type``, ``loc`` and
        ``msg`` only. ``loc`` is coerced to a list because Pydantic returns a
        tuple, which is not JSON-native.
    """
    safe: list[dict[str, Any]] = []
    for error in errors:
        entry: dict[str, Any] = {
            key: value for key, value in error.items() if key not in _UNSAFE_KEYS
        }
        loc = entry.get("loc")
        if isinstance(loc, tuple):
            entry["loc"] = list(loc)
        safe.append(entry)
    return safe
