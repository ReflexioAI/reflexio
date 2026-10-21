"""Finite configuration JSON without rounding opaque numeric metadata.

Pure raw-document representation shared by storage and explicit migration.
No schema validation, historical rewrites, environment discovery or I/O.
"""

from decimal import DecimalException
from typing import Any

import simplejson


def decode_config_document(raw: str) -> dict[str, Any]:
    """Decode an object with exact decimal numbers and value-free failures."""
    try:
        payload = simplejson.loads(raw, use_decimal=True, allow_nan=False)
        if not isinstance(payload, dict):
            raise ValueError("Expected an object")
        return payload
    except (ValueError, TypeError, DecimalException, RecursionError):
        raise ValueError("Configuration document is not a finite JSON object") from None


def encode_config_document(
    payload: dict[str, Any], *, indent: int | None = None
) -> str:
    """Use a stable finite JSON representation for floats and exact decimals.

    Normalize model-generated float notation through the same decoder before
    final encoding, so a subsequent read compares identically (including 1e-6).
    This preserves decimal scale without maintaining a numeric formatter.
    """
    try:
        if not isinstance(payload, dict):
            raise ValueError("Expected an object")
        normalized = simplejson.loads(
            simplejson.dumps(payload, use_decimal=True, allow_nan=False),
            use_decimal=True,
            allow_nan=False,
        )
        return simplejson.dumps(
            normalized,
            use_decimal=True,
            allow_nan=False,
            sort_keys=True,
            indent=indent,
        )
    except (ValueError, TypeError, OverflowError, DecimalException, RecursionError):
        raise ValueError("Configuration document cannot be encoded safely") from None
