"""The two canonical-digest primitives the open-world path persists.

``canonical_json_bytes`` and ``sha256_hex`` are not internal details: their
output is signed into stored rows by the open-world path. Changing
either changes a persisted digest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def canonical_json_bytes(payload: object) -> bytes:
    """Encode the schema's integer/string-only JSON values as RFC 8785 bytes."""
    return _canonical_json(payload).encode("utf-8")


def _canonical_json(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        _reject_surrogates(value)
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, int):
        if not -(2**53) < value < 2**53:
            raise ValueError(
                "RFC 8785 integers must be exactly representable by IEEE 754"
            )
        return str(value)
    if isinstance(value, tuple | list):
        return "[" + ",".join(_canonical_json(item) for item in value) + "]"
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("RFC 8785 object keys must be strings")
        keys = sorted(value, key=lambda key: key.encode("utf-16be"))
        return (
            "{"
            + ",".join(
                f"{_canonical_json(key)}:{_canonical_json(value[key])}" for key in keys
            )
            + "}"
        )
    raise TypeError(f"Unsupported RFC 8785 value: {type(value).__name__}")


def _reject_surrogates(value: str) -> None:
    if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise ValueError("RFC 8785 strings cannot contain surrogate code points")


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
