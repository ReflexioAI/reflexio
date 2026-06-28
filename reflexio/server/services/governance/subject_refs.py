from __future__ import annotations

import hashlib
import hmac


def _hmac_ref(prefix: str, raw: str, secret: str) -> str:
    if not raw:
        raise ValueError("raw value must be non-empty")
    if not secret:
        raise ValueError("secret must be non-empty")
    digest = hmac.new(secret.encode(), raw.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{prefix}_v1_{digest}"


def subject_ref(raw: str, secret: str) -> str:
    return _hmac_ref("subref", raw, secret)


def actor_ref(raw: str, secret: str) -> str:
    return _hmac_ref("actref", raw, secret)


def request_ref(raw: str, secret: str) -> str:
    return _hmac_ref("reqref", raw, secret)


def stable_id(prefix: str, material: str) -> str:
    if not prefix:
        raise ValueError("prefix must be non-empty")
    if not material:
        raise ValueError("material must be non-empty")
    digest = hashlib.sha256(material.encode()).hexdigest()[:32]
    return f"{prefix}_{digest}"
