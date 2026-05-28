"""Minimal Fernet encryption helper for the Braintrust connector.

Reads `REFLEXIO_FERNET_KEYS` (comma-separated) from the environment. When
the env var is unset, both `encrypt` and `decrypt` return the value
unchanged — matching the existing `EncryptManager` behavior in
`reflexio_ext.utils` and keeping OS dev mode workable without rotation
infrastructure.

In enterprise deployments, the env var IS set and the API key is stored
ciphertext-only.
"""

from __future__ import annotations

import logging
import os

from cryptography import fernet
from cryptography.fernet import Fernet, MultiFernet

logger = logging.getLogger(__name__)


_ENV_KEY = "REFLEXIO_FERNET_KEYS"
_MULTI: MultiFernet | None = None
_LOADED = False


def _load() -> MultiFernet | None:
    """Lazy-load the MultiFernet from env on first use."""
    global _MULTI, _LOADED
    if _LOADED:
        return _MULTI
    raw = os.environ.get(_ENV_KEY, "").strip()
    if not raw:
        logger.info(
            "%s is unset — Braintrust connector storing API keys in plaintext "
            "(suitable only for local development).",
            _ENV_KEY,
        )
        _LOADED = True
        return None
    fernets = []
    for k in raw.split(","):
        k = k.strip()
        if not k:
            continue
        try:
            fernets.append(Fernet(k.encode("utf-8")))
        except Exception:  # noqa: BLE001
            logger.warning("Discarding invalid Fernet key in %s", _ENV_KEY)
    if not fernets:
        raise RuntimeError(f"{_ENV_KEY} is set but contains no valid Fernet keys")
    _MULTI = MultiFernet(fernets)
    _LOADED = True
    return _MULTI


def encrypt(value: str) -> str:
    """Encrypt `value` if Fernet keys are configured; otherwise return as-is.

    Args:
        value (str): The plaintext (e.g., a Braintrust API key).

    Returns:
        str: Ciphertext, or `value` unchanged when no keys are configured.
    """
    m = _load()
    if m is None:
        return value
    return m.encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt(value: str) -> str:
    """Decrypt `value` if Fernet keys are configured; otherwise return as-is.

    Args:
        value (str): The (possibly ciphertext) value from storage.

    Returns:
        str: Plaintext.

    Raises:
        InvalidToken: When the token cannot be decoded by any registered key.
    """
    m = _load()
    if m is None:
        return value
    try:
        return m.decrypt(value.encode("utf-8")).decode("utf-8")
    except fernet.InvalidToken:
        # Storage may contain plaintext from a pre-encryption deployment.
        # Caller decides whether to surface the issue.
        raise


def _reset_for_test() -> None:
    """Reset the cached MultiFernet — for tests only."""
    global _MULTI, _LOADED
    _MULTI = None
    _LOADED = False
