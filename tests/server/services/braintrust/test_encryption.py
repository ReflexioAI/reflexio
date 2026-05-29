"""Tests for the Braintrust encryption helper.

Exercises both branches: env-var unset (passthrough) and env-var set
(actual Fernet roundtrip).
"""

import os

import pytest
from cryptography.fernet import Fernet

from reflexio.server.services.braintrust import _encryption


def test_passthrough_when_env_key_unset(monkeypatch) -> None:
    """With no Fernet keys configured, encrypt/decrypt return the value untouched."""
    monkeypatch.delenv("REFLEXIO_FERNET_KEYS", raising=False)
    _encryption._reset_for_test()
    assert _encryption.encrypt("sk-secret") == "sk-secret"
    assert _encryption.decrypt("sk-secret") == "sk-secret"


def test_roundtrip_with_fernet_key(monkeypatch) -> None:
    """With a Fernet key set, encrypt produces ciphertext, decrypt recovers it."""
    key = Fernet.generate_key().decode("utf-8")
    monkeypatch.setenv("REFLEXIO_FERNET_KEYS", key)
    _encryption._reset_for_test()

    plaintext = "sk-customer-braintrust-key"
    ciphertext = _encryption.encrypt(plaintext)
    assert ciphertext != plaintext
    assert _encryption.decrypt(ciphertext) == plaintext


def test_invalid_key_in_env_is_discarded(monkeypatch) -> None:
    """A malformed Fernet key fails closed instead of storing plaintext."""
    monkeypatch.setenv("REFLEXIO_FERNET_KEYS", "not-a-real-fernet-key")
    _encryption._reset_for_test()
    with pytest.raises(RuntimeError, match="no valid Fernet keys"):
        _encryption.encrypt("v")


def test_finalize_env_cleanup() -> None:
    """Reset module state so other tests aren't affected."""
    os.environ.pop("REFLEXIO_FERNET_KEYS", None)
    _encryption._reset_for_test()
