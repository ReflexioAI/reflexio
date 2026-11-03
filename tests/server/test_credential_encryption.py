"""Malformed keys are optional filtering; unexpected codec failures are not."""

from unittest.mock import patch

import pytest
from cryptography.fernet import Fernet

from reflexio.server import credential_encryption


@pytest.mark.parametrize("require_keys", [False, True])
def test_unexpected_constructor_failure_cannot_select_plaintext(require_keys):
    key = Fernet.generate_key().decode()
    with (
        patch.object(
            credential_encryption,
            "Fernet",
            side_effect=RuntimeError("constructor unavailable"),
        ),
        pytest.raises(RuntimeError, match="constructor unavailable"),
    ):
        manager = credential_encryption.EncryptManager(key, require_keys=require_keys)
        manager.encrypt("synthetic-credential")


def test_malformed_rotation_entry_keeps_the_valid_key():
    key = Fernet.generate_key()
    manager = credential_encryption.EncryptManager(
        f"invalid-key,{key.decode()}", require_keys=True
    )
    ciphertext = manager.encrypt("synthetic-credential")
    assert ciphertext is not None
    assert Fernet(key).decrypt(ciphertext.encode()) == b"synthetic-credential"
    assert manager.decrypt(ciphertext) == "synthetic-credential"
