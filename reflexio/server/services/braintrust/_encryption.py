"""Connector policy over the process credential-encryption implementation."""

import os

from cryptography.fernet import InvalidToken

from reflexio.removed_env import reject_removed_env
from reflexio.server.credential_encryption import (
    EncryptManager,
    credential_keys_required,
    looks_like_fernet_token,
)
from reflexio.server.env_utils import env_str


def _manager() -> EncryptManager:
    reject_removed_env(os.environ, canonical="FERNET_KEYS")
    keys = env_str("FERNET_KEYS")
    manager = EncryptManager(keys, require_keys=credential_keys_required(os.environ))
    if keys and manager.multi_fernet is None:
        raise RuntimeError("FERNET_KEYS is set but contains no valid Fernet keys")
    return manager


def encrypt(value: str) -> str:
    """Use canonical encryption policy; optional unkeyed local stores stay usable."""
    encrypted = _manager().encrypt(value)
    if encrypted is None:
        raise RuntimeError("Unable to encrypt Braintrust credential")
    return encrypted


def decrypt(value: str) -> str:
    """Refuse unreadable ciphertext before it can become an outbound credential."""
    manager = _manager()
    if manager.multi_fernet is None and looks_like_fernet_token(value):
        raise InvalidToken
    decrypted = manager.decrypt(value)
    if decrypted is None:
        raise InvalidToken
    return decrypted
