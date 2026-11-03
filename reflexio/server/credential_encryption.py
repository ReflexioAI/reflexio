"""Credential encryption and rotation for explicit process keyrings.

Callers supply key material and policy mappings; there are no ambient env reads.
"""

import logging
from collections.abc import Mapping

from cryptography import fernet
from cryptography.fernet import Fernet, MultiFernet

from reflexio.server.env_utils import EnvBoolError, env_bool

logger = logging.getLogger(__name__)


def credential_keys_required(env: Mapping[str, str]) -> bool:
    """Validate required-write policy without echoing unexpected environment data."""
    try:
        return env_bool("FERNET_REQUIRED", default=False, env=env)
    except EnvBoolError:
        raise EnvBoolError("FERNET_REQUIRED must be 'true' or 'false'") from None


# A Fernet token starts with version byte 0x80, which base64url-encodes to this
# prefix. A stored config row is either such a token or plaintext JSON (starting
# with "{"), so the prefix distinguishes the two unambiguously.
_FERNET_TOKEN_PREFIX = "gAAAAA"  # noqa: S105 — a format marker, not a secret


def looks_like_fernet_token(value: str) -> bool:
    """Report whether a stored value has the shape of a Fernet token.

    Used to tell "ciphertext we cannot read" apart from "plaintext that has not
    been re-encrypted yet" — the two cases a failed decrypt otherwise conflates.

    Args:
        value (str): The stored value (ciphertext or plaintext).

    Returns:
        bool: True if the value looks like a Fernet token.
    """
    return value.strip().startswith(_FERNET_TOKEN_PREFIX)


class EncryptManager:
    multi_fernet: MultiFernet | None = None

    def __init__(self, fernet_keys: str, require_keys: bool = False):
        self.require_keys = require_keys
        fernet_key_list: list[bytes] = []
        fernet_keys_split: list[str] = []
        if fernet_keys:
            fernet_keys_split = fernet_keys.split(",")
        for fernet_key in fernet_keys_split:
            fernet_key = fernet_key.strip()
            if not fernet_key:
                continue
            fernet_key_list += [fernet_key.encode("utf-8")]
        fernets = []
        for k in fernet_key_list:
            try:
                fernets += [Fernet(k)]
            except ValueError:  # noqa: S112, PERF203 — malformed rotation entries
                continue
        if fernets:
            self.multi_fernet = MultiFernet(fernets)

    def rotate(self, encrypted_value: str) -> str | None:
        if not self.multi_fernet:
            return encrypted_value
        try:
            return self.multi_fernet.rotate(encrypted_value.encode("utf-8")).decode(
                "utf-8"
            )
        except fernet.InvalidToken:
            # Token is no longer valid, just ignore
            return None
        except Exception:
            logger.exception("Failed to rotate encrypted value")
            return None

    def encrypt(self, value: str) -> str | None:
        if not self.multi_fernet:
            if self.require_keys:
                raise RuntimeError(
                    "FERNET_REQUIRED is set but no valid FERNET_KEYS are "
                    "configured; refusing to store plaintext."
                )
            return value
        try:
            return self.multi_fernet.encrypt(value.encode("utf-8")).decode("utf-8")
        except Exception:
            logger.exception("Failed to encrypt value")
            return None

    def decrypt_allow_plaintext(self, encrypted_value: str) -> tuple[str, bool]:
        """Decrypt, tolerating a plaintext (not-yet-re-encrypted) value.

        Returns ``(value, decrypted_ok)``, which distinguishes three cases:

        - ``(plaintext, True)`` — a keyed manager decrypted a Fernet token.
        - ``(raw, False)`` — the value is plaintext (still not re-encrypted
          during the migration window); the caller may use it and should signal
          "needs re-encryption".
        - ``("", False)`` — the value IS a Fernet token but cannot be decrypted
          (unkeyed manager, or keys rotated away). Returning the raw ciphertext
          here would make callers treat it as plaintext and fail downstream with
          an unrelated error such as a JSON parse failure, so the value is
          dropped and every caller's ``if not decrypted`` guard fires instead.

        Args:
            encrypted_value (str): The stored value (ciphertext or plaintext).

        Returns:
            tuple[str, bool]: ``(value, decrypted_ok)``.
        """
        if not self.multi_fernet:
            if looks_like_fernet_token(encrypted_value):
                return "", False
            return encrypted_value, False
        try:
            decrypted = self.multi_fernet.decrypt(
                encrypted_value.encode("utf-8")
            ).decode("utf-8")
            return decrypted, True
        except fernet.InvalidToken:
            if looks_like_fernet_token(encrypted_value):
                return "", False
            return encrypted_value, False

    def decrypt(self, encrypted_value: str, ttl: int | None = None) -> str | None:
        if not self.multi_fernet:
            return encrypted_value
        try:
            return self.multi_fernet.decrypt(
                encrypted_value.encode("utf-8"), ttl=ttl
            ).decode("utf-8")
        except fernet.InvalidToken:
            # Token is no longer valid, just ignore
            return None
        except Exception:
            logger.exception("Failed to decrypt value")
            return None
