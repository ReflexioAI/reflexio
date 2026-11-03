"""Connector tests never inherit a developer's credential policy or keyring."""

import pytest


@pytest.fixture(autouse=True)
def isolated_credential_environment(monkeypatch):
    for name in ("FERNET_KEYS", "FERNET_REQUIRED", "REFLEXIO_FERNET_KEYS"):
        monkeypatch.delenv(name, raising=False)
