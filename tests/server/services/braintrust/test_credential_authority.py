"""One process keyring protects connector credentials without a fallback alias."""

from unittest.mock import Mock, patch

import pytest
from cryptography.fernet import Fernet, InvalidToken

from reflexio.server.services.braintrust import _encryption


def test_canonical_ring_encrypts_and_decrypts(monkeypatch):
    key = Fernet.generate_key()
    monkeypatch.setenv("FERNET_KEYS", key.decode())
    encrypted = _encryption.encrypt("synthetic-credential")
    assert encrypted != "synthetic-credential"
    assert Fernet(key).decrypt(encrypted.encode()) == b"synthetic-credential"
    assert _encryption.decrypt(encrypted) == "synthetic-credential"


def test_combined_ring_keeps_historical_connector_ciphertext_readable(monkeypatch):
    current, previous = Fernet.generate_key(), Fernet.generate_key()
    historical = Fernet(previous).encrypt(b"synthetic-credential").decode()
    monkeypatch.setenv("FERNET_KEYS", f"{current.decode()},{previous.decode()}")
    assert _encryption.decrypt(historical) == "synthetic-credential"
    rewritten = _encryption.encrypt("synthetic-credential")
    assert Fernet(current).decrypt(rewritten.encode()) == b"synthetic-credential"


@pytest.mark.parametrize("obsolete", ["", "synthetic-old-value"])
def test_obsolete_assignment_is_rejected_even_alongside_canonical(
    monkeypatch, obsolete
):
    monkeypatch.setenv("REFLEXIO_FERNET_KEYS", obsolete)
    monkeypatch.setenv("FERNET_KEYS", Fernet.generate_key().decode())
    with pytest.raises(ValueError, match="Removed environment names") as error:
        _encryption.encrypt("synthetic-credential")
    assert "synthetic-old-value" not in str(error.value)
    assert "synthetic-credential" not in str(error.value)


def test_required_policy_refuses_plaintext(monkeypatch):
    monkeypatch.setenv("FERNET_REQUIRED", "true")
    with pytest.raises(RuntimeError, match="refusing to store plaintext"):
        _encryption.encrypt("synthetic-credential")


@pytest.mark.parametrize("value", ["ture", "private-policy-value"])
def test_invalid_required_policy_fails_before_plaintext_write(monkeypatch, value):
    monkeypatch.setenv("FERNET_REQUIRED", value)
    with pytest.raises(ValueError, match="FERNET_REQUIRED") as error:
        _encryption.encrypt("synthetic-credential")
    assert value not in str(error.value)


def test_invalid_canonical_key_fails_closed_without_echo(monkeypatch):
    monkeypatch.setenv("FERNET_KEYS", "synthetic-invalid-key")
    with pytest.raises(RuntimeError, match="no valid Fernet keys") as error:
        _encryption.encrypt("synthetic-credential")
    assert "synthetic-invalid-key" not in str(error.value)


def test_unkeyed_reader_cannot_forward_ciphertext_as_a_credential():
    ciphertext = Fernet(Fernet.generate_key()).encrypt(b"synthetic-credential").decode()
    with pytest.raises(InvalidToken):
        _encryption.decrypt(ciphertext)


def test_wrong_key_refuses_ciphertext(monkeypatch):
    ciphertext = Fernet(Fernet.generate_key()).encrypt(b"synthetic-credential").decode()
    monkeypatch.setenv("FERNET_KEYS", Fernet.generate_key().decode())
    with pytest.raises(InvalidToken):
        _encryption.decrypt(ciphertext)


def test_unkeyed_local_plaintext_remains_usable():
    assert _encryption.encrypt("synthetic-credential") == "synthetic-credential"
    assert _encryption.decrypt("synthetic-credential") == "synthetic-credential"


def test_sqlite_saved_ciphertext_survives_reopen_and_keyring_rotation(
    tmp_path, monkeypatch
):
    from reflexio.models.api_schema.braintrust_schema import SelectProjectsRequest
    from reflexio.server.services.braintrust.service import BraintrustConnectorService
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    previous, current = Fernet.generate_key(), Fernet.generate_key()
    monkeypatch.setenv("FERNET_KEYS", previous.decode())
    path = str(tmp_path / "credentials.sqlite")
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        storage = SQLiteStorage(org_id="credential-probe", db_path=path)
    service = BraintrustConnectorService(storage=storage, org_id="credential-probe")
    try:
        assert service.select_projects(
            SelectProjectsRequest(
                api_key="synthetic-credential", workspace_id="workspace", project_ids=[]
            )
        ).success
        saved = storage.get_braintrust_connection("credential-probe")
        assert saved is not None
        assert saved.api_key_enc != "synthetic-credential"
        assert (
            Fernet(previous).decrypt(saved.api_key_enc.encode())
            == b"synthetic-credential"
        )
    finally:
        storage.conn.close()

    monkeypatch.setenv("FERNET_KEYS", f"{current.decode()},{previous.decode()}")
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        reopened = SQLiteStorage(org_id="credential-probe", db_path=path)
    factory = Mock()
    try:
        service = BraintrustConnectorService(
            storage=reopened, org_id="credential-probe", client_factory=factory
        )
        assert service.sync_once().success
        factory.assert_called_once_with("synthetic-credential")
        historical = reopened.get_braintrust_connection("credential-probe")
        assert historical is not None
        assert historical.api_key_enc == saved.api_key_enc
        assert historical.workspace_id == "workspace"
        # A missing historical key must stop before constructing any HTTP client.
        monkeypatch.setenv("FERNET_KEYS", current.decode())
        factory.reset_mock()
        response = service.sync_once()
        assert response.success is False
        assert "Failed to decrypt" in response.msg
        factory.assert_not_called()
    finally:
        reopened.conn.close()
