"""Unit tests for ``ReflexioClient`` config round-tripping."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from reflexio.client import ReflexioClient
from reflexio.models.config_schema import Config, StorageConfigSQLite


def _json_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.json.return_value = payload
    response.content = b"{}"
    response.headers = {"Content-Type": "application/json"}
    return response


@patch("reflexio.client.client.requests.Session")
def test_client_get_config_accepts_unknown_overlay(mock_session_class) -> None:
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    payload = Config(
        storage_config=StorageConfigSQLite(db_path="/tmp/test.db")
    ).model_dump()
    payload["x_extension_config"] = {
        "enabled": True,
        "version": "extension-v1",
    }
    mock_session.request.return_value = _json_response(payload)

    client = ReflexioClient(api_key="test_key", url_endpoint="http://localhost:8000")
    result = client.get_config()

    assert isinstance(result, Config)
    assert result.storage_config == StorageConfigSQLite(db_path="/tmp/test.db")
    assert result.model_dump()["x_extension_config"] == payload["x_extension_config"]


@patch("reflexio.client.client.requests.Session")
def test_client_set_config_preserves_unknown_overlay(mock_session_class) -> None:
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    get_payload = Config(
        storage_config=StorageConfigSQLite(db_path="/tmp/test.db")
    ).model_dump()
    get_payload["x_extension_config"] = {
        "enabled": True,
        "version": "extension-v1",
    }
    mock_session.request.side_effect = [
        _json_response(get_payload),
        _json_response({"success": True, "msg": "Configuration set successfully"}),
    ]

    client = ReflexioClient(api_key="test_key", url_endpoint="http://localhost:8000")
    config = client.get_config()
    response = client.set_config(config)

    assert response == {"success": True, "msg": "Configuration set successfully"}
    assert mock_session.request.call_count == 2
    args, kwargs = mock_session.request.call_args
    assert args[0] == "POST"
    assert args[1].endswith("/api/set_config")
    assert kwargs["json"]["x_extension_config"] == get_payload["x_extension_config"]
