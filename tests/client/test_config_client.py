"""Unit tests for ``ReflexioClient`` config round-tripping."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from reflexio.client import ConfigResponse, OfflineTunerConfigResponse, ReflexioClient
from reflexio.models.config_schema import Config, LLMConfig, StorageConfigSQLite


def _json_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.json.return_value = payload
    response.content = b"{}"
    response.headers = {"Content-Type": "application/json"}
    return response


@patch("reflexio.client.client.httpx.Client")
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

    assert isinstance(result, ConfigResponse)
    assert result.storage_config == StorageConfigSQLite(db_path="/tmp/test.db")
    assert result.model_dump()["x_extension_config"] == payload["x_extension_config"]


@patch("reflexio.client.client.httpx.Client")
def test_client_get_config_types_offline_tuner_overlay(mock_session_class) -> None:
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    payload = Config(
        storage_config=StorageConfigSQLite(db_path="/tmp/test.db")
    ).model_dump()
    payload["offline_tuner_config"] = {"enabled": False}
    mock_session.request.return_value = _json_response(payload)

    client = ReflexioClient(api_key="test_key", url_endpoint="http://localhost:8000")
    result = client.get_config()

    assert isinstance(result.offline_tuner_config, OfflineTunerConfigResponse)
    assert result.offline_tuner_config.enabled is False


@patch("reflexio.client.client.httpx.Client")
def test_client_set_config_does_not_send_response_overlay(mock_session_class) -> None:
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    get_payload = Config(
        storage_config=StorageConfigSQLite(db_path="/tmp/test.db")
    ).model_dump()
    get_payload["x_extension_config"] = {
        "enabled": True,
        "version": "extension-v1",
    }
    get_payload["offline_tuner_config"] = {"enabled": False}
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
    assert "x_extension_config" not in kwargs["json"]
    assert "offline_tuner_config" not in kwargs["json"]


@pytest.mark.parametrize("shape", ["raw", "typed", "mixed"])
@pytest.mark.parametrize("explicit", [False, True])
@patch("reflexio.client.client.httpx.Client")
def test_set_config_distinguishes_evaluator_reset_from_omission(
    mock_session_class, shape, explicit
):
    client = ReflexioClient(api_key="test_key", url_endpoint="http://localhost:8000")
    session = mock_session_class.return_value
    session.request.return_value = _json_response({"success": True})
    llm = {"generation_model_name": "openai/gpt-4o"}
    if explicit:
        llm["evaluation_model_name"] = None
    raw = {"storage_config": None, "llm_config": llm}

    if shape == "mixed":
        raw["llm_config"] = LLMConfig.model_validate(llm)
    client.set_config(Config.model_validate(raw) if shape == "typed" else raw)

    sent = session.request.call_args.kwargs["json"]["llm_config"]
    assert ("evaluation_model_name" in sent) is explicit
    if explicit:
        assert sent["evaluation_model_name"] is None
