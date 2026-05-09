"""Unit tests for ``ReflexioClient.update_config``.

The method ships partial dicts straight to the server without
constructing a ``Config`` first — that's the whole point. We assert
on the wire payload rather than building a real session.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from reflexio.client import ReflexioClient
from reflexio.models.config_schema import Config


@patch("reflexio.client.client.requests.Session")
def test_update_config_skips_client_side_validation(mock_session_class) -> None:
    """A partial dict missing required Config fields must reach the server unchanged.

    ``Config`` requires ``storage_config``; if the client did
    ``Config(**partial)`` like ``set_config`` does, this call would
    blow up with a ``ValidationError`` before any HTTP request. We
    want the dict to flow through the wire untouched so the server
    can do the merge.
    """
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "success": True,
        "msg": "Configuration set successfully",
    }
    mock_response.content = b'{"success": true}'
    mock_response.headers = {"Content-Type": "application/json"}
    mock_session.request.return_value = mock_response

    client = ReflexioClient(api_key="test_key", url_endpoint="http://localhost:8000")
    partial = {"extraction_backend": "classic"}

    result = client.update_config(partial)

    assert result == {
        "success": True,
        "msg": "Configuration set successfully",
    }
    # Exactly one HTTP call to /api/update_config with the partial dict
    # passed through verbatim — no Config wrapping, no field stripping.
    assert mock_session.request.call_count == 1
    args, kwargs = mock_session.request.call_args
    assert args[0] == "POST"
    assert args[1].endswith("/api/update_config")
    assert kwargs["json"] == partial
    # Ensure the dict object itself is not mutated.
    assert partial == {"extraction_backend": "classic"}


@patch("reflexio.client.client.requests.Session")
def test_update_config_passes_nested_object_through(mock_session_class) -> None:
    """Nested top-level dicts (``llm_config``, ``storage_config``) are not deep-merged."""
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    mock_response = MagicMock()
    mock_response.json.return_value = {"success": True, "msg": "ok"}
    mock_response.content = b'{"success": true}'
    mock_response.headers = {"Content-Type": "application/json"}
    mock_session.request.return_value = mock_response

    client = ReflexioClient(api_key="test_key", url_endpoint="http://localhost:8000")
    partial = {"llm_config": {"embedding_model_name": "local/minilm-l6-v2"}}

    client.update_config(partial)

    kwargs = mock_session.request.call_args.kwargs
    assert kwargs["json"] == partial


def test_update_config_rejects_non_dict() -> None:
    """update_config requires a dict — list / Config / None are rejected.

    Catching this client-side gives a clearer error than a server 422
    on a non-mapping body.
    """
    client = ReflexioClient(api_key="test_key", url_endpoint="http://localhost:8000")

    with pytest.raises(TypeError, match="dict"):
        client.update_config([1, 2, 3])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="dict"):
        # A Config model instance is also wrong here — set_config takes
        # those, update_config does not.
        full_config = Config.model_construct()
        client.update_config(full_config)  # type: ignore[arg-type]
