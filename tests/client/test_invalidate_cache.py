"""Unit tests for ``ReflexioClient.invalidate_cache``.

The method wraps ``POST /api/admin/cache/invalidate`` and either
forwards an explicit ``org_id`` token or sends an empty body so the
server resolves the org from the auth header.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


@patch("reflexio.client.client.requests.Session")
def test_invalidate_cache_without_org_id_sends_empty_body(mock_session_class) -> None:
    """No org_id argument → empty JSON body so the server uses auth-resolved org."""
    from reflexio.client import ReflexioClient

    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    mock_response = MagicMock()
    mock_response.json.return_value = {"invalidated": True, "org_id": "test-org"}
    mock_response.content = b'{"invalidated": true}'
    mock_response.headers = {"Content-Type": "application/json"}
    mock_session.request.return_value = mock_response

    client = ReflexioClient(api_key="test_key", url_endpoint="http://localhost:8000")
    result = client.invalidate_cache()

    assert result == {"invalidated": True, "org_id": "test-org"}
    assert mock_session.request.call_count == 1
    args, kwargs = mock_session.request.call_args
    assert args[0] == "POST"
    assert args[1].endswith("/api/admin/cache/invalidate")
    # An empty dict goes on the wire, not None — keeps the request a
    # well-formed JSON POST that pydantic accepts on the server.
    assert kwargs["json"] == {}


@patch("reflexio.client.client.requests.Session")
def test_invalidate_cache_with_org_id_sends_token(mock_session_class) -> None:
    """An explicit org_id is forwarded as a verification token in the body."""
    from reflexio.client import ReflexioClient

    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    mock_response = MagicMock()
    mock_response.json.return_value = {"invalidated": False, "org_id": "test-org"}
    mock_response.content = b'{"invalidated": false}'
    mock_response.headers = {"Content-Type": "application/json"}
    mock_session.request.return_value = mock_response

    client = ReflexioClient(api_key="test_key", url_endpoint="http://localhost:8000")
    result = client.invalidate_cache(org_id="test-org")

    assert result["invalidated"] is False
    assert mock_session.request.call_args.kwargs["json"] == {"org_id": "test-org"}
