from __future__ import annotations

from unittest.mock import MagicMock, patch

from reflexio.client import ReflexioClient


def _json_response(payload: dict[str, object]) -> MagicMock:
    response = MagicMock()
    response.json.return_value = payload
    response.content = b'{"success": true}'
    response.headers = {"Content-Type": "application/json"}
    return response


@patch("reflexio.client.client.requests.Session")
def test_make_request_does_not_persist_per_call_headers(mock_session_class) -> None:
    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session_class.return_value = mock_session
    mock_session.request.return_value = _json_response({"success": True})

    client = ReflexioClient(api_key="normal-key", url_endpoint="http://localhost:8000")
    client.session.headers.update({"User-Agent": "reflexio-test"})

    client._make_request(  # noqa: SLF001 - client header regression guard
        "POST",
        "/api/admin/offline_tuner/run",
        headers={"Authorization": "Bearer admin-key"},
        json={},
    )
    client._make_request("GET", "/api/whoami")  # noqa: SLF001

    first_call = mock_session.request.call_args_list[0]
    second_call = mock_session.request.call_args_list[1]
    assert first_call.kwargs["headers"]["Authorization"] == "Bearer admin-key"
    assert second_call.kwargs["headers"]["Authorization"] == "Bearer normal-key"
    assert client.session.headers == {"User-Agent": "reflexio-test"}
