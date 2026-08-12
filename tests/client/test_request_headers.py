from __future__ import annotations

from unittest.mock import MagicMock, patch

import aiohttp
import pytest

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


@pytest.mark.asyncio
async def test_make_async_request_applies_total_timeout_and_disables_redirects(
    monkeypatch,
) -> None:
    captured = {}

    class FakeResponse:
        status = 200
        headers = {"Content-Type": "application/json"}

        def raise_for_status(self):
            return None

        async def read(self):
            return b'{"success": true}'

        async def json(self, *, content_type=None):
            assert content_type is None
            return {"success": True}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def request(self, method, url, **kwargs):
            captured.update(method=method, url=url, **kwargs)
            return FakeResponse()

    monkeypatch.setattr(aiohttp, "ClientSession", FakeSession)
    client = ReflexioClient(
        api_key="test-key", url_endpoint="http://localhost:8000", timeout=7.5
    )

    response = await client._make_async_request("GET", "/api/whoami")  # noqa: SLF001

    assert response == {"success": True}
    assert captured["allow_redirects"] is False
    assert isinstance(captured["timeout"], aiohttp.ClientTimeout)
    assert captured["timeout"].total == 7.5
