from __future__ import annotations

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock, patch

import httpx
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


def _patch_async_client(monkeypatch, handler, captured: dict) -> None:
    real_async_client = httpx.AsyncClient

    def fake_async_client(**kwargs):
        captured["client_kwargs"] = kwargs
        return real_async_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake_async_client)


@pytest.mark.asyncio
async def test_make_async_request_applies_timeout_and_disables_redirects(
    monkeypatch,
) -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(200, json={"success": True})

    _patch_async_client(monkeypatch, handler, captured)
    client = ReflexioClient(
        api_key="test-key", url_endpoint="http://localhost:8000", timeout=7.5
    )

    response = await client._make_async_request(  # noqa: SLF001
        "POST", "/api/whoami", headers={"X-Extra": "1"}, json={"a": 1}
    )

    assert response == {"success": True}
    assert captured["client_kwargs"] == {
        "timeout": 7.5,
        "follow_redirects": False,
        "trust_env": False,
    }
    request = captured["request"]
    assert request.method == "POST"
    assert str(request.url) == "http://localhost:8000/api/whoami"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert request.headers["X-Extra"] == "1"
    assert request.content == b'{"a":1}'


@pytest.mark.asyncio
async def test_async_transport_does_not_inherit_proxy_or_certificate_environment(
    monkeypatch,
    tmp_path,
) -> None:
    """Keep the former async transport's explicit-endpoint routing policy."""
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing-ca.pem"))
    auth_seen: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            auth_seen.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"success": true}')

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = ReflexioClient(
            api_key="test-key", url_endpoint=f"http://127.0.0.1:{server.server_port}"
        )
        assert await client._make_async_request("GET", "/api/whoami") == {
            "success": True
        }
        assert auth_seen == ["Bearer test-key"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


@pytest.mark.asyncio
async def test_make_async_request_enforces_total_deadline(monkeypatch) -> None:
    async def slow_handler(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, json={"success": True})

    _patch_async_client(monkeypatch, slow_handler, {})
    client = ReflexioClient(
        api_key="test-key", url_endpoint="http://localhost:8000", timeout=0.05
    )

    with pytest.raises(TimeoutError):
        await client._make_async_request("GET", "/api/whoami")  # noqa: SLF001


@pytest.mark.asyncio
async def test_make_async_request_raises_http_status_error(monkeypatch) -> None:
    _patch_async_client(
        monkeypatch, lambda _r: httpx.Response(500, json={"detail": "boom"}), {}
    )
    client = ReflexioClient(api_key="test-key", url_endpoint="http://localhost:8000")

    with pytest.raises(httpx.HTTPStatusError):
        await client._make_async_request("GET", "/api/whoami")  # noqa: SLF001
