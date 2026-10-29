"""Behaviour tests for ``reflexio status check`` and ``reflexio status whoami``.

Each test runs the real Typer command against a real loopback HTTP server, so
the request path, headers, status handling and rendered output are exercised
end to end. Only the server's responses are scripted.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

_WHOAMI = {
    "success": True,
    "org_id": "org-status-test",
    "storage_type": "sqlite",
    "storage_label": "sqlite:///~/.reflexio/data/reflexio.db",
    "storage_configured": True,
    "message": "Storage resolved from local config.",
}


class _Server:
    """A loopback server whose responses are set per path by the test."""

    def __init__(self) -> None:
        self.routes: dict[str, tuple[int, str]] = {}
        self.requests: list[tuple[str, str | None]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server API
                server.requests.append((self.path, self.headers.get("Authorization")))
                status, body = server.routes.get(self.path, (404, '{"detail":"nf"}'))
                payload = body.encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args: object) -> None:
                return

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def server() -> Iterator[_Server]:
    srv = _Server()
    yield srv
    srv.close()


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REFLEXIO_URL", raising=False)
    monkeypatch.delenv("REFLEXIO_API_KEY", raising=False)


def _closed_port_url() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


class TestStatusCheck:
    def test_healthy_server_human_output_sends_bearer_key(
        self, runner, app, server
    ) -> None:
        server.routes["/health"] = (200, '{"status":"ok"}')
        result = runner.invoke(
            app,
            ["--server-url", server.url, "--api-key", "rflx-secret", "status", "check"],
        )
        assert result.exit_code == 0, result.output
        assert f"Connected to {server.url} (healthy)" in result.stdout
        assert server.requests == [("/health", "Bearer rflx-secret")]

    def test_healthy_server_json_envelope_without_key(
        self, runner, app, server
    ) -> None:
        server.routes["/health"] = (200, "{}")
        result = runner.invoke(
            app, ["--json", "--server-url", server.url, "status", "check"]
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == {
            "ok": True,
            "data": {"status": "healthy", "url": server.url},
        }
        # No key configured: no Authorization header is sent.
        assert server.requests == [("/health", None)]

    def test_unhealthy_status_code_is_reported_and_exits_nonzero(
        self, runner, app, server
    ) -> None:
        server.routes["/health"] = (503, '{"status":"degraded"}')
        result = runner.invoke(app, ["--server-url", server.url, "status", "check"])
        assert result.exit_code == 1
        assert f"Error: Server at {server.url} returned 503" in result.stderr
        assert "healthy" not in result.stdout

    def test_unhealthy_json_envelope_on_stderr(self, runner, app, server) -> None:
        server.routes["/health"] = (500, "{}")
        result = runner.invoke(
            app, ["--json", "--server-url", server.url, "status", "check"]
        )
        assert result.exit_code == 1
        envelope = json.loads(result.stderr)
        assert envelope["ok"] is False
        assert envelope["error"]["type"] == "unhealthy"
        assert envelope["error"]["url"] == server.url
        assert "500" in envelope["error"]["message"]

    def test_unreachable_server_renders_network_hint(self, runner, app) -> None:
        url = _closed_port_url()
        result = runner.invoke(app, ["--server-url", url, "status", "check"])
        assert result.exit_code == 1
        assert f"Error: Cannot reach server at {url}" in result.stderr
        assert "reflexio services start" in result.stderr

    def test_unreachable_server_json_envelope(self, runner, app) -> None:
        url = _closed_port_url()
        result = runner.invoke(app, ["--json", "--server-url", url, "status", "check"])
        assert result.exit_code == 1
        envelope = json.loads(result.stderr)
        assert envelope["ok"] is False
        assert envelope["error"]["type"] == "network"
        assert url in envelope["error"]["message"]
        assert "hint" in envelope["error"]

    def test_url_falls_back_to_environment(
        self, runner, app, server, monkeypatch
    ) -> None:
        server.routes["/health"] = (200, "{}")
        monkeypatch.setenv("REFLEXIO_URL", server.url)
        result = runner.invoke(app, ["status", "check"])
        assert result.exit_code == 0, result.output
        assert f"Connected to {server.url} (healthy)" in result.stdout


class TestStatusWhoami:
    def test_human_summary_masks_key_and_shows_storage(
        self, runner, app, server
    ) -> None:
        server.routes["/api/whoami"] = (200, json.dumps(_WHOAMI))
        key = "rflx-abcdefghijklmnop"
        result = runner.invoke(
            app,
            ["--server-url", server.url, "--api-key", key, "status", "whoami"],
        )
        assert result.exit_code == 0, result.output
        out = result.stdout
        assert "org-status-test" in out
        assert "sqlite:///~/.reflexio/data/reflexio.db" in out
        assert "[configured]" in out
        assert "Storage resolved from local config." in out
        assert "rflx-****mnop" in out
        assert key not in out
        assert server.requests == [("/api/whoami", f"Bearer {key}")]

    def test_json_mode_renders_response_envelope(self, runner, app, server) -> None:
        server.routes["/api/whoami"] = (200, json.dumps(_WHOAMI))
        result = runner.invoke(
            app, ["--json", "--server-url", server.url, "status", "whoami"]
        )
        assert result.exit_code == 0, result.output
        envelope = json.loads(result.stdout)
        assert envelope["ok"] is True
        assert envelope["data"]["org_id"] == "org-status-test"
        assert envelope["data"]["storage_configured"] is True

    def test_missing_endpoint_explains_old_server_version(
        self, runner, app, server
    ) -> None:
        # No /api/whoami route: the server answers 404.
        result = runner.invoke(app, ["--server-url", server.url, "status", "whoami"])
        assert result.exit_code == 1
        assert "/api/whoami returned 404" in result.stderr
        assert "predates '/api/whoami'" in result.stderr

    def test_missing_endpoint_json_envelope(self, runner, app, server) -> None:
        result = runner.invoke(
            app, ["--json", "--server-url", server.url, "status", "whoami"]
        )
        assert result.exit_code == 1
        envelope = json.loads(result.stderr)
        assert envelope["error"]["type"] == "api"
        assert "404" in envelope["error"]["message"]

    def test_other_http_errors_are_classified_by_the_shared_handler(
        self, runner, app, server
    ) -> None:
        server.routes["/api/whoami"] = (401, '{"detail":"Invalid API key"}')
        result = runner.invoke(app, ["--server-url", server.url, "status", "whoami"])
        assert result.exit_code != 0
        assert "predates" not in result.stderr
        assert "Error" in result.stderr

    def test_unreachable_server_renders_network_hint(self, runner, app) -> None:
        url = _closed_port_url()
        result = runner.invoke(app, ["--server-url", url, "status", "whoami"])
        assert result.exit_code == 1
        assert "Failed to reach" in result.stderr
        assert "/api/whoami" in result.stderr
        assert "REFLEXIO_URL" in result.stderr
