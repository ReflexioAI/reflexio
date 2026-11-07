"""Transport behaviour of the CLI commands that call the server directly.

``reflexio api``, ``reflexio status check`` and ``reflexio doctor check`` issue
their own httpx calls instead of going through ``ReflexioClient``. These tests
run each command against a scripted loopback server so redirect following,
request bodies and connection-failure mapping are exercised over real HTTP.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class _Server:
    """Loopback server: ``/moved`` and ``/health`` redirect, the rest echo."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, status: int, payload: dict, location: str = "") -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                if location:
                    self.send_header("Location", location)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _handle(self) -> None:
                server.seen.append((self.command, self.path))
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length).decode() if length else ""
                if self.path in ("/moved", "/health"):
                    self._reply(302, {}, location="/health-live")
                elif self.path == "/health-live":
                    self._reply(200, {"status": "healthy"})
                else:
                    self._reply(
                        200,
                        {
                            "method": self.command,
                            "body": json.loads(body) if body else None,
                        },
                    )

            do_GET = do_POST = do_DELETE = _handle  # noqa: N815 - http.server API

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


def test_api_delete_sends_its_json_body(runner, app, server) -> None:
    result = runner.invoke(
        app,
        ["--json", "--server-url", server.url, "api", "--data", '{"ids": ["a"]}']
        + ["DELETE", "/api/x"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["data"] == {
        "method": "DELETE",
        "body": {"ids": ["a"]},
    }


def test_api_follows_redirects(runner, app, server) -> None:
    result = runner.invoke(
        app, ["--json", "--server-url", server.url, "api", "GET", "/moved"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["data"] == {"status": "healthy"}
    assert server.seen == [("GET", "/moved"), ("GET", "/health-live")]


def test_api_unreachable_server_is_a_network_error(runner, app) -> None:
    url = _closed_port_url()
    result = runner.invoke(app, ["--server-url", url, "api", "GET", "/api/x"])
    assert result.exit_code == 4
    assert "Connection failed" in result.stderr


def test_status_check_follows_a_health_redirect(runner, app, server) -> None:
    result = runner.invoke(app, ["--server-url", server.url, "status", "check"])
    assert result.exit_code == 0, result.output
    assert server.seen == [("GET", "/health"), ("GET", "/health-live")]


def _server_health(result_stdout: str) -> dict:
    checks = json.loads(result_stdout)["data"]
    return next(check for check in checks if check["name"] == "server_health")


def test_doctor_follows_a_health_redirect(runner, app, server) -> None:
    result = runner.invoke(
        app, ["--json", "--server-url", server.url, "doctor", "check"]
    )
    health = _server_health(result.stdout)
    assert health["status"] == "pass", health
    assert server.seen == [("GET", "/health"), ("GET", "/health-live")]


def test_doctor_reports_an_unreachable_server(runner, app) -> None:
    url = _closed_port_url()
    result = runner.invoke(app, ["--json", "--server-url", url, "doctor", "check"])
    health = _server_health(result.stdout)
    assert health["status"] == "fail"
    assert health["message"] == f"Cannot connect to {url}"
