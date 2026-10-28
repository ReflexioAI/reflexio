from unittest.mock import patch

import pytest

from reflexio.client.client import ReflexioClient


def test_client_reads_reflexio_url(monkeypatch):
    monkeypatch.delenv("REFLEXIO_API_URL", raising=False)
    monkeypatch.setenv("REFLEXIO_URL", "http://example.test:9999")
    c = ReflexioClient()
    assert "example.test:9999" in c.base_url


def test_client_ignores_legacy_reflexio_api_url(monkeypatch):
    """The legacy ``REFLEXIO_API_URL`` name is no longer honored.

    Only ``REFLEXIO_URL`` is read; with it unset the client must fall back to
    ``BACKEND_URL`` and never pick up the legacy value.
    """
    from reflexio.client.client import BACKEND_URL

    monkeypatch.delenv("REFLEXIO_URL", raising=False)
    monkeypatch.setenv("REFLEXIO_API_URL", "http://legacy.test:1234")
    c = ReflexioClient()
    assert "legacy.test:1234" not in c.base_url
    assert c.base_url == BACKEND_URL


@pytest.mark.parametrize("mode", ["true", " TRUE ", "unknown"])
@pytest.mark.parametrize("target", ["", "   "])
def test_implicit_legacy_target_refused_before_transport(monkeypatch, mode, target):
    monkeypatch.setenv("IS_TEST_ENV", mode)
    monkeypatch.setenv("REFLEXIO_URL", target)
    with patch("reflexio.client.client.requests.Session") as transport:
        with pytest.raises(ValueError, match="REFLEXIO_URL") as error:
            ReflexioClient()
        transport.assert_not_called()
    assert mode.strip() not in str(error.value)


@pytest.mark.parametrize("mode", ["", "false", " FALSE "])
def test_neutral_mode_uses_managed_default(monkeypatch, mode):
    monkeypatch.setenv("IS_TEST_ENV", mode)
    monkeypatch.setenv("REFLEXIO_URL", "   ")
    assert ReflexioClient().base_url == "https://www.reflexio.ai/"


def test_explicit_target_wins_without_mode_routing(monkeypatch):
    monkeypatch.setenv("IS_TEST_ENV", "true")
    monkeypatch.setenv("REFLEXIO_URL", " http://env.test:9000 ")
    assert ReflexioClient().base_url == "http://env.test:9000"
    assert (
        ReflexioClient(url_endpoint=" http://argument.test:9001 ").base_url
        == "http://argument.test:9001"
    )
    assert ReflexioClient(url_endpoint="   ").base_url == "http://env.test:9000"


def test_explicit_test_target_reaches_only_owned_loopback_server(monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    paths = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            paths.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setenv("IS_TEST_ENV", "true")
    monkeypatch.setenv("REFLEXIO_URL", f"http://127.0.0.1:{server.server_port}")
    client = ReflexioClient()
    try:
        assert client._make_request("GET", "/health") == {"status": "ok"}
        assert paths == ["/health"]
    finally:
        client.session.close()
        server.shutdown()
        worker.join(timeout=3)
        server.server_close()


def test_client_endpoint_resolution_does_not_import_server():
    import subprocess
    import sys

    program = """
import os, sys
class RejectServer:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "reflexio.server" or fullname.startswith("reflexio.server."):
            raise AssertionError("Client imported server: " + fullname)
sys.meta_path.insert(0, RejectServer())
os.environ["IS_TEST_ENV"] = "true"
os.environ["REFLEXIO_URL"] = "http://explicit.test"
from reflexio import ReflexioClient
assert ReflexioClient().base_url == "http://explicit.test"
"""
    result = subprocess.run(  # noqa: S603 - fixed owned Python program
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
