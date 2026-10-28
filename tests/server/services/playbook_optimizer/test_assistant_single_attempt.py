"""A failed assistant response never authorizes replay of its side effects."""

import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from reflexio.server.services.playbook_optimizer.assistant_webhook import (
    LocalScriptAssistant,
    LocalScriptFailedError,
    WebhookAssistant,
    WebhookFailedError,
)


@pytest.mark.parametrize(
    "status,body",
    [
        (200, b"not-json"),
        (200, b"[]"),
        (200, b'{"text":"missing"}'),
        (503, b"private-response"),
        (307, b""),
    ],
)
def test_http_dispatches_once_after_side_effect(status, body):
    paths = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            paths.append(self.path)
            self.send_response(status if self.path == "/invoke" else 200)
            if status == 307 and self.path == "/invoke":
                self.send_header("Location", "/redirected")
            self.end_headers()
            self.wfile.write(
                body if self.path == "/invoke" else b'{"content":"replayed"}'
            )

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    assistant = WebhookAssistant(
        url=f"http://127.0.0.1:{server.server_port}/invoke",
        auth_header=None,
        timeout_s=2,
    )
    try:
        with pytest.raises(WebhookFailedError) as error:
            assistant([], [])
        assert "private-response" not in str(error.value)
        assert paths == ["/invoke"]
    finally:
        server.shutdown()
        worker.join(timeout=3)
        server.server_close()


@pytest.mark.parametrize(
    "ending",
    [
        "print('invalid')",
        "print('[]')",
        "print('{}')",
        "raise SystemExit(3)",
        "import time; time.sleep(5)",
    ],
)
def test_script_dispatches_once_after_side_effect(tmp_path, ending):
    effect = tmp_path / "effects"
    script = tmp_path / "assistant.py"
    script.write_text(
        "from pathlib import Path\np=Path("
        + repr(str(effect))
        + ")\nwith p.open('a') as f: f.write('executed\\n')\n"
        + ending
    )
    assistant = LocalScriptAssistant(
        script_path=sys.executable, script_args=[str(script)], timeout_s=1
    )
    with pytest.raises(LocalScriptFailedError):
        assistant([], [])
    assert effect.read_text().splitlines() == ["executed"]
