"""The installed LiteLLM maps provider HTTP failures the way the classifier expects.

``TRANSIENT_ERROR_NAMES`` and ``TRANSIENT_UPSTREAM_STATUS_CODES`` (``failure.py``) match
LiteLLM exception class names and status codes, and the provider circuit
breaker trusts that classification. Neither is checked against the installed
LiteLLM elsewhere: the unit tests build the exceptions by hand. A LiteLLM
upgrade that renames a class or remaps a status (for example a 529 overload, or
a context-window 400) would silently turn outages into charged failures, or the
reverse. These tests send real requests through LiteLLM to a loopback server.
"""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from litellm.exceptions import ContextWindowExceededError

from reflexio.server.llm.failure import (
    OutageWait,
    Transient,
    classify_llm_failure,
)
from reflexio.test_support.llm_mock import unpatched_litellm


def _unavailable(error: BaseException) -> bool:
    return isinstance(classify_llm_failure(error), OutageWait | Transient)


_ROUTES = ("minimax/MiniMax-M3", "openai/gpt-4o-mini", "zai/glm-5.3")

# status -> (provider unavailable?) for an OpenAI-shaped error body.
_STATUS_CASES = {
    400: False,
    401: False,
    404: False,
    408: True,
    429: True,
    500: True,
    502: True,
    503: True,
    504: True,
    529: True,
}

_CONTEXT_OVERFLOW = (
    "This model's maximum context length is 8192 tokens. However, your "
    "messages resulted in 9000 tokens."
)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        pass

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        scenario = self.path.split("/s/", 1)[1].split("/", 1)[0]
        if scenario == "overflow":
            status = 400
            message, code = _CONTEXT_OVERFLOW, "context_length_exceeded"
        else:
            status = int(scenario)
            message, code = f"upstream status {status}", None
        body = json.dumps(
            {"error": {"message": message, "type": "api_error", "code": code}}
        ).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def stub_base() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _failure(model: str, api_base: str) -> Exception:
    import litellm

    with unpatched_litellm(), pytest.raises(Exception) as caught:
        litellm.completion(
            model=model,
            api_base=api_base,
            api_key="loopback-test-key",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8,
            timeout=10,
            num_retries=0,
            max_retries=0,
        )
    return caught.value


@pytest.mark.parametrize("model", _ROUTES)
@pytest.mark.parametrize(("status", "unavailable"), sorted(_STATUS_CASES.items()))
def test_http_status_classification(
    stub_base: str, model: str, status: int, unavailable: bool
) -> None:
    error = _failure(model, f"{stub_base}/s/{status}/v1")
    assert _unavailable(error) is unavailable, (
        f"{model} HTTP {status} raised {type(error).__name__} "
        f"(status_code={getattr(error, 'status_code', None)})"
    )


# The MiniMax route is absent on purpose: LiteLLM's MiniMax adapter raises a
# plain BadRequestError for this OpenAI-style overflow body (1.103.1 and
# 1.105.7 alike), so it is not a typed overflow there.
@pytest.mark.parametrize("model", ["openai/gpt-4o-mini", "zai/glm-5.3"])
def test_context_overflow_is_typed_and_not_an_outage(
    stub_base: str, model: str
) -> None:
    error = _failure(model, f"{stub_base}/s/overflow/v1")
    assert isinstance(error, ContextWindowExceededError)
    assert not _unavailable(error)


@pytest.mark.parametrize("model", _ROUTES)
def test_refused_connection_is_an_outage(model: str) -> None:
    # Port 1 on loopback is never listening: the connection is refused.
    error = _failure(model, "http://127.0.0.1:1/v1")
    assert _unavailable(error), type(error).__name__
