"""Readiness probe: existing compute deadlines must own reranker waits/results."""

import json
import os

import httpx
import pytest

from reflexio.server.llm import call_deadline
from reflexio.server.llm.providers import embedding_service_provider as provider
from reflexio.server.llm.rerank import cross_encoder_reranker as reranker


def _install(monkeypatch, transport):
    url = "http://owned-inference.test"
    monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_URL", url)
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "internal_service")
    monkeypatch.setenv("REFLEXIO_RERANK_ENABLED", "true")
    monkeypatch.setenv("REFLEXIO_RERANK_SERVICE_TIMEOUT_MS", "5000")
    monkeypatch.setattr(
        provider,
        "_configured_model_cache",
        {
            (os.getpid(), url): provider.InferenceServiceCapabilities(
                configured_model="local/owned-embedding",
                configured_reranker_model="owned-reranker",
                reranker_enabled=True,
                reranker_ready=True,
            )
        },
    )
    client = httpx.Client(
        transport=httpx.MockTransport(transport),
        event_hooks={"request": [provider.http_request_deadline]},
    )
    monkeypatch.setattr(provider, "_http_client", lambda: client)
    return client


def test_reranking_clips_transport_and_worker_budget_to_compute_deadline(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(call_deadline, "monotonic", lambda: now[0])
    requests = []

    def transport(request):
        requests.append(request)
        return httpx.Response(200, json={"data": [{"index": 0, "score": 1.0}]})

    with _install(monkeypatch, transport), call_deadline.model_deadline(100.25):
        assert reranker.score_pairs("query", ["doc"]) == [1.0]
    assert len(requests) == 1
    assert requests[0].extensions["timeout"]["read"] <= 0.25
    assert json.loads(requests[0].content)["timeout_ms"] <= 250


@pytest.mark.parametrize("status", [200, 503])
def test_reranking_refuses_result_after_compute_deadline(monkeypatch, status):
    now = [100.0]
    monkeypatch.setattr(call_deadline, "monotonic", lambda: now[0])

    def transport(request):
        now[0] = 101.0
        return httpx.Response(status, json={"data": [{"index": 0, "score": 1.0}]})

    with (
        _install(monkeypatch, transport),
        call_deadline.model_deadline(100.25),
        pytest.raises(call_deadline.CallDeadlineExceededError),
    ):
        reranker.score_pairs("query", ["doc"])


def test_expired_compute_does_not_dispatch_reranking(monkeypatch):
    monkeypatch.setattr(call_deadline, "monotonic", lambda: 100.0)

    def transport(request):
        pytest.fail("expired compute dispatched a reranking request")

    with (
        _install(monkeypatch, transport),
        call_deadline.model_deadline(99.0),
        pytest.raises(call_deadline.CallDeadlineExceededError),
    ):
        reranker.score_pairs("query", ["doc"])


def test_unavailable_reranker_remains_fail_open(monkeypatch):
    monkeypatch.setattr(call_deadline, "monotonic", lambda: 100.0)

    def transport(request):
        return httpx.Response(503)

    with (
        _install(monkeypatch, transport),
        call_deadline.model_deadline(101.0),
        pytest.raises(reranker.CrossEncoderUnavailableError),
    ):
        reranker.score_pairs("query", ["doc"])


def test_reranking_sends_remaining_compute_budget_over_loopback_http(monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from time import monotonic

    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            body = json.dumps({"data": [{"index": 0, "score": 1.0}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_URL", url)
        monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "internal_service")
        monkeypatch.setenv("REFLEXIO_RERANK_ENABLED", "true")
        monkeypatch.setenv("REFLEXIO_RERANK_SERVICE_TIMEOUT_MS", "5000")
        monkeypatch.setattr(
            provider,
            "_configured_model_cache",
            {
                (os.getpid(), url): provider.InferenceServiceCapabilities(
                    configured_model="local/owned-embedding",
                    configured_reranker_model="owned-reranker",
                    reranker_enabled=True,
                    reranker_ready=True,
                )
            },
        )
        with httpx.Client(
            event_hooks={"request": [provider.http_request_deadline]}
        ) as client:
            monkeypatch.setattr(provider, "_http_client", lambda: client)
            with call_deadline.model_deadline(monotonic() + 1.0):
                assert reranker.score_pairs("query", ["doc"]) == [1.0]
        assert len(received) == 1
        assert 1 <= received[0]["timeout_ms"] <= 1000
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


@pytest.mark.parametrize("phase", ["discovery", "transport"])
def test_expired_failure_does_not_become_fail_open_unavailability(monkeypatch, phase):
    now = [100.0]
    monkeypatch.setattr(call_deadline, "monotonic", lambda: now[0])
    requests = []

    def transport(request):
        requests.append(request)
        now[0] = 101.0
        raise httpx.ReadTimeout("synthetic deadline expiry", request=request)

    def discover(*args, **kwargs):
        now[0] = 101.0
        raise provider.EmbeddingUnavailableError("synthetic late discovery failure")

    with _install(monkeypatch, transport):
        if phase == "discovery":
            monkeypatch.setattr(provider, "_configured_model_cache", {})
            monkeypatch.setattr(provider, "_inference_service_health", discover)
        with (
            call_deadline.model_deadline(100.25),
            pytest.raises(call_deadline.CallDeadlineExceededError),
        ):
            reranker.score_pairs("query", ["doc"])
    assert len(requests) == (phase == "transport")
