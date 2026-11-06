"""One captured request allowance owns admission; zero does no optional work."""

from unittest.mock import Mock

import pytest

from reflexio.server.llm import call_deadline
from reflexio.server.llm.rerank import common
from reflexio.server.llm.rerank import cross_encoder_reranker as remote
from reflexio.server.llm.rerank.cross_encoder_model import CrossEncoderRunner


@pytest.mark.parametrize(
    "raw,expected", [(None, 5000), ("", 5000), ("  ", 5000), ("0", 0), ("17", 17)]
)
def test_budget_policy_preserves_default_custom_and_disabled(
    monkeypatch, raw, expected
):
    if raw is None:
        monkeypatch.delenv("REFLEXIO_RERANK_BUDGET_MS", raising=False)
    else:
        monkeypatch.setenv("REFLEXIO_RERANK_BUDGET_MS", raw)
    assert common.reranker_budget_ms() == expected
    assert common.reranker_enabled() is (expected > 0)


@pytest.mark.parametrize("raw", ["-1", "true", "1.5", "secret-invalid", "9" * 400])
def test_invalid_budget_fails_without_echoing_value(monkeypatch, raw):
    monkeypatch.setenv("REFLEXIO_RERANK_BUDGET_MS", raw)
    with pytest.raises(common.CrossEncoderUnavailableError) as exc:
        common.reranker_budget_ms()
    assert raw not in str(exc.value)


@pytest.mark.parametrize(
    "name", ["REFLEXIO_RERANK_ENABLED", "REFLEXIO_RERANK_SERVICE_TIMEOUT_MS"]
)
@pytest.mark.parametrize("value", ["", "false", "0", "2500"])
def test_obsolete_assignment_refuses_even_with_current_budget(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    monkeypatch.setenv("REFLEXIO_RERANK_BUDGET_MS", "2500")
    with pytest.raises(ValueError, match="REFLEXIO_RERANK_BUDGET_MS"):
        common.reranker_budget_ms()


def test_zero_budget_never_discovers_or_loads(monkeypatch):
    monkeypatch.setenv("REFLEXIO_RERANK_BUDGET_MS", "0")
    discover = Mock(side_effect=AssertionError("disabled discovery"))
    monkeypatch.setattr(remote, "resolve_service_configured_reranker_model", discover)
    monkeypatch.setattr(
        remote,
        "inference_http_client",
        Mock(side_effect=AssertionError("disabled HTTP")),
    )
    with pytest.raises(common.CrossEncoderUnavailableError, match="disabled"):
        remote.score_pairs("q", ["d"])
    runner = CrossEncoderRunner(common.RERANK_MODEL)
    monkeypatch.setattr(
        runner, "_get_model", Mock(side_effect=AssertionError("disabled load"))
    )
    assert runner.status() == "disabled"
    assert runner.prewarm() is False
    discover.assert_not_called()


def test_admitted_budget_is_not_reread_after_discovery(monkeypatch):
    monkeypatch.setenv("REFLEXIO_RERANK_BUDGET_MS", "2500")

    def discover():
        monkeypatch.setenv("REFLEXIO_RERANK_BUDGET_MS", "0")
        return common.RERANK_MODEL

    monkeypatch.setattr(remote, "resolve_service_configured_reranker_model", discover)
    post = Mock(return_value=[1.0])
    monkeypatch.setattr(remote, "_score_pairs_remote", post)
    assert remote.score_pairs("q", ["d"]) == [1.0]
    assert post.call_args.kwargs["budget_ms"] == 2500


def test_outer_expiry_still_propagates_and_is_not_fail_open(monkeypatch):
    monkeypatch.setenv("REFLEXIO_RERANK_BUDGET_MS", "5000")
    now = [100.0]
    monkeypatch.setattr(call_deadline, "monotonic", lambda: now[0])

    def discover():
        now[0] += 0.3
        return common.RERANK_MODEL

    monkeypatch.setattr(remote, "resolve_service_configured_reranker_model", discover)
    with (
        call_deadline.model_deadline(100.2),
        pytest.raises(call_deadline.CallDeadlineExceededError),
    ):
        remote.score_pairs("q", ["d"])
    assert call_deadline.model_call_deadline.get() is None


def test_real_http_preserves_request_budget_after_discovery(monkeypatch):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import httpx

    from reflexio.server.llm.providers import embedding_service_provider as provider

    now = [100.0]
    requests = []
    monkeypatch.setattr(call_deadline, "monotonic", lambda: now[0])
    monkeypatch.setenv("REFLEXIO_RERANK_BUDGET_MS", "5000")
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "internal_service")
    monkeypatch.setattr(provider, "_configured_model_cache", {})

    class Handler(BaseHTTPRequestHandler):
        def respond(self, value):
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            requests.append(self.path)
            assert self.path == "/health"
            now[0] += 2.0
            self.respond(
                {
                    "configured_model": "local/owned-embedding",
                    "configured_reranker_model": common.RERANK_MODEL,
                    "reranker_enabled": True,
                    "reranker_ready": True,
                }
            )

        def do_POST(self):
            requests.append(self.path)
            assert self.path == "/v1/rerank"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["timeout_ms"] == 5000
            assert body["documents"] == ["d"]
            self.respond({"data": [{"index": 0, "score": 2.0}]})

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    thread.start()
    monkeypatch.setenv(
        "REFLEXIO_EMBEDDING_SERVICE_URL", f"http://127.0.0.1:{server.server_port}"
    )
    try:
        with httpx.Client(
            event_hooks={"request": [provider.http_request_deadline]}
        ) as client:
            monkeypatch.setattr(provider, "_http_client", lambda: client)
            assert remote.score_pairs("q", ["d"]) == [2.0]
            assert requests == ["/health", "/v1/rerank"]
            monkeypatch.setenv("REFLEXIO_RERANK_BUDGET_MS", "0")
            with pytest.raises(common.CrossEncoderUnavailableError, match="disabled"):
                remote.score_pairs("q", ["d"])
            assert requests == ["/health", "/v1/rerank"]
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
    assert call_deadline.model_call_deadline.get() is None


@pytest.mark.parametrize("budget", ["0", "5000"])
def test_expired_caller_is_not_masked_by_optional_admission(monkeypatch, budget):
    monkeypatch.setenv("REFLEXIO_RERANK_BUDGET_MS", budget)
    monkeypatch.setattr(call_deadline, "monotonic", lambda: 100.0)
    with (
        call_deadline.model_deadline(99.0),
        pytest.raises(call_deadline.CallDeadlineExceededError),
    ):
        remote.score_pairs("q", ["d"])


def test_disabled_direct_model_scoring_never_imports_model(monkeypatch):
    from reflexio.server.llm.rerank import cross_encoder_model as model

    monkeypatch.setenv("REFLEXIO_RERANK_BUDGET_MS", "0")
    load = Mock(side_effect=AssertionError("disabled model import"))
    monkeypatch.setattr(model, "_import_cross_encoder", load)
    with pytest.raises(common.CrossEncoderUnavailableError, match="disabled"):
        model.CrossEncoderRunner(common.RERANK_MODEL).score_pairs("q", ["d"])
    load.assert_not_called()
