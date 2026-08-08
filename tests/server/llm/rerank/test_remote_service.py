from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import httpx
import pytest

import reflexio.server.llm.rerank.cross_encoder_reranker as reranker
from reflexio.server.llm.rerank.common import MULTILINGUAL_RERANK_MODEL
from reflexio.server.llm.rerank.cross_encoder_reranker import (
    CrossEncoderUnavailableError,
)
from reflexio.server.tracing import configure_tracer


class _Response:
    def __init__(self, body: dict[str, Any], status_code: int = 200) -> None:
        self._body = body
        request = httpx.Request("POST", "http://inference/v1/rerank")
        self._response = httpx.Response(status_code, request=request)

    @property
    def status_code(self) -> int:
        return self._response.status_code

    def raise_for_status(self) -> None:
        self._response.raise_for_status()

    def json(self) -> dict[str, Any]:
        return self._body


class _RecordingSpan:
    def __init__(self, name: str, data: dict[str, object]) -> None:
        self.name = name
        self.data = data

    def set_data(self, key: str, value: object) -> None:
        self.data[key] = value


class _RecordingTracer:
    def __init__(self) -> None:
        self.spans: list[_RecordingSpan] = []

    @contextmanager
    def span(self, name: str, **data: object):
        span = _RecordingSpan(name, dict(data))
        self.spans.append(span)
        yield span


def teardown_function() -> None:
    configure_tracer(None)


def _set_model(
    monkeypatch: pytest.MonkeyPatch, model: str = MULTILINGUAL_RERANK_MODEL
) -> None:
    monkeypatch.setattr(
        reranker, "resolve_service_configured_reranker_model", lambda: model
    )


def test_reranker_uses_embedding_service_url_and_discovered_model(monkeypatch) -> None:
    monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_URL", "http://inference:8089/")
    monkeypatch.setenv("REFLEXIO_RERANK_SERVICE_TIMEOUT_MS", "2500")
    _set_model(monkeypatch)

    class _Client:
        def post(self, url: str, *, json: dict[str, Any], timeout: float) -> _Response:
            assert url == "http://inference:8089/v1/rerank"
            assert json == {
                "model": MULTILINGUAL_RERANK_MODEL,
                "query": "数据库",
                "documents": ["PostgreSQL", "weather"],
            }
            assert timeout == 2.5
            return _Response(
                {
                    "data": [
                        {"index": 1, "score": -2.0},
                        {"index": 0, "score": 4.0},
                    ]
                }
            )

    monkeypatch.setattr(reranker, "inference_http_client", lambda: _Client())

    assert reranker.score_pairs("数据库", ["PostgreSQL", "weather"]) == [4.0, -2.0]


def test_separate_reranker_url_override_is_ignored(monkeypatch) -> None:
    monkeypatch.setenv("REFLEXIO_RERANK_SERVICE_URL", "http://obsolete")
    monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_URL", "http://inference")

    assert reranker.reranker_service_url() == "http://inference"


def test_disabled_reranker_makes_no_discovery_or_http_request(monkeypatch) -> None:
    monkeypatch.setenv("REFLEXIO_RERANK_ENABLED", "false")
    monkeypatch.setattr(
        reranker,
        "resolve_service_configured_reranker_model",
        lambda: pytest.fail("model discovery must not run"),
    )
    monkeypatch.setattr(
        reranker,
        "inference_http_client",
        lambda: pytest.fail("HTTP client must not be constructed"),
    )
    with pytest.raises(CrossEncoderUnavailableError, match="disabled"):
        reranker.score_pairs("q", ["doc"])


def test_colocated_discovery_failure_is_silent_fail_open(monkeypatch) -> None:
    monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_URL", raising=False)
    monkeypatch.setattr(
        reranker,
        "resolve_service_configured_reranker_model",
        lambda: (_ for _ in ()).throw(
            reranker.EmbeddingUnavailableError("child is restarting")
        ),
    )

    with pytest.raises(CrossEncoderUnavailableError) as exc_info:
        reranker.score_pairs("q", ["doc"])
    assert exc_info.value.report_failure is False


def test_remote_discovery_failure_is_reported_but_fail_open(monkeypatch) -> None:
    monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_URL", "http://inference")
    monkeypatch.setattr(
        reranker,
        "resolve_service_configured_reranker_model",
        lambda: (_ for _ in ()).throw(reranker.EmbeddingUnavailableError("down")),
    )

    with pytest.raises(CrossEncoderUnavailableError) as exc_info:
        reranker.score_pairs("q", ["doc"])
    assert exc_info.value.report_failure is True


@pytest.mark.parametrize("remote", [False, True])
def test_service_503_classification(monkeypatch, remote: bool) -> None:
    if remote:
        monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_URL", "http://inference")
    else:
        monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_URL", raising=False)
    _set_model(monkeypatch)

    class _Client:
        def post(self, *_args, **_kwargs) -> _Response:
            return _Response({}, status_code=503)

    monkeypatch.setattr(reranker, "inference_http_client", lambda: _Client())

    with pytest.raises(CrossEncoderUnavailableError) as exc_info:
        reranker.score_pairs("q", ["doc"])
    assert exc_info.value.report_failure is remote


@pytest.mark.parametrize("remote", [False, True])
def test_transport_failure_classification(monkeypatch, remote: bool) -> None:
    if remote:
        monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_URL", "http://inference")
    else:
        monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_URL", raising=False)
    _set_model(monkeypatch)

    class _Client:
        def post(self, *_args, **_kwargs) -> _Response:
            raise httpx.WriteError("connection closed while writing")

    monkeypatch.setattr(reranker, "inference_http_client", lambda: _Client())

    with pytest.raises(CrossEncoderUnavailableError) as exc_info:
        reranker.score_pairs("q", ["doc"])
    assert exc_info.value.report_failure is remote


@pytest.mark.parametrize(
    "data",
    [
        None,
        [{"index": 0, "score": 1.0}, {"index": 1, "score": 2.0}],
        [{"index": 0, "score": 1.0}, {"index": 0, "score": 2.0}],
        [{"index": 2, "score": 1.0}],
        [{"index": "0", "score": 1.0}],
        [{"index": 0, "score": True}],
    ],
)
def test_malformed_response_degrades_as_unavailable(monkeypatch, data: Any) -> None:
    monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_URL", "http://inference")
    _set_model(monkeypatch)

    class _Client:
        def post(self, *_args, **_kwargs) -> _Response:
            return _Response({"data": data})

    monkeypatch.setattr(reranker, "inference_http_client", lambda: _Client())
    with pytest.raises(CrossEncoderUnavailableError):
        reranker.score_pairs("q", ["doc"])


@pytest.mark.parametrize(
    ("mode", "expected_outcome", "expected_status"),
    [
        ("success", "success", 200),
        ("transport", "transport_error", None),
        ("http", "http_error", 503),
        ("invalid", "invalid_response", 200),
    ],
)
def test_rerank_api_span_covers_every_request_outcome(
    monkeypatch,
    mode: str,
    expected_outcome: str,
    expected_status: int | None,
) -> None:
    monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_URL", "http://inference")
    _set_model(monkeypatch)
    tracer = _RecordingTracer()
    configure_tracer(tracer)

    class _Client:
        def post(self, *_args, **_kwargs) -> _Response:
            if mode == "transport":
                raise httpx.ReadTimeout("timed out")
            if mode == "http":
                return _Response({}, status_code=503)
            if mode == "invalid":
                return _Response({"data": None})
            return _Response({"data": [{"index": 0, "score": 1.5}]})

    monkeypatch.setattr(reranker, "inference_http_client", lambda: _Client())
    if mode == "success":
        assert reranker.score_pairs("q", ["doc"]) == [1.5]
    else:
        with pytest.raises(CrossEncoderUnavailableError):
            reranker.score_pairs("q", ["doc"])

    assert len(tracer.spans) == 1
    span = tracer.spans[0]
    assert span.name == "search.rerank.api"
    assert span.data["model"] == MULTILINGUAL_RERANK_MODEL
    assert span.data["document_count"] == 1
    assert span.data["outcome"] == expected_outcome
    assert span.data.get("http_status_code") == expected_status
