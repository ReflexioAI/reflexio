from typing import Any
from unittest.mock import patch

import httpx
import pytest

import reflexio.server.llm.rerank.cross_encoder_reranker as reranker
from reflexio.server.llm.rerank.cross_encoder_reranker import (
    RERANK_MODEL,
    CrossEncoderUnavailableError,
)


class _Response:
    def __init__(
        self,
        body: dict[str, Any],
        *,
        status_error: Exception | None = None,
    ) -> None:
        self._body = body
        self._status_error = status_error

    def raise_for_status(self) -> None:
        if self._status_error is not None:
            raise self._status_error

    def json(self) -> dict[str, Any]:
        return self._body


def test_unset_service_url_uses_local_scorer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REFLEXIO_RERANK_SERVICE_URL", raising=False)
    with patch.object(reranker, "_score_pairs_local", return_value=[1.0]) as local:
        assert reranker.score_pairs("q", ["doc"]) == [1.0]
    local.assert_called_once_with("q", ["doc"])


def test_set_service_url_calls_remote_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REFLEXIO_RERANK_SERVICE_URL", "http://rerank.internal:8089/")
    monkeypatch.setenv("REFLEXIO_RERANK_SERVICE_TIMEOUT_MS", "2500")

    def post(url: str, *, json: dict[str, Any], timeout: float) -> _Response:
        assert url == "http://rerank.internal:8089/v1/rerank"
        assert json == {
            "model": RERANK_MODEL,
            "query": "q",
            "documents": ["first", "second"],
        }
        assert timeout == 2.5
        return _Response(
            {
                "data": [
                    {"index": 1, "score": -1.0},
                    {"index": 0, "score": 2},
                ]
            }
        )

    monkeypatch.setattr(reranker.httpx, "post", post)

    assert reranker.score_pairs("q", ["first", "second"]) == [2.0, -1.0]


def test_remote_endpoint_http_error_degrades_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REFLEXIO_RERANK_SERVICE_URL", "http://rerank.internal")
    request = httpx.Request("POST", "http://rerank.internal/v1/rerank")
    response = httpx.Response(503, request=request)
    status_error = httpx.HTTPStatusError(
        "service unavailable", request=request, response=response
    )
    monkeypatch.setattr(
        reranker.httpx,
        "post",
        lambda *_args, **_kwargs: _Response({}, status_error=status_error),
    )

    with pytest.raises(CrossEncoderUnavailableError):
        reranker.score_pairs("q", ["doc"])


@pytest.mark.parametrize(
    "data",
    [
        None,
        [{"index": 0, "score": 1.0}, {"index": 1, "score": 2.0}],
        [{"index": 0, "score": 1.0}, {"index": 0, "score": 2.0}],
        [{"index": 2, "score": 1.0}],
        [{"index": "0", "score": 1.0}],
        [{"index": 0, "score": "1.0"}],
        [{"index": 0, "score": True}],
    ],
)
def test_remote_endpoint_malformed_response_degrades_as_unavailable(
    monkeypatch: pytest.MonkeyPatch, data: Any
) -> None:
    monkeypatch.setenv("REFLEXIO_RERANK_SERVICE_URL", "http://rerank.internal")
    monkeypatch.setattr(
        reranker.httpx,
        "post",
        lambda *_args, **_kwargs: _Response({"data": data}),
    )

    with pytest.raises(CrossEncoderUnavailableError):
        reranker.score_pairs("q", ["doc"])


def test_invalid_remote_timeout_degrades_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REFLEXIO_RERANK_SERVICE_URL", "http://rerank.internal")
    monkeypatch.setenv("REFLEXIO_RERANK_SERVICE_TIMEOUT_MS", "not-an-int")

    with pytest.raises(CrossEncoderUnavailableError):
        reranker.score_pairs("q", ["doc"])
