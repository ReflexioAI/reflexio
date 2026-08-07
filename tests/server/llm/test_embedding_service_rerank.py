from __future__ import annotations

from fastapi.testclient import TestClient

from reflexio.server.llm.embedding_service import create_embedding_app
from reflexio.server.llm.rerank.common import (
    MULTILINGUAL_RERANK_MODEL,
    RERANK_MODEL,
    CrossEncoderUnavailableError,
)


class _Runner:
    def __init__(
        self,
        *,
        scores: list[float] | None = None,
        ready: bool = True,
        error: Exception | None = None,
    ) -> None:
        self.scores = scores or []
        self._ready = ready
        self.error = error
        self.prewarm_calls = 0
        self.score_calls: list[tuple[str, list[str]]] = []

    def score_pairs(self, query: str, docs: list[str]) -> list[float]:
        self.score_calls.append((query, docs))
        if self.error:
            raise self.error
        return self.scores

    def prewarm(self) -> bool:
        self.prewarm_calls += 1
        return self._ready

    def ready(self) -> bool:
        return self._ready

    def status(self) -> str:
        return "ready" if self._ready else "unavailable"


def test_rerank_endpoint_uses_configured_runner_and_model() -> None:
    runner = _Runner(scores=[0.25, -1.5])
    client = TestClient(
        create_embedding_app(
            reranker_model=MULTILINGUAL_RERANK_MODEL,
            reranker_runner=runner,  # type: ignore[arg-type]
        )
    )

    response = client.post(
        "/v1/rerank",
        json={
            "model": MULTILINGUAL_RERANK_MODEL,
            "query": "数据库",
            "documents": ["PostgreSQL", "weather"],
        },
    )

    assert response.status_code == 200
    assert response.json()["data"] == [
        {"index": 0, "score": 0.25},
        {"index": 1, "score": -1.5},
    ]
    assert runner.score_calls == [("数据库", ["PostgreSQL", "weather"])]


def test_rerank_endpoint_rejects_model_not_baked_with_service() -> None:
    runner = _Runner()
    client = TestClient(
        create_embedding_app(
            reranker_model=MULTILINGUAL_RERANK_MODEL,
            reranker_runner=runner,  # type: ignore[arg-type]
        )
    )

    response = client.post(
        "/v1/rerank",
        json={"model": RERANK_MODEL, "query": "q", "documents": ["doc"]},
    )
    assert response.status_code == 400
    assert runner.score_calls == []


def test_rerank_endpoint_reports_unavailable_runner() -> None:
    runner = _Runner(error=CrossEncoderUnavailableError("no model"))
    client = TestClient(create_embedding_app(reranker_runner=runner))  # type: ignore[arg-type]

    response = client.post(
        "/v1/rerank",
        json={"model": RERANK_MODEL, "query": "q", "documents": ["doc"]},
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "no model"


def test_disabled_reranker_skips_prewarm_and_scoring(monkeypatch) -> None:
    monkeypatch.setenv("REFLEXIO_RERANK_ENABLED", "false")
    runner = _Runner()
    with TestClient(create_embedding_app(reranker_runner=runner)) as client:  # type: ignore[arg-type]
        health = client.get("/health")
        reranker_health = client.get("/health/rerank")
        rerank = client.post(
            "/v1/rerank",
            json={"model": RERANK_MODEL, "query": "q", "documents": ["doc"]},
        )

    assert runner.prewarm_calls == 0
    assert runner.score_calls == []
    assert health.json()["reranker_enabled"] is False
    assert reranker_health.status_code == 503
    assert reranker_health.json()["status"] == "disabled"
    assert rerank.status_code == 503


def test_reranker_partial_health_does_not_fail_embedding_health() -> None:
    runner = _Runner(ready=False)
    client = TestClient(create_embedding_app(reranker_runner=runner))  # type: ignore[arg-type]

    embedding_health = client.get("/health")
    reranker_health = client.get("/health/rerank")

    assert embedding_health.status_code == 200
    assert embedding_health.json()["reranker_ready"] is False
    assert reranker_health.status_code == 503
    assert reranker_health.json()["status"] == "unavailable"


def test_service_lifespan_prewarms_runner_directly() -> None:
    runner = _Runner()
    with TestClient(create_embedding_app(reranker_runner=runner)) as client:  # type: ignore[arg-type]
        assert client.get("/health").status_code == 200
    assert runner.prewarm_calls == 1
