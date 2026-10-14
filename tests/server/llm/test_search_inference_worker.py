from types import SimpleNamespace
from typing import cast

from fastapi.testclient import TestClient

from reflexio.server.llm import embedding_service
from reflexio.server.llm.embedding_service import create_embedding_app
from reflexio.server.llm.rerank.common import RERANK_MODEL
from reflexio.server.llm.rerank.cross_encoder_model import CrossEncoderRunner


def test_search_worker_admits_query_and_rerank_but_rejects_documents(monkeypatch):
    monkeypatch.setenv("REFLEXIO_RERANK_ENABLED", "true")
    monkeypatch.setattr(embedding_service, "_ACTIVE_MODEL", None)
    runner = SimpleNamespace(
        prewarm=lambda: True,
        ready=lambda: True,
        status=lambda: "ready",
        score_pairs=lambda _query, docs: [float(i) for i in range(len(docs))],
    )
    app = create_embedding_app(
        default_model="custom/pinned",
        allowed_models={"custom/pinned"},
        model_encoders={"custom/pinned": lambda texts: [[1.0, 0.0] for _ in texts]},
        fixed_dimensions={"custom/pinned": 2},
        model_contract=lambda: {
            "model": "custom/pinned",
            "revision": "fixed",
            "dimensions": 2,
        },
        reranker_runner=cast(CrossEncoderRunner, runner),
        workload="search",
    )
    with TestClient(app) as client:
        assert client.get("/health/model").json()["workload"] == "query"
        assert client.get("/health").json()["workload"] == "search"
        assert client.get("/health").json()["reranker_ready"] is True
        assert (
            client.post(
                "/v1/embeddings",
                json={
                    "model": "custom/pinned",
                    "input": ["question"],
                    "purpose": "query",
                },
            ).status_code
            == 200
        )
        for payload in (
            {"input": ["query: document"], "purpose": "document"},
            {"input": ["a", "b"], "purpose": "query"},
        ):
            assert (
                client.post(
                    "/v1/embeddings", json={"model": "custom/pinned", **payload}
                ).status_code
                == 422
            )
        response = client.post(
            "/v1/rerank",
            json={
                "model": RERANK_MODEL,
                "query": "question",
                "documents": ["a", "b"],
            },
        )
        assert response.status_code == 200
        assert [item["score"] for item in response.json()["data"]] == [0.0, 1.0]
