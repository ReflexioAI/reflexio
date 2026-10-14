import threading
import time
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from anyio.to_thread import run_sync
from fastapi import HTTPException
from fastapi.testclient import TestClient

from reflexio.server.llm import embedding_service
from reflexio.server.llm.embedding_service import create_embedding_app
from reflexio.server.llm.rerank.common import RERANK_MODEL
from reflexio.server.llm.rerank.cross_encoder_model import CrossEncoderRunner


@pytest.fixture
def real_runner(monkeypatch):
    monkeypatch.setenv("REFLEXIO_RERANK_ENABLED", "true")
    monkeypatch.setattr(embedding_service, "_ACTIVE_MODEL", None)
    runner = CrossEncoderRunner(RERANK_MODEL)
    model = Mock()
    model.predict.return_value = [1.0]
    runner._model = model
    return runner, model


def search_app(runner, **kwargs):
    return create_embedding_app(
        default_model="custom/pinned",
        allowed_models={"custom/pinned"},
        model_encoders={"custom/pinned": lambda texts: [[1.0, 0.0] for _ in texts]},
        model_contract=lambda: {
            "model": "custom/pinned",
            "revision": "fixed",
            "dimensions": 2,
        },
        reranker_runner=runner,
        workload="search",
        **kwargs,
    )


def fail_scoring(client):
    assert (
        client.post(
            "/v1/rerank",
            json={"model": RERANK_MODEL, "query": "q", "documents": ["d"]},
        ).status_code
        == 503
    )


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


def test_search_startup_requires_real_prewarm_without_scheduler(real_runner):
    runner, model = real_runner
    model.predict.side_effect = RuntimeError("model unavailable")
    with (
        pytest.raises(RuntimeError, match="startup readiness"),
        TestClient(search_app(runner)),
    ):
        pytest.fail("Unready search worker started")
    assert not runner.ready()


@pytest.mark.parametrize("scheduled", [False, True])
def test_search_health_recovers_actual_runner_after_transient_score_failure(
    real_runner, scheduled
):
    runner, model = real_runner
    submissions = []

    async def submit(encode, texts, priority, timeout_ms, chunk_size):
        submissions.append((priority, timeout_ms, chunk_size))
        return await run_sync(encode, texts)

    app = search_app(runner, inference_submit=submit if scheduled else None)
    with TestClient(app) as client:
        model.predict.side_effect = RuntimeError("transient prediction failure")
        fail_scoring(client)
        assert not runner.ready()
        model.predict.side_effect = None
        assert client.get("/health").json()["reranker_ready"] is True
        assert runner.ready()
        assert model.predict.call_args.args == ([("warmup", "warmup")],)
        response = client.post(
            "/v1/rerank",
            json={"model": RERANK_MODEL, "query": "recovered", "documents": ["d"]},
        )
        assert response.status_code == 200
        assert response.json()["data"] == [{"index": 0, "score": 1.0}]
        calls = model.predict.call_count
        assert client.get("/health").json()["reranker_ready"] is True
        assert model.predict.call_count == calls
    if scheduled:
        assert ("interactive", 1000, 1) in submissions


def test_search_health_failed_probe_does_not_advertise_false_recovery(real_runner):
    runner, model = real_runner
    with TestClient(search_app(runner)) as client:
        model.predict.side_effect = RuntimeError("prediction remains unavailable")
        fail_scoring(client)
        assert client.get("/health").json()["reranker_ready"] is False
        response = client.get("/health/rerank")
        assert response.status_code == 503
        assert response.json()["ready"] is False
        assert not runner.ready()
        assert model.predict.call_count == 4  # Startup, scoring, two health probes.


def test_search_health_scheduler_timeout_stays_unready(real_runner):
    runner, model = real_runner

    async def submit(encode, texts, _priority, timeout_ms, _chunk_size):
        if timeout_ms == 1000:
            raise HTTPException(504, "CPU inference deadline exceeded")
        return await run_sync(encode, texts)

    with TestClient(search_app(runner, inference_submit=submit)) as client:
        model.predict.side_effect = RuntimeError("transient failure")
        fail_scoring(client)
        model.predict.side_effect = None
        assert client.get("/health").json()["reranker_ready"] is False
        assert not runner.ready()
        assert model.predict.call_count == 2


def test_slow_search_recovery_retains_single_flight_after_health_timeout(real_runner):
    runner, model = real_runner
    entered, release = threading.Event(), threading.Event()

    def slow_prediction(*_args, **_kwargs):
        entered.set()
        assert release.wait(5)
        return [1.0]

    with TestClient(search_app(runner)) as client:
        model.predict.side_effect = RuntimeError("transient failure")
        fail_scoring(client)
        model.predict.side_effect = slow_prediction
        try:
            started = time.monotonic()
            assert client.get("/health").json()["reranker_ready"] is False
            assert time.monotonic() - started < 2
            assert entered.is_set()
            calls = model.predict.call_count
            assert client.get("/health/rerank").status_code == 503
            assert model.predict.call_count == calls
        finally:
            release.set()
        deadline = time.monotonic() + 2
        while not runner.ready() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert runner.ready()


@pytest.mark.parametrize("workload", ["shared", "query"])
def test_other_worker_health_does_not_retry_unready_reranker(real_runner, workload):
    runner, model = real_runner
    client = TestClient(create_embedding_app(reranker_runner=runner, workload=workload))
    assert client.get("/health").json()["reranker_ready"] is False
    assert client.get("/health/rerank").status_code == 503
    model.predict.assert_not_called()
