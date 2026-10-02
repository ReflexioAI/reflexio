"""The HTTP boundary must preserve CPU priority, deadline and failures."""

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from reflexio.server.llm.embedding_service import (
    _until_disconnected,
    create_embedding_app,
)


def make_app(submit, ready=True):
    runner = Mock()
    runner.prewarm.return_value = ready
    runner.ready.return_value = ready
    runner.score_pairs.side_effect = lambda _query, docs: [0.5] * len(docs)
    runner.status.return_value = "ready" if ready else "unavailable"
    return create_embedding_app(
        allowed_models={"custom/model"},
        model_encoders={"custom/model": lambda texts: [[1.0, 0.0] for _ in texts]},
        inference_submit=submit,
        reranker_model="custom/reranker",
        reranker_runner=runner,
    )


def test_routes_preserve_priority_deadline_chunking_and_order():
    calls = []

    async def submit(encode, texts, priority, timeout_ms, chunk_size):
        calls.append((texts, priority, timeout_ms, chunk_size))
        return encode(texts)

    with TestClient(make_app(submit)) as client:
        embedded = client.post(
            "/v1/embeddings",
            json={
                "model": "custom/model",
                "input": ["a", "b"],
                "priority": "interactive",
                "timeout_ms": 1234,
            },
        )
        reranked = client.post(
            "/v1/rerank",
            json={
                "model": "custom/reranker",
                "query": "q",
                "documents": ["a", "b"],
                "timeout_ms": 4321,
            },
        )
    assert embedded.status_code == reranked.status_code == 200
    assert calls == [
        (["a", "b"], "interactive", 1234, 1),
        (["a", "b"], "interactive", 4321, 4),
    ]
    assert [row["index"] for row in reranked.json()["data"]] == [0, 1]


@pytest.mark.parametrize("status", [503, 504])
def test_scheduler_failures_reach_both_http_callers(status):
    async def submit(*args):
        raise HTTPException(status, "capacity/deadline")

    with TestClient(make_app(submit)) as client:
        assert (
            client.post(
                "/v1/embeddings", json={"model": "custom/model", "input": "a"}
            ).status_code
            == status
        )
        assert (
            client.post(
                "/v1/rerank",
                json={"model": "custom/reranker", "query": "q", "documents": ["a"]},
            ).status_code
            == status
        )


def test_cpu_startup_refuses_enabled_unavailable_reranker(monkeypatch):
    monkeypatch.setenv("REFLEXIO_RERANK_ENABLED", "true")

    async def submit(*args):
        return []

    with (
        pytest.raises(RuntimeError, match="startup readiness"),
        TestClient(make_app(submit, ready=False)),
    ):
        pass


@pytest.mark.asyncio
async def test_http_disconnect_cancels_the_borrowed_submission():
    disconnected = asyncio.Event()
    cancelled = asyncio.Event()

    request = Mock(spec=Request)
    request.is_disconnected = AsyncMock(side_effect=lambda: disconnected.is_set())

    async def submit():
        try:
            await asyncio.Event().wait()
            return []
        finally:
            cancelled.set()

    task = asyncio.create_task(_until_disconnected(request, submit()))
    await asyncio.sleep(0)
    disconnected.set()
    with pytest.raises(HTTPException) as failure:
        await asyncio.wait_for(task, 1)
    assert failure.value.status_code == 499
    assert cancelled.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,payload",
    [
        ("/v1/embeddings", {"model": "custom/model", "input": "example"}),
        (
            "/v1/rerank",
            {"model": "custom/reranker", "query": "q", "documents": ["example"]},
        ),
    ],
)
async def test_asgi_disconnect_releases_both_cpu_route_submissions(path, payload):
    entered, disconnected, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def submit(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    app = make_app(submit)
    body_sent = False

    async def receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {
                "type": "http.request",
                "body": json.dumps(payload).encode(),
                "more_body": False,
            }
        await disconnected.wait()
        return {"type": "http.disconnect"}

    responses = []

    async def send(message):
        responses.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "server": ("localhost", 8089),
        "client": ("localhost", 1),
    }
    task = asyncio.create_task(app(scope, receive, send))
    await asyncio.wait_for(entered.wait(), 1)
    disconnected.set()
    await asyncio.wait_for(task, 1)
    assert cancelled.is_set()
    assert responses[0]["status"] == 499
