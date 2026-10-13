"""Query isolation: explicit purpose, identical vector space, no shared failover."""

from __future__ import annotations

import threading
import time
from unittest.mock import Mock

import httpx
import pytest

from reflexio.server import search_runtime
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.providers import embedding_service_provider as provider

MODEL = "local/multilingual-e5-small"
CONTRACT = {"model": MODEL, "revision": "pinned-revision", "dimensions": 2}


@pytest.fixture
def routing(monkeypatch):
    calls = []
    bodies = {
        "shared": {**CONTRACT, "workload": "shared"},
        "query": {**CONTRACT, "workload": "query"},
    }

    def handler(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=bodies[request.url.host])
        import json

        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": [1.0, 0.0]}
                    for i in range(len(payload["input"]))
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "internal_service")
    monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_URL", "http://shared")
    monkeypatch.setenv("REFLEXIO_QUERY_EMBEDDING_SERVICE_URL", "http://query/")
    monkeypatch.setattr(provider, "_query_contract_cache", {})
    monkeypatch.setattr(provider, "_query_contract_lock", threading.Lock())
    monkeypatch.setattr(provider, "_http_client", lambda: client)
    yield calls, bodies
    client.close()


def test_explicit_query_routes_and_batches_single_inputs(routing):
    calls, _ = routing
    llm = LiteLLMClient(LiteLLMConfig(model="gpt-4o"))
    assert llm.get_embedding("passage: no routing hint", MODEL, 2, purpose="query") == [
        1.0,
        0.0,
    ]
    assert llm.get_embeddings(["one", "two"], MODEL, 2, purpose="query") == [
        [1.0, 0.0],
        [1.0, 0.0],
    ]
    assert [r.url.host for r in calls if r.method == "POST"] == ["query"] * 3
    assert len([r for r in calls if r.method == "GET"]) == 2
    assert all(b'"purpose":"query"' in r.content for r in calls if r.method == "POST")


def test_document_prefix_and_search_scope_cannot_select_query_worker(routing):
    calls, _ = routing
    token = search_runtime._scope.set(search_runtime.SearchScope())
    try:
        provider.get_service_embeddings(["query: document text"], model=MODEL)
    finally:
        search_runtime._scope.reset(token)
    assert [(r.method, r.url.host) for r in calls] == [("POST", "shared")]


def test_unset_or_blank_query_url_retains_shared_routing(routing, monkeypatch):
    calls, _ = routing
    monkeypatch.setenv("REFLEXIO_QUERY_EMBEDDING_SERVICE_URL", " \t")
    provider.get_service_embeddings(["query: text"], model=MODEL, purpose="query")
    assert [(r.method, r.url.host) for r in calls] == [("POST", "shared")]


def test_local_model_query_route_uses_default_shared_daemon(routing, monkeypatch):
    calls, bodies = routing
    monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_URL", raising=False)
    monkeypatch.delenv("REFLEXIO_EMBEDDING_DAEMON_HOST", raising=False)
    monkeypatch.setenv("EMBEDDING_PORT", "8072")
    bodies["127.0.0.1"] = bodies["shared"]
    assert provider.get_service_embeddings(
        ["text"], model=MODEL, dimensions=2, purpose="query"
    ) == [[1.0, 0.0]]
    assert str(calls[0].url) == "http://127.0.0.1:8072/health/model"
    assert calls[-1].method == "POST" and calls[-1].url.host == "query"


def test_advertised_query_contract_lock_ignores_long_embedding_timeout(
    routing, monkeypatch
):
    _, bodies = routing
    monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_TIMEOUT_MS", "600000")
    monkeypatch.setattr(provider, "_ADVERTISED_CAPABILITY_LOCK_SECONDS", 0.03)
    provider._query_contract_lock.acquire()
    try:
        started = time.monotonic()
        with pytest.raises(provider.EmbeddingUnavailableError, match="busy"):
            provider.validate_query_embedding_service(
                MODEL, 2, advertised=(bodies["shared"], bodies["query"])
            )
        assert time.monotonic() - started < 0.5
    finally:
        provider._query_contract_lock.release()


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "wrong"),
        ("revision", "different"),
        ("dimensions", 3),
        ("revision", ""),
        ("dimensions", True),
        ("workload", "shared"),
    ],
)
def test_incompatible_worker_fails_before_any_embedding(routing, field, value):
    calls, bodies = routing
    bodies["query"][field] = value
    with pytest.raises(provider.EmbeddingUnavailableError):
        provider.get_service_embeddings(
            ["text"], model=MODEL, dimensions=2, purpose="query"
        )
    assert not [r for r in calls if r.method == "POST"]
    assert not provider._query_contract_cache


def test_fresh_contract_drift_is_rejected_and_never_overwrites_cache(routing):
    _, bodies = routing
    provider.validate_query_embedding_service(MODEL, 2)
    previous = dict(provider._query_contract_cache)
    fresh = (
        {**bodies["shared"], "revision": "new"},
        {**bodies["query"], "revision": "new"},
    )
    with pytest.raises(provider.EmbeddingUnavailableError, match="restart"):
        provider.validate_query_embedding_service(MODEL, 2, advertised=fresh)
    assert provider._query_contract_cache == previous


def test_request_model_dimensions_and_same_endpoint_are_rejected(routing, monkeypatch):
    provider.validate_query_embedding_service(MODEL, 2)
    for model, dimensions in [("wrong", 2), (MODEL, 3)]:
        with pytest.raises(provider.EmbeddingUnavailableError, match="requested"):
            provider.validate_query_embedding_service(model, dimensions)
    monkeypatch.setenv("REFLEXIO_QUERY_EMBEDDING_SERVICE_URL", "http://shared")
    with pytest.raises(provider.EmbeddingUnavailableError, match="separate"):
        provider.validate_query_embedding_service(MODEL, 2)


def test_busy_discovery_respects_original_search_deadline(routing):
    token = search_runtime._scope.set(
        search_runtime.SearchScope(deadline=time.monotonic() + 0.03)
    )
    provider._query_contract_lock.acquire()
    try:
        started = time.monotonic()
        with pytest.raises(provider.EmbeddingUnavailableError, match="busy"):
            provider.validate_query_embedding_service(MODEL, 2)
        assert time.monotonic() - started < 0.5
    finally:
        provider._query_contract_lock.release()
        search_runtime._scope.reset(token)


def test_failed_query_never_retries_on_shared_worker(routing, monkeypatch):
    provider.validate_query_embedding_service(MODEL, 2)
    client = Mock()
    client.post.side_effect = httpx.ReadTimeout("query worker down")
    monkeypatch.setattr(provider, "_http_client", lambda: client)
    with pytest.raises(provider.EmbeddingUnavailableError):
        provider.get_service_embeddings(
            ["text"], model=MODEL, dimensions=2, purpose="query"
        )
    client.post.assert_called_once()
    assert client.post.call_args.args[0] == "http://query/v1/embeddings"


@pytest.mark.parametrize("vector", [[1.0], [float("nan"), 0.0], [0.0, float("inf")]])
@pytest.mark.parametrize("dimensions", [2, None])
def test_invalid_query_vectors_fail_without_shared_fallback_or_success_counter(
    routing, monkeypatch, vector, dimensions
):
    provider.validate_query_embedding_service(MODEL, 2)
    client = Mock()
    response = Mock()
    response.json.return_value = {"data": [{"index": 0, "embedding": vector}]}
    client.post.return_value = response
    counter = Mock()
    monkeypatch.setattr(provider, "_http_client", lambda: client)
    monkeypatch.setattr(provider, "increment", counter)
    with pytest.raises(provider.EmbeddingUnavailableError):
        provider.get_service_embeddings(
            ["text"], model=MODEL, dimensions=dimensions, purpose="query"
        )
    client.post.assert_called_once()
    assert client.post.call_args.args[0] == "http://query/v1/embeddings"
    counter.assert_not_called()


def test_query_worker_cannot_also_be_the_document_destination(routing):
    calls, bodies = routing
    bodies["shared"]["workload"] = "query"
    with pytest.raises(provider.EmbeddingUnavailableError, match="Document"):
        provider.get_service_embeddings(["text"], model=MODEL, purpose="query")
    assert not [request for request in calls if request.method == "POST"]
