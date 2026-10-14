import json

import httpx
import pytest

from reflexio.server import search_runtime
from reflexio.server.llm.providers import embedding_service_provider as provider
from reflexio.server.llm.rerank import cross_encoder_reranker as reranker


@pytest.fixture
def endpoints(monkeypatch):
    monkeypatch.setenv("REFLEXIO_RERANK_ENABLED", "true")
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "internal_service")
    monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_URL", "http://shared")
    monkeypatch.setenv("REFLEXIO_QUERY_EMBEDDING_SERVICE_URL", "http://reserved")
    monkeypatch.setenv("REFLEXIO_SEARCH_RERANK_SERVICE_URL", "http://reserved")
    monkeypatch.setattr(provider, "_configured_model_cache", {})
    monkeypatch.setattr(provider, "_query_contract_cache", {})
    calls = []
    settings = {
        "workload": "search",
        "ready": True,
        "status": 200,
        "model": "paired-reranker",
        "payloads": [],
    }

    def handle(request):
        reserved = request.url.host == "reserved"
        if request.url.path == "/health/model":
            return httpx.Response(
                200,
                json={
                    "model": "local/pinned",
                    "revision": "same",
                    "dimensions": 2,
                    "workload": "query" if reserved else "shared",
                },
            )
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "configured_model": "local/pinned",
                    "configured_reranker_model": settings["model"]
                    if reserved
                    else "paired-reranker",
                    "reranker_enabled": True,
                    "reranker_ready": settings["ready"],
                    "workload": settings["workload"] if reserved else "shared",
                },
            )
        calls.append(str(request.url))
        settings["payloads"].append(json.loads(request.content))
        return httpx.Response(
            settings["status"] if reserved else 200,
            json={
                "data": [{"index": 1, "score": -2.0}, {"index": 0, "score": 4.0}],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(provider, "_http_client", lambda: client)
        yield calls, settings


def test_shared_worker_distinguishes_search_from_other_interactive_reranks(
    endpoints, monkeypatch
):
    calls, settings = endpoints
    monkeypatch.delenv("REFLEXIO_SEARCH_RERANK_SERVICE_URL")
    monkeypatch.delenv("REFLEXIO_QUERY_EMBEDDING_SERVICE_URL")
    assert reranker.score_search_pairs("q", ["a", "b"]) == [4.0, -2.0]
    assert reranker.score_pairs("q", ["a", "b"]) == [4.0, -2.0]
    assert calls == ["http://shared/v1/rerank"] * 2
    assert settings["payloads"][0]["priority"] == "search"
    assert "priority" not in settings["payloads"][1]


def test_only_search_reranking_uses_reserved_capacity(endpoints):
    calls, _ = endpoints
    expected = ("paired-reranker", [4.0, -2.0])
    assert reranker.score_search_pairs_with_model("q", ["a", "b"]) == expected
    assert reranker.score_pairs_with_model("q", ["a", "b"]) == expected
    assert calls == ["http://reserved/v1/rerank", "http://shared/v1/rerank"]


@pytest.mark.parametrize(
    "change", [{"workload": "query"}, {"ready": False}, {"model": "other"}]
)
def test_search_route_rejects_incompatible_or_unready_worker(endpoints, change):
    calls, settings = endpoints
    settings.update(change)
    with pytest.raises(reranker.CrossEncoderUnavailableError):
        reranker.score_search_pairs_with_model("q", ["a", "b"])
    assert calls == []


def test_reserved_rerank_failure_does_not_send_work_to_shared(endpoints):
    calls, settings = endpoints
    settings["status"] = 503
    with pytest.raises(reranker.CrossEncoderUnavailableError):
        reranker.score_search_pairs_with_model("q", ["a", "b"])
    assert calls == ["http://reserved/v1/rerank"]


def test_unconfigured_search_reranking_retains_shared_route(endpoints, monkeypatch):
    calls, _ = endpoints
    monkeypatch.delenv("REFLEXIO_SEARCH_RERANK_SERVICE_URL")
    assert reranker.score_search_pairs_with_model("q", ["a", "b"]) == (
        "paired-reranker",
        [4.0, -2.0],
    )
    assert calls == ["http://shared/v1/rerank"]


def test_initially_unready_search_worker_recovers_without_api_restart(endpoints):
    calls, settings = endpoints
    settings["ready"] = False
    with pytest.raises(reranker.CrossEncoderUnavailableError):
        reranker.score_search_pairs_with_model("q", ["a", "b"])
    assert calls == []
    settings["ready"] = True
    assert reranker.score_search_pairs_with_model("q", ["a", "b"]) == (
        "paired-reranker",
        [4.0, -2.0],
    )
    assert calls == ["http://reserved/v1/rerank"]


def test_healthy_search_contract_does_not_repeat_health_discovery(
    endpoints, monkeypatch
):
    reranker.score_search_pairs_with_model("q", ["a", "b"])
    client = provider._http_client()
    monkeypatch.setattr(
        client,
        "get",
        lambda *_args, **_kwargs: pytest.fail("Hot search fetched health"),
    )
    assert reranker.score_search_pairs_with_model("q", ["a", "b"])[1] == [4.0, -2.0]


@pytest.mark.parametrize(
    "route,expected_calls,expected_counter",
    [
        ("reserved", ["http://reserved/v1/rerank"], 1),
        ("shared_search", ["http://shared/v1/rerank"], 0),
        ("shared_generic", ["http://shared/v1/rerank"], 0),
        ("empty", [], 0),
        ("reserved_failure", ["http://reserved/v1/rerank"], 0),
    ],
)
def test_search_worker_counter_proves_successful_nonempty_reserved_scoring(
    endpoints, monkeypatch, route, expected_calls, expected_counter
):
    calls, settings = endpoints
    if route == "shared_search":
        monkeypatch.delenv("REFLEXIO_SEARCH_RERANK_SERVICE_URL")
    if route == "reserved_failure":
        settings["status"] = 503
    score = (
        reranker.score_pairs
        if route == "shared_generic"
        else reranker.score_search_pairs
    )
    scope = search_runtime.SearchScope()
    token = search_runtime._scope.set(scope)
    try:
        if route == "reserved_failure":
            with pytest.raises(reranker.CrossEncoderUnavailableError):
                score("q", ["a", "b"])
        else:
            assert score("q", [] if route == "empty" else ["a", "b"]) == (
                [] if route == "empty" else [4.0, -2.0]
            )
        assert calls == expected_calls
        assert scope.counters == (
            {"search.rerank.search_worker": 1} if expected_counter else {}
        )
    finally:
        search_runtime._scope.reset(token)
