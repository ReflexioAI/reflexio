"""Compute deadlines gate managed embedding work without detaching it."""

import json
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

from reflexio.server.llm import _litellm_embedding as embedding
from reflexio.server.llm import _provider_concurrency as admission
from reflexio.server.llm import call_deadline as deadline
from reflexio.server.llm.providers import embedding_service_provider as service


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(deadline, "monotonic", lambda: now[0])
    return now


def test_nested_budget_tightens_and_expired_unknown_provider_cannot_dispatch(
    monkeypatch, clock
):
    monkeypatch.setattr(admission, "_provider_key", lambda _model: None)
    assert deadline.remaining_call_time(4) == 4
    with deadline.model_deadline(102):
        with deadline.model_deadline(104):
            assert deadline.remaining_call_time(4) == 2
        clock[0] = 102
        with (
            pytest.raises(deadline.CallDeadlineExceededError),
            admission.provider_slot("x"),
        ):
            pytest.fail("expired work dispatched without a recognized provider")


@pytest.mark.parametrize("acquired", [True, False])
def test_provider_wait_consumes_budget_and_releases_only_acquired_capacity(
    monkeypatch, clock, acquired
):
    waits, releases = [], []

    class Permit:
        def acquire(self, *, timeout):
            waits.append(timeout)
            clock[0] = 101
            return acquired

        def release(self):
            releases.append(True)

    monkeypatch.setattr(admission, "_provider_key", lambda _model: "synthetic")
    monkeypatch.setattr(admission, "_get_semaphore", lambda _provider: Permit())
    with (
        deadline.model_deadline(101),
        pytest.raises(deadline.CallDeadlineExceededError),
        admission.provider_slot("x"),
    ):
        pytest.fail("provider dispatched after admission spent the budget")
    assert waits == [1]
    assert len(releases) == int(acquired)


@pytest.fixture
def service_mode(monkeypatch):
    monkeypatch.setattr(
        service, "embedding_provider_mode", lambda *_: "internal_service"
    )
    monkeypatch.setattr(service, "embedding_service_url", lambda *_: "http://offline")
    monkeypatch.setattr(service, "query_embedding_service_url", lambda: "")
    monkeypatch.setattr(service, "_max_texts_per_request", lambda: 1)


@pytest.mark.parametrize("slow_chunk", [0, 1])
def test_real_service_path_does_not_return_late_vectors_or_dispatch_next_chunk(
    monkeypatch, clock, service_mode, slow_chunk
):
    requests = []

    def respond(request):
        requests.append(request)
        assert request.extensions["timeout"]["read"] <= 1
        assert json.loads(request.content)["timeout_ms"] <= 1000
        clock[0] += 1 if len(requests) - 1 == slow_chunk else 0.25
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1]}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(service, "_http_client", lambda: client)
        with (
            deadline.model_deadline(101),
            pytest.raises(deadline.CallDeadlineExceededError),
        ):
            service.get_service_embeddings(["a", "b", "c"], model="local/synthetic")
    assert len(requests) == slow_chunk + 1


def test_expired_reconnect_backoff_does_not_replay(monkeypatch, clock, service_mode):
    requests = []

    def connect_failure(request):
        requests.append(request)
        clock[0] = 100.95
        raise httpx.ConnectError("offline connect failure", request=request)

    def spend_backoff(seconds):
        assert seconds == pytest.approx(0.05)
        clock[0] += seconds

    monkeypatch.setattr(service, "time", SimpleNamespace(sleep=spend_backoff))
    with httpx.Client(transport=httpx.MockTransport(connect_failure)) as client:
        monkeypatch.setattr(service, "_http_client", lambda: client)
        with (
            deadline.model_deadline(101),
            pytest.raises(deadline.CallDeadlineExceededError),
        ):
            service.get_service_embeddings(["a"], model="local/synthetic")
    assert len(requests) == 1


def test_late_capability_response_is_not_cached_and_lock_is_released(
    monkeypatch, clock, service_mode
):
    monkeypatch.setattr(service, "_configured_model_cache", {})

    def late_health(*_args, **_kwargs):
        clock[0] = 101
        return {"configured_model": "local/synthetic"}

    monkeypatch.setattr(service, "_inference_service_health", late_health)
    with (
        deadline.model_deadline(101),
        pytest.raises(deadline.CallDeadlineExceededError),
    ):
        service.resolve_inference_service_capabilities(service_url="http://offline")
    assert not service._configured_model_cache
    assert not service._configured_model_lock.locked()


def test_cloud_dispatch_gets_remaining_budget_and_rejects_late_result(
    monkeypatch, clock
):
    client = cast(Any, embedding.EmbeddingMixin())
    client.config = SimpleNamespace(timeout=5, max_retries=3)
    client._resolve_api_key = lambda *_args, **_kwargs: (None, None, None)
    monkeypatch.setattr(
        embedding, "resolve_service_configured_model", lambda model: model
    )
    monkeypatch.setattr(embedding, "embedding_provider_mode", lambda _model: "cloud")
    monkeypatch.setattr(embedding, "should_use_embedding_service", lambda _model: False)
    monkeypatch.setattr(embedding, "_truncate_for_embedding", lambda text, _model: text)
    monkeypatch.setattr(admission, "_provider_key", lambda _model: "synthetic")
    monkeypatch.setattr(admission, "_semaphores", {})
    sem = admission._get_semaphore("synthetic")
    capacity = sem._value

    def respond(**kwargs):
        assert kwargs["timeout"] == 1
        assert kwargs["num_retries"] == 0
        assert sem._value == capacity - 1
        clock[0] = 101
        assert sem._value == capacity - 1
        return SimpleNamespace(data=[{"index": 0, "embedding": [0.1]}])

    monkeypatch.setattr(embedding.litellm, "embedding", respond)
    with (
        deadline.model_deadline(101),
        pytest.raises(deadline.CallDeadlineExceededError),
    ):
        client.get_embedding("synthetic", model="openai/synthetic")
    assert sem._value == capacity
