from __future__ import annotations

from typing import Any

import httpx
import pytest

from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.providers import embedding_service_provider as provider
from reflexio.server.llm.providers.embedding_service_provider import (
    CUSTOM_EMBEDDING_MODEL,
    EmbeddingUnavailableError,
    embedding_provider_mode,
    get_service_embeddings,
    remote_inference_service_configured,
    resolve_inference_service_capabilities,
    resolve_service_configured_model,
    resolve_service_configured_reranker_model,
)


class _Response:
    def __init__(self, body: dict[str, Any], status_code: int = 200) -> None:
        self._body = body
        self.status_code = status_code
        self.request = httpx.Request("GET", "http://inference/health")
        self.text = str(body)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "failed",
                request=self.request,
                response=httpx.Response(self.status_code, request=self.request),
            )

    def json(self) -> dict[str, Any]:
        return self._body


def _reset_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider, "_configured_model_cache", {})


def test_local_models_always_route_to_separate_service(monkeypatch) -> None:
    monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_URL", raising=False)

    assert embedding_provider_mode("local/minilm-l6-v2") == "local_service"
    assert embedding_provider_mode("local/nomic-embed-text-v1.5") == "local_service"
    assert embedding_provider_mode(CUSTOM_EMBEDDING_MODEL) == "local_service"


def test_whitespace_service_url_is_treated_as_unset(monkeypatch) -> None:
    monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_URL", "  \t")
    monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)

    assert embedding_provider_mode(CUSTOM_EMBEDDING_MODEL) == "local_service"
    assert remote_inference_service_configured() is False


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://127.0.0.1:8072", False),
        ("http://[::1]:8072", False),
        ("http://localhost:8072", False),
        ("http://inference.internal:8089", True),
    ],
)
def test_remote_inference_service_detection(
    monkeypatch, url: str, expected: bool
) -> None:
    monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_URL", url)

    assert remote_inference_service_configured() is expected


def test_removed_inprocess_mode_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "inprocess")

    with pytest.raises(EmbeddingUnavailableError, match="Invalid"):
        embedding_provider_mode("local/minilm-l6-v2")


def test_explicit_cloud_models_bypass_configured_service(monkeypatch) -> None:
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "internal_service")
    monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_URL", "http://inference")

    assert embedding_provider_mode("text-embedding-3-small") == "cloud"


def test_cloud_embedding_configuration_still_routes_custom_to_shared_service(
    monkeypatch,
) -> None:
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "cloud")
    monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_URL", raising=False)

    assert embedding_provider_mode(CUSTOM_EMBEDDING_MODEL) == "local_service"
    assert embedding_provider_mode("text-embedding-3-small") == "cloud"


def test_one_health_request_caches_both_models(monkeypatch) -> None:
    calls = 0

    class _Client:
        def get(self, *_args, **_kwargs) -> _Response:
            nonlocal calls
            calls += 1
            return _Response(
                {
                    "configured_model": "local/multilingual-e5-small",
                    "configured_reranker_model": (
                        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
                    ),
                    "reranker_enabled": True,
                    "reranker_ready": True,
                }
            )

    monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_URL", "http://inference")
    monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
    _reset_cache(monkeypatch)
    monkeypatch.setattr(provider, "_http_client", lambda: _Client())

    assert resolve_service_configured_model() == "local/multilingual-e5-small"
    assert resolve_service_configured_reranker_model() == (
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    )
    capabilities = resolve_inference_service_capabilities()
    assert capabilities.reranker_ready is True
    assert calls == 1


def test_health_cache_is_scoped_by_service_url(monkeypatch) -> None:
    calls: list[str] = []

    class _Client:
        def get(self, url: str, **_kwargs) -> _Response:
            calls.append(url)
            model = "local/a" if "first" in url else "local/b"
            return _Response({"configured_model": model})

    _reset_cache(monkeypatch)
    monkeypatch.setattr(provider, "_http_client", lambda: _Client())
    monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_URL", "http://first")
    assert resolve_service_configured_model() == "local/a"
    monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_URL", "http://second")
    assert resolve_service_configured_model() == "local/b"
    assert calls == ["http://first/health", "http://second/health"]


def test_missing_reranker_model_does_not_break_embedding_discovery(monkeypatch) -> None:
    class _Client:
        def get(self, *_args, **_kwargs) -> _Response:
            return _Response({"configured_model": "local/minilm-l6-v2"})

    monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_URL", "http://inference")
    _reset_cache(monkeypatch)
    monkeypatch.setattr(provider, "_http_client", lambda: _Client())

    assert resolve_service_configured_model() == "local/minilm-l6-v2"
    with pytest.raises(EmbeddingUnavailableError, match="configured_reranker_model"):
        resolve_service_configured_reranker_model()


def test_embedding_response_model_is_ignored_and_indices_are_ordered(
    monkeypatch,
) -> None:
    class _Client:
        def post(self, url: str, *, json: dict[str, Any], timeout: float) -> _Response:
            assert url == "http://inference/v1/embeddings"
            assert json["model"] == "local/multilingual-e5-small"
            assert timeout == 2
            return _Response(
                {
                    "model": "ignored/model",
                    "data": [
                        {"index": 1, "embedding": [0.3, 0.4]},
                        {"index": 0, "embedding": [0.1, 0.2]},
                    ],
                }
            )

    monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_URL", "http://inference")
    monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setattr(provider, "_http_client", lambda: _Client())

    assert get_service_embeddings(
        ["中文", "English"], model="local/multilingual-e5-small"
    ) == [[0.1, 0.2], [0.3, 0.4]]


def test_embedding_service_failure_never_constructs_local_model(monkeypatch) -> None:
    class _Client:
        def post(self, *_args, **_kwargs) -> _Response:
            raise httpx.ConnectError("down")

    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "local_service")
    monkeypatch.setattr(provider, "_http_client", lambda: _Client())
    monkeypatch.setattr(provider.time, "sleep", lambda _seconds: None)

    with pytest.raises(EmbeddingUnavailableError, match="unavailable"):
        get_service_embeddings(["text"], model="local/minilm-l6-v2")


def test_litellm_local_model_uses_http_provider(monkeypatch) -> None:
    calls: list[tuple[list[str], str]] = []
    monkeypatch.setattr(
        "reflexio.server.llm._litellm_embedding.get_service_embeddings",
        lambda texts, *, model, dimensions: (  # noqa: ARG005
            calls.append((texts, model)) or [[0.1, 0.2]]
        ),
    )
    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o"))

    assert client.get_embedding("hello", model="local/minilm-l6-v2") == [0.1, 0.2]
    assert calls == [(["hello"], "local/minilm-l6-v2")]


@pytest.mark.parametrize(
    "data",
    [
        None,
        [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}],
        [{"index": "0", "embedding": [1.0]}],
        [{"index": 0, "embedding": "bad"}],
    ],
)
def test_malformed_embedding_response_fails_closed(monkeypatch, data: Any) -> None:
    class _Client:
        def post(self, *_args, **_kwargs) -> _Response:
            return _Response({"data": data})

    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "local_service")
    monkeypatch.setattr(provider, "_http_client", lambda: _Client())

    with pytest.raises(EmbeddingUnavailableError):
        get_service_embeddings(["text"], model="local/minilm-l6-v2")


def test_capability_discovery_honours_off_mode(monkeypatch) -> None:
    """`off` must short-circuit discovery, not attempt a connection.

    The mode was derived locally here, so `off` -- a validated member of
    _VALID_MODES -- could not reach this path: callers that had explicitly
    disabled embedding still got a connection error against the service they
    had just turned off. Asserts on the message because the failure mode being
    guarded is a *misleading* error, not merely an error.
    """
    from reflexio.server.llm.providers import embedding_service_provider as provider

    monkeypatch.setattr(provider, "_configured_model_cache", {})
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "off")

    def _fail_if_called(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("discovery attempted an HTTP call while provider=off")

    monkeypatch.setattr(provider, "_http_client", _fail_if_called)

    with pytest.raises(provider.EmbeddingUnavailableError, match="disabled"):
        provider.resolve_inference_service_capabilities()
