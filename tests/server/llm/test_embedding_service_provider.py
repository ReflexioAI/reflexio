from __future__ import annotations

from unittest.mock import patch

import pytest

from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.providers.embedding_service_provider import (
    EmbeddingUnavailableError,
    embedding_provider_mode,
    get_service_embeddings,
)


def test_claude_smart_legacy_flag_defaults_to_local_service(monkeypatch) -> None:
    monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_URL", raising=False)
    monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", "1")

    assert embedding_provider_mode("local/nomic-embed-v1.5") == "local_service"


def test_claude_smart_legacy_flag_requires_one(monkeypatch) -> None:
    monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_URL", raising=False)
    monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", "true")

    assert embedding_provider_mode("local/nomic-embed-v1.5") == "inprocess"


def test_local_model_without_opt_in_preserves_inprocess_mode(monkeypatch) -> None:
    monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_URL", raising=False)
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", raising=False)

    assert embedding_provider_mode("local/minilm-l6-v2") == "inprocess"


def test_litellm_client_routes_local_service_embeddings(monkeypatch) -> None:
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "local_service")
    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o"))

    with patch(
        "reflexio.server.llm.litellm_client.get_service_embeddings",
        return_value=[[0.1, 0.2]],
    ) as mocked:
        assert client.get_embedding("hello", model="local/nomic-embed-v1.5") == [
            0.1,
            0.2,
        ]

    mocked.assert_called_once_with(
        ["hello"], model="local/nomic-embed-v1.5", dimensions=None
    )


def test_off_mode_raises_typed_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "off")
    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o"))

    with pytest.raises(EmbeddingUnavailableError):
        client.get_embedding("hello", model="local/nomic-embed-v1.5")


def test_service_response_is_sorted_by_index(monkeypatch) -> None:
    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            }

    class _Client:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url: str, json: dict) -> _Response:  # noqa: A002
            assert url == "http://127.0.0.1:8072/v1/embeddings"
            assert json["model"] == "local/nomic-embed-v1.5"
            return _Response()

    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "local_service")
    monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_URL", raising=False)
    monkeypatch.setattr(
        "reflexio.server.llm.providers.embedding_service_provider.httpx.Client",
        _Client,
    )

    assert get_service_embeddings(["a", "b"], model="local/nomic-embed-v1.5") == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]


def test_service_response_rejects_index_mismatch(monkeypatch) -> None:
    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 1, "embedding": [0.5, 0.6]},
                ]
            }

    class _Client:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url: str, json: dict) -> _Response:  # noqa: A002, ARG002
            return _Response()

    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "local_service")
    monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_URL", raising=False)
    monkeypatch.setattr(
        "reflexio.server.llm.providers.embedding_service_provider.httpx.Client",
        _Client,
    )

    with pytest.raises(EmbeddingUnavailableError, match="duplicate index 1"):
        get_service_embeddings(["a", "b"], model="local/nomic-embed-v1.5")
