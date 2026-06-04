from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.providers.embedding_service_provider import (
    EmbeddingUnavailableError,
    embedding_provider_mode,
    embedding_service_timeout_seconds,
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


def test_local_service_default_timeout_allows_cold_start(monkeypatch) -> None:
    monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_TIMEOUT_MS", raising=False)

    assert embedding_service_timeout_seconds("local_service") == 30


def test_internal_service_keeps_fast_default_timeout(monkeypatch) -> None:
    monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_TIMEOUT_MS", raising=False)

    assert embedding_service_timeout_seconds("internal_service") == 2


def test_embedding_service_timeout_env_overrides_mode_default(monkeypatch) -> None:
    monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_TIMEOUT_MS", "7500")

    assert embedding_service_timeout_seconds("local_service") == 7.5
    assert embedding_service_timeout_seconds("internal_service") == 7.5


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
            assert timeout == 30
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


class TestEmbeddingServiceExceptionScope:
    """Narrow exception scope: real transient errors retry, programming bugs propagate raw.

    The retry loop in ``get_service_embeddings`` previously caught bare
    ``Exception``, meaning a programming bug (``AttributeError``,
    ``TypeError``) would be retried once and then wrapped as
    ``EmbeddingUnavailableError`` — hiding the real defect. These tests pin
    the narrowed scope: only ``httpx.HTTPError``, ``json.JSONDecodeError``,
    and ``ValueError`` (the shape-error signal from
    ``_ordered_embeddings_from_response``) are caught.
    """

    @staticmethod
    def _route_to_local_service(monkeypatch) -> None:
        """Configure env so ``get_service_embeddings`` reaches the retry loop.

        ``REFLEXIO_EMBEDDING_PROVIDER=local_service`` is one of the
        ``_SERVICE_MODES`` that passes the routing guards.
        """
        monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "local_service")
        monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_URL", raising=False)

    def test_programming_bug_propagates_raw_without_retry(self, monkeypatch) -> None:
        """An ``AttributeError`` inside the call body must propagate raw on
        the first attempt — not be caught, retried, and re-wrapped as
        ``EmbeddingUnavailableError``.
        """
        self._route_to_local_service(monkeypatch)
        call_count = {"n": 0}

        class _BrokenClient:
            def __init__(self, timeout: float) -> None:
                self.timeout = timeout

            def __enter__(self) -> _BrokenClient:
                return self

            def __exit__(self, *_args) -> None:
                return None

            def post(self, *_a, **_k):
                call_count["n"] += 1
                raise AttributeError("simulated programming bug")

        monkeypatch.setattr(
            "reflexio.server.llm.providers.embedding_service_provider.httpx.Client",
            _BrokenClient,
        )

        with pytest.raises(AttributeError, match="simulated programming bug"):
            get_service_embeddings(["text"], model="local/nomic-embed-v1.5")
        assert call_count["n"] == 1, "programming bug must NOT be retried"

    def test_httpx_request_error_still_retries(self, monkeypatch) -> None:
        """Network errors continue to retry once then surface as
        ``EmbeddingUnavailableError`` — existing transient-error behavior
        is preserved by the narrowed clause.
        """
        self._route_to_local_service(monkeypatch)
        call_count = {"n": 0}

        class _FlakyClient:
            def __init__(self, timeout: float) -> None:
                self.timeout = timeout

            def __enter__(self) -> _FlakyClient:
                return self

            def __exit__(self, *_args) -> None:
                return None

            def post(self, *_a, **_k):
                call_count["n"] += 1
                raise httpx.RequestError("connection refused")

        monkeypatch.setattr(
            "reflexio.server.llm.providers.embedding_service_provider.httpx.Client",
            _FlakyClient,
        )

        with pytest.raises(EmbeddingUnavailableError):
            get_service_embeddings(["text"], model="local/nomic-embed-v1.5")
        assert call_count["n"] == 2, "transient HTTP error must retry once"
