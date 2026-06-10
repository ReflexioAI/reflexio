from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMClientError,
    LiteLLMConfig,
)
from reflexio.server.llm.providers import embedding_service_provider as esp
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
    monkeypatch.setattr(esp, "_local_service_probe_cache", None)
    monkeypatch.setattr(
        esp.httpx,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            httpx.RequestError("connection refused")
        ),
    )

    assert embedding_provider_mode("local/minilm-l6-v2") == "inprocess"


def test_local_model_auto_uses_reachable_matching_daemon(monkeypatch) -> None:
    class _HealthResponse:
        status_code = 200

        def json(self) -> dict[str, str]:
            return {"active_model": "local/nomic-embed-text-v1.5"}

    monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_URL", raising=False)
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", raising=False)
    monkeypatch.setattr(esp, "_local_service_probe_cache", None)
    monkeypatch.setattr(esp.httpx, "get", lambda *_args, **_kwargs: _HealthResponse())

    assert embedding_provider_mode("local/nomic-embed-text-v1.5") == "local_service"


def test_local_model_auto_avoids_reachable_mismatched_daemon(monkeypatch) -> None:
    class _HealthResponse:
        status_code = 200

        def json(self) -> dict[str, str]:
            return {"active_model": "local/nomic-embed-text-v1.5"}

    monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_URL", raising=False)
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", raising=False)
    monkeypatch.setattr(esp, "_local_service_probe_cache", None)
    monkeypatch.setattr(esp.httpx, "get", lambda *_args, **_kwargs: _HealthResponse())

    assert embedding_provider_mode("local/minilm-l6-v2") == "inprocess"


def test_local_service_probe_timeout_env_is_honored(monkeypatch) -> None:
    class _HealthResponse:
        status_code = 200

        def json(self) -> dict[str, str]:
            return {"active_model": "local/nomic-embed-text-v1.5"}

    observed: dict[str, float] = {}

    def _get(_url: str, *, timeout: float) -> _HealthResponse:
        observed["timeout"] = timeout
        return _HealthResponse()

    monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_URL", raising=False)
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", raising=False)
    monkeypatch.setenv("REFLEXIO_EMBEDDING_LOCAL_SERVICE_PROBE_TIMEOUT_MS", "1250")
    monkeypatch.setattr(esp, "_local_service_probe_cache", None)
    monkeypatch.setattr(esp.httpx, "get", _get)

    assert embedding_provider_mode("local/nomic-embed-text-v1.5") == "local_service"
    assert observed["timeout"] == 1.25


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


def test_local_service_daemon_host_override_changes_url(monkeypatch) -> None:
    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

    class _Client:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url: str, json: dict) -> _Response:  # noqa: A002
            assert url == "http://embedding.internal:8072/v1/embeddings"
            assert json["model"] == "local/nomic-embed-v1.5"
            return _Response()

    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "local_service")
    monkeypatch.setenv("REFLEXIO_EMBEDDING_DAEMON_HOST", "embedding.internal")
    monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_URL", raising=False)
    monkeypatch.setattr(
        "reflexio.server.llm.providers.embedding_service_provider.httpx.Client",
        _Client,
    )

    assert get_service_embeddings(["a"], model="local/nomic-embed-v1.5") == [[0.1, 0.2]]


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

    def test_connect_error_retries_once(self, monkeypatch) -> None:
        """A connection-establishment failure (request never reached the
        server) retries once, then surfaces as ``EmbeddingUnavailableError``.
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
                raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(
            "reflexio.server.llm.providers.embedding_service_provider.httpx.Client",
            _FlakyClient,
        )

        with pytest.raises(EmbeddingUnavailableError):
            get_service_embeddings(["text"], model="local/nomic-embed-v1.5")
        assert call_count["n"] == 2, "connection error must retry once"

    def test_read_timeout_does_not_retry(self, monkeypatch) -> None:
        """A read timeout means the server already received the request and may
        still be encoding it. Retrying would queue a second identical encode and
        amplify load on a saturated daemon, so it must fail fast (one call).
        """
        self._route_to_local_service(monkeypatch)
        call_count = {"n": 0}

        class _SlowClient:
            def __init__(self, timeout: float) -> None:
                self.timeout = timeout

            def __enter__(self) -> _SlowClient:
                return self

            def __exit__(self, *_args) -> None:
                return None

            def post(self, *_a, **_k):
                call_count["n"] += 1
                raise httpx.ReadTimeout("timed out")

        monkeypatch.setattr(
            "reflexio.server.llm.providers.embedding_service_provider.httpx.Client",
            _SlowClient,
        )

        with pytest.raises(EmbeddingUnavailableError):
            get_service_embeddings(["text"], model="local/nomic-embed-v1.5")
        assert call_count["n"] == 1, "read timeout must NOT be retried"


class TestRequestChunking:
    """``get_service_embeddings`` bounds each request to
    ``REFLEXIO_EMBEDDING_SERVICE_MAX_TEXTS_PER_REQUEST`` texts and concatenates
    the per-chunk results in input order, so a single large publish cannot
    exceed the client read timeout.
    """

    @staticmethod
    def _route_to_local_service(monkeypatch) -> None:
        monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "local_service")
        monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_URL", raising=False)

    @staticmethod
    def _client_recording_payloads(payloads: list[dict]) -> type:
        """A fake httpx.Client that records each POST payload and answers with
        one embedding per input text, derived from the text's ``t<n>`` suffix
        so concatenation order is observable.
        """

        class _Response:
            @staticmethod
            def raise_for_status() -> None:
                return None

            @staticmethod
            def json() -> dict:
                texts = payloads[-1]["input"]
                return {
                    "data": [
                        {"index": i, "embedding": [float(text[1:])]}
                        for i, text in enumerate(texts)
                    ]
                }

        class _Client:
            def __init__(self, timeout: float) -> None:
                self.timeout = timeout

            def __enter__(self) -> _Client:
                return self

            def __exit__(self, *_args) -> None:
                return None

            def post(self, _url, *, json):
                payloads.append(json)
                return _Response()

        return _Client

    def test_large_input_is_chunked_and_concatenated_in_order(
        self, monkeypatch
    ) -> None:
        self._route_to_local_service(monkeypatch)
        monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_MAX_TEXTS_PER_REQUEST", "2")
        payloads: list[dict] = []
        monkeypatch.setattr(
            "reflexio.server.llm.providers.embedding_service_provider.httpx.Client",
            self._client_recording_payloads(payloads),
        )

        result = get_service_embeddings(
            ["t0", "t1", "t2", "t3", "t4"], model="local/nomic-embed-v1.5"
        )

        assert [p["input"] for p in payloads] == [["t0", "t1"], ["t2", "t3"], ["t4"]]
        assert result == [[0.0], [1.0], [2.0], [3.0], [4.0]]

    def test_chunk_failure_surfaces_embedding_unavailable(self, monkeypatch) -> None:
        """A failure on a later chunk discards earlier partial results and
        raises ``EmbeddingUnavailableError`` — callers never see a short list.
        """
        self._route_to_local_service(monkeypatch)
        monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_MAX_TEXTS_PER_REQUEST", "2")
        payloads: list[dict] = []
        good_client = self._client_recording_payloads(payloads)

        class _FailsOnSecondChunk(good_client):
            def post(self, _url, *, json):
                if len(payloads) >= 1:
                    raise httpx.ReadTimeout("timed out")
                return super().post(_url, json=json)

        monkeypatch.setattr(
            "reflexio.server.llm.providers.embedding_service_provider.httpx.Client",
            _FailsOnSecondChunk,
        )

        with pytest.raises(EmbeddingUnavailableError):
            get_service_embeddings(["t0", "t1", "t2"], model="local/nomic-embed-v1.5")
        assert [p["input"] for p in payloads] == [["t0", "t1"]]

    @pytest.mark.parametrize("raw", ["abc", "0", "-3"])
    def test_invalid_max_texts_env_falls_back_to_default(
        self, monkeypatch, raw: str
    ) -> None:
        monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_MAX_TEXTS_PER_REQUEST", raw)
        assert esp._max_texts_per_request() == esp._DEFAULT_MAX_TEXTS_PER_REQUEST

    def test_valid_max_texts_env_is_honored(self, monkeypatch) -> None:
        monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_MAX_TEXTS_PER_REQUEST", "8")
        assert esp._max_texts_per_request() == 8


def test_nomic_inprocess_fallback_uses_nomic_embedder(monkeypatch) -> None:
    class _Nomic:
        def embed(self, texts: list[str]) -> list[list[float]]:
            assert texts == ["hello"]
            return [[0.1] * 512]

    class _NomicFactory:
        @staticmethod
        def get() -> _Nomic:
            return _Nomic()

    monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_URL", raising=False)
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", raising=False)
    monkeypatch.setattr(esp, "_local_service_probe_cache", None)
    monkeypatch.setattr(
        esp.httpx,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            httpx.RequestError("connection refused")
        ),
    )
    monkeypatch.setattr(
        "reflexio.server.llm.litellm_client.NomicEmbedder",
        _NomicFactory,
    )

    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o"))

    result = client.get_embedding("hello", model="local/nomic-embed-text-v1.5")

    assert len(result) == 512
    assert result[0] == 0.1


def test_nomic_inprocess_fallback_does_not_use_minilm(monkeypatch) -> None:
    class _BrokenNomic:
        def embed(self, texts: list[str]) -> list[list[float]]:  # noqa: ARG002
            raise RuntimeError("nomic unavailable")

    class _NomicFactory:
        @staticmethod
        def get() -> _BrokenNomic:
            return _BrokenNomic()

    class _MiniLMFactory:
        @staticmethod
        def get():
            raise AssertionError("MiniLM must not handle local/nomic-* models")

    monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_URL", raising=False)
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", raising=False)
    monkeypatch.setattr(esp, "_local_service_probe_cache", None)
    monkeypatch.setattr(
        esp.httpx,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            httpx.RequestError("connection refused")
        ),
    )
    monkeypatch.setattr(
        "reflexio.server.llm.litellm_client.NomicEmbedder",
        _NomicFactory,
    )
    monkeypatch.setattr(
        "reflexio.server.llm.litellm_client.LocalEmbedder",
        _MiniLMFactory,
    )

    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o"))

    with pytest.raises(LiteLLMClientError, match="Nomic embedding generation failed"):
        client.get_embedding("hello", model="local/nomic-embed-text-v1.5")
