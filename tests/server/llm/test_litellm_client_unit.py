"""Unit tests for the LiteLLM client wrapper.

Tests cover initialization, response generation, retry logic, error classification,
embeddings, structured output parsing, config management, image handling, and
prompt caching. All LiteLLM SDK calls are mocked -- no real API requests are made.
"""

import base64
import json
import logging
import struct
import tempfile
import zlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from reflexio.models.config_schema import (
    AnthropicConfig,
    APIKeyConfig,
    AzureOpenAIConfig,
    CustomEndpointConfig,
    DashScopeConfig,
    DeepSeekConfig,
    GeminiConfig,
    MiniMaxConfig,
    MoonshotConfig,
    OpenRouterConfig,
    XAIConfig,
    ZAIConfig,
)
from reflexio.models.config_schema import (
    OpenAIConfig as CommonsOpenAIConfig,
)
from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMClientError,
    LiteLLMConfig,
    _get_embedding_encoding,
    _get_embedding_limit,
    _truncate_for_embedding,
    create_litellm_client,
)

# ---------------------------------------------------------------------------
# Pydantic models used for structured-output tests
# ---------------------------------------------------------------------------


class SampleResponse(BaseModel):
    answer: str
    score: int


class MathResult(BaseModel):
    result: int
    explanation: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_completion_response(content: str = "Hello world") -> MagicMock:
    """Build a mock litellm.completion response."""
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    resp.usage.prompt_tokens_details = None
    resp.usage.cache_creation_input_tokens = None
    resp.usage.cache_read_input_tokens = None
    return resp


def _make_embedding_response(
    embedding: list[float] | None = None,
) -> MagicMock:
    """Build a mock litellm.embedding response."""
    resp = MagicMock()
    resp.data = [{"embedding": embedding or [0.1, 0.2, 0.3], "index": 0}]
    return resp


def _make_batch_embedding_response(
    embeddings: list[list[float]] | None = None,
) -> MagicMock:
    """Build a mock litellm.embedding response for batch."""
    resp = MagicMock()
    if embeddings is None:
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
    resp.data = [{"embedding": emb, "index": i} for i, emb in enumerate(embeddings)]
    return resp


def _build_client(
    config: LiteLLMConfig | None = None,
) -> LiteLLMClient:
    """Instantiate a LiteLLMClient without touching real APIs."""
    if config is None:
        config = LiteLLMConfig(model="gpt-4o")
    return LiteLLMClient(config)


def _create_minimal_png(
    width: int = 2, height: int = 2, color: tuple = (255, 0, 0)
) -> bytes:
    """Create a minimal valid PNG image in memory."""

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk_len = struct.pack(">I", len(data))
        chunk_crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return chunk_len + chunk_type + data + chunk_crc

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = png_chunk(b"IHDR", ihdr_data)
    raw_data = b""
    for _ in range(height):
        raw_data += b"\x00"
        for _ in range(width):
            raw_data += bytes(color)
    compressed_data = zlib.compress(raw_data)
    idat = png_chunk(b"IDAT", compressed_data)
    iend = png_chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


# ===================================================================
# Init tests
# ===================================================================


class TestInit:
    """Initialization of LiteLLMClient."""

    def test_basic_init(self):
        config = LiteLLMConfig(model="gpt-4o")
        client = LiteLLMClient(config)

        assert client.config is config
        assert client.get_model() == "gpt-4o"

    def test_init_no_api_key_config(self):
        config = LiteLLMConfig(model="gpt-4o")
        client = LiteLLMClient(config)

        assert client._api_key is None
        assert client._api_base is None
        assert client._api_version is None

    def test_init_with_openai_api_key_config(self):
        api_key_config = APIKeyConfig(openai=CommonsOpenAIConfig(api_key="sk-test-key"))
        config = LiteLLMConfig(model="gpt-4o", api_key_config=api_key_config)
        client = LiteLLMClient(config)

        assert client._api_key == "sk-test-key"
        assert client._api_base is None

    def test_init_with_azure_config(self):
        azure = AzureOpenAIConfig(
            api_key="az-key",
            endpoint="https://myresource.openai.azure.com/",
            api_version="2024-02-15-preview",
        )
        api_key_config = APIKeyConfig(openai=CommonsOpenAIConfig(azure_config=azure))
        config = LiteLLMConfig(model="azure/gpt-4", api_key_config=api_key_config)
        client = LiteLLMClient(config)

        assert client._api_key == "az-key"
        assert "myresource" in client._api_base
        assert client._api_version == "2024-02-15-preview"

    def test_init_with_anthropic_config(self):
        api_key_config = APIKeyConfig(anthropic=AnthropicConfig(api_key="ant-key"))
        config = LiteLLMConfig(
            model="claude-3-5-sonnet-20241022", api_key_config=api_key_config
        )
        client = LiteLLMClient(config)

        assert client._api_key == "ant-key"

    def test_init_with_custom_provider_gemini(self):
        api_key_config = APIKeyConfig(gemini=GeminiConfig(api_key="gem-key"))
        config = LiteLLMConfig(model="gemini/gemini-pro", api_key_config=api_key_config)
        client = LiteLLMClient(config)

        assert client._api_key == "gem-key"

    def test_init_with_openrouter_config(self):
        api_key_config = APIKeyConfig(openrouter=OpenRouterConfig(api_key="or-key"))
        config = LiteLLMConfig(
            model="openrouter/openai/gpt-4o", api_key_config=api_key_config
        )
        client = LiteLLMClient(config)

        assert client._api_key == "or-key"

    def test_init_with_minimax_config(self):
        api_key_config = APIKeyConfig(minimax=MiniMaxConfig(api_key="mm-key"))
        config = LiteLLMConfig(
            model="minimax/minimax-01", api_key_config=api_key_config
        )
        client = LiteLLMClient(config)

        assert client._api_key == "mm-key"

    def test_init_with_custom_endpoint(self):
        api_key_config = APIKeyConfig(
            custom_endpoint=CustomEndpointConfig(
                model="my-model",
                api_key="ce-key",
                api_base="https://custom.api.com/v1",
            )
        )
        config = LiteLLMConfig(model="gpt-4o", api_key_config=api_key_config)
        client = LiteLLMClient(config)

        assert client._api_key == "ce-key"
        assert client._api_base == "https://custom.api.com/v1"


# ===================================================================
# _resolve_api_key tests
# ===================================================================


class TestResolveApiKey:
    """Tests for _resolve_api_key across different providers."""

    def test_no_api_key_config_returns_nones(self):
        client = _build_client()
        key, base, version = client._resolve_api_key()
        assert key is None
        assert base is None
        assert version is None

    def test_custom_endpoint_priority_for_non_embedding(self):
        api_key_config = APIKeyConfig(
            custom_endpoint=CustomEndpointConfig(
                model="custom-model",
                api_key="ce-key",
                api_base="https://custom.api.com/v1",
            ),
            openai=CommonsOpenAIConfig(api_key="sk-openai"),
        )
        config = LiteLLMConfig(model="gpt-4o", api_key_config=api_key_config)
        client = LiteLLMClient(config)

        key, base, version = client._resolve_api_key(for_embedding=False)
        assert key == "ce-key"
        assert base == "https://custom.api.com/v1"

    def test_custom_endpoint_skipped_for_embedding(self):
        api_key_config = APIKeyConfig(
            custom_endpoint=CustomEndpointConfig(
                model="custom-model",
                api_key="ce-key",
                api_base="https://custom.api.com/v1",
            ),
            openai=CommonsOpenAIConfig(api_key="sk-openai"),
        )
        config = LiteLLMConfig(model="gpt-4o", api_key_config=api_key_config)
        client = LiteLLMClient(config)

        key, base, version = client._resolve_api_key(for_embedding=True)
        assert key == "sk-openai"
        assert base is None

    def test_resolve_for_different_model(self):
        api_key_config = APIKeyConfig(
            anthropic=AnthropicConfig(api_key="ant-key"),
            openai=CommonsOpenAIConfig(api_key="sk-openai"),
        )
        config = LiteLLMConfig(model="gpt-4o", api_key_config=api_key_config)
        client = LiteLLMClient(config)

        key, _, _ = client._resolve_api_key(model="claude-3-5-sonnet")
        assert key == "ant-key"

    def test_resolve_unknown_model_uses_openai(self):
        api_key_config = APIKeyConfig(openai=CommonsOpenAIConfig(api_key="sk-openai"))
        config = LiteLLMConfig(
            model="some-unknown-model", api_key_config=api_key_config
        )
        client = LiteLLMClient(config)

        key, _, _ = client._resolve_api_key()
        assert key == "sk-openai"

    def test_resolve_returns_nones_when_provider_not_configured(self):
        """When a gemini model is used but no gemini config exists."""
        api_key_config = APIKeyConfig(openai=CommonsOpenAIConfig(api_key="sk-openai"))
        config = LiteLLMConfig(model="gemini/gemini-pro", api_key_config=api_key_config)
        client = LiteLLMClient(config)

        key, base, version = client._resolve_api_key()
        assert key is None
        assert base is None
        assert version is None

    @pytest.mark.parametrize(
        ("model", "cfg_attr", "cfg_cls", "expected_key"),
        [
            ("deepseek/deepseek-chat", "deepseek", DeepSeekConfig, "ds-key"),
            ("zai/glm-4", "zai", ZAIConfig, "zai-key"),
            ("moonshot/moonshot-v1", "moonshot", MoonshotConfig, "ms-key"),
            ("xai/grok-1", "xai", XAIConfig, "xai-key"),
        ],
    )
    def test_resolve_simple_provider_prefixes(
        self, model: str, cfg_attr: str, cfg_cls: type, expected_key: str
    ):
        """Each simple-prefix provider returns its api_key with no base or version."""
        api_key_config = APIKeyConfig(**{cfg_attr: cfg_cls(api_key=expected_key)})
        config = LiteLLMConfig(model=model, api_key_config=api_key_config)
        client = LiteLLMClient(config)

        key, base, version = client._resolve_api_key()
        assert key == expected_key
        assert base is None
        assert version is None

    def test_resolve_dashscope_returns_api_base(self):
        """DashScope provider returns api_key and optional api_base."""
        api_key_config = APIKeyConfig(
            dashscope=DashScopeConfig(
                api_key="ds-qwen-key",
                api_base="https://dashscope-intl.aliyuncs.com/v1",
            )
        )
        config = LiteLLMConfig(
            model="dashscope/qwen-turbo", api_key_config=api_key_config
        )
        client = LiteLLMClient(config)

        key, base, version = client._resolve_api_key()
        assert key == "ds-qwen-key"
        assert base == "https://dashscope-intl.aliyuncs.com/v1"
        assert version is None

    def test_resolve_dashscope_unconfigured_returns_nones(self):
        """DashScope prefix with no dashscope config returns all Nones."""
        api_key_config = APIKeyConfig(openai=CommonsOpenAIConfig(api_key="sk-openai"))
        config = LiteLLMConfig(
            model="dashscope/qwen-turbo", api_key_config=api_key_config
        )
        client = LiteLLMClient(config)

        key, base, version = client._resolve_api_key()
        assert key is None
        assert base is None
        assert version is None

    @pytest.mark.parametrize(
        ("model", "prefix"),
        [
            ("deepseek/deepseek-chat", "deepseek"),
            ("zai/glm-4", "zai"),
            ("moonshot/moonshot-v1", "moonshot"),
            ("xai/grok-1", "xai"),
        ],
    )
    def test_resolve_unconfigured_simple_provider_returns_nones(
        self, model: str, prefix: str
    ):
        """Simple-prefix provider with no matching config returns all Nones."""
        api_key_config = APIKeyConfig(openai=CommonsOpenAIConfig(api_key="sk-openai"))
        config = LiteLLMConfig(model=model, api_key_config=api_key_config)
        client = LiteLLMClient(config)

        key, base, version = client._resolve_api_key()
        assert key is None
        assert base is None
        assert version is None


# ===================================================================
# generate_response tests
# ===================================================================


class TestGenerateResponse:
    """Tests for generate_response (single-prompt entry point)."""

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_text_only_prompt(self, mock_completion):
        mock_completion.return_value = _make_completion_response("Paris")
        client = _build_client()

        result = client.generate_response("What is the capital of France?")

        assert result == "Paris"
        call_kwargs = mock_completion.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is the capital of France?"

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_with_system_message(self, mock_completion):
        mock_completion.return_value = _make_completion_response("Yes")
        client = _build_client()

        client.generate_response("Hello", system_message="You are helpful.")

        call_kwargs = mock_completion.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"] == "user"

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_structured_output_pydantic(self, mock_completion):
        json_str = json.dumps({"answer": "ok", "score": 5})
        mock_completion.return_value = _make_completion_response(json_str)
        client = _build_client()

        result = client.generate_response("test", response_format=SampleResponse)

        assert isinstance(result, SampleResponse)
        assert result.answer == "ok"
        assert result.score == 5

    def test_invalid_response_format_raises(self):
        client = _build_client()
        with pytest.raises(LiteLLMClientError, match="Pydantic BaseModel class"):
            client.generate_response("test", response_format={"type": "json_object"})

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_parse_structured_output_disabled(self, mock_completion):
        """When parse_structured_output=False, raw content string should be returned."""
        json_str = json.dumps({"answer": "ok", "score": 5})
        mock_completion.return_value = _make_completion_response(json_str)
        client = _build_client()

        result = client.generate_response(
            "test",
            response_format=SampleResponse,
            parse_structured_output=False,
        )

        assert isinstance(result, str)
        assert result == json_str


# ===================================================================
# generate_chat_response tests
# ===================================================================


class TestGenerateChatResponse:
    """Tests for generate_chat_response (messages-list entry point)."""

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_valid_messages(self, mock_completion):
        mock_completion.return_value = _make_completion_response("Hi there")
        client = _build_client()

        messages = [
            {"role": "system", "content": "Be polite"},
            {"role": "user", "content": "Hello"},
        ]
        result = client.generate_chat_response(messages)

        assert result == "Hi there"

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_system_message_prepended(self, mock_completion):
        mock_completion.return_value = _make_completion_response("ok")
        client = _build_client()

        messages = [{"role": "user", "content": "Hi"}]
        client.generate_chat_response(messages, system_message="Be brief")

        call_kwargs = mock_completion.call_args.kwargs
        sent_msgs = call_kwargs["messages"]
        assert sent_msgs[0]["role"] == "system"
        assert sent_msgs[0]["content"] == "Be brief"
        assert sent_msgs[1]["role"] == "user"

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_system_message_merged_with_existing(self, mock_completion):
        mock_completion.return_value = _make_completion_response("ok")
        client = _build_client()

        messages = [
            {"role": "system", "content": "Existing system msg"},
            {"role": "user", "content": "Hi"},
        ]
        client.generate_chat_response(messages, system_message="Prepend this")

        call_kwargs = mock_completion.call_args.kwargs
        sent_msgs = call_kwargs["messages"]
        assert sent_msgs[0]["role"] == "system"
        assert "Prepend this" in sent_msgs[0]["content"]
        assert "Existing system msg" in sent_msgs[0]["content"]

    def test_invalid_response_format_raises(self):
        client = _build_client()
        messages = [{"role": "user", "content": "Hi"}]
        with pytest.raises(LiteLLMClientError, match="Pydantic BaseModel class"):
            client.generate_chat_response(messages, response_format="not_a_model")


# ===================================================================
# _make_request / retry logic tests
# ===================================================================


class TestMakeRequestRetry:
    """Tests for the retry wrapper around litellm.completion."""

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_success_on_first_try(self, mock_completion):
        mock_completion.return_value = _make_completion_response("ok")
        client = _build_client()

        result = client._make_request([{"role": "user", "content": "hi"}])

        assert result == "ok"
        assert mock_completion.call_count == 1

    @patch("reflexio.server.llm.litellm_client.time.sleep")
    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_success_on_retry(self, mock_completion, mock_sleep):
        mock_completion.side_effect = [
            RuntimeError("temporary failure"),
            _make_completion_response("ok"),
        ]
        config = LiteLLMConfig(model="gpt-4o", max_retries=2, retry_delay=0.01)
        client = LiteLLMClient(config)

        result = client._make_request([{"role": "user", "content": "hi"}])

        assert result == "ok"
        assert mock_completion.call_count == 2
        mock_sleep.assert_called_once()

    @patch("reflexio.server.llm.litellm_client.time.sleep")
    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_all_retries_fail(self, mock_completion, mock_sleep):
        mock_completion.side_effect = RuntimeError("boom")
        config = LiteLLMConfig(model="gpt-4o", max_retries=2, retry_delay=0.01)
        client = LiteLLMClient(config)

        with pytest.raises(LiteLLMClientError, match="failed after 2 retries"):
            client._make_request([{"role": "user", "content": "hi"}])

        assert mock_completion.call_count == 2

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_non_retryable_error_stops_immediately(self, mock_completion):
        mock_completion.side_effect = RuntimeError("invalid_api_key: bad key")
        config = LiteLLMConfig(model="gpt-4o", max_retries=3)
        client = LiteLLMClient(config)

        with pytest.raises(LiteLLMClientError, match="invalid_api_key"):
            client._make_request([{"role": "user", "content": "hi"}])

        assert mock_completion.call_count == 1


# ===================================================================
# _is_non_retryable_error tests
# ===================================================================


class TestIsNonRetryableError:
    """Verify classification of errors as retryable vs non-retryable."""

    @pytest.fixture()
    def client(self):
        return _build_client()

    @pytest.mark.parametrize(
        "error_str",
        [
            "invalid_api_key provided",
            "unauthorized access",
            "permission_denied for resource",
            "quota_exceeded for project",
            "billing account inactive",
            "invalid_request: bad payload",
            "authentication failed",
            "forbidden resource",
            "rate_limit exceeded",
        ],
    )
    def test_non_retryable_patterns(self, client, error_str):
        assert client._is_non_retryable_error(error_str) is True

    @pytest.mark.parametrize(
        "error_str",
        [
            "connection timed out",
            "internal server error",
            "service unavailable",
            "network error",
        ],
    )
    def test_retryable_patterns(self, client, error_str):
        assert client._is_non_retryable_error(error_str) is False


# ===================================================================
# get_embedding tests
# ===================================================================


class TestGetEmbedding:
    """Tests for the single-text embedding endpoint."""

    @patch("reflexio.server.llm.litellm_client.litellm.embedding")
    def test_valid_text(self, mock_embedding):
        mock_embedding.return_value = _make_embedding_response([0.1, 0.2, 0.3])
        client = _build_client()

        result = client.get_embedding("some text")

        assert result == [0.1, 0.2, 0.3]
        call_kwargs = mock_embedding.call_args.kwargs
        assert call_kwargs["model"] == "text-embedding-3-small"
        assert call_kwargs["input"] == ["some text"]

    @patch("reflexio.server.llm.litellm_client.litellm.embedding")
    def test_custom_model(self, mock_embedding):
        mock_embedding.return_value = _make_embedding_response()
        client = _build_client()

        client.get_embedding("text", model="text-embedding-ada-002")

        call_kwargs = mock_embedding.call_args.kwargs
        assert call_kwargs["model"] == "text-embedding-ada-002"

    @patch("reflexio.server.llm.litellm_client.litellm.embedding")
    def test_with_dimensions(self, mock_embedding):
        mock_embedding.return_value = _make_embedding_response([0.1, 0.2])
        client = _build_client()

        client.get_embedding("text", dimensions=256)

        call_kwargs = mock_embedding.call_args.kwargs
        assert call_kwargs["dimensions"] == 256

    @patch("reflexio.server.llm.litellm_client.litellm.embedding")
    def test_embedding_failure_raises(self, mock_embedding):
        mock_embedding.side_effect = RuntimeError("API down")
        client = _build_client()

        with pytest.raises(LiteLLMClientError, match="Embedding generation failed"):
            client.get_embedding("text")

    @patch("reflexio.server.llm.litellm_client.litellm.embedding")
    def test_embedding_with_api_key_config(self, mock_embedding):
        mock_embedding.return_value = _make_embedding_response()
        api_key_config = APIKeyConfig(openai=CommonsOpenAIConfig(api_key="sk-test"))
        config = LiteLLMConfig(model="gpt-4o", api_key_config=api_key_config)
        client = LiteLLMClient(config)

        client.get_embedding("text")

        call_kwargs = mock_embedding.call_args.kwargs
        assert call_kwargs["api_key"] == "sk-test"


# ===================================================================
# get_embeddings (batch) tests
# ===================================================================


class TestGetEmbeddings:
    """Tests for the batch embedding endpoint."""

    @patch("reflexio.server.llm.litellm_client.litellm.embedding")
    def test_batch_embeddings(self, mock_embedding):
        mock_embedding.return_value = _make_batch_embedding_response(
            [[0.1, 0.2], [0.3, 0.4]]
        )
        client = _build_client()

        result = client.get_embeddings(["hello", "world"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]

    def test_empty_list_returns_empty(self):
        client = _build_client()
        result = client.get_embeddings([])
        assert result == []

    @patch("reflexio.server.llm.litellm_client.litellm.embedding")
    def test_batch_embedding_failure_raises(self, mock_embedding):
        mock_embedding.side_effect = RuntimeError("API down")
        client = _build_client()

        with pytest.raises(
            LiteLLMClientError, match="Batch embedding generation failed"
        ):
            client.get_embeddings(["text1", "text2"])

    @patch("reflexio.server.llm.litellm_client.litellm.embedding")
    def test_batch_embeddings_sorted_by_index(self, mock_embedding):
        """Ensure results are sorted by index even if API returns out of order."""
        resp = MagicMock()
        resp.data = [
            {"embedding": [0.3, 0.4], "index": 1},
            {"embedding": [0.1, 0.2], "index": 0},
        ]
        mock_embedding.return_value = resp
        client = _build_client()

        result = client.get_embeddings(["first", "second"])

        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]


# ===================================================================
# Embedding input truncation tests
# ===================================================================


class TestEmbeddingTruncation:
    """Tests for the embedding-input truncation helpers."""

    @pytest.fixture(autouse=True)
    def _reset_caches(self):
        """
        Clear both embedding-helper lru_caches and the warned-models set
        before and after every test so ordering can't leak mocked values.
        """
        from reflexio.server.llm.litellm_client import _TRUNCATION_WARNED_MODELS

        _get_embedding_limit.cache_clear()
        _get_embedding_encoding.cache_clear()
        _TRUNCATION_WARNED_MODELS.clear()
        yield
        _get_embedding_limit.cache_clear()
        _get_embedding_encoding.cache_clear()
        _TRUNCATION_WARNED_MODELS.clear()

    def test_get_embedding_limit_openai_family(self):
        """Known OpenAI embedding models resolve to the 8191 cap."""
        # Patch the registry so the assertion doesn't ride on whatever value
        # the installed litellm build happens to report today.
        with patch(
            "reflexio.server.llm.litellm_client.litellm.get_model_info",
            return_value={"mode": "embedding", "max_input_tokens": 8191},
        ):
            assert _get_embedding_limit("text-embedding-3-small") == 8191
            assert _get_embedding_limit("text-embedding-3-large") == 8191
            assert _get_embedding_limit("text-embedding-ada-002") == 8191

    @pytest.mark.parametrize(
        ("model", "mock_kwargs", "expected"),
        [
            # Unknown text-embedding-* name — litellm has no entry, OpenAI fallback.
            ("text-embedding-bogus", {"side_effect": Exception("unmapped")}, 8191),
            # openai/* prefix, unknown to litellm — fallback.
            ("openai/custom-embed", {"side_effect": Exception("unmapped")}, 8191),
            # azure/* prefix, unknown to litellm — fallback.
            ("azure/custom-embed", {"side_effect": Exception("unmapped")}, 8191),
            # Unknown non-OpenAI provider — no safe fallback, return None.
            ("mystery-provider/embed-v1", {"side_effect": Exception("unmapped")}, None),
            # Registry reports a non-embedding mode — don't trust max_input_tokens.
            (
                "mystery/chat-model",
                {"return_value": {"mode": "chat", "max_input_tokens": 100000}},
                None,
            ),
            # Cohere-style provider with a small cap — return exactly what litellm says.
            (
                "cohere/embed-english-v3.0",
                {"return_value": {"mode": "embedding", "max_input_tokens": 512}},
                512,
            ),
        ],
    )
    def test_get_embedding_limit_variants(self, model, mock_kwargs, expected):
        """Exhaustive table of registry lookup + prefix-fallback outcomes."""
        with patch(
            "reflexio.server.llm.litellm_client.litellm.get_model_info",
            **mock_kwargs,
        ):
            assert _get_embedding_limit(model) == expected

    def test_get_embedding_encoding_unknown_model_falls_back_to_cl100k(self):
        """Unknown model names use the cl100k_base tokenizer as a proxy."""
        encoding = _get_embedding_encoding("totally-custom-model")
        assert encoding.name == "cl100k_base"

    def test_truncate_empty_string_is_pass_through(self):
        assert _truncate_for_embedding("", "text-embedding-3-small") == ""

    def test_truncate_short_text_is_unchanged(self):
        """Text already within the limit is returned unchanged."""
        text = "hello world"
        result = _truncate_for_embedding(text, "text-embedding-3-small")
        assert result == text

    def test_truncate_long_text_is_shortened(self):
        """A tiny override limit exercises the truncation path on short input."""
        text = "word " * 200
        result = _truncate_for_embedding(text, "text-embedding-3-small", max_tokens=10)
        encoding = _get_embedding_encoding("text-embedding-3-small")
        assert len(encoding.encode(result)) <= 10
        assert len(result) < len(text)

    def test_truncate_unknown_provider_is_pass_through(self):
        """Unknown non-OpenAI models skip truncation entirely."""
        text = "word " * 5000
        with patch(
            "reflexio.server.llm.litellm_client.litellm.get_model_info",
            side_effect=Exception("unmapped"),
        ):
            result = _truncate_for_embedding(text, "mystery-provider/embed-v1")
        assert result == text

    def test_truncate_preserves_prefix_from_storage_caller(self):
        """Prefix (`search_document:` / `search_query:`) survives truncation.

        Mirrors how sqlite_storage._base._get_embedding builds its input: the
        prefix is concatenated onto the text before reaching the LLM client,
        so the truncation budget must be spent on the suffix, not the prefix.
        """
        encoding = _get_embedding_encoding("text-embedding-3-small")
        prefix = "search_document: "
        body = encoding.decode(list(range(9000)))
        result = _truncate_for_embedding(prefix + body, "text-embedding-3-small")
        assert result.startswith(prefix)
        assert len(encoding.encode(result)) <= 8191

    def test_truncate_warning_emitted_once_then_debug(self, caplog):
        """First oversized text for a model warns; subsequent calls go to DEBUG."""
        encoding = _get_embedding_encoding("text-embedding-3-small")
        oversized = encoding.decode(list(range(9000)))

        with caplog.at_level(
            logging.WARNING, logger="reflexio.server.llm.litellm_client"
        ):
            _truncate_for_embedding(oversized, "text-embedding-3-small")
            _truncate_for_embedding(oversized, "text-embedding-3-small")

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "Truncating embedding input" in warnings[0].getMessage()
        assert "text-embedding-3-small" in warnings[0].getMessage()

    @patch("reflexio.server.llm.litellm_client.litellm.embedding")
    def test_get_embedding_long_text_is_truncated_before_call(self, mock_embedding):
        """The string reaching litellm.embedding is always under the cap."""
        mock_embedding.return_value = _make_embedding_response([0.1, 0.2, 0.3])
        client = _build_client()
        encoding = _get_embedding_encoding("text-embedding-3-small")
        oversized = encoding.decode(list(range(9000)))

        client.get_embedding(oversized)

        call_kwargs = mock_embedding.call_args.kwargs
        sent_text = call_kwargs["input"][0]
        assert len(encoding.encode(sent_text)) <= 8191
        assert sent_text != oversized

    @patch("reflexio.server.llm.litellm_client.litellm.embedding")
    def test_get_embeddings_truncates_each_element_independently(self, mock_embedding):
        """Batch truncation is per-element: short strings are not rewritten."""
        mock_embedding.return_value = _make_batch_embedding_response(
            [[0.1, 0.2], [0.3, 0.4]]
        )
        client = _build_client()
        encoding = _get_embedding_encoding("text-embedding-3-small")
        short = "hello"
        oversized = encoding.decode(list(range(9000)))

        client.get_embeddings([short, oversized])

        sent_texts = mock_embedding.call_args.kwargs["input"]
        assert sent_texts[0] == short
        assert sent_texts[1] != oversized
        assert len(encoding.encode(sent_texts[1])) <= 8191


# ===================================================================
# Structured output parsing tests
# ===================================================================


class TestMaybeParseStructuredOutput:
    """Tests for _maybe_parse_structured_output."""

    @pytest.fixture()
    def client(self):
        return _build_client()

    def test_no_response_format_returns_raw(self, client):
        result = client._maybe_parse_structured_output("raw text", None, True)
        assert result == "raw text"

    def test_parse_disabled_returns_raw(self, client):
        result = client._maybe_parse_structured_output(
            '{"answer": "ok", "score": 5}', SampleResponse, False
        )
        assert isinstance(result, str)

    def test_none_content_returns_none(self, client):
        result = client._maybe_parse_structured_output(None, SampleResponse, True)
        assert result is None

    def test_already_pydantic_model_returned_as_is(self, client):
        obj = SampleResponse(answer="ok", score=5)
        result = client._maybe_parse_structured_output(obj, SampleResponse, True)
        assert result is obj

    def test_valid_json_parsed(self, client):
        json_str = json.dumps({"answer": "ok", "score": 5})
        result = client._maybe_parse_structured_output(json_str, SampleResponse, True)
        assert isinstance(result, SampleResponse)
        assert result.answer == "ok"

    def test_json_in_markdown_code_block(self, client):
        content = '```json\n{"answer": "ok", "score": 5}\n```'
        result = client._maybe_parse_structured_output(content, SampleResponse, True)
        assert isinstance(result, SampleResponse)
        assert result.score == 5

    def test_python_style_json_sanitized(self, client):
        """Python-style True/False/None and single quotes are sanitized."""
        content = "{'answer': 'ok', 'score': 5}"
        result = client._maybe_parse_structured_output(content, SampleResponse, True)
        assert isinstance(result, SampleResponse)
        assert result.answer == "ok"

    def test_unparseable_returns_raw_content(self, client):
        result = client._maybe_parse_structured_output(
            "totally not json", SampleResponse, True
        )
        assert result == "totally not json"


# ===================================================================
# _extract_json_from_string tests
# ===================================================================


class TestExtractJsonFromString:
    """Tests for JSON extraction from various string formats."""

    @pytest.fixture()
    def client(self):
        return _build_client()

    def test_plain_json_object(self, client):
        content = '{"key": "value"}'
        assert client._extract_json_from_string(content) == '{"key": "value"}'

    def test_json_in_markdown_block(self, client):
        content = '```json\n{"key": "value"}\n```'
        result = client._extract_json_from_string(content)
        assert result == '{"key": "value"}'

    def test_json_in_plain_code_block(self, client):
        content = '```\n{"key": "value"}\n```'
        result = client._extract_json_from_string(content)
        assert result == '{"key": "value"}'

    def test_json_array(self, client):
        content = "Some text before [1, 2, 3] some text after"
        result = client._extract_json_from_string(content)
        assert result == "[1, 2, 3]"

    def test_json_object_in_text(self, client):
        content = 'Here is the result: {"answer": 42} that is all'
        result = client._extract_json_from_string(content)
        assert result == '{"answer": 42}'

    def test_no_json_returns_original(self, client):
        content = "plain text"
        assert client._extract_json_from_string(content) == "plain text"


# ===================================================================
# _sanitize_json_string tests
# ===================================================================


class TestSanitizeJsonString:
    """Tests for Python-to-JSON sanitization."""

    @pytest.fixture()
    def client(self):
        return _build_client()

    def test_single_quotes_to_double(self, client):
        result = client._sanitize_json_string("{'key': 'value'}")
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_python_booleans(self, client):
        result = client._sanitize_json_string('{"flag": True, "other": False}')
        parsed = json.loads(result)
        assert parsed == {"flag": True, "other": False}

    def test_python_none(self, client):
        result = client._sanitize_json_string('{"val": None}')
        parsed = json.loads(result)
        assert parsed == {"val": None}

    def test_trailing_commas(self, client):
        result = client._sanitize_json_string('{"a": 1, "b": 2, }')
        parsed = json.loads(result)
        assert parsed == {"a": 1, "b": 2}

    def test_escaped_apostrophe_in_single_quoted(self, client):
        result = client._sanitize_json_string("{'text': 'didn\\'t work'}")
        parsed = json.loads(result)
        assert parsed["text"] == "didn't work"

    def test_double_quotes_inside_single_quoted_escaped(self, client):
        result = client._sanitize_json_string("{'key': 'he said \"hello\"'}")
        parsed = json.loads(result)
        assert parsed["key"] == 'he said "hello"'


# ===================================================================
# Temperature restriction tests
# ===================================================================


class TestTemperatureRestriction:
    """Tests for _is_temperature_restricted_model."""

    @pytest.fixture()
    def client(self):
        return _build_client()

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5-codex",
            "GPT-5-Mini",
        ],
    )
    def test_restricted_models(self, client, model):
        assert client._is_temperature_restricted_model(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4o",
            "gpt-4o-mini",
            "claude-3-5-sonnet",
            "gemini-pro",
        ],
    )
    def test_non_restricted_models(self, client, model):
        assert client._is_temperature_restricted_model(model) is False

    def test_provider_prefix_stripped(self, client):
        """Model with provider prefix like openrouter/openai/gpt-5-nano."""
        assert (
            client._is_temperature_restricted_model("openrouter/openai/gpt-5-nano")
            is True
        )

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_restricted_model_omits_temperature(self, mock_completion):
        mock_completion.return_value = _make_completion_response("ok")
        config = LiteLLMConfig(model="gpt-5-mini", temperature=0.7)
        client = LiteLLMClient(config)

        client.generate_response("hi")

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["temperature"] == 1.0

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_non_restricted_model_includes_temperature(self, mock_completion):
        mock_completion.return_value = _make_completion_response("ok")
        config = LiteLLMConfig(model="gpt-4o", temperature=0.3)
        client = LiteLLMClient(config)

        client.generate_response("hi")

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.3


# ===================================================================
# Config management tests
# ===================================================================


class TestConfigManagement:
    """Tests for update_config, get_config, get_model."""

    def test_get_config_returns_config(self):
        config = LiteLLMConfig(model="gpt-4o")
        client = LiteLLMClient(config)

        returned = client.get_config()
        assert returned is client.config
        assert returned.model == "gpt-4o"

    def test_get_model(self):
        config = LiteLLMConfig(model="claude-3-5-sonnet")
        client = LiteLLMClient(config)
        assert client.get_model() == "claude-3-5-sonnet"

    def test_update_config_known_keys(self):
        client = _build_client()
        client.update_config(model="gpt-4o-mini", temperature=0.2, max_retries=5)

        assert client.config.model == "gpt-4o-mini"
        assert client.config.temperature == 0.2
        assert client.config.max_retries == 5

    def test_update_config_unknown_key_is_ignored(self):
        client = _build_client()
        client.update_config(nonexistent_param="value")
        assert not hasattr(client.config, "nonexistent_param")


# ===================================================================
# _build_completion_params tests
# ===================================================================


class TestBuildCompletionParams:
    """Tests for _build_completion_params internals."""

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_max_tokens_passed_through(self, mock_completion):
        mock_completion.return_value = _make_completion_response("ok")
        config = LiteLLMConfig(model="gpt-4o", max_tokens=100)
        client = LiteLLMClient(config)

        client.generate_response("hi")

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["max_tokens"] == 100

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_top_p_non_default(self, mock_completion):
        mock_completion.return_value = _make_completion_response("ok")
        config = LiteLLMConfig(model="gpt-4o", top_p=0.9)
        client = LiteLLMClient(config)

        client.generate_response("hi")

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["top_p"] == 0.9

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_top_p_default_not_included(self, mock_completion):
        mock_completion.return_value = _make_completion_response("ok")
        config = LiteLLMConfig(model="gpt-4o", top_p=1.0)
        client = LiteLLMClient(config)

        client.generate_response("hi")

        call_kwargs = mock_completion.call_args.kwargs
        assert "top_p" not in call_kwargs

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_custom_endpoint_overrides_model(self, mock_completion):
        mock_completion.return_value = _make_completion_response("ok")
        api_key_config = APIKeyConfig(
            custom_endpoint=CustomEndpointConfig(
                model="custom-model",
                api_key="ce-key",
                api_base="https://custom.api.com/v1",
            )
        )
        config = LiteLLMConfig(model="gpt-4o", api_key_config=api_key_config)
        client = LiteLLMClient(config)

        client.generate_response("hi")

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["model"] == "custom-model"
        assert call_kwargs["api_key"] == "ce-key"
        assert call_kwargs["api_base"] == "https://custom.api.com/v1"

    def test_invalid_max_retries_fallback(self):
        config = LiteLLMConfig(model="gpt-4o", max_retries=2)
        client = LiteLLMClient(config)

        params, _, _, max_retries = client._build_completion_params(
            [{"role": "user", "content": "hi"}],
            max_retries="invalid",
        )
        assert max_retries == 2  # Falls back to config value

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_different_model_resolves_different_api_key(self, mock_completion):
        mock_completion.return_value = _make_completion_response("ok")
        api_key_config = APIKeyConfig(
            openai=CommonsOpenAIConfig(api_key="sk-openai"),
            anthropic=AnthropicConfig(api_key="ant-key"),
        )
        config = LiteLLMConfig(model="gpt-4o", api_key_config=api_key_config)
        client = LiteLLMClient(config)

        # Use a claude model which differs from the config model
        client.generate_response("hi", model="claude-3-5-sonnet")

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["api_key"] == "ant-key"


# ===================================================================
# _apply_prompt_caching tests
# ===================================================================


class TestApplyPromptCaching:
    """Tests for Anthropic prompt caching."""

    @pytest.fixture()
    def client(self):
        return _build_client()

    def test_non_anthropic_model_unchanged(self, client):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = client._apply_prompt_caching(messages, "gpt-4o")
        assert result == messages

    def test_claude_model_adds_cache_control(self, client):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = client._apply_prompt_caching(messages, "claude-3-5-sonnet")

        assert result[0]["role"] == "system"
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "You are helpful."
        assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        # User message unchanged
        assert result[1] == messages[1]

    def test_anthropic_in_model_name(self, client):
        messages = [{"role": "system", "content": "System msg"}]
        result = client._apply_prompt_caching(messages, "anthropic/claude-3")
        assert isinstance(result[0]["content"], list)

    def test_non_string_system_content_unchanged(self, client):
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Already formatted"}],
            },
        ]
        result = client._apply_prompt_caching(messages, "claude-3-5-sonnet")
        # Should not re-wrap
        assert result[0]["content"] == [{"type": "text", "text": "Already formatted"}]


# ===================================================================
# _build_user_content tests
# ===================================================================


class TestBuildUserContent:
    """Tests for _build_user_content with images."""

    @pytest.fixture()
    def client(self):
        return _build_client()

    def test_text_only(self, client):
        result = client._build_user_content("Hello")
        assert result == "Hello"

    def test_no_images(self, client):
        result = client._build_user_content("Hello", images=None)
        assert result == "Hello"

    def test_image_url(self, client):
        result = client._build_user_content(
            "Describe this", images=["https://example.com/image.png"]
        )
        assert isinstance(result, list)
        assert result[0] == {"type": "text", "text": "Describe this"}
        assert result[1]["type"] == "image_url"
        assert result[1]["image_url"]["url"] == "https://example.com/image.png"

    def test_image_bytes(self, client):
        img_bytes = b"\x89PNG\r\n\x1a\nfakedata"
        result = client._build_user_content(
            "Describe", images=[img_bytes], image_media_type="image/png"
        )
        assert isinstance(result, list)
        assert result[1]["type"] == "image_url"
        assert "data:image/png;base64," in result[1]["image_url"]["url"]

    def test_image_bytes_default_media_type(self, client):
        result = client._build_user_content("Describe", images=[b"fake"])
        assert "data:image/png;base64," in result[1]["image_url"]["url"]

    def test_image_dict_passthrough(self, client):
        img_dict = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,abc"},
        }
        result = client._build_user_content("Describe", images=[img_dict])
        assert result[1] is img_dict

    def test_image_file_path(self, client):
        png_data = _create_minimal_png()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_data)
            tmp_path = f.name

        try:
            result = client._build_user_content("Describe", images=[tmp_path])
            assert isinstance(result, list)
            assert result[1]["type"] == "image_url"
            assert "data:image/png;base64," in result[1]["image_url"]["url"]
        finally:
            Path(tmp_path).unlink()


# ===================================================================
# encode_image_to_base64 tests
# ===================================================================


class TestEncodeImageToBase64:
    """Tests for encode_image_to_base64."""

    @pytest.fixture()
    def client(self):
        return _build_client()

    def test_valid_png(self, client):
        png_data = _create_minimal_png()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_data)
            tmp_path = f.name

        try:
            b64_data, media_type = client.encode_image_to_base64(tmp_path)
            assert media_type == "image/png"
            assert len(b64_data) > 0
            # Verify it's valid base64
            decoded = base64.b64decode(b64_data)
            assert decoded == png_data
        finally:
            Path(tmp_path).unlink()

    def test_file_not_found_raises(self, client):
        with pytest.raises(LiteLLMClientError, match="Image file not found"):
            client.encode_image_to_base64("/nonexistent/path/image.png")

    def test_unsupported_format_raises(self, client):
        with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as f:
            f.write(b"fake bmp data")
            tmp_path = f.name

        try:
            with pytest.raises(LiteLLMClientError, match="Unsupported image format"):
                client.encode_image_to_base64(tmp_path)
        finally:
            Path(tmp_path).unlink()

    def test_jpeg_format(self, client):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake jpeg data")
            tmp_path = f.name

        try:
            _, media_type = client.encode_image_to_base64(tmp_path)
            assert media_type == "image/jpeg"
        finally:
            Path(tmp_path).unlink()


# ===================================================================
# _log_token_usage tests
# ===================================================================


class TestLogTokenUsage:
    """Tests for _log_token_usage."""

    @pytest.fixture()
    def client(self):
        return _build_client()

    def test_no_usage_attribute(self, client):
        response = MagicMock(spec=[])
        # Should not raise
        client._log_token_usage({"model": "gpt-4o"}, response)

    def test_with_cache_details(self, client):
        response = MagicMock()
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 5
        response.usage.total_tokens = 15
        response.usage.prompt_tokens_details = MagicMock(cached_tokens=3)
        response.usage.cache_creation_input_tokens = None
        response.usage.cache_read_input_tokens = None
        # Should not raise
        client._log_token_usage({"model": "gpt-4o"}, response)

    def test_with_anthropic_cache_stats(self, client):
        response = MagicMock()
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 5
        response.usage.total_tokens = 15
        response.usage.prompt_tokens_details = None
        response.usage.cache_creation_input_tokens = 100
        response.usage.cache_read_input_tokens = 50
        # Should not raise
        client._log_token_usage({"model": "claude-3"}, response)


# ===================================================================
# _handle_retry_or_raise tests
# ===================================================================


class TestHandleRetryOrRaise:
    """Tests for _handle_retry_or_raise."""

    def test_non_retryable_error_raises_immediately(self):
        client = _build_client()
        with pytest.raises(LiteLLMClientError, match="API call failed"):
            client._handle_retry_or_raise(
                RuntimeError("invalid_api_key: bad key"),
                {"model": "gpt-4o"},
                attempt=0,
                max_retries=3,
                response_format=None,
                elapsed_seconds=0.5,
            )

    @patch("reflexio.server.llm.litellm_client.time.sleep")
    def test_retryable_error_sleeps(self, mock_sleep):
        config = LiteLLMConfig(model="gpt-4o", retry_delay=1.0)
        client = LiteLLMClient(config)
        # Should not raise, just sleep
        client._handle_retry_or_raise(
            RuntimeError("connection timeout"),
            {"model": "gpt-4o"},
            attempt=0,
            max_retries=3,
            response_format=None,
            elapsed_seconds=0.5,
        )
        mock_sleep.assert_called_once_with(1.0)  # 1.0 * 2^0

    def test_last_attempt_does_not_raise(self):
        """On last attempt, _handle_retry_or_raise just logs; _make_request raises later."""
        client = _build_client()
        # Should not raise for retryable error on last attempt
        client._handle_retry_or_raise(
            RuntimeError("connection timeout"),
            {"model": "gpt-4o"},
            attempt=2,
            max_retries=3,
            response_format=None,
            elapsed_seconds=0.5,
        )


# ===================================================================
# create_litellm_client convenience function tests
# ===================================================================


class TestCreateLiteLLMClient:
    """Tests for the create_litellm_client factory function."""

    def test_basic_creation(self):
        client = create_litellm_client(model="gpt-4o")
        assert client.get_model() == "gpt-4o"
        assert client.config.temperature == 0.7

    def test_with_all_params(self):
        client = create_litellm_client(
            model="claude-3",
            temperature=0.5,
            max_tokens=100,
            timeout=30,
            max_retries=5,
        )
        assert client.config.model == "claude-3"
        assert client.config.temperature == 0.5
        assert client.config.max_tokens == 100
        assert client.config.timeout == 30
        assert client.config.max_retries == 5

    def test_with_api_key_config(self):
        api_key_config = APIKeyConfig(openai=CommonsOpenAIConfig(api_key="sk-test"))
        client = create_litellm_client(model="gpt-4o", api_key_config=api_key_config)
        assert client._api_key == "sk-test"


# ===================================================================
# Additional retry/error handling edge cases
# ===================================================================


class TestRetryErrorEdgeCases:
    """Additional edge cases for retry logic and error handling."""

    @patch("reflexio.server.llm.litellm_client.time.sleep")
    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_exponential_backoff_delay(self, mock_completion, mock_sleep):
        """Verify exponential backoff: delay = retry_delay * 2^attempt."""
        mock_completion.side_effect = [
            RuntimeError("temp failure 1"),
            RuntimeError("temp failure 2"),
            _make_completion_response("ok"),
        ]
        config = LiteLLMConfig(model="gpt-4o", max_retries=3, retry_delay=1.0)
        client = LiteLLMClient(config)

        result = client._make_request([{"role": "user", "content": "hi"}])

        assert result == "ok"
        assert mock_sleep.call_count == 2
        # First retry: 1.0 * 2^0 = 1.0
        assert mock_sleep.call_args_list[0][0][0] == 1.0
        # Second retry: 1.0 * 2^1 = 2.0
        assert mock_sleep.call_args_list[1][0][0] == 2.0

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_non_retryable_stops_on_first_attempt(self, mock_completion):
        """Non-retryable errors should not trigger retries."""
        mock_completion.side_effect = RuntimeError("quota_exceeded for project")
        config = LiteLLMConfig(model="gpt-4o", max_retries=5)
        client = LiteLLMClient(config)

        with pytest.raises(LiteLLMClientError, match="quota_exceeded"):
            client._make_request([{"role": "user", "content": "hi"}])

        # Should only try once
        assert mock_completion.call_count == 1

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_max_retries_zero_treated_as_one(self, mock_completion):
        """max_retries=0 should be treated as at least 1 attempt."""
        mock_completion.return_value = _make_completion_response("ok")
        config = LiteLLMConfig(model="gpt-4o", max_retries=0)
        client = LiteLLMClient(config)

        result = client._make_request([{"role": "user", "content": "hi"}])
        assert result == "ok"

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_negative_max_retries_treated_as_one(self, mock_completion):
        """Negative max_retries should be treated as at least 1."""
        mock_completion.return_value = _make_completion_response("ok")
        config = LiteLLMConfig(model="gpt-4o", max_retries=-1)
        client = LiteLLMClient(config)

        result = client._make_request([{"role": "user", "content": "hi"}])
        assert result == "ok"


class TestTokenUsageLoggingEdgeCases:
    """Edge cases for _log_token_usage."""

    def test_no_prompt_tokens_details(self):
        """Test logging with no prompt_tokens_details."""
        client = _build_client()
        response = MagicMock()
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 5
        response.usage.total_tokens = 15
        response.usage.prompt_tokens_details = None
        response.usage.cache_creation_input_tokens = None
        response.usage.cache_read_input_tokens = None
        # Should not raise
        client._log_token_usage({"model": "gpt-4o"}, response)

    def test_cached_tokens_zero(self):
        """Test logging when cached_tokens is 0 (no cache info appended)."""
        client = _build_client()
        response = MagicMock()
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 5
        response.usage.total_tokens = 15
        details = MagicMock(cached_tokens=0)
        response.usage.prompt_tokens_details = details
        response.usage.cache_creation_input_tokens = None
        response.usage.cache_read_input_tokens = None
        # Should not raise
        client._log_token_usage({"model": "gpt-4o"}, response)


class TestSanitizeJsonEdgeCases:
    """Additional edge cases for _sanitize_json_string."""

    def test_nested_single_quotes(self):
        """Test nested single-quoted strings."""
        client = _build_client()
        result = client._sanitize_json_string("{'items': ['a', 'b', 'c']}")
        parsed = json.loads(result)
        assert parsed == {"items": ["a", "b", "c"]}

    def test_trailing_comma_in_array(self):
        """Test trailing commas before closing bracket."""
        client = _build_client()
        result = client._sanitize_json_string('{"items": [1, 2, 3, ]}')
        parsed = json.loads(result)
        assert parsed == {"items": [1, 2, 3]}

    def test_mixed_python_values(self):
        """Test handling of mixed True/False/None values."""
        client = _build_client()
        result = client._sanitize_json_string('{"a": True, "b": False, "c": None}')
        parsed = json.loads(result)
        assert parsed == {"a": True, "b": False, "c": None}

    def test_boolean_inside_string_not_replaced(self):
        """Test that True/False inside strings are not replaced."""
        client = _build_client()
        result = client._sanitize_json_string('{"msg": "This is True story"}')
        parsed = json.loads(result)
        # "True" inside the string value should remain unchanged
        assert "True" in parsed["msg"]


class TestBuildCompletionParamsEdgeCases:
    """Additional edge cases for _build_completion_params."""

    def test_max_retries_kwarg_overrides_config(self):
        """Test that max_retries kwarg overrides config value."""
        config = LiteLLMConfig(model="gpt-4o", max_retries=2)
        client = LiteLLMClient(config)

        _, _, _, max_retries = client._build_completion_params(
            [{"role": "user", "content": "hi"}],
            max_retries=5,
        )
        assert max_retries == 5

    def test_model_kwarg_overrides_config(self):
        """Test that model kwarg overrides config model."""
        api_key_config = APIKeyConfig(
            anthropic=AnthropicConfig(api_key="ant-key"),
            openai=CommonsOpenAIConfig(api_key="sk-openai"),
        )
        config = LiteLLMConfig(model="gpt-4o", api_key_config=api_key_config)
        client = LiteLLMClient(config)

        params, _, _, _ = client._build_completion_params(
            [{"role": "user", "content": "hi"}],
            model="claude-3-5-sonnet",
        )
        assert params["model"] == "claude-3-5-sonnet"
        assert params["api_key"] == "ant-key"
