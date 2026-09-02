"""Unit tests for the LiteLLM client wrapper.

Tests cover initialization, response generation, retry logic, error classification,
embeddings, structured output parsing, config management, image handling, and
prompt caching. All LiteLLM SDK calls are mocked -- no real API requests are made.
"""

import base64
import json
import logging
import multiprocessing
import struct
import tempfile
import time
import zlib
from pathlib import Path
from typing import Annotated, Any, Literal, cast
from unittest.mock import MagicMock, patch

import litellm
import pytest
from litellm.exceptions import APIConnectionError
from pydantic import BaseModel, Field

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
from reflexio.models.structured_output import find_schema_keyword
from reflexio.server import error_reporting
from reflexio.server.llm._litellm_subprocess import _snapshot_completion_response
from reflexio.server.llm._litellm_types import CompletionResult, ModelProvenance
from reflexio.server.llm._provider_concurrency import ProviderCapSaturatedError
from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMClientError,
    LiteLLMConfig,
    LLMHardTimeoutError,
    StructuredOutputParseError,
    StructuredOutputRepairError,
    ToolCallingChatResponse,
    _CompletionErrorSnapshot,
    _extract_json_from_string,
    _get_embedding_encoding,
    _get_embedding_limit,
    _litellm_completion_worker,
    _sanitize_json_string,
    _truncate_for_embedding,
    create_litellm_client,
)
from reflexio.server.llm.llm_utils import make_strict_json_schema

# ---------------------------------------------------------------------------
# Pydantic models used for structured-output tests
# ---------------------------------------------------------------------------


class SampleResponse(BaseModel):
    answer: str
    score: int


class MathResult(BaseModel):
    result: int
    explanation: str


class SingleListResponse(BaseModel):
    items: list[SampleResponse] = Field(default_factory=list)


class OptionalListResponse(BaseModel):
    items: list[SampleResponse] | None = None


class MultiFieldListResponse(BaseModel):
    items: list[SampleResponse] = Field(default_factory=list)
    source: str


class BareListResponse(BaseModel):
    items: list = Field(default_factory=list)


class OptionalMultiListResponse(BaseModel):
    """Multi-field schema whose fields are ALL optional lists.

    Mirrors ``ProfileDeduplicationOutput``: providers sometimes emit one inner
    array and drop the object wrapping it, and only one field can accept it.
    """

    groups: list[SampleResponse] = Field(default_factory=list)
    unique_ids: list[str] = Field(default_factory=list)


class AmbiguousMultiListResponse(BaseModel):
    """Two optional fields accept the same item shape — placement is a guess."""

    primary_ids: list[str] = Field(default_factory=list)
    secondary_ids: list[str] = Field(default_factory=list)


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
            endpoint="https://myresource.openai.azure.com/",  # type: ignore[arg-type]
            api_version="2024-02-15-preview",
        )
        api_key_config = APIKeyConfig(openai=CommonsOpenAIConfig(azure_config=azure))
        config = LiteLLMConfig(model="azure/gpt-4", api_key_config=api_key_config)
        client = LiteLLMClient(config)

        assert client._api_key == "az-key"
        assert client._api_base is not None and "myresource" in client._api_base
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
                api_base="https://example.com/v1",  # type: ignore[arg-type]
            )
        )
        config = LiteLLMConfig(model="gpt-4o", api_key_config=api_key_config)
        client = LiteLLMClient(config)

        assert client._api_key == "ce-key"
        assert client._api_base == "https://example.com/v1"


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
                api_base="https://example.com/v1",  # type: ignore[arg-type]
            ),
            openai=CommonsOpenAIConfig(api_key="sk-openai"),
        )
        config = LiteLLMConfig(model="gpt-4o", api_key_config=api_key_config)
        client = LiteLLMClient(config)

        key, base, version = client._resolve_api_key(for_embedding=False)
        assert key == "ce-key"
        assert base == "https://example.com/v1"

    def test_custom_endpoint_skipped_for_embedding(self):
        api_key_config = APIKeyConfig(
            custom_endpoint=CustomEndpointConfig(
                model="custom-model",
                api_key="ce-key",
                api_base="https://example.com/v1",  # type: ignore[arg-type]
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

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_opt_in_result_carries_actual_model_and_provider(self, mock_completion):
        response = _make_completion_response("hello")
        response.model = "MiniMax-M3"
        response._hidden_params = {"custom_llm_provider": "minimax"}
        mock_completion.return_value = response
        client = _build_client(
            LiteLLMConfig(
                model="minimax/MiniMax-M3",
                api_key_config=APIKeyConfig(minimax=MiniMaxConfig(api_key="test-key")),
            )
        )

        result = client.generate_response_with_provenance("test")

        assert isinstance(result, CompletionResult)
        assert result.value == "hello"
        assert result.provenance == ModelProvenance(
            model_name="MiniMax-M3",
            provider="minimax",
        )

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_structured_result_provenance_does_not_serialize(self, mock_completion):
        response = _make_completion_response(json.dumps({"answer": "ok", "score": 5}))
        response.model = "gpt-5.4-mini"
        response._hidden_params = {"custom_llm_provider": "openai"}
        mock_completion.return_value = response
        client = _build_client(LiteLLMConfig(model="gpt-5.4-mini"))

        result = client.generate_response_with_provenance(
            "test",
            response_format=SampleResponse,
        )

        assert isinstance(result, CompletionResult)
        assert isinstance(result.value, SampleResponse)
        assert result.value.model_dump() == {"answer": "ok", "score": 5}
        assert "provenance" not in result.value.model_dump()

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_claude_code_does_not_launder_requested_route_as_observed(
        self, mock_completion
    ):
        """Public ModelResponse.model is the requested route; only the stamp counts."""
        response = _make_completion_response("hello")
        response.model = "claude-code/default"
        response._hidden_params = {
            "reflexio_provider": "claude-code",
            "reflexio_cli_binary": "claude",
        }
        mock_completion.return_value = response
        client = _build_client(LiteLLMConfig(model="claude-code/default"))

        result = client.generate_response_with_provenance("test")

        assert isinstance(result, CompletionResult)
        assert result.provenance == ModelProvenance(
            model_name=None,
            provider=None,
        )

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_codex_cli_provenance_keeps_unknown_model(self, mock_completion):
        response = _make_completion_response("hello")
        response.model = None
        response._hidden_params = {
            "reflexio_provider": "claude-code",
            "reflexio_cli_binary": "codex",
        }
        mock_completion.return_value = response
        client = _build_client(LiteLLMConfig(model="claude-code/default"))

        result = client.generate_response_with_provenance("test")

        assert isinstance(result, CompletionResult)
        assert result.provenance == ModelProvenance(
            model_name=None,
            provider=None,
        )

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_claude_cli_provenance_keeps_served_model(self, mock_completion):
        response = _make_completion_response("hello")
        response.model = "claude-sonnet-5"
        response._hidden_params = {
            "reflexio_provider": "claude-code",
            "reflexio_cli_binary": "claude",
            "reflexio_served_model": "claude-sonnet-5",
        }
        mock_completion.return_value = response
        client = _build_client(LiteLLMConfig(model="claude-code/default"))

        result = client.generate_response_with_provenance("test")

        assert isinstance(result, CompletionResult)
        assert result.provenance == ModelProvenance(model_name="claude-sonnet-5")

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_model_name_does_not_imply_provider(self, mock_completion):
        response = _make_completion_response("hello")
        response.model = "gpt-5.4-mini"
        response._hidden_params = {}
        mock_completion.return_value = response
        client = _build_client(
            LiteLLMConfig(
                model="minimax/MiniMax-M3",
                api_key_config=APIKeyConfig(minimax=MiniMaxConfig(api_key="test-key")),
            )
        )

        result = client.generate_response_with_provenance("test")

        assert isinstance(result, CompletionResult)
        assert result.provenance == ModelProvenance(model_name="gpt-5.4-mini")

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_request_side_hidden_model_is_not_treated_as_served(self, mock_completion):
        """LiteLLM may echo the requested model into _hidden_params['model'].

        Without a response body model or reflexio_served_model stamp, model_name
        must stay unknown rather than recording the configured route as actual.
        """
        response = _make_completion_response("hello")
        response.model = None
        response._hidden_params = {
            "model": "minimax/MiniMax-M3",
            "model_id": "minimax/MiniMax-M3",
            "custom_llm_provider": "minimax",
        }
        mock_completion.return_value = response
        client = _build_client(
            LiteLLMConfig(
                model="minimax/MiniMax-M3",
                api_key_config=APIKeyConfig(minimax=MiniMaxConfig(api_key="test-key")),
            )
        )

        result = client.generate_response_with_provenance("test")

        assert isinstance(result, CompletionResult)
        assert result.provenance == ModelProvenance(
            model_name=None,
            provider="minimax",
        )

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_network_fallback_uses_actual_provider(self, mock_completion):
        response = _make_completion_response("hello")
        response.model = "gpt-5.4-mini"
        response._hidden_params = {"custom_llm_provider": "openai"}
        mock_completion.return_value = response
        client = _build_client(
            LiteLLMConfig(
                model="openai/local-model",
                api_key_config=APIKeyConfig(
                    custom_endpoint=CustomEndpointConfig(
                        model="openai/local-model",
                        api_key="test-key",
                        api_base="https://example.com/v1",  # type: ignore[arg-type]
                    )
                ),
            )
        )

        result = client.generate_response_with_provenance("test")

        assert isinstance(result, CompletionResult)
        provenance = cast(CompletionResult[Any], result).provenance
        assert provenance.provider == "openai"
        assert provenance.model_name == "gpt-5.4-mini"

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_azure_provenance_uses_actual_provider(self, mock_completion):
        response = _make_completion_response("hello")
        response.model = "gpt-5.4-mini"
        response._hidden_params = {"custom_llm_provider": "azure"}
        mock_completion.return_value = response
        client = _build_client(
            LiteLLMConfig(
                model="azure/gpt-5.4-mini",
                api_key_config=APIKeyConfig(
                    openai=CommonsOpenAIConfig(
                        azure_config=AzureOpenAIConfig(
                            api_key="test-key",
                            endpoint="https://example.openai.azure.com/",  # type: ignore[arg-type]
                        )
                    )
                ),
            )
        )

        result = client.generate_response_with_provenance("test")

        assert isinstance(result, CompletionResult)
        assert cast(CompletionResult[Any], result).provenance.provider == "azure"

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_fallback_uses_actual_provider(self, mock_completion):
        response = _make_completion_response("hello")
        response.model = "gpt-5.4-mini"
        response._hidden_params = {"custom_llm_provider": "openai"}
        mock_completion.return_value = response
        client = _build_client(
            LiteLLMConfig(
                model="minimax/MiniMax-M3",
                api_key_config=APIKeyConfig(
                    minimax=MiniMaxConfig(api_key="minimax-key"),
                    openai=CommonsOpenAIConfig(api_key="openai-key"),
                ),
            )
        )

        result = client.generate_response_with_provenance("test")

        assert isinstance(result, CompletionResult)
        assert cast(CompletionResult[Any], result).provenance.provider == "openai"

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
# get_embedding tests
# ===================================================================


# Env vars that route embedding calls away from litellm.embedding.
# When any of these is set in the shell, embedding_provider_mode() may resolve
# to "local_service" or "internal_service", causing get_embedding /
# get_embeddings to detour through get_service_embeddings (HTTP to
# 127.0.0.1:8072 by default) before reaching the mocked litellm.embedding.
# Tests below mock litellm.embedding and assert against that code path, so the
# routing env vars must be cleared per-test. Production routing logic is
# covered in tests/server/llm/test_embedding_service_provider.py.
_EMBEDDING_ROUTING_ENV_VARS = (
    "REFLEXIO_EMBEDDING_PROVIDER",
    "REFLEXIO_EMBEDDING_SERVICE_URL",
    "CLAUDE_SMART_USE_LOCAL_EMBEDDING",
)


def _force_litellm_embedding_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear shell-leaked embedding-routing env vars so tests exercise the
    litellm.embedding code path, not the local/internal service branches."""
    for name in _EMBEDDING_ROUTING_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class TestGetEmbedding:
    """Tests for the single-text embedding endpoint."""

    @pytest.fixture(autouse=True)
    def _force_litellm_route(self, monkeypatch):
        _force_litellm_embedding_route(monkeypatch)

    @pytest.fixture(autouse=True)
    def _pin_default_model(self, monkeypatch):
        # These tests exercise the litellm-backed default. Auto-detection
        # in the test environment may resolve to local/* (LocalEmbedder)
        # which short-circuits past the mocked litellm.embedding call.
        monkeypatch.setattr(
            LiteLLMClient,
            "_resolve_default_embedding_model",
            lambda _: "text-embedding-3-small",
        )

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

    @pytest.fixture(autouse=True)
    def _force_litellm_route(self, monkeypatch):
        _force_litellm_embedding_route(monkeypatch)

    @pytest.fixture(autouse=True)
    def _pin_default_model(self, monkeypatch):
        # See TestGetEmbedding._pin_default_model — these tests assert against
        # the litellm-backed default and must not be intercepted by the
        # local-embedder short-circuit.
        monkeypatch.setattr(
            LiteLLMClient,
            "_resolve_default_embedding_model",
            lambda _: "text-embedding-3-small",
        )

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
# Single/batch parity + error-mode tagging
# ===================================================================


class TestEmbeddingSingleBatchParity:
    """``get_embedding`` is a thin wrapper over ``get_embeddings`` — the two
    must agree, and embedding failures must be tagged with the resolved mode."""

    @staticmethod
    def _route_local_service(monkeypatch, *, embed):
        """Force the HTTP inference-service branch with a fake response."""
        from reflexio.server.llm import _litellm_embedding

        monkeypatch.setattr(
            _litellm_embedding,
            "get_service_embeddings",
            lambda texts, **_kwargs: embed(texts),
        )

    def test_get_embedding_matches_get_embeddings_first_element(self, monkeypatch):
        """get_embedding(t) == get_embeddings([t])[0] on the local-embedder path."""
        self._route_local_service(
            monkeypatch,
            embed=lambda texts: [[float(len(t)), 1.0, 2.0] for t in texts],
        )
        client = _build_client()

        single = client.get_embedding("hello", model="local/minilm-l6-v2")
        batch = client.get_embeddings(["hello"], model="local/minilm-l6-v2")

        assert single == batch[0]
        assert single == [5.0, 1.0, 2.0]

    def test_service_embedding_failure_never_falls_back(self, monkeypatch):
        """A service failure propagates and never constructs a local model."""

        def _boom(_texts):
            raise RuntimeError("boom")

        self._route_local_service(monkeypatch, embed=_boom)
        client = _build_client()

        with pytest.raises(RuntimeError, match="boom"):
            client.get_embedding("hello", model="local/minilm-l6-v2")


# ===================================================================
# Default embedding-model resolution tests
# ===================================================================


class TestEmbeddingDefaultResolution:
    """Tests for the caching and routing behavior of the embedding default."""

    @pytest.fixture(autouse=True)
    def _force_litellm_route(self, monkeypatch):
        _force_litellm_embedding_route(monkeypatch)

    def test_resolve_default_embedding_model_is_cached(self, monkeypatch):
        """``_resolve_default_embedding_model`` must hit resolve_model_name once.

        The per-instance cache (``self._default_embedding_model``) is the
        contract documented on ``LiteLLMClient`` — auto-detection is not free
        (it probes env vars and provider registries), so repeated
        ``get_embedding`` / ``get_embeddings`` calls on the same client must
        not re-resolve.
        """
        call_count = {"n": 0}

        def _fake_resolve(*args, **kwargs):
            call_count["n"] += 1
            return "text-embedding-3-small"

        monkeypatch.setattr(
            "reflexio.server.llm._litellm_embedding.resolve_model_name",
            _fake_resolve,
        )

        client = _build_client()
        with patch("reflexio.server.llm.litellm_client.litellm.embedding") as mock_emb:
            mock_emb.return_value = _make_embedding_response([0.1, 0.2])
            client.get_embedding("a")
            client.get_embedding("b")
            client.get_embeddings(["c", "d"])

        assert call_count["n"] == 1

    def test_get_embedding_routes_to_local_when_default_is_local(self, monkeypatch):
        """When the auto-detected default is ``local/…``, ``get_embedding``
        must take the inference-service branch and never call ``litellm.embedding``.
        """
        monkeypatch.setattr(
            LiteLLMClient,
            "_resolve_default_embedding_model",
            lambda _: "local/minilm-l6-v2",
        )
        monkeypatch.setattr(
            "reflexio.server.llm._litellm_embedding.get_service_embeddings",
            lambda *_args, **_kwargs: [[0.9, 0.8, 0.7]],
        )

        client = _build_client()
        with patch("reflexio.server.llm.litellm_client.litellm.embedding") as mock_emb:
            result = client.get_embedding("hello")

        assert result == [0.9, 0.8, 0.7]
        mock_emb.assert_not_called()


# ===================================================================
# Embedding input truncation tests
# ===================================================================


class TestEmbeddingTruncation:
    """Tests for the embedding-input truncation helpers."""

    @pytest.fixture(autouse=True)
    def _force_litellm_route(self, monkeypatch):
        _force_litellm_embedding_route(monkeypatch)

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

    @pytest.fixture(autouse=True)
    def _pin_default_model(self, monkeypatch):
        # See TestGetEmbedding._pin_default_model.
        monkeypatch.setattr(
            LiteLLMClient,
            "_resolve_default_embedding_model",
            lambda _: "text-embedding-3-small",
        )

    def test_get_embedding_limit_openai_family(self):
        """Known OpenAI embedding models resolve to the 8191 cap."""
        # Patch the registry so the assertion doesn't ride on whatever value
        # the installed litellm build happens to report today.
        with patch(
            "reflexio.server.llm._litellm_embedding.litellm.get_model_info",
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
            "reflexio.server.llm._litellm_embedding.litellm.get_model_info",
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
            "reflexio.server.llm._litellm_embedding.litellm.get_model_info",
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

    def test_none_content_raises_parse_error(self, client):
        with pytest.raises(
            StructuredOutputParseError,
            match="Structured output response content was empty",
        ):
            client._maybe_parse_structured_output(None, SampleResponse, True)

    def test_already_pydantic_model_returned_as_is(self, client):
        obj = SampleResponse(answer="ok", score=5)
        result = client._maybe_parse_structured_output(obj, SampleResponse, True)
        assert result is obj

    def test_valid_json_parsed(self, client):
        json_str = json.dumps({"answer": "ok", "score": 5})
        result = client._maybe_parse_structured_output(json_str, SampleResponse, True)
        assert isinstance(result, SampleResponse)
        assert result.answer == "ok"

    def test_once_json_encoded_object_is_parsed(self, client):
        content = json.dumps(json.dumps({"answer": "ok", "score": 5}))
        result = client._maybe_parse_structured_output(content, SampleResponse, True)

        assert isinstance(result, SampleResponse)
        assert result.answer == "ok"

    def test_twice_json_encoded_object_still_fails(self, client):
        payload = json.dumps({"answer": "ok", "score": 5})
        content = json.dumps(json.dumps(payload))

        with pytest.raises(StructuredOutputParseError):
            client._maybe_parse_structured_output(content, SampleResponse, True)

    def test_top_level_list_wrapped_for_single_list_schema(self, client):
        content = json.dumps([{"answer": "ok", "score": 5}])
        result = client._maybe_parse_structured_output(
            content, SingleListResponse, True
        )

        assert isinstance(result, SingleListResponse)
        assert len(result.items) == 1
        assert result.items[0].answer == "ok"

    def test_single_item_object_wrapped_for_single_list_schema(self, client):
        content = json.dumps({"answer": "ok", "score": 5})
        result = client._maybe_parse_structured_output(
            content, SingleListResponse, True
        )

        assert isinstance(result, SingleListResponse)
        assert len(result.items) == 1
        assert result.items[0].answer == "ok"

    def test_invalid_single_item_object_still_fails(self, client):
        content = json.dumps({"answer": "missing score"})

        with pytest.raises(StructuredOutputParseError):
            client._maybe_parse_structured_output(content, SingleListResponse, True)

    def test_top_level_list_wrapped_for_optional_list_schema(self, client):
        content = json.dumps([{"answer": "ok", "score": 5}])
        result = client._maybe_parse_structured_output(
            content, OptionalListResponse, True
        )

        assert isinstance(result, OptionalListResponse)
        assert result.items is not None
        assert len(result.items) == 1
        assert result.items[0].answer == "ok"

    def test_top_level_list_wrapped_for_bare_list_schema(self, client):
        content = json.dumps([{"answer": "ok", "score": 5}])
        result = client._maybe_parse_structured_output(content, BareListResponse, True)

        assert isinstance(result, BareListResponse)
        assert result.items == [{"answer": "ok", "score": 5}]

    def test_top_level_list_not_wrapped_for_multi_field_schema(self, client):
        """A required sibling field means the wrap cannot rebuild a whole object."""
        content = json.dumps([{"answer": "ok", "score": 5}])
        with pytest.raises(StructuredOutputParseError):
            client._maybe_parse_structured_output(content, MultiFieldListResponse, True)

    def test_top_level_list_wrapped_for_multi_field_schema_when_unambiguous(
        self, client
    ):
        """MiniMax drops the wrapper object and returns the inner array alone.

        Only ``groups`` accepts these items and every sibling is optional, so
        the placement is unambiguous — this is the ProfileDeduplicationOutput
        failure that produced 12.5k duplicate error events.
        """
        content = json.dumps([{"answer": "ok", "score": 5}])
        result = client._maybe_parse_structured_output(
            content, OptionalMultiListResponse, True
        )

        assert isinstance(result, OptionalMultiListResponse)
        assert len(result.groups) == 1
        assert result.groups[0].answer == "ok"
        assert result.unique_ids == []

    def test_top_level_list_routed_to_the_only_field_that_accepts_it(self, client):
        """Item shape, not field order, decides which list field receives it."""
        content = json.dumps(["NEW-2", "NEW-3"])
        result = client._maybe_parse_structured_output(
            content, OptionalMultiListResponse, True
        )

        assert result.unique_ids == ["NEW-2", "NEW-3"]
        assert result.groups == []

    def test_top_level_list_not_wrapped_when_two_fields_accept_it(self, client):
        """Ambiguous placement must fail to the repair path, never be guessed."""
        content = json.dumps(["a", "b"])
        with pytest.raises(StructuredOutputParseError):
            client._maybe_parse_structured_output(
                content, AmbiguousMultiListResponse, True
            )

    def test_json_in_markdown_code_block(self, client):
        content = '```json\n{"answer": "ok", "score": 5}\n```'
        result = client._maybe_parse_structured_output(content, SampleResponse, True)
        assert isinstance(result, SampleResponse)
        assert result.score == 5

    def test_structured_json_with_markdown_fence_in_string(self, client):
        content = json.dumps(
            {
                "answer": "Run:\n```bash\nmake package\n```",
                "score": 5,
            }
        )
        result = client._maybe_parse_structured_output(content, SampleResponse, True)
        assert isinstance(result, SampleResponse)
        assert result.score == 5
        assert "make package" in result.answer

    def test_python_style_json_sanitized(self, client):
        """Python-style True/False/None and single quotes are sanitized."""
        content = "{'answer': 'ok', 'score': 5}"
        result = client._maybe_parse_structured_output(content, SampleResponse, True)
        assert isinstance(result, SampleResponse)
        assert result.answer == "ok"

    def test_complete_malformed_json_repaired(self, client):
        """Complete JSON-shaped output with minor syntax errors is repaired."""
        content = '{"answer": "ok" "score": 5}'
        result = client._maybe_parse_structured_output(content, SampleResponse, True)
        assert isinstance(result, SampleResponse)
        assert result.answer == "ok"
        assert result.score == 5

    def test_truncated_json_not_repaired(self, client):
        """Incomplete JSON raises so the structured-output retry loop can retry."""
        content = '{"answer": "ok", "score": 5'
        with pytest.raises(StructuredOutputParseError):
            client._maybe_parse_structured_output(content, SampleResponse, True)

    def test_truncated_list_salvages_only_complete_valid_items(self, client):
        content = (
            '{"items": ['
            '{"answer": "first", "score": 5}, '
            '{"answer": "incomplete", "score"'
        )

        result = client._maybe_parse_structured_output(
            content, SingleListResponse, True
        )

        assert isinstance(result, SingleListResponse)
        assert [(item.answer, item.score) for item in result.items] == [("first", 5)]

    def test_truncated_list_drops_complete_invalid_sibling(self, client):
        content = (
            '{"items": ['
            '{"answer": "missing score"}, '
            '{"answer": "valid", "score": 7}, '
            '{"answer": "incomplete"'
        )

        result = client._maybe_parse_structured_output(
            content, SingleListResponse, True
        )

        assert isinstance(result, SingleListResponse)
        assert [(item.answer, item.score) for item in result.items] == [("valid", 7)]

    def test_prefixed_truncated_json_not_repaired(self, client):
        """Truncation is detected even if the model prefixes the JSON with prose."""
        content = 'Here is the result:\n{"answer": "ok", "score": 5'
        with pytest.raises(StructuredOutputParseError):
            client._maybe_parse_structured_output(content, SampleResponse, True)

    def test_unparseable_raises_structured_output_parse_error(self, client):
        with pytest.raises(StructuredOutputParseError):
            client._maybe_parse_structured_output(
                "totally not json", SampleResponse, True
            )


# Schema-keyword presence is checked via the shared structure-aware
# ``find_schema_keyword`` (reflexio.models.structured_output) — single source of
# truth for the name-map special-casing.


# A minimal discriminated-union output schema, the shape behind PYTHON-FASTAPI-9J.
# Pydantic emits `oneOf` + `discriminator` at properties.decisions.items, which
# strict structured-output endpoints reject.
class _UnifyDecision(BaseModel):
    kind: Literal["unify"] = "unify"
    new_id: str


class _RejectDecision(BaseModel):
    kind: Literal["reject"] = "reject"
    existing_id: int


_ConsolidationDecision = Annotated[
    _UnifyDecision | _RejectDecision, Field(discriminator="kind")
]


class _DiscriminatedOutput(BaseModel):
    decisions: list[_ConsolidationDecision] = Field(default_factory=list)


class TestStrictStructuredOutputRequest:
    """Provider-facing structured output request format."""

    def test_strict_json_schema_strips_provider_unsupported_constraints(self):
        class NestedLabel(BaseModel):
            label: str = Field(min_length=1, max_length=20)

        class BoundedScore(BaseModel):
            score: float = Field(ge=0.0, le=1.0)
            tags: list[str] = Field(min_length=1, max_length=3)
            code: str = Field(min_length=2, max_length=8, pattern="^[a-z]+$")
            nested: NestedLabel

        schema = make_strict_json_schema(BoundedScore.model_json_schema())

        score_schema = schema["properties"]["score"]
        tags_schema = schema["properties"]["tags"]
        code_schema = schema["properties"]["code"]
        assert "minimum" not in score_schema
        assert "maximum" not in score_schema
        assert "minItems" not in tags_schema
        assert "maxItems" not in tags_schema
        assert "minLength" not in code_schema
        assert "maxLength" not in code_schema
        assert "pattern" not in code_schema
        nested_label_schema = schema["$defs"]["NestedLabel"]["properties"]["label"]
        assert "minLength" not in nested_label_schema
        assert "maxLength" not in nested_label_schema
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == {"score", "tags", "code", "nested"}

    def test_supported_model_uses_strict_json_schema_response_format(self):
        client = _build_client(LiteLLMConfig(model="gpt-4o-mini"))

        with patch.object(
            LiteLLMClient,
            "_supports_response_schema",
            return_value=True,
        ):
            params, parser_schema, parse_structured, _, _ = (
                client._build_completion_params(
                    [{"role": "user", "content": "test"}],
                    response_format=SampleResponse,
                )
            )

        provider_format = params["response_format"]
        assert provider_format["type"] == "json_schema"
        assert provider_format["json_schema"]["strict"] is True
        assert provider_format["json_schema"]["schema"]["additionalProperties"] is False
        assert set(provider_format["json_schema"]["schema"]["required"]) == {
            "answer",
            "score",
        }
        assert parser_schema is SampleResponse
        assert parse_structured is True

    def test_explicit_none_model_falls_back_to_config_default(self):
        # Callers that forward an optional model (e.g. the eval judges pass
        # ``model=rubric.get("judge_model")``) may hand over a literal None;
        # it must resolve to the config default instead of crashing on
        # ``None.lower()`` during API-key resolution.
        client = _build_client(LiteLLMConfig(model="gpt-4o-mini"))

        params, _, _, _, _ = client._build_completion_params(
            [{"role": "user", "content": "test"}],
            model=None,
        )

        assert params["model"] == "gpt-4o-mini"

    def test_unsupported_model_keeps_pydantic_response_format(self):
        # A provider that is neither natively response-schema-capable nor on the
        # OpenAI-compatible allowlist keeps the raw Pydantic model so LiteLLM can
        # apply its own provider-specific handling (e.g. tool-calling).
        client = _build_client(LiteLLMConfig(model="ollama/llama3"))

        with patch.object(
            LiteLLMClient,
            "_supports_response_schema",
            return_value=False,
        ):
            params, parser_schema, _, _, _ = client._build_completion_params(
                [{"role": "user", "content": "test"}],
                response_format=SampleResponse,
            )

        assert params["response_format"] is SampleResponse
        assert parser_schema is SampleResponse

    def test_zai_uses_coding_endpoint_and_prompt_backed_json_mode(self):
        client = _build_client(LiteLLMConfig(model="zai/glm-5.2"))
        messages = [{"role": "user", "content": "test"}]

        params, parser_schema, parse_structured, _, _ = client._build_completion_params(
            messages,
            response_format=SampleResponse,
        )

        assert params["api_base"] == "https://api.z.ai/api/coding/paas/v4"
        assert params["response_format"] == {"type": "json_object"}
        assert "response_format" in params["allowed_openai_params"]
        assert params["messages"][0]["role"] == "system"
        instruction = params["messages"][0]["content"]
        assert "Return ONLY a JSON object" in instruction
        assert '"answer"' in instruction
        assert '"score"' in instruction
        assert messages == [{"role": "user", "content": "test"}]
        assert parser_schema is SampleResponse
        assert parse_structured is True

    def test_minimax_carries_the_schema_in_the_prompt_not_response_format(self):
        """MiniMax ignores ``response_format``, so the schema must be in the prompt.

        Measured against the live API: two identical calls differing only in
        ``drop_params`` both returned free prose rather than JSON, so a
        ``json_schema`` response_format is discarded however it is sent. The
        symptom was an analyst inventing a different set of field names on
        every run -- answering from the prompt alone, never having been given
        a schema. Asserting the field NAMES appear in the instruction is the
        point: a bare ``{"type": "json_object"}`` would leave the model to
        guess them, which is the defect this guards.
        """
        client = _build_client(
            LiteLLMConfig(
                model="minimax/MiniMax-M3",
                api_key_config=APIKeyConfig(minimax=MiniMaxConfig(api_key="mm-key")),
            )
        )
        messages = [{"role": "user", "content": "test"}]

        params, parser_schema, parse_structured, _, _ = client._build_completion_params(
            messages,
            response_format=SampleResponse,
        )

        assert params["response_format"] == {"type": "json_object"}
        instruction = params["messages"][0]["content"]
        assert params["messages"][0]["role"] == "system"
        assert "Return ONLY a JSON object" in instruction
        assert '"answer"' in instruction
        assert '"score"' in instruction
        # The caller's messages are not mutated, and local parsing stays typed.
        assert messages == [{"role": "user", "content": "test"}]
        assert parser_schema is SampleResponse
        assert parse_structured is True

    def test_zai_tool_turn_leaves_tools_free_and_constrains_only_terminus(self):
        client = _build_client(LiteLLMConfig(model="zai/glm-5.2"))
        messages = [
            {"role": "system", "content": "Use tools when needed."},
            {"role": "user", "content": "test"},
        ]
        tool_specs = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look something up.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        params, parser_schema, _, _, _ = client._build_completion_params(
            messages,
            response_format=SampleResponse,
            tools=tool_specs,
        )

        assert "response_format" not in params
        assert params["tools"] == tool_specs
        system_content = params["messages"][0]["content"]
        assert system_content.startswith("Use tools when needed.")
        assert (
            "When you are not calling a tool and are ready to finish" in system_content
        )
        assert messages[0]["content"] == "Use tools when needed."
        assert parser_schema is SampleResponse

    def test_zai_preserves_explicit_api_base_and_allowed_params(self):
        client = _build_client(LiteLLMConfig(model="zai/glm-5.2"))

        params, _, _, _, _ = client._build_completion_params(
            [{"role": "user", "content": "test"}],
            response_format=SampleResponse,
            api_base="https://example.test/v4",
            allowed_openai_params=["seed"],
        )

        assert params["api_base"] == "https://example.test/v4"
        assert params["allowed_openai_params"] == ["seed", "response_format"]

    def test_zai_custom_endpoint_takes_precedence_over_builtin_default(self):
        api_key_config = APIKeyConfig(
            custom_endpoint=CustomEndpointConfig(
                model="zai/glm-5.2",
                api_key="custom-key",
                api_base="https://example.com/v1",  # type: ignore[arg-type]
            )
        )
        client = _build_client(
            LiteLLMConfig(
                model="zai/glm-5.2",
                api_key_config=api_key_config,
            )
        )

        params, _, _, _, _ = client._build_completion_params(
            [{"role": "user", "content": "test"}]
        )

        assert params["api_base"] == "https://example.com/v1"

    def test_zai_strict_response_format_false_preserves_passthrough(self):
        client = _build_client(LiteLLMConfig(model="zai/glm-5.2"))
        messages = [{"role": "user", "content": "test"}]

        params, _, _, _, _ = client._build_completion_params(
            messages,
            response_format=SampleResponse,
            strict_response_format=False,
        )

        assert params["response_format"] is SampleResponse
        assert params["messages"] == messages

    def test_openai_compatible_underreported_provider_uses_strict_schema(self):
        # Covers the _JSON_SCHEMA_PROVIDER_ALLOWLIST mechanism: a provider that
        # genuinely accepts a json_schema response_format but that LiteLLM
        # reports as unsupported must receive our own normalized strict schema,
        # not the raw Pydantic model.
        #
        # The allowlist is EMPTY in production -- minimax was its only member
        # and moved to the prompt path, having turned out to ignore
        # response_format outright. So this patches a member in to exercise the
        # mechanism itself. Without that, the code path would have no coverage
        # at all and the next provider added to it would be unguarded.
        client = _build_client(LiteLLMConfig(model="minimax/MiniMax-M3"))

        with (
            patch.object(
                LiteLLMClient,
                "_supports_response_schema",
                return_value=False,
            ),
            patch.object(
                LiteLLMClient,
                "_JSON_SCHEMA_PROVIDER_ALLOWLIST",
                frozenset({"minimax"}),
            ),
            patch.object(
                LiteLLMClient,
                "_PROMPT_SCHEMA_PROVIDER_ALLOWLIST",
                frozenset({"zai"}),
            ),
        ):
            params, parser_schema, parse_structured, _, _ = (
                client._build_completion_params(
                    [{"role": "user", "content": "test"}],
                    response_format=SampleResponse,
                )
            )

        provider_format = params["response_format"]
        assert provider_format["type"] == "json_schema"
        assert provider_format["json_schema"]["strict"] is True
        assert parser_schema is SampleResponse
        assert parse_structured is True

    def test_discriminated_union_strips_oneof_for_underreported_provider(self):
        # The exact shape behind PYTHON-FASTAPI-9J: a discriminated-union output
        # whose Pydantic schema carries `oneOf`/`discriminator` at
        # properties.<field>.items. On an under-reported OpenAI-compatible
        # provider (minimax), the schema actually sent must contain neither
        # (strict structured-output endpoints reject `oneOf`).

        # Sanity: the raw Pydantic schema really does emit the rejected keyword.
        raw_items = _DiscriminatedOutput.model_json_schema()["properties"]["decisions"][
            "items"
        ]
        assert "oneOf" in raw_items

        client = _build_client(LiteLLMConfig(model="minimax/MiniMax-M3"))
        # ``_DiscriminatedOutput`` is intentionally a non-base double (emits oneOf)
        # to exercise make_strict's prod backstop. The by-construction boundary
        # guard would (correctly) raise on it under pytest, so patch it to a no-op
        # here — this test asserts the make_strict fallback, not the guard.
        # The json-schema allowlist is EMPTY in production (minimax moved to the
        # prompt path), so a member is patched in to exercise the mechanism.
        with (
            patch.object(
                LiteLLMClient, "_supports_response_schema", return_value=False
            ),
            patch.object(
                LiteLLMClient,
                "_JSON_SCHEMA_PROVIDER_ALLOWLIST",
                frozenset({"minimax"}),
            ),
            patch.object(
                LiteLLMClient, "_PROMPT_SCHEMA_PROVIDER_ALLOWLIST", frozenset({"zai"})
            ),
            patch(
                "reflexio.server.llm._litellm_structured_output.assert_provider_safe_schema"
            ),
        ):
            params, _, _, _, _ = client._build_completion_params(
                [{"role": "user", "content": "test"}],
                response_format=_DiscriminatedOutput,
            )

        provider_format = params["response_format"]
        assert provider_format["type"] == "json_schema"
        sent_schema = provider_format["json_schema"]["schema"]
        assert not find_schema_keyword(sent_schema, "oneOf")
        assert not find_schema_keyword(sent_schema, "discriminator")
        # The variants are preserved as `anyOf` so generation stays constrained.
        assert "anyOf" in sent_schema["properties"]["decisions"]["items"]

    def test_real_minimax_gate_normalizes_without_mocking_predicate(self):
        # The bug slipped because every strict-schema test PATCHED
        # _supports_response_schema. This one does NOT — it exercises the real
        # litellm capability lookup + allowlist for the actual default prod model
        # (minimax). With the discriminated-union output, the response_format
        # actually built must be a normalized dict with no oneOf/discriminator.
        client = _build_client(LiteLLMConfig(model="minimax/MiniMax-M3"))
        # Non-base double exercises the make_strict backstop; patch the
        # by-construction guard (it would raise under pytest) — see the sibling
        # test above for why.
        with patch(
            "reflexio.server.llm._litellm_structured_output.assert_provider_safe_schema"
        ):
            params, _, _, _, _ = client._build_completion_params(
                [{"role": "user", "content": "test"}],
                response_format=_DiscriminatedOutput,
            )
        # minimax now carries its schema in the PROMPT -- it ignores
        # response_format outright -- so the normalization invariant moved
        # transport with it. It did not stop mattering: an unfolded `oneOf`
        # trips assert_provider_safe_schema and raises before the request is
        # even built, so a prompt-only provider could otherwise never carry a
        # discriminated union at all.
        assert params["response_format"] == {"type": "json_object"}
        instruction = params["messages"][0]["content"]
        # `prompt_schema_instruction` renders "<prose>\n\n<json.dumps(schema)>",
        # and json.dumps(indent=2) emits no blank lines, so the first blank
        # line is an unambiguous separator.
        prose, _, schema_text = instruction.partition("\n\n")
        assert "JSON Schema" in prose
        schema = json.loads(schema_text)
        assert not find_schema_keyword(schema, "oneOf")
        assert not find_schema_keyword(schema, "discriminator")

    def test_finder_ignores_property_named_like_a_keyword(self):
        # A field literally named `oneOf`/`discriminator` is a property NAME, not a
        # schema keyword, and must not trip the strict-schema guard (the finder is
        # context-aware — CodeRabbit false-positive fix).
        class _TrickyNames(BaseModel):
            oneOf: str = ""  # noqa: N815  (deliberately a keyword-like field name)
            discriminator: int = 0

        schema = make_strict_json_schema(_TrickyNames.model_json_schema())
        assert "oneOf" in schema["properties"]
        assert "discriminator" in schema["properties"]
        assert not find_schema_keyword(schema, "oneOf")
        assert not find_schema_keyword(schema, "discriminator")

    def test_strict_response_format_can_be_disabled_per_call(self):
        client = _build_client(LiteLLMConfig(model="gpt-4o-mini"))

        with patch.object(
            LiteLLMClient,
            "_supports_response_schema",
            return_value=True,
        ):
            params, _, _, _, _ = client._build_completion_params(
                [{"role": "user", "content": "test"}],
                response_format=SampleResponse,
                strict_response_format=False,
            )

        assert params["response_format"] is SampleResponse


# ===================================================================
# Retry-on-parse-failure tests
# ===================================================================


class TestStructuredOutputRetry:
    """Tests for retry behaviour when _maybe_parse_structured_output raises."""

    def _make_mock_response(self, content: str) -> MagicMock:
        """Build a mock litellm.completion response with given content."""
        choice = MagicMock()
        choice.message.content = content
        choice.message.tool_calls = None
        choice.finish_reason = "stop"
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        resp.usage.prompt_tokens_details = None
        resp.usage.cache_creation_input_tokens = None
        resp.usage.cache_read_input_tokens = None
        return resp

    def test_structured_output_parse_failure_retries_and_succeeds(self):
        """Malformed JSON on first attempt, valid on second — retry eventually succeeds."""
        call_count = 0
        valid_json = '{"answer": "ok", "score": 42}'

        def fake_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            content = "not valid json {{{{" if call_count == 1 else valid_json
            return self._make_mock_response(content)

        client = _build_client(
            LiteLLMConfig(model="gpt-4o-mini", max_retries=3, retry_delay=0)
        )

        with patch("litellm.completion", side_effect=fake_completion):
            result = client.generate_chat_response(
                messages=[{"role": "user", "content": "test"}],
                response_format=SampleResponse,
            )

        assert call_count == 2
        assert isinstance(result, SampleResponse)
        assert result.answer == "ok"
        assert result.score == 42

    def test_structured_output_parse_failure_all_retries_exhausted_raises(self):
        """Every attempt returns malformed content — raises LiteLLMClientError.

        Post-refactor, client-side parse-retry is decoupled from
        ``max_retries`` (which now feeds litellm's ``num_retries`` for
        transport-level errors). A malformed structured response is treated
        as a 200 by litellm, so it falls through to our explicit one-shot
        parse-retry. After that single retry also fails, we surface the
        error. With max_retries=2 we still expect exactly 2 total calls.
        """
        call_count = 0

        def fake_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            return self._make_mock_response("not valid json at all {{{{")

        client = _build_client(
            LiteLLMConfig(model="gpt-4o-mini", max_retries=2, retry_delay=0)
        )

        with (
            patch("litellm.completion", side_effect=fake_completion),
            pytest.raises(LiteLLMClientError),
        ):
            client.generate_chat_response(
                messages=[{"role": "user", "content": "test"}],
                response_format=SampleResponse,
            )

        assert call_count == 2

    def test_structured_output_parse_failure_extra_retry_at_default_budget(self):
        """With max_retries=1, a parse failure still gets one extra attempt and can recover."""
        call_count = 0
        valid_json = '{"answer": "ok", "score": 42}'

        def fake_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            content = "not valid json {{{{" if call_count == 1 else valid_json
            return self._make_mock_response(content)

        client = _build_client(
            LiteLLMConfig(model="gpt-4o-mini", max_retries=1, retry_delay=0)
        )

        with patch("litellm.completion", side_effect=fake_completion):
            result = client.generate_chat_response(
                messages=[{"role": "user", "content": "test"}],
                response_format=SampleResponse,
            )

        assert call_count == 2
        assert isinstance(result, SampleResponse)
        assert result.answer == "ok"

    def test_non_parse_error_gets_no_extra_retry_at_default_budget(self):
        """A non-parse error with max_retries=1 still runs exactly once — the extra retry is parse-only."""
        call_count = 0

        def fake_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("boom")

        client = _build_client(
            LiteLLMConfig(model="gpt-4o-mini", max_retries=1, retry_delay=0)
        )

        with (
            patch("litellm.completion", side_effect=fake_completion),
            pytest.raises(LiteLLMClientError),
        ):
            client.generate_chat_response(
                messages=[{"role": "user", "content": "test"}],
                response_format=SampleResponse,
            )

        assert call_count == 1

    def _request_end_failure_records(self, caplog):
        return [
            r
            for r in caplog.records
            if "event=llm_request_end" in r.getMessage()
            and "success=False" in r.getMessage()
        ]

    def test_transient_upstream_error_logged_at_warning(self, caplog):
        """A transient upstream failure (provider timeout / connection / 529
        overload) logs the request-end failure at WARNING, not ERROR — callers
        own fatality and most degrade gracefully — but still raises."""

        def fake_completion(**kwargs):
            raise TimeoutError("provider hung")  # incl. our LLMHardTimeoutError

        client = _build_client(
            LiteLLMConfig(model="minimax/MiniMax-M3", max_retries=1, retry_delay=0)
        )
        with (
            patch("litellm.completion", side_effect=fake_completion),
            caplog.at_level(logging.DEBUG),
            pytest.raises(LiteLLMClientError),
        ):
            client.generate_chat_response(
                messages=[{"role": "user", "content": "test"}],
                response_format=SampleResponse,
            )

        ends = self._request_end_failure_records(caplog)
        assert ends, "expected a request-end failure log record"
        assert all(r.levelno == logging.WARNING for r in ends)
        assert not any(r.levelno == logging.ERROR for r in ends)

    def test_unexpected_error_still_logged_at_error(self, caplog):
        """A genuinely-unexpected error (not a known transient upstream type)
        stays at ERROR."""

        def fake_completion(**kwargs):
            raise RuntimeError("boom")

        client = _build_client(
            LiteLLMConfig(model="gpt-4o-mini", max_retries=1, retry_delay=0)
        )
        with (
            patch("litellm.completion", side_effect=fake_completion),
            caplog.at_level(logging.DEBUG),
            pytest.raises(LiteLLMClientError),
        ):
            client.generate_chat_response(
                messages=[{"role": "user", "content": "test"}],
                response_format=SampleResponse,
            )

        ends = self._request_end_failure_records(caplog)
        assert ends, "expected a request-end failure log record"
        assert all(r.levelno == logging.ERROR for r in ends)

    def test_parse_exhaustion_logs_request_end_failure(self, caplog):
        """Blind-retry exhaustion (no validator) still emits the request-end
        failure record — litellm saw a 200, so only this layer can log it."""

        def fake_completion(**kwargs):
            choice = MagicMock()
            choice.message.content = '{"answer": "bad", "sco'  # truncated JSON
            choice.message.tool_calls = None
            choice.finish_reason = "stop"
            resp = MagicMock()
            resp.choices = [choice]
            resp.usage = MagicMock(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            )
            resp.usage.prompt_tokens_details = None
            resp.usage.cache_creation_input_tokens = None
            resp.usage.cache_read_input_tokens = None
            return resp

        client = _build_client(LiteLLMConfig(model="primary-model"))
        with (
            patch("litellm.completion", side_effect=fake_completion),
            caplog.at_level(logging.DEBUG),
            pytest.raises(LiteLLMClientError),
        ):
            client.generate_chat_response(
                messages=[{"role": "user", "content": "test"}],
                response_format=SampleResponse,
            )

        ends = self._request_end_failure_records(caplog)
        assert ends, "expected a request-end failure log record"
        assert all(r.levelno == logging.ERROR for r in ends)


class TestStructuredOutputRepair:
    """Tests for opt-in corrective repair of structured output."""

    def _make_mock_response(
        self,
        content: str,
        *,
        finish_reason: str = "stop",
        model: str = "served-model",
    ) -> MagicMock:
        choice = MagicMock()
        choice.message.content = content
        choice.message.tool_calls = None
        choice.finish_reason = finish_reason
        resp = MagicMock()
        resp.choices = [choice]
        resp.model = model
        resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        resp.usage.prompt_tokens_details = None
        resp.usage.cache_creation_input_tokens = None
        resp.usage.cache_read_input_tokens = None
        return resp

    @staticmethod
    def _score_validator(output: BaseModel) -> list[str]:
        assert isinstance(output, SampleResponse)
        if output.score == 42:
            return []
        return [f"score must be 42, got {output.score}"]

    def test_validator_repairs_semantic_failure_on_same_model(self):
        calls: list[dict[str, Any]] = []
        original_messages = [{"role": "user", "content": "test"}]
        responses = [
            '{"answer": "bad", "score": 1}',
            '{"answer": "ok", "score": 42}',
        ]

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return self._make_mock_response(responses[len(calls) - 1])

        client = _build_client(LiteLLMConfig(model="primary-model"))

        with patch("litellm.completion", side_effect=fake_completion):
            result = client.generate_chat_response(
                messages=original_messages,
                response_format=SampleResponse,
                structured_output_validator=self._score_validator,
            )

        assert isinstance(result, SampleResponse)
        assert result.score == 42
        assert [call["model"] for call in calls] == ["primary-model", "primary-model"]
        repair_messages = calls[1]["messages"]
        assert [m["role"] for m in repair_messages] == ["user", "assistant", "user"]
        assert '"score":1' in repair_messages[1]["content"].replace(" ", "")
        assert "score must be 42" in repair_messages[2]["content"]
        assert original_messages == [{"role": "user", "content": "test"}]

    def test_validator_advances_to_fallback_rung_with_original_prompt(self):
        """The owned walk advances to the fallback rung after the primary rung's
        same-model repair budget is exhausted. The fallback rung is a FRESH task:
        it receives the ORIGINAL prompt, never the primary's repair conversation,
        and no ``fallbacks`` kwarg is ever handed to litellm."""
        calls: list[dict[str, Any]] = []
        responses = [
            '{"answer": "bad", "score": 1}',  # primary: semantic fail
            '{"answer": "still bad", "score": 2}',  # primary repair: semantic fail
            '{"answer": "ok", "score": 42}',  # fallback-a: valid
        ]

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return self._make_mock_response(responses[len(calls) - 1])

        client = _build_client(
            LiteLLMConfig(
                model="primary-model",
                fallback_models=[
                    "local/embedder",  # dropped: no litellm completion route
                    "primary-model",  # dropped: self-reference
                    "fallback-a",
                    "fallback-b",
                ],
            )
        )

        with patch("litellm.completion", side_effect=fake_completion):
            result = client.generate_chat_response(
                messages=[{"role": "user", "content": "test"}],
                response_format=SampleResponse,
                structured_output_validator=self._score_validator,
            )

        assert isinstance(result, SampleResponse)
        assert result.score == 42
        # primary rung (initial + one same-model repair), then advance to fallback-a.
        assert [call["model"] for call in calls] == [
            "primary-model",
            "primary-model",
            "fallback-a",
        ]
        # No fallbacks delegated to litellm on any rung — the walk owns advancement.
        assert all("fallbacks" not in call for call in calls)
        # The fallback rung gets the ORIGINAL prompt, not the [user, assistant,
        # user] repair conversation from the primary rung.
        assert [m["role"] for m in calls[2]["messages"]] == ["user"]
        assert calls[2]["messages"][0]["content"] == "test"

    def test_validator_exhaustion_raises_typed_error_with_latest_response(self):
        responses = [
            '{"answer": "bad", "score": 1}',
            '{"answer": "latest secret", "score": 2}',
        ]

        def fake_completion(**kwargs):
            return self._make_mock_response(responses.pop(0))

        client = _build_client(LiteLLMConfig(model="primary-model"))

        with (
            patch("litellm.completion", side_effect=fake_completion),
            pytest.raises(StructuredOutputRepairError) as exc_info,
        ):
            client.generate_chat_response(
                messages=[{"role": "user", "content": "test"}],
                response_format=SampleResponse,
                structured_output_validator=self._score_validator,
            )

        err = exc_info.value
        assert err.failure_kind == "semantic"
        assert err.model == "primary-model"
        assert err.raw_content == '{"answer": "latest secret", "score": 2}'
        assert isinstance(err.parsed_output, SampleResponse)
        assert err.parsed_output.score == 2
        assert err.validation_errors == ("score must be 42, got 2",)
        assert "latest secret" not in str(err)

    def test_validator_requires_structured_parsing(self):
        client = _build_client(LiteLLMConfig(model="primary-model"))

        with pytest.raises(ValueError):
            client.generate_chat_response(
                messages=[{"role": "user", "content": "test"}],
                structured_output_validator=self._score_validator,
            )

        with pytest.raises(ValueError):
            client.generate_chat_response(
                messages=[{"role": "user", "content": "test"}],
                response_format=SampleResponse,
                parse_structured_output=False,
                structured_output_validator=self._score_validator,
            )

    def test_validator_does_not_repair_intermediate_tool_calls(self):
        calls: list[dict[str, Any]] = []
        tool_call = MagicMock()
        tool_call.function.name = "lookup"
        choice = MagicMock()
        choice.message.content = None
        choice.message.tool_calls = [tool_call]
        choice.finish_reason = "tool_calls"
        response = MagicMock()
        response.choices = [choice]
        response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        response.usage.prompt_tokens_details = None
        response.usage.cache_creation_input_tokens = None
        response.usage.cache_read_input_tokens = None

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return response

        def fail_validator(output: BaseModel) -> list[str]:
            raise AssertionError(f"validator should not run for tool call: {output!r}")

        client = _build_client(LiteLLMConfig(model="primary-model"))

        with patch("litellm.completion", side_effect=fake_completion):
            result = client.generate_chat_response(
                messages=[{"role": "user", "content": "test"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                            },
                        },
                    }
                ],
                response_format=SampleResponse,
                structured_output_validator=fail_validator,
            )

        assert isinstance(result, ToolCallingChatResponse)
        assert result.tool_calls == [tool_call]
        assert len(calls) == 1
        assert calls[0]["messages"] == [{"role": "user", "content": "test"}]

    def test_repair_triggers_on_parse_failure_first_attempt(self):
        calls: list[dict[str, Any]] = []
        responses = [
            '{"answer": "bad", "sco',  # truncated JSON -> parse error
            '{"answer": "ok", "score": 42}',
        ]

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return self._make_mock_response(responses[len(calls) - 1])

        client = _build_client(LiteLLMConfig(model="primary-model"))

        with patch("litellm.completion", side_effect=fake_completion):
            result = client.generate_chat_response(
                messages=[{"role": "user", "content": "test"}],
                response_format=SampleResponse,
                structured_output_validator=self._score_validator,
            )

        assert isinstance(result, SampleResponse)
        assert result.score == 42
        assert len(calls) == 2
        repair_messages = calls[1]["messages"]
        assert [m["role"] for m in repair_messages] == ["user", "assistant", "user"]
        # The malformed output is echoed back verbatim for the corrective turn.
        assert repair_messages[1]["content"] == '{"answer": "bad", "sco'
        assert "truncated" in repair_messages[2]["content"]

    def test_repair_error_keeps_initial_parse_failure_provenance(self):
        responses = [
            ('{"answer": "bad", "sco', "served-primary"),
            ('{"answer": "still bad", "score": 1}', "served-repair"),
        ]

        def fake_completion(**_kwargs):
            content, model = responses.pop(0)
            return self._make_mock_response(content, model=model)

        client = _build_client(LiteLLMConfig(model="primary-model"))

        with (
            patch("litellm.completion", side_effect=fake_completion),
            pytest.raises(StructuredOutputRepairError) as exc_info,
        ):
            client.generate_chat_response(
                messages=[{"role": "user", "content": "test"}],
                response_format=SampleResponse,
                structured_output_validator=self._score_validator,
            )

        err = exc_info.value
        assert err.first_parsed_provenance is not None
        assert err.first_parsed_provenance.model_name == "served-repair"

    def test_repair_names_schema_error_without_echoing_field_content(self):
        calls: list[dict[str, Any]] = []
        responses = [
            '{"answer": "customer text"}',
            '{"answer": "ok", "score": 42}',
        ]

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return self._make_mock_response(responses[len(calls) - 1])

        client = _build_client(LiteLLMConfig(model="primary-model"))

        with patch("litellm.completion", side_effect=fake_completion):
            result = client.generate_chat_response(
                messages=[{"role": "user", "content": "test"}],
                response_format=SampleResponse,
                structured_output_validator=self._score_validator,
            )

        assert isinstance(result, SampleResponse)
        repair_instruction = calls[1]["messages"][-1]["content"]
        assert "score: missing" in repair_instruction
        assert "customer text" not in repair_instruction

    def test_repair_echo_replaces_length_truncated_output(self):
        calls: list[dict[str, Any]] = []
        responses = [
            '{"answer": "bad", "sco',
            '{"answer": "ok", "score": 42}',
        ]

        def fake_completion(**kwargs):
            calls.append(kwargs)
            index = len(calls) - 1
            return self._make_mock_response(
                responses[index],
                finish_reason="length" if index == 0 else "stop",
            )

        client = _build_client(LiteLLMConfig(model="primary-model"))

        with patch("litellm.completion", side_effect=fake_completion):
            result = client.generate_chat_response(
                messages=[{"role": "user", "content": "test"}],
                response_format=SampleResponse,
                structured_output_validator=self._score_validator,
            )

        assert isinstance(result, SampleResponse)
        # A length-truncated response is NOT echoed back; the placeholder
        # tells the model its previous output overflowed instead.
        repair_echo = calls[1]["messages"][1]["content"]
        assert repair_echo.startswith("(output truncated at")

    def test_refusal_short_circuits_repair(self):
        calls: list[dict[str, Any]] = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            resp = self._make_mock_response('{"answer": "no", "score": 1}')
            resp.choices[0].message.refusal = "I cannot help with that."
            return resp

        client = _build_client(LiteLLMConfig(model="primary-model"))

        with (
            patch("litellm.completion", side_effect=fake_completion),
            pytest.raises(StructuredOutputRepairError) as exc_info,
        ):
            client.generate_chat_response(
                messages=[{"role": "user", "content": "test"}],
                response_format=SampleResponse,
                structured_output_validator=self._score_validator,
            )

        assert exc_info.value.failure_kind == "refusal"
        assert len(calls) == 1

    def test_repair_transport_failure_raises_client_error_not_repair_error(self):
        """Callers that keep the first parsed output (e.g. the consolidator)
        rely on repair-turn transport failures surfacing as LiteLLMClientError."""
        calls: list[dict[str, Any]] = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return self._make_mock_response('{"answer": "bad", "score": 1}')
            raise RuntimeError("connection dropped")

        client = _build_client(LiteLLMConfig(model="primary-model"))

        with (
            patch("litellm.completion", side_effect=fake_completion),
            pytest.raises(LiteLLMClientError) as exc_info,
        ):
            client.generate_chat_response(
                messages=[{"role": "user", "content": "test"}],
                response_format=SampleResponse,
                structured_output_validator=self._score_validator,
            )

        assert not isinstance(exc_info.value, StructuredOutputRepairError)
        assert exc_info.value.first_parsed_provenance is not None
        assert exc_info.value.first_parsed_provenance.model_name == "served-model"
        assert len(calls) == 2

    def test_exhaustion_keeps_latest_parsed_output_after_final_parse_failure(self):
        """Within a single rung: when the corrective turn fails to PARSE, the typed
        error's parsed_output rolls forward to the most recent attempt that DID
        parse (the initial semantic-fail), not None."""
        responses = [
            '{"answer": "first", "score": 2}',  # initial: parses, semantic failure
            '{"answer": "esc", "sco',  # repair turn: parse failure
        ]
        served_models = ["served-primary", "served-repair"]

        def fake_completion(**kwargs):
            return self._make_mock_response(
                responses.pop(0), model=served_models.pop(0)
            )

        client = _build_client(LiteLLMConfig(model="primary-model"))

        with (
            patch("litellm.completion", side_effect=fake_completion),
            pytest.raises(StructuredOutputRepairError) as exc_info,
        ):
            client.generate_chat_response(
                messages=[{"role": "user", "content": "test"}],
                response_format=SampleResponse,
                structured_output_validator=self._score_validator,
            )

        err = exc_info.value
        assert err.failure_kind == "parse"
        assert err.raw_content == '{"answer": "esc", "sco'
        assert isinstance(err.parsed_output, SampleResponse)
        assert err.parsed_output.score == 2
        assert err.first_parsed_provenance is not None
        assert err.first_parsed_provenance.model_name == "served-primary"

    def test_ladder_preserves_first_parsed_provenance_across_rungs(self):
        """Salvage attribution must match the first parse of the whole walk.

        A shared validator closure keeps the first parsed *content* across rungs.
        Without ladder-wide first_parsed_provenance, the consolidator would pair
        that content with the last rung's model.
        """
        # Per rung: initial semantic fail + same-model repair semantic fail.
        responses = [
            ('{"answer": "primary", "score": 1}', "served-primary"),
            ('{"answer": "primary-repair", "score": 2}', "served-primary-repair"),
            ('{"answer": "fallback", "score": 3}', "served-fallback"),
            ('{"answer": "fallback-repair", "score": 4}', "served-fallback-repair"),
        ]

        def fake_completion(**_kwargs):
            content, model = responses.pop(0)
            return self._make_mock_response(content, model=model)

        client = _build_client(
            LiteLLMConfig(model="primary-model", fallback_models=["fallback-model"])
        )

        with (
            patch("litellm.completion", side_effect=fake_completion),
            pytest.raises(StructuredOutputRepairError) as exc_info,
        ):
            client.generate_chat_response(
                messages=[{"role": "user", "content": "test"}],
                response_format=SampleResponse,
                structured_output_validator=self._score_validator,
            )

        err = exc_info.value
        assert err.model == "fallback-model"
        assert err.first_parsed_provenance is not None
        assert err.first_parsed_provenance.model_name == "served-primary"

    def test_ladder_preserves_first_parsed_when_final_rung_cap_saturates(self):
        """Fail-closed cap on the last rung must not drop first-parsed attribution.

        ProviderCapSaturatedError is not a LiteLLMClientError subclass. The outer
        ladder must wrap it and keep ladder-wide first_parsed_provenance so
        consolidator salvage pairs primary content with the primary served model.
        """
        from reflexio.server.llm._provider_concurrency import (  # noqa: PLC0415
            ProviderCapSaturatedError,
        )

        call_count = 0

        def fake_completion(**_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return self._make_mock_response(
                    '{"answer": "primary", "score": 1}', model="served-primary"
                )
            # Same-model repair + every later rung: fail-closed provider cap.
            raise ProviderCapSaturatedError("provider cap saturated")

        client = _build_client(
            LiteLLMConfig(model="primary-model", fallback_models=["fallback-model"])
        )

        with (
            patch("litellm.completion", side_effect=fake_completion),
            pytest.raises(LiteLLMClientError) as exc_info,
        ):
            client.generate_chat_response(
                messages=[{"role": "user", "content": "test"}],
                response_format=SampleResponse,
                structured_output_validator=self._score_validator,
            )

        err = exc_info.value
        assert not isinstance(err, StructuredOutputRepairError)
        assert err.first_parsed_provenance is not None
        assert err.first_parsed_provenance.model_name == "served-primary"


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
        assert _extract_json_from_string(content) == '{"key": "value"}'

    def test_json_in_markdown_block(self, client):
        content = '```json\n{"key": "value"}\n```'
        result = _extract_json_from_string(content)
        assert result == '{"key": "value"}'

    def test_json_in_plain_code_block(self, client):
        content = '```\n{"key": "value"}\n```'
        result = _extract_json_from_string(content)
        assert result == '{"key": "value"}'

    def test_json_array(self, client):
        content = "Some text before [1, 2, 3] some text after"
        result = _extract_json_from_string(content)
        assert result == "[1, 2, 3]"

    def test_json_object_in_text(self, client):
        content = 'Here is the result: {"answer": 42} that is all'
        result = _extract_json_from_string(content)
        assert result == '{"answer": 42}'

    def test_json_object_ignores_stray_braces_in_text(self, client):
        content = 'Result {not json}: {"answer": 42, "why": "{kept}"} trailing {x}'
        result = _extract_json_from_string(content)
        assert result == '{"answer": 42, "why": "{kept}"}'

    def test_json_array_ignores_stray_brackets_in_text(self, client):
        content = 'Candidates [not json] then [{"answer": 42}] trailing [x]'
        result = _extract_json_from_string(content)
        assert result == '[{"answer": 42}]'

    def test_json_object_with_markdown_fence_in_string(self, client):
        content = json.dumps(
            {
                "key": "Use:\n```bash\nsupabase start\n```",
                "value": 42,
            }
        )
        result = _extract_json_from_string(content)
        assert json.loads(result) == {
            "key": "Use:\n```bash\nsupabase start\n```",
            "value": 42,
        }

    def test_no_json_returns_original(self, client):
        content = "plain text"
        assert _extract_json_from_string(content) == "plain text"


# ===================================================================
# _sanitize_json_string tests
# ===================================================================


class TestSanitizeJsonString:
    """Tests for Python-to-JSON sanitization."""

    @pytest.fixture()
    def client(self):
        return _build_client()

    def test_single_quotes_to_double(self, client):
        result = _sanitize_json_string("{'key': 'value'}")
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_python_booleans(self, client):
        result = _sanitize_json_string('{"flag": True, "other": False}')
        parsed = json.loads(result)
        assert parsed == {"flag": True, "other": False}

    def test_python_none(self, client):
        result = _sanitize_json_string('{"val": None}')
        parsed = json.loads(result)
        assert parsed == {"val": None}

    def test_trailing_commas(self, client):
        result = _sanitize_json_string('{"a": 1, "b": 2, }')
        parsed = json.loads(result)
        assert parsed == {"a": 1, "b": 2}

    def test_escaped_apostrophe_in_single_quoted(self, client):
        result = _sanitize_json_string("{'text': 'didn\\'t work'}")
        parsed = json.loads(result)
        assert parsed["text"] == "didn't work"

    def test_double_quotes_inside_single_quoted_escaped(self, client):
        result = _sanitize_json_string("{'key': 'he said \"hello\"'}")
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
            "gpt-5.4-mini",
            "gpt-5-nano",
            "gpt-5-codex",
            "GPT-5.4-Mini",
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
        config = LiteLLMConfig(model="gpt-5.4-mini", temperature=0.7)
        client = LiteLLMClient(config)

        client.generate_response("hi")

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["temperature"] == 1.0

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_non_restricted_model_default_injects_seed_only(
        self, mock_completion, monkeypatch
    ):
        """Default behavior: seed=42 is injected, but caller-configured
        temperature flows through unchanged (the temperature override is
        opt-in via an explicit REFLEXIO_LLM_SEED env var)."""
        monkeypatch.delenv("REFLEXIO_LLM_SEED", raising=False)
        mock_completion.return_value = _make_completion_response("ok")
        config = LiteLLMConfig(model="gpt-4o", temperature=0.3)
        client = LiteLLMClient(config)

        client.generate_response("hi")

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["seed"] == 42
        assert call_kwargs["drop_params"] is True
        assert call_kwargs["temperature"] == 0.3

    def test_drop_params_is_scoped_to_completion_params(self, monkeypatch):
        """Best-effort seed should not change LiteLLM's process-global setting."""
        monkeypatch.setattr(litellm, "drop_params", False)
        client = LiteLLMClient(LiteLLMConfig(model="gpt-4o"))

        params, _, _, _, _ = client._build_completion_params(
            [{"role": "user", "content": "hi"}]
        )

        assert params["drop_params"] is True
        assert litellm.drop_params is False

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_explicit_seed_env_forces_temperature_zero(
        self, mock_completion, monkeypatch
    ):
        """Explicit REFLEXIO_LLM_SEED opt-in: seed is injected AND temperature
        is forced to 0.0 on non-restricted models for reproducible sampling."""
        monkeypatch.setenv("REFLEXIO_LLM_SEED", "7")
        mock_completion.return_value = _make_completion_response("ok")
        config = LiteLLMConfig(model="gpt-4o", temperature=0.3)
        client = LiteLLMClient(config)

        client.generate_response("hi")

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["seed"] == 7
        assert call_kwargs["temperature"] == 0.0

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_invalid_seed_env_falls_back_to_default(self, mock_completion, monkeypatch):
        """A non-integer REFLEXIO_LLM_SEED falls back to seed=42 so the
        'always inject a seed' contract holds; the temperature override still
        fires because the operator explicitly opted into determinism."""
        monkeypatch.setenv("REFLEXIO_LLM_SEED", "not-an-int")
        mock_completion.return_value = _make_completion_response("ok")
        config = LiteLLMConfig(model="gpt-4o", temperature=0.3)
        client = LiteLLMClient(config)

        client.generate_response("hi")

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["seed"] == 42
        assert call_kwargs["temperature"] == 0.0


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
    def test_minimax_gets_default_max_tokens_cap(self, mock_completion):
        """Unset max_tokens on a MiniMax model applies the provider cap.

        MiniMax-M3 with unbounded output deterministically stalls into the
        120s litellm timeout (prod consolidator/document-expansion outage,
        2026-07-14). The provider-level default in model_defaults bounds it.
        """
        mock_completion.return_value = _make_completion_response("ok")
        client = LiteLLMClient(LiteLLMConfig(model="minimax/MiniMax-M3"))

        client.generate_response("hi")

        assert mock_completion.call_args.kwargs["max_tokens"] == 8192

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_minimax_explicit_max_tokens_beats_provider_cap(self, mock_completion):
        """Config/call-site max_tokens overrides the provider default cap."""
        mock_completion.return_value = _make_completion_response("ok")
        client = LiteLLMClient(
            LiteLLMConfig(model="minimax/MiniMax-M3", max_tokens=100)
        )

        client.generate_response("hi")

        assert mock_completion.call_args.kwargs["max_tokens"] == 100

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_unmapped_provider_stays_unbounded(self, mock_completion):
        """Providers without a default cap keep omitting max_tokens."""
        mock_completion.return_value = _make_completion_response("ok")
        client = LiteLLMClient(LiteLLMConfig(model="gpt-4o"))

        client.generate_response("hi")

        assert "max_tokens" not in mock_completion.call_args.kwargs

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
    def test_minimax_m3_floor_is_120(self, mock_completion):
        """MiniMax-M3's floor was lowered from 240s to the 120s default
        so a hung primary is abandoned sooner and the
        fallback is reached faster."""
        mock_completion.return_value = _make_completion_response("ok")
        client = LiteLLMClient(LiteLLMConfig(model="minimax/MiniMax-M3"))

        client.generate_response("hi")

        assert mock_completion.call_args.kwargs["timeout"] == 120

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_model_timeout_floor_does_not_lower_higher_config(self, mock_completion):
        """A configured timeout above the floor is preserved."""
        mock_completion.return_value = _make_completion_response("ok")
        client = LiteLLMClient(LiteLLMConfig(model="minimax/MiniMax-M3", timeout=600))

        client.generate_response("hi")

        assert mock_completion.call_args.kwargs["timeout"] == 600

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_explicit_timeout_kwarg_beats_model_floor(self, mock_completion):
        """A per-call timeout kwarg bypasses the floor entirely."""
        mock_completion.return_value = _make_completion_response("ok")
        client = LiteLLMClient(LiteLLMConfig(model="minimax/MiniMax-M3"))

        client.generate_chat_response([{"role": "user", "content": "hi"}], timeout=90)

        assert mock_completion.call_args.kwargs["timeout"] == 90

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_model_without_floor_keeps_config_timeout(self, mock_completion):
        mock_completion.return_value = _make_completion_response("ok")
        client = LiteLLMClient(LiteLLMConfig(model="gpt-4o"))

        client.generate_response("hi")

        assert mock_completion.call_args.kwargs["timeout"] == 120

    @patch("reflexio.server.llm.litellm_client.litellm.completion")
    def test_custom_endpoint_overrides_model(self, mock_completion):
        mock_completion.return_value = _make_completion_response("ok")
        api_key_config = APIKeyConfig(
            custom_endpoint=CustomEndpointConfig(
                model="custom-model",
                api_key="ce-key",
                api_base="https://example.com/v1",  # type: ignore[arg-type]
            )
        )
        config = LiteLLMConfig(model="gpt-4o", api_key_config=api_key_config)
        client = LiteLLMClient(config)

        client.generate_response("hi")

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["model"] == "custom-model"
        assert call_kwargs["api_key"] == "ce-key"
        assert call_kwargs["api_base"] == "https://example.com/v1"

    def test_invalid_max_retries_fallback(self):
        config = LiteLLMConfig(model="gpt-4o", max_retries=2)
        client = LiteLLMClient(config)

        params, _, _, max_retries, _ = client._build_completion_params(
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

    def test_claude_code_prefix_is_noop(self, client):
        """claude-code/* routes through the CLI, which cannot accept cache_control blocks."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = client._apply_prompt_caching(messages, "claude-code/default")
        assert result == messages
        assert isinstance(result[0]["content"], str)


# ===================================================================
# claude-code provider routing (API-key resolution guard)
# ===================================================================


class TestResolveByPrefixClaudeCode:
    """claude-code/* must not hit any API-key field in APIKeyConfig."""

    def test_claude_code_returns_all_none(self):
        cfg = APIKeyConfig(
            openai=CommonsOpenAIConfig(api_key="sk-test"),
            anthropic=AnthropicConfig(api_key="ant-test"),
        )
        client = _build_client(
            LiteLLMConfig(model="claude-code/default", api_key_config=cfg)
        )
        assert client._resolve_by_prefix("claude-code/default") == (None, None, None)

    def test_claude_code_does_not_fall_through_to_anthropic_branch(self):
        """Guard against regressions where 'claude' substring match swallows claude-code/*."""
        cfg = APIKeyConfig(anthropic=AnthropicConfig(api_key="ant-test"))
        client = _build_client(
            LiteLLMConfig(model="claude-code/default", api_key_config=cfg)
        )
        api_key, _, _ = client._resolve_by_prefix("claude-code/default")
        assert api_key is None


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
# max_retries clamping edge cases (guard clause in _build_completion_params)
# ===================================================================


class TestMaxRetriesClamping:
    """max_retries clamping in _build_completion_params.

    The completion path now forces ``num_retries=0`` on litellm (the fallback
    list, not same-model retry, is the resilience mechanism — see
    PYTHON-FASTAPI-62), so the clamped value is no longer forwarded. We assert
    the clamp on the value ``_build_completion_params`` returns, which the
    per-call ``max_retries`` override API still validates.
    """

    def test_max_retries_zero_treated_as_one(self):
        """max_retries=0 should be clamped to at least 1 in the returned value."""
        client = LiteLLMClient(LiteLLMConfig(model="gpt-4o", max_retries=0))

        _, _, _, max_retries, _ = client._build_completion_params(
            [{"role": "user", "content": "hi"}]
        )
        assert max_retries == 1

    def test_negative_max_retries_treated_as_one(self):
        """Negative max_retries should be clamped to at least 1."""
        client = LiteLLMClient(LiteLLMConfig(model="gpt-4o", max_retries=-1))

        _, _, _, max_retries, _ = client._build_completion_params(
            [{"role": "user", "content": "hi"}]
        )
        assert max_retries == 1


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
        result = _sanitize_json_string("{'items': ['a', 'b', 'c']}")
        parsed = json.loads(result)
        assert parsed == {"items": ["a", "b", "c"]}

    def test_trailing_comma_in_array(self):
        """Test trailing commas before closing bracket."""
        result = _sanitize_json_string('{"items": [1, 2, 3, ]}')
        parsed = json.loads(result)
        assert parsed == {"items": [1, 2, 3]}

    def test_mixed_python_values(self):
        """Test handling of mixed True/False/None values."""
        result = _sanitize_json_string('{"a": True, "b": False, "c": None}')
        parsed = json.loads(result)
        assert parsed == {"a": True, "b": False, "c": None}

    def test_boolean_inside_string_not_replaced(self):
        """Test that True/False inside strings are not replaced."""
        result = _sanitize_json_string('{"msg": "This is True story"}')
        parsed = json.loads(result)
        # "True" inside the string value should remain unchanged
        assert "True" in parsed["msg"]


class TestBuildCompletionParamsEdgeCases:
    """Additional edge cases for _build_completion_params."""

    def test_max_retries_kwarg_overrides_config(self):
        """Test that max_retries kwarg overrides config value."""
        config = LiteLLMConfig(model="gpt-4o", max_retries=2)
        client = LiteLLMClient(config)

        _, _, _, max_retries, _ = client._build_completion_params(
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

        params, _, _, _, _ = client._build_completion_params(
            [{"role": "user", "content": "hi"}],
            model="claude-3-5-sonnet",
        )
        assert params["model"] == "claude-3-5-sonnet"
        assert params["api_key"] == "ant-key"

    def test_local_embedding_model_is_dropped_from_fallbacks(self):
        """A ``local/*`` in-process embedding model must be filtered out of the
        fallback list: it has no litellm completion route (it is served
        in-process), so handing it to litellm's fallback ladder raises
        ``BadRequestError: LLM Provider NOT provided``.
        Valid generation fallbacks are preserved."""
        config = LiteLLMConfig(
            model="gpt-4o",
            fallback_models=["local/nomic-embed-text-v1.5", "gpt-5-mini"],
        )
        client = LiteLLMClient(config)

        ladder = client._resolve_ladder()
        assert "local/nomic-embed-text-v1.5" not in ladder
        assert ladder == ["gpt-4o", "gpt-5-mini"]


class TestConfigDefaults:
    def test_max_retries_default_is_three(self):
        assert LiteLLMConfig(model="x").max_retries == 3

    def test_fallback_models_default_is_empty_without_env_var(self, monkeypatch):
        """Default is OFF so local reflexio + claude-smart never silently
        route to an unintended provider."""
        monkeypatch.delenv("REFLEXIO_LLM_FALLBACK_MODELS", raising=False)
        assert LiteLLMConfig(model="x").fallback_models == []

    def test_fallback_models_reads_env_var_when_set(self, monkeypatch):
        """Production opt-in: REFLEXIO_LLM_FALLBACK_MODELS=gpt-5.4-mini in
        the deploy env enables the fallback for every chat call site."""
        monkeypatch.setenv("REFLEXIO_LLM_FALLBACK_MODELS", "gpt-5.4-mini")
        assert LiteLLMConfig(model="x").fallback_models == ["gpt-5.4-mini"]

    def test_fallback_models_env_var_is_comma_separated(self, monkeypatch):
        monkeypatch.setenv("REFLEXIO_LLM_FALLBACK_MODELS", "gpt-5.4-mini, gpt-5-nano")
        assert LiteLLMConfig(model="x").fallback_models == [
            "gpt-5.4-mini",
            "gpt-5-nano",
        ]

    def test_fallback_models_can_be_overridden_at_construction(self):
        cfg = LiteLLMConfig(model="x", fallback_models=["gpt-5.4-mini", "gpt-5-nano"])
        assert cfg.fallback_models == ["gpt-5.4-mini", "gpt-5-nano"]

    def test_fallback_models_supports_empty_list_to_disable(self):
        assert LiteLLMConfig(model="x", fallback_models=[]).fallback_models == []


class TestPerCallOverrides:
    def test_per_call_max_retries_forwards_to_make_request(self, monkeypatch):
        client = LiteLLMClient(LiteLLMConfig(model="x", max_retries=3))
        seen_kwargs: dict[str, Any] = {}
        monkeypatch.setattr(
            client,
            "_make_request",
            lambda _messages, **kw: (
                seen_kwargs.update(kw) or CompletionResult("ok", ModelProvenance())
            ),
        )
        client.generate_chat_response(
            [{"role": "user", "content": "hi"}], max_retries=7
        )
        assert seen_kwargs.get("max_retries") == 7

    def test_per_call_fallback_models_forwards_to_make_request(self, monkeypatch):
        client = LiteLLMClient(LiteLLMConfig(model="x"))
        seen_kwargs: dict[str, Any] = {}
        monkeypatch.setattr(
            client,
            "_make_request",
            lambda _messages, **kw: (
                seen_kwargs.update(kw) or CompletionResult("ok", ModelProvenance())
            ),
        )
        client.generate_chat_response(
            [{"role": "user", "content": "hi"}], fallback_models=["claude-x"]
        )
        assert seen_kwargs.get("fallback_models") == ["claude-x"]

    def test_per_call_overrides_optional_default_to_config(self, monkeypatch):
        # When caller doesn't pass, neither flows through -- _make_request reads
        # the config defaults.
        client = LiteLLMClient(LiteLLMConfig(model="x"))
        seen_kwargs: dict[str, Any] = {}
        monkeypatch.setattr(
            client,
            "_make_request",
            lambda _messages, **kw: (
                seen_kwargs.update(kw) or CompletionResult("ok", ModelProvenance())
            ),
        )
        client.generate_chat_response([{"role": "user", "content": "hi"}])
        assert "max_retries" not in seen_kwargs
        assert "fallback_models" not in seen_kwargs


# ===================================================================
# litellm.completion integration: retries + fallback delegation
# ===================================================================


def test_subprocess_snapshot_preserves_provenance_metadata():
    response = _make_completion_response("ok")
    response.model = "claude-sonnet-5"
    response._hidden_params = {
        "reflexio_provider": "claude-code",
        "reflexio_cli_binary": "claude",
        "reflexio_served_model": "claude-sonnet-5",
    }

    snapshot = _snapshot_completion_response(response)

    assert snapshot.model == "claude-sonnet-5"
    assert snapshot._hidden_params == response._hidden_params


class TestLitellmIntegration:
    """Assert _make_request hands the right knobs to litellm.completion."""

    @staticmethod
    def _messages() -> list[dict[str, Any]]:
        return [{"role": "user", "content": "hi"}]

    def test_completion_forces_num_retries_zero(self, monkeypatch):
        """The completion path disables litellm same-model retries regardless of
        config: retrying a *hung* primary num_retries+1 times is what made the
        fallback unreachable and produced the 490s in PYTHON-FASTAPI-62. The
        fallback list is the resilience mechanism instead."""
        client = LiteLLMClient(LiteLLMConfig(model="x", max_retries=3))
        captured: dict[str, Any] = {}

        def _fake(**params):
            captured.update(params)
            return _make_completion_response("ok")

        monkeypatch.setattr("litellm.completion", _fake)
        client.generate_chat_response(self._messages())
        assert captured.get("num_retries") == 0

    def test_config_fallback_used_when_primary_fails(self, monkeypatch):
        """Config-explicit fallback (opt-in at construction): the owned walk
        advances to it when the primary fails, and NEVER hands ``fallbacks`` to
        litellm."""
        client = LiteLLMClient(
            LiteLLMConfig(model="minimax/MiniMax-M3", fallback_models=["gpt-5.4-mini"])
        )
        calls: list[dict[str, Any]] = []

        def _fake(**params):
            calls.append(params)
            if params["model"] == "minimax/MiniMax-M3":
                raise APIConnectionError(message="x", llm_provider="minimax", model="m")
            return _make_completion_response("ok")

        monkeypatch.setattr("litellm.completion", _fake)
        client.generate_chat_response(self._messages())
        assert [c["model"] for c in calls] == ["minimax/MiniMax-M3", "gpt-5.4-mini"]
        assert all("fallbacks" not in c for c in calls)

    def test_no_fallbacks_when_env_var_unset(self, monkeypatch):
        """Local reflexio / claude-smart safety check: with no env var and no
        explicit construction arg, the primary serves alone and no ``fallbacks``
        kwarg is ever passed."""
        monkeypatch.delenv("REFLEXIO_LLM_FALLBACK_MODELS", raising=False)
        client = LiteLLMClient(LiteLLMConfig(model="claude-code/claude-sonnet-4-6"))
        captured: dict[str, Any] = {}

        def _fake(**params):
            captured.update(params)
            return _make_completion_response("ok")

        monkeypatch.setattr("litellm.completion", _fake)
        client.generate_chat_response(self._messages())
        assert "fallbacks" not in captured

    def test_env_var_enables_fallback_globally(self, monkeypatch):
        """Production-style: set the env var and every LiteLLMClient picks it up.
        The owned walk uses it (advances on primary failure) without a
        ``fallbacks`` kwarg."""
        monkeypatch.setenv("REFLEXIO_LLM_FALLBACK_MODELS", "gpt-5.4-mini")
        client = LiteLLMClient(LiteLLMConfig(model="minimax/MiniMax-M3"))
        calls: list[dict[str, Any]] = []

        def _fake(**params):
            calls.append(params)
            if params["model"] == "minimax/MiniMax-M3":
                raise APIConnectionError(message="x", llm_provider="minimax", model="m")
            return _make_completion_response("ok")

        monkeypatch.setattr("litellm.completion", _fake)
        client.generate_chat_response(self._messages())
        assert [c["model"] for c in calls] == ["minimax/MiniMax-M3", "gpt-5.4-mini"]
        assert all("fallbacks" not in c for c in calls)

    def test_per_call_override_wins_over_config(self, monkeypatch):
        client = LiteLLMClient(LiteLLMConfig(model="x", max_retries=3))
        calls: list[dict[str, Any]] = []

        def _fake(**params):
            calls.append(params)
            if params["model"] == "x":
                raise APIConnectionError(message="x", llm_provider="x", model="x")
            return _make_completion_response("ok")

        monkeypatch.setattr("litellm.completion", _fake)
        client.generate_chat_response(
            self._messages(), max_retries=7, fallback_models=["gpt-5.4-mini"]
        )
        # The per-call fallback override drives the walk; num_retries is forced to
        # 0 on every rung regardless of the max_retries override, and no
        # ``fallbacks`` kwarg is delegated to litellm.
        assert [c["model"] for c in calls] == ["x", "gpt-5.4-mini"]
        assert all(c.get("num_retries") == 0 for c in calls)
        assert all("fallbacks" not in c for c in calls)

    def test_fallback_self_reference_deduped(self, monkeypatch):
        """If primary equals a fallback entry, that entry is dropped from the
        resolved ladder."""
        client = LiteLLMClient(
            LiteLLMConfig(
                model="gpt-5.4-mini", fallback_models=["gpt-5.4-mini", "gpt-5-nano"]
            )
        )
        assert client._resolve_ladder() == ["gpt-5.4-mini", "gpt-5-nano"]

    def test_empty_fallbacks_omits_kwarg(self, monkeypatch):
        client = LiteLLMClient(LiteLLMConfig(model="x", fallback_models=[]))
        captured: dict[str, Any] = {}

        def _fake(**params):
            captured.update(params)
            return _make_completion_response("ok")

        monkeypatch.setattr("litellm.completion", _fake)
        client.generate_chat_response(self._messages())
        # The owned walk never delegates a fallback chain to litellm.
        assert "fallbacks" not in captured

    def test_completion_has_client_side_hard_timeout(self, monkeypatch):
        monkeypatch.setenv("REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS", "0")
        client = LiteLLMClient(LiteLLMConfig(model="x", timeout=cast(Any, 0.01)))

        def _slow(**_params):
            time.sleep(1)
            return _make_completion_response("late")

        monkeypatch.setattr("litellm.completion", _slow)

        start = time.perf_counter()
        with pytest.raises(LiteLLMClientError, match="hard timeout"):
            client.generate_chat_response(self._messages())
        # A single subprocess spawn/kill cycle (the blind same-model hard-timeout
        # retry was removed) — still far below the 1s the blocked call would take.
        assert time.perf_counter() - start < 1.0

    @pytest.mark.skipif(
        multiprocessing.get_start_method() != "fork",
        reason="the isolated worker sees the litellm.completion monkeypatch only "
        "under the fork start method (spawn re-imports real litellm); the "
        "drain-before-join behavior under test is exercised on Linux CI/prod",
    )
    def test_large_result_does_not_deadlock_hard_timeout(self, monkeypatch):
        """A large completion payload overflows the OS pipe buffer feeding the
        result queue. If the parent joined the child before draining the queue,
        the child's queue-feeder thread would block on the full pipe, the child
        could not exit, and a finished-but-large result would trip a *false*
        hard timeout. The queue must be drained before join."""
        monkeypatch.setenv("REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS", "0")
        client = LiteLLMClient(LiteLLMConfig(model="x"))

        big = "x" * (2 * 1024 * 1024)  # 2 MB, far exceeds the ~64KB pipe buffer

        def _big(**_params):
            choice = MagicMock()
            choice.message.content = big
            choice.message.tool_calls = None
            choice.finish_reason = "stop"
            resp = MagicMock()
            resp.choices = [choice]
            resp.usage = None
            resp._hidden_params = {}
            resp.model = "x"
            return resp

        monkeypatch.setattr("litellm.completion", _big)
        params = {"model": "x", "messages": self._messages(), "timeout": 0.5}
        # Force the subprocess path (a monkeypatched completion is not module
        # "litellm", so isolation falls to the short-timeout branch).
        assert client._should_process_isolate_completion(0.5, 0.0)

        start = time.perf_counter()
        payload = client._completion_with_hard_timeout(params, hard_timeout=10.0)
        elapsed = time.perf_counter() - start

        assert payload.choices[0].message.content == big
        # Draining-before-join returns immediately; the pre-fix deadlock would
        # instead burn the full hard_timeout and raise LLMHardTimeoutError.
        assert elapsed < 8.0

    def test_worker_snapshots_litellm_api_connection_error(self, monkeypatch):
        """LiteLLM exceptions can dump but fail to load across process queues."""

        def _raise_connection_error(**_params):
            raise APIConnectionError(
                "connection failed",
                llm_provider="openai",
                model="gpt-5-mini",
            )

        monkeypatch.setattr("litellm.completion", _raise_connection_error)
        process_context = multiprocessing.get_context()
        result_queue = process_context.Queue(maxsize=1)

        try:
            _litellm_completion_worker({"model": "gpt-5-mini"}, result_queue)
            status, payload = result_queue.get(timeout=1.0)
        finally:
            result_queue.close()
            result_queue.join_thread()

        assert status == "error"
        assert isinstance(payload, _CompletionErrorSnapshot)
        assert payload.type_name == "APIConnectionError"
        assert "connection failed" in payload.message
        assert payload.model == "gpt-5-mini"
        assert payload.llm_provider == "openai"

    def test_isolated_worker_error_retains_upstream_type(self, monkeypatch):
        """The parent wrapper must retain enough type data for log severity."""
        client = LiteLLMClient(LiteLLMConfig(model="gpt-5-mini"))
        result_queue = MagicMock()
        result_queue.get.return_value = (
            "error",
            _CompletionErrorSnapshot(
                type_name="APIConnectionError",
                message="connection failed",
                model="gpt-5-mini",
                llm_provider="openai",
            ),
        )
        process = MagicMock()
        process.is_alive.return_value = False
        process_context = MagicMock()
        process_context.Queue.return_value = result_queue
        process_context.Process.return_value = process
        monkeypatch.setattr(multiprocessing, "get_context", lambda: process_context)
        monkeypatch.setattr(
            client, "_should_process_isolate_completion", lambda *_args: True
        )
        params = {
            "model": "gpt-5-mini",
            "messages": self._messages(),
            "timeout": 0.5,
        }

        with pytest.raises(LiteLLMClientError) as exc_info:
            client._completion_with_hard_timeout(params, hard_timeout=5.0)

        assert exc_info.value.upstream_error_type == "APIConnectionError"

    def test_hard_timeout_not_retried_at_client_level(self, monkeypatch):
        """A hard timeout is NOT retried at the client level. Same-model retry of
        a hang is exactly what produced the 490s in PYTHON-FASTAPI-62; the
        fallback ladder inside the litellm call is the resilience path instead.
        The timeout surfaces as LiteLLMClientError after a single attempt."""
        client = LiteLLMClient(LiteLLMConfig(model="x"))
        attempts: list[int] = []

        def _always_timeout(params, hard_timeout):
            attempts.append(1)
            raise LLMHardTimeoutError("LLM request exceeded hard timeout")

        monkeypatch.setattr(client, "_completion_with_hard_timeout", _always_timeout)

        with pytest.raises(LiteLLMClientError, match="hard timeout"):
            client.generate_chat_response(self._messages())
        assert len(attempts) == 1

    def test_hard_timeout_is_per_rung_single_attempt(self, monkeypatch):
        """Each rung now owns a per-SINGLE-ATTEMPT hard timeout (not the old
        ladder-wide ``(1 + len(fallbacks)) * per_attempt``). num_retries is 0 and
        no ``fallbacks`` kwarg is delegated — the walk advances between rungs.
        """
        monkeypatch.setenv("REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS", "5")
        client = LiteLLMClient(
            LiteLLMConfig(model="minimax/MiniMax-M3", fallback_models=["gpt-5-mini"])
        )
        captured: dict[str, Any] = {}

        def _capture(params, hard_timeout):
            captured["hard_timeout"] = hard_timeout
            captured["timeout"] = params.get("timeout")
            captured["num_retries"] = params.get("num_retries")
            captured["fallbacks"] = params.get("fallbacks")
            return _make_completion_response("ok")

        monkeypatch.setattr(client, "_completion_with_hard_timeout", _capture)
        client.generate_chat_response(self._messages())

        # MiniMax-M3 floor is 120s; the primary rung's hard timeout is a SINGLE
        # attempt plus one grace buffer — it does NOT scale with fallback count.
        assert captured["timeout"] == 120
        assert captured["num_retries"] == 0
        assert captured["fallbacks"] is None
        assert captured["hard_timeout"] == pytest.approx(120 + 5)

    @pytest.mark.parametrize("fallback_models", [[], ["a"], ["a", "b"]])
    def test_hard_timeout_does_not_scale_with_rung_count(
        self, monkeypatch, fallback_models
    ):
        """The per-rung hard timeout is constant regardless of how many fallbacks
        follow — each rung is bounded to its own single attempt + one grace. A
        hung primary is abandoned after one attempt-worth of time, then the walk
        advances (the PYTHON-FASTAPI-62 fix, now client-owned)."""
        monkeypatch.setenv("REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS", "5")
        client = LiteLLMClient(
            LiteLLMConfig(model="x", timeout=30, fallback_models=fallback_models)
        )
        captured: dict[str, Any] = {}

        def _capture(params, hard_timeout):
            captured["hard_timeout"] = hard_timeout
            return _make_completion_response("ok")

        monkeypatch.setattr(client, "_completion_with_hard_timeout", _capture)
        client.generate_chat_response(self._messages())

        # Only the primary rung runs (it succeeds); its bound is 30 + 5 grace,
        # independent of the fallback count.
        assert captured["hard_timeout"] == pytest.approx(30 + 5)

    def test_fallback_response_returned_when_primary_fails(self, monkeypatch):
        """End-to-end: when the primary fails and a fallback is configured, the
        fallback rung's response is what _make_request returns — via the owned
        walk, with no ``fallbacks`` kwarg handed to litellm."""
        client = LiteLLMClient(
            LiteLLMConfig(model="minimax/MiniMax-M3", fallback_models=["gpt-5-mini"])
        )

        def _fake(**params):
            if params["model"] == "minimax/MiniMax-M3":
                raise APIConnectionError(
                    message="primary down", llm_provider="minimax", model="m"
                )
            return _make_completion_response("from-fallback")

        monkeypatch.setattr("litellm.completion", _fake)
        result = client.generate_chat_response(self._messages())
        assert result == "from-fallback"

    def test_invalid_hard_timeout_grace_env_falls_back(self, monkeypatch, caplog):
        monkeypatch.setenv("REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS", "not-a-float")
        client = LiteLLMClient(LiteLLMConfig(model="x"))

        assert client._hard_timeout_grace_seconds() == 5.0
        assert "Invalid REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS" in caplog.text

    def test_parse_failure_triggers_one_explicit_retry(self, monkeypatch):
        """Pre-refactor parse-retry preserved: when client-side Pydantic
        re-validation raises StructuredOutputParseError, the call retries
        ONCE. LiteLLM's num_retries can't catch this because litellm sees a
        successful 200 — the parse failure is post-hoc."""

        class Strict(BaseModel):
            required_field: str

        client = LiteLLMClient(LiteLLMConfig(model="x"))
        attempts: list[str] = []
        responses_in_order = [
            _make_completion_response("{}"),  # malformed: missing required_field
            _make_completion_response('{"required_field": "ok"}'),
        ]

        def _fake(**params):
            attempts.append(params["model"])
            return responses_in_order.pop(0)

        monkeypatch.setattr("litellm.completion", _fake)

        result = client.generate_chat_response(
            self._messages(), response_format=Strict, parse_structured_output=True
        )
        assert isinstance(result, Strict)
        assert len(attempts) == 2  # initial + one parse-retry

    def test_parse_failure_only_retries_once(self, monkeypatch):
        """If the parse-retry ALSO returns malformed output, the error
        surfaces — no infinite parse-retry loop."""

        class Strict(BaseModel):
            required_field: str

        client = LiteLLMClient(LiteLLMConfig(model="x"))
        attempts: list[str] = []

        def _always_malformed(**params):
            attempts.append(params["model"])
            return _make_completion_response("{}")

        monkeypatch.setattr("litellm.completion", _always_malformed)

        with pytest.raises(LiteLLMClientError):
            client.generate_chat_response(
                self._messages(),
                response_format=Strict,
                parse_structured_output=True,
            )
        assert len(attempts) == 2  # initial + one parse-retry, then give up


# ===================================================================
# Reflexio-owned per-rung fallback walk (L1 contract)
# ===================================================================


class TestOwnedFallbackWalk:
    """Contract for the reflexio-owned ladder walk: no ``fallbacks`` ever reaches
    litellm, each rung rebuilds its own transport, parse-exhaustion advances, the
    provider slot is acquired per rung, and a loop-driven fallback signal fires
    with a reason."""

    def _messages(self):
        return [{"role": "user", "content": "hi"}]

    def test_no_fallbacks_kwarg_ever_passed_to_litellm(self, monkeypatch):
        client = LiteLLMClient(
            LiteLLMConfig(model="minimax/MiniMax-M3", fallback_models=["zai/glm-5.2"])
        )
        seen = []

        def _fake(**params):
            seen.append(params)
            return _make_completion_response("ok")

        monkeypatch.setattr("litellm.completion", _fake)
        client.generate_chat_response(self._messages())
        assert all("fallbacks" not in p for p in seen)

    def test_mixed_ladder_no_longer_raises_and_each_rung_gets_own_transport(
        self, monkeypatch
    ):
        # minimax (native json_schema) primary FAILS → zai (prompt-backed) serves.
        client = LiteLLMClient(
            LiteLLMConfig(model="minimax/MiniMax-M3", fallback_models=["zai/glm-5.2"])
        )
        calls = []

        def _fake(**params):
            calls.append(params)
            if params["model"] == "minimax/MiniMax-M3":
                raise APIConnectionError(
                    message="boom", llm_provider="minimax", model="MiniMax-M3"
                )
            return _make_completion_response('{"value": "ok"}')

        monkeypatch.setattr("litellm.completion", _fake)

        class Out(BaseModel):
            value: str

        client.generate_chat_response(self._messages(), response_format=Out)
        assert [c["model"] for c in calls] == ["minimax/MiniMax-M3", "zai/glm-5.2"]
        # minimax rung got a strict json_schema response_format; zai rung did NOT
        assert calls[0].get("response_format") is not None
        assert calls[1].get("response_format") in (None, {"type": "json_object"})
        # zai rung got the schema directive injected into the system message
        assert any(
            m["role"] == "system" and "value" in m.get("content", "")
            for m in calls[1]["messages"]
        )

    def test_hard_timeout_advances_from_minimax_to_glm(self, monkeypatch, caplog):
        """A killed MiniMax request must advance to the configured GLM rung."""
        client = LiteLLMClient(
            LiteLLMConfig(model="minimax/MiniMax-M3", fallback_models=["zai/glm-5.2"])
        )
        calls: list[str] = []

        def _complete(params, _hard_timeout):
            model = params["model"]
            calls.append(model)
            if model == "minimax/MiniMax-M3":
                raise LLMHardTimeoutError("LLM request exceeded hard timeout")
            return _make_completion_response("from-glm")

        monkeypatch.setattr(client, "_completion_with_hard_timeout", _complete)
        with caplog.at_level(logging.INFO):
            result = client.generate_chat_response(self._messages())

        assert result == "from-glm"
        assert calls == ["minimax/MiniMax-M3", "zai/glm-5.2"]
        assert any(
            "event=llm_fallback_used" in record.message
            and "served_model=zai/glm-5.2" in record.message
            and "reason=transport_error" in record.message
            for record in caplog.records
        )

    def test_isolated_transport_failure_is_warning_when_fallback_serves(
        self, monkeypatch, caplog
    ):
        """A subprocess wrapper must not turn a recoverable outage into ERROR."""
        client = LiteLLMClient(
            LiteLLMConfig(
                model="minimax/MiniMax-M3",
                fallback_models=["zai/glm-5.2"],
            )
        )

        def _complete(params, _hard_timeout):
            if params["model"] == "minimax/MiniMax-M3":
                raise LiteLLMClientError(
                    "isolated worker failed",
                    upstream_error_type="APIConnectionError",
                )
            return _make_completion_response("from-glm")

        monkeypatch.setattr(client, "_completion_with_hard_timeout", _complete)
        with caplog.at_level(logging.INFO):
            result = client.generate_chat_response(self._messages())

        assert result == "from-glm"
        failed = [
            record
            for record in caplog.records
            if "event=llm_request_end" in record.message
            and "success=False" in record.message
        ]
        assert len(failed) == 1
        assert failed[0].levelno == logging.WARNING

    def test_all_rungs_fail_raises_last_error(self, monkeypatch):
        client = LiteLLMClient(
            LiteLLMConfig(model="minimax/MiniMax-M3", fallback_models=["zai/glm-5.2"])
        )

        def _fake(**params):
            raise APIConnectionError(
                message=f"down:{params['model']}",
                llm_provider="x",
                model=params["model"],
            )

        monkeypatch.setattr("litellm.completion", _fake)
        with pytest.raises(LiteLLMClientError, match=r"zai/glm-5\.2"):
            client.generate_chat_response(self._messages())

    def test_parse_exhausted_primary_advances_to_fallback(self, monkeypatch):
        # primary returns HTTP 200 with malformed JSON on BOTH its attempts
        # (initial + one same-model parse-retry), then advances to the fallback.
        client = LiteLLMClient(
            LiteLLMConfig(model="minimax/MiniMax-M3", fallback_models=["zai/glm-5.2"])
        )
        calls = []

        def _fake(**params):
            calls.append(params["model"])
            if params["model"] == "minimax/MiniMax-M3":
                return _make_completion_response("not json")
            return _make_completion_response('{"value": "ok"}')

        monkeypatch.setattr("litellm.completion", _fake)

        class Out(BaseModel):
            value: str

        result = cast(
            Out,
            client.generate_chat_response(self._messages(), response_format=Out),
        )
        assert result.value == "ok"
        # exactly: primary + one same-model parse-retry, then one fallback try
        assert calls == ["minimax/MiniMax-M3", "minimax/MiniMax-M3", "zai/glm-5.2"]

    def test_provider_slot_acquired_per_rung(self, monkeypatch):
        client = LiteLLMClient(
            LiteLLMConfig(model="minimax/MiniMax-M3", fallback_models=["zai/glm-5.2"])
        )
        slots = []
        import reflexio.server.llm._litellm_text_generation as tg

        real_slot = tg.provider_slot

        from contextlib import contextmanager

        @contextmanager
        def _spy(model):
            slots.append(model)
            with real_slot(model):
                yield

        monkeypatch.setattr(tg, "provider_slot", _spy)

        def _fake(**params):
            if params["model"] == "minimax/MiniMax-M3":
                raise APIConnectionError(
                    message="boom", llm_provider="minimax", model="m"
                )
            return _make_completion_response("ok")

        monkeypatch.setattr("litellm.completion", _fake)
        client.generate_chat_response(self._messages())
        assert slots == ["minimax/MiniMax-M3", "zai/glm-5.2"]

    def test_cap_saturation_is_advance_worthy(self, monkeypatch):
        """A fail-closed provider-cap saturation on a rung is caught by the walk
        as advance-worthy (part of the error taxonomy), not surfaced raw."""
        client = LiteLLMClient(
            LiteLLMConfig(model="minimax/MiniMax-M3", fallback_models=["zai/glm-5.2"])
        )
        from contextlib import contextmanager

        import reflexio.server.llm._litellm_text_generation as tg

        @contextmanager
        def _cap_primary(model):
            if model == "minimax/MiniMax-M3":
                raise ProviderCapSaturatedError("cap saturated")
            yield

        monkeypatch.setattr(tg, "provider_slot", _cap_primary)
        calls = []

        def _fake(**params):
            calls.append(params["model"])
            return _make_completion_response("ok")

        monkeypatch.setattr("litellm.completion", _fake)
        result = client.generate_chat_response(self._messages())
        assert result == "ok"
        # primary never reached litellm (cap saturated); fallback served.
        assert calls == ["zai/glm-5.2"]

    def test_fallback_signal_fires_with_reason(self, monkeypatch, caplog):
        client = LiteLLMClient(
            LiteLLMConfig(model="minimax/MiniMax-M3", fallback_models=["zai/glm-5.2"])
        )

        def _fake(**params):
            if params["model"] == "minimax/MiniMax-M3":
                raise APIConnectionError(message="x", llm_provider="minimax", model="m")
            return _make_completion_response("ok")

        monkeypatch.setattr("litellm.completion", _fake)
        with caplog.at_level(logging.INFO):
            client.generate_chat_response(self._messages())
        assert any("event=llm_fallback_used" in r.message for r in caplog.records)
        assert any("served_model=zai/glm-5.2" in r.message for r in caplog.records)

    def test_no_fallback_signal_when_primary_serves(self, monkeypatch, caplog):
        client = LiteLLMClient(
            LiteLLMConfig(model="minimax/MiniMax-M3", fallback_models=["zai/glm-5.2"])
        )
        monkeypatch.setattr(
            "litellm.completion", lambda **_p: _make_completion_response("ok")
        )
        with caplog.at_level(logging.INFO):
            client.generate_chat_response(self._messages())
        assert not any("event=llm_fallback_used" in r.message for r in caplog.records)

    def test_custom_endpoint_short_circuits_ladder_to_single_rung(
        self, monkeypatch, caplog
    ):
        """A custom endpoint is a single-model pin — fallback_models must not
        re-pin every rung to the SAME ce.model (wasted rung timeouts) nor let
        the success branch log a false ``served_model`` for a rung that never
        actually ran."""
        api_key_config = APIKeyConfig(
            custom_endpoint=CustomEndpointConfig(
                model="ce-model",
                api_key="ce-key",
                api_base="https://example.com/v1",  # type: ignore[arg-type]
            )
        )
        client = LiteLLMClient(
            LiteLLMConfig(
                model="minimax/MiniMax-M3",
                fallback_models=["zai/glm-5.2"],
                api_key_config=api_key_config,
            )
        )
        assert client._resolve_ladder(fallback_models=["zai/glm-5.2"]) == ["ce-model"]

        calls = []

        def _fake(**params):
            calls.append(params["model"])
            return _make_completion_response("ok")

        monkeypatch.setattr("litellm.completion", _fake)
        with caplog.at_level(logging.INFO):
            client.generate_chat_response(self._messages())
        assert calls == ["ce-model"]
        assert not any("event=llm_fallback_used" in r.message for r in caplog.records)


# ===================================================================
# Fallback observability: error-reporting tags + structured log line
# ===================================================================


class TestFallbackObservability:
    """Verify error tags fire when a fallback rung served the request, and stay
    silent when the primary served. Detection is now authoritative and
    loop-driven: the owned walk knows exactly which rung served, so it calls
    ``_emit_fallback_signal`` with the primary, the served rung, and a reason —
    no more response-model diffing.
    """

    @staticmethod
    def _install_recording_reporter(monkeypatch) -> dict[str, str]:
        tags: dict[str, str] = {}

        class RecordingReporter:
            def set_error_tags(self, values) -> None:
                tags.update(values)

        monkeypatch.setattr(error_reporting, "_error_reporter", RecordingReporter())
        return tags

    def test_error_tag_set_when_fallback_serves(self, monkeypatch):
        tags = self._install_recording_reporter(monkeypatch)
        client = LiteLLMClient(
            LiteLLMConfig(model="minimax/MiniMax-M3", fallback_models=["gpt-5.4-mini"])
        )

        def _fake(**params):
            if params["model"] == "minimax/MiniMax-M3":
                raise APIConnectionError(message="x", llm_provider="minimax", model="m")
            return _make_completion_response("ok")

        monkeypatch.setattr("litellm.completion", _fake)
        client.generate_chat_response([{"role": "user", "content": "hi"}])

        assert tags.get("llm.fallback_used") == "true"
        assert tags.get("llm.primary_model") == "minimax/MiniMax-M3"
        assert tags.get("llm.fallback_model") == "gpt-5.4-mini"

    def test_error_tag_not_set_when_primary_served(self, monkeypatch):
        tags = self._install_recording_reporter(monkeypatch)
        client = LiteLLMClient(
            LiteLLMConfig(model="minimax/MiniMax-M3", fallback_models=["gpt-5.4-mini"])
        )

        # The primary serves on the first rung — no fallback occurred.
        monkeypatch.setattr(
            "litellm.completion", lambda **_p: _make_completion_response("ok")
        )
        client.generate_chat_response([{"role": "user", "content": "hi"}])

        assert "llm.fallback_used" not in tags

    def test_error_reason_tag_reflects_failure_class(self, monkeypatch):
        """The new ``llm.fallback_reason`` tag distinguishes an outage from a
        broken-but-reachable primary. A transport error on the primary tags the
        fallback with ``transport_error``."""
        tags = self._install_recording_reporter(monkeypatch)
        client = LiteLLMClient(
            LiteLLMConfig(model="minimax/MiniMax-M3", fallback_models=["gpt-5.4-mini"])
        )

        def _fake(**params):
            if params["model"] == "minimax/MiniMax-M3":
                raise APIConnectionError(message="x", llm_provider="minimax", model="m")
            return _make_completion_response("ok")

        monkeypatch.setattr("litellm.completion", _fake)
        client.generate_chat_response([{"role": "user", "content": "hi"}])

        assert tags.get("llm.fallback_reason") == "transport_error"

    def test_cli_route_resolution_is_not_reported_as_fallback(self, monkeypatch):
        tags = self._install_recording_reporter(monkeypatch)
        client = LiteLLMClient(LiteLLMConfig(model="claude-code/default"))
        response = _make_completion_response("ok")
        response.model = "claude-sonnet-5"
        response._hidden_params = {
            "reflexio_provider": "claude-code",
            "reflexio_cli_binary": "claude",
            "reflexio_served_model": "claude-sonnet-5",
        }
        monkeypatch.setattr("litellm.completion", lambda **_p: response)

        client.generate_chat_response([{"role": "user", "content": "hi"}])

        assert "llm.fallback_used" not in tags

    def test_real_network_fallback_from_cli_primary_is_still_reported(
        self, monkeypatch
    ):
        """Fallback tags fire only when the ladder advances past the primary.

        Served-model metadata on a successful CLI primary response is not a
        fallback signal (see test_cli_route_resolution_is_not_reported_as_fallback).
        A transport failure on the CLI primary that reaches a later rung is.
        """
        tags = self._install_recording_reporter(monkeypatch)
        client = LiteLLMClient(
            LiteLLMConfig(
                model="claude-code/default",
                fallback_models=["gpt-5.4-mini"],
            )
        )
        response = _make_completion_response("ok")
        response.model = "gpt-5.4-mini"
        response._hidden_params = {"custom_llm_provider": "openai"}

        def _fake(**params):
            if params["model"] == "claude-code/default":
                raise APIConnectionError(
                    message="cli unreachable",
                    llm_provider="claude-code",
                    model="claude-code/default",
                )
            return response

        monkeypatch.setattr("litellm.completion", _fake)

        client.generate_chat_response([{"role": "user", "content": "hi"}])

        assert tags.get("llm.fallback_used") == "true"
        assert tags.get("llm.primary_model") == "claude-code/default"
        assert tags.get("llm.fallback_model") == "gpt-5.4-mini"
        assert tags.get("llm.fallback_reason") == "transport_error"


class TestEmbeddingRetries:
    """Embedding calls get num_retries parity with chat. Cross-model
    fallback is intentionally NOT added — embedding vector spaces are
    model-specific; switching mid-call would silently corrupt the index."""

    @staticmethod
    def _force_litellm_route(monkeypatch):
        """Neutralize the embedding-service router so the call reaches
        litellm.embedding. The CI/local env may have
        CLAUDE_SMART_USE_LOCAL_EMBEDDING=1 or REFLEXIO_EMBEDDING_SERVICE_URL
        set, both of which divert to the HTTP service path.
        """
        monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
        monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_URL", raising=False)
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", raising=False)

    def test_get_embedding_passes_num_retries(self, monkeypatch):
        from types import SimpleNamespace

        self._force_litellm_route(monkeypatch)
        client = LiteLLMClient(LiteLLMConfig(model="x", max_retries=3))
        captured: dict[str, Any] = {}

        def _fake(**params):
            captured.update(params)
            return SimpleNamespace(data=[{"embedding": [0.1] * 1536, "index": 0}])

        monkeypatch.setattr("litellm.embedding", _fake)
        monkeypatch.setattr(
            client,
            "_resolve_default_embedding_model",
            lambda: "text-embedding-3-small",
        )
        client.get_embedding("hello")
        assert captured.get("num_retries") == 3

    def test_get_embeddings_batch_passes_num_retries(self, monkeypatch):
        from types import SimpleNamespace

        self._force_litellm_route(monkeypatch)
        client = LiteLLMClient(LiteLLMConfig(model="x", max_retries=3))
        captured: dict[str, Any] = {}

        def _fake(**params):
            captured.update(params)
            return SimpleNamespace(
                data=[
                    {"embedding": [0.1] * 1536, "index": 0},
                    {"embedding": [0.2] * 1536, "index": 1},
                ]
            )

        monkeypatch.setattr("litellm.embedding", _fake)
        monkeypatch.setattr(
            client,
            "_resolve_default_embedding_model",
            lambda: "text-embedding-3-small",
        )
        client.get_embeddings(["a", "b"])
        assert captured.get("num_retries") == 3

    def test_embedding_never_receives_fallbacks_kwarg(self, monkeypatch):
        """Even with fallback_models set on the config, embedding calls
        MUST NOT receive a fallbacks kwarg — vector spaces are model-
        specific and silent cross-model fallback would corrupt the index."""
        from types import SimpleNamespace

        self._force_litellm_route(monkeypatch)
        client = LiteLLMClient(
            LiteLLMConfig(
                model="x",
                max_retries=3,
                fallback_models=["gpt-5.4-mini"],
            )
        )
        captured: dict[str, Any] = {}

        def _fake(**params):
            captured.update(params)
            return SimpleNamespace(data=[{"embedding": [0.1] * 1536, "index": 0}])

        monkeypatch.setattr("litellm.embedding", _fake)
        monkeypatch.setattr(
            client,
            "_resolve_default_embedding_model",
            lambda: "text-embedding-3-small",
        )
        client.get_embedding("hi")
        assert "fallbacks" not in captured


def test_generate_chat_response_does_not_mutate_caller_messages() -> None:
    """Merging a ``system_message`` must not mutate the caller's message dicts.

    ``final_messages = list(messages)`` is a shallow copy that shares the caller's
    dict objects, so merging into ``final_messages[0]`` in place used to corrupt
    the caller's list and re-prepend the system message on reuse/retry.
    """
    client = _build_client()
    original = [
        {"role": "system", "content": "orig-system"},
        {"role": "user", "content": "hi"},
    ]

    completion = CompletionResult("ok", ModelProvenance())
    with patch.object(client, "_make_request", return_value=completion) as mock_req:
        client.generate_chat_response(original, system_message="injected")

    # The caller's first dict is untouched...
    assert original[0]["content"] == "orig-system"
    # ...while _make_request received a NEW merged system dict.
    sent = mock_req.call_args[0][0]
    assert sent[0] is not original[0]
    assert sent[0]["content"] == "injected\n\norig-system"

    # A second call must not double-prepend onto the caller's data.
    with patch.object(client, "_make_request", return_value=completion):
        client.generate_chat_response(original, system_message="injected")
    assert original[0]["content"] == "orig-system"


class TestTruncatedListSalvage:
    """Recover complete items from a cut-off list without inventing content."""

    class _Item(BaseModel):
        name: str
        score: int

    class _Wrapper(BaseModel):
        items: list["TestTruncatedListSalvage._Item"]

    def test_recovers_complete_items_before_the_cut(self, caplog):
        from reflexio.server.llm._litellm_structured_output import (
            _salvage_complete_single_list_items,
        )

        # Third item is cut mid-object; the first two are complete.
        truncated = (
            '{"items": [{"name": "a", "score": 1}, {"name": "b", "score": 2}, '
            '{"name": "c", "sco'
        )

        with caplog.at_level(
            logging.WARNING,
            logger="reflexio.server.llm._litellm_structured_output",
        ):
            salvaged = _salvage_complete_single_list_items(self._Wrapper, truncated)

        assert salvaged is not None
        typed_salvaged = cast(TestTruncatedListSalvage._Wrapper, salvaged)
        assert [(i.name, i.score) for i in typed_salvaged.items] == [
            ("a", 1),
            ("b", 2),
        ]
        assert (
            "event=structured_output_salvaged schema=_Wrapper salvaged=2" in caplog.text
        )

    def test_item_failing_its_own_schema_is_dropped_not_coerced(self):
        from reflexio.server.llm._litellm_structured_output import (
            _salvage_complete_single_list_items,
        )

        truncated = (
            '{"items": [{"name": "a", "score": "not-an-int"}, '
            '{"name": "b", "score": 2}, {"name": "c", "sco'
        )

        salvaged = _salvage_complete_single_list_items(self._Wrapper, truncated)

        assert salvaged is not None
        typed_salvaged = cast(TestTruncatedListSalvage._Wrapper, salvaged)
        assert [(i.name, i.score) for i in typed_salvaged.items] == [("b", 2)]

    def test_truncation_with_no_complete_item_returns_none(self):
        """Never report a cut-off response as an empty success."""
        from reflexio.server.llm._litellm_structured_output import (
            _salvage_complete_single_list_items,
        )

        assert (
            _salvage_complete_single_list_items(self._Wrapper, '{"items": [{"na')
            is None
        )

    def test_empty_closed_list_returns_none(self):
        """An empty list plus detected truncation is not a "found nothing" answer."""
        from reflexio.server.llm._litellm_structured_output import (
            _salvage_complete_single_list_items,
        )

        assert (
            _salvage_complete_single_list_items(self._Wrapper, '{"items": [], "extra')
            is None
        )

    def test_schema_without_a_single_list_field_is_not_salvaged(self):
        from reflexio.server.llm._litellm_structured_output import (
            _salvage_complete_single_list_items,
        )

        class Pair(BaseModel):
            left: str
            right: str

        assert _salvage_complete_single_list_items(Pair, '{"left": "a", "rig') is None


class TestSafeValidationErrors:
    """Diagnostics must never carry model-authored content."""

    def test_json_syntax_error_reports_position_only(self):
        from reflexio.server.llm._litellm_structured_output import (
            _safe_validation_errors,
        )

        try:
            json.loads('{"secret": "customer data", }')
        except json.JSONDecodeError as exc:
            errors = _safe_validation_errors(exc)
        else:  # pragma: no cover - the payload above is invalid by construction
            pytest.fail("expected a JSONDecodeError")

        assert len(errors) == 1
        assert "json_invalid" in errors[0]
        assert "customer data" not in errors[0]

    def test_validation_error_reports_paths_and_codes_only(self):
        from pydantic import ValidationError

        from reflexio.server.llm._litellm_structured_output import (
            _safe_validation_errors,
        )

        class Model(BaseModel):
            count: int

        try:
            Model.model_validate({"count": "customer data"})
        except ValidationError as exc:
            errors = _safe_validation_errors(exc)
        else:  # pragma: no cover - the payload above is invalid by construction
            pytest.fail("expected a ValidationError")

        assert errors == ("count: int_parsing",)
        assert all("customer data" not in error for error in errors)
