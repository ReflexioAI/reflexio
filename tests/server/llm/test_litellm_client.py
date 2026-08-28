"""Offline tests for the LiteLLM client.

Tests build a client from a literal config and check model selection,
API-key routing, Azure endpoint fields, and image encoding. A loopback
integration check also exercises the installed LiteLLM HTTP transport without
external credentials or paid API calls.

The tests that actually call OpenAI or Anthropic live in
``tests/e2e_tests/test_litellm_client_real_llm.py``: that path is the only one
``llm_mock._is_e2e_test_run`` exempts from the session-wide
``litellm.completion`` patch, so a live test placed here would silently assert
against the mock.
"""

import json
import os
import struct
import tempfile
import threading
import zlib
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from pydantic import BaseModel, HttpUrl

from reflexio.models.config_schema import (
    AnthropicConfig,
    APIKeyConfig,
    AzureOpenAIConfig,
    CustomEndpointConfig,
    OpenAIConfig,
    OpenRouterConfig,
)
from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMClientError,
    LiteLLMConfig,
    ToolCallingChatResponse,
    _sanitize_json_string,
    create_litellm_client,
)
from reflexio.test_support.llm_mock import unpatched_litellm


@pytest.mark.integration
def test_installed_litellm_transport_round_trip(monkeypatch):
    """Exercise Reflexio -> real LiteLLM SDK -> local HTTP -> parsed responses."""
    import litellm

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            message = {"role": "assistant", "content": "offline-response"}
            finish = "stop"
            if body.get("tools"):
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_test",
                            "type": "function",
                            "function": {
                                "name": "finish",
                                "arguments": '{"answer":42}',
                            },
                        }
                    ],
                }
                finish = "tool_calls"
            elif body.get("response_format"):
                message["content"] = '{"answer":42,"explanation":"loopback"}'
            payload = json.dumps(
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": message, "finish_reason": finish}
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
    monkeypatch.setenv("REFLEXIO_BLOCK_PRIVATE_URLS", "false")
    monkeypatch.setattr(litellm, "callbacks", [])
    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = LiteLLMClient(
                LiteLLMConfig(
                    model="openai/gpt-4o-mini",
                    timeout=10,
                    max_retries=0,
                    fallback_models=[],
                    api_key_config=APIKeyConfig(
                        custom_endpoint=CustomEndpointConfig(
                            model="openai/gpt-4o-mini",
                            api_key="loopback-test-key",
                            api_base=HttpUrl(
                                f"http://127.0.0.1:{server.server_port}/v1"
                            ),
                        )
                    ),
                )
            )
            with unpatched_litellm():
                assert client.generate_response("hello") == "offline-response"
                structured = client.generate_response(
                    "answer", response_format=MathResult
                )
                assert isinstance(structured, MathResult)
                assert structured.answer == 42
                tool = {
                    "type": "function",
                    "function": {
                        "name": "finish",
                        "parameters": MathResult.model_json_schema(),
                    },
                }
                result = client.generate_chat_response(
                    [{"role": "user", "content": "finish"}],
                    tools=[tool],
                    tool_choice="required",
                )
                assert isinstance(result, ToolCallingChatResponse)
                assert result.tool_calls is not None
                assert result.tool_calls[0].function.name == "finish"
                assert json.loads(result.tool_calls[0].function.arguments) == {
                    "answer": 42
                }
            assert len(requests) == 3
            assert all(path == "/v1/chat/completions" for path, _ in requests)
            assert requests[1][1]["response_format"]["type"] == "json_schema"
        finally:
            server.shutdown()
            thread.join(timeout=5)


def create_minimal_png(
    width: int = 10, height: int = 10, color: tuple = (255, 0, 0)
) -> bytes:
    """
    Create a minimal valid PNG image in memory.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        color: RGB tuple for the fill color.

    Returns:
        PNG image as bytes.
    """

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk_len = struct.pack(">I", len(data))
        chunk_crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return chunk_len + chunk_type + data + chunk_crc

    # PNG signature
    signature = b"\x89PNG\r\n\x1a\n"

    # IHDR chunk (image header)
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = png_chunk(b"IHDR", ihdr_data)

    # IDAT chunk (image data)
    raw_data = b""
    for _ in range(height):
        raw_data += b"\x00"  # Filter byte (none)
        for _ in range(width):
            raw_data += bytes(color)

    compressed_data = zlib.compress(raw_data)
    idat = png_chunk(b"IDAT", compressed_data)

    # IEND chunk (image end)
    iend = png_chunk(b"IEND", b"")

    return signature + ihdr + idat + iend


# Pydantic models for structured output tests
class MathResult(BaseModel):
    """Simple math result model."""

    answer: int
    explanation: str


class ColorAnalysis(BaseModel):
    """Color analysis result model."""

    primary_color: str
    is_solid: bool


@pytest.fixture
def test_image_bytes() -> bytes:
    """Create a test PNG image as bytes (solid red)."""
    return create_minimal_png(width=50, height=50, color=(255, 0, 0))


@pytest.fixture
def test_image_file(test_image_bytes: bytes) -> Generator[str, None, None]:
    """Create a temporary PNG image file."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(test_image_bytes)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestLiteLLMClientConfiguration:
    """Test client configuration and setup."""

    def test_create_client_with_config(self):
        """Test creating a client with explicit config."""
        config = LiteLLMConfig(
            model="gpt-5.4-mini",
            temperature=0.5,
            max_tokens=100,
            timeout=30,
            max_retries=2,
        )
        client = LiteLLMClient(config)

        assert client.get_model() == "gpt-5.4-mini"
        assert client.get_config().temperature == 0.5
        assert client.get_config().max_tokens == 100

    def test_create_client_with_factory_function(self):
        """Test creating a client with the factory function."""
        client = create_litellm_client(
            model="claude-sonnet-4-5-20250929",
            temperature=0.3,
            max_tokens=200,
        )

        assert client.get_model() == "claude-sonnet-4-5-20250929"
        assert client.get_config().temperature == 0.3

    def test_update_config(self):
        """Test updating client configuration."""
        client = create_litellm_client(model="gpt-5.4-mini", temperature=0.5)

        client.update_config(temperature=0.8, max_tokens=500)

        assert client.get_config().temperature == 0.8
        assert client.get_config().max_tokens == 500

    def test_update_config_ignores_unknown_params(self):
        """Test that unknown config parameters are ignored."""
        client = create_litellm_client(model="gpt-5.4-mini")

        # Should not raise, just log warning
        client.update_config(unknown_param="value")

        # Original config should be unchanged
        assert client.get_model() == "gpt-5.4-mini"


class TestLiteLLMClientImageEncoding:
    """Test image encoding utilities."""

    def test_encode_image_to_base64_from_file(self, test_image_file: str):
        """Test encoding an image file to base64."""
        client = create_litellm_client(model="gpt-5.4-mini")
        base64_data, media_type = client.encode_image_to_base64(test_image_file)

        assert isinstance(base64_data, str)
        assert len(base64_data) > 0
        assert media_type == "image/png"

    def test_encode_image_to_base64_nonexistent_file(self):
        """Test that encoding a nonexistent file raises error."""
        client = create_litellm_client(model="gpt-5.4-mini")

        with pytest.raises(LiteLLMClientError, match="Image file not found"):
            client.encode_image_to_base64("/nonexistent/path/image.png")

    def test_encode_image_to_base64_unsupported_format(self, tmp_path: Path):
        """Test that unsupported format raises error."""
        client = create_litellm_client(model="gpt-5.4-mini")
        unsupported_file = tmp_path / "test.bmp"
        unsupported_file.write_bytes(b"fake image data")

        with pytest.raises(LiteLLMClientError, match="Unsupported image format"):
            client.encode_image_to_base64(str(unsupported_file))

    def test_supported_image_formats(self):
        """Test that supported image formats are correctly defined."""
        client = create_litellm_client(model="gpt-5.4-mini")

        assert ".jpg" in client.SUPPORTED_IMAGE_FORMATS
        assert ".jpeg" in client.SUPPORTED_IMAGE_FORMATS
        assert ".png" in client.SUPPORTED_IMAGE_FORMATS
        assert ".gif" in client.SUPPORTED_IMAGE_FORMATS
        assert ".webp" in client.SUPPORTED_IMAGE_FORMATS


class TestLiteLLMClientModelSwitching:
    """Test switching between different models."""

    def test_same_interface_different_models(self):
        """Test that the interface is consistent across models."""
        openai_client = create_litellm_client(model="gpt-5.4-mini")
        claude_client = create_litellm_client(model="claude-sonnet-4-5-20250929")

        # Both should have the same methods
        assert hasattr(openai_client, "generate_response")
        assert hasattr(openai_client, "generate_chat_response")
        assert hasattr(openai_client, "get_embedding")
        assert hasattr(openai_client, "update_config")
        assert hasattr(openai_client, "get_model")
        assert hasattr(openai_client, "get_config")

        assert hasattr(claude_client, "generate_response")
        assert hasattr(claude_client, "generate_chat_response")
        assert hasattr(claude_client, "get_embedding")
        assert hasattr(claude_client, "update_config")
        assert hasattr(claude_client, "get_model")
        assert hasattr(claude_client, "get_config")


class TestLiteLLMClientEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_images_list(self):
        """Test that empty images list works like no images."""
        # Should not raise - empty list is falsy
        # This is a configuration test, not an API call test
        client = create_litellm_client(model="gpt-5.4-mini")
        assert client.get_model() == "gpt-5.4-mini"

    def test_dict_based_response_format_raises_error(self):
        """Test that dict-based response_format raises error."""
        client = create_litellm_client(model="gpt-5.4-mini")
        # Dict-based formats are no longer supported - must use Pydantic models
        with pytest.raises(
            LiteLLMClientError,
            match="response_format must be a Pydantic BaseModel class",
        ):
            client.generate_response(
                "test prompt", response_format={"type": "json_object"}
            )

    def test_dict_based_response_format_raises_error_chat(self):
        """Test that dict-based response_format raises error for chat responses."""
        client = create_litellm_client(model="gpt-5.4-mini")
        messages = [{"role": "user", "content": "test message"}]
        with pytest.raises(
            LiteLLMClientError,
            match="response_format must be a Pydantic BaseModel class",
        ):
            client.generate_chat_response(
                messages, response_format={"type": "json_object"}
            )

    def test_config_defaults(self):
        """Test that config has sensible defaults."""
        config = LiteLLMConfig(model="gpt-5.4-mini")

        assert config.temperature == 0.7
        assert config.max_tokens is None
        assert config.timeout == 120
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.top_p == 1.0


class TestLiteLLMClientAPIKeyOverride:
    """Test API key configuration override functionality."""

    def test_create_client_with_openai_api_key_config(self):
        """Test creating a client with OpenAI API key config override."""
        api_key_config = APIKeyConfig(
            openai=OpenAIConfig(api_key="test-openai-key-12345")
        )
        config = LiteLLMConfig(
            model="gpt-5.4-mini",
            temperature=0.5,
            api_key_config=api_key_config,
        )
        client = LiteLLMClient(config)

        assert client.get_model() == "gpt-5.4-mini"
        assert client.get_config().api_key_config == api_key_config
        # Verify the API key was resolved correctly
        assert client._api_key == "test-openai-key-12345"
        assert client._api_base is None
        assert client._api_version is None

    def test_create_client_with_anthropic_api_key_config(self):
        """Test creating a client with Anthropic API key config override."""
        api_key_config = APIKeyConfig(
            anthropic=AnthropicConfig(api_key="test-anthropic-key-67890")
        )
        config = LiteLLMConfig(
            model="claude-sonnet-4-5-20250929",
            temperature=0.5,
            api_key_config=api_key_config,
        )
        client = LiteLLMClient(config)

        assert client.get_model() == "claude-sonnet-4-5-20250929"
        assert client.get_config().api_key_config == api_key_config
        # Verify the API key was resolved correctly for Claude model
        assert client._api_key == "test-anthropic-key-67890"
        assert client._api_base is None
        assert client._api_version is None

    def test_create_client_with_azure_openai_config(self, monkeypatch):
        """Test creating a client with Azure OpenAI configuration."""
        monkeypatch.setattr(
            "reflexio.models.api_schema.validators.socket.getaddrinfo",
            lambda _host, port, *_args, **_kwargs: [
                (2, 1, 0, "", ("93.184.216.34", port or 443))
            ],
        )
        api_key_config = APIKeyConfig(
            openai=OpenAIConfig(
                azure_config=AzureOpenAIConfig(
                    api_key="test-azure-key-11111",
                    endpoint="https://test-resource.openai.azure.com/",  # type: ignore[arg-type]
                    api_version="2024-02-15-preview",
                    deployment_name="gpt-4o-deployment",
                )
            )
        )
        config = LiteLLMConfig(
            model="azure/gpt-4o",
            temperature=0.5,
            api_key_config=api_key_config,
        )
        client = LiteLLMClient(config)

        assert client.get_model() == "azure/gpt-4o"
        # Verify the Azure config was resolved correctly
        assert client._api_key == "test-azure-key-11111"
        assert client._api_base == "https://test-resource.openai.azure.com/"
        assert client._api_version == "2024-02-15-preview"

    def test_create_client_with_factory_and_api_key_config(self):
        """Test creating a client using the factory function with API key config."""
        api_key_config = APIKeyConfig(
            openai=OpenAIConfig(api_key="factory-test-key-99999")
        )
        client = create_litellm_client(
            model="gpt-5.4-mini",
            temperature=0.3,
            max_tokens=200,
            api_key_config=api_key_config,
        )

        assert client.get_model() == "gpt-5.4-mini"
        assert client.get_config().temperature == 0.3
        assert client._api_key == "factory-test-key-99999"

    def test_api_key_resolution_openai_model(self):
        """Test that OpenAI models resolve to OpenAI API key."""
        api_key_config = APIKeyConfig(
            openai=OpenAIConfig(api_key="openai-key"),
            anthropic=AnthropicConfig(api_key="anthropic-key"),
        )
        config = LiteLLMConfig(
            model="gpt-4o-mini",
            api_key_config=api_key_config,
        )
        client = LiteLLMClient(config)

        # OpenAI model should resolve to OpenAI key
        assert client._api_key == "openai-key"

    def test_api_key_resolution_claude_model(self):
        """Test that Claude models resolve to Anthropic API key."""
        api_key_config = APIKeyConfig(
            openai=OpenAIConfig(api_key="openai-key"),
            anthropic=AnthropicConfig(api_key="anthropic-key"),
        )
        config = LiteLLMConfig(
            model="claude-3-5-sonnet-20241022",
            api_key_config=api_key_config,
        )
        client = LiteLLMClient(config)

        # Claude model should resolve to Anthropic key
        assert client._api_key == "anthropic-key"

    def test_api_key_resolution_azure_model(self):
        """Test that Azure models resolve to Azure OpenAI config."""
        api_key_config = APIKeyConfig(
            openai=OpenAIConfig(
                api_key="direct-openai-key",
                azure_config=AzureOpenAIConfig(
                    api_key="azure-key",
                    endpoint="https://example.com/",  # type: ignore[arg-type]
                    api_version="2024-02-15-preview",
                ),
            ),
        )
        config = LiteLLMConfig(
            model="azure/gpt-4",
            api_key_config=api_key_config,
        )
        client = LiteLLMClient(config)

        # Azure model should resolve to Azure config
        assert client._api_key == "azure-key"
        assert client._api_base == "https://example.com/"
        assert client._api_version == "2024-02-15-preview"

    def test_no_api_key_config_returns_none(self):
        """Test that client without API key config has None resolved keys."""
        config = LiteLLMConfig(model="gpt-5.4-mini")
        client = LiteLLMClient(config)

        assert client._api_key is None
        assert client._api_base is None
        assert client._api_version is None

    def test_missing_provider_key_returns_none(self):
        """Test that missing provider key in config returns None."""
        # Config with only Anthropic key, but using OpenAI model
        api_key_config = APIKeyConfig(
            anthropic=AnthropicConfig(api_key="anthropic-only-key")
        )
        config = LiteLLMConfig(
            model="gpt-5.4-mini",
            api_key_config=api_key_config,
        )
        client = LiteLLMClient(config)

        # OpenAI model but no OpenAI key configured
        assert client._api_key is None

    def test_api_key_config_with_both_providers(self):
        """Test that config with both providers resolves correctly based on model."""
        api_key_config = APIKeyConfig(
            openai=OpenAIConfig(api_key="openai-shared-key"),
            anthropic=AnthropicConfig(api_key="anthropic-shared-key"),
        )

        # Create client for OpenAI model
        openai_config = LiteLLMConfig(
            model="gpt-4o-mini",
            api_key_config=api_key_config,
        )
        openai_client = LiteLLMClient(openai_config)
        assert openai_client._api_key == "openai-shared-key"

        # Create client for Claude model
        claude_config = LiteLLMConfig(
            model="claude-3-5-haiku-20241022",
            api_key_config=api_key_config,
        )
        claude_client = LiteLLMClient(claude_config)
        assert claude_client._api_key == "anthropic-shared-key"

    def test_create_client_with_openrouter_api_key_config(self):
        """Test creating a client with OpenRouter API key config override."""
        api_key_config = APIKeyConfig(
            openrouter=OpenRouterConfig(api_key="test-openrouter-key-12345")
        )
        config = LiteLLMConfig(
            model="openrouter/openai/gpt-4o",
            temperature=0.5,
            api_key_config=api_key_config,
        )
        client = LiteLLMClient(config)

        assert client.get_model() == "openrouter/openai/gpt-4o"
        assert client.get_config().api_key_config == api_key_config
        # Verify the API key was resolved correctly
        assert client._api_key == "test-openrouter-key-12345"
        assert client._api_base is None
        assert client._api_version is None

    def test_api_key_resolution_openrouter_model(self):
        """Test that OpenRouter models resolve to OpenRouter API key."""
        api_key_config = APIKeyConfig(
            openai=OpenAIConfig(api_key="openai-key"),
            anthropic=AnthropicConfig(api_key="anthropic-key"),
            openrouter=OpenRouterConfig(api_key="openrouter-key"),
        )
        config = LiteLLMConfig(
            model="openrouter/anthropic/claude-3.5-sonnet",
            api_key_config=api_key_config,
        )
        client = LiteLLMClient(config)

        # OpenRouter model should resolve to OpenRouter key, not Anthropic
        assert client._api_key == "openrouter-key"

    def test_openrouter_missing_key_returns_none(self):
        """Test that OpenRouter model without OpenRouter config returns None."""
        api_key_config = APIKeyConfig(
            openai=OpenAIConfig(api_key="openai-key"),
        )
        config = LiteLLMConfig(
            model="openrouter/openai/gpt-4o",
            api_key_config=api_key_config,
        )
        client = LiteLLMClient(config)

        # OpenRouter model but no OpenRouter key configured
        assert client._api_key is None


class TestSanitizeJsonString:
    """Unit tests for the module-level _sanitize_json_string helper."""

    def test_single_quotes_to_double(self):
        """Single-quoted JSON keys and values are converted to double quotes."""
        result = _sanitize_json_string("{'key': 'value'}")
        assert result == '{"key": "value"}'

    def test_python_booleans(self):
        """Python True/False/None are converted to JSON true/false/null."""
        result = _sanitize_json_string('{"a": True, "b": False, "c": None}')
        assert result == '{"a": true, "b": false, "c": null}'

    def test_python_booleans_inside_strings_preserved(self):
        """True/False/None inside quoted strings are NOT converted."""
        result = _sanitize_json_string('{"msg": "True story about None"}')
        assert result == '{"msg": "True story about None"}'

    def test_trailing_commas(self):
        """Trailing commas before } or ] are removed."""
        result = _sanitize_json_string('{"a": 1, "b": 2,}')
        assert result == '{"a": 1, "b": 2}'

    def test_trailing_comma_in_array(self):
        """Trailing commas in arrays are removed."""
        result = _sanitize_json_string("[1, 2, 3,]")
        assert result == "[1, 2, 3]"

    def test_escaped_apostrophe_in_single_quoted_string(self):
        """Escaped apostrophes inside single-quoted strings are handled."""
        import json

        result = _sanitize_json_string("{'text': 'didn\\'t work'}")
        parsed = json.loads(result)
        assert parsed["text"] == "didn't work"

    def test_double_quotes_inside_single_quoted_string(self):
        """Double quotes inside single-quoted strings are escaped."""
        import json

        result = _sanitize_json_string("{'text': 'he said \"hello\"'}")
        parsed = json.loads(result)
        assert parsed["text"] == 'he said "hello"'

    def test_mixed_all_issues(self):
        """Combined: single quotes, Python booleans, trailing comma."""
        import json

        result = _sanitize_json_string(
            "{'is_success': True, 'failure_type': None, 'reason': 'ok',}"
        )
        parsed = json.loads(result)
        assert parsed == {"is_success": True, "failure_type": None, "reason": "ok"}

    def test_valid_json_passthrough(self):
        """Already-valid JSON passes through unchanged."""
        original = '{"is_success": true, "count": 42}'
        result = _sanitize_json_string(original)
        assert result == original

    def test_word_boundary_prevents_partial_replacement(self):
        """Words containing True/False as substrings are not replaced."""
        result = _sanitize_json_string('{"TrueValue": 1, "isFalsey": 2}')
        assert '"TrueValue"' in result
        assert '"isFalsey"' in result
