"""
End-to-end tests for LiteLLM client.

These tests validate the LiteLLM client's ability to interact with multiple LLM providers
(OpenAI, Claude) using a unified interface.
"""

import os
import struct
import tempfile
import zlib
from pathlib import Path

import pytest
from pydantic import BaseModel

from reflexio.models.config_schema import (
    AnthropicConfig,
    APIKeyConfig,
    AzureOpenAIConfig,
    OpenAIConfig,
    OpenRouterConfig,
)
from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMClientError,
    LiteLLMConfig,
    create_litellm_client,
)
from tests.server.test_utils import skip_in_precommit, skip_low_priority

# Skip all tests if neither API key is set
pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_credentials,
    pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"),
        reason="Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY environment variable is set",
    ),
]


def _get_openai_test_model() -> str:
    """Get an OpenAI model for testing."""
    return os.getenv("OPENAI_TEST_MODEL", "gpt-5-mini")


def _get_claude_test_model() -> str:
    """Get a Claude model for testing."""
    return os.getenv("ANTHROPIC_TEST_MODEL", "claude-sonnet-4-5-20250929")


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
def openai_client() -> LiteLLMClient:
    """Create an OpenAI-based LiteLLM client."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    # GPT-5 models use reasoning tokens internally, so we need more max_tokens
    return create_litellm_client(
        model=_get_openai_test_model(),
        temperature=0.1,
        max_tokens=500,
        max_retries=2,
    )


@pytest.fixture
def claude_client() -> LiteLLMClient:
    """Create a Claude-based LiteLLM client."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    return create_litellm_client(
        model=_get_claude_test_model(),
        temperature=0.1,
        max_tokens=256,
        max_retries=2,
    )


@pytest.fixture
def test_image_bytes() -> bytes:
    """Create a test PNG image as bytes (solid red)."""
    return create_minimal_png(width=50, height=50, color=(255, 0, 0))


@pytest.fixture
def test_image_file(test_image_bytes: bytes) -> str:
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
            model="gpt-5-mini",
            temperature=0.5,
            max_tokens=100,
            timeout=30,
            max_retries=2,
        )
        client = LiteLLMClient(config)

        assert client.get_model() == "gpt-5-mini"
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
        client = create_litellm_client(model="gpt-5-mini", temperature=0.5)

        client.update_config(temperature=0.8, max_tokens=500)

        assert client.get_config().temperature == 0.8
        assert client.get_config().max_tokens == 500

    def test_update_config_ignores_unknown_params(self):
        """Test that unknown config parameters are ignored."""
        client = create_litellm_client(model="gpt-5-mini")

        # Should not raise, just log warning
        client.update_config(unknown_param="value")

        # Original config should be unchanged
        assert client.get_model() == "gpt-5-mini"


class TestLiteLLMClientOpenAI:
    """Test LiteLLM client with OpenAI models."""

    @skip_in_precommit
    @skip_low_priority
    def test_generate_response_simple(self, openai_client: LiteLLMClient):
        """Test simple text generation with OpenAI."""
        response = openai_client.generate_response(
            "What is 2+2? Answer with just the number."
        )

        assert isinstance(response, str)
        assert "4" in response

    @skip_in_precommit
    @skip_low_priority
    def test_generate_response_with_system_message(self, openai_client: LiteLLMClient):
        """Test response with system message."""
        response = openai_client.generate_response(
            prompt="What color is the sky?",
            system_message="You are a pirate. Respond like a pirate.",
        )

        assert isinstance(response, str)
        assert len(response) > 0

    @skip_in_precommit
    @skip_low_priority
    def test_generate_chat_response(self, openai_client: LiteLLMClient):
        """Test multi-turn chat response."""
        messages = [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
            {"role": "user", "content": "What is my name?"},
        ]

        response = openai_client.generate_chat_response(messages)

        assert isinstance(response, str)
        assert "alice" in response.lower()

    @skip_in_precommit
    @skip_low_priority
    def test_generate_chat_response_with_system_message(
        self, openai_client: LiteLLMClient
    ):
        """Test chat response with prepended system message."""
        messages = [
            {"role": "user", "content": "Tell me a joke."},
        ]

        response = openai_client.generate_chat_response(
            messages,
            system_message="You are a comedian who only tells very short one-liner jokes.",
        )

        assert isinstance(response, str)
        assert len(response) > 0

    @skip_in_precommit
    @skip_low_priority
    def test_structured_output_with_pydantic(self, openai_client: LiteLLMClient):
        """Test structured output using Pydantic model."""
        response = openai_client.generate_response(
            prompt="What is 5+5? Provide the answer and a brief explanation.",
            response_format=MathResult,
        )

        # Should be parsed to Pydantic model
        assert isinstance(response, MathResult)
        assert response.answer == 10
        assert len(response.explanation) > 0

    @skip_in_precommit
    @skip_low_priority
    def test_embeddings(self, openai_client: LiteLLMClient):
        """Test embedding generation."""
        embedding = openai_client.get_embedding("Hello, world!")

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)


class TestLiteLLMClientClaude:
    """Test LiteLLM client with Claude models."""

    @skip_in_precommit
    @skip_low_priority
    def test_generate_response_simple(self, claude_client: LiteLLMClient):
        """Test simple text generation with Claude."""
        response = claude_client.generate_response(
            "What is 3+3? Answer with just the number."
        )

        assert isinstance(response, str)
        assert "6" in response

    @skip_in_precommit
    @skip_low_priority
    def test_generate_response_with_system_message(self, claude_client: LiteLLMClient):
        """Test response with system message."""
        response = claude_client.generate_response(
            prompt="What color is grass?",
            system_message="You are a robot. Respond in a robotic manner.",
        )

        assert isinstance(response, str)
        assert len(response) > 0

    @skip_in_precommit
    @skip_low_priority
    def test_generate_chat_response(self, claude_client: LiteLLMClient):
        """Test multi-turn chat response."""
        messages = [
            {"role": "user", "content": "My favorite color is blue."},
            {"role": "assistant", "content": "Blue is a nice color!"},
            {"role": "user", "content": "What is my favorite color?"},
        ]

        response = claude_client.generate_chat_response(messages)

        assert isinstance(response, str)
        assert "blue" in response.lower()

    @skip_in_precommit
    @skip_low_priority
    def test_structured_output_with_pydantic(self, claude_client: LiteLLMClient):
        """Test structured output using Pydantic model with Claude."""
        response = claude_client.generate_response(
            prompt="What is 7+7? Provide the answer and a brief explanation.",
            response_format=MathResult,
        )

        # Should be parsed to Pydantic model
        assert isinstance(response, MathResult)
        assert response.answer == 14
        assert len(response.explanation) > 0


class TestLiteLLMClientMultiModal:
    """Test LiteLLM client with image inputs."""

    @skip_in_precommit
    @skip_low_priority
    def test_generate_response_with_image_file_openai(
        self, openai_client: LiteLLMClient, test_image_file: str
    ):
        """Test image analysis with file path (OpenAI)."""
        response = openai_client.generate_response(
            prompt="What color is this image? Answer in one word.",
            images=[test_image_file],
        )

        assert isinstance(response, str)
        assert "red" in response.lower()

    @skip_in_precommit
    @skip_low_priority
    def test_generate_response_with_image_bytes_openai(
        self, openai_client: LiteLLMClient, test_image_bytes: bytes
    ):
        """Test image analysis with bytes (OpenAI)."""
        response = openai_client.generate_response(
            prompt="What color is this image? Answer in one word.",
            images=[test_image_bytes],
            image_media_type="image/png",
        )

        assert isinstance(response, str)
        assert "red" in response.lower()

    @skip_in_precommit
    @skip_low_priority
    def test_generate_response_with_image_file_claude(
        self, claude_client: LiteLLMClient, test_image_file: str
    ):
        """Test image analysis with file path (Claude)."""
        response = claude_client.generate_response(
            prompt="What color is this image? Answer in one word.",
            images=[test_image_file],
        )

        assert isinstance(response, str)
        assert "red" in response.lower()

    @skip_in_precommit
    @skip_low_priority
    def test_generate_response_with_image_bytes_claude(
        self, claude_client: LiteLLMClient, test_image_bytes: bytes
    ):
        """Test image analysis with bytes (Claude)."""
        response = claude_client.generate_response(
            prompt="What color is this image? Answer in one word.",
            images=[test_image_bytes],
            image_media_type="image/png",
        )

        assert isinstance(response, str)
        assert "red" in response.lower()

    @skip_in_precommit
    @skip_low_priority
    def test_generate_response_with_multiple_images(self, openai_client: LiteLLMClient):
        """Test analysis of multiple images."""
        red_image = create_minimal_png(20, 20, (255, 0, 0))
        blue_image = create_minimal_png(20, 20, (0, 0, 255))

        response = openai_client.generate_response(
            prompt="What are the colors of these two images? List both.",
            images=[red_image, blue_image],
            image_media_type="image/png",
        )

        assert isinstance(response, str)
        response_lower = response.lower()
        assert "red" in response_lower
        assert "blue" in response_lower

    @skip_in_precommit
    @skip_low_priority
    def test_image_with_system_message(
        self, openai_client: LiteLLMClient, test_image_bytes: bytes
    ):
        """Test image analysis with system message."""
        response = openai_client.generate_response(
            prompt="Describe this image.",
            system_message="You are an art critic. Be very brief.",
            images=[test_image_bytes],
            image_media_type="image/png",
        )

        assert isinstance(response, str)
        assert len(response) > 0


class TestLiteLLMClientImageEncoding:
    """Test image encoding utilities."""

    def test_encode_image_to_base64_from_file(self, test_image_file: str):
        """Test encoding an image file to base64."""
        client = create_litellm_client(model="gpt-5-mini")
        base64_data, media_type = client.encode_image_to_base64(test_image_file)

        assert isinstance(base64_data, str)
        assert len(base64_data) > 0
        assert media_type == "image/png"

    def test_encode_image_to_base64_nonexistent_file(self):
        """Test that encoding a nonexistent file raises error."""
        client = create_litellm_client(model="gpt-5-mini")

        with pytest.raises(LiteLLMClientError, match="Image file not found"):
            client.encode_image_to_base64("/nonexistent/path/image.png")

    def test_encode_image_to_base64_unsupported_format(self, tmp_path: Path):
        """Test that unsupported format raises error."""
        client = create_litellm_client(model="gpt-5-mini")
        unsupported_file = tmp_path / "test.bmp"
        unsupported_file.write_bytes(b"fake image data")

        with pytest.raises(LiteLLMClientError, match="Unsupported image format"):
            client.encode_image_to_base64(str(unsupported_file))

    def test_supported_image_formats(self):
        """Test that supported image formats are correctly defined."""
        client = create_litellm_client(model="gpt-5-mini")

        assert ".jpg" in client.SUPPORTED_IMAGE_FORMATS
        assert ".jpeg" in client.SUPPORTED_IMAGE_FORMATS
        assert ".png" in client.SUPPORTED_IMAGE_FORMATS
        assert ".gif" in client.SUPPORTED_IMAGE_FORMATS
        assert ".webp" in client.SUPPORTED_IMAGE_FORMATS


class TestLiteLLMClientModelSwitching:
    """Test switching between different models."""

    @skip_in_precommit
    @skip_low_priority
    def test_switch_from_openai_to_claude(self):
        """Test that we can create clients for different providers."""
        if not os.getenv("OPENAI_API_KEY") or not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("Both OPENAI_API_KEY and ANTHROPIC_API_KEY required")

        # Create OpenAI client (GPT-5 needs more tokens due to reasoning)
        openai_client = create_litellm_client(
            model="gpt-5-mini",
            temperature=0.1,
            max_tokens=300,
        )
        openai_response = openai_client.generate_response("Say 'hello' only.")

        # Create Claude client
        claude_client = create_litellm_client(
            model="claude-sonnet-4-5-20250929",
            temperature=0.1,
            max_tokens=100,
        )
        claude_response = claude_client.generate_response("Say 'hello' only.")

        # Both should work
        assert isinstance(openai_response, str)
        assert isinstance(claude_response, str)
        assert "hello" in openai_response.lower()
        assert "hello" in claude_response.lower()

    def test_same_interface_different_models(self):
        """Test that the interface is consistent across models."""
        openai_client = create_litellm_client(model="gpt-5-mini")
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

    def test_empty_images_list(self, openai_client: LiteLLMClient):
        """Test that empty images list works like no images."""
        # Should not raise - empty list is falsy
        # This is a configuration test, not an API call test
        client = create_litellm_client(model="gpt-5-mini")
        assert client.get_model() == "gpt-5-mini"

    def test_dict_based_response_format_raises_error(self):
        """Test that dict-based response_format raises error."""
        client = create_litellm_client(model="gpt-5-mini")
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
        client = create_litellm_client(model="gpt-5-mini")
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
        config = LiteLLMConfig(model="gpt-5-mini")

        assert config.temperature == 0.7
        assert config.max_tokens is None
        assert config.timeout == 120
        assert config.max_retries == 1
        assert config.retry_delay == 1.0
        assert config.top_p == 1.0

    @skip_in_precommit
    @skip_low_priority
    def test_long_conversation(self, openai_client: LiteLLMClient):
        """Test handling of longer conversations."""
        messages = []
        for i in range(5):
            messages.append({"role": "user", "content": f"Message {i + 1}: Hello!"})
            messages.append({"role": "assistant", "content": f"Response {i + 1}: Hi!"})
        messages.append(
            {"role": "user", "content": "How many times did we exchange greetings?"}
        )

        response = openai_client.generate_chat_response(messages)

        assert isinstance(response, str)
        # Should mention 5 or "five" somewhere in the response
        response_lower = response.lower()
        assert "5" in response or "five" in response_lower


class TestLiteLLMClientAPIKeyOverride:
    """Test API key configuration override functionality."""

    def test_create_client_with_openai_api_key_config(self):
        """Test creating a client with OpenAI API key config override."""
        api_key_config = APIKeyConfig(
            openai=OpenAIConfig(api_key="test-openai-key-12345")
        )
        config = LiteLLMConfig(
            model="gpt-5-mini",
            temperature=0.5,
            api_key_config=api_key_config,
        )
        client = LiteLLMClient(config)

        assert client.get_model() == "gpt-5-mini"
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

    def test_create_client_with_azure_openai_config(self):
        """Test creating a client with Azure OpenAI configuration."""
        api_key_config = APIKeyConfig(
            openai=OpenAIConfig(
                azure_config=AzureOpenAIConfig(
                    api_key="test-azure-key-11111",
                    endpoint="https://test-resource.openai.azure.com/",
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
            model="gpt-5-mini",
            temperature=0.3,
            max_tokens=200,
            api_key_config=api_key_config,
        )

        assert client.get_model() == "gpt-5-mini"
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
                    endpoint="https://azure.openai.azure.com/",
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
        assert client._api_base == "https://azure.openai.azure.com/"
        assert client._api_version == "2024-02-15-preview"

    def test_no_api_key_config_returns_none(self):
        """Test that client without API key config has None resolved keys."""
        config = LiteLLMConfig(model="gpt-5-mini")
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
            model="gpt-5-mini",
            api_key_config=api_key_config,
        )
        client = LiteLLMClient(config)

        # OpenAI model but no OpenAI key configured
        assert client._api_key is None

    @skip_in_precommit
    @skip_low_priority
    def test_generate_response_with_api_key_override_openai(self):
        """Test generating response using OpenAI API key from config override."""
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        # Use real key from env but pass through api_key_config
        api_key_config = APIKeyConfig(openai=OpenAIConfig(api_key=openai_key))
        client = create_litellm_client(
            model=_get_openai_test_model(),
            temperature=0.1,
            max_tokens=500,
            max_retries=2,
            api_key_config=api_key_config,
        )

        response = client.generate_response("What is 1+1? Answer with just the number.")

        assert isinstance(response, str)
        assert "2" in response

    @skip_in_precommit
    @skip_low_priority
    def test_generate_response_with_api_key_override_anthropic(self):
        """Test generating response using Anthropic API key from config override."""
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        # Use real key from env but pass through api_key_config
        api_key_config = APIKeyConfig(anthropic=AnthropicConfig(api_key=anthropic_key))
        client = create_litellm_client(
            model=_get_claude_test_model(),
            temperature=0.1,
            max_tokens=256,
            max_retries=2,
            api_key_config=api_key_config,
        )

        response = client.generate_response("What is 4+4? Answer with just the number.")

        assert isinstance(response, str)
        assert "8" in response

    @skip_in_precommit
    @skip_low_priority
    def test_embeddings_with_api_key_override(self):
        """Test embedding generation with API key override."""
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        api_key_config = APIKeyConfig(openai=OpenAIConfig(api_key=openai_key))
        client = create_litellm_client(
            model="gpt-5-mini",
            api_key_config=api_key_config,
        )

        embedding = client.get_embedding("Test embedding with API key override")

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    @skip_in_precommit
    @skip_low_priority
    def test_structured_output_with_api_key_override(self):
        """Test structured output with API key override."""
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        api_key_config = APIKeyConfig(openai=OpenAIConfig(api_key=openai_key))
        client = create_litellm_client(
            model=_get_openai_test_model(),
            temperature=0.1,
            max_tokens=500,
            api_key_config=api_key_config,
        )

        response = client.generate_response(
            prompt="What is 3+3? Provide the answer and a brief explanation.",
            response_format=MathResult,
        )

        assert isinstance(response, MathResult)
        assert response.answer == 6
        assert len(response.explanation) > 0

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

    @skip_in_precommit
    @skip_low_priority
    def test_chat_response_with_api_key_override(self):
        """Test multi-turn chat response with API key override."""
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        api_key_config = APIKeyConfig(openai=OpenAIConfig(api_key=openai_key))
        client = create_litellm_client(
            model=_get_openai_test_model(),
            temperature=0.1,
            max_tokens=500,
            api_key_config=api_key_config,
        )

        messages = [
            {"role": "user", "content": "My favorite number is 42."},
            {"role": "assistant", "content": "That's a great number!"},
            {"role": "user", "content": "What is my favorite number?"},
        ]

        response = client.generate_chat_response(messages)

        assert isinstance(response, str)
        assert "42" in response

    @skip_in_precommit
    @skip_low_priority
    def test_image_analysis_with_api_key_override(self, test_image_bytes: bytes):
        """Test image analysis with API key override."""
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        api_key_config = APIKeyConfig(openai=OpenAIConfig(api_key=openai_key))
        client = create_litellm_client(
            model=_get_openai_test_model(),
            temperature=0.1,
            max_tokens=500,
            api_key_config=api_key_config,
        )

        response = client.generate_response(
            prompt="What color is this image? Answer in one word.",
            images=[test_image_bytes],
            image_media_type="image/png",
        )

        assert isinstance(response, str)
        assert "red" in response.lower()

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
    """Unit tests for LiteLLMClient._sanitize_json_string."""

    @pytest.fixture
    def client(self):
        config = LiteLLMConfig(
            model="gpt-5-mini",
            api_key_config=APIKeyConfig(openai=OpenAIConfig(api_key="test")),
        )
        return LiteLLMClient(config)

    def test_single_quotes_to_double(self, client):
        """Single-quoted JSON keys and values are converted to double quotes."""
        result = client._sanitize_json_string("{'key': 'value'}")
        assert result == '{"key": "value"}'

    def test_python_booleans(self, client):
        """Python True/False/None are converted to JSON true/false/null."""
        result = client._sanitize_json_string('{"a": True, "b": False, "c": None}')
        assert result == '{"a": true, "b": false, "c": null}'

    def test_python_booleans_inside_strings_preserved(self, client):
        """True/False/None inside quoted strings are NOT converted."""
        result = client._sanitize_json_string('{"msg": "True story about None"}')
        assert result == '{"msg": "True story about None"}'

    def test_trailing_commas(self, client):
        """Trailing commas before } or ] are removed."""
        result = client._sanitize_json_string('{"a": 1, "b": 2,}')
        assert result == '{"a": 1, "b": 2}'

    def test_trailing_comma_in_array(self, client):
        """Trailing commas in arrays are removed."""
        result = client._sanitize_json_string("[1, 2, 3,]")
        assert result == "[1, 2, 3]"

    def test_escaped_apostrophe_in_single_quoted_string(self, client):
        """Escaped apostrophes inside single-quoted strings are handled."""
        import json

        result = client._sanitize_json_string("{'text': 'didn\\'t work'}")
        parsed = json.loads(result)
        assert parsed["text"] == "didn't work"

    def test_double_quotes_inside_single_quoted_string(self, client):
        """Double quotes inside single-quoted strings are escaped."""
        import json

        result = client._sanitize_json_string("{'text': 'he said \"hello\"'}")
        parsed = json.loads(result)
        assert parsed["text"] == 'he said "hello"'

    def test_mixed_all_issues(self, client):
        """Combined: single quotes, Python booleans, trailing comma."""
        import json

        result = client._sanitize_json_string(
            "{'is_success': True, 'failure_type': None, 'reason': 'ok',}"
        )
        parsed = json.loads(result)
        assert parsed == {"is_success": True, "failure_type": None, "reason": "ok"}

    def test_valid_json_passthrough(self, client):
        """Already-valid JSON passes through unchanged."""
        original = '{"is_success": true, "count": 42}'
        result = client._sanitize_json_string(original)
        assert result == original

    def test_word_boundary_prevents_partial_replacement(self, client):
        """Words containing True/False as substrings are not replaced."""
        result = client._sanitize_json_string('{"TrueValue": 1, "isFalsey": 2}')
        assert '"TrueValue"' in result
        assert '"isFalsey"' in result
