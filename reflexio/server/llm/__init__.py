"""
LLM client package providing unified access to multiple LLM providers.

This package uses LiteLLM as the backend to provide a consistent interface
for OpenAI, Claude, Azure OpenAI, and other LLM providers.
"""

from .litellm_client import (
    LiteLLMClient,
    LiteLLMClientError,
    LiteLLMConfig,
    create_litellm_client,
)
from .model_defaults import (
    ModelRole,
    resolve_model_name,
    validate_llm_availability,
)

__all__ = [
    "LiteLLMClient",
    "LiteLLMConfig",
    "LiteLLMClientError",
    "ModelRole",
    "create_litellm_client",
    "resolve_model_name",
    "validate_llm_availability",
]
