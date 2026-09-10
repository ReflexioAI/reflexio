"""
LLM client package providing unified access to multiple LLM providers.

This package uses LiteLLM as the backend to provide a consistent interface
for OpenAI, Claude, Azure OpenAI, and other LLM providers.
"""

from .litellm_client import (
    STRUCTURED_OUTPUT_CORRECTION_PREFIX,
    LiteLLMClient,
    LiteLLMClientError,
    LiteLLMConfig,
    ProviderRequestGuardError,
    StructuredOutputRepairError,
    StructuredOutputValidator,
    ToolCallingChatResponse,
    create_litellm_client,
    is_structured_output_correction_turn,
    structured_output_correction_turn,
    structured_output_repair_idempotency_key,
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
    "ProviderRequestGuardError",
    "StructuredOutputRepairError",
    "StructuredOutputValidator",
    "ModelRole",
    "ToolCallingChatResponse",
    "create_litellm_client",
    "STRUCTURED_OUTPUT_CORRECTION_PREFIX",
    "is_structured_output_correction_turn",
    "resolve_model_name",
    "structured_output_correction_turn",
    "structured_output_repair_idempotency_key",
    "validate_llm_availability",
]
