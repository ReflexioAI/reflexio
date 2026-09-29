"""Public exports loaded on demand so inference stays independent of app code."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

_EXPORTS = {
    name: module
    for module, names in {
        "reflexio.server.llm.litellm_client": (
            "STRUCTURED_OUTPUT_CORRECTION_PREFIX",
            "LiteLLMClient",
            "LiteLLMClientError",
            "LiteLLMConfig",
            "ProviderRequestGuardError",
            "StructuredOutputRepairError",
            "StructuredOutputValidator",
            "ToolCallingChatResponse",
            "create_litellm_client",
            "is_structured_output_correction_turn",
            "structured_output_correction_turn",
            "structured_output_repair_idempotency_key",
        ),
        "reflexio.server.llm.model_defaults": (
            "ModelRole",
            "resolve_model_name",
            "validate_llm_availability",
        ),
    }.items()
    for name in names
}

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


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
