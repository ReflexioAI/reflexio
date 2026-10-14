"""Complete-request admission at the existing provider dispatch boundary."""

import json
import math
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from typing import Any

import litellm

from reflexio.server.llm._litellm_types import (
    ModelProvenance,
    ProviderRequestGuard,
    ProviderRequestGuardError,
)
from reflexio.server.llm.token_accounting import RunTokenTotals

# Scoped to one logical extraction; copied worker contexts carry the same
# durable coordinator callback. Other LLM callers keep their existing behavior.
extraction_provider_guard: ContextVar[ProviderRequestGuard | None] = ContextVar(
    "extraction_provider_guard", default=None
)
extraction_usage_observer: ContextVar[
    Callable[[ModelProvenance, RunTokenTotals], None] | None
] = ContextVar("extraction_usage_observer", default=None)
extraction_request_identity: ContextVar[dict[str, Any] | None] = ContextVar(
    "extraction_request_identity", default=None
)


class InputBudgetConfigurationError(ProviderRequestGuardError):
    """A safe effective model/endpoint capacity could not be established."""


class InputBudgetExceededError(ProviderRequestGuardError):
    """The complete prompt plus reserved output exceeds its admission budget."""


class ProviderInputOverflowError(InputBudgetExceededError):
    """An explicitly classified provider rejection, before fallback/retry."""

    def __init__(self, identity: dict[str, Any] | None):
        super().__init__("Provider rejected extraction input capacity")
        self.identity = dict(identity or {})


def declared_context_limits(raw: str) -> dict[str, int]:
    """Parse endpoint-specific operator declarations without permissive coercion.

    Keys are the actual model for ordinary provider endpoints, or
    ``model@api_base`` for a custom endpoint. Empty values mean no declarations.
    """
    try:
        value = json.loads(raw.strip() or "{}")
    except (ValueError, TypeError) as exc:
        raise InputBudgetConfigurationError(
            "Invalid context-limit declarations"
        ) from exc
    if not isinstance(value, dict) or any(
        not isinstance(key, str)
        or not key.strip()
        or type(limit) is not int
        or limit <= 0
        for key, limit in value.items()
    ):
        raise InputBudgetConfigurationError("Context limits require positive integers")
    return value


def supply_extraction_output_reservation(params: dict[str, Any]) -> None:
    """Bound uncapped native requests without replacing configured output caps.

    Reserve and request the advertised native output maximum. Custom endpoints
    must supply their own cap; another deployment's metadata is not authority.
    """
    if (
        params.get("max_completion_tokens") is not None
        or params.get("max_tokens") is not None
    ):
        return
    if params.get("api_base") or params.get("base_url"):
        raise InputBudgetConfigurationError(
            "Custom endpoint requires an explicit output reservation"
        )
    try:
        maximum = litellm.get_model_info(params["model"]).get("max_output_tokens")
    except Exception as exc:
        raise InputBudgetConfigurationError(
            "Effective output reservation is required"
        ) from exc
    if type(maximum) is not int or maximum <= 0:
        raise InputBudgetConfigurationError("Effective output reservation is required")
    key = "max_completion_tokens" if "max_completion_tokens" in params else "max_tokens"
    params[key] = maximum


def measure_request_input_tokens(params: Mapping[str, Any]) -> int:
    """Count complete built input with margin, without rejecting for capacity.

    This is the same counter used by admission. Pure compression renderers need
    an oversized integer so they can drop whole evidence groups; configuration
    and counting failures still raise. Output reservation and context capacity
    remain the admission guard's responsibility.
    """
    model = params.get("model")
    if not isinstance(model, str) or not model:
        raise InputBudgetConfigurationError("Missing effective model")
    messages = params.get("messages")
    if not isinstance(messages, list):
        raise InputBudgetConfigurationError("Complete provider messages are required")
    try:
        tokens = litellm.token_counter(
            model=model,
            messages=messages,
            tools=params.get("tools"),
            tool_choice=params.get("tool_choice"),
        )
        response_format = params.get("response_format")
        if response_format:
            if hasattr(response_format, "model_json_schema"):
                response_format = response_format.model_json_schema()
            tokens += litellm.token_counter(
                model=model,
                text=json.dumps(response_format, ensure_ascii=False, sort_keys=True),
            )
    except Exception as exc:
        raise InputBudgetConfigurationError("Complete prompt counting failed") from exc
    return math.ceil(tokens * 1.2)


def request_input_capacity(
    params: Mapping[str, Any],
    *,
    context_limits: Mapping[str, int],
    prompt_target: int = 65536,
) -> int:
    """Return the input ceiling after the actual native output reservation.

    This is provider-free and does not count messages. Pure consumer preparation
    uses the same capacity rules as dispatch; custom routes need declarations.
    """
    model = params.get("model")
    if not isinstance(model, str) or not model or prompt_target <= 0:
        raise InputBudgetConfigurationError("Missing effective model or prompt target")
    api_base = params.get("api_base") or params.get("base_url")
    key = f"{model}@{api_base}" if api_base else model
    context_limit = context_limits.get(key)
    if context_limit is None:
        if api_base:
            raise InputBudgetConfigurationError(
                "Custom endpoint requires a context declaration"
            )
        try:
            info = litellm.get_model_info(model)
        except Exception as exc:
            raise InputBudgetConfigurationError(
                "Unknown model context capacity"
            ) from exc
        context_limit = info.get("max_input_tokens")
        # A generic max_tokens can describe OUTPUT capacity; do not treat it as
        # a context-window declaration when max_input_tokens is absent.
    if type(context_limit) is not int or context_limit <= 0:
        raise InputBudgetConfigurationError("Unknown model context capacity")
    output_tokens = params.get("max_completion_tokens", params.get("max_tokens"))
    if type(output_tokens) is not int or output_tokens <= 0:
        raise InputBudgetConfigurationError("Effective output reservation is required")
    return min(prompt_target, context_limit - output_tokens)


def check_request_input_budget(
    params: Mapping[str, Any],
    *,
    context_limits: Mapping[str, int],
    prompt_target: int = 65536,
) -> int:
    """Check each actual built request, including repairs and fallback rungs."""
    capacity = request_input_capacity(
        params, context_limits=context_limits, prompt_target=prompt_target
    )
    budgeted_tokens = measure_request_input_tokens(params)
    if budgeted_tokens > capacity:
        raise InputBudgetExceededError(
            "Complete extraction prompt exceeds input capacity"
        )
    return budgeted_tokens


def extraction_request_guard(
    *,
    context_limits: Mapping[str, int],
    reserve_attempt: Callable[[], None],
    prompt_target: int = 65536,
) -> Callable[[dict[str, Any], float, tuple[str, ...], bool], None]:
    """Reserve durable quota only after request admission, before dispatch."""

    def guard(
        params: dict[str, Any],
        _hard_timeout: float,
        _ladder: tuple[str, ...],
        _parse_structured_output: bool,
    ) -> None:
        check_request_input_budget(
            params, context_limits=context_limits, prompt_target=prompt_target
        )
        try:
            reserve_attempt()
        except ProviderRequestGuardError:
            raise
        except Exception as exc:
            raise ProviderRequestGuardError(
                "Durable extraction attempt reservation failed"
            ) from exc

    return guard
