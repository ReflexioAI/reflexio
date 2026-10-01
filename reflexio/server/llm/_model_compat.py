"""Capabilities for qualified models newer than LiteLLM's bundled catalog.

Sources: OpenAI/Anthropic verified 2026-09-30; MiniMax/Z.ai 2026-10-01.
https://developers.openai.com/api/docs/models
https://platform.claude.com/docs/en/models/overview
https://platform.claude.com/docs/en/models/opus-5-5/migration-guide
https://platform.minimax.io/docs/api-reference/text-openai-api
https://docs.z.ai/guides/llm/glm-5.3
https://docs.z.ai/guides/vlm/glm-5.3-flash
Only direct provider IDs are covered; third-party routes retain their own policy.
"""

from typing import Any

import litellm

from reflexio.server.llm._litellm_types import LiteLLMClientError

OPENAI_MODELS = frozenset({"gpt-6-astra", "gpt-6.1-sol", "gpt-6-luna"})
ANTHROPIC_MODELS = frozenset(
    {"claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5"}
)
MINIMAX_PREVIEW_MODEL = "minimax/MiniMax-M3.1-Flash-Preview"
GLM_MODELS = frozenset({"zai/glm-5.3", "zai/glm-5.3-flash", "zai/glm-5.3-flashx"})
SECONDARY_MODELS = GLM_MODELS | {MINIMAX_PREVIEW_MODEL}


def qualified_model_id(model: str) -> str | None:
    """Recognize qualified IDs and their supported direct-provider routes."""
    if model in SECONDARY_MODELS:
        return model
    provider, _, name = model.partition("/")
    if not name:
        name = provider
        provider = ""
    if name in OPENAI_MODELS and provider in ("", "openai"):
        return name
    if name in ANTHROPIC_MODELS and provider in ("", "anthropic"):
        return name
    return None


def ensure_model_capabilities(model: str) -> None:
    """Supply missing catalog entries without overwriting provider data/prices.

    The fallback contains capabilities, not invented prices. Do not persist it
    across LiteLLM catalog reloads: a subsequently downloaded entry wins.
    """
    name = qualified_model_id(model)
    if name is None or name in litellm.model_cost:
        return
    if name in SECONDARY_MODELS:
        # These providers use prompt schemas in Reflexio; native strict JSON
        # schema support has not been established. Register capabilities only.
        litellm.register_model(
            {
                name: {
                    "litellm_provider": name.split("/", 1)[0],
                    "mode": "chat",
                    "supports_function_calling": True,
                    "supports_reasoning": True,
                }
            },
            persist_across_reloads=False,
        )
        return
    is_openai = name in OPENAI_MODELS
    capabilities: dict[str, Any] = {
        "litellm_provider": "openai" if is_openai else "anthropic",
        "mode": "chat",
        "max_tokens": 128000,
        "max_output_tokens": 128000,
        # Follow LiteLLM's GPT-6 Astra catalog convention: reserve max output
        # within the 1,050,000 total context window.
        "max_input_tokens": 922000 if is_openai else 1000000,
        "supports_function_calling": True,
        "supports_response_schema": True,
        "supports_reasoning": True,
        "supports_tool_choice": True,
    }
    if is_openai:
        capabilities["supports_none_reasoning_effort"] = name == "gpt-6-luna"
    else:
        capabilities.update(
            supports_native_structured_output=True,
            supports_output_config=True,
            supports_adaptive_thinking=True,
            thinking_always_on=True,
            supports_forced_tool_use=False,
            supports_sampling_params=False,
            supports_assistant_prefill=False,
        )
    litellm.register_model({name: capabilities}, persist_across_reloads=False)


def apply_model_request_policy(params: dict[str, Any]) -> None:
    """Apply documented sampling, reasoning, and tool constraints."""
    qualified = qualified_model_id(params["model"])
    if qualified is None:
        return
    if qualified in SECONDARY_MODELS:
        extra = params.get("extra_body") or {}
        thinking = extra.get("thinking", params.get("thinking"))
        effort = extra.get("reasoning_effort", params.get("reasoning_effort"))
        allowed_efforts = (
            {"low", "medium", "high", "xhigh", "max"}
            if qualified == MINIMAX_PREVIEW_MODEL
            else {"low", "high", "max"}
        )
        if isinstance(thinking, dict) and thinking.get("type") == "disabled":
            raise LiteLLMClientError(
                f"{qualified} requires thinking to remain enabled."
            )
        if effort is not None and (
            not isinstance(effort, str) or effort not in allowed_efforts
        ):
            raise LiteLLMClientError(
                f"{qualified} supports reasoning_effort values: "
                f"{', '.join(sorted(allowed_efforts))}."
            )
        if (
            qualified == MINIMAX_PREVIEW_MODEL
            and extra.get("reasoning_split", params.get("reasoning_split")) is False
        ):
            raise LiteLLMClientError(f"{qualified} requires reasoning_split=True.")
        if "reasoning_effort" in params:
            # LiteLLM 1.103.1 omits this documented field from both provider
            # adapters' supported parameter lists; otherwise drop_params loses it.
            params["allowed_openai_params"] = list(
                dict.fromkeys(
                    [*params.get("allowed_openai_params", []), "reasoning_effort"]
                )
            )
        return
    extra = dict(params.get("extra_body") or {})
    effort = extra.get("reasoning_effort", params.get("reasoning_effort"))
    if isinstance(effort, dict):
        effort = effort.get("effort")
    if isinstance(extra.get("reasoning"), dict):
        effort = extra["reasoning"].get("effort", effort)
    sampling_allowed = qualified == "gpt-6-luna" and effort == "none"
    for overrides in (params, extra):
        overrides.pop("top_k", None)
        if not sampling_allowed:
            for sampling_param in ("temperature", "top_p"):
                overrides.pop(sampling_param, None)
    if "extra_body" in params:
        params["extra_body"] = extra
    tool_choice = extra.get("tool_choice", params.get("tool_choice"))
    if qualified not in ANTHROPIC_MODELS:
        return
    if "tool_choice" in extra:
        # The native override wins; do not let LiteLLM validate or map the
        # superseded top-level choice before it overlays the final body.
        params.pop("tool_choice", None)
    response_format = params.get("response_format")
    if (
        isinstance(response_format, dict)
        and response_format.get("type") == "json_schema"
    ):
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        # LiteLLM's response_format mapper still emits the beta output_format
        # field. Reuse its schema normalization with the documented GA field.
        output_format = (
            AnthropicConfig().map_response_format_to_anthropic_output_format(
                response_format
            )
        )
        # Provider-specific output_config overwrites LiteLLM's effort mapping.
        # Reuse that mapping before merging the schema and caller overrides.
        mapped = AnthropicConfig().map_openai_params(
            non_default_params={"reasoning_effort": params.get("reasoning_effort")},
            optional_params={},
            model=qualified,
            drop_params=True,
        )
        params["output_config"] = {
            **mapped.get("output_config", {}),
            **params.get("output_config", {}),
            **extra.pop("output_config", {}),
            "format": output_format,
        }
        params.pop("response_format")
    if tool_choice is None:
        return
    choice_type = (
        tool_choice.get("type") if isinstance(tool_choice, dict) else tool_choice
    )
    if choice_type not in ("auto", "none"):
        raise LiteLLMClientError(
            f"{params['model']} does not support forced tool selection; "
            "use tool_choice='auto' or 'none'."
        )


def model_completion(params: dict[str, Any]) -> Any:
    """Keep signed Claude blocks using a request-local LiteLLM logging hook."""
    if qualified_model_id(params["model"]) not in ANTHROPIC_MODELS:
        return litellm.completion(**params)
    import datetime
    import json
    import uuid

    from litellm.litellm_core_utils.rules import Rules
    from litellm.utils import function_setup

    request_params = {
        **params,
        "litellm_call_id": params.get("litellm_call_id", str(uuid.uuid4())),
    }
    # Provide a request-local logger: LiteLLM skips its second setup. Raw
    # response blocks are otherwise discarded by its Anthropic transformation.
    logging_obj, request = function_setup(
        "completion",
        Rules(),
        datetime.datetime.now(),
        is_async_call=False,
        **request_params,
    )
    caller_logger = request.get("logger_fn")
    original_content = None
    originals = {}
    for message in params["messages"]:
        fields = message.get("provider_specific_fields") or {}
        original = fields.get("reflexio_anthropic_content")
        if isinstance(original, list):
            tool_ids = frozenset(
                b["id"] for b in original if b.get("type") == "tool_use"
            )
            if tool_ids:
                originals[tool_ids] = original

    def capture(details: dict[str, Any]) -> None:
        nonlocal original_content
        if details.get("log_event_type") == "pre_api_call" and originals:
            body = (details.get("additional_args") or {}).get("complete_input_dict")
            if isinstance(body, dict):
                # Replace only the matching assistant blocks in the final body.
                # LiteLLM still owns all user/tool-result/cache transformations.
                for message in body.get("messages", []):
                    if message.get("role") != "assistant":
                        continue
                    tool_ids = frozenset(
                        b["id"]
                        for b in message.get("content", [])
                        if isinstance(b, dict) and b.get("type") == "tool_use"
                    )
                    if tool_ids in originals:
                        message["content"] = originals[tool_ids]
        if details.get("log_event_type") == "post_api_call":
            raw = details.get("original_response")
            if isinstance(raw, str):
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = None
                if isinstance(body, dict) and isinstance(body.get("content"), list):
                    original_content = body["content"]
        if callable(caller_logger):
            caller_logger(details)

    logging_obj.logger_fn = capture
    response = litellm.completion(
        **{**request, "litellm_logging_obj": logging_obj, "logger_fn": capture}
    )
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict) and original_content is not None:
        hidden["original_response"] = original_content
    return response
