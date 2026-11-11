"""Native complete-request admission through the actual completion client."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from reflexio.server.llm import input_budget
from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMClientError,
    LiteLLMConfig,
)


@pytest.fixture
def transport(monkeypatch):
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None, refusal=None),
                finish_reason="stop",
            )
        ],
        usage=None,
        model="fixture",
        _hidden_params={},
    )
    dispatch = Mock(return_value=response)
    counter = Mock(return_value=100)
    monkeypatch.setattr(input_budget.litellm, "token_counter", counter)
    monkeypatch.setattr(
        input_budget.litellm,
        "get_model_info",
        Mock(return_value={"max_input_tokens": 1000, "max_output_tokens": 200}),
    )
    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o", max_tokens=20, fallback_models=[])
    )
    monkeypatch.setattr(client, "_completion_with_hard_timeout", dispatch)
    return client, dispatch, counter


def test_complete_capacity_refusal_precedes_dispatch_and_scope_resets(
    transport, monkeypatch
):
    client, dispatch, counter = transport
    counter.return_value = 1000
    with (
        pytest.raises(input_budget.InputBudgetExceededError),
        input_budget.model_input_budget(),
    ):
        client.generate_chat_response([{"role": "user", "content": "full evidence"}])
    dispatch.assert_not_called()
    assert (
        client.generate_chat_response(
            [{"role": "user", "content": "ordinary unscoped request"}]
        )
        == "ok"
    )
    assert dispatch.call_count == 1


def test_native_default_output_is_reserved_on_the_actual_dispatch(transport):
    client, dispatch, _ = transport
    client.config.max_tokens = None
    with input_budget.model_input_budget():
        assert (
            client.generate_chat_response([{"role": "user", "content": "evidence"}])
            == "ok"
        )
    assert dispatch.call_args.args[0]["max_tokens"] == 200


def test_smaller_fallback_cannot_bypass_complete_budget(transport, monkeypatch):
    client, dispatch, _ = transport
    client.config.fallback_models = ["gpt-4o-mini"]
    dispatch.side_effect = LiteLLMClientError("fixture primary transport failed")
    monkeypatch.setattr(
        input_budget.litellm,
        "get_model_info",
        lambda model: {
            "max_input_tokens": 1000 if model == "gpt-4o" else 100,
            "max_output_tokens": 20,
        },
    )
    with (
        pytest.raises(input_budget.InputBudgetExceededError),
        input_budget.model_input_budget(),
    ):
        client.generate_chat_response([{"role": "user", "content": "same evidence"}])
    assert dispatch.call_count == 1
    assert dispatch.call_args.args[0]["model"] == "gpt-4o"


@pytest.mark.parametrize("provider", ["claude-code", "openclaw"])
def test_opaque_host_retains_its_own_capacity_owner(transport, monkeypatch, provider):
    client, dispatch, counter = transport
    client.config.max_tokens = None
    monkeypatch.setattr(
        input_budget.litellm,
        "get_model_info",
        Mock(side_effect=AssertionError("host has no verified native capacity")),
    )
    with input_budget.model_input_budget():
        assert (
            client.generate_chat_response(
                [{"role": "user", "content": "complete host evidence"}],
                model=f"{provider}/default",
            )
            == "ok"
        )
    counter.assert_not_called()
    assert "max_tokens" not in dispatch.call_args.args[0]


@pytest.mark.parametrize("cap", [None, 20])
@pytest.mark.parametrize(
    "model", ["gpt-4o", "minimax/MiniMax-M3", "zai/glm-5.3-flashx"]
)
def test_custom_endpoint_requires_its_own_declaration_and_output_policy(
    transport, monkeypatch, cap, model
):
    client, dispatch, _ = transport
    client.config.max_tokens = cap
    import json

    monkeypatch.setenv(
        "REFLEXIO_LLM_CONTEXT_LIMITS_JSON",
        json.dumps({f"{model}@https://custom.invalid": 100_000}),
    )
    if cap is None:
        with (
            pytest.raises(
                input_budget.InputBudgetConfigurationError, match="output reservation"
            ),
            input_budget.model_input_budget(),
        ):
            client.generate_chat_response(
                [{"role": "user", "content": "evidence"}],
                api_base="https://custom.invalid",
                model=model,
            )
        dispatch.assert_not_called()
    else:
        with input_budget.model_input_budget():
            assert (
                client.generate_chat_response(
                    [{"role": "user", "content": "evidence"}],
                    api_base="https://custom.invalid",
                    model=model,
                )
                == "ok"
            )
        assert dispatch.call_args.args[0]["api_base"] == "https://custom.invalid"
        assert dispatch.call_args.args[0]["max_tokens"] == cap


def test_corrective_repair_growth_is_readmitted_before_dispatch(transport, monkeypatch):
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: int

    client, dispatch, counter = transport
    dispatch.return_value.choices[0].message.content = '{"value": 0}'

    def count(**kwargs):
        messages = kwargs.get("messages", [])
        return (
            1000
            if any(message.get("role") == "assistant" for message in messages)
            else 100
            if messages
            else 0
        )

    counter.side_effect = count
    with (
        pytest.raises(input_budget.InputBudgetExceededError),
        input_budget.model_input_budget(),
    ):
        client.generate_chat_response(
            [{"role": "user", "content": "evidence"}],
            response_format=Answer,
            structured_output_validator=lambda answer: (
                ["Repair the fixture value."] if answer.value == 0 else []
            ),
        )
    assert dispatch.call_count == 1
    assert any(
        any(
            message.get("role") == "assistant"
            for message in call.kwargs.get("messages", [])
        )
        for call in counter.call_args_list
    )


def test_nested_budget_scope_restores_captured_outer_declarations(
    transport, monkeypatch
):
    client, dispatch, _ = transport
    monkeypatch.setenv("REFLEXIO_LLM_CONTEXT_LIMITS_JSON", '{"gpt-4o": 1000}')
    with input_budget.model_input_budget():
        monkeypatch.setenv("REFLEXIO_LLM_CONTEXT_LIMITS_JSON", '{"gpt-4o": 100}')
        with (
            pytest.raises(input_budget.InputBudgetExceededError),
            input_budget.model_input_budget(),
        ):
            client.generate_chat_response([{"role": "user", "content": "evidence"}])
        assert (
            client.generate_chat_response([{"role": "user", "content": "evidence"}])
            == "ok"
        )
    assert dispatch.call_count == 1


@pytest.mark.parametrize("model", ["gpt-4o", "anthropic/claude-sonnet-5-5"])
def test_schema_and_tools_are_included_in_complete_request_admission(transport, model):
    client, dispatch, counter = transport
    counter.side_effect = lambda **kwargs: 900 if "text" in kwargs else 100
    tools = [{"type": "function", "function": {"name": "check"}}]
    from pydantic import BaseModel

    class Answer(BaseModel):
        content: str

    schema = Answer
    with (
        pytest.raises(input_budget.InputBudgetExceededError),
        input_budget.model_input_budget(),
    ):
        client.generate_chat_response(
            [{"role": "user", "content": "evidence"}],
            tools=tools,
            model=model,
            response_format=schema,
            parse_structured_output=False,
        )
    dispatch.assert_not_called()
    assert counter.call_args_list[0].kwargs["tools"] == tools
    assert any("text" in call.kwargs for call in counter.call_args_list)


@pytest.mark.parametrize("model", ["minimax/MiniMax-M3", "zai/glm-5.3-flashx"])
def test_native_provider_output_default_is_preserved(transport, monkeypatch, model):
    from reflexio.server.llm.model_defaults import default_completion_api_base

    client, dispatch, _ = transport
    client.config.max_tokens = None
    monkeypatch.setattr(
        input_budget.litellm,
        "get_model_info",
        Mock(return_value={"max_input_tokens": 100_000, "max_output_tokens": 10_000}),
    )
    overrides = {"model": model}
    if route := default_completion_api_base(model):
        overrides["api_base"] = route
    with input_budget.model_input_budget():
        assert (
            client.generate_chat_response(
                [{"role": "user", "content": "evidence"}], **overrides
            )
            == "ok"
        )
    assert dispatch.call_args.args[0]["max_tokens"] == 8192
