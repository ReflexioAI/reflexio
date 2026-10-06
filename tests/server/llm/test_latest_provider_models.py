"""Qualify actual provider adapters offline, including two-turn tool exchanges."""

import json
import pickle
from inspect import getattr_static
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import litellm
import pytest
from pydantic import BaseModel, ConfigDict

from reflexio.server.llm._litellm_subprocess import _picklable_completion_result
from reflexio.server.llm._litellm_types import StructuredOutputRepairError
from reflexio.server.llm._model_compat import (
    ANTHROPIC_MODELS,
    OPENAI_MODELS,
    SECONDARY_MODELS,
    ensure_model_capabilities,
    model_completion,
)
from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMClientError,
    LiteLLMConfig,
    ToolCallingChatResponse,
)
from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.llm.tools import Tool, ToolRegistry, run_tool_loop
from reflexio.test_support.llm_mock import (
    litellm_is_patched,
    patched_litellm,
    unpatched_litellm,
)

MODELS = sorted(OPENAI_MODELS | ANTHROPIC_MODELS) + ["claude-haiku-4-5-20251001"]


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str


class LookupArgs(BaseModel):
    key: str


@pytest.fixture(autouse=True)
def offline_catalog(monkeypatch):
    """Use only the wheel's catalog; downloaded metadata must not mask gaps."""
    bundled = (
        Path(litellm.__file__).parent / "model_prices_and_context_window_backup.json"
    )
    monkeypatch.setattr(litellm, "model_cost", json.loads(bundled.read_text()))
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
    monkeypatch.delenv("REFLEXIO_LLM_SEED", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-placeholder")
    LiteLLMClient._supports_response_schema.cache_clear()
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    LiteLLMClient._supports_response_schema.cache_clear()
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("prefixed", [False, True])
def test_offline_model_capabilities_and_sampling(model, prefixed, monkeypatch):
    provider = "openai" if model in OPENAI_MODELS else "anthropic"
    routed = f"{provider}/{model}" if prefixed else model
    client = LiteLLMClient(LiteLLMConfig(model=routed, temperature=0.3, top_p=0.8))
    monkeypatch.setenv("REFLEXIO_LLM_SEED", "7")
    params = client._build_completion_params(
        [{"role": "user", "content": "hello"}], temperature=0.2, top_p=0.5, top_k=4
    )[0]
    assert client._supports_response_schema(routed)
    assert litellm.supports_function_calling(model=routed)
    assert litellm.get_llm_provider(routed)[1] == provider
    if model in OPENAI_MODELS | ANTHROPIC_MODELS:
        assert not {"temperature", "top_p", "top_k"}.intersection(params)
    else:
        assert params["temperature"] == 0.0
        assert params["top_p"] == 0.5


@pytest.mark.parametrize("prefixed", [False, True])
def test_luna_explicit_no_reasoning_preserves_sampling(prefixed, transport):
    requests, responses = transport
    responses.append(_chat_response("gpt-6-luna", "done"))
    model = "openai/gpt-6-luna" if prefixed else "gpt-6-luna"
    client = LiteLLMClient(LiteLLMConfig(model=model))
    assert (
        client.generate_chat_response(
            [{"role": "user", "content": "hello"}],
            reasoning_effort="none",
            temperature=0.2,
            top_p=0.5,
        )
        == "done"
    )
    assert requests[0][1]["temperature"] == 0.2
    assert requests[0][1]["top_p"] == 0.5
    assert requests[0][1]["reasoning_effort"] == "none"


@pytest.mark.parametrize("model", sorted(ANTHROPIC_MODELS))
@pytest.mark.parametrize("effort", ["low", {"effort": "low"}])
@pytest.mark.parametrize("output_config", [None, {"effort": "high"}])
def test_claude_schema_preserves_explicit_reasoning_effort(
    model, effort, output_config, transport
):
    requests, responses = transport
    responses.append(_provider_response(model))
    client = LiteLLMClient(LiteLLMConfig(model=model))
    assert client.generate_chat_response(
        [{"role": "user", "content": "Answer with done."}],
        response_format=Answer,
        reasoning_effort=effort,
        **({"output_config": output_config} if output_config else {}),
    ) == Answer(answer="done")
    assert requests[0][1]["output_config"]["effort"] == (
        "high" if output_config else "low"
    )
    assert requests[0][1]["output_config"]["format"]["type"] == "json_schema"


@pytest.mark.parametrize("model", sorted(OPENAI_MODELS))
@pytest.mark.parametrize("tools", [False, True])
def test_openai_sampling_restrictions_cover_extra_body(model, tools, transport):
    requests, responses = transport
    responses.append(
        _provider_response(model) if tools else _chat_response(model, "done")
    )
    extra_body = {"temperature": 0.2, "top_p": 0.5, "top_k": 2}
    client = LiteLLMClient(LiteLLMConfig(model=model))
    client.generate_chat_response(
        [{"role": "user", "content": "Answer with done."}],
        extra_body=extra_body,
        **(
            {
                "tools": [
                    Tool(
                        name="lookup", args_model=LookupArgs, handler=lambda *_: {}
                    ).openai_spec()
                ],
                "response_format": Answer,
            }
            if tools
            else {}
        ),
    )
    assert not {"temperature", "top_p", "top_k"}.intersection(requests[0][1])
    assert extra_body == {"temperature": 0.2, "top_p": 0.5, "top_k": 2}


@pytest.mark.parametrize(
    "effort,extra_effort,tools,expected_sampling",
    [
        ("none", {"reasoning_effort": "high"}, False, False),
        ("high", {"reasoning_effort": "none"}, False, True),
        ({"effort": "none"}, {}, False, True),
        ("low", {"reasoning": {"effort": "high"}}, True, False),
        (None, {"reasoning": {"effort": "none"}}, True, True),
    ],
)
def test_luna_sampling_uses_effective_extra_body_effort(
    effort, extra_effort, tools, expected_sampling, transport
):
    requests, responses = transport
    model = "gpt-6-luna"
    responses.append(
        _provider_response(model) if tools else _chat_response(model, "done")
    )
    extra = {"temperature": 0.2, "top_p": 0.5, "top_k": 2, **extra_effort}
    client = LiteLLMClient(LiteLLMConfig(model=model))
    client.generate_chat_response(
        [{"role": "user", "content": "Answer with done."}],
        extra_body=extra,
        reasoning_effort=effort,
        **(
            {
                "tools": [
                    Tool(
                        name="lookup", args_model=LookupArgs, handler=lambda *_: {}
                    ).openai_spec()
                ],
                "response_format": Answer,
            }
            if tools
            else {}
        ),
    )
    body = requests[0][1]
    assert "top_k" not in body
    assert ("temperature" in body) is expected_sampling
    assert ("top_p" in body) is expected_sampling


def test_capability_fallback_preserves_downloaded_metadata(monkeypatch):
    entry = {"litellm_provider": "openai", "input_cost_per_token": 0.123}
    monkeypatch.setitem(litellm.model_cost, "gpt-6.1-sol", entry)
    ensure_model_capabilities("openai/gpt-6.1-sol")
    assert litellm.model_cost["gpt-6.1-sol"] is entry
    assert entry == {"litellm_provider": "openai", "input_cost_per_token": 0.123}
    ensure_model_capabilities("anthropic/claude-future-99")
    assert "claude-future-99" not in litellm.model_cost


@pytest.mark.parametrize("model", sorted(ANTHROPIC_MODELS))
@pytest.mark.parametrize(
    "choice", ["required", {"type": "function", "function": {"name": "lookup"}}]
)
def test_forced_claude_tools_fail_before_transport(model, choice):
    client = LiteLLMClient(LiteLLMConfig(model=model))
    with (
        patch("litellm.completion") as completion,
        pytest.raises(
            LiteLLMClientError, match="does not support forced tool selection"
        ),
    ):
        client.generate_chat_response(
            [{"role": "user", "content": "hello"}], tools=[], tool_choice=choice
        )
    completion.assert_not_called()


@pytest.mark.parametrize("model", sorted(ANTHROPIC_MODELS))
@pytest.mark.parametrize(
    "choice", [{"type": "any"}, {"type": "tool", "name": "lookup"}]
)
def test_forced_claude_extra_body_tools_fail_before_transport(model, choice):
    client = LiteLLMClient(LiteLLMConfig(model=model))
    with (
        patch("litellm.completion") as completion,
        pytest.raises(LiteLLMClientError, match="forced tool selection"),
    ):
        client.generate_chat_response(
            [{"role": "user", "content": "hello"}],
            tools=[],
            tool_choice="auto",
            extra_body={"tool_choice": choice},
        )
    completion.assert_not_called()


@pytest.mark.parametrize("model", sorted(ANTHROPIC_MODELS))
def test_claude_extra_body_effort_keeps_schema(model, transport):
    requests, responses = transport
    responses.append(_provider_response(model))
    extra = {"output_config": {"effort": "high"}}
    client = LiteLLMClient(LiteLLMConfig(model=model))
    assert client.generate_chat_response(
        [{"role": "user", "content": "Answer with done."}],
        response_format=Answer,
        reasoning_effort="low",
        extra_body=extra,
    ) == Answer(answer="done")
    assert requests[0][1]["output_config"]["effort"] == "high"
    assert requests[0][1]["output_config"]["format"]["type"] == "json_schema"
    assert extra == {"output_config": {"effort": "high"}}


def _provider_response(model, *, tool_turn=False, text='{"answer":"done"}'):
    if model not in OPENAI_MODELS:
        content = [{"type": "text", "text": text}]
        if tool_turn:
            content = [
                {"type": "thinking", "thinking": "", "signature": "signed-thinking"},
                {"type": "text", "text": "Looking up the key."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "lookup",
                    "input": {"key": "test"},
                },
            ]
        return {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": "tool_use" if tool_turn else "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 20, "output_tokens": 10},
        }
    output = [
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
    ]
    if tool_turn:
        output = [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [],
                "encrypted_content": "encrypted-reasoning",
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": '{"key":"test"}',
                "status": "completed",
            },
        ]
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": model,
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
    }


@pytest.fixture
def transport(monkeypatch):
    requests = []
    responses = []

    def send(_client, request, **_kwargs):
        assert request.method == "POST", (
            f"Unexpected network request: {request.method} {request.url}"
        )
        requests.append((str(request.url), json.loads(request.content)))
        assert responses, f"Unexpected extra provider request: {request.url}"
        payload = responses.pop(0)
        status, payload = payload if isinstance(payload, tuple) else (200, payload)
        return httpx.Response(status, json=payload, request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(
        LiteLLMClient, "_should_process_isolate_completion", lambda *_a: False
    )
    with unpatched_litellm(), monkeypatch.context() as completion_patch:
        # An enterprise + OSS selection can install nested session patchers.
        # Bind the exported real entry point explicitly so this test cannot
        # pass through a remaining session mock.
        from litellm.main import completion

        completion_patch.setattr(litellm, "completion", completion)
        yield requests, responses
    assert not responses, "The expected provider exchange did not complete"


def test_transport_teardown_preserves_the_session_mock():
    with patched_litellm():
        patches = pytest.MonkeyPatch()
        fixture = getattr_static(transport, "__wrapped__")(patches)
        try:
            next(fixture)
            with pytest.raises(StopIteration):
                next(fixture)
        finally:
            fixture.close()
            patches.undo()
        assert litellm_is_patched(), (
            "Adapter teardown must not disable later unit mocks"
        )


@pytest.mark.parametrize("model", sorted(ANTHROPIC_MODELS))
def test_signed_block_capture_preserves_caller_logger(model, transport):
    requests, responses = transport
    native = _provider_response(model, tool_turn=True)
    responses.append(native)
    ensure_model_capabilities(model)
    events = []
    response = model_completion(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Look up first."}],
            "logger_fn": lambda details: events.append(details["log_event_type"]),
        }
    )
    assert len(requests) == 1
    assert "pre_api_call" in events
    assert "post_api_call" in events
    assert response._hidden_params["original_response"] == native["content"]


@pytest.mark.parametrize("model", MODELS)
def test_real_adapter_structured_output(model, transport):
    requests, responses = transport
    responses.append(_provider_response(model))
    client = LiteLLMClient(LiteLLMConfig(model=model))
    if model in OPENAI_MODELS:
        # Function tools require the real SDK's Responses bridge.
        result = client.generate_chat_response(
            [{"role": "user", "content": "Answer with done."}],
            tools=[
                Tool(
                    name="lookup", args_model=LookupArgs, handler=lambda *_: {}
                ).openai_spec()
            ],
            response_format=Answer,
        )
        assert isinstance(result, ToolCallingChatResponse)
        assert isinstance(result.parsed_output, Answer)
        assert result.parsed_output.answer == "done"
        assert requests[0][0].endswith("/responses")
        schema = requests[0][1]["text"]["format"]
    else:
        result = client.generate_chat_response(
            [{"role": "user", "content": "Answer with done."}], response_format=Answer
        )
        assert result == Answer(answer="done")
        assert requests[0][0].endswith("/messages")
        schema = (
            requests[0][1]["output_config"]["format"]
            if model in ANTHROPIC_MODELS
            else requests[0][1]["output_format"]
        )
    assert schema["type"] == "json_schema"
    assert schema["schema"]["additionalProperties"] is False
    assert requests[0][1]["model"] == model
    if model in OPENAI_MODELS | ANTHROPIC_MODELS:
        assert not {"temperature", "top_p", "top_k"}.intersection(requests[0][1])


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("snapshot", [False, True])
def test_real_adapter_two_turn_tool_loop(model, snapshot, transport, monkeypatch):
    requests, responses = transport
    responses.extend(
        [_provider_response(model, tool_turn=True), _provider_response(model)]
    )
    client = LiteLLMClient(LiteLLMConfig(model=model))
    monkeypatch.setattr(
        "reflexio.server.llm.tools.resolve_model_name", lambda **_: model
    )
    monkeypatch.setattr(
        "reflexio.server.llm._litellm_text_generation.resolve_model_name",
        lambda *_a, **_k: model,
    )
    if snapshot:
        real_completion = litellm.completion

        def force_snapshot(**params):
            response = real_completion(**params)
            # Unpicklable unrelated state forces the production snapshot branch.
            raw = SimpleNamespace(**response.__dict__, unpicklable=lambda: None)
            return pickle.loads(pickle.dumps(_picklable_completion_result(raw)))  # noqa: S301 — trusted fixture round-trip

        monkeypatch.setattr(litellm, "completion", force_snapshot)
    calls = []

    def lookup(args: BaseModel, _ctx):
        assert isinstance(args, LookupArgs)
        calls.append(args.key)
        return {"value": "found"}

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="lookup",
            args_model=LookupArgs,
            handler=lookup,
        )
    )
    result = run_tool_loop(
        client,
        [{"role": "user", "content": "Look up test and answer."}],
        registry,
        ModelRole.EXTRACTION_AGENT,
        response_format=Answer,
        max_steps=3,
    )
    assert calls == ["test"]
    assert result.finished_reason == "structured_output"
    assert result.structured_output == Answer(answer="done")
    assert len(requests) == 2
    if model in OPENAI_MODELS:
        assert all(url.endswith("/responses") for url, _ in requests)
        items = requests[1][1]["input"]
        assert any(item.get("type") == "function_call_output" for item in items)
        assert any(
            item.get("encrypted_content") == "encrypted-reasoning" for item in items
        )
    else:
        assistant = next(
            m for m in requests[1][1]["messages"] if m["role"] == "assistant"
        )
        assert {
            "type": "thinking",
            "thinking": "",
            "signature": "signed-thinking",
        } in assistant["content"]
        assert {"type": "text", "text": "Looking up the key."} in assistant["content"]


def _chat_response(model, text):
    return {
        "id": "chatcmpl_1",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }


@pytest.mark.parametrize("model", sorted(OPENAI_MODELS))
def test_openai_text_and_custom_endpoints_keep_chat_route(model, transport):
    requests, responses = transport
    responses.extend([_chat_response(model, "done"), _chat_response(model, "custom")])
    client = LiteLLMClient(LiteLLMConfig(model=f"openai/{model}"))
    assert (
        client.generate_chat_response([{"role": "user", "content": "hello"}]) == "done"
    )
    result = client.generate_chat_response(
        [{"role": "user", "content": "hello"}],
        api_base="https://custom.example/v1",
        tools=[
            Tool(
                name="lookup", args_model=LookupArgs, handler=lambda *_: {}
            ).openai_spec()
        ],
    )
    assert isinstance(result, ToolCallingChatResponse)
    assert result.content == "custom"
    assert requests[0][0] == "https://api.openai.com/v1/chat/completions"
    assert requests[1][0] == "https://custom.example/v1/chat/completions"


def test_real_adapter_rebuilds_cross_provider_fallback(transport):
    requests, responses = transport
    responses.extend(
        [
            (
                503,
                {
                    "type": "error",
                    "error": {
                        "type": "overloaded_error",
                        "message": "temporary outage",
                    },
                },
            ),
            _provider_response("gpt-6.1-sol"),
        ]
    )
    client = LiteLLMClient(
        LiteLLMConfig(
            model="anthropic/claude-sonnet-5-5", fallback_models=["openai/gpt-6.1-sol"]
        )
    )
    result = client.generate_chat_response(
        [{"role": "user", "content": "hello"}],
        response_format=Answer,
        tools=[
            Tool(
                name="lookup", args_model=LookupArgs, handler=lambda *_: {}
            ).openai_spec()
        ],
    )
    assert isinstance(result, ToolCallingChatResponse)
    assert result.parsed_output == Answer(answer="done")
    assert len(requests) == 2
    assert requests[0][0].endswith("/messages")
    assert requests[0][1]["output_config"]["format"]["type"] == "json_schema"
    assert requests[1][0].endswith("/responses")
    assert requests[1][1]["text"]["format"]["type"] == "json_schema"
    assert "output_config" not in requests[1][1]
    assert all("temperature" not in payload for _, payload in requests)


def test_real_adapter_corrective_retry_keeps_schema_and_sampling_policy(transport):
    requests, responses = transport
    responses.extend(
        [
            _provider_response("claude-opus-5-5", text='{"wrong":"value"}'),
            _provider_response("claude-opus-5-5"),
        ]
    )
    client = LiteLLMClient(LiteLLMConfig(model="claude-opus-5-5"))
    assert client.generate_chat_response(
        [{"role": "user", "content": "hello"}], response_format=Answer
    ) == Answer(answer="done")
    assert len(requests) == 2
    assert requests[0][1]["output_config"] == requests[1][1]["output_config"]
    assert len(requests[1][1]["messages"]) > len(requests[0][1]["messages"])
    assert all("temperature" not in payload for _, payload in requests)


@pytest.mark.parametrize("snapshot", [False, True])
def test_real_adapter_refusal_stops_without_retry(snapshot, transport, monkeypatch):
    requests, responses = transport
    refusal = _provider_response("claude-opus-5-5", text="I cannot help with that.")
    refusal["stop_reason"] = "refusal"
    responses.append(refusal)
    client = LiteLLMClient(LiteLLMConfig(model="claude-opus-5-5"))
    if snapshot:
        completion = client._completion_with_hard_timeout

        def force_snapshot(params, timeout):
            raw = completion(params, timeout)
            raw = SimpleNamespace(
                **raw.__dict__,
                _hidden_params=raw._hidden_params,
                unpicklable=lambda: None,
            )
            return pickle.loads(pickle.dumps(_picklable_completion_result(raw)))  # noqa: S301 — trusted fixture

        monkeypatch.setattr(client, "_completion_with_hard_timeout", force_snapshot)
    with pytest.raises(StructuredOutputRepairError) as caught:
        client.generate_chat_response(
            [{"role": "user", "content": "hello"}],
            response_format=Answer,
            structured_output_validator=lambda _: [],
        )
    assert caught.value.failure_kind == "refusal"
    assert len(requests) == 1


@pytest.mark.parametrize("model", sorted(ANTHROPIC_MODELS))
@pytest.mark.parametrize("snapshot", [False, True])
def test_interleaved_signed_thinking_keeps_exact_provider_block_order(
    model, snapshot, transport, monkeypatch
):
    requests, responses = transport
    first = _provider_response(model, tool_turn=True)
    blocks = [
        {"type": "thinking", "thinking": "", "signature": "signed-1"},
        {"type": "text", "text": "First lookup."},
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "lookup",
            "input": {"key": "first"},
        },
        {"type": "thinking", "thinking": "", "signature": "signed-2"},
        {
            "type": "tool_use",
            "id": "toolu_2",
            "name": "lookup",
            "input": {"key": "second"},
        },
    ]
    first["content"] = blocks
    responses.extend([first, _provider_response(model)])
    client = LiteLLMClient(LiteLLMConfig(model=model))
    monkeypatch.setattr(
        "reflexio.server.llm.tools.resolve_model_name", lambda **_: model
    )
    monkeypatch.setattr(
        "reflexio.server.llm._litellm_text_generation.resolve_model_name",
        lambda *_a, **_k: model,
    )
    if snapshot:
        completion = client._completion_with_hard_timeout

        def force_snapshot(params, timeout):
            raw = completion(params, timeout)
            raw = SimpleNamespace(
                **raw.__dict__,
                _hidden_params=raw._hidden_params,
                unpicklable=lambda: None,
            )
            return pickle.loads(pickle.dumps(_picklable_completion_result(raw)))  # noqa: S301 — trusted fixture

        monkeypatch.setattr(client, "_completion_with_hard_timeout", force_snapshot)
    calls = []

    def lookup(args: BaseModel, _ctx):
        assert isinstance(args, LookupArgs)
        calls.append(args.key)
        return {"value": args.key}

    registry = ToolRegistry()
    registry.register(Tool(name="lookup", args_model=LookupArgs, handler=lookup))
    result = run_tool_loop(
        client,
        [{"role": "user", "content": "Look up first and second."}],
        registry,
        ModelRole.EXTRACTION_AGENT,
        response_format=Answer,
        max_steps=3,
    )
    assert result.structured_output == Answer(answer="done")
    assert calls == ["first", "second"]
    assert len(requests) == 2
    assistant = next(m for m in requests[1][1]["messages"] if m["role"] == "assistant")
    assert assistant["content"] == blocks


@pytest.mark.parametrize("model", sorted(SECONDARY_MODELS))
def test_secondary_capabilities_without_downloaded_catalog(model, monkeypatch):
    monkeypatch.delitem(litellm.model_cost, model, raising=False)
    ensure_model_capabilities(model)
    assert litellm.supports_function_calling(model=model)
    assert litellm.supports_reasoning(model=model)
    assert not LiteLLMClient._supports_response_schema(model)
    entry = litellm.model_cost[model]
    assert "input_cost_per_token" not in entry
    ensure_model_capabilities(model)
    assert litellm.model_cost[model] is entry


@pytest.mark.parametrize("model", sorted(SECONDARY_MODELS))
@pytest.mark.parametrize("extra_body", [False, True])
@pytest.mark.parametrize(
    "knob",
    [
        {"thinking": {"type": "disabled"}},
        {"reasoning_effort": "none"},
        {"reasoning_effort": {"effort": "low"}},
    ],
)
def test_secondary_disabled_thinking_fails_before_transport(model, extra_body, knob):
    client = LiteLLMClient(LiteLLMConfig(model=model))
    with patch("litellm.completion") as completion, pytest.raises(LiteLLMClientError):
        client.generate_chat_response(
            [{"role": "user", "content": "hello"}],
            **({"extra_body": knob} if extra_body else knob),
        )
    completion.assert_not_called()


@pytest.mark.parametrize("model", sorted(SECONDARY_MODELS))
@pytest.mark.parametrize("api_base", [None, "https://example.test/v1"])
def test_secondary_reasoning_effort_reaches_actual_adapter(model, api_base, transport):
    requests, responses = transport
    responses.append(_chat_response(model.split("/", 1)[1], '{"answer":"done"}'))
    client = LiteLLMClient(LiteLLMConfig(model=model))
    result = client.generate_chat_response(
        [{"role": "user", "content": "Answer with done."}],
        response_format=Answer,
        reasoning_effort="low",
        **({"api_base": api_base} if api_base else {}),
    )
    assert result == Answer(answer="done")
    url, body = requests[0]
    default_url = (
        "https://api.minimax.io/v1/chat/completions"
        if model.startswith("minimax/")
        else (
            "https://api.z.ai/api/paas/v4/chat/completions"
            if model == "zai/glm-5.3-flashx"
            else "https://api.z.ai/api/coding/paas/v4/chat/completions"
        )
    )
    assert url == (f"{api_base}/chat/completions" if api_base else default_url)
    assert body["model"] == model.split("/", 1)[1]
    assert body["reasoning_effort"] == "low"
    assert body.get("response_format") == {"type": "json_object"}
    assert "answer" in str(body["messages"])
    assert body["temperature"] == 0.7


@pytest.mark.parametrize(
    ("model", "knob"),
    [
        ("minimax/MiniMax-M3.1-Flash-Preview", {"reasoning_split": False}),
        ("zai/glm-5.3", {"reasoning_effort": "medium"}),
        ("zai/glm-5.3-flash", {"reasoning_effort": "medium"}),
        ("zai/glm-5.3-flashx", {"reasoning_effort": "medium"}),
    ],
)
@pytest.mark.parametrize("extra_body", [False, True])
def test_secondary_invalid_model_specific_knobs(model, knob, extra_body):
    client = LiteLLMClient(LiteLLMConfig(model=model))
    with patch("litellm.completion") as completion, pytest.raises(LiteLLMClientError):
        client.generate_chat_response(
            [{"role": "user", "content": "hello"}],
            **({"extra_body": knob} if extra_body else knob),
        )
    completion.assert_not_called()


@pytest.mark.parametrize("model", sorted(SECONDARY_MODELS))
def test_secondary_tool_loop_through_actual_subprocess(model, monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    requests = []
    name = model.split("/", 1)[1]
    first = _chat_response(name, "Looking up the key.")
    first["choices"][0]["finish_reason"] = "tool_calls"
    first["choices"][0]["message"].update(
        reasoning_content="Retain this reasoning.",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"key":"test"}'},
            }
        ],
    )
    responses = [first, _chat_response(name, '{"answer":"done"}')]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            if not responses:
                self.send_error(500, "Unexpected extra request")
                return
            payload = json.dumps(responses.pop(0)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = LiteLLMClient(LiteLLMConfig(model=model, timeout=30, fallback_models=[]))
    client._api_base = f"http://127.0.0.1:{server.server_port}/v1"
    generate = client.generate_chat_response_with_provenance

    def with_low_effort(*args, **kwargs):
        kwargs.setdefault("reasoning_effort", "low")
        return generate(*args, **kwargs)

    monkeypatch.setattr(
        client, "generate_chat_response_with_provenance", with_low_effort
    )
    monkeypatch.setattr(
        "reflexio.server.llm.tools.resolve_model_name", lambda **_: model
    )
    monkeypatch.setattr(
        "reflexio.server.llm._litellm_text_generation.resolve_model_name",
        lambda *_a, **_k: model,
    )
    monkeypatch.setattr(
        LiteLLMClient, "_should_process_isolate_completion", lambda *_: True
    )
    calls = []
    registry = ToolRegistry()

    def lookup(args, _ctx):
        calls.append(args.key)
        return {"value": "found"}

    registry.register(Tool(name="lookup", args_model=LookupArgs, handler=lookup))
    try:
        with unpatched_litellm(), monkeypatch.context() as completion_patch:
            from litellm.main import completion

            completion_patch.setattr(litellm, "completion", completion)
            result = run_tool_loop(
                client,
                [{"role": "user", "content": "Look up test and answer."}],
                registry,
                ModelRole.EXTRACTION_AGENT,
                response_format=Answer,
                max_steps=3,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert result.structured_output == Answer(answer="done")
    assert calls == ["test"]
    assert len(requests) == 2
    assert not responses
    assert all(body["reasoning_effort"] == "low" for body in requests)
    assistant = next(m for m in requests[1]["messages"] if m["role"] == "assistant")
    assert assistant["reasoning_content"] == "Retain this reasoning."
    assert assistant["content"] == "Looking up the key."
    assert assistant["tool_calls"][0]["id"] == "call_1"
