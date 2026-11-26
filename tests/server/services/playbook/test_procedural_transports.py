"""Actual single-tool parsers and network adapters preserve complete V8 batches."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import litellm
import pytest

from reflexio.server.llm._litellm_types import (
    CompletionResult,
    LiteLLMConfig,
    ModelProvenance,
    ToolCallingChatResponse,
)
from reflexio.server.llm._model_compat import ANTHROPIC_MODELS
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.providers.cli_tool_protocol import (
    parse_tool_use,
    render_tools_instruction,
)
from reflexio.server.llm.providers.native_cli_provider import (
    _build_model_response_with_tool_call,
)
from reflexio.server.llm.providers.openclaw_provider import _tool_call_response
from reflexio.server.services.playbook.procedural import codec
from reflexio.server.services.playbook.procedural.models import (
    ContractError,
    Document,
    Input,
    Observation,
    Page,
    Source,
)
from reflexio.server.services.playbook.procedural.transport import (
    BATCH_TOOL,
    IncompleteBatchError,
    StageClient,
)
from reflexio.server.services.playbook.procedural.workflow import run_pipeline
from reflexio.test_support.llm_mock import inline_test_completion, unpatched_litellm

WITNESS = "EXCERPT_1_1_1"


def _member(
    name: str, arguments: dict[str, Any], identity: str = "member"
) -> dict[str, Any]:
    return {"id": identity, "name": name, "arguments": arguments}


def _finish(status: str = "complete") -> dict[str, Any]:
    return _member(
        "finish_batch", {"status": status, "reason": "Complete batch."}, "done"
    )


def _observation() -> dict[str, Any]:
    return _member(
        "record_observation",
        {
            "topic": "Release process",
            "evidence": WITNESS,
            "observation": "Read original source.",
        },
    )


def _entry(
    name: str = "deploy-service", *, target: str | None = None
) -> dict[str, Any]:
    args = {
        "group": "coupled",
        "name": name,
        "trigger": "Preparing a release",
        "entry": "Use staging before production.",
        "evidence": WITNESS,
    }
    if target:
        args["target"] = target
    return _member("revise_procedure" if target else "create_procedure", args)


def _verdict(unit: str, *, state: str = "pass") -> dict[str, Any]:
    return _member(
        "review_result",
        {
            "unit": unit,
            "decision": "accept",
            "result_support": "pass",
            "scope_conditions": state,
            "useful_nonduplicate": "pass",
            "preservation_and_correction": "pass",
            "resource_consistency": "pass",
            "reason": "Original source supports complete result.",
        },
        unit,
    )


def _input(existing: list[dict[str, Any]] | None = None) -> Input:
    return Input(
        owner="owner",
        sources=[
            Source(
                interaction_id=2,
                request_id="request",
                owner="owner",
                role="User",
                request_source="app",
                text="Use staging before production.",
            )
        ],
        existing=existing or [],
        agent_context="Release assistant",
        definition="Learn recurring release procedures",
        tool_context="",
    )


def _cli_client(
    model: str, batches: list[list[dict[str, Any]]]
) -> tuple[StageClient, MagicMock]:
    """Script terminal CLI text, then use the unchanged real parser/response adapter."""
    native = MagicMock(spec=LiteLLMClient)
    native.config = LiteLLMConfig(model=model, timeout=120, fallback_models=[])
    native.get_completion_model_ladder.side_effect = lambda **kwargs: (kwargs["model"],)
    native.get_completion_request_contract.side_effect = lambda **kwargs: {
        "model": kwargs["model"],
        "custom_llm_provider": None,
    }
    pending = iter(batches)

    def complete(**kwargs: Any) -> CompletionResult:
        kwargs["provider_request_guard"]({}, 240, (kwargs["model"],), False)
        assert len(kwargs["tools"]) == 1
        assert kwargs["tools"][0]["function"]["name"] == BATCH_TOOL
        assert BATCH_TOOL in render_tools_instruction(kwargs["tools"])
        text = json.dumps({"tool": BATCH_TOOL, "args": {"calls": next(pending)}})
        parsed = parse_tool_use(text, {BATCH_TOOL})
        assert parsed is not None
        response = (
            _tool_call_response(model, parsed)
            if model.startswith("openclaw/")
            else _build_model_response_with_tool_call(
                model=model, terminal_text=text, elapsed_seconds=0, tool_use=parsed
            )
        )
        message = response.choices[0].message
        assert message.tool_calls is not None
        assert len(message.tool_calls) == 1
        return CompletionResult(
            value=ToolCallingChatResponse(
                content=message.content,
                tool_calls=message.tool_calls,
                finish_reason=response.choices[0].finish_reason,
            ),
            provenance=ModelProvenance(
                model_name=model, provider=model.partition("/")[0]
            ),
            usage=getattr(response, "usage", None),
        )

    native.generate_chat_response_with_provenance.side_effect = complete
    return StageClient(native, model), native


def _source(client: StageClient) -> list[Observation]:
    return client.ask(
        "source",
        "Read original source",
        {},
        {"record_observation": Observation},
        lambda batch: [value for _, value in batch],
        lambda _: None,
        evidence_choices=[WITNESS],
    )


@pytest.mark.parametrize("model", ["local-cli/codex/test-model", "openclaw/test-model"])
def test_single_cli_turn_preserves_entry_and_resource_through_complete_workflow(
    model: str,
) -> None:
    resource = _member(
        "attach_resource",
        {
            "group": "coupled",
            "document_name": "deploy-service",
            "path": "references/check.md",
            "content": "Use staging before production.",
            "evidence": WITNESS,
        },
        "resource",
    )
    client, native = _cli_client(
        model,
        [
            [_observation(), _finish()],
            [_entry(), resource, _finish()],
            [_verdict("U0"), _finish()],
        ],
    )
    receipt = run_pipeline(_input(), {2: ["Use staging before production."]}, client)
    assert receipt["outcome"] == "accept"
    document = codec.decode(receipt["rendered"]["W0"]["content"])
    assert document is not None
    assert [page.path for page in document.pages] == ["SKILL.md", "references/check.md"]
    assert native.generate_chat_response_with_provenance.call_count == 3
    assert {
        call.kwargs["model"]
        for call in native.generate_chat_response_with_provenance.call_args_list
    } == {model}


@pytest.mark.parametrize("state,accepted", [("pass", True), ("fail", False)])
def test_cli_coupled_revision_and_retirement_are_reduced_together(
    state: str, accepted: bool
) -> None:
    document = Document(
        name="deploy-service",
        pages=[Page(path="SKILL.md", content="Deploy after approval.")],
    )
    existing = [
        {
            "user_playbook_id": 31,
            "user_id": "owner",
            "content": codec.encode(document),
            "trigger": "Preparing a release",
            "source_interaction_ids": [1],
            "source_span": "Old instruction.",
        },
        {
            "user_playbook_id": 32,
            "user_id": "owner",
            "content": "Old duplicate.",
            "trigger": "Old release",
            "source_interaction_ids": [1],
            "source_span": "Old duplicate.",
        },
    ]
    retirement = _member(
        "retire_procedure",
        {"group": "coupled", "target": "P2", "evidence": WITNESS},
        "retire",
    )
    client, _ = _cli_client(
        "local-cli/codex/test-model",
        [
            [_observation(), _finish()],
            [_entry(target="P1"), retirement, _finish()],
            [_verdict("U0"), _verdict("U1", state=state), _finish()],
        ],
    )
    receipt = run_pipeline(
        _input(existing), {2: ["Use staging before production."]}, client
    )
    assert receipt["retired_ids"] == ([32] if accepted else [])
    assert receipt["accepted"] == (["W0"] if accepted else [])
    assert receipt["plan"]["keep_existing"] == ([] if accepted else [31, 32])


def test_cli_terminal_incomplete_overrides_bad_prior_members_without_repair() -> None:
    client, native = _cli_client(
        "local-cli/codex/test-model",
        [[_member("foreign", {"bad": True}), _finish("incomplete")]],
    )
    with pytest.raises(IncompleteBatchError):
        _source(client)
    assert native.generate_chat_response_with_provenance.call_count == 1


@pytest.mark.parametrize(
    "fault", ["no_finish", "duplicate_ids", "foreign_function", "wrong_evidence"]
)
def test_cli_partial_or_invalid_member_batches_never_complete(fault: str) -> None:
    batch = [_observation(), _finish()]
    if fault == "no_finish":
        batch.pop()
    elif fault == "duplicate_ids":
        batch[1]["id"] = batch[0]["id"]
    elif fault == "foreign_function":
        batch[0]["name"] = "execute_shell"
    else:
        batch[0]["arguments"]["evidence"] = "EXCERPT_foreign"
    client, native = _cli_client("local-cli/codex/test-model", [batch, batch])
    with pytest.raises((ContractError, ValueError)):
        _source(client)
    assert native.generate_chat_response_with_provenance.call_count == 2
    repair = native.generate_chat_response_with_provenance.call_args.kwargs["messages"]
    assert repair[2]["tool_calls"][0]["function"]["name"] == BATCH_TOOL
    assert len([message for message in repair if message["role"] == "tool"]) == 1
    assert repair[3]["tool_call_id"] == repair[2]["tool_calls"][0]["id"]


def test_network_primary_cli_fallback_wraps_before_admission_without_retargeting() -> (
    None
):
    client, native = _cli_client(
        "local-cli/codex/test-model", [[_observation(), _finish()]]
    )
    native.config = LiteLLMConfig(
        model="minimax/MiniMax-M3", fallback_models=["local-cli/codex/test-model"]
    )
    native.get_completion_model_ladder.side_effect = lambda **kwargs: (
        kwargs["model"],
        "local-cli/codex/test-model",
    )
    mixed = StageClient(native, "minimax/MiniMax-M3")
    assert all(mixed.single_call.values())
    assert len(_source(mixed)) == 1
    request = native.generate_chat_response_with_provenance.call_args.kwargs
    assert (
        request["model"] == "minimax/MiniMax-M3"
        and request["tools"][0]["function"]["name"] == BATCH_TOOL
    )
    assert mixed.model_ladders["source"] == [
        "minimax/MiniMax-M3",
        "local-cli/codex/test-model",
    ]
    assert request["fallback_models"] == ["local-cli/codex/test-model"]


def test_explicit_http_transport_overrides_openclaw_shaped_model_label() -> None:
    _, native = _cli_client("openclaw/opaque-http-model", [])
    native.get_completion_request_contract.side_effect = lambda **kwargs: {
        "model": kwargs["model"],
        "custom_llm_provider": "openai",
    }
    client = StageClient(native, "openclaw/opaque-http-model")
    assert not any(client.single_call.values())


@pytest.fixture
def real_adapter(monkeypatch):
    """Reuse the real LiteLLM adapters while rejecting every unscripted HTTP call."""
    requests, replies = [], []
    bundled = (
        Path(litellm.__file__).parent / "model_prices_and_context_window_backup.json"
    )
    monkeypatch.setattr(litellm, "model_cost", json.loads(bundled.read_text()))
    monkeypatch.setattr(litellm, "callbacks", [])
    for name in (
        "BRAINTRUST_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "REFLEXIO_LLM_SEED",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-placeholder")
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    LiteLLMClient._supports_response_schema.cache_clear()
    litellm.in_memory_llm_clients_cache.flush_cache()

    def send(_client, request, **_kwargs):
        assert request.method == "POST"
        requests.append((str(request.url), json.loads(request.content)))
        assert replies, "Unexpected provider request"
        return httpx.Response(200, json=replies.pop(0), request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    with unpatched_litellm(), monkeypatch.context() as patch:
        from litellm.main import completion

        patch.setattr(litellm, "completion", completion)
        patch.setattr(
            LiteLLMClient, "_run_completion", staticmethod(inline_test_completion)
        )
        yield requests, replies
    assert not replies
    LiteLLMClient._supports_response_schema.cache_clear()
    litellm.in_memory_llm_clients_cache.flush_cache()


def _anthropic(
    model: str, *, invalid: bool = False, signed: bool = False, plain: bool = False
) -> dict[str, Any]:
    content = (
        [{"type": "text", "text": "No tool return."}]
        if plain
        else [
            {
                "type": "tool_use",
                "id": "toolu_return",
                "name": "record_observation" if invalid else "finish_batch",
                "input": {"topic": "Incomplete declaration"}
                if invalid
                else {"status": "complete", "reason": "No useful learning."},
            }
        ]
    )
    if signed:
        content = [
            {"type": "thinking", "thinking": "", "signature": "signed-block"},
            {"type": "text", "text": "Original interleaved explanation."},
            *content,
        ]
    return {
        "id": "msg_return",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": "end_turn" if plain else "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 10},
    }


def _responses(model: str, *, invalid: bool) -> dict[str, Any]:
    calls: list[dict[str, Any]] = [
        {
            "type": "function_call",
            "id": "fc_return",
            "call_id": "call_return",
            "name": "record_observation" if invalid else "finish_batch",
            "arguments": '{"topic":"Incomplete declaration"}'
            if invalid
            else '{"status":"complete","reason":"No useful learning."}',
            "status": "completed",
        }
    ]
    if invalid:
        calls.insert(
            0,
            {
                "type": "reasoning",
                "id": "rs_return",
                "summary": [],
                "encrypted_content": "encrypted-reasoning",
            },
        )
    return {
        "id": "resp_return",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": model,
        "output": calls,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
    }


@pytest.mark.parametrize("model", sorted(ANTHROPIC_MODELS))
def test_supported_claude_models_prepare_and_execute_with_auto_selection(
    model: str, real_adapter
) -> None:
    requests, replies = real_adapter
    replies.append(_anthropic(model))
    client = StageClient(
        LiteLLMClient(LiteLLMConfig(model=model, max_tokens=1024, fallback_models=[])),
        model,
    )
    assert _source(client) == []
    assert len(requests) == 1
    assert requests[0][1]["model"] == model
    assert requests[0][1]["tool_choice"] == {"type": "auto"}
    assert client.physical_calls == 1


@pytest.mark.parametrize("model", sorted(ANTHROPIC_MODELS))
def test_auto_selection_does_not_allow_a_plain_claude_answer_as_success(
    model: str, real_adapter
) -> None:
    requests, replies = real_adapter
    replies.append(_anthropic(model, plain=True))
    client = StageClient(
        LiteLLMClient(LiteLLMConfig(model=model, max_tokens=1024, fallback_models=[])),
        model,
    )
    with pytest.raises(ContractError, match="complete terminal"):
        _source(client)
    assert len(requests) == 1


def test_signed_interleaved_anthropic_blocks_survive_real_adapter_structural_repair(
    real_adapter,
) -> None:
    requests, replies = real_adapter
    model = "claude-opus-5-5"
    original = _anthropic(model, invalid=True, signed=True)
    replies.extend([original, _anthropic(model)])
    client = StageClient(
        LiteLLMClient(LiteLLMConfig(model=model, max_tokens=1024, fallback_models=[])),
        model,
    )
    assert _source(client) == []
    assert len(requests) == 2
    assistant = next(
        message
        for message in requests[1][1]["messages"]
        if message["role"] == "assistant"
    )
    assert assistant["content"] == original["content"]


def test_encrypted_responses_reasoning_survives_real_adapter_structural_repair(
    real_adapter,
) -> None:
    requests, replies = real_adapter
    model = "gpt-6.1-sol"
    replies.extend([_responses(model, invalid=True), _responses(model, invalid=False)])
    client = StageClient(
        LiteLLMClient(LiteLLMConfig(model=model, max_tokens=1024, fallback_models=[])),
        model,
    )
    assert _source(client) == []
    assert len(requests) == 2
    items = requests[1][1]["input"]
    assert any(item.get("encrypted_content") == "encrypted-reasoning" for item in items)
    assert any(
        item.get("type") == "function_call_output"
        and item.get("call_id") == "call_return"
        for item in items
    )


def test_cli_truncated_outer_delivery_is_terminal_without_repair() -> None:
    client, native = _cli_client(
        "local-cli/codex/test-model", [[_observation(), _finish()]]
    )
    original = native.generate_chat_response_with_provenance.side_effect

    def truncated(**kwargs: Any) -> CompletionResult:
        result = original(**kwargs)
        result.value.finish_reason = "length"
        return result

    native.generate_chat_response_with_provenance.side_effect = truncated
    with pytest.raises(ContractError, match="complete terminal"):
        _source(client)
    assert native.generate_chat_response_with_provenance.call_count == 1


def test_existing_frozen_fallback_ladder_is_not_replaced_mid_operation() -> None:
    client, native = _cli_client("local-cli/codex/test-model", [[_finish()]])
    native.config.fallback_models = ["unselected-provider/model"]
    assert _source(client) == []
    assert (
        native.generate_chat_response_with_provenance.call_args.kwargs[
            "fallback_models"
        ]
        == []
    )
