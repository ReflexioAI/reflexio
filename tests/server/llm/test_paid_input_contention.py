"""Only contention without an observed model response is a free input yield."""

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from reflexio.server.llm import tools
from reflexio.server.llm._litellm_types import CompletionResult, ModelProvenance
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.services.storage.error import (
    ExecutionInputBusyError,
    execution_input_is_busy,
)

pytestmark = pytest.mark.unit


class _Args(BaseModel):
    value: str


class _Batch(BaseModel):
    items: list[_Args]


class _Call(BaseModel):
    tool: str
    value: str


class _Plan(BaseModel):
    reasoning: str
    next_call: _Call


@pytest.mark.parametrize(
    "protocol,stage",
    [
        ("native", "tool"),
        ("batch", "tool"),
        ("multi", "tool"),
        ("native", "later_admission"),
        ("multi", "later_admission"),
        ("native", "first_admission"),
        ("batch", "first_admission"),
        ("multi", "first_admission"),
    ],
)
def test_paid_response_classification_covers_all_tool_protocols(
    monkeypatch, protocol, stage
):
    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini"))
    monkeypatch.setattr(
        tools, "supports_tool_calling", lambda _model: protocol == "native"
    )
    calls = []
    responses = []

    def generate(**_kwargs):
        calls.append(1)
        if stage == "first_admission" or (
            stage == "later_admission" and len(calls) == 2
        ):
            raise ExecutionInputBusyError("Synthetic pre-provider input admission")
        if protocol == "native":
            value = SimpleNamespace(
                content=None,
                tool_calls=[
                    {
                        "id": "synthetic",
                        "type": "function",
                        "function": {
                            "name": "emit",
                            "arguments": json.dumps({"value": "synthetic"}),
                        },
                    }
                ],
            )
        elif protocol == "batch":
            value = _Batch(items=[_Args(value="synthetic")])
        else:
            value = _Plan(
                reasoning="Synthetic turn",
                next_call=_Call(tool="emit", value="synthetic"),
            )
        responses.append(1)
        return CompletionResult(value=value, provenance=ModelProvenance())

    def emit(_args, _ctx):
        if stage == "tool":
            raise ExecutionInputBusyError("Synthetic tool input lock")
        return {"ok": True}

    monkeypatch.setattr(client, "generate_chat_response_with_provenance", generate)
    expected = (
        ExecutionInputBusyError
        if stage == "first_admission"
        else tools.ModelTurnInputBusyError
    )
    with pytest.raises(expected) as raised:
        tools.run_tool_loop(
            client=client,
            messages=[],
            model_role=ModelRole.EXTRACTION_AGENT,
            registry=tools.ToolRegistry(
                [tools.Tool(name="emit", args_model=_Args, handler=emit)]
            ),
            max_steps=3,
            fallback_schema=_Batch,
            fallback_tool_name="emit",
            multi_stage_schema=_Plan if protocol == "multi" else None,
        )
    assert len(responses) == (0 if stage == "first_admission" else 1)
    assert execution_input_is_busy(raised.value) is (stage == "first_admission")
    if stage != "first_admission":
        assert raised.value.__cause__ is None
