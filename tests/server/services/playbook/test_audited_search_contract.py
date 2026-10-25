"""Advertised search limits must agree with native fail-closed enforcement."""

import json
from types import SimpleNamespace
from typing import cast

import pytest

from reflexio.server.llm._litellm_types import ModelProvenance
from reflexio.server.llm.litellm_client import LiteLLMClient, ToolCallingChatResponse
from reflexio.server.services.playbook.components.reconciliation import (
    Audit,
    AuditedConsolidationAbstainedError,
    Reconciler,
)

pytestmark = pytest.mark.unit
COLLECTIONS = {"search_playbooks": [], "search_traces": [], "search_profiles": []}


def tools(count):
    return ToolCallingChatResponse(
        content=None,
        tool_calls=[
            {
                "id": f"search-{number}",
                "type": "function",
                "function": {
                    "name": "search_playbooks",
                    "arguments": json.dumps({"query": "authorization"}),
                },
            }
            for number in range(count)
        ],
        finish_reason="tool_calls",
    )


class ScriptedClient:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.requests = []

    def generate_chat_response_with_provenance(self, **kwargs):
        kwargs["provider_request_guard"]()
        self.requests.append(kwargs)
        return SimpleNamespace(value=next(self.outputs), provenance=ModelProvenance())


def ask(engine, stage="reconcile-audit"):
    return engine.ask(stage, {}, Audit, lambda _: None, COLLECTIONS)


@pytest.mark.parametrize(
    "stage", ["reconcile-method", "reconcile-audit", "reconcile-revise"]
)
def test_all_stages_advertise_the_enforced_tool_contract(stage):
    client = ScriptedClient([tools(3), Audit(findings=[])])
    engine = Reconciler(cast(LiteLLMClient, client), "scripted")
    assert ask(engine, stage).findings == []
    prompt = client.requests[0]["messages"][0]["content"]
    assert "at most 3 tool calls in one response" in prompt
    assert "at most 2 tool-use rounds" in prompt
    assert "do not invent facts" in prompt
    assert engine.calls == 2
    assert sum(m["role"] == "tool" for m in client.requests[1]["messages"]) == 3


def test_four_tool_calls_still_abstain_without_partial_search_execution():
    client = ScriptedClient([tools(4)])
    engine = Reconciler(cast(LiteLLMClient, client), "scripted")
    with pytest.raises(
        AuditedConsolidationAbstainedError, match="tool allowance exceeded"
    ):
        ask(engine)
    assert engine.calls == 1 and len(client.requests) == 1
    assert not any(m["role"] == "tool" for m in client.requests[0]["messages"])


def test_two_tool_rounds_require_a_final_structured_audit_without_more_tools():
    client = ScriptedClient([tools(3), tools(3), Audit(findings=[])])
    engine = Reconciler(cast(LiteLLMClient, client), "scripted")
    assert ask(engine).findings == []
    assert engine.calls == 3
    assert client.requests[2]["tools"] is None
    assert client.requests[2]["response_format"] is Audit


def test_third_tool_round_cannot_bypass_the_advertised_limit():
    client = ScriptedClient([tools(1), tools(1), tools(1)])
    engine = Reconciler(cast(LiteLLMClient, client), "scripted")
    with pytest.raises(
        AuditedConsolidationAbstainedError, match="tool allowance exceeded"
    ):
        ask(engine)
    assert engine.calls == 3


def test_physical_allowance_can_still_stop_before_a_final_audit():
    client = ScriptedClient([tools(1), tools(1), Audit(findings=[])])
    engine = Reconciler(cast(LiteLLMClient, client), "scripted", max_calls=2)
    with pytest.raises(
        AuditedConsolidationAbstainedError, match="physical-request allowance"
    ):
        ask(engine)
    assert engine.calls == 2 and len(client.requests) == 2
