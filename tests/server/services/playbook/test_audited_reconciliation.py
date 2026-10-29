"""Meaningful partition, audit, repair and physical-budget invariants."""

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from reflexio.server.llm._litellm_types import ModelProvenance
from reflexio.server.llm.input_budget import extraction_provider_guard
from reflexio.server.llm.litellm_client import LiteLLMClient, ProviderRequestGuardError
from reflexio.server.services.extraction.overflow_coordinator import (
    OverflowSuspendedError,
    current_overflow_coordinator,
)
from reflexio.server.services.playbook.components.reconciliation import (
    AuditedConsolidationAbstainedError,
    Decisions,
    Reconciler,
    Replace,
    project,
    validate_decisions,
)

pytestmark = pytest.mark.unit
REF = {"session_id": "request", "turn": 17}
OUTPUT = {
    "trigger": "before submitting",
    "content": "check authorization",
    "rationale": "See cited evidence.",
    "evidence_refs": [REF],
}


def payload():
    return {
        "new": [{"id": "NEW-0", **OUTPUT}],
        "existing": [{"id": "EXISTING-0", **OUTPUT}],
        "sessions": [
            {
                "id": "request",
                "messages": [
                    {"turn": 17, "role": "user", "content": "check authorization"}
                ],
            }
        ],
    }


def replacement(**updates):
    return {
        "kind": "replace",
        "new_ids": ["NEW-0"],
        "existing_ids": ["EXISTING-0"],
        "outputs": [OUTPUT],
        "reason": "same prerequisite",
        **updates,
    }


class ScriptedClient:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.requests = []

    def generate_chat_response_with_provenance(self, **kwargs):
        kwargs["provider_request_guard"]()
        self.requests.append(kwargs)
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return SimpleNamespace(value=json.dumps(output), provenance=ModelProvenance())


class CheckpointCoordinator:
    def __init__(self):
        self.checkpoint = {"completed_steps": {}}

    def step_key(self, stage, data):
        return hashlib.sha256(
            json.dumps([stage, data], sort_keys=True).encode()
        ).hexdigest()

    def completed_step(self, key):
        return self.checkpoint["completed_steps"].get(key)

    def complete_step(self, key, output, provenance):
        self.checkpoint["completed_steps"][key] = {
            "output": output,
            "model_provenance": asdict(provenance),
        }

    def save_checkpoint(self, checkpoint):
        self.checkpoint = checkpoint


def test_multioutput_replace_is_one_unordered_action():
    batch = Decisions.model_validate(
        {
            "decisions": [
                replacement(
                    outputs=[
                        OUTPUT,
                        {**OUTPUT, "content": "report authorization result"},
                    ]
                )
            ]
        }
    )
    validate_decisions(
        batch, {"NEW-0": {}}, {"EXISTING-0": {}}, {("request", 17): object()}
    )
    proposed = project(batch, {"EXISTING-0": OUTPUT})
    assert len(proposed) == 2
    assert all(row["source_ids"] == ["NEW-0", "EXISTING-0"] for row in proposed)


@pytest.mark.parametrize(
    "actions,match",
    [
        ([], "Every NEW"),
        ([replacement(), replacement()], "Every NEW"),
        (
            [
                replacement(),
                {
                    "kind": "remove",
                    "new_ids": [],
                    "existing_ids": ["EXISTING-0"],
                    "reason": "obsolete",
                    "evidence_refs": [REF],
                },
            ],
            "EXISTING",
        ),
        (
            [
                replacement(
                    outputs=[
                        {
                            **OUTPUT,
                            "evidence_refs": [
                                {"session_id": "other-owner", "turn": 17}
                            ],
                        }
                    ]
                )
            ],
            "Evidence",
        ),
    ],
)
def test_invalid_batches_cannot_be_applied(actions, match):
    batch = Decisions.model_validate({"decisions": actions})
    with pytest.raises(ValueError, match=match):
        validate_decisions(
            batch, {"NEW-0": {}}, {"EXISTING-0": {}}, {("request", 17): object()}
        )


def test_audit_revision_requires_final_independent_audit():
    client = ScriptedClient(
        [
            {"decisions": [replacement()]},
            {
                "findings": [
                    {
                        "reason": "lost reporting",
                        "evidence_refs": [REF],
                        "playbook_ids": ["NEW-0"],
                    }
                ]
            },
            {
                "decisions": [
                    replacement(
                        outputs=[
                            {**OUTPUT, "content": "check and report authorization"}
                        ]
                    )
                ]
            },
            {"findings": []},
        ]
    )
    result = Reconciler(cast(LiteLLMClient, client), "scripted").run(
        payload(), {("request", 17): object()}
    )
    assert isinstance(result.decisions[0], Replace)
    assert result.decisions[0].outputs[0].content == "check and report authorization"
    assert len(client.requests) == 4
    audit = json.loads(client.requests[-1]["messages"][1]["content"])
    assert audit["proposed_memory"][0]["content"] == "check and report authorization"
    assert audit["sessions"][0]["messages"][0]["cited_by_proposals"] == [0]


def test_physical_budget_cannot_promote_unaudited_revision():
    client = ScriptedClient(
        [
            {"decisions": [replacement()]},
            {
                "findings": [
                    {
                        "reason": "lost clause",
                        "evidence_refs": [REF],
                        "playbook_ids": ["NEW-0"],
                    }
                ]
            },
            {"decisions": [replacement()]},
        ]
    )
    engine = Reconciler(cast(LiteLLMClient, client), "scripted", max_calls=4)
    # Simulate a provider retry consuming the fourth physical allowance.
    original = client.generate_chat_response_with_provenance

    def retried(**kwargs):
        if len(client.requests) == 2:
            kwargs["provider_request_guard"]()
        return original(**kwargs)

    client.generate_chat_response_with_provenance = retried
    with pytest.raises(ProviderRequestGuardError):
        engine.run(payload(), {("request", 17): object()})
    assert engine.calls == 4
    assert len(client.requests) == 3


def test_schema_repair_uses_same_physical_budget():
    client = ScriptedClient(
        [{"decisions": []}, {"decisions": [replacement()]}, {"findings": []}]
    )
    engine = Reconciler(cast(LiteLLMClient, client), "scripted", max_calls=3)
    engine.run(payload(), {("request", 17): object()})
    assert engine.calls == 3
    assert client.requests[1]["response_format"] is Decisions


def test_targeted_revision_cannot_reinstate_consumed_existing_by_omission():
    create = {
        "kind": "create",
        "new_ids": ["NEW-0"],
        "outputs": [OUTPUT],
        "reason": "unrelated",
    }
    client = ScriptedClient(
        [
            {"decisions": [replacement()]},
            {
                "findings": [
                    {
                        "reason": "defect",
                        "evidence_refs": [REF],
                        "playbook_ids": ["NEW-0"],
                    }
                ]
            },
            {"decisions": [create]},
            {"decisions": [create]},
        ]
    )
    with pytest.raises(ValueError, match="previously changed"):
        Reconciler(cast(LiteLLMClient, client), "scripted").run(
            payload(), {("request", 17): object()}
        )


def test_resume_reuses_validated_steps_without_resetting_physical_allowance():
    coordinator = CheckpointCoordinator()
    token = current_overflow_coordinator.set(cast(Any, coordinator))
    try:
        first = ScriptedClient([{"decisions": [replacement()]}, RuntimeError("busy")])
        with pytest.raises(RuntimeError, match="busy"):
            Reconciler(cast(LiteLLMClient, first), "scripted", max_calls=3).run(
                payload(), {("request", 17): object()}
            )
        assert coordinator.checkpoint["audited_consolidation_budget"]["calls"] == 2
        second = ScriptedClient([{"findings": []}])
        resumed = Reconciler(cast(LiteLLMClient, second), "scripted", max_calls=3)
        resumed.run(payload(), {("request", 17): object()})
        assert resumed.calls == 3 and len(second.requests) == 1
        complete = ScriptedClient([])
        Reconciler(cast(LiteLLMClient, complete), "scripted", max_calls=3).run(
            payload(), {("request", 17): object()}
        )
        assert complete.requests == []
        changed = payload()
        changed["new"][0]["content"] = "changed advice"
        with pytest.raises(AuditedConsolidationAbstainedError, match="binding changed"):
            Reconciler(cast(LiteLLMClient, complete), "scripted", max_calls=3).run(
                changed, {("request", 17): object()}
            )
    finally:
        current_overflow_coordinator.reset(token)


def test_suspension_before_dispatch_does_not_consume_method_allowance():
    coordinator = CheckpointCoordinator()
    admitted = []

    def outer_guard(*_args):
        admitted.append("checked")
        if len(admitted) == 1:
            raise OverflowSuspendedError("continue on next worker")

    coordinator_token = current_overflow_coordinator.set(cast(Any, coordinator))
    guard_token = extraction_provider_guard.set(outer_guard)
    try:
        client = ScriptedClient([{"decisions": [replacement()]}, {"findings": []}])
        first = Reconciler(cast(LiteLLMClient, client), "scripted", max_calls=2)
        with pytest.raises(OverflowSuspendedError):
            first.run(payload(), {("request", 17): object()})
        assert first.calls == 0 and client.requests == []
        assert extraction_provider_guard.get() is outer_guard
        resumed = Reconciler(cast(LiteLLMClient, client), "scripted", max_calls=2)
        resumed.run(payload(), {("request", 17): object()})
        assert resumed.calls == 2 and len(admitted) == 3
        assert extraction_provider_guard.get() is outer_guard
    finally:
        extraction_provider_guard.reset(guard_token)
        current_overflow_coordinator.reset(coordinator_token)


def _tool_turn(arguments, name="search_playbooks", *, count=1):
    from reflexio.server.llm._litellm_types import ToolCallingChatResponse

    return ToolCallingChatResponse(
        content=None,
        tool_calls=[
            {
                "id": f"call_{index + 1}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
            for index in range(count)
        ],
        finish_reason="tool_calls",
    )


class ToolScriptedClient(ScriptedClient):
    """Returns ``ToolCallingChatResponse`` items unserialized."""

    def generate_chat_response_with_provenance(self, **kwargs):
        kwargs["provider_request_guard"]()
        self.requests.append(json.loads(json.dumps(kwargs, default=str)))
        output = next(self.outputs)
        if not isinstance(output, dict):
            return SimpleNamespace(value=output, provenance=ModelProvenance())
        return SimpleNamespace(value=json.dumps(output), provenance=ModelProvenance())


def test_tool_ceiling_allows_two_rounds_then_requires_json_and_independent_audit():
    client = ToolScriptedClient(
        [
            _tool_turn('{"query": "authorization"}', count=3),
            _tool_turn('{"query": "authorization"}', count=3),
            {"decisions": [replacement()]},
            {"findings": []},
        ]
    )
    engine = Reconciler(cast(LiteLLMClient, client), "scripted", max_calls=4)
    engine.run(payload(), {("request", 17): object()})
    assert engine.calls == 4
    final = client.requests[2]
    assert final["tools"] is None
    assert final["response_format"] == str(Decisions)
    assert len([m for m in final["messages"] if m.get("role") == "tool"]) == 6
    for request in (client.requests[0], client.requests[3]):
        instructions = request["messages"][0]["content"]
        assert "at most 3 tool calls in each response, across all tools" in instructions
        assert "at most 2 tool rounds per stage" in instructions
        assert "you may return final JSON immediately" in instructions
        assert "correcting a malformed tool call" in instructions
    assert client.requests[3]["messages"] != final["messages"]


@pytest.mark.parametrize("excess", ["calls", "rounds"])
def test_tool_instructions_do_not_relax_enforced_ceiling(excess):
    tool = _tool_turn('{"query": "authorization"}')
    if excess == "calls":
        outputs = [_tool_turn('{"query": "authorization"}', count=4)]
    else:
        outputs = [tool, tool, tool]
    client = ToolScriptedClient(outputs)
    engine = Reconciler(cast(LiteLLMClient, client), "scripted")
    with pytest.raises(AuditedConsolidationAbstainedError, match="allowance exceeded"):
        engine.run(payload(), {("request", 17): object()})
    assert len(client.requests) == len(outputs)


def test_tool_rounds_and_physical_budget_survive_checkpoint_resume():
    coordinator = CheckpointCoordinator()
    admissions = []

    def suspend_second(*_args):
        admissions.append(True)
        if len(admissions) == 2:
            raise OverflowSuspendedError("yield before provider dispatch")

    coordinator_token = current_overflow_coordinator.set(cast(Any, coordinator))
    guard_token = extraction_provider_guard.set(suspend_second)
    try:
        first = ToolScriptedClient([_tool_turn('{"query": "authorization"}')])
        with pytest.raises(OverflowSuspendedError):
            Reconciler(cast(LiteLLMClient, first), "scripted", max_calls=4).run(
                payload(), {("request", 17): object()}
            )
        assert coordinator.checkpoint["audited_conversation"]["turn"] == 1
        assert coordinator.checkpoint["audited_consolidation_budget"]["calls"] == 1
        resumed_client = ToolScriptedClient(
            [
                _tool_turn('{"query": "authorization"}'),
                {"decisions": [replacement()]},
                {"findings": []},
            ]
        )
        resumed = Reconciler(
            cast(LiteLLMClient, resumed_client), "scripted", max_calls=4
        )
        resumed.run(payload(), {("request", 17): object()})
        assert resumed.calls == 4
        assert resumed_client.requests[1]["tools"] is None
        assert (
            len(
                [
                    m
                    for m in resumed_client.requests[1]["messages"]
                    if m.get("role") == "tool"
                ]
            )
            == 2
        )
    finally:
        extraction_provider_guard.reset(guard_token)
        current_overflow_coordinator.reset(coordinator_token)


@pytest.mark.parametrize(
    ("arguments", "name", "error"),
    [
        ("null", "search_playbooks", "JSON object"),
        ("[]", "search_playbooks", "JSON object"),
        ("{}", "search_playbooks", "'query'"),
        ("not json", "search_playbooks", "JSON object"),
        ('{"query": 7}', "search_playbooks", "'query'"),
        ('{"query": "  "}', "search_playbooks", "'query'"),
        ('{"query": "auth", "limit": "3"}', "search_playbooks", "'limit'"),
        ('{"query": "auth", "limit": 11}', "search_playbooks", "'limit'"),
        ('{"query": "auth", "extra": 1}', "search_playbooks", "Only"),
        ('{"query": "auth"}', "search_everything", "Unknown tool"),
    ],
)
def test_malformed_tool_arguments_are_returned_to_the_model(arguments, name, error):
    """A malformed read-only call is the model's mistake to correct, not a crash.

    Before this, ``set(None)`` / ``args["query"]`` raised TypeError/KeyError,
    which the abstention handler masked as a semantic abstention and the
    durable worker retried forever.
    """
    client = ToolScriptedClient(
        [
            _tool_turn(arguments, name),
            {"decisions": [replacement()]},
            {"findings": []},
        ]
    )
    engine = Reconciler(cast(LiteLLMClient, client), "scripted")
    batch = engine.run(payload(), {("request", 17): object()})

    assert [d.kind for d in batch.decisions] == ["replace"]
    tool_messages = [
        m for m in client.requests[1]["messages"] if m.get("role") == "tool"
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert error in json.loads(tool_messages[0]["content"])["error"]
    assert engine.calls == 3


@pytest.mark.parametrize("name", [["search_playbooks"], {"name": "x"}, None, 7])
def test_non_string_tool_name_is_returned_to_the_model(name):
    """An unhashable model-authored tool name must not raise TypeError from
    the collection lookup; it is a correctable unknown-tool error."""
    client = ToolScriptedClient(
        [
            _tool_turn('{"query": "auth"}', name),
            {"decisions": [replacement()]},
            {"findings": []},
        ]
    )
    engine = Reconciler(cast(LiteLLMClient, client), "scripted")
    batch = engine.run(payload(), {("request", 17): object()})

    assert [d.kind for d in batch.decisions] == ["replace"]
    tool = next(m for m in client.requests[1]["messages"] if m.get("role") == "tool")
    assert "Unknown tool" in json.loads(tool["content"])["error"]


def test_well_formed_tool_call_still_searches_the_collection():
    client = ToolScriptedClient(
        [
            _tool_turn('{"query": "authorization", "limit": 1}'),
            {"decisions": [replacement()]},
            {"findings": []},
        ]
    )
    engine = Reconciler(cast(LiteLLMClient, client), "scripted")
    engine.run(payload(), {("request", 17): object()})
    tool = next(m for m in client.requests[1]["messages"] if m.get("role") == "tool")
    rows = json.loads(tool["content"])
    assert len(rows) == 1 and rows[0]["content"] == "check authorization"


@pytest.mark.integration
def test_process_returned_tool_call_reaches_search_and_independent_audit(tmp_path):
    """A fresh parent must not depend on a provider model's lazy serializer.

    Warming the tool-call class in this pytest process hides the defect, so
    both the provider child and reconciliation parent run in fresh interpreters.
    Only the provider response is synthetic; the tool history, search and audit
    use the ordinary reconciliation path.
    """
    probe = tmp_path / "tool_call_transport.py"
    probe.write_text(
        textwrap.dedent("""\
        import pickle
        import subprocess
        import sys

        if len(sys.argv) > 1:
            from litellm.types.utils import ModelResponse

            response = ModelResponse(choices=[{
                "message": {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "transported-call", "type": "function",
                    "function": {"name": "search_playbooks", "arguments":
                        '{"query": "authorization", "limit": 1}'},
                }]},
                "finish_reason": "tool_calls",
            }])
            sys.stdout.buffer.write(pickle.dumps(response))
        else:
            child = subprocess.run(
                [sys.executable, __file__, "provider"], capture_output=True,
                check=True, timeout=30,
            )
            response = pickle.loads(child.stdout)
            call = response.choices[0].message.tool_calls[0]

            from reflexio.server.llm._litellm_types import ToolCallingChatResponse
            from reflexio.server.services.playbook.components.reconciliation import Reconciler
            from tests.server.services.playbook.test_audited_reconciliation import (
                ToolScriptedClient, payload, replacement,
            )

            client = ToolScriptedClient([
                ToolCallingChatResponse(content=None, tool_calls=[call], finish_reason="tool_calls"),
                {"decisions": [replacement()]}, {"findings": []},
            ])
            engine = Reconciler(client, "scripted")
            batch = engine.run(payload(), {("request", 17): object()})
            assert [d.kind for d in batch.decisions] == ["replace"]
            history = client.requests[1]["messages"]
            assistant = next(m for m in history if m.get("tool_calls"))
            assert assistant["tool_calls"] == [{
                "id": "transported-call", "type": "function",
                "function": {"name": "search_playbooks", "arguments":
                    '{"query": "authorization", "limit": 1}'},
            }]
            import json
            tool = next(m for m in history if m.get("role") == "tool")
            assert tool["tool_call_id"] == "transported-call"
            assert json.loads(tool["content"])[0]["content"] == "check authorization"
            assert client.requests[2]["messages"] != history  # independent full audit
            assert engine.calls == 3
        """),
        encoding="utf-8",
    )
    result = subprocess.run(  # noqa: S603 -- owned synthetic script, no provider access.
        [sys.executable, str(probe)],
        cwd=Path(__file__).resolve().parents[4],
        # The temporary script imports test helpers, which are absent from wheels.
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[4])},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("error", [TypeError("bug"), KeyError("bug")])
def test_unexpected_errors_are_not_masked_as_abstentions(error):
    """Only deterministic rejections abstain; a bug keeps its own type."""
    from reflexio.server.services.playbook.components.reconciliation import (
        _run_audited,
    )

    client = ScriptedClient([error])
    engine = Reconciler(cast(LiteLLMClient, client), "scripted")
    with pytest.raises(type(error)) as raised:
        _run_audited(engine, payload(), {("request", 17): object()})
    assert not isinstance(raised.value, AuditedConsolidationAbstainedError)


def test_deterministic_rejections_still_abstain():
    from reflexio.server.llm.litellm_client import StructuredOutputRepairError
    from reflexio.server.services.playbook.components.reconciliation import (
        _run_audited,
    )

    repair = StructuredOutputRepairError(
        "exhausted", failure_kind="semantic", model="scripted"
    )
    for outputs in (
        [{"decisions": []}, {"decisions": []}],  # repaired once, still invalid
        [repair],
    ):
        engine = Reconciler(cast(LiteLLMClient, ScriptedClient(outputs)), "scripted")
        with pytest.raises(AuditedConsolidationAbstainedError) as raised:
            _run_audited(engine, payload(), {("request", 17): object()})
        assert isinstance(raised.value.__cause__, ValueError | type(repair))


def test_execution_and_durable_binding_use_one_frozen_prompt_bundle(monkeypatch):
    from copy import deepcopy
    from pathlib import Path

    from reflexio.server.services.playbook.components import reconciliation

    supplied = reconciliation.load_reconciliation_prompts(
        Path(reconciliation.__file__).with_name("reconciliation_prompts")
    )
    original = dict(supplied)
    coordinator = CheckpointCoordinator()
    token = current_overflow_coordinator.set(cast(Any, coordinator))
    try:
        client = ScriptedClient([{"decisions": [replacement()]}, {"findings": []}])
        engine = Reconciler(cast(LiteLLMClient, client), "scripted", prompts=supplied)
        supplied["reconcile-method"] = "Changed after admission"

        def unexpected_reload(*_args, **_kwargs):
            raise AssertionError("A running method must not reread its prompt assets")

        monkeypatch.setattr(Path, "read_text", unexpected_reload)
        engine.run(payload(), {("request", 17): object()})
        assert client.requests[0]["messages"][0]["content"].startswith(
            original["reconcile-method"]
        )
        assert client.requests[1]["messages"][0]["content"].startswith(
            original["reconcile-audit"]
        )
        saved = deepcopy(coordinator.checkpoint)
        changed = ScriptedClient([])
        with pytest.raises(AuditedConsolidationAbstainedError, match="binding changed"):
            Reconciler(cast(LiteLLMClient, changed), "scripted", prompts=supplied).run(
                payload(), {("request", 17): object()}
            )
        assert changed.requests == [] and coordinator.checkpoint == saved
    finally:
        current_overflow_coordinator.reset(token)
