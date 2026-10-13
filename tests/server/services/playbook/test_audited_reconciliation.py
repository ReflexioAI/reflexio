"""Meaningful partition, audit, repair and physical-budget invariants."""

import hashlib
import json
from dataclasses import asdict
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
