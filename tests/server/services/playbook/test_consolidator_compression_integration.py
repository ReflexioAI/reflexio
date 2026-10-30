"""Shared compression, native audit, and SQLite continuation under one owner."""

import json
import socket
import time
from copy import deepcopy
from dataclasses import replace

import litellm
import pytest

from reflexio.server.llm import LiteLLMClientError
from reflexio.server.llm.input_budget import (
    InputBudgetConfigurationError,
    InputBudgetExceededError,
    check_request_input_budget,
)
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.extraction.overflow_coordinator import (
    OverflowCoordinator,
    OverflowSuspendedError,
)
from reflexio.server.services.playbook.components.reconciliation import Reconciler
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookEvidenceSource,
)
from reflexio.server.services.storage.storage_base._extraction_stream import (
    OverflowCheckpointConflictError,
)
from tests.server.services.playbook.test_audited_reconciliation import (
    payload,
    replacement,
)
from tests.server.services.test_durable_window_pipeline import _overflow_coordinator
from tests.server.services.test_durable_window_pipeline import (
    pipeline as imported_pipeline,
)

pytestmark = pytest.mark.integration
durable_pipeline = imported_pipeline
MODEL = "minimax/MiniMax-M3"


@pytest.fixture(autouse=True)
def network_denied(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("This integration gate must not access the network")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setenv("MINIMAX_API_KEY", "scripted-only")


class ScriptedClient(LiteLLMClient):
    def __init__(self):
        super().__init__(
            LiteLLMConfig(
                model=MODEL, max_tokens=8192, max_retries=1, fallback_models=[]
            )
        )
        self.calls = []
        self.fail_consumer = False

    def _completion_with_hard_timeout(self, params, hard_timeout):
        self.calls.append(deepcopy(params))
        data = json.loads(params["messages"][1]["content"])
        if "input_packets" in data:
            ref = data["input_packets"][0]["evidence"][0]["source_id"]
            value = {"points": [{"evidence_refs": [ref]}]}
        elif self.fail_consumer:
            raise RuntimeError("scripted interruption after completed compression")
        else:
            value = (
                {"findings": []}
                if "proposed_memory" in data
                else {"decisions": [replacement()]}
            )
        return litellm.ModelResponse(
            model=MODEL,
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps(value)},
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )


def evidence(user, *, large=False):
    data = payload()
    for row in [*data["new"], *data["existing"], *data["sessions"]]:
        row["user_id"] = user
    message = data["sessions"][0]["messages"][0]
    message["request_source"] = "automated-transcript"
    if large:
        message["content"] += "\n" + "\n".join(
            f"Tool observation {n}: record value {n * n}." for n in range(7000)
        )
    return data, {
        ("request", 17): PlaybookEvidenceSource(
            interaction_id=17,
            request_id="request",
            evidence_texts=(message["content"],),
            role=message["role"],
            request_source=message["request_source"],
        )
    }


def owner(storage, window, token):
    parent = storage.get_agent_run(f"window:{window.window_id}")
    assert parent is not None
    return OverflowCoordinator(
        storage,
        window,
        token,
        parent.generation_request_snapshot["overflow"],
        source_digest="captured-source",
        # Capacity belongs to this scripted fixture, not a live provider claim.
        context_limits={MODEL: 1000000},
    )


def run(client, coordinator, data, references):
    with coordinator.provider_scope():
        actor = Reconciler(
            client,
            MODEL,
            source_scope={
                "org_id": coordinator.storage.org_id,
                "project_id": coordinator.window.project_id,
                "user_id": coordinator.window.user_id,
            },
        )
        result = actor.run(data, references)
        return actor, result


@pytest.mark.parametrize("model", ["claude-code/default", "openclaw/default"])
def test_opaque_cli_consolidation_retains_audit_and_existing_owner(model, monkeypatch):
    from reflexio.server.llm.input_budget import extraction_provider_guard
    from reflexio.server.services.playbook.components import context_compression

    def reject_native_capacity(*args, **kwargs):
        raise AssertionError("Opaque CLI adapters must not claim native capacity")

    monkeypatch.setattr(context_compression, "_budgets", reject_native_capacity)
    client = ScriptedClient()
    client.config.model = model
    data, refs = evidence("user")
    before = deepcopy(data)
    reservations = []
    token = extraction_provider_guard.set(lambda *args: reservations.append(args))
    try:
        actor = Reconciler(
            client,
            model,
            source_scope={"org_id": "org", "project_id": "project", "user_id": "user"},
        )
        result = actor.run(data, refs)
    finally:
        extraction_provider_guard.reset(token)
    assert result.decisions and data == before
    assert len(client.calls) == actor.calls == len(reservations) == 2
    assert all(call["model"] == model for call in client.calls)
    assert "proposed_memory" in json.loads(client.calls[-1]["messages"][1]["content"])
    assert actor.citation_sources == refs


def test_small_input_uses_passthrough_and_keeps_the_independent_audit(durable_pipeline):
    storage, window, token = _overflow_coordinator(durable_pipeline)
    coordinator = owner(storage, window, token)
    data, refs = evidence(window.user_id)
    before = deepcopy(data)
    client = ScriptedClient()
    actor, result = run(client, coordinator, data, refs)
    assert result.decisions and data == before
    assert len(client.calls) == actor.calls == 2
    assert "proposed_memory" in json.loads(client.calls[-1]["messages"][1]["content"])
    assert coordinator.checkpoint["reconciliation_context"]["nodes"] == {}
    assert len(coordinator.checkpoint["reconciliation_context"]["attempt_ids"]) == 2


def test_restart_reuses_committed_compression_and_retains_unknown_attempts(
    durable_pipeline,
):
    storage, window, token = _overflow_coordinator(durable_pipeline)
    first = owner(storage, window, token)
    data, refs = evidence(window.user_id, large=True)
    client = ScriptedClient()
    client.fail_consumer = True
    with pytest.raises(LiteLLMClientError):
        run(client, first, data, refs)
    calls_before = len(client.calls)
    compression_before = sum(
        "input_packets" in json.loads(call["messages"][1]["content"])
        for call in client.calls
    )
    assert compression_before > 1
    assert first.checkpoint["reconciliation_context"]["nodes"]
    assert any(attempt["usage"] is None for attempt in first._manifest["attempts"])
    resumed = owner(storage, window, token)
    client.fail_consumer = False
    actor, result = run(client, resumed, data, refs)
    assert result.decisions
    assert len(client.calls) == calls_before + 2
    assert actor.calls == len(client.calls) == len(resumed._manifest["attempts"])
    assert actor.citation_sources is not None
    selected = actor.citation_sources[("request", 17)].evidence_texts
    assert sum(map(len, selected)) < len(data["sessions"][0]["messages"][0]["content"])
    assert all(text in refs[("request", 17)].evidence_texts[0] for text in selected)


def test_completed_audit_replay_does_not_dispatch_again(durable_pipeline):
    storage, window, token = _overflow_coordinator(durable_pipeline)
    coordinator = owner(storage, window, token)
    data, refs = evidence(window.user_id)
    client = ScriptedClient()
    _, original = run(client, coordinator, data, refs)
    count = len(client.calls)
    _, repeated = run(client, owner(storage, window, token), data, refs)
    assert repeated == original
    assert len(client.calls) == count


def test_resume_rejects_changed_evidence_without_dispatch(durable_pipeline):
    storage, window, token = _overflow_coordinator(durable_pipeline)
    coordinator = owner(storage, window, token)
    data, refs = evidence(window.user_id)
    client = ScriptedClient()
    run(client, coordinator, data, refs)
    changed = deepcopy(data)
    changed["sessions"][0]["messages"][0]["content"] += " Additional correction."
    with pytest.raises(InputBudgetConfigurationError, match="contract changed"):
        run(client, owner(storage, window, token), changed, refs)
    assert len(client.calls) == 2


def test_compression_cannot_spend_the_calls_reserved_for_draft_and_audit(
    durable_pipeline,
):
    storage, window, token = _overflow_coordinator(durable_pipeline)
    coordinator = owner(storage, window, token)
    data, refs = evidence(window.user_id, large=True)
    client = ScriptedClient()
    with coordinator.provider_scope():
        actor = Reconciler(
            client,
            MODEL,
            max_calls=2,
            source_scope={
                "org_id": storage.org_id,
                "project_id": window.project_id,
                "user_id": window.user_id,
            },
        )
        with pytest.raises(
            InputBudgetConfigurationError, match="reserve draft and audit"
        ):
            actor.run(data, refs)
    assert client.calls == []
    assert coordinator.checkpoint["audited_consolidation_budget"]["calls"] == 0


def test_actual_followup_growth_is_rejected_before_reserving_an_attempt(
    durable_pipeline,
):
    storage, window, token = _overflow_coordinator(durable_pipeline)
    coordinator = owner(storage, window, token)
    data, refs = evidence(window.user_id)
    client = ScriptedClient()
    actor, _ = run(client, coordinator, data, refs)
    params = deepcopy(client.calls[-1])
    params["messages"].append(
        {
            "role": "tool",
            "tool_call_id": "oversized",
            "content": "large tool result " * 40000,
        }
    )
    with pytest.raises(InputBudgetExceededError):
        actor.guard(params, 1.0, (MODEL,), False)
    assert (
        len(client.calls) == actor.calls == len(coordinator._manifest["attempts"]) == 2
    )


def test_admitted_legacy_conversation_keeps_its_protocol_and_spent_calls(
    durable_pipeline,
):
    storage, window, token = _overflow_coordinator(durable_pipeline)
    coordinator = owner(storage, window, token)
    data, refs = evidence(window.user_id)
    client = ScriptedClient()
    with coordinator.provider_scope():
        original = Reconciler(client, MODEL).run(data, refs)
    assert "reconciliation_context" not in coordinator.checkpoint
    actor, repeated = run(client, owner(storage, window, token), data, refs)
    assert repeated == original
    assert actor.source_scope is None
    assert len(client.calls) == actor.calls == 2


def test_source_scope_must_match_the_parent_owner(durable_pipeline):
    storage, window, token = _overflow_coordinator(durable_pipeline)
    coordinator = owner(storage, window, token)
    data, refs = evidence("another-user")
    client = ScriptedClient()
    with coordinator.provider_scope():
        actor = Reconciler(
            client,
            MODEL,
            source_scope={
                "org_id": storage.org_id,
                "project_id": window.project_id,
                "user_id": "another-user",
            },
        )
        with pytest.raises(InputBudgetConfigurationError, match="source owner changed"):
            actor.run(data, refs)
    assert client.calls == []


def test_committed_selection_cannot_be_rebound_to_other_inputs(durable_pipeline):
    storage, window, token = _overflow_coordinator(durable_pipeline)
    coordinator = owner(storage, window, token)
    data, refs = evidence(window.user_id, large=True)
    client = ScriptedClient()
    client.fail_consumer = True
    with pytest.raises(LiteLLMClientError):
        run(client, coordinator, data, refs)
    before = coordinator.checkpoint
    altered = deepcopy(before)
    node = next(iter(altered["reconciliation_context"]["nodes"].values()))
    node["input_refs_digest"] = "f" * 64
    with pytest.raises(OverflowCheckpointConflictError, match="immutable"):
        coordinator.save_checkpoint(altered)
    assert owner(storage, window, token).checkpoint == before


def test_legacy_conversation_saved_before_first_attempt_drains_unchanged(
    durable_pipeline,
):
    storage, window, token = _overflow_coordinator(durable_pipeline)
    coordinator = owner(storage, window, token)
    data, refs = evidence(window.user_id)
    client = ScriptedClient()
    coordinator._deadline = time.monotonic() + 1
    with coordinator.provider_scope(), pytest.raises(OverflowSuspendedError):
        Reconciler(client, MODEL).run(data, refs)
    assert coordinator.checkpoint["audited_conversation"]
    assert "audited_consolidation_budget" not in coordinator.checkpoint
    actor, result = run(client, owner(storage, window, token), data, refs)
    assert result.decisions
    assert actor.source_scope is None
    assert len(client.calls) == actor.calls == 2
    assert "reconciliation_context" not in coordinator.checkpoint


def test_smaller_model_capacity_triggers_compression_instead_of_passthrough(
    durable_pipeline,
):
    storage, window, token = _overflow_coordinator(durable_pipeline)
    coordinator = owner(storage, window, token)
    coordinator.context_limits = {MODEL: 16384}
    data, refs = evidence(window.user_id)
    original = data["sessions"][0]["messages"][0]["content"]
    text = (
        original
        + "\n"
        + "\n".join(f"Tool observation {n}: record value {n * n}." for n in range(1000))
    )
    data["sessions"][0]["messages"][0]["content"] = text
    refs[("request", 17)] = replace(refs[("request", 17)], evidence_texts=(text,))
    client = ScriptedClient()
    actor, result = run(client, coordinator, data, refs)
    assert result.decisions
    assert coordinator.checkpoint["reconciliation_context"]["nodes"]
    assert len(client.calls) == actor.calls > 2
    assert all(
        check_request_input_budget(
            params, context_limits=coordinator.context_limits, prompt_target=32768
        )
        <= 8192
        for params in client.calls
    )
