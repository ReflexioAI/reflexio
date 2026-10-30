"""Mutable profile context cannot reset quota or masquerade as policy drift."""

import json
from copy import deepcopy
from dataclasses import replace
from unittest.mock import Mock

import pytest

from reflexio.models.api_schema.domain import UserProfile
from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.extraction import overflow_pipeline
from tests.server.services.test_durable_window_pipeline import payload
from tests.server.services.test_durable_window_pipeline import pipeline as pipeline
from tests.server.services.test_overflow_policy_revalidation import (
    _make_due,
    _respond,
    _state,
)

pytestmark = pytest.mark.integration


def _interrupt(pipeline, monkeypatch, after_child):
    engine, worker, http = pipeline
    storage = engine.get_storage()
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv(
        "REFLEXIO_LLM_CONTEXT_LIMITS_JSON",
        '{"minimax/MiniMax-M3":1000000,"gpt-4o":65536}',
    )
    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.token_counter", lambda **_: 100
    )
    dispatch = Mock(side_effect=_respond)
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", dispatch)
    extract = overflow_pipeline.extract_overflow_window
    first = True

    def interrupt_once(**kwargs):
        nonlocal first
        if first:
            first = False
            execute = kwargs["execute_call"]

            def interrupt(call):
                if after_child:
                    execute(call)
                raise RuntimeError("Interrupted before parent output commit")

            kwargs["execute_call"] = interrupt
        return extract(**kwargs)

    monkeypatch.setattr(overflow_pipeline, "extract_overflow_window", interrupt_once)
    assert http.post(
        "/api/publish_interaction", json=payload("context-retry", force=True)
    ).json()["success"]
    assert worker.drain_org(storage.org_id, 1, 300) == 0
    with storage._stream_sql(read_only=True) as db:
        window_id = db.query(
            "SELECT window_id FROM extraction_cursors WHERE user_id='u' AND kind='profile'"
        )[0]["window_id"]
    return storage, worker, dispatch, window_id


def _add_context(storage, user_id="u"):
    storage.add_user_profile(
        user_id,
        [
            UserProfile(
                profile_id="context-edit",
                user_id=user_id,
                content="The user has an additional existing preference.",
                last_modified_timestamp=100,
                generated_from_request_id="context-edit",
            )
        ],
        skip_embedding=True,
    )


@pytest.mark.parametrize("after_child", [False, True])
@pytest.mark.parametrize("context_user", ["u", "other-user"])
def test_context_retry_preserves_quota_and_receipts(
    pipeline, monkeypatch, after_child, context_user
):
    storage, worker, dispatch, window_id = _interrupt(
        pipeline, monkeypatch, after_child
    )
    _, before, checkpoint = _state(storage, window_id)
    frozen = deepcopy(before.generation_request_snapshot["overflow"])
    calls = dispatch.call_count
    _add_context(storage, context_user)
    _make_due(storage)
    assert worker.drain_org(storage.org_id, 1, 300) == 1
    assert storage.extraction_status("u", "context-retry")["status"] == "done"
    _, after, current = _state(storage, window_id)
    manifest = after.generation_request_snapshot["overflow"]
    assert manifest["attempts"][: len(frozen["attempts"])] == frozen["attempts"]
    assert manifest["attempt_limit"] == frozen["attempt_limit"]
    for key, receipt in checkpoint["completed_steps"].items():
        assert current["completed_steps"][key] == receipt
    assert (
        current["extraction_instruction_contract"]
        == checkpoint["extraction_instruction_contract"]
    )
    if context_user == "u":
        assert current["extraction_policy"] != checkpoint["extraction_policy"]
        # Even a completed old child must run again under the new context.
        assert dispatch.call_count == calls + 2
    else:
        assert current["extraction_policy"] == checkpoint["extraction_policy"]
        assert dispatch.call_count == calls + (1 if after_child else 2)


@pytest.mark.parametrize("hold", ["legacy", "active", "quota"])
def test_context_change_cannot_bypass_unknown_continuation_or_quota(
    pipeline, monkeypatch, hold
):
    storage, worker, dispatch, window_id = _interrupt(pipeline, monkeypatch, True)
    _, run, _ = _state(storage, window_id)
    snapshot = deepcopy(run.generation_request_snapshot)
    manifest = snapshot["overflow"]
    checkpoint = manifest["checkpoint"]
    if hold == "legacy":
        checkpoint.pop("extraction_instruction_contract", None)
    elif hold == "active":
        checkpoint["active_child"] = {"key": "in-flight-context"}
    else:
        original = manifest["attempts"][0]
        manifest["attempts"] = [
            {**deepcopy(original), "id": f"consumed-{index}"} for index in range(128)
        ]
    with storage.conn:
        storage.conn.execute(
            "UPDATE _agent_runs SET generation_request_snapshot=? WHERE id=?",
            (json.dumps(snapshot), run.id),
        )
    calls = dispatch.call_count
    _add_context(storage)
    _make_due(storage)
    assert worker.drain_org(storage.org_id, 1, 300) == 0
    cursor, current, _ = _state(storage, window_id)
    assert cursor["completed_seq"] == 0 and cursor["window_id"] == window_id
    assert dispatch.call_count == calls
    assert (
        current.generation_request_snapshot["overflow"]["attempts"]
        == manifest["attempts"]
    )
    assert cursor["last_error"] == (
        "input_budget_exceeded"
        if hold == "quota"
        else "input_budget_configuration_error"
    )


def test_unchanged_legacy_context_acquires_instruction_witness(pipeline, monkeypatch):
    storage, worker, dispatch, window_id = _interrupt(pipeline, monkeypatch, True)
    _, run, _ = _state(storage, window_id)
    snapshot = deepcopy(run.generation_request_snapshot)
    snapshot["overflow"]["checkpoint"].pop("extraction_instruction_contract")
    with storage.conn:
        storage.conn.execute(
            "UPDATE _agent_runs SET generation_request_snapshot=? WHERE id=?",
            (json.dumps(snapshot), run.id),
        )
    calls = dispatch.call_count
    _make_due(storage)
    assert worker.drain_org(storage.org_id, 1, 300) == 1
    assert dispatch.call_count == calls + 1
    _, _, checkpoint = _state(storage, window_id)
    assert checkpoint["extraction_instruction_contract"]


def test_template_identity_change_cannot_hide_behind_equal_rendering():
    from tests.server.services.test_overflow_carried_evidence import coordinator_fixture

    coordinator, build, execute = coordinator_fixture()
    saved = []

    def build_call():
        call = build()
        return replace(
            call,
            context_free_messages=deepcopy(call.messages),
            template_identities={
                "template": {"active_version": "v2" if saved else "v1"}
            },
        )

    def execute_first(call):
        result = execute(call)
        saved.append(deepcopy(coordinator.checkpoint))
        return result

    with pytest.raises(InputBudgetConfigurationError, match="instructions changed"):
        overflow_pipeline.extract_overflow_window(
            coordinator=coordinator,
            client=Mock(spec=LiteLLMClient),
            build_call=build_call,
            execute_call=execute_first,
            final_validator=None,
        )
    assert len(saved) == 1 and coordinator.checkpoint == saved[0]
