"""Existing operator retirement unblocks a terminal V8 window without rerolling it."""

import json
from copy import deepcopy

import litellm
import pytest
import yaml

from reflexio.server import metering_events
from reflexio.server.services.extraction.input_recovery import (
    apply_input_budget_recovery,
    plan_input_budget_recovery,
)
from reflexio.server.services.storage.storage_base import AgentRunStatus
from reflexio.server.services.storage.storage_base._extraction_stream import (
    MAX_EXTRACTION_WINDOW_ATTEMPTS,
    OverflowCheckpointConflictError,
)
from tests.server.services.playbook.test_procedural_window_integration import cli_setup
from tests.server.services.test_durable_window_pipeline import payload
from tests.server.services.test_durable_window_pipeline import pipeline as pipeline
from tests.server.services.test_window_retry_ceiling_integration import _drain_once

pytestmark = pytest.mark.integration


def _cursor(storage):
    with storage._stream_sql(read_only=True) as db:
        return db.query(
            "SELECT completed_seq,window_id,attempts,retry_at,last_error "
            "FROM extraction_cursors WHERE user_id=? AND kind=?",
            ("u", "playbook"),
        )[0]


@pytest.mark.parametrize("terminal_kind", ["malformed", "incomplete"])
def test_operator_retirement_preserves_terminal_failure_and_unblocks_future_window(
    pipeline, monkeypatch, tmp_path, terminal_kind
):
    engine, storage, window, token, client, calls, existing, wire = cli_setup(
        pipeline, monkeypatch, tmp_path
    )
    worker = pipeline[1]
    http = pipeline[2]
    storage.release_user_extraction(window.user_id, token)
    monkeypatch.setattr(
        "reflexio.lib._base.create_generation_litellm_client", lambda _: client
    )
    valid_new_operation = False

    def terminal_then_future(params, timeout):
        body = yaml.safe_load(params["messages"][1]["content"])
        if "source_observations" in body and not valid_new_operation:
            calls.append(params)
            arguments = (
                "{invalid-json"
                if terminal_kind == "malformed"
                else json.dumps(
                    {
                        "calls": [
                            {
                                "id": "incomplete",
                                "name": "finish_batch",
                                "arguments": {
                                    "status": "incomplete",
                                    "reason": "Unable to finish this operation.",
                                },
                            }
                        ]
                    }
                )
            )
            return litellm.ModelResponse(
                model=params["model"],
                choices=[
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"malformed-{len(calls)}",
                                    "type": "function",
                                    "function": {
                                        "name": "return_procedural_batch",
                                        "arguments": arguments,
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            )
        return wire(params, timeout)

    monkeypatch.setattr(client, "_completion_with_hard_timeout", terminal_then_future)
    financial_events = []
    monkeypatch.setattr(
        metering_events,
        "record_usage_event_strict",
        lambda **event: financial_events.append(event),
    )
    for _ in range(MAX_EXTRACTION_WINDOW_ATTEMPTS + 2):
        _drain_once(worker, storage)
    cursor = _cursor(storage)
    assert cursor["window_id"] == window.window_id
    assert cursor["attempts"] == MAX_EXTRACTION_WINDOW_ATTEMPTS
    assert cursor["retry_at"] == float("inf")
    assert cursor["last_error"].startswith("retry_exhausted:")
    failed_calls = 3 if terminal_kind == "malformed" else 2
    assert len(calls) == failed_calls  # Incomplete is not structurally repaired.
    failed = storage.get_agent_run(f"window:{window.window_id}")
    assert failed is not None and failed.status == AgentRunStatus.FAILED
    assert failed.committed_output is None
    checkpoint = deepcopy(failed.generation_request_snapshot["procedural_execution"])
    assert checkpoint["failed"] and checkpoint["physical_calls"] == failed_calls
    sources = deepcopy(failed.generation_request_snapshot["source_interaction_digests"])
    assert (
        storage.get_user_playbooks(user_id="u", agent_version="v")[0].content
        == existing[0].content
    )

    # A later publish does not silently reset or skip a parked head window.
    future = payload("after-v8-retirement", count=5, force=True)
    future["agent_version"] = "v"
    for turn in future["interaction_data_list"]:
        turn["content"] = (
            "The obsolete deploy command is no longer valid; require explicit approval."
        )
    assert http.post("/api/publish_interaction", json=future).json()["success"]
    _drain_once(worker, storage)
    assert _cursor(storage) == cursor
    assert len(calls) == failed_calls
    assert not financial_events

    # Exercise the actual existing operator preview, owned apply and stale CAS.
    receipt = plan_input_budget_recovery(storage, window.window_id, action="retire")
    assert receipt["recoverable"] and receipt["hold"] == "retry_exhausted"
    assert receipt["cursor_attempts"] == MAX_EXTRACTION_WINDOW_ATTEMPTS
    apply_input_budget_recovery(storage, receipt)
    with pytest.raises(OverflowCheckpointConflictError):
        apply_input_budget_recovery(storage, receipt)
    retired = storage.get_agent_run(failed.id)
    assert retired is not None and retired.status == AgentRunStatus.CANCELLED
    assert retired.committed_output is None
    assert retired.generation_request_snapshot["procedural_execution"] == checkpoint
    assert retired.generation_request_snapshot["source_interaction_digests"] == sources
    assert len(calls) == failed_calls
    assert not financial_events
    assert storage.pending_extraction_effects() == []
    with storage._stream_sql(read_only=True) as db:
        retired_window = db.query(
            "SELECT completed,effects_done,derived_claimed,effects,end_seq FROM extraction_windows WHERE window_id=?",
            (window.window_id,),
        )[0]
    effects = json.loads(retired_window["effects"])
    assert effects["retired"] and effects["skipped"] and effects["billing"] is None
    assert (
        retired_window["completed"]
        == retired_window["effects_done"]
        == retired_window["derived_claimed"]
        == 1
    )
    assert _cursor(storage)["completed_seq"] == retired_window["end_seq"]
    assert _cursor(storage)["window_id"] is None

    # Only the new admitted operation may execute; no old charge is refunded.
    valid_new_operation = True
    for _ in range(4):
        _drain_once(worker, storage)
    assert storage.extraction_status("u", "after-v8-retirement")["status"] == "done"
    assert len(calls) == failed_calls + 3
    assert storage.get_user_playbooks(user_id="u", agent_version="v") == []
    assert storage.pending_extraction_effects() == []
    final_failed = storage.get_agent_run(failed.id)
    assert final_failed is not None
    assert (
        final_failed.generation_request_snapshot["procedural_execution"] == checkpoint
    )
    with storage._stream_sql(read_only=True) as db:
        completed = db.query(
            "SELECT window_id FROM extraction_windows WHERE user_id=? AND kind=? AND completed=1 ORDER BY end_seq",
            ("u", "playbook"),
        )
    assert len(completed) == 2 and completed[1]["window_id"] != window.window_id
    legitimate = storage.get_agent_run(f"window:{completed[1]['window_id']}")
    assert legitimate is not None and legitimate.status == AgentRunStatus.FINALIZED
    assert legitimate.binding.request_id == "after-v8-retirement"
    assert (
        legitimate.generation_request_snapshot["procedural_execution"]["physical_calls"]
        == 3
    )


@pytest.mark.parametrize("failure_kind", ["capacity", "presentation"])
def test_compiler_input_hold_rearms_without_rerolling_reader_or_refunding_attempts(
    pipeline, monkeypatch, failure_kind
):
    from reflexio.server.llm import input_budget
    from reflexio.server.services.playbook.procedural import native, transport
    from tests.server.services.playbook.test_procedural_window_integration import setup

    _, storage, window, token, client, calls, _ = setup(pipeline, monkeypatch)
    worker = pipeline[1]
    storage.release_user_extraction(window.user_id, token)
    monkeypatch.setattr(
        "reflexio.lib._base.create_generation_litellm_client", lambda _: client
    )
    original_guard = input_budget.check_request_input_budget
    repaired = False

    def capacity(request, **kwargs):
        original_guard(request, **kwargs)
        if (
            failure_kind == "capacity"
            and not repaired
            and "source_observations"
            in yaml.safe_load(request["messages"][1]["content"])
        ):
            raise input_budget.InputBudgetExceededError(
                "synthetic pre-dispatch compiler hold"
            )

    monkeypatch.setattr(
        "reflexio.server.services.extraction.overflow_coordinator.check_request_input_budget",
        capacity,
    )
    real_dump = transport.yaml.dump

    def broken_renderer(body, **kwargs):
        if (
            failure_kind == "presentation"
            and not repaired
            and "source_observations" in body
        ):
            return "owner: wrong-source\n"
        return real_dump(body, **kwargs)

    monkeypatch.setattr(transport.yaml, "dump", broken_renderer)
    for _ in range(MAX_EXTRACTION_WINDOW_ATTEMPTS + 2):
        _drain_once(worker, storage)
        if _cursor(storage)["retry_at"] == float("inf"):
            break
    held = storage.get_agent_run(f"window:{window.window_id}")
    assert _cursor(storage)["retry_at"] == float("inf")
    reason = (
        "input_budget_exceeded"
        if failure_kind == "capacity"
        else "input_budget_configuration_error"
    )
    assert reason in _cursor(storage)["last_error"]
    assert len(calls) == 1
    frozen = deepcopy(held.generation_request_snapshot["overflow"])
    assert len(frozen["attempts"]) == 1
    assert not frozen["checkpoint"][native.KEY].get("failed_steps")
    reader = deepcopy(frozen["checkpoint"][native.KEY]["reader_steps"])
    receipt = plan_input_budget_recovery(storage, window.window_id, action="rearm")
    assert receipt["recoverable"]
    repaired = True
    apply_input_budget_recovery(storage, receipt)
    after_rearm = storage.get_agent_run(held.id)
    assert (
        after_rearm.generation_request_snapshot["overflow"]["attempts"]
        == frozen["attempts"]
    )
    for _ in range(3):
        _drain_once(worker, storage)
    completed = storage.get_agent_run(held.id)
    final = completed.generation_request_snapshot["overflow"]
    assert len(calls) == len(final["attempts"]) == 3
    assert final["checkpoint"][native.KEY]["reader_steps"] == reader
    assert _cursor(storage)["completed_seq"] == window.end_seq
    assert _cursor(storage)["window_id"] is None
    assert len(storage.get_user_playbooks(user_id="u", agent_version="v")) == 1
