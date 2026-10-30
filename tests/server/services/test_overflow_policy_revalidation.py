"""A saved overflow output survives a policy-hash change only if it passes today's checks."""

import json
import logging
import math
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.extraction import overflow_pipeline
from reflexio.server.services.extraction.input_recovery import (
    apply_input_budget_recovery,
    plan_input_budget_recovery,
)
from reflexio.server.services.profile.components import extractor as profile_extractor
from tests.server.services.test_durable_window_pipeline import payload
from tests.server.services.test_durable_window_pipeline import pipeline as pipeline

_PROFILE = {
    "content": "The user prefers concise answers.",
    "time_to_live": "infinity",
    "source_span": "I prefer concise answers with details 0",
}


def _respond(params, hard_timeout):
    schema = getattr(params.get("response_format"), "__name__", "") or json.dumps(
        params.get("response_format"), default=str
    )
    if "ProfileDeduplicationOutput" in schema or any(
        "unique_ids" in str(message.get("content", ""))
        for message in params["messages"]
    ):
        content = {"duplicate_groups": [], "unique_ids": ["NEW-0"], "deletions": []}
    else:
        content = {"profiles": [deepcopy(_PROFILE)]}
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(content), tool_calls=None, refusal=None
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
        model="minimax/MiniMax-M3",
        _hidden_params={"custom_llm_provider": "minimax"},
    )


def _extraction_calls(dispatch: Mock) -> int:
    return sum(
        getattr(call.args[0].get("response_format"), "__name__", "")
        == "StructuredProfilesOutput"
        for call in dispatch.call_args_list
    )


def _state(storage, window_id):
    with storage._stream_sql(read_only=True) as db:
        cursor = db.query(
            "SELECT completed_seq,window_id,attempts,retry_at,last_error "
            "FROM extraction_cursors WHERE user_id=? AND kind=?",
            ("u", "profile"),
        )[0]
    run = storage.get_agent_run(f"window:{window_id}")
    return cursor, run, run.generation_request_snapshot["overflow"]["checkpoint"]


def _make_due(storage, *, cursor=True):
    with storage._stream_sql() as db:
        if cursor:
            db.query("UPDATE extraction_cursors SET retry_at=0 WHERE user_id=?", ("u",))
        db.query(
            "UPDATE learning_work SET due_at=0 WHERE org_id=? AND user_id=?",
            (storage.org_id, "u"),
        )


@pytest.mark.parametrize("change", ["unchanged", "reworded", "incompatible"])
def test_saved_output_is_revalidated_under_changed_policy(
    pipeline, monkeypatch, caplog, change
):
    engine, worker, client = pipeline
    storage = engine.get_storage()
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", "true")
    monkeypatch.setenv(
        "REFLEXIO_LLM_CONTEXT_LIMITS_JSON",
        '{"minimax/MiniMax-M3":1000000,"gpt-4o":65536}',
    )
    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.token_counter", lambda **_: 100
    )
    dispatch = Mock(side_effect=_respond)
    monkeypatch.setattr(LiteLLMClient, "_completion_with_deadline", dispatch)
    # Fail once AFTER the parent output is committed, so the retry replays it.
    original_extract = overflow_pipeline.extract_overflow_window
    failures = []

    def extract_then_fail(**kwargs):
        result = original_extract(**kwargs)
        if not failures:
            failures.append(result.run_id)
            raise RuntimeError("injected failure after saved output")
        return result

    monkeypatch.setattr(overflow_pipeline, "extract_overflow_window", extract_then_fail)
    assert client.post(
        "/api/publish_interaction", json=payload("policy-drift", force=True)
    ).json()["success"]
    assert worker.drain_org(storage.org_id, 1, 300) == 0
    window_id = failures[0].removeprefix("window:")
    cursor, run, checkpoint = _state(storage, window_id)
    assert cursor["window_id"] == window_id and cursor["completed_seq"] == 0
    assert run.status.value == "agent_completed" and run.committed_output
    saved_policy = checkpoint["extraction_policy"]
    assert saved_policy and checkpoint["extraction_children"]
    extraction_calls = _extraction_calls(dispatch)

    if change != "unchanged":
        # A wording-only edit to the overflow instruction moves the policy hash.
        monkeypatch.setattr(
            profile_extractor,
            "PROFILE_SOURCE_EVIDENCE_INSTRUCTION",
            profile_extractor.PROFILE_SOURCE_EVIDENCE_INSTRUCTION + " Quote exactly.",
        )
    original_validator = profile_extractor._profile_evidence_validation_errors
    if change == "incompatible":
        # The current contract rejects what the saved output contains.
        monkeypatch.setattr(
            profile_extractor,
            "_profile_evidence_validation_errors",
            lambda *_args: ["profiles[0]: rejected by the current contract"],
        )
    caplog.clear()
    _make_due(storage)
    with caplog.at_level(logging.INFO):
        completed = worker.drain_org(storage.org_id, 1, 300)
    revalidated = [
        r for r in caplog.records if "event=extraction_policy_revalidated" in r.msg
    ]
    parked = [r for r in caplog.records if r.msg.startswith("Extraction input parked")]
    cursor, run, checkpoint = _state(storage, window_id)
    assert _extraction_calls(dispatch) == extraction_calls

    if change != "incompatible":
        assert completed == 1 and cursor["completed_seq"] == 10
        assert cursor["window_id"] is None and cursor["attempts"] == 0
        assert storage.count_all_profiles() == 1
        assert not parked
        if change == "unchanged":
            assert not revalidated
            assert checkpoint["extraction_policy"] == saved_policy
        else:
            assert len(revalidated) == 1
            assert revalidated[0].levelno == logging.WARNING
            current = checkpoint["extraction_policy"]
            assert current != saved_policy
            assert revalidated[0].args[2:] == (saved_policy[:12], current[:12])
            assert {
                c["policy"] for c in checkpoint["extraction_children"].values()
            } == {current}
            # No customer content reaches the operational line.
            assert "concise" not in revalidated[0].getMessage()
        return

    # Genuinely incompatible: held for an operator instead of retried forever.
    assert completed == 0 and storage.count_all_profiles() == 0
    assert not revalidated
    assert cursor["window_id"] == window_id and cursor["completed_seq"] == 0
    assert math.isinf(cursor["retry_at"])
    assert cursor["last_error"] == "input_budget_configuration_error"
    assert checkpoint["extraction_policy"] == saved_policy
    assert run.status.value == "failed" and run.committed_output
    assert len(parked) == 1 and parked[0].levelno == logging.ERROR
    assert storage.extraction_status("u", "policy-drift") == {
        "status": "failed",
        "reason": "input_budget_configuration_error",
    }
    # Parked means no further attempts, hence no repeated ERROR.
    caplog.clear()
    _make_due(storage, cursor=False)
    assert worker.drain_org(storage.org_id, 1, 300) == 0
    assert not [
        r for r in caplog.records if r.msg.startswith("Extraction input parked")
    ]
    assert _state(storage, window_id)[0]["attempts"] == cursor["attempts"]

    # The operator re-arm path accepts the parked configuration hold.
    receipt = plan_input_budget_recovery(storage, window_id)
    assert receipt["recoverable"] and receipt["reason"] == "ready"
    monkeypatch.setattr(
        profile_extractor, "_profile_evidence_validation_errors", original_validator
    )
    apply_input_budget_recovery(storage, receipt)
    _make_due(storage, cursor=False)
    assert worker.drain_org(storage.org_id, 1, 300) == 1
    cursor, _, checkpoint = _state(storage, window_id)
    assert cursor["completed_seq"] == 10 and cursor["window_id"] is None
    assert checkpoint["extraction_policy"] != saved_policy
    assert storage.count_all_profiles() == 1
    assert _extraction_calls(dispatch) == extraction_calls


def test_backoff_configuration_failure_is_not_operator_recoverable(pipeline):
    from tests.server.services.test_durable_window_pipeline import (
        _overflow_coordinator,
    )

    storage, window, token = _overflow_coordinator(pipeline, actual_source=True)
    storage.fail_extraction_input_budget(
        window, token, "input_budget_configuration_error"
    )
    receipt = plan_input_budget_recovery(storage, window.window_id)
    assert receipt["reason"] == "not_held" and not receipt["recoverable"]


@pytest.mark.parametrize("after_rearm", ["unchanged", "corrected"])
def test_incomplete_policy_mismatch_stops_retrying_without_resetting_evidence(
    pipeline, monkeypatch, after_rearm
):
    from reflexio.server.services.storage.storage_base._extraction_stream import (
        MAX_EXTRACTION_WINDOW_ATTEMPTS,
    )

    engine, worker, client = pipeline
    storage = engine.get_storage()
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", "true")
    monkeypatch.setenv(
        "REFLEXIO_LLM_CONTEXT_LIMITS_JSON",
        '{"minimax/MiniMax-M3":1000000,"gpt-4o":65536}',
    )
    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.token_counter", lambda **_: 100
    )
    dispatch = Mock(side_effect=_respond)
    monkeypatch.setattr(LiteLLMClient, "_completion_with_deadline", dispatch)
    original_extract = overflow_pipeline.extract_overflow_window
    first = True

    def interrupt_incomplete(**kwargs):
        nonlocal first
        if first:
            first = False

            def interrupt(_call):
                raise RuntimeError("interrupted before a child output was committed")

            kwargs["execute_call"] = interrupt
        return original_extract(**kwargs)

    monkeypatch.setattr(
        overflow_pipeline, "extract_overflow_window", interrupt_incomplete
    )
    assert client.post(
        "/api/publish_interaction", json=payload("incomplete", force=True)
    ).json()["success"]
    assert worker.drain_org(storage.org_id, 1, 300) == 0
    with storage._stream_sql(read_only=True) as db:
        window_id = db.query(
            "SELECT window_id FROM extraction_cursors WHERE user_id='u' AND kind='profile'"
        )[0]["window_id"]
    _, initial_run, initial_checkpoint = _state(storage, window_id)
    frozen = deepcopy(initial_run.generation_request_snapshot["overflow"])
    assert initial_checkpoint["extraction_policy"]
    assert initial_run.committed_output is None
    instruction = profile_extractor.PROFILE_SOURCE_EVIDENCE_INSTRUCTION
    monkeypatch.setattr(
        profile_extractor,
        "PROFILE_SOURCE_EVIDENCE_INSTRUCTION",
        instruction + " Quote exactly.",
    )
    for _ in range(MAX_EXTRACTION_WINDOW_ATTEMPTS + 3):
        _make_due(storage, cursor=False)
        with storage._stream_sql() as db:
            db.query("UPDATE extraction_cursors SET retry_at=0 WHERE retry_at<1e300")
        worker.drain_org(storage.org_id, 1, 300)
    cursor, run, checkpoint = _state(storage, window_id)
    assert cursor["retry_at"] == float("inf")
    assert cursor["attempts"] == MAX_EXTRACTION_WINDOW_ATTEMPTS
    assert cursor["completed_seq"] == 0 and cursor["window_id"] == window_id
    assert checkpoint == frozen["checkpoint"]
    assert run.generation_request_snapshot["overflow"]["attempts"] == frozen["attempts"]
    assert run.committed_output is None and storage.count_all_profiles() == 0
    dispatch.assert_not_called()
    assert storage.extraction_status("u", "incomplete") == {
        "status": "failed",
        "reason": "retry_exhausted",
    }
    receipt = plan_input_budget_recovery(storage, window_id)
    assert receipt["recoverable"], receipt
    apply_input_budget_recovery(storage, receipt)
    if after_rearm == "corrected":
        monkeypatch.setattr(
            profile_extractor, "PROFILE_SOURCE_EVIDENCE_INSTRUCTION", instruction
        )
    _make_due(storage)
    worker.drain_org(storage.org_id, 1, 300)
    cursor, run, checkpoint = _state(storage, window_id)
    if after_rearm == "unchanged":
        assert cursor["retry_at"] == float("inf")
        assert cursor["completed_seq"] == 0
        assert checkpoint == frozen["checkpoint"]
        dispatch.assert_not_called()
    else:
        assert storage.extraction_status("u", "incomplete")["status"] == "done"
        assert storage.count_all_profiles() == 1


def test_policy_change_between_chunks_preserves_completed_child():
    from reflexio.server.llm.input_budget import InputBudgetConfigurationError
    from tests.server.services.test_overflow_carried_evidence import coordinator_fixture

    coordinator, build, execute = coordinator_fixture()
    saved = []

    def build_changed():
        call = build()
        if saved:
            call.messages.append(
                {"role": "system", "content": "A changed extraction contract"}
            )
        return call

    def execute_first(call):
        result = execute(call)
        saved.append(deepcopy(coordinator.checkpoint))
        return result

    client = Mock(spec=LiteLLMClient)
    with pytest.raises(
        InputBudgetConfigurationError, match="policy changed after checkpoint"
    ):
        overflow_pipeline.extract_overflow_window(
            coordinator=coordinator,
            client=client,
            build_call=build_changed,
            execute_call=execute_first,
            final_validator=None,
        )
    assert len(saved) == 1
    assert coordinator.checkpoint == saved[0]
    assert coordinator.checkpoint["extraction_children"]
    assert coordinator.checkpoint["completed_steps"]
    assert not client.mock_calls
