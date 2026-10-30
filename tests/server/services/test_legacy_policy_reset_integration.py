"""An operator reset restarts a legacy checkpoint held by a policy mismatch.

Checkpoints saved before the instruction witness existed cannot prove a
context-only policy change, so ordinary retries and re-arms park them again.
"""

import json
from copy import deepcopy
from unittest.mock import Mock

import pytest

from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.extraction import overflow_pipeline
from reflexio.server.services.extraction.input_recovery import (
    _inspect,
    apply_input_budget_recovery,
    plan_input_budget_recovery,
)
from reflexio.server.services.profile.components import extractor as profile_extractor
from reflexio.server.services.storage.storage_base._extraction_stream import (
    MAX_EXTRACTION_WINDOW_ATTEMPTS,
    OverflowCheckpointConflictError,
)
from tests.server.services.test_durable_window_pipeline import payload
from tests.server.services.test_durable_window_pipeline import pipeline as pipeline
from tests.server.services.test_overflow_policy_revalidation import (
    _make_due,
    _respond,
    _state,
)

_HOLD = "retry_exhausted:input_budget_configuration_error"


def _edit_checkpoint(storage, window_id, edit):
    run_id = f"window:{window_id}"
    with storage._stream_sql() as db:
        row = db.query(
            "SELECT generation_request_snapshot FROM _agent_runs WHERE id=?",
            (run_id,),
        )[0]
        snapshot = json.loads(row["generation_request_snapshot"])
        edit(snapshot["overflow"]["checkpoint"])
        db.query(
            "UPDATE _agent_runs SET generation_request_snapshot=? WHERE id=?",
            (json.dumps(snapshot), run_id),
        )


def _park_legacy_mismatch(pipeline, monkeypatch):
    """Interrupt a window, strip its witness, then change the policy until parked."""
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
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", dispatch)
    original_extract = overflow_pipeline.extract_overflow_window
    first = True

    def interrupt_first(**kwargs):
        nonlocal first
        if first:
            first = False

            def interrupt(_call):
                raise RuntimeError("interrupted before a child output was committed")

            kwargs["execute_call"] = interrupt
        return original_extract(**kwargs)

    monkeypatch.setattr(overflow_pipeline, "extract_overflow_window", interrupt_first)
    assert client.post(
        "/api/publish_interaction", json=payload("legacy", force=True)
    ).json()["success"]
    assert worker.drain_org(storage.org_id, 1, 300) == 0
    with storage._stream_sql(read_only=True) as db:
        window_id = db.query(
            "SELECT window_id FROM extraction_cursors WHERE user_id='u' AND kind='profile'"
        )[0]["window_id"]

    # A checkpoint written before the instruction witness existed, with the
    # active child mapping its legacy run had pinned to the old policy.
    def make_legacy(checkpoint):
        checkpoint.pop("extraction_instruction_contract")
        checkpoint["extraction_children"] = {
            "chunk-0": {"policy": checkpoint["extraction_policy"]}
        }

    _edit_checkpoint(storage, window_id, make_legacy)
    _, run, checkpoint = _state(storage, window_id)
    assert checkpoint["extraction_policy"] and run.committed_output is None
    monkeypatch.setattr(
        profile_extractor,
        "PROFILE_SOURCE_EVIDENCE_INSTRUCTION",
        profile_extractor.PROFILE_SOURCE_EVIDENCE_INSTRUCTION + " Quote exactly.",
    )
    for _ in range(MAX_EXTRACTION_WINDOW_ATTEMPTS + 3):
        _make_due(storage, cursor=False)
        with storage._stream_sql() as db:
            db.query("UPDATE extraction_cursors SET retry_at=0 WHERE retry_at<1e300")
        worker.drain_org(storage.org_id, 1, 300)
    cursor, run, checkpoint = _state(storage, window_id)
    assert cursor["retry_at"] == float("inf") and cursor["last_error"] == _HOLD
    dispatch.assert_not_called()
    return storage, worker, window_id, dispatch


def test_reset_restarts_legacy_checkpoint_under_current_policy(pipeline, monkeypatch):
    storage, worker, window_id, dispatch = _park_legacy_mismatch(pipeline, monkeypatch)
    cursor, run, checkpoint = _state(storage, window_id)
    frozen = deepcopy(run.generation_request_snapshot["overflow"])
    assert checkpoint.get("extraction_children")

    # A plain re-arm leaves the pin in place, so the window parks again.
    rearm = plan_input_budget_recovery(storage, window_id)
    assert rearm["recoverable"] and rearm["action"] == "rearm"
    apply_input_budget_recovery(storage, rearm)
    _make_due(storage)
    worker.drain_org(storage.org_id, 1, 300)
    cursor, run, checkpoint = _state(storage, window_id)
    assert cursor["retry_at"] == float("inf") and cursor["last_error"] == _HOLD
    assert checkpoint == frozen["checkpoint"]
    dispatch.assert_not_called()

    receipt = plan_input_budget_recovery(
        storage, window_id, action="reset_legacy_policy"
    )
    assert receipt["recoverable"], receipt
    assert (
        receipt["action"] == "reset_legacy_policy"
        and receipt["hold"] == "retry_exhausted"
    )
    apply_input_budget_recovery(storage, receipt)
    reset_cursor, run, checkpoint = _state(storage, window_id)
    manifest = run.generation_request_snapshot["overflow"]
    assert "extraction_policy" not in checkpoint
    assert "extraction_children" not in checkpoint
    assert checkpoint["completed_steps"] == frozen["checkpoint"]["completed_steps"]
    assert manifest["attempts"] == frozen["attempts"]
    assert manifest["revision"] == frozen["revision"] + 1
    assert reset_cursor["attempts"] == cursor["attempts"]
    assert reset_cursor["retry_at"] != float("inf")
    with pytest.raises(OverflowCheckpointConflictError):
        apply_input_budget_recovery(storage, receipt)

    _make_due(storage)
    worker.drain_org(storage.org_id, 1, 300)
    assert storage.extraction_status("u", "legacy")["status"] == "done"
    assert storage.count_all_profiles() == 1
    assert dispatch.called
    _, run, checkpoint = _state(storage, window_id)
    assert checkpoint["extraction_instruction_contract"]


def _set_witness(storage, window_id):
    _edit_checkpoint(
        storage,
        window_id,
        lambda checkpoint: checkpoint.update(extraction_instruction_contract="w"),
    )


def _set_active_child(storage, window_id):
    _edit_checkpoint(
        storage, window_id, lambda checkpoint: checkpoint.update(active_child="c")
    )


def _set_committed(storage, window_id):
    with storage._stream_sql() as db:
        db.query(
            "UPDATE _agent_runs SET committed_output='{}' WHERE id=?",
            (f"window:{window_id}",),
        )


def _set_other_hold(storage, window_id):
    with storage._stream_sql() as db:
        db.query(
            "UPDATE extraction_cursors SET last_error=? WHERE window_id=?",
            ("retry_exhausted:TimeoutError@x.py:1", window_id),
        )


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (_set_witness, "not_legacy_policy_checkpoint"),
        (_set_active_child, "active_continuation"),
        (_set_committed, "committed_output"),
        (_set_other_hold, "not_configuration_hold"),
    ],
)
def test_reset_refuses_windows_it_cannot_restart_safely(
    pipeline, monkeypatch, mutate, reason
):
    storage, _, window_id, _ = _park_legacy_mismatch(pipeline, monkeypatch)
    mutate(storage, window_id)
    before = _state(storage, window_id)
    receipt = plan_input_budget_recovery(
        storage, window_id, action="reset_legacy_policy"
    )
    assert receipt["reason"] == reason and not receipt["recoverable"]
    with pytest.raises(OverflowCheckpointConflictError):
        apply_input_budget_recovery(storage, receipt)
    # A forged receipt cannot bypass the storage guard either.
    window, current = _inspect(storage, window_id, action="reset_legacy_policy")
    token = storage.claim_user_extraction("u", "test-reset", 300)
    assert token is not None
    try:
        with pytest.raises(OverflowCheckpointConflictError):
            storage.reset_legacy_extraction_policy(
                window,
                token,
                expected_revision=current["parent_revision"],
                source_digest=current["source_digest"],
                expected_attempts=current["cursor_attempts"],
                expected_last_error=current["last_error"],
                expected_input_protocol=current["input_protocol"],
            )
    finally:
        storage.release_user_extraction("u", token)
    assert _state(storage, window_id) == before
