"""An operator retire closes a parked window unextracted so later inputs proceed."""

import pytest

from reflexio.server.services.extraction.input_recovery import (
    _inspect,
    apply_input_budget_recovery,
    plan_input_budget_recovery,
)
from reflexio.server.services.storage.storage_base._extraction_stream import (
    OverflowCheckpointConflictError,
)
from tests.server.services.test_durable_window_pipeline import payload
from tests.server.services.test_durable_window_pipeline import pipeline as pipeline
from tests.server.services.test_legacy_policy_reset_integration import (
    _HOLD,
    _park_legacy_mismatch,
)
from tests.server.services.test_overflow_policy_revalidation import _make_due, _state


def _window_row(storage, window_id):
    with storage._stream_sql(read_only=True) as db:
        return db.query(
            "SELECT end_seq,completed,effects_done,derived_claimed,outcome "
            "FROM extraction_windows WHERE window_id=?",
            (window_id,),
        )[0]


def test_retire_releases_the_user_without_learning_the_window(pipeline, monkeypatch):
    _, _, client = pipeline
    storage, worker, window_id, dispatch = _park_legacy_mismatch(pipeline, monkeypatch)
    _, parked_run, _ = _state(storage, window_id)
    journal = parked_run.generation_request_snapshot["overflow"]["attempts"]
    end_seq = _window_row(storage, window_id)["end_seq"]

    receipt = plan_input_budget_recovery(storage, window_id, action="retire")
    assert receipt["recoverable"], receipt
    assert receipt["action"] == "retire" and receipt["last_error"] == _HOLD
    apply_input_budget_recovery(storage, receipt)

    cursor, run, checkpoint = _state(storage, window_id)
    assert cursor["completed_seq"] == end_seq and cursor["window_id"] is None
    assert cursor["attempts"] == 0 and cursor["last_error"] is None
    assert cursor["retry_at"] != float("inf")
    row = _window_row(storage, window_id)
    assert row["completed"] == 1 and row["effects_done"] == 1
    assert row["derived_claimed"] == 1 and row["outcome"] is None
    assert run.status == "cancelled" and run.committed_output is None
    assert checkpoint == {"stage": "retired"}
    assert run.generation_request_snapshot["overflow"]["attempts"] == journal
    assert storage.pending_extraction_effects(limit=10) == []
    assert storage.count_all_profiles() == 0
    dispatch.assert_not_called()
    with pytest.raises(OverflowCheckpointConflictError):
        apply_input_budget_recovery(storage, receipt)

    # The user's next conversation is selected and learned as usual.
    assert client.post(
        "/api/publish_interaction", json=payload("after-retire", force=True)
    ).json()["success"]
    for _ in range(4):
        _make_due(storage)
        worker.drain_org(storage.org_id, 1, 300)
    assert storage.extraction_status("u", "after-retire") == {
        "status": "done",
        "reason": "covered",
    }
    profiles = storage.get_all_profiles(user_id="u")
    assert profiles and dispatch.called
    # The retired request is never extracted as its own window.
    assert {p.generated_from_request_id for p in profiles} == {"after-retire"}


def _rearm_hold(storage, window_id):
    with storage._stream_sql() as db:
        db.query(
            "UPDATE extraction_cursors SET retry_at=0 WHERE window_id=?", (window_id,)
        )


def test_retire_refuses_a_window_that_is_not_parked(pipeline, monkeypatch):
    storage, _, window_id, _ = _park_legacy_mismatch(pipeline, monkeypatch)
    _rearm_hold(storage, window_id)
    before = _state(storage, window_id)
    receipt = plan_input_budget_recovery(storage, window_id, action="retire")
    assert receipt["reason"] == "not_held" and not receipt["recoverable"]
    with pytest.raises(OverflowCheckpointConflictError):
        apply_input_budget_recovery(storage, receipt)
    # A forged receipt cannot bypass the storage guard either.
    window, current = _inspect(storage, window_id, action="retire")
    token = storage.claim_user_extraction("u", "test-retire", 300)
    assert token is not None
    try:
        with pytest.raises(OverflowCheckpointConflictError):
            storage.retire_extraction_window(
                window,
                token,
                expected_attempts=current["cursor_attempts"],
                expected_last_error=current["last_error"],
            )
    finally:
        storage.release_user_extraction("u", token)
    assert _state(storage, window_id) == before
    assert _window_row(storage, window_id)["completed"] == 0


def test_retire_refuses_a_stale_hold(pipeline, monkeypatch):
    storage, _, window_id, _ = _park_legacy_mismatch(pipeline, monkeypatch)
    window, current = _inspect(storage, window_id, action="retire")
    before = _state(storage, window_id)
    token = storage.claim_user_extraction("u", "test-retire", 300)
    assert token is not None
    try:
        with pytest.raises(OverflowCheckpointConflictError):
            storage.retire_extraction_window(
                window,
                token,
                expected_attempts=current["cursor_attempts"] + 1,
                expected_last_error=current["last_error"],
            )
    finally:
        storage.release_user_extraction("u", token)
    assert _state(storage, window_id) == before


def test_retire_needs_no_recoverable_parent(pipeline, monkeypatch):
    """A window whose history cannot be re-armed can still be retired."""
    storage, _, window_id, _ = _park_legacy_mismatch(pipeline, monkeypatch)
    with storage._stream_sql() as db:
        db.query("DELETE FROM _agent_runs WHERE id=?", (f"window:{window_id}",))
    rearm = plan_input_budget_recovery(storage, window_id, action="reset_legacy_policy")
    assert rearm["reason"] == "missing_parent_unknown_history"
    receipt = plan_input_budget_recovery(storage, window_id, action="retire")
    assert receipt["recoverable"], receipt
    apply_input_budget_recovery(storage, receipt)
    row = _window_row(storage, window_id)
    assert row["completed"] == 1 and row["effects_done"] == 1
    with storage._stream_sql(read_only=True) as db:
        cursor = db.query(
            "SELECT completed_seq,window_id FROM extraction_cursors "
            "WHERE user_id='u' AND kind='profile'"
        )[0]
    assert cursor == {"completed_seq": row["end_seq"], "window_id": None}
