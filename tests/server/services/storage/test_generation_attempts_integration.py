"""Admitted batch ownership and cancellation through actual SQLite storage."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any, cast

import pytest

from reflexio.server.services.operation_state_utils import OperationStateManager
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def manager(storage):
    return OperationStateManager(storage, "attempt-org", "playbook_generation")


def test_concurrent_cold_claims_across_storage_instances(tmp_path):
    path = str(tmp_path / "shared.db")
    cells = [SQLiteStorage(org_id="attempt-org", db_path=path) for _ in range(2)]
    gate = Barrier(2)

    def claim(cell):
        contender = manager(cell)
        gate.wait(timeout=5)
        return contender, contender.claim_progress({})

    with ThreadPoolExecutor(max_workers=2) as workers:
        outcomes = list(workers.map(claim, cells))
    assert sorted(success for _, success in outcomes) == [False, True]
    winner = next(contender for contender, success in outcomes if success)
    state = winner.get_progress()
    assert state is not None
    assert state["request_params"]["operation_id"] == winner.operation_id
    winner.finalize_progress(0, 0)
    assert winner.storage.get_operation_state(winner._cancellation_key()) is None


def test_rejected_claim_preserves_the_admitted_attempt(tmp_path):
    storage = SQLiteStorage(org_id="attempt-org", db_path=str(tmp_path / "batch.db"))
    first, second = manager(storage), manager(storage)
    assert first.claim_progress({"agent_version": "v1"})
    first.initialize_progress(total_users=2, request_params={"agent_version": "v1"})
    first.update_progress("u1", count=0, success=True, total_users=2)
    assert not second.claim_progress({"agent_version": "other"})
    second.mark_progress_failed("rejected contender")
    assert second.get_attempt_progress(second.operation_id) is None
    with pytest.raises(RuntimeError, match="not admitted"):
        second.finalize_progress(0)
    state = first.get_progress()
    assert state is not None
    assert state["request_params"]["operation_id"] == first.operation_id
    assert state["status"] == "in_progress"
    assert state["processed_user_ids"] == ["u1"]
    first.finalize_progress(1, 3)
    final = first.get_progress()
    assert final is not None
    assert final["pending_request_queue"] == []


def test_cancellation_during_selection_survives_batch_initialization(tmp_path):
    storage = SQLiteStorage(org_id="attempt-org", db_path=str(tmp_path / "cancel.db"))
    admitted = manager(storage)
    assert admitted.claim_progress({})
    assert manager(storage).request_cancellation()
    admitted.initialize_progress(total_users=2, request_params={})
    assert admitted.is_cancellation_requested()
    admitted.mark_cancelled()
    admitted.finalize_progress(0, 0)
    assert storage.get_operation_state(admitted._cancellation_key()) is None
    state = admitted.get_progress()
    assert state is not None
    assert state["status"] == "cancelled"


def test_replaced_worker_cannot_write_new_attempt_progress(tmp_path):
    storage = SQLiteStorage(org_id="attempt-org", db_path=str(tmp_path / "replace.db"))
    old = manager(storage)
    assert old.claim_progress({})
    expired = old.get_progress()
    assert expired is not None
    expired["started_at"] = 0
    storage.update_operation_state(old._progress_key(), expired)
    new = manager(storage)
    assert new.claim_progress({})
    before = new.get_progress()
    with pytest.raises(RuntimeError, match="no longer owns"):
        old.set_current_item("late-user")
    old.finalize_progress(1, 99)
    receipt = old.get_attempt_progress(old.operation_id)
    assert receipt is not None and receipt["finished_operation_id"] == old.operation_id
    old.mark_progress_failed("old failure")
    assert old.get_attempt_progress(old.operation_id) == receipt
    assert new.get_progress() == before
    # Cancellation is scoped to the admitted attempt, rather than old flags.
    storage.upsert_operation_state(
        old._cancellation_key(), {"cancellation_requested": True}
    )
    assert not new.is_cancellation_requested()


@pytest.mark.parametrize("stage", ["selection", "preparation", "no_work"])
def test_admitted_early_failure_retains_attempt_and_releases_claim(
    tmp_path, monkeypatch, stage
):
    from unittest.mock import MagicMock

    from reflexio.models.api_schema.service_schemas import (
        RerunPlaybookGenerationRequest,
    )
    from reflexio.server.services.playbook.service import PlaybookGenerationService

    storage = SQLiteStorage(org_id="attempt-org", db_path=str(tmp_path / "early.db"))
    admitted = manager(storage)
    assert admitted.claim_progress({})
    payload = RerunPlaybookGenerationRequest(agent_version="v1")
    payload._operation_id = admitted.operation_id
    context = MagicMock(org_id="attempt-org", storage=storage)
    service = PlaybookGenerationService(llm_client=MagicMock(), request_context=context)

    def select(_request):
        if stage == "selection":
            raise RuntimeError("selection failed")
        return [] if stage == "no_work" else ["user"]

    def prepare(_request):
        raise RuntimeError("preparation failed")

    monkeypatch.setattr(service, "_get_rerun_user_ids", select)
    monkeypatch.setattr(service, "EMITS_LEARNING_BILLING", False)
    monkeypatch.setattr(service, "_pre_process_rerun", prepare)
    # Match lib._generation's reflective dispatch of the rerun request schema.
    response = cast(Any, service).run_rerun(payload)
    assert response.success is False
    state = admitted.get_progress()
    assert state is not None
    assert state["status"] == "failed"
    assert state["request_params"]["operation_id"] == admitted.operation_id
    assert not state["in_progress"]
    assert storage.get_operation_state(admitted._cancellation_key()) is None
    assert manager(storage).claim_progress({})
