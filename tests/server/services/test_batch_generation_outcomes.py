"""Batch completion must reflect every selected user's outcome."""

from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from reflexio.models.api_schema.service_schemas import OperationStatus
from reflexio.server.services.base_generation._batch_progress import BatchProgressMixin
from reflexio.server.services.operation_state_utils import OperationStateManager


class BatchService(BatchProgressMixin[dict]):
    def __init__(self, failed_users=(), cancel_after=None, generated_per_user=2):
        self._is_batch_mode = False
        self.failed_users = failed_users
        self.cancel_after = cancel_after
        self.generated_per_user = generated_per_user
        self.generated = []
        self.rows = {}
        storage = MagicMock()
        self.mock_storage = storage
        storage.get_operation_state.side_effect = lambda key: deepcopy(
            self.rows.get(key)
        )
        storage.create_operation_state.side_effect = self._save_state
        storage.upsert_operation_state.side_effect = self._save_state
        storage.update_operation_state.side_effect = self._save_state
        storage.try_acquire_in_progress_lock.side_effect = self._claim
        storage.clear_in_progress_lock_if_owner.side_effect = self._replace_owned
        storage.delete_operation_state.side_effect = lambda key: self.rows.pop(
            key, None
        )
        self.manager = OperationStateManager(storage, "batch-org", "batch-test")

    def _save_state(self, key, state):
        self.rows[key] = deepcopy(state)

    def _claim(self, key, request_id, **_kwargs):
        current = self.rows.get(key, {})
        if current.get("in_progress") or current.get("status") == "in_progress":
            return {"acquired": False}
        self.rows[key] = {"current_request_id": request_id, "in_progress": True}
        return {"acquired": True}

    def _replace_owned(self, key, request_id, state):
        if self.rows.get(key, {}).get("current_request_id") != request_id:
            return False
        self._save_state(key, state)
        return True

    def _get_base_service_name(self):
        return "batch-test"

    def _create_state_manager(self):
        return self.manager

    def _get_rerun_user_ids(self, request):
        return request["users"]

    def _build_rerun_request_params(self, request):
        return request

    def _pre_process_rerun(self, request):
        pass

    def _create_run_request_for_item(self, user_id, request):
        return {"user": user_id}

    def _create_rerun_response(self, success, msg, count):
        return {"success": success, "msg": msg, "count": count}

    def _get_generated_count(self, request, processed_user_ids=None):
        return sum(u in (processed_user_ids or []) for u in self.generated)

    def run(self, request):
        user = request["user"]
        if user in self.failed_users:
            raise RuntimeError("invalid model evidence")
        self.generated.extend([user] * self.generated_per_user)
        if self.cancel_after == user:
            assert self.manager.request_cancellation()


@pytest.mark.parametrize("failed", [(), ("a",), ("a", "b")])
def test_rerun_outcome_and_counts(failed):
    service = BatchService(failed_users=failed)
    response = service.run_rerun({"users": ["a", "b"]})
    state = service.manager.get_progress()
    assert state is not None
    assert response["success"] is (not failed)
    assert response["count"] == (2 - len(failed)) * 2
    assert state["status"] == (
        OperationStatus.FAILED.value if failed else OperationStatus.COMPLETED.value
    )
    assert state["processed_users"] == 2 - len(failed)
    assert state["failed_users"] == len(failed)
    assert state["stats"]["total_generated"] == response["count"]
    if failed:
        assert state["error_message"] == f"Generation failed for {len(failed)} user(s)"
    assert not service._is_batch_mode


def test_cancellation_retains_output_and_statistics():
    service = BatchService(cancel_after="a")
    response = service.run_rerun({"users": ["a", "b"]})
    state = service.manager.get_progress()
    assert state is not None
    assert response["success"] is False
    assert response["count"] == 2
    assert state["status"] == OperationStatus.CANCELLED.value
    assert state["processed_users"] == 1
    assert state["progress_percentage"] == 50
    assert state["stats"] == {"total_interactions_processed": 1, "total_generated": 2}
    assert service.generated == ["a", "a"]
    assert not service.manager.is_cancellation_requested()


def test_successful_zero_output_is_complete():
    service = BatchService(generated_per_user=0)
    response = service.run_rerun({"users": ["a"]})
    assert response["success"] is True
    assert response["count"] == 0
    state = service.manager.get_progress()
    assert state is not None
    assert state["status"] == OperationStatus.COMPLETED.value


def test_progress_write_failure_cannot_count_user_as_success(monkeypatch):
    service = BatchService()
    update = service.manager.update_progress

    def fail_success_write(*args, **kwargs):
        if kwargs["success"]:
            raise RuntimeError("progress write failed")
        return update(*args, **kwargs)

    monkeypatch.setattr(service.manager, "update_progress", fail_success_write)
    response = service.run_rerun({"users": ["a"]})
    state = service.manager.get_progress()
    assert state is not None
    assert response["success"] is False
    assert state["processed_users"] == 0
    assert state["failed_users"] == 1
    assert state["status"] == OperationStatus.FAILED.value
    # Output commits are retained and counted, but completion was not confirmed.
    assert service.generated == ["a", "a"]
    assert response["count"] == 2
    assert state["stats"]["total_generated"] == 2


@pytest.mark.parametrize(
    "failure", ["cold_collision", "other_table", "in_scope", "other_code"]
)
def test_only_cold_operation_slot_collision_retries_claim(failure):
    from types import SimpleNamespace

    from reflexio.server.services.storage.error import StorageError

    service = BatchService()
    key = service.manager._progress_key()
    service.rows[key] = {"current_request_id": "winner", "in_progress": True}

    class DatabaseCollisionError(RuntimeError):
        pgcode: str
        diag: SimpleNamespace

    cause = DatabaseCollisionError("database collision")
    cause.pgcode = "40001" if failure == "other_code" else "23505"
    cause.diag = SimpleNamespace(
        table_name="other" if failure == "other_table" else "_operation_state",
        constraint_name="_operation_state_pkey",
    )
    error = StorageError("claim failed")
    error.__cause__ = cause
    storage = service.mock_storage
    storage._scope_conn = object() if failure == "in_scope" else None
    storage.try_acquire_in_progress_lock.side_effect = [error, {"acquired": False}]
    if failure == "cold_collision":
        assert not service.manager.claim_progress({})
        assert storage.try_acquire_in_progress_lock.call_count == 2
    else:
        with pytest.raises(StorageError, match="claim failed"):
            service.manager.claim_progress({})
        assert storage.try_acquire_in_progress_lock.call_count == 1
