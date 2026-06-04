"""Contract tests for OperationMixin — run against every local storage backend."""

import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# TestOperationStateCRUD
# ---------------------------------------------------------------------------


class TestOperationStateCRUD:
    def test_create_and_get_operation_state(self, storage):
        state = {"status": "running", "progress": 50}
        storage.create_operation_state("svc_a", state)

        result = storage.get_operation_state("svc_a")
        assert result is not None
        assert result["operation_state"]["status"] == "running"
        assert result["operation_state"]["progress"] == 50

    def test_get_nonexistent_returns_none(self, storage):
        assert storage.get_operation_state("missing") is None

    def test_upsert_creates_new(self, storage):
        state = {"status": "idle"}
        storage.upsert_operation_state("svc_new", state)

        result = storage.get_operation_state("svc_new")
        assert result is not None
        assert result["operation_state"]["status"] == "idle"

    def test_upsert_updates_existing(self, storage):
        storage.create_operation_state("svc_up", {"status": "running", "progress": 10})
        storage.upsert_operation_state("svc_up", {"status": "done", "progress": 100})

        result = storage.get_operation_state("svc_up")
        assert result is not None
        assert result["operation_state"]["status"] == "done"
        assert result["operation_state"]["progress"] == 100

    def test_update_operation_state(self, storage):
        storage.create_operation_state("svc_upd", {"status": "running"})
        storage.update_operation_state(
            "svc_upd", {"status": "completed", "result": "ok"}
        )

        result = storage.get_operation_state("svc_upd")
        assert result is not None
        assert result["operation_state"]["status"] == "completed"
        assert result["operation_state"]["result"] == "ok"

    def test_delete_operation_state(self, storage):
        storage.create_operation_state("svc_del", {"status": "running"})
        storage.delete_operation_state("svc_del")

        assert storage.get_operation_state("svc_del") is None

    def test_delete_all_operation_states(self, storage):
        storage.create_operation_state("svc_1", {"status": "a"})
        storage.create_operation_state("svc_2", {"status": "b"})

        storage.delete_all_operation_states()

        assert storage.get_operation_state("svc_1") is None
        assert storage.get_operation_state("svc_2") is None

    def test_get_all_operation_states(self, storage):
        storage.create_operation_state("svc_x", {"status": "x"})
        storage.create_operation_state("svc_y", {"status": "y"})

        all_states = storage.get_all_operation_states()
        assert len(all_states) == 2


# ---------------------------------------------------------------------------
# TestPendingRequestQueue — R2
# ---------------------------------------------------------------------------


class TestPendingRequestQueue:
    """Contract: when the lock is held, blocked requests queue FIFO with payloads."""

    def _state(self, storage, key):
        record = storage.get_operation_state(key)
        if record is None:
            return None
        return record.get("operation_state", record)

    def test_first_acquire_creates_empty_queue(self, storage):
        result = storage.try_acquire_in_progress_lock("svc_lock_1", "req_1")
        assert result["acquired"] is True

        state = self._state(storage, "svc_lock_1")
        assert state is not None
        assert state.get("pending_request_queue", []) == []

    def test_blocked_request_appends_to_queue(self, storage):
        storage.try_acquire_in_progress_lock("svc_lock_2", "req_1")
        result = storage.try_acquire_in_progress_lock(
            "svc_lock_2", "req_2", payload={"user_id": "user_b"}
        )
        assert result["acquired"] is False

        state = self._state(storage, "svc_lock_2")
        queue = state.get("pending_request_queue", [])
        assert len(queue) == 1
        assert queue[0]["request_id"] == "req_2"
        assert queue[0]["payload"] == {"user_id": "user_b"}

    def test_multiple_blocked_requests_queue_fifo(self, storage):
        storage.try_acquire_in_progress_lock("svc_lock_3", "req_1")
        storage.try_acquire_in_progress_lock(
            "svc_lock_3", "req_2", payload={"user_id": "user_b"}
        )
        storage.try_acquire_in_progress_lock(
            "svc_lock_3", "req_3", payload={"user_id": "user_c"}
        )

        state = self._state(storage, "svc_lock_3")
        queue = state.get("pending_request_queue", [])
        assert [q["request_id"] for q in queue] == ["req_2", "req_3"]
        assert queue[0]["payload"] == {"user_id": "user_b"}
        assert queue[1]["payload"] == {"user_id": "user_c"}

    def test_queue_ignores_duplicate_request_id(self, storage):
        """If the same request_id retries while blocked (e.g. publish retry)
        we should not enqueue it twice. Pre-fix, the single slot just got
        overwritten; queue must dedupe by request_id."""
        storage.try_acquire_in_progress_lock("svc_lock_4", "req_1")
        storage.try_acquire_in_progress_lock(
            "svc_lock_4", "req_2", payload={"user_id": "user_b"}
        )
        storage.try_acquire_in_progress_lock(
            "svc_lock_4", "req_2", payload={"user_id": "user_b_v2"}
        )

        state = self._state(storage, "svc_lock_4")
        queue = state.get("pending_request_queue", [])
        # Only one entry for req_2 — second attempt is a noop.
        assert [q["request_id"] for q in queue] == ["req_2"]

    def test_holder_retry_does_not_self_enqueue(self, storage):
        """Holder calling try_acquire again for its own request_id is a noop."""
        storage.try_acquire_in_progress_lock("svc_lock_5", "req_1")
        result = storage.try_acquire_in_progress_lock("svc_lock_5", "req_1")
        # Same request — treated as already-acquired.
        assert result["acquired"] is True

        state = self._state(storage, "svc_lock_5")
        assert state.get("pending_request_queue", []) == []
