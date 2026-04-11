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
