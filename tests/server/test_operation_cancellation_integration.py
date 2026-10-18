"""Exact generation cancellation crosses HTTP and real storage ownership."""

from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from reflexio.lib._operations import OperationsMixin
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.auth import default_get_org_id
from reflexio.server.cache import reflexio_cache
from reflexio.server.rate_limit import limiter
from reflexio.server.routes.system import router
from reflexio.server.services.operation_state_utils import OperationStateManager
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage


@pytest.fixture
def cancellation_cell(tmp_path, monkeypatch):
    stores = {
        org: SQLiteStorage(org_id=org, db_path=str(tmp_path / f"{org}.db"))
        for org in ("owner", "other")
    }

    def api_for_org(*, org_id):
        api = object.__new__(OperationsMixin)
        api.request_context = cast(
            RequestContext,
            SimpleNamespace(
                org_id=org_id,
                storage=stores[org_id],
                is_storage_configured=lambda: True,
            ),
        )
        return api

    def authenticated_org(request: Request):
        return request.headers.get("x-test-org", "owner")

    monkeypatch.setattr(reflexio_cache, "get_reflexio", api_for_org)
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(router)
    app.dependency_overrides[default_get_org_id] = authenticated_org
    try:
        with TestClient(app, client=(f"cancellation-{uuid4().hex}", 50000)) as client:
            yield client, stores
    finally:
        for storage in stores.values():
            storage.conn.close()


@pytest.mark.parametrize("service", ["profile_generation", "playbook_generation"])
def test_exact_attempt_cancellation_and_next_admission(cancellation_cell, service):
    client, stores = cancellation_cell
    admitted = OperationStateManager(stores["owner"], "owner", service)
    assert admitted.claim_progress({"user_id": "launch_test_user"})
    response = client.post(
        f"/api/cancel_operation/{admitted.operation_id}", json={"service_name": service}
    )
    assert response.status_code == 200, response.text
    assert response.json()["cancelled_services"] == [service]
    assert admitted.is_cancellation_requested()
    admitted.mark_cancelled()
    admitted.finalize_progress(0, 0)
    # Terminal progress needs no raw row/lock deletion to admit another attempt.
    successor = OperationStateManager(stores["owner"], "owner", service)
    assert successor.claim_progress({"user_id": "launch_test_next"})
    assert not successor.is_cancellation_requested()


def test_wrong_or_foreign_attempt_cannot_cancel_current_owner(cancellation_cell):
    client, stores = cancellation_cell
    admitted = OperationStateManager(stores["owner"], "owner", "profile_generation")
    other = OperationStateManager(stores["other"], "other", "profile_generation")
    assert admitted.claim_progress({})
    assert other.claim_progress({})
    for attempt, org in ((str(uuid4()), "owner"), (admitted.operation_id, "other")):
        response = client.post(
            f"/api/cancel_operation/{attempt}",
            headers={"x-test-org": org},
            json={"service_name": "profile_generation"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["cancelled_services"] == []
        assert not admitted.is_cancellation_requested()
        assert not other.is_cancellation_requested()


def test_old_attempt_cannot_cancel_a_successor(cancellation_cell):
    client, stores = cancellation_cell
    old = OperationStateManager(stores["owner"], "owner", "profile_generation")
    assert old.claim_progress({})
    old.finalize_progress(0, 0)
    current = OperationStateManager(stores["owner"], "owner", "profile_generation")
    assert current.claim_progress({})
    response = client.post(
        f"/api/cancel_operation/{old.operation_id}",
        json={"service_name": "profile_generation"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["cancelled_services"] == []
    assert not current.is_cancellation_requested()


def test_progress_recovery_is_not_worker_completion(cancellation_cell):
    client, stores = cancellation_cell
    admitted = OperationStateManager(stores["owner"], "owner", "profile_generation")
    assert admitted.claim_progress({"user_id": "launch_test_user"})
    state = admitted.get_progress()
    assert state is not None
    state["started_at"] = 0
    stores["owner"].update_operation_state(admitted._progress_key(), state)
    path = "/api/generation_operations/profile_generation"
    observed = client.get(path).json()["operation_status"]
    assert observed["status"] == "in_progress"
    assert not observed.get("finished_operation_id")
    # The legacy UI endpoint still performs its historical metadata recovery.
    recovered = client.get("/api/get_operation_status").json()["operation_status"]
    assert recovered["status"] == "failed"
    assert not recovered.get("finished_operation_id")
    response = client.post(
        f"/api/cancel_operation/{admitted.operation_id}",
        json={"service_name": "profile_generation"},
    )
    assert response.json()["cancelled_services"] == ["profile_generation"]
    assert admitted.is_cancellation_requested()
    admitted.mark_cancelled()
    assert not client.get(path).json()["operation_status"].get("finished_operation_id")
    admitted.finalize_progress(0, 0)
    assert (
        client.get(path).json()["operation_status"]["finished_operation_id"]
        == admitted.operation_id
    )


def test_owner_failure_acknowledges_finished_execution(cancellation_cell):
    client, stores = cancellation_cell
    admitted = OperationStateManager(stores["owner"], "owner", "profile_generation")
    assert admitted.claim_progress({})
    admitted.mark_progress_failed("selection failed")
    observed = client.get("/api/generation_operations/profile_generation").json()[
        "operation_status"
    ]
    assert observed["status"] == "failed"
    assert observed["finished_operation_id"] == admitted.operation_id
    response = client.post(
        f"/api/cancel_operation/{admitted.operation_id}",
        json={"service_name": "profile_generation"},
    )
    assert response.json()["cancelled_services"] == []


def test_completed_attempt_survives_successor_admission(cancellation_cell):
    client, stores = cancellation_cell
    old = OperationStateManager(stores["owner"], "owner", "profile_generation")
    assert old.claim_progress({"user_id": "private-user", "source": "private-filter"})
    old.mark_cancelled()
    old.finalize_progress(4, 9)
    completed = old.get_attempt_progress(old.operation_id)
    old.finalize_progress(100, 100)
    assert old.get_attempt_progress(old.operation_id) == completed
    current = OperationStateManager(stores["owner"], "owner", "profile_generation")
    assert current.claim_progress({"user_id": "successor"})
    response = client.get(
        f"/api/generation_operations/profile_generation/{old.operation_id}"
    )
    assert response.status_code == 200, response.text
    completion = response.json()["operation_status"]
    assert completion["status"] == "cancelled"
    assert completion["finished_operation_id"] == old.operation_id
    assert completion["request_params"] == {"operation_id": old.operation_id}
    assert completion["stats"]["total_generated"] == 9
    # No user/filter metadata or identifier lists enter the retained receipt.
    assert old.operation_id is not None
    receipt = stores["owner"].get_operation_state(old._completion_key(old.operation_id))
    assert receipt is not None
    assert "private-user" not in str(receipt) and "private-filter" not in str(receipt)
    foreign = client.get(
        f"/api/generation_operations/profile_generation/{old.operation_id}",
        headers={"x-test-org": "other"},
    )
    assert foreign.status_code == 200 and foreign.json() == {"success": True}
    assert not current.is_cancellation_requested()


def test_displaced_worker_finishes_only_its_own_execution(cancellation_cell):
    client, stores = cancellation_cell
    old = OperationStateManager(stores["owner"], "owner", "profile_generation")
    assert old.claim_progress({})
    state = old.get_progress()
    assert state is not None
    state["started_at"] = 0
    stores["owner"].update_operation_state(old._progress_key(), state)
    current = OperationStateManager(stores["owner"], "owner", "profile_generation")
    assert current.claim_progress({})
    before = current.get_progress()
    with pytest.raises(RuntimeError, match="no longer owns"):
        old.set_current_item("late-user")
    response = client.post(
        f"/api/cancel_operation/{old.operation_id}",
        json={"service_name": "profile_generation"},
    )
    assert response.json()["cancelled_services"] == ["profile_generation"]
    assert old.is_cancellation_requested() and not current.is_cancellation_requested()
    old.mark_cancelled()
    old.finalize_progress(0, 0)
    assert current.get_progress() == before
    response = client.get(
        f"/api/generation_operations/profile_generation/{old.operation_id}"
    )
    assert response.status_code == 200
    assert (
        response.json()["operation_status"]["finished_operation_id"] == old.operation_id
    )
    assert response.json()["operation_status"]["status"] == "cancelled"
    progress = current.get_progress()
    assert (
        progress is not None
        and progress["request_params"]["operation_id"] == current.operation_id
    )


def test_receipt_write_failure_rolls_back_completion(cancellation_cell, monkeypatch):
    _, stores = cancellation_cell
    owner = OperationStateManager(stores["owner"], "owner", "profile_generation")
    assert owner.claim_progress({})
    before = owner.get_progress()

    receipt_before = owner.get_attempt_progress(owner.operation_id)
    replace = stores["owner"].clear_in_progress_lock_if_owner

    def fail_mirror(key, operation_id, state):
        if key == owner._progress_key():
            raise RuntimeError("receipt storage failed")
        return replace(key, operation_id, state)

    monkeypatch.setattr(stores["owner"], "clear_in_progress_lock_if_owner", fail_mirror)
    with pytest.raises(RuntimeError, match="receipt storage failed"):
        owner.finalize_progress(0, 0)
    assert owner.get_progress() == before
    assert owner.operation_id is not None
    assert owner.get_attempt_progress(owner.operation_id) == receipt_before


@pytest.mark.parametrize(
    "payload", [{}, {"service_name": ""}, {"service_name": "other"}]
)
def test_exact_cancellation_requires_a_generation_service(cancellation_cell, payload):
    client, _ = cancellation_cell
    response = client.post(f"/api/cancel_operation/{uuid4()}", json=payload)
    assert response.status_code == 422


def test_failed_admission_does_not_leave_claim_or_execution(
    cancellation_cell, monkeypatch
):
    _, stores = cancellation_cell
    owner = OperationStateManager(stores["owner"], "owner", "profile_generation")

    def fail_receipt(*_args, **_kwargs):
        raise RuntimeError("admission storage failed")

    with monkeypatch.context() as patch:
        patch.setattr(stores["owner"], "create_operation_state", fail_receipt)
        with pytest.raises(RuntimeError, match="admission storage failed"):
            owner.claim_progress({})
    assert owner.get_progress() is None
    assert owner.get_attempt_progress(owner.operation_id) is None
    next_owner = OperationStateManager(stores["owner"], "owner", "profile_generation")
    assert next_owner.claim_progress({})


def test_finished_execution_cannot_be_resurrected(cancellation_cell):
    _, stores = cancellation_cell
    owner = OperationStateManager(stores["owner"], "owner", "profile_generation")
    assert owner.claim_progress({})
    owner.finalize_progress(0)
    before = owner.get_attempt_progress(owner.operation_id)
    with pytest.raises(RuntimeError, match="no longer owns"):
        owner.set_current_item("late-user")
    assert owner.get_attempt_progress(owner.operation_id) == before


def test_pre_upgrade_progress_is_not_execution_completion(cancellation_cell):
    client, stores = cancellation_cell
    stores["owner"].upsert_operation_state(
        "profile_generation::owner::progress",
        {"status": "failed", "request_params": {}, "completed_at": 1},
    )
    response = client.get("/api/generation_operations/profile_generation")
    assert response.status_code == 200
    assert response.json()["success"] is False
    assert "no durable execution record" in response.json()["msg"]
