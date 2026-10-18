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


@pytest.mark.parametrize(
    "payload", [{}, {"service_name": ""}, {"service_name": "other"}]
)
def test_exact_cancellation_requires_a_generation_service(cancellation_cell, payload):
    client, _ = cancellation_cell
    response = client.post(f"/api/cancel_operation/{uuid4()}", json=payload)
    assert response.status_code == 422
