"""HTTP-level tests for /stall_state endpoints."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from reflexio.server.api import create_app
from reflexio.server.api_endpoints.request_context import (
    RequestContext,
    get_request_context,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage


@pytest.fixture()
def storage():
    """A real SQLiteStorage instance in a temporary directory."""
    with (
        tempfile.TemporaryDirectory() as tmp,
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
    ):
        yield SQLiteStorage(org_id="test-stall", db_path=f"{tmp}/reflexio.db")


@pytest.fixture()
def client(storage):
    """TestClient wired to the same storage instance as the ``storage`` fixture."""
    app = create_app(get_org_id=lambda: "test-stall-org")

    # Build a fake RequestContext whose .storage is our real SQLiteStorage.
    fake_ctx = RequestContext.__new__(RequestContext)
    fake_ctx.org_id = "test-stall-org"
    fake_ctx.storage = storage

    app.dependency_overrides[get_request_context] = lambda: fake_ctx

    return TestClient(app, raise_server_exceptions=True)


def test_get_stall_state_clean(client: TestClient):
    resp = client.get("/stall_state")
    assert resp.status_code == 200
    body = resp.json()
    assert body["stalled"] is False
    assert body["reason"] is None


def test_get_stall_state_when_stalled(client: TestClient, storage):
    storage.upsert_stall_state(
        reason="billing_error",
        stalled_at=datetime.now(UTC),
        reset_estimate=None,
        error_message="credit exhausted",
    )
    resp = client.get("/stall_state")
    assert resp.status_code == 200
    body = resp.json()
    assert body["stalled"] is True
    assert body["reason"] == "billing_error"
    assert body["notified_in_cc"] is False


def test_post_notified_flips_flag(client: TestClient, storage):
    storage.upsert_stall_state(
        reason="auth_error",
        stalled_at=datetime.now(UTC),
        reset_estimate=None,
        error_message="login",
    )
    resp = client.post("/stall_state/notified")
    assert resp.status_code == 200
    assert resp.json()["notified_in_cc"] is True
