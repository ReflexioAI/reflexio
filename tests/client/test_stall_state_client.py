"""ReflexioClient methods for stall_state."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from reflexio.client import ReflexioClient
from reflexio.server.api import create_app
from reflexio.server.api_endpoints.request_context import RequestContext, get_request_context
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage


@pytest.fixture()
def storage():
    """A real SQLiteStorage instance in a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
            yield SQLiteStorage(org_id="test-stall-client", db_path=f"{tmp}/reflexio.db")


@pytest.fixture()
def client_with_server(storage):
    """ReflexioClient pointed at a TestClient-backed FastAPI app.

    The app's ``get_request_context`` dependency is overridden to use the
    same ``SQLiteStorage`` instance as the ``storage`` fixture, so tests can
    seed state via ``storage.upsert_stall_state(...)`` and observe it through
    the client.
    """
    app = create_app(get_org_id=lambda: "test-stall-client-org")

    fake_ctx = RequestContext.__new__(RequestContext)
    fake_ctx.org_id = "test-stall-client-org"
    fake_ctx.storage = storage

    app.dependency_overrides[get_request_context] = lambda: fake_ctx

    http_client = TestClient(app, raise_server_exceptions=True)

    # Wrap the TestClient in a requests.Session-compatible shim so ReflexioClient
    # can call it via its normal self.session.request(...) path.
    class _TestClientSession:
        """Thin adapter making TestClient look like a requests.Session."""

        max_redirects = 0
        headers: dict = {}

        def update(self, d: dict) -> None:  # noqa: D401
            self.headers.update(d)

        def request(self, method: str, url: str, **kwargs) -> object:
            # Strip the base_url prefix; TestClient works with paths only.
            path = url.replace("http://testserver", "")
            kwargs.pop("timeout", None)
            return http_client.request(method, path, **kwargs)

    reflexio_client = ReflexioClient.__new__(ReflexioClient)
    reflexio_client.base_url = "http://testserver"
    reflexio_client.api_key = ""
    reflexio_client.timeout = 30
    reflexio_client.session = _TestClientSession()

    return reflexio_client


def test_get_stall_state_when_clean(client_with_server, storage):
    state = client_with_server.get_stall_state()
    assert state.stalled is False


def test_get_stall_state_when_stalled(client_with_server, storage):
    storage.upsert_stall_state(
        reason="billing_error",
        stalled_at=datetime.now(timezone.utc),
        reset_estimate=None,
        error_message="x",
    )
    state = client_with_server.get_stall_state()
    assert state.stalled is True
    assert state.reason == "billing_error"


def test_mark_stall_notified_flips_flag(client_with_server, storage):
    storage.upsert_stall_state(
        reason="auth_error",
        stalled_at=datetime.now(timezone.utc),
        reset_estimate=None,
        error_message="x",
    )
    client_with_server.mark_stall_notified()
    state = client_with_server.get_stall_state()
    assert state.notified_in_cc is True
