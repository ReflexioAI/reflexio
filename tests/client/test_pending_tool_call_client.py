"""ReflexioClient methods for pending tool calls (human questions)."""
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from reflexio.client import ReflexioClient
from reflexio.models.config_schema import Config, StorageConfigSQLite
from reflexio.server.api import create_app
from reflexio.server.api_endpoints.request_context import (
    RequestContext,
    get_request_context,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import (
    PendingToolCallRecord,
    PendingToolCallStatus,
)

ORG_ID = "test-pending-tool-call-org"


@pytest.fixture()
def storage():
    """A real SQLiteStorage instance in a temporary directory."""
    with (
        tempfile.TemporaryDirectory() as tmp,
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
    ):
        yield SQLiteStorage(org_id=ORG_ID, db_path=f"{tmp}/reflexio.db")


@pytest.fixture()
def client_with_server(storage):
    """ReflexioClient pointed at a TestClient-backed FastAPI app.

    The app's ``get_request_context`` dependency is overridden to use the
    same ``SQLiteStorage`` instance as the ``storage`` fixture, so tests can
    seed pending tool calls via ``storage.create_pending_tool_call(...)`` and
    operate on them through the client.
    """
    app = create_app(get_org_id=lambda: ORG_ID)

    fake_ctx = RequestContext.__new__(RequestContext)
    fake_ctx.org_id = ORG_ID
    fake_ctx.storage = storage
    fake_ctx.configurator = MagicMock()
    fake_ctx.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite()
    )

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
            path = url.replace("http://testserver", "")
            kwargs.pop("timeout", None)
            return http_client.request(method, path, **kwargs)

    reflexio_client = ReflexioClient.__new__(ReflexioClient)
    reflexio_client.base_url = "http://testserver"
    reflexio_client.api_key = ""
    reflexio_client.timeout = 30
    reflexio_client.session = _TestClientSession()

    return reflexio_client


def _seed_ask_human(storage, call_id: str = "ptc-1") -> PendingToolCallRecord:
    """Insert a pending ``ask_human`` tool call and return the stored record."""
    now = datetime.now(UTC)
    future = now + timedelta(hours=1)
    record = PendingToolCallRecord(
        id=call_id,
        org_id=ORG_ID,
        scope={"user_id": "alice"},
        scope_hash=f"hash-{call_id}",
        tool_name="ask_human",
        dedup_key=f"dedup-{call_id}",
        status=PendingToolCallStatus.PENDING,
        question_text="What deployment target is canonical?",
        user_id="alice",
        created_at=now,
        expires_at=future,
        cache_until=future,
    )
    return storage.create_pending_tool_call(record)


def test_list_and_get_pending_tool_call(client_with_server, storage):
    _seed_ask_human(storage)

    listed = client_with_server.list_pending_tool_calls(status="pending")
    assert len(listed.pending_tool_calls) == 1
    row = listed.pending_tool_calls[0]
    assert row.id == "ptc-1"
    assert row.tool_name == "ask_human"
    assert row.status == PendingToolCallStatus.PENDING

    fetched = client_with_server.get_pending_tool_call("ptc-1")
    assert fetched.id == "ptc-1"
    assert fetched.question_text == "What deployment target is canonical?"


def test_resolve_with_answer(client_with_server, storage):
    _seed_ask_human(storage)

    resolved = client_with_server.resolve_pending_tool_call("ptc-1", answer="AWS ECS")
    assert resolved.status == PendingToolCallStatus.RESOLVED
    assert resolved.result == {"answer": "AWS ECS"}


def test_resolve_with_raw_result(client_with_server, storage):
    _seed_ask_human(storage)

    resolved = client_with_server.resolve_pending_tool_call(
        "ptc-1", result={"answer": "GCP", "extra": True}
    )
    assert resolved.status == PendingToolCallStatus.RESOLVED
    assert resolved.result == {"answer": "GCP", "extra": True}


def test_resolve_requires_exactly_one_of_result_or_answer(client_with_server, storage):
    _seed_ask_human(storage)

    with pytest.raises(ValueError):
        client_with_server.resolve_pending_tool_call("ptc-1")
    with pytest.raises(ValueError):
        client_with_server.resolve_pending_tool_call(
            "ptc-1", result={"answer": "x"}, answer="y"
        )


def test_update_answer_after_resolve(client_with_server, storage):
    _seed_ask_human(storage)
    client_with_server.resolve_pending_tool_call("ptc-1", answer="AWS ECS")

    edited = client_with_server.update_pending_tool_call_answer(
        "ptc-1", "Google Cloud Run"
    )
    assert edited.status == PendingToolCallStatus.RESOLVED
    assert edited.result == {"answer": "Google Cloud Run"}


def test_mark_not_applicable(client_with_server, storage):
    _seed_ask_human(storage)

    na = client_with_server.mark_pending_tool_call_not_applicable("ptc-1")
    assert na.status == PendingToolCallStatus.RESOLVED
    assert na.result is not None
    assert na.result.get("not_applicable") is True


def test_cancel_pending_tool_call(client_with_server, storage):
    _seed_ask_human(storage)

    cancelled = client_with_server.cancel_pending_tool_call("ptc-1")
    assert cancelled.status == PendingToolCallStatus.CANCELLED
