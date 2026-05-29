"""HTTP-level tests for pending tool call endpoints."""

from __future__ import annotations

import hashlib
import hmac
import tempfile
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from reflexio.models.config_schema import (
    Config,
    PendingToolCallConfig,
    StorageConfigSQLite,
)
from reflexio.server.api import create_app
from reflexio.server.api_endpoints.request_context import (
    RequestContext,
    get_request_context,
)
from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import (
    AgentBinding,
    AgentRunRecord,
    AgentRunStatus,
    PendingToolCallRecord,
    PendingToolCallStatus,
    RunToolDependencyRecord,
    build_pending_tool_call_dedup_key,
    build_scope_hash,
    human_feedback_scope,
)


@pytest.fixture
def storage():
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
    ):
        yield SQLiteStorage(org_id="org_1", db_path=f"{temp_dir}/reflexio.db")


@pytest.fixture
def client(storage):
    app = create_app(get_org_id=lambda: "org_1")

    fake_ctx = RequestContext.__new__(RequestContext)
    fake_ctx.org_id = "org_1"
    fake_ctx.storage = storage
    fake_ctx.configurator = MagicMock()
    fake_ctx.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite()
    )
    app.state.fake_ctx = fake_ctx

    app.dependency_overrides[get_request_context] = lambda: fake_ctx
    return TestClient(app, raise_server_exceptions=True)


def _agent_run(run_id: str, status: AgentRunStatus) -> AgentRunRecord:
    return AgentRunRecord(
        id=run_id,
        binding=AgentBinding(
            org_id="org_1",
            extractor_kind="profile",
            extractor_name="default_profile_extractor",
            user_id="user_1",
            request_id="request_1",
            agent_version="v1",
            source="api",
            source_interaction_ids=[1, 2],
            window_start_interaction_id=1,
            window_end_interaction_id=2,
            extractor_config_hash="hash_1",
        ),
        status=status,
        generation_request_snapshot={"request_id": "request_1"},
    )


def _pending_call(
    call_id: str,
    *,
    org_id: str = "org_1",
    user_id: str | None = "user_1",
    now: datetime,
) -> PendingToolCallRecord:
    scope = human_feedback_scope(org_id)
    question = "What is the deployment target?"
    return PendingToolCallRecord(
        id=call_id,
        org_id=org_id,
        user_id=user_id,
        scope=scope,
        scope_hash=build_scope_hash(scope),
        tool_name="ask_human",
        dedup_key=build_pending_tool_call_dedup_key(
            tool_name="ask_human",
            question_text=question,
        ),
        status=PendingToolCallStatus.PENDING,
        question_text=question,
        args={"question": question},
        tags=["deployment"],
        expires_at=now + timedelta(hours=1),
        cache_until=now + timedelta(minutes=5),
    )


def test_list_pending_tool_calls_filters_to_request_org(client, storage):
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_pending_tool_call(_pending_call("ptc_1", now=now))
    storage.create_pending_tool_call(
        _pending_call("ptc_other", org_id="org_2", now=now)
    )

    response = client.get("/api/pending_tool_calls")

    assert response.status_code == 200
    body = response.json()
    assert [item["id"] for item in body["pending_tool_calls"]] == ["ptc_1"]
    assert body["pending_tool_calls"][0]["scope"] == {
        "org_id": "org_1",
        "scope_kind": "org",
    }
    assert body["pending_tool_calls"][0]["user_id"] == "user_1"


def test_list_pending_tool_calls_migrates_once_on_missing_schema_cache(
    client,
    storage,
):
    storage.migrate = MagicMock()
    storage.list_pending_tool_calls = MagicMock(
        side_effect=[
            StorageError(
                "APIError: Could not find the table 'org_1._pending_tool_calls' "
                "in the schema cache"
            ),
            [],
        ]
    )

    response = client.get("/api/pending_tool_calls")

    assert response.status_code == 200
    assert response.json() == {"pending_tool_calls": []}
    storage.migrate.assert_called_once()
    assert storage.list_pending_tool_calls.call_count == 2


def test_get_pending_tool_call_enforces_org_scope(client, storage):
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_pending_tool_call(_pending_call("ptc_1", now=now))
    storage.create_pending_tool_call(
        _pending_call("ptc_other", org_id="org_2", now=now)
    )

    assert client.get("/api/pending_tool_calls/ptc_1").status_code == 200
    assert client.get("/api/pending_tool_calls/ptc_other").status_code == 404


def test_resolve_pending_tool_call_is_idempotent_and_schedules_resume(
    client,
    storage,
):
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_agent_run(_agent_run("run_1", AgentRunStatus.FINALIZED_PENDING_TOOL))
    storage.create_pending_tool_call(_pending_call("ptc_1", now=now))
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(run_id="run_1", pending_tool_call_id="ptc_1")
    )

    payload = {"result": {"answer": "AWS ECS"}, "valid_for_seconds": 3600}
    first = client.post("/api/pending_tool_calls/ptc_1/resolve", json=payload)
    second = client.post("/api/pending_tool_calls/ptc_1/resolve", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["status"] == "resolved"
    assert first.json()["result"] == {"answer": "AWS ECS"}
    run = storage.get_agent_run("run_1")
    assert run is not None
    assert run.status == AgentRunStatus.RESUME_READY


def test_resolve_pending_tool_call_rejects_different_result(client, storage):
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_pending_tool_call(_pending_call("ptc_1", now=now))

    first = client.post(
        "/api/pending_tool_calls/ptc_1/resolve",
        json={"result": {"answer": "AWS ECS"}},
    )
    second = client.post(
        "/api/pending_tool_calls/ptc_1/resolve",
        json={"result": {"answer": "GCP"}},
    )

    assert first.status_code == 200
    assert second.status_code == 409


def test_resolve_pending_tool_call_accepts_configured_hmac_signature(client, storage):
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_pending_tool_call(_pending_call("ptc_1", now=now))
    secret = "test-secret"
    client.app.state.fake_ctx.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite(),
        pending_tool_call_config=PendingToolCallConfig(hmac_secrets=[secret]),
    )
    body = b'{"result":{"answer":"AWS ECS"}}'
    timestamp = str(int(time.time()))
    digest = hmac.new(
        secret.encode("utf-8"),
        timestamp.encode("utf-8") + b"." + body,
        hashlib.sha256,
    ).hexdigest()

    response = client.post(
        "/api/pending_tool_calls/ptc_1/resolve",
        content=body,
        headers={
            "content-type": "application/json",
            "x-reflexio-timestamp": timestamp,
            "x-reflexio-signature": f"sha256={digest}",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "resolved"


def test_resolve_pending_tool_call_rejects_missing_hmac_signature(client, storage):
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_pending_tool_call(_pending_call("ptc_1", now=now))
    client.app.state.fake_ctx.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite(),
        pending_tool_call_config=PendingToolCallConfig(hmac_secrets=["test-secret"]),
    )

    response = client.post(
        "/api/pending_tool_calls/ptc_1/resolve",
        json={"result": {"answer": "AWS ECS"}},
    )

    assert response.status_code == 401


def test_cancel_pending_tool_call_is_idempotent(client, storage):
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_pending_tool_call(_pending_call("ptc_1", now=now))

    first = client.post("/api/pending_tool_calls/ptc_1/cancel")
    second = client.post("/api/pending_tool_calls/ptc_1/cancel")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["status"] == "cancelled"
