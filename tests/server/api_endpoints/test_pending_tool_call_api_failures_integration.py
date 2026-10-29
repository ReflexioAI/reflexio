"""Pending tool call endpoints: failure, conflict and authentication paths.

Real SQLite storage behind the real FastAPI app. Storage faults (backend
errors, unsupported backends, lost write races) are injected on the real
storage object so each HTTP mapping is observed end to end, and every test
checks the persisted row was left as the endpoint claims.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import uuid
from dataclasses import replace
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
    PendingToolCallRecord,
    PendingToolCallStatus,
    build_pending_tool_call_dedup_key,
    build_scope_hash,
    human_feedback_scope,
    not_applicable_tool_result,
)

pytestmark = pytest.mark.integration

NOW = datetime(2026, 5, 28, tzinfo=UTC)
SECRET = "pending-hmac-secret"


@pytest.fixture
def org_id() -> str:
    return f"ptc-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def storage(tmp_path, org_id):
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        yield SQLiteStorage(org_id=org_id, db_path=str(tmp_path / "ptc.db"))


@pytest.fixture
def ctx(storage, org_id) -> RequestContext:
    context = RequestContext.__new__(RequestContext)
    context.org_id = org_id
    context.storage = storage
    context.configurator = MagicMock()
    context.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite()
    )
    return context


@pytest.fixture
def client(ctx, org_id, tmp_path) -> TestClient:
    app = create_app(get_org_id=lambda: org_id)
    app.dependency_overrides[get_request_context] = lambda: ctx
    return TestClient(app, raise_server_exceptions=True, client=(tmp_path.name, 50000))


def _call(
    call_id: str,
    org_id: str,
    *,
    tool_name: str = "ask_human",
    status: PendingToolCallStatus = PendingToolCallStatus.PENDING,
    result: dict | None = None,
) -> PendingToolCallRecord:
    scope = human_feedback_scope(org_id)
    question = f"Question for {call_id}?"
    resolved = status == PendingToolCallStatus.RESOLVED
    return PendingToolCallRecord(
        id=call_id,
        org_id=org_id,
        user_id="user_1",
        scope=scope,
        scope_hash=build_scope_hash(scope),
        tool_name=tool_name,
        dedup_key=build_pending_tool_call_dedup_key(
            tool_name=tool_name, question_text=question
        ),
        status=status,
        question_text=question,
        args={"question": question},
        result=result,
        resolved_at=NOW if resolved else None,
        valid_until=NOW + timedelta(hours=1) if resolved else None,
        expires_at=NOW + timedelta(hours=1),
        cache_until=NOW + timedelta(minutes=5),
    )


def _status(storage, call_id: str) -> PendingToolCallStatus:
    record = storage.get_pending_tool_call(call_id)
    assert record is not None
    return record.status


# --------------------------------------------------------------------------
# Storage availability
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("get", "/api/pending_tool_calls", None),
        ("get", "/api/pending_tool_calls/ptc_1", None),
        ("post", "/api/pending_tool_calls/ptc_1/resolve", {"result": {"a": 1}}),
        ("patch", "/api/pending_tool_calls/ptc_1/answer", {"answer": "x"}),
        ("post", "/api/pending_tool_calls/ptc_1/not_applicable", {}),
        ("post", "/api/pending_tool_calls/ptc_1/cancel", None),
    ],
)
def test_every_endpoint_returns_503_without_storage(client, ctx, method, path, body):
    ctx.storage = None
    response = client.request(method, path, json=body)
    assert response.status_code == 503
    assert response.json()["detail"] == "Storage not configured"


_READ_ENDPOINTS = [
    ("list_pending_tool_calls", "get", "/api/pending_tool_calls", None),
    ("get_pending_tool_call", "get", "/api/pending_tool_calls/ptc_1", None),
    (
        "get_pending_tool_call",
        "post",
        "/api/pending_tool_calls/ptc_1/resolve",
        {"result": {"answer": "x"}},
    ),
    (
        "get_pending_tool_call",
        "patch",
        "/api/pending_tool_calls/ptc_1/answer",
        {"answer": "x"},
    ),
    (
        "get_pending_tool_call",
        "post",
        "/api/pending_tool_calls/ptc_1/not_applicable",
        {},
    ),
    ("get_pending_tool_call", "post", "/api/pending_tool_calls/ptc_1/cancel", None),
]


@pytest.mark.parametrize(("storage_method", "method", "path", "body"), _READ_ENDPOINTS)
def test_unsupported_backend_maps_to_503(
    client, storage, monkeypatch, storage_method, method, path, body
):
    monkeypatch.setattr(
        storage, storage_method, MagicMock(side_effect=NotImplementedError)
    )
    response = client.request(method, path, json=body)
    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Pending tool calls are not supported by the configured storage backend"
    )


@pytest.mark.parametrize(("storage_method", "method", "path", "body"), _READ_ENDPOINTS)
def test_backend_read_error_maps_to_503_with_its_message(
    client, storage, monkeypatch, storage_method, method, path, body
):
    storage.migrate = MagicMock()
    monkeypatch.setattr(
        storage,
        storage_method,
        MagicMock(side_effect=StorageError("connection reset by peer")),
    )
    response = client.request(method, path, json=body)
    assert response.status_code == 503
    assert response.json()["detail"] == "connection reset by peer"
    # Only a missing pending-tool-call table triggers the migrate-and-retry path.
    storage.migrate.assert_not_called()


def test_missing_table_without_a_migrate_hook_is_not_retried(client, storage, ctx):
    class NoMigrateStorage:
        def __init__(self) -> None:
            self.calls = 0

        def get_pending_tool_call(self, _call_id):
            self.calls += 1
            raise StorageError('relation "_pending_tool_calls" does not exist')

    fake = NoMigrateStorage()
    ctx.storage = fake
    response = client.get("/api/pending_tool_calls/ptc_1")
    assert response.status_code == 503
    assert "_pending_tool_calls" in response.json()["detail"]
    assert fake.calls == 1


@pytest.mark.parametrize(
    ("storage_method", "method", "path", "body", "status"),
    [
        (
            "resolve_pending_tool_call",
            "post",
            "/api/pending_tool_calls/ptc_1/resolve",
            {"result": {"answer": "x"}},
            PendingToolCallStatus.PENDING,
        ),
        (
            "update_resolved_pending_tool_call_result",
            "patch",
            "/api/pending_tool_calls/ptc_1/answer",
            {"answer": "new"},
            PendingToolCallStatus.RESOLVED,
        ),
        (
            "mark_pending_tool_call_not_applicable",
            "post",
            "/api/pending_tool_calls/ptc_1/not_applicable",
            {},
            PendingToolCallStatus.PENDING,
        ),
        (
            "cancel_pending_tool_call",
            "post",
            "/api/pending_tool_calls/ptc_1/cancel",
            None,
            PendingToolCallStatus.PENDING,
        ),
    ],
)
def test_backend_write_error_maps_to_503_and_leaves_row_unchanged(
    client, storage, org_id, monkeypatch, storage_method, method, path, body, status
):
    storage.create_pending_tool_call(
        _call("ptc_1", org_id, status=status, result={"answer": "old"})
    )
    monkeypatch.setattr(
        storage, storage_method, MagicMock(side_effect=StorageError("write failed"))
    )
    response = client.request(method, path, json=body)
    assert response.status_code == 503
    assert response.json()["detail"] == "write failed"
    assert _status(storage, "ptc_1") == status


# --------------------------------------------------------------------------
# HMAC authentication
# --------------------------------------------------------------------------


def _signed(body: bytes, *, timestamp: int, secret: str = SECRET, prefix=True):
    digest = hmac.new(
        secret.encode(), str(timestamp).encode() + b"." + body, hashlib.sha256
    ).hexdigest()
    return {
        "content-type": "application/json",
        "x-reflexio-timestamp": str(timestamp),
        "x-reflexio-signature": f"sha256={digest}" if prefix else digest,
    }


@pytest.fixture
def hmac_ctx(ctx):
    ctx.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite(),
        pending_tool_call_config=PendingToolCallConfig(
            hmac_secrets=["rotated-out-secret", SECRET]
        ),
    )
    return ctx


_RESOLVE_BODY = b'{"result":{"answer":"x"}}'


@pytest.mark.parametrize(
    ("make_headers", "detail"),
    [
        (
            lambda: {"x-reflexio-timestamp": "yesterday", "x-reflexio-signature": "a"},
            "Invalid HMAC timestamp",
        ),
        (
            lambda: _signed(_RESOLVE_BODY, timestamp=int(time.time()) - 3600),
            "Expired HMAC timestamp",
        ),
        (
            lambda: _signed(_RESOLVE_BODY, timestamp=int(time.time()), secret="x"),
            "Invalid HMAC signature",
        ),
    ],
    ids=["bad-timestamp", "expired", "wrong-secret"],
)
def test_rejected_signatures_do_not_resolve(
    client, storage, org_id, hmac_ctx, make_headers, detail
):
    storage.create_pending_tool_call(_call("ptc_1", org_id))
    # Built at run time: a collection-time timestamp could expire before use.
    response = client.post(
        "/api/pending_tool_calls/ptc_1/resolve",
        content=_RESOLVE_BODY,
        headers={"content-type": "application/json", **make_headers()},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == detail
    assert _status(storage, "ptc_1") == PendingToolCallStatus.PENDING


def test_unprefixed_signature_from_any_configured_secret_is_accepted(
    client, storage, org_id, hmac_ctx
):
    storage.create_pending_tool_call(_call("ptc_1", org_id))
    body = json.dumps({"result": {"answer": "signed"}}).encode()
    response = client.post(
        "/api/pending_tool_calls/ptc_1/resolve",
        content=body,
        headers=_signed(body, timestamp=int(time.time()), prefix=False),
    )
    assert response.status_code == 200
    stored = storage.get_pending_tool_call("ptc_1")
    assert stored is not None
    assert stored.status == PendingToolCallStatus.RESOLVED
    assert stored.result == {"answer": "signed"}


# --------------------------------------------------------------------------
# State conflicts
# --------------------------------------------------------------------------


def test_resolving_a_cancelled_call_conflicts(client, storage, org_id):
    storage.create_pending_tool_call(
        _call("ptc_1", org_id, status=PendingToolCallStatus.CANCELLED)
    )
    response = client.post(
        "/api/pending_tool_calls/ptc_1/resolve", json={"result": {"answer": "x"}}
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "Pending tool call is cancelled"
    assert _status(storage, "ptc_1") == PendingToolCallStatus.CANCELLED


def _lose_race(storage, monkeypatch, method: str, winner: PendingToolCallRecord):
    """The write affects no row because a concurrent writer already moved it."""

    real_cancel = storage.cancel_pending_tool_call
    real_resolve = storage.resolve_pending_tool_call

    def lost(*_args, **_kwargs):
        if winner.status == PendingToolCallStatus.CANCELLED:
            real_cancel(winner.id)
        else:
            real_resolve(
                winner.id,
                result=winner.result or {},
                resolved_at=NOW,
                valid_for_seconds=3600,
            )
        # Falls through to None: this caller's own write matched no row.

    monkeypatch.setattr(storage, method, lost)


@pytest.mark.parametrize(
    ("winner_result", "expected_status", "detail"),
    [
        ({"answer": "same"}, 200, None),
        (
            {"answer": "other"},
            409,
            "Pending tool call already resolved with a different result",
        ),
    ],
    ids=["same-result", "different-result"],
)
def test_resolve_after_a_lost_race_rechecks_the_winner(
    client, storage, org_id, monkeypatch, winner_result, expected_status, detail
):
    storage.create_pending_tool_call(_call("ptc_1", org_id))
    winner = replace(
        _call("ptc_1", org_id),
        status=PendingToolCallStatus.RESOLVED,
        result=winner_result,
    )
    _lose_race(storage, monkeypatch, "resolve_pending_tool_call", winner)

    response = client.post(
        "/api/pending_tool_calls/ptc_1/resolve", json={"result": {"answer": "same"}}
    )

    assert response.status_code == expected_status
    if detail:
        assert response.json()["detail"] == detail
    else:
        assert response.json()["result"] == {"answer": "same"}


def test_resolve_after_losing_to_a_cancellation_conflicts(
    client, storage, org_id, monkeypatch
):
    storage.create_pending_tool_call(_call("ptc_1", org_id))
    winner = replace(_call("ptc_1", org_id), status=PendingToolCallStatus.CANCELLED)
    _lose_race(storage, monkeypatch, "resolve_pending_tool_call", winner)

    response = client.post(
        "/api/pending_tool_calls/ptc_1/resolve", json={"result": {"answer": "x"}}
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Pending tool call is cancelled"
    assert _status(storage, "ptc_1") == PendingToolCallStatus.CANCELLED


@pytest.mark.parametrize(
    ("returned", "detail"),
    [
        ({"status": PendingToolCallStatus.EXPIRED}, "Pending tool call is expired"),
        (
            {"status": PendingToolCallStatus.RESOLVED, "result": {"answer": "other"}},
            "Pending tool call already resolved with a different result",
        ),
    ],
    ids=["not-resolved", "different-result"],
)
def test_resolve_rejects_an_unexpected_write_result(
    client, storage, org_id, monkeypatch, returned, detail
):
    storage.create_pending_tool_call(_call("ptc_1", org_id))
    monkeypatch.setattr(
        storage,
        "resolve_pending_tool_call",
        MagicMock(return_value=replace(_call("ptc_1", org_id), **returned)),
    )
    with patch(
        "reflexio.server.services.extraction.resume_scheduler.wake_resume_schedulers"
    ) as wake:
        response = client.post(
            "/api/pending_tool_calls/ptc_1/resolve",
            json={"result": {"answer": "mine"}},
        )
    assert response.status_code == 409
    assert response.json()["detail"] == detail
    wake.assert_not_called()


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/pending_tool_calls/ptc_1/answer", {"answer": "x"}),
        ("/api/pending_tool_calls/ptc_1/not_applicable", {}),
    ],
)
def test_only_ask_human_calls_can_be_managed(client, storage, org_id, path, body):
    storage.create_pending_tool_call(
        _call(
            "ptc_1",
            org_id,
            tool_name="attach_pending_info_request",
            status=PendingToolCallStatus.RESOLVED,
            result={"answer": "kept"},
        )
    )
    response = client.request(
        "patch" if path.endswith("answer") else "post", path, json=body
    )
    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Only ask_human tool calls can be managed from this endpoint"
    )
    stored = storage.get_pending_tool_call("ptc_1")
    assert stored is not None and stored.result == {"answer": "kept"}


def test_answer_update_requires_a_resolved_call(client, storage, org_id):
    storage.create_pending_tool_call(_call("ptc_1", org_id))
    response = client.patch(
        "/api/pending_tool_calls/ptc_1/answer", json={"answer": "early"}
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "Pending tool call is pending"
    assert _status(storage, "ptc_1") == PendingToolCallStatus.PENDING


def test_blank_answer_is_rejected(client, storage, org_id):
    storage.create_pending_tool_call(
        _call(
            "ptc_1",
            org_id,
            status=PendingToolCallStatus.RESOLVED,
            result={"answer": "kept"},
        )
    )
    response = client.patch(
        "/api/pending_tool_calls/ptc_1/answer", json={"answer": "   "}
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "Answer cannot be empty"
    stored = storage.get_pending_tool_call("ptc_1")
    assert stored is not None and stored.result == {"answer": "kept"}


def test_answer_update_lost_to_a_cancellation_conflicts(
    client, storage, org_id, monkeypatch
):
    storage.create_pending_tool_call(
        _call(
            "ptc_1",
            org_id,
            status=PendingToolCallStatus.RESOLVED,
            result={"answer": "kept"},
        )
    )
    monkeypatch.setattr(
        storage,
        "update_resolved_pending_tool_call_result",
        MagicMock(return_value=None),
    )
    monkeypatch.setattr(
        storage,
        "get_pending_tool_call",
        MagicMock(
            side_effect=[
                _call(
                    "ptc_1",
                    org_id,
                    status=PendingToolCallStatus.RESOLVED,
                    result={"answer": "kept"},
                ),
                _call("ptc_1", org_id, status=PendingToolCallStatus.EXPIRED),
            ]
        ),
    )
    response = client.patch(
        "/api/pending_tool_calls/ptc_1/answer", json={"answer": "new"}
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "Pending tool call is expired"


@pytest.mark.parametrize(
    ("returned", "detail"),
    [
        ({"status": PendingToolCallStatus.CANCELLED}, "Pending tool call is cancelled"),
        (
            {"status": PendingToolCallStatus.RESOLVED, "result": {"answer": "other"}},
            "Pending tool call already resolved with a different result",
        ),
    ],
    ids=["not-resolved", "different-result"],
)
def test_answer_update_rejects_an_unexpected_write_result(
    client, storage, org_id, monkeypatch, returned, detail
):
    storage.create_pending_tool_call(
        _call(
            "ptc_1",
            org_id,
            status=PendingToolCallStatus.RESOLVED,
            result={"answer": "kept"},
        )
    )
    monkeypatch.setattr(
        storage,
        "update_resolved_pending_tool_call_result",
        MagicMock(return_value=replace(_call("ptc_1", org_id), **returned)),
    )
    response = client.patch(
        "/api/pending_tool_calls/ptc_1/answer", json={"answer": "mine"}
    )
    assert response.status_code == 409
    assert response.json()["detail"] == detail


def test_not_applicable_rejects_a_cancelled_call(client, storage, org_id):
    storage.create_pending_tool_call(
        _call("ptc_1", org_id, status=PendingToolCallStatus.CANCELLED)
    )
    response = client.post("/api/pending_tool_calls/ptc_1/not_applicable", json={})
    assert response.status_code == 409
    assert response.json()["detail"] == "Pending tool call is cancelled"
    assert _status(storage, "ptc_1") == PendingToolCallStatus.CANCELLED


def test_not_applicable_overrides_a_resolved_answer(client, storage, org_id):
    storage.create_pending_tool_call(
        _call(
            "ptc_1",
            org_id,
            status=PendingToolCallStatus.RESOLVED,
            result={"answer": "previous"},
        )
    )
    response = client.post("/api/pending_tool_calls/ptc_1/not_applicable", json={})
    assert response.status_code == 200
    stored = storage.get_pending_tool_call("ptc_1")
    assert stored is not None
    assert stored.status == PendingToolCallStatus.RESOLVED
    assert stored.result == not_applicable_tool_result()


@pytest.mark.parametrize(
    ("returned", "detail"),
    [
        ({"status": PendingToolCallStatus.EXPIRED}, "Pending tool call is expired"),
        (
            {"status": PendingToolCallStatus.RESOLVED, "result": {"answer": "other"}},
            "Pending tool call could not be marked not applicable",
        ),
    ],
    ids=["not-resolved", "different-result"],
)
def test_not_applicable_rejects_an_unexpected_write_result(
    client, storage, org_id, monkeypatch, returned, detail
):
    storage.create_pending_tool_call(_call("ptc_1", org_id))
    monkeypatch.setattr(
        storage,
        "mark_pending_tool_call_not_applicable",
        MagicMock(return_value=replace(_call("ptc_1", org_id), **returned)),
    )
    response = client.post("/api/pending_tool_calls/ptc_1/not_applicable", json={})
    assert response.status_code == 409
    assert response.json()["detail"] == detail


def test_cancelling_a_resolved_call_conflicts(client, storage, org_id):
    storage.create_pending_tool_call(
        _call(
            "ptc_1",
            org_id,
            status=PendingToolCallStatus.RESOLVED,
            result={"answer": "kept"},
        )
    )
    response = client.post("/api/pending_tool_calls/ptc_1/cancel")
    assert response.status_code == 409
    assert response.json()["detail"] == "Pending tool call is resolved"
    assert _status(storage, "ptc_1") == PendingToolCallStatus.RESOLVED


def test_other_org_calls_are_invisible_to_every_mutation(client, storage):
    storage.create_pending_tool_call(_call("ptc_foreign", "some-other-org"))
    for method, path, body in [
        ("post", "/api/pending_tool_calls/ptc_foreign/resolve", {"result": {"a": 1}}),
        ("post", "/api/pending_tool_calls/ptc_foreign/not_applicable", {}),
        ("post", "/api/pending_tool_calls/ptc_foreign/cancel", None),
    ]:
        response = client.request(method, path, json=body)
        assert response.status_code == 404, path
    assert _status(storage, "ptc_foreign") == PendingToolCallStatus.PENDING
