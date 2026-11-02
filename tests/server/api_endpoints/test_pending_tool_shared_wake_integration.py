"""Committed HTTP callbacks signal the existing durable owner, without dispatch."""

import json
import sqlite3
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
from reflexio.server.services.durable_learning import local
from reflexio.server.services.durable_learning.scheduler import DurableLearningScheduler
from reflexio.server.services.extraction.resume_worker import ExtractionResumeWorker
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import (
    PendingToolCallStatus,
    not_applicable_tool_result,
)
from tests.server.api_endpoints.test_pending_tool_call_api_failures_integration import (
    _call,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def isolated_owners(monkeypatch):
    # Keep real thread lifecycle and wake delivery, without consuming the row or
    # touching model dispatch. Restore the prior process-owned registry exactly.
    monkeypatch.setattr(DurableLearningScheduler, "_run_once", lambda _: 3600)
    with local._lock:
        saved = local._schedulers, local._contexts, local._server_scheduler
        local._schedulers = {}
        local._contexts = type(local._contexts)()
        local._server_scheduler = None
    try:
        yield
    finally:
        with local._lock:
            schedulers = list(local._schedulers.values())
            if local._server_scheduler is not None:
                schedulers.append(local._server_scheduler)
            local._schedulers, local._contexts, local._server_scheduler = saved
        for scheduler in schedulers:
            scheduler.stop()


def _context(storage, directory):
    ctx = RequestContext.__new__(RequestContext)
    ctx.org_id = storage.org_id
    ctx.storage = storage
    ctx.storage_base_dir = str(directory)
    ctx.configurator = MagicMock()
    ctx.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite(db_path=storage.db_path),
        pending_tool_call_config=PendingToolCallConfig(enabled=True),
    )
    return ctx


def _owner(ctx, mode):
    if mode == "local":
        local.ensure_local_extraction(ctx)
        return local._schedulers[ctx.storage_base_dir]
    scheduler = DurableLearningScheduler(
        request_context_factory=lambda _: ctx, org_ids_provider=lambda: ()
    )
    scheduler.start()
    local.adopt_server_scheduler(scheduler)
    return scheduler


@pytest.mark.parametrize("owner_mode", ["server", "local"])
@pytest.mark.parametrize("operation", ["resolve", "answer", "not_applicable"])
@pytest.mark.parametrize("fail_write", [False, True], ids=["commit", "rollback"])
def test_pending_callback_wakes_shared_owner_only_after_committed_change(
    tmp_path, isolated_owners, monkeypatch, owner_mode, operation, fail_write
):
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        store = SQLiteStorage(org_id="callback-org", db_path=str(tmp_path / "work.db"))
    ctx = _context(store, tmp_path)
    existing = _call(
        "callback",
        ctx.org_id,
        status=(
            PendingToolCallStatus.RESOLVED
            if operation == "answer"
            else PendingToolCallStatus.PENDING
        ),
        result={"answer": "before"} if operation == "answer" else None,
    )
    store.create_pending_tool_call(existing)
    scheduler = _owner(ctx, owner_mode)
    observed = []
    original_wake = scheduler.wake

    def committed_row():
        with sqlite3.connect(store.db_path) as reader:
            row = reader.execute(
                "SELECT status, result FROM _pending_tool_calls WHERE id=?",
                (existing.id,),
            ).fetchone()
        return row[0], json.loads(row[1]) if row[1] is not None else None

    def observe_wake():
        observed.append(committed_row())
        original_wake()

    monkeypatch.setattr(scheduler, "wake", observe_wake)
    monkeypatch.setattr(
        ExtractionResumeWorker,
        "__init__",
        MagicMock(side_effect=AssertionError("callback dispatched a separate worker")),
    )
    if fail_write:
        # Abort the real SQLite mutation. Its surrounding APP transaction must
        # roll back before the HTTP error; no mocked storage return can prove it.
        with store.commit_scope():
            store.conn.execute(
                "CREATE TRIGGER reject_callback BEFORE UPDATE ON _pending_tool_calls "
                "WHEN NEW.id = 'callback' BEGIN "
                "SELECT RAISE(ABORT, 'callback rollback'); END"
            )
    app = create_app(get_org_id=lambda: ctx.org_id)
    app.dependency_overrides[get_request_context] = lambda: ctx
    client = TestClient(app)
    if operation == "answer":
        response = client.patch(
            f"/api/pending_tool_calls/{existing.id}/answer", json={"answer": "after"}
        )
        expected = {"answer": "after"}
    else:
        response = client.post(
            f"/api/pending_tool_calls/{existing.id}/{operation}",
            json={"result": {"answer": "after"}} if operation == "resolve" else {},
        )
        expected = (
            {"answer": "after"}
            if operation == "resolve"
            else not_applicable_tool_result()
        )
    assert response.status_code == (503 if fail_write else 200)
    assert observed == ([] if fail_write else [("resolved", expected)])
    assert committed_row() == (
        (existing.status.value, existing.result)
        if fail_write
        else ("resolved", expected)
    )
    assert scheduler.is_running()


@pytest.mark.parametrize("owner_mode", ["server", "local"])
def test_shared_wake_targets_existing_owner_through_stop_and_restart(
    tmp_path, isolated_owners, monkeypatch, owner_mode
):
    store = SQLiteStorage(org_id="lifetime-org", db_path=str(tmp_path / "work.db"))
    ctx = _context(store, tmp_path)
    scheduler = _owner(ctx, owner_mode)
    observed = []
    original_wake = scheduler.wake

    def observe_wake():
        observed.append(scheduler)
        original_wake()

    monkeypatch.setattr(scheduler, "wake", observe_wake)
    monkeypatch.setattr(
        ctx.configurator,
        "get_config",
        MagicMock(side_effect=AssertionError("wake loaded config")),
    )
    local.wake_extraction(ctx)
    assert observed == [scheduler]
    foreign_directory = _context(store, tmp_path / "other")
    local.wake_extraction(foreign_directory)
    # The server owns its fleet. A local directory consumer must not receive a
    # callback intended for another config directory, even for the same org.
    before_stop = 2 if owner_mode == "server" else 1
    assert len(observed) == before_stop
    scheduler.stop()
    assert not scheduler.is_running()
    local.wake_extraction(ctx)
    assert len(observed) == before_stop
    scheduler.start()
    local.wake_extraction(ctx)
    assert observed == [scheduler] * (before_stop + 1)
    assert scheduler.is_running()


def test_callback_without_owner_never_creates_or_registers_consumer(
    tmp_path, isolated_owners, monkeypatch
):
    store = SQLiteStorage(org_id="unowned-org", db_path=str(tmp_path / "work.db"))
    ctx = _context(store, tmp_path)
    monkeypatch.setattr(
        ctx.configurator,
        "get_config",
        MagicMock(side_effect=AssertionError("wake loaded config")),
    )
    local.wake_extraction(ctx)
    assert local._schedulers == {}
    assert len(local._contexts) == 0
    assert local._server_scheduler is None
