"""Pre-admission readiness refusals stay retryable through the public facade."""

from __future__ import annotations

import asyncio
import logging
import time
from functools import partial
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.server.services.storage.error import ReadinessUnavailableError
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage


@pytest.fixture(autouse=True)
def stop_background_scheduler(monkeypatch):
    monkeypatch.setattr(
        "reflexio.server.services.durable_learning.local.ensure_local_extraction",
        lambda _: None,
    )
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "off")


def payload(request_id: str) -> dict:
    return {
        "request_id": request_id,
        "user_id": "readiness-user",
        "session_id": "readiness-session",
        "interaction_data_list": [
            {"content": "I prefer concise answers.", "created_at": 1}
        ],
    }


def test_readiness_refusal_is_retryable_without_admission_then_recovers(
    tmp_path, monkeypatch, caplog, worker_id
):
    reflexio = Reflexio(org_id=f"ready-{worker_id}", storage_base_dir=str(tmp_path))
    storage = reflexio.get_storage()
    assert isinstance(storage, SQLiteStorage)
    original_prepare = storage.prepare_interaction_embeddings
    error = ReadinessUnavailableError("credential-bearing provider diagnostics")
    monkeypatch.setattr(
        storage, "prepare_interaction_embeddings", Mock(side_effect=error)
    )
    monkeypatch.setattr(
        "reflexio.server.work_scope.current_project_id", lambda: "readiness-project"
    )
    caplog.set_level(logging.WARNING)

    with pytest.raises(ReadinessUnavailableError) as refusal:
        reflexio.publish_interaction(payload("ready-request"), defer_learning=True)
    assert refusal.value is error
    assert storage.get_request("ready-request") is None
    assert storage.get_user_interaction("readiness-user") == []
    assert storage.claim_extraction("ready-worker", 300) is None
    assert storage.conn.execute("SELECT COUNT(*) FROM learning_work").fetchone()[0] == 0
    assert (
        storage.conn.execute("SELECT COUNT(*) FROM extraction_cursors").fetchone()[0]
        == 0
    )
    records = [r for r in caplog.records if r.name.endswith("generation_service")]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert "event=publish_readiness_unavailable" in records[0].getMessage()
    assert "admitted=false" in records[0].getMessage()
    assert "retryable=true" in records[0].getMessage()
    assert "release_blocking=true" in records[0].getMessage()
    assert "project_id=readiness-project" in records[0].getMessage()
    assert not records[0].exc_info
    assert "credential-bearing" not in caplog.text

    monkeypatch.setattr(storage, "prepare_interaction_embeddings", original_prepare)
    response = reflexio.publish_interaction(
        payload("ready-request"), defer_learning=True
    )
    assert response.success is True
    assert storage.get_request("ready-request") is not None
    assert len(storage.get_user_interaction("readiness-user")) == 1
    assert storage.claim_extraction("ready-worker", 300) is not None
    work = storage.conn.execute(
        "SELECT highwater, revision FROM learning_work"
    ).fetchall()

    # A replay needs no new embedding admission, even while a tenant warms up.
    pending = Mock(side_effect=error)
    monkeypatch.setattr(storage, "prepare_interaction_embeddings", pending)
    replay = reflexio.publish_interaction(payload("ready-request"), defer_learning=True)
    assert replay.success is True and replay.replayed
    assert len(storage.get_user_interaction("readiness-user")) == 1
    assert (
        storage.conn.execute("SELECT highwater, revision FROM learning_work").fetchall()
        == work
    )
    changed = payload("ready-request")
    changed["interaction_data_list"][0]["content"] = "A different publish"
    conflict = reflexio.publish_interaction(changed, defer_learning=True)
    assert not conflict.success and conflict.error_reason == "request_id_conflict"
    assert len(storage.get_user_interaction("readiness-user")) == 1


@pytest.mark.parametrize("error", [RuntimeError("broken"), TimeoutError("broken")])
def test_unexpected_publish_failure_still_reports_error(
    tmp_path, monkeypatch, caplog, worker_id, error
):
    reflexio = Reflexio(org_id=f"fault-{worker_id}", storage_base_dir=str(tmp_path))
    monkeypatch.setattr(
        reflexio.get_storage(),
        "prepare_interaction_embeddings",
        Mock(side_effect=error),
    )
    caplog.set_level(logging.WARNING)
    response = reflexio.publish_interaction(
        payload("fault-request"), defer_learning=True
    )
    assert not response.success
    records = [r for r in caplog.records if r.name.endswith("generation_service")]
    assert any(r.levelno == logging.ERROR and r.exc_info for r in records)
    assert not any("publish_readiness_unavailable" in r.getMessage() for r in records)


@pytest.mark.parametrize("failure_phase", ["worker_wakeup", "coverage_read"])
def test_readiness_failure_after_commit_preserves_success(
    tmp_path, monkeypatch, worker_id, failure_phase
):
    reflexio = Reflexio(org_id=f"committed-{worker_id}", storage_base_dir=str(tmp_path))
    storage = reflexio.get_storage()
    failure = Mock(
        side_effect=ReadinessUnavailableError("dependency became unavailable")
    )
    if failure_phase == "worker_wakeup":
        monkeypatch.setattr(
            "reflexio.server.services.durable_learning.local.ensure_local_extraction",
            failure,
        )
    else:
        monkeypatch.setattr(storage, "extraction_report", failure)
    response = reflexio.publish_interaction(
        payload("committed-request"), defer_learning=True
    )
    assert response.success and response.request_id == "committed-request"
    assert storage.get_request("committed-request") is not None
    assert len(storage.get_user_interaction("readiness-user")) == 1
    assert storage.claim_extraction("committed-worker", 300) is not None
    failure.assert_called()


def test_standalone_http_readiness_refusal_then_retry_and_post_commit_success(
    tmp_path, monkeypatch, worker_id
):
    """The shared app owns 503 mapping; no enterprise handler is installed."""
    from reflexio.server.api import create_app
    from reflexio.server.api_endpoints import publisher_api
    from reflexio.server.services.durable_learning import waiting

    org_id = f"standalone-readiness-{worker_id}"
    reflexio = Reflexio(org_id=org_id, storage_base_dir=str(tmp_path))
    storage = reflexio.get_storage()
    assert isinstance(storage, SQLiteStorage)
    original_prepare = storage.prepare_interaction_embeddings
    unready = [True]

    @SQLiteStorage.handle_exceptions
    def prepare_when_ready(owner, interactions):
        if unready[0]:
            raise ReadinessUnavailableError("private provider diagnostics")
        original_prepare(interactions)

    monkeypatch.setattr(
        storage, "prepare_interaction_embeddings", partial(prepare_when_ready, storage)
    )
    monkeypatch.setattr(publisher_api, "get_reflexio", lambda **_: reflexio)
    monkeypatch.setattr(waiting, "operation_limit_value", lambda _: 1)
    client = TestClient(
        create_app(get_org_id=lambda: org_id), raise_server_exceptions=False
    )

    def assert_ingestion_permit_released():
        acquired = asyncio.run(
            waiting.acquire_ingestion(org_id, time.monotonic() + 0.1)
        )
        assert acquired, "the HTTP outcome must release its ingestion permit"
        waiting.release_ingestion(org_id)

    try:
        refused = client.post(
            "/api/publish_interaction", json=payload("standalone-request")
        )
        assert refused.status_code == 503
        assert refused.json() == {"detail": "embedding_readiness_unavailable"}
        assert refused.headers["Cache-Control"] == "no-store"
        assert refused.headers["Retry-After"] == "30"
        assert "private provider" not in refused.text
        assert storage.get_request("standalone-request") is None
        assert storage.get_user_interaction("readiness-user") == []
        assert storage.claim_extraction("standalone-worker", 300) is None
        assert_ingestion_permit_released()

        unready[0] = False
        accepted = client.post(
            "/api/publish_interaction", json=payload("standalone-request")
        )
        assert accepted.status_code == 200 and accepted.json()["success"] is True
        assert len(storage.get_user_interaction("readiness-user")) == 1
        assert_ingestion_permit_released()

        # A dependency becoming unavailable after commit must preserve success.
        monkeypatch.setattr(
            "reflexio.server.services.durable_learning.local.ensure_local_extraction",
            Mock(side_effect=ReadinessUnavailableError("post-commit readiness")),
        )
        committed = client.post(
            "/api/publish_interaction", json=payload("committed-http-request")
        )
        assert committed.status_code == 200 and committed.json()["success"] is True
        assert storage.get_request("committed-http-request") is not None
        assert len(storage.get_user_interaction("readiness-user")) == 2
        assert_ingestion_permit_released()
    finally:
        client.close()
