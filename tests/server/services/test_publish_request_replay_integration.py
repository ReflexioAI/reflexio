"""A caller-supplied request_id is idempotent across retries.

Real SQLite storage, mocked LLM. A retry of a committed publish is replayed
with the original identifiers and stores nothing; the same id reused for a
different user or payload is an explicit conflict, logged at WARNING.
"""

from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.api_schema.service_schemas import (
    InteractionData,
    PublishUserInteractionRequest,
    RetrievedLearning,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.generation_service import (
    GenerationService,
    PublishRequestIdConflictError,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

_GENERATION_LOGGER = "reflexio.server.services.generation_service"


@pytest.fixture(autouse=True)
def _no_embeddings_or_scheduler(monkeypatch):
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "off")
    monkeypatch.setattr(
        "reflexio.server.services.durable_learning.local.ensure_local_extraction",
        lambda _: None,
    )


def _service(tmp_path, worker_id: str) -> GenerationService:
    return GenerationService(
        llm_client=LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini")),
        request_context=RequestContext(
            org_id=f"replay_{worker_id}", storage_base_dir=str(tmp_path)
        ),
    )


def _request(
    *,
    user_id: str = "u1",
    content: str = "hello",
    session_id: str = "s1",
    created_at: int = 1_700_000_000,
) -> PublishUserInteractionRequest:
    return PublishUserInteractionRequest(
        request_id="caller-id",
        user_id=user_id,
        session_id=session_id,
        interaction_data_list=[
            InteractionData(role="User", content=content, created_at=created_at),
            InteractionData(role="Agent", content="hi", created_at=created_at),
        ],
    )


def _counts(store: SQLiteStorage) -> tuple[int, int]:
    return (
        store.conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0],
        store.conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0],
    )


def test_identical_retry_replays_original_ids_without_new_rows(
    tmp_path, monkeypatch, worker_id
):
    svc = _service(tmp_path, worker_id)
    store = svc.storage
    assert isinstance(store, SQLiteStorage)
    first = svc.run(_request(), defer_learning=True)
    assert not first.replayed and len(first.interaction_ids) == 2
    rows = _counts(store)

    # The retry admits nothing and records no second successful publish.
    metered = Mock()
    monkeypatch.setattr(
        "reflexio.server.services.generation_service.record_usage_event", metered
    )
    # A retry whose client re-stamped created_at is still the same publish.
    replay = svc.run(_request(created_at=1_700_000_999), defer_learning=True)

    assert replay.replayed
    assert replay.request_id == "caller-id"
    assert replay.interaction_ids == first.interaction_ids
    assert _counts(store) == rows
    events = [c.kwargs["event_name"] for c in metered.call_args_list]
    assert "publish_request_succeeded" not in events
    assert "publish_request_failed" not in events


@pytest.mark.parametrize(
    "changed",
    [
        {"user_id": "someone-else"},
        {"content": "a different message"},
        {"session_id": "another-session"},
    ],
    ids=["other-user", "other-content", "other-session"],
)
def test_reused_id_for_different_publish_is_a_warning_conflict(
    tmp_path, worker_id, caplog, changed
):
    svc = _service(tmp_path, worker_id)
    store = svc.storage
    assert isinstance(store, SQLiteStorage)
    svc.run(_request(), defer_learning=True)
    rows = _counts(store)

    caplog.set_level(logging.WARNING, logger=_GENERATION_LOGGER)
    caplog.clear()
    with pytest.raises(PublishRequestIdConflictError, match="different publish"):
        svc.run(_request(**changed), defer_learning=True)

    assert _counts(store) == rows
    records = [r for r in caplog.records if r.name == _GENERATION_LOGGER]
    assert [r.levelno for r in records] == [logging.WARNING]
    assert "publish_request_id_conflict" in records[0].getMessage()


def test_concurrent_commit_after_preflight_is_replayed(
    tmp_path, monkeypatch, worker_id
):
    """The unique-key arbiter inside the transaction also yields a replay."""
    svc = _service(tmp_path, worker_id)
    store = svc.storage
    assert isinstance(store, SQLiteStorage)
    request = _request()
    committed = {}

    def concurrent_commit(_interactions):
        # Our preflight already missed; an identical publish commits now.
        with monkeypatch.context() as nested:
            nested.setattr(store, "prepare_interaction_embeddings", lambda _: None)
            committed["first"] = svc.run(request, defer_learning=True)

    monkeypatch.setattr(store, "prepare_interaction_embeddings", concurrent_commit)
    result = svc.run(request, defer_learning=True)

    assert result.replayed
    assert result.interaction_ids == committed["first"].interaction_ids
    assert _counts(store) == (1, 2)


def test_genuine_failure_still_reports_at_error(
    tmp_path, monkeypatch, worker_id, caplog
):
    """Only the conflict is demoted; an unexpected failure stays an ERROR."""
    svc = _service(tmp_path, worker_id)
    store = svc.storage
    assert isinstance(store, SQLiteStorage)
    monkeypatch.setattr(
        store,
        "add_user_interactions_bulk",
        Mock(side_effect=RuntimeError("disk on fire")),
    )
    caplog.set_level(logging.WARNING, logger=_GENERATION_LOGGER)
    with pytest.raises(RuntimeError, match="disk on fire"):
        svc.run(_request(), defer_learning=True)
    assert any(
        r.name == _GENERATION_LOGGER and r.levelno == logging.ERROR
        for r in caplog.records
    )


def test_library_response_reports_replay_and_conflict(tmp_path, worker_id):
    reflexio = Reflexio(
        org_id=f"replay_lib_{worker_id}", storage_base_dir=str(tmp_path)
    )
    first = reflexio.publish_interaction(_request(), defer_learning=True)
    assert first.success and not first.replayed

    replay = reflexio.publish_interaction(_request(), defer_learning=True)
    assert replay.success and replay.replayed
    assert replay.request_id == first.request_id
    assert replay.error_reason is None

    conflict = reflexio.publish_interaction(
        _request(content="changed"), defer_learning=True
    )
    assert not conflict.success and not conflict.replayed
    assert conflict.error_reason == "request_id_conflict"
    assert conflict.request_id == "caller-id"


def test_participant_without_receipt_for_existing_id_is_a_conflict(tmp_path, worker_id):
    """The external-admission replay shares the same conflict class."""
    svc = _service(tmp_path, worker_id)
    svc.run(_request(), defer_learning=True)
    participant = Mock()
    participant.claim.return_value = None
    with pytest.raises(PublishRequestIdConflictError):
        svc.run(_request(), defer_learning=True, admission_participant=participant)
    participant.complete.assert_not_called()


def _receipt_request(receipt: str) -> PublishUserInteractionRequest:
    request = _request()
    request.interaction_data_list[1].retrieved_learnings = [
        RetrievedLearning(
            kind="user_playbook", learning_id="p1", exposure_receipt=receipt
        )
    ]
    return request


def test_exposure_receipt_round_trip_still_replays(tmp_path, worker_id):
    """Storage keeps only a receipt's digest; the identical retry must match."""
    svc = _service(tmp_path, worker_id)
    first = svc.run(_receipt_request("signed-receipt"), defer_learning=True)
    replay = svc.run(_receipt_request("signed-receipt"), defer_learning=True)
    assert replay.replayed and replay.interaction_ids == first.interaction_ids
    with pytest.raises(PublishRequestIdConflictError):
        svc.run(_receipt_request("another-receipt"), defer_learning=True)


def test_empty_role_retry_replays(tmp_path, worker_id):
    """An empty role is stored and read back as "User"; a retry still replays."""
    svc = _service(tmp_path, worker_id)
    store = svc.storage
    assert isinstance(store, SQLiteStorage)
    request = PublishUserInteractionRequest(
        request_id="caller-id",
        user_id="u1",
        session_id="s1",
        interaction_data_list=[InteractionData(role="", content="hello")],
    )
    first = svc.run(request, defer_learning=True)
    rows = _counts(store)

    replay = svc.run(request.model_copy(deep=True), defer_learning=True)

    assert replay.replayed
    assert replay.interaction_ids == first.interaction_ids
    assert _counts(store) == rows


def test_two_workers_racing_the_insert_admit_once(tmp_path, worker_id, monkeypatch):
    """Both attempts finish preparation before either inserts; one admits.

    Two services own separate SQLite connections to one database, like two
    server workers. The barrier holds both after embedding preparation, the last
    step before the admission transaction, so neither has seen the other's row.
    """
    import threading

    services = [_service(tmp_path, worker_id) for _ in range(2)]
    assert services[0].storage is not services[1].storage
    barrier = threading.Barrier(2, timeout=30)
    original = SQLiteStorage.prepare_interaction_embeddings

    def prepare_then_wait(self, interactions):
        original(self, interactions)
        barrier.wait()

    monkeypatch.setattr(
        SQLiteStorage, "prepare_interaction_embeddings", prepare_then_wait
    )
    results = {}

    def publish(index: int) -> None:
        results[index] = services[index].run(_request(), defer_learning=True)

    threads = [threading.Thread(target=publish, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert sorted(r.replayed for r in results.values()) == [False, True]
    assert results[0].interaction_ids == results[1].interaction_ids
    store = services[0].storage
    assert isinstance(store, SQLiteStorage)
    assert _counts(store) == (1, 2)
