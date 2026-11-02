"""Input contention yields a turn without consuming its window failure budget."""

from unittest.mock import Mock

import pytest

from reflexio.server.services.durable_learning.worker import DurableLearningWorker
from reflexio.server.services.profile.service import ProfileGenerationService
from reflexio.server.services.storage.error import ExecutionInputBusyError, StorageError
from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
    SQLiteStreamSQL,
)
from reflexio.server.services.storage.storage_base import LeaseLostError
from reflexio.server.services.storage.storage_base._extraction_stream import (
    MAX_EXTRACTION_WINDOW_ATTEMPTS,
)
from tests.server.services.test_durable_window_pipeline import payload
from tests.server.services.test_durable_window_pipeline import pipeline as pipeline
from tests.server.services.test_window_readiness_integration import _state

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("wrapped", [False, True])
def test_busy_persist_yields_without_timer_or_retry_spending_and_reuses_saved_compute(
    pipeline, monkeypatch, wrapped
):
    engine, worker, http = pipeline
    assert http.post(
        "/api/publish_interaction", json=payload("input-busy", force=True)
    ).json()["success"]
    storage = engine.get_storage()
    with storage._stream_sql(read_only=True) as db:
        clock = [db.now()]
    monkeypatch.setattr(SQLiteStreamSQL, "now", lambda _: clock[0])
    ready = False
    persist = ProfileGenerationService.persist_generation
    compute = Mock(wraps=ProfileGenerationService.compute_generation)
    monkeypatch.setattr(
        ProfileGenerationService,
        "compute_generation",
        lambda self, *args, **kw: compute(self, *args, **kw),
    )

    def write(service, plan):
        if not ready:
            if wrapped:
                try:
                    raise ExecutionInputBusyError("synthetic row is held")
                except ExecutionInputBusyError as error:
                    raise StorageError("explicit wrapper") from error
            raise ExecutionInputBusyError("synthetic row is held")
        return persist(service, plan)

    monkeypatch.setattr(ProfileGenerationService, "persist_generation", write)
    saved = None
    for _ in range(MAX_EXTRACTION_WINDOW_ATTEMPTS + 1):
        assert worker.drain_org("stream-test", 1, 300) == 0
        cursor, windows, work = _state(storage)
        assert cursor["attempts"] == 0 and cursor["last_error"] is None
        assert cursor["retry_at"] != float("inf")
        assert len(windows) == 1 and not windows[0]["completed"]
        saved = windows[0]["outcome"] if saved is None else saved
        assert saved is not None and windows[0]["outcome"] == saved
        assert compute.call_count == 1, "A busy writer must retain its saved compute"
        assert work == {"due_at": clock[0], "lease_token": None}
        assert storage.count_all_profiles() == 0
        clock[0] += 1
    ready = True
    assert worker.drain_org("stream-test", 1, 300) == 1
    assert compute.call_count == 1
    cursor, windows, _ = _state(storage)
    assert cursor["attempts"] == 0 and windows[0]["completed"]
    count = storage.count_all_profiles()
    assert count > 0
    assert worker.drain_org("stream-test", 1, 300) == 1
    assert _state(storage)[1][0]["effects_done"]
    assert worker.drain_org("stream-test", 1, 300) == 0
    assert storage.count_all_profiles() == count


@pytest.mark.parametrize("withdrawal", ["replacement", "erasure"])
def test_busy_yield_requires_the_actual_live_owner(pipeline, withdrawal):
    engine, _, http = pipeline
    assert http.post(
        "/api/publish_interaction", json=payload("stale-busy", force=True)
    ).json()["success"]
    storage = engine.get_storage()
    user, token = storage.claim_extraction("old", 300)
    window = storage.prepare_extraction(user, token)
    assert window is not None
    if withdrawal == "replacement":
        with storage._stream_sql() as db:
            db.query("UPDATE learning_work SET lease_until=0 WHERE user_id=?", (user,))
        replacement = storage.claim_user_extraction(user, "new", 300)
        assert replacement is not None and replacement != token
        before = _state(storage)
    else:
        storage.clear_user_data(user)
    with pytest.raises(LeaseLostError):
        DurableLearningWorker._handle_window_failure(
            storage, window, token, ExecutionInputBusyError("busy")
        )
    storage.release_user_extraction(user, token)
    if withdrawal == "replacement":
        assert _state(storage) == before
        storage.release_user_extraction(user, replacement)
    else:
        with storage._stream_sql(read_only=True) as db:
            assert (
                db.query("SELECT user_id FROM learning_work WHERE user_id=?", (user,))
                == []
            )


def test_unrelated_failure_while_handling_busy_keeps_its_own_retry_classification(
    pipeline, monkeypatch
):
    engine, worker, http = pipeline
    assert http.post(
        "/api/publish_interaction", json=payload("not-busy", force=True)
    ).json()["success"]

    def unrelated(_service, *_args, **_kwargs):
        try:
            raise ExecutionInputBusyError("earlier handling context")
        except ExecutionInputBusyError:
            raise RuntimeError("different failure") from None

    monkeypatch.setattr(ProfileGenerationService, "compute_generation", unrelated)
    assert worker.drain_org("stream-test", 1, 300) == 0
    cursor, _, _ = _state(engine.get_storage())
    assert cursor["attempts"] == 1
    assert "RuntimeError" in cursor["last_error"]
