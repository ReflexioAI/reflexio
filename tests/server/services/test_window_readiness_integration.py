"""Dependency readiness defers a durable window, never its failure budget."""

import logging
from unittest.mock import Mock

import pytest

from reflexio.server.services.durable_learning.window_executor import WindowExecutor
from reflexio.server.services.durable_learning.worker import DurableLearningWorker
from reflexio.server.services.profile.service import ProfileGenerationService
from reflexio.server.services.storage.error import ReadinessUnavailableError
from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
    SQLiteStreamSQL,
)
from reflexio.server.services.storage.storage_base._extraction_stream import (
    MAX_EXTRACTION_WINDOW_ATTEMPTS,
)
from tests.server.services.test_durable_window_pipeline import payload
from tests.server.services.test_durable_window_pipeline import pipeline as pipeline


def _state(storage):
    with storage._stream_sql(read_only=True) as db:
        cursor = db.query(
            "SELECT window_id,attempts,retry_at,last_error FROM extraction_cursors "
            "WHERE user_id=? AND kind=?",
            ("u", "profile"),
        )[0]
        windows = db.query(
            "SELECT window_id,outcome,completed,effects_done FROM extraction_windows WHERE user_id=?",
            ("u",),
        )
        work = db.query(
            "SELECT due_at,lease_token FROM learning_work WHERE user_id=?",
            ("u",),
        )[0]
    return cursor, windows, work


@pytest.mark.parametrize("phase", ["compute", "persist"])
def test_readiness_defers_without_parking_and_recovers_once(
    pipeline, monkeypatch, caplog, phase
):
    from reflexio.server import background_work

    monkeypatch.setattr(
        background_work,
        "_classifier",
        lambda exc: isinstance(exc, ReadinessUnavailableError),
    )
    monkeypatch.setattr(background_work, "_streaks", {})
    engine, worker, http = pipeline
    assert http.post(
        "/api/publish_interaction", json=payload("readiness", force=True)
    ).json()["success"]
    storage = engine.get_storage()
    with storage._stream_sql(read_only=True) as db:
        clock = [db.now()]
    monkeypatch.setattr(SQLiteStreamSQL, "now", lambda _: clock[0])
    monkeypatch.setattr(background_work, "_monotonic", lambda: clock[0])
    ready = False
    method = "compute_generation" if phase == "compute" else "persist_generation"
    original = getattr(ProfileGenerationService, method)

    def qualified(service, *args, **kwargs):
        if not ready:
            raise ReadinessUnavailableError("SENTINEL_CUSTOMER_TEXT")
        return original(service, *args, **kwargs)

    monkeypatch.setattr(ProfileGenerationService, method, qualified)
    compute = Mock(wraps=ProfileGenerationService.compute_generation)
    # A descriptor-preserving wrapper counts real compute calls, including
    # refusals, without changing method binding.
    monkeypatch.setattr(
        ProfileGenerationService,
        "compute_generation",
        lambda self, *args, **kwargs: compute(self, *args, **kwargs),
    )
    window_id = None
    saved_outcome = None
    with caplog.at_level(logging.WARNING):
        for attempt in range(MAX_EXTRACTION_WINDOW_ATTEMPTS + 1):
            if attempt == 3:
                clock[0] += 301  # a persistent outage must still escalate
            assert worker.drain_org("stream-test", 1, 300) == 0
            cursor, windows, work = _state(storage)
            window_id = window_id or cursor["window_id"]
            assert cursor["window_id"] == window_id
            assert cursor["attempts"] == 0
            assert cursor["retry_at"] != float("inf")
            assert cursor["last_error"] is None
            assert len(windows) == 1 and not windows[0]["completed"]
            assert work == {"due_at": clock[0] + 5, "lease_token": None}
            assert storage.count_all_profiles() == 0
            assert len(background_work._streaks) == 1
            if phase == "persist":
                saved_outcome = saved_outcome or windows[0]["outcome"]
                assert saved_outcome is not None
                assert windows[0]["outcome"] == saved_outcome
                assert compute.call_count == 1
            calls = compute.call_count
            assert worker.drain_org("stream-test", 1, 300) == 0
            assert compute.call_count == calls, "deferral must honor its due time"
            clock[0] += 6
    assert "SENTINEL_CUSTOMER_TEXT" not in caplog.text
    assert any(
        record.levelno == logging.ERROR and "escalated=true" in record.message
        for record in caplog.records
    )
    ready = True
    assert worker.drain_org("stream-test", 1, 300) == 1
    cursor, windows, _ = _state(storage)
    assert windows[0]["window_id"] == window_id and windows[0]["completed"]
    assert cursor["attempts"] == 0
    assert background_work._streaks == {}, (
        "only completed work clears its failure streak"
    )
    assert storage.extraction_status("u", "readiness")["status"] == "done"
    count = storage.count_all_profiles()
    assert count > 0
    assert worker.drain_org("stream-test", 1, 300) == 1  # owed effects, not computation
    assert _state(storage)[1][0]["effects_done"]
    assert worker.drain_org("stream-test", 1, 300) == 0
    assert storage.count_all_profiles() == count
    if phase == "persist":
        assert compute.call_count == 1


@pytest.mark.parametrize("owner_change", ["replacement", "erasure"])
def test_stale_readiness_deferral_cannot_change_new_owner_or_restore_erased_work(
    pipeline, owner_change
):
    engine, _, http = pipeline
    assert http.post(
        "/api/publish_interaction", json=payload("stale", force=True)
    ).json()["success"]
    storage = engine.get_storage()
    user, token = storage.claim_extraction("old", 300)
    window = storage.prepare_extraction(user, token)
    assert window is not None
    if owner_change == "replacement":
        with storage._stream_sql() as db:
            db.query("UPDATE learning_work SET lease_until=0 WHERE user_id=?", (user,))
        replacement = storage.claim_user_extraction(user, "new", 300)
        assert replacement is not None and replacement != token
        before = _state(storage)
    else:
        storage.clear_user_data(user)
    result = DurableLearningWorker._handle_window_failure(
        storage, window, token, ReadinessUnavailableError("not ready")
    )
    storage.release_user_extraction(user, token)
    assert result == "readiness_deferred"
    if owner_change == "replacement":
        assert _state(storage) == before
        assert before[2]["lease_token"] == replacement
        storage.release_user_extraction(user, replacement)
    else:
        with storage._stream_sql(read_only=True) as db:
            for sql in (
                "SELECT user_id FROM learning_work WHERE user_id=?",
                "SELECT user_id FROM extraction_windows WHERE user_id=?",
                "SELECT user_id FROM extraction_cursors WHERE user_id=?",
            ):
                assert db.query(sql, (user,)) == []


def test_readiness_during_effect_delivery_never_recomputes_completed_window(
    pipeline, monkeypatch
):
    engine, worker, http = pipeline
    assert http.post(
        "/api/publish_interaction", json=payload("effects-ready", force=True)
    ).json()["success"]
    assert worker.drain_org("stream-test", 1, 300) == 1
    storage = engine.get_storage()
    before = _state(storage)
    assert before[1][0]["completed"] and not before[1][0]["effects_done"]
    compute = Mock(
        side_effect=AssertionError("Completed output must not be recomputed")
    )
    monkeypatch.setattr(ProfileGenerationService, "compute_generation", compute)
    deliver = WindowExecutor.deliver
    refused = Mock(side_effect=ReadinessUnavailableError("not qualified"))
    monkeypatch.setattr(WindowExecutor, "deliver", refused)
    worker.drain_org("stream-test", 1, 300)
    after = _state(storage)
    assert after[0] == before[0]
    assert after[1] == before[1]
    assert after[2]["lease_token"] is None
    assert storage.extraction_status("u", "effects-ready")["status"] == "done"
    with storage._stream_sql(read_only=True) as db:
        now = db.now()
    monkeypatch.setattr(SQLiteStreamSQL, "now", lambda _: now + 6)
    monkeypatch.setattr(WindowExecutor, "deliver", deliver)
    assert worker.drain_org("stream-test", 1, 300) == 1
    assert _state(storage)[1][0]["effects_done"]
    compute.assert_not_called()
