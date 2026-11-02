"""Real SQLite questions exercise isolated, bounded shared housekeeping."""

import logging
from datetime import UTC, datetime, timedelta
from threading import BoundedSemaphore, Event
from unittest.mock import MagicMock

import pytest

from reflexio.server.services.durable_learning import worker as workers
from reflexio.server.services.durable_learning.discovery import discover_work
from reflexio.server.services.durable_learning.expiry import ExpiryHousekeeping
from reflexio.server.services.durable_learning.scheduler import DurableLearningScheduler
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import (
    PendingToolCallRecord,
    PendingToolCallStatus,
    build_scope_hash,
    human_feedback_scope,
)
from tests.server.services.durable_learning.test_owned_resume_selection_integration import (
    _context,
)
from tests.server.services.storage import test_resume_candidate_selection as candidates

store = candidates.store
pytestmark = pytest.mark.integration


def _question(storage, identity):
    scope = human_feedback_scope(storage.org_id)
    return storage.create_pending_tool_call(
        PendingToolCallRecord(
            id=identity,
            org_id=storage.org_id,
            user_id="one",
            scope=scope,
            scope_hash=build_scope_hash(scope),
            tool_name="ask_human",
            dedup_key=identity,
            status=PendingToolCallStatus.PENDING,
            question_text="Synthetic expiry question",
            expires_at=datetime.now(UTC) - timedelta(seconds=1),
            cache_until=datetime.now(UTC) + timedelta(hours=1),
        )
    )


def test_partial_discovery_keeps_expiry_outside_model_capacity(store, monkeypatch):
    question = _question(store, "expiry-only")
    context = _context(store)

    def unavailable():
        raise OSError("synthetic learning source unavailable")

    monkeypatch.setattr(store, "list_extraction_orgs", unavailable)
    scheduler = DurableLearningScheduler(
        request_context_factory=lambda _: context,
        org_ids_provider=lambda: discover_work(store),
    )
    budget = BoundedSemaphore(1)
    assert budget.acquire(blocking=False)
    monkeypatch.setattr(workers, "_budget", budget)
    reserve = MagicMock(wraps=workers._reserve)
    monkeypatch.setattr(workers, "_reserve", reserve)
    claim = MagicMock(side_effect=AssertionError("Expiry must not claim a user"))
    monkeypatch.setattr(store, "claim_user_extraction", claim)
    try:
        scheduler._run_once()
        assert (
            store.get_pending_tool_call(question.id).status
            == PendingToolCallStatus.EXPIRED
        )
        claim.assert_not_called()
        reserve.assert_not_called()
    finally:
        budget.release()


def test_expiry_gate_logs_known_work_once_and_reenable_needs_no_callback(store, caplog):
    caplog.set_level(logging.INFO)
    question = _question(store, "gated")
    context = _context(store)
    config = context.configurator.get_config()
    config.pending_tool_call_config.enabled = False
    scheduler = DurableLearningScheduler(
        request_context_factory=lambda _: context,
        org_ids_provider=lambda: discover_work(store),
    )
    for _ in range(3):
        scheduler._run_once()
    assert (
        store.get_pending_tool_call(question.id).status == PendingToolCallStatus.PENDING
    )
    assert (
        sum("durable_expiry_skipped_gate_disabled" in r.message for r in caplog.records)
        == 1
    )
    config.pending_tool_call_config.enabled = True
    for _ in range(2):
        scheduler._run_once()
    assert (
        store.get_pending_tool_call(question.id).status == PendingToolCallStatus.EXPIRED
    )


def test_expiry_first_org_error_is_value_free_and_does_not_block_second(store, caplog):
    question = _question(store, "second-org")
    context = _context(store)

    def factory(org):
        if org == "bad-first":
            raise ValueError("SENTINEL_PRIVATE_QUESTION_AND_CREDENTIAL")
        return context

    ExpiryHousekeeping(factory, Event()).run(["bad-first", store.org_id])
    assert (
        store.get_pending_tool_call(question.id).status == PendingToolCallStatus.EXPIRED
    )
    assert "SENTINEL_PRIVATE_QUESTION_AND_CREDENTIAL" not in caplog.text
    assert all(r.exc_info is None for r in caplog.records)


def test_expiry_cursor_defers_new_ids_and_advances_past_refused_head(store):
    _question(store, "a")
    _question(store, "b")
    cursor, first = store._expired_question_page()
    assert first["id"] == "a" and cursor.through_id == "b"
    _question(store, "z")
    cursor, second = store._expired_question_page(cursor=cursor)
    assert second["id"] == "b" and cursor.through_id == "b"
    assert store._expired_question_page(cursor=cursor)[1] is None
    cursor, first = store._expired_question_page()
    assert first["id"] == "a" and cursor.through_id == "z"
    assert set(first) == {"id", "org_id", "project_id"}
    assert store.get_pending_tool_call("a").status == PendingToolCallStatus.PENDING


def test_expiry_is_org_scoped_even_when_another_org_shares_the_file(store):
    other = SQLiteStorage(org_id="other-org", db_path=store.db_path)
    try:
        own = _question(store, "own-question")
        foreign = _question(other, "foreign-question")
        _, descriptor = store._expired_question_page()
        assert descriptor["id"] == own.id and descriptor["org_id"] == store.org_id
        assert not store._expire_pending_question(foreign.id)
        assert store._expire_pending_question(own.id)
        assert other.get_pending_tool_call(foreign.id) == foreign
    finally:
        other.conn.close()


def test_stopping_housekeeping_never_constructs_another_context(store):
    event = Event()
    factory = MagicMock(
        side_effect=AssertionError("Stopped housekeeping cannot borrow")
    )
    event.set()
    ExpiryHousekeeping(factory, event).run([store.org_id])
    factory.assert_not_called()
