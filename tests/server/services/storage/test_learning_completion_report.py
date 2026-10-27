"""Acceptance-to-coverage invariants against real SQLite storage."""

import json

import pytest

from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
    SQLiteStreamSQL,
    migrate_extraction_stream,
)
from reflexio.server.services.storage.storage_base._extraction_stream import (
    LeaseLostError,
)
from tests.server.services.storage.test_storage_contract_extraction_stream import (
    drain,
    publish,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def clock(monkeypatch):
    value = [1000.0]
    monkeypatch.setattr(SQLiteStreamSQL, "now", lambda _: value[0])
    return value


def report(storage, clock, *, page_size=100, max_requests=10_000):
    clock[0] = 1300.0
    return storage.learning_completion_report(
        1000, 1100, page_size=page_size, max_requests=max_requests
    )


@pytest.mark.parametrize(
    ("delay", "outcome"),
    [
        (0, "on_time"),
        (60, "on_time"),
        (60.01, "completed_late"),
        (120, "completed_late"),
    ],
)
def test_completion_includes_queue_delay_and_survives_ack(
    storage, clock, delay, outcome
):
    publish(storage, 2, width=2, stride=1)
    clock[0] += delay
    windows = drain(storage)
    assert len(windows) == 1
    row = storage.conn.execute(
        "SELECT completed_at,manifest,effects_done FROM extraction_windows"
    ).fetchone()
    assert row["completed_at"] == 1000 + delay
    assert row["effects_done"] == 1
    assert json.loads(row["manifest"]) == []
    result = report(storage, clock)
    assert result[outcome] == result["eligible_admissions"] == 1
    assert result["one_minute_fraction_lower_bound"] == (outcome == "on_time")


def test_all_required_kinds_and_all_publishes_not_worker_attempts(storage, clock):
    publish(storage, 1, request_id="a", width=2, stride=1, playbook=True)
    publish(storage, 1, request_id="b", width=2, stride=1, playbook=True)
    user, token = storage.claim_extraction("test", 300)
    window = storage.prepare_extraction(user, token)
    clock[0] = 1020
    storage.complete_extraction(window, token, {})
    storage.ack_extraction_effects(window.window_id)
    result = report(storage, clock)
    assert result["overdue_unfinished"] == 2
    clock[0] = 1070
    drain(storage)
    result = report(storage, clock, page_size=1)
    assert result["completed_late"] == 2
    assert result["on_time"] == 0


def test_waiting_retrying_and_parked_work_stay_in_denominator(storage, clock):
    publish(storage, 1, request_id="tail", user="tail")
    publish(storage, 2, request_id="retry", user="retry", width=2, stride=1)
    claim = storage.claim_extraction("test", 300)
    assert claim
    user, token = claim
    window = storage.prepare_extraction(user, token)
    storage.retry_extraction(window, token, "ProviderFailure")
    storage.conn.execute(
        "UPDATE extraction_cursors SET retry_at=2000 WHERE user_id='retry'"
    )
    storage.conn.commit()
    publish(storage, 1, request_id="parked", user="parked")
    storage.conn.execute(
        "UPDATE extraction_cursors SET last_error='retry_exhausted:Failure',retry_at=1e999 WHERE user_id='parked'"
    )
    result = report(storage, clock, page_size=1)
    assert result["eligible_admissions"] == result["overdue_unfinished"] == 3
    assert result["current_reasons"] == {
        "awaiting_coverage": 1,
        "retrying": 1,
        "retry_exhausted": 1,
    }


def test_missing_history_and_erasure_are_never_on_time(storage, clock):
    publish(storage, 2, request_id="old", width=2, stride=1)
    drain(storage)
    storage.conn.execute("UPDATE extraction_windows SET completed_at=NULL")
    storage.conn.commit()
    publish(storage, 2, request_id="erased", user="erased")
    storage.conn.execute(
        "DELETE FROM interactions WHERE user_id='erased' AND ingestion_seq=2"
    )
    storage.conn.commit()
    publish(storage, 1, request_id="disabled", user="disabled", eligible=False)
    publish(storage, 0, request_id="empty", user="empty")
    result = report(storage, clock)
    assert result["retained_admissions"] == 4
    assert result["eligible_admissions"] == result["unknown"] == 1
    assert result["excluded"] == 3
    assert result["on_time"] == 0


def test_context_overlap_does_not_replace_first_completion(storage, clock):
    publish(storage, 2, request_id="first", width=2, stride=1)
    clock[0] = 1020
    drain(storage)
    clock[0] = 1030
    publish(storage, 1, request_id="second", width=2, stride=1)
    clock[0] = 1100
    drain(storage)
    result = report(storage, clock, page_size=1)
    assert result["on_time"] == 1
    assert result["completed_late"] == 1


def test_suspension_and_retry_preserve_acceptance_clock(storage, clock):
    publish(storage, 2, width=2, stride=1)
    user, token = storage.claim_extraction("first", 300)
    original = storage.prepare_extraction(user, token)
    storage.suspend_extraction(original, token)
    clock[0] = 1030
    user, token = storage.claim_extraction("resumed", 300)
    resumed = storage.prepare_extraction(user, token)
    assert resumed.window_id == original.window_id
    storage.retry_extraction(resumed, token, "ProviderFailure")
    clock[0] = 1070
    drain(storage)
    result = report(storage, clock)
    assert result["retained_admissions"] == result["completed_late"] == 1


def test_completions_after_observation_cutoff_remain_unfinished(storage, clock):
    publish(storage, 2, width=2, stride=1)
    clock[0] = 1400
    drain(storage)
    # Models a page observing a completion made after the report's first clock.
    assert report(storage, clock)["overdue_unfinished"] == 1


def test_fenced_and_rolled_back_completions_cannot_create_success(storage, clock):
    publish(storage, 2, width=2, stride=1)
    user, token = storage.claim_extraction("test", 300)
    window = storage.prepare_extraction(user, token)
    with pytest.raises(RuntimeError, match="rollback"), storage.commit_scope():
        storage.complete_extraction(window, token, {})
        raise RuntimeError("rollback")
    assert (
        storage.conn.execute("SELECT completed_at FROM extraction_windows").fetchone()[
            0
        ]
        is None
    )
    with pytest.raises(LeaseLostError):
        storage.complete_extraction(window, "stale-token", {})
    assert report(storage, clock)["overdue_unfinished"] == 1


def test_report_rejects_partial_paging_and_immature_intervals(
    storage, clock, monkeypatch
):
    for index in range(3):
        publish(storage, 1, request_id=f"r{index}")
    with pytest.raises(ValueError, match="mature"):
        storage.learning_completion_report(1000, 1100)
    with pytest.raises(RuntimeError, match="max_requests"):
        report(storage, clock, page_size=1, max_requests=2)
    from reflexio.server.services.storage.storage_base import learning_completion

    original = learning_completion._page
    page = None

    def repeated(*args):
        nonlocal page
        if page is None:
            page = original(*args)
        return page

    monkeypatch.setattr(learning_completion, "_page", repeated)
    with pytest.raises(RuntimeError, match="did not advance"):
        report(storage, clock, page_size=1)


def test_sqlite_upgrade_does_not_invent_old_completion_times(storage, clock):
    publish(storage, 2, width=2, stride=1)
    drain(storage)
    storage.conn.execute("ALTER TABLE extraction_windows DROP COLUMN completed_at")
    migrate_extraction_stream(storage.conn)
    migrate_extraction_stream(storage.conn)
    assert report(storage, clock)["unknown"] == 1


def test_report_never_fetches_learning_content_and_uses_indexes(storage, clock):
    publish(storage, 1000, request_id="older")
    clock[0] = 1010
    publish(storage, 2, width=2, stride=1)
    statements = []
    storage.conn.set_trace_callback(statements.append)
    report(storage, clock)
    storage.conn.set_trace_callback(None)
    statement = next(sql for sql in statements if sql.startswith("WITH admissions"))
    assert "manifest" not in statement and "outcome" not in statement
    plan = " ".join(
        row[3] for row in storage.conn.execute("EXPLAIN QUERY PLAN " + statement)
    )
    assert "extraction_admission_time" in plan
    assert "extraction_window_completion" in plan
    assert plan.count("extraction_request_arrivals") >= 2
