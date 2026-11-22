# ruff: noqa: S608 -- Fixed synthetic rows in fixture-owned disposable databases.
"""Queued-only closure is independently proved, never inferred from counters."""

import json

import pytest

from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
    SQLiteStreamSQL,
)
from reflexio.test_support.transition_closure import CASES, derive, mutate, seed

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _deny_providers(monkeypatch):
    def deny(*_a, **_kw):
        raise AssertionError("No provider is allowed in transition closure")

    for name in ("completion", "acompletion", "embedding", "aembedding"):
        monkeypatch.setattr(f"litellm.{name}", deny)


@pytest.mark.parametrize("history", [False, True])
def test_authentic_fresh_queue_is_allowed_after_virgin_or_completed_cursor(
    storage, history
):
    with storage.commit_scope(), storage._stream_sql() as db:
        seed(db, storage.org_id, "", history=history)
    result = derive(storage)
    assert result.ready, result.blockers
    assert result.counts["queued_admissions"] == 1
    assert result.counts["org_work_requiring_fleet_reconciliation"] == 1
    # Raw observation is intentionally still conservative.
    assert (
        storage.transition_inventory_page("work").items[0].execution_bearing
        is not False
    )


@pytest.mark.parametrize("case", CASES)
def test_missing_authentication_or_generation_never_becomes_queued_only(storage, case):
    with storage.commit_scope(), storage._stream_sql() as db:
        seed(db, storage.org_id, "")
        mutate(db, storage.org_id, "", case)
    result = derive(storage)
    assert not result.ready, (case, result.counts)
    assert result.blockers


def test_replayed_admission_page_is_refused_without_unbounded_iteration(
    storage, monkeypatch
):
    with storage.commit_scope(), storage._stream_sql() as db:
        seed(db, storage.org_id, "")
    query = SQLiteStreamSQL.query
    saved = []
    reads = []

    def replay(db, sql, params=()):
        if sql.startswith("SELECT r.user_id,r.request_id,r.learning_admission"):
            reads.append(1)
            if saved:
                return saved[0]
            rows = query(db, sql, params)
            saved.append(rows)
            return rows
        return query(db, sql, params)

    monkeypatch.setattr(SQLiteStreamSQL, "query", replay)
    result = derive(storage)
    assert not result.ready
    assert "observation_did_not_advance" in result.blockers
    assert len(reads) == 2


@pytest.mark.parametrize("missing_coverage", [False, True])
def test_processed_admission_coverage_is_proven_in_bounded_batches(
    storage, monkeypatch, missing_coverage
):
    count = 121
    with storage.commit_scope(), storage._stream_sql() as db:
        seed(db, storage.org_id, "", history=True)
        for seq in range(3, count + 1):
            admitted_at = 3 if missing_coverage and seq == count else 1
            admission = {
                "profile": {"eligible": True},
                "playbook": {"eligible": True},
                "min_seq": seq,
                "max_seq": seq,
                "admitted_at": admitted_at,
            }
            request = f"request-{seq:04}"
            db.query(
                f"INSERT INTO {db.table('requests')}(request_id,user_id,created_at,session_id,learning_admission) "
                "VALUES(?,'user','2026-01-01','session',?)",
                (request, json.dumps(admission)),
            )
            db.query(
                f"INSERT INTO {db.table('interactions')}(interaction_id,user_id,created_at,request_id,ingestion_seq) "
                "VALUES(?,'user','2026-01-01',?,?)",
                (seq, request, seq),
            )
        db.query(f"UPDATE {db.table('learning_work')} SET highwater=?", (count,))
        db.query(
            f"UPDATE {db.table('extraction_cursors')} SET completed_seq=?,started=1",
            (count,),
        )
        db.query(f"UPDATE {db.table('extraction_windows')} SET end_seq=?", (count,))
        db.query(
            f"INSERT INTO {db.table('extraction_windows')}(window_id,user_id,kind,org_id,project_id,predecessor,end_seq,manifest,policy,completed,effects_done,completed_at) "
            "VALUES('playbook-past','user','playbook',?,'',0,?,'[]','{}',1,1,2)",
            (storage.org_id, count),
        )

    query = SQLiteStreamSQL.query
    coverage_batches = []

    def count_queries(db, sql, params=()):
        if sql.startswith("WITH pending_coverage") and "w.end_seq>=p.seq" in sql:
            coverage_batches.append((len(params) - 3) // 4)
        return query(db, sql, params)

    monkeypatch.setattr(SQLiteStreamSQL, "query", count_queries)
    result = derive(storage)
    assert result.ready is not missing_coverage, result.blockers
    assert result.blockers == (
        ("processed_admission_coverage_unproven",) if missing_coverage else ()
    )
    assert result.counts["admissions"] == count
    assert result.counts["queued_admissions"] == 0
    assert coverage_batches == [100, 100, 40]


def test_cursor_completion_still_requires_exact_window_end(storage):
    with storage.commit_scope(), storage._stream_sql() as db:
        seed(db, storage.org_id, "", history=True)
        db.query(f"UPDATE {db.table('extraction_windows')} SET end_seq=2")
    result = derive(storage)
    assert not result.ready
    assert "cursor_completion_unproven" in result.blockers
