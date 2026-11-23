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


@pytest.mark.parametrize("inputs", [(), (None,), (None, None)])
def test_unadmitted_request_and_unsequenced_inputs_are_not_durable_intake(
    storage, inputs
):
    with storage.commit_scope(), storage._stream_sql() as db:
        db.query(
            f"INSERT INTO {db.table('requests')}(request_id,user_id,created_at,session_id) "
            "VALUES('shell','user','2026-01-01','session')"
        )
        for index, sequence in enumerate(inputs, 1):
            db.query(
                f"INSERT INTO {db.table('interactions')}(interaction_id,user_id,created_at,request_id,ingestion_seq) "
                "VALUES(?,'user','2026-01-01','shell',?)",
                (index, sequence),
            )
        # Use the real execution selector, not just the closure implementation.
        for kind in ("profile", "playbook"):
            assert storage._inputs(db, "user", kind, 0, 10) == []
    result = derive(storage)
    assert result.ready, result.blockers
    assert result.counts["admissions"] == 0
    assert result.counts["queued_admissions"] == 0


@pytest.mark.parametrize("admission", ["null", "{}", "[]", '"not-admission"'])
def test_nonintake_does_not_accept_json_null_or_malformed_admission(storage, admission):
    with storage.commit_scope(), storage._stream_sql() as db:
        db.query(
            f"INSERT INTO {db.table('requests')}(request_id,user_id,created_at,session_id,learning_admission) "
            "VALUES('shell','user','2026-01-01','session',?)",
            (admission,),
        )
    result = derive(storage)
    assert not result.ready
    assert "admission_authentication_unproven" in result.blockers


@pytest.mark.parametrize("sequences", [(1,), (None, 1)])
def test_unadmitted_request_with_any_sequenced_input_still_refuses(storage, sequences):
    with storage.commit_scope(), storage._stream_sql() as db:
        db.query(
            f"INSERT INTO {db.table('requests')}(request_id,user_id,created_at,session_id) "
            "VALUES('shell','user','2026-01-01','session')"
        )
        for index, sequence in enumerate(sequences, 1):
            db.query(
                f"INSERT INTO {db.table('interactions')}(interaction_id,user_id,created_at,request_id,ingestion_seq) "
                "VALUES(?,'user','2026-01-01','shell',?)",
                (index, sequence),
            )
    result = derive(storage)
    assert not result.ready
    assert "admission_authentication_unproven" in result.blockers


def test_nonintake_does_not_hide_another_users_input(storage):
    with storage.commit_scope(), storage._stream_sql() as db:
        db.query(
            f"INSERT INTO {db.table('requests')}(request_id,user_id,created_at,session_id) "
            "VALUES('shell','user','2026-01-01','session')"
        )
        db.query(
            f"INSERT INTO {db.table('interactions')}(interaction_id,user_id,created_at,request_id) "
            "VALUES(1,'other','2026-01-01','shell')"
        )
    result = derive(storage)
    assert not result.ready
    assert "orphan_source_input" in result.blockers


@pytest.mark.parametrize("source", ["running", "window", "terminal-history"])
def test_nonintake_keeps_independent_execution_checks(storage, source):
    with storage.commit_scope(), storage._stream_sql() as db:
        db.query(
            f"INSERT INTO {db.table('requests')}(request_id,user_id,created_at,session_id) "
            "VALUES('shell','user','2026-01-01','session')"
        )
        if source == "window":
            db.query(
                f"INSERT INTO {db.table('extraction_windows')}(window_id,user_id,kind,org_id,project_id,predecessor,end_seq,manifest,policy) "
                "VALUES('window','user','profile',?,'',0,1,'[]','{}')",
                (storage.org_id,),
            )
        else:
            db.query(
                f"INSERT INTO {db.table('_agent_runs')}(id,org_id,user_id,extractor_kind,request_id,status) "
                "VALUES('run',?,'user','profile','shell',?)",
                (storage.org_id, "running" if source == "running" else "finalized"),
            )
    result = derive(storage)
    assert "admission_authentication_unproven" not in result.blockers
    if source == "terminal-history":
        assert result.ready, result.blockers
    else:
        assert not result.ready
        assert (
            "runs_not_closed" if source == "running" else "windows_not_closed"
        ) in result.blockers


def test_deleted_allocation_tail_is_not_required_or_reused(storage):
    with storage.commit_scope(), storage._stream_sql() as db:
        seed(db, storage.org_id, "")
    # Exercise the real deletion owner. It removes requests/inputs while the
    # monotonic allocator and its virgin counters deliberately remain intact.
    storage.delete_request("request")
    result = derive(storage)
    assert result.ready, result.blockers
    with storage.commit_scope(), storage._stream_sql() as db:
        assert (
            db.query(f"SELECT highwater FROM {db.table('learning_work')}")[0][
                "highwater"
            ]
            == 2
        )
        storage.lock_extraction_stream("user")
        db.query(
            f"INSERT INTO {db.table('requests')}(request_id,user_id,created_at,session_id) "
            "VALUES('next','user','2026-01-01','session')"
        )
        db.query(
            f"INSERT INTO {db.table('interactions')}(interaction_id,user_id,created_at,request_id) "
            "VALUES(3,'user','2026-01-01','next')"
        )
        storage.admit_extraction(
            "user",
            "next",
            [3],
            {"profile": {"eligible": True}, "playbook": {"eligible": False}},
        )
        assert (
            db.query(f"SELECT ingestion_seq FROM {db.table('interactions')}")[0][
                "ingestion_seq"
            ]
            == 3
        )
    result = derive(storage)
    assert result.ready, result.blockers
    assert result.counts["queued_admissions"] == 1


@pytest.mark.parametrize(
    "case", ["negative", "below-input", "missing-owner", "below-cursor", "zero-input"]
)
def test_retained_source_and_cursor_still_require_allocation_bounds(storage, case):
    with storage.commit_scope(), storage._stream_sql() as db:
        seed(db, storage.org_id, "")
        if case == "missing-owner":
            db.query(f"DELETE FROM {db.table('learning_work')}")
        elif case == "below-cursor":
            db.query(
                f"UPDATE {db.table('extraction_cursors')} SET completed_seq=3,started=1"
            )
        elif case == "zero-input":
            db.query(
                f"UPDATE {db.table('interactions')} SET ingestion_seq=0 WHERE interaction_id=1"
            )
        else:
            db.query(
                f"UPDATE {db.table('learning_work')} SET highwater=?",
                (-1 if case == "negative" else 1,),
            )
    result = derive(storage)
    assert not result.ready
    assert (
        "cursor_highwater_unproven"
        if case == "below-cursor"
        else "org_work_source_tail_unproven"
    ) in result.blockers


@pytest.mark.parametrize("nonintake", [True, False])
def test_nonintake_is_excluded_before_the_unchanged_page_limit(
    storage, monkeypatch, nonintake
):
    monkeypatch.setattr(
        "reflexio.server.services.storage.storage_base.transition_closure._MAX_PAGES", 2
    )
    with storage.commit_scope(), storage._stream_sql() as db:
        seed(db, storage.org_id, "")
        for index in range(201):
            db.query(
                f"INSERT INTO {db.table('requests')}(request_id,user_id,created_at,session_id,learning_admission) "
                "VALUES(?,'user','2026-01-01','session',?)",
                (f"shell-{index:04}", None if nonintake else "{}"),
            )
    result = derive(storage)
    if nonintake:
        assert result.ready, result.blockers
        assert result.counts["admissions"] == 1
        assert result.counts["queued_admissions"] == 1
    else:
        assert not result.ready
        assert "observation_limit_reached" in result.blockers
