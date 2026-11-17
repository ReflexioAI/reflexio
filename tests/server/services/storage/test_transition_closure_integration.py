# ruff: noqa: S608 -- Fixed synthetic rows in fixture-owned disposable databases.
"""Queued-only closure is independently proved, never inferred from counters."""

import pytest

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
    from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
        SQLiteStreamSQL,
    )

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
