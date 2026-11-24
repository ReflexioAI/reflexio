# ruff: noqa: S608 -- Fixed synthetic rows in fixture-owned disposable databases.
"""Queued-only closure is independently proved, never inferred from counters."""

import json
import sqlite3

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
@pytest.mark.parametrize("restricted", [False, True])
def test_processed_admission_coverage_is_proven_in_bounded_batches(
    storage, monkeypatch, missing_coverage, restricted
):
    if restricted:
        storage.conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 503)
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
            coverage_batches.append((len(params) - 3) // 5)
        return query(db, sql, params)

    monkeypatch.setattr(SQLiteStreamSQL, "query", count_queries)
    result = derive(storage)
    assert result.ready is not missing_coverage, result.blockers
    assert result.blockers == (
        ("processed_admission_coverage_unproven",) if missing_coverage else ()
    )
    assert result.counts["admissions"] == count
    assert result.counts["queued_admissions"] == 0
    cap = SQLiteStreamSQL(storage.conn).transition_coverage_batch_limit()
    assert coverage_batches == [min(cap, 240 - start) for start in range(0, 240, cap)]
    if restricted:
        assert cap == 100


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


@pytest.mark.parametrize("erased", [False, True])
@pytest.mark.parametrize("missing_date", [False, True])
def test_closed_detached_history_proves_numeric_floor_not_publication(
    storage, erased, missing_date
):
    from dataclasses import asdict

    from reflexio.server.services.storage.storage_base.transition_inventory import (
        _projection,
    )
    from reflexio.test_support.transition_closure import seed_closed_window_history

    with storage.commit_scope(), storage._stream_sql() as db:
        seed_closed_window_history(db, storage.org_id, "", erased=erased)
        if missing_date:
            db.query("UPDATE extraction_windows SET completed_at=NULL")
        table, _keys, scope, columns = _projection(db, "runs", db.now())
        row = db.query(
            f"SELECT {columns} FROM {table} r WHERE {scope}", (storage.org_id,)
        )[0]
        assert row["inert_history"]
        assert (
            not row["published"]
            and not row["legacy_published"]
            and not row["window_published"]
        )
    result = derive(storage)
    assert result.ready, result.blockers
    assert result.counts["windows"] == 1
    window = storage.transition_inventory_page("windows").items[0]
    assert window.inert_history and window.execution_bearing is False
    item = storage.transition_inventory_page("runs").items[0]
    assert item.inert_history and item.execution_bearing is False
    assert item.saved_output is not erased
    assert "PRIVATE_SAVED_CANARY" not in json.dumps(asdict(item))


@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE extraction_windows SET completed=0",
        "UPDATE extraction_windows SET effects_done=0",
        "UPDATE extraction_windows SET invalidated=1",
        "UPDATE extraction_cursors SET window_id='closed'",
        "UPDATE learning_work SET lease_until=9999999999",
        "UPDATE _agent_runs SET status='agent_completed'",
        "UPDATE _agent_runs SET extractor_kind='playbook'",
        "UPDATE _agent_runs SET pending_tool_call_ids='[\"call\"]'",
        "UPDATE extraction_windows SET effects=json_set(effects,'$.billing',json('{}'))",
        "UPDATE extraction_windows SET effects=json_set(effects,'$.billing.pipeline','playbook')",
        'UPDATE _agent_runs SET generation_request_snapshot=\'{"overflow":{"checkpoint":{"active_child":"child"}}}\'',
        'UPDATE extraction_windows SET effects=\'{"version":1,"skipped":true,"billing":null,"request_ids":[" \\t\\n"]}\'',
        'UPDATE extraction_windows SET effects=\'{"version":1,"skipped":true,"billing":null,"request_ids":[],"acknowledged_window_run_id":"window:closed"}\'',
    ],
)
def test_closed_history_never_hides_remaining_execution_or_new_formats(
    storage, mutation
):
    from reflexio.test_support.transition_closure import seed_closed_window_history

    with storage.commit_scope(), storage._stream_sql() as db:
        seed_closed_window_history(db, storage.org_id, "")
        db.query(mutation)
    assert not derive(storage).ready
    item = storage.transition_inventory_page("runs").items[0]
    assert not item.inert_history


@pytest.mark.parametrize("erased", [False, True])
def test_closed_history_and_nonintake_agree_with_attributed_cursor_reader(
    storage, erased
):
    from reflexio.server.services.storage.storage_base.transition_inventory import (
        TransitionInventoryAttribution,
    )
    from reflexio.test_support.transition_closure import seed_closed_window_history

    with storage.commit_scope(), storage._stream_sql() as db:
        seed_closed_window_history(db, storage.org_id, "", erased=erased)
        db.query(
            "INSERT INTO requests(request_id,user_id,created_at,session_id) VALUES('shell','history-user','2026-01-01','session')"
        )
        db.query(
            "INSERT INTO interactions(interaction_id,user_id,created_at,request_id) VALUES(1,'history-user','2026-01-01','shell')"
        )
    assert derive(storage).ready
    page = storage.transition_inventory_page(
        "cursors",
        attribution=TransitionInventoryAttribution(
            storage.org_id, "", storage._durable_work_route()
        ),
    )
    assert page.items[0].execution_bearing is False, page.items[0].reasons


def test_erased_numeric_floor_is_not_positive_processed_admission_coverage(storage):
    from reflexio.test_support.transition_closure import seed_closed_window_history

    with storage.commit_scope(), storage._stream_sql() as db:
        seed_closed_window_history(db, storage.org_id, "", erased=True)
        db.query(
            "INSERT INTO requests(request_id,user_id,created_at,session_id,learning_admission) VALUES('live','history-user','2026-01-01','session',?)",
            (
                json.dumps(
                    {
                        "profile": {"eligible": True},
                        "playbook": {"eligible": False},
                        "min_seq": 1,
                        "max_seq": 1,
                        "admitted_at": 0,
                    }
                ),
            ),
        )
        db.query(
            "INSERT INTO interactions(interaction_id,user_id,created_at,request_id,ingestion_seq) VALUES(1,'history-user','2026-01-01','live',1)"
        )
    result = derive(storage)
    assert not result.ready
    assert "processed_admission_coverage_unproven" in result.blockers


def test_closed_parent_lookup_uses_window_primary_key_and_unrelated_cursor_sets(
    storage,
):
    from reflexio.server.services.storage.storage_base.transition_inventory import (
        _projection,
    )
    from reflexio.test_support.transition_closure import seed_closed_window_history

    with storage.commit_scope(), storage._stream_sql() as db:
        seed_closed_window_history(db, storage.org_id, "")
        table, _keys, scope, columns = _projection(db, "runs", db.now())
        plan = db.query(
            f"EXPLAIN QUERY PLAN SELECT {columns} FROM {table} r WHERE {scope}",
            (storage.org_id,),
        )
    details = [row["detail"] for row in plan]
    assert any(
        "SEARCH w USING INDEX" in detail and "window_id=?" in detail
        for detail in details
    )
    assert any("LIST SUBQUERY" in detail for detail in details)
    assert not any("CORRELATED LIST SUBQUERY" in detail for detail in details)


@pytest.mark.parametrize("history", [False, True])
def test_original_clockless_admission_retains_native_queue_semantics(storage, history):
    from reflexio.server.services.storage.storage_base.transition_inventory import (
        TransitionInventoryAttribution,
    )

    with storage.commit_scope(), storage._stream_sql() as db:
        seed(db, storage.org_id, "", history=history)
        rows = db.query("SELECT request_id,learning_admission FROM requests")
        for row in rows:
            admission = json.loads(row["learning_admission"])
            del admission["admitted_at"]
            db.query(
                "UPDATE requests SET learning_admission=? WHERE request_id=?",
                (json.dumps(admission), row["request_id"]),
            )
    result = derive(storage)
    assert result.ready, result.blockers
    assert result.counts["queued_admissions"] == 1
    page = storage.transition_inventory_page(
        "cursors",
        attribution=TransitionInventoryAttribution(
            storage.org_id, "", storage._durable_work_route()
        ),
    )
    assert all("cursor_source_unproven" not in item.reasons for item in page.items)
    with storage._stream_sql() as db:
        inputs = storage._inputs(db, "user", "profile", 0, 100)
    assert [row["ingestion_seq"] for row in inputs] == [1, 2]
    assert all(
        "admitted_at" not in json.loads(row["learning_admission"]) for row in inputs
    )


@pytest.mark.parametrize("stamp", [None, True, -1, "old", float("inf")])
def test_present_invalid_admission_clock_is_not_a_legacy_format(storage, stamp):
    with storage.commit_scope(), storage._stream_sql() as db:
        seed(db, storage.org_id, "")
        admission = json.loads(
            db.query("SELECT learning_admission FROM requests")[0]["learning_admission"]
        )
        admission["admitted_at"] = stamp
        db.query("UPDATE requests SET learning_admission=?", (json.dumps(admission),))
    assert "admission_authentication_unproven" in derive(storage).blockers


@pytest.mark.parametrize(
    "case", ["exact", "wrong-request", "nul-prefix", "retired", "early-date"]
)
def test_clocked_admission_needs_exact_undated_ack_or_valid_dated_coverage(
    storage, case
):
    from reflexio.server.services.storage.storage_base.transition_inventory import (
        TransitionInventoryAttribution,
    )

    with storage.commit_scope(), storage._stream_sql() as db:
        seed(db, storage.org_id, "", history=True)
        effect = {
            "version": 1,
            "skipped": True,
            "billing": None,
            "request_ids": [
                {
                    "wrong-request": "different",
                    "nul-prefix": "request-1\0suffix",
                }.get(case, "request-1")
            ],
        }
        if case == "retired":
            effect["retired"] = True
        db.query(
            "UPDATE extraction_windows SET completed_at=?,effects=?,derived_claimed=1 WHERE window_id='past'",
            (0 if case == "early-date" else None, json.dumps(effect)),
        )

    closure = derive(storage)
    assert closure.ready is (case == "exact"), closure.blockers
    if case != "exact":
        assert "processed_admission_coverage_unproven" in closure.blockers
    page = storage.transition_inventory_page(
        "cursors",
        attribution=TransitionInventoryAttribution(
            storage.org_id, "", storage._durable_work_route()
        ),
    )
    profile = next(item for item in page.items if item.key[1] == "profile")
    assert ("cursor_source_unproven" not in profile.reasons) is (case == "exact")


@pytest.mark.parametrize("clockless", [False, True])
def test_native_retention_hole_preserves_range_but_selects_only_retained_inputs(
    storage, clockless
):
    from reflexio.models.api_schema.domain.entities import DeleteUserInteractionRequest
    from tests.server.services.storage.test_storage_contract_extraction_stream import (
        publish,
    )

    items = publish(storage, 3, width=2, stride=1)
    storage.delete_user_interaction(
        DeleteUserInteractionRequest(
            user_id="u", interaction_id=items[1].interaction_id
        )
    )
    if clockless:
        with storage.commit_scope(), storage._stream_sql() as db:
            admission = json.loads(
                db.query("SELECT learning_admission FROM requests")[0][
                    "learning_admission"
                ]
            )
            del admission["admitted_at"]
            db.query(
                "UPDATE requests SET learning_admission=?", (json.dumps(admission),)
            )
    closure = derive(storage)
    assert closure.ready, closure.blockers
    user, token = storage.claim_extraction("retained-native-owner", 300)
    window = storage.prepare_extraction(user, token)
    assert window is not None
    assert [item["seq"] for item in window.manifest] == [1, 3]
    assert [item["interaction_id"] for item in window.manifest] == [
        items[0].interaction_id,
        items[2].interaction_id,
    ]
    assert window.predecessor == 0 and window.end_seq == 3
    with storage._stream_sql() as db:
        admission = json.loads(
            db.query("SELECT learning_admission FROM requests")[0]["learning_admission"]
        )
        assert (admission["min_seq"], admission["max_seq"]) == (1, 3)
        assert db.query("SELECT highwater FROM learning_work")[0]["highwater"] == 3


@pytest.mark.parametrize("capped", [False, True])
def test_source_inventory_uses_bounded_large_pages_without_raising_row_cap(
    storage, monkeypatch, capped
):
    from reflexio.server.services.storage.storage_base import (
        transition_closure as owner,
    )
    from reflexio.server.services.storage.storage_base import (
        transition_inventory as inventory_owner,
    )

    with storage.commit_scope():
        storage.conn.executemany(
            "INSERT INTO _agent_runs(id,org_id,extractor_kind,user_id,request_id,status,generation_request_snapshot,service_config_snapshot,agent_context_snapshot) VALUES(?,?,'profile','historical-user','request','finalized','{}','{}','')",
            [(f"history-{i:04}", storage.org_id) for i in range(501)],
        )
    read = inventory_owner.read_transition_inventory_page_db
    calls = []

    def observe(db, source, **kwargs):
        if source == "runs":
            calls.append(kwargs["limit"])
        return read(db, source, **kwargs)

    monkeypatch.setattr(inventory_owner, "read_transition_inventory_page_db", observe)
    if capped:
        monkeypatch.setattr(owner, "_MAX_PAGES", 1)
    result = derive(storage)
    if capped:
        assert result.blockers == ("observation_limit_reached",)
        assert calls == [500]
    else:
        assert result.ready, result.blockers
        assert result.counts["runs"] == 501
        assert calls == [500, 500, 500]


@pytest.mark.parametrize("virgin", [False, True])
@pytest.mark.parametrize("bad", [False, True])
def test_thousand_cursor_proofs_are_batched_and_a_bad_batch_certifies_nothing(
    storage, monkeypatch, virgin, bad
):
    from reflexio.server.services.storage.storage_base import (
        transition_closure as owner,
    )

    users = [f"batch-user-{i:04}" for i in range(1000)]
    with storage.commit_scope():
        storage.conn.executemany(
            "INSERT INTO learning_work(org_id,user_id,highwater) VALUES(?,?,?)",
            [(storage.org_id, user, 0 if virgin else 1) for user in users],
        )
        storage.conn.executemany(
            "INSERT INTO extraction_cursors(user_id,kind,project_id,completed_seq,started) VALUES(?,'profile','',?,?)",
            [(user, 0 if virgin else 1, 0 if virgin else 1) for user in users],
        )
        if not virgin:
            storage.conn.executemany(
                "INSERT INTO extraction_windows(window_id,user_id,kind,org_id,project_id,predecessor,end_seq,manifest,policy,completed,effects_done,completed_at) VALUES(?,?,'profile',?,'',0,1,'[]','{}',1,1,1)",
                [
                    (f"batch-window-{i:04}", user, storage.org_id)
                    for i, user in enumerate(users)
                ],
            )
        if bad and virgin:
            storage.conn.execute(
                "INSERT INTO extraction_windows(window_id,user_id,kind,org_id,project_id,predecessor,end_seq,manifest,policy,completed,effects_done,completed_at) VALUES('unexpected-window',?,'profile',?,'',0,1,'[]','{}',1,1,1)",
                (users[0], storage.org_id),
            )
        elif bad:
            storage.conn.execute(
                "UPDATE extraction_windows SET end_seq=2 WHERE window_id='batch-window-0000'"
            )
    query = SQLiteStreamSQL.query
    batches = []

    def counted(db, sql, params=()):
        if sql.startswith("WITH virgin_cursors"):
            batches.append((len(params) - 2) // 2)
        elif sql.startswith("WITH pending_coverage"):
            batches.append((len(params) - 3) // 5)
        return query(db, sql, params)

    monkeypatch.setattr(SQLiteStreamSQL, "query", counted)
    result = derive(storage)
    assert result.ready is not bad, result.blockers
    assert result.counts["cursors"] == 1000
    cap = SQLiteStreamSQL(storage.conn).transition_coverage_batch_limit()
    assert batches == [min(cap, 1000 - start) for start in range(0, 1000, cap)]
    if bad:
        assert "cursor_completion_unproven" in result.blockers
    # The failed batch must not certify its otherwise-good siblings.
    batches.clear()
    with storage._stream_sql() as db:
        blockers = set()
        frontiers = owner._cursor_frontiers(
            db, storage.org_id, "", db.now(), blockers, {}
        )
    assert len(frontiers) == (1000 - cap if bad else 1000)
    assert (
        (users[1], "profile") in frontiers
        if not bad
        else (users[1], "profile") not in frontiers
    )


@pytest.mark.parametrize("status", ["finalized", "failed", "running"])
def test_first_native_admission_does_not_confuse_generic_history_with_window_lineage(
    storage, status
):
    from tests.server.services.storage.test_storage_contract_extraction_stream import (
        publish,
    )

    with storage.commit_scope():
        storage.conn.execute(
            "INSERT INTO _agent_runs(id,org_id,extractor_kind,user_id,request_id,status,generation_request_snapshot,committed_output) VALUES('old-generic',?,'profile','u','old-request',?,'{}','{}')",
            (storage.org_id, status),
        )
    publish(storage, 2, width=2, stride=1)
    result = derive(storage)
    assert result.ready is (status == "finalized"), result.blockers
    if status != "finalized":
        assert "cursor_completion_unproven" in result.blockers
