"""Transition inspection uses real SQLite metadata, never runnable selectors."""

import json
from dataclasses import asdict

import pytest

from reflexio.server.services.storage.storage_base.transition_inventory import (
    INVENTORY_SOURCES,
    TransitionInventoryScopeError,
)

pytestmark = pytest.mark.integration


def seed_run(
    storage, run_id, *, status="running", snapshot=None, output=None, org=None
):
    with storage.commit_scope():
        storage.conn.execute(
            "INSERT INTO _agent_runs(id,org_id,extractor_kind,user_id,request_id,status,"
            "generation_request_snapshot,service_config_snapshot,agent_context_snapshot,committed_output) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                org or storage.org_id,
                "profile",
                "user",
                "request",
                status,
                json.dumps(
                    snapshot or {"output_schema_name": "StructuredProfilesOutput"}
                ),
                '{"secret":"NEVER_RETURN_CONFIG"}',
                "NEVER_RETURN_CONTEXT",
                json.dumps(output) if output is not None else None,
            ),
        )


def test_all_sources_require_an_empty_page_and_preserve_all_run_states(storage):
    states = (
        "running",
        "finalization_failed",
        "finalized_pending_tool",
        "cancelled",
        "arbitrary",
    )
    for index, state in enumerate(states):
        seed_run(storage, str(index), status=state)
    seed_run(storage, "other-org", org="other")
    for source in INVENTORY_SOURCES:
        page = storage.transition_inventory_page(source, limit=1)
        if not page.items:
            assert page.complete
            assert page.next_after is None
    after = None
    items = []
    for _ in range(10):
        page = storage.transition_inventory_page("runs", limit=2, after=after)
        items.extend(page.items)
        if page.complete:
            break
        assert page.items
        after = page.next_after
    else:
        pytest.fail("inventory did not exhaust")
    assert [item.key for item in items] == [(str(i),) for i in range(5)]
    assert [item.execution_bearing for item in items] == [True, True, True, False, None]
    assert all(not item.witness_known for item in items)
    assert items[-1].state is None


def test_witness_projection_is_content_free_and_missing_instruction_stays_unknown(
    storage,
):
    seed_run(
        storage,
        "overflow",
        snapshot={
            "output_schema_name": "StructuredProfilesOutput",
            "source_interaction_ids": [123],
            "secret_source": "NEVER_RETURN_SOURCE",
            "overflow": {
                "checkpoint": {
                    "stage": "reduce",
                    "children": {"one": {"output": "NEVER_RETURN_CHILD"}},
                    "extraction_instruction_contract": "a" * 64,
                    "extraction_policy": "b" * 64,
                }
            },
        },
        output={"output": "NEVER_RETURN_OUTPUT"},
    )
    seed_run(storage, "ordinary")
    page = storage.transition_inventory_page("runs")
    assert not page.complete
    ordinary, overflow = page.items
    assert not ordinary.witness_known
    assert overflow.witness_known
    assert overflow.instruction_digest == "a" * 64
    assert "overflow_requires_inspection" in overflow.reasons
    assert overflow.saved_output
    assert "NEVER_RETURN" not in json.dumps([asdict(item) for item in page.items])


def test_parked_stranded_window_effects_and_live_ownership_are_inventoried(storage):
    with storage.commit_scope():
        storage.conn.execute(
            "INSERT INTO learning_work(org_id,user_id,highwater,pending_since,lease_until) VALUES(?,?,?,?,?)",
            (storage.org_id, "user", 20, 1, 99999999999.0),
        )
        storage.conn.execute(
            "INSERT INTO extraction_cursors(user_id,kind,started,completed_seq,attempts,retry_at,last_error) "
            "VALUES(?,?,?,?,?,?,?)",
            ("user", "profile", 10, 0, 12, float("inf"), "NEVER_RETURN_ERROR"),
        )
        for key, completed, effects_done in (
            ("stranded", 0, 0),
            ("debt", 1, 0),
            ("inert", 1, 1),
        ):
            storage.conn.execute(
                "INSERT INTO extraction_windows(window_id,user_id,kind,org_id,predecessor,end_seq,manifest,policy,completed,effects_done) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    key,
                    "user",
                    "profile",
                    storage.org_id,
                    len(key),
                    20,
                    "[]",
                    "{}",
                    completed,
                    effects_done,
                ),
            )
    with pytest.raises(TransitionInventoryScopeError):
        storage.transition_inventory_page("cursors")
    windows = {
        item.key[0]: item for item in storage.transition_inventory_page("windows").items
    }
    assert "unfinished_window" in windows["stranded"].reasons
    assert "effects_debt" in windows["debt"].reasons
    assert windows["inert"].reasons == ()
    assert windows["inert"].live_owner
    work = storage.transition_inventory_page("work").items[0]
    assert work.live_owner
    assert "pending_admission" in work.reasons
    assert "unproven_admission" in work.reasons


def test_dependencies_and_preparations_do_not_need_to_be_runnable(storage):
    seed_run(storage, "held", status="failed")
    with storage.commit_scope():
        storage.conn.execute(
            "INSERT INTO _pending_tool_calls(id,org_id,scope_hash,tool_name,dedup_key,status,question_text,expires_at,cache_until) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (
                "question",
                storage.org_id,
                "hash",
                "tool",
                "dedup",
                "pending",
                "NEVER_RETURN_QUESTION",
                "2100-01-01T00:00:00+00:00",
                "2100-01-01T00:00:00+00:00",
            ),
        )
        storage.conn.execute(
            "INSERT INTO _run_tool_dependencies(run_id,pending_tool_call_id) VALUES(?,?)",
            ("held", "question"),
        )
        storage.conn.execute(
            "INSERT INTO _agent_output_preparations(run_id,org_id,preparation) VALUES(?,?,?)",
            (
                "held",
                storage.org_id,
                '{"output_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","secret":"NEVER_RETURN_PREP"}',
            ),
        )
    run = storage.transition_inventory_page("runs").items[0]
    assert run.execution_bearing
    assert "dependency" in run.reasons
    assert "preparation" in run.reasons
    assert storage.transition_inventory_page("dependencies").items[0].reasons == (
        "waiting_dependency",
    )
    assert storage.transition_inventory_page("preparations").items[0].reasons == (
        "saved_preparation",
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"limit": 0},
        {"limit": 501},
        {"limit": True},
        {"limit": 1.5},
        {"after": ("a", "b")},
        {"after": ["a"]},
        {"after": (1,)},
    ],
)
def test_invalid_page_bounds_and_cursor_are_refused(storage, kwargs):
    with pytest.raises(ValueError):
        storage.transition_inventory_page("runs", **kwargs)


def test_missing_metadata_table_is_a_failure_not_an_empty_inventory(storage):
    with storage.commit_scope():
        storage.conn.execute("DROP TABLE _agent_output_preparations")
    with pytest.raises(Exception, match="_agent_output_preparations"):
        storage.transition_inventory_page("runs")


def test_published_terminal_preparation_is_inert_but_pending_finance_remains(storage):
    output_id = "a" * 32
    seed_run(
        storage,
        "published",
        status="agent_completed",
        output={"publication": {"id": output_id}},
    )
    seed_run(
        storage,
        "debt",
        status="agent_completed",
        output={"publication": {"id": "b" * 32}},
    )
    with storage.commit_scope():
        for run_id, receipt_id, ids, state, facts in (
            ("published", "c" * 32, "[]", "accepted", None),
            (
                "debt",
                "d" * 32,
                '["learning"]',
                "pending",
                '{"pipeline":"profile","source":null,"created_at":1}',
            ),
        ):
            storage.conn.execute(
                "INSERT INTO _agent_output_receipts(receipt_id,org_id,run_id,output_id,entity_type,learning_ids,delivery_state,delivery_facts) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    receipt_id,
                    storage.org_id,
                    run_id,
                    output_id if run_id == "published" else "b" * 32,
                    "profile",
                    ids,
                    state,
                    facts,
                ),
            )
        storage.conn.execute("UPDATE _agent_runs SET status='finalized'")
        storage.conn.execute(
            "INSERT INTO _agent_output_preparations(run_id,org_id,preparation) VALUES(?,?,?)",
            ("published", storage.org_id, json.dumps({"output_id": output_id})),
        )
    runs = {
        item.run_id: item for item in storage.transition_inventory_page("runs").items
    }
    assert not runs["published"].execution_bearing
    assert runs["debt"].reasons == ("debt",)
    assert (
        not storage.transition_inventory_page("preparations").items[0].execution_bearing
    )
    assert [
        item.execution_bearing
        for item in storage.transition_inventory_page("receipts").items
    ] == [False, True]


def test_legacy_finalization_receipt_proves_old_saved_output_published(storage):
    seed_run(storage, "legacy", status="finalized", output={"profiles": []})
    seed_run(storage, "unproven", status="finalized", output={"profiles": []})
    with storage.commit_scope():
        storage.conn.execute(
            "INSERT INTO _agent_run_finalization_receipts(run_id,entity_type,learning_ids) VALUES(?,?,?)",
            ("legacy", "profile", "[]"),
        )
    legacy, unproven = storage.transition_inventory_page("runs").items
    assert legacy.execution_bearing is False
    assert unproven.execution_bearing is False
    assert unproven.inert_history
    with storage._stream_sql() as db:
        from reflexio.server.services.storage.storage_base.transition_inventory import (
            _projection,
        )

        table, _keys, scope, columns = _projection(db, "runs", db.now())
        facts = {
            row["run_id"]: row
            for row in db.query(
                f"SELECT {columns} FROM {table} r WHERE {scope}",  # noqa: S608 -- Actual native projection and fixed test scope.
                (storage.org_id,),
            )
        }
    assert facts["legacy"]["legacy_published"]
    assert not facts["unproven"]["legacy_published"]
    assert not facts["unproven"]["published"]


def test_ignored_cursor_never_reports_a_complete_inventory(storage, monkeypatch):
    from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
        SQLiteStreamSQL,
    )

    seed_run(storage, "first")
    first = storage.transition_inventory_page("runs", limit=1)
    original = SQLiteStreamSQL.query

    def ignored_cursor(db, sql, params=()):
        if " AND r.id>?" in sql:
            return original(
                db, sql.replace(" AND r.id>?", ""), (*params[:-2], params[-1])
            )
        return original(db, sql, params)

    monkeypatch.setattr(SQLiteStreamSQL, "query", ignored_cursor)
    with pytest.raises(ValueError, match="did not advance"):
        storage.transition_inventory_page("runs", after=first.next_after, limit=1)


def test_shared_route_cursors_refuse_even_a_page_beyond_all_keys(storage):
    with storage.commit_scope():
        for user, kind in (("a", "playbook"), ("a", "profile"), ("b", "profile")):
            storage.conn.execute(
                "INSERT INTO extraction_cursors(user_id,kind,last_error) VALUES(?,?,?)",
                (user, kind, "held"),
            )
    with pytest.raises(TransitionInventoryScopeError) as error:
        storage.transition_inventory_page(
            "cursors", after=("z", "profile", ""), limit=1
        )
    assert str(error.value) == "Transition inventory source attribution is unavailable"


def test_shared_route_orphan_dependency_never_exposes_identifiers_or_exhaustion(
    storage,
):
    # Synthetic historical/corrupt route: current writes enforce foreign keys.
    storage.conn.execute("PRAGMA foreign_keys=OFF")
    with storage.commit_scope():
        storage.conn.execute(
            "INSERT INTO _run_tool_dependencies(run_id,pending_tool_call_id) VALUES(?,?)",
            ("FOREIGN_RUN", "FOREIGN_QUESTION"),
        )
    storage.conn.execute("PRAGMA foreign_keys=ON")
    with pytest.raises(TransitionInventoryScopeError) as error:
        storage.transition_inventory_page("dependencies", after=("z", "z"), limit=1)
    assert "FOREIGN" not in str(error.value)


def seed_window(storage, key, *, org=None, project=""):
    with storage.commit_scope():
        storage.conn.execute(
            "INSERT INTO extraction_windows(window_id,user_id,kind,org_id,project_id,predecessor,end_seq,manifest,policy) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (key, "FOREIGN_USER", "profile", org, project, len(key), 20, "[]", "{}"),
        )


def test_shared_route_null_org_window_refuses_before_returning_identifiers(storage):
    seed_window(storage, "FOREIGN_WINDOW")
    with pytest.raises(TransitionInventoryScopeError) as error:
        storage.transition_inventory_page("windows", after=("z",), limit=1)
    assert "FOREIGN" not in str(error.value)


def test_project_scoped_metadata_never_returns_foreign_or_unbound_identifiers(
    storage, monkeypatch
):
    from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
        SQLiteStreamSQL,
    )
    from reflexio.server.services.storage.storage_base import transition_inventory

    # Exercise the native project's SQL predicates on disposable SQLite tables.
    # The shared closed-window proof now also binds the native parent's project.
    storage.conn.execute(
        "ALTER TABLE _agent_runs ADD COLUMN project_id TEXT DEFAULT ''"
    )
    monkeypatch.setattr(transition_inventory, "current_project_id", lambda: "requested")
    monkeypatch.setattr(
        SQLiteStreamSQL,
        "transition_project_scope",
        lambda _self, _source: ("r.project_id", "r.project_id=?"),
    )
    monkeypatch.setattr(
        SQLiteStreamSQL,
        "transition_unbound_project",
        lambda _self: "COALESCE(r.project_id,'')=''",
    )
    seed_window(storage, "FOREIGN_NAMED", org=storage.org_id, project="another")
    seed_window(storage, "FOREIGN_ORG", org="another", project="requested")
    assert storage.transition_inventory_page("windows").complete
    seed_window(storage, "LEGACY_UNBOUND", org=storage.org_id)
    with pytest.raises(TransitionInventoryScopeError) as error:
        storage.transition_inventory_page("windows", after=("z",))
    assert "LEGACY" not in str(error.value)


def test_inventory_uses_independent_metadata_owner_and_never_mutates(storage):
    seed_run(storage, "one")
    changes = storage.conn.total_changes
    for source in INVENTORY_SOURCES:
        storage.transition_inventory_page(source)
    assert storage.conn.total_changes == changes
    with storage.commit_scope(), pytest.raises(ValueError, match="independent writer"):
        storage.transition_inventory_page("runs")


def test_shared_route_foreign_cursor_cannot_prove_work_inert(storage):
    with storage.commit_scope():
        storage.conn.execute(
            "INSERT INTO learning_work(org_id,user_id,highwater,pending_since) VALUES(?,?,?,?)",
            (storage.org_id, "same-user", 20, 0),
        )
        storage.conn.execute(
            "INSERT INTO learning_work(org_id,user_id,highwater,pending_since) VALUES(?,?,?,?)",
            ("foreign-org", "same-user", 20, 0),
        )
        storage.conn.execute(
            "INSERT INTO extraction_cursors(user_id,kind,completed_seq) VALUES(?,?,?)",
            ("same-user", "profile", 20),
        )
    work = storage.transition_inventory_page("work").items[0]
    assert work.execution_bearing is None
    assert work.reasons == ("unproven_admission",)


@pytest.mark.parametrize("after", [None, ("zzzz",)])
@pytest.mark.parametrize("status", ["running", "finalized"])
def test_null_primary_key_refuses_even_after_keyset_boundary(storage, after, status):
    # SQLite TEXT PRIMARY KEY does not imply NOT NULL in ordinary rowid tables.
    seed_run(storage, None, status=status)
    with pytest.raises(TransitionInventoryScopeError, match="inventory|Inventory"):
        storage.transition_inventory_page("runs", after=after)


@pytest.mark.parametrize("after", [None, ("zzzz", "zzzz")])
def test_null_nonleading_inventory_key_cannot_be_paginated_away(
    storage, monkeypatch, after
):
    from reflexio.server.services.storage.storage_base.transition_inventory import (
        read_transition_inventory_page_db,
    )

    seed_run(storage, "valid", status="finalized")
    with storage.commit_scope():
        storage.conn.execute(
            "CREATE TEMP VIEW null_inventory_dependencies AS SELECT 'valid' AS run_id,NULL AS pending_tool_call_id,1 AS resolved_at,1 AS consumed_at"
        )
    with storage._independent_writer_metadata_sql() as db:
        original = db.table
        monkeypatch.setattr(
            db,
            "table",
            lambda name: (
                "null_inventory_dependencies"
                if name == "_run_tool_dependencies"
                else original(name)
            ),
        )
        with pytest.raises(TransitionInventoryScopeError):
            read_transition_inventory_page_db(
                db, "dependencies", org_id=storage.org_id, project_id="", after=after
            )


@pytest.mark.parametrize(
    "state,inert",
    [
        ("finalized", True),
        ("failed", False),
        ("cancelled", True),
        ("agent_completed", False),
        ("finalizing", False),
    ],
)
def test_generic_terminal_saved_output_is_history_only_when_native_owners_cannot_replay(
    storage, state, inert
):
    seed_run(
        storage,
        "standalone",
        status=state,
        output={"output": "PRIVATE_RETAINED_OUTPUT"},
    )
    item = storage.transition_inventory_page("runs").items[0]
    assert item.inert_history is inert
    assert item.saved_output
    assert (
        item.execution_bearing is False
        if inert
        else item.execution_bearing is not False
    )
    assert "PRIVATE_RETAINED_OUTPUT" not in json.dumps(asdict(item))


@pytest.mark.parametrize("unknown", [None, True])
@pytest.mark.parametrize(
    "field",
    ["live_owner", "dependency", "preparation", "debt", "window", "terminal_history"],
)
def test_finalized_history_python_and_sql_both_refuse_unknown_or_active_facts(
    storage, field, unknown
):
    from reflexio.server.services.storage.storage_base.transition_sources import (
        RUN_EXECUTION_FACTS,
        generic_closed_run_inert,
        generic_closed_run_inert_sql,
    )

    facts: dict[str, bool | None] = dict.fromkeys(RUN_EXECUTION_FACTS, False)
    facts["terminal_history"] = True
    facts[field] = (
        unknown if field != "terminal_history" else (None if unknown is None else False)
    )
    columns = ",".join(f"? AS {key}" for key in facts)
    with storage._stream_sql() as db:
        observed = db.query(
            f"SELECT {generic_closed_run_inert_sql()} AS inert FROM (SELECT {columns}) facts",  # noqa: S608 -- Fixed shared source predicate and test-only column names.
            tuple(facts.values()),
        )[0]["inert"]
    assert not generic_closed_run_inert(facts)
    assert not observed


def test_finalized_generic_history_does_not_ignore_a_foreign_user_live_token(storage):
    seed_run(storage, "standalone", status="finalized", output={"output": "retained"})
    with storage.commit_scope():
        storage.conn.execute("UPDATE _agent_runs SET user_work_token='parent-token'")
        storage.conn.execute(
            "INSERT INTO learning_work(org_id,user_id,lease_token,lease_until) VALUES(?,'other','parent-token',9999999999)",
            (storage.org_id,),
        )
    item = storage.transition_inventory_page("runs").items[0]
    assert not item.inert_history
    assert "live_owner" in item.reasons


def test_run_window_debt_lookup_uses_exact_window_primary_key(storage):
    from reflexio.server.services.storage.storage_base.transition_inventory import (
        _projection,
    )

    with storage._stream_sql() as db:
        columns = _projection(db, "runs", db.now())[3]
        prefix = f"EXISTS(SELECT 1 FROM {db.table('extraction_windows')} w WHERE "  # noqa: S608 -- Fixed native predicates and registered fixture tables.
        debt = prefix + columns.rsplit(prefix, 1)[1].removesuffix(" AS window")
        plan = db.query(
            f"EXPLAIN QUERY PLAN SELECT {debt} FROM {db.table('_agent_runs')} r"  # noqa: S608 -- Fixed native predicates and registered fixture tables.
        )
    details = [row["detail"] for row in plan]
    assert any(
        "SEARCH w USING INDEX" in detail and "window_id=?" in detail
        for detail in details
    ), details
    assert not any(detail == "SCAN w" for detail in details), details


def test_indexed_run_window_debt_preserves_exact_scope_and_terminal_semantics(storage):
    from reflexio.server.services.storage.storage_base.transition_inventory import (
        _projection,
    )

    cases = [
        ("window:matching", "matching", storage.org_id, 0, 0, True),
        ("window:null-org", "null-org", None, 1, 0, True),
        ("window:foreign", "foreign", "foreign", 0, 0, False),
        ("window:done", "done", storage.org_id, 1, 1, False),
        ("generic", "generic", storage.org_id, 0, 0, False),
        ("window:suffix:extra", "suffix", storage.org_id, 0, 0, False),
        ("window:", "", storage.org_id, 0, 0, True),
        ("prefixwindow:z", "z", storage.org_id, 0, 0, False),
    ]
    with storage.commit_scope(), storage._stream_sql() as db:
        for index, (run, window, org, completed, effects, _expected) in enumerate(
            cases
        ):
            db.query(
                "INSERT INTO _agent_runs(id,org_id,extractor_kind,user_id,request_id,status,generation_request_snapshot) VALUES(?,?,'profile',?,'request','finalized','{}')",
                (run, storage.org_id, f"user-{index}"),
            )
            db.query(
                "INSERT INTO extraction_windows(window_id,user_id,kind,org_id,project_id,predecessor,end_seq,manifest,policy,completed,effects_done) VALUES(?,?,'profile',?,'',?,?,'[]','{}',?,?)",
                (
                    window,
                    f"user-{index}",
                    org,
                    index * 3,
                    index * 3 + 1,
                    completed,
                    effects,
                ),
            )
        prefix = f"EXISTS(SELECT 1 FROM {db.table('extraction_windows')} w WHERE "  # noqa: S608 -- Fixed native predicates and registered fixture tables.
        debt = prefix + _projection(db, "runs", db.now())[3].rsplit(prefix, 1)[
            1
        ].removesuffix(" AS window")
        old = "EXISTS(SELECT 1 FROM extraction_windows w WHERE r.id='window:'||w.window_id AND (w.org_id=r.org_id OR w.org_id IS NULL) AND (w.completed=0 OR w.effects_done=0))"
        rows = db.query(
            f"SELECT r.id,{debt} AS indexed,{old} AS previous FROM _agent_runs r"  # noqa: S608 -- Fixed native predicates and registered fixture tables.
        )
    expected = {run: value for run, _window, _org, _completed, _effects, value in cases}
    assert len(rows) == len(cases)
    assert all(
        bool(row["indexed"]) is expected[row["id"]]
        and row["indexed"] == row["previous"]
        for row in rows
    )


@pytest.mark.parametrize("overflow", [None, {}, {"version": 1}, "unknown"])
def test_cancelled_overflow_is_not_generic_inert_history(storage, overflow):
    seed_run(
        storage,
        "cancelled-generic",
        status="cancelled",
        snapshot={"overflow": overflow},
        output={"retained": "synthetic"},
    )
    item = storage.transition_inventory_page("runs").items[0]
    assert not item.inert_history
    assert item.execution_bearing is not False


@pytest.mark.parametrize(
    "user,token,work_org,work_user,work_token,until,expected",
    [
        ("subject", None, "target", "subject", "other", 2, True),
        ("subject", "shared", "target", "another", "shared", 2, True),
        ("subject", "shared", "target", "subject", "shared", 0, False),
        ("subject", "shared", "foreign", "subject", "shared", 2, False),
        ("subject", None, "target", "another", None, 2, False),
        (None, "shared", "target", "another", "shared", 2, True),
        (None, None, "target", "another", None, 2, False),
        ("subject", "", "target", "another", "", 2, True),
    ],
)
def test_live_owner_sets_preserve_exists_null_and_foreign_identity_semantics(
    storage, user, token, work_org, work_user, work_token, until, expected
):
    from reflexio.server.services.storage.storage_base.transition_sources import (
        live_work_sql,
    )

    with storage.commit_scope(), storage._stream_sql() as db:
        db.query(
            "INSERT INTO learning_work(org_id,user_id,lease_token,lease_until) VALUES(?,?,?,?)",
            (work_org, work_user, work_token, until),
        )
        indexed = live_work_sql(db, alias="r", org_sql="r.org_id", now=1)
        old = "EXISTS(SELECT 1 FROM learning_work u WHERE u.org_id=r.org_id AND (u.user_id=r.user_id OR u.lease_token=r.user_work_token) AND u.lease_until>1)"
        row = db.query(
            f"SELECT {indexed} AS indexed,{old} AS previous FROM (SELECT CAST(? AS TEXT) AS org_id,CAST(? AS TEXT) AS user_id,CAST(? AS TEXT) AS user_work_token) r",  # noqa: S608 -- Fixed native predicate and typed synthetic fixture identity.
            ("target", user, token),
        )[0]
    assert bool(row["indexed"]) is expected
    assert row["indexed"] == row["previous"]


def test_private_closure_projection_keeps_classification_and_public_witnesses(storage):
    from reflexio.server.services.storage.storage_base.transition_inventory import (
        _item,
        _projection,
    )

    seed_run(
        storage,
        "metadata",
        status="finalized",
        snapshot={
            "output_schema_name": "metadata_schema",
            "overflow": {
                "version": 1,
                "checkpoint": {
                    "stage": "retired",
                    "extraction_instruction_contract": "a" * 64,
                    "extraction_policy": "b" * 64,
                },
            },
        },
        output={"retained": "synthetic"},
    )
    with storage.commit_scope():
        storage.conn.execute(
            "UPDATE _agent_runs SET extractor_config_hash=?", ("c" * 64,)
        )
    public = storage.transition_inventory_page("runs").items[0]
    with storage._stream_sql() as db:
        columns = _projection(db, "runs", db.now(), include_witnesses=False)[3]
        row = db.query(f"SELECT {columns} FROM _agent_runs r")[0]  # noqa: S608 -- Fixed private projection on the fixture table.
    private = _item("runs", public.key, row)
    assert (
        private.reasons,
        private.execution_bearing,
        private.live_owner,
        private.inert_history,
        private.overflow,
    ) == (
        public.reasons,
        public.execution_bearing,
        public.live_owner,
        public.inert_history,
        public.overflow,
    )
    assert public.overflow and private.overflow
    assert public.schema_name == "metadata_schema"
    assert public.instruction_digest == "a" * 64
    assert public.policy_digest == "b" * 64
    assert public.configuration_digest == "c" * 64
    assert (
        private.schema_name
        is private.instruction_digest
        is private.policy_digest
        is private.configuration_digest
        is private.revision
        is None
    )
    assert " AS schema_name" not in columns
