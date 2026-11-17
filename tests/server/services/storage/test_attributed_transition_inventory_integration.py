"""Positive cursor history requires real source and completion metadata."""

import json

import pytest

from reflexio.server.services.storage.storage_base.transition_inventory import (
    TransitionInventoryAttribution,
    TransitionInventoryScopeError,
)

pytestmark = pytest.mark.integration


def attribution(storage):
    # Internal synthetic owner binding; the Enterprise owner separately verifies
    # the real complete roster. No public JSON verification interface exists.
    return TransitionInventoryAttribution(
        storage.org_id, "", storage._durable_work_route()
    )


def seed_completed(storage):
    with storage.commit_scope():
        storage.conn.execute(
            "INSERT INTO requests(request_id,user_id,created_at,session_id,learning_admission) VALUES(?,?,?,?,?)",
            (
                "request",
                "user",
                "2026-01-01",
                "session",
                json.dumps(
                    {
                        "profile": {"eligible": True},
                        "playbook": {"eligible": False},
                        "min_seq": 1,
                        "max_seq": 1,
                        "admitted_at": 1,
                    }
                ),
            ),
        )
        storage.conn.execute(
            "INSERT INTO interactions(interaction_id,user_id,request_id,created_at,ingestion_seq) VALUES(1,'user','request','2026-01-01',1)"
        )
        storage.conn.execute(
            "INSERT INTO extraction_cursors(user_id,kind,completed_seq,started) VALUES('user','profile',1,1)"
        )
        storage.conn.execute(
            "INSERT INTO extraction_windows(window_id,user_id,kind,org_id,predecessor,end_seq,manifest,policy,completed,effects_done,completed_at) VALUES('window','user','profile',?,0,1,'[]','{}',1,1,2)",
            (storage.org_id,),
        )


def test_retained_cursor_requires_attribution_and_exact_completed_source(storage):
    seed_completed(storage)
    with pytest.raises(TransitionInventoryScopeError):
        storage.transition_inventory_page("cursors")
    page = storage.transition_inventory_page(
        "cursors", attribution=attribution(storage), limit=1
    )
    assert not page.complete
    assert page.items[0].execution_bearing is False
    assert page.items[0].reasons == ()
    assert page.next_after == ("user", "profile", "")
    assert storage.transition_inventory_page(
        "cursors", attribution=attribution(storage), after=page.next_after
    ).complete


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("DELETE FROM extraction_windows", "cursor_completion_unproven"),
        ("UPDATE extraction_windows SET effects_done=0", "cursor_completion_unproven"),
        ("UPDATE extraction_windows SET end_seq=2", "cursor_completion_unproven"),
        ("UPDATE extraction_windows SET user_id='other'", "cursor_completion_unproven"),
        ("UPDATE extraction_windows SET kind='playbook'", "cursor_completion_unproven"),
        ("UPDATE requests SET learning_admission=NULL", "cursor_source_unproven"),
        ("DELETE FROM interactions", "cursor_source_unproven"),
        ("UPDATE extraction_cursors SET last_error='held'", "held_cursor"),
        ("UPDATE extraction_cursors SET attempts=1", "unfinished_cursor"),
    ],
)
def test_missing_or_changed_completion_never_becomes_inert(storage, mutation, reason):
    seed_completed(storage)
    with storage.commit_scope():
        storage.conn.execute(mutation)
    item = storage.transition_inventory_page(
        "cursors", attribution=attribution(storage)
    ).items[0]
    assert item.execution_bearing is not False
    assert reason in item.reasons


def test_new_admitted_source_after_cursor_is_unfinished(storage):
    seed_completed(storage)
    with storage.commit_scope():
        storage.conn.execute(
            "UPDATE requests SET learning_admission=?",
            (
                json.dumps(
                    {
                        "profile": {"eligible": True},
                        "min_seq": 1,
                        "max_seq": 2,
                        "admitted_at": 1,
                    }
                ),
            ),
        )
    item = storage.transition_inventory_page(
        "cursors", attribution=attribution(storage)
    ).items[0]
    assert item.execution_bearing is not False
    assert "uncovered_cursor_source" in item.reasons


def test_foreign_window_refuses_even_beyond_last_cursor(storage):
    seed_completed(storage)
    with storage.commit_scope():
        storage.conn.execute(
            "INSERT INTO extraction_windows(window_id,user_id,kind,org_id,predecessor,end_seq,manifest,policy) "
            "VALUES('foreign','user','profile','foreign',2,3,'[]','{}')"
        )
    with pytest.raises(TransitionInventoryScopeError):
        storage.transition_inventory_page(
            "cursors", attribution=attribution(storage), after=("z", "z", "z")
        )


def test_wrong_writer_binding_is_refused(storage):
    seed_completed(storage)
    with pytest.raises(TransitionInventoryScopeError, match="binding"):
        storage.transition_inventory_page(
            "cursors",
            attribution=TransitionInventoryAttribution(storage.org_id, "", object()),
        )


@pytest.mark.parametrize("column", ["user_id", "kind", "project_id"])
@pytest.mark.parametrize("after", [None, ("zzzz", "zzzz", "zzzz")])
def test_null_attributed_cursor_key_refuses_before_keyset_filter(
    storage, monkeypatch, column, after
):
    from reflexio.server.services.storage.storage_base.transition_inventory import (
        _attributed_cursors,
    )

    values = {"user_id": "'u'", "kind": "'profile'", "project_id": "''"}
    values[column] = "NULL"
    with storage.commit_scope():
        storage.conn.execute(
            "CREATE TEMP VIEW null_attributed_cursors AS SELECT "
            + ",".join(f"{value} AS {name}" for name, value in values.items())  # noqa: S608 -- Fixed negative-test column/constant choices.
            + ",NULL AS window_id,0 AS started,0 AS attempts,0 AS retry_at,NULL AS last_error,0 AS completed_seq"
        )
    with storage._independent_writer_metadata_sql() as db:
        original = db.table
        monkeypatch.setattr(
            db,
            "table",
            lambda name: (
                "null_attributed_cursors"
                if name == "extraction_cursors"
                else original(name)
            ),
        )
        with pytest.raises(TransitionInventoryScopeError):
            _attributed_cursors(db, attribution(storage), after, 100)
