"""Database collation owns keyset order; Python only rejects repeated keys."""

import pytest

from reflexio.server.services.storage.storage_base import transition_inventory as owner

pytestmark = pytest.mark.integration


@pytest.fixture(params=["work", "cursors"])
def collated_inventory(storage, monkeypatch, request):
    # Test-only views use an actual non-binary DB collation without changing
    # production schema, query predicates, or any stored cursor representation.
    with storage.commit_scope():
        for user in ("a", "B", "c"):
            storage.conn.execute(
                "INSERT INTO learning_work(org_id,user_id,highwater) VALUES(?,?,1)",
                (storage.org_id, user),
            )
            storage.conn.execute(
                "INSERT INTO extraction_cursors(user_id,kind,attempts) VALUES(?,'profile',1)",
                (user,),
            )
        storage.conn.execute(
            "CREATE TEMP VIEW collated_inventory_work AS SELECT org_id,"
            "user_id COLLATE NOCASE AS user_id,highwater,pending_since,revision,"
            "last_turn,due_at,lease_token,lease_owner,lease_until,position_json FROM learning_work"
        )
        storage.conn.execute(
            "CREATE TEMP VIEW collated_inventory_cursors AS SELECT "
            "user_id COLLATE NOCASE AS user_id,kind,completed_seq,started,"
            "project_id,window_id,attempts,retry_at,last_error,last_turn FROM extraction_cursors"
        )
    with storage._independent_writer_metadata_sql() as db:
        original_table = db.table
        views = {
            "learning_work": "collated_inventory_work",
            "extraction_cursors": "collated_inventory_cursors",
        }
        monkeypatch.setattr(
            db, "table", lambda name: views.get(name, original_table(name))
        )
        attribution = owner.TransitionInventoryAttribution(
            storage.org_id, "", storage._durable_work_route()
        )

        def read(*, after=None, limit=100):
            if request.param == "cursors":
                return owner._attributed_cursors(db, attribution, after, limit)
            return owner.read_transition_inventory_page_db(
                db,
                "work",
                org_id=storage.org_id,
                project_id="",
                after=after,
                limit=limit,
            )

        yield db, read, request.param


def test_database_order_can_differ_from_python_within_and_across_pages(
    collated_inventory,
):
    _, read, source = collated_inventory
    page = read()
    assert [item.key[0] for item in page.items] == ["a", "B", "c"]
    assert page.items[1].key < page.items[0].key  # The old Python guard rejects this.
    assert not page.complete
    assert all(item.execution_bearing is not False for item in page.items)
    first = read(limit=1)
    after = first.next_after
    second = read(after=after, limit=1)
    assert second.items[0].key[0] == "B"
    assert second.next_after < after
    assert not second.complete
    third = read(after=second.next_after, limit=1)
    assert third.items[0].key[0] == "c"
    assert third.next_after == (("c", "profile", "") if source == "cursors" else ("c",))
    assert not third.complete
    empty = read(after=third.next_after, limit=1)
    assert empty.complete and empty.next_after is None and empty.items == ()


@pytest.mark.parametrize("repetition", ["adjacent", "nonadjacent", "after"])
def test_repeated_keys_still_refuse_instead_of_false_exhaustion(
    collated_inventory, monkeypatch, repetition
):
    db, read, source = collated_inventory
    original_query = db.query

    def repeated_query(sql, params=()):
        rows = original_query(sql, params)
        assert all(row["key0"] is not None for row in rows)
        if repetition == "after":
            return [{**rows[0], "key0": "a"}]
        if repetition == "adjacent":
            return [rows[0], rows[0]]
        return [*rows, rows[0]]

    monkeypatch.setattr(db, "query", repeated_query)
    after = (
        (("a", "profile", "") if source == "cursors" else ("a",))
        if repetition == "after"
        else None
    )
    with pytest.raises(ValueError, match="did not advance"):
        read(after=after)
