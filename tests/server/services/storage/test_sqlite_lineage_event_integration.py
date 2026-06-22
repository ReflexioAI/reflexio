import pytest

from reflexio.models.api_schema.domain.entities import LineageEvent
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def _evt(**kw):
    base = {
        "org_id": "org-42",
        "entity_type": "user_playbook",
        "entity_id": "UP-1",
        "op": "merge",
        "prov_relation": "wasDerivedFrom",
        "source_ids": ["UP-1"],
        "actor": "consolidator",
        "request_id": "req-7",
        "reason": "dup",
    }
    base.update(kw)
    return LineageEvent(**base)


def test_append_then_get(tmp_path):
    s = SQLiteStorage(org_id="org-42", db_path=str(tmp_path / "t.db"))
    s.migrate()
    eid = s.append_lineage_event(_evt())
    assert eid > 0
    rows = s.get_lineage_events(entity_type="user_playbook", entity_id="UP-1")
    assert len(rows) == 1 and rows[0].op == "merge" and rows[0].created_at > 0


def test_append_is_idempotent_on_unique_key(tmp_path):
    s = SQLiteStorage(org_id="org-42", db_path=str(tmp_path / "t.db"))
    s.migrate()
    first = s.append_lineage_event(_evt())
    again = s.append_lineage_event(_evt())  # same (entity_id, op, request_id)
    assert first == again
    assert len(s.get_lineage_events(entity_id="UP-1")) == 1


# ---------------------------------------------------------------------------
# B3 idempotency-key hazard: empty request_id collapses distinct events
# ---------------------------------------------------------------------------


def test_empty_request_id_collapses_second_event(tmp_path):
    """Two events with the same (org, entity_type, entity_id, op) and request_id=''
    collapse to one row via ON CONFLICT DO NOTHING.

    This is the storage-level mechanism that non-empty request_ids protect against.
    When request_id is '' (empty string), the unique key
    (org_id, entity_type, entity_id, op, request_id) treats all events for a
    given entity+op as the same row — the second INSERT is silently dropped.

    This test proves the hazard honestly: it is NOT that two optimizer passes on
    the same incumbent collide (they cannot — each supersede_record uses a FRESH
    successor_id as entity_id), but that any two events for the SAME entity_id
    and op with request_id='' would collapse.  Non-empty, distinct request_ids
    prevent this for run-correlation events.
    """
    s = SQLiteStorage(org_id="org-42", db_path=str(tmp_path / "t.db"))
    s.migrate()

    # Two events for the same entity+op but both with request_id="" → only one persists
    e1 = _evt(
        entity_id="UP-hazard", op="revise", request_id="", prov_relation="wasRevisionOf"
    )
    e2 = _evt(
        entity_id="UP-hazard", op="revise", request_id="", prov_relation="wasRevisionOf"
    )
    s.append_lineage_event(e1)
    s.append_lineage_event(e2)

    rows = s.get_lineage_events(entity_type="user_playbook", entity_id="UP-hazard")
    assert len(rows) == 1, (
        "both events share request_id='' → ON CONFLICT DO NOTHING silently drops the second"
    )


def test_distinct_nonempty_request_ids_both_persist(tmp_path):
    """Two events for the same (entity_type, entity_id, op) with DISTINCT non-empty
    request_ids both persist — the unique key discriminates on request_id.

    This is the correct behaviour for run-correlation: each operation run stamps
    its own non-empty id so independent events are recorded independently.
    """
    s = SQLiteStorage(org_id="org-42", db_path=str(tmp_path / "t.db"))
    s.migrate()

    e1 = _evt(
        entity_id="UP-safe",
        op="revise",
        request_id="run-aaa",
        prov_relation="wasRevisionOf",
    )
    e2 = _evt(
        entity_id="UP-safe",
        op="revise",
        request_id="run-bbb",
        prov_relation="wasRevisionOf",
    )
    s.append_lineage_event(e1)
    s.append_lineage_event(e2)

    rows = s.get_lineage_events(entity_type="user_playbook", entity_id="UP-safe")
    assert len(rows) == 2, (
        "distinct non-empty request_ids must produce two separate rows"
    )
    stored_request_ids = {r.request_id for r in rows}
    assert stored_request_ids == {"run-aaa", "run-bbb"}
