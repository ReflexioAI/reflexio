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
