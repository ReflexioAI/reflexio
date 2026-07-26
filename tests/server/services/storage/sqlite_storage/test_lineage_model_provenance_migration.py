from __future__ import annotations

import sqlite3

import pytest

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration

_MODEL_COLUMNS = {
    "model_name",
    "provider",
}


def test_fresh_database_has_model_provenance_columns(tmp_path) -> None:
    storage = SQLiteStorage(org_id="org", db_path=str(tmp_path / "fresh.db"))
    storage.migrate()
    columns = {
        row["name"] for row in storage.conn.execute("PRAGMA table_info(lineage_event)")
    }
    assert columns >= _MODEL_COLUMNS
    assert "requested_model" not in columns


def test_legacy_database_upgrade_adds_nullable_columns_without_backfill(
    tmp_path,
) -> None:
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE lineage_event (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            org_id TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            op TEXT NOT NULL,
            prov_relation TEXT NOT NULL DEFAULT '',
            source_ids TEXT NOT NULL DEFAULT '[]',
            actor TEXT NOT NULL DEFAULT '',
            request_id TEXT NOT NULL DEFAULT '',
            reason TEXT NOT NULL DEFAULT '',
            created_at INTEGER NOT NULL,
            UNIQUE (org_id, entity_type, entity_id, op, request_id)
        );
        INSERT INTO lineage_event (
            org_id, entity_type, entity_id, op, created_at
        ) VALUES ('org', 'profile', 'legacy-profile', 'create', 1);
    """)
    conn.commit()
    conn.close()

    storage = SQLiteStorage(org_id="org", db_path=str(db_path))
    storage.migrate()
    columns = {
        row["name"] for row in storage.conn.execute("PRAGMA table_info(lineage_event)")
    }
    assert columns >= _MODEL_COLUMNS

    event = storage.get_lineage_events(entity_id="legacy-profile")[0]
    assert event.model_name is None
    assert event.provider is None
