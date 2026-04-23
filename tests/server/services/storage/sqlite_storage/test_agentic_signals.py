"""Task 2.4: agentic signal columns persist through profiles + user_playbooks."""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def _get_columns(db_path: str, table: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {
            row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
    finally:
        conn.close()


def test_fresh_schema_has_agentic_signal_columns(tmp_path):
    """Fresh SQLiteStorage DBs include source_span/notes/reader_angle on both tables."""
    db_path = str(tmp_path / "fresh.db")
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        SQLiteStorage(org_id="test_fresh", db_path=db_path)
    assert {"source_span", "notes", "reader_angle"} <= _get_columns(db_path, "profiles")
    assert {"source_span", "notes", "reader_angle"} <= _get_columns(
        db_path, "user_playbooks"
    )


def test_migration_adds_columns_to_legacy_db(tmp_path):
    """A pre-existing DB without the new columns gets them added at startup.

    The legacy schema simulates a DB created just before the agentic signal
    columns were introduced — all existing columns are present, but
    source_span/notes/reader_angle are absent.
    """
    db_path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(db_path)
    # Profiles table without source_span/notes/reader_angle
    conn.execute(
        "CREATE TABLE profiles ("
        "profile_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, "
        "content TEXT NOT NULL DEFAULT '', "
        "last_modified_timestamp INTEGER NOT NULL, "
        "generated_from_request_id TEXT NOT NULL DEFAULT '', "
        "profile_time_to_live TEXT NOT NULL DEFAULT 'infinity', "
        "expiration_timestamp INTEGER NOT NULL DEFAULT 4102444800, "
        "custom_features TEXT, embedding TEXT, "
        "source TEXT DEFAULT '', status TEXT, extractor_names TEXT, "
        "expanded_terms TEXT, "
        "created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')))"
    )
    # user_playbooks table without source_span/notes/reader_angle
    conn.execute(
        "CREATE TABLE user_playbooks ("
        "user_playbook_id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "user_id TEXT, playbook_name TEXT NOT NULL DEFAULT '', "
        "created_at TEXT NOT NULL, request_id TEXT NOT NULL, "
        "agent_version TEXT NOT NULL DEFAULT '', "
        "content TEXT NOT NULL DEFAULT '', trigger TEXT, rationale TEXT, "
        "blocking_issue TEXT, source_interaction_ids TEXT, "
        "status TEXT, source TEXT, embedding TEXT, expanded_terms TEXT)"
    )
    conn.commit()
    conn.close()

    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        SQLiteStorage(org_id="test_legacy", db_path=db_path)

    assert {"source_span", "notes", "reader_angle"} <= _get_columns(db_path, "profiles")
    assert {"source_span", "notes", "reader_angle"} <= _get_columns(
        db_path, "user_playbooks"
    )
