"""SQLite storage tests for the per-request metadata field added in F2."""

import pytest

from reflexio.models.api_schema.domain.entities import Request
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage


@pytest.fixture
def storage(tmp_path):
    db_path = tmp_path / "test.db"
    return SQLiteStorage(org_id="0", db_path=str(db_path))


def test_sqlite_persists_request_metadata(storage):
    r = Request(
        request_id="r1",
        user_id="u1",
        session_id="s1",
        metadata={"reflexio_retrieval_enabled": True},
    )
    storage.add_request(r)
    got = storage.get_request("r1")
    assert got is not None
    assert got.metadata == {"reflexio_retrieval_enabled": True}


def test_sqlite_default_empty_metadata(storage):
    r = Request(request_id="r2", user_id="u1", session_id="s1")
    storage.add_request(r)
    got = storage.get_request("r2")
    assert got is not None
    assert got.metadata == {}


def test_sqlite_metadata_accepts_nested_values(storage):
    r = Request(
        request_id="r3",
        user_id="u1",
        session_id="test_session",
        metadata={"reflexio_retrieval_enabled": False, "tags": ["a", "b"]},
    )
    storage.add_request(r)
    got = storage.get_request("r3")
    assert got is not None
    assert got.metadata["tags"] == ["a", "b"]


def test_sqlite_get_requests_by_session_carries_metadata(storage):
    """End-to-end check that the bulk-fetch read path round-trips metadata."""
    r1 = Request(
        request_id="r4",
        user_id="u1",
        session_id="s2",
        metadata={"reflexio_retrieval_enabled": True},
    )
    r2 = Request(
        request_id="r5",
        user_id="u1",
        session_id="s2",
        metadata={"reflexio_retrieval_enabled": True},
    )
    storage.add_request(r1)
    storage.add_request(r2)
    rows = storage.get_requests_by_session("u1", "s2")
    assert {r.request_id for r in rows} == {"r4", "r5"}
    for r in rows:
        assert r.metadata == {"reflexio_retrieval_enabled": True}


def test_sqlite_migration_adds_metadata_column_to_existing_db(tmp_path):
    """Simulate a pre-F2 DB (no metadata column), then run migration via
    SQLiteStorage init and confirm:
      - The metadata column now exists.
      - A row inserted under the old schema reads back as {} (not None, not error).
    """
    import sqlite3

    db_path = tmp_path / "pre_f2.db"
    # Create the pre-F2 requests table (without the metadata column).
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE requests (
            request_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT '',
            agent_version TEXT NOT NULL DEFAULT '',
            session_id TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO requests (request_id, user_id, created_at, source, "
        "agent_version, session_id) VALUES (?,?,?,?,?,?)",
        ("legacy-r1", "u1", "2020-01-01T00:00:00", "", "", "legacy-s1"),
    )
    conn.commit()
    conn.close()

    # SQLiteStorage init must run the migration. It should ALTER the table
    # to add the metadata column and NOT crash on the pre-existing row.
    storage = SQLiteStorage(org_id="0", db_path=str(db_path))

    # Confirm the column exists now.
    cols = [
        r[1] for r in storage.conn.execute("PRAGMA table_info(requests)").fetchall()
    ]
    assert "metadata" in cols

    # Confirm the pre-existing legacy row reads back with empty metadata.
    got = storage.get_request("legacy-r1")
    assert got is not None
    assert got.metadata == {}
