"""Migration test: requests.session_id becomes required and non-empty.

Covers the in-place table rebuild in
``SQLiteStorageBase._migrate_request_session_id_required`` — the riskiest part
of the session-id-required change, since it drops and recreates ``requests``.
"""

import sqlite3

import pytest

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

# The pre-migration ``requests`` schema: session_id was nullable.
_LEGACY_REQUESTS_DDL = """
CREATE TABLE requests (
    request_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    agent_version TEXT NOT NULL DEFAULT '',
    session_id TEXT,
    metadata TEXT NOT NULL DEFAULT '{}'
);
"""


def _seed_legacy_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(_LEGACY_REQUESTS_DDL)
    conn.executemany(
        "INSERT INTO requests (request_id, user_id, created_at, source, session_id) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            ("r-null", "u1", "2026-01-01T00:00:00+00:00", "web", None),
            ("r-blank", "u1", "2026-01-01T00:00:01+00:00", "web", "   "),
            ("r-valid", "u1", "2026-01-01T00:00:02+00:00", "web", "s-valid"),
        ],
    )
    conn.commit()
    conn.close()


def _session_ids(db_path: str) -> dict[str, str]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return {
            row["request_id"]: row["session_id"]
            for row in conn.execute("SELECT request_id, session_id FROM requests")
        }
    finally:
        conn.close()


def test_migration_backfills_legacy_blank_sessions(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    _seed_legacy_db(db_path)

    SQLiteStorage(org_id="0", db_path=db_path)  # construction triggers migrate()

    sessions = _session_ids(db_path)
    assert len(sessions) == 3  # no rows dropped during the rebuild
    assert sessions["r-valid"] == "s-valid"  # pre-existing value untouched
    # NULL/blank rows are backfilled per-row (NOT grouped into a shared session).
    assert sessions["r-null"].startswith("legacy-")
    assert sessions["r-blank"].startswith("legacy-")
    assert sessions["r-null"] != sessions["r-blank"]


def test_migration_enforces_not_null_and_non_empty(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    _seed_legacy_db(db_path)
    SQLiteStorage(org_id="0", db_path=db_path)

    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO requests (request_id, user_id, created_at, session_id) "
                "VALUES ('x', 'u', '2026-01-01T00:00:00+00:00', NULL)"
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO requests (request_id, user_id, created_at, session_id) "
                "VALUES ('y', 'u', '2026-01-01T00:00:00+00:00', '   ')"
            )
    finally:
        conn.close()


def test_migration_is_idempotent(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    _seed_legacy_db(db_path)

    SQLiteStorage(org_id="0", db_path=db_path)
    first = _session_ids(db_path)

    # Re-opening detects the required schema with no blanks and no-ops, so the
    # backfilled legacy ids must be preserved (not regenerated).
    SQLiteStorage(org_id="0", db_path=db_path)
    second = _session_ids(db_path)
    assert first == second


def test_migration_noop_on_fresh_db(tmp_path):
    """A fresh DB is created with the required schema; no rebuild needed."""
    db_path = str(tmp_path / "fresh.db")
    SQLiteStorage(org_id="0", db_path=db_path)

    conn = sqlite3.connect(db_path)
    try:
        (sql,) = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='requests'"
        ).fetchone()
    finally:
        conn.close()
    assert "CHECK (trim(session_id) != '')" in sql
