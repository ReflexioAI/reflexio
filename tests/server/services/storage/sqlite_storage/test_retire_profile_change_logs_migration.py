"""Migration test: ``profile_change_logs`` is retired via a reversible RENAME.

Covers ``SQLiteStorageBase._migrate_retire_profile_change_logs`` plus the
deletion of the table's ``CREATE TABLE`` from ``_DDL`` (Lineage B3 Task 8). The
two invariants under test:

  (a) an existing legacy ``profile_change_logs`` table is renamed to
      ``profile_change_logs_retired_20260623`` (data preserved, recoverable);
  (b) the table is NOT recreated by a second ``migrate()`` — i.e. the headline
      trap (``executescript(_DDL)`` resurrecting an empty table) does not bite.
"""

import sqlite3

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

_RETIRED_NAME = "profile_change_logs_retired_20260623"

# The pre-migration legacy schema (matches the deleted ``_DDL`` block).
_LEGACY_PCL_DDL = """
CREATE TABLE profile_change_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    added_profiles TEXT NOT NULL DEFAULT '[]',
    removed_profiles TEXT NOT NULL DEFAULT '[]',
    mentioned_profiles TEXT NOT NULL DEFAULT '[]'
);
"""


def _seed_legacy_pcl(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_LEGACY_PCL_DDL)
        conn.execute(
            "INSERT INTO profile_change_logs (user_id, request_id, created_at) "
            "VALUES ('u1', 'r1', 1)"
        )
        conn.commit()
    finally:
        conn.close()


def _table_names(db_path: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        conn.close()


def _row_count(db_path: str, table: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
    finally:
        conn.close()


def test_legacy_table_is_renamed_and_data_preserved(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    _seed_legacy_pcl(db_path)

    SQLiteStorage(org_id="0", db_path=db_path)  # construction triggers migrate()

    tables = _table_names(db_path)
    assert "profile_change_logs" not in tables  # renamed away
    assert _RETIRED_NAME in tables  # reversible: data still recoverable
    assert _row_count(db_path, _RETIRED_NAME) == 1  # seeded row preserved


def test_table_not_recreated_by_second_migrate(tmp_path):
    """The ``_DDL`` deletion must hold: re-running migrate() does not resurrect it."""
    db_path = str(tmp_path / "legacy.db")
    _seed_legacy_pcl(db_path)

    SQLiteStorage(org_id="0", db_path=db_path)
    # Second open re-runs migrate() — executescript(_DDL) must NOT recreate it,
    # and the rename guard must no-op (source table already gone).
    SQLiteStorage(org_id="0", db_path=db_path)

    tables = _table_names(db_path)
    assert "profile_change_logs" not in tables
    assert _RETIRED_NAME in tables
    assert _row_count(db_path, _RETIRED_NAME) == 1  # not clobbered by re-run


def test_fresh_db_has_neither_table(tmp_path):
    """A fresh DB never creates ``profile_change_logs`` (removed from _DDL) and
    has nothing to rename."""
    db_path = str(tmp_path / "fresh.db")
    SQLiteStorage(org_id="0", db_path=db_path)

    tables = _table_names(db_path)
    assert "profile_change_logs" not in tables
    assert _RETIRED_NAME not in tables
