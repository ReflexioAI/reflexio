"""Migration test: ``playbook_aggregation_change_logs`` is retired via a reversible RENAME.

Covers ``SQLiteStorageBase._migrate_retire_playbook_aggregation_change_logs`` plus
the deletion of the table's ``CREATE TABLE`` from ``_DDL`` (Lineage Track B Task 4).
The two invariants under test:

  (a) an existing legacy ``playbook_aggregation_change_logs`` table is renamed to
      ``playbook_aggregation_change_logs_retired_20260624`` (data preserved, recoverable);
  (b) the table is NOT recreated by a second ``migrate()`` — i.e. the headline
      trap (``executescript(_DDL)`` resurrecting an empty table) does not bite.
"""

import sqlite3

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

_RETIRED_NAME = "playbook_aggregation_change_logs_retired_20260624"

# The pre-migration legacy schema (matches the deleted ``_DDL`` block).
_LEGACY_PACL_DDL = """
CREATE TABLE playbook_aggregation_change_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at INTEGER NOT NULL,
    playbook_name TEXT NOT NULL,
    agent_version TEXT NOT NULL,
    run_mode TEXT NOT NULL,
    added_playbooks TEXT NOT NULL DEFAULT '[]',
    removed_playbooks TEXT NOT NULL DEFAULT '[]',
    updated_playbooks TEXT NOT NULL DEFAULT '[]'
);
"""


def _seed_legacy_pacl(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_LEGACY_PACL_DDL)
        conn.execute(
            "INSERT INTO playbook_aggregation_change_logs "
            "(created_at, playbook_name, agent_version, run_mode) "
            "VALUES (1, 'pb', 'v1', 'incremental')"
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
    _seed_legacy_pacl(db_path)

    SQLiteStorage(org_id="0", db_path=db_path)  # construction triggers migrate()

    tables = _table_names(db_path)
    assert "playbook_aggregation_change_logs" not in tables  # renamed away
    assert _RETIRED_NAME in tables  # reversible: data still recoverable
    assert _row_count(db_path, _RETIRED_NAME) == 1  # seeded row preserved


def test_table_not_recreated_by_second_migrate(tmp_path):
    """The ``_DDL`` deletion must hold: re-running migrate() does not resurrect it."""
    db_path = str(tmp_path / "legacy.db")
    _seed_legacy_pacl(db_path)

    SQLiteStorage(org_id="0", db_path=db_path)
    # Second open re-runs migrate() — executescript(_DDL) must NOT recreate it,
    # and the rename guard must no-op (source table already gone).
    SQLiteStorage(org_id="0", db_path=db_path)

    tables = _table_names(db_path)
    assert "playbook_aggregation_change_logs" not in tables
    assert _RETIRED_NAME in tables
    assert _row_count(db_path, _RETIRED_NAME) == 1  # not clobbered by re-run


def test_fresh_db_has_neither_table(tmp_path):
    """A fresh DB never creates ``playbook_aggregation_change_logs`` (removed from _DDL)
    and has nothing to rename."""
    db_path = str(tmp_path / "fresh.db")
    SQLiteStorage(org_id="0", db_path=db_path)

    tables = _table_names(db_path)
    assert "playbook_aggregation_change_logs" not in tables
    assert _RETIRED_NAME not in tables
