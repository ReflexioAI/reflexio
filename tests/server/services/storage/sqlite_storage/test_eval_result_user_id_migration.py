"""Migration test: a pre-existing DB whose ``agent_success_evaluation_result``
table predates the ``user_id`` column must still migrate cleanly.

Regression for an ordering bug: ``_DDL`` builds
``idx_eval_identity_created_at_desc`` on
``agent_success_evaluation_result(user_id, ...)``, while the column is added by
``_migrate_eval_result_user_id``. When that helper ran *after*
``executescript(_DDL)``, an upgraded DB raised
``sqlite3.OperationalError: no such column: user_id`` on every boot and never
reached the backfill — leaving the DB permanently stuck. The helper now runs
before ``_DDL``.
"""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

# Pre-``user_id`` schema for the eval-result table (and its old index, which did
# not reference user_id). Mirrors the columns shipped before the per-user change.
_LEGACY_EVAL_DDL = """
CREATE TABLE agent_success_evaluation_result (
    result_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    agent_version TEXT NOT NULL DEFAULT '',
    evaluation_name TEXT NOT NULL DEFAULT '',
    is_success INTEGER NOT NULL DEFAULT 0,
    failure_type TEXT,
    failure_reason TEXT,
    created_at TEXT NOT NULL,
    regular_vs_shadow TEXT NOT NULL DEFAULT 'regular',
    embedding BLOB
);
CREATE INDEX idx_eval_identity_created_at_desc
    ON agent_success_evaluation_result
    (session_id, evaluation_name, agent_version, created_at DESC);
"""


def _seed_legacy_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(_LEGACY_EVAL_DDL)
    conn.execute(
        "INSERT INTO agent_success_evaluation_result "
        "(result_id, session_id, evaluation_name, created_at) "
        "VALUES ('r1', 's1', 'eval', '2026-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()


def _eval_columns(db_path: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(agent_success_evaluation_result)"
            )
        }
    finally:
        conn.close()


def test_migrate_adds_user_id_and_tags_on_legacy_eval_table(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    _seed_legacy_db(db_path)

    # Without the fix this raises OperationalError: no such column: user_id.
    SQLiteStorage(org_id="0", db_path=db_path)  # construction triggers migrate()

    cols = _eval_columns(db_path)
    assert "user_id" in cols  # backfill ran
    assert "tags" in cols
    # Existing row preserved with the NOT NULL DEFAULT '' backfill value.
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT result_id, user_id FROM agent_success_evaluation_result"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [("r1", "")]


def test_migrate_is_idempotent_on_legacy_eval_table(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    _seed_legacy_db(db_path)

    SQLiteStorage(org_id="0", db_path=db_path)
    SQLiteStorage(org_id="0", db_path=db_path)  # second boot must not raise

    assert "user_id" in _eval_columns(db_path)
    assert "tags" in _eval_columns(db_path)


def test_current_evaluation_index_migration_does_not_wait_for_writer(tmp_path):
    db_path = str(tmp_path / "current.db")
    writer = SQLiteStorage(org_id="0", db_path=db_path)
    reader = SQLiteStorage(org_id="0", db_path=db_path)
    try:
        with writer.commit_scope():
            reader._migrate_eval_result_user_id()
    finally:
        reader.conn.close()
        writer.conn.close()


@pytest.mark.parametrize("stale", [False, True])
def test_evaluation_index_migration_repairs_missing_or_stale_index(tmp_path, stale):
    store = SQLiteStorage(org_id="0", db_path=str(tmp_path / "repair.db"))
    try:
        store.conn.execute("DROP INDEX idx_eval_identity_created_at_desc")
        if stale:
            store.conn.execute(
                "CREATE INDEX idx_eval_identity_created_at_desc "
                "ON agent_success_evaluation_result(session_id, created_at)"
            )
        store.conn.commit()
        store._migrate_eval_result_user_id()
        assert [
            (row[2], row[3])
            for row in store.conn.execute(
                "PRAGMA index_xinfo(idx_eval_identity_created_at_desc)"
            )
            if row[5]
        ] == [
            ("user_id", 0),
            ("session_id", 0),
            ("evaluation_name", 0),
            ("agent_version", 0),
            ("created_at", 1),
        ]
    finally:
        store.conn.close()


def test_concurrent_evaluation_index_upgrades_tolerate_another_creator(tmp_path):
    db_path = str(tmp_path / "concurrent.db")
    seed = sqlite3.connect(db_path)
    try:
        seed.execute(
            "CREATE TABLE agent_success_evaluation_result "
            "(user_id TEXT, session_id TEXT, evaluation_name TEXT, "
            "agent_version TEXT, created_at INTEGER)"
        )
    finally:
        seed.close()
    barrier = Barrier(2)

    class CoordinatedConnection(sqlite3.Connection):
        def execute(self, sql, parameters=(), /):
            cursor = super().execute(sql, parameters)
            if sql.startswith("DROP INDEX"):
                # Both connections saw a missing index and dropped it before
                # either creates it, reproducing separate-worker startup.
                barrier.wait(timeout=5)
            return cursor

    def migrate(_index):
        connection = sqlite3.connect(db_path, factory=CoordinatedConnection)
        connection.row_factory = sqlite3.Row
        storage = SQLiteStorage.__new__(SQLiteStorage)
        storage.conn = connection
        try:
            storage._migrate_eval_result_user_id()
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(migrate, range(2))) == [None, None]
