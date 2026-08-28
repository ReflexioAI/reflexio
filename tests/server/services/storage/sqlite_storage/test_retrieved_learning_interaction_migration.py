"""SQLite migration coverage for interaction-attributed learning verdicts."""

import multiprocessing
import sqlite3
from unittest.mock import Mock, patch

import pytest

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.sqlite_storage._base import SQLiteStorageBase

_LEGACY_DDL = """
CREATE TABLE retrieved_learning_evaluation (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    agent_version TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL,
    learning_id TEXT NOT NULL,
    is_relevant INTEGER,
    relevance_reason TEXT NOT NULL DEFAULT '',
    impact TEXT,
    impact_reason TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    governance_subject_ref TEXT,
    UNIQUE (user_id, session_id, kind, learning_id)
);
CREATE INDEX idx_rle_created_at_result_id
    ON retrieved_learning_evaluation(created_at DESC, result_id DESC);
"""


def _seed_legacy_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(_LEGACY_DDL)
    conn.execute(
        """INSERT INTO retrieved_learning_evaluation (
               user_id, session_id, agent_version, kind, learning_id,
               is_relevant, relevance_reason, impact, impact_reason, created_at
           ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
        ("u1", "s1", "v1", "profile", "p1", 1, "relevant", "positive", "helped", 10),
    )
    conn.commit()
    conn.close()


def test_migration_preserves_legacy_rows_and_changes_identity(tmp_path) -> None:
    db_path = str(tmp_path / "legacy.db")
    _seed_legacy_db(db_path)

    SQLiteStorage(org_id="0", db_path=db_path)
    # A second startup proves the migration is idempotent.
    SQLiteStorage(org_id="0", db_path=db_path)

    conn = sqlite3.connect(db_path)
    try:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(retrieved_learning_evaluation)")
        }
        assert {
            "interaction_id",
            "interaction_created_at",
            "diagnosis",
            "evaluated_playbook_digest",
            "diagnosis_evidence_complete",
        }.issubset(columns)
        legacy = conn.execute(
            """SELECT learning_id, interaction_id, interaction_created_at, diagnosis, evaluated_playbook_digest, diagnosis_evidence_complete
               FROM retrieved_learning_evaluation WHERE result_id = 1"""
        ).fetchone()
        assert legacy == ("p1", None, None, None, None, 0)

        values = ("u1", "s1", "v1", 20, 200, "profile", "p1", 1, "", "positive", "", 20)
        conn.execute(
            """INSERT INTO retrieved_learning_evaluation (
                   user_id, session_id, agent_version, interaction_id,
                   interaction_created_at, kind, learning_id, is_relevant,
                   relevance_reason, impact, impact_reason, created_at
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            values,
        )
        conn.execute(
            """INSERT INTO retrieved_learning_evaluation (
                   user_id, session_id, agent_version, interaction_id,
                   interaction_created_at, kind, learning_id, is_relevant,
                   relevance_reason, impact, impact_reason, created_at
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (*values[:3], 21, 201, *values[5:]),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO retrieved_learning_evaluation (
                       user_id, session_id, agent_version, interaction_id,
                       interaction_created_at, kind, learning_id, is_relevant,
                       relevance_reason, impact, impact_reason, created_at
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                values,
            )
    finally:
        conn.close()


def _initialize_diagnosis_worker(db_path, index, ready, inspected, competing, results):
    original_migration = SQLiteStorageBase._migrate_playbook_diagnosis

    def migrate(storage):
        # Both processes finish older migrations before racing the new one.
        ready.wait(timeout=15)
        if index == 1:
            assert inspected.wait(timeout=15)
        conn = storage.conn

        def execute(statement, *args):
            if index == 1 and statement == "BEGIN IMMEDIATE":
                competing.set()
            cursor = conn.execute(statement, *args)
            if statement == "PRAGMA table_info(retrieved_learning_evaluation)":
                rows = cursor.fetchall()
                if index == 0:
                    inspected.set()
                    assert competing.wait(timeout=15)
                else:
                    # Without a write lock both readers see the old schema.
                    competing.set()
                return iter(rows)
            return cursor

        proxy = Mock(wraps=conn)
        proxy.execute.side_effect = execute
        with patch.object(storage, "conn", proxy):
            original_migration(storage)

    try:
        with patch.object(SQLiteStorageBase, "_migrate_playbook_diagnosis", migrate):
            storage = SQLiteStorage(org_id="0", db_path=db_path)
            storage.conn.close()
        results.put(None)
    except Exception as exc:
        results.put(f"{type(exc).__name__}: {exc}")


def _pre_diagnosis_storage(db_path):
    storage = SQLiteStorage(org_id="0", db_path=db_path)
    with storage.conn:
        for column in (
            "diagnosis",
            "evaluated_playbook_digest",
            "diagnosis_evidence_complete",
        ):
            storage.conn.execute(
                f"ALTER TABLE retrieved_learning_evaluation DROP COLUMN {column}"
            )
    return storage


def test_diagnosis_migration_serializes_concurrent_startup(tmp_path):
    db_path = str(tmp_path / "concurrent.db")
    storage = _pre_diagnosis_storage(db_path)
    storage.conn.close()

    context = multiprocessing.get_context("spawn")
    ready = context.Barrier(2)
    inspected = context.Event()
    competing = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_initialize_diagnosis_worker,
            args=(db_path, index, ready, inspected, competing, results),
        )
        for index in range(2)
    ]
    try:
        for process in processes:
            process.start()
        outcomes = [results.get(timeout=30) for _ in processes]
        assert outcomes == [None, None], outcomes
        for process in processes:
            process.join(timeout=5)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        results.close()
        results.join_thread()


def test_diagnosis_migration_rolls_back_partial_upgrade(tmp_path):
    storage = _pre_diagnosis_storage(str(tmp_path / "rollback.db"))
    alterations = 0

    def deny_second_alter(action, *_args):
        nonlocal alterations
        if action == sqlite3.SQLITE_ALTER_TABLE:
            alterations += 1
            if alterations == 2:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    try:
        storage.conn.set_authorizer(deny_second_alter)
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            storage._migrate_playbook_diagnosis()
        storage.conn.set_authorizer(None)
        assert not storage.conn.in_transaction
        columns = {
            row["name"]
            for row in storage.conn.execute(
                "PRAGMA table_info(retrieved_learning_evaluation)"
            )
        }
        assert "diagnosis" not in columns
        # Rollback leaves the connection usable for a complete retry.
        storage._migrate_playbook_diagnosis()
        storage.conn.execute(
            "SELECT diagnosis, evaluated_playbook_digest, diagnosis_evidence_complete "
            "FROM retrieved_learning_evaluation"
        )
    finally:
        storage.conn.close()
