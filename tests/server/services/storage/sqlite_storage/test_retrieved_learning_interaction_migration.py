"""SQLite migration coverage for interaction-attributed learning verdicts."""

import sqlite3

import pytest

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

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
        assert {"interaction_id", "interaction_created_at"}.issubset(columns)
        legacy = conn.execute(
            """SELECT learning_id, interaction_id, interaction_created_at
               FROM retrieved_learning_evaluation WHERE result_id = 1"""
        ).fetchone()
        assert legacy == ("p1", None, None)

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
