"""Regression test for legacy SQLite candidate tables missing metadata_json."""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import patch

from reflexio.models.api_schema.domain import (
    PlaybookOptimizationCandidate,
    PlaybookOptimizationJob,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

_LEGACY_CANDIDATE_DDL = """
CREATE TABLE playbook_optimization_candidates (
    candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    candidate_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    parent_candidate_ids TEXT NOT NULL DEFAULT '[]',
    aggregate_score REAL,
    is_winner INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX idx_poc_job ON playbook_optimization_candidates(job_id);
"""


def _candidate_columns(db_path: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(playbook_optimization_candidates)"
            ).fetchall()
        }
    finally:
        conn.close()


def test_migration_adds_candidate_metadata_column_and_persists_metadata(tmp_path) -> None:
    db_path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(_LEGACY_CANDIDATE_DDL)
    conn.commit()
    conn.close()

    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        storage = SQLiteStorage(org_id="legacy-opt-test", db_path=db_path)

    assert "metadata_json" in _candidate_columns(db_path)

    job = storage.create_playbook_optimization_job(
        PlaybookOptimizationJob(target_kind="user_playbook", target_id=10)
    )
    metadata_json = json.dumps(
        {
            "rollback_baseline": {"user_playbook_id": 10},
            "frozen_selection_set": {
                "target_user_playbook_id": 10,
                "sessions": [],
            },
        }
    )
    storage.insert_playbook_optimization_candidate(
        PlaybookOptimizationCandidate(
            job_id=job.job_id,
            candidate_index=0,
            content="candidate",
            metadata_json=metadata_json,
        )
    )

    [persisted] = storage.list_playbook_optimization_candidates(job.job_id)
    assert json.loads(persisted.metadata_json) == json.loads(metadata_json)
