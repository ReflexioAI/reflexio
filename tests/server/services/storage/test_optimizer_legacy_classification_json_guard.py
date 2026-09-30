"""Upgrading a 0.2.28-era DB must survive a malformed ``metadata_json``.

``_classify_legacy_playbook_optimization_jobs`` runs in the storage constructor.
It used ``json_valid(x) AND json_type(x, ...)`` as a guard, but SQLite does not
short-circuit AND in a result-column expression: it evaluates ``json_type`` on a
non-JSON value and raises "malformed JSON". The 0.2.28 schema stored
``metadata_json`` as a plain ``TEXT NOT NULL DEFAULT '{}'`` and its model as a
plain ``str``, so nothing ever enforced validity -- one such row stopped the
backend from starting after an upgrade.

The DDL below is copied verbatim from reflexio 0.2.28 (26bed34), the runtime
every claude-smart install up to 0.2.51 vendors, so this exercises the real
pre-migration shape rather than a reconstruction of it.
"""

import sqlite3
from pathlib import Path

import pytest

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

_PIN_OPTIMIZER_DDL = """
CREATE TABLE playbook_optimization_jobs (
    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_kind TEXT NOT NULL,
    target_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    best_candidate_id INTEGER,
    successor_target_id INTEGER,
    decision_reason TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE TABLE playbook_optimization_candidates (
    candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    candidate_index INTEGER NOT NULL DEFAULT 0,
    content TEXT NOT NULL,
    parent_candidate_ids TEXT NOT NULL DEFAULT '[]',
    aggregate_score REAL,
    is_winner INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL
);
CREATE TABLE playbook_optimization_evaluations (
    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    candidate_id INTEGER NOT NULL,
    target_kind TEXT NOT NULL,
    target_id INTEGER NOT NULL,
    scenario_user_playbook_id INTEGER,
    source_interaction_ids TEXT NOT NULL DEFAULT '[]',
    score REAL NOT NULL DEFAULT 0.0,
    verdict TEXT NOT NULL DEFAULT 'tie',
    likert INTEGER NOT NULL DEFAULT 0,
    rationale TEXT NOT NULL DEFAULT '',
    asi_json TEXT NOT NULL DEFAULT '{}',
    incumbent_rollout_json TEXT NOT NULL DEFAULT '[]',
    candidate_rollout_json TEXT NOT NULL DEFAULT '[]',
    created_at INTEGER NOT NULL
);
CREATE TABLE playbook_optimization_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL
);
"""

_GEPA = (
    '{"source_window_count": 3, "train_window_count": 2, "validation_window_count": 1}'
)
_TUNER = '{"offline_tuner": {"round": 1}}'


def _legacy_db(
    tmp_path: Path, job_metadata: str, candidate_metadata: str | None = None
) -> Path:
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(_PIN_OPTIMIZER_DDL)
    conn.execute(
        """INSERT INTO playbook_optimization_jobs
           (job_id, target_kind, target_id, status, metadata_json,
            created_at, updated_at)
           VALUES (1, 'user_playbook', 1, 'completed', ?, 1, 1)""",
        (job_metadata,),
    )
    if candidate_metadata is not None:
        conn.execute(
            """INSERT INTO playbook_optimization_candidates
               (job_id, content, metadata_json, created_at)
               VALUES (1, 'candidate', ?, 1)""",
            (candidate_metadata,),
        )
    conn.commit()
    conn.close()
    return db_path


def _optimizer_kind(db_path: Path, org_id: str) -> str:
    storage = SQLiteStorage(org_id=org_id, db_path=str(db_path))
    try:
        row = storage.conn.execute(
            "SELECT optimizer_kind FROM playbook_optimization_jobs WHERE job_id = 1"
        ).fetchone()
        assert row is not None
        return row["optimizer_kind"]
    finally:
        storage.conn.close()


@pytest.mark.parametrize(
    ("job_metadata", "candidate_metadata"),
    [
        ("not json", None),
        ("", None),
        ("{}", "not json"),
        ("{}", ""),
    ],
    ids=["job-non-json", "job-empty", "candidate-non-json", "candidate-empty"],
)
def test_malformed_metadata_does_not_stop_the_upgrade(
    tmp_path: Path, job_metadata: str, candidate_metadata: str | None
) -> None:
    """A malformed value must be classified, not crash the constructor.

    The two JOB cases are regression tests: against the old AND guards they
    raised "malformed JSON", because those guards were result columns and
    SQLite evaluates both operands there.

    The two CANDIDATE cases are hardening, and are labelled so rather than
    claimed as regressions: that guard sits in a subquery WHERE clause, which
    short-circuited on SQLite 3.53.4, so these pass against the old code too.
    SQLite does not promise an evaluation order for WHERE terms, though, and a
    planner change would turn the same row into a startup crash -- these pin
    the behaviour that the CASE rewrite now guarantees.
    """
    db_path = _legacy_db(tmp_path, job_metadata, candidate_metadata)

    assert _optimizer_kind(db_path, "malformed") == "optimizer_legacy_unknown"


@pytest.mark.parametrize(
    ("job_metadata", "candidate_metadata", "expected"),
    [
        (_GEPA, None, "gepa"),
        (_TUNER, None, "offline_tuner_legacy"),
        ("{}", '{"rollback_baseline": 1}', "offline_tuner_legacy"),
        ("{}", None, "optimizer_legacy_unknown"),
    ],
    ids=["gepa", "tuner-job", "tuner-candidate", "no-signature"],
)
def test_valid_metadata_is_still_classified(
    tmp_path: Path,
    job_metadata: str,
    candidate_metadata: str | None,
    expected: str,
) -> None:
    """The lazy guard must not change what valid JSON classifies as."""
    db_path = _legacy_db(tmp_path, job_metadata, candidate_metadata)

    assert _optimizer_kind(db_path, "valid") == expected
