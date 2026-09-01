"""The SQLite allowlist rebuild must work in the REMOVAL direction (OQ-3).

The rebuild's trigger predicate was written for the expand direction only: it
returns early when every REQUIRED literal is already present. Removing a literal
from the target tuple leaves an old database satisfying that predicate, so the
rebuild never runs and the permissive CHECK survives forever. These tests pin
the negative predicate and the row remediation that make removal work.

The databases here are built the way the rest of this suite builds a legacy
database: open a real ``SQLiteStorage`` so the full current schema exists, then
replace the two optimizer tables with their pre-Phase-7 permissive definitions.
Reopening the storage is what runs the rebuild.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pytest

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


_LEGACY_JOBS_DDL = """
CREATE TABLE playbook_optimization_jobs (
    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
    optimizer_kind TEXT NOT NULL DEFAULT 'optimizer_legacy_unknown'
        CHECK (optimizer_kind IN (
            'gepa', 'offline_tuner_replay', 'offline_tuner_open_world',
            'offline_tuner_legacy', 'optimizer_legacy_unknown'
        )),
    target_kind TEXT NOT NULL,
    target_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    best_candidate_id INTEGER,
    successor_target_id INTEGER,
    decision_reason TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    discovery_key TEXT,
    attempt_key TEXT,
    lease_owner TEXT,
    lease_fence INTEGER NOT NULL DEFAULT 0 CHECK (lease_fence >= 0),
    lease_expires_at INTEGER,
    stage TEXT CHECK (stage IS NULL OR stage IN (
        'evidence_frozen', 'discovery_analyzed', 'candidate_generated',
        'replay_running', 'replay_evaluated', 'held_out_analyzed',
        'publishing', 'applied', 'abstained', 'failed'
    )),
    terminal_outcome TEXT CHECK (terminal_outcome IS NULL OR terminal_outcome IN (
        'applied', 'replay_unsupported', 'replay_failed', 'governance_erased',
        'no_grounded_hypothesis', 'analyst_unqualified',
        'heldout_evidence_failed', 'stale_incumbent', 'governance_invalidated',
        'infrastructure_failure', 'insufficient_negative_evidence',
        'insufficient_positive_evidence', 'insufficient_coverage',
        'deployment_unsupported', 'incomplete_replay_scope',
        'insufficient_replay_cases', 'replay_inconclusive',
        'candidate_regressed', 'candidate_did_not_improve', 'incumbent_changed',
        'generation_failed', 'publication_failed'
    )),
    expected_population_manifest_digest TEXT,
    generation_selection_manifest_digest TEXT,
    replay_manifest_digest TEXT,
    candidate_content_digest TEXT,
    search_projection_digest TEXT,
    publication_scope_digest TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
)
"""

_LEGACY_ARTIFACTS_DDL = """
CREATE TABLE playbook_optimization_artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    artifact_kind TEXT NOT NULL CHECK (artifact_kind IN (
        'expected_population_manifest', 'generation_selection',
        'replay_manifest', 'candidate', 'candidate_search_projection',
        'open_world_evidence_bundle', 'open_world_discovery_memo',
        'open_world_candidate', 'open_world_attempt_decision'
    )),
    content_json TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    UNIQUE (job_id, artifact_kind),
    FOREIGN KEY (job_id) REFERENCES playbook_optimization_jobs(job_id)
        ON DELETE CASCADE
)
"""


def _write_legacy_optimizer_tables(db_path: Path, org_id: str) -> None:
    """Leave ``db_path`` holding the pre-Phase-7 permissive optimizer tables."""
    initial = SQLiteStorage(org_id=org_id, db_path=str(db_path))
    initial.conn.close()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("DROP INDEX IF EXISTS idx_poa_job")
    conn.execute("DROP TABLE playbook_optimization_artifacts")
    conn.execute("DROP INDEX IF EXISTS idx_poj_target")
    conn.execute("DROP INDEX IF EXISTS idx_poj_status")
    conn.execute("DROP INDEX IF EXISTS uq_poj_active_discovery")
    conn.execute("DROP INDEX IF EXISTS uq_poj_active_attempt")
    conn.execute("DROP INDEX IF EXISTS uq_poj_active_target")
    conn.execute("DROP TABLE playbook_optimization_jobs")
    conn.execute(_LEGACY_JOBS_DDL)
    conn.execute(_LEGACY_ARTIFACTS_DDL)
    conn.execute(
        """INSERT INTO playbook_optimization_jobs (
               job_id, optimizer_kind, target_kind, target_id, status,
               metadata_json, stage, terminal_outcome, created_at, updated_at
           ) VALUES (1, 'offline_tuner_replay', 'user_playbook', 1, 'skipped',
                     '{"offline_tuner": {}}', 'replay_evaluated',
                     'replay_inconclusive', 1, 1)"""
    )
    conn.execute(
        """INSERT INTO playbook_optimization_artifacts (
               artifact_id, job_id, artifact_kind, content_json, content_digest,
               created_at, updated_at
           ) VALUES (1, 1, 'replay_manifest', '{"a":1}', ?, 1, 1)""",
        ("a" * 64,),
    )
    conn.execute(
        """INSERT INTO playbook_optimization_artifacts (
               artifact_id, job_id, artifact_kind, content_json, content_digest,
               created_at, updated_at
           ) VALUES (2, 1, 'candidate', '{"b":2}', ?, 1, 1)""",
        ("b" * 64,),
    )
    conn.commit()
    conn.close()


def _table_sql(conn: sqlite3.Connection, name: str) -> str:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    assert row is not None
    return str(row["sql"])


def test_the_rebuild_trigger_is_not_satisfied_by_a_legacy_schema(
    tmp_path: Path,
) -> None:
    """The whole OQ-3 defect in one assertion.

    A positive 'all required present' predicate returns early against this
    database, because every SURVIVING literal is still in its table_sql. Only a
    predicate that also asks 'is any RETIRED literal still present?' fires.
    """
    from reflexio.server.services.storage.sqlite_storage import _base

    db_path = tmp_path / "legacy-trigger.db"
    _write_legacy_optimizer_tables(db_path, "legacy-trigger")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        jobs_sql = _table_sql(conn, "playbook_optimization_jobs")
        artifacts_sql = _table_sql(conn, "playbook_optimization_artifacts")
    finally:
        conn.close()

    # The positive half of the predicate is satisfied: nothing here is missing.
    assert "'offline_tuner_open_world'" in jobs_sql
    assert all(
        kind in artifacts_sql
        for kind in (
            "'expected_population_manifest'",
            "'open_world_attempt_decision'",
        )
    )
    # Only the negative half can see that the schema is stale.
    assert any(literal in jobs_sql for literal in _base._RETIRED_OPTIMIZER_JOB_LITERALS)
    assert any(
        literal in artifacts_sql for literal in _base._RETIRED_ARTIFACT_KIND_LITERALS
    )


def test_the_rebuild_contracts_the_allowlist_on_an_existing_database(
    tmp_path: Path,
) -> None:
    """After the rebuild the CHECK must refuse the retired literal."""
    db_path = tmp_path / "legacy-contract.db"
    _write_legacy_optimizer_tables(db_path, "legacy-contract")
    storage = SQLiteStorage(org_id="legacy-contract", db_path=str(db_path))
    try:
        jobs_sql = _table_sql(storage.conn, "playbook_optimization_jobs")
        artifacts_sql = _table_sql(storage.conn, "playbook_optimization_artifacts")
        assert "offline_tuner_replay" not in jobs_sql
        assert "replay_evaluated" not in jobs_sql
        assert "replay_inconclusive" not in jobs_sql
        assert "offline_tuner_legacy" in jobs_sql
        # replay_manifest_digest is a COLUMN name, not a vocabulary literal.
        assert "'replay_manifest'" not in artifacts_sql

        with pytest.raises(sqlite3.IntegrityError):
            storage.conn.execute(
                """INSERT INTO playbook_optimization_jobs (
                       optimizer_kind, target_kind, target_id, created_at,
                       updated_at
                   ) VALUES ('offline_tuner_replay', 'user_playbook', 2, 1, 1)"""
            )
        storage.conn.rollback()
        with pytest.raises(sqlite3.IntegrityError):
            storage.conn.execute(
                """INSERT INTO playbook_optimization_artifacts (
                       job_id, artifact_kind, content_json, content_digest,
                       created_at, updated_at
                   ) VALUES (1, 'replay_manifest', '{}', ?, 1, 1)""",
                ("c" * 64,),
            )
        storage.conn.rollback()
    finally:
        storage.conn.close()


def test_the_rebuild_remediates_the_rows_it_would_otherwise_reject(
    tmp_path: Path,
) -> None:
    """A table copy fails on a row carrying a retired value.

    The remediation is documented, not incidental: relabel the job to
    'offline_tuner_legacy' (the accurate surviving label, retained by OQ-1
    option A), null the retired stage and terminal_outcome (both nullable), and
    delete the artifact rows, whose kind column is NOT NULL and part of a
    UNIQUE key so there is no surviving value to map onto.
    """
    db_path = tmp_path / "legacy-remediation.db"
    _write_legacy_optimizer_tables(db_path, "legacy-remediation")
    storage = SQLiteStorage(org_id="legacy-remediation", db_path=str(db_path))
    try:
        row = storage.conn.execute(
            "SELECT optimizer_kind, stage, terminal_outcome "
            "FROM playbook_optimization_jobs WHERE job_id = 1"
        ).fetchone()
        assert row is not None
        assert row["optimizer_kind"] == "offline_tuner_legacy"
        assert row["stage"] is None
        assert row["terminal_outcome"] is None

        kinds = [
            artifact["artifact_kind"]
            for artifact in storage.conn.execute(
                "SELECT artifact_kind FROM playbook_optimization_artifacts "
                "WHERE job_id = 1 ORDER BY artifact_id"
            ).fetchall()
        ]
        assert kinds == ["candidate"]
    finally:
        storage.conn.close()


@pytest.mark.parametrize("status", ["pending", "running"])
def test_the_rebuild_leaves_no_relabelled_job_active(
    tmp_path: Path,
    status: str,
) -> None:
    """A relabelled replay job must not survive as an ACTIVE legacy job.

    ``_classify_legacy_playbook_optimization_jobs`` retires every active
    ``offline_tuner_legacy`` / ``optimizer_legacy_unknown`` job, and it runs
    BEFORE this rebuild. Without the rebuild's own remediation, a running replay
    job is reopened as a running ``offline_tuner_legacy`` job -- a kind with no
    producer or consumer -- holding ``uq_poj_active_target`` /
    ``uq_poj_active_discovery`` / ``uq_poj_active_attempt`` until the NEXT open
    lets that sweep see it. One open must be enough.
    """
    db_path = tmp_path / f"legacy-active-{status}.db"
    _write_legacy_optimizer_tables(db_path, f"legacy-active-{status}")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT INTO playbook_optimization_jobs (
               job_id, optimizer_kind, target_kind, target_id, status,
               discovery_key, attempt_key, lease_owner, lease_expires_at,
               created_at, updated_at
           ) VALUES (2, 'offline_tuner_replay', 'user_playbook', 2, ?,
                     'discovery-2', 'attempt-2', 'worker-a', 9999, 1, 1)""",
        (status,),
    )
    conn.commit()
    conn.close()

    storage = SQLiteStorage(org_id=f"legacy-active-{status}", db_path=str(db_path))
    try:
        row = storage.conn.execute(
            "SELECT optimizer_kind, status, decision_reason, lease_owner, "
            "lease_expires_at FROM playbook_optimization_jobs WHERE job_id = 2"
        ).fetchone()
        assert row is not None
        assert row["optimizer_kind"] == "offline_tuner_legacy"
        assert row["status"] == "skipped"
        assert row["decision_reason"] == "retired_by_replay_redesign"
        assert row["lease_owner"] is None
        assert row["lease_expires_at"] is None
        assert (
            storage.conn.execute(
                "SELECT COUNT(*) FROM playbook_optimization_jobs "
                "WHERE status IN ('pending', 'running')"
            ).fetchone()[0]
            == 0
        )
    finally:
        storage.conn.close()


def test_a_settled_relabelled_job_keeps_its_own_status_and_reason(
    tmp_path: Path,
) -> None:
    """The remediation must scope to ACTIVE rows, not rewrite settled history."""
    db_path = tmp_path / "legacy-settled.db"
    _write_legacy_optimizer_tables(db_path, "legacy-settled")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT INTO playbook_optimization_jobs (
               job_id, optimizer_kind, target_kind, target_id, status,
               decision_reason, created_at, updated_at
           ) VALUES (3, 'offline_tuner_replay', 'user_playbook', 3, 'completed',
                     'published_successor', 1, 1)"""
    )
    conn.commit()
    conn.close()

    storage = SQLiteStorage(org_id="legacy-settled", db_path=str(db_path))
    try:
        row = storage.conn.execute(
            "SELECT optimizer_kind, status, decision_reason "
            "FROM playbook_optimization_jobs WHERE job_id = 3"
        ).fetchone()
        assert row is not None
        assert row["optimizer_kind"] == "offline_tuner_legacy"
        assert row["status"] == "completed"
        assert row["decision_reason"] == "published_successor"
    finally:
        storage.conn.close()


def test_the_destroyed_replay_artifacts_are_logged_with_their_count(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Deleting ``replay_manifest`` rows destroys their ``content_json``.

    The sibling ``optimizer_kind`` vocabulary retains its retired labels so
    historical rows survive; this vocabulary cannot, because ``artifact_kind`` is
    NOT NULL and part of ``UNIQUE (job_id, artifact_kind)`` and the tenant
    contract drops the literal too. Destruction is therefore correct -- but it
    must not be silent.
    """
    db_path = tmp_path / "legacy-artifact-log.db"
    _write_legacy_optimizer_tables(db_path, "legacy-artifact-log")

    with caplog.at_level(
        logging.WARNING,
        logger="reflexio.server.services.storage.sqlite_storage._base",
    ):
        storage = SQLiteStorage(org_id="legacy-artifact-log", db_path=str(db_path))
        storage.conn.close()

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
        and "replay_manifest" in record.getMessage()
    ]
    assert len(warnings) == 1, caplog.text
    assert "destroyed 1 " in warnings[0]


def test_the_rebuild_is_idempotent(tmp_path: Path) -> None:
    """A second run must return early, or every open costs a table copy."""
    db_path = tmp_path / "legacy-idempotent.db"
    _write_legacy_optimizer_tables(db_path, "legacy-idempotent")
    first_storage = SQLiteStorage(org_id="legacy-idempotent", db_path=str(db_path))
    try:
        first_jobs = _table_sql(first_storage.conn, "playbook_optimization_jobs")
        first_artifacts = _table_sql(
            first_storage.conn, "playbook_optimization_artifacts"
        )
        first_job_row_id = first_storage.conn.execute(
            "SELECT job_id FROM playbook_optimization_jobs"
        ).fetchone()["job_id"]
    finally:
        first_storage.conn.close()

    second_storage = SQLiteStorage(org_id="legacy-idempotent", db_path=str(db_path))
    try:
        assert (
            _table_sql(second_storage.conn, "playbook_optimization_jobs") == first_jobs
        )
        assert (
            _table_sql(second_storage.conn, "playbook_optimization_artifacts")
            == first_artifacts
        )
        assert (
            second_storage.conn.execute(
                "SELECT job_id FROM playbook_optimization_jobs"
            ).fetchone()["job_id"]
            == first_job_row_id
        )
    finally:
        second_storage.conn.close()
