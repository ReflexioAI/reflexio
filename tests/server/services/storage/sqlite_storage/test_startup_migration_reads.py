"""Current migrations must not compete with ordinary SQLite writers."""

import pytest

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage


@pytest.mark.parametrize(
    "migration",
    [
        "_classify_legacy_playbook_optimization_jobs",
        "_migrate_session_outcomes_schema",
        "_migrate_playbook_diagnosis",
        "migrate",
    ],
)
def test_current_migration_reads_under_independent_writer(tmp_path, migration):
    path = str(tmp_path / "current.db")
    writer = SQLiteStorage(org_id="0", db_path=path)
    reader = SQLiteStorage(org_id="0", db_path=path)
    try:
        # A short busy timeout makes an accidental write fail promptly; no
        # elapsed-time assertion depends on machine load.
        reader.conn.execute("PRAGMA busy_timeout=100")
        with writer.commit_scope():
            getattr(reader, migration)()
            assert not reader.conn.in_transaction
    finally:
        reader.conn.close()
        writer.conn.close()


@pytest.mark.parametrize("index", ["discovery", "attempt", "target"])
@pytest.mark.parametrize("mode", ["missing", "nonunique", "changed_predicate"])
def test_optimizer_classification_repairs_noncurrent_index(tmp_path, index, mode):
    store = SQLiteStorage(org_id="0", db_path=str(tmp_path / "repair.db"))
    name = f"uq_poj_active_{index}"
    try:
        original = store.conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = ?", (name,)
        ).fetchone()[0]
        store.conn.execute(f"DROP INDEX {name}")
        if mode == "nonunique":
            # Same name, but a nonunique index offers no duplicate protection.
            store.conn.execute(
                f"CREATE INDEX {name} ON playbook_optimization_jobs(job_id)"
            )
        elif mode == "changed_predicate":
            # Whitespace inside a quoted status changes the filter's meaning.
            store.conn.execute(original.replace("'pending'", "' pending'"))
        store.conn.commit()
        store._classify_legacy_playbook_optimization_jobs()
        repaired = store.conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = ?", (name,)
        ).fetchone()[0]
        assert " ".join(repaired.split()) == " ".join(original.split())
    finally:
        store.conn.close()


@pytest.mark.parametrize("kind", ["offline_tuner_legacy", "optimizer_legacy_unknown"])
def test_current_optimizer_indexes_do_not_hide_legacy_active_rows(tmp_path, kind):
    store = SQLiteStorage(org_id="0", db_path=str(tmp_path / "legacy.db"))
    try:
        store.conn.execute(
            "INSERT INTO playbook_optimization_jobs "
            "(optimizer_kind, target_kind, target_id, status, created_at, updated_at) "
            "VALUES (?, 'user_playbook', 1, 'running', 1, 1)",
            (kind,),
        )
        store.conn.commit()
        store._classify_legacy_playbook_optimization_jobs()
        row = store.conn.execute(
            "SELECT status, decision_reason FROM playbook_optimization_jobs"
        ).fetchone()
        assert tuple(row) == ("skipped", "retired_by_replay_redesign")
    finally:
        store.conn.close()


def test_current_outcome_schema_still_repairs_blank_subject(tmp_path):
    store = SQLiteStorage(org_id="0", db_path=str(tmp_path / "subject.db"))
    try:
        store.conn.execute(
            "INSERT INTO session_outcomes "
            "(outcome_id, outcome_revision, user_id, session_id, outcome, occurred_at, "
            "source, outcome_contract_digest, finalized_trajectory_digest, "
            "governance_subject_ref, created_at) "
            "VALUES ('outcome', 1, 'user', 'session', 'unknown', 1, 'test', "
            "'contract', 'trajectory', '   ', 1)"
        )
        store.conn.commit()
        store._migrate_session_outcomes_schema()
        assert store.conn.execute(
            "SELECT governance_subject_ref FROM session_outcomes"
        ).fetchone()[0] == store._subject_ref_for_user_id("user")
    finally:
        store.conn.close()


def test_initialization_restores_missing_stall_singleton(tmp_path):
    store = SQLiteStorage(org_id="0", db_path=str(tmp_path / "stall.db"))
    try:
        store.conn.execute("DELETE FROM stall_state")
        store.conn.commit()
        store.migrate()
        assert tuple(
            store.conn.execute("SELECT id, stalled FROM stall_state").fetchone()
        ) == (1, 0)
    finally:
        store.conn.close()
