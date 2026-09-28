"""Upgrade real historical SQLite rows without inventing progress evidence."""

from unittest.mock import patch

from reflexio.models.api_schema.service_schemas import UserProfile
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.sqlite_storage._base import SQLiteStorageBase


def test_historical_rows_and_receipts_survive_reopen_upgrade(tmp_path):
    path = str(tmp_path / "historical.db")
    # The base CREATE TABLE is the pre-upgrade schema. Suppress only this new
    # migration while creating it; all other ordinary initialization still runs.
    with patch.object(SQLiteStorageBase, "_migrate_agent_runs_schema"):
        storage = SQLiteStorage(org_id="org", db_path=path)
    try:
        columns = [r[1] for r in storage.conn.execute("PRAGMA table_info(_agent_runs)")]
        assert "last_progress_at" not in columns
        for identifier, status in (
            ("old-running", "running"),
            ("old-output", "finalization_failed"),
        ):
            storage.conn.execute(
                """INSERT INTO _agent_runs
                (id,org_id,extractor_kind,request_id,status,committed_output,
                 claimed_by,claimed_at,updated_at)
                VALUES (?,'org','profile','request',?,'{"profiles":[]}',
                  'old-owner','2026-08-01T00:00:00+00:00','2026-08-01T00:00:00+00:00')""",
                (identifier, status),
            )
        storage.conn.execute(
            "INSERT INTO _agent_run_finalization_receipts(run_id,entity_type,learning_ids) VALUES ('old-output','profile','[]')"
        )
        storage.conn.commit()
        before = [
            dict(r)
            for r in storage.conn.execute("SELECT * FROM _agent_runs ORDER BY id")
        ]
        receipts = [
            dict(r)
            for r in storage.conn.execute(
                "SELECT * FROM _agent_run_finalization_receipts"
            )
        ]
    finally:
        storage.conn.close()
    for _ in range(2):
        upgraded = SQLiteStorage(org_id="org", db_path=path)
        try:
            after = [
                dict(r)
                for r in upgraded.conn.execute("SELECT * FROM _agent_runs ORDER BY id")
            ]
            assert [{key: row[key] for key in columns} for row in after] == before
            assert all(
                row["progress_stage"] is None
                and row["last_progress_at"] is None
                and row["recovery_operation_id"] is None
                and row["recovery_history"] == "[]"
                for row in after
            )
            assert [
                dict(r)
                for r in upgraded.conn.execute(
                    "SELECT * FROM _agent_run_finalization_receipts"
                )
            ] == receipts
        finally:
            upgraded.conn.close()


def test_sqlite_nonempty_receipt_detects_deleted_result(tmp_path):
    storage = SQLiteStorage(org_id="org", db_path=str(tmp_path / "receipt.db"))
    try:
        storage.conn.execute(
            """INSERT INTO _agent_runs (id,org_id,extractor_kind,request_id,status)
            VALUES ('run','org','profile','request','finalization_failed')"""
        )
        storage.conn.commit()
        profile = UserProfile(
            profile_id="receipt-result",
            user_id="user",
            content="Synthetic result",
            last_modified_timestamp=1_700_000_000,
            generated_from_request_id="request",
            embedding=[0.0] * 512,
        )
        storage.add_user_profile("user", [profile], skip_embedding=True)
        storage.save_agent_run_finalization_receipt(
            run_id="run", entity_type="profile", learning_ids=[profile.profile_id]
        )
        facts = storage.get_agent_run_recovery_facts("run")
        assert facts["receipt_count"] == 1 and facts["receipt_missing_results"] == 0
        storage.conn.execute(
            "DELETE FROM profiles WHERE profile_id=?", (profile.profile_id,)
        )
        storage.conn.commit()
        facts = storage.get_agent_run_recovery_facts("run")
        assert facts["receipt_exists"] and facts["receipt_missing_results"] == 1
    finally:
        storage.conn.close()
