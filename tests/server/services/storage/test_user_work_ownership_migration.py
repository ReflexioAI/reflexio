"""Private owner state upgrades preserve live work and commit both fields together."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from threading import Barrier

import pytest

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def _columns(storage, table):
    return {row["name"] for row in storage.conn.execute(f"PRAGMA table_info({table})")}


def _ownership_columns(storage):
    return (
        "user_work_token" in _columns(storage, "_agent_runs"),
        "position_json" in _columns(storage, "learning_work"),
    )


@pytest.fixture
def legacy_storage(tmp_path):
    storage = SQLiteStorage(
        org_id="synthetic_owner", db_path=str(tmp_path / "owner.db")
    )
    try:
        for table, name in (
            ("_agent_runs", "user_work_token"),
            ("learning_work", "position_json"),
        ):
            if name in _columns(storage, table):
                storage.conn.execute(f"ALTER TABLE {table} DROP COLUMN {name}")
        storage.conn.commit()
        token = storage.claim_user_extraction("synthetic_user", "original_owner", 300)
        assert token is not None
        storage.conn.execute(
            "INSERT INTO _agent_runs (id,org_id,user_id,extractor_kind,request_id,status,"
            "claimed_by,claimed_at,updated_at,resume_attempts,generation_request_snapshot) "
            "VALUES ('legacy_run',?,'synthetic_user','profile','synthetic_request',"
            "'resuming','original_worker','2026-01-01T00:00:00Z',"
            "'2026-01-01T00:00:00Z',3,'{}')",
            (storage.org_id,),
        )
        storage.conn.commit()
        yield storage, token
    finally:
        storage.conn.close()


def _legacy_state(storage):
    lease = storage.conn.execute(
        "SELECT lease_token,lease_owner,lease_until,last_turn FROM learning_work "
        "WHERE org_id=? AND user_id='synthetic_user'",
        (storage.org_id,),
    ).fetchone()
    run = storage.conn.execute(
        "SELECT status,claimed_by,claimed_at,updated_at,resume_attempts "
        "FROM _agent_runs WHERE id='legacy_run'"
    ).fetchone()
    return tuple(lease), tuple(run)


def test_fresh_owner_fields_stay_private(tmp_path):
    storage = SQLiteStorage(
        org_id="synthetic_fresh", db_path=str(tmp_path / "fresh.db")
    )
    try:
        assert _ownership_columns(storage) == (True, True)
    finally:
        storage.conn.close()


def test_owner_upgrade_preserves_live_work_and_repeated_state(legacy_storage):
    storage, token = legacy_storage
    before = _legacy_state(storage)
    storage.migrate()
    assert _ownership_columns(storage) == (True, True)
    assert _legacy_state(storage) == before
    storage.fence_user_extraction("synthetic_user", token)
    position = json.dumps(
        {
            "last_class": "continuation",
            "continuation": {"after_id": "run_2", "through_id": "run_9"},
        }
    )
    storage.conn.execute(
        "UPDATE _agent_runs SET user_work_token=? WHERE id='legacy_run'", (token,)
    )
    storage.conn.execute(
        "UPDATE learning_work SET position_json=? WHERE org_id=?",
        (position, storage.org_id),
    )
    storage.conn.commit()
    storage.migrate()
    assert _legacy_state(storage) == before
    assert (
        storage.conn.execute(
            "SELECT position_json FROM learning_work WHERE org_id=?", (storage.org_id,)
        ).fetchone()[0]
        == position
    )
    assert (
        storage.conn.execute(
            "SELECT user_work_token FROM _agent_runs WHERE id='legacy_run'"
        ).fetchone()[0]
        == token
    )
    run = storage.get_agent_run("legacy_run")
    assert run is not None
    assert "user_work_token" not in asdict(run)


def test_owner_column_failure_rolls_back_both_fields(legacy_storage):
    storage, _token = legacy_storage
    before = _legacy_state(storage)

    def deny_second_column(action, _first, second, _database, _trigger):
        return (
            sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_ALTER_TABLE and second == "learning_work"
            else sqlite3.SQLITE_OK
        )

    storage.conn.set_authorizer(deny_second_column)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            storage.migrate()
    finally:
        storage.conn.set_authorizer(None)
    assert _ownership_columns(storage) == (False, False)
    assert _legacy_state(storage) == before
    storage.migrate()
    assert _ownership_columns(storage) == (True, True)


def test_concurrent_owner_upgrade_keeps_one_live_lease(legacy_storage):
    storage, token = legacy_storage
    other = SQLiteStorage(org_id=storage.org_id, db_path=storage.db_path)
    try:
        for table, name in (
            ("_agent_runs", "user_work_token"),
            ("learning_work", "position_json"),
        ):
            if name in _columns(storage, table):
                storage.conn.execute(f"ALTER TABLE {table} DROP COLUMN {name}")
        storage.conn.commit()
        before = _legacy_state(storage)
        ready = Barrier(2)

        def migrate(target):
            ready.wait(timeout=5)
            return target.migrate()

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(migrate, target) for target in (storage, other)]
            assert all(future.result(timeout=10) for future in futures)
        assert _ownership_columns(storage) == (True, True)
        assert _legacy_state(storage) == before
        storage.fence_user_extraction("synthetic_user", token)
    finally:
        other.conn.close()
