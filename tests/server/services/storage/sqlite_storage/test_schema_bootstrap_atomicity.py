"""A failed base-schema bootstrap must preserve data and permit a clean retry."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage, _ddl
from reflexio.server.services.storage.sqlite_storage.migrations import _steps


@pytest.mark.parametrize("existing_data", [False, True])
def test_failed_base_schema_is_rolled_back_and_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing_data: bool
) -> None:
    database = tmp_path / "bootstrap.db"
    if existing_data:
        with closing(sqlite3.connect(database)) as connection:
            connection.execute("CREATE TABLE preserved (value TEXT)")
            connection.execute("INSERT INTO preserved VALUES ('keep')")
            connection.commit()

    with monkeypatch.context() as broken:
        broken.setattr(_steps, "_DDL", _ddl._DDL + "\nINVALID SQL;")
        with pytest.raises(sqlite3.OperationalError, match="syntax error"):
            SQLiteStorage(org_id="bootstrap-test", db_path=str(database))

    with closing(sqlite3.connect(database)) as observer:
        tables = observer.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
        assert tables == (
            [("preserved",)] if existing_data else [("_learning_policy_authority",)]
        )
        if not existing_data:
            retained = json.loads(
                observer.execute(
                    "SELECT authority FROM _learning_policy_authority"
                ).fetchone()[0]
            )
            assert retained["phase"] == "sealed" and retained["bootstrap"] is True
        if existing_data:
            assert observer.execute("SELECT value FROM preserved").fetchall() == [
                ("keep",)
            ]

    storage = SQLiteStorage(org_id="bootstrap-test", db_path=str(database))
    try:
        assert storage.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        if not existing_data:
            current = storage.read_learning_policy_authority()
            assert current is not None
            assert current.authority_nonce == retained["authority_nonce"]
            assert current.initialization_nonce == retained["initialization_nonce"]
            assert current.phase == "open" and not current.bootstrap
        assert (
            storage.conn.execute(
                "SELECT name FROM sqlite_master WHERE name = 'interactions'"
            ).fetchone()
            is not None
        )
        if existing_data:
            assert (
                storage.conn.execute("SELECT value FROM preserved").fetchone()[0]
                == "keep"
            )
    finally:
        storage.conn.close()
