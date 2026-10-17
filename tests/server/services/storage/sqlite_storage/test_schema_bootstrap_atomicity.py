"""A failed base-schema bootstrap must preserve data and permit a clean retry."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage, _base


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
        broken.setattr(_base, "_DDL", _base._DDL + "\nINVALID SQL;")
        with pytest.raises(sqlite3.OperationalError, match="syntax error"):
            SQLiteStorage(org_id="bootstrap-test", db_path=str(database))

    with closing(sqlite3.connect(database)) as observer:
        tables = observer.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
        assert tables == ([("preserved",)] if existing_data else [])
        if existing_data:
            assert observer.execute("SELECT value FROM preserved").fetchall() == [
                ("keep",)
            ]

    storage = SQLiteStorage(org_id="bootstrap-test", db_path=str(database))
    try:
        assert storage.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
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
