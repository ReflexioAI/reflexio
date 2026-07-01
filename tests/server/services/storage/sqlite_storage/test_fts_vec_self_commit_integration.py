"""Characterization test: ``_fts_upsert`` / ``_vec_upsert`` self-commit.

Pins the FtsVec day-one guard a later verbatim move (peeling
``SQLiteFtsVecMixin`` out of ``sqlite_storage/_base.py``) must preserve: these
helpers commit internally, so a write is immediately durable and visible from a
SECOND connection with no outer commit from the caller.

Falsifiability: if the move drops the helper's internal ``self.conn.commit()``,
the write stays in the first connection's uncommitted transaction and a fresh
second connection sees zero rows — the "visible" assertions below fail.
"""

import sqlite3

import pytest

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration

_ROWID = 987654


def _store(tmp_path) -> tuple[SQLiteStorage, str]:
    db_path = str(tmp_path / "ftsvec.db")
    return SQLiteStorage(org_id="org-ftsvec", db_path=db_path), db_path


def test_fts_upsert_is_visible_from_second_connection(tmp_path) -> None:
    storage, db_path = _store(tmp_path)

    # No outer transaction/commit from the test — the helper must self-commit.
    storage._fts_upsert(  # type: ignore[attr-defined]
        "user_playbooks_fts", _ROWID, search_text="self committing row"
    )

    probe = sqlite3.connect(db_path)
    try:
        found = probe.execute(
            "SELECT rowid FROM user_playbooks_fts WHERE rowid = ?", (_ROWID,)
        ).fetchall()
    finally:
        probe.close()

    assert len(found) == 1, (
        "_fts_upsert must self-commit so the row is visible from a second connection"
    )


def test_vec_upsert_is_visible_from_second_connection(tmp_path) -> None:
    storage, db_path = _store(tmp_path)
    if not storage._has_sqlite_vec:  # type: ignore[attr-defined]
        pytest.skip("sqlite-vec extension not loaded in this environment")

    embedding = [0.0] * storage.embedding_dimensions
    storage._vec_upsert("user_playbooks_vec", _ROWID, embedding)  # type: ignore[attr-defined]

    # A vec0 virtual table can only be queried with the extension loaded.
    import sqlite_vec  # type: ignore[import-untyped]

    probe = sqlite3.connect(db_path)
    try:
        probe.enable_load_extension(True)
        sqlite_vec.load(probe)
        probe.enable_load_extension(False)
        found = probe.execute(
            "SELECT rowid FROM user_playbooks_vec WHERE rowid = ?", (_ROWID,)
        ).fetchall()
    finally:
        probe.close()

    assert len(found) == 1, (
        "_vec_upsert must self-commit so the row is visible from a second connection"
    )
