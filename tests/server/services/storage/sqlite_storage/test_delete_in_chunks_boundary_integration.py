"""Characterization test: ``_delete_in_chunks`` batches its ``IN (...)`` deletes.

Pins the chunk-boundary behavior a later verbatim move (peeling
``SQLiteDeletionMixin`` out of ``sqlite_storage/_base.py``) must preserve.

Modern sqlite raises ``SQLITE_LIMIT_VARIABLE_NUMBER`` to 32766, so a naive
single-statement ``IN`` over ~1200 ids would NOT overflow here and the chunk
guard would look vacuous. To keep the test falsifiable across builds, the
fixture caps the connection's variable limit to 999 (the classic old-build
default). Chunking uses ``RETENTION_DELETE_CHUNK`` (500) params per statement,
so it stays under the cap while any un-chunked collapse would exceed it.

Falsifiability: if a later move collapses the batching into a single
``DELETE ... WHERE col IN (<all ids>)``, the >999-id cases raise
``sqlite3.OperationalError: too many SQL variables`` and the "keep" assertion
catches an over-broad delete. A stub that no-ops fails the deletion assertion.
"""

import sqlite3

import pytest

from reflexio.server.services.storage.retention_mixin import RETENTION_DELETE_CHUNK
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration

# Classic pre-3.32 default; below any un-chunked collapse of the id lists used
# here, but above the per-chunk parameter count (RETENTION_DELETE_CHUNK).
_OLD_SQLITE_VARIABLE_LIMIT = 999


def _store(tmp_path) -> SQLiteStorage:
    storage = SQLiteStorage(org_id="org-chunk", db_path=str(tmp_path / "chunk.db"))
    storage.conn.setlimit(
        sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, _OLD_SQLITE_VARIABLE_LIMIT
    )
    return storage


def _make_scratch_table(storage: SQLiteStorage) -> None:
    storage.conn.execute(
        "CREATE TABLE chunk_scratch (id INTEGER PRIMARY KEY, tag TEXT)"
    )
    storage.conn.commit()


def _insert_ids(storage: SQLiteStorage, ids: list[int], tag: str) -> None:
    storage.conn.executemany(
        "INSERT INTO chunk_scratch (id, tag) VALUES (?, ?)",
        [(i, tag) for i in ids],
    )
    storage.conn.commit()


@pytest.mark.parametrize(
    "delete_count",
    [
        RETENTION_DELETE_CHUNK - 1,  # under one chunk
        RETENTION_DELETE_CHUNK,  # exactly one chunk boundary
        RETENTION_DELETE_CHUNK + 1,  # one past the boundary (2 chunks)
        RETENTION_DELETE_CHUNK * 2 + 200,  # 3 chunks, > the capped variable limit
    ],
)
def test_delete_in_chunks_deletes_all_across_chunk_boundaries(
    tmp_path, delete_count: int
) -> None:
    storage = _store(tmp_path)
    _make_scratch_table(storage)

    # Ids to delete plus a disjoint "keep" set that must survive.
    delete_ids = list(range(1, delete_count + 1))
    keep_ids = list(range(1_000_000, 1_000_000 + 50))
    _insert_ids(storage, delete_ids, "del")
    _insert_ids(storage, keep_ids, "keep")

    # Must not raise "too many SQL variables" even for id lists > the cap.
    storage._delete_in_chunks("chunk_scratch", "id", delete_ids)  # type: ignore[attr-defined]

    remaining = storage.conn.execute("SELECT id FROM chunk_scratch").fetchall()
    remaining_ids = {row["id"] for row in remaining}

    assert remaining_ids == set(keep_ids), (
        "all target ids must be deleted and only the keep set may remain"
    )


def test_delete_in_chunks_empty_values_is_noop(tmp_path) -> None:
    storage = _store(tmp_path)
    _make_scratch_table(storage)
    _insert_ids(storage, [1, 2, 3], "keep")

    storage._delete_in_chunks("chunk_scratch", "id", [])  # type: ignore[attr-defined]

    count = storage.conn.execute("SELECT COUNT(*) AS c FROM chunk_scratch").fetchone()
    assert count["c"] == 3


def test_delete_in_chunks_survives_where_single_statement_overflows(
    tmp_path,
) -> None:
    """A single un-chunked ``IN`` over these ids overflows the cap; chunking must not."""
    storage = _store(tmp_path)
    _make_scratch_table(storage)
    ids = list(range(1, 1500))
    _insert_ids(storage, ids, "del")

    # Sanity: a single-statement IN over the same ids DOES overflow the cap, so
    # the chunked helper's success below is a real guard, not a vacuous pass.
    placeholders = ",".join("?" for _ in ids)
    with pytest.raises(sqlite3.OperationalError, match="too many SQL variables"):
        storage.conn.execute(
            f"DELETE FROM chunk_scratch WHERE id IN ({placeholders})",  # noqa: S608
            ids,
        )
    storage.conn.rollback()

    storage._delete_in_chunks("chunk_scratch", "id", ids)  # type: ignore[attr-defined]
    count = storage.conn.execute("SELECT COUNT(*) AS c FROM chunk_scratch").fetchone()
    assert count["c"] == 0
