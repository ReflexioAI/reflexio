"""SQLite storage tests for the per-request metadata field added in F2."""

import pytest

from reflexio.models.api_schema.domain.entities import Request
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage


@pytest.fixture
def storage(tmp_path):
    db_path = tmp_path / "test.db"
    return SQLiteStorage(org_id="0", db_path=str(db_path))


def test_sqlite_persists_request_metadata(storage):
    r = Request(
        request_id="r1",
        user_id="u1",
        session_id="s1",
        metadata={"reflexio_retrieval_enabled": True},
    )
    storage.add_request(r)
    got = storage.get_request("r1")
    assert got is not None
    assert got.metadata == {"reflexio_retrieval_enabled": True}


def test_sqlite_default_empty_metadata(storage):
    r = Request(request_id="r2", user_id="u1", session_id="s1")
    storage.add_request(r)
    got = storage.get_request("r2")
    assert got is not None
    assert got.metadata == {}


def test_sqlite_metadata_accepts_nested_values(storage):
    r = Request(
        request_id="r3",
        user_id="u1",
        metadata={"reflexio_retrieval_enabled": False, "tags": ["a", "b"]},
    )
    storage.add_request(r)
    got = storage.get_request("r3")
    assert got is not None
    assert got.metadata["tags"] == ["a", "b"]


def test_sqlite_get_requests_by_session_carries_metadata(storage):
    """End-to-end check that the bulk-fetch read path round-trips metadata."""
    r1 = Request(
        request_id="r4",
        user_id="u1",
        session_id="s2",
        metadata={"reflexio_retrieval_enabled": True},
    )
    r2 = Request(
        request_id="r5",
        user_id="u1",
        session_id="s2",
        metadata={"reflexio_retrieval_enabled": True},
    )
    storage.add_request(r1)
    storage.add_request(r2)
    rows = storage.get_requests_by_session("u1", "s2")
    assert {r.request_id for r in rows} == {"r4", "r5"}
    for r in rows:
        assert r.metadata == {"reflexio_retrieval_enabled": True}
