"""Contract test: Request.metadata round-trips through every storage backend.

Parametrized via the existing ``storage`` fixture (see ``conftest.py`` for the
backend rotation). Adding a new backend automatically inherits this test.
"""

import pytest

from reflexio.models.api_schema.domain.entities import Request
from reflexio.server.services.storage.storage_base import BaseStorage

pytestmark = pytest.mark.integration


def test_request_metadata_roundtrips(storage: BaseStorage) -> None:
    r = Request(
        request_id="contract-meta-r1",
        user_id="contract-meta-u1",
        session_id="contract-meta-s1",
        metadata={"reflexio_retrieval_enabled": True},
    )
    storage.add_request(r)
    got = storage.get_request("contract-meta-r1")
    assert got is not None
    assert got.metadata == {"reflexio_retrieval_enabled": True}


def test_request_metadata_empty_default(storage: BaseStorage) -> None:
    r = Request(request_id="contract-meta-r2", user_id="contract-meta-u1")
    storage.add_request(r)
    got = storage.get_request("contract-meta-r2")
    assert got is not None
    assert got.metadata == {}


def test_request_metadata_get_requests_by_session_includes_metadata(
    storage: BaseStorage,
) -> None:
    r1 = Request(
        request_id="contract-meta-r3",
        user_id="contract-meta-u1",
        session_id="contract-meta-s2",
        metadata={"reflexio_retrieval_enabled": True},
    )
    r2 = Request(
        request_id="contract-meta-r4",
        user_id="contract-meta-u1",
        session_id="contract-meta-s2",
        metadata={"reflexio_retrieval_enabled": True},
    )
    storage.add_request(r1)
    storage.add_request(r2)
    rows = storage.get_requests_by_session("contract-meta-u1", "contract-meta-s2")
    assert {r.request_id for r in rows} == {"contract-meta-r3", "contract-meta-r4"}
    for r in rows:
        assert r.metadata == {"reflexio_retrieval_enabled": True}


def test_request_metadata_nested_values_roundtrip(storage: BaseStorage) -> None:
    r = Request(
        request_id="contract-meta-r5",
        user_id="contract-meta-u1",
        metadata={"reflexio_retrieval_enabled": True, "nested": {"k": [1, 2, 3]}},
    )
    storage.add_request(r)
    got = storage.get_request("contract-meta-r5")
    assert got is not None
    assert got.metadata == {
        "reflexio_retrieval_enabled": True,
        "nested": {"k": [1, 2, 3]},
    }
