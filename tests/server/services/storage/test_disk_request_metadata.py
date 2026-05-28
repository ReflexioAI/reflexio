"""Disk storage tests for the per-request metadata field added in F2.

This file primarily exists as a regression guard: DiskStorage uses Pydantic's
native serialization (``model_dump`` on write, ``model_validate`` on read)
via the YAML-frontmatter file format, so adding fields to ``Request`` should
round-trip automatically. The tests below lock that behavior in so future
refactors of the disk-storage serialization path don't silently drop new
fields.

The ``DiskStorage`` constructor depends on the ``qmd`` CLI for search indexing,
which isn't available in unit-test environments. We patch the ``QMDClient``
constructor with a ``MagicMock`` — same pattern as
``test_disk_storage_reflection_methods.py``.
"""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.domain.entities import Request
from reflexio.server.services.storage.disk_storage import DiskStorage

pytestmark = pytest.mark.integration


@pytest.fixture
def storage() -> Generator[DiskStorage]:
    """Yield a DiskStorage instance with QMDClient stubbed out.

    Mirrors the fixture in ``test_disk_storage_reflection_methods.py`` —
    the ``qmd`` CLI dependency keeps disk storage out of the parametrized
    contract fixture, so per-file tests stub QMD directly.
    """
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch(
            "reflexio.server.services.storage.disk_storage._base.QMDClient",
            return_value=MagicMock(),
        ),
    ):
        yield DiskStorage(org_id="metadata_roundtrip", base_dir=temp_dir)


def test_disk_storage_persists_request_metadata(storage: DiskStorage) -> None:
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


def test_disk_storage_default_empty_metadata(storage: DiskStorage) -> None:
    r = Request(request_id="r2", user_id="u1")
    storage.add_request(r)
    got = storage.get_request("r2")
    assert got is not None
    assert got.metadata == {}


def test_disk_storage_metadata_nested_values(storage: DiskStorage) -> None:
    r = Request(
        request_id="r3",
        user_id="u1",
        metadata={"reflexio_retrieval_enabled": False, "tags": ["a", "b"]},
    )
    storage.add_request(r)
    got = storage.get_request("r3")
    assert got is not None
    assert got.metadata["tags"] == ["a", "b"]
    assert got.metadata["reflexio_retrieval_enabled"] is False


def test_disk_storage_get_requests_by_session_carries_metadata(
    storage: DiskStorage,
) -> None:
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
