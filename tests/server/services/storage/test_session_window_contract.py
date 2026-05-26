"""Contract tests for get_session_ids_in_window across storage backends.

This file defines its own parametrized ``storage`` fixture (shadowing the
conftest one) so the new method is exercised against BOTH SQLite and
Disk backends without enrolling pre-existing contract tests against the
Disk backend (which currently has unrelated failures in retention and
stall_state).
"""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.internal_schema import SessionDescriptor
from reflexio.models.api_schema.service_schemas import (
    Request,
)
from reflexio.server.services.storage.storage_base import BaseStorage

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", "disk"])
def storage(request: pytest.FixtureRequest) -> Generator[BaseStorage]:
    """Yield a fresh, isolated storage instance for each backend."""
    backend = request.param

    with tempfile.TemporaryDirectory() as temp_dir:
        if backend == "sqlite":
            from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

            with patch.object(
                SQLiteStorage, "_get_embedding", return_value=[0.0] * 512
            ):
                yield SQLiteStorage(
                    org_id="contract_test_session_window",
                    db_path=f"{temp_dir}/reflexio.db",
                )
        elif backend == "disk":
            from reflexio.server.services.storage.disk_storage import DiskStorage

            yield DiskStorage(
                org_id="contract_test_session_window", base_dir=temp_dir
            )


def _seed_request(storage: BaseStorage, user_id: str, session_id: str, ts: int) -> str:
    """Insert one Request at ``ts`` and return its request_id."""
    req = Request(
        request_id=f"req_{session_id}_{ts}",
        user_id=user_id,
        created_at=ts,
        source="test",
        agent_version="v1",
        session_id=session_id,
    )
    storage.add_request(req)
    return req.request_id


def test_returns_distinct_sessions_in_window(storage: BaseStorage) -> None:
    _seed_request(storage, "u1", "s1", ts=1000)
    _seed_request(storage, "u1", "s1", ts=1005)
    _seed_request(storage, "u2", "s2", ts=2000)
    _seed_request(storage, "u3", "s3", ts=9999)

    out = storage.get_session_ids_in_window(from_ts=500, to_ts=5000)

    assert isinstance(out, list)
    assert all(isinstance(d, SessionDescriptor) for d in out)
    session_ids = {d.session_id for d in out}
    assert session_ids == {"s1", "s2"}
    assert sum(1 for d in out if d.session_id == "s1") == 1


def test_empty_window_returns_empty_list(storage: BaseStorage) -> None:
    out = storage.get_session_ids_in_window(from_ts=0, to_ts=1)
    assert out == []


def test_window_boundaries_are_inclusive(storage: BaseStorage) -> None:
    _seed_request(storage, "u1", "edge_low", ts=100)
    _seed_request(storage, "u1", "edge_high", ts=200)
    out = storage.get_session_ids_in_window(from_ts=100, to_ts=200)
    assert {d.session_id for d in out} == {"edge_low", "edge_high"}
