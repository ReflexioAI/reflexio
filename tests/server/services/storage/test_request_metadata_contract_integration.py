"""Contract test: Request.metadata round-trips through the SQLite storage backend."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain.entities import (
    Interaction,
    Request,
    UserActionType,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import BaseStorage

pytestmark = pytest.mark.integration


@pytest.fixture
def storage() -> Generator[BaseStorage]:
    """Yield a fresh, isolated SQLite storage instance."""
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
    ):
        yield SQLiteStorage(
            org_id="contract_test_request_metadata",
            db_path=f"{temp_dir}/reflexio.db",
        )


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
    r = Request(
        request_id="contract-meta-r2",
        user_id="contract-meta-u1",
        session_id="test_session",
    )
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
        session_id="test_session",
        metadata={"reflexio_retrieval_enabled": True, "nested": {"k": [1, 2, 3]}},
    )
    storage.add_request(r)
    got = storage.get_request("contract-meta-r5")
    assert got is not None
    assert got.metadata == {
        "reflexio_retrieval_enabled": True,
        "nested": {"k": [1, 2, 3]},
    }


def test_evaluation_only_requests_are_excluded_from_learning_windows(
    storage: BaseStorage,
) -> None:
    normal = Request(
        request_id="contract-eval-normal",
        user_id="contract-eval-u1",
        created_at=1000,
        source="api",
        agent_version="v1",
        session_id="contract-eval-s1",
    )
    evaluation_only = Request(
        request_id="contract-eval-only",
        user_id="contract-eval-u1",
        created_at=1001,
        source="api",
        agent_version="v1",
        session_id="contract-eval-s1",
        evaluation_only=True,
    )
    storage.add_request(normal)
    storage.add_request(evaluation_only)
    storage.add_user_interactions_bulk(
        "contract-eval-u1",
        [
            Interaction(
                user_id="contract-eval-u1",
                request_id=normal.request_id,
                created_at=1000,
                content="Normal interaction can teach extraction",
                user_action=UserActionType.NONE,
            ),
            Interaction(
                user_id="contract-eval-u1",
                request_id=evaluation_only.request_id,
                created_at=1001,
                content="Evaluation-only interaction must not teach extraction",
                user_action=UserActionType.NONE,
            ),
        ],
    )

    got = storage.get_request(evaluation_only.request_id)
    assert got is not None
    assert got.evaluation_only is True
    session_requests = storage.get_requests_by_session(
        "contract-eval-u1", "contract-eval-s1"
    )
    assert {request.request_id for request in session_requests} == {
        normal.request_id,
        evaluation_only.request_id,
    }

    _, new_groups = storage.get_operation_state_with_new_request_interaction(
        "contract-eval-state", "contract-eval-u1", sources=["api"]
    )
    grouped, flat = storage.get_last_k_interactions_grouped(
        "contract-eval-u1", 10, sources=["api"]
    )

    assert {group.request.request_id for group in new_groups} == {normal.request_id}
    assert {group.request.request_id for group in grouped} == {normal.request_id}
    assert {interaction.request_id for interaction in flat} == {normal.request_id}
