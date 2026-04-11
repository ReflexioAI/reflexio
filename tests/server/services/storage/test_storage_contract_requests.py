"""Contract tests for request CRUD and session queries across all storage backends."""

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.service_schemas import (
    Interaction,
    Request,
    UserActionType,
)
from reflexio.server.services.storage.storage_base import BaseStorage

pytestmark = pytest.mark.integration


def _make_request(
    request_id: str,
    user_id: str,
    session_id: str | None = None,
) -> Request:
    return Request(
        request_id=request_id,
        user_id=user_id,
        created_at=int(datetime.now(UTC).timestamp()),
        source="test",
        agent_version="v1",
        session_id=session_id,
    )


class TestRequestCRUD:
    def test_add_and_get_request(self, storage: BaseStorage) -> None:
        req = _make_request("r1", "u1")
        storage.add_request(req)

        result = storage.get_request("r1")
        assert result is not None
        assert result.request_id == "r1"
        assert result.user_id == "u1"
        assert result.source == "test"

    def test_get_nonexistent_request_returns_none(self, storage: BaseStorage) -> None:
        assert storage.get_request("missing") is None

    def test_delete_request(self, storage: BaseStorage) -> None:
        storage.add_request(_make_request("r1", "u1"))
        assert storage.get_request("r1") is not None

        storage.delete_request("r1")
        assert storage.get_request("r1") is None

    def test_delete_all_requests(self, storage: BaseStorage) -> None:
        storage.add_request(_make_request("r1", "u1"))
        storage.add_request(_make_request("r2", "u2"))

        storage.delete_all_requests()

        assert storage.get_request("r1") is None
        assert storage.get_request("r2") is None

    def test_delete_requests_by_ids(self, storage: BaseStorage) -> None:
        storage.add_request(_make_request("r1", "u1"))
        storage.add_request(_make_request("r2", "u1"))
        storage.add_request(_make_request("r3", "u1"))

        deleted = storage.delete_requests_by_ids(["r1", "r2"])
        assert deleted == 2
        assert storage.get_request("r1") is None
        assert storage.get_request("r2") is None
        assert storage.get_request("r3") is not None


class TestSessionQueries:
    def test_get_sessions_groups_by_session(self, storage: BaseStorage) -> None:
        req = _make_request("r1", "u1", session_id="s1")
        storage.add_request(req)

        interaction = Interaction(
            interaction_id=1,
            user_id="u1",
            request_id="r1",
            content="hello",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action=UserActionType.NONE,
            user_action_description="",
            interacted_image_url="",
        )
        storage.add_user_interaction("u1", interaction)

        sessions = storage.get_sessions(user_id="u1", session_id="s1")
        assert "s1" in sessions
        items = sessions["s1"]
        assert len(items) == 1
        assert items[0].request.request_id == "r1"
        assert items[0].session_id == "s1"

    def test_get_requests_by_session(self, storage: BaseStorage) -> None:
        storage.add_request(_make_request("r1", "u1", session_id="s1"))
        storage.add_request(_make_request("r2", "u1", session_id="s1"))

        result = storage.get_requests_by_session("u1", "s1")
        assert len(result) == 2
        ids = {r.request_id for r in result}
        assert ids == {"r1", "r2"}
