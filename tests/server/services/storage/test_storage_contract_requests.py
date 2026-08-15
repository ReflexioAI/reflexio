"""Contract tests for request CRUD and session queries across all storage backends."""

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.service_schemas import (
    Interaction,
    Request,
    UserActionType,
)
from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.storage_base import BaseStorage

pytestmark = pytest.mark.integration


def _make_request(
    request_id: str,
    user_id: str,
    session_id: str = "s-default",
    source: str = "test",
) -> Request:
    return Request(
        request_id=request_id,
        user_id=user_id,
        created_at=int(datetime.now(UTC).timestamp()),
        source=source,
        agent_version="v1",
        session_id=session_id,
    )


class TestRequestCRUD:
    def test_add_request_rejects_legacy_source(self, storage: BaseStorage) -> None:
        request = _make_request("legacy-source-write", "u1", source="Legacy Source")

        with pytest.raises(StorageError):
            storage.add_request(request)

        assert storage.get_request(request.request_id) is None

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

    def test_get_sessions_filters_source_before_limit(
        self, storage: BaseStorage
    ) -> None:
        newer = _make_request("r-new", "u1", source="web")
        newer.created_at = 1_700_000_200
        older_match = _make_request("r-old", "u1", source="cli")
        older_match.created_at = 1_700_000_100
        storage.add_request(newer)
        storage.add_request(older_match)

        sessions = storage.get_sessions(source="cli", top_k=1)
        request_ids = [
            item.request.request_id for items in sessions.values() for item in items
        ]

        assert request_ids == ["r-old"]

    def test_get_requests_by_session(self, storage: BaseStorage) -> None:
        storage.add_request(_make_request("r1", "u1", session_id="s1"))
        storage.add_request(_make_request("r2", "u1", session_id="s1"))

        result = storage.get_requests_by_session("u1", "s1")
        assert len(result) == 2
        ids = {r.request_id for r in result}
        assert ids == {"r1", "r2"}

    def test_retrieval_experiment_assignment_uses_earliest_tagged_request(
        self, storage: BaseStorage
    ) -> None:
        first = _make_request("r1", "u1", session_id="s1")
        first.created_at = 1_700_000_000
        first.retrieval_experiment_id = "exp-1"
        first.retrieval_experiment_arm = "holdout"
        later = _make_request("r2", "u1", session_id="s1")
        later.created_at = 1_700_000_100
        later.retrieval_experiment_id = "exp-1"
        later.retrieval_experiment_arm = "treatment"
        storage.add_request(later)
        storage.add_request(first)

        assignments = storage.get_retrieval_experiment_assignments("exp-1")

        assert assignments == {("u1", "s1"): "holdout"}
        stored = storage.get_request("r1")
        assert stored is not None
        assert stored.retrieval_experiment_id == "exp-1"
        assert stored.retrieval_experiment_arm == "holdout"

    def test_retrieval_experiment_output_tokens_use_stored_non_user_counts(
        self, storage: BaseStorage
    ) -> None:
        treatment = _make_request("r1", "u1", session_id="s1")
        treatment.retrieval_experiment_id = "exp-1"
        treatment.retrieval_experiment_arm = "treatment"
        treatment_continued = _make_request("r4", "u1", session_id="s1")
        treatment_continued.retrieval_experiment_id = "exp-1"
        treatment_continued.retrieval_experiment_arm = "treatment"
        holdout = _make_request("r2", "u2", session_id="s2")
        holdout.retrieval_experiment_id = "exp-1"
        holdout.retrieval_experiment_arm = "holdout"
        other = _make_request("r3", "u3", session_id="s3")
        other.retrieval_experiment_id = "exp-2"
        other.retrieval_experiment_arm = "treatment"
        for request in (treatment, treatment_continued, holdout, other):
            storage.add_request(request)

        interactions = [
            Interaction(
                user_id="u1",
                request_id="r1",
                role="\t UsEr \r\n",
                content="ignored",
            ),
            Interaction(
                user_id="u1", request_id="r1", role="Assistant", content="hello 世界"
            ),
            Interaction(
                user_id="u1", request_id="r1", role="tool", content="tool output"
            ),
            Interaction(
                user_id="u1", request_id="r4", role="system", content="continued output"
            ),
            Interaction(
                user_id="u2", request_id="r2", role="USER", content="only input"
            ),
            Interaction(
                user_id="u3",
                request_id="r3",
                role="assistant",
                content="other experiment",
            ),
        ]
        for interaction in interactions:
            storage.add_user_interaction(interaction.user_id, interaction)

        counts = storage.get_retrieval_experiment_output_token_counts("exp-1")

        assert counts == {
            ("u1", "s1"): sum(
                interaction.token_count or 0
                for interaction in interactions
                if interaction.request_id in {"r1", "r4"}
                and interaction.role.strip().lower() != "user"
            ),
            ("u2", "s2"): 0,
        }
