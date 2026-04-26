"""Contract tests for profile and interaction CRUD across all storage backends."""

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.service_schemas import (
    DeleteUserInteractionRequest,
    DeleteUserProfileRequest,
    Interaction,
    ProfileTimeToLive,
    UserActionType,
    UserProfile,
)
from reflexio.server.services.storage.storage_base import BaseStorage

pytestmark = pytest.mark.integration


def _make_profile(user_id: str, profile_id: str, content: str) -> UserProfile:
    return UserProfile(
        user_id=user_id,
        profile_id=profile_id,
        content=content,
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id=f"req_{profile_id}",
        profile_time_to_live=ProfileTimeToLive.INFINITY,
        source="test",
    )


def _make_interaction(
    user_id: str,
    interaction_id: int,
    content: str,
    request_id: str,
) -> Interaction:
    return Interaction(
        interaction_id=interaction_id,
        user_id=user_id,
        request_id=request_id,
        content=content,
        created_at=int(datetime.now(UTC).timestamp()),
        user_action=UserActionType.NONE,
        user_action_description="",
        interacted_image_url="",
    )


class TestProfileCRUD:
    def test_add_and_get_profile(self, storage: BaseStorage) -> None:
        profile = _make_profile("u1", "p1", "likes sushi")
        storage.add_user_profile("u1", [profile])

        result = storage.get_user_profile("u1")
        assert len(result) == 1
        assert result[0].content == "likes sushi"
        assert result[0].profile_id == "p1"

    def test_get_nonexistent_user_returns_empty(self, storage: BaseStorage) -> None:
        assert storage.get_user_profile("nonexistent") == []

    def test_get_all_profiles_across_users(self, storage: BaseStorage) -> None:
        storage.add_user_profile("u1", [_make_profile("u1", "p1", "likes sushi")])
        storage.add_user_profile("u2", [_make_profile("u2", "p2", "likes pizza")])

        profiles = storage.get_all_profiles()
        assert len(profiles) == 2
        ids = {p.profile_id for p in profiles}
        assert ids == {"p1", "p2"}

    def test_get_all_profiles_respects_limit(self, storage: BaseStorage) -> None:
        for i in range(3):
            storage.add_user_profile(
                f"u{i}", [_make_profile(f"u{i}", f"p{i}", f"content {i}")]
            )

        profiles = storage.get_all_profiles(limit=2)
        assert len(profiles) == 2

    def test_delete_profile(self, storage: BaseStorage) -> None:
        storage.add_user_profile("u1", [_make_profile("u1", "p1", "likes sushi")])
        assert len(storage.get_user_profile("u1")) == 1

        storage.delete_user_profile(
            DeleteUserProfileRequest(user_id="u1", profile_id="p1")
        )
        assert storage.get_user_profile("u1") == []

    def test_update_profile_by_id(self, storage: BaseStorage) -> None:
        storage.add_user_profile("u1", [_make_profile("u1", "p1", "likes sushi")])

        updated = _make_profile("u1", "p1", "now prefers ramen")
        storage.update_user_profile_by_id("u1", "p1", updated)

        result = storage.get_user_profile("u1")
        assert len(result) == 1
        assert result[0].content == "now prefers ramen"

    def test_delete_all_profiles_for_user(self, storage: BaseStorage) -> None:
        storage.add_user_profile("u1", [_make_profile("u1", "p1", "likes sushi")])
        storage.add_user_profile("u2", [_make_profile("u2", "p2", "likes pizza")])

        storage.delete_all_profiles_for_user("u1")

        assert storage.get_user_profile("u1") == []
        assert len(storage.get_user_profile("u2")) == 1

    def test_delete_all_profiles(self, storage: BaseStorage) -> None:
        storage.add_user_profile("u1", [_make_profile("u1", "p1", "likes sushi")])
        storage.add_user_profile("u2", [_make_profile("u2", "p2", "likes pizza")])

        storage.delete_all_profiles()

        assert storage.get_user_profile("u1") == []
        assert storage.get_user_profile("u2") == []

    def test_count_all_profiles(self, storage: BaseStorage) -> None:
        assert storage.count_all_profiles() == 0

        storage.add_user_profile("u1", [_make_profile("u1", "p1", "likes sushi")])
        storage.add_user_profile("u2", [_make_profile("u2", "p2", "likes pizza")])
        storage.add_user_profile("u2", [_make_profile("u2", "p3", "likes ramen")])

        assert storage.count_all_profiles() == 3

        storage.delete_user_profile(
            DeleteUserProfileRequest(user_id="u2", profile_id="p2")
        )
        assert storage.count_all_profiles() == 2


class TestInteractionCRUD:
    def test_add_and_get_interaction(self, storage: BaseStorage) -> None:
        interaction = _make_interaction("u1", 1, "clicked item", "req1")
        storage.add_user_interaction("u1", interaction)

        result = storage.get_user_interaction("u1")
        assert len(result) == 1
        assert result[0].content == "clicked item"

    def test_add_interactions_bulk(self, storage: BaseStorage) -> None:
        interactions = [
            _make_interaction("u1", i, f"action {i}", f"req{i}") for i in range(1, 4)
        ]
        storage.add_user_interactions_bulk("u1", interactions)

        result = storage.get_user_interaction("u1")
        assert len(result) == 3

    def test_get_all_interactions(self, storage: BaseStorage) -> None:
        storage.add_user_interaction("u1", _make_interaction("u1", 1, "a1", "req1"))
        storage.add_user_interaction("u2", _make_interaction("u2", 2, "a2", "req2"))

        result = storage.get_all_interactions()
        assert len(result) == 2
        ids = {i.interaction_id for i in result}
        assert ids == {1, 2}

    def test_count_all_interactions(self, storage: BaseStorage) -> None:
        for i in range(1, 4):
            storage.add_user_interaction(
                "u1", _make_interaction("u1", i, f"a{i}", f"req{i}")
            )

        assert storage.count_all_interactions() == 3

    def test_delete_interaction(self, storage: BaseStorage) -> None:
        storage.add_user_interaction("u1", _make_interaction("u1", 1, "a1", "req1"))
        assert len(storage.get_user_interaction("u1")) == 1

        storage.delete_user_interaction(
            DeleteUserInteractionRequest(user_id="u1", interaction_id=1)
        )
        assert storage.get_user_interaction("u1") == []

    def test_delete_all_interactions_for_user(self, storage: BaseStorage) -> None:
        storage.add_user_interaction("u1", _make_interaction("u1", 1, "a1", "req1"))
        storage.add_user_interaction("u2", _make_interaction("u2", 2, "a2", "req2"))

        storage.delete_all_interactions_for_user("u1")

        assert storage.get_user_interaction("u1") == []
        assert len(storage.get_user_interaction("u2")) == 1

    def test_delete_all_interactions(self, storage: BaseStorage) -> None:
        storage.add_user_interaction("u1", _make_interaction("u1", 1, "a1", "req1"))
        storage.add_user_interaction("u2", _make_interaction("u2", 2, "a2", "req2"))

        storage.delete_all_interactions()

        assert storage.get_user_interaction("u1") == []
        assert storage.get_user_interaction("u2") == []

    def test_delete_oldest_interactions(self, storage: BaseStorage) -> None:
        now = int(datetime.now(UTC).timestamp())
        for i in range(1, 6):
            interaction = Interaction(
                interaction_id=i,
                user_id="u1",
                request_id=f"req{i}",
                content=f"action {i}",
                created_at=now + i,
                user_action=UserActionType.NONE,
                user_action_description="",
                interacted_image_url="",
            )
            storage.add_user_interaction("u1", interaction)

        assert storage.count_all_interactions() == 5

        deleted = storage.delete_oldest_interactions(2)
        assert deleted == 2
        assert storage.count_all_interactions() == 3
