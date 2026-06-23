"""Contract tests for profile and interaction CRUD across all storage backends."""

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.domain.enums import Status
from reflexio.models.api_schema.service_schemas import (
    Citation,
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


class TestGetAllGeneratedProfiles:
    """Contract: get_all_generated_profiles returns every profile with a
    non-empty generated_from_request_id (any status) — the bulk form of the
    per-id read that reconstruct_profile_change_log uses for the "added" side.
    """

    def test_returns_gfr_bearing_profiles_and_matches_per_id_union(
        self, storage: BaseStorage
    ) -> None:
        uid = "u-gen-all"
        p1 = _make_profile(uid, "g-p1", "c1")  # gfr=req_g-p1
        p2 = _make_profile(uid, "g-p2", "c2")  # gfr=req_g-p2
        p_nogfr = UserProfile(
            user_id=uid,
            profile_id="g-p3",
            content="c3",
            last_modified_timestamp=int(datetime.now(UTC).timestamp()),
            generated_from_request_id="",  # no run — must be excluded
            profile_time_to_live=ProfileTimeToLive.INFINITY,
            source="test",
        )
        storage.add_user_profile(uid, [p1, p2, p_nogfr])

        got = {p.profile_id for p in storage.get_all_generated_profiles()}

        # gfr-bearing profiles present; the empty-gfr one excluded.
        assert {"g-p1", "g-p2"} <= got
        assert "g-p3" not in got

        # Equivalent to the union of the per-id reads it replaces (robust to any
        # other profiles already in this storage).
        union = {
            p.profile_id
            for r in storage.get_distinct_generated_from_request_ids()
            for p in storage.get_profiles_by_generated_from_request_id(r)
        }
        assert got == union


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

    def test_update_user_profile_tags_round_trip(self, storage: BaseStorage) -> None:
        storage.add_user_profile("u1", [_make_profile("u1", "p1", "likes sushi")])
        assert storage.get_user_profile("u1")[0].tags is None  # untagged until tagged

        storage.update_user_profile_tags("u1", "p1", ["food", "japanese"])

        result = storage.get_user_profile("u1")
        assert result[0].tags == ["food", "japanese"]
        # Tags-only update must not disturb content.
        assert result[0].content == "likes sushi"

        storage.update_user_profile_tags("u1", "p1", [])
        assert storage.get_user_profile("u1")[0].tags == []

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


class TestGetProfilesByIds:
    """Contract tests for get_profiles_by_ids (used by ReflectionService)."""

    def test_returns_only_requested_ids(self, storage: BaseStorage) -> None:
        storage.add_user_profile(
            "u1",
            [
                _make_profile("u1", "p1", "a"),
                _make_profile("u1", "p2", "b"),
                _make_profile("u1", "p3", "c"),
            ],
        )
        result = storage.get_profiles_by_ids("u1", ["p1", "p3"])
        assert {p.profile_id for p in result} == {"p1", "p3"}

    def test_empty_ids_returns_empty_list(self, storage: BaseStorage) -> None:
        storage.add_user_profile("u1", [_make_profile("u1", "p1", "a")])
        assert storage.get_profiles_by_ids("u1", []) == []

    def test_unknown_ids_silently_skipped(self, storage: BaseStorage) -> None:
        storage.add_user_profile("u1", [_make_profile("u1", "p1", "a")])
        result = storage.get_profiles_by_ids("u1", ["p1", "missing"])
        assert len(result) == 1
        assert result[0].profile_id == "p1"

    def test_filters_by_user_id(self, storage: BaseStorage) -> None:
        storage.add_user_profile("u1", [_make_profile("u1", "p1", "a")])
        storage.add_user_profile("u2", [_make_profile("u2", "p2", "b")])
        # Asking u1 for u2's profile_id returns nothing.
        assert storage.get_profiles_by_ids("u1", ["p2"]) == []

    def test_default_status_filter_excludes_archived(
        self, storage: BaseStorage
    ) -> None:
        storage.add_user_profile(
            "u1",
            [
                _make_profile("u1", "p1", "current"),
                _make_profile("u1", "p2", "to-archive"),
            ],
        )
        storage.archive_profile_by_id("u1", "p2")
        # Default status_filter = [None] → CURRENT only.
        result = storage.get_profiles_by_ids("u1", ["p1", "p2"])
        assert {p.profile_id for p in result} == {"p1"}

    def test_explicit_status_filter_includes_archived(
        self, storage: BaseStorage
    ) -> None:
        storage.add_user_profile("u1", [_make_profile("u1", "p1", "x")])
        storage.archive_profile_by_id("u1", "p1")
        result = storage.get_profiles_by_ids(
            "u1", ["p1"], status_filter=[Status.ARCHIVED]
        )
        assert len(result) == 1
        assert result[0].profile_id == "p1"


class TestArchiveProfileById:
    """Contract tests for archive_profile_by_id (used by ReflectionService)."""

    def test_archives_current_profile(self, storage: BaseStorage) -> None:
        storage.add_user_profile("u1", [_make_profile("u1", "p1", "old content")])
        assert storage.archive_profile_by_id("u1", "p1") is True

        # Status filter [None] excludes ARCHIVED rows.
        assert storage.get_user_profile("u1", status_filter=[None]) == []
        archived = storage.get_user_profile("u1", status_filter=[Status.ARCHIVED])
        assert len(archived) == 1
        assert archived[0].profile_id == "p1"

    def test_returns_false_for_missing_profile(self, storage: BaseStorage) -> None:
        assert storage.archive_profile_by_id("u1", "does-not-exist") is False

    def test_returns_false_when_already_archived(self, storage: BaseStorage) -> None:
        storage.add_user_profile("u1", [_make_profile("u1", "p1", "old content")])
        assert storage.archive_profile_by_id("u1", "p1") is True
        # Second call: row exists but status != None.
        assert storage.archive_profile_by_id("u1", "p1") is False

    def test_returns_false_for_wrong_user(self, storage: BaseStorage) -> None:
        storage.add_user_profile("u1", [_make_profile("u1", "p1", "old content")])
        assert storage.archive_profile_by_id("u2", "p1") is False
        # u1's row is untouched.
        current = storage.get_user_profile("u1", status_filter=[None])
        assert len(current) == 1


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

    def test_interaction_citations_round_trip(self, storage: BaseStorage) -> None:
        """Citations attached to an Interaction must round-trip through storage.

        Covers both INSERT branches (with and without an assigned interaction_id)
        and the JSON serialization in ``_profiles.py`` plus the deserialization
        in ``_row_to_interaction``.
        """
        interaction = _make_interaction("u1", 1, "answered with cite", "req1")
        interaction.citations = [
            Citation(
                kind="playbook",
                real_id="pb_42",
                tag="r1-ab12",
                title="rule X",
            ),
            Citation(
                kind="profile",
                real_id="prof_7",
                tag="p1-cd34",
                title="user role",
            ),
        ]
        storage.add_user_interaction("u1", interaction)

        result = storage.get_user_interaction("u1")
        assert len(result) == 1
        assert result[0].citations == interaction.citations

    def test_interaction_with_no_citations_round_trips_as_empty(
        self, storage: BaseStorage
    ) -> None:
        """An Interaction without citations comes back with ``citations=[]``.

        Pre-migration rows have ``citations IS NULL`` in SQLite; verify
        ``_row_to_interaction`` parses that as an empty list rather than
        raising or surfacing ``None``.
        """
        interaction = _make_interaction("u1", 1, "no citations", "req1")
        storage.add_user_interaction("u1", interaction)

        result = storage.get_user_interaction("u1")
        assert len(result) == 1
        assert result[0].citations == []
