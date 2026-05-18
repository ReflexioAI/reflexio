"""Contract tests for clear_user_data — per-``user_id`` row deletion.

The method is the data-isolation primitive used by paired-protocol
harnesses (e.g. SWE-bench) to share a single backend across parallel
tasks without one task's clear-all nuking another in-flight task's
rows. These tests pin the cross-entity invariants every storage backend
must satisfy.
"""

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    Interaction,
    ProfileTimeToLive,
    Request,
    UserActionType,
    UserPlaybook,
    UserProfile,
)
from reflexio.server.services.storage.storage_base import BaseStorage

pytestmark = pytest.mark.integration


def _make_request(request_id: str, user_id: str) -> Request:
    return Request(
        request_id=request_id,
        user_id=user_id,
        created_at=int(datetime.now(UTC).timestamp()),
        source="test",
        agent_version="v1",
        session_id=f"sess_{request_id}",
    )


def _make_interaction(
    user_id: str, interaction_id: int, request_id: str
) -> Interaction:
    return Interaction(
        interaction_id=interaction_id,
        user_id=user_id,
        request_id=request_id,
        content=f"content-{interaction_id}",
        created_at=int(datetime.now(UTC).timestamp()),
        user_action=UserActionType.NONE,
        user_action_description="",
        interacted_image_url="",
    )


def _make_profile(user_id: str, profile_id: str) -> UserProfile:
    return UserProfile(
        user_id=user_id,
        profile_id=profile_id,
        content=f"prefs-{profile_id}",
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id=f"req_{profile_id}",
        profile_time_to_live=ProfileTimeToLive.INFINITY,
        source="test",
    )


def _make_user_playbook(
    user_playbook_id: int, user_id: str, request_id: str
) -> UserPlaybook:
    return UserPlaybook(
        user_playbook_id=user_playbook_id,
        user_id=user_id,
        playbook_name="fb",
        agent_version="v1",
        request_id=request_id,
        content=f"playbook-{user_playbook_id}",
        created_at=1_700_000_000 + user_playbook_id,
        source="test",
        source_interaction_ids=[],
    )


def _make_agent_playbook(agent_playbook_id: int) -> AgentPlaybook:
    return AgentPlaybook(
        agent_playbook_id=agent_playbook_id,
        playbook_name="fb",
        agent_version="v1",
        content=f"agent-pb-{agent_playbook_id}",
        created_at=1_700_000_000 + agent_playbook_id,
    )


def _seed_user(storage: BaseStorage, user_id: str, suffix: str) -> None:
    """Seed one user's worth of cross-entity data.

    Adds a request, an interaction, a user playbook, and a profile for
    ``user_id``. ``suffix`` namespaces the row keys so two users can be
    seeded into the same backend without primary-key collisions.
    """
    storage.add_request(_make_request(f"req_{suffix}", user_id))
    storage.add_user_interaction(
        user_id, _make_interaction(user_id, 1000 + ord(suffix[0]), f"req_{suffix}")
    )
    storage.save_user_playbooks(
        [_make_user_playbook(2000 + ord(suffix[0]), user_id, f"req_{suffix}")]
    )
    storage.add_user_profile(user_id, [_make_profile(user_id, f"pid_{suffix}")])


class TestClearUserData:
    def test_clears_only_target_user_rows(self, storage: BaseStorage) -> None:
        """Sibling user data must survive clear_user_data on another user."""
        _seed_user(storage, "userA", "a")
        _seed_user(storage, "userB", "b")
        # Shared cross-project agent playbook — must survive.
        storage.save_agent_playbooks([_make_agent_playbook(9001)])

        counts = storage.clear_user_data("userA")

        # All four user-scoped entity types report at least one deletion.
        for key in ("interactions", "user_playbooks", "profiles", "requests"):
            assert key in counts, f"missing {key} in deleted_counts"
            assert counts[key] >= 1, (
                f"expected {key} deletion >= 1 for userA, got {counts[key]}"
            )

        # userA fully wiped.
        assert storage.get_user_interaction("userA") == []
        assert storage.get_user_profile("userA") == []
        assert storage.get_user_playbooks(user_id="userA") == []
        assert storage.get_request("req_a") is None

        # userB completely untouched.
        assert len(storage.get_user_interaction("userB")) == 1
        assert len(storage.get_user_profile("userB")) == 1
        assert len(storage.get_user_playbooks(user_id="userB")) == 1
        assert storage.get_request("req_b") is not None

    def test_agent_playbooks_unchanged(self, storage: BaseStorage) -> None:
        """agent_playbooks have no user_id column and must never be touched."""
        _seed_user(storage, "userA", "a")
        storage.save_agent_playbooks(
            [_make_agent_playbook(9001), _make_agent_playbook(9002)]
        )
        before = storage.get_agent_playbooks(limit=1000)
        assert len(before) == 2

        storage.clear_user_data("userA")

        after = storage.get_agent_playbooks(limit=1000)
        assert len(after) == len(before)
        assert {p.agent_playbook_id for p in after} == {
            p.agent_playbook_id for p in before
        }

    def test_clear_unknown_user_is_noop(self, storage: BaseStorage) -> None:
        """Clearing an unknown user_id returns zero counts and does not raise."""
        _seed_user(storage, "userA", "a")

        counts = storage.clear_user_data("does_not_exist")

        # No rows removed for the unknown user.
        assert all(v == 0 for v in counts.values()), counts
        # userA still intact.
        assert len(storage.get_user_interaction("userA")) == 1
        assert len(storage.get_user_profile("userA")) == 1
        assert len(storage.get_user_playbooks(user_id="userA")) == 1
        assert storage.get_request("req_a") is not None

    def test_returned_counts_match_seeded_rows(self, storage: BaseStorage) -> None:
        """Per-entity counts must reflect actual seeded row counts for the user."""
        # Seed userA with two of each.
        storage.add_request(_make_request("ra1", "userA"))
        storage.add_request(_make_request("ra2", "userA"))
        storage.add_user_interaction("userA", _make_interaction("userA", 5001, "ra1"))
        storage.add_user_interaction("userA", _make_interaction("userA", 5002, "ra2"))
        storage.save_user_playbooks(
            [
                _make_user_playbook(6001, "userA", "ra1"),
                _make_user_playbook(6002, "userA", "ra2"),
            ]
        )
        storage.add_user_profile(
            "userA",
            [
                _make_profile("userA", "pa1"),
                _make_profile("userA", "pa2"),
            ],
        )

        counts = storage.clear_user_data("userA")

        assert counts["interactions"] == 2
        assert counts["user_playbooks"] == 2
        assert counts["profiles"] == 2
        assert counts["requests"] == 2
