"""Disk-storage coverage for ``clear_user_data`` (BaseStorage default impl).

The shared ``storage`` fixture in conftest.py only exercises SQLite
(QMD-binary dependency keeps disk out of the parametrized fixture).
This file mocks out ``QMDClient`` so the default ``BaseStorage``
``clear_user_data`` composition can be verified end-to-end on the disk
backend without requiring the ``qmd`` CLI.
"""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

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

pytestmark = pytest.mark.integration


@pytest.fixture
def disk_storage() -> Generator:
    """Yield a DiskStorage instance with QMDClient stubbed out."""
    from reflexio.server.services.storage.disk_storage import DiskStorage

    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch(
            "reflexio.server.services.storage.disk_storage._base.QMDClient",
            return_value=MagicMock(),
        ),
    ):
        yield DiskStorage(org_id="contract_test_disk", base_dir=temp_dir)


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


def _seed_user(storage, user_id: str, suffix: str) -> None:
    storage.add_request(_make_request(f"req_{suffix}", user_id))
    storage.add_user_interaction(
        user_id, _make_interaction(user_id, 1000 + ord(suffix[0]), f"req_{suffix}")
    )
    storage.save_user_playbooks(
        [_make_user_playbook(2000 + ord(suffix[0]), user_id, f"req_{suffix}")]
    )
    storage.add_user_profile(user_id, [_make_profile(user_id, f"pid_{suffix}")])


class TestClearUserDataDisk:
    def test_clears_only_target_user_rows(self, disk_storage) -> None:
        _seed_user(disk_storage, "userA", "a")
        _seed_user(disk_storage, "userB", "b")
        disk_storage.save_agent_playbooks([_make_agent_playbook(9001)])

        counts = disk_storage.clear_user_data("userA")

        for key in ("interactions", "user_playbooks", "profiles", "requests"):
            assert key in counts
            assert counts[key] >= 1

        assert disk_storage.get_user_interaction("userA") == []
        assert disk_storage.get_user_profile("userA") == []
        assert disk_storage.get_user_playbooks(user_id="userA") == []
        assert disk_storage.get_request("req_a") is None

        assert len(disk_storage.get_user_interaction("userB")) == 1
        assert len(disk_storage.get_user_profile("userB")) == 1
        assert len(disk_storage.get_user_playbooks(user_id="userB")) == 1
        assert disk_storage.get_request("req_b") is not None

    def test_agent_playbooks_unchanged(self, disk_storage) -> None:
        _seed_user(disk_storage, "userA", "a")
        disk_storage.save_agent_playbooks(
            [_make_agent_playbook(9001), _make_agent_playbook(9002)]
        )
        before = disk_storage.get_agent_playbooks(limit=1000)
        assert len(before) == 2

        disk_storage.clear_user_data("userA")

        after = disk_storage.get_agent_playbooks(limit=1000)
        assert {p.agent_playbook_id for p in after} == {
            p.agent_playbook_id for p in before
        }

    def test_clear_unknown_user_is_noop(self, disk_storage) -> None:
        _seed_user(disk_storage, "userA", "a")

        counts = disk_storage.clear_user_data("does_not_exist")

        assert all(v == 0 for v in counts.values()), counts
        assert len(disk_storage.get_user_interaction("userA")) == 1
