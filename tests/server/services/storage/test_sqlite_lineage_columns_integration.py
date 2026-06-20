import time

import pytest

from reflexio.models.api_schema.domain import AgentPlaybook, PlaybookStatus
from reflexio.models.api_schema.domain.entities import (
    ProfileTimeToLive,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.api_schema.domain.enums import Status
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def test_user_playbook_pointers_roundtrip(tmp_path):
    s = SQLiteStorage(org_id="test-org", db_path=str(tmp_path / "t.db"))
    s.migrate()
    pb = UserPlaybook(user_id="u1", agent_version="v1", request_id="r1",
                      content="c", merged_into=42, superseded_by=None)
    s.save_user_playbooks([pb])
    got = s.get_user_playbook_by_id(pb.user_playbook_id)
    assert got is not None and got.merged_into == 42 and got.superseded_by is None


def test_agent_playbook_pointers_roundtrip(tmp_path):
    """F007: agent_playbook merged_into/superseded_by survive save -> _row_to_agent_playbook."""
    s = SQLiteStorage(org_id="test-org", db_path=str(tmp_path / "t.db"))
    s.migrate()
    [ap] = s.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="support",
                agent_version="v1",
                content="c",
                playbook_status=PlaybookStatus.PENDING,
                status=Status.SUPERSEDED,
                merged_into=7,
                superseded_by=11,
            )
        ]
    )
    got = s.get_agent_playbook_by_id(ap.agent_playbook_id, include_tombstones=True)
    assert got is not None and got.merged_into == 7 and got.superseded_by == 11


def test_profile_pointers_roundtrip(tmp_path):
    """F007: profile merged_into/superseded_by (TEXT) survive save -> _row_to_profile."""
    s = SQLiteStorage(org_id="test-org", db_path=str(tmp_path / "t.db"))
    s.migrate()
    now = int(time.time())
    profile = UserProfile(
        profile_id="profile-ptr-1",
        user_id="u1",
        content="p",
        last_modified_timestamp=now,
        generated_from_request_id="req-ptr-1",
        profile_time_to_live=ProfileTimeToLive.INFINITY,
        expiration_timestamp=now + 10_000_000,
        status=Status.SUPERSEDED,
        merged_into="survivor-profile-id",
        superseded_by="successor-profile-id",
    )
    s.add_user_profile("u1", [profile])
    got = s.get_profile_by_id(profile.profile_id, include_tombstones=True)
    assert got is not None
    assert got.merged_into == "survivor-profile-id"
    assert got.superseded_by == "successor-profile-id"
