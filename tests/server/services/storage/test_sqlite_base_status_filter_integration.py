"""Integration tests for Task 3: tombstone status filtering in SQLite storage."""

import pytest

from reflexio.models.api_schema.service_schemas import Status, UserPlaybook
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def _seed(tmp_path):
    s = SQLiteStorage(org_id="test-org", db_path=str(tmp_path / "t.db"))
    s.migrate()
    cur = UserPlaybook(user_id="u1", agent_version="v1", request_id="r1", content="cur")
    tomb = UserPlaybook(
        user_id="u1",
        agent_version="v1",
        request_id="r1",
        content="old",
        status=Status.MERGED,
        merged_into=0,
    )
    s.save_user_playbooks([cur, tomb])
    return s, cur.user_playbook_id, tomb.user_playbook_id


def test_default_reads_exclude_tombstones(tmp_path):
    s, cur_id, tomb_id = _seed(tmp_path)
    ids = {p.user_playbook_id for p in s.get_user_playbooks(user_id="u1")}
    assert cur_id in ids and tomb_id not in ids
    assert s.count_user_playbooks(user_id="u1") == 1


def test_get_by_id_excludes_tombstone_by_default_but_include_flag_returns_it(tmp_path):
    s, cur_id, tomb_id = _seed(tmp_path)
    assert s.get_user_playbook_by_id(tomb_id) is None
    got = s.get_user_playbook_by_id(tomb_id, include_tombstones=True)
    assert got is not None and got.status is Status.MERGED


def test_superseded_tombstone_also_excluded_by_default(tmp_path):
    s = SQLiteStorage(org_id="test-org", db_path=str(tmp_path / "t.db"))
    s.migrate()
    superseded = UserPlaybook(
        user_id="u2",
        agent_version="v1",
        request_id="r2",
        content="old",
        status=Status.SUPERSEDED,
        superseded_by=99,
    )
    s.save_user_playbooks([superseded])
    ids = {p.user_playbook_id for p in s.get_user_playbooks(user_id="u2")}
    assert superseded.user_playbook_id not in ids
    assert s.count_user_playbooks(user_id="u2") == 0


def test_get_agent_playbook_by_id_excludes_tombstone_by_default(tmp_path):
    """F006: get_agent_playbook_by_id hides MERGED agent playbooks unless include_tombstones=True."""
    from reflexio.models.api_schema.domain import AgentPlaybook, PlaybookStatus

    s = SQLiteStorage(org_id="test-org", db_path=str(tmp_path / "t.db"))
    s.migrate()
    [cur] = s.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="support",
                agent_version="v1",
                content="current",
                playbook_status=PlaybookStatus.PENDING,
            )
        ]
    )
    [tomb] = s.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="support",
                agent_version="v1",
                content="old",
                playbook_status=PlaybookStatus.PENDING,
                status=Status.MERGED,
                merged_into=cur.agent_playbook_id,
            )
        ]
    )

    # Default: tombstone hidden, current visible.
    assert s.get_agent_playbook_by_id(tomb.agent_playbook_id) is None
    assert s.get_agent_playbook_by_id(cur.agent_playbook_id) is not None

    # With flag: tombstone returned.
    got = s.get_agent_playbook_by_id(tomb.agent_playbook_id, include_tombstones=True)
    assert got is not None and got.status is Status.MERGED


def test_get_profile_by_id_excludes_tombstone_by_default(tmp_path):
    """get_profile_by_id should hide MERGED profiles unless include_tombstones=True."""
    import time

    from reflexio.models.api_schema.service_schemas import (
        ProfileTimeToLive,
        UserProfile,
    )

    s = SQLiteStorage(org_id="test-org", db_path=str(tmp_path / "t.db"))
    s.migrate()
    now = int(time.time())
    far_future = now + 10_000_000
    tomb_profile = UserProfile(
        profile_id="profile-tomb-1",
        user_id="u3",
        content="old profile",
        last_modified_timestamp=now,
        generated_from_request_id="req-tomb-1",
        profile_time_to_live=ProfileTimeToLive.INFINITY,
        expiration_timestamp=far_future,
        status=Status.MERGED,
        merged_into="some-other-id",
    )
    s.add_user_profile("u3", [tomb_profile])

    # Default: should return None (tombstone hidden)
    result = s.get_profile_by_id(tomb_profile.profile_id)
    assert result is None

    # With flag: should return the tombstone
    result_with_flag = s.get_profile_by_id(
        tomb_profile.profile_id, include_tombstones=True
    )
    assert result_with_flag is not None
    assert result_with_flag.status is Status.MERGED
