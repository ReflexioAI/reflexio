"""Contract tests for AgentPlaybookMixin — run against every local storage backend."""

import pytest

from reflexio.models.api_schema.domain.enums import Status
from reflexio.models.api_schema.service_schemas import AgentPlaybook, UserPlaybook

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user_playbook(
    user_playbook_id: int,
    user_id: str,
    playbook_name: str,
    agent_version: str,
) -> UserPlaybook:
    return UserPlaybook(
        user_playbook_id=user_playbook_id,
        user_id=user_id,
        playbook_name=playbook_name,
        agent_version=agent_version,
        request_id=f"req-{user_playbook_id}",
        content=f"content-{user_playbook_id}",
        created_at=1_700_000_000 + user_playbook_id,
        source="test",
    )


def _make_agent_playbook(
    playbook_id: int,
    playbook_name: str,
    agent_version: str,
) -> AgentPlaybook:
    return AgentPlaybook(
        agent_playbook_id=playbook_id,
        playbook_name=playbook_name,
        agent_version=agent_version,
        content=f"content-{playbook_id}",
        created_at=1_700_000_000 + playbook_id,
    )


# ---------------------------------------------------------------------------
# TestUserPlaybookCRUD
# ---------------------------------------------------------------------------


class TestUserPlaybookCRUD:
    def test_save_and_get_user_playbooks(self, storage):
        rfs = [
            _make_user_playbook(1, "u1", "fb", "v1"),
            _make_user_playbook(2, "u2", "fb", "v1"),
        ]
        storage.save_user_playbooks(rfs)

        result = storage.get_user_playbooks(playbook_name="fb")
        assert len(result) == 2

    def test_count_user_playbooks(self, storage):
        rfs = [
            _make_user_playbook(1, "u1", "fb", "v1"),
            _make_user_playbook(2, "u2", "fb", "v1"),
            _make_user_playbook(3, "u3", "fb", "v1"),
        ]
        storage.save_user_playbooks(rfs)

        assert storage.count_user_playbooks(playbook_name="fb") == 3

    def test_delete_user_playbook(self, storage):
        storage.save_user_playbooks([_make_user_playbook(1, "u1", "fb", "v1")])

        saved = storage.get_user_playbooks(playbook_name="fb")
        assert len(saved) == 1

        storage.delete_user_playbook(saved[0].user_playbook_id)
        assert storage.count_user_playbooks(playbook_name="fb") == 0

    def test_delete_all_user_playbooks(self, storage):
        rfs = [
            _make_user_playbook(1, "u1", "fb", "v1"),
            _make_user_playbook(2, "u2", "fb", "v1"),
            _make_user_playbook(3, "u3", "fb", "v1"),
        ]
        storage.save_user_playbooks(rfs)

        storage.delete_all_user_playbooks()
        assert storage.count_user_playbooks() == 0

    def test_get_user_playbooks_filters_by_playbook_name(self, storage):
        storage.save_user_playbooks(
            [
                _make_user_playbook(1, "u1", "alpha", "v1"),
                _make_user_playbook(2, "u2", "alpha", "v1"),
                _make_user_playbook(3, "u3", "beta", "v1"),
            ]
        )

        alpha = storage.get_user_playbooks(playbook_name="alpha")
        beta = storage.get_user_playbooks(playbook_name="beta")

        assert len(alpha) == 2
        assert len(beta) == 1
        assert all(rf.playbook_name == "alpha" for rf in alpha)
        assert beta[0].playbook_name == "beta"

    def test_delete_all_user_playbooks_by_playbook_name(self, storage):
        storage.save_user_playbooks(
            [
                _make_user_playbook(1, "u1", "alpha", "v1"),
                _make_user_playbook(2, "u2", "alpha", "v1"),
                _make_user_playbook(3, "u3", "beta", "v1"),
            ]
        )

        storage.delete_all_user_playbooks_by_playbook_name("alpha")

        assert storage.count_user_playbooks(playbook_name="alpha") == 0
        assert storage.count_user_playbooks(playbook_name="beta") == 1


class TestGetUserPlaybooksByIds:
    """Contract tests for get_user_playbooks_by_ids (used by ReflectionService)."""

    def test_returns_only_requested_ids(self, storage):
        storage.save_user_playbooks(
            [
                _make_user_playbook(1, "u1", "fb", "v1"),
                _make_user_playbook(2, "u1", "fb", "v1"),
                _make_user_playbook(3, "u1", "fb", "v1"),
            ]
        )
        # Storage assigns ids on insert; round-trip to discover them.
        ids = sorted(
            p.user_playbook_id
            for p in storage.get_user_playbooks(user_id="u1", status_filter=[None])
        )
        target = [ids[0], ids[2]]
        result = storage.get_user_playbooks_by_ids("u1", target)
        assert {p.user_playbook_id for p in result} == set(target)

    def test_empty_ids_returns_empty_list(self, storage):
        storage.save_user_playbooks([_make_user_playbook(1, "u1", "fb", "v1")])
        assert storage.get_user_playbooks_by_ids("u1", []) == []

    def test_unknown_ids_silently_skipped(self, storage):
        storage.save_user_playbooks([_make_user_playbook(1, "u1", "fb", "v1")])
        existing_id = storage.get_user_playbooks(user_id="u1", status_filter=[None])[
            0
        ].user_playbook_id
        result = storage.get_user_playbooks_by_ids("u1", [existing_id, 99_999])
        assert {p.user_playbook_id for p in result} == {existing_id}

    def test_filters_by_user_id(self, storage):
        storage.save_user_playbooks(
            [
                _make_user_playbook(1, "u1", "fb", "v1"),
                _make_user_playbook(2, "u2", "fb", "v1"),
            ]
        )
        u2_id = storage.get_user_playbooks(user_id="u2", status_filter=[None])[
            0
        ].user_playbook_id
        # Asking u1 for u2's playbook id returns nothing.
        assert storage.get_user_playbooks_by_ids("u1", [u2_id]) == []

    def test_default_status_filter_excludes_archived(self, storage):
        storage.save_user_playbooks(
            [
                _make_user_playbook(1, "u1", "fb", "v1"),
                _make_user_playbook(2, "u1", "fb", "v1"),
            ]
        )
        ids = sorted(
            p.user_playbook_id
            for p in storage.get_user_playbooks(user_id="u1", status_filter=[None])
        )
        storage.archive_user_playbook_by_id("u1", ids[1])
        result = storage.get_user_playbooks_by_ids("u1", ids)
        assert {p.user_playbook_id for p in result} == {ids[0]}

    def test_explicit_status_filter_includes_archived(self, storage):
        storage.save_user_playbooks([_make_user_playbook(1, "u1", "fb", "v1")])
        upid = storage.get_user_playbooks(user_id="u1", status_filter=[None])[
            0
        ].user_playbook_id
        storage.archive_user_playbook_by_id("u1", upid)
        result = storage.get_user_playbooks_by_ids(
            "u1", [upid], status_filter=[Status.ARCHIVED]
        )
        assert len(result) == 1
        assert result[0].user_playbook_id == upid


class TestArchiveUserPlaybookById:
    """Contract tests for archive_user_playbook_by_id (used by ReflectionService)."""

    def test_archives_current_playbook(self, storage):
        storage.save_user_playbooks([_make_user_playbook(1, "u1", "fb", "v1")])
        assert storage.archive_user_playbook_by_id("u1", 1) is True

        # Status filter excludes archived rows.
        current = storage.get_user_playbooks(user_id="u1", status_filter=[None])
        assert current == []
        archived = storage.get_user_playbooks(
            user_id="u1", status_filter=[Status.ARCHIVED]
        )
        assert len(archived) == 1
        assert archived[0].user_playbook_id == 1

    def test_returns_false_for_missing_playbook(self, storage):
        assert storage.archive_user_playbook_by_id("u1", 999) is False

    def test_returns_false_when_already_archived(self, storage):
        storage.save_user_playbooks([_make_user_playbook(1, "u1", "fb", "v1")])
        assert storage.archive_user_playbook_by_id("u1", 1) is True
        assert storage.archive_user_playbook_by_id("u1", 1) is False

    def test_returns_false_for_wrong_user(self, storage):
        storage.save_user_playbooks([_make_user_playbook(1, "u1", "fb", "v1")])
        assert storage.archive_user_playbook_by_id("u2", 1) is False
        # u1's row untouched.
        current = storage.get_user_playbooks(user_id="u1", status_filter=[None])
        assert len(current) == 1


# ---------------------------------------------------------------------------
# TestAgentPlaybookCRUD
# ---------------------------------------------------------------------------


class TestAgentPlaybookCRUD:
    def test_save_and_get_agent_playbooks(self, storage):
        fbs = [
            _make_agent_playbook(1, "fb", "v1"),
            _make_agent_playbook(2, "fb", "v1"),
        ]
        storage.save_agent_playbooks(fbs)

        result = storage.get_agent_playbooks(playbook_name="fb")
        assert len(result) == 2

    def test_delete_agent_playbook(self, storage):
        storage.save_agent_playbooks([_make_agent_playbook(1, "fb", "v1")])

        saved = storage.get_agent_playbooks(playbook_name="fb")
        assert len(saved) == 1

        storage.delete_agent_playbook(saved[0].agent_playbook_id)
        assert storage.get_agent_playbooks(playbook_name="fb") == []

    def test_delete_all_agent_playbooks(self, storage):
        storage.save_agent_playbooks(
            [
                _make_agent_playbook(1, "fb", "v1"),
                _make_agent_playbook(2, "fb", "v1"),
            ]
        )

        storage.delete_all_agent_playbooks()
        assert storage.get_agent_playbooks() == []
