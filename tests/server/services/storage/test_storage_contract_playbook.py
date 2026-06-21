"""Contract tests for AgentPlaybookMixin — run against every local storage backend."""

import time

import pytest

from reflexio.models.api_schema.domain.enums import Status
from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    AgentPlaybookSourceWindow,
    UserPlaybook,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user_playbook(
    user_playbook_id: int,
    user_id: str,
    playbook_name: str,
    agent_version: str,
    source_interaction_ids: list[int] | None = None,
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
        source_interaction_ids=source_interaction_ids or [],
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

    def test_update_user_playbook_tags_round_trip(self, storage):
        storage.save_user_playbooks([_make_user_playbook(1, "u1", "fb", "v1")])
        saved = storage.get_user_playbooks(user_id="u1", status_filter=[None])
        assert saved[0].tags is None  # untagged until the tagging pass runs

        storage.update_user_playbook(saved[0].user_playbook_id, tags=["safety", "ux"])

        result = storage.get_user_playbooks(user_id="u1", status_filter=[None])
        assert result[0].tags == ["safety", "ux"]
        assert result[0].content == saved[0].content

        storage.update_user_playbook(saved[0].user_playbook_id, tags=[])
        assert (
            storage.get_user_playbooks(user_id="u1", status_filter=[None])[0].tags == []
        )

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


class TestAgentPlaybookSourceWindows:
    def test_source_windows_round_trip_and_legacy_ids(self, storage):
        storage.set_source_windows_for_agent_playbook(
            10,
            [
                AgentPlaybookSourceWindow(
                    user_playbook_id=2, source_interaction_ids=[20, 21]
                ),
                AgentPlaybookSourceWindow(
                    user_playbook_id=3, source_interaction_ids=[30]
                ),
            ],
        )

        assert storage.get_source_user_playbook_ids_for_agent_playbook(10) == [2, 3]
        windows = storage.get_source_windows_for_agent_playbook(10)
        assert [w.user_playbook_id for w in windows] == [2, 3]
        assert [w.source_interaction_ids for w in windows] == [[20, 21], [30]]

    def test_legacy_id_writer_creates_empty_source_windows(self, storage):
        storage.set_source_user_playbook_ids_for_agent_playbook(10, [2, 3, 2])

        assert storage.get_source_user_playbook_ids_for_agent_playbook(10) == [2, 3]
        assert storage.get_source_windows_for_agent_playbook(10) == [
            AgentPlaybookSourceWindow(user_playbook_id=2, source_interaction_ids=[]),
            AgentPlaybookSourceWindow(user_playbook_id=3, source_interaction_ids=[]),
        ]

    def test_source_windows_survive_user_playbook_delete(self, storage):
        playbook = _make_user_playbook(2, "u1", "fb", "v1", source_interaction_ids=[20])
        storage.save_user_playbooks([playbook])
        storage.set_source_windows_for_agent_playbook(
            10,
            [
                AgentPlaybookSourceWindow(
                    user_playbook_id=playbook.user_playbook_id,
                    source_interaction_ids=[20],
                )
            ],
        )

        storage.delete_user_playbooks_by_ids([playbook.user_playbook_id])

        assert storage.get_source_windows_for_agent_playbook(10) == [
            AgentPlaybookSourceWindow(
                user_playbook_id=playbook.user_playbook_id,
                source_interaction_ids=[20],
            )
        ]

    def test_batch_source_user_playbook_ids_round_trip(self, storage):
        storage.set_source_windows_for_agent_playbook(
            10,
            [
                AgentPlaybookSourceWindow(
                    user_playbook_id=2, source_interaction_ids=[20]
                ),
                AgentPlaybookSourceWindow(
                    user_playbook_id=3, source_interaction_ids=[30]
                ),
            ],
        )
        storage.set_source_windows_for_agent_playbook(
            11,
            [
                AgentPlaybookSourceWindow(
                    user_playbook_id=3, source_interaction_ids=[31]
                )
            ],
        )

        # Duplicate ids in the request are deduped; an agent playbook with no
        # source rows still appears with an empty list so callers get a
        # complete map.
        result = storage.get_source_user_playbook_ids_for_agent_playbooks(
            [10, 11, 10, 12]
        )

        assert result == {10: [2, 3], 11: [3], 12: []}

    def test_batch_source_user_playbook_ids_empty_input(self, storage):
        assert storage.get_source_user_playbook_ids_for_agent_playbooks([]) == {}


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

    def test_update_agent_playbook_tags_round_trip(self, storage):
        storage.save_agent_playbooks([_make_agent_playbook(1, "fb", "v1")])
        saved = storage.get_agent_playbooks(playbook_name="fb")
        assert saved[0].tags is None  # untagged until the tagging pass runs

        storage.update_agent_playbook(saved[0].agent_playbook_id, tags=["a", "b"])

        result = storage.get_agent_playbooks(playbook_name="fb")
        assert result[0].tags == ["a", "b"]
        assert result[0].content == saved[0].content

        storage.update_agent_playbook(saved[0].agent_playbook_id, tags=[])
        assert storage.get_agent_playbooks(playbook_name="fb")[0].tags == []

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


class TestDashboardPlaybooksTimeSeries:
    """The dashboard playbooks chart must count both playbook tables."""

    def test_playbooks_time_series_includes_agent_playbooks(self, storage):
        # total_playbooks counts user_playbooks + agent_playbooks, so the time
        # series that feeds the chart must include both — otherwise the chart
        # undercounts versus the stat card. Use a current-time created_at so the
        # rows fall inside the dashboard look-back window.
        now = int(time.time())
        storage.save_user_playbooks(
            [
                UserPlaybook(
                    user_playbook_id=1,
                    user_id="u1",
                    playbook_name="fb",
                    agent_version="v1",
                    request_id="req-1",
                    content="user pb",
                    created_at=now,
                    source="test",
                    source_interaction_ids=[],
                )
            ]
        )
        storage.save_agent_playbooks(
            [
                AgentPlaybook(
                    agent_playbook_id=1,
                    playbook_name="fb",
                    agent_version="v1",
                    content="agent pb",
                    created_at=now,
                )
            ]
        )

        stats = storage.get_dashboard_stats(days_back=30)

        assert stats["current_period"]["total_playbooks"] == 2
        # The series must contain BOTH playbooks (regression: it previously
        # queried only user_playbooks and would have length 1 here).
        assert len(stats["playbooks_time_series"]) == 2
