"""Disk-storage coverage for the four methods added for ReflectionService.

The shared ``storage`` fixture in conftest.py only exercises SQLite
(QMD-binary dependency keeps disk out of the parametrized fixture).
This file mocks out ``QMDClient`` so the new disk-storage methods can
be exercised without requiring the ``qmd`` CLI.
"""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.domain.enums import (
    ProfileTimeToLive,
    Status,
)
from reflexio.models.api_schema.service_schemas import (
    AgentPlaybookSourceWindow,
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
        yield DiskStorage(org_id="contract_test", base_dir=temp_dir)


def _make_profile(user_id: str, profile_id: str, content: str) -> UserProfile:
    return UserProfile(
        profile_id=profile_id,
        user_id=user_id,
        content=content,
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id="seed_req",
        profile_time_to_live=ProfileTimeToLive.INFINITY,
        custom_features={},
        source="seed",
    )


def _make_playbook(
    user_playbook_id: int, user_id: str, content: str = "rule"
) -> UserPlaybook:
    return UserPlaybook(
        user_playbook_id=user_playbook_id,
        user_id=user_id,
        agent_version="v1",
        request_id=f"seed_{user_playbook_id}",
        playbook_name="fb",
        content=content,
        trigger="when X",
        rationale="because Y",
        source="seed",
    )


# ---------------------------------------------------------------------------
# get_profiles_by_ids
# ---------------------------------------------------------------------------


class TestDiskGetProfilesByIds:
    def test_returns_only_requested_ids(self, disk_storage):
        disk_storage.add_user_profile(
            "u1",
            [
                _make_profile("u1", "p1", "a"),
                _make_profile("u1", "p2", "b"),
                _make_profile("u1", "p3", "c"),
            ],
        )
        result = disk_storage.get_profiles_by_ids("u1", ["p1", "p3"])
        assert {p.profile_id for p in result} == {"p1", "p3"}

    def test_empty_ids_returns_empty_list(self, disk_storage):
        disk_storage.add_user_profile("u1", [_make_profile("u1", "p1", "a")])
        assert disk_storage.get_profiles_by_ids("u1", []) == []

    def test_unknown_ids_silently_skipped(self, disk_storage):
        disk_storage.add_user_profile("u1", [_make_profile("u1", "p1", "a")])
        result = disk_storage.get_profiles_by_ids("u1", ["p1", "missing"])
        assert {p.profile_id for p in result} == {"p1"}

    def test_filters_by_user_id(self, disk_storage):
        disk_storage.add_user_profile("u1", [_make_profile("u1", "p1", "a")])
        disk_storage.add_user_profile("u2", [_make_profile("u2", "p2", "b")])
        # Asking u1 for u2's profile_id returns nothing — disk storage
        # scopes by directory, not just by row contents.
        assert disk_storage.get_profiles_by_ids("u1", ["p2"]) == []

    def test_default_status_filter_excludes_archived(self, disk_storage):
        disk_storage.add_user_profile(
            "u1",
            [
                _make_profile("u1", "p1", "current"),
                _make_profile("u1", "p2", "to-archive"),
            ],
        )
        disk_storage.archive_profile_by_id("u1", "p2")
        result = disk_storage.get_profiles_by_ids("u1", ["p1", "p2"])
        assert {p.profile_id for p in result} == {"p1"}

    def test_explicit_status_filter_includes_archived(self, disk_storage):
        disk_storage.add_user_profile("u1", [_make_profile("u1", "p1", "x")])
        disk_storage.archive_profile_by_id("u1", "p1")
        result = disk_storage.get_profiles_by_ids(
            "u1", ["p1"], status_filter=[Status.ARCHIVED]
        )
        assert {p.profile_id for p in result} == {"p1"}


# ---------------------------------------------------------------------------
# archive_profile_by_id
# ---------------------------------------------------------------------------


class TestDiskArchiveProfileById:
    def test_archives_current_profile(self, disk_storage):
        disk_storage.add_user_profile("u1", [_make_profile("u1", "p1", "c")])
        assert disk_storage.archive_profile_by_id("u1", "p1") is True

        current = disk_storage.get_user_profile("u1", status_filter=[None])
        archived = disk_storage.get_user_profile("u1", status_filter=[Status.ARCHIVED])
        assert current == []
        assert {p.profile_id for p in archived} == {"p1"}

    def test_returns_false_for_missing_profile(self, disk_storage):
        assert disk_storage.archive_profile_by_id("u1", "nope") is False

    def test_returns_false_when_already_archived(self, disk_storage):
        disk_storage.add_user_profile("u1", [_make_profile("u1", "p1", "c")])
        assert disk_storage.archive_profile_by_id("u1", "p1") is True
        assert disk_storage.archive_profile_by_id("u1", "p1") is False

    def test_returns_false_for_wrong_user(self, disk_storage):
        # On disk storage the user_id scoping happens at the directory
        # level — u1's file lives in u1's dir, so asking u2 to archive
        # it finds no file.
        disk_storage.add_user_profile("u1", [_make_profile("u1", "p1", "c")])
        assert disk_storage.archive_profile_by_id("u2", "p1") is False
        # u1's row is untouched.
        current = disk_storage.get_user_profile("u1", status_filter=[None])
        assert {p.profile_id for p in current} == {"p1"}


# ---------------------------------------------------------------------------
# get_user_playbooks_by_ids
# ---------------------------------------------------------------------------


class TestDiskGetUserPlaybooksByIds:
    def test_returns_only_requested_ids(self, disk_storage):
        disk_storage.save_user_playbooks(
            [
                _make_playbook(1, "u1"),
                _make_playbook(2, "u1"),
                _make_playbook(3, "u1"),
            ]
        )
        result = disk_storage.get_user_playbooks_by_ids("u1", [1, 3])
        assert {p.user_playbook_id for p in result} == {1, 3}

    def test_empty_ids_returns_empty_list(self, disk_storage):
        disk_storage.save_user_playbooks([_make_playbook(1, "u1")])
        assert disk_storage.get_user_playbooks_by_ids("u1", []) == []

    def test_unknown_ids_silently_skipped(self, disk_storage):
        disk_storage.save_user_playbooks([_make_playbook(1, "u1")])
        result = disk_storage.get_user_playbooks_by_ids("u1", [1, 99_999])
        assert {p.user_playbook_id for p in result} == {1}

    def test_filters_by_user_id(self, disk_storage):
        # Disk-storage user_playbooks aren't directory-scoped per user,
        # so user_id filtering must happen at read time.
        disk_storage.save_user_playbooks(
            [
                _make_playbook(1, "u1"),
                _make_playbook(2, "u2"),
            ]
        )
        assert disk_storage.get_user_playbooks_by_ids("u1", [2]) == []

    def test_default_status_filter_excludes_archived(self, disk_storage):
        disk_storage.save_user_playbooks(
            [
                _make_playbook(1, "u1"),
                _make_playbook(2, "u1"),
            ]
        )
        disk_storage.archive_user_playbook_by_id("u1", 2)
        result = disk_storage.get_user_playbooks_by_ids("u1", [1, 2])
        assert {p.user_playbook_id for p in result} == {1}

    def test_explicit_status_filter_includes_archived(self, disk_storage):
        disk_storage.save_user_playbooks([_make_playbook(1, "u1")])
        disk_storage.archive_user_playbook_by_id("u1", 1)
        result = disk_storage.get_user_playbooks_by_ids(
            "u1", [1], status_filter=[Status.ARCHIVED]
        )
        assert {p.user_playbook_id for p in result} == {1}


# ---------------------------------------------------------------------------
# archive_user_playbook_by_id
# ---------------------------------------------------------------------------


class TestDiskArchiveUserPlaybookById:
    def test_archives_current_playbook(self, disk_storage):
        disk_storage.save_user_playbooks([_make_playbook(1, "u1")])
        assert disk_storage.archive_user_playbook_by_id("u1", 1) is True

        current = disk_storage.get_user_playbooks(user_id="u1", status_filter=[None])
        archived = disk_storage.get_user_playbooks(
            user_id="u1", status_filter=[Status.ARCHIVED]
        )
        assert current == []
        assert {p.user_playbook_id for p in archived} == {1}

    def test_returns_false_for_missing_playbook(self, disk_storage):
        assert disk_storage.archive_user_playbook_by_id("u1", 999) is False

    def test_returns_false_when_already_archived(self, disk_storage):
        disk_storage.save_user_playbooks([_make_playbook(1, "u1")])
        assert disk_storage.archive_user_playbook_by_id("u1", 1) is True
        assert disk_storage.archive_user_playbook_by_id("u1", 1) is False

    def test_returns_false_for_wrong_user(self, disk_storage):
        disk_storage.save_user_playbooks([_make_playbook(1, "u1")])
        assert disk_storage.archive_user_playbook_by_id("u2", 1) is False
        current = disk_storage.get_user_playbooks(user_id="u1", status_filter=[None])
        assert {p.user_playbook_id for p in current} == {1}


class TestDiskAgentPlaybookSourceWindows:
    def test_source_windows_round_trip(self, disk_storage):
        disk_storage.set_source_windows_for_agent_playbook(
            10,
            [
                AgentPlaybookSourceWindow(
                    user_playbook_id=2, source_interaction_ids=[20, 21]
                )
            ],
        )

        assert disk_storage.get_source_user_playbook_ids_for_agent_playbook(10) == [2]
        assert disk_storage.get_source_windows_for_agent_playbook(10) == [
            AgentPlaybookSourceWindow(
                user_playbook_id=2, source_interaction_ids=[20, 21]
            )
        ]

    def test_reads_legacy_id_only_map(self, disk_storage):
        path = disk_storage._entity_path(  # noqa: SLF001
            disk_storage._agent_playbook_source_map_dir(),  # noqa: SLF001
            "10",
        )
        path.write_text(
            '{"user_playbook_ids": [2, 3]}',
            encoding="utf-8",
        )

        assert disk_storage.get_source_windows_for_agent_playbook(10) == [
            AgentPlaybookSourceWindow(user_playbook_id=2, source_interaction_ids=[]),
            AgentPlaybookSourceWindow(user_playbook_id=3, source_interaction_ids=[]),
        ]

    def test_reads_legacy_list_map(self, disk_storage):
        path = disk_storage._entity_path(  # noqa: SLF001
            disk_storage._agent_playbook_source_map_dir(),  # noqa: SLF001
            "10",
        )
        path.write_text("[2, 3]", encoding="utf-8")

        assert disk_storage.get_source_windows_for_agent_playbook(10) == [
            AgentPlaybookSourceWindow(user_playbook_id=2, source_interaction_ids=[]),
            AgentPlaybookSourceWindow(user_playbook_id=3, source_interaction_ids=[]),
        ]
