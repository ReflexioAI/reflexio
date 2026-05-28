"""Contract tests for ExtrasMixin — run against every local storage backend."""

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.domain.enums import UserActionType
from reflexio.models.api_schema.service_schemas import (
    Citation,
    Interaction,
    ProfileChangeLog,
    UserProfile,
)
from reflexio.server.services.storage.storage_base._extras import ExtrasMixin

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profile_change_log(user_id: str, change_description: str) -> ProfileChangeLog:
    return ProfileChangeLog(
        id=0,
        user_id=user_id,
        request_id=f"req-{user_id}",
        created_at=1_700_000_000,
        added_profiles=[
            UserProfile(
                user_id=user_id,
                profile_id=f"prof-{user_id}",
                content=change_description,
                last_modified_timestamp=1_700_000_000,
                generated_from_request_id=f"req-{user_id}",
            )
        ],
        removed_profiles=[],
        mentioned_profiles=[],
    )


# ---------------------------------------------------------------------------
# TestProfileChangeLogs
# ---------------------------------------------------------------------------


class TestProfileChangeLogs:
    def test_add_and_get_profile_change_logs(self, storage):
        storage.add_profile_change_log(_make_profile_change_log("u1", "added greeting"))
        storage.add_profile_change_log(
            _make_profile_change_log("u2", "added preference")
        )

        logs = storage.get_profile_change_logs()
        assert len(logs) == 2

    def test_delete_profile_change_log_for_user(self, storage):
        storage.add_profile_change_log(_make_profile_change_log("u1", "log for u1"))
        storage.add_profile_change_log(_make_profile_change_log("u2", "log for u2"))

        storage.delete_profile_change_log_for_user("u1")

        logs = storage.get_profile_change_logs()
        assert len(logs) == 1
        assert logs[0].user_id == "u2"

    def test_delete_all_profile_change_logs(self, storage):
        storage.add_profile_change_log(_make_profile_change_log("u1", "log 1"))
        storage.add_profile_change_log(_make_profile_change_log("u2", "log 2"))

        storage.delete_all_profile_change_logs()
        assert storage.get_profile_change_logs() == []


# ---------------------------------------------------------------------------
# TestPlaybookApplicationStats
# ---------------------------------------------------------------------------


def _make_interaction(
    request_id: str,
    created_at: int,
    citations: list[Citation],
) -> Interaction:
    return Interaction(
        interaction_id=0,
        user_id="u1",
        request_id=request_id,
        created_at=created_at,
        role="Assistant",
        content="answer",
        user_action=UserActionType.NONE,
        user_action_description="",
        interacted_image_url="",
        shadow_content="",
        expert_content="",
        tools_used=[],
        citations=citations,
    )


class TestPlaybookApplicationStats:
    def test_empty_when_no_citations(self, storage):
        # Backends that have no implementation return [] from the default; SQLite
        # returns [] when no interactions carry citations. Either way: empty.
        assert storage.get_playbook_application_stats(days_back=30) == []

    def test_aggregates_by_kind_and_real_id(self, storage):
        if not _backend_supports_application_stats(storage):
            pytest.skip("Backend does not implement get_playbook_application_stats")
        now = int(datetime.now(tz=UTC).timestamp())
        # Two interactions cite playbook 42; one also cites profile p-99.
        storage._insert_interaction(
            _make_interaction(
                "r1",
                now - 100,
                [
                    Citation(
                        kind="playbook", real_id="42", tag="s1-2a", title="timeline"
                    ),
                    Citation(
                        kind="profile", real_id="p-99", tag="p1-99", title="terse"
                    ),
                ],
            )
        )
        storage._insert_interaction(
            _make_interaction(
                "r2",
                now,
                [
                    Citation(
                        kind="playbook", real_id="42", tag="s1-2a", title="timeline"
                    )
                ],
            )
        )

        stats = storage.get_playbook_application_stats(days_back=30)
        assert len(stats) == 2

        # Most-applied row sorts first.
        top = stats[0]
        assert top.kind == "playbook"
        assert top.real_id == "42"
        assert top.applied_count == 2
        assert top.title == "timeline"
        # last_applied_at should be the LATER of the two interactions.
        assert top.last_applied_at == now

        profile_row = stats[1]
        assert profile_row.kind == "profile"
        assert profile_row.real_id == "p-99"
        assert profile_row.applied_count == 1

    def test_respects_days_back_window(self, storage):
        if not _backend_supports_application_stats(storage):
            pytest.skip("Backend does not implement get_playbook_application_stats")
        now = int(datetime.now(tz=UTC).timestamp())
        old = now - 60 * 24 * 60 * 60  # 60 days ago
        storage._insert_interaction(
            _make_interaction(
                "r_old",
                old,
                [Citation(kind="playbook", real_id="42", tag="s1-2a", title="old")],
            )
        )
        # 30-day window excludes the 60-day-old citation.
        assert storage.get_playbook_application_stats(days_back=30) == []
        # 90-day window includes it.
        stats = storage.get_playbook_application_stats(days_back=90)
        assert len(stats) == 1 and stats[0].applied_count == 1

    def test_counts_duplicate_citations_once_per_interaction(self, storage):
        if not _backend_supports_application_stats(storage):
            pytest.skip("Backend does not implement get_playbook_application_stats")
        now = int(datetime.now(tz=UTC).timestamp())
        storage._insert_interaction(
            _make_interaction(
                "r_duplicate",
                now,
                [
                    Citation(
                        kind="playbook", real_id="42", tag="s1-2a", title="timeline"
                    ),
                    Citation(
                        kind="playbook", real_id="42", tag="s1-2a", title="timeline"
                    ),
                ],
            )
        )

        stats = storage.get_playbook_application_stats(days_back=30)

        assert len(stats) == 1
        assert stats[0].applied_count == 1


def _backend_supports_application_stats(storage) -> bool:
    """True when the storage backend has a real (non-default) implementation.

    The default in ``ExtrasMixin`` returns ``[]`` for any input — backends
    that haven't been wired up yet (supabase, postgres) hit that path
    and have nothing to test.
    """
    return (
        storage.__class__.get_playbook_application_stats
        is not ExtrasMixin.get_playbook_application_stats
    )
