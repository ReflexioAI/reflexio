"""Contract tests for ExtrasMixin — run against every local storage backend."""

import pytest

from reflexio.models.api_schema.service_schemas import (
    ProfileChangeLog,
    UserProfile,
)

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
