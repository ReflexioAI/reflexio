"""Integration tests for Reflexio.storage_stats."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.domain.entities import (
    NEVER_EXPIRES_TIMESTAMP,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive
from reflexio.models.api_schema.retriever_schema import StorageStatsRequest

pytestmark = pytest.mark.integration


@pytest.fixture
def reflexio_with_data(tmp_path):
    """Reflexio instance seeded with two profiles and one playbook."""
    from reflexio.lib.reflexio_lib import Reflexio

    reflexio = Reflexio(org_id="stats-test", storage_base_dir=str(tmp_path))
    storage = reflexio._get_storage()
    storage.add_user_profile(
        "u_with",
        [
            UserProfile(
                user_id="u_with",
                profile_id="p_old",
                content="old content",
                profile_time_to_live=ProfileTimeToLive.INFINITY,
                last_modified_timestamp=1_700_000_000,
                expiration_timestamp=NEVER_EXPIRES_TIMESTAMP,
                source="test",
                generated_from_request_id="r",
            ),
            UserProfile(
                user_id="u_with",
                profile_id="p_new",
                content="new content",
                profile_time_to_live=ProfileTimeToLive.INFINITY,
                last_modified_timestamp=1_700_001_000,
                expiration_timestamp=NEVER_EXPIRES_TIMESTAMP,
                source="test",
                generated_from_request_id="r",
            ),
        ],
    )
    storage.save_user_playbooks(
        [
            UserPlaybook(
                user_playbook_id=0,
                user_id="u_with",
                agent_version="v1",
                request_id="r",
                playbook_name="p",
                content="content",
                trigger="trigger",
            )
        ]
    )
    return reflexio


def test_storage_stats_counts_match(reflexio_with_data):
    response = reflexio_with_data.storage_stats(StorageStatsRequest(user_id="u_with"))
    assert response.success is True
    assert response.profile_count == 2
    assert response.playbook_count == 1


def test_storage_stats_returns_timestamp_range(reflexio_with_data):
    response = reflexio_with_data.storage_stats(StorageStatsRequest(user_id="u_with"))
    expected_oldest = datetime.fromtimestamp(1_700_000_000, tz=UTC)
    expected_newest = datetime.fromtimestamp(1_700_001_000, tz=UTC)
    assert response.oldest_profile_modified == expected_oldest
    assert response.newest_profile_modified == expected_newest


def test_reflexio_storage_stats_empty_user(tmp_path):
    """Empty user returns success with zeros and null timestamps."""
    from reflexio.lib.reflexio_lib import Reflexio

    reflexio = Reflexio(org_id="stats-empty", storage_base_dir=str(tmp_path))
    response = reflexio.storage_stats(StorageStatsRequest(user_id="ghost"))
    assert response.success is True
    assert response.profile_count == 0
    assert response.playbook_count == 0
    assert response.oldest_profile_modified is None
    assert response.newest_profile_modified is None
