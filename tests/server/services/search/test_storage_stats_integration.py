"""Integration tests for storage_stats — Reflexio facade + tool handler."""

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
from reflexio.server.services.extraction.plan import ExtractionCtx
from reflexio.server.services.extraction.tools import (
    StorageStatsArgs,
    _handle_storage_stats,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def storage_with_data(tmp_path):
    """Storage seeded with two profiles (different timestamps) + one playbook."""
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    storage = SQLiteStorage(org_id="stats-test", db_path=str(tmp_path / "stats.db"))
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
    return storage


def test_handler_counts_match(storage_with_data):
    ctx = ExtractionCtx(user_id="u_with", agent_version="v1", extractor_name="p")
    result = _handle_storage_stats(StorageStatsArgs(), storage_with_data, ctx)
    assert result["profile_count"] == 2
    assert result["playbook_count"] == 1
    assert ctx.search_count == 0  # storage_stats does NOT bump search_count


def test_handler_returns_iso_timestamp_range(storage_with_data):
    ctx = ExtractionCtx(user_id="u_with", agent_version="v1", extractor_name="p")
    result = _handle_storage_stats(StorageStatsArgs(), storage_with_data, ctx)
    expected_oldest = datetime.fromtimestamp(1_700_000_000, tz=UTC).isoformat()
    expected_newest = datetime.fromtimestamp(1_700_001_000, tz=UTC).isoformat()
    assert result["oldest_profile_modified"] == expected_oldest
    assert result["newest_profile_modified"] == expected_newest


def test_handler_returns_null_timestamps_for_empty_user(storage_with_data):
    ctx = ExtractionCtx(user_id="u_no_data", agent_version="v1", extractor_name="p")
    result = _handle_storage_stats(StorageStatsArgs(), storage_with_data, ctx)
    assert result["profile_count"] == 0
    assert result["playbook_count"] == 0
    assert result["oldest_profile_modified"] is None
    assert result["newest_profile_modified"] is None


def test_reflexio_storage_stats_facade(tmp_path):
    """The Reflexio facade method should populate every response field correctly."""
    from reflexio.lib.reflexio_lib import Reflexio

    reflexio = Reflexio(org_id="stats-facade", storage_base_dir=str(tmp_path))
    storage = reflexio._get_storage()
    storage.add_user_profile(
        "u_face",
        [
            UserProfile(
                user_id="u_face",
                profile_id="p1",
                content="profile one",
                profile_time_to_live=ProfileTimeToLive.INFINITY,
                last_modified_timestamp=1_700_000_000,
                expiration_timestamp=NEVER_EXPIRES_TIMESTAMP,
                source="test",
                generated_from_request_id="r",
            ),
        ],
    )
    response = reflexio.storage_stats(StorageStatsRequest(user_id="u_face"))
    assert response.success is True
    assert response.profile_count == 1
    assert response.playbook_count == 0
    assert response.oldest_profile_modified is not None
    assert response.newest_profile_modified is not None
    assert response.oldest_profile_modified == response.newest_profile_modified


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
