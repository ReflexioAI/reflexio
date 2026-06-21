"""Integration test: GET /api/profile_change_log serves the legacy get_profile_change_logs path.

The endpoint was reverted from reconstruction-backed (B3 Task 3) back to the
legacy storage read, because the production write-side emits ``hard_delete``
events with no linkage — not the ``revise``/``merge`` events the reconstruction
expects — so reconstruction cannot reproduce the legacy log yet.

These tests assert that the endpoint reads from the legacy ``profile_change_logs``
table (via storage.add_profile_change_log / get_profile_change_logs).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.domain.entities import ProfileChangeLog, UserProfile
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive
from reflexio.models.api_schema.retriever_schema import ProfileChangeLogViewResponse
from reflexio.server.cache.reflexio_cache import get_reflexio

pytestmark = pytest.mark.integration


def _make_profile(user_id: str, profile_id: str, content: str) -> UserProfile:
    return UserProfile(
        user_id=user_id,
        profile_id=profile_id,
        content=content,
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id=f"req_{profile_id}",
        profile_time_to_live=ProfileTimeToLive.INFINITY,
    )


def test_endpoint_returns_legacy_change_log(client_with_org):
    """GET /api/profile_change_log returns data from the legacy profile_change_logs table.

    Seeds a ProfileChangeLog entry directly via storage.add_profile_change_log,
    then asserts the endpoint returns:
    - success=True
    - one change-log entry with the correct added/removed profiles
    - mentioned_profiles=[]
    - response parseable by ProfileChangeLogViewResponse
    """
    client, org_id = client_with_org
    storage = get_reflexio(org_id=org_id).request_context.storage
    assert storage is not None, "storage must be configured in integration test fixture"

    old_p = _make_profile(
        user_id="u-legacy-test", profile_id="p-old-1", content="stale profile text"
    )
    new_p = _make_profile(
        user_id="u-legacy-test", profile_id="p-new-1", content="updated profile text"
    )

    log_entry = ProfileChangeLog(
        id=0,
        user_id="u-legacy-test",
        request_id="req-legacy-test",
        added_profiles=[new_p],
        removed_profiles=[old_p],
        mentioned_profiles=[],
    )
    storage.add_profile_change_log(log_entry)

    resp = client.get("/api/profile_change_log")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    parsed = ProfileChangeLogViewResponse(**body)
    assert parsed.success is True

    logs = parsed.profile_change_logs
    assert len(logs) == 1

    row = logs[0]
    assert row.request_id == "req-legacy-test"

    assert len(row.added_profiles) == 1
    assert row.added_profiles[0].profile_id == "p-new-1"
    assert row.added_profiles[0].content == "updated profile text"

    assert len(row.removed_profiles) == 1
    assert row.removed_profiles[0].profile_id == "p-old-1"
    assert row.removed_profiles[0].content == "stale profile text"

    assert row.mentioned_profiles == []


def test_endpoint_response_is_parseable_by_schema(client_with_org):
    """GET /api/profile_change_log always returns a response parseable by
    ProfileChangeLogViewResponse regardless of storage contents.
    """
    client, _ = client_with_org

    resp = client.get("/api/profile_change_log")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    parsed = ProfileChangeLogViewResponse(**body)
    assert parsed.success is True
    assert isinstance(parsed.profile_change_logs, list)
