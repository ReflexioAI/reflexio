"""Integration test: GET /api/profile_change_log uses read-side reconstruction.

Phase B3 / Task 3: Verifies that the endpoint now calls
``reconstruct_profile_change_log`` (read from lineage_event) instead of the
legacy ``get_profile_change_logs`` (reads profile_change_logs table).

The response shape must be byte-identical to the legacy path:
  - success = True
  - profile_change_logs: list of ProfileChangeLogView
  - each entry has added_profiles / removed_profiles / mentioned_profiles=[]
  - parseable by ProfileChangeLogViewResponse
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.domain.entities import LineageContext, UserProfile
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive
from reflexio.models.api_schema.retriever_schema import ProfileChangeLogViewResponse
from reflexio.server.cache.reflexio_cache import get_reflexio

pytestmark = pytest.mark.integration


def _make_profile(
    user_id: str,
    profile_id: str,
    content: str,
) -> UserProfile:
    return UserProfile(
        user_id=user_id,
        profile_id=profile_id,
        content=content,
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id=f"req_{profile_id}",
        profile_time_to_live=ProfileTimeToLive.INFINITY,
    )


def _seed_supersede(
    storage,
    *,
    user_id: str,
    old_id: str,
    new_id: str,
    request_id: str,
    old_content: str = "old facts",
    new_content: str = "new facts",
) -> tuple[UserProfile, UserProfile]:
    old = _make_profile(user_id=user_id, profile_id=old_id, content=old_content)
    new = _make_profile(user_id=user_id, profile_id=new_id, content=new_content)
    storage.add_user_profile(user_id, [old])
    storage.add_user_profile(user_id, [new])
    ctx = LineageContext(op_kind="revise", actor="reflection", request_id=request_id)
    storage.supersede_record(
        entity_type="profile",
        incumbent_id=old_id,
        successor_id=new_id,
        context=ctx,
    )
    return old, new


def test_endpoint_returns_reconstructed_change_log(client_with_org):
    """GET /api/profile_change_log returns reconstructed view from lineage_event.

    Seeds a supersede into storage (writing only to lineage_event, NOT to
    profile_change_logs), then asserts the endpoint returns:
    - success=True
    - one change-log entry with the correct added/removed profiles
    - mentioned_profiles=[]
    - response parseable by ProfileChangeLogViewResponse
    """
    client, org_id = client_with_org
    storage = get_reflexio(org_id=org_id).request_context.storage

    old_p, new_p = _seed_supersede(
        storage,
        user_id="u-endpoint-test",
        old_id="p-old-1",
        new_id="p-new-1",
        request_id="req-endpoint-test",
        old_content="stale profile text",
        new_content="updated profile text",
    )

    resp = client.get("/api/profile_change_log")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    # Must parse cleanly — shape-identical to the legacy response.
    parsed = ProfileChangeLogViewResponse(**body)
    assert parsed.success is True

    logs = parsed.profile_change_logs
    assert len(logs) == 1

    row = logs[0]
    assert row.request_id == "req-endpoint-test"

    assert len(row.added_profiles) == 1
    assert row.added_profiles[0].profile_id == "p-new-1"
    assert row.added_profiles[0].content == "updated profile text"

    assert len(row.removed_profiles) == 1
    assert row.removed_profiles[0].profile_id == "p-old-1"
    assert row.removed_profiles[0].content == "stale profile text"

    # mentioned_profiles is always [] in Stage-1 — same as legacy path.
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
