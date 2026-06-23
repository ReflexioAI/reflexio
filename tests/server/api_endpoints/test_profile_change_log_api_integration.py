"""Integration test: GET /api/profile_change_log serves the lineage reconstruction.

B3 Task 3 repoints the endpoint (via the lib facade ``get_profile_change_logs``)
from the legacy ``profile_change_logs`` table to ``reconstruct_profile_change_log``
(lineage_event linkage + survivor/tombstone content). The earlier revert to the
legacy path is now resolved: the dedup write-side soft-supersedes superseded
profiles (``supersede_profiles_by_ids`` → ``status_change``/``superseded`` events)
instead of hard-deleting without linkage, so reconstruction can reproduce the
removals.

These tests seed via the REAL dedup write-side and deliberately do NOT write the
legacy table — proving the endpoint is served from reconstruction.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.domain.entities import UserProfile
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive
from reflexio.models.api_schema.retriever_schema import ProfileChangeLogViewResponse
from reflexio.server.cache.reflexio_cache import get_reflexio

pytestmark = pytest.mark.integration


def _make_profile(
    user_id: str, profile_id: str, content: str, request_id: str
) -> UserProfile:
    return UserProfile(
        user_id=user_id,
        profile_id=profile_id,
        content=content,
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id=request_id,
        profile_time_to_live=ProfileTimeToLive.INFINITY,
    )


def test_endpoint_serves_reconstructed_change_log(client_with_org):
    """GET /api/profile_change_log returns the RECONSTRUCTED view (B3 Task 3).

    Seeds via the real dedup write-side:
      - survivor profile carries generated_from_request_id == the dedup run id
        (the immutable "added" signal),
      - the incumbent is soft-deleted via supersede_profiles_by_ids, which emits
        a status_change/superseded lineage event under that run id (the "removed"
        signal).

    It does NOT write the legacy table — so if the endpoint still read the legacy
    table it would return zero rows. The endpoint must return the reconstructed
    row with a schema-parseable shape.
    """
    client, org_id = client_with_org
    storage = get_reflexio(org_id=org_id).request_context.storage
    assert storage is not None, "storage must be configured in integration test fixture"

    user_id = "u-recon-test"
    run_id = "req-recon-test"
    old_p = _make_profile(user_id, "p-old-1", "stale profile text", request_id="seed")
    new_p = _make_profile(user_id, "p-new-1", "updated profile text", request_id=run_id)

    storage.add_user_profile(user_id, [old_p])
    storage.add_user_profile(user_id, [new_p])
    # Dedup soft-delete: emits status_change(to_status="superseded", request_id=run_id).
    storage.supersede_profiles_by_ids(user_id, ["p-old-1"], run_id)

    resp = client.get("/api/profile_change_log")
    assert resp.status_code == 200, resp.text

    parsed = ProfileChangeLogViewResponse(**resp.json())
    assert parsed.success is True

    rows = {row.request_id: row for row in parsed.profile_change_logs}
    assert run_id in rows, (
        "endpoint must serve the reconstructed row for the dedup run; "
        "an empty/missing row means it is still reading the legacy table"
    )
    row = rows[run_id]

    assert [p.profile_id for p in row.added_profiles] == ["p-new-1"]
    assert row.added_profiles[0].content == "updated profile text"

    assert [p.profile_id for p in row.removed_profiles] == ["p-old-1"]
    assert row.removed_profiles[0].content == "stale profile text"


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
