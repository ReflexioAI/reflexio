"""Integration tests: B3 empty-request_id invariant.

Verifies that:

1. Two dedup runs with DIFFERENT request_ids reconstruct as TWO separate
   change-log rows (not merged/collapsed).
2. Empty-request_id group is SKIPPED — the new model guards against merging
   unrelated runs under "".
3. ReflectionServiceRequest.request_id has a non-empty default_factory so the
   production path always supplies a non-empty id.
4. supersede_record and merge_records raise ValueError on empty/None
   context.request_id (production raise, not a test-only helper).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from reflexio.lib._profiles import reconstruct_profile_change_log
from reflexio.models.api_schema.domain.entities import LineageContext, UserProfile
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.sqlite_storage._lineage import (
    _EMPTY_REQUEST_ID_MSG,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store(tmp_path) -> SQLiteStorage:
    s = SQLiteStorage(org_id=f"rid-org-{tmp_path.name}", db_path=str(tmp_path / "r.db"))
    s.migrate()
    return s


def _make_profile(
    user_id: str = "u1",
    profile_id: str = "p1",
    content: str = "c",
    request_id: str = "",
) -> UserProfile:
    return UserProfile(
        user_id=user_id,
        profile_id=profile_id,
        content=content,
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id=request_id or f"gen_{profile_id}",
        profile_time_to_live=ProfileTimeToLive.INFINITY,
    )


def _seed_dedup_run(
    s: SQLiteStorage,
    *,
    user_id: str,
    old_id: str,
    new_id: str,
    request_id: str,
    old_content: str = "old",
    new_content: str = "new",
) -> tuple[UserProfile, UserProfile]:
    """Seed a dedup run using the stable signals model.

    - new profile carries generated_from_request_id == request_id
    - old profile is soft-deleted via supersede_profiles_by_ids (status_change+superseded)
    """
    old = _make_profile(
        user_id=user_id, profile_id=old_id, content=old_content, request_id="seed"
    )
    new = _make_profile(
        user_id=user_id, profile_id=new_id, content=new_content, request_id=request_id
    )
    s.add_user_profile(user_id, [old])
    s.add_user_profile(user_id, [new])
    if request_id:
        s.supersede_profiles_by_ids(user_id, [old_id], request_id)
    return old, new


# ---------------------------------------------------------------------------
# Test 1 — Two distinct request_ids reconstruct as two separate rows
# ---------------------------------------------------------------------------


def test_distinct_request_ids_produce_two_separate_rows(tmp_path):
    """Two dedup runs with different request_ids reconstruct as TWO rows.

    The grouping key is request_id.  When each dedup run carries its own
    unique id, reconstruction must yield one row per id.
    """
    s = _store(tmp_path)
    _seed_dedup_run(
        s,
        user_id="u1",
        old_id="p-old-a",
        new_id="p-new-a",
        request_id="req-aaa",
    )
    _seed_dedup_run(
        s,
        user_id="u1",
        old_id="p-old-b",
        new_id="p-new-b",
        request_id="req-bbb",
    )

    result = reconstruct_profile_change_log(s)
    assert result.success
    req_ids = {row.request_id for row in result.profile_change_logs}
    assert "req-aaa" in req_ids, "expected row for req-aaa"
    assert "req-bbb" in req_ids, "expected row for req-bbb"


# ---------------------------------------------------------------------------
# Test 2 — Empty request_id collapses unrelated events into one group
# (documents the collapse risk; proves the invariant must be enforced)
# ---------------------------------------------------------------------------


def test_empty_request_id_group_is_skipped(tmp_path):
    """GUARD: profiles with generated_from_request_id='' produce no reconstruction row.

    The new time-travel-stable model explicitly skips the empty-string
    generated_from_request_id so unrelated runs can never be merged under "".
    The distinct-column query excludes empty-string values at the DB level.

    This replaces the old "collapse" documentation: the new behavior is a
    hard skip, enforced in reconstruct_profile_change_log.

    Seeds rows with TRULY empty generated_from_request_id (bypassing the
    _make_profile helper's ``or`` fallback by constructing directly) to ensure
    the skip is exercised, not vacuously passing on an empty DB.
    """
    s = _store(tmp_path)
    # Add profiles with TRULY empty generated_from_request_id (bypass the
    # _make_profile helper's `or` fallback by constructing directly).
    p1 = UserProfile(
        user_id="u1",
        profile_id="p-empty-1",
        content="c1",
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id="",
        profile_time_to_live=ProfileTimeToLive.INFINITY,
    )
    p2 = UserProfile(
        user_id="u1",
        profile_id="p-empty-2",
        content="c2",
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id="",
        profile_time_to_live=ProfileTimeToLive.INFINITY,
    )
    s.add_user_profile("u1", [p1, p2])
    # Verify the seed was actually persisted (non-vacuous guard): query the
    # DB directly to confirm empty generated_from_request_id rows are present.
    rows = s.conn.execute(
        "SELECT profile_id FROM profiles WHERE generated_from_request_id = ''",
    ).fetchall()
    assert len(rows) == 2, (
        "test setup failed: expected 2 profiles with empty generated_from_request_id "
        f"in storage, got {len(rows)}"
    )

    result = reconstruct_profile_change_log(s)
    assert result.success
    # Empty request_id group is skipped entirely — the seeded rows are in storage
    # but the empty-string key is excluded before grouping.
    req_ids = {row.request_id for row in result.profile_change_logs}
    assert "" not in req_ids, "empty request_id must be skipped"
    assert result.profile_change_logs == [], "empty generated_from_request_id → no rows"


# ---------------------------------------------------------------------------
# Test 3 — ReflectionServiceRequest has a non-empty default_factory
# (production safety: no production call path arrives with an empty id)
# ---------------------------------------------------------------------------


def test_reflection_service_request_default_request_id_is_nonempty():
    """ReflectionServiceRequest.request_id default_factory yields a non-empty str.

    The default_factory is ``lambda: uuid.uuid4().hex``.  This test proves that
    a bare ``ReflectionServiceRequest(user_id="u")`` always carries a non-empty
    request_id — so the production reflection→supersede path is guarded by
    construction.
    """
    from reflexio.server.services.reflection.reflection_service_utils import (
        ReflectionServiceRequest,
    )

    req = ReflectionServiceRequest(user_id="u1")
    assert req.request_id != "", "default request_id must be non-empty"
    assert len(req.request_id) > 0

    # Two independently-constructed requests must not share the same id.
    req2 = ReflectionServiceRequest(user_id="u1")
    assert req.request_id != req2.request_id, (
        "each ReflectionServiceRequest must mint a unique request_id by default"
    )


def test_reflection_service_request_explicit_request_id_preserved():
    """An explicitly-supplied request_id is passed through unchanged."""
    from reflexio.server.services.reflection.reflection_service_utils import (
        ReflectionServiceRequest,
    )

    fixed_id = uuid.uuid4().hex
    req = ReflectionServiceRequest(user_id="u1", request_id=fixed_id)
    assert req.request_id == fixed_id


# ---------------------------------------------------------------------------
# Test 4 — supersede_record and merge_records raise on empty request_id
# (production guards in _lineage.py, not a test-only helper)
# ---------------------------------------------------------------------------


def test_supersede_record_raises_on_empty_request_id(tmp_path):
    """supersede_record raises ValueError when context.request_id is empty."""
    s = _store(tmp_path)
    ctx = LineageContext(op_kind="revise", actor="test", request_id="")
    with pytest.raises(ValueError, match=_EMPTY_REQUEST_ID_MSG):
        s.supersede_record(
            entity_type="user_playbook",
            incumbent_id="1",
            successor_id="2",
            context=ctx,
        )


def test_supersede_record_raises_on_none_request_id(tmp_path):
    """supersede_record raises ValueError when context.request_id is None."""
    s = _store(tmp_path)
    # LineageContext allows None for request_id field
    ctx = LineageContext(op_kind="revise", actor="test", request_id=None)
    with pytest.raises(ValueError, match=_EMPTY_REQUEST_ID_MSG):
        s.supersede_record(
            entity_type="user_playbook",
            incumbent_id="1",
            successor_id="2",
            context=ctx,
        )


def test_merge_records_raises_on_empty_request_id(tmp_path):
    """merge_records raises ValueError when context.request_id is empty."""
    s = _store(tmp_path)
    ctx = LineageContext(op_kind="merge", actor="test", request_id="")
    with pytest.raises(ValueError, match=_EMPTY_REQUEST_ID_MSG):
        s.merge_records(
            entity_type="user_playbook",
            survivor_id="1",
            source_ids=["2"],
            context=ctx,
        )


def test_merge_records_raises_on_none_request_id(tmp_path):
    """merge_records raises ValueError when context.request_id is None."""
    s = _store(tmp_path)
    ctx = LineageContext(op_kind="merge", actor="test", request_id=None)
    with pytest.raises(ValueError, match=_EMPTY_REQUEST_ID_MSG):
        s.merge_records(
            entity_type="user_playbook",
            survivor_id="1",
            source_ids=["2"],
            context=ctx,
        )


# ---------------------------------------------------------------------------
# Test 5 — whitespace-only request_id is also rejected (F009)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ws_id", ["   ", "\t", "\n"])
def test_supersede_record_raises_on_whitespace_request_id(tmp_path, ws_id):
    """supersede_record raises ValueError when context.request_id is whitespace-only."""
    s = _store(tmp_path)
    ctx = LineageContext(op_kind="revise", actor="test", request_id=ws_id)
    with pytest.raises(ValueError, match=_EMPTY_REQUEST_ID_MSG):
        s.supersede_record(
            entity_type="user_playbook",
            incumbent_id="1",
            successor_id="2",
            context=ctx,
        )


@pytest.mark.parametrize("ws_id", ["   ", "\t", "\n"])
def test_merge_records_raises_on_whitespace_request_id(tmp_path, ws_id):
    """merge_records raises ValueError when context.request_id is whitespace-only."""
    s = _store(tmp_path)
    ctx = LineageContext(op_kind="merge", actor="test", request_id=ws_id)
    with pytest.raises(ValueError, match=_EMPTY_REQUEST_ID_MSG):
        s.merge_records(
            entity_type="user_playbook",
            survivor_id="1",
            source_ids=["2"],
            context=ctx,
        )
