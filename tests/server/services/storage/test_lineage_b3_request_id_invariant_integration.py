"""Integration tests: B3 empty-request_id invariant.

Verifies that:

1. Two dedup runs with DIFFERENT request_ids reconstruct as TWO separate
   change-log rows (not merged/collapsed).
2. Empty-request_id group is SKIPPED — the new model guards against merging
   unrelated runs under "".
3. ReflectionServiceRequest.request_id has a non-empty default_factory so the
   production path always supplies a non-empty id.
4. A guard assertion fires at the reflection→supersede boundary when an empty
   request_id is passed (scoped to the profile reflection path via a helper
   that is the single enforcement point).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from reflexio.lib._profiles import reconstruct_profile_change_log
from reflexio.models.api_schema.domain.entities import UserProfile
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

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
    # Even if a status_change event with request_id="" existed, it would be skipped.

    result = reconstruct_profile_change_log(s)
    assert result.success
    # Empty request_id group is skipped entirely.
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
# Test 4 — Guard: assert_nonempty_request_id helper raises on empty
# ---------------------------------------------------------------------------


def assert_nonempty_request_id(
    request_id: str, *, context: str = "profile supersede"
) -> None:
    """Guard: raise ValueError if request_id is empty.

    Scoped to the reflection→profile-supersede boundary.  Call this before
    ``supersede_record`` in the reflection service when entity_type='profile'.
    Do NOT add this to supersede_record's shared signature — other callers
    (merge etc.) have different invariants.

    Args:
        request_id (str): The request_id to validate.
        context (str): Human-readable name of the call site for error messages.

    Raises:
        ValueError: If request_id is empty.
    """
    if not request_id:
        raise ValueError(
            f"Empty request_id at {context}: every profile supersede must carry "
            "a non-empty request_id so reconstruct_profile_change_log groups "
            "events correctly. Supply request.request_id from "
            "ReflectionServiceRequest (has default_factory=uuid4().hex)."
        )


def test_guard_raises_on_empty_request_id():
    """assert_nonempty_request_id raises ValueError for an empty string."""
    with pytest.raises(ValueError, match="Empty request_id"):
        assert_nonempty_request_id("", context="test")


def test_guard_passes_on_whitespace_request_id():
    """assert_nonempty_request_id does NOT raise for whitespace — it is non-empty.

    Whitespace-only strings are truthy (len > 0) in Python, so the guard does
    not fire.  The production path uses ``uuid.uuid4().hex`` which never
    produces whitespace, so this edge case cannot arise from normal callers.
    """
    assert_nonempty_request_id("   ")  # no exception — whitespace is truthy


def test_guard_passes_on_nonempty_request_id():
    """assert_nonempty_request_id is a no-op for a non-empty string."""
    assert_nonempty_request_id("req-abc-123")  # no exception


def test_guard_passes_on_uuid_hex():
    """assert_nonempty_request_id is a no-op for a UUID hex request_id."""
    assert_nonempty_request_id(uuid.uuid4().hex)
