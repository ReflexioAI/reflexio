"""Task 1.8: clear_user_data must reach EXPIRED and expired profile rows.

Before the fix, BaseStorage.clear_user_data used get_user_profile with
``_all_statuses`` that omitted Status.EXPIRED and the unconditional
``expiration_timestamp >= now`` filter hid any profile with a past expiry.
EXPIRED tombstones (expiration_timestamp in the past) were silently skipped
by the GDPR erasure path.

Note: SQLiteStorage has its own clear_user_data override that reads profiles
directly via ``SELECT … WHERE user_id = ?`` (no status/expiry filter), so
the SQLite override already handles EXPIRED profiles.  The bug lives in
BaseStorage.clear_user_data (used by the Supabase/Postgres backends).
We call BaseStorage.clear_user_data(s, …) directly here so the failing
assertion targets the broken base-class code path.

Two test bodies:
  * test_base_clear_user_data_erases_expired_profile — the bug: EXPIRED
    profile survives BaseStorage.clear_user_data (FAILS before fix, PASSES
    after).
  * test_get_user_profile_include_expired_param — the new param:
    include_expired=True surfaces the EXPIRED row; False (default) hides it
    (fails BEFORE fix with TypeError on unknown kwarg, PASSES after).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain.entities import UserProfile
from reflexio.models.api_schema.domain.enums import Status
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import BaseStorage

pytestmark = pytest.mark.integration

_USER = "u_erasure_1_8"
_PAST_EXP = 1  # epoch timestamp well in the past
_FAR_FUTURE = 10_000_000_000  # far future so profile won't expire naturally


def _mk_profile(pid: str, *, exp: int) -> UserProfile:
    return UserProfile(
        profile_id=pid,
        user_id=_USER,
        content=f"pii-{pid}",
        last_modified_timestamp=1,
        generated_from_request_id="req1",
        expiration_timestamp=exp,
    )


def test_base_clear_user_data_erases_expired_profile(tmp_path):
    """EXPIRED profile must be physically absent after BaseStorage.clear_user_data.

    We call BaseStorage.clear_user_data(s, …) directly (bypassing the
    SQLiteStorage override) to exercise the base-class code path shared
    with Supabase/Postgres.

    Before the fix, BaseStorage.clear_user_data calls
    get_user_profile(status_filter=_all_statuses) which (a) omits
    Status.EXPIRED from the list and (b) applies expiration_timestamp >= now
    unconditionally, so the EXPIRED row is never enumerated and survives
    erasure.  After the fix, Status.EXPIRED is in _all_statuses and
    include_expired=True is passed, so the row is enumerated and deleted.
    """
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        s = SQLiteStorage(db_path=str(tmp_path / "t.db"), org_id="org_test_1_8")

        # Add two profiles: one that will become EXPIRED, one that stays CURRENT.
        s.add_user_profile(_USER, [_mk_profile("p_expired", exp=_PAST_EXP)])
        s.add_user_profile(_USER, [_mk_profile("p_current", exp=_FAR_FUTURE)])

        # Transition p_expired → status=EXPIRED via the expiry sweep.
        swept = s.expire_active_profiles(now=1_000_000)
        assert swept == 1, f"Expected 1 profile tombstoned, got {swept}"

        # Sanity: EXPIRED profile is hidden from default get_user_profile.
        live = s.get_user_profile(_USER)
        assert all(p.profile_id != "p_expired" for p in live), (
            "EXPIRED profile should not appear in default get_user_profile"
        )

        # Run on-demand GDPR erasure using the BASE CLASS implementation.
        # (SQLiteStorage overrides clear_user_data with raw SQL that already
        # catches EXPIRED rows; we call the base class to target the broken path.)
        counts = BaseStorage.clear_user_data(s, _USER)

        # The combined profile count (hard-deleted + purged) must cover both rows.
        total = counts.get("profiles", 0) + counts.get("purged_profiles", 0)
        assert total >= 2, (
            f"Expected at least 2 profiles processed by BaseStorage.clear_user_data, "
            f"got counts={counts}"
        )

        # p_expired must no longer exist — even with include_tombstones=True.
        gone = s.get_profile_by_id("p_expired", include_tombstones=True)
        assert gone is None, (
            "EXPIRED profile must be physically absent after BaseStorage.clear_user_data; "
            f"got status={gone.status if gone else None!r}"
        )

        # p_current must also be erased (no regression on normal profiles).
        current_gone = s.get_profile_by_id("p_current", include_tombstones=True)
        assert current_gone is None, (
            "CURRENT profile must also be absent after BaseStorage.clear_user_data; "
            f"got {current_gone!r}"
        )


def test_get_user_profile_include_expired_param(tmp_path):
    """include_expired=True returns the EXPIRED row; False (default) does not.

    Directly tests the new keyword on get_user_profile.  Before the fix,
    the kwarg doesn't exist, so this test raises TypeError (wrapped in
    StorageError by the handle_exceptions decorator).
    """
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        s = SQLiteStorage(db_path=str(tmp_path / "t.db"), org_id="org_param_test")

        s.add_user_profile(_USER, [_mk_profile("px", exp=_PAST_EXP)])
        s.expire_active_profiles(now=1_000_000)

        # include_expired=True with EXPIRED status_filter must surface the row.
        with_expired = s.get_user_profile(
            _USER,
            status_filter=[Status.EXPIRED],
            include_expired=True,
        )
        assert any(p.profile_id == "px" for p in with_expired), (
            "get_user_profile(include_expired=True) must return the EXPIRED row"
        )

        # include_expired=False (default) must not return the EXPIRED row even
        # when the status_filter explicitly includes EXPIRED.
        without_expired = s.get_user_profile(
            _USER,
            status_filter=[Status.EXPIRED],
            include_expired=False,
        )
        assert all(p.profile_id != "px" for p in without_expired), (
            "get_user_profile(include_expired=False) must not return the EXPIRED row"
        )
