"""Task 1.3: expire_active_profiles tombstones TTL-expired active profiles."""

from __future__ import annotations

import pytest

from reflexio.models.api_schema.domain.entities import UserProfile
from reflexio.models.api_schema.domain.enums import Status
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def _p(pid: str, exp: int) -> UserProfile:
    return UserProfile(
        profile_id=pid,
        user_id="u1",
        content="c",
        last_modified_timestamp=1,
        generated_from_request_id="r1",
        expiration_timestamp=exp,
    )


def test_expire_active_profiles_tombstones_only_expired(tmp_path):
    s = SQLiteStorage(db_path=str(tmp_path / "t.db"), org_id="org_test")
    s.add_user_profile("u1", [_p("old", exp=100)])  # expired
    s.add_user_profile("u1", [_p("fresh", exp=10_000)])  # not expired
    n = s.expire_active_profiles(now=1000)
    assert n == 1
    # expired profile is hidden from normal reads
    assert s.get_profile_by_id("old", include_tombstones=False) is None
    # but visible when tombstones included, with EXPIRED status
    row = s.get_profile_by_id("old", include_tombstones=True)
    assert row is not None
    assert row.status == Status.EXPIRED
    # fresh profile is unaffected
    assert s.get_profile_by_id("fresh", include_tombstones=False) is not None


def test_expire_active_profiles_idempotent(tmp_path):
    s = SQLiteStorage(db_path=str(tmp_path / "t.db"), org_id="org_test")
    s.add_user_profile("u1", [_p("old", exp=100)])
    assert s.expire_active_profiles(now=1000) == 1
    assert s.expire_active_profiles(now=1000) == 0  # already tombstoned
