"""Task 2.2: delete_expired_share_links physically deletes expired share links."""

from __future__ import annotations

import pytest

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def _link(n: int, expires_at: int | None) -> dict:
    return {
        "token": f"shr_test.{n}",
        "resource_type": "profile",
        "resource_id": f"r{n}",
        "expires_at": expires_at,
        "created_by_email": None,
    }


def test_delete_expired_share_links(tmp_path):
    s = SQLiteStorage(db_path=str(tmp_path / "t.db"), org_id="org_test")
    s.create_share_link(**_link(1, expires_at=1))  # long expired
    s.create_share_link(**_link(2, expires_at=10_000))  # fresh
    s.create_share_link(**_link(3, expires_at=None))  # never expires
    assert s.delete_expired_share_links(now=1000, grace_seconds=0) == 1
    assert {lnk.id for lnk in s.get_share_links()} == {2, 3}


def test_delete_expired_share_links_respects_grace(tmp_path):
    s = SQLiteStorage(db_path=str(tmp_path / "t.db"), org_id="org_test")
    s.create_share_link(
        **_link(1, expires_at=900)
    )  # expired_at=900, now=1000, grace=200 → cutoff=800 → not past grace
    s.create_share_link(
        **_link(2, expires_at=700)
    )  # expired_at=700, cutoff=800 → past grace, deleted
    assert s.delete_expired_share_links(now=1000, grace_seconds=200) == 1
    remaining = {lnk.id for lnk in s.get_share_links()}
    assert len(remaining) == 1  # only expires_at=900 row survives


def test_delete_expired_share_links_preserves_null(tmp_path):
    s = SQLiteStorage(db_path=str(tmp_path / "t.db"), org_id="org_test")
    s.create_share_link(**_link(1, expires_at=None))
    s.create_share_link(**_link(2, expires_at=None))
    assert s.delete_expired_share_links(now=999_999_999, grace_seconds=0) == 0
    assert len(s.get_share_links()) == 2


def test_delete_expired_share_links_empty(tmp_path):
    s = SQLiteStorage(db_path=str(tmp_path / "t.db"), org_id="org_test")
    assert s.delete_expired_share_links(now=1000, grace_seconds=0) == 0


def test_delete_expired_share_links_respects_limit(tmp_path):
    s = SQLiteStorage(db_path=str(tmp_path / "t.db"), org_id="org_test")
    for i in range(5):
        s.create_share_link(**_link(i, expires_at=i + 1))
    # All 5 are expired (now=1000), but limit=3
    deleted = s.delete_expired_share_links(now=1000, grace_seconds=0, limit=3)
    assert deleted == 3
    assert len(s.get_share_links()) == 2
