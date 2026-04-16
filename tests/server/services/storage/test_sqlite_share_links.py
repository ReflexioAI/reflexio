"""Tests for SQLite share link storage implementation."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage


@pytest.fixture
def storage(tmp_path):
    """Create a fresh SQLiteStorage in a temp dir."""
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        yield SQLiteStorage(org_id="test-org", db_path=str(tmp_path / "reflexio.db"))


class TestCreateShareLink:
    def test_creates_and_returns_link(self, storage):
        link = storage.create_share_link(
            token="shr_Mw.abc",
            resource_type="profile",
            resource_id="p1",
            expires_at=None,
            created_by_email=None,
        )
        assert link.id is not None
        assert link.token == "shr_Mw.abc"
        assert link.resource_type == "profile"
        assert link.resource_id == "p1"
        assert link.org_id == "test-org"
        assert link.created_at is not None

    def test_creates_with_expires_at(self, storage):
        link = storage.create_share_link(
            token="shr_Mw.exp",
            resource_type="profile",
            resource_id="p1",
            expires_at=9999999999,
            created_by_email="a@b.com",
        )
        assert link.expires_at == 9999999999
        assert link.created_by_email == "a@b.com"


class TestGetShareLinkByToken:
    def test_found(self, storage):
        created = storage.create_share_link(
            token="shr_Mw.abc",
            resource_type="profile",
            resource_id="p1",
            expires_at=None,
            created_by_email=None,
        )
        found = storage.get_share_link_by_token("shr_Mw.abc")
        assert found is not None
        assert found.id == created.id

    def test_not_found(self, storage):
        assert storage.get_share_link_by_token("shr_doesnotexist") is None


class TestGetShareLinkByResource:
    def test_found(self, storage):
        created = storage.create_share_link(
            token="shr_Mw.abc",
            resource_type="profile",
            resource_id="p1",
            expires_at=None,
            created_by_email=None,
        )
        found = storage.get_share_link_by_resource("profile", "p1")
        assert found is not None
        assert found.id == created.id

    def test_not_found(self, storage):
        assert storage.get_share_link_by_resource("profile", "missing") is None


class TestGetShareLinks:
    def test_empty(self, storage):
        assert storage.get_share_links() == []

    def test_multiple(self, storage):
        storage.create_share_link(
            token="shr_Mw.a",
            resource_type="profile",
            resource_id="p1",
            expires_at=None,
            created_by_email=None,
        )
        storage.create_share_link(
            token="shr_Mw.b",
            resource_type="profile",
            resource_id="p2",
            expires_at=None,
            created_by_email=None,
        )
        links = storage.get_share_links()
        assert len(links) == 2


class TestDeleteShareLink:
    def test_deletes_existing(self, storage):
        link = storage.create_share_link(
            token="shr_Mw.a",
            resource_type="profile",
            resource_id="p1",
            expires_at=None,
            created_by_email=None,
        )
        assert storage.delete_share_link(link.id) is True
        assert storage.get_share_link_by_token("shr_Mw.a") is None

    def test_missing_returns_false(self, storage):
        assert storage.delete_share_link(99999) is False


class TestDeleteAllShareLinks:
    def test_empty(self, storage):
        assert storage.delete_all_share_links() == 0

    def test_deletes_all(self, storage):
        for i in range(3):
            storage.create_share_link(
                token=f"shr_Mw.{i}",
                resource_type="profile",
                resource_id=f"p{i}",
                expires_at=None,
                created_by_email=None,
            )
        assert storage.delete_all_share_links() == 3
        assert storage.get_share_links() == []
