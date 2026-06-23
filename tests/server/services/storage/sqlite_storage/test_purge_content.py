"""Task 1: has_inbound_lineage_refs query for purge-vs-hard-delete decision."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain.entities import LineageContext, UserProfile
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def _ctx(rid: str = "r1") -> LineageContext:
    return LineageContext(op_kind="revise", actor="test", reason="t", request_id=rid)


def _profile(pid: str, uid: str, content: str) -> UserProfile:
    return UserProfile(
        profile_id=pid,
        user_id=uid,
        content=content,
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id="req-1",
    )


@pytest.fixture
def storage(tmp_path):
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        yield SQLiteStorage(org_id="0", db_path=str(tmp_path / "t.db"))


def test_has_inbound_lineage_refs_true_when_pointed_to(storage):
    # Two profiles; B supersedes A → A.superseded_by=B, so B has an inbound ref.
    storage.add_user_profile("alice", [_profile("A", "alice", "old")])
    storage.add_user_profile("alice", [_profile("B", "alice", "new")])
    storage.supersede_record(
        entity_type="profile", incumbent_id="A", successor_id="B", context=_ctx()
    )
    assert (
        storage.has_inbound_lineage_refs(entity_type="profile", entity_id="B") is True
    )
    assert (
        storage.has_inbound_lineage_refs(entity_type="profile", entity_id="A") is False
    )


def test_has_inbound_lineage_refs_true_when_merged_into(storage):
    # Two profiles; A is merged into B → A.merged_into=B, so B has an inbound ref.
    storage.add_user_profile("bob", [_profile("C", "bob", "old")])
    storage.add_user_profile("bob", [_profile("D", "bob", "new")])
    storage.merge_records(
        entity_type="profile",
        survivor_id="D",
        source_ids=["C"],
        context=_ctx(rid="r2"),
    )
    assert (
        storage.has_inbound_lineage_refs(entity_type="profile", entity_id="D") is True
    )
    assert (
        storage.has_inbound_lineage_refs(entity_type="profile", entity_id="C") is False
    )
