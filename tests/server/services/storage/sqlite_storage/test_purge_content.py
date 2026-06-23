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


def _profile(storage: SQLiteStorage, pid: str, uid: str, content: str) -> UserProfile:
    return UserProfile(
        profile_id=pid,
        user_id=uid,
        content=content,
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id="req-1",
    )


def test_has_inbound_lineage_refs_true_when_pointed_to(tmp_path):
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        s = SQLiteStorage(org_id="0", db_path=str(tmp_path / "t.db"))
    # Two profiles; B supersedes A → A.superseded_by=B, so B has an inbound ref.
    s.add_user_profile("alice", [_profile(s, "A", "alice", "old")])
    s.add_user_profile("alice", [_profile(s, "B", "alice", "new")])
    s.supersede_record(
        entity_type="profile", incumbent_id="A", successor_id="B", context=_ctx()
    )
    assert s.has_inbound_lineage_refs(entity_type="profile", entity_id="B") is True
    assert s.has_inbound_lineage_refs(entity_type="profile", entity_id="A") is False
