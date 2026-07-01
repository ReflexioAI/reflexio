"""Task 2.3: delete_expired_pending_tool_calls deletes only terminal 'expired' rows.

RESOLVED rows (with live valid_until) are never deleted, even if their
expires_at is in the past — they hold cached results used by resumable
extraction resume paths.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import (
    PendingToolCallRecord,
    PendingToolCallStatus,
    build_pending_tool_call_dedup_key,
    build_scope_hash,
)

pytestmark = pytest.mark.integration


def _seed_call(
    s: SQLiteStorage,
    *,
    id: str,
    status: str,
    expires_at: str | None = None,
    valid_until: str | None = None,
    org_id: str = "org_test",
) -> PendingToolCallRecord:
    """Insert a _pending_tool_calls row with arbitrary status for test setup."""
    now = datetime.now(UTC)
    scope: dict[str, str] = {"org_id": org_id, "scope_kind": "org"}
    record = PendingToolCallRecord(
        id=id,
        org_id=org_id,
        scope=scope,
        scope_hash=build_scope_hash(scope),
        tool_name="ask_human",
        dedup_key=build_pending_tool_call_dedup_key(
            tool_name="ask_human", question_text=f"q_{id}"
        ),
        status=PendingToolCallStatus(status),
        question_text=f"q_{id}",
        expires_at=(
            datetime.fromisoformat(expires_at).replace(tzinfo=UTC)
            if expires_at
            else now + timedelta(hours=1)
        ),
        cache_until=now + timedelta(hours=1),
        valid_until=(
            datetime.fromisoformat(valid_until).replace(tzinfo=UTC)
            if valid_until
            else None
        ),
    )
    return s.create_pending_tool_call(record)


def test_delete_expired_pending_tool_calls_spares_resolved(tmp_path):
    """Expired row is deleted; RESOLVED row with live valid_until is preserved."""
    s = SQLiteStorage(db_path=str(tmp_path / "t.db"), org_id="org_test")
    now = datetime.now(UTC)
    _seed_call(
        s,
        id="a",
        status="expired",
        expires_at=(now - timedelta(days=2)).isoformat(),
    )
    _seed_call(
        s,
        id="b",
        status="resolved",
        valid_until=(now + timedelta(days=5)).isoformat(),
        expires_at=(now - timedelta(days=2)).isoformat(),  # past expires_at but RESOLVED
    )
    deleted = s.delete_expired_pending_tool_calls(
        now=int(now.timestamp()), grace_seconds=86400
    )
    assert deleted == 1
    assert s.get_pending_tool_call("a") is None
    assert s.get_pending_tool_call("b") is not None  # live cached result preserved


def test_delete_expired_within_grace_not_deleted(tmp_path):
    """Expired row inside the grace window is not deleted yet."""
    s = SQLiteStorage(db_path=str(tmp_path / "t.db"), org_id="org_test")
    now = datetime.now(UTC)
    # expires_at 30 min ago, grace = 1 h → cutoff 1 h ago → not past cutoff
    _seed_call(
        s,
        id="c",
        status="expired",
        expires_at=(now - timedelta(minutes=30)).isoformat(),
    )
    deleted = s.delete_expired_pending_tool_calls(
        now=int(now.timestamp()), grace_seconds=3600
    )
    assert deleted == 0
    assert s.get_pending_tool_call("c") is not None


def test_delete_expired_pending_tool_calls_empty(tmp_path):
    """Returns 0 on an empty table."""
    s = SQLiteStorage(db_path=str(tmp_path / "t.db"), org_id="org_test")
    assert s.delete_expired_pending_tool_calls(now=1_000_000, grace_seconds=0) == 0


def test_delete_expired_pending_tool_calls_ignores_pending_status(tmp_path):
    """Pending rows (not yet expired) are never physically deleted."""
    s = SQLiteStorage(db_path=str(tmp_path / "t.db"), org_id="org_test")
    now = datetime.now(UTC)
    # PENDING row whose expires_at is in the past — must NOT be deleted
    _seed_call(
        s,
        id="d",
        status="pending",
        expires_at=(now - timedelta(days=3)).isoformat(),
    )
    deleted = s.delete_expired_pending_tool_calls(
        now=int(now.timestamp()), grace_seconds=0
    )
    assert deleted == 0
    assert s.get_pending_tool_call("d") is not None
