"""Integration tests: Class B direct-delete sweep in _gc_tick (Task 2.1).

Key invariant: Class B (pending-tool-call reclamation — and, in enterprise,
share-link reclamation) runs even when ``lineage_gc.enabled=False``. Class A
(profile expiry sweep + tombstone GC) must NOT run when
``lineage_gc.enabled=False``.

Both are gated independently:
  - Class A runs under ``if cfg.lineage_gc.enabled``
  - Class B runs under ``if cfg.expiry_reclamation.enabled``
  - The scheduler STARTS when EITHER flag is True.

Exercised here via ``delete_expired_pending_tool_calls`` — the only Class B
target OSS itself implements. ``delete_expired_share_links`` is an
enterprise-only Class B target (§9.1 of the project-scoped-tenancy design);
the scheduler's ``getattr``-guarded dispatch (see
``reflexio.server.services.lineage.gc_scheduler._CLASS_B_SWEEPS``) skips it
cleanly when a backend does not implement it, so it needs no OSS coverage.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reflexio.models.api_schema.domain.entities import UserProfile
from reflexio.models.config_schema import ExpiryReclamationConfig, LineageGCConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.lineage.gc_scheduler import (
    LineageGCScheduler,
    maybe_start_lineage_gc,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import (
    PendingToolCallRecord,
    PendingToolCallStatus,
    build_pending_tool_call_dedup_key,
    build_scope_hash,
)

pytestmark = pytest.mark.integration

_ORG_ID = "class_b_gc_test_org"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def org_id() -> str:
    return _ORG_ID


def _make_ctx_factory(
    tmp_path,
    *,
    lineage_gc_enabled: bool,
    expiry_reclamation_enabled: bool,
) -> tuple[SQLiteStorage, Callable[[str], RequestContext]]:
    """Build (storage, factory) pair for the given flag combination."""
    storage = SQLiteStorage(org_id=_ORG_ID, db_path=str(tmp_path / "gc_b_test.db"))
    cfg = SimpleNamespace(
        lineage_gc=LineageGCConfig(
            enabled=lineage_gc_enabled, tombstone_grace_window_days=1
        ),
        expiry_reclamation=ExpiryReclamationConfig(enabled=expiry_reclamation_enabled),
    )

    def factory(org_id: str) -> RequestContext:
        ctx = RequestContext.__new__(RequestContext)
        ctx.org_id = org_id
        ctx.storage = storage
        ctx.storage_base_dir = None
        configurator = MagicMock()
        configurator.get_config.return_value = cfg
        ctx.configurator = configurator
        return ctx

    return storage, factory


_CLASS_B_CALL_ID = "call_class_b_test"


def _seed_expired_pending_tool_call(storage: SQLiteStorage) -> None:
    """Create a terminal 'expired' pending tool call, past the 1-day Class B grace."""
    now = datetime.now(UTC)
    scope = {"org_id": storage.org_id, "scope_kind": "org"}
    storage.create_pending_tool_call(
        PendingToolCallRecord(
            id=_CLASS_B_CALL_ID,
            org_id=storage.org_id,
            scope=scope,
            scope_hash=build_scope_hash(scope),
            tool_name="ask_human",
            dedup_key=build_pending_tool_call_dedup_key(
                tool_name="ask_human", question_text="q_class_b"
            ),
            status=PendingToolCallStatus("expired"),
            question_text="q_class_b",
            expires_at=now - timedelta(days=2),
            cache_until=now - timedelta(days=2),
        )
    )


def _pending_tool_call_exists(storage: SQLiteStorage) -> bool:
    """Return whether the seeded pending tool call still exists in storage."""
    return storage.get_pending_tool_call(_CLASS_B_CALL_ID) is not None


def _seed_active_profile(storage: SQLiteStorage) -> UserProfile:
    """Seed an active profile with a far-past expiration (eligible for expiry sweep)."""
    p = UserProfile(
        profile_id="p_class_b_active",
        user_id="u_class_b",
        content="test profile",
        last_modified_timestamp=1,
        generated_from_request_id="r_class_b_profile",
        expiration_timestamp=100,  # far in the past
    )
    storage.add_user_profile("u_class_b", [p])
    return p


# ---------------------------------------------------------------------------
# Test 1: Class B reclaims even when lineage_gc is DISABLED
# ---------------------------------------------------------------------------


def test_class_b_runs_when_lineage_gc_disabled(tmp_path, org_id):
    """Class B sweep deletes expired pending tool calls when lineage_gc.enabled=False.

    Also asserts that Class A (profile expiry sweep) does NOT run, confirming
    that the two guards are independent.
    """
    storage, factory = _make_ctx_factory(
        tmp_path,
        lineage_gc_enabled=False,
        expiry_reclamation_enabled=True,
    )

    # Seed a long-expired (terminal 'expired' status) pending tool call.
    _seed_expired_pending_tool_call(storage)
    assert _pending_tool_call_exists(storage)

    # Seed an active profile with a past expiration — if Class A ran it would
    # be tombstoned.
    profile = _seed_active_profile(storage)

    sched = LineageGCScheduler(request_context_factory=factory, bootstrap_org_id=org_id)
    sched._gc_tick([org_id])

    # Class B: pending tool call must be reclaimed.
    assert not _pending_tool_call_exists(storage), (
        "Class B must delete the expired pending tool call even when "
        "lineage_gc is disabled"
    )

    # Class A must NOT have run: the active profile must still be active (not tombstoned).
    row = storage.get_profile_by_id(profile.profile_id, include_tombstones=True)
    assert row is not None, "Profile must still exist — Class A must not have run"
    assert row.status is None, (
        "Profile status must remain None (CURRENT); expire_active_profiles must not "
        "have been called (lineage_gc.enabled=False gates Class A)"
    )


# ---------------------------------------------------------------------------
# Test 2: Both disabled → scheduler does not start
# ---------------------------------------------------------------------------


def test_scheduler_does_not_start_when_both_disabled(tmp_path, org_id):
    """maybe_start_lineage_gc returns None when both flags are False."""
    _, factory = _make_ctx_factory(
        tmp_path,
        lineage_gc_enabled=False,
        expiry_reclamation_enabled=False,
    )
    result = maybe_start_lineage_gc(factory, bootstrap_org_id=org_id)
    assert result is None, (
        "Scheduler must not start when lineage_gc.enabled=False and "
        "expiry_reclamation.enabled=False"
    )


# ---------------------------------------------------------------------------
# Test 3: Scheduler starts when only expiry_reclamation is enabled
# ---------------------------------------------------------------------------


def test_scheduler_starts_when_only_expiry_reclamation_enabled(tmp_path, org_id):
    """maybe_start_lineage_gc starts when expiry_reclamation.enabled=True even
    with lineage_gc.enabled=False."""
    _, factory = _make_ctx_factory(
        tmp_path,
        lineage_gc_enabled=False,
        expiry_reclamation_enabled=True,
    )
    sched = maybe_start_lineage_gc(factory, bootstrap_org_id=org_id)
    assert sched is not None, (
        "Scheduler must start when expiry_reclamation.enabled=True "
        "(even with lineage_gc.enabled=False)"
    )
    sched.stop(timeout_seconds=1.0)


# ---------------------------------------------------------------------------
# Test 4: Class B tick is a no-op when expiry_reclamation is disabled
# ---------------------------------------------------------------------------


def test_class_b_does_not_run_when_disabled(tmp_path, org_id):
    """When expiry_reclamation.enabled=False the pending tool call is NOT deleted."""
    storage, factory = _make_ctx_factory(
        tmp_path,
        lineage_gc_enabled=False,
        expiry_reclamation_enabled=False,
    )
    _seed_expired_pending_tool_call(storage)
    assert _pending_tool_call_exists(storage)

    sched = LineageGCScheduler(request_context_factory=factory, bootstrap_org_id=org_id)
    sched._gc_tick([org_id])

    assert _pending_tool_call_exists(storage), (
        "Class B must not run when expiry_reclamation.enabled=False"
    )
